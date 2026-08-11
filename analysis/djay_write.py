"""Scrittura di hot cue nel database live di djay Pro (Mac, versione App Store).

Reverse engineering fatto per diff mirati (before/after, un solo cambiamento
alla volta) sul file reale dell'utente, con autorizzazione esplicita. Il
database (`~/Music/djay/djay Media Library.djayMediaLibrary/MediaLibrary.db`)
è YapDatabase (SQLite: tabella `database2`, colonne collection/key/data). Ogni
traccia ha una riga in `mediaItemUserData` (collection) il cui campo `data` è
un blob binario proprietario (firma `TSAF`) contenente, fra l'altro, l'array
`cuePoints`.

Formato del blob (verificato su più esperimenti before/after reali):
- offset 8: uint32 LE, "totale oggetti serializzati" nel blob. Aumenta di 1
  per ogni NUOVO cue point aggiunto quando l'array esiste già (verificato:
  +2 per +2 cue). Alla primissima creazione dell'array il delta è diverso
  (osservato +4 per i primi 3 cue): per questo lo scrittore supporta SOLO
  l'aggiunta a tracce che hanno GIÀ almeno un cue point in djay Pro.
- il pattern b"\\x05\\x06\\n\\x00\\x00" precede un uint32 LE che è la
  lunghezza dell'array cuePoints.
- ogni cue oltre al primo è un blocco compatto a lunghezza fissa (24 byte):
    b"+\\x05\\x0f\\x13\\x00" + float32 LE (time)
    + b"\\x05\\x10\\x13\\x00" + float32 LE (endTime, -1.0 = non impostato)
    + b"\\x05\\x11\\x0f" + uint8 (number = slot/pad, determina il colore
      tramite la palette fissa di djay Pro — non c'è un campo colore separato)
    + b"\\x05\\x12\\x00"
  inseriti subito prima di b"+\\x08ADCAudioAlignmentFingerprint".

Non tocca MAI il database live direttamente senza: (1) un backup fresco del
bundle .djayMediaLibrary, (2) una verifica di round-trip che rilegge il nuovo
blob e conferma che i cue esistenti sono bit-per-bit invariati e i nuovi cue
hanno i valori attesi.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import struct
import subprocess
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

MAGIC = b"TSAF"
_TOTAL_COUNT_OFFSET = 8
_ARRAY_COUNT_PREFIX = b"\x05\x06\n\x00\x00"
_FINGERPRINT_MARKER = b"+\x08ADCAudioAlignmentFingerprint"
_CUE_ENTRY_LEN = 24

DEFAULT_LIBRARY_PATH = (
    Path.home() / "Music" / "djay" / "djay Media Library.djayMediaLibrary"
)


@dataclass
class CuePoint:
    time: float
    number: int             # slot/pad 0-7: determina il colore in djay Pro
    end_time: float = -1.0  # -1.0 = non impostato (hot cue semplice, non loop)


@dataclass
class WriteResult:
    ok: bool
    message: str
    track_uuid: str | None = None
    backup_path: Path | None = None
    cues_before: list[CuePoint] = field(default_factory=list)
    cues_after: list[CuePoint] = field(default_factory=list)


class DjayWriteError(Exception):
    """Errore che deve fermare l'operazione senza scrivere nulla."""


# --------------------------------------------------------------------------
# Lettura / parsing (usato sia per la verifica round-trip sia per l'anteprima)
# --------------------------------------------------------------------------

def _db_path(library_path: Path) -> Path:
    return library_path / "MediaLibrary.db"


def _open_readonly(db_file: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)


def read_cue_points(blob: bytes) -> list[CuePoint]:
    """Estrae i cue point esistenti da un blob mediaItemUserData.

    Il primo cue point del blob è sempre codificato "per esteso" (nomi dei
    campi in chiaro: "ADCCuePoint", "time", "endTime", "number", con
    lunghezza fissa nota); quelli successivi usano il formato compatto a 24
    byte scritto da `append_cues`. Entrambi vengono decodificati con offset
    diretti (nessuna euristica), verificati byte per byte con dati reali.
    """
    if not blob.startswith(MAGIC):
        raise DjayWriteError("Il blob non ha la firma TSAF attesa: formato inatteso.")

    fp_idx = blob.find(_FINGERPRINT_MARKER)
    scan_end = fp_idx if fp_idx != -1 else len(blob)

    cues: list[CuePoint] = []

    # Primo cue, verboso. Struttura a offset fissi (verificata sui dati reali):
    #   "ADCCuePoint\x00" \x13\x00 <time f32> \x08time\x00
    #   \x13\x00 <endTime f32> \x08endTime\x00 <number u8> \x08number\x00
    verbose_idx = blob.find(b"ADCCuePoint\x00")
    if verbose_idx != -1 and verbose_idx < scan_end:
        v = verbose_idx + len(b"ADCCuePoint\x00")
        if (blob[v:v + 2] == b"\x13\x00"
                and blob[v + 6:v + 12] == b"\x08time\x00"
                and blob[v + 12:v + 14] == b"\x13\x00"
                and blob[v + 18:v + 27] == b"\x08endTime\x00"
                and blob[v + 28:v + 36] == b"\x08number\x00"):
            time_val = struct.unpack("<f", blob[v + 2:v + 6])[0]
            end_val = struct.unpack("<f", blob[v + 14:v + 18])[0]
            number = blob[v + 27]
            cues.append(CuePoint(time=time_val, number=number, end_time=end_val))

    # Cue compatti successivi, 24 byte ciascuno a partire dal marcatore '+'.
    #   +[0] \x05\x0f[1:3] \x13\x00[3:5] <time f32>[5:9]
    #   \x05\x10\x13\x00[9:13] <endTime f32>[13:17]
    #   \x05\x11\x0f[17:20] <number>[20] \x05\x12\x00[21:24]
    i = 0
    while True:
        i = blob.find(b"+\x05\x0f\x13\x00", i, scan_end)
        if i == -1:
            break
        if (blob[i + 9:i + 13] == b"\x05\x10\x13\x00"
                and blob[i + 17:i + 20] == b"\x05\x11\x0f"
                and blob[i + 21:i + 24] == b"\x05\x12\x00"):
            time_val = struct.unpack("<f", blob[i + 5:i + 9])[0]
            end_val = struct.unpack("<f", blob[i + 13:i + 17])[0]
            number = blob[i + 20]
            cues.append(CuePoint(time=time_val, number=number, end_time=end_val))
            i += _CUE_ENTRY_LEN
        else:
            i += 1
    return cues


def _find_array_count_offset(blob: bytes) -> int:
    matches = [m.start() for m in re.finditer(re.escape(_ARRAY_COUNT_PREFIX), blob)]
    if len(matches) != 1:
        raise DjayWriteError(
            f"Atteso esattamente un array cuePoints, trovati {len(matches)}: "
            "struttura del blob diversa da quella verificata, mi fermo."
        )
    return matches[0] + len(_ARRAY_COUNT_PREFIX)


def _find_fingerprint_offset(blob: bytes) -> int:
    matches = [m.start() for m in re.finditer(re.escape(_FINGERPRINT_MARKER), blob)]
    if len(matches) != 1:
        raise DjayWriteError(
            f"Atteso esattamente un campo audioAlignmentFingerprint, trovati "
            f"{len(matches)}: mi fermo."
        )
    return matches[0]


def append_cues(blob: bytes, new_cues: list[CuePoint]) -> bytes:
    """Accoda nuovi cue point al blob, preservando tutto il resto invariato.

    Richiede che il blob abbia GIÀ un array cuePoints non vuoto (traccia con
    almeno un cue esistente in djay Pro) — è l'unico caso verificato byte per
    byte; altrimenti solleva DjayWriteError senza modificare nulla.
    """
    if not new_cues:
        raise DjayWriteError("Nessun cue da aggiungere.")

    count_offset = _find_array_count_offset(blob)
    old_count = struct.unpack("<I", blob[count_offset:count_offset + 4])[0]
    if old_count == 0:
        raise DjayWriteError(
            "La traccia non ha ancora nessun cue point in djay Pro: questo "
            "scrittore supporta solo l'aggiunta a tracce che ne hanno già "
            "almeno uno (il comportamento sulla primissima creazione "
            "dell'array non è stato verificato in modo sicuro)."
        )

    fp_offset = _find_fingerprint_offset(blob)
    if fp_offset <= count_offset:
        raise DjayWriteError("Ordine dei campi nel blob inatteso: mi fermo.")

    entries = b"".join(
        b"+\x05\x0f\x13\x00" + struct.pack("<f", c.time)
        + b"\x05\x10\x13\x00" + struct.pack("<f", c.end_time)
        + b"\x05\x11\x0f" + struct.pack("<B", c.number)
        + b"\x05\x12\x00"
        for c in new_cues
    )

    new_blob = blob[:fp_offset] + entries + blob[fp_offset:]

    # Aggiorna il contatore dell'array cuePoints (stesso offset, il testo
    # prima di count_offset non cambia lunghezza).
    new_count = old_count + len(new_cues)
    new_blob = (
        new_blob[:count_offset]
        + struct.pack("<I", new_count)
        + new_blob[count_offset + 4:]
    )

    # Aggiorna il contatore "totale oggetti" a offset 8 (+1 per cue aggiunto,
    # verificato quando l'array esiste già).
    old_total = struct.unpack("<I", new_blob[8:12])[0]
    new_total = old_total + len(new_cues)
    new_blob = new_blob[:8] + struct.pack("<I", new_total) + new_blob[12:]

    return new_blob


def verify_roundtrip(original: bytes, updated: bytes, new_cues: list[CuePoint]) -> None:
    """Verifica di sicurezza: rilegge `updated` e conferma che tutto combaci.

    Solleva DjayWriteError (senza permettere la scrittura) se qualcosa non
    torna: cue esistenti alterati, nuovi cue mancanti/sbagliati, o coda del
    blob (dopo i cue) diversa dall'originale.
    """
    before_cues = read_cue_points(original)
    after_cues = read_cue_points(updated)

    if len(after_cues) != len(before_cues) + len(new_cues):
        raise DjayWriteError(
            f"Verifica fallita: attesi {len(before_cues) + len(new_cues)} cue "
            f"dopo la scrittura, trovati {len(after_cues)}."
        )

    for i, c in enumerate(before_cues):
        got = after_cues[i]
        if abs(got.time - c.time) > 1e-4 or got.number != c.number:
            raise DjayWriteError(
                f"Verifica fallita: il cue esistente #{i} risulta alterato "
                f"({c} -> {got})."
            )

    for i, c in enumerate(new_cues):
        got = after_cues[len(before_cues) + i]
        if abs(got.time - c.time) > 1e-4 or got.number != c.number:
            raise DjayWriteError(
                f"Verifica fallita: il nuovo cue #{i} non risulta scritto "
                f"correttamente ({c} -> {got})."
            )

    # La coda del blob (impronta audio + tutto il resto, dopo i cue) deve
    # restare byte per byte identica, solo spostata in avanti.
    fp_before = _find_fingerprint_offset(original)
    fp_after = _find_fingerprint_offset(updated)
    tail_before = original[fp_before:]
    tail_after = updated[fp_after:]
    if tail_before != tail_after:
        raise DjayWriteError(
            "Verifica fallita: la parte del blob dopo i cue point (impronta "
            "audio e altri campi) non è rimasta identica."
        )


# --------------------------------------------------------------------------
# Individuazione della traccia (path del file -> uuid interno di djay Pro)
# --------------------------------------------------------------------------

def find_track_uuid(db_file: Path, filepath: Path) -> str:
    """Trova l'uuid di djay Pro per il file dato.

    Cerca in localMediaItemLocations/globalMediaItemLocations un file:// URI
    che, decodificato, combacia col path assoluto del file. Se più uuid
    puntano allo stesso path (visto in pratica: entry duplicate/orfane),
    seleziona quello con un array cuePoints già popolato — è l'unico modo
    verificato per capire qual è la entry che djay Pro usa davvero.
    """
    target = str(filepath.resolve())
    conn = _open_readonly(db_file)
    try:
        candidates: list[str] = []
        for coll in ("localMediaItemLocations", "globalMediaItemLocations"):
            rows = conn.execute(
                "SELECT key, data FROM database2 WHERE collection=?", (coll,)
            ).fetchall()
            for key, data in rows:
                idx = data.find(b"file://")
                if idx == -1:
                    continue
                end = data.find(b"\x00", idx)
                uri = data[idx:end].decode("utf-8", errors="replace")
                if urllib.parse.unquote(uri[len("file://"):]) == target:
                    candidates.append(key)

        if not candidates:
            raise DjayWriteError(
                "Nessuna traccia trovata in djay Pro per questo file. "
                "Deve essere già presente nella tua libreria djay Pro."
            )

        populated = []
        for uuid in candidates:
            row = conn.execute(
                "SELECT data FROM database2 WHERE collection='mediaItemUserData' AND key=?",
                (uuid,),
            ).fetchone()
            if row is None:
                continue
            try:
                offset = _find_array_count_offset(row[0])
                count = struct.unpack("<I", row[0][offset:offset + 4])[0]
            except DjayWriteError:
                count = 0
            if count > 0:
                populated.append(uuid)

        if len(populated) == 1:
            return populated[0]
        if len(populated) == 0:
            raise DjayWriteError(
                "Trovata la traccia in djay Pro ma senza nessun cue point "
                "esistente. Aggiungine almeno uno manualmente in djay Pro "
                "prima di usare questa funzione (vedi limite noto)."
            )
        raise DjayWriteError(
            f"Trovate {len(populated)} tracce diverse in djay Pro con cue "
            "point per lo stesso file (probabile duplicato nella libreria "
            "djay Pro): non posso scegliere in automatico. Consolida le "
            "voci duplicate in djay Pro prima di riprovare."
        )
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Backup e scrittura effettiva
# --------------------------------------------------------------------------

def is_djay_running() -> bool:
    try:
        out = subprocess.run(
            ["pgrep", "-x", "djay Pro"], capture_output=True, text=True,
        )
        return out.returncode == 0
    except Exception:
        return False  # best-effort: se il check fallisce non blocchiamo da soli


def backup_library(library_path: Path = DEFAULT_LIBRARY_PATH) -> Path:
    """Copia l'intero bundle .djayMediaLibrary (db+wal+shm) prima di scrivere."""
    if not library_path.exists():
        raise DjayWriteError(f"Libreria djay Pro non trovata in {library_path}")
    dest_root = Path.home() / "Music" / "Wavecut" / "Backups"
    stamp = time.strftime("%Y-%m-%d-(%H-%M-%S)")
    dest = dest_root / f"{stamp}-djayPro-backup" / library_path.name
    dest.mkdir(parents=True, exist_ok=True)
    for f in library_path.glob("MediaLibrary.db*"):
        shutil.copy2(f, dest / f.name)
    return dest


def _require_library(library_path: Path) -> Path:
    db_file = _db_path(library_path)
    if not db_file.exists():
        raise DjayWriteError(
            f"Libreria djay Pro non trovata in {library_path}. "
            "Questa funzione è verificata solo per l'installazione Mac App Store."
        )
    return db_file


def preview_write(
    filepath: Path, new_cues: list[CuePoint], library_path: Path = DEFAULT_LIBRARY_PATH,
) -> WriteResult:
    """Calcola cosa verrebbe scritto, verifica il round-trip, ma NON scrive nulla."""
    db_file = _require_library(library_path)
    uuid = find_track_uuid(db_file, filepath)

    conn = _open_readonly(db_file)
    try:
        row = conn.execute(
            "SELECT data FROM database2 WHERE collection='mediaItemUserData' AND key=?",
            (uuid,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise DjayWriteError("Riga mediaItemUserData non trovata per questa traccia.")

    original = row[0]
    updated = append_cues(original, new_cues)
    verify_roundtrip(original, updated, new_cues)

    return WriteResult(
        ok=True, message="Anteprima verificata: pronta per la scrittura.",
        track_uuid=uuid,
        cues_before=read_cue_points(original),
        cues_after=read_cue_points(updated),
    )


def write_new_cues(
    filepath: Path, new_cues: list[CuePoint],
    library_path: Path = DEFAULT_LIBRARY_PATH,
    allow_while_running: bool = False,
) -> WriteResult:
    """Scrive davvero i nuovi cue nel database live di djay Pro.

    Sempre, in quest'ordine: verifica che djay Pro non sia in esecuzione
    (a meno di override esplicito), backup del bundle intero, calcolo del
    nuovo blob, verifica di round-trip, e SOLO SE tutto torna, UPDATE della
    riga nel database live.
    """
    if not allow_while_running and is_djay_running():
        raise DjayWriteError(
            "djay Pro risulta in esecuzione: chiudilo prima di scrivere, per "
            "evitare conflitti con l'auto-salvataggio dell'app."
        )

    db_file = _require_library(library_path)
    backup_path = backup_library(library_path)
    uuid = find_track_uuid(db_file, filepath)

    conn = _open_readonly(db_file)
    try:
        row = conn.execute(
            "SELECT data FROM database2 WHERE collection='mediaItemUserData' AND key=?",
            (uuid,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise DjayWriteError("Riga mediaItemUserData non trovata per questa traccia.")

    original = row[0]
    updated = append_cues(original, new_cues)
    verify_roundtrip(original, updated, new_cues)  # si ferma qui se qualcosa non torna

    write_conn = sqlite3.connect(str(db_file))
    try:
        write_conn.execute(
            "UPDATE database2 SET data=? WHERE collection='mediaItemUserData' AND key=?",
            (updated, uuid),
        )
        write_conn.commit()
        write_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        write_conn.close()

    return WriteResult(
        ok=True, message=f"Scritti {len(new_cues)} nuovi cue point.",
        track_uuid=uuid, backup_path=backup_path,
        cues_before=read_cue_points(original),
        cues_after=read_cue_points(updated),
    )
