"""Scrittura di hot cue, memory cue e loop nella libreria di rekordbox 6/7.

Il gemello di `djay_write` per il software che ha preso il posto di djay
Pro. La differenza sta tutta in COME si arriva al database: quello di djay
è uno SQLite in chiaro con dentro un blob da decodificare a mano (tutto
`djay_write` è quel lavoro), mentre `master.db` di rekordbox è cifrato con
SQLCipher ma dentro ha tabelle normali. La chiave e le bindings le porta
`pyrekordbox`, quindi qui non c'è nessun formato da reverse-engineerare:
si scrivono righe in `djmdCue`, che è un dato leggibile e verificabile.

Cosa fa pyrekordbox al posto nostro, ed è la ragione per cui ci si appoggia:
- apre il database cifrato senza chiedere chiavi all'utente;
- RIFIUTA il commit se rekordbox è in esecuzione (`get_rekordbox_pid`);
- tiene la contabilità degli USN, i numeri di sequenza con cui rekordbox
  riconosce le modifiche per la sincronizzazione cloud. Scriverli a mano
  sarebbe il modo più veloce per confondere una libreria vera.

Quello che resta a noi è il significato dei campi, misurato sui record veri
della libreria (87k brani, 315k cue letti):
- `Kind` è il pad: 0 = memory cue, 1..8 = hot cue A..H;
- `OutMsec` = -1 vuol dire "non è un loop"; un loop ha l'uscita vera;
- `InFrame` è l'istante in frame da 1/150 di secondo, TRONCATO e non
  arrotondato — misurato sui cue di rekordbox stesso: 1051 ms → 157 (non
  158), 147491 → 22123 (non 22124), e sui loop 19384 → 2907 (non 2908);
- `Comment` è l'etichetta che si legge nella forma d'onda.

Un backup del database si fa comunque prima di scrivere, ma UNO al giorno:
`master.db` qui pesa quasi un giga, e una copia per ogni brano riempirebbe
il disco in una serata.
"""

from __future__ import annotations

import shutil
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .cue_export import RB_HOT_CUES, RekordboxMarker

# Il frame di rekordbox: 1/150 di secondo (vedi il docstring del modulo).
FRAMES_PER_SECOND = 150

# Quanti backup nostri tenere: rekordbox ha già i suoi (master.backup*.db),
# questi servono a tornare indietro da una scrittura NOSTRA.
BACKUPS_KEPT = 2

MEMORY_CUE = 0


class RekordboxWriteError(Exception):
    """Errore che deve fermare l'operazione senza scrivere nulla."""


@dataclass
class RekordboxResult:
    """Cosa c'era e cosa ci sarebbe (o c'è) dopo la scrittura."""
    ok: bool
    message: str
    track_id: str | None = None
    title: str = ""
    backup_path: Path | None = None
    cues_before: int = 0
    loops_before: int = 0
    written: int = 0
    removed: int = 0
    pads_taken: list[int] = field(default_factory=list)
    hot_cues: int = 0
    memory_cues: int = 0
    loops: int = 0


def available() -> tuple[bool, str]:
    """Se si può scrivere in rekordbox da qui, e altrimenti perché no.

    Come `vocals.available()`: la pagina lo chiede e lo dice, invece di
    esplodere su un import mancante. `poetry install --without rekordbox` è
    una scelta legittima.
    """
    try:
        import pyrekordbox  # noqa: F401
    except ImportError as trouble:
        return False, (f"pyrekordbox non è installato ({trouble}): "
                       "`poetry install --with rekordbox`.")
    path = database_path()
    if path is None:
        return False, ("Nessuna libreria rekordbox trovata su questo "
                       "computer.")
    if not path.is_file():
        return False, f"La libreria rekordbox non è dov'era attesa: {path}"
    return True, str(path)


def database_path() -> Path | None:
    """Il `master.db` di questo computer, come lo trova rekordbox stesso."""
    try:
        from pyrekordbox.config import get_config
    except ImportError:
        return None
    for version in ("rekordbox7", "rekordbox6", "rekordbox5"):
        try:
            found = get_config(version, "db_path")
        except Exception:
            continue
        if found:
            return Path(found)
    return None


def is_rekordbox_running() -> bool:
    """Se rekordbox è aperto adesso: scrivergli sotto i piedi significa
    farsi sovrascrivere al suo primo salvataggio automatico."""
    try:
        from pyrekordbox.utils import get_rekordbox_pid
    except ImportError:
        return False
    return bool(get_rekordbox_pid())


def backup_database(db_path: Path, when: date | None = None) -> Path:
    """Copia `master.db` accanto all'originale, una volta al giorno.

    Il nome porta la data: se la copia di oggi c'è già la si tiene, perché
    la prima copia del giorno è quella "prima che DjCaddy toccasse niente"
    — ed è quella che si vuole per tornare indietro. Delle più vecchie
    restano le ultime `BACKUPS_KEPT`.
    """
    stamp = (when or date.today()).isoformat()
    backup = db_path.with_name(f"{db_path.name}.djcaddy-{stamp}.bak")
    if not backup.exists():
        shutil.copy2(db_path, backup)
    old = sorted(db_path.parent.glob(f"{db_path.name}.djcaddy-*.bak"))
    for stale in old[:-BACKUPS_KEPT]:
        stale.unlink(missing_ok=True)
    return backup


def _composed(text: str) -> str:
    """Accenti in un carattere solo: il percorso che macOS scrive sul disco
    e quello registrato da rekordbox possono differire solo per questo."""
    return unicodedata.normalize("NFC", text)


def cue_row_values(marker: RekordboxMarker) -> dict:
    """I campi di `djmdCue` per un marcatore — la parte PURA, senza database.

    Tenuta a parte perché è l'unico pezzo in cui si decide qualcosa: il
    resto è aprire, cercare e consegnare a pyrekordbox.
    """
    start_ms = max(0, round(marker.start * 1000))
    values = {
        "InMsec": start_ms,
        "InFrame": start_ms * FRAMES_PER_SECOND // 1000,
        "InMpegFrame": 0,
        "InMpegAbs": 0,
        "Kind": marker.pad if marker.pad else MEMORY_CUE,
        "Color": -1,
        "ColorTableIndex": 0,
        "ActiveLoop": 0,
        "Comment": (marker.label or "")[:255],
        "BeatLoopSize": 0,
        "CueMicrosec": 0,
        "InPointSeekInfo": "",
        "OutPointSeekInfo": "",
        "rb_local_deleted": 0,
    }
    if marker.end is None:
        values.update({"OutMsec": -1, "OutFrame": 0})
    else:
        end_ms = max(start_ms, round(marker.end * 1000))
        values.update({
            "OutMsec": end_ms,
            "OutFrame": end_ms * FRAMES_PER_SECOND // 1000,
        })
    values.update({"OutMpegFrame": 0, "OutMpegAbs": 0})
    return values


def fit_to_free_pads(markers: list[RekordboxMarker],
                     taken: set[int]) -> list[RekordboxMarker]:
    """Riassegna i pad evitando quelli che rekordbox ha già occupati.

    Serve solo quando si AGGIUNGE: due cue sullo stesso Kind sono due righe
    valide per il database ma un pad solo sul controller, e quale delle due
    si veda non lo decide nessuno. Chi non trova un pad libero diventa un
    memory cue — non si perde niente, ed è il motivo per cui in rekordbox
    non serve scartare righe come in djay.

    L'ordine dei pad segue il tempo: il primo marcatore prende il primo pad
    libero, non "il suo".
    """
    free = [pad for pad in range(1, RB_HOT_CUES + 1) if pad not in taken]
    out = []
    for marker in sorted(markers, key=lambda m: m.start):
        if marker.pad is None:
            out.append(marker)
            continue
        pad = free.pop(0) if free else None
        out.append(replace_pad(marker, pad))
    return out


def replace_pad(marker: RekordboxMarker, pad: int | None) -> RekordboxMarker:
    """Lo stesso marcatore su un altro pad (o su nessuno)."""
    return RekordboxMarker(marker.row_id, marker.start, marker.end,
                           marker.label, pad)


def check_markers(markers: list[RekordboxMarker]) -> None:
    """Si ferma PRIMA di aprire il database se il piano non sta in piedi."""
    if not markers:
        raise RekordboxWriteError("Nessun marcatore da scrivere.")
    pads = [m.pad for m in markers if m.pad is not None]
    if any(not 1 <= pad <= RB_HOT_CUES for pad in pads):
        raise RekordboxWriteError(
            f"Pad fuori intervallo: rekordbox ne ha {RB_HOT_CUES} (A..H).")
    if len(pads) != len(set(pads)):
        raise RekordboxWriteError("Due marcatori sullo stesso pad.")
    for m in markers:
        if m.start < 0:
            raise RekordboxWriteError("Un marcatore prima dell'inizio del brano.")
        if m.end is not None and m.end <= m.start:
            raise RekordboxWriteError("Un loop che finisce prima di cominciare.")


def _open(db_path: Path | None):
    from pyrekordbox import Rekordbox6Database

    path = db_path or database_path()
    if path is None or not Path(path).is_file():
        raise RekordboxWriteError(
            "Nessuna libreria rekordbox trovata su questo computer.")
    try:
        return Rekordbox6Database(path=str(path), unlock=True)
    except Exception as trouble:                      # chiave, versione, permessi
        raise RekordboxWriteError(
            f"La libreria rekordbox non si apre: {trouble}") from trouble


def _find_track(db, filepath: Path):
    """La riga di `djmdContent` del brano, cercata per percorso assoluto.

    `FolderPath` porta il percorso INTERO del file, nome compreso (il nome
    inganna). Il ripiego sul nome file non c'è apposta: due cartelle possono
    contenere lo stesso nome, e scrivere i cue sul brano sbagliato è peggio
    che dire "non l'ho trovato".
    """
    from sqlalchemy import select
    from pyrekordbox.db6.tables import DjmdContent

    wanted = _composed(str(filepath))
    row = db.session.execute(
        select(DjmdContent).where(DjmdContent.FolderPath == str(filepath))
    ).scalars().first()
    if row is not None:
        return row
    for candidate in db.session.execute(
            select(DjmdContent).where(
                DjmdContent.FileNameL == filepath.name)).scalars():
        if _composed(candidate.FolderPath or "") == wanted:
            return candidate
    return None


def _existing_cues(db, track_id: str):
    from sqlalchemy import select
    from pyrekordbox.db6.tables import DjmdCue

    return db.session.execute(
        select(DjmdCue).where(DjmdCue.ContentID == track_id,
                              DjmdCue.rb_local_deleted == 0)
    ).scalars().all()


def _fitted(existing, markers, replace: bool) -> list[RekordboxMarker]:
    """I marcatori come verranno scritti davvero: con `replace` i pad sono
    tutti liberi perché quello che c'era se ne va."""
    if replace:
        return sorted(markers, key=lambda m: m.start)
    return fit_to_free_pads(markers, {c.Kind for c in existing if c.Kind})


def _summary(track, existing, fitted, replace: bool) -> RekordboxResult:
    loops = [c for c in existing if (c.OutMsec or -1) != -1]
    return RekordboxResult(
        ok=True,
        message="Anteprima verificata: pronta per la scrittura.",
        track_id=track.ID,
        title=track.Title or (track.FileNameL or ""),
        cues_before=len(existing) - len(loops),
        loops_before=len(loops),
        written=len(fitted),
        removed=len(existing) if replace else 0,
        pads_taken=[] if replace
        else sorted({c.Kind for c in existing if c.Kind}),
        hot_cues=sum(1 for m in fitted if m.pad),
        memory_cues=sum(1 for m in fitted if m.pad is None and m.end is None),
        loops=sum(1 for m in fitted if m.end is not None),
    )


def preview_write(filepath: Path, markers: list[RekordboxMarker],
                  replace: bool = False,
                  db_path: Path | None = None) -> RekordboxResult:
    """Cosa cambierebbe, senza cambiare niente: il database non viene scritto.

    Verifica le tre cose che possono andare storte prima di toccare la
    libreria: il piano non sta in piedi, il brano in rekordbox non c'è, il
    database non si apre.
    """
    check_markers(markers)
    db = _open(db_path)
    try:
        track = _find_track(db, filepath)
        if track is None:
            raise RekordboxWriteError(
                "Questo brano non è nella libreria di rekordbox: importalo "
                "lì una volta, poi i cue si possono scrivere.")
        existing = _existing_cues(db, track.ID)
        return _summary(track, existing,
                        _fitted(existing, markers, replace), replace)
    finally:
        db.close()


def write_cues(filepath: Path, markers: list[RekordboxMarker],
               replace: bool = False,
               db_path: Path | None = None) -> RekordboxResult:
    """Scrive davvero i marcatori nella libreria di rekordbox.

    Sempre in quest'ordine: verifica del piano, rekordbox chiuso, backup del
    giorno, ricerca del brano, e solo alla fine le righe nuove. Con
    `replace` i cue già presenti su QUEL brano vengono marcati cancellati
    (`rb_local_deleted`), che è il modo in cui rekordbox stesso li toglie
    senza rompere i riferimenti; senza, i nuovi si aggiungono ai vecchi.

    Il commit lo fa pyrekordbox, che rifiuta di nuovo se rekordbox si è
    aperto nel frattempo e sistema gli USN.
    """
    check_markers(markers)
    if is_rekordbox_running():
        raise RekordboxWriteError(
            "rekordbox risulta in esecuzione: chiudilo prima di scrivere, "
            "per non farti sovrascrivere al suo primo salvataggio.")

    path = db_path or database_path()
    if path is None or not Path(path).is_file():
        raise RekordboxWriteError(
            "Nessuna libreria rekordbox trovata su questo computer.")
    backup = backup_database(Path(path))

    from pyrekordbox.db6.tables import DjmdCue

    db = _open(path)
    try:
        track = _find_track(db, filepath)
        if track is None:
            raise RekordboxWriteError(
                "Questo brano non è nella libreria di rekordbox: importalo "
                "lì una volta, poi i cue si possono scrivere.")
        existing = _existing_cues(db, track.ID)
        fitted = _fitted(existing, markers, replace)
        result = _summary(track, existing, fitted, replace)

        if replace:
            for cue in existing:
                cue.rb_local_deleted = 1
        for marker in fitted:
            db.add(DjmdCue(
                ID=str(db.generate_unused_id(DjmdCue, is_28_bit=True)),
                ContentID=track.ID, ContentUUID=track.UUID,
                UUID=str(uuid.uuid4()), **cue_row_values(marker)))
        db.commit()
    finally:
        db.close()

    result.backup_path = backup
    result.message = (
        f"{result.written} marcatori scritti su «{result.title}» in "
        f"rekordbox."
        + (f" I {result.removed} precedenti sono stati tolti."
           if replace and result.removed else ""))
    return result
