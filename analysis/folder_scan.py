"""Scansione di una cartella: cosa c'è dentro, per tipo di file.

Deriva da `folder_analysis.py` del progetto Essentia-to-Metadata, ma separa
la scansione (qui) dalla ricerca dei duplicati (in `duplicates.py`), che è
un'operazione molto più costosa e va lanciata a parte.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# Estensioni audio riconosciute -> etichetta di formato mostrata nei conteggi.
AUDIO_FORMATS = {
    ".mp3": "MP3",
    ".flac": "FLAC",
    ".wav": "WAV",
    ".aiff": "AIFF",
    ".aif": "AIFF",
    ".m4a": "M4A",
    ".mp4": "M4A",
    ".m4b": "M4A",
    ".aac": "AAC",
    ".ogg": "OGG",
    ".oga": "OGG",
    ".opus": "OPUS",
    ".alac": "ALAC",
    ".wma": "WMA",
    ".wv": "WAVPACK",
    ".ape": "APE",
    ".mpc": "MUSEPACK",
    ".dsf": "DSD",
}

OTHER = "OTHER"
NO_EXTENSION = "(senza estensione)"
# Su volumi non-macOS (exFAT, FAT32, NTFS) il Finder non può scrivere gli
# attributi estesi dentro al file e li mette in un file affiancato "._<nome>".
# Portano l'estensione del brano ma sono AppleDouble da 4 KB: non sono audio,
# e vanno tenuti fuori sia dai conteggi audio sia dalla ricerca duplicati —
# altrimenti, avendo tutti la stessa dimensione, si presenterebbero a
# migliaia come "duplicati". Su una libreria reale sono 88.115 file su
# 116.381, cioè i tre quarti di quello che una scansione ingenua conta.
APPLEDOUBLE = "AppleDouble (._ macOS)"

# Cartella in cui finiscono i duplicati messi da parte. Va SALTATA dalle
# scansioni successive: sta dentro la libreria, e senza escluderla la seconda
# analisi ritrova quello che la prima aveva gia' spostato.
QUARANTINE_DIRNAME = "_DUPLICATES_TO_DELETE"

# I primi quattro byte di un file AppleDouble. Serve a NON cancellare per
# sbaglio un file che si chiama "._qualcosa" senza esserlo davvero: il nome
# da solo non basta a giustificare una cancellazione.
APPLEDOUBLE_MAGIC = b"\x00\x05\x16\x07"


@dataclass
class ScannedFile:
    path: Path
    size: int
    fmt: str                      # etichetta di formato, oppure OTHER
    # Quando il file e' stato scritto l'ultima volta. Costa nulla — la
    # `stat()` che da' la dimensione la porta gia' con se' — e per decidere
    # se un file di contorno serve ancora e' spesso l'unico indizio che c'e':
    # una copertina scaricata anni fa insieme al brano e una salvata ieri si
    # somigliano in tutto tranne che in questo.
    mtime: float = 0.0

    @property
    def is_audio(self) -> bool:
        return self.fmt not in (OTHER, APPLEDOUBLE)


@dataclass
class FolderScan:
    """Esito della scansione. `unreadable` tiene i file che il filesystem non
    ha lasciato leggere: vanno mostrati, non nascosti, perché sono anche i
    file che la ricerca duplicati non potrà valutare."""
    root: Path
    files: list[ScannedFile] = field(default_factory=list)
    unreadable: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def audio(self) -> list[ScannedFile]:
        return [f for f in self.files if f.is_audio]

    def counts_by_format(self) -> Counter:
        return Counter(f.fmt for f in self.files)

    def counts_by_extension(self) -> Counter:
        return Counter(extension_of(f.path) for f in self.files)

    def size_by_extension(self) -> Counter:
        totals = Counter()
        for f in self.files:
            totals[extension_of(f.path)] += f.size
        return totals

    def files_with_extension(self, ext: str) -> list[ScannedFile]:
        """I file di una sola estensione, dal piu' grosso al piu' piccolo.

        L'ordine e' quello: da qui si decide cosa cancellare per fare posto,
        e la domanda e' sempre "chi occupa di piu'".
        """
        return sorted((f for f in self.files if extension_of(f.path) == ext),
                      key=lambda f: f.size, reverse=True)

    def size_by_format(self) -> Counter:
        totals = Counter()
        for f in self.files:
            totals[f.fmt] += f.size
        return totals

    def total_size(self) -> int:
        return sum(f.size for f in self.files)


def is_metadata_sidecar(path: Path) -> bool:
    """File "._<nome>" scritto da macOS accanto al brano su volumi non-macOS.

    Non contiene audio: solo Finder Info e resource fork. Verificato su un
    file reale — 4096 byte in tutto, di cui 259 non nulli, e nessuno di essi
    è audio. Non c'è modo di analizzarlo perché non c'è niente da analizzare;
    l'unica cosa sensata è non guardarlo proprio.
    """
    return path.name.startswith("._")


def extension_of(path: Path) -> str:
    """L'estensione con cui il file viene contato ed elencato.

    Sempre minuscola: ".JPG" e ".jpg" sono la stessa cosa, e contarle a parte
    spezzerebbe in due mucchi quello che l'utente vede come uno solo.
    """
    return path.suffix.lower() or NO_EXTENSION


def format_of(path: Path) -> str:
    if is_metadata_sidecar(path):
        return APPLEDOUBLE
    return AUDIO_FORMATS.get(path.suffix.lower(), OTHER)


def scan_folder(root: Path, audio_only: bool = False, progress=None,
                skip_dirs=(QUARANTINE_DIRNAME,)) -> FolderScan:
    """Elenca ricorsivamente i file sotto `root`.

    `skip_dirs` esclude intere cartelle: di default la quarantena, che
    altrimenti verrebbe rianalizzata a ogni giro.

    `audio_only` tiene solo i formati audio riconosciuti. `progress`, se
    passato, viene chiamato ogni tanto con il numero di file visti finora:
    su una libreria da decine di migliaia di file la scansione non è
    istantanea e in una UI serve poterlo dire.
    """
    scan = FolderScan(root=root)
    skip = set(skip_dirs)
    seen = 0
    for path in root.rglob("*"):
        try:
            if not path.is_file():
                continue
            if skip & set(path.relative_to(root).parts[:-1]):
                continue
            fmt = format_of(path)
            if audio_only and fmt in (OTHER, APPLEDOUBLE):
                continue
            info = path.stat()
            scan.files.append(ScannedFile(path=path, size=info.st_size, fmt=fmt,
                                          mtime=info.st_mtime))
        except OSError as e:
            scan.unreadable.append((path, str(e)))
        seen += 1
        if progress is not None and seen % 500 == 0:
            progress(seen)
    if progress is not None:
        progress(seen)
    return scan


def human_size(num_bytes: float) -> str:
    """Dimensione leggibile. Usa multipli di 1024 (come il Finder di macOS
    non fa, ma come fa `du`): l'importante è che sia coerente ovunque."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024 or unit == "TB":
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


# --------------------------------------------------------------------------
# Sidecar AppleDouble: individuazione e rimozione
# --------------------------------------------------------------------------

@dataclass
class SidecarReport:
    """Cosa è stato trovato cercando i "._<nome>".

    `confirmed` sono quelli che hanno ANCHE il contenuto giusto e si possono
    togliere senza pensarci. `unverified` si chiamano così ma dentro c'è
    altro: restano dove sono e vengono solo segnalati, perché cancellare in
    base al nome è esattamente il modo in cui si perde un file per sbaglio.
    """
    confirmed: list[Path] = field(default_factory=list)
    unverified: list[Path] = field(default_factory=list)
    freed_bytes: int = 0
    unreadable: list[tuple[Path, str]] = field(default_factory=list)
    # Quante voci sono state ATTRAVERSATE in tutto, sidecar o no. Serve a
    # distinguere "ho guardato e non ce n'erano" da "non ho potuto guardare":
    # `rglob` inghiotte gli errori di percorso, quindi una cartella
    # illeggibile — un volume USB che nega l'accesso, come è successo qui —
    # produce un rapporto vuoto identico a quello di una cartella pulita.
    walked: int = 0
    root_error: str | None = None

    @property
    def looked_properly(self) -> bool:
        return self.root_error is None


def is_appledouble_content(path: Path) -> bool:
    """Legge i primi byte per confermare che è davvero un AppleDouble."""
    try:
        with path.open("rb") as fh:
            return fh.read(4) == APPLEDOUBLE_MAGIC
    except OSError:
        return False


def find_sidecars(root: Path, progress=None) -> SidecarReport:
    """Cerca i sidecar "._<nome>" sotto `root`, verificandone il contenuto.

    Comprende anche quelli delle CARTELLE (senza estensione): sono la stessa
    cosa e la stessa spazzatura.
    """
    report = SidecarReport()
    try:
        next(iter(os.scandir(root)), None)
    except OSError as e:
        report.root_error = str(e)
        return report

    seen = 0
    for path in root.rglob("*"):
        report.walked += 1
        if not path.name.startswith("._"):
            continue
        try:
            if not path.is_file():
                continue
            if is_appledouble_content(path):
                report.confirmed.append(path)
                report.freed_bytes += path.stat().st_size
            else:
                report.unverified.append(path)
        except OSError as e:
            report.unreadable.append((path, str(e)))
        seen += 1
        if progress is not None and seen % 500 == 0:
            progress(seen)
    if progress is not None:
        progress(seen)
    return report


def delete_sidecars(paths, dry_run: bool = True) -> tuple[int, int, list[tuple[Path, str]]]:
    """Cancella i sidecar. Ritorna (quanti, byte liberati, errori).

    A differenza dei duplicati qui si cancella davvero invece di mettere in
    quarantena, e la ragione è che questi file non contengono niente da
    salvare: solo l'etichetta di quarantena del browser che ha scaricato il
    brano. Metterli da parte vorrebbe dire spostare spazzatura, e su un
    volume non-macOS il sistema li ricrea comunque alla prossima occasione.

    Per sicurezza il contenuto viene ricontrollato subito prima: se nel
    frattempo il file non è più un AppleDouble, viene saltato.
    """
    removed, freed, errors = 0, 0, []
    for path in paths:
        try:
            if not is_appledouble_content(path):
                errors.append((path, "non è (più) un AppleDouble: saltato"))
                continue
            size = path.stat().st_size
            if not dry_run:
                path.unlink()
            removed += 1
            freed += size
        except OSError as e:
            errors.append((path, str(e)))
    return removed, freed, errors


# --------------------------------------------------------------------------
# Cancellazione di quello che musica non e'
# --------------------------------------------------------------------------

# Estensioni che non contengono audio ma NON sono spazzatura: buttarle fa
# danno, e il danno non si vede subito. Il valore dice perche', perche' "non
# cancellare" senza una ragione e' un avviso che si impara a ignorare.
NOT_JUNK = {
    ".xml": "a rekordbox library export looks exactly like this",
    ".nml": "a Traktor collection, cue points and loops included",
    ".m3u": "a playlist", ".m3u8": "a playlist", ".pls": "a playlist",
    ".cue": "a cue sheet says where to cut a continuous mix",
    ".db": "an application's database (djay, rekordbox)",
    ".edb": "a rekordbox database", ".pdb": "a rekordbox database",
    ".json": "the analyses this very app writes are .json files",
    ".asd": "Ableton's analysis: it comes back by itself, but slowly",
}


def delete_files(paths, dry_run: bool = True) -> tuple[int, int, list[tuple[Path, str]]]:
    """Cancella file che non sono audio. Ritorna (quanti, byte, errori).

    Il rifiuto di toccare un file audio sta QUI e non nella pagina che
    chiama, ed e' voluto: la pagina decide cosa mostrare, ma la garanzia che
    un brano non venga cancellato per sbaglio non puo' dipendere da quale
    elenco l'ha proposto. Il formato viene riletto dal percorso subito prima
    di cancellare, come i sidecar rileggono il proprio contenuto.

    Si cancella davvero invece di mettere in quarantena — come per i sidecar
    e per lo stesso motivo: la quarantena sta sullo stesso volume, quindi
    spostarci una copertina da 2 MB non libera i 2 MB. Chi vuole ripensarci
    ha la lista sotto gli occhi prima di confermare.
    """
    removed, freed, errors = 0, 0, []
    for item in paths:
        path = Path(getattr(item, "path", item))
        try:
            if format_of(path) not in (OTHER, APPLEDOUBLE):
                errors.append((path, "e' un file audio: saltato"))
                continue
            size = path.stat().st_size
            if not dry_run:
                path.unlink()
            removed += 1
            freed += size
        except OSError as e:
            errors.append((path, str(e)))
    return removed, freed, errors


# --------------------------------------------------------------------------
# File audio illeggibili
# --------------------------------------------------------------------------

@dataclass
class BadFile:
    path: Path
    size: int
    reason: str


@dataclass
class IntegrityReport:
    """Esito del controllo di leggibilità sui file audio.

    `bad` sono quelli che nessun lettore riuscirà ad aprire; `missing` quelli
    spariti dal percorso mentre li si controllava. Restano separati perché
    richiedono due rimedi diversi: riscaricare, oppure niente.
    """
    checked: int = 0
    bad: list[BadFile] = field(default_factory=list)
    missing: list[Path] = field(default_factory=list)


# Che demuxer imporre a ffmpeg quando il riconoscimento automatico sbaglia,
# in base all'estensione. Serve solo come SECONDO tentativo: al primo giro si
# lascia decidere a ffmpeg, che di norma ci azzecca.
_DEMUXERS = {
    ".mp3": "mp3", ".mp2": "mp3",
    ".flac": "flac",
    ".wav": "wav",
    ".aif": "aiff", ".aiff": "aiff",
    ".m4a": "mp4", ".m4b": "mp4", ".mp4": "mp4", ".aac": "aac",
    ".ogg": "ogg", ".oga": "ogg", ".opus": "ogg",
    ".wma": "asf",
    ".ape": "ape", ".wv": "wv", ".mpc": "mpc",
}


def _probe(path: Path, demuxer: str | None, timeout: float) -> tuple[bool, str]:
    """Un giro di ffprobe. Torna (ha_funzionato, ultima riga d'errore)."""
    forced = ["-f", demuxer] if demuxer else []
    out = subprocess.run(
        ["ffprobe", "-v", "error", *forced, "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=timeout,
    )
    if out.returncode == 0 and out.stdout.strip():
        return True, ""
    lines = (out.stderr or "").strip().splitlines()
    return False, lines[-1].split(": ")[-1] if lines else "il decoder rifiuta il file"


def _decoder_error(path: Path, timeout: float = 60.0) -> str | None:
    """Chiede a ffprobe se il file si apre. None se sì, altrimenti il motivo.

    Se il riconoscimento automatico fallisce si RIPROVA imponendo il demuxer
    che l'estensione suggerisce, e solo un secondo rifiuto conta. Non è uno
    scrupolo teorico: nella libreria reale i pool per DJ (Mixx-It, DMC,
    Music Factory) distribuiscono MP3 incapsulati in un contenitore WAV, cioè
    tag ID3, poi header RIFF/WAVE, poi i frame MPEG. ffmpeg salta l'ID3 per
    riconoscere il formato, vede RIFF e sceglie il demuxer wav; quello però
    rilegge dall'offset 0, trova "ID3" e si ferma. Sono file sanissimi —
    misurati 79 su 92 respinti, tutti con durata giusta e audio che decodifica
    senza un errore — e macOS e rekordbox li riproducono senza storie. Al
    primo giro finivano in quarantena come illeggibili.
    """
    demuxer = _DEMUXERS.get(path.suffix.lower())
    try:
        ok, reason = _probe(path, None, timeout)
        if not ok and demuxer:
            ok, reason = _probe(path, demuxer, timeout)
    except FileNotFoundError:
        return None            # ffprobe non c'è: meglio tacere che accusare
    except subprocess.TimeoutExpired:
        return "il decoder non risponde"
    except OSError as e:
        return f"decoder non eseguibile: {e}"
    return None if ok else reason


def _as_path(item) -> Path:
    """Percorso di un elemento, che sia un ScannedFile, un Path o una stringa.

    Deliberatamente per attributo e non con `isinstance`: gli oggetti tenuti
    nella sessione di Streamlit sono stati creati da un'istanza PRECEDENTE di
    questo modulo, e dopo un ricaricamento `isinstance` contro la classe
    appena importata risponde False anche se l'oggetto è quello giusto.
    """
    return Path(getattr(item, "path", item))


def check_readable(path: Path, deep: bool = False) -> str | None:
    """None se il file sembra un audio valido, altrimenti il motivo.

    Senza `deep` si ferma all'intestazione: costa nulla e prende i casi
    grossolani (file vuoto, pagina di errore HTML salvata come .mp3,
    download troncato). NON prende però il caso peggiore, quello di un file
    con intestazione perfetta e stream rovinato: misurato su un brano reale
    della libreria, mutagen ne legge tranquillamente 219 secondi di durata
    mentre il decoder si rifiuta di aprirlo.

    Con `deep` la domanda la si gira a ffprobe, che è lo stesso decoder che
    userà il tagging. Definitivo, ma va pagato a file.
    """
    try:
        size = path.stat().st_size
    except OSError as e:
        return f"non leggibile: {e}"
    if size == 0:
        return "file vuoto"

    header = _header_error(path)
    if not deep or not _has_ffprobe():
        return header

    # In profondità l'ultima parola è del DECODER, non di mutagen: mutagen
    # legge i tag, e un file che non gli torna può benissimo suonare. Misurati
    # 3 file su 36 respinti da mutagen che ffprobe apre senza problemi.
    decoder = _decoder_error(path)
    if decoder is None:
        return None
    return header or decoder


@lru_cache(maxsize=1)
def _has_ffprobe() -> bool:
    return shutil.which("ffprobe") is not None


def _header_error(path: Path) -> str | None:
    """I controlli a costo zero sull'intestazione, via mutagen."""
    import mutagen

    try:
        audio = mutagen.File(path)
    except Exception as e:                              # noqa: BLE001
        return f"intestazione illeggibile: {type(e).__name__}"
    if audio is None:
        return "nessuna intestazione audio riconosciuta"
    info = getattr(audio, "info", None)
    if info is None:
        return "nessuna informazione di stream"
    length = getattr(info, "length", None)
    if length is not None and length <= 0:
        return "durata nulla"
    return None


# Quanti ffprobe tenere in volo insieme. Thread e non processi: il lavoro lo
# fa un sottoprocesso esterno, quindi il GIL non è in mezzo e ogni thread non
# fa altro che aspettare.
CHECK_THREADS = 8


def check_integrity(files, progress=None, workers: int = CHECK_THREADS) -> IntegrityReport:
    """Quali file non si aprono davvero. Interpella SEMPRE il decoder.

    Non c'è più una modalità "solo intestazioni": era veloce e sbagliata.
    Mutagen legge i tag, e su questa libreria ne condannava 31 che il decoder
    apre senza fiatare — MP4 battezzati ".mp3", audio che comincia oltre la
    finestra in cui mutagen lo cerca, contenitori che non conosce. Dato che
    da questo elenco si decide cosa mettere in quarantena, un "forse" non è
    un risultato utilizzabile. La velocità la si recupera in parallelo.
    """
    report = IntegrityReport()
    total = len(files)
    paths = [_as_path(item) for item in files]

    def esamina(path: Path):
        if not path.exists():
            return path, None, True
        reason = check_readable(path, deep=True)
        if reason is None:
            return path, None, False
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        return path, BadFile(path=path, size=size, reason=reason), False

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, (path, bad, missing) in enumerate(pool.map(esamina, paths), 1):
            if missing:
                report.missing.append(path)
            elif bad is not None:
                report.bad.append(bad)
            report.checked += 1
            if progress is not None and (i % 50 == 0 or i == total):
                progress(i, total)
    return report


# --------------------------------------------------------------------------
# Durate: per stanare i medley
# --------------------------------------------------------------------------

@dataclass
class TrackDuration:
    path: Path
    size: int
    seconds: float


@dataclass
class DurationReport:
    """Durate lette dalle intestazioni.

    Si legge UNA volta e si filtra poi in memoria: leggere l'intestazione
    costa circa 8 ms a file (misurato), che su una libreria da 90.000 brani
    sono una dozzina di minuti — improponibili a ogni movimento di uno
    slider, istantanei se fatti una volta sola.
    """
    tracks: list[TrackDuration] = field(default_factory=list)
    unknown: list[Path] = field(default_factory=list)

    def longer_than(self, minutes: float) -> list[TrackDuration]:
        cutoff = minutes * 60
        return sorted((t for t in self.tracks if t.seconds > cutoff),
                      key=lambda t: t.seconds, reverse=True)

    def between(self, low_minutes: float, high_minutes: float) -> list[TrackDuration]:
        """I brani in una fascia di durata, estremo alto compreso.

        Il basso e' escluso e l'alto no: cosi' "da 10 a 30" non ripesca chi
        dura esattamente 10 minuti (che il filtro "piu' lungo di 10" gia'
        lasciava fuori) ma tiene chi ne dura esattamente 30, che altrimenti
        sfuggirebbe a qualunque fascia scelta.
        """
        low, high = low_minutes * 60, high_minutes * 60
        return sorted((t for t in self.tracks if low < t.seconds <= high),
                      key=lambda t: t.seconds, reverse=True)

    @property
    def longest_minutes(self) -> float:
        return max((t.seconds for t in self.tracks), default=0.0) / 60


def read_durations(files, progress=None) -> DurationReport:
    """Durata di ogni file audio, dall'intestazione.

    I file di cui non si riesce a stabilire la durata finiscono in `unknown`
    invece di essere considerati lunghi zero: uno zero inventato li farebbe
    sparire da qualunque filtro "più lungo di", che è il verso sbagliato in
    cui sbagliare quando poi si propone di spostarli.
    """
    import mutagen

    report = DurationReport()
    total = len(files)
    for i, item in enumerate(files, 1):
        path = _as_path(item)
        try:
            audio = mutagen.File(path)
            seconds = getattr(getattr(audio, "info", None), "length", None)
        except Exception:
            seconds = None
        if seconds is None or seconds <= 0:
            report.unknown.append(path)
        else:
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            report.tracks.append(
                TrackDuration(path=path, size=size, seconds=float(seconds)))
        if progress is not None and (i % 200 == 0 or i == total):
            progress(i, total)
    return report


def human_duration(seconds: float) -> str:
    """Durata come HH:MM:SS, ore comprese anche quando sono zero.

    Le ore ci sono SEMPRE, e con due cifre, perché la tabella tratta questa
    colonna come testo e la ordina alfabeticamente: scrivendo "15:00" per
    quindici minuti e "1:10:57" per un'ora e dieci, il primo finirebbe dopo
    il secondo. Con "00:15:00" e "01:10:57" l'ordine alfabetico coincide con
    quello cronologico. Le due cifre servono al caso limite di un file
    rovinato che dichiara una durata assurda: senza, "10:00:00" verrebbe
    prima di "09:00:00".
    """
    seconds = int(round(max(0.0, seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
