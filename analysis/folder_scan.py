"""Scansione di una cartella: cosa c'è dentro, per tipo di file.

Deriva da `folder_analysis.py` del progetto Essentia-to-Metadata, ma separa
la scansione (qui) dalla ricerca dei duplicati (in `duplicates.py`), che è
un'operazione molto più costosa e va lanciata a parte.
"""

from __future__ import annotations

import subprocess
from collections import Counter
from dataclasses import dataclass, field
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
        return Counter(f.path.suffix.lower() or "(senza estensione)" for f in self.files)

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
            scan.files.append(ScannedFile(path=path, size=path.stat().st_size, fmt=fmt))
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
    seen = 0
    for path in root.rglob("._*"):
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


def _decoder_error(path: Path, timeout: float = 60.0) -> str | None:
    """Chiede a ffprobe se il file si apre. None se sì, altrimenti il motivo.

    È il controllo che conta davvero, perché ffprobe usa lo stesso decoder
    che userà il tagging: se lo rifiuta qui, lo rifiuterà anche là. Costa
    circa 0,05 s a file su disco locale.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        return None            # ffprobe non c'è: meglio tacere che accusare
    except subprocess.TimeoutExpired:
        return "il decoder non risponde"
    except OSError as e:
        return f"decoder non eseguibile: {e}"
    if out.returncode != 0 or not out.stdout.strip():
        last = (out.stderr or "").strip().splitlines()
        return last[-1].split(": ")[-1] if last else "il decoder rifiuta il file"
    return None


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
    import mutagen

    try:
        size = path.stat().st_size
    except OSError as e:
        return f"non leggibile: {e}"
    if size == 0:
        return "file vuoto"
    try:
        audio = mutagen.File(path)
    except Exception as e:
        return f"intestazione illeggibile: {type(e).__name__}"
    if audio is None:
        return "nessuna intestazione audio riconosciuta"
    info = getattr(audio, "info", None)
    if info is None:
        return "nessuna informazione di stream"
    length = getattr(info, "length", None)
    if length is not None and length <= 0:
        return "durata nulla"
    return _decoder_error(path) if deep else None


def check_integrity(files, deep: bool = False, progress=None) -> IntegrityReport:
    """Controlla la leggibilità di una lista di file audio già scansionati."""
    report = IntegrityReport()
    total = len(files)
    for i, item in enumerate(files, 1):
        path = item.path if isinstance(item, ScannedFile) else Path(item)
        if not path.exists():
            report.missing.append(path)
        else:
            reason = check_readable(path, deep=deep)
            if reason is not None:
                try:
                    size = path.stat().st_size
                except OSError:
                    size = 0
                report.bad.append(BadFile(path=path, size=size, reason=reason))
        report.checked += 1
        if progress is not None and (i % 200 == 0 or i == total):
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
        path = item.path if isinstance(item, ScannedFile) else Path(item)
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
