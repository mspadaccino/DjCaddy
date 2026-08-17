"""Scansione di una cartella: cosa c'è dentro, per tipo di file.

Deriva da `folder_analysis.py` del progetto Essentia-to-Metadata, ma separa
la scansione (qui) dalla ricerca dei duplicati (in `duplicates.py`), che è
un'operazione molto più costosa e va lanciata a parte.
"""

from __future__ import annotations

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


def scan_folder(root: Path, audio_only: bool = False, progress=None) -> FolderScan:
    """Elenca ricorsivamente i file sotto `root`.

    `audio_only` tiene solo i formati audio riconosciuti. `progress`, se
    passato, viene chiamato ogni tanto con il numero di file visti finora:
    su una libreria da decine di migliaia di file la scansione non è
    istantanea e in una UI serve poterlo dire.
    """
    scan = FolderScan(root=root)
    seen = 0
    for path in root.rglob("*"):
        try:
            if not path.is_file():
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
