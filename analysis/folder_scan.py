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
