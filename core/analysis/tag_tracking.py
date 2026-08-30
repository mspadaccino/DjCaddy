"""Elenco dei brani già passati dal tagging, per non rianalizzarli.

Formato: un percorso assoluto per riga, UTF-8 — lo stesso di
`.essentia_processed.txt` dello script `tag_music.py`, così il file di
quest'ultimo si può copiare qui sopra tale e quale quando la sua analisi è
finita, e le decine di migliaia di brani già fatti non si rifanno.

Il file vive nella radice del repo (ed è in .gitignore: sono dati, non
codice). Le ripetizioni non danno fastidio — l'originale ne ha circa un
quinto, 113.000 righe per 90.000 percorsi distinti — perché in memoria
diventa un insieme.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Accanto al codice, come faceva lo script originale col suo tracking.
DEFAULT_TRACKING_FILE = Path(__file__).resolve().parent.parent.parent / ".wavecut_tag_progress.txt"


@dataclass
class ProcessedTracker:
    """Chi è già stato taggato. Un solo file, letto all'avvio e aggiornato
    in coda man mano che i brani vengono completati."""
    tracking_file: Path = DEFAULT_TRACKING_FILE
    _done: set[str] = field(default_factory=set, init=False)
    lines_read: int = field(default=0, init=False)
    existed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        try:
            with self.tracking_file.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    entry = line.strip()
                    if entry:
                        self.lines_read += 1
                        self._done.add(entry)
            self.existed = True
        except OSError:
            pass

    def __contains__(self, path) -> bool:
        return str(path) in self._done

    def __len__(self) -> int:
        return len(self._done)

    @property
    def duplicate_lines(self) -> int:
        """Quante righe erano ripetute: utile solo da mostrare, il
        funzionamento non ne risente."""
        return self.lines_read - len(self._done)

    def pending(self, paths) -> list[Path]:
        """Quali dei percorsi dati non sono ancora stati fatti."""
        return [p for p in paths if str(p) not in self._done]

    def mark(self, path) -> None:
        """Segna un brano come fatto, in memoria e in coda al file.
        Ripetere la stessa marcatura non riscrive nulla."""
        entry = str(path)
        if entry in self._done:
            return
        self._done.add(entry)
        try:
            self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
            with self.tracking_file.open("a", encoding="utf-8") as fh:
                fh.write(f"{entry}\n")
        except OSError:
            pass   # perdere una riga di tracking non deve fermare l'analisi
