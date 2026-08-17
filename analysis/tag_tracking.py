"""Elenco dei brani già passati dal tagging, per non rianalizzarli.

Formato: un percorso assoluto per riga, UTF-8 — lo stesso di
`.essentia_processed.txt` dello script `tag_music.py`, così il lavoro già
fatto (90.000 brani e passa) non va buttato.

Wavecut LEGGE le fonti esterne ma non ci scrive mai dentro: quel file è
aperto in append da uno script che può essere in esecuzione in questo
momento, e due processi che ci scrivono insieme sono un ottimo modo per
rovinarlo. Le righe nuove finiscono quindi sempre e solo nel tracking di
Wavecut, e il totale è l'unione delle fonti. Quando lo script esterno ha
finito basta aggiungere il suo file alle fonti (o copiarlo dentro il repo):
i duplicati non danno fastidio, l'insieme li assorbe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TRACKING_FILE = Path.home() / ".wavecut" / "essentia_processed.txt"

# Il tracking dello script originale, ancora in uso mentre gira.
LEGACY_TRACKING_FILE = (
    Path.home() / "PycharmProjects" / "Essentia-to-Metadata" / ".essentia_processed.txt"
)


@dataclass
class SourceStats:
    path: Path
    lines: int = 0
    exists: bool = False


@dataclass
class ProcessedTracker:
    """Chi è già stato fatto, letto da più fonti e scritto in una sola.

    `tracking_file` è l'unico file su cui si scrive. `sources` sono in sola
    lettura (di default il tracking dello script originale).
    """
    tracking_file: Path = DEFAULT_TRACKING_FILE
    sources: list[Path] = field(default_factory=lambda: [LEGACY_TRACKING_FILE])
    _done: set[str] = field(default_factory=set, init=False)
    stats: list[SourceStats] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        for source in [*self.sources, self.tracking_file]:
            stat = SourceStats(path=source)
            # Il file può crescere mentre lo leggiamo: l'ultima riga può
            # risultare troncata. Al peggio un brano viene rianalizzato, il
            # che è innocuo — molto meglio che tenere un lock su un file di
            # qualcun altro.
            try:
                with source.open("r", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        entry = line.strip()
                        if entry:
                            stat.lines += 1
                            self._done.add(entry)
                stat.exists = True
            except OSError:
                pass
            self.stats.append(stat)

    def __contains__(self, path) -> bool:
        return str(path) in self._done

    def __len__(self) -> int:
        return len(self._done)

    @property
    def total_lines(self) -> int:
        """Righe lette in tutto: confrontata con `len()` dice quanta
        ripetizione c'è nelle fonti (nell'originale è circa un quinto)."""
        return sum(s.lines for s in self.stats)

    def pending(self, paths) -> list[Path]:
        """Quali dei percorsi dati non sono ancora stati fatti."""
        return [p for p in paths if str(p) not in self._done]

    def mark(self, path) -> None:
        """Segna un brano come fatto, in memoria e in coda al tracking di
        Wavecut. Ripetere la stessa marcatura non riscrive nulla."""
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
