"""Strutture dati condivise dell'analisi."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Boundary:
    """Un confine di frase suggerito all'interno di una traccia."""

    time: float          # secondi dall'inizio del brano
    confidence: float    # 0..1, forza relativa del picco di novelty
    label: str = ""      # etichetta indicativa (es. "Rise" / "Fall" / "Shift")

    def to_dict(self) -> dict:
        return {"time": self.time, "confidence": self.confidence, "label": self.label}

    @classmethod
    def from_dict(cls, d: dict) -> "Boundary":
        return cls(time=d["time"], confidence=d["confidence"], label=d.get("label", ""))


@dataclass
class TrackAnalysis:
    """Risultato dell'analisi per singola traccia.

    Le feature per-file (genre, bpm, rms, boundaries) sono cacheabili: non
    dipendono dal resto della libreria. La `vibe` invece è library-relative
    (percentili di energia) e viene assegnata nel secondo passaggio, quindi
    NON viene messa in cache.
    """

    path: Path
    genre: str
    bpm: float | None
    rms: float | None
    boundaries: list[Boundary] = field(default_factory=list)
    error: str | None = None
    vibe: str | None = None

    # --- serializzazione per la cache (solo la parte per-file) ---
    def to_dict(self) -> dict:
        return {
            "genre": self.genre,
            "bpm": self.bpm,
            "rms": self.rms,
            "boundaries": [b.to_dict() for b in self.boundaries],
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, path: Path, d: dict) -> "TrackAnalysis":
        return cls(
            path=path,
            genre=d["genre"],
            bpm=d["bpm"],
            rms=d["rms"],
            boundaries=[Boundary.from_dict(b) for b in d.get("boundaries", [])],
            error=d.get("error"),
        )

    # --- riga per il report CLI ---
    def to_row(self) -> dict:
        return {
            "path": str(self.path),
            "genre": self.genre,
            "bpm": round(self.bpm, 1) if self.bpm is not None else "",
            "vibe": self.vibe or "",
            "boundaries": ";".join(f"{b.time:.2f}" for b in self.boundaries),
            "error": self.error or "",
        }
