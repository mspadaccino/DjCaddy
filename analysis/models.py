"""Strutture dati condivise dell'analisi."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Tassonomia delle sezioni (struttura dance/house/techno "da club").
# "Groove" è il fallback per ciò che non ricade chiaramente nelle altre.
INTRO = "Intro"
BUILDUP = "Build-up"
DROP = "Drop/Chorus"
BREAKDOWN = "Breakdown"
OUTRO = "Outro"
GROOVE = "Groove"
SECTION_LABELS = [INTRO, BUILDUP, DROP, BREAKDOWN, OUTRO, GROOVE]


def format_elapsed(time: float) -> str:
    """Formatta un istante come tempo trascorso dall'inizio `MM:SS.d`."""
    total = round(max(0.0, time), 1)
    minutes, seconds = divmod(total, 60)
    return f"{int(minutes):02d}:{seconds:04.1f}"


def format_remaining(time: float, duration: float | None) -> str:
    """Formatta un istante come tempo rimanente `-MM:SS.d` dalla fine del brano.

    Coerente con la visualizzazione di default di djay Pro (countdown, decimi).
    Se la durata non è nota, ripiega sui secondi assoluti dall'inizio.
    """
    if duration is None:
        return f"{time:.2f}"
    remaining = round(max(0.0, duration - time), 1)
    minutes, seconds = divmod(remaining, 60)
    return f"-{int(minutes):02d}:{seconds:04.1f}"


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
class Section:
    """Una sezione strutturale classificata (fra due boundary)."""

    start: float          # secondi dall'inizio
    end: float            # secondi dall'inizio
    label: str            # una delle SECTION_LABELS
    energy: float = 0.0   # 0..1, energia relativa al massimo del brano
    bars: float | None = None  # lunghezza stimata in battute (da BPM)
    vocal: bool = False   # presenza di voce (euristico, correggibile a mano)
    vocal_score: float = 0.0   # 0..1, quota di energia armonica in banda voce

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "label": self.label,
                "energy": self.energy, "bars": self.bars,
                "vocal": self.vocal, "vocal_score": self.vocal_score}

    @classmethod
    def from_dict(cls, d: dict) -> "Section":
        return cls(start=d["start"], end=d["end"], label=d["label"],
                   energy=d.get("energy", 0.0), bars=d.get("bars"),
                   vocal=d.get("vocal", False), vocal_score=d.get("vocal_score", 0.0))


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
    duration: float | None = None   # secondi, per il tempo rimanente
    boundaries: list[Boundary] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    error: str | None = None
    vibe: str | None = None

    # --- serializzazione per la cache (solo la parte per-file) ---
    def to_dict(self) -> dict:
        return {
            "genre": self.genre,
            "bpm": self.bpm,
            "rms": self.rms,
            "duration": self.duration,
            "boundaries": [b.to_dict() for b in self.boundaries],
            "sections": [s.to_dict() for s in self.sections],
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, path: Path, d: dict) -> "TrackAnalysis":
        return cls(
            path=path,
            genre=d["genre"],
            bpm=d["bpm"],
            rms=d["rms"],
            duration=d.get("duration"),
            boundaries=[Boundary.from_dict(b) for b in d.get("boundaries", [])],
            sections=[Section.from_dict(s) for s in d.get("sections", [])],
            error=d.get("error"),
        )

    # --- riga per il report CLI ---
    def to_row(self) -> dict:
        return {
            "path": str(self.path),
            "genre": self.genre,
            "bpm": round(self.bpm, 1) if self.bpm is not None else "",
            "vibe": self.vibe or "",
            "boundaries": "; ".join(
                f"{format_remaining(b.time, self.duration)}({b.label} {b.confidence:.2f})"
                for b in self.boundaries
            ),
            "sections": "; ".join(
                f"{s.label}{' 🎤' if s.vocal else ''} "
                f"{format_remaining(s.start, self.duration)}"
                + (f" [{s.bars:.0f}b]" if s.bars else "")
                for s in self.sections
            ),
            "error": self.error or "",
        }
