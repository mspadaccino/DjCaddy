"""Vibe = bucket di tempo + energia relativa alla libreria.

Le soglie di energia sono percentili calcolati sull'intera libreria
analizzata (approccio two-pass: prima si estraggono tutte le feature, poi si
derivano le soglie). Bucket e percentili sono configurabili qui.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# --- Configurazione bucket di tempo (BPM -> nome) ---
TEMPO_BUCKETS: list[tuple[float, float, str]] = [
    (0, 100, "Warm-Up"),
    (100, 118, "Groove"),
    (118, 128, "Peak-Time"),
    (128, 999, "High-Energy-Tempo"),
]
UNKNOWN_TEMPO = "Unknown-Tempo"

# Percentili usati per dividere l'energia (RMS) in Low/Mid/High
ENERGY_PERCENTILES = (33, 66)
ENERGY_LABELS = ("Low", "Mid", "High")
UNKNOWN_ENERGY = "Unknown-Energy"


def bpm_to_tempo_bucket(bpm: float | None) -> str:
    if bpm is None:
        return UNKNOWN_TEMPO
    for low, high, name in TEMPO_BUCKETS:
        if low <= bpm < high:
            return name
    return UNKNOWN_TEMPO


def build_energy_labeler(rms_values: list[float]) -> Callable[[float | None], str]:
    """Ritorna una funzione RMS -> Low/Mid/High basata sui percentili
    calcolati sull'intera libreria analizzata."""
    if not rms_values:
        return lambda rms: UNKNOWN_ENERGY

    low_thresh, high_thresh = np.percentile(rms_values, ENERGY_PERCENTILES)

    def labeler(rms: float | None) -> str:
        if rms is None:
            return UNKNOWN_ENERGY
        if rms <= low_thresh:
            return ENERGY_LABELS[0]
        if rms <= high_thresh:
            return ENERGY_LABELS[1]
        return ENERGY_LABELS[2]

    return labeler


def compute_vibe(tempo_bucket: str, energy_label: str) -> str:
    """Combina bucket di tempo ed energia nella vibe finale (es. 'Peak-Time-High')."""
    return f"{tempo_bucket}-{energy_label}"
