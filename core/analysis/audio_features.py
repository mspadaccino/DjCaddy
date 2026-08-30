"""Caricamento audio (librosa) e calcolo di BPM + RMS energy.

L'audio viene caricato una sola volta per file (a `ANALYSIS_SR`, mono) e lo
stesso segnale serve sia qui, per BPM/RMS, sia in `structure.py` per la
segmentazione. BPM e RMS sono calcolati sui primi `ANALYSIS_DURATION_SECONDS`
del brano, per velocità e per aderenza alla prima versione dello strumento.

I file mp3 richiedono ffmpeg installato a livello di sistema
(`brew install ffmpeg`): librosa li decodifica via audioread.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ANALYSIS_SR = 22050              # sample rate comune per feature e struttura
ANALYSIS_DURATION_SECONDS = 60   # finestra per BPM/RMS


def load_audio(filepath: Path, sr: int = ANALYSIS_SR, mono: bool = True,
               duration: float | None = None):
    """Carica un file audio. Solleva l'eccezione al chiamante (gestione a monte)."""
    import librosa

    y, sr = librosa.load(str(filepath), sr=sr, mono=mono, duration=duration)
    return y, sr


def compute_bpm_rms(y: np.ndarray, sr: int) -> tuple[float, float]:
    """Calcola BPM stimato e RMS energy medio del segnale passato."""
    import librosa

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo) if not hasattr(tempo, "__len__") else float(np.atleast_1d(tempo)[0])
    rms = float(np.mean(librosa.feature.rms(y=y)))
    return bpm, rms
