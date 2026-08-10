"""Waveform colorata per bande di frequenza, in stile djay Pro / Serato.

L'altezza codifica il volume, il colore la composizione in frequenza:
  - rosso  = bassi   (< LOW_HZ)
  - verde  = medi    (LOW_HZ..HIGH_HZ)
  - blu    = alti    (>= HIGH_HZ)

`compute_frequency_waveform` carica il conto pesante (STFT via librosa) e
delega la parte numerica pura a helper testabili senza audio.
"""

from __future__ import annotations

import numpy as np

LOW_HZ = 200.0
HIGH_HZ = 2000.0

_N_FFT = 2048
_HOP = 512


def _scale(x: np.ndarray) -> np.ndarray:
    """Normalizza una banda in [0, 1] usando il 99° percentile (robusto ai picchi)."""
    if x.size == 0:
        return x
    ref = np.percentile(x, 99)
    if ref <= 0:
        ref = x.max() if x.max() > 0 else 1.0
    return np.clip(x / ref, 0.0, 1.0)


def _colors(lo: np.ndarray, mid: np.ndarray, hi: np.ndarray) -> list[str]:
    r = (_scale(lo) * 255).astype(int)
    g = (_scale(mid) * 255).astype(int)
    b = (_scale(hi) * 255).astype(int)
    return [f"#{ri:02x}{gi:02x}{bi:02x}" for ri, gi, bi in zip(r, g, b)]


def _build(S: np.ndarray, freqs: np.ndarray, times: np.ndarray, points: int):
    """Da spettrogramma di ampiezza a (tempi, ampiezza normalizzata, colori hex).

    Pura (solo numpy): l'STFT è a monte. Aggrega i frame in al più `points` bin.
    """
    n = S.shape[1]
    if n == 0:
        return np.array([]), np.array([]), []

    lo = S[freqs < LOW_HZ].sum(axis=0)
    mid = S[(freqs >= LOW_HZ) & (freqs < HIGH_HZ)].sum(axis=0)
    hi = S[freqs >= HIGH_HZ].sum(axis=0)
    amp = S.sum(axis=0)

    bins = min(points, n)
    chunks = np.array_split(np.arange(n), bins)

    def agg(a: np.ndarray) -> np.ndarray:
        return np.array([a[c].mean() for c in chunks])

    lo, mid, hi, amp, t = agg(lo), agg(mid), agg(hi), agg(amp), agg(times)
    amp_n = amp / (amp.max() if amp.max() > 0 else 1.0)
    return t, amp_n, _colors(lo, mid, hi)


def compute_frequency_waveform(y: np.ndarray, sr: int, points: int = 1600):
    """Ritorna (tempi[s], ampiezza in [0,1], colori hex) per il disegno."""
    import librosa

    if y is None or y.size == 0:
        return np.array([]), np.array([]), []

    S = np.abs(librosa.stft(y, n_fft=_N_FFT, hop_length=_HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_N_FFT)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=_HOP)
    return _build(S, freqs, times, points)
