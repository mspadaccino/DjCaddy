"""Waveform colorata per bande di frequenza, in stile djay Pro / Serato.

L'altezza codifica il volume, il colore la composizione in frequenza:
  - rosso  = bassi   (< LOW_HZ)
  - verde  = medi    (LOW_HZ..HIGH_HZ)
  - blu    = alti    (>= HIGH_HZ)

`compute_frequency_waveform` carica il conto pesante (STFT via librosa) e
delega la parte numerica pura a helper testabili senza audio.
"""

from __future__ import annotations

import subprocess

import numpy as np

LOW_HZ = 200.0
HIGH_HZ = 2000.0

# Quante colonne disegna l'onda del lettore, e a che frequenza si legge
# l'audio per ricavarla. Mille campioni al secondo sono un millesimo di
# quelli veri e bastano per il PROFILO: quello che si guarda nel dock e'
# dove il brano sale e dove stacca, non la forma della singola oscillazione.
ENVELOPE_POINTS = 800
ENVELOPE_RATE = 1000


def envelope(path: str) -> tuple[list[float], float] | None:
    """Il profilo di ampiezza del brano — (peaks 0..1, durata in secondi) —
    o None se non si riesce a leggerlo.

    E' l'onda del lettore in fondo alla pagina, e sta in core perche' i
    lettori sono DUE — il dock di Streamlit e quello dell'app Qt — e il
    criterio di parita' e' proprio "stesso brano, stessa forma d'onda":
    condividere il conto la garantisce per costruzione.

    Decodifica con ffmpeg e non con librosa: misurato sullo stesso brano da
    17 MB, 0,37 s contro 1,31 s, e ffprobe e' gia' quello che usa il
    controllo di integrita'. Se ffmpeg manca o il file e' illeggibile si
    torna None: sopra si ricade su un lettore senza onda, perche' l'onda e'
    un di piu' e non deve poter impedire l'ascolto.
    """
    try:
        raw = subprocess.run(
            ["ffmpeg", "-v", "quiet", "-i", path, "-ac", "1",
             "-ar", str(ENVELOPE_RATE), "-f", "s16le", "-"],
            capture_output=True, timeout=120).stdout
    except Exception:
        return None

    samples = np.frombuffer(raw, dtype="<i2")
    if samples.size < ENVELOPE_POINTS:
        return None
    usable = samples.size - samples.size % ENVELOPE_POINTS
    peaks = np.abs(samples[:usable].reshape(ENVELOPE_POINTS, -1)).max(axis=1)
    loudest = peaks.max() or 1
    return (peaks / loudest).round(3).tolist(), samples.size / ENVELOPE_RATE

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
