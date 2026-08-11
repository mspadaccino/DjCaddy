"""Segmentazione strutturale: individuazione dei phrase boundary.

Metodo (Foote novelty su matrice di auto-similarità):
  1. si stima il beat e si allineano le feature (MFCC + chroma) al beat, così
     i confini cadono sulla griglia musicale, che è ciò che serve per gli hot
     cue di un DJ e riduce la dimensione della matrice a valori trattabili;
  2. si costruisce una self-similarity matrix (coseno) fra i frame beat-sync;
  3. si convolve un kernel a scacchiera gaussiano lungo la diagonale, ottenendo
     una novelty curve i cui picchi segnano i cambi di sezione;
  4. si estraggono i picchi (con distanza minima in beat), li si converte in
     timestamp e si assegna una label indicativa dal delta di energia.

La confidenza è l'altezza del picco di novelty normalizzata in [0, 1].
Questa è la parte meno affidabile dell'analisi: i timestamp sono suggerimenti
da validare a orecchio nell'app Streamlit.
"""

from __future__ import annotations

import numpy as np

from .models import Boundary

# Parametri (in "beat", non secondi: seguono la griglia musicale)
KERNEL_HALF_BEATS = 16     # semi-lato del kernel a scacchiera (~4 battute in 4/4)
MIN_GAP_BEATS = 16         # distanza minima fra due boundary (~4 battute)
EDGE_GAP_BEATS = 4         # distanza minima dai bordi del brano (~1 battuta)
PEAK_DELTA = 0.10          # soglia relativa di prominenza del picco
MIN_BEATS = 2 * KERNEL_HALF_BEATS + 1  # brani troppo corti: nessun boundary


def _checkerboard_kernel(half: int) -> np.ndarray:
    """Kernel a scacchiera gaussiano di lato (2*half+1)."""
    axis = np.arange(-half, half + 1)
    xx, yy = np.meshgrid(axis, axis)
    sigma = half / 2.0
    gauss = np.exp(-0.5 * (xx ** 2 + yy ** 2) / (sigma ** 2))
    sign = np.sign(xx) * np.sign(yy)
    return gauss * sign


def _novelty(ssm: np.ndarray, half: int) -> np.ndarray:
    """Novelty curve: correlazione del kernel a scacchiera lungo la diagonale."""
    kernel = _checkerboard_kernel(half)
    n = ssm.shape[0]
    padded = np.pad(ssm, half, mode="edge")
    nov = np.empty(n)
    size = 2 * half + 1
    for i in range(n):
        window = padded[i:i + size, i:i + size]
        nov[i] = np.sum(window * kernel)
    nov = np.maximum(nov, 0.0)
    peak = nov.max()
    if peak > 0:
        nov = nov / peak
    return nov


def _keep_away_from_edges(peaks, last_beat_idx: int, gap_beats: int = EDGE_GAP_BEATS):
    """Scarta i picchi che ritaglierebbero una sezione più corta di una battuta.

    `fix_frames` aggiunge due frame sintetici, uno a 0 e uno a fine traccia,
    che non sono beat rilevati; con il padding `edge` della SSM producono
    picchi di novelty spuri proprio sui bordi. Su un brano reale usciva un
    boundary a 0.046 s, cioè una sezione di 46 ms: inutile da ascoltare e,
    peggio, uno degli 8 pad hot-cue di djay Pro sprecato su un doppione del
    cue a inizio brano.

    La soglia è UNA BATTUTA, non la distanza minima fra due boundary
    (MIN_GAP_BEATS, ~4 battute): sui bordi serve solo a togliere le schegge,
    e l'outro di un brano da club dura spesso pochi secondi — con la soglia
    piena sparirebbe, insieme al punto di mix-out che a un DJ serve. Pura,
    testabile.
    """
    return [p for p in peaks if gap_beats <= p <= last_beat_idx - gap_beats]


def _energy_label(rms_sync: np.ndarray, idx: int, w: int = 8) -> str:
    """Etichetta indicativa dal cambio di energia intorno al boundary."""
    before = rms_sync[max(0, idx - w):idx]
    after = rms_sync[idx:idx + w]
    if before.size == 0 or after.size == 0:
        return "Shift"
    b, a = float(before.mean()), float(after.mean())
    if b <= 0:
        return "Rise" if a > 0 else "Shift"
    ratio = a / b
    if ratio >= 1.15:
        return "Rise"
    if ratio <= 0.87:
        return "Fall"
    return "Shift"


def detect_boundaries(y: np.ndarray, sr: int) -> list[Boundary]:
    """Ritorna i phrase boundary suggeriti per il segnale (traccia intera)."""
    import librosa

    if y is None or y.size == 0:
        return []

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    if beats is None or len(beats) < MIN_BEATS:
        return []

    # Feature timbriche + armoniche, allineate al beat
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    feat = np.vstack([mfcc, chroma])

    # Confini di beat coerenti con `sync`: includono inizio (0) e fine traccia,
    # così la colonna p di `sync` inizia ESATTAMENTE a beat_times[p]
    # (evita l'off-by-one di un beat fra colonne sincronizzate e timestamp).
    beat_frames = librosa.util.fix_frames(beats, x_min=0, x_max=feat.shape[1])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    sync = librosa.util.sync(feat, beat_frames, aggregate=np.median)

    # Self-similarity coseno fra i frame beat-sync
    sync_n = librosa.util.normalize(sync, axis=0)
    ssm = np.dot(sync_n.T, sync_n)

    n = ssm.shape[0]
    half = min(KERNEL_HALF_BEATS, (n - 1) // 2)
    if half < 2:
        return []

    nov = _novelty(ssm, half)

    # Energia beat-sync per l'etichetta indicativa (stessi confini)
    rms = librosa.feature.rms(y=y)[0][np.newaxis, :]
    rms_sync = librosa.util.sync(rms, beat_frames, aggregate=np.mean)[0]

    peaks = librosa.util.peak_pick(
        nov,
        pre_max=MIN_GAP_BEATS // 2,
        post_max=MIN_GAP_BEATS // 2,
        pre_avg=MIN_GAP_BEATS,
        post_avg=MIN_GAP_BEATS,
        delta=PEAK_DELTA,
        wait=MIN_GAP_BEATS,
    )

    boundaries: list[Boundary] = []
    last_beat_idx = len(beat_times) - 1
    for p in _keep_away_from_edges(peaks, last_beat_idx):
        col = int(min(p, last_beat_idx))          # colonna sync -> onset esatto del beat p
        time = float(beat_times[col])
        conf = float(nov[int(min(p, n - 1))])
        label = _energy_label(rms_sync, int(min(p, len(rms_sync) - 1)))
        boundaries.append(Boundary(time=time, confidence=conf, label=label))

    return boundaries
