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

# Aggancio dei boundary al battito (vedi snap_to_local_beat).
GRID_HOP = 256             # hop dell'inviluppo di onset: ~11.6 ms a 22050 Hz
GRID_BINS = 192            # risoluzione della fase nel ripiegamento
GRID_MIN_CONTRAST = 3.0    # sotto questo il brano non ha una pulsazione netta
                           # (tempo suonato, non programmato): niente aggancio
GRID_SPAN_BEATS = 16       # ampiezza della finestra in cui si misura la fase


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


def _fold(env: np.ndarray, times: np.ndarray, period: float):
    """Ripiega l'inviluppo di onset modulo `period`.

    Ritorna (contrasto, fase): il picco dell'istogramma è la fase della
    pulsazione, e quanto svetta sul fondo dice se a quel periodo il brano è
    davvero periodico. Pura, testabile.
    """
    if period <= 0 or times.size == 0:
        return 0.0, 0.0
    bins = ((times % period) / period * GRID_BINS).astype(int) % GRID_BINS
    fold = np.bincount(bins, weights=env, minlength=GRID_BINS)
    mean = fold.mean()
    peak = int(np.argmax(fold))
    return (float(fold[peak] / mean) if mean else 0.0,
            (peak + 0.5) / GRID_BINS * period)


def snap_to_local_beat(time: float, period: float, onset_env: np.ndarray,
                       onset_times: np.ndarray, span_beats: int = GRID_SPAN_BEATS) -> float:
    """Porta `time` sul battito più vicino, misurando la fase LOCALMENTE.

    La fase si ricava da una finestra di poche battute attorno al punto, non
    da tutto il brano: anche conoscendo il tempo con precisione, su sette
    minuti un errore minimo fa scivolare una griglia globale di oltre cento
    millisecondi, mentre in una finestra così stretta non fa in tempo a
    farsi sentire.
    """
    if period <= 0 or onset_env.size == 0:
        return time
    half = span_beats * period
    window = (onset_times >= time - half) & (onset_times <= time + half)
    if not window.any():
        return time
    env = onset_env[window]
    _, phase = _fold(env / (env.max() or 1.0),
                     onset_times[window] - (time - half), period)
    phase += time - half
    return float(phase + round((time - phase) / period) * period)


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


def detect_boundaries(y: np.ndarray, sr: int, bpm: float | None = None) -> list[Boundary]:
    """Ritorna i phrase boundary suggeriti per il segnale (traccia intera).

    `bpm`, se noto da fonte esterna (djay Pro lo tiene nella sua libreria),
    è la griglia su cui agganciare i boundary: è quella che l'utente vede
    disegnata in console, quindi allinearsi a quella è ciò che conta. Senza,
    i boundary restano dove li mette il beat tracker — vedi il commento
    sull'aggancio più sotto: una griglia indovinata fa più danni che altro.
    """
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

    # I tempi dei battiti di librosa sono quantizzati al frame (~23 ms) e
    # seguono una stima di tempo che può essere sbagliata di qualche punto
    # percentuale: i cue finiscono così sparsi attorno al battito invece che
    # sopra. Si riprende quindi la fase dall'inviluppo di onset, alla
    # risoluzione fine, e ci si aggancia — ma solo se il brano ha una
    # pulsazione netta, altrimenti è meglio non toccare niente.
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=GRID_HOP)
    onset_times = librosa.frames_to_time(
        np.arange(len(onset_env)), sr=sr, hop_length=GRID_HOP)
    # Ci si aggancia SOLO con un BPM di provenienza esterna. Provare a
    # stimarselo qui peggiora le cose, ed è stato misurato: agganciando a una
    # griglia dedotta da noi la forza di onset sul cue CALA (0.235 -> 0.117 e
    # 0.198 -> 0.073 su due brani), mentre col BPM di djay Pro sale
    # (0.113 -> 0.152 e 0.202 -> 0.230). Una griglia col tempo sbagliato è
    # peggio di nessuna griglia: sposta i cue con sicurezza nel posto errato.
    quantize = False
    if bpm and bpm > 0:
        period = 60.0 / bpm
        contrast, _ = _fold(onset_env / (onset_env.max() or 1.0), onset_times, period)
        quantize = contrast >= GRID_MIN_CONTRAST

    boundaries: list[Boundary] = []
    last_beat_idx = len(beat_times) - 1
    for p in _keep_away_from_edges(peaks, last_beat_idx):
        col = int(min(p, last_beat_idx))          # colonna sync -> onset esatto del beat p
        time = float(beat_times[col])
        if quantize:
            time = max(0.0, snap_to_local_beat(time, period, onset_env, onset_times))
        conf = float(nov[int(min(p, n - 1))])
        label = _energy_label(rms_sync, int(min(p, len(rms_sync) - 1)))
        boundaries.append(Boundary(time=time, confidence=conf, label=label))

    return boundaries
