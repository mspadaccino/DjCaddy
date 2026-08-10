"""Rilevamento vocal via source separation (Demucs).

Isola lo stem vocale e ne restituisce l'inviluppo di energia nel tempo, così le
sezioni con voce si riconoscono dall'energia reale del cantato (accurato).

È pesante: scarica un modello alla prima esecuzione e gira una rete neurale
sull'intero brano. È best-effort: se `demucs` non è installato o la separazione
fallisce, ritorna None e il flag vocal resta manuale (checkbox nell'app).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_HOP = 512
_model = None  # modello Demucs riusato fra le tracce (caricarlo è costoso)

# Estrazione delle regioni cantate dall'inviluppo vocale
VOCAL_FLOOR = 0.15     # frazione del picco vocale sopra cui c'è voce
MERGE_GAP_S = 1.0      # unisce regioni separate da pause brevi
MIN_REGION_S = 1.5     # scarta sprazzi troppo brevi


def vocal_regions(envelope, floor: float = VOCAL_FLOOR,
                  merge_gap: float = MERGE_GAP_S,
                  min_region: float = MIN_REGION_S) -> list[tuple[float, float]]:
    """Da (times, rms) dello stem vocale agli intervalli [start, end] cantati.

    Pura (numpy): soglia relativa al picco del brano, poi unisce le pause brevi
    e scarta i frammenti troppo corti.
    """
    times, rms = envelope
    if rms.size == 0:
        return []
    peak = float(rms.max())
    if peak <= 0:
        return []

    active = rms >= floor * peak
    # Intervalli contigui di frame attivi
    spans: list[list[float]] = []
    in_run = False
    for i, on in enumerate(active):
        if on and not in_run:
            start = float(times[i])
            in_run = True
        elif not on and in_run:
            spans.append([start, float(times[i])])
            in_run = False
    if in_run:
        spans.append([start, float(times[-1])])

    # Unisce regioni separate da pause < merge_gap
    merged: list[list[float]] = []
    for st, en in spans:
        if merged and st - merged[-1][1] <= merge_gap:
            merged[-1][1] = en
        else:
            merged.append([st, en])

    # Scarta i frammenti più corti di min_region
    return [(st, en) for st, en in merged if en - st >= min_region]


def available() -> bool:
    """True se l'API di basso livello di Demucs (+ torch) è importabile."""
    try:
        import torch  # noqa: F401
        import demucs.apply  # noqa: F401
        import demucs.pretrained  # noqa: F401
        return True
    except Exception:
        return False


def _device() -> str:
    """cpu di default (affidabile); override con DJ_DEMUCS_DEVICE (es. mps, cuda)."""
    import os

    import torch

    forced = os.environ.get("DJ_DEMUCS_DEVICE")
    if forced:
        return forced
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_model():
    global _model
    if _model is None:
        from demucs.pretrained import get_model
        _model = get_model("htdemucs")
        _model.cpu().eval()
    return _model


def vocal_envelope(filepath: Path):
    """Ritorna (times[s], rms) dello stem vocale, oppure None se non disponibile.

    Usa l'API di basso livello (pretrained + apply_model) e carica l'audio con
    librosa, così non dipende da `demucs.api` (assente in alcune build).
    """
    try:
        import librosa
        import torch
        from demucs.apply import apply_model

        model = _get_model()
        sr = int(model.samplerate)
        channels = int(model.audio_channels)

        y, _ = librosa.load(str(filepath), sr=sr, mono=False)
        y = np.asarray(y, dtype="float32")
        if y.ndim == 1:                       # file mono
            y = np.tile(y, (channels, 1)) if channels > 1 else y[np.newaxis, :]
        elif y.shape[0] != channels:          # adatta il numero di canali
            y = y.mean(0, keepdims=True) if channels == 1 else np.tile(y[:1], (channels, 1))

        wav = torch.from_numpy(y)
        ref = wav.mean(0)
        wav_n = (wav - ref.mean()) / (ref.std() + 1e-8)   # normalizzazione come in demucs

        with torch.no_grad():
            sources = apply_model(model, wav_n[None], device=_device(), progress=False)[0]
        sources = sources * ref.std() + ref.mean()

        idx = list(model.sources).index("vocals")
        mono = sources[idx].mean(0).cpu().numpy().astype("float32")
        if mono.size == 0:
            return None
        rms = librosa.feature.rms(y=mono, hop_length=_HOP)[0]
        times = librosa.frames_to_time(np.arange(rms.size), sr=sr, hop_length=_HOP)
        return times, rms
    except Exception:
        return None
