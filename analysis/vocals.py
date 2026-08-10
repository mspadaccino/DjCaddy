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
_separator = None  # istanza Demucs riusata fra le tracce (il modello è pesante)

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
    try:
        import demucs.api  # noqa: F401
        return True
    except Exception:
        return False


def _get_separator():
    global _separator
    if _separator is None:
        from demucs.api import Separator
        _separator = Separator(model="htdemucs")
    return _separator


def vocal_envelope(filepath: Path):
    """Ritorna (times[s], rms) dello stem vocale, oppure None se non disponibile."""
    try:
        import librosa

        sep = _get_separator()
        _, stems = sep.separate_audio_file(str(filepath))
        vocals = stems["vocals"]                       # tensor (canali, campioni)
        sr = int(sep.samplerate)
        mono = vocals.mean(dim=0).detach().cpu().numpy().astype("float32")
        if mono.size == 0:
            return None
        rms = librosa.feature.rms(y=mono, hop_length=_HOP)[0]
        times = librosa.frames_to_time(np.arange(rms.size), sr=sr, hop_length=_HOP)
        return times, rms
    except Exception:
        return None
