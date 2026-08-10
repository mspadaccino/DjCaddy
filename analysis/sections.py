"""Classificazione delle sezioni strutturali (Intro/Build-up/Drop/...).

Euristica basata sull'arco di energia e sulla presenza di basso, adatta alla
struttura a blocchi della dance/house/techno "da club". Non è ML: è un insieme
di regole ordinate con soglie regolabili, da confermare a orecchio nell'app.

Ingresso: i boundary già rilevati da `structure.py` definiscono i tagli; qui si
caratterizza ogni sezione e le si assegna un'etichetta della tassonomia.
"""

from __future__ import annotations

import numpy as np

from .models import (
    BREAKDOWN,
    BUILDUP,
    DROP,
    GROOVE,
    INTRO,
    OUTRO,
    Boundary,
    Section,
)

# Soglie (regolabili). Energia relativa al massimo delle sezioni del brano.
HIGH_ENERGY = 0.70   # >= -> candidato Drop
LOW_ENERGY = 0.45    # <= -> candidato Breakdown
BASS_FULL = 0.45     # quota di energia sotto 200 Hz per dire "basso pieno"

_LOW_HZ = 200.0
_N_FFT = 2048
_HOP = 512
_BEATS_PER_BAR = 4

# Una sezione è "vocal" se almeno questa frazione è coperta da regioni cantate
# (dallo stem vocale di Demucs, vedi vocals.py).
VOCAL_SECTION_COVER = 0.10


def _label(energy_rel: np.ndarray, bass_ratio: np.ndarray) -> list[str]:
    """Assegna le etichette alle sezioni date le feature relative. Pura, testabile."""
    n = len(energy_rel)
    if n == 0:
        return []
    if n == 1:
        return [GROOVE]

    labels = [GROOVE] * n
    labels[0] = INTRO
    labels[-1] = OUTRO

    # Drop: energia alta + basso pieno
    for i in range(1, n - 1):
        if energy_rel[i] >= HIGH_ENERGY and bass_ratio[i] >= BASS_FULL:
            labels[i] = DROP

    # Breakdown: energia bassa + basso ridotto, in mezzo al brano
    for i in range(1, n - 1):
        if labels[i] == GROOVE and energy_rel[i] <= LOW_ENERGY and bass_ratio[i] < BASS_FULL:
            labels[i] = BREAKDOWN

    # Build-up: sezione in salita subito prima di un Drop
    for i in range(1, n - 1):
        if labels[i] == GROOVE and labels[i + 1] == DROP and energy_rel[i] < energy_rel[i + 1]:
            labels[i] = BUILDUP

    return labels


def _section_features(y: np.ndarray, sr: int, starts: list[float], ends: list[float]):
    """Energia media e quota di basso per ogni sezione (via STFT)."""
    import librosa

    S = np.abs(librosa.stft(y, n_fft=_N_FFT, hop_length=_HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_N_FFT)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=_HOP)

    total = S.sum(axis=0)
    low = S[freqs < _LOW_HZ].sum(axis=0)

    energy = np.zeros(len(starts))
    bass_ratio = np.zeros(len(starts))
    for i, (a, b) in enumerate(zip(starts, ends)):
        mask = (times >= a) & (times < b)
        if not mask.any():                       # sezione più corta di un frame
            mask = np.zeros_like(times, dtype=bool)
            mask[min(int(np.searchsorted(times, a)), len(times) - 1)] = True
        e = float(total[mask].mean())
        energy[i] = e
        bass_ratio[i] = float(low[mask].mean()) / e if e > 0 else 0.0

    peak = energy.max() if energy.size and energy.max() > 0 else 1.0
    return energy / peak, bass_ratio


def annotate_vocals(sections: list[Section], regions: list[tuple[float, float]]) -> None:
    """Marca ogni sezione in base alla copertura da parte delle regioni cantate.

    `vocal_score` = frazione della sezione coperta da voce (0..1); `vocal` è
    True se la copertura supera VOCAL_SECTION_COVER.
    """
    for s in sections:
        length = s.end - s.start
        if length <= 0:
            continue
        covered = sum(
            max(0.0, min(e, s.end) - max(st, s.start))
            for st, e in regions if e > s.start and st < s.end
        )
        s.vocal_score = covered / length
        s.vocal = s.vocal_score >= VOCAL_SECTION_COVER


def classify_sections(
    y: np.ndarray, sr: int, boundaries: list[Boundary],
    duration: float | None, bpm: float | None,
) -> list[Section]:
    """Costruisce e classifica le sezioni fra i boundary (0..durata)."""
    if y is None or y.size == 0 or not duration or duration <= 0:
        return []

    cuts = [0.0] + [b.time for b in boundaries]
    starts = sorted({max(0.0, min(c, duration)) for c in cuts})
    ends = starts[1:] + [duration]

    energy_rel, bass_ratio = _section_features(y, sr, starts, ends)
    labels = _label(energy_rel, bass_ratio)

    bar_seconds = (_BEATS_PER_BAR * 60.0 / bpm) if bpm else None
    sections: list[Section] = []
    for i in range(len(starts)):
        bars = (ends[i] - starts[i]) / bar_seconds if bar_seconds else None
        sections.append(Section(
            start=starts[i], end=ends[i], label=labels[i],
            energy=float(energy_rel[i]), bars=bars,
        ))
    return sections
