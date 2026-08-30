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

# Soglie (regolabili). Tutte relative al brano stesso, non assolute: così le
# regole reggono fra generi e masterizzazioni diverse.
HIGH_ENERGY = 0.70   # energia / max del brano: >= -> candidato Drop
BASS_PRESENT = 0.50  # energia sotto 200 Hz / max del brano: >= -> "il basso c'è"
RISING = 0.35        # crescita interna alla sezione per dire "sta salendo"

_LOW_HZ = 200.0
_N_FFT = 2048
_HOP = 512
_BEATS_PER_BAR = 4

# Una sezione è "vocal" se almeno questa frazione è coperta da regioni cantate
# (dallo stem vocale di Demucs, vedi vocals.py).
VOCAL_SECTION_COVER = 0.10


def _label(
    energy_rel: np.ndarray,
    bass_level: np.ndarray,
    slope: np.ndarray | None = None,
) -> list[str]:
    """Assegna le etichette alle sezioni date le feature relative. Pura, testabile.

    Tutte e tre le feature sono rapportate al massimo del brano (vedi
    `_section_features`), quindi le soglie non dipendono dal genere né dal
    volume di masterizzazione:

    - `energy_rel`: quanto va forte la sezione rispetto al picco del brano;
    - `bass_level`: quanto basso c'è rispetto al momento di massimo basso —
      è il segnale che dice se kick e sub ci sono o sono stati tolti, ed è
      ciò che separa un drop da un breakdown o dalla coda di un build-up;
    - `slope`: quanto la sezione cresce DENTRO di sé, l'unica cosa che
      distingue un build-up da un groove qualsiasi (per energia media si
      somigliano). Se non viene passata si ricade sulle regole di vicinato.
    """
    n = len(energy_rel)
    if n == 0:
        return []
    if n == 1:
        return [GROOVE]
    if slope is None:
        slope = np.zeros(n)
    if n == 2:
        # Con due sole sezioni non c'è un arco da riconoscere: per un DJ la
        # lettura utile resta "da qui si entra, qui si esce".
        return [INTRO, OUTRO]

    # Drop: va forte E il basso c'è. Una sezione ancora in forte salita è la
    # CODA di un build-up, non un drop, anche se arriva già forte.
    is_drop = [
        energy_rel[i] >= HIGH_ENERGY and bass_level[i] >= BASS_PRESENT
        and slope[i] < RISING
        for i in range(n)
    ]
    labels = [DROP if is_drop[i] else GROOVE for i in range(n)]

    # Breakdown: il basso sparisce e l'energia CALA rispetto a prima. Non si
    # richiede che l'energia sia bassa in assoluto: nella house/disco il
    # breakdown tiene su archi e voce, cala solo il fondo (verificato su
    # brani reali, dove l'energia resta anche oltre 0.6 mentre il basso
    # crolla a 0.07). Va deciso PRIMA del build-up: una sezione senza basso
    # appena prima di un drop è il breakdown classico, mentre la salita si
    # riconosce semmai dalla pendenza.
    for i in range(1, n - 1):
        if (labels[i] == GROOVE and slope[i] < RISING
                and bass_level[i] < BASS_PRESENT
                and energy_rel[i] < HIGH_ENERGY
                and energy_rel[i] < energy_rel[i - 1]):
            labels[i] = BREAKDOWN

    # Build-up: o sale visibilmente dentro di sé verso qualcosa di più
    # grosso, o (senza informazione di pendenza) è la sezione più bassa
    # subito prima di un drop.
    for i in range(n - 1):
        if labels[i] != GROOVE:
            continue
        if slope[i] >= RISING and energy_rel[i + 1] >= energy_rel[i]:
            labels[i] = BUILDUP
        elif is_drop[i + 1] and energy_rel[i] < energy_rel[i + 1]:
            labels[i] = BUILDUP

    # Bordi: per un DJ la prima e l'ultima sezione sono le zone di mix, a
    # meno che non siano chiaramente dei drop (brani che aprono o chiudono
    # in pieno: chiamarli Intro/Outro sarebbe fuorviante in console).
    if labels[0] != DROP:
        labels[0] = INTRO
    if labels[-1] != DROP:
        labels[-1] = OUTRO

    return labels


def _section_features(y: np.ndarray, sr: int, starts: list[float], ends: list[float]):
    """Energia, quantità di basso e crescita interna di ogni sezione (STFT).

    Energia e basso sono normalizzati sul massimo del brano, non su valori
    assoluti: `bass_level` risponde a "quanto basso c'è rispetto al momento
    di massimo basso di QUESTO brano", che è la domanda giusta per capire se
    il kick c'è o è stato tolto. (La versione precedente usava invece la
    QUOTA di spettro sotto i 200 Hz: su brani reali quel numero non supera
    ~0.33 anche a basso pieno — e su un disco anni '70 arriva a 0.13 — così
    la soglia non veniva mai raggiunta e nessuna sezione era mai un Drop.)

    La crescita (`slope`) confronta l'ultimo terzo della sezione col primo,
    rapportata alla sua energia media: è adimensionale e serve a riconoscere
    i build-up, che per energia MEDIA sono indistinguibili da un groove ma
    dentro di sé salgono.
    """
    import librosa

    S = np.abs(librosa.stft(y, n_fft=_N_FFT, hop_length=_HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_N_FFT)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=_HOP)

    total = S.sum(axis=0)
    low = S[freqs < _LOW_HZ].sum(axis=0)

    energy = np.zeros(len(starts))
    bass = np.zeros(len(starts))
    slope = np.zeros(len(starts))
    for i, (a, b) in enumerate(zip(starts, ends)):
        mask = (times >= a) & (times < b)
        if not mask.any():                       # sezione più corta di un frame
            mask = np.zeros_like(times, dtype=bool)
            mask[min(int(np.searchsorted(times, a)), len(times) - 1)] = True
        e = float(total[mask].mean())
        energy[i] = e
        bass[i] = float(low[mask].mean())

        frames = np.flatnonzero(mask)
        third = frames.size // 3
        if third and e > 0:
            head = float(total[frames[:third]].mean())
            tail = float(total[frames[-third:]].mean())
            slope[i] = (tail - head) / e

    def _relative(v: np.ndarray) -> np.ndarray:
        peak = v.max() if v.size and v.max() > 0 else 1.0
        return v / peak

    return _relative(energy), _relative(bass), slope


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

    energy_rel, bass_level, slope = _section_features(y, sr, starts, ends)
    labels = _label(energy_rel, bass_level, slope)

    raw = [
        Section(start=starts[i], end=ends[i], label=labels[i],
                energy=float(energy_rel[i]))
        for i in range(len(starts))
    ]
    sections = _merge_consecutive(raw)   # tieni solo i CAMBI di phrase

    bar_seconds = (_BEATS_PER_BAR * 60.0 / bpm) if bpm else None
    for s in sections:
        s.bars = (s.end - s.start) / bar_seconds if bar_seconds else None
    return sections


def _merge_consecutive(sections: list[Section]) -> list[Section]:
    """Collassa sezioni adiacenti con la stessa etichetta in un'unica phrase.

    Il tag resta solo dove la phrase CAMBIA: niente ripetizioni dello stesso
    tipo una dopo l'altra. Pura, testabile.
    """
    if not sections:
        return []
    merged = [sections[0]]
    for s in sections[1:]:
        prev = merged[-1]
        if s.label == prev.label:
            prev.end = s.end
            prev.energy = max(prev.energy, s.energy)
        else:
            merged.append(s)
    return merged
