"""Il Chapter Builder: la playlist ripartita nei cinque capitoli di un set.

Qui c'è la logica — a chi tocca quale capitolo, in che ordine dentro al
capitolo, e le aree colorate che la lavagna disegna sotto le schede. Le
tabelle, i bottoni e lo stato di sessione stanno nelle app.

I capitoli stessi — quote, fasce, colori — stanno in `core.analysis.arc`,
perché li legge anche il Journey, che fa il lavoro opposto: qui una
playlist fatta si ripartisce nell'arco, là l'arco sceglie i brani. I nomi
si riesportano da qui per chi li ha sempre importati da qui.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.analysis.arc import CHAPTER_COLORS, CHAPTERS, chapter_score
from core.analysis.mixing import camelot_distance

__all__ = ["CHAPTERS", "CHAPTER_COLORS", "assign_chapters",
           "board_chapter_regions"]


def assign_chapters(frame: pd.DataFrame,
                    playlist: list[int]) -> list[list[int]]:
    """Assign playlist tracks to five chapters and sort each one.

    Tracks are assigned by percentile affinity to the chapter's emotional
    profile, then each chapter is sorted by BPM direction with a greedy
    Camelot walk within same-BPM tracks. Chapters are chained so that the
    first track of one connects harmonically to the last of the previous.
    """
    n = len(playlist)
    if n == 0:
        return [[] for _ in CHAPTERS]

    bpms = np.array([float(frame.at[i, "bpm"] or 0) for i in playlist])
    arousals = np.array([float(frame.at[i, "energy"] or 0) for i in playlist])
    valences = np.array([float(frame.at[i, "valence_rank"] or 0)
                         for i in playlist])
    grooves = np.array([float(frame.at[i, "danceability"] or 0)
                        for i in playlist])

    def _percentile_rank(arr: np.ndarray) -> np.ndarray:
        order = arr.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(arr), dtype=float)
        return ranks / max(1, len(arr) - 1)

    pct_bpm = _percentile_rank(bpms)
    pct_arousal = arousals
    pct_valence = valences
    pct_groove = _percentile_rank(grooves)

    scores = np.zeros((n, len(CHAPTERS)), dtype=np.float64)
    for j, ch in enumerate(CHAPTERS):
        for k in range(n):
            scores[k, j] = chapter_score(
                pct_bpm[k], pct_arousal[k], pct_valence[k], pct_groove[k], ch)

    quotas = [max(1, round(ch["quota"] * n)) for ch in CHAPTERS]
    overshoot = sum(quotas) - n
    while overshoot > 0:
        biggest = max(range(len(quotas)), key=lambda j: quotas[j])
        quotas[biggest] -= 1
        overshoot -= 1
    while overshoot < 0:
        smallest = min(range(len(quotas)), key=lambda j: quotas[j])
        quotas[smallest] += 1
        overshoot += 1

    assigned: list[list[int]] = [[] for _ in CHAPTERS]
    taken = set()
    for j in range(len(CHAPTERS)):
        # Il pareggio si scioglie sull'indice del brano nella libreria, non
        # sulla sua posizione nella playlist: quella cambia da un giro
        # all'altro — un "Apply" la riscrive nell'ordine dei capitoli — e un
        # pareggio sciolto sulla posizione dava un risultato diverso ogni
        # volta che si rifaceva "Create chapters" sugli stessi brani.
        candidates = [(scores[k, j], playlist[k], k)
                      for k in range(n) if k not in taken]
        candidates.sort()
        for _, _, k in candidates[:quotas[j]]:
            assigned[j].append(playlist[k])
            taken.add(k)

    # Sort each chapter: BPM sets the direction (ascending for the
    # rising arc, descending for Release), then within same-BPM tracks
    # a greedy Camelot walk picks the smoothest harmonic path.
    # The last chapter (Release) reverses direction.
    sorted_chapters: list[list[int]] = []
    for ci, ch_tracks in enumerate(assigned):
        if not ch_tracks:
            sorted_chapters.append(ch_tracks)
            continue
        descending = (ci == len(CHAPTERS) - 1)  # Release
        ordered = _bpm_then_camelot(frame, ch_tracks, descending)
        # At chapter boundaries, rotate so the first track is
        # Camelot-compatible with the last of the previous chapter.
        if sorted_chapters and sorted_chapters[-1] and len(ordered) > 1:
            tail_key = frame.at[sorted_chapters[-1][-1], "camelot"]
            best_start = min(range(len(ordered)), key=lambda k: (
                camelot_distance(tail_key,
                                 frame.at[ordered[k], "camelot"]),
                abs(frame.at[ordered[k], "bpm"]
                    - frame.at[ordered[0], "bpm"])))
            if best_start > 0:
                ordered = ordered[best_start:] + ordered[:best_start]
        sorted_chapters.append(ordered)

    return sorted_chapters


def _bpm_then_camelot(frame: pd.DataFrame, tracks: list[int],
                      descending: bool) -> list[int]:
    """Sort tracks by BPM direction, then Camelot-walk within same-BPM groups.

    Tracks are bucketed by rounded BPM; within each bucket a greedy
    nearest-neighbour walk on Camelot distance keeps harmonic flow.
    """
    by_bpm: dict[int, list[int]] = {}
    for i in tracks:
        bpm = round(float(frame.at[i, "bpm"] or 0))
        by_bpm.setdefault(bpm, []).append(i)

    bpm_keys = sorted(by_bpm.keys(), reverse=descending)
    result: list[int] = []
    for bpm_val in bpm_keys:
        group = by_bpm[bpm_val]
        if len(group) <= 1:
            result.extend(group)
            continue
        # Greedy Camelot walk within this BPM bucket
        if result:
            prev_key = frame.at[result[-1], "camelot"]
            start = min(group, key=lambda i:
                        camelot_distance(prev_key, frame.at[i, "camelot"]))
        else:
            start = group[0]
        walked = [start]
        remaining = set(group) - {start}
        while remaining:
            cur_key = frame.at[walked[-1], "camelot"]
            nxt = min(remaining, key=lambda i:
                      camelot_distance(cur_key, frame.at[i, "camelot"]))
            walked.append(nxt)
            remaining.discard(nxt)
        result.extend(walked)
    return result


def board_chapter_regions(ch_lookup: dict[int, str] | None,
                          playlist: list[int]) -> list[dict]:
    """Chapter shading regions for the board graph.

    Returns a list of {start, end, color, name} dicts where start/end
    are 0-based positions in the playlist. Consecutive tracks in the
    same chapter form one region.
    """
    if not ch_lookup:
        return []
    regions: list[dict] = []
    prev_name = None
    for pos, idx in enumerate(playlist):
        name = ch_lookup.get(idx)
        if name != prev_name:
            if regions:
                regions[-1]["end"] = pos - 1
            if name:
                regions.append({"start": pos, "end": pos,
                                "color": CHAPTER_COLORS.get(name, "#888"),
                                "name": name})
            prev_name = name
        elif name and regions:
            regions[-1]["end"] = pos
    return regions
