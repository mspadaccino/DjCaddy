"""La radio: una playlist da un GRUPPO di brani, non da uno solo.

Quick List parte da un seme e risponde "cosa gli sta vicino". Qui si parte da un insieme — i preferiti, un lazo sulla mappa —
e si risponde "cosa va in quella direzione": è la modalità "radio" dei
servizi di streaming, riscritta sugli embedding della mappa.

Quattro regole, una per ogni modo in cui la versione ingenua fallisce:

- **Il gusto del gruppo è la media dei suoi vettori** — ma se il gruppo ha
  due anime (techno e bossa nova nei preferiti) la media sta in mezzo al
  nulla. Allora si spezza in parti (`split`) e ogni parte ha il suo profilo;
  si pesca da ciascuna in proporzione a quanto è grande.
- **Niente doppioni.** A vicinanza pura escono venti versioni dello stesso
  brano. Ogni candidato paga quanto somiglia a quelli già presi (maximal
  marginal relevance): la manopola `variety` dice quanto paga. Chi suona
  quasi identico a un preso o a un seme (`twin_min`) non entra proprio.
- **Deriva.** A ogni brano preso il profilo si sposta un po' verso di lui
  (`drift`): a zero la playlist resta attorno al gruppo, alta diventa un
  viaggio che se ne allontana.
- **I no contano.** I brani scartati (`negatives`) spingono il profilo
  dall'altra parte, alla Rocchio, e non vengono riproposti.

Il modulo è puro: prende matrici e indici, restituisce indici in ordine di
scelta. L'ordine mixabile lo dà poi `magic_sort`, non lui.
"""

from __future__ import annotations

import numpy as np

# Quanto pesa la media dei no rispetto a quella dei sì nel profilo. Meno di
# uno perché i no sono pochi e rumorosi: un brano tolto dice "non questo",
# non "il contrario di questo".
NEGATIVE_WEIGHT = 0.5

# Sotto questo guadagno di coesione una parte in più non spiega niente: si
# starebbe spezzando un gruppo che è uno solo.
SPLIT_GAIN = 0.03


def unit(vectors) -> np.ndarray:
    """I vettori a norma uno: da qui in poi il coseno è un prodotto scalare."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-9)


def _centroid(vectors: np.ndarray) -> np.ndarray:
    return unit(vectors.mean(axis=0))[0]


def _kmeans(vectors: np.ndarray, k: int, rounds: int = 10) -> np.ndarray:
    """K-means sferico, con i centri iniziali presi il più lontano possibile
    l'uno dall'altro — e dal primo vettore, non dal caso: lo stesso gruppo
    dà le stesse parti a ogni chiamata."""
    centers = [vectors[0]]
    while len(centers) < k:
        nearest = np.max(vectors @ np.array(centers).T, axis=1)
        centers.append(vectors[int(np.argmin(nearest))])
    centers = np.array(centers)
    labels = np.zeros(len(vectors), dtype=int)
    for _ in range(rounds):
        fresh = np.argmax(vectors @ centers.T, axis=1)
        if np.array_equal(fresh, labels) and _ > 0:
            break
        labels = fresh
        for part in range(k):
            members = vectors[labels == part]
            if len(members):
                centers[part] = _centroid(members)
    return labels


def _cohesion(vectors: np.ndarray, labels: np.ndarray) -> float:
    """Quanto ogni vettore sta vicino al centro della sua parte, in media."""
    total = 0.0
    for part in np.unique(labels):
        members = vectors[labels == part]
        total += float((members @ _centroid(members)).sum())
    return total / len(vectors)


def split(vectors, max_parts: int = 3,
          gain: float = SPLIT_GAIN) -> list[list[int]]:
    """Le anime del gruppo: liste di posizioni in `vectors`, una per parte.

    Si prova con due parti, poi tre, e si tiene la divisione solo finché
    ogni parte in più rende il gruppo sensibilmente più coeso. Un gruppo
    che è uno solo resta uno solo.
    """
    matrix = unit(vectors)
    if len(matrix) < 2:
        return [list(range(len(matrix)))]
    best = np.zeros(len(matrix), dtype=int)
    score = _cohesion(matrix, best)
    for k in range(2, min(max_parts, len(matrix)) + 1):
        labels = _kmeans(matrix, k)
        if len(np.unique(labels)) < k:
            break
        fresh = _cohesion(matrix, labels)
        if fresh - score < gain:
            break
        best, score = labels, fresh
    return [[int(i) for i in np.flatnonzero(best == part)]
            for part in np.unique(best)]


def tune(embeddings, seeds, pool=None, k: int = 20, variety: float = 0.5,
         drift: float = 0.0, negatives=(), twin_min: float = 0.97,
         max_parts: int = 3) -> list[int]:
    """I `k` brani che la radio propone da `seeds`, in ordine di scelta.

    `pool` limita i candidati (i brani che passano i filtri); senza, tutta
    la libreria. Semi e negativi non si ripropongono mai.
    """
    matrix = unit(embeddings)
    seeds = [int(i) for i in dict.fromkeys(seeds)]
    negatives = [int(i) for i in negatives]
    if not seeds or k <= 0 or not len(matrix):
        return []
    out = set(seeds) | set(negatives)
    candidates = np.arange(len(matrix)) if pool is None \
        else np.asarray(list(pool), dtype=int)
    candidates = np.array([c for c in candidates if int(c) not in out],
                          dtype=int)
    if not len(candidates):
        return []

    parts = [[seeds[i] for i in part] for part in split(matrix[seeds], max_parts)]
    profiles = [_centroid(matrix[part]) for part in parts]
    if negatives:
        away = _centroid(matrix[negatives])
        profiles = [unit(p - NEGATIVE_WEIGHT * away)[0] for p in profiles]

    rows = matrix[candidates]
    relevance = [rows @ p for p in profiles]
    # Quanto ogni candidato somiglia al più vicino fra semi e presi: è la
    # penalità della diversità, e cresce di un prodotto scalare a scelta.
    closest = np.max(rows @ matrix[seeds].T, axis=1)
    taken_from = [0] * len(parts)
    picked: list[int] = []
    alive = np.ones(len(candidates), dtype=bool)
    while len(picked) < k and alive.any():
        alive &= closest < twin_min
        if not alive.any():
            break
        part = min(range(len(parts)),
                   key=lambda p: (taken_from[p] / len(parts[p]), p))
        score = relevance[part] - variety * closest
        score[~alive] = -np.inf
        best = int(np.argmax(score))
        chosen = int(candidates[best])
        picked.append(chosen)
        taken_from[part] += 1
        alive[best] = False
        closest = np.maximum(closest, rows @ rows[best])
        if drift > 0:
            profiles[part] = unit((1 - drift) * profiles[part]
                                  + drift * rows[best])[0]
            relevance[part] = rows @ profiles[part]
    return picked
