"""Il Journey: da un brano a un altro in `n` passi, seguendo l'arco.

Quick List risponde "cosa gli sta vicino", il Chain Maker "cosa viene
dopo", magic sort "in che ordine". Manca "come arrivo da qui a lì": un
brano di partenza, uno di arrivo se lo si conosce, una lunghezza, e la
forma che la serata deve avere. Questo modulo la risponde con una fila di
brani presi dalla libreria — non da una playlist che esiste già — dove
ogni brano mixa col successivo e l'energia segue l'arco.

Tre pezzi, ognuno con un motivo:

- **Il corridoio.** Su novantamila brani non si cerca fra tutti: si
  tengono i `reach` più economici da raggiungere sommando il costo dalla
  partenza e quello dall'arrivo — un'ellisse fra i due, che è quel che
  "sulla strada" vuol dire — o una palla attorno alla partenza se l'arrivo
  non c'è. Le copie della stessa canzone entrano una volta sola.
- **Viterbi a strati.** Uno strato per posizione: lo strato 0 è la
  partenza, l'ultimo l'arrivo. Il costo di un cammino è la somma delle
  transizioni `D` più, a ogni posizione, quanto il brano sta fuori dal
  capitolo che l'arco assegna a quella posizione, pesato da `w_arc`.
  Sul corridoio la soluzione è esatta, ed è lo stesso costo di tutto il
  resto della pagina: gli stessi pesi, le stesse tre distanze.
- **Niente ripetizioni.** Viterbi da solo trova un CAMMINO, e un cammino
  può tornare su un brano: da A a B e di nuovo ad A costa due passi
  piccoli. Le transizioni fra gemelli — stesso master, altro nome — si
  vietano prima; i ritorni si tolgono dopo: un brano che compare due volte
  resta dove compare la prima volta, si vieta negli strati seguenti, e si
  rifà. Ogni giro chiude almeno uno strato a un brano, quindi finisce.

Il modulo è puro: prende il costo e delle matrici, torna indici in ordine
di scaletta. L'arco lo legge da `arc.py`, lo stesso dei capitoli.
"""

from __future__ import annotations

import numpy as np

from . import arc
from .mixing import TransitionCost

# Sotto questa distanza di suono due brani sono lo stesso master: il
# passaggio dall'uno all'altro non è una transizione, è un doppione. La
# soglia è quella della radio (coseno 0.97).
TWIN_SOUND = 0.03

# Quanto largo il corridoio, per brano chiesto, e fra quali limiti: sotto
# i 150 il Viterbi non ha scelta, sopra i 1500 le matrici pesano e la
# strada non migliora più.
REACH_PER_STEP = 30
REACH_MIN, REACH_MAX = 150, 1500


def corridor(cost: TransitionCost, start: int, end: int | None,
             pool=None, reach: int = REACH_MIN, song_of=None) -> list[int]:
    """I `reach` brani più a portata fra `start` ed `end`, senza i due e
    senza le copie della stessa canzone (la prima che si incontra resta).

    Con l'arrivo, il costo è la somma dei due: chi sta in mezzo costa poco
    da entrambi. Senza, è il costo dalla partenza e basta.
    """
    candidates = (np.arange(len(cost.bpm)) if pool is None
                  else np.asarray(list(pool), dtype=int))
    ends = {int(start)} | ({int(end)} if end is not None else set())
    candidates = candidates[~np.isin(candidates, list(ends))]
    if not len(candidates):
        return []
    distance = cost.to(int(start), candidates)
    if end is not None:
        distance = distance + cost.to(int(end), candidates)
    order = candidates[np.argsort(distance, kind="stable")]
    if song_of is None:
        return [int(i) for i in order[:reach]]
    seen = {song_of(i) for i in ends}
    out: list[int] = []
    for i in order:
        song = song_of(int(i))
        if song in seen:
            continue
        seen.add(song)
        out.append(int(i))
        if len(out) >= reach:
            break
    return out


def _viterbi(hops: np.ndarray, extra: np.ndarray, banned: np.ndarray,
             first: int, last: int | None) -> list[int] | None:
    """Il cammino più economico a `n` strati sulla matrice `hops` (R × R),
    con `extra[t, j]` da pagare per stare in `j` allo strato `t` e
    `banned[t, j]` dove non si può stare. Parte da `first`; finisce in
    `last` se dato, altrove dove costa meno. None se non c'è cammino."""
    n, size = extra.shape
    best = np.full((n, size), np.inf, dtype=np.float64)
    back = np.zeros((n, size), dtype=int)
    best[0, first] = 0.0
    for t in range(1, n):
        reach = best[t - 1][:, None] + hops
        back[t] = np.argmin(reach, axis=0)
        best[t] = reach[back[t], np.arange(size)] + extra[t]
        best[t, banned[t]] = np.inf
    finish = last if last is not None else int(np.argmin(best[n - 1]))
    if not np.isfinite(best[n - 1, finish]):
        return None
    path = [finish]
    for t in range(n - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return path[::-1]


def plan(cost: TransitionCost, start: int, n: int, end: int | None = None,
         pool=None, arc_values: np.ndarray | None = None,
         w_arc: float = 0.0, song_of=None, reach: int | None = None,
         twin: float = TWIN_SOUND) -> list[int]:
    """La fila di `n` brani da `start` a `end`, in ordine di scaletta.

    `pool` limita i candidati (i brani che passano i filtri); la partenza e
    l'arrivo entrano comunque. `arc_values` è la matrice N × 4 di
    `arc.measures` sulla libreria, e `w_arc` quanto conta stare nel
    capitolo giusto rispetto a una transizione: a 0 l'arco non parla.
    `song_of` tiene fuori le copie di un brano già in fila; `twin` è la
    distanza di suono sotto cui due brani non sono una transizione.

    Torna meno di `n` brani solo quando non c'è modo di farne di più: un
    corridoio troppo stretto, o nessun cammino che eviti i gemelli.
    """
    start = int(start)
    n = int(n)
    if end is not None and int(end) == start:
        end = None
    if n <= 1 or not len(cost.bpm):
        return [start]
    if end is not None and n == 2:
        return [start, int(end)]

    if reach is None:
        reach = min(REACH_MAX, max(REACH_MIN, REACH_PER_STEP * n))
    inner = corridor(cost, start, end, pool=pool, reach=reach,
                     song_of=song_of)
    ring = [start] + inner + ([int(end)] if end is not None else [])
    if len(ring) < n:
        # Non c'è abbastanza da cui scegliere: si dà quello che si può,
        # nell'ordine del corridoio, che è già "dal più vicino".
        return ring

    hops = cost.matrix(ring).astype(np.float64)
    hops[cost.sound_matrix(ring) < twin] = np.inf
    np.fill_diagonal(hops, np.inf)             # restare fermi non è un passo

    size = len(ring)
    chapters = arc.chapters_along(n)
    extra = np.zeros((n, size), dtype=np.float64)
    if arc_values is not None and w_arc > 0:
        values = np.asarray(arc_values, dtype=float)[ring]
        for t, chapter in enumerate(chapters):
            extra[t] = w_arc * arc.arc_costs(values, chapter)

    first, last = 0, (size - 1 if end is not None else None)
    banned = np.zeros((n, size), dtype=bool)
    banned[1:, first] = True
    if last is not None:
        banned[:-1, last] = True
        banned[-1, :last] = True

    while True:
        path = _viterbi(hops, extra, banned, first, last)
        if path is None:
            return [start] + ([int(end)] if end is not None else [])
        places: dict[int, list[int]] = {}
        for t, node in enumerate(path):
            places.setdefault(node, []).append(t)
        repeated = {node: at for node, at in places.items() if len(at) > 1}
        if not repeated:
            return [ring[node] for node in path]
        # Un brano ripetuto resta dove compare la PRIMA volta e si vieta
        # dopo: un cammino che torna indietro si raddrizza tenendo l'andata
        # e cercando altro al posto del ritorno. Ogni giro chiude almeno
        # uno strato a un brano, quindi finisce.
        for node, at in repeated.items():
            banned[at[0] + 1:, node] = True
