"""Quanto costa passare da un brano all'altro, e cosa ci si costruisce sopra.

Il cuore della sezione Map: una playlist non è una lista di testo ma un
grafo orientato pesato, dove i nodi sono i brani e l'arco A→B vale quanto
è facile mixarli. Il costo è quello della specifica:

    D(A,B) = w1·d_sound(A,B) + w2·d_bpm(A,B) + w3·d_camelot(A,B)

Le tre distanze sono tutte normalizzate 0..1, altrimenti i pesi non
sarebbero confrontabili: `w1=w2=w3` deve voler dire "contano uguale".

Tutto qui dentro è puro: prende numeri, restituisce numeri. Niente audio,
niente Essentia, niente Streamlit — si prova con array finti.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Ruota Camelot
# --------------------------------------------------------------------------

# Nome della tonalità → codice Camelot. Le chiavi sono (tonica, modo) con la
# tonica normalizzata a bemolle, perché i tag del mondo reale scrivono la
# stessa nota in tre modi diversi (F#, Gb, Fis).
_ENHARMONIC = {
    "C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb",
    "CIS": "Db", "DIS": "Eb", "FIS": "Gb", "GIS": "Ab", "AIS": "Bb",
    "E#": "F", "B#": "C", "Cb": "B", "Fb": "E",
}

_MAJOR = ["B", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E"]
_MINOR = ["Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "Gb", "Db"]

CAMELOT: dict[tuple[str, str], str] = {}
for _i, _note in enumerate(_MAJOR, start=1):
    CAMELOT[(_note, "major")] = f"{_i}B"
for _i, _note in enumerate(_MINOR, start=1):
    CAMELOT[(_note, "minor")] = f"{_i}A"


def to_camelot(key: str | None, scale: str | None = None) -> str | None:
    """Da "F# minor", "Gbm", "11A" al codice Camelot. None se non si capisce.

    Accetta già i codici Camelot perché è così che molti tag di libreria
    portano la tonalità: chi ha già "8A" nel file non deve essere costretto
    a farla ricalcolare.
    """
    if not key:
        return None
    text = str(key).strip()
    if not text:
        return None

    upper = text.upper().replace(" ", "")
    if len(upper) in (2, 3) and upper[-1] in "AB" and upper[:-1].isdigit():
        number = int(upper[:-1])
        if 1 <= number <= 12:
            return f"{number}{upper[-1]}"

    # "F# minor" / "Gbm" / "F#" (maggiore sottinteso)
    parts = text.replace("-", " ").split()
    note = parts[0]
    rest = " ".join(parts[1:]).lower() if len(parts) > 1 else ""
    if not rest and len(note) > 1 and note.endswith(("m", "M")):
        # "Gbm": la m finale è il modo, non parte della nota
        if note[-1] == "m":
            rest, note = "minor", note[:-1]
    mode = str(scale or rest or "major").lower()
    mode = "minor" if mode.startswith(("min", "m")) and not mode.startswith("maj") else "major"

    note = note[0].upper() + note[1:].replace("♯", "#").replace("♭", "b")
    note = _ENHARMONIC.get(note.upper(), _ENHARMONIC.get(note, note))
    return CAMELOT.get((note, mode))


# Costo di una transizione sulla ruota, in funzione di quanti passi di ruota
# separano i due codici. Passo 0 e passo 1 sono le mosse gratuite del mixaggio
# armonico (8A→8A, 8A→9A): è la specifica a volerle a costo zero. Da lì in
# poi il costo sale in fretta, perché due chiavi a tre passi di distanza si
# scontrano davvero.
_RING_COST = {0: 0.0, 1: 0.0, 2: 0.4, 3: 0.6, 4: 0.8, 5: 0.9, 6: 1.0}

# Quanto si paga a cambiare lettera (maggiore↔minore) quando NON si è sullo
# stesso numero. Sullo stesso numero è il relativo (8A→8B): gratis.
_MODE_PENALTY = 0.2

# Senza tonalità non si può né premiare né punire: mezzo costo, così un brano
# senza chiave non scala in cima alle proposte solo perché non si sa nulla.
UNKNOWN_KEY_COST = 0.5


def camelot_distance(a: str | None, b: str | None) -> float:
    """Costo 0..1 di passare dalla chiave `a` alla chiave `b`."""
    if not a or not b:
        return UNKNOWN_KEY_COST
    try:
        n1, l1 = int(a[:-1]), a[-1].upper()
        n2, l2 = int(b[:-1]), b[-1].upper()
    except (ValueError, IndexError):
        return UNKNOWN_KEY_COST
    steps = min((n1 - n2) % 12, (n2 - n1) % 12)
    cost = _RING_COST[steps]
    if l1 != l2 and steps != 0:
        cost = min(1.0, cost + _MODE_PENALTY)
    return cost


# --------------------------------------------------------------------------
# BPM
# --------------------------------------------------------------------------

# Oltre questo scarto il pitch fader non basta più: è la soglia dove il costo
# arriva a metà, non dove la transizione diventa impossibile.
BPM_TOLERANCE = 0.06


def bpm_distance(a: float | None, b: float | None,
                 tolerance: float = BPM_TOLERANCE) -> float:
    """Costo 0..1 dello scarto di tempo, a ottave ripiegate.

    Un brano a 128 e uno a 64 si mixano in half-time: contarli lontanissimi
    sarebbe sbagliato in una libreria che tiene insieme hip hop e techno.
    """
    if not a or not b:
        return UNKNOWN_KEY_COST
    relative = min(abs(b * factor - a) / a for factor in (0.5, 1.0, 2.0))
    if relative <= tolerance:
        return 0.5 * relative / tolerance
    return min(1.0, 0.5 + 0.5 * (relative - tolerance) / tolerance)


# --------------------------------------------------------------------------
# Scarti con segno
# --------------------------------------------------------------------------

# Le distanze qui sopra dicono QUANTO due brani sono lontani; queste dicono
# DA CHE PARTE. Servono a mostrarlo, non a ordinare: un set sale, tiene e
# lascia cadere, e mettere la direzione dentro al punteggio significherebbe
# decidere quale delle tre — che è la scelta del DJ, non del programma.


def bpm_shift(a: float | None, b: float | None) -> float | None:
    """Di quanto cambia il tempo passando da `a` a `b`, col segno.

    A ottave ripiegate come `bpm_distance`, e per la stessa ragione: da 128
    a 64 si va in half-time senza cambiare passo, e segnarlo come −64 farebbe
    leggere una frenata dove non c'è.
    """
    if not a or not b:
        return None
    folded = min((b * factor for factor in (0.5, 1.0, 2.0)),
                 key=lambda value: abs(value - a))
    return folded - a


def camelot_shift(a: str | None, b: str | None) -> tuple[int, bool] | None:
    """Passi con segno sulla ruota da `a` a `b`, e se cambia il modo.

    Positivo è orario (8A→9A), che è la mossa con cui si alza di un grado
    senza stonare. Si prende sempre la via breve, perché la ruota si chiude:
    da 12A a 1A è un passo avanti, non undici indietro.

    Il cambio di modo viaggia a parte perché non è un passo: 8A→8B sono zero
    passi e un salto al relativo maggiore, che è una mossa vera.
    """
    if not a or not b:
        return None
    try:
        n1, l1 = int(a[:-1]), a[-1].upper()
        n2, l2 = int(b[:-1]), b[-1].upper()
    except (ValueError, IndexError):
        return None
    return (n2 - n1 + 6) % 12 - 6, l1 != l2


# --------------------------------------------------------------------------
# Costo della transizione
# --------------------------------------------------------------------------

class TransitionCost:
    """La funzione D(A,B) applicata a una libreria intera.

    Si costruisce una volta sui vettori della libreria e poi si interroga
    per indice. `vectors` sono gli embedding (N × 1280): la distanza di
    suono è 1 − coseno fra i due, tagliata a 0..1. Prima era la distanza
    sulla mappa in due dimensioni, che dell'embedding è un'ombra: due brani
    vicini nell'ombra non lo sono per forza, e viceversa. Misurare nelle
    1280 dimensioni fa di "cosa gli somiglia" il caso di Quick List coi
    pesi 1, 0, 0 — e la mappa torna a servire a guardare, non a misurare.

    Su 90.000 brani la matrice dei vettori è mezzo giga: non si copia (è
    una vista dello store) e si legge per righe — tutte quando i bersagli
    sono la libreria intera, le sole che servono altrimenti.
    """

    def __init__(self, vectors, bpm, camelot,
                 w_sound: float = 1.0, w_bpm: float = 1.0, w_key: float = 1.0):
        self.vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        self.norms = np.maximum(np.linalg.norm(self.vectors, axis=1), 1e-9)
        self.bpm = list(bpm)
        self.camelot = list(camelot)
        self.w_sound, self.w_bpm, self.w_key = w_sound, w_bpm, w_key

    def sound_distances(self, vector, targets) -> np.ndarray:
        """1 − coseno fra `vector` e ogni bersaglio, tagliato a 0..1."""
        targets = np.asarray(list(targets), dtype=int)
        vector = np.asarray(vector, dtype=np.float32)
        scale = max(float(np.linalg.norm(vector)), 1e-9)
        # Indicizzare la matrice copia le righe chieste: su mezza libreria
        # sono centinaia di mega. Da un quarto in su costa meno leggerla
        # tutta e tenere della risposta solo quello che serve.
        if len(targets) * 4 >= len(self.vectors):
            dots = (self.vectors @ vector)[targets]
        else:
            dots = self.vectors[targets] @ vector
        cosine = dots / (self.norms[targets] * scale)
        return np.clip(1.0 - cosine, 0.0, 1.0)

    def to(self, source: int, targets) -> np.ndarray:
        """Il costo D(source, t) per ogni t in `targets`."""
        return self.from_point((self.vectors[source], self.bpm[source],
                                self.camelot[source]), targets)

    def from_point(self, point, targets) -> np.ndarray:
        """Il costo da un punto che non è un brano: `(vettore, bpm,
        camelot)`. Serve alla tendenza del Chain Maker, che cerca vicino a
        dove la catena STA ANDANDO, non a dove è arrivata."""
        targets = np.asarray(list(targets), dtype=int)
        if not len(targets):
            return np.empty(0, dtype=np.float32)
        vector, bpm, camelot = point
        sound = self.sound_distances(vector, targets)
        tempo = np.array([bpm_distance(bpm, self.bpm[t]) for t in targets])
        key = np.array([camelot_distance(camelot, self.camelot[t])
                        for t in targets])
        return (self.w_sound * sound + self.w_bpm * tempo + self.w_key * key) / \
            max(1e-9, self.w_sound + self.w_bpm + self.w_key)

    def ahead(self, previous: int, last: int, trend: float) -> tuple:
        """Dove sarebbe il prossimo brano se la catena continuasse dritta.

        Da `previous` a `last` c'è una direzione, nel suono e nel tempo;
        `trend` dice quanto proseguirla: a 0 si sta fermi su `last`, a 1 si
        fa un altro passo uguale. La tonalità non ha una direzione — è una
        ruota — e resta quella di `last`.
        """
        vector = self.vectors[last] + trend * (self.vectors[last]
                                               - self.vectors[previous])
        bpm = self.bpm[last]
        if bpm and self.bpm[previous]:
            bpm = bpm + trend * (bpm - self.bpm[previous])
        return vector, bpm, self.camelot[last]

    def between(self, a: int, b: int) -> float:
        return float(self.to(a, [b])[0])

    def parts(self, a: int, b: int) -> dict:
        """Le tre distanze separate: serve a mostrare PERCHÉ un brano è vicino."""
        return {
            "sound": float(self.sound_distances(self.vectors[a], [b])[0]),
            "bpm": bpm_distance(self.bpm[a], self.bpm[b]),
            "key": camelot_distance(self.camelot[a], self.camelot[b]),
        }


def nearest(cost: TransitionCost, seed: int, k: int = 20,
            pool=None, ahead=None) -> list[tuple[int, float]]:
    """I `k` brani con la transizione più economica da `seed`.

    `pool` limita la ricerca a un sottoinsieme (i brani che passano i filtri
    della pagina): senza, si cerca in tutta la libreria. Con `ahead` — un
    punto da `TransitionCost.ahead` — si misura da lì invece che dal seme,
    che resta comunque escluso.
    """
    candidates = np.arange(len(cost.bpm)) if pool is None else np.asarray(list(pool), dtype=int)
    candidates = candidates[candidates != seed]
    if not len(candidates):
        return []
    costs = (cost.from_point(ahead, candidates) if ahead is not None
             else cost.to(seed, candidates))
    order = np.argsort(costs)[:k]
    return [(int(candidates[i]), float(costs[i])) for i in order]


# --------------------------------------------------------------------------
# Playlist disegnata sulla mappa
# --------------------------------------------------------------------------

def resample_path(points, step: float) -> np.ndarray:
    """Punti equidistanti lungo la spezzata `points`, passo `step`.

    Il tratto che arriva dal mouse ha vertici fitti dove si è disegnato piano
    e radi dove si è disegnato veloce: campionarlo a passo costante evita che
    la velocità della mano decida quanti brani entrano in playlist.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return pts
    segments = np.diff(pts, axis=0)
    lengths = np.hypot(segments[:, 0], segments[:, 1])
    travelled = np.concatenate([[0.0], np.cumsum(lengths)])
    total = travelled[-1]
    if total <= 0:
        return pts[:1]
    wanted = np.arange(0.0, total + step, max(step, 1e-9))
    return np.column_stack([np.interp(wanted, travelled, pts[:, 0]),
                            np.interp(wanted, travelled, pts[:, 1])])


def closed_shape(points, tolerance: float = 0.2) -> bool:
    """Il tratto disegnato torna dove era partito?

    Serve a capire cosa si sta chiedendo, senza doverlo dichiarare: un tratto
    che si richiude è un RECINTO — "prendi quello che c'è dentro" — mentre
    uno aperto è un PERCORSO — "prendi quello che tocco, nell'ordine in cui
    lo tocco". Sono due domande diverse e lo stesso gesto le fa entrambe.

    Chiuso vuol dire che la distanza fra il primo e l'ultimo punto è piccola
    RISPETTO A QUANTO SI È GIRATO: un cerchio largo può lasciare un varco di
    parecchi pixel e restare inequivocabilmente un cerchio.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return False
    steps = np.diff(pts, axis=0)
    length = float(np.hypot(steps[:, 0], steps[:, 1]).sum())
    if length <= 0:
        return False
    gap = float(np.hypot(*(pts[0] - pts[-1])))
    return gap <= tolerance * length


def along_path(coords, points, radius: float, pool=None,
               step: float | None = None) -> list[int]:
    """I brani entro `radius` dal tratto disegnato, nell'ordine del tratto.

    Un brano vicino a due punti del tratto entra una volta sola, al primo:
    l'ordine di una playlist è il racconto che si è disegnato, e tornare
    indietro a riprendere lo stesso brano lo spezzerebbe.
    """
    coords = np.asarray(coords, dtype=np.float32)
    if not len(coords):
        return []
    samples = resample_path(points, step if step is not None else radius / 2)
    allowed = None if pool is None else np.asarray(list(pool), dtype=int)

    taken: list[int] = []
    seen: set[int] = set()
    for x, y in samples:
        delta = coords - np.array([x, y], dtype=np.float32)
        near = np.flatnonzero(np.hypot(delta[:, 0], delta[:, 1]) <= radius)
        if allowed is not None:
            near = np.intersect1d(near, allowed, assume_unique=False)
        # Dentro lo stesso campione l'ordine è per vicinanza al punto: due
        # brani presi insieme non hanno un "prima" dato dal tratto.
        for i in sorted(near, key=lambda j: float(np.hypot(*(coords[j] - [x, y])))):
            if int(i) not in seen:
                seen.add(int(i))
                taken.append(int(i))
    return taken


# --------------------------------------------------------------------------
# Magic sort: il commesso viaggiatore, addomesticato
# --------------------------------------------------------------------------

def magic_sort(cost: TransitionCost, indices, start: int | None = None,
               passes: int = 2) -> list[int]:
    """Ordina `indices` in modo che ogni brano si fonda col successivo.

    È il cammino minimo che tocca tutti i nodi una volta: TSP aperto, che è
    NP-difficile. Si fa quello che si fa sempre — vicino più vicino per avere
    un percorso, poi 2-opt per raddrizzarne gli incroci — perché qui i nodi
    sono le decine di brani di una serata, non una libreria intera, e su
    quelle dimensioni la soluzione esatta non varrebbe l'attesa.
    """
    nodes = [int(i) for i in indices]
    if len(nodes) <= 2:
        return nodes

    # Matrice dei costi una volta sola: il 2-opt la interroga O(n²) volte per
    # passata, e ricalcolare bpm/camelot ogni volta si sentirebbe.
    matrix = np.array([cost.to(node, nodes) for node in nodes], dtype=np.float32)

    begin = nodes.index(start) if start in nodes else 0
    route = [begin]
    left = set(range(len(nodes))) - {begin}
    while left:
        current = route[-1]
        nxt = min(left, key=lambda j: matrix[current, j])
        route.append(nxt)
        left.discard(nxt)

    def length(order: list[int]) -> float:
        return float(sum(matrix[a, b] for a, b in zip(order, order[1:])))

    # 2-opt: si rovescia il tratto fra due posizioni se accorcia il totale.
    # Il primo nodo resta fermo — è il brano da cui l'utente vuole partire.
    for _ in range(passes):
        improved = False
        best = length(route)
        for i in range(1, len(route) - 1):
            for j in range(i + 1, len(route)):
                candidate = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                value = length(candidate)
                if value < best - 1e-6:
                    route, best, improved = candidate, value, True
        if not improved:
            break

    return [nodes[i] for i in route]


def sorted_after(cost: TransitionCost, playlist: list[int],
                 group: list[int]) -> list[int]:
    """Il gruppo ordinato per attaccarsi a quello che c'è già.

    Magic sort da solo sceglie da dove partire, e va bene finché la playlist
    comincia lì. In coda a una playlist esistente no: il primo del gruppo
    finisce dietro all'ultimo di prima, e se lo si lascia scegliere alla
    cieca quella giuntura è l'unico salto della serata. Si parte dal brano
    del gruppo che costa meno raggiungere da lì.
    """
    if not playlist:
        return magic_sort(cost, group)
    tail = playlist[-1]
    return magic_sort(cost, group,
                      start=min(group, key=lambda i: cost.between(tail, i)))
