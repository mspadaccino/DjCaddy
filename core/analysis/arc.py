"""L'arco di un set: cinque capitoli, ognuno con una quota e un bersaglio.

È la forma che una serata dovrebbe avere — Intro, Buildup, Tension, Climax,
Release — scritta come dati: quanta parte del set spetta a ogni capitolo, e
in che fascia di tempo, energia, valence e groove stanno i suoi brani.

La leggono due strumenti che fanno cose opposte con la stessa forma:

- il **Chapter Builder** (`core/viz/chapters.py`) prende una playlist già
  fatta e dice quale brano va in quale capitolo — l'arco applicato DOPO;
- il **Journey** (`core/analysis/journey.py`) cerca nella libreria i brani
  che realizzano l'arco fra un brano di partenza e uno di arrivo — l'arco
  applicato PRIMA.

Una definizione sola, perché se il Journey inseguisse una curva e i
capitoli ne applicassero un'altra, il primo produrrebbe una playlist che
i secondi poi rimescolano.

Le quattro misure sono tutte fra 0 e 1 e tutte RELATIVE: energia e valence
sono già ranghi sulla libreria; tempo e groove lo diventano qui, con
`measures`. Il Chapter Builder li prende invece in percentile della
playlist — su una playlist fatta è la scala giusta, e cambiarla cambierebbe
i capitoli che la gente vede — ma il Journey sceglie nella libreria intera,
e "tempo nel quarto più alto" prima che i brani siano scelti ha senso solo
sulla libreria.
"""

from __future__ import annotations

import numpy as np

from .energy import ranks

# Le quattro misure di un capitolo, nell'ordine delle colonne di `measures`.
MEASURES = ("bpm", "arousal", "valence", "groove")

CHAPTERS = [
    {"name": "Intro",   "icon": "🌅", "quota": 0.15,
     "bpm": (0.00, 0.15), "arousal": (0.00, 0.25),
     "valence": (0.30, 0.50), "groove": (0.20, 0.40)},
    {"name": "Buildup", "icon": "🔨", "quota": 0.25,
     "bpm": (0.15, 0.40), "arousal": (0.25, 0.60),
     "valence": (0.40, 0.60), "groove": (0.75, 0.95)},
    {"name": "Tension", "icon": "🌀", "quota": 0.20,
     "bpm": (0.40, 0.70), "arousal": (0.60, 0.80),
     "valence": (0.00, 0.20), "groove": (0.80, 1.00)},
    {"name": "Climax",  "icon": "⚡", "quota": 0.25,
     "bpm": (0.70, 1.00), "arousal": (0.85, 1.00),
     "valence": (0.75, 1.00), "groove": (0.00, 0.90)},
    {"name": "Release", "icon": "🌙", "quota": 0.15,
     "bpm": (0.30, 0.50), "arousal": (0.30, 0.50),
     "valence": (0.20, 0.45), "groove": (0.30, 0.60)},
]

CHAPTER_COLORS = {
    "Intro":   "#8e9aa6",
    "Buildup": "#f2a33c",
    "Tension": "#7b4fbf",
    "Climax":  "#e0503b",
    "Release": "#3d9be0",
}


def _outside(value: float, low: float, high: float) -> float:
    """Di quanto `value` esce dalla fascia: zero dentro, la distanza dal
    bordo più vicino fuori."""
    if low <= value <= high:
        return 0.0
    return min(abs(value - low), abs(value - high))


def chapter_score(bpm: float, arousal: float, valence: float,
                  groove: float, chapter: dict) -> float:
    """Quanto un brano sta fuori dal capitolo: 0 (dentro su tutte e quattro
    le misure) fino a 4 (il caso peggiore). La somma delle quattro uscite,
    che è ciò che il Chapter Builder ha sempre minimizzato."""
    return (_outside(bpm, *chapter["bpm"])
            + _outside(arousal, *chapter["arousal"])
            + _outside(valence, *chapter["valence"])
            + _outside(groove, *chapter["groove"]))


def chapter_at(position: float) -> int:
    """In quale capitolo cade un punto del set, da 0 (l'apertura) a 1 (la
    chiusura), secondo le quote: i primi 15% sono Intro, e così via."""
    edge = 0.0
    for n, chapter in enumerate(CHAPTERS):
        edge += chapter["quota"]
        if position < edge:
            return n
    return len(CHAPTERS) - 1


def chapters_along(n: int) -> list[int]:
    """Il capitolo di ognuna delle `n` posizioni di un set, in fila.

    Con una posizione sola è l'apertura. Le quote si leggono sull'intervallo
    chiuso, così l'ultimo brano cade sempre in Release e il primo in Intro,
    che è il minimo che un arco deve garantire.
    """
    if n <= 0:
        return []
    if n == 1:
        return [0]
    return [chapter_at(k / (n - 1) * (1 - 1e-9)) for k in range(n)]


def measures(bpm, energy, valence_rank, danceability) -> np.ndarray:
    """Le quattro misure di ogni brano sulla scala della libreria: una
    matrice N × 4, colonne come `MEASURES`, tutto fra 0 e 1.

    Tempo e groove diventano ranghi sulla libreria — quello che si passa —
    perché i loro numeri grezzi non stanno in nessuna fascia; energia e
    valence arrivano già come ranghi e passano come sono. Un brano senza
    una misura sta a metà: nessuna fascia lo attira, nessuna lo respinge.
    """
    columns = [ranks(bpm),
               np.asarray(energy, dtype=float),
               np.asarray(valence_rank, dtype=float),
               ranks(danceability)]
    matrix = np.column_stack(columns) if len(columns[0]) else np.empty((0, 4))
    return np.nan_to_num(matrix, nan=0.5)


def arc_costs(values: np.ndarray, chapter: int) -> np.ndarray:
    """Quanto ogni riga di `values` (N × 4, da `measures`) sta fuori dal
    capitolo, fra 0 e 1: `chapter_score` diviso per quattro, così il costo
    dell'arco è confrontabile con le tre distanze del costo di transizione.
    """
    out = np.zeros(len(values), dtype=np.float32)
    for column, name in enumerate(MEASURES):
        low, high = CHAPTERS[chapter][name]
        v = values[:, column]
        out += np.where(v < low, low - v, np.where(v > high, v - high, 0.0))
    return out / len(MEASURES)
