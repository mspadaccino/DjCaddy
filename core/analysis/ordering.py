"""Gli ordinamenti semplici della playlist: per tempo, per energia, per
tonalità.

Il Magic sort minimizza il costo di transizione lungo la fila; questi
mettono in fila per UNA misura, e sono STABILI: due brani pari sulla misura
restano nell'ordine in cui erano. È quello che rende gli ordinamenti
componibili — prima per energia, poi per tempo, e dentro ogni tempo l'ordine
per energia sopravvive — e per lo stesso motivo il Magic sort, che parte
dal primo brano della fila, parte da quello che l'ordinamento prima gli ha
messo davanti.

La tonalità si ordina lungo la ruota Camelot: 1A, 1B, 2A, 2B … 12B. Numeri
vicini mixano, e lo stesso numero nei due modi è la relativa — così una
playlist ordinata per tonalità è anche una fila che si può suonare. Chi non
ha la misura va in coda, in qualunque verso si ordini: "non si sa" non è né
il più basso né il più alto.
"""

from __future__ import annotations

import math

import pandas as pd

from core.analysis.mixing import wheel_position

# Le misure per cui si può ordinare, con la colonna del frame che le porta.
FIELDS = {"bpm": "bpm", "energy": "energy", "key": "camelot"}


def camelot_rank(code) -> float | None:
    """La posizione lungo la ruota: 1A → 2, 1B → 3, 2A → 4 … 12B → 25."""
    place = wheel_position(code)
    if place is None:
        return None
    number, mode = place
    return float(number * 2 + mode)


def _measure(value, field: str) -> float | None:
    if field == "key":
        return camelot_rank(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def order_by(frame: pd.DataFrame, indices: list[int], field: str,
             descending: bool = False) -> list[int]:
    """`indices` riordinati per `field`, stabilmente, i senza-misura in coda."""
    column = FIELDS[field]
    known, unknown = [], []
    for i in indices:
        value = _measure(frame.at[i, column], field) \
            if column in frame else None
        (unknown if value is None else known).append((value, i))
    known.sort(key=lambda pair: pair[0], reverse=descending)
    return [i for _, i in known] + [i for _, i in unknown]
