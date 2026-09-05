"""La vista dello scaffale: una riga per playlist, com'è fatta la serata.

La scheda Playlist mostra UNA scaletta; per sapere se `house_climax` copre
l'ora e mezza di picco, se `funky_intro` è tutta in 8A o se lo stesso disco
sta in tre playlist bisognava aprirle una per una. Qui si legge tutto in
una tabella: quanti brani, la corsa dei BPM, l'energia media (un rango di
libreria, come ovunque), le tonalità coperte lungo la ruota, la durata, e
quanti brani stanno anche altrove — con i nomi, per il tooltip.

Solo lettura e solo numeri: i gesti sulle playlist stanno nella scheda
Playlist, e questa vista serve a capire dove manca materiale, non a
muoverlo.
"""

from __future__ import annotations

import math

import pandas as pd

from core.analysis.ordering import camelot_rank

# Le colonne della tabella, nell'ordine in cui si leggono. Quelle con
# l'underscore viaggiano ma non si mostrano.
COLUMNS = ("playlist", "tracks", "BPM", "energy", "keys", "length",
           "shared", "_shared_told")

_BARS = "▁▂▃▄▅▆▇█"


def energy_bar(mean: float | None) -> str:
    """L'energia media come numero e barretta: «0.52 ▅». Un rango, come
    ovunque nell'app — 0 il decimo più calmo che si possiede, 1 il più
    tirato."""
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "—"
    step = min(len(_BARS) - 1, int(mean * len(_BARS)))
    return f"{mean:.2f} {_BARS[step]}"


def length_told(seconds: float) -> str:
    """«1h 12m» o «38m»: la scala di una serata, non dei secondi."""
    minutes = int(round(seconds / 60))
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m"


def shared_tracks(playlists: dict[str, list[str]]) -> dict[str, list[str]]:
    """Per ogni brano che sta in più di una playlist, i nomi di tutte."""
    homes: dict[str, list[str]] = {}
    for name, paths in playlists.items():
        for path in dict.fromkeys(paths):
            homes.setdefault(path, []).append(name)
    return {path: names for path, names in homes.items() if len(names) > 1}


def _span(values: pd.Series) -> str:
    known = pd.to_numeric(values, errors="coerce").dropna()
    if not len(known):
        return "—"
    low, high = int(round(known.min())), int(round(known.max()))
    return str(low) if low == high else f"{low}–{high}"


def _keys(values: pd.Series) -> str:
    codes = {str(v) for v in values if isinstance(v, str) and v}
    ranked = sorted(codes, key=lambda c: (camelot_rank(c) is None,
                                          camelot_rank(c) or 0))
    return " ".join(ranked) if ranked else "—"


def shelf_rows(playlists: dict[str, list[str]], frame: pd.DataFrame,
               at_path: dict[str, int]) -> pd.DataFrame:
    """La tabella: una riga per playlist, nell'ordine dei nomi dato.

    I brani che non stanno sulla mappa contano nel numero ma non nelle
    misure — non si sa niente di loro — e non possono fare da doppi.
    """
    shared = shared_tracks(playlists)
    rows = []
    for name, paths in playlists.items():
        known = frame.loc[[at_path[p] for p in paths if p in at_path]] \
            if paths else frame.iloc[0:0]
        doubles = [p for p in paths if p in shared]
        told = "\n".join(
            f"{frame.at[at_path[p], 'name'] if p in at_path else p} — also in "
            + ", ".join(n for n in shared[p] if n != name)
            for p in doubles)
        energy = pd.to_numeric(known.get("energy", pd.Series(dtype=float)),
                               errors="coerce").dropna()
        seconds = pd.to_numeric(known.get("duration",
                                          pd.Series(dtype=float)),
                                errors="coerce").fillna(0).sum()
        rows.append({
            "playlist": name,
            "tracks": len(paths),
            "BPM": _span(known.get("bpm", pd.Series(dtype=float))),
            "energy": energy_bar(float(energy.mean()) if len(energy)
                                 else None),
            "keys": _keys(known.get("camelot", pd.Series(dtype=object))),
            "length": length_told(float(seconds)),
            "shared": len(doubles),
            "_shared_told": told,
        })
    return pd.DataFrame(rows, columns=list(COLUMNS))


def shelf_summary(playlists: dict[str, list[str]], frame: pd.DataFrame,
                  at_path: dict[str, int]) -> str:
    """La riga sotto la tabella: quante playlist, quanti brani, quanto
    dura la serata tutta insieme, quanti brani stanno in più posti."""
    paths = [p for ps in playlists.values() for p in ps]
    seconds = sum(
        float(frame.at[at_path[p], "duration"] or 0)
        for p in paths if p in at_path
        and not pd.isna(frame.at[at_path[p], "duration"]))
    shared = shared_tracks(playlists)
    told = (f"{len(playlists)} playlist(s) · {len(paths)} track(s) · "
            f"{length_told(seconds)}")
    if shared:
        told += f" · {len(shared)} track(s) in more than one playlist"
    return told
