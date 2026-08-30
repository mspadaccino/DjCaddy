"""La libreria in mano alla pagina Map: lo store, il frame e le tavole.

È il `library_frame` della pagina Streamlit più le colonne che là si
aggiungono al momento del disegno: qui si prepara tutto in un giro solo,
perché il giro gira in un thread del pool (87k righe non si leggono sul
filo della UI) e i pannelli ricevono l'oggetto pronto.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.analysis import energy, mood_scale
from core.analysis.map_store import MapStore
from core.analysis.mixing import TransitionCost


@dataclass
class Library:
    """Quello che ogni pannello della pagina riceve, già derivato.

    `cost` è il costo di transizione coi pesi CONDIVISI di Quick List,
    Sounds like it e Chain Maker: il pannello dei pesi lo muta, tutti lo
    interrogano — come gli slider unici della pagina Streamlit. La playlist
    invece legge i suoi scarti a pesi fermi (1,1,1) e si costruisce il
    proprio, come la sua sezione di là.
    """

    store: MapStore
    frame: pd.DataFrame
    common: dict[str, int]
    at_path: dict[str, int]
    cost: TransitionCost

    @property
    def placed(self) -> int:
        return len(self.frame)


def library_frame(store: MapStore) -> pd.DataFrame | None:
    """I brani piazzati con le misure derivate, o None se non ce ne sono.

    Energia e valence sono RANGHI sulla libreria intera, non numeri per
    brano: si calcolano su tutte le righe e si tagliano ai piazzati, come
    nella pagina Streamlit. Le liste dei generi e dei mood servono ai
    filtri; il numero firmato della valence al solo asse dei quadranti.
    """
    placed = store.placed
    if not placed:
        return None
    frame = pd.DataFrame(store.rows[:placed])
    frame["index"] = np.arange(len(frame))
    frame["energy"] = energy.from_rows(store.rows)[:placed]
    valence = np.asarray(mood_scale.from_rows(store.rows), dtype=float)
    frame["valence"] = valence[:placed]
    frame["valence_rank"] = energy.ranks(valence)[:placed]
    frame["x"], frame["y"] = store.coords[:, 0], store.coords[:, 1]
    frame["genre_list"] = frame["genres"].fillna("").str.split("; ")
    # Una mappa fatta prima che il mood si registrasse non ha la colonna:
    # meglio un filtro che non propone niente di un errore a metà pagina.
    moods = frame["moods"] if "moods" in frame \
        else pd.Series("", index=frame.index)
    frame["mood_list"] = moods.fillna("").str.split("; ")
    return frame


def load_library() -> tuple[MapStore, Library | None]:
    """Lo store e, se c'è qualcosa di piazzato, la libreria pronta."""
    store = MapStore.load()
    frame = library_frame(store)
    if frame is None:
        return store, None
    common = (mood_scale.popularity(list(frame["moods"]))
              if "moods" in frame else {})
    at_path = {row["path"]: i for i, row in enumerate(store.rows[:len(frame)])}
    cost = TransitionCost(store.coords[:len(frame)], frame["bpm"].tolist(),
                          frame["camelot"].tolist())
    return store, Library(store, frame, common, at_path, cost)
