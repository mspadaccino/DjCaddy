"""I filtri della pagina Map: la regola, senza i widget.

La pagina disegna slider e ruota; qui c'è cosa vogliono dire — quali brani
passano, e fra quali estremi ha senso tendere una corsa. Sono la stessa
domanda per Streamlit e per Qt: quali brani sto guardando.
"""

from __future__ import annotations

import pandas as pd

from core.analysis.arc import CHAPTERS
from core.analysis.duplicates import folded


def span(frame: pd.DataFrame, column: str,
         floor: float, ceiling: float) -> tuple[float, float]:
    """Gli estremi veri di una colonna, per non offrire una corsa vuota.

    Uno slider 0..200 su una libreria che sta fra 110 e 130 è quasi tutto
    corsa morta. I due estremi devono comunque restare diversi fra loro,
    anche quando la colonna è vuota o porta un valore solo: uno slider che
    parte e finisce nello stesso punto non si disegna.
    """
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if not len(values):
        return (floor, ceiling)
    low, high = float(values.min()), float(values.max())
    return (low, high) if high > low else (low, low + 1.0)


def filter_tracks(frame: pd.DataFrame, genres: list[str], moods: list[str],
                  keys: list[str], bpm: tuple[float, float],
                  groove: tuple[float, float],
                  genre_depth: int | None = None,
                  energy: tuple[float, float] | None = None,
                  valence: tuple[float, float] | None = None) -> pd.DataFrame:
    """I brani che passano i filtri della pagina.

    `frame` deve portare `genre_list` e `mood_list` (le etichette già
    spezzate in liste): un brano resta se porta ALMENO UNO dei generi scelti
    e almeno uno dei mood scelti — le etichette sono multi-label apposta, e
    "Minimal" e "Deep House" possono essere vere dello stesso brano. Una
    lista vuota vuol dire "tutti".

    I generi di un brano sono in ordine di forza, il primo è quello che il
    modello sente di più: `genre_depth` dice quanti guardarne dall'alto —
    1 è "solo il genere principale", `None` è tutti, che è com'era.

    `energy` e `valence` sono intervalli sui RANGHI di libreria (colonne
    `energy` e `valence_rank`, 0..1): "il quarto più calmo che possiedi",
    non un numero assoluto. Senza intervallo non si guardano — le mappe
    fatte prima di queste due colonne non le hanno.

    Un brano senza BPM o senza groove non viene escluso da un intervallo su
    quel valore: non sappiamo dove cade, e farlo sparire sarebbe rispondere
    "no" a una domanda che non è stata posta.
    """
    kept = frame
    if genres:
        wanted = set(genres)
        kept = kept[kept["genre_list"].map(
            lambda tags: bool(wanted & set(tags[:genre_depth])))]
    if moods:
        wanted = set(moods)
        kept = kept[kept["mood_list"].map(
            lambda tags: bool(wanted & set(tags)))]
    if keys:
        kept = kept[kept["camelot"].isin(keys)]
    kept = kept[kept["bpm"].isna() | kept["bpm"].between(*bpm)]
    kept = kept[kept["danceability"].isna()
                | kept["danceability"].between(*groove)]
    for column, span_ in (("energy", energy), ("valence_rank", valence)):
        if span_ is not None and column in kept:
            kept = kept[kept[column].isna() | kept[column].between(*span_)]
    return kept


def chapter_ranges(chapter: dict, frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Le bande di un capitolo dell'arco come intervalli per i filtri.

    In `arc.CHAPTERS` le quattro misure sono percentili di libreria. Energia
    e mood lo sono anche sugli slider, e passano come sono; tempo e groove
    sugli slider sono numeri veri, e il percentile si legge sulla colonna:
    "il 15% più lento" di questa libreria è un BPM preciso, e cambia con la
    libreria. Una colonna vuota lascia l'intervallo tutto aperto sui bordi
    che i filtri offrono.
    """
    out = {"energy": chapter["arousal"], "valence": chapter["valence"]}
    for name, column, floor, ceiling in (("bpm", "bpm", 60.0, 200.0),
                                        ("groove", "danceability", 0.0, 1.0)):
        values = pd.to_numeric(frame[column], errors="coerce").dropna() \
            if column in frame else pd.Series(dtype=float)
        if len(values):
            low, high = chapter[name]
            out[name] = (float(values.quantile(low)),
                         float(values.quantile(high)))
        else:
            out[name] = span(frame, column, floor, ceiling) \
                if column in frame else (floor, ceiling)
    return out


def chapter_named(name: str) -> dict | None:
    return next((c for c in CHAPTERS if c["name"] == name), None)


def matching_tracks(frame: pd.DataFrame, pool, words: list[str]) -> list[int]:
    """Le posizioni che contengono TUTTE le parole, nel nome o nella cartella.

    A parole sparse e non a sottostringa: "madonna lucky" deve trovare
    "Madonna - Lucky Star (Extended Dance Remix)", che una ricerca contigua
    non trova. L'ordine non conta — chi cerca ricorda i pezzi di un titolo,
    non come sono disposti.

    Si guarda nel nome del file e nella cartella perché è lì che stanno
    artista e titolo: la mappa non conserva i tag, e in una libreria da DJ il
    nome del file li porta quasi sempre entrambi.
    """
    inside = frame.loc[list(pool)]
    hay = (inside["name"].fillna("") + " "
           + inside["folder"].fillna("")).map(folded)
    keep = pd.Series(True, index=hay.index)
    for word in words:
        # Anche le parole cercate, non solo il testo in cui si cerca: farlo
        # fare a chi chiama vuol dire che la funzione è giusta solo finché
        # tutti si ricordano di farlo. Vale per le maiuscole e vale per gli
        # accenti — un nome che arriva dal disco di un Mac ha la tilde
        # staccata dalla lettera e non combacia con quella che si digita.
        keep &= hay.str.contains(folded(word), regex=False)
    return keep[keep].index.tolist()
