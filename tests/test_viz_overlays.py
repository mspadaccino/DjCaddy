"""`overlay_figure`: i soli contorni che l'app Qt incolla sulla nuvola.

Il contratto che questi test proteggono è quello del canale a due tempi di
PlotlyView: nuvola una volta, contorni a ogni gesto, e la pagina che ne
esce dev'essere la STESSA figura che `build_figure` produce in un colpo
solo — stessi tracciati, nello stesso ordine, con la stessa annotazione.
"""

import numpy as np
import pandas as pd
import plotly.io as pio

from core.viz.map_figure import EMPTY_CLOUD, build_figure, overlay_figure


def _spelled(part) -> str:
    """Un tracciato (o un'annotazione) come JSON: i dict grezzi portano
    array numpy, che non si lasciano confrontare con `==`."""
    return pio.json.to_json_plotly(part.to_plotly_json())

COORDS = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                   [3.0, 0.0], [4.0, 0.0]])

MARKS = {"playlist": [0, 1], "seed": 2, "seed_name": "two.mp3",
         "selected": [3], "chained": [1], "mixes": [4], "alike": [0],
         "pl_selection": [1], "playing": 4}


def _cloud() -> pd.DataFrame:
    """Cinque punti disegnabili, con le colonne che `build_figure` legge."""
    return pd.DataFrame({
        "x": COORDS[:, 0], "y": COORDS[:, 1],
        "_size": [7.0] * 5,
        "genre_key": ["House", "House", "Techno", "Techno", "Dub"],
        "index": range(5),
        "name": [f"t{i}.mp3" for i in range(5)],
        "bpm": [120, 121, 122, 123, 124],
        "camelot": ["8A"] * 5,
        "genres": ["House"] * 5,
    })


def test_overlays_carry_no_cloud():
    figure = overlay_figure(COORDS, MARKS, dark=True)
    # Nessun tracciato porta customdata: niente brani cliccabili, solo segni.
    assert all(trace.customdata is None for trace in figure.data)
    names = {trace.name for trace in figure.data}
    assert {"playlist", "selected", "in the chain", "mixes out of it",
            "sounds like it", "current PL selection", "seed"} <= names
    # Il brano in ascolto non è un tracciato: è un'annotazione, che sta
    # sopra il canvas dei punti e che un lasso non attenua.
    assert "playing" not in names
    assert [a.name for a in figure.layout.annotations
            if a.name == "playing"] == ["playing"]
    # La playlist ha UN segno: il percorso. L'anello che marcava lo stesso
    # insieme è stato tolto come doppione.
    assert "in the playlist" not in names


def test_seed_name_is_the_only_annotation_with_text():
    figure = overlay_figure(COORDS, MARKS, dark=True)
    notes = [a for a in figure.layout.annotations if a.text]
    assert len(notes) == 1
    assert "two.mp3" in notes[0].text


def test_no_marks_means_no_traces():
    figure = overlay_figure(COORDS, {"playlist": [], "seed": None},
                            dark=False)
    assert len(figure.data) == 0
    assert not figure.layout.annotations


def test_cloud_plus_overlays_equals_the_full_figure():
    """Nuvola + contorni deve dare la figura intera, pezzo per pezzo: è
    esattamente l'incollatura che fa il JS di PlotlyView."""
    drawn = _cloud()
    top = ["House", "Techno"]
    full = build_figure(drawn, top, COORDS, dark=True, **MARKS)
    cloud = build_figure(drawn, top, COORDS, playlist=[], seed=None,
                         dark=True)
    overlays = overlay_figure(COORDS, MARKS, dark=True)

    glued = ([_spelled(t) for t in cloud.data]
             + [_spelled(t) for t in overlays.data])
    assert glued == [_spelled(t) for t in full.data]

    notes = ([_spelled(n) for n in cloud.layout.annotations]
             + [_spelled(n) for n in overlays.layout.annotations])
    assert notes == [_spelled(n) for n in full.layout.annotations]


def test_genre_labels_can_be_switched_off():
    """`labels=False` spegne i nomi dei generi scritti sulla nuvola: dove i
    gruppi si accavallano le scritte coprono i punti, e il flag in pagina
    le toglie senza toccare i tracciati."""
    drawn = pd.DataFrame({
        "x": [0.0, 1, 2, 5, 6, 7], "y": [0.0] * 6, "_size": [7.0] * 6,
        "genre_key": ["House"] * 3 + ["Techno"] * 3,
        "index": range(6), "name": [f"t{i}" for i in range(6)],
        "bpm": [120] * 6, "camelot": ["8A"] * 6, "genres": ["x"] * 6})
    coords = np.column_stack([drawn["x"], drawn["y"]])
    lit = build_figure(drawn, ["House", "Techno"], coords,
                       playlist=[], seed=None)
    off = build_figure(drawn, ["House", "Techno"], coords,
                       playlist=[], seed=None, labels=False)
    assert len(lit.layout.annotations) == 2
    assert not off.layout.annotations
    assert len(off.data) == len(lit.data)


def test_colour_by_none_greys_the_whole_cloud():
    """Il livello "none": nessuna chiave di genere, quindi nessun colore —
    tutta la nuvola in un solo tracciato grigio, senza etichette, e in
    legenda "tracks" invece di un "other" che non ha un principale."""
    from core.viz.map_figure import GENRE_LEVELS, SKIN, genre_level

    assert GENRE_LEVELS["none"] == "none"
    assert genre_level("Electronic - House", "none") == ""

    drawn = _cloud().assign(genre_key="")
    figure = build_figure(drawn, [], COORDS, playlist=[], seed=None)
    assert len(figure.data) == 1
    assert figure.data[0].name == "tracks"
    assert figure.data[0].marker.color == SKIN["light"]["other"]
    assert not figure.layout.annotations


def test_empty_cloud_never_grows():
    """EMPTY_CLOUD è condiviso: se qualcuno lo riempisse, ogni figura di
    contorni porterebbe punti fantasma."""
    overlay_figure(COORDS, MARKS, dark=True)
    assert len(EMPTY_CLOUD) == 0
