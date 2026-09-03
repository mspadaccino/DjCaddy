"""I filtri della pagina Map, estratti in `core.viz.filters` (Fase 1).

Il comportamento è quello che il blocco dentro `render_filters` ha sempre
avuto: qui è scritto come test perché quel blocco, inline fra i widget, non
era fotografabile con uno snapshot.
"""

import numpy as np
import pandas as pd

from core.viz.filters import filter_tracks, span


def _tracks() -> pd.DataFrame:
    frame = pd.DataFrame([
        {"camelot": "8A", "bpm": 120.0, "danceability": 0.5,
         "genres": "House; Minimal", "moods": "Dark; Deep"},
        {"camelot": "9A", "bpm": 126.0, "danceability": 0.8,
         "genres": "Disco", "moods": "Happy"},
        {"camelot": "8B", "bpm": np.nan, "danceability": np.nan,
         "genres": "Techno", "moods": "Dark"},
        {"camelot": "3A", "bpm": 150.0, "danceability": 0.2,
         "genres": "House", "moods": "Party"},
    ])
    frame["genre_list"] = frame["genres"].str.split("; ")
    frame["mood_list"] = frame["moods"].str.split("; ")
    return frame


def _kept(**changes) -> list[int]:
    asked = {"genres": [], "moods": [], "keys": [],
             "bpm": (60.0, 200.0), "groove": (0.0, 1.0), **changes}
    return filter_tracks(_tracks(), **asked).index.tolist()


def test_no_filter_lets_everything_through():
    assert _kept() == [0, 1, 2, 3]


def test_a_track_stays_if_it_carries_any_of_the_chosen_genres():
    # Multi-label apposta: "Minimal" e "Deep House" possono essere veri
    # dello stesso brano, e basta uno dei generi scelti.
    assert _kept(genres=["Minimal", "Disco"]) == [0, 1]


def test_the_moods_follow_the_same_any_of_rule():
    assert _kept(moods=["Dark"]) == [0, 2]


def test_the_keys_narrow_to_the_chosen_ones():
    assert _kept(keys=["8A", "8B"]) == [0, 2]


def test_a_bpm_range_drops_what_falls_outside():
    assert _kept(bpm=(115.0, 130.0)) == [0, 1, 2]


def test_a_track_without_the_measure_is_not_excluded_by_a_range_on_it():
    """Non sappiamo dove cade, e farlo sparire sarebbe rispondere "no" a una
    domanda che non è stata posta."""
    assert 2 in _kept(bpm=(119.0, 121.0))
    assert 2 in _kept(groove=(0.45, 0.55))


def test_the_filters_combine():
    assert _kept(genres=["House"], bpm=(115.0, 130.0)) == [0]


def test_the_span_reads_the_real_extremes():
    assert span(_tracks(), "bpm", 60.0, 200.0) == (120.0, 150.0)


def test_an_empty_column_falls_back_to_the_offered_bounds():
    empty = pd.DataFrame({"bpm": [np.nan, np.nan]})
    assert span(empty, "bpm", 60.0, 200.0) == (60.0, 200.0)


def test_a_single_value_still_makes_a_drawable_slider():
    # Uno slider che parte e finisce nello stesso punto non si disegna.
    flat = pd.DataFrame({"bpm": [128.0, 128.0]})
    assert span(flat, "bpm", 60.0, 200.0) == (128.0, 129.0)


def test_genre_depth_looks_only_at_the_strongest_labels():
    # Il brano 0 è "House; Minimal": Minimal è il suo secondo genere.
    assert _kept(genres=["Minimal"]) == [0]
    assert _kept(genres=["Minimal"], genre_depth=1) == []
    assert _kept(genres=["Minimal"], genre_depth=2) == [0]
    assert _kept(genres=["House"], genre_depth=1) == [0, 3]
