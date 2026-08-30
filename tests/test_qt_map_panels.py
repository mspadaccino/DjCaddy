"""Le parti pure dei pannelli della pagina Map Qt: righe di tabella e
libreria derivata. Girano solo col gruppo `qt` installato, come gli altri
test Qt; niente finestre — qui si provano funzioni, non widget."""

import os

import pytest

pytest.importorskip("PySide6", reason="gruppo poetry `qt` non installato")
pytest.importorskip("pytestqt", reason="gruppo poetry `qt` non installato")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd

from core.analysis.map_profile import EMBEDDING_DIM, TrackProfile
from core.analysis.map_store import MapStore
from core.analysis.mixing import TransitionCost
from qt_app.pages.map.library import library_frame
from qt_app.pages.map.playlist_panel import playlist_rows
from qt_app.pages.map.set_builder import numbered_rows


def library() -> pd.DataFrame:
    return pd.DataFrame([
        {"name": "one.mp3", "bpm": 124.0, "camelot": "8A", "energy": 0.65,
         "danceability": 0.61, "valence_rank": 0.9,
         "moods": "happy", "genres": "Electronic - House",
         "top_genre": "Electronic - House",
         "folder": "/x", "path": "/x/one.mp3", "duration": 300.0},
        {"name": "two.mp3", "bpm": 98.0, "camelot": "3B", "energy": 0.10,
         "danceability": 0.30, "valence_rank": 0.2,
         "moods": "dark", "genres": "Electronic - Techno",
         "top_genre": "Electronic - Techno",
         "folder": "/x", "path": "/x/two.mp3", "duration": 200.0},
        {"name": "three.mp3", "bpm": 120.0, "camelot": "8B", "energy": 0.5,
         "danceability": 0.5, "valence_rank": 0.5,
         "moods": "", "genres": "", "top_genre": "—",
         "folder": "/y", "path": "/y/three.mp3", "duration": 250.0},
    ])


def cost_of(frame: pd.DataFrame) -> TransitionCost:
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    return TransitionCost(coords, frame["bpm"].tolist(),
                          frame["camelot"].tolist())


# --- la tabella della playlist ---

def test_playlist_rows_number_and_cost_from_previous():
    frame = library()
    table = playlist_rows(frame, cost_of(frame), [0, 2], common={},
                          ch_lookup=None)
    assert list(table["#"]) == [1, 2]
    assert pd.isna(table.at[0, "from previous"])
    assert table.at[1, "from previous"] > 0
    assert "chapter" not in table.columns
    assert list(table["_path"]) == ["/x/one.mp3", "/y/three.mp3"]


def test_playlist_rows_chapter_pills():
    frame = library()
    table = playlist_rows(frame, cost_of(frame), [0, 2], common={},
                          ch_lookup={0: "Intro", 2: "Climax"})
    assert table.at[0, "chapter"] == ["Intro"]
    assert table.at[1, "chapter"] == ["Climax"]
    # La colonna sta subito dopo il numero, come nella tabella Streamlit.
    assert list(table.columns[:3]) == ["#", "chapter", "file"]


def test_playlist_rows_track_missing_a_chapter_stays_blank():
    frame = library()
    table = playlist_rows(frame, cost_of(frame), [0, 1], common={},
                          ch_lookup={0: "Intro"})
    assert table.at[1, "chapter"] == []


# --- le righe numerate della selezione ---

def test_numbered_rows_keep_the_given_order():
    frame = library()
    table = numbered_rows(frame, [2, 0], common={})
    assert list(table["#"]) == [1, 2]
    assert list(table["file"]) == ["three.mp3", "one.mp3"]
    assert list(table["_path"]) == ["/y/three.mp3", "/x/one.mp3"]


# --- la libreria derivata ---

def _profile(path, vector, bpm=128.0):
    return TrackProfile(path=path, duration=300.0, bpm=bpm, camelot="8A",
                        embedding=np.full(EMBEDDING_DIM, vector,
                                          dtype=np.float32))


def test_library_frame_carries_the_derived_columns(tmp_path):
    for name, vector in (("a.mp3", 1.0), ("b.mp3", 2.0)):
        (tmp_path / name).write_bytes(b"x")
    store = MapStore.load(tmp_path / "map")
    store.append([_profile(tmp_path / "a.mp3", 1.0, bpm=120.0),
                  _profile(tmp_path / "b.mp3", 2.0, bpm=124.0)])
    store.set_coords(np.array([[0.0, 0.0], [1.0, 1.0]]))

    frame = library_frame(store)
    for column in ("index", "energy", "valence", "valence_rank",
                   "x", "y", "genre_list", "mood_list"):
        assert column in frame.columns, column
    assert list(frame["index"]) == [0, 1]
    assert list(frame["x"]) == [0.0, 1.0]


def test_library_frame_is_none_before_the_projection(tmp_path):
    (tmp_path / "a.mp3").write_bytes(b"x")
    store = MapStore.load(tmp_path / "map")
    store.append([_profile(tmp_path / "a.mp3", 1.0)])
    assert library_frame(store) is None
