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
from qt_app.pages.map.playlist_panel import (double_marks, playlist_doubles,
                                             playlist_rows)
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


# --- i possibili doppioni della playlist ---

def test_playlist_doubles_reads_the_same_song_through_its_disguises():
    """Numero di traccia in testa e parentesi del mix non distinguono: sono
    i travestimenti tipici dello stesso pezzo arrivato da fonti diverse."""
    paths = ["/a/07 New Order - Ruined In A Day.mp3",
             "/b/New Order - Ruined In A Day (club mix).mp3",
             "/c/Totally Else.mp3"]
    groups, pairs = playlist_doubles(paths, vectors=None)
    assert groups == [[0, 1]]
    assert pairs == []


def test_playlist_doubles_hears_twins_only_above_the_threshold():
    paths = ["/a/one.mp3", "/b/two.mp3", "/c/three.mp3"]
    vectors = np.array([
        [1.0, 0.0],
        [0.9, 0.43589],     # coseno 0.9 col primo: vicino, non gemello
        [0.9999, 0.01]])    # coseno ~1 col primo: gemello
    groups, pairs = playlist_doubles(paths, vectors)
    assert groups == []
    assert [(a, b) for a, b, _ in pairs] == [(0, 2)]
    assert pairs[0][2] > 0.99


def test_playlist_doubles_name_group_absorbs_its_own_sound_pair():
    """Due file dello stesso pezzo suonano anche uguali: la coppia per suono
    dentro un gruppo per nome non si ripete — vale il segnale più forte. Il
    gemello che si chiama in tutt'altro modo invece si aggiunge."""
    paths = ["/a/07 Foo - Bar.mp3", "/b/Foo - Bar (edit).mp3",
             "/c/Totally Else.mp3"]
    groups, pairs = playlist_doubles(paths, np.ones((3, 4)))
    assert groups == [[0, 1]]
    assert [(a, b) for a, b, _ in pairs] == [(0, 2), (1, 2)]


def test_double_marks_tint_name_over_sound_and_write_the_summary():
    from qt_app import theme

    paths = ["/a/07 Foo - Bar.mp3", "/b/Foo - Bar (edit).mp3",
             "/c/Totally Else.mp3"]
    marks, told = double_marks(paths, np.ones((3, 4)))
    assert marks[paths[0]][0] is theme.TWIN_NAME_ROW
    assert marks[paths[1]][0] is theme.TWIN_NAME_ROW
    assert marks[paths[2]][0] is theme.TWIN_SOUND_ROW
    assert "#2" in marks[paths[0]][1]          # il compagno per nome
    assert "#1" in marks[paths[2]][1]          # uno dei gemelli per suono
    assert "2 row(s) share a song name" in told
    assert "1 sound nearly identical" in told


def test_double_marks_stay_silent_on_a_clean_playlist():
    marks, told = double_marks(["/a/one.mp3", "/b/two.mp3"], None)
    assert marks == {}
    assert told is None


# --- le righe numerate della selezione ---

def test_numbered_rows_keep_the_given_order():
    frame = library()
    table = numbered_rows(frame, [2, 0], common={})
    assert list(table["#"]) == [1, 2]
    assert list(table["file"]) == ["three.mp3", "one.mp3"]
    assert list(table["_path"]) == ["/y/three.mp3", "/x/one.mp3"]


# --- il gruppo del lasso: i bottoni mandano le sole righe spuntate ---

def test_group_buttons_send_only_the_ticked_rows(qtbot):
    """Il lasso arriva già tutto spuntato — è già una scelta — ma da lì in
    poi comandano le caselle: tolta una spunta, i bottoni lavorano su meno.
    Prima partiva il gruppo intero comunque, e la colonna ✓ era un
    ornamento."""
    from qt_app.pages.map.library import Library
    from qt_app.pages.map.set_builder import SetBuilderPanel
    from qt_app.state import AppState

    frame = library()
    at_path = {frame.at[i, "path"]: i for i in range(len(frame))}
    lib = Library(store=None, frame=frame, common={}, at_path=at_path,
                  cost=cost_of(frame))
    panel = SetBuilderPanel(AppState(), wire_table=lambda table: None)
    qtbot.addWidget(panel)
    panel.set_library(lib)
    panel.set_choice(None, [0, 1, 2], [0, 1, 2])

    assert panel._group_table.selected_paths() == [
        "/x/one.mp3", "/x/two.mp3", "/y/three.mp3"]

    heard = []
    panel.append_playlist.connect(heard.append)
    panel._group_table.toggle_pick(0)           # via il primo
    panel._on_plain_append()
    assert heard == [[1, 2]]

    heard.clear()
    panel._on_sort_append()                     # ordina, ma i soli spuntati
    assert len(heard) == 1 and sorted(heard[0]) == [1, 2]


def test_unticking_survives_a_knob_touch(qtbot):
    """Rifare la tabella a ogni giro rimetterebbe le spunte appena tolte: si
    rifà solo quando il GRUPPO cambia."""
    from qt_app.pages.map.library import Library
    from qt_app.pages.map.set_builder import SetBuilderPanel
    from qt_app.state import AppState

    frame = library()
    at_path = {frame.at[i, "path"]: i for i in range(len(frame))}
    lib = Library(store=None, frame=frame, common={}, at_path=at_path,
                  cost=cost_of(frame))
    panel = SetBuilderPanel(AppState(), wire_table=lambda table: None)
    qtbot.addWidget(panel)
    panel.set_library(lib)
    panel.set_choice(None, [0, 1], [0, 1])
    panel._group_table.toggle_pick(0)
    panel.set_choice(None, [0, 1], [0, 1])      # stesso gruppo, altro giro
    assert panel._group_table.selected_paths() == ["/x/two.mp3"]
    panel.set_choice(None, [0, 2], [0, 2])      # gruppo NUOVO: si riparte
    assert panel._group_table.selected_paths() == [
        "/x/one.mp3", "/y/three.mp3"]


def test_every_pickable_list_can_be_taken_or_cleared_in_one_gesture(qtbot):
    """Select all / none sulle tre tabelle a spunte di Build a set: le liste
    arrivano a venti righe, e prenderle tutte non è un gesto da fare riga
    per riga."""
    from PySide6.QtWidgets import QPushButton

    from qt_app.pages.map.library import Library
    from qt_app.pages.map.set_builder import SetBuilderPanel
    from qt_app.state import AppState

    frame = library()
    at_path = {frame.at[i, "path"]: i for i in range(len(frame))}
    lib = Library(store=None, frame=frame, common={}, at_path=at_path,
                  cost=cost_of(frame))
    panel = SetBuilderPanel(AppState(), wire_table=lambda table: None)
    qtbot.addWidget(panel)
    panel.set_library(lib)

    labels = [b.text() for b in panel.findChildren(QPushButton)]
    assert labels.count("Select all") == 3      # gruppo, Quick List, Sounds
    assert labels.count("Select none") == 3

    panel.set_choice(None, [0, 1, 2], [0, 1, 2])
    for table in (panel._group_table, panel._mixes_table,
                  panel._alike_table):
        table.set_tracks(numbered_rows(frame, [0, 1, 2], common={}))
        table.set_all_picked(True)
        assert len(table.selected_paths()) == 3
        table.set_all_picked(False)
        assert table.selected_paths() == []


def test_reset_brings_back_the_button_that_makes_the_list(qtbot):
    """Fatta la lista, il bottone che la fa spariva e non tornava più senza
    cambiare seme: Reset riporta la schermata di partenza, spunte e anelli
    compresi, lasciando i settaggi dove sono."""
    from qt_app.pages.map.library import Library
    from qt_app.pages.map.set_builder import SetBuilderPanel
    from qt_app.state import AppState

    frame = library()
    at_path = {frame.at[i, "path"]: i for i in range(len(frame))}
    lib = Library(store=None, frame=frame, common={}, at_path=at_path,
                  cost=cost_of(frame))
    panel = SetBuilderPanel(AppState(), wire_table=lambda table: None)
    qtbot.addWidget(panel)
    panel.set_library(lib)
    panel.set_pool(np.array([0, 1, 2]))
    panel.set_choice(0, [], [0, 1, 2])

    rings = []
    panel.suggestions_changed.connect(lambda m, a: rings.append((m, a)))
    panel._on_ask_mixes()
    assert panel._mixes_ask.isHidden()           # il bottone si è fatto da parte
    panel._mixes_table.set_all_picked(True)
    assert panel._mixes_table.selected_paths()
    assert panel._tabs.tabText(0).endswith(")")  # il conteggio sulla linguetta

    rings.clear()
    panel._on_reset_mixes()
    assert not panel._mixes_ask.isHidden()
    assert panel._mixes_table.isHidden()
    assert panel._mixes_table.selected_paths() == []
    assert rings and rings[-1][0] == []          # anelli tolti dalla mappa
    assert not panel._tabs.tabText(0).endswith(")")
    # I settaggi restano: è la lista che riparte, non la pagina.
    assert panel._count.value() == 20

    # E si può rifare, che è tutto il punto.
    panel._on_ask_mixes()
    assert panel._mixes_ask.isHidden()


def test_reset_of_one_list_leaves_the_other_alone(qtbot):
    """Quick List e Sounds like it sono due domande diverse sullo stesso
    seme: chiuderne una non chiude l'altra."""
    from qt_app.pages.map.library import Library
    from qt_app.pages.map.set_builder import SetBuilderPanel
    from qt_app.state import AppState

    frame = library()
    at_path = {frame.at[i, "path"]: i for i in range(len(frame))}
    lib = Library(store=None, frame=frame, common={}, at_path=at_path,
                  cost=cost_of(frame))
    panel = SetBuilderPanel(AppState(), wire_table=lambda table: None)
    qtbot.addWidget(panel)
    panel.set_library(lib)
    panel.set_choice(0, [], [0, 1, 2])

    panel._on_ask_mixes()
    asked = panel._asked_mixes
    panel._on_reset_alike()
    assert panel._asked_mixes == asked
    assert panel._asked_alike is None


# --- la ricerca per nome ---

def test_search_picker_list_shows_only_with_matches(qtbot):
    """Da ferma la lista non si disegna: un riquadro vuoto sotto il campo
    ruberebbe altezza alle tabelle — nel Chain Maker se la mangiava ai
    candidati."""
    from qt_app.pages.map.set_builder import SearchPicker

    picker = SearchPicker("type a name")
    qtbot.addWidget(picker)
    picker.set_universe(library(), [0, 1, 2])
    assert picker._list.isHidden()
    picker._search.setText("one")
    assert not picker._list.isHidden()
    assert picker._list.count() == 1
    picker._search.setText("non c'è")
    assert picker._list.isHidden()


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
