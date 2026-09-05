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
from core.analysis.shelf import Shelf
from qt_app.pages.map.library import library_frame
from qt_app.pages.map.playlist_panel import (appended, double_marks,
                                             playlist_doubles, playlist_rows)
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


def _fan(*degrees) -> np.ndarray:
    """Vettori unitari a ventaglio: la distanza di suono cresce con
    l'angolo."""
    angles = np.radians(degrees)
    return np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)


def cost_of(frame: pd.DataFrame) -> TransitionCost:
    return TransitionCost(_fan(0, 15, 30), frame["bpm"].tolist(),
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


# --- l'aggiunta in coda: chi c'è già non rientra, e si racconta ---

def test_appended_skips_what_is_already_there_and_tells():
    merged, skipped = appended(["/a/one.mp3", "/b/two.mp3"],
                               ["/b/two.mp3", "/c/three.mp3"])
    assert merged == ["/a/one.mp3", "/b/two.mp3", "/c/three.mp3"]
    assert skipped == ["/b/two.mp3"]


def test_appended_catches_the_double_inside_the_same_batch():
    """La stessa mandata può portare due volte lo stesso file (un m3u8 che
    lo elenca due volte): la seconda copia non entra e va nel resoconto —
    una volta sola, anche se i tentativi erano tre."""
    merged, skipped = appended([], ["/a/x.mp3", "/a/x.mp3", "/b/y.mp3",
                                    "/a/x.mp3"])
    assert merged == ["/a/x.mp3", "/b/y.mp3"]
    assert skipped == ["/a/x.mp3"]


def test_appended_stays_silent_when_everything_is_new():
    merged, skipped = appended(["/a/one.mp3"], ["/b/two.mp3"])
    assert merged == ["/a/one.mp3", "/b/two.mp3"]
    assert skipped == []


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


def test_double_marks_tint_only_the_extra_copies():
    """La prima occorrenza resta pulita: «via tutto il tinto» deve essere
    un gesto sicuro — di ogni pezzo ne resta una. Dove nome e suono valgono
    insieme veste il nome, che è il segnale più forte."""
    from qt_app import theme

    paths = ["/a/07 Foo - Bar.mp3", "/b/Foo - Bar (edit).mp3",
             "/c/Totally Else.mp3"]
    marks, told = double_marks(paths, np.ones((3, 4)))
    assert paths[0] not in marks               # il keeper non si tinge
    assert marks[paths[1]][0] is theme.TWIN_NAME_ROW
    assert marks[paths[2]][0] is theme.TWIN_SOUND_ROW
    assert "Copy of #1" in marks[paths[1]][1]
    assert "#1" in marks[paths[2]][1] and "#2" in marks[paths[2]][1]
    assert "1 repeat a song name from an earlier row" in told
    assert "1 sound nearly identical to an earlier row" in told


def test_double_marks_leave_the_first_take_of_a_trio_clean():
    paths = ["/a/Foo - Bar.mp3", "/b/07 Foo - Bar.mp3",
             "/c/Foo - Bar (club mix).mp3"]
    marks, told = double_marks(paths, None)
    assert paths[0] not in marks
    assert set(marks) == {paths[1], paths[2]}
    # Entrambe le copie puntano al keeper, non l'una all'altra.
    assert "Copy of #1" in marks[paths[1]][1]
    assert "Copy of #1" in marks[paths[2]][1]
    assert "2 repeat a song name from an earlier row" in told


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
    """Select all / none sulle quattro tabelle a spunte di Build a set (più
    il gruppo): le liste arrivano a venti righe, e prenderle tutte non è un
    gesto da fare riga per riga."""
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
    assert labels.count("Select all") == 4      # gruppo, Quick, Journey, Radio
    assert labels.count("Select none") == 4

    panel.set_choice(None, [0, 1, 2], [0, 1, 2])
    for table in (panel._group_table, panel._mixes_table,
                  panel._journey_table, panel._radio_table):
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
    panel.suggestions_changed.connect(rings.append)
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
    assert rings and rings[-1] == []             # anelli tolti dalla mappa
    assert not panel._tabs.tabText(0).endswith(")")
    # I settaggi restano: è la lista che riparte, non la pagina.
    assert panel._count.value() == 20

    # E si può rifare, che è tutto il punto.
    panel._on_ask_mixes()
    assert panel._mixes_ask.isHidden()


def test_reset_of_one_list_leaves_the_other_alone(qtbot, tmp_path,
                                                  monkeypatch):
    """Quick List e Radio sono due domande diverse sullo stesso seme:
    chiuderne una non chiude l'altra."""
    monkeypatch.setattr("qt_app.state._save_favourites", lambda paths: None)
    panel, state, _ = _radio_panel(qtbot, tmp_path)
    panel._radio_from.setCurrentIndex(1)
    panel.set_choice(0, [], [0])

    panel._on_ask_mixes()
    panel._on_ask_radio()
    asked = panel._asked_mixes
    panel._on_reset_radio()
    assert panel._asked_mixes == asked
    assert panel._radio_key is None


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


# --- salva / salva con nome ---
def test_save_rewrites_the_known_file_without_a_dialog(
        qtbot, tmp_path, monkeypatch):
    """«Save» è spento finché la playlist non ha un file; da quando una
    «Save as…» lo ha scelto resta acceso e riscrive quel file senza
    riaprire il dialogo — solo una conferma — anche subito dopo aver
    scritto. Al no non scrive."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    from pathlib import Path

    from qt_app.pages.map.library import Library
    from qt_app.pages.map.playlist_panel import PlaylistPanel
    from qt_app.state import AppState

    class _FakeStore:
        # `PlaylistPanel.set_library` e `_refresh` vogliono coordinate ed
        # embedding — qui bastano placeholder, nessuna riga li confronta.
        coords = np.zeros((3, 2))
        embeddings = np.zeros((3, 4))

    frame = library()
    at_path = {frame.at[i, "path"]: i for i in range(len(frame))}
    lib = Library(store=_FakeStore(), frame=frame, common={}, at_path=at_path,
                  cost=cost_of(frame))
    state = AppState()
    panel = PlaylistPanel(state, wire_table=lambda table: None,
                          shelf=Shelf(tmp_path / "shelf"))
    qtbot.addWidget(panel)
    panel.set_library(lib)
    monkeypatch.setattr(panel, "_tracks_for_export", lambda: [
        {"path": Path(p), "name": Path(p).stem, "artist": "",
         "duration": 1.0} for p in state.playlist])
    assert not panel._save_again.isEnabled()

    state.set_playlist(["/x/one.mp3"])
    assert not panel._save_again.isEnabled()   # nessun file ancora
    out = tmp_path / "set.m3u8"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "")))
    panel._save_m3u8.trigger()
    assert "one.mp3" in out.read_text()
    assert panel._save_again.isEnabled()       # il file c'è: acceso

    state.set_playlist(["/x/one.mp3", "/x/two.mp3"])
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: pytest.fail(
                            "«Save» must not open the dialog")))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(
        lambda *a, **k: QMessageBox.StandardButton.Yes))
    panel._save_again.click()
    assert "two.mp3" in out.read_text()
    assert panel._save_again.isEnabled()

    state.set_playlist(["/x/one.mp3"])
    monkeypatch.setattr(QMessageBox, "question", staticmethod(
        lambda *a, **k: QMessageBox.StandardButton.Cancel))
    panel._save_again.click()
    assert "two.mp3" in out.read_text()        # al no, il file resta com'era

    state.set_playlist([])
    assert not panel._save_again.isEnabled()   # niente da scrivere


def test_a_loaded_playlist_becomes_the_file_save_writes_to(
        qtbot, tmp_path, monkeypatch):
    """Caricare un .m3u8 e ritoccarlo: «Save» riscrive QUEL file, senza
    dialogo — prima il pannello lo dimenticava e il tasto restava spento."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    from pathlib import Path

    from qt_app.pages.map.library import Library
    from qt_app.pages.map.playlist_panel import PlaylistPanel
    from qt_app.state import AppState

    class _FakeStore:
        coords = np.zeros((3, 2))
        embeddings = np.zeros((3, 4))

    frame = library()
    at_path = {frame.at[i, "path"]: i for i in range(len(frame))}
    lib = Library(store=_FakeStore(), frame=frame, common={}, at_path=at_path,
                  cost=cost_of(frame))
    state = AppState()
    panel = PlaylistPanel(state, wire_table=lambda table: None,
                          shelf=Shelf(tmp_path / "shelf"))
    qtbot.addWidget(panel)
    panel.set_library(lib)
    monkeypatch.setattr(panel, "_tracks_for_export", lambda: [
        {"path": Path(p), "name": Path(p).stem, "artist": "",
         "duration": 1.0} for p in state.playlist])

    loaded = tmp_path / "night.m3u8"
    loaded.write_text("#EXTM3U\n/x/one.mp3\n", "utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(loaded), "")))
    # Il riquadro delle scelte: si risponde «Load as new playlist», che è
    # il primo bottone aggiunto.
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    monkeypatch.setattr(QMessageBox, "clickedButton",
                        lambda self: self.buttons()[0])
    panel._on_load()
    assert state.playlist == ["/x/one.mp3"]
    assert panel._save_again.isEnabled()

    state.set_playlist(["/x/one.mp3", "/x/two.mp3"])
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: pytest.fail(
                            "«Save» must not open the dialog")))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(
        lambda *a, **k: QMessageBox.StandardButton.Yes))
    panel._save_again.click()
    assert "two.mp3" in loaded.read_text()


# --- Magic sort ---

def test_magic_sort_only_reorders_ticked_rows(qtbot, tmp_path):
    """Con righe spuntate, il Magic sort tocca solo quelle: le altre — qui
    "a", in mezzo — restano nel loro slot. È il vincolo locale: si spunta
    un tratto della scaletta e solo quel tratto si riordina."""
    from qt_app.pages.map.library import Library
    from qt_app.pages.map.playlist_panel import PlaylistPanel
    from qt_app.state import AppState

    class _FakeStore:
        coords = np.zeros((4, 2))
        embeddings = np.zeros((4, 4))

    names = ["b", "a", "d", "c"]
    angles = [10, 0, 30, 20]
    frame = pd.DataFrame([
        {"name": n, "bpm": 120.0, "camelot": "8A", "energy": 0.5,
         "danceability": 0.5, "valence_rank": 0.5, "moods": "", "genres": "",
         "top_genre": "—", "folder": "/x", "path": f"/x/{n}.mp3",
         "duration": 200.0}
        for n in names])
    cost = TransitionCost(_fan(*angles), frame["bpm"].tolist(),
                          frame["camelot"].tolist())
    at_path = {frame.at[i, "path"]: i for i in range(len(frame))}
    lib = Library(store=_FakeStore(), frame=frame, common={}, at_path=at_path,
                  cost=cost)
    state = AppState()
    panel = PlaylistPanel(state, wire_table=lambda table: None,
                          shelf=Shelf(tmp_path / "shelf"))
    qtbot.addWidget(panel)
    panel.set_library(lib)
    state.set_playlist(["/x/b.mp3", "/x/a.mp3", "/x/d.mp3", "/x/c.mp3"])

    panel._table.set_picked({"/x/b.mp3", "/x/d.mp3", "/x/c.mp3"})
    panel._on_magic_sort()

    assert state.playlist == ["/x/b.mp3", "/x/a.mp3", "/x/c.mp3", "/x/d.mp3"]


def test_magic_sort_with_nothing_ticked_reorders_the_whole_playlist(qtbot, tmp_path):
    from qt_app.pages.map.library import Library
    from qt_app.pages.map.playlist_panel import PlaylistPanel
    from qt_app.state import AppState

    class _FakeStore:
        coords = np.zeros((4, 2))
        embeddings = np.zeros((4, 4))

    names = ["b", "a", "d", "c"]
    angles = [10, 0, 30, 20]
    frame = pd.DataFrame([
        {"name": n, "bpm": 120.0, "camelot": "8A", "energy": 0.5,
         "danceability": 0.5, "valence_rank": 0.5, "moods": "", "genres": "",
         "top_genre": "—", "folder": "/x", "path": f"/x/{n}.mp3",
         "duration": 200.0}
        for n in names])
    cost = TransitionCost(_fan(*angles), frame["bpm"].tolist(),
                          frame["camelot"].tolist())
    at_path = {frame.at[i, "path"]: i for i in range(len(frame))}
    lib = Library(store=_FakeStore(), frame=frame, common={}, at_path=at_path,
                  cost=cost)
    state = AppState()
    panel = PlaylistPanel(state, wire_table=lambda table: None,
                          shelf=Shelf(tmp_path / "shelf"))
    qtbot.addWidget(panel)
    panel.set_library(lib)
    state.set_playlist(["/x/b.mp3", "/x/a.mp3", "/x/d.mp3", "/x/c.mp3"])

    panel._on_magic_sort()

    assert state.playlist == ["/x/b.mp3", "/x/a.mp3", "/x/c.mp3", "/x/d.mp3"]


# --- Radio e appunti ---

def _radio_library():
    """Otto brani in un piano, a ventaglio: due attorno all'asse x, due
    attorno all'asse y, gli altri in mezzo o fuori. Venti gradi l'uno
    dall'altro almeno, che è sotto la soglia dei gemelli. Gli embedding
    stanno in uno store finto, che è tutto quello che la radio chiede allo
    store."""
    from types import SimpleNamespace
    from qt_app.pages.map.library import Library

    angles = [0, 20, 40, 90, 70, 110, -30, 150]
    vectors = np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))]
                        for a in angles], dtype=np.float32)
    frame = pd.DataFrame([
        {"name": f"t{n}.mp3", "bpm": 124.0, "camelot": "8A", "energy": 0.5,
         "danceability": 0.5, "valence": 0.1, "valence_rank": 0.5,
         "moods": "", "genres": "House", "top_genre": "House",
         "folder": "/r", "path": f"/r/t{n}.mp3", "duration": 200.0}
        for n in range(8)])
    at_path = {frame.at[i, "path"]: i for i in range(len(frame))}
    cost = TransitionCost(vectors, frame["bpm"].tolist(),
                          frame["camelot"].tolist())
    store = SimpleNamespace(embeddings=vectors)
    return Library(store=store, frame=frame, common={}, at_path=at_path,
                   cost=cost)


def _radio_panel(qtbot, tmp_path):
    from core.analysis.journal import Journal
    from qt_app.pages.map.set_builder import SetBuilderPanel
    from qt_app.state import AppState, _save_favourites

    state = AppState()
    state.favourites = []
    journal = Journal(tmp_path / "choices.jsonl")
    panel = SetBuilderPanel(state, wire_table=lambda table: None,
                            journal=journal)
    qtbot.addWidget(panel)
    panel.set_library(_radio_library())
    panel.set_pool(np.arange(8))
    return panel, state, journal


def test_radio_from_the_map_selection_tunes_around_the_group(qtbot, tmp_path,
                                                             monkeypatch):
    monkeypatch.setattr("qt_app.state._save_favourites", lambda paths: None)
    panel, state, _ = _radio_panel(qtbot, tmp_path)
    panel._radio_from.setCurrentIndex(1)               # Map selection
    panel.set_choice(None, [0, 1], [0, 1])
    assert panel._radio_ask.isEnabled()
    panel._variety.setValue(0.0)
    panel._count.setValue(5)
    panel._on_ask_radio()
    # Attorno all'asse x, senza i semi: resta fuori solo la 7, a 150 gradi.
    assert set(panel._radio_shown) == {2, 3, 4, 5, 6}
    assert len(panel._radio_table.selected_paths()) == 5    # tutte spuntate
    assert panel._tabs.tabText(3).endswith(f"({len(panel._radio_shown)})")


def test_radio_flag_switches_to_the_favourites(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr("qt_app.state._save_favourites", lambda paths: None)
    panel, state, _ = _radio_panel(qtbot, tmp_path)
    panel.set_choice(None, [0, 1], [0, 1])
    assert not panel._radio_ask.isEnabled()            # niente preferiti
    state.toggle_favourite("/r/t3.mp3")
    assert panel._radio_ask.isEnabled()
    panel._variety.setValue(0.0)
    panel._count.setValue(5)
    panel._on_ask_radio()
    # Attorno all'asse y: fuori la 0 e la 6, le più lontane dai 90 gradi.
    assert set(panel._radio_shown) == {1, 2, 4, 5, 7}
    # Un preferito in più cambia il gruppo: la lista si chiude da sé.
    state.toggle_favourite("/r/t0.mp3")
    assert panel._radio_key is None and not panel._radio_ask.isHidden()


def test_radio_again_turns_the_unticked_into_noes(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr("qt_app.state._save_favourites", lambda paths: None)
    panel, state, journal = _radio_panel(qtbot, tmp_path)
    panel._radio_from.setCurrentIndex(1)
    panel.set_choice(0, [], [0])                       # il seme solo
    panel._variety.setValue(0.0)
    panel._on_ask_radio()
    first = panel._radio_shown[0]
    row = list(panel._radio_table.model_.frame["_path"]).index(
        panel._lib.frame.at[first, "path"])
    panel._radio_table.toggle_pick(row)                # via il primo
    panel._on_radio_again()
    assert first not in panel._radio_shown
    assert panel._radio_negatives == [first]
    heard = []
    panel.replace_playlist.connect(heard.append)
    panel._send_radio("replace")
    assert heard and first not in heard[0]
    kinds = [line["kind"] for line in journal.read()]
    assert kinds == ["radio_again", "radio_sent"]
    assert journal.read()[1]["negatives"] == [f"/r/t{first}.mp3"]


def test_chain_pick_is_noted_with_the_roster_it_came_from(qtbot, tmp_path,
                                                          monkeypatch):
    monkeypatch.setattr("qt_app.state._save_favourites", lambda paths: None)
    panel, state, journal = _radio_panel(qtbot, tmp_path)
    panel._on_start_by_name(0)
    panel._roster_table.toggle_pick(0)
    panel._on_roster_add()
    panel._on_chain_send()
    lines = journal.read()
    assert [line["kind"] for line in lines] == ["pick", "chain_sent"]
    pick = lines[0]
    assert pick["source"]["path"] == "/r/t0.mp3"
    assert pick["chosen"] == [pick["shown"][0]["path"]]
    assert pick["shown"][0]["rank"] == 0 and pick["weights"] == [1.0, 1.0, 1.0]
    assert lines[1]["walk"] == ["/r/t0.mp3", pick["chosen"][0]]


def test_chain_trend_looks_ahead_of_the_source(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr("qt_app.state._save_favourites", lambda paths: None)
    panel, state, _ = _radio_panel(qtbot, tmp_path)
    # Catena 1 (20°) → 2 (40°). Ferma, da 2 viene prima la 4 (70°) e poi
    # la 0 (0°). Con la tendenza il punto sta verso i 60°: la 4 resta prima
    # e la 3 (90°) passa davanti alla 0.
    panel._on_chain_reorder(["/r/t1.mp3", "/r/t2.mp3"])
    assert panel._roster_told.text().endswith("</b>")
    still = list(panel._roster_table.model_.frame["_path"])
    panel._trend.setValue(1.0)
    assert "looking ahead" in panel._roster_told.text()
    ahead = list(panel._roster_table.model_.frame["_path"])
    assert still[:2] == ["/r/t4.mp3", "/r/t0.mp3"]
    assert ahead[:2] == ["/r/t4.mp3", "/r/t3.mp3"]


def test_auto_chain_grows_the_chain_and_is_noted_apart_from_picks(
        qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr("qt_app.state._save_favourites", lambda paths: None)
    panel, state, journal = _radio_panel(qtbot, tmp_path)
    panel._on_start_by_name(0)
    panel._auto_steps.setValue(3)
    panel._on_auto_chain()
    walk = panel._walk()
    assert len(walk) == 4 and walk[0] == "/r/t0.mp3"
    assert panel._source == walk[-1]
    lines = journal.read()
    assert [line["kind"] for line in lines] == ["auto_chain"]
    assert lines[0]["added"] == walk[1:] and lines[0]["steps"] == 3


# --- i pesi: slider, condivisi fino al Magic sort della playlist ---

def test_weight_sliders_move_by_tenths_and_tell_the_page(qtbot):
    """La riga dei pesi sta sopra le schede di destra, fuori da Build a
    set: è della pagina, perché la legge anche la Playlist."""
    from qt_app.pages.map.weights import WeightsBar

    bar = WeightsBar()
    qtbot.addWidget(bar)
    assert bar.weights() == (1.0, 1.0, 1.0)
    heard = []
    bar.changed.connect(lambda: heard.append(bar.weights()))
    bar._bpm.setValue(0.3)
    bar._key.setValue(2.0)
    assert heard[-1] == (1.0, 0.3, 2.0)
    assert bar._bpm._told.text() == "0.3"
    bar.set_weights(2.0, 1.0, 0.0)
    assert bar.weights() == (2.0, 1.0, 0.0)


def test_the_builder_takes_the_weights_from_the_page(qtbot, tmp_path,
                                                     monkeypatch):
    """Build a set non ha più slider suoi: riceve i pesi con `set_weights`,
    li scrive nel costo condiviso e rifà le liste aperte con quelli."""
    monkeypatch.setattr("qt_app.state._save_favourites", lambda paths: None)
    panel, state, journal = _radio_panel(qtbot, tmp_path)
    assert panel.weights() == (1.0, 1.0, 1.0)
    panel.set_weights(1.0, 0.3, 2.0)
    cost = panel._lib.cost
    assert (cost.w_sound, cost.w_bpm, cost.w_key) == (1.0, 0.3, 2.0)
    assert panel.weights() == (1.0, 0.3, 2.0)
    # E il quaderno annota i pesi della pagina, non i suoi.
    panel._on_start_by_name(0)
    panel._roster_table.toggle_pick(0)
    panel._on_roster_add()
    assert journal.read()[-1]["weights"] == [1.0, 0.3, 2.0]


def test_the_playlist_reads_the_shared_cost_and_its_weights(qtbot, tmp_path):
    """Il Magic sort e i numeri in tabella seguono gli slider di Build a
    set: prima la playlist si faceva un costo suo a pesi fermi, e muovere
    i pesi non le cambiava niente."""
    from qt_app.pages.map.library import Library
    from qt_app.pages.map.playlist_panel import PlaylistPanel
    from qt_app.state import AppState

    class _FakeStore:
        coords = np.zeros((3, 2))
        embeddings = np.zeros((3, 4))

    frame = library()
    at_path = {frame.at[i, "path"]: i for i in range(len(frame))}
    cost = cost_of(frame)
    lib = Library(store=_FakeStore(), frame=frame, common={}, at_path=at_path,
                  cost=cost)
    state = AppState()
    panel = PlaylistPanel(state, wire_table=lambda table: None,
                          shelf=Shelf(tmp_path / "shelf"))
    qtbot.addWidget(panel)
    panel.set_library(lib)
    assert panel._cost is cost
    state.set_playlist(["/x/one.mp3", "/x/two.mp3"])   # 8A→3B, 124→98
    all_three = panel._worst.text()
    cost.w_sound, cost.w_bpm, cost.w_key = 1.0, 0.0, 0.0
    panel.refresh_costs()
    sound_only = panel._worst.text()
    assert all_three != sound_only


# --- la lavagna sotto la tabella ---
def test_the_board_sits_under_the_table_in_the_same_tab(qtbot, tmp_path):
    """La lavagna è la vista della playlist e sta nella sua scheda, sotto la
    tabella e con uno splitter fra i due — non in una scheda «Chapters» a
    parte, dove sembrava un accessorio dei capitoli."""
    from PySide6.QtWidgets import QSplitter

    from qt_app.pages.map.library import Library
    from qt_app.pages.map.playlist_panel import PlaylistPanel
    from qt_app.state import AppState

    class _FakeStore:
        coords = np.zeros((3, 2))
        embeddings = np.zeros((3, 4))

    frame = library()
    at_path = {frame.at[i, "path"]: i for i in range(len(frame))}
    lib = Library(store=_FakeStore(), frame=frame, common={}, at_path=at_path,
                  cost=cost_of(frame))
    state = AppState()
    panel = PlaylistPanel(state, wire_table=lambda table: None,
                          shelf=Shelf(tmp_path / "shelf"))
    qtbot.addWidget(panel)
    panel.set_library(lib)
    assert not hasattr(panel, "board_widget")
    split = panel._playlist_controls
    assert isinstance(split, QSplitter) and split.count() == 2
    assert split.widget(0).isAncestorOf(panel._table)
    assert split.widget(1).isAncestorOf(panel._board)
    assert split.widget(1).isAncestorOf(panel._ch_create)
    # Vuota, la playlist non mostra né tabella né lavagna; piena, tutte e due.
    assert split.isHidden()
    state.set_playlist(["/x/one.mp3", "/x/two.mp3"])
    assert not split.isHidden()


# --- il Journey ---

def test_journey_from_the_seed_to_a_track_picked_by_name(qtbot, tmp_path,
                                                         monkeypatch):
    """Da 0 (0°) a 3 (90°) in quattro: la strada dritta passa per 2 (40°) e
    4 (70°). La fila esce spuntata, il conteggio va sulla linguetta, e
    mandarla alla playlist la scrive nel quaderno con i suoi estremi."""
    from qt_app.pages.map.set_builder import TAB_JOURNEY

    monkeypatch.setattr("qt_app.state._save_favourites", lambda paths: None)
    panel, state, journal = _radio_panel(qtbot, tmp_path)
    panel.set_choice(0, [], [0])
    assert panel._journey_ask.isEnabled()
    assert "open" in panel._journey_end_told.text()
    panel._on_journey_end(3)
    assert "t3.mp3" in panel._journey_end_told.text()
    panel._journey_count.setValue(4)
    panel._journey_arc.setValue(0.0)
    panel._on_ask_journey()
    assert panel._journey_shown == [0, 2, 4, 3]
    assert len(panel._journey_table.selected_paths()) == 4
    assert panel._tabs.tabText(TAB_JOURNEY).endswith("(4)")
    assert "chapter" not in panel._journey_table.model_.frame.columns

    # L'arco acceso scrive il capitolo di ogni posizione.
    panel._journey_arc.setValue(0.5)
    assert list(panel._journey_table.model_.frame["chapter"]) == \
        [["Intro"], ["Buildup"], ["Climax"], ["Release"]]

    heard = []
    panel.replace_playlist.connect(heard.append)
    panel._send_journey("replace")
    assert heard == [[0, 2, 4, 3]]
    line = journal.read()[-1]
    assert line["kind"] == "journey_sent"
    assert (line["start"], line["end"]) == ("/r/t0.mp3", "/r/t3.mp3")
    assert line["count"] == 4 and line["arc"] == 0.5


def test_journey_closes_when_the_start_moves_and_can_leave_from_the_chain(
        qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr("qt_app.state._save_favourites", lambda paths: None)
    panel, state, _ = _radio_panel(qtbot, tmp_path)
    panel.set_choice(0, [], [0])
    panel._journey_count.setValue(3)
    panel._on_ask_journey()
    assert panel._journey_key is not None and panel._journey_ask.isHidden()
    # Un altro seme: il viaggio di prima parlava di un altro punto.
    panel.set_choice(1, [], [1])
    assert panel._journey_key is None and not panel._journey_ask.isHidden()
    assert panel._journey_shown == []

    # Dalla catena: senza catena non si parte; con la catena, dall'ultimo.
    panel._journey_from.setCurrentIndex(1)
    assert not panel._journey_ask.isEnabled()
    assert "chain is empty" in panel._journey_told.text()
    panel._on_chain_reorder(["/r/t1.mp3", "/r/t2.mp3"])
    assert panel._journey_ask.isEnabled()
    assert "t2.mp3" in panel._journey_told.text()
    panel._on_ask_journey()
    assert panel._journey_shown[0] == 2 and len(panel._journey_shown) == 3

    # Dalla playlist: l'ultimo in fila, e la partenza segue la playlist.
    panel._journey_from.setCurrentIndex(2)
    assert not panel._journey_ask.isEnabled()
    assert "playlist is empty" in panel._journey_told.text()
    state.set_playlist(["/r/t5.mp3", "/r/t3.mp3"])
    assert "t3.mp3" in panel._journey_told.text()
    assert panel._journey_ask.isEnabled()


def test_radio_from_the_playlist_tunes_around_what_is_in_it(qtbot, tmp_path,
                                                           monkeypatch):
    """La playlist come gruppo: i suoi brani sono i semi, non si ripropongono,
    e una playlist che cambia chiude la lista come farebbe un preferito in
    più."""
    from qt_app.pages.map.set_builder import RADIO_PLAYLIST

    monkeypatch.setattr("qt_app.state._save_favourites", lambda paths: None)
    panel, state, journal = _radio_panel(qtbot, tmp_path)
    panel._radio_from.setCurrentIndex(RADIO_PLAYLIST)
    assert not panel._radio_ask.isEnabled()
    assert "playlist is empty" in panel._radio_told.text()
    state.set_playlist(["/r/t0.mp3", "/r/t1.mp3"])
    assert panel._radio_ask.isEnabled()
    panel._variety.setValue(0.0)
    panel._count.setValue(5)
    panel._on_ask_radio()
    # Attorno all'asse x, senza i due semi: resta fuori solo la 7, a 150°.
    assert set(panel._radio_shown) == {2, 3, 4, 5, 6}
    heard = []
    panel.append_playlist.connect(heard.append)
    panel._send_radio("append")
    assert heard and set(heard[0]) == {2, 3, 4, 5, 6}
    assert journal.read()[-1]["source"] == "Playlist"
    # La playlist cambia: la lista di prima parlava di un altro gruppo.
    state.set_playlist(["/r/t0.mp3", "/r/t1.mp3", "/r/t2.mp3"])
    assert panel._radio_key is None and not panel._radio_ask.isHidden()


# --- lo scaffale ---

def _shelf_panel(qtbot, tmp_path):
    from qt_app.pages.map.library import Library
    from qt_app.pages.map.playlist_panel import PlaylistPanel
    from qt_app.state import AppState

    class _FakeStore:
        coords = np.zeros((3, 2))
        embeddings = np.zeros((3, 4))

    frame = library()
    at_path = {frame.at[i, "path"]: i for i in range(len(frame))}
    lib = Library(store=_FakeStore(), frame=frame, common={}, at_path=at_path,
                  cost=cost_of(frame))
    state = AppState()
    shelf = Shelf(tmp_path / "shelf")
    panel = PlaylistPanel(state, wire_table=lambda table: None, shelf=shelf)
    qtbot.addWidget(panel)
    panel.set_library(lib)
    return panel, state, shelf


def test_an_empty_shelf_starts_with_the_default_playlist(qtbot, tmp_path):
    panel, state, shelf = _shelf_panel(qtbot, tmp_path)
    assert shelf.names() == ["Playlist"]
    assert panel.current_name() == "Playlist"
    assert panel._names.currentText() == "Playlist"
    assert state.playlist == []


def test_every_change_is_written_to_the_shelf_at_once(qtbot, tmp_path):
    panel, state, shelf = _shelf_panel(qtbot, tmp_path)
    state.set_playlist(["/x/one.mp3", "/x/two.mp3"])
    assert shelf.read("Playlist") == ["/x/one.mp3", "/x/two.mp3"]


def test_switching_keeps_both_playlists_as_they_were(qtbot, tmp_path,
                                                     monkeypatch):
    from PySide6.QtWidgets import QInputDialog
    panel, state, shelf = _shelf_panel(qtbot, tmp_path)
    state.set_playlist(["/x/one.mp3"])

    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("house_intro", True)))
    panel._new.click()
    assert panel.current_name() == "house_intro"
    assert state.playlist == []                    # il tavolo è vuoto
    assert shelf.read("Playlist") == ["/x/one.mp3"]   # l'altra aspetta
    state.set_playlist(["/x/two.mp3"])

    panel._names.setCurrentText("Playlist")
    assert state.playlist == ["/x/one.mp3"]
    assert shelf.read("house_intro") == ["/x/two.mp3"]
    assert shelf.active() == "Playlist"


def test_the_active_playlist_comes_back_at_the_next_start(qtbot, tmp_path,
                                                          monkeypatch):
    from PySide6.QtWidgets import QInputDialog
    panel, state, shelf = _shelf_panel(qtbot, tmp_path)
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("funky_climax", True)))
    panel._new.click()
    state.set_playlist(["/y/three.mp3"])

    again, state2, _ = _shelf_panel(qtbot, tmp_path)
    assert again.current_name() == "funky_climax"
    assert state2.playlist == ["/y/three.mp3"]


def test_a_taken_or_bad_name_is_refused(qtbot, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QInputDialog, QMessageBox
    panel, state, shelf = _shelf_panel(qtbot, tmp_path)
    answers = iter([("Playlist", True), ("a/b", True), ("", False)])
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: next(answers)))
    warned = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warned.append(a[2])))
    panel._new.click()
    assert len(warned) == 2                        # preso, poi non valido
    assert shelf.names() == ["Playlist"]           # e alla fine niente


def test_rename_keeps_the_tracks_and_the_tab_name(qtbot, tmp_path,
                                                  monkeypatch):
    from PySide6.QtWidgets import QInputDialog
    panel, state, shelf = _shelf_panel(qtbot, tmp_path)
    state.set_playlist(["/x/one.mp3"])
    told = []
    panel.shelf_changed.connect(told.append)
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("dance_buildup", True)))
    panel._rename.click()
    assert shelf.names() == ["dance_buildup"]
    assert shelf.read("dance_buildup") == ["/x/one.mp3"]
    assert state.playlist == ["/x/one.mp3"]
    assert told == ["dance_buildup"]


def test_deleting_the_last_playlist_leaves_an_empty_default(qtbot, tmp_path,
                                                            monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    panel, state, shelf = _shelf_panel(qtbot, tmp_path)
    state.set_playlist(["/x/one.mp3"])
    monkeypatch.setattr(QMessageBox, "question", staticmethod(
        lambda *a, **k: QMessageBox.StandardButton.Yes))
    panel._delete.click()
    assert shelf.names() == ["Playlist"]
    assert state.playlist == []
    assert panel.current_name() == "Playlist"


def test_a_loaded_file_enters_the_shelf_under_its_own_name(qtbot, tmp_path,
                                                           monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    panel, state, shelf = _shelf_panel(qtbot, tmp_path)
    state.set_playlist(["/x/two.mp3"])

    loaded = tmp_path / "night.m3u8"
    loaded.write_text("#EXTM3U\n/x/one.mp3\n", "utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(loaded), "")))
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    monkeypatch.setattr(QMessageBox, "clickedButton",
                        lambda self: self.buttons()[0])
    panel._on_load()
    assert panel.current_name() == "night"
    assert state.playlist == ["/x/one.mp3"]
    assert shelf.read("night") == ["/x/one.mp3"]
    assert shelf.read("Playlist") == ["/x/two.mp3"]   # quella di prima resta


def test_the_whole_shelf_goes_into_one_rekordbox_library(qtbot, tmp_path,
                                                         monkeypatch):
    """«Save as library» chiede: questa playlist o lo scaffale. Lo scaffale
    esce come una cartella «DjCaddy» con una playlist per nome, i brani
    letti dai file dello scaffale, quelli fuori mappa lasciati fuori."""
    import xml.etree.ElementTree as ET

    from PySide6.QtWidgets import QFileDialog, QMessageBox
    panel, state, shelf = _shelf_panel(qtbot, tmp_path)
    state.set_playlist(["/x/one.mp3", "/x/two.mp3"])
    shelf.write("house_buildup", ["/x/two.mp3", "/nowhere/ghost.mp3"])

    out = tmp_path / "shelf.xml"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "")))
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    # Qt riordina i bottoni per ruolo: si cerca «The whole shelf» per nome.
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: next(
        b for b in self.buttons() if b.text() == "The whole shelf"))
    monkeypatch.setattr("qt_app.pages.map.playlist_panel.read_title_artist",
                        lambda path: (path.stem, ""))
    panel._save_xml.trigger()

    root = ET.fromstring(out.read_text("utf-8"))
    folder = root.find("PLAYLISTS/NODE/NODE[@Name='DjCaddy']")
    assert [n.get("Name") for n in folder.findall("NODE")] == [
        "house_buildup", "Playlist"]
    assert root.find("COLLECTION").get("Entries") == "2"   # il fantasma no
    buildup = folder.find("NODE[@Name='house_buildup']")
    assert buildup.get("Entries") == "1"
    assert not panel._save_again.isEnabled()   # lo scaffale non è «il file»
