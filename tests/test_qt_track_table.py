"""PandasModel e delegate della TrackTable: il contratto della Fase 2.

Girano solo con il gruppo `qt` installato: senza PySide6 (o senza
pytest-qt) si saltano, perché la CI del profilo Streamlit non deve
pretendere Qt. Offscreen, così la suite non fa lampeggiare finestre.
"""

import os

import pytest

pytest.importorskip("PySide6", reason="gruppo poetry `qt` non installato")
pytest.importorskip("pytestqt", reason="gruppo poetry `qt` non installato")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
from PySide6.QtCore import QModelIndex, Qt

from core.viz.track_columns import (ENERGY_COLORS, EMOTION_COLORS,
                                    GROOVE_COLORS, KEY_COLORS, energy_level)
from qt_app.widgets.track_table import (CHECK_A_COLUMN, CHECK_B_COLUMN,
                                        CHECK_COLUMN, PILLS_ROLE,
                                        PLAY_A_COLUMN, PLAY_B_COLUMN,
                                        PLAY_COLUMN, PandasModel, PillDelegate,
                                        TrackTable, pill_color, track_frame)


def library() -> pd.DataFrame:
    """Tre brani con le colonne che `core.viz.reading` legge."""
    return pd.DataFrame([
        {"name": "one.mp3", "bpm": 124.0, "camelot": "8A", "energy": 0.65,
         "danceability": 0.61, "valence_rank": 0.9,
         "moods": "happy; summer", "genres": "Electronic - House",
         "folder": "/x", "path": "/x/one.mp3"},
        {"name": "two.mp3", "bpm": 98.0, "camelot": "3B", "energy": 0.10,
         "danceability": 0.30, "valence_rank": 0.2,
         "moods": "dark", "genres": "Electronic - Techno; Electronic - Dub",
         "folder": "/x", "path": "/x/two.mp3"},
        {"name": "three.mp3", "bpm": None, "camelot": None, "energy": None,
         "danceability": None, "valence_rank": None,
         "moods": "", "genres": "", "folder": "/y", "path": "/y/three.mp3"},
    ])


def shown() -> pd.DataFrame:
    return track_frame(library(), common={})


# --- il modello ---

def test_model_hides_underscore_columns():
    model = PandasModel(shown())
    names = [model.headerData(c, Qt.Orientation.Horizontal)
             for c in range(model.columnCount())]
    assert "_path" not in names
    assert "file" in names and "key" in names
    assert model.rowCount() == 3


def test_model_formats_display_values():
    model = PandasModel(shown())
    names = [model.headerData(c, Qt.Orientation.Horizontal)
             for c in range(model.columnCount())]
    bpm, genres = names.index("BPM"), names.index("genres")
    # Il BPM intero senza il ".0" che pandas appiccica ai float con NaN.
    assert model.data(model.index(0, bpm)) == "124"
    assert model.data(model.index(2, bpm)) == ""
    # Le liste si leggono unite nel testo piano, intere nel ruolo pastiglie.
    assert model.data(model.index(1, genres)) == \
        "Electronic - Techno; Electronic - Dub"
    assert model.data(model.index(1, genres), PILLS_ROLE) == \
        ["Electronic - Techno", "Electronic - Dub"]


def test_model_sorts_numbers_with_missing_last():
    model = PandasModel(shown())
    names = [model.headerData(c, Qt.Orientation.Horizontal)
             for c in range(model.columnCount())]
    bpm = names.index("BPM")
    model.sort(bpm, Qt.SortOrder.AscendingOrder)
    files = list(model.frame["file"])
    assert files[:2] == ["two.mp3", "one.mp3"]   # 98 prima di 124
    assert files[2] == "three.mp3"               # senza BPM in fondo


def test_model_reorders_rows_on_drop():
    model = PandasModel(shown(), reorderable=True)
    heard = []
    model.order_changed.connect(heard.append)

    data = model.mimeData([model.index(2, 0)])
    # In testa: riga 2 lasciata cadere prima della riga 0. False a ragion
    # veduta — il modello ha già spostato, la vista non deve rimuovere.
    assert model.dropMimeData(data, Qt.DropAction.MoveAction,
                              0, 0, QModelIndex()) is False
    assert list(model.frame["file"]) == ["three.mp3", "one.mp3", "two.mp3"]
    assert heard == [["/y/three.mp3", "/x/one.mp3", "/x/two.mp3"]]


def test_model_flags_gate_the_drag():
    still = PandasModel(shown())
    assert not still.flags(still.index(0, 0)) & Qt.ItemFlag.ItemIsDragEnabled
    moving = PandasModel(shown(), reorderable=True)
    assert moving.flags(moving.index(0, 0)) & Qt.ItemFlag.ItemIsDragEnabled
    # Si lascia cadere FRA le righe (indice invalido), non sopra una riga.
    assert moving.flags(QModelIndex()) & Qt.ItemFlag.ItemIsDropEnabled
    assert not moving.flags(moving.index(0, 0)) & Qt.ItemFlag.ItemIsDropEnabled


# --- i colori delle pastiglie: le stesse scale di core/viz ---

def test_pill_colors_come_from_core_viz():
    assert pill_color("key", "8A") == KEY_COLORS["8A"]
    assert pill_color("energy", "7") == ENERGY_COLORS[6]
    assert pill_color("groove", "0.61") == GROOVE_COLORS[61]
    assert pill_color("emotion", "↑") == EMOTION_COLORS[0]
    assert pill_color("genres", "House", {"House": "#e0503b"}) == "#e0503b"


def test_pill_colors_stay_neutral_off_scale():
    assert pill_color("key", "13A") is None
    assert pill_color("energy", "0") is None
    assert pill_color("energy", "boh") is None
    assert pill_color("groove", "1.50") is None
    assert pill_color("genres", "House", {}) is None


def test_energy_pill_matches_core_reading():
    # La pastiglia in tabella scrive quello che `core.viz` scrive ovunque.
    table = shown()
    assert table["energy"].iloc[0] == [energy_level(0.65)]


# --- il delegate ---

def test_delegate_reads_cell_values(qtbot):
    table = TrackTable()
    qtbot.addWidget(table)
    table.set_tracks(shown(), {})
    model = table.model_
    names = [model.headerData(c, Qt.Orientation.Horizontal)
             for c in range(model.columnCount())]
    key = names.index("key")
    assert isinstance(table.itemDelegateForColumn(key), PillDelegate)
    assert PillDelegate._values(model.index(0, key)) == ["8A"]
    assert PillDelegate._values(model.index(2, key)) == []


def test_double_click_names_the_track(qtbot):
    table = TrackTable()
    qtbot.addWidget(table)
    table.set_tracks(shown(), {})
    with qtbot.waitSignal(table.row_activated) as heard:
        table._on_double_click(table.model_.index(1, 0))
    assert heard.args == ["/x/two.mp3"]


# --- la Fase 3: capitoli e selezione ---

def test_chapter_pill_uses_the_chapter_colors():
    from core.viz.chapters import CHAPTER_COLORS
    assert pill_color("chapter", "Intro") == CHAPTER_COLORS["Intro"]
    assert pill_color("chapter", "Climax") == CHAPTER_COLORS["Climax"]
    assert pill_color("chapter", "boh") is None


def test_selected_paths_follow_the_row_selection(qtbot):
    table = TrackTable()
    qtbot.addWidget(table)
    table.set_tracks(shown(), {})
    with qtbot.waitSignal(table.selection_paths_changed) as heard:
        table.selectRow(1)
    assert heard.args == [["/x/two.mp3"]]
    assert table.selected_paths() == ["/x/two.mp3"]
    table.clearSelection()
    assert table.selected_paths() == []


def test_checkable_grows_the_check_column_on_a_copy(qtbot):
    table = TrackTable(checkable=True)
    qtbot.addWidget(table)
    frame = shown()
    table.set_tracks(frame, {})
    assert table.model_.headerData(
        0, Qt.Orientation.Horizontal) == CHECK_COLUMN
    assert table.model_.headerData(
        1, Qt.Orientation.Horizontal) == PLAY_COLUMN
    # Il frame di chi chiama resta suo: le colonne nascono su una copia.
    assert CHECK_COLUMN not in frame.columns
    assert PLAY_COLUMN not in frame.columns


def test_every_table_leads_with_the_play_column(qtbot):
    table = TrackTable()
    qtbot.addWidget(table)
    table.set_tracks(shown(), {})
    assert table.model_.headerData(
        0, Qt.Orientation.Horizontal) == PLAY_COLUMN


def test_play_cell_sounds_the_row_without_selecting_it(qtbot):
    table = TrackTable()
    qtbot.addWidget(table)
    table.resize(640, 220)     # basta la geometria: visualRect non
                               # vuole una finestra vera, e una
                               # finestra vera lascia paint in coda
                               # che crollano al teardown (segfault)
    table.set_tracks(shown(), {})
    cell = table.visualRect(table.model_.index(1, 0)).center()
    with qtbot.waitSignal(table.play_requested) as heard:
        qtbot.mouseClick(table.viewport(), Qt.MouseButton.LeftButton,
                         pos=cell)
    assert heard.args == ["/x/two.mp3"]
    assert table.selected_paths() == []     # suonare non è scegliere


def test_play_a_and_play_b_sound_different_files(qtbot):
    """Le righe a due file (i duplicati B/C) hanno un ▶ per file: quello di
    A non deve mai suonare B, e viceversa."""
    table = TrackTable(checkable=True)
    qtbot.addWidget(table)
    table.resize(640, 220)
    table.set_tracks(pd.DataFrame([{
        CHECK_A_COLUMN: "", PLAY_A_COLUMN: "", "file A": "/x/a.mp3",
        CHECK_B_COLUMN: "", PLAY_B_COLUMN: "", "file B": "/y/a.mp3",
        "_path": "/y/a.mp3", "_path2": "/x/a.mp3"}]))
    cell_a = table.visualRect(table.model_.index(0, 1)).center()   # ▶ A
    cell_b = table.visualRect(table.model_.index(0, 4)).center()   # ▶ B
    with qtbot.waitSignal(table.play_requested) as heard:
        qtbot.mouseClick(table.viewport(), Qt.MouseButton.LeftButton,
                         pos=cell_a)
    assert heard.args == ["/x/a.mp3"]
    with qtbot.waitSignal(table.play_requested) as heard:
        qtbot.mouseClick(table.viewport(), Qt.MouseButton.LeftButton,
                         pos=cell_b)
    assert heard.args == ["/y/a.mp3"]
    assert table.selected_paths() == []         # suonare non è scegliere


def test_check_click_toggles_the_row_without_clearing_the_rest(qtbot):
    table = TrackTable(checkable=True)
    qtbot.addWidget(table)
    table.resize(640, 220)     # basta la geometria: visualRect non
                               # vuole una finestra vera, e una
                               # finestra vera lascia paint in coda
                               # che crollano al teardown (segfault)
    table.set_tracks(shown(), {})

    def tick(row):
        cell = table.visualRect(table.model_.index(row, 0)).center()
        qtbot.mouseClick(table.viewport(), Qt.MouseButton.LeftButton,
                         pos=cell)

    tick(0)
    tick(1)                     # la seconda spunta NON svuota la prima
    assert table.selected_paths() == ["/x/one.mp3", "/x/two.mp3"]
    tick(0)                     # rispuntare toglie la sola riga
    assert table.selected_paths() == ["/x/two.mp3"]


def test_a_plain_click_highlights_but_leaves_the_ticks_alone(qtbot):
    """Cliccare una riga per guardarla non deve disfare la scelta: il clic
    evidenzia, la casella prende."""
    table = TrackTable(checkable=True)
    qtbot.addWidget(table)
    table.resize(640, 220)     # basta la geometria: visualRect non
                               # vuole una finestra vera, e una
                               # finestra vera lascia paint in coda
                               # che crollano al teardown (segfault)
    table.set_tracks(shown(), {})
    table.toggle_pick(0)
    table.toggle_pick(1)
    # Un clic su una cella qualunque della terza riga (la colonna del nome).
    cell = table.visualRect(table.model_.index(2, 2)).center()
    qtbot.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, pos=cell)
    assert table.selected_paths() == ["/x/one.mp3", "/x/two.mp3"]
    assert table.selectionModel().isRowSelected(2)   # l'evidenziazione c'è


def test_playing_row_wears_the_yellow(qtbot):
    """La riga del brano in ascolto si tinge (BackgroundRole): per percorso,
    quindi il giallo segue il brano nei riordini e se ne va a fine ascolto."""
    from qt_app import theme

    table = TrackTable()
    qtbot.addWidget(table)
    table.set_tracks(shown(), {})
    model = table.model_
    ground = Qt.ItemDataRole.BackgroundRole
    assert model.data(model.index(1, 2), ground) is None
    table.set_playing("/x/two.mp3")
    assert model.data(model.index(1, 2), ground) == theme.PLAYING_ROW
    assert model.data(model.index(0, 2), ground) is None   # solo la sua
    table.set_playing(None)
    assert model.data(model.index(1, 2), ground) is None


def test_marked_rows_wear_their_tint_but_the_playing_yellow_wins(qtbot):
    """I possibili doppioni si tingono per percorso (BackgroundRole) e
    spiegano il perché nel tooltip; il giallo dell'ascolto resta più forte,
    e a marks vuoti la tinta se ne va."""
    from qt_app import theme

    table = TrackTable()
    qtbot.addWidget(table)
    table.set_tracks(shown(), {})
    model = table.model_
    ground = Qt.ItemDataRole.BackgroundRole
    tip = Qt.ItemDataRole.ToolTipRole
    table.set_marks({"/x/two.mp3": (theme.TWIN_NAME_ROW, "same song as #1")})
    assert model.data(model.index(1, 2), ground) == theme.TWIN_NAME_ROW
    assert model.data(model.index(1, 2), tip) == "same song as #1"
    assert model.data(model.index(0, 2), ground) is None
    assert model.data(model.index(0, 2), tip) is None
    table.set_playing("/x/two.mp3")
    assert model.data(model.index(1, 2), ground) == theme.PLAYING_ROW
    table.set_playing(None)
    assert model.data(model.index(1, 2), ground) == theme.TWIN_NAME_ROW
    table.set_marks({})
    assert model.data(model.index(1, 2), ground) is None


def test_ticks_survive_a_refresh_and_prune_the_missing(qtbot):
    table = TrackTable(checkable=True)
    qtbot.addWidget(table)
    table.set_tracks(shown(), {})
    table.toggle_pick(0)
    table.toggle_pick(2)
    table.set_tracks(shown(), {})                    # stesso frame, rifatto
    assert table.selected_paths() == ["/x/one.mp3", "/y/three.mp3"]
    smaller = shown()
    table.set_tracks(smaller[smaller["_path"] != "/x/one.mp3"], {})
    assert table.selected_paths() == ["/y/three.mp3"]
