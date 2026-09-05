"""Il pannello dei filtri della pagina Map: macro generi e generi collegati.
Gira solo col gruppo `qt` installato, senza finestre."""

import os

import pytest

pytest.importorskip("PySide6", reason="gruppo poetry `qt` non installato")
pytest.importorskip("pytestqt", reason="gruppo poetry `qt` non installato")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
from PySide6.QtCore import Qt


def _frame() -> pd.DataFrame:
    rows = [
        ("Electronic - House; Electronic - Deep House", "happy"),
        ("Electronic - Techno", "dark"),
        ("Rock - Punk", "angry"),
        ("Funk / Soul - Disco; Electronic - House", "happy"),
    ]
    frame = pd.DataFrame({
        "genres": [g for g, _ in rows], "moods": [m for _, m in rows],
        "bpm": [124.0, 130.0, 170.0, 118.0],
        "danceability": [0.6, 0.7, 0.3, 0.5],
        "camelot": ["8A", "9A", "3B", "5A"],
    })
    frame["genre_list"] = frame["genres"].str.split("; ")
    frame["mood_list"] = frame["moods"].str.split("; ")
    return frame


def _options(checklist) -> list[str]:
    return [checklist._list.item(i).text()
            for i in range(checklist._list.count())]


def _tick(checklist, name: str, on: bool = True) -> None:
    for i in range(checklist._list.count()):
        item = checklist._list.item(i)
        if item.text() == name:
            item.setCheckState(Qt.CheckState.Checked if on
                               else Qt.CheckState.Unchecked)


def _panel(qtbot):
    from qt_app.pages.map.filters import FiltersPanel
    panel = FiltersPanel()
    qtbot.addWidget(panel)
    panel.set_frame(_frame())
    return panel


def test_macro_genres_are_the_first_half_of_the_labels(qtbot):
    panel = _panel(qtbot)
    assert _options(panel._macros) == ["Electronic", "Rock", "Funk / Soul"]
    assert len(_options(panel._genres)) == 5


def test_ticking_a_macro_hides_every_other_leaf(qtbot):
    panel = _panel(qtbot)
    _tick(panel._macros, "Electronic")
    assert _options(panel._genres) == [
        "Electronic - House", "Electronic - Deep House", "Electronic - Techno"]
    assert "Rock - Punk" not in _options(panel._genres)


def test_a_macro_alone_lets_all_its_tracks_through(qtbot):
    panel = _panel(qtbot)
    _tick(panel._macros, "Electronic")
    assert list(panel.kept(_frame()).index) == [0, 1, 3]
    _tick(panel._macros, "Rock")
    assert list(panel.kept(_frame()).index) == [0, 1, 2, 3]


def test_a_leaf_under_the_macro_narrows_it_further(qtbot):
    panel = _panel(qtbot)
    _tick(panel._macros, "Electronic")
    _tick(panel._genres, "Electronic - Techno")
    assert list(panel.kept(_frame()).index) == [1]


def test_unticking_the_macro_brings_every_leaf_back_and_keeps_the_tick(qtbot):
    panel = _panel(qtbot)
    _tick(panel._macros, "Electronic")
    _tick(panel._genres, "Electronic - Techno")
    _tick(panel._macros, "Electronic", on=False)
    assert len(_options(panel._genres)) == 5
    assert panel._genres.checked() == ["Electronic - Techno"]
    assert list(panel.kept(_frame()).index) == [1]


def test_a_tick_on_a_leaf_that_disappears_falls_with_it(qtbot):
    panel = _panel(qtbot)
    _tick(panel._genres, "Rock - Punk")
    _tick(panel._macros, "Electronic")
    assert panel._genres.checked() == []
    assert list(panel.kept(_frame()).index) == [0, 1, 3]


def test_reset_clears_the_macros_and_restores_the_leaves(qtbot):
    panel = _panel(qtbot)
    _tick(panel._macros, "Rock")
    panel._on_reset()
    assert panel._macros.checked() == []
    assert len(_options(panel._genres)) == 5
    assert len(panel.kept(_frame())) == 4


def test_look_at_narrows_the_genres_to_the_strongest_ones(qtbot):
    from qt_app.pages.map.filters import GENRE_DEPTHS
    panel = _panel(qtbot)
    # Il brano 3 è "Funk / Soul - Disco; Electronic - House": Electronic è
    # il suo secondo macro genere.
    _tick(panel._macros, "Electronic")
    assert list(panel.kept(_frame()).index) == [0, 1, 3]
    panel._depth.setCurrentIndex(0)                # the 1st genre only
    assert panel.genre_depth() == 1
    assert list(panel.kept(_frame()).index) == [0, 1]
    _tick(panel._genres, "Electronic - Deep House")
    assert list(panel.kept(_frame()).index) == []  # è il secondo del brano 0
    panel._on_reset()
    assert panel.genre_depth() is None
    assert panel._depth.currentText() == GENRE_DEPTHS[-1][0]


# --- gli slider a due maniglie ---

def test_the_ranges_open_on_the_true_extremes_of_the_library(qtbot):
    panel = _panel(qtbot)
    assert panel._bpm.values() == (118.0, 170.0)
    assert panel._groove.values() == (0.3, 0.7)
    assert panel._bpm._low.text() == "118.0"
    assert panel._groove._high.text() == "0.700"
    assert len(panel.kept(_frame())) == 4


def test_a_range_narrows_the_tracks_and_reset_opens_it_again(qtbot):
    panel = _panel(qtbot)
    panel._bpm.set_values(120.0, 130.0)
    assert list(panel.kept(_frame()).index) == [0, 1]
    panel._on_reset()
    assert panel._bpm.values() == (118.0, 170.0)
    assert len(panel.kept(_frame())) == 4


def test_dragging_a_handle_moves_only_that_end_and_tells(qtbot):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    from qt_app.widgets.range_slider import RangeSlider
    slider = RangeSlider(decimals=1)
    qtbot.addWidget(slider)
    slider.set_span(100.0, 200.0)
    slider.resize(300, 30)
    bar = slider._bar
    bar.resize(214, 20)          # 200 pixel utili fra le due maniglie
    heard = []
    slider.valuesChanged.connect(lambda lo, hi: heard.append((lo, hi)))

    def press(x):
        bar.mousePressEvent(QMouseEvent(
            QMouseEvent.Type.MouseButtonPress, QPointF(x, 10), QPointF(x, 10),
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier))

    press(7 + 50)                # vicino alla maniglia bassa: la sposta
    assert slider.values() == (125.0, 200.0)
    press(7 + 150)               # vicino a quella alta
    assert slider.values() == (125.0, 175.0)
    press(7 + 130)               # la bassa non supera l'alta
    assert slider.values() == (125.0, 175.0) or slider.values()[0] <= 175.0
    assert heard[-1] == slider.values()
    assert slider._low.text() == f"{slider.values()[0]:.1f}"
