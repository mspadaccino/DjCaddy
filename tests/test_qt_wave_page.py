"""La pagina Wave Qt: le righe della tabella cue e i due widget del disegno.

Girano solo col gruppo `qt` installato, offscreen come gli altri test Qt.
La logica delle righe (fine, battiti, slot) è quella di core già coperta da
`test_cue_export`: qui si prova la composizione della vista e i gesti.
"""

import os

import pytest

pytest.importorskip("PySide6", reason="gruppo poetry `qt` non installato")
pytest.importorskip("pytestqt", reason="gruppo poetry `qt` non installato")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt

from qt_app.pages.wave.cue_table import CueTable, view_rows
from qt_app.widgets.wave_review import WaveReview


def rows():
    return [
        {"id": "sec0", "kind": "phrase_start", "tag": "Intro", "start": 0.0},
        {"id": "sec1", "kind": "phrase_start", "tag": "Drop/Chorus",
         "start": 30.0},
        {"id": "vs0", "kind": "vocal_start", "tag": "Vocal start",
         "start": 10.0},
        {"id": "ve0", "kind": "vocal_end", "tag": "Vocal end", "start": 20.0},
    ]


# --- la vista delle righe ---------------------------------------------------

def test_view_rows_sorted_by_time_with_ends_beats_slots():
    shown = view_rows(rows(), bpm=120.0, analysis_end={"sec1": 60.0})

    assert [r["id"] for r in shown] == ["sec0", "vs0", "ve0", "sec1"]
    by_id = {r["id"]: r for r in shown}
    # La fine di sec0 è l'inizio di sec1; l'ultima tiene la fine rilevata.
    assert by_id["sec0"]["end"] == 30.0
    assert by_id["sec1"]["end"] == 60.0
    # Beats solo sulle frasi: 30 s a 120 BPM sono 60 battiti.
    assert by_id["sec0"]["beats"] == 60.0
    assert by_id["vs0"]["beats"] is None
    # Gli slot rekordbox: frasi sui pad A..H, la coppia vocale su UN loop.
    assert by_id["sec0"]["slot"] == "Hot cue A"
    assert by_id["sec1"]["slot"] == "Hot cue B"
    assert by_id["vs0"]["slot"] == by_id["ve0"]["slot"] == "Loop"


def test_view_rows_moving_a_start_moves_the_previous_end():
    moved = rows()
    moved[1]["start"] = 12.0        # il Drop trascinato molto prima
    shown = view_rows(moved, bpm=None, analysis_end={"sec1": 60.0})
    by_id = {r["id"]: r for r in shown}
    assert by_id["sec0"]["end"] == 12.0
    assert [r["id"] for r in shown][:2] == ["sec0", "vs0"]


def test_view_rows_no_bpm_no_beats():
    shown = view_rows(rows(), bpm=None, analysis_end={})
    assert all(r["beats"] is None for r in shown)


# --- la tabella -------------------------------------------------------------

def test_cue_table_rows_and_ids(qtbot):
    table = CueTable()
    qtbot.addWidget(table)
    shown = view_rows(rows(), bpm=120.0, analysis_end={"sec1": 60.0})
    table.set_rows(shown)

    assert table.rowCount() == 4
    assert [table.row_id(i) for i in range(4)] == \
        ["sec0", "vs0", "ve0", "sec1"]
    # Start in mm:ss, End della prima frase = inizio della seconda.
    assert table.item(0, 3).text() == "00:00.0"
    assert table.item(0, 4).text() == "00:30.0"
    assert table.item(3, 6).text() == "Hot cue B"


def test_cue_table_start_edit_emits_seconds(qtbot):
    table = CueTable()
    qtbot.addWidget(table)
    table.set_rows(view_rows(rows(), bpm=None, analysis_end={}))

    with qtbot.waitSignal(table.start_edited) as caught:
        table.item(0, 3).setText("1:07.3")
    assert caught.args == ["sec0", pytest.approx(67.3)]


def test_cue_table_bad_start_is_rejected(qtbot):
    table = CueTable()
    qtbot.addWidget(table)
    table.set_rows(view_rows(rows(), bpm=None, analysis_end={}))

    with qtbot.waitSignal(table.start_rejected):
        table.item(0, 3).setText("boh")


def test_cue_table_tag_edit_emits_label(qtbot):
    table = CueTable()
    qtbot.addWidget(table)
    table.set_rows(view_rows(rows(), bpm=None, analysis_end={}))

    with qtbot.waitSignal(table.tag_edited) as caught:
        table.item(1, 2).setText("Outro")
    assert caught.args == ["vs0", "Outro"]


def test_cue_table_play_and_delete_clicks(qtbot):
    table = CueTable()
    qtbot.addWidget(table)
    table.set_rows(view_rows(rows(), bpm=None, analysis_end={}))

    with qtbot.waitSignal(table.play_clicked) as caught:
        table.cellClicked.emit(2, 1)
    assert caught.args == ["ve0"]
    with qtbot.waitSignal(table.delete_clicked) as caught:
        table.cellClicked.emit(0, 7)
    assert caught.args == ["sec0"]


# --- l'onda -----------------------------------------------------------------

def test_wave_review_click_seeks_proportionally(qtbot):
    wave = WaveReview()
    qtbot.addWidget(wave)
    wave.resize(400, 200)
    wave.set_wave([0.5] * 100, ["#ff0000"] * 100, duration=200.0)

    with qtbot.waitSignal(wave.seek_requested) as caught:
        qtbot.mouseClick(wave, Qt.MouseButton.LeftButton, pos=wave.rect().center())
    assert caught.args[0] == pytest.approx(100.0, abs=2.0)


def test_wave_review_draws_with_markers_and_regions(qtbot):
    """Il paint non deve cadere con dati veri: marker, regioni, posizione.
    `grab()` forza un rendering completo anche offscreen."""
    wave = WaveReview()
    qtbot.addWidget(wave)
    wave.resize(400, 200)
    wave.set_wave([0.1, 0.9] * 400, ["#102030", "#a0b0c0"] * 400, 180.0)
    wave.set_regions([(10.0, 20.0), (100.0, 130.0)])
    wave.set_markers([{"t": 0.0, "label": "Intro", "color": "#8e9aa6"},
                      {"t": 90.0, "label": "Drop/Chorus", "color": "#e0503b"}])
    wave.set_position(45.0)
    image = wave.grab()
    assert not image.isNull() and image.width() > 0


def test_wave_review_without_wave_does_not_crash(qtbot):
    wave = WaveReview()
    qtbot.addWidget(wave)
    wave.resize(300, 200)
    assert not wave.grab().isNull()
