"""Il frontend HTML dentro Qt: il payload arriva anche se mandato prima
che la pagina sia pronta. Gira solo col gruppo `qt` installato."""

import json
import os

import pytest

pytest.importorskip("PySide6", reason="gruppo poetry `qt` non installato")
pytest.importorskip("pytestqt", reason="gruppo poetry `qt` non installato")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd


def _frame() -> pd.DataFrame:
    frame = pd.DataFrame({"genres": ["Electronic - House"], "moods": ["happy"],
                          "bpm": [124.0], "danceability": [0.6],
                          "camelot": ["8A"]})
    frame["genre_list"] = frame["genres"].str.split("; ")
    frame["mood_list"] = frame["moods"].str.split("; ")
    return frame


def test_a_payload_sent_before_the_page_is_ready_is_not_lost(qtbot):
    """La pagina si dichiara pronta PRIMA che il canale col Qt si apra: il
    «ready» finiva nel vuoto e Qt non rimandava il payload arrivato troppo
    presto. La ruota Camelot dei filtri restava grigia — lo scenario è
    quello vero: il pannello si costruisce e la libreria arriva subito."""
    from qt_app.pages.map.filters import FiltersPanel

    panel = FiltersPanel()
    qtbot.addWidget(panel)
    # Grande quanto nella pagina: a widget minuscolo la pagina web carica
    # con un altro tempo e la corsa non si vede.
    panel.resize(420, 760)
    panel.show()
    panel.set_frame(_frame())          # → set_keys([]) a pagina non pronta
    wheel = panel._wheel

    seen = {}

    def coloured() -> bool:
        wheel.page().runJavaScript(
            "(function () { var p = document.querySelector('.slice path'); "
            "return p ? p.getAttribute('fill') : null; })()",
            lambda value: seen.__setitem__("fill", value))
        fill = seen.get("fill") or ""
        # Vuoto finché la pagina non c'è, grigio finché il payload non
        # arriva: si aspetta il colore vero.
        return fill.startswith("#") and fill != "#c7ccd4"

    qtbot.waitUntil(coloured, timeout=8000)
    assert seen["fill"] == "#89e6c7"   # 1B, la prima fetta: il suo colore


def test_the_payload_is_sent_again_when_the_channel_opens(qtbot):
    """La regola, senza corse: il payload dato prima si rimanda sia al
    «ready» della pagina sia all'apertura del canale — il secondo è quello
    che arriva di sicuro, e basta lui a colorare la ruota."""
    from qt_app.widgets.wheel_view import WheelView

    wheel = WheelView()
    qtbot.addWidget(wheel)
    wheel.set_keys(["8A"])
    sent = []
    wheel.set_payload = sent.append
    wheel._on_event({"type": "channel"})
    wheel._on_event({"type": "ready"})
    wheel._on_event({"type": "height", "height": 10})
    assert len(sent) == 2 and all(p["selected"] == ["8A"] for p in sent)


def test_the_board_fits_the_height_qt_gives_it(qtbot):
    """La lavagna entra tutta nel riquadro: l'SVG è alto quanto la pagina,
    non quanto la sua proporzione vorrebbe, così un divisore portato in
    basso non taglia più il fondo del disegno."""
    from qt_app.widgets.board_view import BoardView

    board = BoardView()
    qtbot.addWidget(board)
    board.resize(900, 260)
    with qtbot.waitSignal(board.loadFinished, timeout=8000):
        board.show()
    # Un solo giro di misura dopo che la pagina ha impaginato: una raffica
    # di runJavaScript dentro waitUntil affamava la pagina e la misura non
    # arrivava mai.
    qtbot.wait(400)
    seen = {}
    # Una stringa, non una lista: gli array del JS arrivano vuoti a Qt.
    board.page().runJavaScript(
        "(function () { var s = document.querySelector('#wrap svg'); "
        "return JSON.stringify(s ? [Math.round(s.getBoundingClientRect()"
        ".height), window.innerHeight] : null); })()",
        lambda value: seen.__setitem__("size", value))
    qtbot.waitUntil(lambda: "size" in seen, timeout=4000)
    drawn, page = json.loads(seen["size"])
    assert abs(drawn - page) <= 2      # alto quanto il riquadro, non 900·H/W
