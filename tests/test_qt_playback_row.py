"""La fila di ascolto: da dove ⏮ e ⏭ del lettore vanno avanti e indietro.

Un ▶ singolo porta con sé le righe della tabella in cui lo si è premuto e
si ferma a fine brano; «Play all» porta la playlist e continua da sola.
In entrambi i casi si salta dentro la stessa fila, e ai bordi ci si ferma.
"""

from __future__ import annotations

from qt_app.state import AppState


def _heard(state: AppState) -> list:
    out: list = []
    state.now_playing_changed.connect(out.append)
    return out


def test_a_single_play_carries_its_row_and_skips_inside_it(qapp):
    state = AppState()
    heard = _heard(state)
    state.play("/b.mp3", ["/a.mp3", "/b.mp3", "/c.mp3"])
    assert state.can_skip(-1) and state.can_skip(1)
    state.skip(1)
    assert state.now_playing == "/c.mp3"
    assert not state.can_skip(1)
    state.skip(1)                                   # al bordo: fermo
    assert state.now_playing == "/c.mp3"
    state.skip(-1)
    state.skip(-1)
    assert state.now_playing == "/a.mp3"
    assert heard == ["/b.mp3", "/c.mp3", "/b.mp3", "/a.mp3"]


def test_a_single_play_ends_where_it_is(qapp):
    """Fine brano dopo un ▶ singolo: stop, anche se la fila continua —
    è il comportamento di sempre del ▶ di una riga."""
    state = AppState()
    state.play("/a.mp3", ["/a.mp3", "/b.mp3"])
    state.advance()
    assert state.now_playing is None
    assert not state.can_skip(1)


def test_a_play_without_a_row_has_nowhere_to_go(qapp):
    state = AppState()
    state.play("/a.mp3")
    assert not state.can_skip(-1) and not state.can_skip(1)


def test_play_all_continues_on_its_own_even_after_a_manual_skip(qapp):
    state = AppState()
    state.play_queue(["/a.mp3", "/b.mp3", "/c.mp3"])
    assert state.now_playing == "/a.mp3" and not state.can_skip(-1)
    state.skip(1)
    assert state.now_playing == "/b.mp3"
    state.advance()                                 # fine brano: da solo
    assert state.now_playing == "/c.mp3"
    state.advance()
    assert state.now_playing is None


def test_the_dock_buttons_follow_the_row(qapp, qtbot):
    from qt_app.widgets.player_dock import PlayerDock

    state = AppState()
    dock = PlayerDock(state)
    qtbot.addWidget(dock)
    state.play("/a.mp3", ["/a.mp3", "/b.mp3"])
    assert not dock._back.isEnabled() and dock._forward.isEnabled()
    dock._forward.click()
    assert state.now_playing == "/b.mp3"
    assert dock._back.isEnabled() and not dock._forward.isEnabled()
