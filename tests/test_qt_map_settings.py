"""Map settings: i brani spariti dal disco si trovano e si tolgono da qui.
Gira solo col gruppo `qt` installato, senza finestre."""

import os

import pytest

pytest.importorskip("PySide6", reason="gruppo poetry `qt` non installato")
pytest.importorskip("pytestqt", reason="gruppo poetry `qt` non installato")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from core.analysis.map_profile import EMBEDDING_DIM, TrackProfile
from core.analysis.map_store import MapStore


def _profile(path, vector):
    return TrackProfile(path=path, duration=300.0, bpm=128.0, camelot="8A",
                        embedding=np.full(EMBEDDING_DIM, vector, dtype=np.float32))


def _store_with_a_ghost(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    kept, gone = lib / "a.mp3", lib / "b.mp3"
    kept.write_bytes(b"x")
    gone.write_bytes(b"y")
    store = MapStore.load(tmp_path / "map")
    store.append([_profile(kept, 1.0), _profile(gone, 2.0)])
    store.set_coords(np.array([[0.0, 0.0], [1.0, 0.0]]))
    gone.unlink()
    return store, lib, gone


def test_the_ghosts_are_counted_and_removed_on_request(qtbot, tmp_path):
    from qt_app.pages.map.settings import SettingsDialog

    store, lib, gone = _store_with_a_ghost(tmp_path)
    dialog = SettingsDialog()
    qtbot.addWidget(dialog)
    dialog._folder = lib
    dialog.set_store(store)
    qtbot.waitUntil(lambda: not dialog._checking and dialog._missing != [])
    assert dialog._missing == [str(gone)]
    assert dialog._prune.isEnabled()
    assert "1 track(s)" in dialog._missing_told.text()

    heard = []
    dialog.library_changed.connect(lambda: heard.append(True))
    dialog._remove_missing()
    qtbot.waitUntil(lambda: bool(heard))
    assert len(store) == 1 and store.rows[0]["path"].endswith("a.mp3")
    assert not dialog._prune.isEnabled()
    assert "Removed 1" in dialog._missing_told.text()


def test_nothing_is_checked_when_the_folder_is_not_reachable(qtbot, tmp_path):
    from qt_app.pages.map.settings import SettingsDialog

    store, lib, _ = _store_with_a_ghost(tmp_path)
    dialog = SettingsDialog()
    qtbot.addWidget(dialog)
    dialog._folder = tmp_path / "unplugged"
    dialog.set_store(store)
    assert not dialog._checking
    assert not dialog._prune.isEnabled()
    assert "not reachable" in dialog._missing_told.text()
    assert len(store) == 2
