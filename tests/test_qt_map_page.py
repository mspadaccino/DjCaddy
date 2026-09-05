"""La pagina Map si costruisce intera: il fumo che un NameError dentro
`_build` deve far vedere PRIMA che l'app si apra. Gira solo col gruppo
`qt` installato, senza libreria e senza toccare i file dell'utente."""

import os

import pytest

pytest.importorskip("PySide6", reason="gruppo poetry `qt` non installato")
pytest.importorskip("pytestqt", reason="gruppo poetry `qt` non installato")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_the_map_page_builds_with_every_tab(qtbot, tmp_path, monkeypatch):
    # Niente lettura della mappa vera (gira nel pool) e niente scaffale o
    # preset in ~/Documents: la pagina si costruisce e basta.
    monkeypatch.setattr("qt_app.pages.map.page.run_in_pool",
                        lambda *a, **k: None)
    monkeypatch.setattr("core.analysis.shelf.user_dir", lambda: tmp_path)
    monkeypatch.setattr("core.analysis.presets.user_dir", lambda: tmp_path)
    monkeypatch.setattr("qt_app.state._save_favourites", lambda paths: None)
    monkeypatch.setattr("qt_app.state._load_favourites", lambda: [])
    from qt_app.pages.map.page import MapPage
    from qt_app.state import AppState

    page = MapPage(AppState())
    qtbot.addWidget(page)
    tabs = [page._panels.tabText(i) for i in range(page._panels.count())]
    assert tabs == ["🔎 Filters", "🎛️ Build a set", "🎵 Playlist",
                    "📚 Shelf", "★ Favourites"]
    assert (tmp_path / "Playlists" / "Playlist.m3u8").exists()
