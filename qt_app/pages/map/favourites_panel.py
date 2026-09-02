"""La scheda Favourites: i brani preferiti, con tutti i dettagli soliti.

La stella si mette e si toglie da qui come da Quick List, Sounds like it,
Chain Maker e dalla Playlist — è la stessa domanda ovunque venga fatta
(`AppState.favourites`, `TrackTable.favorite_requested`). Qui è anche il
gesto che porta i preferiti nella scaletta: si spuntano le righe volute e
"Add selected tracks to playlist" le manda in coda, come ogni altra lista
di Build a set.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from core.viz.track_columns import genre_colors
from qt_app import theme
from qt_app.state import AppState
from qt_app.widgets.track_table import TrackTable

from .library import Library
from .set_builder import numbered_rows


def _dim(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("dim")
    label.setWordWrap(True)
    return label


class FavouritesPanel(QWidget):
    """I preferiti: la tabella si rifà da sola a ogni tocco di stella,
    ovunque sia stato dato — non solo da qui."""

    append_playlist = Signal(list)          # gli INDICI di libreria scelti

    def __init__(self, state: AppState, wire_table, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._lib: Library | None = None
        self._build(wire_table)
        state.favourites_changed.connect(lambda _: self._refresh())

    def _build(self, wire_table) -> None:
        self._title = QLabel("<b>Favourites</b>")
        self._reset = QPushButton("🗑 Clear")
        self._reset.setToolTip("Clear every favourite. The audio files and "
                               "the playlist are untouched.")
        self._reset.clicked.connect(self._state.clear_favourites)
        header = QHBoxLayout()
        header.addWidget(self._title, stretch=1)
        header.addWidget(self._reset)

        self._empty = _dim(
            "No favourites yet: click the ☆ next to ▶ on the seed row, or "
            "the ★ column of Quick List, Sounds like it, Chain Maker and "
            "the Playlist, to add a track here.")

        self._table = TrackTable(checkable=True, favouritable=True)
        wire_table(self._table)

        pick_all = QPushButton("Select all")
        pick_all.clicked.connect(lambda: self._table.set_all_picked(True))
        pick_none = QPushButton("Select none")
        pick_none.clicked.connect(lambda: self._table.set_all_picked(False))
        pick_row = QHBoxLayout()
        pick_row.addWidget(pick_all)
        pick_row.addWidget(pick_none)
        pick_row.addStretch(1)

        self._add = QPushButton("➕ Add selected tracks to playlist")
        self._add.clicked.connect(self._on_add)

        box = QVBoxLayout(self)
        box.addLayout(header)
        box.addWidget(self._empty)
        box.addWidget(self._table, stretch=1)
        box.addLayout(pick_row)
        box.addWidget(self._add)

    # ------------------------------------------------------------------
    def set_library(self, lib: Library) -> None:
        self._lib = lib
        self._refresh()

    def _refresh(self) -> None:
        if self._lib is None:
            return
        at_path = self._lib.at_path
        indices = [at_path[p] for p in self._state.favourites if p in at_path]
        has = bool(indices)
        self._title.setText(
            f"<b>Favourites — {len(indices)} track(s)</b>" if has
            else "<b>Favourites</b>")
        self._empty.setVisible(not has)
        self._table.setVisible(has)
        self._add.setDisabled(not has)
        self._reset.setDisabled(not has)
        if not has:
            return
        frame, common = self._lib.frame, self._lib.common
        table = numbered_rows(frame, indices, common)
        self._table.set_tracks(
            table, genre_colors(frame, table["genres"], dark=theme.DARK))

    def _on_add(self) -> None:
        at_path = self._lib.at_path
        wanted = [at_path[p] for p in self._table.selected_paths()
                 if p in at_path]
        if wanted:
            self.append_playlist.emit(wanted)
