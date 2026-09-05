"""La scheda Shelf: tutte le playlist in una tabella, com'è fatta la serata.

Una riga per playlist, i numeri di `core.viz.shelf_view`. Si rifà quando
lo scaffale cambia — una playlist che si riscrive, un nome nuovo — ma solo
se la scheda si vede: altrimenti si segna e si rifà alla prossima
apertura. Il doppio clic su una riga porta quella playlist sul tavolo: la
vista è anche il modo più corto di girare fra le scalette.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QHeaderView, QLabel,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from core.analysis.shelf import Shelf
from core.viz.shelf_view import COLUMNS, shelf_rows, shelf_summary
from qt_app import theme

from .library import Library

SHOWN = [c for c in COLUMNS if not c.startswith("_")]


class ShelfPanel(QWidget):
    """`open_requested` porta il nome della playlist doppio-cliccata."""

    open_requested = Signal(str)

    def __init__(self, shelf: Shelf, parent=None) -> None:
        super().__init__(parent)
        self._shelf = shelf
        self._lib: Library | None = None
        self._dirty = True

        self._table = QTableWidget(0, len(SHOWN))
        self._table.setHorizontalHeaderLabels(SHOWN)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setAlternatingRowColors(True)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(SHOWN.index("playlist"),
                                    QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(SHOWN.index("keys"),
                                    QHeaderView.ResizeMode.Stretch)
        self._table.setToolTip(theme.hint(
            "One row per playlist on the shelf. BPM is the span, energy the "
            "mean rank across your library with a bar (0 the calmest tenth "
            "you own, 1 the hardest), keys the ones covered along the "
            "Camelot wheel, shared how many of its tracks are also in "
            "another playlist — hover the number for which. Double-click a "
            "row to bring that playlist onto the table."))
        self._table.itemDoubleClicked.connect(self._on_double_click)

        self._summary = QLabel("")
        self._summary.setObjectName("dim")
        self._summary.setWordWrap(True)

        box = QVBoxLayout(self)
        box.addWidget(self._table, stretch=1)
        box.addWidget(self._summary)

    def set_library(self, lib: Library) -> None:
        self._lib = lib
        self.invalidate()

    def invalidate(self) -> None:
        """Lo scaffale è cambiato: si rifà ora se si vede, sennò dopo."""
        self._dirty = True
        if self.isVisible():
            self.refresh()

    def showEvent(self, event) -> None:  # noqa: N802 (nome Qt)
        super().showEvent(event)
        if self._dirty:
            self.refresh()

    def refresh(self) -> None:
        self._dirty = False
        if self._lib is None:
            return
        playlists = {name: self._shelf.read(name)
                     for name in self._shelf.names()}
        rows = shelf_rows(playlists, self._lib.frame, self._lib.at_path)
        self._table.setRowCount(len(rows))
        # Dizionari e non namedtuple: itertuples rinomina le colonne che
        # cominciano per underscore, e `_shared_told` sparirebbe.
        for r, row in enumerate(rows.to_dict("records")):
            for c, column in enumerate(SHOWN):
                item = QTableWidgetItem(str(row[column]))
                if column in ("tracks", "shared", "BPM", "length"):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if column == "shared" and row["_shared_told"]:
                    item.setToolTip(row["_shared_told"])
                self._table.setItem(r, c, item)
        self._summary.setText(
            shelf_summary(playlists, self._lib.frame, self._lib.at_path))

    def _on_double_click(self, item: QTableWidgetItem) -> None:
        name = self._table.item(item.row(), SHOWN.index("playlist"))
        if name is not None:
            self.open_requested.emit(name.text())
