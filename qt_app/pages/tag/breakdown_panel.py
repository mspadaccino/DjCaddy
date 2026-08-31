"""La scomposizione dei tag: ogni elemento contato uno per uno.

Un tag è più cose insieme: "Electronic - House; Electronic - Tech House"
sono due generi su due livelli. La domanda "quanta house ho" non si
risponde guardando le stringhe intere — il conto lo fa
`core.analysis.tag_breakdown`, qui si sceglie una riga e si vedono i suoi
brani.
"""

from __future__ import annotations

import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QPushButton,
                               QTabWidget, QTableView, QVBoxLayout, QWidget)

from core.analysis.tag_breakdown import as_text, build_breakdown
from qt_app import theme
from qt_app.pages.common import dim, reveal_in_files
from qt_app.state import AppState
from qt_app.widgets.track_table import PandasModel, TrackTable


def file_name(field: str, kinds: list[str]) -> str:
    """Un nome di file leggibile anche con molte righe scelte."""
    clean = [t.replace("/", "-").replace(" ", "_") for t in kinds[:3]]
    rest = f"_and_{len(kinds) - 3}_more" if len(kinds) > 3 else ""
    return f"{field.lower()}_{'+'.join(clean)}{rest}.txt"


def spelled_kinds(kinds: list[str]) -> str:
    """I tipi scelti in forma leggibile: oltre i tre si conta, non si elenca."""
    if len(kinds) <= 3:
        return ", ".join(kinds)
    return f"{', '.join(kinds[:3])} and {len(kinds) - 3} more"


class _FieldPane(QWidget):
    """Un campo (GENRE o COMMENT): i suoi elementi, e i brani di ognuno."""

    def __init__(self, field: str, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._field = field
        self._breakdown = None
        self._kinds: list[str] = []

        self._told = dim("")
        self._types = QTableView()
        self._types_model = PandasModel(parent=self._types)
        self._types.setModel(self._types_model)
        self._types.verticalHeader().setVisible(False)
        self._types.verticalHeader().setDefaultSectionSize(24)
        self._types.setShowGrid(False)
        self._types.setAlternatingRowColors(True)
        self._types.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._types.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        self._types.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        # Una volta sola: set_frame azzera la selezione ma il selectionModel
        # resta lo stesso — riconnettersi a ogni giro moltiplica le chiamate.
        self._types.selectionModel().selectionChanged.connect(
            lambda *_: self._on_picked())

        self._picked_told = dim("")
        self._picked_told.setVisible(False)
        self._tracks = TrackTable(library_menu=False)
        self._tracks.play_requested.connect(state.play)
        self._tracks.row_activated.connect(state.play)
        self._tracks.reveal_requested.connect(reveal_in_files)
        self._tracks.setVisible(False)
        self._save = QPushButton("⬇ Save these paths as .txt")
        self._save.clicked.connect(self._on_save)
        self._save.setVisible(False)

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)
        box.addWidget(self._told)
        box.addWidget(self._types, stretch=2)
        box.addWidget(self._picked_told)
        box.addWidget(self._tracks, stretch=3)
        box.addWidget(self._save)

    def set_breakdown(self, breakdown) -> None:
        self._breakdown = breakdown
        rows = breakdown.rows()
        if not rows:
            self._told.setText(f"No {self._field.lower()} to break down "
                               "here.")
            self._types_model.set_frame(pd.DataFrame())
            return
        self._told.setText(
            f"{len(breakdown.counts):,} distinct across "
            f"{breakdown.tracks_with_any:,} tracks · "
            f"{breakdown.tracks_without:,} with no {self._field.lower()} at "
            "all. Click a row to see its tracks; ⌘/ctrl extends.")
        self._types_model.set_frame(pd.DataFrame(rows))
        self._types.setColumnWidth(0, 260)
        self._tracks.setVisible(False)
        self._picked_told.setVisible(False)
        self._save.setVisible(False)

    def _on_picked(self) -> None:
        if self._breakdown is None:
            return
        listed = self._breakdown.rows()
        rows = sorted({i.row() for i in
                       self._types.selectionModel().selectedRows()})
        self._kinds = [listed[i]["Type"] for i in rows if i < len(listed)]
        if not self._kinds:
            self._tracks.setVisible(False)
            self._picked_told.setVisible(False)
            self._save.setVisible(False)
            return
        tracks = self._breakdown.tracks_of(self._kinds)
        told = sum(listed[i]["Tracks"] for i in rows if i < len(listed))
        self._picked_told.setText(
            f"{spelled_kinds(self._kinds)} — {len(tracks):,} track(s)"
            + (f" — the rows add up to {told:,}, but a track carrying "
               "several of them is listed once"
               if told != len(tracks) else ""))
        self._picked_told.setVisible(True)
        self._tracks.set_tracks(pd.DataFrame(
            [{"file": b.name, "folder": str(b.parent), "_path": str(b)}
             for b in tracks], columns=["file", "folder", "_path"]))
        self._tracks.setVisible(True)
        self._save.setText(f"⬇ Save these {len(tracks):,} paths as .txt")
        self._save.setVisible(True)

    def _on_save(self) -> None:
        if self._breakdown is None or not self._kinds:
            return
        tracks = self._breakdown.tracks_of(self._kinds)
        label = ", ".join(self._kinds)
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Save paths", file_name(self._field, self._kinds),
            "Text (*.txt)")
        if not chosen:
            return
        try:
            from pathlib import Path
            Path(chosen).write_text(
                as_text(tracks, f"{self._field} = {label}"))
        except OSError as trouble:
            self._picked_told.setText(f"Could not save: {trouble}")
            self._picked_told.setStyleSheet(f"color: {theme.PRIMARY};")


class BreakdownPanel(QWidget):
    """Le due scomposizioni (genere e commento), una tab a testa."""

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._genre = _FieldPane("GENRE", state)
        self._comment = _FieldPane("COMMENT", state)
        tabs = QTabWidget()
        tabs.addTab(self._genre, "Genre")
        tabs.addTab(self._comment, "Comment")
        note = dim(
            "Every tag broken into its parts and counted. `;` separates one "
            "element from the next and ` - ` the parent genre from the "
            "child, so Electronic - House counts under both. `/` and `&` "
            "are left alone: they belong inside names like Funk / Soul.")
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)
        box.addWidget(note)
        box.addWidget(tabs, stretch=1)

    def set_coverage(self, readable) -> None:
        self._genre.set_breakdown(build_breakdown(readable, "genre"))
        self._comment.set_breakdown(build_breakdown(readable, "comment"))
