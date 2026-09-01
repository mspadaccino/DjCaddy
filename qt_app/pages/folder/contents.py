"""Cosa contiene la cartella: formati, estensioni, e la pulizia mirata.

L'estensione dice cos'è il file, non un giudizio su di esso: gli elenchi
audio si guardano e si ascoltano ma non si spuntano — l'audio esce solo per
via di quarantena, dalle altre sezioni. Quello che audio non è (copertine,
`.nfo`, i resti di un download) si può cancellare da qui, dopo conferma.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPushButton,
                               QTableView, QVBoxLayout, QWidget)

from core.analysis.folder_scan import (AUDIO_FORMATS, NOT_JUNK, delete_files,
                                       human_size)
from qt_app import theme
from qt_app.pages.common import ConfirmBar, dim, reveal_in_files
from qt_app.state import AppState
from qt_app.widgets.track_table import PandasModel, TrackTable
from qt_app.workers import run_in_pool


def listed_rows(files) -> pd.DataFrame:
    """Le righe di un elenco per estensione, più grossi per primi."""
    return pd.DataFrame(
        [{"file": f.path.name, "folder": str(f.path.parent),
          "size": human_size(f.size),
          "modified": (datetime.fromtimestamp(f.mtime).strftime("%Y-%m-%d")
                       if f.mtime else "—"),
          "kind": f.fmt, "_path": str(f.path), "_bytes": f.size}
         for f in files],
        columns=["file", "folder", "size", "modified", "kind",
                 "_path", "_bytes"])


class ContentsPanel(QWidget):
    """Formati e estensioni, con l'elenco dei file di ognuna."""

    rescan_needed = Signal()

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._scan = None
        self._ext: str | None = None

        self._formats = QTableView()
        self._formats_model = PandasModel(parent=self._formats)
        self._formats.setModel(self._formats_model)
        self._formats.verticalHeader().setVisible(False)
        self._formats.verticalHeader().setDefaultSectionSize(24)
        self._formats.setShowGrid(False)
        self._formats.setAlternatingRowColors(True)
        self._formats.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)

        self._pick = QComboBox()
        self._pick.currentIndexChanged.connect(lambda _: self._on_pick())
        pick_row = QHBoxLayout()
        pick_row.addWidget(QLabel("Show the files with extension"))
        pick_row.addWidget(self._pick, stretch=1)

        self._told = dim("")
        self._told.setVisible(False)
        self._warn = dim("")
        theme.style(self._warn, lambda: f"color: {theme.WARN};")
        self._warn.setVisible(False)

        # Due tabelle, una spuntabile e una no: l'audio si guarda e si
        # ascolta, il resto si può condannare.
        self._audio_files = TrackTable(library_menu=False)
        self._audio_files.play_requested.connect(state.play)
        self._audio_files.row_activated.connect(state.play)
        self._audio_files.reveal_requested.connect(reveal_in_files)
        self._audio_files.setVisible(False)
        self._other_files = TrackTable(checkable=True, library_menu=False,
                                       playable=False)
        self._other_files.reveal_requested.connect(reveal_in_files)
        self._other_files.selection_paths_changed.connect(
            lambda _: self._refresh_doomed())
        self._other_files.setVisible(False)

        # I due bottoni della scelta in blocco, come nei duplicati: una
        # cartella di copertine da buttare sono centinaia di righe, e
        # spuntarle a una a una non è un gesto.
        self._pick_all = QPushButton("Select all")
        self._pick_all.clicked.connect(
            lambda: self._other_files.set_all_picked(True))
        self._pick_none = QPushButton("Select none")
        self._pick_none.clicked.connect(
            lambda: self._other_files.set_all_picked(False))
        self._pick_row = QWidget()
        pick_all_row = QHBoxLayout(self._pick_row)
        pick_all_row.setContentsMargins(0, 0, 0, 0)
        pick_all_row.addWidget(self._pick_all)
        pick_all_row.addWidget(self._pick_none)
        pick_all_row.addStretch(1)
        self._pick_row.setVisible(False)

        self._doomed_told = dim("")
        self._doomed_told.setVisible(False)
        self._confirm = ConfirmBar("Delete them", primary=True)
        self._confirm.activated.connect(self._on_delete)
        self._confirm.setVisible(False)
        self._unreadable = dim("")
        self._unreadable.setVisible(False)

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)
        box.addWidget(self._formats, stretch=1)
        box.addLayout(pick_row)
        box.addWidget(self._told)
        box.addWidget(self._warn)
        box.addWidget(self._audio_files, stretch=2)
        box.addWidget(self._pick_row)
        box.addWidget(self._other_files, stretch=2)
        box.addWidget(self._doomed_told)
        box.addWidget(self._confirm)
        box.addWidget(self._unreadable)

    # ------------------------------------------------------------------
    def set_scan(self, scan) -> None:
        self._scan = scan
        sizes = scan.size_by_format()
        self._formats_model.set_frame(pd.DataFrame(
            [{"format": fmt, "files": n, "size": human_size(sizes[fmt])}
             for fmt, n in scan.counts_by_format().most_common()]))
        self._formats.setColumnWidth(0, 260)

        by_ext = scan.counts_by_extension()
        ext_sizes = scan.size_by_extension()
        self._pick.blockSignals(True)
        self._pick.clear()
        self._pick.addItem("—", None)
        for ext, n in by_ext.most_common():
            self._pick.addItem(
                f"{ext} · {n:,} files · {human_size(ext_sizes[ext])}", ext)
        # L'estensione che si stava guardando resta in mano dopo un rescan.
        if self._ext is not None:
            at = self._pick.findData(self._ext)
            self._pick.setCurrentIndex(max(0, at))
        self._pick.blockSignals(False)
        self._on_pick()

        self._unreadable.setVisible(bool(scan.unreadable))
        if scan.unreadable:
            listed = "; ".join(f"{p} ({e})" for p, e in scan.unreadable[:5])
            self._unreadable.setText(
                f"⚠️ {len(scan.unreadable)} unreadable entries while "
                f"walking: {listed}"
                + ("…" if len(scan.unreadable) > 5 else ""))

    # ------------------------------------------------------------------
    def _on_pick(self) -> None:
        self._ext = self._pick.currentData()
        self._confirm.setVisible(False)
        self._doomed_told.setVisible(False)
        self._warn.setVisible(False)
        if self._scan is None or self._ext is None:
            self._told.setVisible(False)
            self._audio_files.setVisible(False)
            self._other_files.setVisible(False)
            self._pick_row.setVisible(False)
            return
        listed = self._scan.files_with_extension(self._ext)
        listed_bytes = sum(f.size for f in listed)
        is_audio = self._ext in AUDIO_FORMATS
        table = listed_rows(listed)
        self._told.setVisible(True)
        if is_audio:
            self._told.setText(
                f"{len(listed):,} files, {human_size(listed_bytes)} in "
                "all, biggest first. These are music, so there is nothing "
                "to tick: audio leaves by way of quarantine — the other "
                "sections — and never from here.")
            self._audio_files.set_tracks(table)
            self._audio_files.setVisible(True)
            self._other_files.setVisible(False)
            self._pick_row.setVisible(False)
        else:
            self._told.setText(
                f"{len(listed):,} files, {human_size(listed_bytes)} in "
                "all, biggest first. Tick what you want gone: these are "
                "DELETED, not quarantined — a folder on the same drive "
                "would free nothing. There is no undo.")
            if self._ext in NOT_JUNK:
                self._warn.setText(
                    f"{self._ext} files hold no audio, but they are not "
                    f"clutter either: {NOT_JUNK[self._ext]}. Look before "
                    "you tick.")
                self._warn.setVisible(True)
            self._other_files.set_tracks(table)
            self._other_files.set_all_picked(False)
            self._other_files.setVisible(True)
            self._pick_row.setVisible(bool(len(table)))
            self._audio_files.setVisible(False)
            self._refresh_doomed()

    def _refresh_doomed(self) -> None:
        frame = self._other_files.model_.frame
        picked = set(self._other_files.selected_paths())
        if not len(frame) or "_bytes" not in frame:
            return
        doomed = frame[frame["_path"].isin(picked)]
        freed = int(doomed["_bytes"].sum()) if len(doomed) else 0
        self._doomed_told.setText(
            f"{len(doomed):,} of {len(frame):,} ticked · "
            f"{human_size(freed)} would be freed")
        self._doomed_told.setVisible(True)
        self._confirm.setVisible(bool(len(doomed)))
        self._confirm.set_ask(
            f"Delete these {len(doomed):,} files ({human_size(freed)}) "
            "for good")

    def _on_delete(self) -> None:
        doomed = [Path(p) for p in self._other_files.selected_paths()]
        if not doomed:
            return

        def _job():
            return delete_files(doomed, dry_run=False)

        def _done(result) -> None:
            removed, freed, errors = result
            self._told.setText(
                f"{removed:,} files deleted, {human_size(freed)} freed."
                + (f" {len(errors)} skipped: "
                   + "; ".join(f"{q} ({e})" for q, e in errors[:3])
                   if errors else ""))
            self.rescan_needed.emit()

        run_in_pool(_job, _done,
                    lambda t: self._told.setText(f"Deleting failed: {t}"))
