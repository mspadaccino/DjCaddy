"""La pagina Tag analysis: genere e mood nei tag, con i modelli Essentia.

Il punto di partenza non è il registro di cosa è stato tentato ma i FILE:
si legge cosa contengono adesso, e chi ha i tag incompleti diventa la coda
di lavoro — la stessa regola della pagina Streamlit.

Layout: la coda a sinistra (copertura, filtro, spunte), a destra le tab
Run / Breakdown / Background job / Environment. È lo scarto deliberato
dalla pagina-fiume di Streamlit, come i pannelli della pagina Map in
Fase 3: le funzioni sono le stesse, disposte per una finestra.

Su Windows l'analisi non esiste (essentia-tensorflow non pubblica wheel):
la pagina lo dice subito, e lettura dei tag, scomposizione e stato del job
restano utilizzabili — Windows è piattaforma di consumo, il Mac analizza.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QProgressBar,
                               QPushButton, QSplitter, QTabWidget,
                               QTableView, QVBoxLayout, QWidget)

from core.analysis.essentia_tags import (MODEL_DIR, MODELS, available,
                                         find_taggable, missing_models,
                                         scan_coverage)
from core.analysis.tag_tracking import DEFAULT_TRACKING_FILE, ProcessedTracker
from qt_app import theme
from qt_app.pages.common import FolderRow, Metric, dim, reveal_in_files
from qt_app.state import AppState
from qt_app.widgets.track_table import PandasModel, TrackTable
from qt_app.workers import Progress, run_in_pool

from .breakdown_panel import BreakdownPanel
from .job_panel import JobPanel
from .run_panel import RunPanel

# Sotto questa soglia i tag si leggono da soli appena scegli la cartella
# (~12 ms l'uno); sopra, il tempo lo si dichiara e lo si fa chiedere.
AUTO_READ_BELOW = 2000

FILTERS = ["genre or comment", "genre", "comment", "both",
           "everything (no filter)"]


def queue_rows(selected) -> pd.DataFrame:
    """Le righe della tabella della coda, da una lista di TagCoverage."""
    return pd.DataFrame(
        [{"file": c.path.name,
          "GENRE": c.genre or "❌ missing",
          "COMMENT": c.comment or "❌ missing",
          "folder": str(c.path.parent), "_path": str(c.path)}
         for c in selected],
        columns=["file", "GENRE", "COMMENT", "folder", "_path"])


def filtered_coverage(coverage, choice: str):
    """Chi entra in coda con il filtro scelto (le voci di FILTERS)."""
    if choice.startswith("everything"):
        return list(coverage.readable)
    return coverage.missing(
        genre=choice in ("genre or comment", "genre", "both"),
        comment=choice in ("genre or comment", "comment", "both"),
        require_both=choice == "both")


class TagPage(QWidget):
    """Coda di lavoro dai file, analisi coi modelli, job in background."""

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._root: Path | None = None
        self._scope: list[Path] = []
        self._coverage = None
        self._selected: list = []
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        self._folder = FolderRow("Folder to analyze")
        self._folder.chosen.connect(self._on_folder)

        self._status = dim("Choose a folder to start: every audio track "
                           "inside it, subfolders included. Analysis writes "
                           "genre and mood into the file tags — the mood "
                           "goes into the default comment field, the one "
                           "rekordbox actually displays.")
        if sys.platform == "win32" and not available():
            self._status.setText(
                "Analysis needs the Mac: essentia-tensorflow has no Windows "
                "wheels, so tagging cannot run here. Reading tags, the "
                "breakdown and the job state still work — this machine "
                "consumes the library, the Mac analyzes it.")
            self._status.setStyleSheet("color: #ffb454;")

        self._bar = QProgressBar()
        self._bar.setVisible(False)
        self._read = QPushButton("")
        self._read.clicked.connect(self._on_read_tags)
        self._read.setVisible(False)

        # --- sinistra: la copertura e la coda ---
        self._read_told = Metric("Tracks read")
        self._with_genre = Metric("With genre")
        self._with_comment = Metric(
            "With comment", "Only the default comment counts — the one "
                            "rekordbox shows.")
        self._complete = Metric("Complete")
        numbers = QHBoxLayout()
        for metric in (self._read_told, self._with_genre,
                       self._with_comment, self._complete):
            numbers.addWidget(metric)

        self._choice = QComboBox()
        self._choice.addItems(FILTERS)
        self._choice.currentTextChanged.connect(lambda _: self._refresh_queue())
        self._all = QPushButton("Select all")
        self._all.clicked.connect(lambda: self._queue_table.set_all_picked(True))
        self._none = QPushButton("Select none")
        self._none.clicked.connect(lambda: self._queue_table.set_all_picked(False))
        choice_row = QHBoxLayout()
        choice_row.addWidget(QLabel("Work on tracks missing…"))
        choice_row.addWidget(self._choice, stretch=1)
        choice_row.addWidget(self._all)
        choice_row.addWidget(self._none)

        self._queue_told = dim("")
        self._queue_table = TrackTable(checkable=True, library_menu=False)
        self._queue_table.play_requested.connect(self._state.play)
        self._queue_table.row_activated.connect(self._state.play)
        self._queue_table.reveal_requested.connect(reveal_in_files)
        self._queue_table.selection_paths_changed.connect(
            lambda _: self._push_queue())

        self._unreadable_told = dim("")
        self._unreadable_told.setVisible(False)

        left_box = QWidget()
        left = QVBoxLayout(left_box)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(6)
        left.addLayout(numbers)
        left.addLayout(choice_row)
        left.addWidget(self._queue_told)
        left.addWidget(self._queue_table, stretch=1)
        left.addWidget(self._unreadable_told)

        # --- destra: i pannelli ---
        self._run = RunPanel()
        self._run.tags_written.connect(self._on_tags_written)
        self._breakdown = BreakdownPanel(self._state)
        self._job = JobPanel(lambda: (self._run.settings_box.settings(),
                                      self._run.settings_box.workers.value()))
        panels = QTabWidget()
        panels.addTab(self._run, "⚙️ Run")
        panels.addTab(self._breakdown, "🧩 Breakdown")
        panels.addTab(self._job, "🔁 Background job")
        panels.addTab(self._environment(), "🧪 Environment")

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left_box)
        split.addWidget(panels)
        split.setSizes([820, 680])

        top = QHBoxLayout()
        top.addWidget(self._folder, stretch=1)
        top.addWidget(self._read)

        box = QVBoxLayout(self)
        box.setContentsMargins(8, 8, 8, 8)
        box.setSpacing(6)
        box.addLayout(top)
        box.addWidget(self._status)
        box.addWidget(self._bar)
        box.addWidget(split, stretch=1)

    def _environment(self) -> QWidget:
        """Essentia, i modelli, il file dei progressi: lo stato del banco."""
        told = []
        told.append("`essentia` is importable in this environment."
                    if available() else
                    "`essentia` is NOT importable here. It comes with a "
                    "plain `poetry install`."
                    + (" On Windows there is no wheel at all: analysis "
                       "needs the Mac." if sys.platform == "win32" else ""))
        missing = missing_models()
        told.append(f"{len(missing)} model file(s) missing from "
                    f"{MODEL_DIR}." if missing else
                    f"All {len(MODELS)} model files found in {MODEL_DIR}.")
        tracker = ProcessedTracker()
        told.append(
            f"Progress file: {len(tracker):,} tracks recorded "
            f"({tracker.duplicate_lines:,} repeated lines absorbed) · "
            f"{DEFAULT_TRACKING_FILE}" if tracker.existed else
            "No progress file yet. Copy the one from the standalone script "
            f"over {DEFAULT_TRACKING_FILE} — same format, one absolute "
            "path per line.")

        table = QTableView()
        model = PandasModel(pd.DataFrame(
            [{"file": name, "purpose": purpose,
              "present": "—" if name in missing else "✓"}
             for name, purpose in MODELS.items()]), parent=table)
        table.setModel(model)
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(24)
        table.setShowGrid(False)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        table.setColumnWidth(0, 320)
        table.setColumnWidth(1, 220)

        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)
        for line in told:
            box.addWidget(dim(line))
        box.addWidget(table, stretch=1)
        return page

    # ------------------------------------------------------------------
    # la cartella e la lettura dei tag
    # ------------------------------------------------------------------
    def _on_folder(self, root: Path) -> None:
        self._root = root
        self._coverage = None
        self._job.set_root(root)
        self._status.setStyleSheet("")
        self._status.setText(f"Listing audio files under {root.name}…")
        run_in_pool(lambda root=root: find_taggable(root), self._on_listed,
                    lambda t: self._status.setText(
                        f"Could not list the folder: {t}"))

    def _on_listed(self, scope: list[Path]) -> None:
        self._scope = scope
        if not scope:
            self._status.setText("No audio tracks in this folder.")
            return
        self._status.setText(
            f"{len(scope):,} track(s) under {self._root.name} ready.")
        if len(scope) <= AUTO_READ_BELOW:
            self._on_read_tags()
        else:
            self._read.setText(f"Read the tags of {len(scope):,} tracks")
            self._read.setVisible(True)
            self._status.setText(
                f"{len(scope):,} tracks — reading their tags takes about "
                f"{len(scope) * 0.012 / 60:.0f} minutes. It happens once; "
                "after that the filters and the breakdown are instant. The "
                "background job does NOT need this reading.")

    def _on_read_tags(self) -> None:
        if not self._scope:
            return
        self._read.setVisible(False)
        self._bar.setVisible(True)
        self._bar.setRange(0, len(self._scope))
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._bar.setFormat("Reading tags…")
        progress = Progress(self)
        progress.count.connect(self._on_read_progress)
        scope = list(self._scope)

        def _job():
            return scan_coverage(scope, progress=progress.count.emit)

        run_in_pool(_job, self._on_coverage,
                    lambda t: (self._bar.setVisible(False),
                               self._status.setText(
                                   f"Reading the tags failed: {t}")))

    def _on_read_progress(self, done: int, total: int) -> None:
        self._bar.setValue(done)
        self._bar.setFormat(
            f"Read {done:,}/{total:,} — about "
            f"{max(0, total - done) * 0.012 / 60:.0f} min left")

    def _on_coverage(self, coverage) -> None:
        self._bar.setVisible(False)
        self._coverage = coverage
        readable = coverage.readable
        if not readable:
            self._status.setText("None of these files had readable tags.")
            return
        with_genre = sum(c.has_genre for c in readable)
        with_comment = sum(c.has_comment for c in readable)
        complete = sum(c.complete for c in readable)
        self._read_told.show_(f"{len(readable):,}")
        self._with_genre.show_(f"{with_genre:,}",
                               f"{with_genre / len(readable):.0%}")
        self._with_comment.show_(f"{with_comment:,}",
                                 f"{with_comment / len(readable):.0%}")
        self._complete.show_(f"{complete:,}",
                             f"{complete / len(readable):.0%}")
        self._unreadable_told.setVisible(bool(coverage.unreadable))
        if coverage.unreadable:
            self._unreadable_told.setText(
                f"⚠️ {len(coverage.unreadable)} file(s) whose tags could "
                "not be read. Not a tagging problem — no reader opens "
                "these. Folder analysis → Unreadable files deals with "
                "them.")
        self._breakdown.set_coverage(readable)
        self._refresh_queue()

    # ------------------------------------------------------------------
    # la coda
    # ------------------------------------------------------------------
    def _refresh_queue(self) -> None:
        if self._coverage is None:
            return
        self._selected = filtered_coverage(self._coverage,
                                           self._choice.currentText())
        self._queue_table.set_tracks(queue_rows(self._selected))
        if not self._selected:
            self._queue_told.setText(
                "Every track here already has what this filter looks for — "
                "nothing is queued. To go over these tracks anyway, pick "
                "“everything (no filter)” and turn on Overwrite in the "
                "settings if the point is to replace what they carry.")
        else:
            self._queue_told.setText(
                f"{len(self._selected):,} tracks match — all ticked. "
                "Untick whatever you want left alone; ▶ plays a row.")
        self._queue_table.set_all_picked(bool(self._selected))

    def _push_queue(self) -> None:
        self._run.set_queue(
            [Path(p) for p in self._queue_table.selected_paths()])

    def _on_tags_written(self) -> None:
        """Dopo un salvataggio i file dicono cose nuove: si rilegge."""
        if self._scope:
            self._on_read_tags()
