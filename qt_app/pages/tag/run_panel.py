"""Le impostazioni e l'analisi a mano: la parte "guarda prima di scrivere".

Come di là: l'analisi non tocca i file — i risultati compaiono in tabella,
il salvataggio è un clic a parte, così l'analisi non si paga due volte. Le
colonne GENRE/COMMENT proposte si riformattano al volo quando cambiano le
impostazioni di formato (le soglie no: quelle lavorano durante l'analisi).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QGridLayout, QHBoxLayout, QLabel, QProgressBar,
                               QPushButton, QSpinBox, QVBoxLayout, QWidget)

from core.analysis.essentia_tags import (GENRE_FORMATS, TagSettings,
                                         analyze_many, available,
                                         build_tag_values, default_workers,
                                         missing_models, write_tags)
from core.analysis.tag_tracking import ProcessedTracker
from qt_app import theme
from qt_app.pages.common import dim, spelled
from qt_app.widgets.track_table import TrackTable
from qt_app.workers import Progress, run_in_pool

# Secondi a brano misurati su questa macchina (M5, 10 core, 24 analisi), per
# numero di processi: dicono quanto ci vorrà PRIMA di partire.
SECONDS_PER_TRACK = {1: 8.2, 2: 5.7, 3: 5.0, 5: 4.1, 8: 3.7}


def seconds_each(workers: int) -> float:
    known = min(SECONDS_PER_TRACK, key=lambda k: abs(k - workers))
    return SECONDS_PER_TRACK[known]


def proposed_rows(analyzed: list[tuple], settings: TagSettings) -> list[dict]:
    """Le righe della tabella dei risultati, formattate con le impostazioni
    CORRENTI: cambiare il formato del genere non richiede una nuova analisi."""
    rows = []
    for path, tags in analyzed:
        values = build_tag_values(tags, settings)
        rows.append({
            "file": path.name,
            "GENRE": values.genre or "—",
            "COMMENT": values.mood or "—",
            "confidence": (values.genre_confidence or "")[:60],
            "_path": str(path),
        })
    return rows


class SettingsBox(QWidget):
    """Le stesse opzioni che il terminale chiedeva, come controlli."""

    changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(4)

        self.do_genres = QCheckBox("Write genre")
        self.do_genres.setChecked(True)
        self.top_genres = QSpinBox()
        self.top_genres.setRange(1, 10)
        self.top_genres.setValue(3)
        self.genre_threshold = QDoubleSpinBox()
        self.genre_threshold.setRange(0.0, 1.0)
        self.genre_threshold.setSingleStep(0.01)
        self.genre_threshold.setValue(0.15)
        self.genre_threshold.setToolTip(
            "Minimum activation. If nothing clears it the single best label "
            "is written anyway — a track always has a genre.")
        self.genre_format = QComboBox()
        self.genre_format.addItems(list(GENRE_FORMATS))
        self.genre_format.setToolTip(
            'How "Rock---Alternative Rock" is written out.')

        self.do_moods = QCheckBox("Write mood")
        self.do_moods.setChecked(True)
        self.moods_in_tag = QSpinBox()
        self.moods_in_tag.setRange(1, 5)
        self.moods_in_tag.setValue(3)
        self.mood_threshold = QDoubleSpinBox()
        self.mood_threshold.setRange(0.0, 1.0)
        self.mood_threshold.setDecimals(3)
        self.mood_threshold.setSingleStep(0.005)
        self.mood_threshold.setValue(0.05)
        self.mood_threshold.setToolTip(
            "Lower than the genre one on purpose: mood activations are much "
            "smaller. Nothing is invented if none clear it.")
        self.confidence_tags = QCheckBox("Also write confidence tags")
        self.confidence_tags.setChecked(True)
        self.confidence_tags.setToolTip(
            "Percentages in a SEPARATE field (ESSENTIA_GENRE / "
            "ESSENTIA_MOOD), beside the tag itself. djay Pro does not "
            "display these.")
        self.confidence_in_comment = QCheckBox("Percentages in the comment")
        self.confidence_in_comment.setToolTip(
            'Puts them in the comment djay Pro actually shows: "Happy 87%; '
            'Deep 62%" instead of "Happy; Deep".')

        self.overwrite = QCheckBox("Overwrite tags that are already there")
        self.overwrite.setToolTip(
            "Off: a track that already has a genre keeps it.")
        self.max_seconds = QSpinBox()
        self.max_seconds.setRange(0, 1200)
        self.max_seconds.setSingleStep(30)
        self.max_seconds.setValue(300)
        self.max_seconds.setToolTip(
            "0 = the whole track. Do not go below 300 without reason: "
            "measured on a disco-house track with a one-minute intro, 120s "
            "gave the wrong genre and mood.")
        self.workers = QSpinBox()
        self.workers.setRange(1, max(2, os.cpu_count() or 2))
        self.workers.setValue(default_workers())
        self.workers.setToolTip(
            "How many analyses run in parallel, each in its own process. "
            "Half the cores is the sweet spot; each holds ~1.3 GB of "
            "models.")

        grid.addWidget(self.do_genres, 0, 0)
        grid.addWidget(self.do_moods, 0, 2)
        grid.addWidget(QLabel("How many genres"), 1, 0)
        grid.addWidget(self.top_genres, 1, 1)
        grid.addWidget(QLabel("Moods in the comment"), 1, 2)
        grid.addWidget(self.moods_in_tag, 1, 3)
        grid.addWidget(QLabel("Genre threshold"), 2, 0)
        grid.addWidget(self.genre_threshold, 2, 1)
        grid.addWidget(QLabel("Mood threshold"), 2, 2)
        grid.addWidget(self.mood_threshold, 2, 3)
        grid.addWidget(QLabel("Genre format"), 3, 0)
        grid.addWidget(self.genre_format, 3, 1)
        grid.addWidget(self.confidence_tags, 3, 2, 1, 2)
        grid.addWidget(self.overwrite, 4, 0, 1, 2)
        grid.addWidget(self.confidence_in_comment, 4, 2, 1, 2)
        grid.addWidget(QLabel("Seconds of audio"), 5, 0)
        grid.addWidget(self.max_seconds, 5, 1)
        grid.addWidget(QLabel("Tracks at the same time"), 5, 2)
        grid.addWidget(self.workers, 5, 3)

        for control in (self.do_genres, self.do_moods, self.confidence_tags,
                        self.confidence_in_comment, self.overwrite):
            control.toggled.connect(lambda _: self.changed.emit())
        for control in (self.top_genres, self.moods_in_tag, self.max_seconds,
                        self.workers):
            control.valueChanged.connect(lambda _: self.changed.emit())
        for control in (self.genre_threshold, self.mood_threshold):
            control.valueChanged.connect(lambda _: self.changed.emit())
        self.genre_format.currentTextChanged.connect(
            lambda _: self.changed.emit())

    def settings(self) -> TagSettings:
        return TagSettings(
            genres=self.do_genres.isChecked(),
            moods=self.do_moods.isChecked(),
            top_genres=self.top_genres.value(),
            genre_threshold=self.genre_threshold.value(),
            genre_format=self.genre_format.currentText(),
            mood_threshold=self.mood_threshold.value(),
            moods_in_tag=self.moods_in_tag.value(),
            confidence_tags=self.confidence_tags.isChecked(),
            confidence_in_comment=self.confidence_in_comment.isChecked(),
            overwrite=self.overwrite.isChecked(),
            max_seconds=self.max_seconds.value(),
        )


class RunPanel(QWidget):
    """Impostazioni, coda spuntata, analisi e salvataggio."""

    tags_written = Signal()     # la pagina rilegge la copertura

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._queue: list[Path] = []
        self._analyzed: list[tuple] = []
        self._failures: list[dict] = []

        self.settings_box = SettingsBox()
        self.settings_box.changed.connect(self._refresh_proposed)

        self._batch = QSpinBox()
        self._batch.setRange(1, 1)
        self._batch.setToolTip(
            "Defaults to the whole queue. Lower it to try a handful first "
            "and see what comes back before committing to hours.")
        self._batch.valueChanged.connect(lambda _: self._refresh_eta())
        self.settings_box.workers.valueChanged.connect(
            lambda _: self._refresh_eta())
        self._analyze = QPushButton("Analyze")
        self._analyze.setStyleSheet(
            theme.primary_button())
        self._analyze.clicked.connect(self._on_analyze)
        batch_row = QHBoxLayout()
        batch_row.addWidget(QLabel("How many to analyze now"))
        batch_row.addWidget(self._batch)
        batch_row.addStretch(1)
        batch_row.addWidget(self._analyze)

        self._eta = dim("")
        self._bar = QProgressBar()
        self._bar.setVisible(False)
        self._trouble = dim("")
        self._trouble.setVisible(False)

        self._told = dim("Nothing analyzed yet: results appear here, and "
                         "saving them is a separate click — nothing touches "
                         "the files at this stage.")
        self._results = TrackTable(checkable=True, library_menu=False)
        self._results.setVisible(False)
        self._save = QPushButton("💾 Save tags")
        self._save.setStyleSheet(
            theme.primary_button())
        self._save.clicked.connect(self._on_save)
        self._save.setVisible(False)
        self._results.selection_paths_changed.connect(
            lambda _: self._refresh_save())

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)
        box.addWidget(self.settings_box)
        box.addLayout(batch_row)
        box.addWidget(self._eta)
        box.addWidget(self._bar)
        box.addWidget(self._trouble)
        box.addWidget(self._told)
        box.addWidget(self._results, stretch=1)
        box.addWidget(self._save)

    # ------------------------------------------------------------------
    def set_queue(self, queue: list[Path]) -> None:
        self._queue = list(queue)
        some = bool(queue)
        self._batch.setEnabled(some)
        self._analyze.setEnabled(some and available()
                                 and not missing_models())
        if some:
            self._batch.setMaximum(len(queue))
            self._batch.setValue(len(queue))
        self._refresh_eta()

    def _refresh_eta(self) -> None:
        if not self._queue:
            self._eta.setText(
                "Nothing is queued: tick tracks in the table on the left."
                if available() and not missing_models() else
                "`essentia` is not importable or model files are missing — "
                "see Environment. Nothing can be analyzed here.")
            return
        workers = self.settings_box.workers.value()
        each = seconds_each(workers)
        batch = self._batch.value()
        self._eta.setText(
            f"About {each:.0f}s per track at {workers} at a time — roughly "
            f"{spelled(batch * each)} for {batch:,}. Each process holds its "
            f"own copy of the models, about {workers * 1.3:.1f} GB in "
            "total."
            + (" That is hours: long runs are what the background job is "
               "for." if batch * each > 3600 else ""))

    # ------------------------------------------------------------------
    def _on_analyze(self) -> None:
        todo = self._queue[:self._batch.value()]
        if not todo:
            return
        settings = self.settings_box.settings()
        workers = self.settings_box.workers.value()
        self._analyze.setEnabled(False)
        self._bar.setVisible(True)
        self._bar.setRange(0, len(todo))
        self._bar.setValue(0)
        self._bar.setFormat("Loading models…")
        self._bar.setTextVisible(True)
        progress = Progress(self)
        progress.count.connect(self._on_progress)

        def _job():
            done, failures = [], []
            for i, (path, tags, error) in enumerate(
                    analyze_many(todo, settings, workers=workers), 1):
                progress.count.emit(i, len(todo))
                if error is None:
                    done.append((path, tags))
                else:
                    failures.append({"file": path.name,
                                     "folder": str(path.parent),
                                     "error": error})
            return done, failures

        run_in_pool(_job, self._on_analyzed, self._on_failed)

    def _on_progress(self, done: int, total: int) -> None:
        self._bar.setValue(done)
        self._bar.setFormat(f"{done}/{total}")

    def _on_failed(self, trouble: Exception) -> None:
        self._analyze.setEnabled(True)
        self._bar.setVisible(False)
        self._trouble.setText(f"The analysis failed: {trouble}")
        self._trouble.setStyleSheet(f"color: {theme.PRIMARY};")
        self._trouble.setVisible(True)

    def _on_analyzed(self, result) -> None:
        self._analyze.setEnabled(True)
        self._bar.setVisible(False)
        self._analyzed, self._failures = result
        self._trouble.setVisible(bool(self._failures))
        if self._failures:
            listed = "\n".join(f"  {f['file']} — {f['error']}"
                               for f in self._failures[:10])
            self._trouble.setStyleSheet("color: #ffb454;")
            self._trouble.setText(
                f"{len(self._failures)} track(s) could not be analyzed:\n"
                + listed
                + ("\n  …" if len(self._failures) > 10 else ""))
        self._refresh_proposed()

    def _refresh_proposed(self) -> None:
        if not self._analyzed:
            return
        rows = proposed_rows(self._analyzed, self.settings_box.settings())
        frame = pd.DataFrame(rows,
                             columns=["file", "GENRE", "COMMENT",
                                      "confidence", "_path"])
        self._results.set_tracks(frame)
        self._results.set_all_picked(True)
        self._results.setVisible(True)
        self._told.setText(
            "What it found — save when it looks right. The formatting "
            "follows the settings above (thresholds excepted: those are "
            "applied while analyzing).")
        self._save.setVisible(True)
        self._refresh_save()

    def _refresh_save(self) -> None:
        picked = len(self._results.selected_paths())
        self._save.setEnabled(bool(picked))
        self._save.setText(f"💾 Save tags to {picked} file(s)")

    # ------------------------------------------------------------------
    def _on_save(self) -> None:
        chosen = set(self._results.selected_paths())
        pending = [(p, t) for p, t in self._analyzed if str(p) in chosen]
        if not pending:
            return
        settings = self.settings_box.settings()
        self._save.setEnabled(False)
        self._bar.setVisible(True)
        self._bar.setRange(0, len(pending))
        self._bar.setValue(0)
        self._bar.setFormat("Writing…")
        progress = Progress(self)
        progress.count.connect(self._on_progress)

        def _job():
            tracker = ProcessedTracker()
            written, problems = 0, []
            for i, (path, tags) in enumerate(pending, 1):
                progress.count.emit(i, len(pending))
                try:
                    if write_tags(path, tags, settings):
                        written += 1
                    tracker.mark(path)
                except Exception as e:                     # noqa: BLE001
                    problems.append({"file": path.name,
                                     "error": f"{type(e).__name__}: {e}"})
            return written, problems

        run_in_pool(_job, self._on_saved, self._on_failed)

    def _on_saved(self, result) -> None:
        written, problems = result
        self._bar.setVisible(False)
        self._save.setEnabled(True)
        self._analyzed = []
        self._results.setVisible(False)
        self._save.setVisible(False)
        self._told.setText(
            f"Tags written to {written} file(s)."
            + (f" {len(problems)} could not be written: "
               + "; ".join(p["file"] for p in problems[:5])
               if problems else ""))
        self.tags_written.emit()
