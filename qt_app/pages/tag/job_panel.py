"""Il job in background: la cartella intera, macinata da un processo a sé.

Lo stesso lavoro della corsa a mano, lanciato come `tag_cli.py` staccato:
sopravvive alla chiusura dell'app e scrive i tag man mano, quindi ignora le
spunte — va su tutto quello che la cartella ancora deve. Il polling è un
QTimer sul file di stato, come per il job della mappa.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QHBoxLayout, QProgressBar, QPushButton,
                               QVBoxLayout, QWidget)

from core.analysis.tag_job import TAG_CLI_PATH, load_state
from qt_app.pages.common import Metric, dim, spelled


class JobPanel(QWidget):
    """Stato del job corrente, esito dell'ultimo, e il bottone di lancio.

    `current_settings` è una callable che porta (TagSettings, workers) dal
    pannello Run: il job usa le stesse manopole della corsa a mano.
    """

    def __init__(self, current_settings, parent=None) -> None:
        super().__init__(parent)
        self._current_settings = current_settings
        self._root: Path | None = None

        self._timer = QTimer(self)
        self._timer.setInterval(2000)
        self._timer.timeout.connect(self._on_tick)

        self._note = dim(
            "Two ways to run, and this is the one for a lot of tracks: the "
            "job works through the folder on its own and writes as it goes, "
            "skipping whatever the progress file already records — so it "
            "can be stopped and restarted without losing work. It ignores "
            "the picking on the left; for only the ticked tracks, use "
            "Analyze in the Run tab.")
        self._bar = QProgressBar()
        self._bar.setTextVisible(False)
        self._bar.setVisible(False)
        self._told = dim("")

        self._written = Metric("Written")
        self._failed = Metric("Failed")
        self._each = Metric("Per track")
        self._left = Metric("Left")
        self._numbers = QWidget()
        numbers = QHBoxLayout(self._numbers)
        numbers.setContentsMargins(0, 0, 0, 0)
        for metric in (self._written, self._failed, self._each, self._left):
            numbers.addWidget(metric)
        self._numbers.setVisible(False)

        self._launch = QPushButton("▶ Start the job on the whole folder")
        self._launch.clicked.connect(self._on_launch)
        self._launch.setEnabled(False)

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)
        box.addWidget(self._note)
        box.addWidget(self._bar)
        box.addWidget(self._numbers)
        box.addWidget(self._told)
        box.addWidget(self._launch)
        box.addStretch(1)

    # ------------------------------------------------------------------
    def set_root(self, root: Path | None) -> None:
        self._root = root
        self._on_tick()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._timer.start()
        self._on_tick()

    def hideEvent(self, event) -> None:
        self._timer.stop()
        super().hideEvent(event)

    # ------------------------------------------------------------------
    def _on_tick(self) -> None:
        state = load_state()
        running = state is not None and state.running
        self._bar.setVisible(running)
        self._numbers.setVisible(running)
        self._launch.setEnabled(self._root is not None and not running)
        if running:
            self._bar.setRange(0, max(1, state.total))
            self._bar.setValue(state.done)
            self._written.show_(f"{state.written:,}")
            self._failed.show_(f"{state.failed:,}")
            self._each.show_(f"{state.seconds_each:.1f}s")
            self._left.show_(spelled(state.eta_seconds))
            self._told.setText(
                f"{state.done:,}/{state.total:,} · {state.current[:50]}\n"
                f"Running as process {state.pid} on {state.folder} — "
                "closing this window does not stop it. The job writes the "
                "tags as it goes, so what is done is already saved.")
            return
        if state is not None and state.total:
            self._told.setText(
                f"Last job: {state.written:,} written, {state.failed:,} "
                f"failed out of {state.total:,}."
                + (f" {len(state.errors)} error(s) kept — the first: "
                   + "; ".join(f"{e.get('file', '?')}"
                               for e in state.errors[:3])
                   if state.errors else ""))
        elif self._root is None:
            self._told.setText("Choose a folder above to enable the job.")
        else:
            self._told.setText("No job has run yet.")

    def _on_launch(self) -> None:
        if self._root is None:
            return
        settings, workers = self._current_settings()
        log = Path(tempfile.gettempdir()) / "djcaddy_tag_job.log"
        command = [sys.executable, str(TAG_CLI_PATH), str(self._root),
                   "--workers", str(workers),
                   "--genre-format", settings.genre_format,
                   "--max-seconds", str(settings.max_seconds)]
        if settings.overwrite:
            command.append("--overwrite")
        if settings.confidence_in_comment:
            command.append("--confidence-in-comment")
        if not settings.genres:
            command.append("--no-genres")
        if not settings.moods:
            command.append("--no-moods")
        # start_new_session: staccato dall'app, così chiuderla non porta
        # giù anche il lavoro.
        with open(log, "w") as out:
            subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT,
                             start_new_session=True,
                             cwd=TAG_CLI_PATH.parent)
        self._told.setText(f"Started. Output in {log}.")
        QTimer.singleShot(1500, self._on_tick)
