"""Map settings: quanto è grande la mappa, la proiezione, e il job che la
costruisce — governato da qui invece che da Terminal.app.

Il job resta `map_cli.py` in un processo staccato con lo stato su file,
identico a come lo lancia Streamlit: qui cambia solo il polling (un QTimer
al posto del rerun) e il monitor (una finestra interna che rilegge il log,
al posto di `osascript` che apriva il Terminale — che su Windows non
esiste). Pausa e ripresa passano per SIGSTOP/SIGCONT e compaiono solo dove
i segnali ci sono; `caffeinate` ha già la sua guardia in core.
"""

from __future__ import annotations

import signal as posix_signal
import subprocess
import sys
import time
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QDialog, QDoubleSpinBox,
                               QFileDialog, QHBoxLayout, QLabel,
                               QPlainTextEdit, QProgressBar, QPushButton,
                               QSpinBox, QVBoxLayout)

from core.analysis.essentia_tags import (MODEL_DIR, available, find_taggable,
                                         missing_models)
from core.analysis.map_job import (DEFAULT_MAP_LOG, MAP_CLI_PATH, caffeinated,
                                   load_map_state, pause_job, process_state,
                                   resume_job, stop_job)
from core.analysis.map_projection import ProjectionSettings
from core.analysis.map_projection import available as umap_available
from core.analysis.map_projection import project
from core.analysis.map_profile import default_workers
from core.analysis.map_store import MapStore
from qt_app import theme
from qt_app.pages.common import spelled
from qt_app.workers import run_in_pool

# I segnali di pausa esistono solo sui POSIX: su Windows i due bottoni non
# si disegnano — un job si può comunque fermare.
CAN_PAUSE = hasattr(posix_signal, "SIGSTOP")


def _dim(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setObjectName("dim")
    label.setWordWrap(True)
    return label


class _Progress(QObject):
    """Il filo di ritorno di un lavoro nel pool: frasi, non percentuali —
    UMAP non dice a che punto è, e inventarlo sarebbe peggio."""
    told = Signal(str)


class SettingsDialog(QDialog):
    """Infos, proiezione, aggiunta brani, e il monitor del job.

    `library_changed` dice alla pagina che i file della mappa sono cambiati
    (una proiezione rifatta, un job finito): è lei che ricarica.
    """

    library_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Map settings")
        self.setModal(False)
        self.resize(680, 760)
        self._store: MapStore | None = None
        self._folder: Path | None = None
        self._queue: list[Path] = []
        self._was_running = False
        self._projecting = False

        self._timer = QTimer(self)
        self._timer.setInterval(2000)
        self._timer.timeout.connect(self._on_tick)

        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        self._infos = QLabel("")
        self._infos.setWordWrap(True)
        self._infos.setTextFormat(Qt.TextFormat.RichText)
        self._warnings = _dim("")
        theme.style(self._warnings, lambda: f"color: {theme.WARN};")

        # --- la proiezione ---
        self._neighbors = QSpinBox()
        self._neighbors.setRange(5, 100)
        self._neighbors.setValue(ProjectionSettings.n_neighbors)
        self._neighbors.setToolTip(
            "How many tracks define what 'nearby' means. Low: many small "
            "recognisable clusters. High: one continent with soft edges.")
        self._min_dist = QDoubleSpinBox()
        self._min_dist.setRange(0.0, 0.9)
        self._min_dist.setSingleStep(0.05)
        self._min_dist.setValue(ProjectionSettings.min_dist)
        self._min_dist.setToolTip("How tightly points may pack. Low: dense "
                                  "separated clumps, easier to aim at.")
        self._project = QPushButton("↻ Recompute the projection")
        self._project.clicked.connect(self._on_project)
        self._project_told = _dim(
            "The projection is what turns 1280 dimensions into a picture. "
            "Recompute it whenever new tracks come in — and after changing "
            "either knob.")
        projection = QHBoxLayout()
        projection.addWidget(QLabel("Neighbours"))
        projection.addWidget(self._neighbors)
        projection.addWidget(QLabel("Min distance"))
        projection.addWidget(self._min_dist)
        projection.addWidget(self._project, stretch=1)

        # --- aggiungere brani ---
        self._choose = QPushButton("📁 Choose the folder to analyze…")
        self._choose.clicked.connect(self._on_choose)
        self._folder_told = _dim(
            "About 5 seconds per track on one process, 2–3 with several. A "
            "whole library is hours, which is what the background job is "
            "for: it survives closing the app and picks up where it left "
            "off.")
        self._workers = QSpinBox()
        self._workers.setRange(1, 12)
        self._workers.setValue(default_workers())
        self._workers.setToolTip("Each process holds its own copy of the "
                                 "models, about 1.3 GB. Half the cores is "
                                 "the sweet spot.")
        self._awake = QCheckBox("Keep the Mac awake until the job is done")
        self._awake.setChecked(True)
        self._awake.setToolTip("A sleeping Mac freezes the job: it stays "
                               "alive without working.")
        self._awake.setVisible(sys.platform == "darwin")
        self._launch = QPushButton("▶ Add all in the background")
        self._launch.setEnabled(False)
        self._launch.clicked.connect(self._on_launch)
        workers_row = QHBoxLayout()
        workers_row.addWidget(QLabel("Analyses in parallel"))
        workers_row.addWidget(self._workers)
        workers_row.addWidget(self._awake, stretch=1)

        # --- il job ---
        self._bar = QProgressBar()
        self._bar.setTextVisible(False)
        self._job_told = QLabel("")
        self._job_told.setWordWrap(True)
        self._pause = QPushButton("⏸ Pause")
        self._pause.setToolTip("Freezes the job where it is, models still "
                               "loaded, so resuming is instant — but the "
                               "processes keep holding their memory. For a "
                               "break, not for the night.")
        self._pause.clicked.connect(self._on_pause)
        self._resume = QPushButton("▶ Resume")
        self._resume.clicked.connect(self._on_resume)
        self._stop = QPushButton("⏹ Stop")
        self._stop.setToolTip("Ends the job and frees the memory. What is "
                              "already on the map stays there.")
        self._stop.clicked.connect(self._on_stop)
        job_row = QHBoxLayout()
        if CAN_PAUSE:
            job_row.addWidget(self._pause)
            job_row.addWidget(self._resume)
        job_row.addWidget(self._stop)
        job_row.addStretch(1)

        # Il monitor interno: quello che il job stampa, riletto dal log — al
        # posto del Terminale aperto con osascript.
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(150)

        box = QVBoxLayout(self)
        box.addWidget(self._infos)
        box.addWidget(self._warnings)
        box.addWidget(self._project_told)
        box.addLayout(projection)
        box.addSpacing(10)
        box.addWidget(QLabel("<b>Add tracks to the map</b>"))
        box.addWidget(self._folder_told)
        box.addWidget(self._choose)
        box.addLayout(workers_row)
        box.addWidget(self._launch)
        box.addSpacing(10)
        box.addWidget(self._bar)
        box.addWidget(self._job_told)
        box.addLayout(job_row)
        box.addWidget(self._log, stretch=1)

    # ------------------------------------------------------------------
    def set_store(self, store: MapStore) -> None:
        self._store = store
        placed = store.placed
        waiting = len(store) - placed
        dims = store.embeddings.shape[1] if len(store) else 0
        self._infos.setText(
            f"<b>{len(store):,}</b> tracks analyzed · <b>{placed:,}</b> "
            f"placed on the map"
            + (f" · <b>{waiting:,}</b> waiting for the next projection"
               if waiting else "")
            + f" · {dims}-D embedding<br>"
            f"Stored in <code>{store.directory}</code> — append-only, which "
            "is what makes the build interruptible.")

        troubles = []
        if not available():
            troubles.append("`essentia` is not importable here, so no track "
                            "can be analyzed. The map itself still opens"
                            + (" (analysis needs the Mac)."
                               if sys.platform == "win32" else "."))
        if available() and missing_models():
            troubles.append(f"Model files missing from {MODEL_DIR} — see "
                            "Tag Maker.")
        if not umap_available():
            troubles.append("`umap-learn` is not importable here, so the "
                            "projection cannot be recomputed.")
        self._warnings.setText("\n".join(troubles))
        self._warnings.setVisible(bool(troubles))
        self._project.setEnabled(umap_available() and len(store) > 0
                                 and not self._projecting)
        self._refresh_queue()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._timer.start()
        self._on_tick()

    def hideEvent(self, event) -> None:
        self._timer.stop()
        super().hideEvent(event)

    # ------------------------------------------------------------------
    # la proiezione
    # ------------------------------------------------------------------
    def _on_project(self) -> None:
        if self._store is None or not len(self._store):
            return
        store = self._store
        settings = ProjectionSettings(n_neighbors=self._neighbors.value(),
                                      min_dist=self._min_dist.value())
        progress = _Progress(self)
        progress.told.connect(self._project_told.setText)
        began = time.perf_counter()
        self._projecting = True
        self._project.setEnabled(False)
        self._project_told.setText(f"Projecting {len(store):,} tracks…")

        def _job():
            def announce(label: str) -> None:
                progress.told.emit(
                    f"{label} — {spelled(time.perf_counter() - began)} so far")
            coords = project(store.embeddings, settings, on_step=announce)
            store.set_coords(coords)
            return len(coords)

        def _done(count: int) -> None:
            self._projecting = False
            self._project.setEnabled(True)
            self._project_told.setText(
                f"{count:,} tracks placed in "
                f"{spelled(time.perf_counter() - began)}.")
            self.library_changed.emit()

        def _failed(trouble: Exception) -> None:
            self._projecting = False
            self._project.setEnabled(True)
            self._project_told.setText(f"The projection failed: {trouble}")

        run_in_pool(_job, _done, _failed)

    # ------------------------------------------------------------------
    # aggiungere brani
    # ------------------------------------------------------------------
    def _on_choose(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Folder with the tracks to analyze")
        if not chosen:
            return
        self._folder = Path(chosen)
        self._folder_told.setText(f"Listing audio files under "
                                  f"{self._folder.name}…")
        run_in_pool(lambda: find_taggable(self._folder),
                    self._on_listed, lambda t: self._folder_told.setText(
                        f"Could not list the folder: {t}"))

    def _on_listed(self, found: list[Path]) -> None:
        self._found = found
        self._refresh_queue()

    def _refresh_queue(self) -> None:
        if self._folder is None or self._store is None \
                or not hasattr(self, "_found"):
            return
        self._queue = self._store.pending(self._found)
        self._folder_told.setText(
            f"<b>{len(self._found):,} track(s)</b> under "
            f"<code>{self._folder.name}</code> — {len(self._queue):,} not "
            "on the map yet.")
        blocked = not available() or bool(missing_models())
        state = load_map_state()
        running = state is not None and state.running
        self._launch.setEnabled(bool(self._queue) and not blocked
                                and not running)
        self._launch.setText(
            f"▶ Add all {len(self._queue):,} in the background"
            if self._queue else "▶ Nothing to add: everything is on the map")

    def _on_launch(self) -> None:
        if not self._queue or self._folder is None:
            return
        command = [sys.executable, str(MAP_CLI_PATH), str(self._folder),
                   "--workers", str(self._workers.value()), "--project"]
        if self._awake.isChecked():
            command = caffeinated(command)
        with open(DEFAULT_MAP_LOG, "w") as out:
            subprocess.Popen(command, stdout=out,
                             stderr=subprocess.STDOUT,
                             start_new_session=True,
                             cwd=MAP_CLI_PATH.parent)
        self._launch.setEnabled(False)
        self._job_told.setText(f"Started. Output in {DEFAULT_MAP_LOG}.")
        QTimer.singleShot(1500, self._on_tick)

    # ------------------------------------------------------------------
    # il monitor
    # ------------------------------------------------------------------
    def _on_tick(self) -> None:
        state = load_map_state()
        running = state is not None and state.running
        if self._was_running and not running:
            # Finito mentre lo si guardava: la pagina ha qualcosa di nuovo
            # da mostrare, e il contatore della coda va rifatto.
            self.library_changed.emit()
            self._refresh_queue()
        self._was_running = running

        job_widgets = (self._bar, self._pause, self._resume, self._stop,
                       self._log)
        if state is None:
            for widget in job_widgets:
                widget.setVisible(False)
            self._job_told.setText("")
            return

        if not running:
            for widget in job_widgets:
                widget.setVisible(False)
            if state.total:
                self._job_told.setText(
                    f"Last job: {state.written:,} added, {state.failed:,} "
                    f"failed out of {state.total:,}."
                    + (f" {len(state.errors)} error(s) kept in the log."
                       if state.errors else ""))
            return

        how = process_state(state.pid)
        paused = how == "paused"
        self._bar.setVisible(True)
        self._bar.setRange(0, max(1, state.total))
        self._bar.setValue(state.done)
        self._pause.setVisible(CAN_PAUSE and not paused)
        self._resume.setVisible(CAN_PAUSE and paused)
        self._stop.setVisible(True)
        self._log.setVisible(True)
        left = "—" if paused else spelled(state.eta_seconds)
        self._job_told.setText(
            (f"⏸ paused at {state.done:,}/{state.total:,}" if paused else
             f"{state.done:,}/{state.total:,} · {state.current[:50]}")
            + f" · on the map {state.written:,} · failed {state.failed:,}"
            + f" · left {left}\n"
            f"{'Paused' if paused else 'Running'} as process {state.pid} on "
            f"{state.folder} — closing this window does not stop it.")
        self._read_log()

    def _read_log(self) -> None:
        try:
            with open(DEFAULT_MAP_LOG, "rb") as handle:
                handle.seek(0, 2)
                size = handle.tell()
                handle.seek(max(0, size - 8192))
                tail = handle.read().decode("utf-8", "replace")
        except OSError:
            return
        # Il \r del contatore in piazza pulita: ogni riga sovrascritta del
        # terminale diventa una riga vera, e si legge l'ultima.
        lines = tail.replace("\r", "\n").splitlines()[-40:]
        text = "\n".join(lines)
        if text != self._log.toPlainText():
            self._log.setPlainText(text)
            self._log.verticalScrollBar().setValue(
                self._log.verticalScrollBar().maximum())

    def _on_pause(self) -> None:
        state = load_map_state()
        if state is not None and pause_job(state.pid):
            self._on_tick()

    def _on_resume(self) -> None:
        state = load_map_state()
        if state is not None and resume_job(state.pid):
            self._on_tick()

    def _on_stop(self) -> None:
        state = load_map_state()
        if state is not None:
            stop_job(state.pid)
            QTimer.singleShot(600, self._on_tick)
