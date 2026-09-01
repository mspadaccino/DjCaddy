"""La pagina Folder analysis: cosa c'è in una cartella e cosa è di troppo.

Le operazioni restano separate come di là, ognuna col suo bottone: la
scansione è veloce e si rifà a piacere, duplicati/durate/integrità leggono
i file per davvero e si lanciano solo quando servono. Nessun file audio si
cancella da qui: i duplicati e gli illeggibili vanno in quarantena, e solo
ciò che musica non è (sidecar, clutter per estensione) si elimina, sempre
dopo conferma.

Le cinque sezioni della pagina-fiume di Streamlit diventano cinque tab —
lo scarto di layout della Fase 4 — e dopo un'azione che sposta o cancella
la scansione riparte da sola: era comunque la prima cosa da rifare.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QPushButton,
                               QTabWidget, QVBoxLayout, QWidget)

from core.analysis.duplicates import QUARANTINE_DIRNAME
from core.analysis.folder_scan import human_size, scan_folder
from qt_app import theme
from qt_app.pages.common import FolderRow, Metric, dim
from qt_app.state import AppState
from qt_app.workers import Progress, run_in_pool

from .cleanup_panel import JunkPanel, UnreadablePanel
from .contents import ContentsPanel
from .duplicates_panel import DuplicatesPanel
from .filtering_panel import FilteringPanel


class FolderPage(QWidget):
    """Scansione in cima, le cinque sezioni sotto, ognuna col suo passo."""

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._root: Path | None = None
        self._scanning = False

        self._folder = FolderRow("Folder to analyze")
        self._folder.chosen.connect(self._on_folder)
        self._audio_only = QCheckBox("Audio files only")
        self._audio_only.setToolTip(
            "Off: counts every file, so you also see artwork, .DS_Store, "
            "cue sheets and the rest. Duplicate search always considers "
            "audio files only.")
        self._scan = QPushButton("Scan folder")
        theme.style(self._scan, theme.primary_button)
        self._scan.clicked.connect(self._on_scan)
        self._scan.setEnabled(False)
        top = QHBoxLayout()
        top.addWidget(self._folder, stretch=1)
        top.addWidget(self._audio_only)
        top.addWidget(self._scan)

        self._status = dim(
            "Count what a folder actually contains, then look for "
            "duplicates. No audio file is ever deleted here: what you tick "
            f"is MOVED to a `{QUARANTINE_DIRNAME}/` folder that you empty "
            "yourself once rekordbox still looks right.")

        self._files = Metric("Files")
        self._audio = Metric("Audio files")
        self._size = Metric("Total size")
        numbers = QHBoxLayout()
        for metric in (self._files, self._audio, self._size):
            numbers.addWidget(metric)
        self._numbers = QWidget()
        self._numbers.setLayout(numbers)
        numbers.setContentsMargins(0, 0, 0, 0)
        self._numbers.setVisible(False)

        self._contents = ContentsPanel(state)
        self._duplicates = DuplicatesPanel(state)
        self._filtering = FilteringPanel(state)
        self._junk = JunkPanel()
        self._unreadable = UnreadablePanel(state)
        for panel in (self._contents, self._duplicates, self._filtering,
                      self._junk, self._unreadable):
            panel.rescan_needed.connect(self._on_scan)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._contents, "📊 Contents")
        self._tabs.addTab(self._duplicates, "👯 Duplicates")
        self._tabs.addTab(self._filtering, "⏱ Library filtering")
        self._tabs.addTab(self._junk, "🧹 Junk files")
        self._tabs.addTab(self._unreadable, "🩺 Unreadable")
        self._tabs.setVisible(False)

        box = QVBoxLayout(self)
        box.setContentsMargins(8, 8, 8, 8)
        box.setSpacing(6)
        box.addLayout(top)
        box.addWidget(self._status)
        box.addWidget(self._numbers)
        box.addWidget(self._tabs, stretch=1)

    # ------------------------------------------------------------------
    def _on_folder(self, root: Path) -> None:
        self._root = root
        self._scan.setEnabled(True)
        self._status.setText(f"Press Scan folder to look inside "
                             f"{root.name}.")

    def _on_scan(self) -> None:
        if self._root is None or self._scanning:
            return
        self._scanning = True
        root = self._root
        audio_only = self._audio_only.isChecked()
        self._scan.setEnabled(False)
        self._status.setText("Walking the folder tree…")
        progress = Progress(self)
        progress.text.connect(self._status.setText)

        def _job():
            return scan_folder(
                root, audio_only=audio_only,
                progress=lambda n: progress.text.emit(
                    f"{n:,} files seen…"))

        run_in_pool(_job, self._on_scanned, self._on_scan_failed)

    def _on_scan_failed(self, trouble: Exception) -> None:
        self._scanning = False
        self._scan.setEnabled(True)
        self._status.setText(f"The scan failed: {trouble}")

    def _on_scanned(self, scan) -> None:
        self._scanning = False
        self._scan.setEnabled(True)
        self._status.setText(
            f"Scanned {self._root}. The sections below each have their own "
            "button: the heavy reads only run when asked.")
        self._numbers.setVisible(True)
        self._files.show_(f"{len(scan.files):,}")
        self._audio.show_(f"{len(scan.audio):,}")
        self._size.show_(human_size(scan.total_size()))
        self._tabs.setVisible(True)
        self._contents.set_scan(scan)
        self._duplicates.set_scan(scan, self._root)
        self._filtering.set_scan(scan, self._root)
        self._junk.set_root(self._root)
        self._unreadable.set_scan(scan, self._root)
