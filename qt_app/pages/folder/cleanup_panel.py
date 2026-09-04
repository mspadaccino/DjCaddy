"""Junk files e file illeggibili: le due pulizie a bassa ambiguità.

I sidecar `._<nome>` di macOS si CANCELLANO (dentro non c'è niente da
tenere, e il Finder li rifà comunque), sempre dopo la verifica del
contenuto. I file che nessun lettore apre invece erano musica: vanno in
quarantena, non nel cestino — la lista è quello che serve per riscaricarli.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QHBoxLayout, QProgressBar, QPushButton,
                               QVBoxLayout, QWidget)

from core.analysis.duplicates import (QUARANTINE_DIRNAME,
                                      apply_quarantine_plan,
                                      build_quarantine_plan)
from core.analysis.folder_scan import (CHECK_THREADS, check_integrity,
                                       delete_sidecars, find_sidecars,
                                       human_size)
from qt_app import theme
from qt_app.pages.common import ConfirmBar, Metric, dim, reveal_in_files
from qt_app.state import AppState
from qt_app.widgets.track_table import TrackTable
from qt_app.workers import Progress, run_in_pool

# Secondi a file col controllo in parallelo, misurati su questa libreria
# via USB: 5,8 s per 800 file con 8 thread, contro 40,3 s in fila.
SECONDS_PER_CHECK = 0.0073


class JunkPanel(QWidget):
    """I sidecar AppleDouble: trovati per contenuto, cancellati su conferma."""

    rescan_needed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._root: Path | None = None
        self._sidecars = None

        self._find = QPushButton("Find sidecar files")
        theme.style(self._find, theme.primary_button)
        self._find.clicked.connect(self._on_find)
        self._find.setEnabled(False)
        self._told = dim(
            "Small companion files macOS leaves beside the real ones, "
            "named `._<name>`. They carry the track's extension but hold "
            "no audio, and nothing plays them. Deleted, not quarantined: "
            "there is nothing inside worth keeping, and macOS recreates "
            "them next time the Finder touches a folder. Each file's "
            "content is re-checked immediately before removal.")
        head = QHBoxLayout()
        head.addWidget(self._told, stretch=1)
        head.addWidget(self._find)

        self._found = Metric("Sidecars found",
                             "Name starts with ._ AND the content is a "
                             "real AppleDouble.")
        self._space = Metric("Space they take")
        numbers = QHBoxLayout()
        numbers.addWidget(self._found)
        numbers.addWidget(self._space)
        self._numbers = QWidget()
        self._numbers.setLayout(numbers)
        numbers.setContentsMargins(0, 0, 0, 0)
        self._numbers.setVisible(False)

        self._warn = dim("")
        theme.style(self._warn, lambda: f"color: {theme.WARN};")
        self._warn.setVisible(False)
        self._listed = TrackTable(library_menu=False, playable=False)
        self._listed.reveal_requested.connect(reveal_in_files)
        self._listed.setVisible(False)
        self._confirm = ConfirmBar("Delete sidecar files", primary=True)
        self._confirm.activated.connect(self._on_delete)
        self._confirm.setVisible(False)

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)
        box.addLayout(head)
        box.addWidget(self._numbers)
        box.addWidget(self._warn)
        box.addWidget(self._listed, stretch=1)
        box.addWidget(self._confirm)
        box.addStretch(0)

    # ------------------------------------------------------------------
    def set_root(self, root: Path) -> None:
        self._root = root
        self._sidecars = None
        self._find.setEnabled(True)
        for widget in (self._numbers, self._warn, self._listed,
                       self._confirm):
            widget.setVisible(False)

    def _on_find(self) -> None:
        root = self._root
        self._find.setEnabled(False)
        run_in_pool(lambda: find_sidecars(root), self._on_found,
                    lambda t: (self._find.setEnabled(True),
                               self._told.setText(f"The walk failed: {t}")))

    def _on_found(self, report) -> None:
        self._find.setEnabled(True)
        self._sidecars = report
        self._numbers.setVisible(True)
        self._found.show_(f"{len(report.confirmed):,}")
        self._space.show_(human_size(report.freed_bytes))

        troubles = []
        if report.unverified:
            troubles.append(
                f"{len(report.unverified)} file(s) are named like a "
                "sidecar but are not one. They will NOT be touched — "
                "deleting on the strength of a name is how a real file "
                "gets lost.")
        if not report.looked_properly:
            troubles.append(
                f"The folder could not be read, so this says nothing "
                f"about what is in it: {report.root_error}. Press the "
                "button again once it is reachable.")
        self._warn.setText("\n".join(troubles))
        self._warn.setVisible(bool(troubles))

        if report.looked_properly and not report.confirmed:
            self._told.setText(
                f"No sidecar files here — {report.walked:,} entries "
                "walked and none of them was one.")
        self._listed.setVisible(bool(report.confirmed))
        self._confirm.setVisible(bool(report.confirmed))
        if report.confirmed:
            self._listed.set_tracks(pd.DataFrame(
                [{"path": str(p), "_path": str(p)}
                 for p in report.confirmed],
                columns=["path", "_path"]))
            self._confirm.set_ask(
                f"Delete these {len(report.confirmed):,} files "
                f"({human_size(report.freed_bytes)})")

    def _on_delete(self) -> None:
        if self._sidecars is None:
            return
        doomed = list(self._sidecars.confirmed)

        def _job():
            return delete_sidecars(doomed, dry_run=False)

        def _done(result) -> None:
            removed, freed, errors = result
            self._told.setText(
                f"{removed:,} sidecar files deleted, {human_size(freed)} "
                "freed."
                + (f" {len(errors)} skipped." if errors else ""))
            self.rescan_needed.emit()

        run_in_pool(_job, _done,
                    lambda t: self._told.setText(f"Deleting failed: {t}"))


class UnreadablePanel(QWidget):
    """I file che nessun lettore apre: decoder alla mano, poi quarantena."""

    rescan_needed = Signal()

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._root: Path | None = None
        self._audio = []
        self._integrity = None

        self._check = QPushButton("Check audio files")
        theme.style(self._check, theme.primary_button)
        self._check.clicked.connect(self._on_check)
        self._check.setEnabled(False)
        self._told = dim("")
        head = QHBoxLayout()
        head.addWidget(self._told, stretch=1)
        head.addWidget(self._check)
        self._bar = QProgressBar()
        self._bar.setVisible(False)

        self._checked = Metric("Checked")
        self._bad = Metric("Unreadable")
        self._missing = Metric(
            "Vanished", "Listed by the scan but gone by the time we "
                        "looked — moved or renamed in the meantime. "
                        "Nothing to do.")
        numbers = QHBoxLayout()
        for metric in (self._checked, self._bad, self._missing):
            numbers.addWidget(metric)
        self._numbers = QWidget()
        self._numbers.setLayout(numbers)
        numbers.setContentsMargins(0, 0, 0, 0)
        self._numbers.setVisible(False)

        self._listed = TrackTable(library_menu=False)
        self._listed.wire_play(state.play, on_activate=False)
        self._listed.reveal_requested.connect(reveal_in_files)
        self._listed.setVisible(False)
        self._confirm = ConfirmBar("Quarantine unreadable files",
                                   primary=True)
        self._confirm.activated.connect(self._on_quarantine)
        self._confirm.setVisible(False)

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)
        box.addLayout(head)
        box.addWidget(self._bar)
        box.addWidget(self._numbers)
        box.addWidget(self._listed, stretch=1)
        box.addWidget(self._confirm)

    # ------------------------------------------------------------------
    def set_scan(self, scan, root: Path) -> None:
        self._root = root
        self._audio = scan.audio
        self._integrity = None
        self._check.setEnabled(bool(self._audio))
        self._check.setText(f"Check {len(self._audio):,} audio files")
        self._told.setText(
            "Tracks that no player will open — usually a download that "
            "arrived truncated, or an error page saved with an .mp3 name. "
            "Every file is opened with the decoder, the same one the "
            "tagging uses: about "
            f"{len(self._audio) * SECONDS_PER_CHECK / 60:.0f} min for "
            f"{len(self._audio):,} files, {CHECK_THREADS} at a time. "
            "Unlike junk these were meant to be music, so they are moved "
            "to quarantine, not deleted.")
        for widget in (self._numbers, self._listed, self._confirm):
            widget.setVisible(False)

    def _on_check(self) -> None:
        audio = list(self._audio)
        self._check.setEnabled(False)
        self._bar.setVisible(True)
        self._bar.setRange(0, max(1, len(audio)))
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._bar.setFormat("Checking…")
        progress = Progress(self)
        progress.count.connect(self._on_progress)

        def _job():
            return check_integrity(audio, progress=progress.count.emit)

        run_in_pool(_job, self._on_report,
                    lambda t: (self._bar.setVisible(False),
                               self._check.setEnabled(True),
                               self._told.setText(f"The check failed: {t}")))

    def _on_progress(self, done: int, total: int) -> None:
        self._bar.setRange(0, max(1, total))
        self._bar.setValue(done)
        self._bar.setFormat(f"Checked {done:,}/{total:,}…")

    def _on_report(self, report) -> None:
        self._bar.setVisible(False)
        self._check.setEnabled(True)
        self._integrity = report
        self._numbers.setVisible(True)
        self._checked.show_(f"{report.checked:,}")
        self._bad.show_(f"{len(report.bad):,}",
                        human_size(sum(b.size for b in report.bad)))
        self._missing.show_(f"{len(report.missing):,}")

        if not report.bad:
            self._told.setText("Every file opened correctly.")
            self._listed.setVisible(False)
            self._confirm.setVisible(False)
            return
        bad_bytes = sum(b.size for b in report.bad)
        self._told.setText(
            f"Moving them to {self._root / QUARANTINE_DIRNAME} keeps the "
            "folder structure, so you can see which album each came from "
            "and re-download just those.")
        self._listed.set_tracks(pd.DataFrame(
            [{"file": b.path.name, "folder": str(b.path.parent),
              "size": human_size(b.size), "why": b.reason,
              "_path": str(b.path)}
             for b in report.bad]))
        self._listed.setVisible(True)
        self._confirm.setVisible(True)
        self._confirm.set_ask(
            f"Move these {len(report.bad):,} unreadable files to "
            f"quarantine ({human_size(bad_bytes)})")

    def _on_quarantine(self) -> None:
        if self._integrity is None:
            return
        plan = build_quarantine_plan(
            [b.path for b in self._integrity.bad], self._root)
        if not plan:
            return

        def _job():
            return apply_quarantine_plan(plan, dry_run=False)

        def _done(result) -> None:
            moved, errors = result
            self._told.setText(
                f"{moved:,} files moved."
                + (f" {len(errors)} could not be moved." if errors else ""))
            self.rescan_needed.emit()

        run_in_pool(_job, _done,
                    lambda t: self._told.setText(f"The move failed: {t}"))
