"""Library filtering: mix interi e frammenti fuori dalla collezione.

Due segnali — quanto dura il file e come si chiama — perché per i megamix
nessuno dei due basta da solo, e per i file corti la durata è l'unico che
funziona. I file sotto il minuto si possono ANCHE ascoltare nel finale:
un mp3 tagliato a metà dichiara la durata che ha davvero, quindi solo
l'audio dice se muore sul più bello o sfuma.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel,
                               QLineEdit, QProgressBar, QPushButton,
                               QSpinBox, QVBoxLayout, QWidget)

from core.analysis.duplicates import (QUARANTINE_DIRNAME,
                                      apply_quarantine_plan,
                                      build_quarantine_plan)
from core.analysis.folder_scan import (human_duration, human_size,
                                       read_durations)
from core.analysis.mix_names import (DEFAULT_KEYWORDS, NOT_BY_DEFAULT,
                                     looks_like_a_mashup, matching_words,
                                     parse_keywords)
from core.analysis.truncation import inspect
from qt_app import theme
from qt_app.pages.common import ConfirmBar, Metric, dim, reveal_in_files
from qt_app.state import AppState
from qt_app.widgets.track_table import TrackTable
from qt_app.workers import Progress, run_in_pool

RULES = ["both signals fire", "either one fires", "the duration alone"]


def filter_reasons(tracks, low: int, high: int, keywords, vs_rule: bool,
                   rule: str) -> tuple[dict[Path, list[str]], int, int]:
    """Chi entra in lista e perché: ({path: [ragioni]}, per durata, per nome).

    `rule` è una voce di RULES. La stessa aritmetica della pagina
    Streamlit: l'AND si allarga abbassando la soglia, l'OR è la rete larga
    che prende anche i mashup, la sola durata è l'unica via ai file corti.
    I due conteggi sono i bacini dei singoli segnali PRIMA della regola,
    come le metriche di là: dicono quanto pesca ciascuno.
    """
    low_seconds, high_seconds = low * 60.0, high * 60.0
    by_length = {t.path for t in tracks
                 if low_seconds <= t.seconds <= high_seconds}
    by_name: dict[Path, list[str]] = {}
    for track in tracks:
        hits = matching_words(track.path.name, keywords)
        if hits:
            by_name.setdefault(track.path, []).append(
                "name: " + ", ".join(hits))
        if vs_rule and looks_like_a_mashup(track.path.name):
            by_name.setdefault(track.path, []).append("two “vs”")

    reasons: dict[Path, list[str]] = {}
    for track in tracks:
        long_enough, named = track.path in by_length, track.path in by_name
        if rule == "the duration alone":
            kept = long_enough
        elif rule == "both signals fire":
            kept = long_enough and named
        else:
            kept = long_enough or named
        if not kept:
            continue
        reasons[track.path] = (
            ([f"{low}–{high} min"] if long_enough else [])
            + by_name.get(track.path, []))
    return reasons, len(by_length), len(by_name)


class FilteringPanel(QWidget):
    """Durate in memoria, criteri dal vivo, quarantena su conferma."""

    rescan_needed = Signal()

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._root: Path | None = None
        self._audio = []
        self._durations = None
        self._verdicts: dict[Path, object] = {}
        self._matching = []
        self._reasons: dict[Path, list[str]] = {}

        self._read = QPushButton("Read durations")
        theme.style(self._read, theme.primary_button)
        self._read.clicked.connect(self._on_read)
        self._read.setEnabled(False)
        self._bar = QProgressBar()
        self._bar.setVisible(False)
        self._told = dim(
            "Files that do not belong in a collection meant for mixing "
            "single tracks: whole sets and megamixes at the long end, "
            "fragments, previews and truncated downloads at the short end.")
        head = QHBoxLayout()
        head.addWidget(self._told, stretch=1)
        head.addWidget(self._read)

        self._low = QSpinBox()
        self._low.setToolTip(
            "Around 10 minutes is the useful lower line for megamixes — "
            "half of them are longer, while only 1% of extended versions "
            "are. Start it at 0 for the other end of the problem: 0–1 "
            "catches the fragments and truncated downloads.")
        self._high = QSpinBox()
        self._words = QLineEdit(", ".join(DEFAULT_KEYWORDS))
        self._words.setToolTip(
            "Comma separated, matched anywhere in the file name, case "
            "insensitive. Words deliberately left out, measured on this "
            "library:\n" + "\n".join(f"• {w.strip()} — {why}"
                                     for w, why in NOT_BY_DEFAULT.items()))
        self._vs = QCheckBox('Catch "A vs B - song vs song"')
        self._vs.setChecked(True)
        self._vs.setToolTip(
            "Two 'vs' in one name means two artists and two titles — a "
            "mashup. One 'vs' alone is usually a real collaboration: "
            "measured here, one catches 2,312 files, two catch 679 and "
            "twelve out of twelve sampled were mashups.")
        self._rule = QComboBox()
        self._rule.addItems(RULES)
        self._rule.setToolTip(
            "Both: long enough AND named like a mix — near-certain; widen "
            "it by lowering the duration until only the name matters. "
            "Either: the wider net, which also shows mashups. Duration "
            "alone: the only way to reach the short files.")
        for control in (self._low, self._high):
            control.valueChanged.connect(lambda _: self._refresh())
        self._words.textChanged.connect(lambda _: self._refresh())
        self._vs.toggled.connect(lambda _: self._refresh())
        self._rule.currentTextChanged.connect(lambda _: self._refresh())

        knobs = QHBoxLayout()
        knobs.addWidget(QLabel("Duration between (minutes)"))
        knobs.addWidget(self._low)
        knobs.addWidget(QLabel("and"))
        knobs.addWidget(self._high)
        knobs.addWidget(QLabel("…or the name contains"))
        knobs.addWidget(self._words, stretch=1)
        self._knobs = QWidget()
        self._knobs.setLayout(knobs)
        knobs.setContentsMargins(0, 0, 0, 0)
        self._knobs.setVisible(False)

        rule_row = QHBoxLayout()
        rule_row.addWidget(QLabel("A file is listed when…"))
        rule_row.addWidget(self._rule)
        rule_row.addWidget(self._vs, stretch=1)
        self._listen = QPushButton("Listen to how the short ones end")
        self._listen.setToolTip(
            "Only for files under a minute: deliberate samples and "
            "interrupted downloads sit side by side there. Two things tell "
            "them apart — whether the file ends at full volume instead of "
            "dying away, and whether it still claims to be a whole song "
            "(ID3 tags live at the head of the file).")
        self._listen.clicked.connect(self._on_listen)
        rule_row.addWidget(self._listen)
        self._rule_row = QWidget()
        self._rule_row.setLayout(rule_row)
        rule_row.setContentsMargins(0, 0, 0, 0)
        self._rule_row.setVisible(False)

        self._matching_told = Metric("Matching")
        self._by_length_told = Metric("By duration")
        self._by_name_told = Metric("By name")
        self._space_told = Metric("Space")
        numbers = QHBoxLayout()
        for metric in (self._matching_told, self._by_length_told,
                       self._by_name_told, self._space_told):
            numbers.addWidget(metric)
        self._numbers = QWidget()
        self._numbers.setLayout(numbers)
        numbers.setContentsMargins(0, 0, 0, 0)
        self._numbers.setVisible(False)

        pick_all = QPushButton("Select all")
        pick_all.clicked.connect(lambda: self._table.set_all_picked(True))
        pick_none = QPushButton("Select none")
        pick_none.clicked.connect(lambda: self._table.set_all_picked(False))
        self._count = dim("")
        picks = QHBoxLayout()
        picks.addWidget(pick_all)
        picks.addWidget(pick_none)
        picks.addWidget(self._count, stretch=1)
        self._picks = QWidget()
        self._picks.setLayout(picks)
        picks.setContentsMargins(0, 0, 0, 0)
        self._picks.setVisible(False)

        self._table = TrackTable(checkable=True, library_menu=False)
        self._table.wire_play(state.play)
        self._table.reveal_requested.connect(reveal_in_files)
        self._table.selection_paths_changed.connect(
            lambda _: self._refresh_chosen())
        self._table.setVisible(False)

        self._confirm = ConfirmBar("Quarantine these", primary=True)
        self._confirm.activated.connect(self._on_quarantine)
        self._confirm.setVisible(False)

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)
        box.addLayout(head)
        box.addWidget(self._bar)
        box.addWidget(self._knobs)
        box.addWidget(self._rule_row)
        box.addWidget(self._numbers)
        box.addWidget(self._picks)
        box.addWidget(self._table, stretch=1)
        box.addWidget(self._confirm)

    # ------------------------------------------------------------------
    def set_scan(self, scan, root: Path) -> None:
        self._root = root
        self._audio = scan.audio
        self._durations = None
        self._verdicts = {}
        self._read.setEnabled(bool(self._audio))
        self._read.setText(
            f"Read durations of {len(self._audio):,} files")
        for widget in (self._knobs, self._rule_row, self._numbers,
                       self._picks, self._table, self._confirm):
            widget.setVisible(False)

    # ------------------------------------------------------------------
    def _on_read(self) -> None:
        audio = list(self._audio)
        self._read.setEnabled(False)
        self._bar.setVisible(True)
        self._bar.setRange(0, max(1, len(audio)))
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._bar.setFormat("Reading headers…")
        progress = Progress(self)
        progress.count.connect(self._on_progress)

        def _job():
            return read_durations(audio, progress=progress.count.emit)

        run_in_pool(_job, self._on_durations,
                    lambda t: (self._bar.setVisible(False),
                               self._read.setEnabled(True),
                               self._told.setText(f"Reading failed: {t}")))

    def _on_progress(self, done: int, total: int) -> None:
        self._bar.setRange(0, max(1, total))
        self._bar.setValue(done)
        self._bar.setFormat(f"{done:,}/{total:,}")

    def _on_durations(self, durations) -> None:
        self._bar.setVisible(False)
        self._read.setEnabled(True)
        self._durations = durations
        if not durations.tracks:
            self._told.setText("No readable durations here.")
            return
        ceiling = max(30, int(durations.longest_minutes) + 1)
        for control in (self._low, self._high):
            control.blockSignals(True)
            control.setRange(0, ceiling)
        self._low.setValue(min(10, ceiling))
        self._high.setValue(ceiling)
        for control in (self._low, self._high):
            control.blockSignals(False)
        if durations.unknown:
            self._told.setText(
                f"{len(durations.unknown):,} file(s) had no readable "
                "duration and are left out of this filter entirely.")
        for widget in (self._knobs, self._rule_row, self._numbers,
                       self._picks, self._table):
            widget.setVisible(True)
        self._refresh()

    # ------------------------------------------------------------------
    def _refresh(self) -> None:
        if self._durations is None:
            return
        keywords = parse_keywords(self._words.text())
        self._reasons, by_length, by_name = filter_reasons(
            self._durations.tracks, self._low.value(), self._high.value(),
            keywords, self._vs.isChecked(), self._rule.currentText())
        self._matching = sorted(
            (t for t in self._durations.tracks if t.path in self._reasons),
            key=lambda t: t.seconds, reverse=True)
        total_bytes = sum(t.size for t in self._matching)
        self._matching_told.show_(f"{len(self._matching):,}")
        self._by_length_told.show_(f"{by_length:,}",
                                   f"{self._low.value()}–"
                                   f"{self._high.value()} min")
        self._by_name_told.show_(f"{by_name:,}")
        self._space_told.show_(human_size(total_bytes))

        shorts = sum(1 for t in self._matching if t.seconds <= 60)
        self._listen.setVisible(bool(shorts))
        if shorts:
            self._listen.setText(
                f"Listen to how the {shorts:,} short ones end")

        self._table.set_tracks(pd.DataFrame(
            [{"duration": human_duration(t.seconds),
              "verdict": (self._verdicts[t.path].verdict
                          if t.path in self._verdicts else "—"),
              "why": " + ".join(self._reasons[t.path]),
              "file": t.path.name, "folder": str(t.path.parent),
              "size": human_size(t.size),
              "_path": str(t.path), "_bytes": t.size}
             for t in self._matching],
            columns=["duration", "verdict", "why", "file", "folder",
                     "size", "_path", "_bytes"]))
        self._table.set_all_picked(bool(self._matching))
        self._refresh_chosen()

    def _refresh_chosen(self) -> None:
        frame = self._table.model_.frame
        picked = set(self._table.selected_paths())
        if "_bytes" not in frame:
            return
        rows = frame[frame["_path"].isin(picked)]
        freed = int(rows["_bytes"].sum()) if len(rows) else 0
        total = int(frame["_bytes"].sum()) if len(frame) else 0
        self._count.setText(
            f"{len(rows):,} of {len(frame):,} selected · "
            f"{human_size(freed)} would be freed "
            f"(of {human_size(total)} matching)")
        self._confirm.setVisible(bool(len(rows)))
        self._confirm.set_ask(
            f"Move these {len(rows):,} files to quarantine "
            f"({human_size(freed)}) — a match might be a set you actually "
            "want, and you can put it back from there")

    # ------------------------------------------------------------------
    def _on_listen(self) -> None:
        shorts = [t for t in self._matching if t.seconds <= 60]
        if not shorts:
            return
        self._listen.setEnabled(False)
        self._bar.setVisible(True)
        self._bar.setRange(0, len(shorts))
        self._bar.setValue(0)
        self._bar.setFormat("Decoding…")
        progress = Progress(self)
        progress.count.connect(self._on_progress)

        def _job():
            verdicts = {}
            for i, track in enumerate(shorts, 1):
                progress.count.emit(i, len(shorts))
                verdicts[track.path] = inspect(track.path, track.seconds)
            return verdicts

        def _done(verdicts) -> None:
            self._bar.setVisible(False)
            self._listen.setEnabled(True)
            self._verdicts.update(verdicts)
            self._refresh()

        run_in_pool(_job, _done,
                    lambda t: (self._bar.setVisible(False),
                               self._listen.setEnabled(True),
                               self._told.setText(f"Decoding failed: {t}")))

    # ------------------------------------------------------------------
    def _on_quarantine(self) -> None:
        chosen = [Path(p) for p in self._table.selected_paths()]
        plan = build_quarantine_plan(chosen, self._root)
        if not plan:
            return

        def _job():
            return apply_quarantine_plan(plan, dry_run=False)

        def _done(result) -> None:
            moved, errors = result
            self._told.setText(
                f"{moved:,} tracks moved to "
                f"{self._root / QUARANTINE_DIRNAME}."
                + (f" {len(errors)} could not be moved." if errors else ""))
            self.rescan_needed.emit()

        run_in_pool(_job, _done,
                    lambda t: self._told.setText(f"The move failed: {t}"))
