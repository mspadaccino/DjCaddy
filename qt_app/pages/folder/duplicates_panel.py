"""I duplicati: gruppi per MD5, spunte per livello, quarantena.

Stessa gerarchia della pagina Streamlit: A (stesso MD5, stessa cartella —
spuntati in partenza), B (stesso file in cartelle diverse — spunte da
mettere, perché lì si disfa un ordine che magari serve), C (nomi simili,
contenuti diversi), più i gruppi ROTTI, identici perché ugualmente vuoti,
esclusi da tutto.

Nei livelli B e C la riga mette a confronto DUE file e la casella è una per
ciascuno: `keep` è una proposta, non una decisione (vedi `DuplicateGroup`),
e chi dei due se ne va lo dice l'utente — anche il presunto originale.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QLineEdit, QProgressBar,
                               QPushButton, QTabWidget, QVBoxLayout, QWidget)

from core.analysis.duplicates import (QUARANTINE_DIRNAME,
                                      apply_quarantine_plan,
                                      build_quarantine_plan, duplicates_of,
                                      find_duplicates, write_csv)
from core.analysis.folder_scan import human_size
from qt_app import theme
from qt_app.pages.common import ConfirmBar, dim, reveal_in_files
from qt_app.state import AppState
from qt_app.widgets.track_table import (CHECK_A_COLUMN, CHECK_B_COLUMN,
                                        TrackTable)
from qt_app.workers import Progress, run_in_pool


def duplicate_rows(groups, full_paths: bool = False) -> pd.DataFrame:
    """Le righe delle tabelle dei duplicati, una per copia di troppo.

    Con `full_paths` le due copie sono mostrate per intero invece che come
    nome più una cartella sola: nei livelli B e C stanno in cartelle
    DIVERSE, e vedere solo quella di una delle due non dice dove sia
    l'altra — che è proprio l'informazione che serve per decidere. Lì la
    riga porta anche una casella per file (`✓ A` sul presunto originale,
    `✓ B` sulla copia): si mette in quarantena l'uno o l'altro, e la
    proposta di `keep` resta una proposta.
    """
    pair = ([CHECK_A_COLUMN, "file A", CHECK_B_COLUMN, "file B"] if full_paths
            else ["folder", "keep", "duplicate"])
    return pd.DataFrame(
        [{**({CHECK_A_COLUMN: "", "file A": str(g.keep),
              CHECK_B_COLUMN: "", "file B": str(dup)}
             if full_paths else
             {"folder": str(g.folder), "keep": g.keep.name,
              "duplicate": dup.name}),
          "size": human_size(g.size), "copies": g.copies,
          "md5": (g.md5 or "")[:12], "_path": str(dup),
          **({"_path2": str(g.keep)} if full_paths else {}),
          "_bytes": g.size}
         for g in groups for dup in g.duplicates],
        columns=[*pair, "size", "copies", "md5", "_path",
                 *(["_path2"] if full_paths else []), "_bytes"])


def files_and_bytes(frame: pd.DataFrame,
                    only: set[str] | None = None) -> tuple[list[str], int]:
    """I file distinti di una tabella e quanto pesano in tutto.

    Una riga ne porta uno (livello A) o due (B e C), e lo stesso file torna
    su più righe quando il gruppo ha tre copie: contarlo due volte gonfierebbe
    lo spazio che si libera. Con `only` si contano solo i percorsi dati.
    """
    seen: dict[str, int] = {}
    if not len(frame) or "_path" not in frame:
        return [], 0
    for _, row in frame.iterrows():
        for field in ("_path2", "_path"):
            path = row.get(field)
            if isinstance(path, str) and (only is None or path in only):
                seen.setdefault(path, int(row["_bytes"]))
    return list(seen), sum(seen.values())


class _Section(QWidget):
    """Una tabella spuntabile con Select all/none e il conto di cosa libera."""

    changed = Signal()

    def __init__(self, state: AppState, note: str, parent=None) -> None:
        super().__init__(parent)
        self._note = dim(note)
        self._count = dim("")
        pick_all = QPushButton("Select all")
        pick_all.clicked.connect(lambda: self.table.set_all_picked(True))
        pick_none = QPushButton("Select none")
        pick_none.clicked.connect(lambda: self.table.set_all_picked(False))
        self.table = TrackTable(checkable=True, library_menu=False)
        self.table.play_requested.connect(state.play)
        self.table.row_activated.connect(state.play)
        self.table.reveal_requested.connect(reveal_in_files)
        self.table.selection_paths_changed.connect(self._on_picked)

        row = QHBoxLayout()
        row.addWidget(pick_all)
        row.addWidget(pick_none)
        row.addWidget(self._count, stretch=1)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)
        box.addWidget(self._note)
        box.addLayout(row)
        box.addWidget(self.table, stretch=1)

    def set_rows(self, rows: pd.DataFrame, preselect: bool) -> None:
        self._rows = rows
        self.table.set_tracks(rows)
        self.table.set_all_picked(preselect and bool(len(rows)))

    def chosen(self) -> tuple[list[Path], int]:
        picked, freed = files_and_bytes(self.table.model_.frame,
                                        set(self.table.selected_paths()))
        return [Path(p) for p in picked], freed

    def _on_picked(self, _paths) -> None:
        chosen, freed = self.chosen()
        listed, total = files_and_bytes(self.table.model_.frame)
        self._count.setText(
            f"{len(chosen):,} of {len(listed):,} selected · "
            f"{human_size(freed)} would be freed "
            f"(of {human_size(total)} in this section)")
        self.changed.emit()


class DuplicatesPanel(QWidget):
    """La ricerca, i tre livelli, il report CSV e la quarantena."""

    rescan_needed = Signal()

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._root: Path | None = None
        self._audio = []
        self._report = None

        self._find = QPushButton("Find duplicates")
        self._find.setStyleSheet(
            theme.primary_button())
        self._find.clicked.connect(self._on_find)
        self._find.setEnabled(False)
        self._bar = QProgressBar()
        self._bar.setVisible(False)
        self._told = dim(
            "Files are grouped by size first, and only same-size files get "
            "hashed — two files of different size cannot be identical. "
            f"The `{QUARANTINE_DIRNAME}/` folder is skipped so a second "
            "run does not re-find what the first set aside.")
        find_row = QHBoxLayout()
        find_row.addWidget(self._told, stretch=1)
        find_row.addWidget(self._find)

        self._level_a = _Section(
            state, "Byte-identical files sitting side by side. Ticked by "
                   "default.")
        self._level_b = _Section(
            state, "Byte-identical copies sitting in different folders — "
                   "often deliberate. Nothing is ticked to begin with, "
                   "because this is the section where a folder layout you "
                   "rely on would be undone. The two files are the same "
                   "bytes, so tick whichever of the pair you want to move: "
                   "right-click shows either one in the file manager.")
        self._level_c = _Section(
            state, "These are NOT the same file: different edits, remixes "
                   "or rips that happen to be named alike. Nothing is "
                   "ticked, and here a tick loses a version you do not "
                   "have elsewhere — right-click shows either file in the "
                   "file manager, and look at the sizes before choosing.")
        for section in (self._level_a, self._level_b, self._level_c):
            section.changed.connect(self._refresh_plan)

        self._broken_note = dim("")
        self._broken = TrackTable(library_menu=False)
        self._broken.play_requested.connect(state.play)
        self._broken.reveal_requested.connect(reveal_in_files)
        broken_box = QWidget()
        broken_lay = QVBoxLayout(broken_box)
        broken_lay.setContentsMargins(0, 0, 0, 0)
        broken_lay.addWidget(self._broken_note)
        broken_lay.addWidget(self._broken, stretch=1)

        self._levels = QTabWidget()
        self._levels.addTab(self._level_a, "A · same folder")
        self._levels.addTab(self._level_b, "B · other folders")
        self._levels.addTab(self._level_c, "C · similar name")
        self._levels.addTab(broken_box, "⚠ broken")
        self._levels.setVisible(False)

        self._csv_name = QLineEdit()
        self._csv = QPushButton("Write CSV report")
        self._csv.clicked.connect(self._on_csv)
        csv_row = QHBoxLayout()
        csv_row.addWidget(QLabel("Report file name"))
        csv_row.addWidget(self._csv_name, stretch=1)
        csv_row.addWidget(self._csv)
        self._csv_row = QWidget()
        self._csv_row.setLayout(csv_row)
        self._csv_row.setVisible(False)
        csv_row.setContentsMargins(0, 0, 0, 0)

        self._plan_told = dim("")
        self._plan_told.setVisible(False)
        self._confirm = ConfirmBar("Move to quarantine", primary=True)
        self._confirm.activated.connect(self._on_quarantine)
        self._confirm.setVisible(False)

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)
        box.addLayout(find_row)
        box.addWidget(self._bar)
        box.addWidget(self._levels, stretch=1)
        box.addWidget(self._csv_row)
        box.addWidget(self._plan_told)
        box.addWidget(self._confirm)

    # ------------------------------------------------------------------
    def set_scan(self, scan, root: Path) -> None:
        self._root = root
        self._audio = scan.audio
        self._report = None
        self._find.setEnabled(bool(self._audio))
        self._find.setText(
            f"Find duplicates among {len(self._audio):,} audio files")
        self._csv_name.setText(f"{root.name}_duplicates.csv")
        self._levels.setVisible(False)
        self._csv_row.setVisible(False)
        self._plan_told.setVisible(False)
        self._confirm.setVisible(False)

    # ------------------------------------------------------------------
    def _on_find(self) -> None:
        audio = list(self._audio)
        self._find.setEnabled(False)
        self._bar.setVisible(True)
        self._bar.setRange(0, 1)
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._bar.setFormat("Hashing…")
        progress = Progress(self)
        progress.count.connect(self._on_progress)

        def _job():
            return find_duplicates(audio, progress=progress.count.emit)

        run_in_pool(_job, self._on_report,
                    lambda t: (self._bar.setVisible(False),
                               self._find.setEnabled(True),
                               self._told.setText(
                                   f"The search failed: {t}")))

    def _on_progress(self, done: int, total: int) -> None:
        self._bar.setRange(0, max(1, total))
        self._bar.setValue(done)
        self._bar.setFormat(f"Hashing {done:,}/{total:,} candidate files…")

    def _on_report(self, report) -> None:
        self._bar.setVisible(False)
        self._find.setEnabled(True)
        self._report = report

        a_files = duplicates_of(report.same_folder)
        b_files = duplicates_of(report.other_folder)
        recoverable = sum(g.wasted_bytes for g in report.same_folder)
        self._told.setText(
            f"A — same folder: {len(report.same_folder):,} groups, "
            f"{len(a_files):,} duplicate files · "
            f"B — other folders: {len(report.other_folder):,} groups, "
            f"{len(b_files):,} files · "
            f"C — similar name: {len(report.similar_name):,} groups. "
            f"{report.hashed_files:,} files hashed · "
            f"{human_size(recoverable)} recoverable from level A alone.")

        self._levels.setVisible(True)
        self._level_a.set_rows(duplicate_rows(report.same_folder),
                               preselect=True)
        self._level_b.set_rows(
            duplicate_rows(report.other_folder, full_paths=True),
            preselect=False)
        table_c = duplicate_rows(report.similar_name, full_paths=True)
        self._level_c.set_rows(
            table_c.drop(columns=["md5"]) if len(table_c) else table_c,
            preselect=False)
        self._levels.setTabText(
            0, f"A · same folder ({len(report.same_folder)})")
        self._levels.setTabText(
            1, f"B · other folders ({len(report.other_folder)})")
        self._levels.setTabText(
            2, f"C · similar name ({len(report.similar_name)})")

        broken_files = sum(len(g.paths) for g in report.broken)
        self._levels.setTabText(3, f"⚠ broken ({broken_files})")
        if report.broken:
            self._broken_note.setText(
                f"{broken_files:,} files in {len(report.broken):,} groups "
                "are identical because they are equally BROKEN, not because "
                "they are the same music — excluded from everything above. "
                "They are different tracks, so removing one would destroy "
                "the only trace of a song you are missing. Use Unreadable "
                "files to quarantine them one by one.")
            self._broken.set_tracks(pd.DataFrame(
                [{"file": p.name, "folder": str(p.parent),
                  "size": human_size(g.size), "why": g.reason,
                  "_path": str(p)}
                 for g in report.broken for p in g.paths]))
        else:
            self._broken_note.setText("No broken groups.")
            self._broken.set_tracks(pd.DataFrame(
                columns=["file", "folder", "size", "why", "_path"]))

        self._csv_row.setVisible(True)
        self._refresh_plan()

    # ------------------------------------------------------------------
    def _refresh_plan(self) -> None:
        if self._report is None or self._root is None:
            return
        chosen_a, bytes_a = self._level_a.chosen()
        chosen_b, bytes_b = self._level_b.chosen()
        chosen_c, bytes_c = self._level_c.chosen()
        plan = build_quarantine_plan(chosen_a + chosen_b + chosen_c,
                                     self._root)
        self._plan_told.setVisible(True)
        self._confirm.setVisible(bool(plan))
        if not plan:
            self._plan_told.setText(
                "Nothing ticked yet — select the files to move in the "
                "tables above. Quarantine keeps the original folder "
                "structure, nothing is deleted: check djay Pro still sees "
                "everything, then empty the folder yourself.")
            return
        self._plan_told.setText(
            f"{len(plan):,} files ticked, freeing "
            f"{human_size(bytes_a + bytes_b + bytes_c)} — {len(chosen_a):,} "
            f"from A ({human_size(bytes_a)}), {len(chosen_b):,} from B "
            f"({human_size(bytes_b)}), {len(chosen_c):,} from C "
            f"({human_size(bytes_c)}). They move into "
            f"{self._root / QUARANTINE_DIRNAME}, keeping their folder "
            "structure so you can see where each came from and put it "
            "back.")
        self._confirm.set_ask(
            f"I have read the list and want to move these {len(plan):,} "
            "files")

    def _on_quarantine(self) -> None:
        chosen_a, _ = self._level_a.chosen()
        chosen_b, _ = self._level_b.chosen()
        chosen_c, _ = self._level_c.chosen()
        plan = build_quarantine_plan(chosen_a + chosen_b + chosen_c,
                                     self._root)
        if not plan:
            return

        def _job():
            return apply_quarantine_plan(plan, dry_run=False)

        def _done(result) -> None:
            moved, errors = result
            self._plan_told.setText(
                f"{moved:,} files moved to "
                f"{self._root / QUARANTINE_DIRNAME}."
                + (f" {len(errors)} could not be moved: "
                   + "; ".join(f"{p} ({e})" for p, e in errors[:3])
                   if errors else ""))
            self.rescan_needed.emit()

        run_in_pool(_job, _done,
                    lambda t: self._plan_told.setText(
                        f"The move failed: {t}"))

    def _on_csv(self) -> None:
        if self._report is None or self._root is None:
            return
        try:
            written = write_csv(self._report.all_groups(),
                                self._root / self._csv_name.text())
            self._plan_told.setText(f"Report written: {written}")
            self._plan_told.setVisible(True)
        except OSError as trouble:
            self._plan_told.setText(f"Could not write the report: {trouble}")
            self._plan_told.setVisible(True)
