"""La tabella cue: frasi e marcatori vocali, correggibili al posto giusto.

`view_rows` è la parte pura — dalle righe correnti (id, kind, tag, start)
alle righe da mostrare in ordine di tempo, con fine, battiti e slot
ricalcolati — e usa le stesse regole di core della pagina Streamlit
(`phrase_ends`, `plan_rekordbox_markers`): la colonna deve dire le stesse
di là. `CueTable` la disegna: Tag a tendina, Start editabile in mm:ss,
il ▶ che suona la riga e il 🗑 che la toglie.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHeaderView,
                               QStyledItemDelegate, QTableWidget,
                               QTableWidgetItem)

from core.analysis.cue_export import (PHRASE_START, phrase_ends,
                                      plan_rekordbox_markers, wants_hot)
from core.analysis.models import (SECTION_LABELS, VOCAL_END, VOCAL_START,
                                  format_elapsed, parse_mmss)

# Una frase è una riga sola (il suo inizio); la fine è informazione nella
# colonna "End", non un tag a sé — stesse opzioni della pagina Streamlit.
TAG_OPTIONS = list(SECTION_LABELS) + [VOCAL_START, VOCAL_END]

_HEADERS = ["#", "▶", "Tag", "Start (mm:ss)", "End (mm:ss)", "Beats",
            "Hot", "slot", "🗑"]
_TIPS = {
    "End (mm:ss)": "Dove finisce la frase, solo come riferimento: NON "
                   "diventa un cue. Le sezioni sono contigue, quindi "
                   "coincide con l'inizio della frase successiva. "
                   "Le righe vocali hanno la loro fine nella riga gemella.",
    "Beats": "Lunghezza della frase in battiti, ricalcolata dalla fine "
             "corrente.",
    "Hot": "Spuntata, questa riga chiede uno degli 8 pad hot cue (A-H); "
           "spenta diventa un memory cue, che non ha numero e non si "
           "esaurisce mai. È una RICHIESTA: i pad sono otto, quindi la nona "
           "riga spuntata scende a memory da sola — la colonna qui accanto "
           "dice sempre dove finisce davvero.",
    "slot": "Dove finisce questa riga una volta scritta: gli INIZI di frase "
            "sui pad hot-cue 1-8 (la posizione decide il colore), le "
            "regioni vocali sugli slot loop 1-8 (inizio e fine nello "
            "stesso slot). Assegnati in ordine di tempo. Vuoto = non "
            "viene scritta.",
    "Start (mm:ss)": "Formato mm:ss o mm:ss.d, es. 1:07.3",
}
_COL_PLAY, _COL_TAG, _COL_START = 1, 2, 3
_COL_HOT, _COL_SLOT, _COL_DEL = 6, 7, 8


def view_rows(rows: list[dict], bpm: float | None,
              analysis_end: dict) -> list[dict]:
    """Dalle righe correnti alle righe mostrate, in ordine di tempo.

    `rows`: [{id, kind, tag, start}] — il tag e lo start sono quelli DOPO
    gli edit. Ritorna le stesse righe ordinate per start con in più `end`
    (le frasi: l'inizio della successiva, l'ultima la fine rilevata),
    `beats` (solo le frasi, dal BPM) e `slot` (dove finisce la riga
    in rekordbox).
    """
    ordered = sorted(rows, key=lambda r: r["start"])
    kinds = {r["id"]: r["kind"] for r in ordered}
    starts = {r["id"]: r["start"] for r in ordered}
    ends = phrase_ends(kinds, starts, analysis_end)
    plan = plan_rekordbox_markers([
        {"id": r["id"], "kind": r["kind"], "start": r["start"],
         "hot": r.get("hot")}
        for r in ordered])
    beat_seconds = (60.0 / bpm) if bpm else None
    out = []
    for r in ordered:
        end = ends.get(r["id"])
        beats = (round((end - r["start"]) / beat_seconds, 1)
                 if r["kind"] == PHRASE_START and beat_seconds
                 and end is not None else None)
        out.append({**r, "end": end, "beats": beats,
                    "hot": wants_hot(r),
                    "slot": plan.slot_label.get(r["id"], "")})
    return out


class _TagDelegate(QStyledItemDelegate):
    """La tendina dei tag, al posto della SelectboxColumn."""

    def createEditor(self, parent, option, index):
        box = QComboBox(parent)
        box.addItems(TAG_OPTIONS)
        return box

    def setEditorData(self, editor, index) -> None:
        editor.setCurrentText(index.data() or "")

    def setModelData(self, editor, model, index) -> None:
        model.setData(index, editor.currentText())


class CueTable(QTableWidget):
    """Le righe cue del brano in revisione, coi loro tre gesti.

    I segnali portano l'id STABILE della riga (`sec3`, `vs1`…), non la
    posizione: la tabella si riordina per tempo a ogni edit, e una
    posizione ricordata indicherebbe subito la riga sbagliata.
    """

    play_clicked = Signal(str)          # rid
    delete_clicked = Signal(str)        # rid
    tag_edited = Signal(str, str)       # rid, etichetta nuova
    start_edited = Signal(str, float)   # rid, secondi nuovi
    start_rejected = Signal()           # formato non valido: si ricarica
    hot_toggled = Signal(str, bool)     # rid, pad sì/no

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setColumnCount(len(_HEADERS))
        for i, name in enumerate(_HEADERS):
            item = QTableWidgetItem(name)
            if name in _TIPS:
                item.setToolTip(_TIPS[name])
            self.setHorizontalHeaderItem(i, item)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(26)
        self.setShowGrid(False)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.SelectedClicked)
        self.setItemDelegateForColumn(_COL_TAG, _TagDelegate(self))
        for column, width in ((0, 40), (_COL_PLAY, 30), (_COL_TAG, 130),
                              (_COL_START, 110), (4, 110), (5, 64),
                              (_COL_HOT, 44), (_COL_SLOT, 110),
                              (_COL_DEL, 30)):
            self.setColumnWidth(column, width)
        self.horizontalHeader().setSectionResizeMode(
            _COL_TAG, QHeaderView.ResizeMode.Stretch)
        self.cellClicked.connect(self._on_cell_clicked)
        self.itemChanged.connect(self._on_item_changed)
        self._filling = False

    # ------------------------------------------------------------------
    def set_rows(self, shown: list[dict]) -> None:
        """Ripopola dalla vista di `view_rows`. L'id vive nella riga."""
        self._filling = True
        self.setRowCount(len(shown))
        for at, row in enumerate(shown):
            def cell(text: str, editable: bool = False) -> QTableWidgetItem:
                item = QTableWidgetItem(text)
                flags = (Qt.ItemFlag.ItemIsEnabled
                         | Qt.ItemFlag.ItemIsSelectable)
                if editable:
                    flags |= Qt.ItemFlag.ItemIsEditable
                item.setFlags(flags)
                return item

            number = cell(str(at + 1))
            number.setData(Qt.ItemDataRole.UserRole, row["id"])
            self.setItem(at, 0, number)
            play = cell("▶")
            play.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(at, _COL_PLAY, play)
            self.setItem(at, _COL_TAG, cell(row["tag"], editable=True))
            self.setItem(at, _COL_START,
                         cell(format_elapsed(row["start"]), editable=True))
            self.setItem(at, 4, cell(format_elapsed(row["end"])
                                     if row["end"] is not None else ""))
            self.setItem(at, 5, cell("" if row["beats"] is None
                                     else f"{row['beats']:g}"))
            hot = cell("")
            hot.setFlags(hot.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            hot.setCheckState(Qt.CheckState.Checked if row.get("hot")
                              else Qt.CheckState.Unchecked)
            self.setItem(at, _COL_HOT, hot)
            self.setItem(at, _COL_SLOT, cell(row["slot"]))
            trash = cell("🗑")
            trash.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(at, _COL_DEL, trash)
        self._filling = False

    def row_id(self, at: int) -> str | None:
        item = self.item(at, 0)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    # ------------------------------------------------------------------
    def _on_cell_clicked(self, at: int, column: int) -> None:
        rid = self.row_id(at)
        if rid is None:
            return
        if column == _COL_PLAY:
            self.play_clicked.emit(rid)
        elif column == _COL_DEL:
            self.delete_clicked.emit(rid)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._filling:
            return
        rid = self.row_id(item.row())
        if rid is None:
            return
        if item.column() == _COL_TAG:
            self.tag_edited.emit(rid, item.text())
        elif item.column() == _COL_HOT:
            self.hot_toggled.emit(
                rid, item.checkState() == Qt.CheckState.Checked)
        elif item.column() == _COL_START:
            seconds = parse_mmss(item.text())
            if seconds is None:
                # Come di là: il valore precedente resta, e lo si dice.
                self.start_rejected.emit()
            else:
                self.start_edited.emit(rid, float(seconds))
