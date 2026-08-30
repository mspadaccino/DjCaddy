"""Le tabelle dei brani: QTableView vestito con le pastiglie di core/viz.

È il posto dove Qt restituisce la flessibilità chiesta: Streamlit non sa
riordinare le righe col mouse, QTableView sì (InternalMove), e virtualizza
le righe senza serializzare niente. Le colonne, i colori e la lettura della
riga NON stanno qui: vengono da `core.viz.track_columns`, che è lo stesso
posto da cui li prende Streamlit — stessa tavolozza, stessa regola, per
costruzione.

Tre pezzi: `PandasModel` espone un DataFrame alla vista (le colonne che
cominciano con `_` viaggiano ma non si vedono: `_path` serve a risalire al
brano); `PillDelegate` disegna le pastiglie; `TrackTable` li mette insieme.
"""

from __future__ import annotations

import json

import pandas as pd

from PySide6.QtCore import (QAbstractTableModel, QMimeData, QModelIndex,
                            QRect, Qt, Signal)
from PySide6.QtGui import QColor, QFontMetrics, QPainter
from PySide6.QtWidgets import (QAbstractItemView, QMenu, QStyle,
                               QStyledItemDelegate, QStyleOptionButton,
                               QTableView)

from core.viz.chapters import CHAPTER_COLORS
from core.viz.track_columns import (ENERGY_COLORS, EMOTION_COLORS,
                                    EMOTION_OPTIONS, GROOVE_COLORS,
                                    KEY_COLORS, LEVELS, READING_ORDER, reading)
from qt_app import theme

# Il ruolo con cui il delegate chiede il valore GREZZO — la lista di
# pastiglie — mentre DisplayRole resta il testo piano, per le colonne senza
# delegate e per chi copia.
PILLS_ROLE = Qt.ItemDataRole.UserRole + 1

_ROWS_MIME = "application/x-wavecut-rows"

# Il nome della colonna di spunta delle tabelle `checkable`.
CHECK_COLUMN = "✓"

# La colonnina del play, su OGNI tabella: è il gemello del ▶ che
# `play_table` mette nelle tabelle Streamlit. Un clic lì suona la riga e
# non tocca la selezione — sentire un brano non è sceglierlo.
PLAY_COLUMN = "▶"


def pill_color(column: str, value: str,
               genres: dict[str, str] | None = None) -> str | None:
    """Il colore della pastiglia per un valore, o None se resta neutra.

    Le scale sono quelle di `core.viz.track_columns`, agganciate al valore
    con la stessa regola con cui Streamlit le aggancia alla posizione
    nell'elenco delle opzioni: un valore fuori scala non è un errore, è una
    pastiglia che non si colora.
    """
    if column == "key":
        return KEY_COLORS.get(value)
    if column == "energy":
        try:
            step = int(value)
        except ValueError:
            return None
        return ENERGY_COLORS[step - 1] if 1 <= step <= LEVELS else None
    if column == "groove":
        try:
            step = round(float(value) * 100)
        except ValueError:
            return None
        return GROOVE_COLORS[step] if 0 <= step < len(GROOVE_COLORS) else None
    if column == "emotion":
        if value in EMOTION_OPTIONS:
            return EMOTION_COLORS[EMOTION_OPTIONS.index(value)]
        return None
    if column == "genres":
        return (genres or {}).get(value)
    if column == "chapter":
        return CHAPTER_COLORS.get(value)
    return None


def track_frame(rows: pd.DataFrame, common: dict[str, int]) -> pd.DataFrame:
    """Le righe scelte come le scrivono tutte le tabelle, più `_path`.

    La lettura è `core.viz.reading` — gli stessi campi con gli stessi nomi
    di ogni tabella Streamlit, perché il brano che si guarda qui è quello
    che un momento dopo sta di là.
    """
    listed = [reading(row, common) for _, row in rows.iterrows()]
    out = (pd.DataFrame(listed, columns=READING_ORDER) if listed
           else pd.DataFrame(columns=READING_ORDER))
    out["_path"] = list(rows["path"]) if len(rows) else []
    return out


class PandasModel(QAbstractTableModel):
    """Un DataFrame come modello: niente copie per riga, niente widget.

    `reorderable` accende il trascinamento delle righe: il drop riordina il
    frame e `order_changed` racconta il nuovo ordine dei `_path` — è il
    segnale su cui la playlist si aggiornerà in Fase 3.
    """

    order_changed = Signal(list)

    def __init__(self, frame: pd.DataFrame | None = None,
                 reorderable: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._frame = frame if frame is not None else pd.DataFrame()
        self._reorderable = reorderable

    # --- il frame ---
    @property
    def frame(self) -> pd.DataFrame:
        return self._frame

    def set_frame(self, frame: pd.DataFrame) -> None:
        self.beginResetModel()
        self._frame = frame
        self.endResetModel()

    def _shown(self) -> list:
        """Le colonne visibili: quelle col nome che non comincia per `_`."""
        return [c for c in self._frame.columns if not str(c).startswith("_")]

    def path_at(self, row: int) -> str | None:
        if "_path" in self._frame and 0 <= row < len(self._frame):
            return str(self._frame["_path"].iloc[row])
        return None

    # --- dimensioni e dati ---
    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._frame)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._shown())

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self._frame[self._shown()[index.column()]].iloc[index.row()]
        if role == PILLS_ROLE:
            return value
        if role == Qt.ItemDataRole.DisplayRole:
            if value is None or (not isinstance(value, list) and pd.isna(value)):
                return ""
            if isinstance(value, list):
                return "; ".join(str(v) for v in value)
            if isinstance(value, float) and value.is_integer():
                return str(int(value))       # i BPM senza il ".0" di pandas
            return str(value)
        return None

    def headerData(self, section: int, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._shown()[section])
        return str(section + 1)

    # --- ordinamento per colonna ---
    def sort(self, column: int, order=Qt.SortOrder.AscendingOrder) -> None:
        if not len(self._frame) or column >= len(self._shown()):
            return
        name = self._shown()[column]
        # La chiave e non la colonna: le celle a pastiglie portano LISTE, e
        # pandas non sa confrontare una lista con un'altra (né con un NaN).
        flat = self._frame[name].map(
            lambda v: "; ".join(str(x) for x in v) if isinstance(v, list)
            else v)
        numbers = pd.to_numeric(flat, errors="coerce")
        key = numbers if numbers.notna().any() else \
            flat.fillna("").astype(str).str.lower()
        self.beginResetModel()
        self._frame = self._frame.iloc[
            key.argsort(kind="stable")[::1 if
                                       order == Qt.SortOrder.AscendingOrder
                                       else -1]]
        self.endResetModel()

    # --- riordino per trascinamento ---
    def flags(self, index: QModelIndex):
        if not index.isValid():
            # Il posto FRA le righe: è lì che si lascia cadere.
            return (Qt.ItemFlag.ItemIsDropEnabled if self._reorderable
                    else Qt.ItemFlag.NoItemFlags)
        allowed = (Qt.ItemFlag.ItemIsEnabled
                   | Qt.ItemFlag.ItemIsSelectable)
        if self._reorderable:
            allowed |= Qt.ItemFlag.ItemIsDragEnabled
        return allowed

    def supportedDropActions(self):
        return Qt.DropAction.MoveAction

    def mimeTypes(self) -> list[str]:
        return [_ROWS_MIME]

    def mimeData(self, indexes) -> QMimeData:
        rows = sorted({i.row() for i in indexes if i.isValid()})
        data = QMimeData()
        data.setData(_ROWS_MIME, json.dumps(rows).encode())
        return data

    def dropMimeData(self, data, action, row, column, parent) -> bool:
        if not data.hasFormat(_ROWS_MIME) or not self._reorderable:
            return False
        moved = json.loads(bytes(data.data(_ROWS_MIME)).decode())
        target = (parent.row() if parent.isValid()
                  else row if row >= 0 else len(self._frame))
        staying = [i for i in range(len(self._frame)) if i not in moved]
        slot = target - sum(1 for i in moved if i < target)
        order = staying[:slot] + moved + staying[slot:]
        self.beginResetModel()
        self._frame = self._frame.iloc[order]
        self.endResetModel()
        if "_path" in self._frame:
            self.order_changed.emit([str(p) for p in self._frame["_path"]])
        # False a ragion veduta: con True la vista completerebbe il Move
        # RIMUOVENDO le righe di partenza, che qui sono già state spostate.
        return False


class PillDelegate(QStyledItemDelegate):
    """Le pastiglie in una cella: tinte piene, testo scuro, angoli tondi.

    `resolver` porta il colore per un valore (vedi `pill_color`); una cella
    può portarne più d'una — i generi — e chi non ha colore resta su una
    pastiglia neutra, come di là.
    """

    def __init__(self, resolver, parent=None) -> None:
        super().__init__(parent)
        self._resolver = resolver

    @staticmethod
    def _values(index: QModelIndex) -> list[str]:
        raw = index.data(PILLS_ROLE)
        if raw is None or (not isinstance(raw, list) and pd.isna(raw)):
            return []
        listed = raw if isinstance(raw, list) else [raw]
        return [str(v) for v in listed if str(v)]

    def paint(self, painter: QPainter, option, index: QModelIndex) -> None:
        if option.state & QStyle.StateFlag.State_Selected:
            # Il colore di modulo, non option.palette.highlight(): il
            # pennello del temporaneo di shiboken ha fatto crollare la
            # suite (segfault nel paint) appena i test hanno cominciato
            # a selezionare righe.
            painter.fillRect(option.rect, theme.SELECTED_ROW)
        values = self._values(index)
        if not values:
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = painter.font()
        font.setPointSizeF(max(font.pointSizeF() - 1.5, 8.0))
        painter.setFont(font)
        metrics = QFontMetrics(font)
        tall = 18
        x = option.rect.left() + 4
        y = option.rect.top() + (option.rect.height() - tall) // 2
        for value in values:
            wide = metrics.horizontalAdvance(value) + 12
            if x + wide > option.rect.right() and x > option.rect.left() + 4:
                break                    # meglio una pastiglia in meno che a metà
            color = self._resolver(value)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(color or theme.RAISED))
            painter.drawRoundedRect(x, y, wide, tall, tall / 2, tall / 2)
            painter.setPen(QColor(theme.PILL_INK if color else theme.INK))
            painter.drawText(x, y, wide, tall, Qt.AlignmentFlag.AlignCenter,
                             value)
            x += wide + 4
        painter.restore()

    def sizeHint(self, option, index: QModelIndex):
        size = super().sizeHint(option, index)
        metrics = QFontMetrics(option.font)
        wide = 8 + sum(metrics.horizontalAdvance(v) + 16
                       for v in self._values(index))
        size.setWidth(max(wide, 40))
        size.setHeight(max(size.height(), 24))
        return size


class CheckDelegate(QStyledItemDelegate):
    """La casella dei brani PRESI, staccata dall'evidenziazione del clic.

    All'inizio la spunta rifletteva la selezione della vista, e non
    funzionava: cliccare una riga per guardarla svuotava le spunte messe
    prima — un gesto di lettura che disfa una scelta. Adesso i presi sono
    un insieme della tabella (per percorso), si toccano SOLO dalla casella,
    e il clic normale resta quello che è: evidenzia la riga su cui si sta,
    senza toccare quello che si è già preso.
    """

    def paint(self, painter: QPainter, option, index: QModelIndex) -> None:
        if option.state & QStyle.StateFlag.State_Selected:
            # Il colore di modulo, non option.palette.highlight(): il
            # pennello del temporaneo di shiboken ha fatto crollare la
            # suite (segfault nel paint) appena i test hanno cominciato
            # a selezionare righe.
            painter.fillRect(option.rect, theme.SELECTED_ROW)
        box = QStyleOptionButton()
        box.palette = option.palette
        side = 16
        box.rect = QRect(option.rect.center().x() - side // 2,
                         option.rect.center().y() - side // 2, side, side)
        table = self.parent()
        box.state = QStyle.StateFlag.State_Enabled | (
            QStyle.StateFlag.State_On
            if table.is_row_picked(index.row())
            else QStyle.StateFlag.State_Off)
        option.widget.style().drawControl(
            QStyle.ControlElement.CE_CheckBox, box, painter, option.widget)


class PlayDelegate(QStyledItemDelegate):
    """Il ▶ di riga: si disegna e basta, il gesto lo raccoglie la vista.

    Un glifo spento (FADED) perché è un comando presente su ogni riga: in
    inchiostro pieno sarebbe una colonna che urla. Il clic arriva da
    `TrackTable.mousePressEvent`, che suona la riga senza selezionarla.
    """

    def paint(self, painter: QPainter, option, index: QModelIndex) -> None:
        if option.state & QStyle.StateFlag.State_Selected:
            # Il colore di modulo, non option.palette.highlight(): il
            # pennello del temporaneo di shiboken ha fatto crollare la
            # suite (segfault nel paint) appena i test hanno cominciato
            # a selezionare righe.
            painter.fillRect(option.rect, theme.SELECTED_ROW)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = painter.font()
        font.setPointSizeF(max(font.pointSizeF() - 2.0, 8.0))
        painter.setFont(font)
        painter.setPen(QColor(theme.FADED))
        painter.drawText(option.rect, Qt.AlignmentFlag.AlignCenter, "▶")
        painter.restore()


class TrackTable(QTableView):
    """La tabella dei brani, già vestita: pastiglie, sort, trascinamento.

    `row_activated` porta il `_path` della riga doppio-cliccata: è il
    segnale a cui il lettore si aggancia. Gli altri segnali escono dal menu
    contestuale — le stesse quattro voci su ogni tabella della pagina — e
    dalla selezione delle righe: chi li ascolta decide cosa significano (per
    la playlist, la selezione È il canale che cerchia i brani sulla mappa).
    """

    row_activated = Signal(str)
    play_requested = Signal(str)
    seed_requested = Signal(str)
    add_requested = Signal(list)            # i _path delle righe scelte
    reveal_requested = Signal(str)
    selection_paths_changed = Signal(list)  # i _path delle righe selezionate

    # Larghezze di partenza per le colonne che si conoscono: misurarle sui
    # dati (resizeColumnsToContents) visita OGNI riga, e una tabella da
    # novantamila righe si pianterebbe proprio nel gesto che Qt deve rendere
    # gratis.
    _WIDTHS = {CHECK_COLUMN: 30, PLAY_COLUMN: 30, "#": 40, "file": 320,
               "BPM": 52, "key": 52,
               "energy": 60, "groove": 64, "emotion": 64, "mood": 120,
               "genres": 240, "cost": 56, "sound": 56, "bpm cost": 66,
               "key cost": 62, "similarity": 72, "copies": 56, "chapter": 84,
               "from previous": 94, "Δbpm": 52, "Δkey": 48, "Δenergy": 62,
               "Δgroove": 62}

    def __init__(self, reorderable: bool = False, checkable: bool = False,
                 parent=None) -> None:
        super().__init__(parent)
        self._model = PandasModel(reorderable=reorderable, parent=self)
        self.setModel(self._model)
        self._checkable = checkable
        self._check_delegate = CheckDelegate(self) if checkable else None
        self._play_delegate = PlayDelegate(self)
        self._delegates: dict[str, PillDelegate] = {}
        self._genre_colors: dict[str, str] = {}
        # I brani PRESI (colonna ✓), per percorso: sopravvivono ai ridisegni
        # della tabella — quello che sparisce dal frame cade da solo.
        self._picked: set[str] = set()

        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(26)
        self.setShowGrid(False)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSortingEnabled(True)
        self.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        self.horizontalHeader().setStretchLastSection(True)

        if reorderable:
            self.setDragEnabled(True)
            self.setAcceptDrops(True)
            self.setDropIndicatorShown(True)
            self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
            self.setDefaultDropAction(Qt.DropAction.MoveAction)
            self.setDragDropOverwriteMode(False)

        self.doubleClicked.connect(self._on_double_click)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_menu)
        if not checkable:
            # Dove non ci sono caselle la scelta È la selezione; dove ci
            # sono, la racconta `toggle_pick` e il clic evidenzia soltanto.
            self.selectionModel().selectionChanged.connect(
                lambda *_: self.selection_paths_changed.emit(
                    self.selected_paths()))

    @property
    def model_(self) -> PandasModel:
        """Il modello col suo tipo vero, senza il cast di `model()`."""
        return self._model

    def set_tracks(self, frame: pd.DataFrame,
                   genre_colors: dict[str, str] | None = None) -> None:
        """Mostra un frame di `track_frame`, e veste le colonne che conosce."""
        if PLAY_COLUMN not in frame.columns:
            # Su una COPIA: il frame è di chi chiama, e una colonna comparsa
            # di soprassalto in casa sua sarebbe una sorpresa cattiva.
            frame = frame.copy()
            frame.insert(0, PLAY_COLUMN, "")
            if self._checkable:
                frame.insert(0, CHECK_COLUMN, "")   # ✓ prima, ▶ subito dopo
        self._model.set_frame(frame)
        if self._checkable:
            # I presi che il frame nuovo non porta più non sono più presi.
            still = self._picked & {str(p) for p in frame.get("_path", [])}
            if still != self._picked:
                self._picked = still
                self.selection_paths_changed.emit(self.selected_paths())
        # La mappa dei generi si legge DAL VIVO nel resolver: cambia a ogni
        # selezione, e un delegate che la catturasse alla prima chiamata
        # colorerebbe per sempre coi generi di allora.
        self._genre_colors = dict(genre_colors or {})
        shown = [c for c in frame.columns if not str(c).startswith("_")]
        for name in ("key", "energy", "groove", "emotion", "genres", "chapter"):
            if name not in shown:
                continue
            if name not in self._delegates:
                self._delegates[name] = PillDelegate(
                    (lambda v, n=name:
                     pill_color(n, v, self._genre_colors)), self)
            self.setItemDelegateForColumn(
                shown.index(name), self._delegates[name])
        if self._check_delegate is not None and CHECK_COLUMN in shown:
            self.setItemDelegateForColumn(
                shown.index(CHECK_COLUMN), self._check_delegate)
        if PLAY_COLUMN in shown:
            self.setItemDelegateForColumn(
                shown.index(PLAY_COLUMN), self._play_delegate)
        for name, width in self._WIDTHS.items():
            if name in shown:
                self.setColumnWidth(shown.index(name), width)

    def selected_paths(self) -> list[str]:
        """I `_path` delle righe scelte, dall'alto in basso.

        Su una tabella `checkable` sono le righe SPUNTATE — la scelta vive
        nelle caselle, e il clic che evidenzia non la tocca. Sulle altre
        resta la selezione della vista (una riga, ⌘/ctrl per estendere).
        """
        if self._checkable:
            return [p for p in (self._model.path_at(r)
                                for r in range(self._model.rowCount()))
                    if p in self._picked]
        rows = sorted({i.row() for i in self.selectionModel().selectedRows()})
        return [p for p in (self._model.path_at(r) for r in rows) if p]

    def is_row_picked(self, row: int) -> bool:
        return self._model.path_at(row) in self._picked

    def toggle_pick(self, row: int) -> None:
        """Commuta la spunta di una riga — il solo gesto che tocca i presi."""
        path = self._model.path_at(row)
        if path is None:
            return
        self._picked.symmetric_difference_update({path})
        self.viewport().update()
        self.selection_paths_changed.emit(self.selected_paths())

    def clear_picks(self) -> None:
        """Via spunte ed evidenziazione, in silenzio: serve a chi le sta
        rimpiazzando con un gesto più recente, non a segnalare un gesto."""
        self._picked.clear()
        self.blockSignals(True)
        self.clearSelection()
        self.blockSignals(False)
        self.viewport().update()

    def mousePressEvent(self, event) -> None:
        """I due campi-comando davanti alla riga, prima del clic normale.

        Il ▶ suona la riga e NON la seleziona — sentire un brano non è
        sceglierlo. La casella COMMUTA la spunta della sola riga, e il clic
        altrove resta il solito: evidenzia la riga su cui si sta, e le
        spunte non le tocca.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            index = self.indexAt(event.position().toPoint())
            name = (self._model.headerData(index.column(),
                                           Qt.Orientation.Horizontal)
                    if index.isValid() else None)
            if name == PLAY_COLUMN:
                path = self._model.path_at(index.row())
                if path:
                    self.play_requested.emit(path)
                return
            if self._checkable and name == CHECK_COLUMN:
                self.toggle_pick(index.row())
                return
        super().mousePressEvent(event)

    def _on_double_click(self, index: QModelIndex) -> None:
        path = self._model.path_at(index.row())
        if path:
            self.row_activated.emit(path)

    def _on_menu(self, at) -> None:
        index = self.indexAt(at)
        path = self._model.path_at(index.row()) if index.isValid() else None
        if path is None:
            return
        # "Add" prende le righe selezionate se la riga cliccata è fra loro,
        # altrimenti la sola riga cliccata: è la regola dei menu contestuali
        # ovunque — il tasto destro fuori dalla selezione parla di quella riga.
        picked = self.selected_paths()
        added = picked if path in picked else [path]
        menu = QMenu(self)
        menu.addAction("▶ Play",
                       lambda: self.play_requested.emit(path))
        menu.addAction("◎ Use as seed",
                       lambda: self.seed_requested.emit(path))
        menu.addAction(f"➕ Add to playlist ({len(added)})",
                       lambda: self.add_requested.emit(list(added)))
        menu.addSeparator()
        menu.addAction("📂 Show in file manager",
                       lambda: self.reveal_requested.emit(path))
        menu.exec(self.viewport().mapToGlobal(at))
