"""Il pannello dei filtri della pagina Map: i widget attorno alla regola.

La regola — quali brani passano — sta in `core.viz.filters.filter_tracks`,
uguale per le due app. Qui ci sono la ruota Camelot (il frontend riusato),
le due liste spuntabili di generi e mood, gli intervalli di BPM e groove, e
un segnale solo: `changed`, quando la domanda "quali brani sto guardando"
cambia. I filtri restringono TUTTO quello che la pagina propone — i punti,
le proposte, la rosa — che è il motivo per cui il pannello è uno.
"""

from __future__ import annotations

import pandas as pd

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QDoubleSpinBox, QGridLayout, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QPushButton, QVBoxLayout, QWidget)

from core.viz.filters import filter_tracks, span
from qt_app.widgets.wheel_view import WheelView


class CheckList(QWidget):
    """Una lista spuntabile con la ricerca sopra: il multiselect di Qt.

    Le voci arrivano ordinate per frequenza, come nei menu Streamlit: quello
    che si filtra più spesso sta in cima. La ricerca nasconde e basta — le
    spunte restano dove sono, anche fuori vista.
    """

    changed = Signal()

    def __init__(self, label: str, parent=None) -> None:
        super().__init__(parent)
        self._search = QLineEdit()
        self._search.setPlaceholderText(f"filter {label}…")
        self._search.setClearButtonEnabled(True)
        self._list = QListWidget()
        self._list.setUniformItemSizes(True)

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)
        box.addWidget(QLabel(label))
        box.addWidget(self._search)
        box.addWidget(self._list, stretch=1)

        self._search.textChanged.connect(self._on_search)
        self._list.itemChanged.connect(lambda _: self.changed.emit())

    def set_options(self, names: list[str]) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        for name in names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self._list.addItem(item)
        self._list.blockSignals(False)

    def checked(self) -> list[str]:
        return [self._list.item(i).text() for i in range(self._list.count())
                if self._list.item(i).checkState() == Qt.CheckState.Checked]

    def clear_checks(self) -> None:
        self._list.blockSignals(True)
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(Qt.CheckState.Unchecked)
        self._list.blockSignals(False)

    def _on_search(self, text: str) -> None:
        wanted = text.casefold()
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setHidden(bool(wanted) and wanted not in item.text().casefold())


def _range_row(decimals: int, step: float) -> tuple[QDoubleSpinBox, QDoubleSpinBox, QWidget]:
    """Due caselle per un intervallo: da–a. Qt non ha uno slider a due
    maniglie, e due numeri scritti si leggono meglio di due maniglie."""
    low, high = QDoubleSpinBox(), QDoubleSpinBox()
    for spin in (low, high):
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
    row = QWidget()
    box = QHBoxLayout(row)
    box.setContentsMargins(0, 0, 0, 0)
    box.addWidget(low)
    box.addWidget(QLabel("–"))
    box.addWidget(high)
    return low, high, row


class FiltersPanel(QWidget):
    """La ruota, le liste e gli intervalli; `kept(frame)` applica la regola."""

    changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._keys: list[str] = []

        # Un gesto sui filtri ridisegna la nuvola intera: mezzo secondo che
        # non va pagato a ogni lettera scritta o casella spuntata di fila.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(350)
        self._debounce.timeout.connect(self.changed.emit)

        told = QLabel("Filters narrow the map, the suggestions and the "
                      "roster. Nothing picked means everything passes.")
        told.setObjectName("dim")
        told.setWordWrap(True)

        self._wheel = WheelView()
        self._wheel.setMinimumHeight(240)
        self._wheel.height_suggested.connect(
            lambda h: self._wheel.setFixedHeight(max(220, min(340, h))))
        self._wheel.key_toggled.connect(self._on_key)
        wheel_told = QLabel("Pick the keys you want to land on.")
        wheel_told.setObjectName("dim")

        self._genres = CheckList("Genres")
        self._moods = CheckList("Moods")
        for picker in (self._genres, self._moods):
            picker.changed.connect(self._debounce.start)

        # I decimali coprono la precisione con cui lo store scrive i numeri
        # (BPM a un decimale, danceability a tre): una casella che
        # arrotondasse taglierebbe fuori i brani sul bordo — a corsa tutta
        # aperta ne sparivano due su 87mila, che è il modo subdolo di
        # sbagliare.
        self._bpm_low, self._bpm_high, bpm_row = _range_row(1, 1.0)
        self._gr_low, self._gr_high, gr_row = _range_row(3, 0.01)
        for spin in (self._bpm_low, self._bpm_high, self._gr_low, self._gr_high):
            spin.valueChanged.connect(lambda _: self._debounce.start())

        reset = QPushButton("↺ Reset the filters")
        reset.clicked.connect(self._on_reset)

        self._count = QLabel("")
        self._count.setObjectName("dim")
        self._count.setWordWrap(True)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(QLabel("BPM"), 0, 0)
        grid.addWidget(bpm_row, 0, 1)
        grid.addWidget(QLabel("Groove"), 1, 0)
        grid.addWidget(gr_row, 1, 1)

        box = QVBoxLayout(self)
        box.addWidget(told)
        box.addWidget(self._wheel)
        box.addWidget(wheel_told)
        box.addWidget(self._genres, stretch=3)
        box.addWidget(self._moods, stretch=2)
        box.addLayout(grid)
        box.addWidget(reset)
        box.addWidget(self._count)

    # --- la libreria detta le opzioni e le corse ---
    def set_frame(self, frame: pd.DataFrame) -> None:
        genre_counts = pd.Series(
            [g for tags in frame["genre_list"] for g in tags if g]
        ).value_counts()
        mood_counts = pd.Series(
            [m for tags in frame["mood_list"] for m in tags if m]
        ).value_counts()
        self._genres.set_options(list(genre_counts.index))
        self._moods.set_options(list(mood_counts.index))

        tempo = span(frame, "bpm", 60.0, 200.0)
        swing = span(frame, "danceability", 0.0, 1.0)
        for spin, (low, high), at in (
                (self._bpm_low, tempo, tempo[0]),
                (self._bpm_high, tempo, tempo[1]),
                (self._gr_low, swing, swing[0]),
                (self._gr_high, swing, swing[1])):
            spin.blockSignals(True)
            spin.setRange(low, high)
            spin.setValue(at)
            spin.blockSignals(False)

        self._keys = []
        self._wheel.set_keys(self._keys)

    # --- la regola, applicata ---
    def kept(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = filter_tracks(
            frame, self._genres.checked(), self._moods.checked(), self._keys,
            (self._bpm_low.value(), self._bpm_high.value()),
            (self._gr_low.value(), self._gr_high.value()))
        self._count.setText(
            f"{len(out):,} of {len(frame):,} tracks pass — the map, the "
            "suggestions and the roster all come from these.")
        return out

    # --- i gesti ---
    def _on_key(self, code: str) -> None:
        self._keys = ([k for k in self._keys if k != code]
                      if code in self._keys else self._keys + [code])
        self._wheel.set_keys(self._keys)
        self._debounce.start()

    def _on_reset(self) -> None:
        self._keys = []
        self._wheel.set_keys(self._keys)
        self._genres.clear_checks()
        self._moods.clear_checks()
        for spin, edge in ((self._bpm_low, self._bpm_low.minimum()),
                           (self._bpm_high, self._bpm_high.maximum()),
                           (self._gr_low, self._gr_low.minimum()),
                           (self._gr_high, self._gr_high.maximum())):
            spin.blockSignals(True)
            spin.setValue(edge)
            spin.blockSignals(False)
        self._debounce.start()
