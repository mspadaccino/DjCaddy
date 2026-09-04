"""I pesi del costo di transizione: la riga sopra le schede di destra.

I tre pesi definiscono `D` — quanto contano suono, tempo e chiave quando si
dice che due brani mixano — e `D` lo leggono sei cose in due schede
diverse: Quick List, Chain Maker, Journey e il magic sort di Radio Mix in
Build a set; il magic sort e la colonna «from previous» nella Playlist.
Finché gli slider stavano dentro Build a set, chi premeva Magic sort sulla
Playlist veniva ordinato con pesi che non vedeva. Stanno qui, fuori da
ogni scheda, perché la posizione di un comando deve dire dove arriva.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QWidget

from qt_app import theme


class WeightSlider(QWidget):
    """Un peso 0..2 a passi di un decimo: lo slider, col nome davanti e il
    numero dietro. Uno slider e non una casella perché il peso si ASSAGGIA
    — si trascina guardando la lista cambiare — e la casella chiedeva un
    clic per ogni decimo."""

    valueChanged = Signal(float)

    def __init__(self, name: str, why: str, parent=None) -> None:
        super().__init__(parent)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 20)
        self._slider.setValue(10)
        self._slider.setMinimumWidth(90)
        self._told = QLabel("1.0")
        self._told.setFixedWidth(24)
        self.setToolTip(why)
        self._slider.valueChanged.connect(self._on_moved)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(QLabel(f"w·{name}"))
        row.addWidget(self._slider)
        row.addWidget(self._told)

    def value(self) -> float:
        return self._slider.value() / 10

    def setValue(self, value: float) -> None:
        self._slider.setValue(round(value * 10))

    def _on_moved(self, raw: int) -> None:
        self._told.setText(f"{raw / 10:.1f}")
        self.valueChanged.emit(raw / 10)


class WeightsBar(QWidget):
    """La riga dei tre pesi. `changed` dice che uno si è mosso; `weights()`
    li dà nell'ordine del costo: suono, tempo, chiave."""

    changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        title = QLabel("Transition cost")
        title.setToolTip(theme.hint(
            "How two tracks are judged to mix, everywhere on this page: "
            "Quick List, the chain, the Journey, Radio Mix's order, and "
            "the playlist's Magic sort and «from previous» column all use "
            "these three weights. Divided by their sum, so they are "
            "proportions: 1,1,1 means the same as 2,2,2."))
        self._sound = WeightSlider(
            "sound", theme.hint(
                "How much the acoustic distance counts — cosine in the 1280 "
                "dimensions of the embedding, not on the flattened map. "
                "Alone, with BPM and key at 0, a list is «what sounds like "
                "it»."))
        self._bpm = WeightSlider(
            "BPM", theme.hint("How much the tempo gap counts. Beyond ±6% "
                              "the cost climbs fast."))
        self._key = WeightSlider(
            "key", theme.hint("How much harmonic distance counts. Adjacent "
                              "or relative keys cost nothing."))
        for slider in (self._sound, self._bpm, self._key):
            slider.valueChanged.connect(lambda _: self.changed.emit())
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        row.addWidget(title)
        row.addWidget(self._sound)
        row.addWidget(self._bpm)
        row.addWidget(self._key)
        row.addStretch(1)

    def weights(self) -> tuple[float, float, float]:
        return (self._sound.value(), self._bpm.value(), self._key.value())

    def set_weights(self, sound: float, bpm: float, key: float) -> None:
        for slider, value in ((self._sound, sound), (self._bpm, bpm),
                              (self._key, key)):
            slider.setValue(value)
