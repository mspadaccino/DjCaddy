"""Uno slider a due maniglie: un intervallo da–a che si trascina.

Qt non ne ha uno, e i filtri numerici della mappa erano due caselle da
scrivere. Un intervallo però si ASSAGGIA — si stringe guardando il
contatore dei brani scendere — e una casella chiede un clic per passo. Le
maniglie stanno sulla barra, i due numeri accanto: «118.0 – 124.0» a
colpo d'occhio si legge meglio della posizione di due pallini.

I valori sono float e si arrotondano ai decimali chiesti quando li muove
il mouse; quelli messi da fuori (`set_span`, `set_values`) restano esatti.
A corsa tutta aperta i due estremi sono i minimi e massimi veri della
colonna, così nessun brano sul bordo cade fuori per un arrotondamento.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from qt_app import theme

_RADIUS = 7          # raggio della maniglia, in pixel
_TRACK = 4           # spessore della barra


class _Bar(QWidget):
    """La barra con le due maniglie. `moved` porta i due valori."""

    moved = Signal(float, float)

    def __init__(self, decimals: int, parent=None) -> None:
        super().__init__(parent)
        self._decimals = decimals
        self._min, self._max = 0.0, 1.0
        self._low, self._high = 0.0, 1.0
        self._grabbed: str | None = None
        self.setFixedHeight(2 * _RADIUS + 6)
        self.setMinimumWidth(80)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        theme.bus().changed.connect(self.update)

    # --- i numeri ---
    def set_span(self, low: float, high: float) -> None:
        """La corsa, e le maniglie ai suoi estremi."""
        self._min, self._max = low, max(high, low + 1e-9)
        self._low, self._high = low, high
        self.update()

    def set_values(self, low: float, high: float) -> None:
        low = min(max(low, self._min), self._max)
        high = min(max(high, self._min), self._max)
        self._low, self._high = min(low, high), max(low, high)
        self.update()

    def values(self) -> tuple[float, float]:
        return (self._low, self._high)

    def span(self) -> tuple[float, float]:
        return (self._min, self._max)

    # --- pixel e valori ---
    def _x_of(self, value: float) -> float:
        usable = self.width() - 2 * _RADIUS
        return _RADIUS + usable * (value - self._min) / (self._max - self._min)

    def _value_at(self, x: float) -> float:
        usable = max(self.width() - 2 * _RADIUS, 1)
        frac = min(max((x - _RADIUS) / usable, 0.0), 1.0)
        return round(self._min + frac * (self._max - self._min),
                     self._decimals)

    # --- il mouse ---
    def mousePressEvent(self, event) -> None:  # noqa: N802 (nome Qt)
        x = event.position().x()
        near_low = abs(x - self._x_of(self._low))
        near_high = abs(x - self._x_of(self._high))
        # A maniglie sovrapposte vince quella verso cui si va.
        if near_low == near_high:
            self._grabbed = "low" if x < self._x_of(self._low) else "high"
        else:
            self._grabbed = "low" if near_low < near_high else "high"
        self._drag(x)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._grabbed:
            self._drag(event.position().x())

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._grabbed = None

    def _drag(self, x: float) -> None:
        value = self._value_at(x)
        if self._grabbed == "low":
            low, high = min(value, self._high), self._high
        else:
            low, high = self._low, max(value, self._low)
        if (low, high) != (self._low, self._high):
            self._low, self._high = low, high
            self.update()
            self.moved.emit(low, high)

    # --- il disegno ---
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        y = self.height() / 2
        x0, x1 = self._x_of(self._low), self._x_of(self._high)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.RAISED))
        painter.drawRoundedRect(
            QRectF(_RADIUS, y - _TRACK / 2, self.width() - 2 * _RADIUS,
                   _TRACK), _TRACK / 2, _TRACK / 2)
        painter.setBrush(QColor(theme.PRIMARY))
        painter.drawRoundedRect(QRectF(x0, y - _TRACK / 2, x1 - x0, _TRACK),
                                _TRACK / 2, _TRACK / 2)
        painter.setPen(QPen(QColor(theme.PRIMARY), 1.5))
        painter.setBrush(QColor(theme.INK))
        for x in (x0, x1):
            painter.drawEllipse(QRectF(x - _RADIUS, y - _RADIUS,
                                       2 * _RADIUS, 2 * _RADIUS))


class RangeSlider(QWidget):
    """La barra coi due numeri accanto. `valuesChanged` porta l'intervallo
    a ogni movimento di una maniglia — non per i valori messi da fuori,
    come uno spinbox con i segnali bloccati."""

    valuesChanged = Signal(float, float)

    def __init__(self, decimals: int, parent=None) -> None:
        super().__init__(parent)
        self._decimals = decimals
        self._bar = _Bar(decimals)
        self._low = QLabel()
        self._high = QLabel()
        self._low.setAlignment(Qt.AlignmentFlag.AlignRight
                               | Qt.AlignmentFlag.AlignVCenter)
        width = 30 + 8 * decimals
        for label in (self._low, self._high):
            label.setFixedWidth(width)
        self._bar.moved.connect(self._on_moved)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(self._low)
        row.addWidget(self._bar, stretch=1)
        row.addWidget(self._high)
        self._tell()

    def set_span(self, low: float, high: float) -> None:
        """La corsa nuova, tutta aperta: gli estremi veri della colonna."""
        self._bar.set_span(low, high)
        self._tell()

    def set_values(self, low: float, high: float) -> None:
        self._bar.set_values(low, high)
        self._tell()

    def reset(self) -> None:
        self.set_values(*self._bar.span())

    def values(self) -> tuple[float, float]:
        return self._bar.values()

    def span(self) -> tuple[float, float]:
        return self._bar.span()

    def _tell(self) -> None:
        low, high = self._bar.values()
        self._low.setText(f"{low:.{self._decimals}f}")
        self._high.setText(f"{high:.{self._decimals}f}")

    def _on_moved(self, low: float, high: float) -> None:
        self._tell()
        self.valuesChanged.emit(low, high)
