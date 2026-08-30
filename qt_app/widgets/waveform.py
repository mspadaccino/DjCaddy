"""L'onda del lettore: le stesse barre del canvas Streamlit, in QPainter.

L'onda È la barra di avanzamento: la parte già suonata si colora, e un clic
sposta la riproduzione in quel punto. I numeri del disegno — 800 colonne,
barra alta `peak * (H-4)` con un minimo di un pixel, mezzo pixel d'aria fra
una barra e l'altra, il resto al 35% d'opacità — sono copiati uno per uno
dal canvas di `streamlit_app/views/components.py`, perché il criterio di
accettazione è il confronto fianco a fianco: stesso brano, stessa forma
d'onda, stessa risposta al clic.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

from qt_app import theme

HEIGHT = 56


class WaveformBar(QWidget):
    """Le barre di un brano, con la porzione suonata colorata.

    `set_position` ridisegna solo quando cambia la BARRA raggiunta, non a
    ogni millisecondo: il player manda la posizione dieci volte al secondo,
    e nove volte su dieci l'onda sarebbe identica.
    """

    seek_requested = Signal(float)      # secondi

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._peaks: list[float] = []
        self._duration = 0.0
        self._position = 0.0
        self._done = QColor(theme.PRIMARY)
        self._rest = QColor(theme.FADED)
        self._rest.setAlphaF(0.35)
        self.setFixedHeight(HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_wave(self, peaks: list[float], duration: float) -> None:
        self._peaks = list(peaks)
        self._duration = float(duration)
        self._position = 0.0
        self.update()

    def clear(self) -> None:
        self.set_wave([], 0.0)

    def set_position(self, seconds: float) -> None:
        before = self._played_bars()
        self._position = seconds
        if self._played_bars() != before:
            self.update()

    def _played_bars(self) -> int:
        if not self._duration or not self._peaks:
            return 0
        return int(self._position / self._duration * len(self._peaks))

    def paintEvent(self, event) -> None:
        if not self._peaks:
            return
        painter = QPainter(self)
        width, height = self.width(), self.height()
        step = width / len(self._peaks)
        bar = max(step - 0.5, 0.5)
        middle = height / 2
        played = self._played_bars()
        for i, peak in enumerate(self._peaks):
            tall = max(1.0, peak * (height - 4))
            painter.fillRect(
                int(i * step), int(middle - tall / 2), max(int(bar), 1),
                max(int(tall), 1),
                self._done if i <= played else self._rest)
        painter.end()

    def mousePressEvent(self, event) -> None:
        if self._duration and self.width():
            where = event.position().x() / self.width()
            self.seek_requested.emit(
                min(self._duration, max(0.0, where * self._duration)))
