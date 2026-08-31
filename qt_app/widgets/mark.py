"""Il marchio 30×30 dell'header: statico, o che pulsa a tempo di un job.

Un `QTimer` disegna il ciclo di pulsazione (0.97s, ≈124 bpm) solo mentre
serve — acceso e spento da `set_pulsing`, non da un'animazione sempre viva
che consumerebbe cicli per un marchio fermo la maggior parte del tempo.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QElapsedTimer, QTimer
from PySide6.QtGui import QPainter
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QWidget

from .mark_svg import build_mark_svg

_PULSE_MS = 970  # 0.97s per ciclo, ~124 bpm
_PULSE_AMPLITUDE = 0.035


class PulsingMark(QWidget):
    def __init__(self, size: int = 30, parent=None) -> None:
        super().__init__(parent)
        self._size = size
        self.setFixedSize(size, size)
        self._pulsing = False
        self._clock = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self.update)

    def set_pulsing(self, pulsing: bool) -> None:
        if pulsing == self._pulsing:
            return
        self._pulsing = pulsing
        if pulsing:
            self._clock.start()
            self._timer.start()
        else:
            self._timer.stop()
            self.update()

    def _pulse_scale(self) -> float:
        if not self._pulsing:
            return 1.0
        phase = (self._clock.elapsed() % _PULSE_MS) / _PULSE_MS
        return 1.0 + _PULSE_AMPLITUDE * 0.5 * (1 - math.cos(2 * math.pi * phase))

    def paintEvent(self, event) -> None:  # noqa: N802 (nome Qt)
        svg = build_mark_svg(self._size, pulse=self._pulse_scale())
        renderer = QSvgRenderer(svg)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(painter)
