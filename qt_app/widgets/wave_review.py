"""L'onda grande della revisione: bande di frequenza, marker, regioni.

Il disegno replica numero per numero il canvas del player CCv2 di
`streamlit_app/views/wave_analysis.py`, perché il criterio di accettazione
della Fase 4 è il confronto fianco a fianco: fondo #0f0f12 ad angoli tondi,
bande vocali rosa al 32%, barre simmetriche attorno al centro (altezza
`amp * metà`, larghezza `larghezza/n`), marker di sezione con la linea a
mezza opacità, il triangolino alla base e l'etichetta da 10px, playhead
giallo largo 2, tooltip in alto col tempo trascorso e quello che manca.

La parte statica (fondo, regioni, barre, marker) vive in un QPixmap e si
ricostruisce solo quando cambiano i dati o la misura: il playhead, che si
muove dieci volte al secondo, costa un blit e una riga.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from core.analysis.models import format_elapsed

HEIGHT = 200

_BACKGROUND = QColor("#0f0f12")
_REGION = QColor(255, 93, 177, 82)          # rgba(255,93,177,0.32)
_PLAYHEAD = QColor("#ffe14d")
_TOOLTIP_BACK = QColor(15, 15, 18, 235)     # rgba(15,15,18,0.92)
_TOOLTIP_EDGE = QColor("#333333")
_TOOLTIP_INK = QColor("#eeeeee")


class WaveReview(QWidget):
    """La waveform della pagina Wave: si clicca per saltare, come di là."""

    seek_requested = Signal(float)      # secondi

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._amp: list[float] = []
        self._colors: list[str] = []
        self._duration = 0.0
        self._markers: list[dict] = []              # {"t","label","color"}
        self._regions: list[tuple[float, float]] = []
        self._position = 0.0
        self._hover: float | None = None            # x del mouse, in punti
        self._wave: QPixmap | None = None
        self.setFixedHeight(HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)

    # ------------------------------------------------------------------
    def set_wave(self, amp: list[float], colors: list[str],
                 duration: float) -> None:
        self._amp = list(amp)
        self._colors = list(colors)
        self._duration = float(duration)
        self._position = 0.0
        self._invalidate()

    def clear(self) -> None:
        self.set_wave([], [], 0.0)
        self._markers, self._regions = [], []

    def set_markers(self, markers: list[dict]) -> None:
        self._markers = list(markers)
        self._invalidate()

    def set_regions(self, regions) -> None:
        self._regions = [(float(a), float(b)) for a, b in regions]
        self._invalidate()

    def set_position(self, seconds: float) -> None:
        self._position = seconds
        self.update()

    def _invalidate(self) -> None:
        self._wave = None
        self.update()

    # ------------------------------------------------------------------
    def _rebuild(self) -> None:
        """La parte che non si muove, sul pixmap: l'offscreen del canvas."""
        ratio = self.devicePixelRatioF()
        wide, tall = self.width(), self.height()
        pixmap = QPixmap(int(wide * ratio), int(tall * ratio))
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_BACKGROUND)
        painter.drawRoundedRect(0, 0, wide, tall, 6, 6)

        duration = self._duration or 1.0
        middle = tall / 2

        painter.setBrush(_REGION)
        for begin, end in self._regions:
            x0 = begin / duration * wide
            x1 = end / duration * wide
            painter.drawRect(int(x0), 0, max(1, int(x1 - x0)), tall)

        bars = len(self._amp)
        if bars:
            pen = QPen()
            pen.setWidthF(max(1.0, wide / bars))
            for i, amp in enumerate(self._amp):
                x = i / bars * wide
                high = amp * middle
                pen.setColor(QColor(self._colors[i] if i < len(self._colors)
                                    else "#888888"))
                painter.setPen(pen)
                painter.drawLine(QPointF(x, middle - high),
                                 QPointF(x, middle + high))

        font = painter.font()
        font.setPointSizeF(10.0)
        painter.setFont(font)
        for mark in self._markers:
            x = float(mark["t"]) / duration * wide
            color = QColor(mark.get("color") or "#ffffff")
            faded = QColor(color)
            faded.setAlphaF(0.5)
            painter.setPen(QPen(faded, 1))
            painter.drawLine(QPointF(x, 0), QPointF(x, tall))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawPolygon(QPolygonF([QPointF(x - 4, tall - 1),
                                           QPointF(x + 4, tall - 1),
                                           QPointF(x, tall - 9)]))
            painter.setPen(color)
            painter.drawText(QPointF(x + 5, 12), str(mark.get("label", "")))
        painter.end()
        self._wave = pixmap

    def paintEvent(self, event) -> None:
        ratio = self.devicePixelRatioF()
        if (self._wave is None
                or self._wave.width() != int(self.width() * ratio)):
            self._rebuild()
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._wave)

        if self._duration:
            x = self._position / self._duration * self.width()
            painter.setPen(QPen(_PLAYHEAD, 2))
            painter.drawLine(QPointF(x, 0), QPointF(x, self.height()))

        if self._hover is not None and self._duration:
            self._draw_tooltip(painter)
        painter.end()

    def _draw_tooltip(self, painter: QPainter) -> None:
        at = min(1.0, max(0.0, self._hover / max(1, self.width())))
        seconds = at * self._duration
        text = (f"{format_elapsed(seconds)} "
                f"(-{format_elapsed(max(0.0, self._duration - seconds))})")
        font = painter.font()
        font.setPointSizeF(11.0)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        wide = metrics.horizontalAdvance(text) + 14
        tall = metrics.height() + 6
        x = min(max(0.0, self._hover - wide / 2), self.width() - wide)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(_TOOLTIP_EDGE, 1))
        painter.setBrush(_TOOLTIP_BACK)
        painter.drawRoundedRect(int(x), 6, wide, tall, 4, 4)
        painter.setPen(_TOOLTIP_INK)
        painter.drawText(int(x), 6, wide, tall,
                         Qt.AlignmentFlag.AlignCenter, text)

    # ------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        if self._duration and self.width():
            where = event.position().x() / self.width()
            self.seek_requested.emit(
                min(self._duration, max(0.0, where * self._duration)))

    def mouseMoveEvent(self, event) -> None:
        self._hover = event.position().x()
        self.update()

    def leaveEvent(self, event) -> None:
        self._hover = None
        self.update()
