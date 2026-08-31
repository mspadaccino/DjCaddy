"""Lo splash che copre l'apertura della mappa: marchio che si compone in
loop di 4s finché il caricamento vero non finisce.

Le percentuali di ogni riga vengono dai `@keyframes` del riferimento
`reference/DjCaddy Logo.dc.html` (dcDot/dcPath/dcRing/dcBar/dcCue/dcWave/
dcFade), interpolati linearmente fra le tappe indicate — è lo stesso
risultato del comportamento di default di CSS fra due keyframe adiacenti,
qui riletto un fotogramma alla volta invece che dal motore del browser.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QElapsedTimer, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QRadialGradient
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QWidget

from .mark_svg import build_mark_svg

_DURATION_MS = 4000.0
_FRAME_MS = 33  # ~30fps, di più non serve per un loop decorativo

# cx, cy, r, colore, ritardo(s) — dal riferimento (26 punti della mappa).
_DOTS = [
    (96, 118, 5, "#9184D9", .05), (168, 86, 3.5, "#9184D9", .12),
    (234, 152, 6, "#FF8F4D", .2), (122, 214, 4, "#9184D9", .26),
    (292, 96, 4.5, "#9184D9", .33), (196, 266, 5.5, "#9184D9", .4),
    (72, 322, 4, "#3EE5A0", .46), (286, 330, 4.5, "#9184D9", .53),
    (146, 404, 6, "#9184D9", .6), (248, 446, 3.5, "#9184D9", .66),
    (352, 404, 5, "#FF6A2B", .73), (368, 228, 4, "#9184D9", .8),
    (424, 120, 5, "#9184D9", .86), (486, 72, 3.5, "#9184D9", .93),
    (556, 126, 5.5, "#9184D9", 1.0), (628, 88, 4, "#FF8F4D", 1.06),
    (702, 146, 4.5, "#9184D9", 1.13), (784, 104, 5, "#9184D9", 1.2),
    (812, 212, 3.5, "#9184D9", 1.26), (736, 286, 6, "#9184D9", 1.33),
    (806, 368, 4, "#9184D9", 1.4), (664, 392, 5, "#3EE5A0", 1.46),
    (722, 452, 3.5, "#9184D9", 1.53), (576, 446, 4.5, "#9184D9", 1.6),
    (486, 384, 4, "#9184D9", 1.66), (616, 316, 3.5, "#9184D9", 1.73),
]

_PATH_D = "M96 118 C 210 190 168 330 286 330 S 470 470 616 316 S 700 150 784 104"

_DOT_OPACITY = [(0, 0), (6, 0), (22, .55), (100, .55)]
_PATH_DASHOFFSET = [(0, 1200), (12, 1200), (52, 0), (100, 0)]
_PATH_OPACITY = [(0, 0), (12, 0), (18, 1), (52, 1), (100, 1)]
_RING_DASHOFFSET = [(0, 200), (42, 0), (100, 0)]
_RING_OPACITY = [(0, 0), (10, 1), (100, 1)]
_BAR_SCALE = [(0, 0), (12, 0), (38, 1), (52, .88), (62, 1.05), (74, .94),
             (86, 1), (100, 1)]
_BAR_DELAYS = (.1, .2, .3, .4)
_CUE_OPACITY = [(0, 0), (44, 0), (58, 1), (70, 1), (100, 1)]
_CUE_SCALE = [(0, .3), (44, .3), (58, 1.35), (70, 1), (100, 1)]
_WAVE_SCALE = [(0, 0), (46, 0), (94, 1), (100, 1)]
_FADE_OPACITY = [(0, 0), (52, 0), (74, 1), (100, 1)]
_FADE_TRANSLATE = [(0, 7), (52, 7), (74, 0), (100, 0)]

_PULSE_DELAY_MS = 1600
_PULSE_MS = 970
_PULSE_AMPLITUDE = 0.035

_WAVE_BANDS = [  # altezza, colore, periodo del tratteggio, opacità
    (7, "#5B8CFF", 6, .85), (10, "#3EE5A0", 5, .9), (7, "#FF4B4B", 7, .85),
]


def _kf(pct: float, stops: list[tuple[float, float]]) -> float:
    if pct <= stops[0][0]:
        return stops[0][1]
    if pct >= stops[-1][0]:
        return stops[-1][1]
    for (p0, v0), (p1, v1) in zip(stops, stops[1:]):
        if p0 <= pct <= p1:
            f = (pct - p0) / (p1 - p0) if p1 != p0 else 1.0
            return v0 + (v1 - v0) * f
    return stops[-1][1]


def _phase(elapsed_ms: float, delay_s: float = 0.0) -> float:
    """La percentuale (0-100) dentro il ciclo di 4s, tenendo conto del
    ritardo: prima che il ritardo scada l'elemento resta al fotogramma 0%,
    come nella spiegazione "dcBar 4s .Xs infinite" del riferimento."""
    t = elapsed_ms - delay_s * 1000
    if t < 0:
        return 0.0
    return (t % _DURATION_MS) / _DURATION_MS * 100.0


class SplashScreen(QWidget):
    """Finestra frameless 880×520: si chiude quando il caricamento reale
    finisce, non a fine ciclo (`main.py` lo comanda dall'esterno)."""

    WIDTH = 880
    HEIGHT = 520

    def __init__(self, status_text: str, parent=None) -> None:
        # StaysOnTop: senza, una finestra frameless nata da terminale può
        # aprirsi dietro chi l'ha lanciata, senza rubargli il fuoco — e
        # sparire dietro sembra, da fuori, non essere mai apparsa.
        flags = (Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window
                | Qt.WindowType.WindowStaysOnTopHint)
        super().__init__(parent, flags)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setFixedSize(self.WIDTH, self.HEIGHT)
        self._status_text = status_text

        # Costruiti una volta sola: risolvere una famiglia mancante (Inter
        # non è installato ovunque) costa più di un QFont vuoto, e qui
        # tocca farlo 30 volte al secondo se non si tiene la risposta.
        self._wordmark_font = QFont("Inter", 40)
        self._wordmark_font.setWeight(QFont.Weight.DemiBold)
        self._wordmark_font.setLetterSpacing(
            QFont.SpacingType.PercentageSpacing, 96.5)
        self._status_font = QFont("Menlo", 11)

        self._clock = QElapsedTimer()
        self._clock.start()
        self._timer = QTimer(self)
        self._timer.setInterval(_FRAME_MS)
        self._timer.timeout.connect(self.update)
        self._timer.start()

    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 (nome Qt)
        elapsed = self._clock.elapsed()
        t = _phase(elapsed)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0e1117"))

        self._paint_dots(painter, elapsed)
        self._paint_path(painter, t)
        self._paint_mark(painter, elapsed, t)
        self._paint_wordmark(painter, t)
        self._paint_waveform(painter, t)
        self._paint_status(painter, t)

    def _paint_dots(self, painter: QPainter, elapsed: float) -> None:
        for cx, cy, r, color, delay in _DOTS:
            opacity = _kf(_phase(elapsed, delay), _DOT_OPACITY)
            if opacity <= 0:
                continue
            # Un radiale al posto del gaussian blur SVG: più semplice, e
            # visivamente lo stesso alone morbido dietro il marchio.
            glow = r + 6
            gradient = QRadialGradient(cx, cy, glow)
            base = QColor(color)
            base.setAlphaF(opacity)
            edge = QColor(color)
            edge.setAlphaF(0.0)
            gradient.setColorAt(0.0, base)
            gradient.setColorAt(0.55, base)
            gradient.setColorAt(1.0, edge)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(gradient)
            painter.drawEllipse(QRectF(cx - glow, cy - glow, glow * 2, glow * 2))

    def _paint_path(self, painter: QPainter, t: float) -> None:
        offset = _kf(t, _PATH_DASHOFFSET)
        opacity = _kf(t, _PATH_OPACITY)
        if opacity <= 0:
            return
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.WIDTH}" '
            f'height="{self.HEIGHT}" viewBox="0 0 {self.WIDTH} {self.HEIGHT}">'
            f'<path d="{_PATH_D}" fill="none" stroke="#FF6A2B" '
            f'stroke-width="2.5" stroke-linecap="round" '
            f'stroke-dasharray="1200" stroke-dashoffset="{offset}" '
            f'opacity="{opacity}"/></svg>'
        ).encode("utf-8")
        QSvgRenderer(svg).render(painter)

    def _pulse_scale(self, elapsed: float) -> float:
        if elapsed < _PULSE_DELAY_MS:
            return 1.0
        phase = ((elapsed - _PULSE_DELAY_MS) % _PULSE_MS) / _PULSE_MS
        return 1.0 + _PULSE_AMPLITUDE * 0.5 * (1 - math.cos(2 * math.pi * phase))

    def _paint_mark(self, painter: QPainter, elapsed: float, t: float) -> None:
        bar_scales = tuple(
            _kf(_phase(elapsed, delay), _BAR_SCALE) for delay in _BAR_DELAYS)
        svg = build_mark_svg(
            184,
            ring_dashoffset=_kf(t, _RING_DASHOFFSET),
            ring_opacity=_kf(t, _RING_OPACITY),
            bar_scales=bar_scales,
            cue_scale=_kf(t, _CUE_SCALE),
            cue_opacity=_kf(t, _CUE_OPACITY),
            pulse=self._pulse_scale(elapsed),
        )
        x = (self.WIDTH - 184) / 2
        y = self._content_top()
        QSvgRenderer(svg).render(painter, QRectF(x, y, 184, 184))

    def _content_top(self) -> float:
        # marchio 184 + gap 26 + wordmark ~40 + gap 26 + waveform 26,
        # centrati nei 520px di altezza.
        total = 184 + 26 + 40 + 26 + 26
        return (self.HEIGHT - total) / 2

    def _paint_wordmark(self, painter: QPainter, t: float) -> None:
        opacity = _kf(t, _FADE_OPACITY)
        if opacity <= 0:
            return
        dy = _kf(t, _FADE_TRANSLATE)
        painter.setFont(self._wordmark_font)
        y = self._content_top() + 184 + 26 + dy

        metrics = painter.fontMetrics()
        dj_w = metrics.horizontalAdvance("Dj")
        caddy_w = metrics.horizontalAdvance("Caddy")
        x0 = (self.WIDTH - dj_w - caddy_w) / 2

        painter.setOpacity(opacity)
        painter.setPen(QColor("#FAFAFA"))
        painter.drawText(QRectF(x0, y, dj_w, 44),
                         int(Qt.AlignmentFlag.AlignVCenter), "Dj")
        painter.setPen(QColor("#FF6A2B"))
        painter.drawText(QRectF(x0 + dj_w, y, caddy_w, 44),
                         int(Qt.AlignmentFlag.AlignVCenter), "Caddy")
        painter.setOpacity(1.0)

    def _paint_waveform(self, painter: QPainter, t: float) -> None:
        reveal = 300 * _kf(t, _WAVE_SCALE)
        if reveal <= 0:
            return
        x0 = (self.WIDTH - 300) / 2
        y = self._content_top() + 184 + 26 + 40 + 26
        painter.save()
        painter.setClipRect(QRectF(x0, y, reveal, 26))
        row_y = y
        for height, color, period, opacity in _WAVE_BANDS:
            painter.setOpacity(opacity)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(color))
            x = x0
            while x < x0 + 300:
                painter.drawRect(QRectF(x, row_y, 2, height))
                x += period
            row_y += height + 1
        painter.restore()
        painter.setOpacity(1.0)

    def _paint_status(self, painter: QPainter, t: float) -> None:
        opacity = _kf(t, _FADE_OPACITY)
        if opacity <= 0:
            return
        painter.setFont(self._status_font)
        painter.setOpacity(opacity)
        painter.setPen(QColor("#808495"))
        rect = QRectF(0, self.HEIGHT - 22 - 16, self.WIDTH, 16)
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter),
                         self._status_text)
        painter.setOpacity(1.0)
