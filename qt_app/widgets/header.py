"""La riga del marchio sopra la barra delle schede — nient'altro, di
proposito: il marchio a 30px non deve competere con la tab attiva.
"""

from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from qt_app.state import AppState

from .mark import PulsingMark

_WORDMARK = '<span style="color:#FAFAFA;">Dj</span><span style="color:#FF6A2B;">Caddy</span>'


class HeaderBar(QWidget):
    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(56)

        self._mark = PulsingMark(30)
        self._mark.set_pulsing(state.analysis_running)

        wordmark = QLabel(_WORDMARK)
        font = QFont("Inter", 20)
        font.setWeight(QFont.Weight.DemiBold)
        font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 96.5)
        wordmark.setFont(font)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(12)
        layout.addWidget(self._mark)
        layout.addWidget(wordmark)
        layout.addStretch(1)

        state.analysis_running_changed.connect(self._mark.set_pulsing)
