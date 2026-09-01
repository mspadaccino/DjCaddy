"""La riga del marchio sopra la barra delle schede, e l'interruttore del
tema all'altro capo — nient'altro, di proposito: il marchio a 30px non deve
competere con la tab attiva.

L'interruttore sta qui perché il tema è dell'app intera, non di una pagina:
in fondo a una sola scheda sembrerebbe una preferenza di quella scheda.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from qt_app import theme
from qt_app.state import AppState

from .mark import PulsingMark

# L'arancio del marchio è del marchio, non del tema: resta quello sui due
# fondi. Il "Dj" invece è l'inchiostro della pagina, e cambia con lei.
_BRAND = "#FF6A2B"


def _wordmark() -> str:
    return (f'<span style="color:{theme.INK};">Dj</span>'
            f'<span style="color:{_BRAND};">Caddy</span>')


class ThemeSwitch(QPushButton):
    """Il sole o la luna: dice DOVE si va, non dove si è — un bottone
    mostra l'azione che compie."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(36, 28)
        self.setFlat(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # `padding: 0` non è pignoleria: quello del foglio dell'app (0.9em
        # per lato) lascerebbe al glifo dieci pixel su trentasei, e il sole
        # si vedeva come una scheggia gialla.
        theme.style(self, lambda: (
            "QPushButton { background: transparent; border: none;"
            " padding: 0; font-size: 15px; }"
            f"QPushButton:hover {{ background: {theme.RAISED};"
            " border-radius: 6px; }"))
        self.clicked.connect(lambda: theme.set_dark(not theme.is_dark()))
        theme.bus().changed.connect(self._retitle)
        self._retitle()

    def _retitle(self) -> None:
        self.setText("☀️" if theme.is_dark() else "🌙")
        self.setToolTip(theme.hint(
            "Switch to the light theme" if theme.is_dark()
            else "Switch to the dark theme"))


class HeaderBar(QWidget):
    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(56)

        self._mark = PulsingMark(30)
        self._mark.set_pulsing(state.analysis_running)

        self._wordmark = QLabel(_wordmark())
        font = QFont("Inter", 20)
        font.setWeight(QFont.Weight.DemiBold)
        font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 96.5)
        self._wordmark.setFont(font)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(12)
        layout.addWidget(self._mark)
        layout.addWidget(self._wordmark)
        layout.addStretch(1)
        layout.addWidget(ThemeSwitch())

        state.analysis_running_changed.connect(self._mark.set_pulsing)
        theme.bus().changed.connect(
            lambda: self._wordmark.setText(_wordmark()))
