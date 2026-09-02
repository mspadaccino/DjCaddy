"""La riga del marchio sopra la barra delle schede, e all'altro capo la
guida e l'interruttore del tema — nient'altro, di proposito: il marchio a
30px non deve competere con la tab attiva.

Stanno qui per lo stesso motivo: valgono per l'app intera, non per una
pagina, e in fondo a una sola scheda sembrerebbero preferenze di quella
scheda.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QKeySequence
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


def _icon_button_sheet() -> str:
    """Il foglio dei bottoncini dell'intestazione.

    `padding: 0` non è pignoleria: quello del foglio dell'app (0.9em per
    lato) lascerebbe al glifo dieci pixel su trentasei, e il sole si vedeva
    come una scheggia gialla.
    """
    return ("QPushButton { background: transparent; border: none;"
            " padding: 0; font-size: 15px; }"
            f"QPushButton:hover {{ background: {theme.RAISED};"
            " border-radius: 6px; }")


class GuideButton(QPushButton):
    """La guida dell'app, a un clic da ogni scheda.

    La finestra si costruisce al primo clic e poi si tiene: leggere e
    impaginare il README non è caro, ma rifarlo a ogni apertura butterebbe
    via il punto in cui si era rimasti."""

    def __init__(self, parent=None) -> None:
        super().__init__("📖", parent)
        self.setFixedSize(36, 28)
        self.setFlat(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setShortcut(QKeySequence.StandardKey.HelpContents)
        self.setToolTip(theme.hint(
            "Open the guide — what each tab does, and what every number "
            "means"))
        theme.style(self, _icon_button_sheet)
        self._window = None
        self.clicked.connect(self._on_click)

    def _on_click(self) -> None:
        from qt_app.widgets.help_window import HelpWindow

        if self._window is None:
            self._window = HelpWindow(self.window())
        self._window.show()
        self._window.raise_()
        self._window.activateWindow()


class ThemeSwitch(QPushButton):
    """Il sole o la luna: dice DOVE si va, non dove si è — un bottone
    mostra l'azione che compie."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(36, 28)
        self.setFlat(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        theme.style(self, _icon_button_sheet)
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
        layout.addWidget(GuideButton())
        layout.addWidget(ThemeSwitch())

        state.analysis_running_changed.connect(self._mark.set_pulsing)
        theme.bus().changed.connect(
            lambda: self._wordmark.setText(_wordmark()))
