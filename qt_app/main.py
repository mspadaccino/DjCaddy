"""Entry point dell'app Qt: la finestra, le quattro tab, il lettore.

    poetry run python -m qt_app.main

Le tab sono le stesse sezioni, nello stesso ordine, del menu Streamlit; il
lettore sta sotto di tutte, fuori dalle tab, che è come st.bottom lo tiene
su ogni pagina. Dalla Fase 4 le pagine sono tutte vere, e si apre su Wave
analysis come il menu di là.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Lanciato come script (`python qt_app/main.py`) sul path c'è solo qt_app/:
# senza la radice non si importano né `core` né `qt_app` stesso. Con `-m`
# non serve, ma non disturba.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Prima di creare la QApplication, non dopo: Chromium esige che il contesto
# OpenGL condiviso sia deciso all'avvio, e l'import è quello che lo decide.
from PySide6 import QtWebEngineWidgets  # noqa: F401  (l'import È l'effetto)
from PySide6.QtWidgets import (QApplication, QMainWindow, QTabWidget,
                               QVBoxLayout, QWidget)

from qt_app.pages.folder import FolderPage
from qt_app.pages.map import MapPage
from qt_app.pages.tag import TagPage
from qt_app.pages.wave import WavePage
from qt_app.state import AppState
from qt_app.theme import apply_theme
from qt_app.widgets.player_dock import PlayerDock


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Wavecut")
        self.state = AppState(self)
        # Il lettore nasce prima delle pagine: la pagina Wave gli si
        # aggancia (posizione e salti) fin dal costruttore.
        self.player = PlayerDock(self.state)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.wave_page = WavePage(self.state, self.player)
        self.tabs.addTab(self.wave_page, "🌊 Cue analysis")
        self.tabs.addTab(TagPage(self.state), "🏷️ Tag analysis")
        self.tabs.addTab(FolderPage(self.state), "📁 Folder analysis")
        self.map_page = MapPage(self.state)
        self.tabs.addTab(self.map_page, "🗺️ Map")

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(self.tabs, stretch=1)
        layout.addWidget(self.player)
        self.setCentralWidget(central)


def main() -> int:
    QApplication.setApplicationName("Wavecut")
    app = QApplication(sys.argv)
    apply_theme(app)
    window = MainWindow()
    # Dentro lo schermo, sempre: il lettore compare in FONDO alla finestra,
    # e una finestra più alta dello schermo nasconde esattamente lui — si
    # vedeva solo a tutto schermo. 1500×940 resta il massimo, non la misura.
    #
    # Il Dock a scomparsa è il caso subdolo: non toglie spazio alla
    # geometria "disponibile", ma quando riappare passa SOPRA la finestra —
    # cioè sopra il lettore. Se il sistema non ha già scontato una barra in
    # basso (available e schermo finiscono alla stessa riga), il margine
    # glielo lasciamo noi, e la finestra si appoggia in alto.
    screen = app.primaryScreen()
    available = screen.availableGeometry()
    dock_gap = 96 if available.bottom() == screen.geometry().bottom() else 12
    wide = min(1500, available.width() - 24)
    # setGeometry parla del CLIENT, non della cornice: il fondo — dove vive
    # il lettore — atterra esattamente a dock_gap dal bordo dello schermo,
    # qualunque altezza abbia la barra del titolo. Con move()+resize() la
    # cornice spingeva il client in giù di ~28 px, e il margine lasciato al
    # Dock se li mangiava proprio lì.
    top = available.y() + 40
    tall = min(940, available.bottom() - dock_gap - top)
    window.setGeometry(available.x() + (available.width() - wide) // 2,
                       top, wide, tall)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
