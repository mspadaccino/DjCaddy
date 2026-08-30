"""Il tema scuro dell'app: gli stessi colori della pagina Streamlit.

I quattro fondi e i due inchiostri vengono da lì — `SKIN["dark"]` in
`core.viz.map_figure` per carta e riquadro della mappa, il tema scuro di
Streamlit per il resto — perché il criterio del parallel run è "stessa cosa,
più fluida": due tavolozze diverse renderebbero ogni confronto un confronto
fra temi invece che fra app.

Fusion come stile di base, e non quello nativo del Mac: è l'unico che
rispetta la QPalette fino in fondo su tutte le piattaforme, quindi è quello
che renderà l'app uguale su macOS e Win11.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

# Da Streamlit scuro: fondo pagina, riquadri, testo. PLOT è il fondo della
# mappa (SKIN["dark"]["plot"]), staccato di poco dal fondo pagina: quel poco
# basta a dire dove finisce il testo e comincia il territorio.
BACKGROUND = "#0e1117"
PLOT = "#161a22"
RAISED = "#262730"
INK = "#fafafa"
FADED = "#808495"
PRIMARY = "#ff4b4b"

# L'inchiostro DENTRO le pastiglie colorate: scuro sempre, perché le scale di
# `core.viz.track_columns` non scendono mai sotto metà luminosità apposta.
PILL_INK = "#1b1f27"

_QSS = f"""
QMainWindow, QWidget {{
    background: {BACKGROUND};
    color: {INK};
}}

QTabWidget::pane {{ border: none; }}
QTabBar::tab {{
    background: transparent; color: {FADED};
    padding: 0.45em 1.1em; border: none;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ color: {INK}; border-bottom: 2px solid {PRIMARY}; }}
QTabBar::tab:hover {{ color: {INK}; }}

QTableView {{
    background: {PLOT}; alternate-background-color: {BACKGROUND};
    color: {INK}; border: none; border-radius: 6px;
    gridline-color: transparent;
    selection-background-color: {RAISED}; selection-color: {INK};
}}
QHeaderView::section {{
    background: {BACKGROUND}; color: {FADED};
    border: none; padding: 0.3em 0.5em;
}}
QTableCornerButton::section {{ background: {BACKGROUND}; border: none; }}

QPlainTextEdit, QTextEdit {{
    background: {PLOT}; color: {FADED};
    border: none; border-radius: 6px;
}}

QPushButton {{
    background: {RAISED}; color: {INK};
    border: none; border-radius: 6px; padding: 0.35em 0.9em;
}}
QPushButton:hover {{ background: #33343f; }}
QPushButton:pressed {{ background: {PRIMARY}; }}

QSplitter::handle {{ background: {BACKGROUND}; }}
QSplitter::handle:horizontal {{ width: 6px; }}
QSplitter::handle:vertical {{ height: 6px; }}

QScrollBar {{ background: {BACKGROUND}; border: none; }}
QScrollBar:vertical {{ width: 10px; }}
QScrollBar:horizontal {{ height: 10px; }}
QScrollBar::handle {{ background: {RAISED}; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:hover {{ background: #3a3b47; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QLabel#dim {{ color: {FADED}; }}
"""


def apply_theme(app: QApplication) -> None:
    """Veste l'applicazione: stile Fusion, palette scura, QSS."""
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BACKGROUND))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(INK))
    palette.setColor(QPalette.ColorRole.Base, QColor(PLOT))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(BACKGROUND))
    palette.setColor(QPalette.ColorRole.Text, QColor(INK))
    palette.setColor(QPalette.ColorRole.Button, QColor(RAISED))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(INK))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(RAISED))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(INK))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(FADED))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(RAISED))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(INK))
    app.setPalette(palette)

    app.setStyleSheet(_QSS)
