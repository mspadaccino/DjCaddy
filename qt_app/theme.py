"""I due temi dell'app — lo scuro e il chiaro — e l'interruttore fra loro.

I colori vengono dalla pagina Streamlit, tema per tema: lo scuro dal suo
`[theme.dark]`, il chiaro dal chiaro di Streamlit, e le carte dei grafici da
`SKIN` in `core.viz.map_figure` — perché il criterio del parallel run è
"stessa cosa, più fluida": due tavolozze diverse renderebbero ogni confronto
un confronto fra temi invece che fra app.

Il tema si cambia a finestra aperta, e questo modulo è il modo in cui il
cambio arriva dappertutto. Tre canali, che coprono tre modi di essere
colorati:

* il FOGLIO DELL'APP e la palette — riscritti da capo a ogni cambio, e da
  soli vestono la maggior parte dei widget;
* i FOGLI IN LINEA, che sarebbero cotti nel costruttore: passano da
  `style()`, che li ricalcola al cambio invece di lasciarli com'erano;
* chi DISEGNA da sé o vive dentro una pagina web — tabelle, mappa, lavagna,
  ruota: si iscrive a `bus().changed` e si ridisegna.

Chi legge `theme.INK` dentro un `paintEvent` non ha bisogno di niente: i
nomi qui sono variabili di modulo, e `use()` le riscrive.

Fusion come stile di base, e non quello nativo del Mac: è l'unico che
rispetta la QPalette fino in fondo su tutte le piattaforme, quindi è quello
che renderà l'app uguale su macOS e Win11.
"""

from __future__ import annotations

import weakref
from typing import Callable

from PySide6.QtCore import QObject, QSettings, Qt, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QWidget

# I colori che NON cambiano col tema, perché sono leggibili su tutti e due i
# fondi e cambiarli vorrebbe dire due rossi diversi per la stessa cosa: il
# rosso dell'azione, il grigio del testo secondario (3,3:1 sul bianco, 5,2:1
# sul nero) e l'inchiostro DENTRO le pastiglie colorate — scuro sempre,
# perché le scale di `core.viz.track_columns` non scendono mai sotto metà
# luminosità apposta.
PRIMARY = "#ff4b4b"
FADED = "#808495"
PILL_INK = "#1b1f27"

# Quello che invece cambia. Le due tavolozze si leggono in parallelo, riga
# per riga: stesso ruolo, stessa posizione, un valore per tema.
#
# BACKGROUND è il fondo della finestra ED È lo stesso `SKIN[...]["paper"]`
# della mappa: la carta del grafico e la pagina sono la stessa superficie,
# senza cucitura attorno al disegno. PLOT è il fondo degli elenchi, staccato
# di poco: quel poco basta a dire dove finisce la pagina e comincia la
# tabella. RAISED è quello che si solleva — bottoni, lettore, pastiglie
# senza colore.
#
# Le righe-segnale delle tabelle sono trasparenti, e sul bianco un velo rende
# meno che sul nero: le stesse tinte, un filo più cariche.
_THEMES = {
    "dark": {
        "BACKGROUND": "#0e1117",
        "PLOT": "#161a22",
        "RAISED": "#262730",
        "INK": "#fafafa",
        "HOVER": "#33343f",
        "BAR_HOVER": "#3a3b47",
        "WARN": "#ffb454",
        "OK": "#3fbf7f",
        "SELECTED_ROW": QColor(255, 75, 75, 76),
        "PLAYING_ROW": QColor(255, 233, 77, 46),
        "TWIN_NAME_ROW": QColor(255, 160, 60, 64),
        "TWIN_SOUND_ROW": QColor(150, 120, 255, 60),
    },
    "light": {
        "BACKGROUND": "#ffffff",
        "PLOT": "#f2f5f9",
        "RAISED": "#e4e9f0",
        "INK": "#1b1f27",
        "HOVER": "#d6dce6",
        "BAR_HOVER": "#c3cad6",
        "WARN": "#a86a00",
        "OK": "#1f7d55",
        "SELECTED_ROW": QColor(255, 75, 75, 64),
        "PLAYING_ROW": QColor(240, 190, 0, 92),
        "TWIN_NAME_ROW": QColor(255, 150, 40, 80),
        "TWIN_SOUND_ROW": QColor(140, 110, 255, 64),
    },
}

# I nomi che tutta l'app legge. Non sono costanti: `use()` li riscrive, ed è
# per questo che ovunque si scrive `theme.INK` e non `from ... import INK` —
# un nome importato una volta resterebbe al tema di allora.
DARK = True
BACKGROUND = PLOT = RAISED = INK = HOVER = BAR_HOVER = WARN = OK = ""
SELECTED_ROW = PLAYING_ROW = TWIN_NAME_ROW = TWIN_SOUND_ROW = QColor()

_SETTINGS = ("DjCaddy", "DjCaddy")


class _Bus(QObject):
    """Il filo che dice "il tema è cambiato" a chi si disegna da sé."""

    changed = Signal()


_bus: _Bus | None = None


def bus() -> _Bus:
    """Il filo, creato al primo che lo chiede: il modulo si importa prima
    della QApplication, e un QObject di modulo nascerebbe troppo presto."""
    global _bus
    if _bus is None:
        _bus = _Bus()
    return _bus


def is_dark() -> bool:
    return DARK


def use(dark: bool) -> None:
    """Installa una delle due tavolozze nei nomi di modulo."""
    global DARK
    DARK = bool(dark)
    globals().update(_THEMES["dark" if DARK else "light"])


use(True)


def hint(text: str) -> str:
    """Un tooltip che va a capo: il testo piano Qt lo scrive su una riga
    sola, lunga quanto lo schermo — da rich text si impagina. Le parti
    descrittive della pagina vivono nei tooltip apposta (lo spazio è dei
    grafici e delle tabelle), quindi qui passano tutte."""
    return "<qt>" + text.replace("\n", "<br>") + "</qt>"


def primary_button() -> str:
    """Il foglio del bottone che compie l'azione: rosso, e SPENTO quando è
    disabilitato.

    Il foglio in linea vince su quello dell'app, quindi la riga `:disabled`
    va ripetuta qui: senza, un bottone inerte resta rosso pieno e invita a
    un clic che non fa niente — sembrava che non funzionasse, e invece
    aspettava la spunta di conferma.
    """
    return (f"QPushButton {{ background: {PRIMARY}; color: white; }}"
            f"QPushButton:disabled {{ background: {RAISED};"
            f" color: {FADED}; }}")


# I fogli in linea che seguono il tema: il widget e la funzione che sa
# scrivere il suo foglio COL tema di adesso. Riferimenti deboli, perché
# questa lista vive quanto il modulo — cioè più di qualunque widget.
_STYLED: list[tuple[weakref.ref, Callable[[], str]]] = []


def style(widget: QWidget, sheet: Callable[[], str]) -> None:
    """Veste `widget` col foglio che `sheet()` scrive: adesso, e di nuovo a
    ogni cambio di tema.

    Si passa la FUNZIONE, non il foglio già scritto: è il solo modo perché i
    colori dentro vengano riletti dopo, e il QSS è pieno di graffe che con un
    template a segnaposto andrebbero protette una per una.
    """
    widget.setStyleSheet(sheet())
    _STYLED.append((weakref.ref(widget), sheet))


def _rewrite_styles() -> None:
    """Riscrive i fogli in linea, dimenticando i widget che non ci sono più:
    l'oggetto Python vivo con la parte C++ già distrutta — un dialogo chiuso,
    per dire — alza RuntimeError invece di sparire."""
    alive = []
    for ref, sheet in _STYLED:
        widget = ref()
        if widget is None:
            continue
        try:
            widget.setStyleSheet(sheet())
        except RuntimeError:
            continue
        alive.append((ref, sheet))
    _STYLED[:] = alive


def _qss() -> str:
    # La riga selezionata nelle tabelle: il rosso primario a bassa opacità.
    # Prima era RAISED, che dal fondo alternato si distingueva appena — e la
    # riga scelta è un'informazione, non una sfumatura. Una volta sola nella
    # tavolozza, in due forme: il QColor va nella palette (è quello che
    # leggono i delegate), questa stringa nel QSS.
    selected = (f"rgba({SELECTED_ROW.red()}, {SELECTED_ROW.green()}, "
                f"{SELECTED_ROW.blue()}, {SELECTED_ROW.alphaF():.2f})")
    return f"""
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
    selection-background-color: {selected}; selection-color: {INK};
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
QPushButton:hover {{ background: {HOVER}; }}
QPushButton:pressed {{ background: {PRIMARY}; }}
QPushButton:disabled {{ background: {RAISED}; color: {FADED}; }}

QSplitter::handle {{ background: {BACKGROUND}; }}
QSplitter::handle:horizontal {{ width: 6px; }}
QSplitter::handle:vertical {{ height: 6px; }}

QScrollBar {{ background: {BACKGROUND}; border: none; }}
QScrollBar:vertical {{ width: 10px; }}
QScrollBar:horizontal {{ height: 10px; }}
QScrollBar::handle {{ background: {RAISED}; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:hover {{ background: {BAR_HOVER}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QLabel#dim {{ color: {FADED}; }}
"""


def _dress(app: QApplication) -> None:
    """Palette, schema di sistema e foglio dell'app, col tema di adesso."""
    app.setStyle("Fusion")

    # La barra del titolo la disegna il sistema e non guarda la QPalette:
    # su un Mac in modalità chiara restava una striscia bianca sopra le
    # schede, con la finestra tutta scura sotto. Dichiarare lo schema (Qt
    # 6.8+) è il modo previsto per dirlo al sistema, e su macOS diventa
    # l'aspetto della finestra, cornice compresa. Prima della palette:
    # cambiando schema Qt ne installa una sua, e la nostra deve venire dopo.
    app.styleHints().setColorScheme(
        Qt.ColorScheme.Dark if DARK else Qt.ColorScheme.Light)

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BACKGROUND))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(INK))
    palette.setColor(QPalette.ColorRole.Base, QColor(PLOT))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(BACKGROUND))
    palette.setColor(QPalette.ColorRole.Text, QColor(INK))
    palette.setColor(QPalette.ColorRole.Button, QColor(RAISED))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(INK))
    palette.setColor(QPalette.ColorRole.Highlight, SELECTED_ROW)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(INK))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(FADED))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(RAISED))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(INK))
    app.setPalette(palette)

    app.setStyleSheet(_qss())


def apply_theme(app: QApplication) -> None:
    """Veste l'applicazione col tema scelto l'ultima volta."""
    use(QSettings(*_SETTINGS).value("theme", "dark") != "light")
    _dress(app)


def set_dark(dark: bool) -> None:
    """Passa all'altro tema, subito e per la prossima volta.

    L'ordine conta: prima i nomi di modulo (tutto il resto li legge), poi
    l'app, poi i fogli in linea, e per ultimo il segnale — chi si ridisegna
    da sé lo fa quando i colori nuovi sono già tutti al loro posto.
    """
    if bool(dark) == DARK:
        return
    use(dark)
    QSettings(*_SETTINGS).setValue("theme", "dark" if DARK else "light")
    app = QApplication.instance()
    if app is not None:
        _dress(app)
    _rewrite_styles()
    bus().changed.emit()
