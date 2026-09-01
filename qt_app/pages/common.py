"""I pezzi che ogni pagina rifà uguale: etichette, metriche, conferme.

Nati con la Fase 4 (tre pagine nuove in un colpo); la pagina Map importa da
qui quello che prima definiva in casa — `reveal_in_files`, `spelled` — così
la copia resta una.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QWidget)

from qt_app import theme


def reveal_in_files(path: str) -> None:
    """Mostra il file nel gestore della piattaforma: Finder, Esplora, o la
    cartella e basta dove un "seleziona questo" non esiste."""
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", "-R", path])
        elif sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", path])
        else:
            subprocess.Popen(["xdg-open", str(Path(path).parent)])
    except OSError:
        pass


def spelled(seconds: float) -> str:
    """Una durata come la si direbbe: secondi, minuti o ore."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} hours"


def dim(text: str = "") -> QLabel:
    """La riga di spiegazione spenta, che va a capo da sé."""
    label = QLabel(text)
    label.setObjectName("dim")
    label.setWordWrap(True)
    return label


class Metric(QLabel):
    """L'equivalente di st.metric: titolo spento, valore grande, nota.

    Il titolo è fisso; `show_` cambia valore e nota. La nota è la riga
    piccola sotto il valore (in Streamlit era il `delta` usato come
    percentuale, non come variazione).
    """

    def __init__(self, title: str, tooltip: str = "",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        if tooltip:
            self.setToolTip(tooltip)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.show_("—")

    def show_(self, value: str, note: str = "") -> None:
        self.setText(
            f"<span style='color:{theme.FADED}; font-size:11px;'>"
            f"{self._title}</span><br>"
            f"<span style='font-size:20px;'>{value}</span>"
            + (f"<br><span style='color:{theme.FADED}; font-size:11px;'>"
               f"{note}</span>" if note else ""))


class ConfirmBar(QWidget):
    """La coppia spunta-di-conferma + bottone delle azioni che pesano.

    Stesso patto della pagina Streamlit: il bottone si accende solo dopo la
    spunta, e la spunta si spegne da sola dopo l'uso — la conferma vale per
    UN gesto, non per sempre.
    """

    activated = Signal()

    def __init__(self, button_text: str, primary: bool = False,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._check = QCheckBox("")
        self._button = QPushButton(button_text)
        self._button.setEnabled(False)
        if primary:
            theme.style(self._button, theme.primary_button)
        self._check.toggled.connect(self._button.setEnabled)
        self._button.clicked.connect(self._on_click)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._check, stretch=1)
        row.addWidget(self._button)

    def set_ask(self, text: str) -> None:
        self._check.setText(text)

    def reset(self) -> None:
        self._check.setChecked(False)

    def _on_click(self) -> None:
        self.activated.emit()
        self.reset()


class FolderRow(QWidget):
    """La riga "scegli la cartella": campo + sfoglia, come `pick_folder`.

    `chosen` porta il percorso quando diventa una cartella vera — scritto a
    mano e confermato con Invio, o preso dal dialogo di sistema.
    """

    chosen = Signal(object)     # Path

    def __init__(self, label: str = "Folder",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._edit = QLineEdit()
        self._edit.setPlaceholderText("Select library root")
        self._edit.returnPressed.connect(self._on_typed)
        browse = QPushButton("📁 Browse…")
        browse.clicked.connect(self._on_browse)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel(label))
        row.addWidget(self._edit, stretch=1)
        row.addWidget(browse)

    def path(self) -> Path | None:
        text = self._edit.text().strip()
        if not text:
            return None
        folder = Path(text).expanduser()
        return folder if folder.is_dir() else None

    def _on_typed(self) -> None:
        folder = self.path()
        if folder is not None:
            self.chosen.emit(folder)

    def _on_browse(self) -> None:
        picked = QFileDialog.getExistingDirectory(self, "Choose a folder")
        if picked:
            self._edit.setText(picked)
            self.chosen.emit(Path(picked))
