"""La guida dentro l'app: indice a sinistra, testo a destra, ricerca sopra.

Il testo è quello di `core.guide` — il README meno i capitoli sul codice —
e non una copia scritta a parte: la finestra mostra sempre la guida vera.

Due dettagli meritano una riga. Ai titoli non si va cercando il testo nel
documento, perché "Build a set" compare anche in mezzo alla prosa: si va per
POSIZIONE, chiedendo a ogni blocco se è un titolo (`headingLevel`), che è
l'unica domanda che non si può sbagliare. E il salto porta il titolo IN
CIMA alla vista invece che dove capita: `ensureCursorVisible` da solo lo
lascerebbe in fondo allo schermo, con il capitolo appena aperto fuori campo.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QTextCursor, QTextDocument
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLineEdit, QListWidget,
                               QListWidgetItem, QSplitter, QTextBrowser,
                               QVBoxLayout, QWidget)

from core import guide
from qt_app import theme


class HelpWindow(QDialog):
    """La guida. Non modale: si tiene aperta di fianco mentre si lavora."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("DjCaddy — Guide")
        self.setModal(False)
        self.resize(1040, 780)

        text = guide.guide()
        self._chapters = guide.contents(text)
        # Dallo slug di un link al titolo a cui porta: è come si segue un
        # `[…](#anchor)` senza che il markdown debba avere ancore vere.
        self._by_anchor = {guide.anchor(title): title
                           for _, title in self._chapters}

        self._build(text)

    # ------------------------------------------------------------------
    def _build(self, text: str) -> None:
        self._index = QListWidget()
        self._index.setFixedWidth(258)
        for level, title in self._chapters:
            item = QListWidgetItem(("    " if level > 2 else "") + title)
            item.setData(Qt.ItemDataRole.UserRole, title)
            if level == 2:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self._index.addItem(item)
        self._index.currentItemChanged.connect(self._on_index)

        self._text = QTextBrowser()
        self._text.setOpenLinks(False)          # i link li seguiamo noi
        self._text.anchorClicked.connect(self._on_link)
        # Sul DOCUMENTO e non sul widget: `QTextEdit.setMarkdown` prende
        # solo il testo, e il dialetto — quello che porta le tabelle — si
        # sceglie qui.
        self._text.document().setMarkdown(
            text, QTextDocument.MarkdownFeature.MarkdownDialectGitHub)

        self._find = QLineEdit()
        self._find.setPlaceholderText("Search the guide…")
        self._find.setClearButtonEnabled(True)
        self._find.returnPressed.connect(self._on_find)

        right = QWidget()
        column = QVBoxLayout(right)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(8)
        column.addWidget(self._find)
        column.addWidget(self._text, stretch=1)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._index)
        split.addWidget(right)
        split.setStretchFactor(1, 1)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(split)

        theme.style(self, lambda: (
            f"QTextBrowser, QListWidget {{ background: {theme.PLOT};"
            f" color: {theme.INK}; border: none; border-radius: 8px;"
            " padding: 10px; }"
            f"QListWidget::item {{ padding: 3px 2px; }}"
            f"QListWidget::item:selected {{ background: {theme.RAISED};"
            f" color: {theme.INK}; border-radius: 4px; }}"
            f"QLineEdit {{ background: {theme.PLOT}; color: {theme.INK};"
            f" border: 1px solid {theme.RAISED}; border-radius: 8px;"
            " padding: 6px 10px; }"))

    # ------------------------------------------------------------------
    # muoversi
    # ------------------------------------------------------------------
    def _heading_at(self, title: str) -> int | None:
        """La posizione del BLOCCO che è quel titolo, non del testo uguale."""
        block = self._text.document().begin()
        while block.isValid():
            if (block.blockFormat().headingLevel()
                    and block.text().strip() == title):
                return block.position()
            block = block.next()
        return None

    def go_to(self, title: str) -> None:
        where = self._heading_at(title)
        if where is None:
            return
        cursor = QTextCursor(self._text.document())
        cursor.setPosition(where)
        self._text.setTextCursor(cursor)
        # Il titolo in cima alla vista: setTextCursor lo rende visibile, ma
        # di solito all'ultima riga, col capitolo ancora tutto sotto.
        bar = self._text.verticalScrollBar()
        bar.setValue(bar.value() + self._text.cursorRect().top() - 8)

    def _on_index(self, item: QListWidgetItem | None) -> None:
        if item is not None:
            self.go_to(item.data(Qt.ItemDataRole.UserRole))

    def _on_link(self, url: QUrl) -> None:
        if url.scheme() in ("http", "https"):
            QDesktopServices.openUrl(url)
            return
        title = self._by_anchor.get(url.toString().lstrip("#")
                                    or url.fragment())
        if title:
            self.go_to(title)

    def _on_find(self) -> None:
        needle = self._find.text().strip()
        if not needle:
            return
        # Ricerca circolare: arrivati in fondo si ricomincia da capo, che è
        # quello che fa ogni campo "cerca" e quello che ci si aspetta.
        if not self._text.find(needle):
            self._text.moveCursor(QTextCursor.MoveOperation.Start)
            self._text.find(needle)
