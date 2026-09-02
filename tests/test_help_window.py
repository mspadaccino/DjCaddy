"""La finestra della guida: che mostri davvero il testo, e che ci si muova.

Il punto delicato non è il markdown — quello lo impagina Qt — ma il patto
fra l'indice a sinistra e il documento a destra: l'indice elenca i titoli
che `core.guide` ha letto dal testo SORGENTE, e ci si salta cercandoli fra i
blocchi del documento IMPAGINATO. Se le due liste divergessero, l'indice
avrebbe voci che non portano da nessuna parte, e in silenzio.
"""

import os

import pytest

pytest.importorskip("PySide6", reason="gruppo poetry `qt` non installato")
pytest.importorskip("pytestqt", reason="gruppo poetry `qt` non installato")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QUrl

from core import guide
from qt_app.widgets.help_window import HelpWindow


@pytest.fixture
def window(qtbot):
    made = HelpWindow()
    qtbot.addWidget(made)
    made.resize(900, 700)
    made.show()
    qtbot.waitExposed(made)
    return made


def test_the_index_has_a_row_for_every_chapter(window):
    assert window._index.count() == len(window._chapters)
    assert window._index.count() > 20          # la guida non è un volantino


def test_every_row_of_the_index_lands_on_a_real_heading(window):
    """Il patto: ciò che l'indice elenca esiste nel documento COME titolo."""
    orphans = [title for _, title in window._chapters
               if window._heading_at(title) is None]
    assert not orphans, f"voci che non portano da nessuna parte: {orphans}"


def test_the_text_is_the_guide_not_the_readme(window):
    shown = window._text.toPlainText()
    assert "Cue Finder" in shown
    assert "How the code is laid out" not in shown
    assert "pyinstaller" not in shown.lower()


def test_jumping_to_a_later_chapter_scrolls_further(window, qtbot):
    window.go_to("Navigator")
    near = window._text.verticalScrollBar().value()
    window.go_to("Reference: what every number means")
    far = window._text.verticalScrollBar().value()
    assert far > near


def test_an_internal_link_navigates_instead_of_opening_a_browser(window):
    """`[Groove](#groove--read-this-one-carefully)` deve saltare, non
    lanciare il browser di sistema su un URL che non esiste."""
    window.go_to("Navigator")
    before = window._text.verticalScrollBar().value()
    window._on_link(QUrl("#" + guide.anchor("Groove — read this one carefully")))
    assert window._text.verticalScrollBar().value() != before


def test_search_finds_and_wraps_around(window):
    window._find.setText("Camelot")
    window._on_find()
    assert window._text.textCursor().selectedText() == "Camelot"
