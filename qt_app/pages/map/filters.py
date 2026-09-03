"""Il pannello dei filtri della pagina Map: i widget attorno alla regola.

La regola — quali brani passano — sta in `core.viz.filters.filter_tracks`,
uguale per le due app. Qui ci sono la ruota Camelot (il frontend riusato),
le liste spuntabili di macro generi, generi e mood, gli intervalli di BPM
e groove, e un segnale solo: `changed`, quando la domanda "quali brani sto
guardando" cambia.

I generi sono a due livelli — "Electronic - House" — e le due liste sono
collegate: spuntato un macro genere, la lista dei generi mostra SOLO le
sue foglie, le altre spariscono. Un macro spuntato senza foglie spuntate
fa passare tutti i suoi brani; con delle foglie spuntate, solo quelle.
Senza macro spuntati la lista dei generi è completa, come sempre.

Un brano porta fino a quattro generi, in ordine di forza. Il menu «Look
at» dice quanti guardarne: solo il principale, i primi due, i primi tre o
tutti. Vale per i macro e per le foglie insieme — è la stessa lista di
etichette, letta più o meno in profondità. I filtri restringono TUTTO quello che la pagina propone — i punti,
le proposte, la rosa — che è il motivo per cui il pannello è uno.
"""

from __future__ import annotations

import pandas as pd

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QGridLayout,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QPushButton, QVBoxLayout,
                               QWidget)

from core.viz.filters import filter_tracks, span
from core.viz.map_figure import genre_level
from qt_app import theme
from qt_app.widgets.wheel_view import WheelView


# Quanti generi guardare, dall'alto: il testo del menu e la profondità che
# la regola riceve. `None` è tutti.
GENRE_DEPTHS = (("the 1st genre only", 1), ("the top 2", 2),
                ("the top 3", 3), ("all its genres", None))


class CheckList(QWidget):
    """Una lista spuntabile con la ricerca sopra: il multiselect di Qt.

    Le voci arrivano ordinate per frequenza, come nei menu Streamlit: quello
    che si filtra più spesso sta in cima. La ricerca nasconde e basta — le
    spunte restano dove sono, anche fuori vista.
    """

    changed = Signal()

    def __init__(self, label: str, parent=None) -> None:
        super().__init__(parent)
        self._search = QLineEdit()
        self._search.setPlaceholderText(f"filter {label}…")
        self._search.setClearButtonEnabled(True)
        self._list = QListWidget()
        self._list.setUniformItemSizes(True)

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)
        box.addWidget(QLabel(label))
        box.addWidget(self._search)
        box.addWidget(self._list, stretch=1)

        self._search.textChanged.connect(self._on_search)
        self._list.itemChanged.connect(lambda _: self.changed.emit())

    def set_options(self, names: list[str], keep: bool = False) -> None:
        """Le voci nuove. Con `keep` le spunte sopravvivono alle voci che
        restano: serve quando la lista si restringe, non quando cambia
        libreria."""
        kept = set(self.checked()) if keep else set()
        self._list.blockSignals(True)
        self._list.clear()
        for name in names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if name in kept
                               else Qt.CheckState.Unchecked)
            self._list.addItem(item)
        self._list.blockSignals(False)
        self._on_search(self._search.text())

    def checked(self) -> list[str]:
        return [self._list.item(i).text() for i in range(self._list.count())
                if self._list.item(i).checkState() == Qt.CheckState.Checked]

    def clear_checks(self) -> None:
        self._list.blockSignals(True)
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(Qt.CheckState.Unchecked)
        self._list.blockSignals(False)

    def _on_search(self, text: str) -> None:
        wanted = text.casefold()
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setHidden(bool(wanted) and wanted not in item.text().casefold())


def _range_row(decimals: int, step: float) -> tuple[QDoubleSpinBox, QDoubleSpinBox, QWidget]:
    """Due caselle per un intervallo: da–a. Qt non ha uno slider a due
    maniglie, e due numeri scritti si leggono meglio di due maniglie."""
    low, high = QDoubleSpinBox(), QDoubleSpinBox()
    for spin in (low, high):
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
    row = QWidget()
    box = QHBoxLayout(row)
    box.setContentsMargins(0, 0, 0, 0)
    box.addWidget(low)
    box.addWidget(QLabel("–"))
    box.addWidget(high)
    return low, high, row


class FiltersPanel(QWidget):
    """La ruota, le liste e gli intervalli; `kept(frame)` applica la regola."""

    changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._keys: list[str] = []

        # Un gesto sui filtri ridisegna la nuvola intera: mezzo secondo che
        # non va pagato a ogni lettera scritta o casella spuntata di fila.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(350)
        self._debounce.timeout.connect(self.changed.emit)

        # Un quadrato fisso, centrato: l'SVG della ruota si allarga quanto
        # gli si dà e tiene la proporzione — largo quanto la colonna
        # chiederebbe più altezza di quanta ne ha, e usciva tagliato.
        self._wheel = WheelView()
        self._wheel.setFixedSize(300, 306)
        self._wheel.setToolTip(theme.hint(
            "Pick the keys you want to land on. Nothing picked means "
            "every key is welcome."))
        self._wheel.key_toggled.connect(self._on_key)
        wheel_row = QHBoxLayout()
        wheel_row.addStretch(1)
        wheel_row.addWidget(self._wheel)
        wheel_row.addStretch(1)

        self._macros = CheckList("Macro genres")
        self._macros.setToolTip(theme.hint(
            "The first half of the Discogs label — Electronic, Rock, Funk / "
            "Soul. Tick one and the Genres list below shows only what sits "
            "under it; every track of the macro genre passes unless you "
            "narrow it further there."))
        self._macros.changed.connect(self._on_macros)
        self._genres = CheckList("Genres")
        self._all_genres: list[str] = []
        self._moods = CheckList("Moods")
        for picker in (self._genres, self._moods):
            picker.changed.connect(self._debounce.start)
        self._depth = QComboBox()
        for text, _ in GENRE_DEPTHS:
            self._depth.addItem(text)
        self._depth.setCurrentIndex(len(GENRE_DEPTHS) - 1)
        self._depth.setToolTip(theme.hint(
            "A track carries up to four genres, strongest first. This says "
            "how many of them the genre filters look at: the 1st only means "
            "a track passes only if the chosen genre (or macro genre) is "
            "its main one; all its genres means any of them will do — the "
            "old behaviour."))
        self._depth.currentIndexChanged.connect(
            lambda _: self._debounce.start())
        depth_row = QHBoxLayout()
        depth_row.setContentsMargins(0, 0, 0, 0)
        depth_row.addWidget(QLabel("Look at"))
        depth_row.addWidget(self._depth)
        depth_row.addStretch(1)
        # Tre colonne fianco a fianco: macro generi, generi, mood. Sono la
        # stessa domanda posta su tre vocabolari, e in colonna si rubavano
        # l'altezza a vicenda.
        lists_row = QHBoxLayout()
        lists_row.addWidget(self._macros, stretch=1)
        lists_row.addWidget(self._genres, stretch=1)
        lists_row.addWidget(self._moods, stretch=1)

        # I decimali coprono la precisione con cui lo store scrive i numeri
        # (BPM a un decimale, danceability a tre): una casella che
        # arrotondasse taglierebbe fuori i brani sul bordo — a corsa tutta
        # aperta ne sparivano due su 87mila, che è il modo subdolo di
        # sbagliare.
        self._bpm_low, self._bpm_high, bpm_row = _range_row(1, 1.0)
        self._gr_low, self._gr_high, gr_row = _range_row(3, 0.01)
        for spin in (self._bpm_low, self._bpm_high, self._gr_low, self._gr_high):
            spin.valueChanged.connect(lambda _: self._debounce.start())

        reset = QPushButton("↺ Reset the filters")
        reset.clicked.connect(self._on_reset)

        self._count = QLabel("")
        self._count.setObjectName("dim")
        self._count.setWordWrap(True)
        self._count.setToolTip(theme.hint(
            "Filters narrow the map, the suggestions and the roster. "
            "Nothing picked means everything passes. A track carrying ANY "
            "of the chosen genres (or moods) stays: tracks are multi-label "
            "on purpose."))

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(QLabel("BPM"), 0, 0)
        grid.addWidget(bpm_row, 0, 1)
        grid.addWidget(QLabel("Groove"), 1, 0)
        grid.addWidget(gr_row, 1, 1)

        box = QVBoxLayout(self)
        box.addLayout(wheel_row)
        box.addLayout(depth_row)
        box.addLayout(lists_row, stretch=1)
        box.addLayout(grid)
        box.addWidget(reset)
        box.addWidget(self._count)

    # --- la libreria detta le opzioni e le corse ---
    def set_frame(self, frame: pd.DataFrame) -> None:
        genre_counts = pd.Series(
            [g for tags in frame["genre_list"] for g in tags if g]
        ).value_counts()
        mood_counts = pd.Series(
            [m for tags in frame["mood_list"] for m in tags if m]
        ).value_counts()
        self._all_genres = list(genre_counts.index)
        macro_counts = pd.Series(
            [genre_level(g, "parent") for g in self._all_genres]
        ).value_counts() if self._all_genres else pd.Series(dtype=int)
        self._macros.set_options(list(macro_counts.index))
        self._genres.set_options(self._all_genres)
        self._moods.set_options(list(mood_counts.index))

        tempo = span(frame, "bpm", 60.0, 200.0)
        swing = span(frame, "danceability", 0.0, 1.0)
        for spin, (low, high), at in (
                (self._bpm_low, tempo, tempo[0]),
                (self._bpm_high, tempo, tempo[1]),
                (self._gr_low, swing, swing[0]),
                (self._gr_high, swing, swing[1])):
            spin.blockSignals(True)
            spin.setRange(low, high)
            spin.setValue(at)
            spin.blockSignals(False)

        self._keys = []
        self._wheel.set_keys(self._keys)

    def _under_macros(self) -> list[str]:
        """Le foglie dei macro generi spuntati — tutte, senza macro."""
        macros = set(self._macros.checked())
        if not macros:
            return list(self._all_genres)
        return [g for g in self._all_genres
                if genre_level(g, "parent") in macros]

    def genres_wanted(self) -> list[str]:
        """I generi che la regola riceve: le foglie spuntate, o tutte quelle
        dei macro spuntati, o niente (cioè tutti)."""
        ticked = self._genres.checked()
        if ticked:
            return ticked
        return self._under_macros() if self._macros.checked() else []

    def genre_depth(self) -> int | None:
        return GENRE_DEPTHS[self._depth.currentIndex()][1]

    # --- la regola, applicata ---
    def kept(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = filter_tracks(
            frame, self.genres_wanted(), self._moods.checked(), self._keys,
            (self._bpm_low.value(), self._bpm_high.value()),
            (self._gr_low.value(), self._gr_high.value()),
            genre_depth=self.genre_depth())
        self._count.setText(
            f"{len(out):,} of {len(frame):,} tracks pass · ⓘ")
        return out

    # --- i gesti ---
    def _on_macros(self) -> None:
        # La lista dei generi si restringe alle foglie dei macro spuntati;
        # le spunte sulle foglie che restano sopravvivono, le altre cadono
        # con la voce.
        self._genres.set_options(self._under_macros(), keep=True)
        self._debounce.start()

    def _on_key(self, code: str) -> None:
        self._keys = ([k for k in self._keys if k != code]
                      if code in self._keys else self._keys + [code])
        self._wheel.set_keys(self._keys)
        self._debounce.start()

    def _on_reset(self) -> None:
        self._keys = []
        self._wheel.set_keys(self._keys)
        self._macros.clear_checks()
        self._genres.set_options(self._all_genres)
        self._depth.blockSignals(True)
        self._depth.setCurrentIndex(len(GENRE_DEPTHS) - 1)
        self._depth.blockSignals(False)
        self._moods.clear_checks()
        for spin, edge in ((self._bpm_low, self._bpm_low.minimum()),
                           (self._bpm_high, self._bpm_high.maximum()),
                           (self._gr_low, self._gr_low.minimum()),
                           (self._gr_high, self._gr_high.maximum())):
            spin.blockSignals(True)
            spin.setValue(edge)
            spin.blockSignals(False)
        self._debounce.start()
