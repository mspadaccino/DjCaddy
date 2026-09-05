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

In testa ci sono i preset (`core.analysis.presets`): un nome per tutto
quello che il pannello imposta, più quello che la pagina gli affida con
`bind_extras` — i tre pesi del costo, che stanno fuori dal pannello ma
fanno parte della stessa domanda. Scegliere un preset rimette tutto.
"""

from __future__ import annotations

import pandas as pd

from PySide6.QtCore import Qt, QTimer, Signal
from collections.abc import Callable

from PySide6.QtWidgets import (QComboBox, QGridLayout, QHBoxLayout,
                               QInputDialog, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton,
                               QVBoxLayout, QWidget)

from core.analysis.presets import Presets
from core.viz.filters import filter_tracks, span
from core.viz.map_figure import genre_level
from qt_app import theme
from qt_app.widgets.range_slider import RangeSlider
from qt_app.widgets.wheel_view import WheelView


# Quanti generi guardare, dall'alto: il testo del menu e la profondità che
# la regola riceve. `None` è tutti.
GENRE_DEPTHS = (("the 1st genre only", 1), ("the top 2", 2),
                ("the top 3", 3), ("all its genres", None))

# La voce del menu dei preset quando nessuno è scelto.
NO_PRESET = "— presets —"


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
        self.set_checked([])

    def set_checked(self, names: list[str]) -> None:
        """Spuntate queste e nessun'altra, in silenzio."""
        wanted = set(names)
        self._list.blockSignals(True)
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setCheckState(Qt.CheckState.Checked if item.text() in wanted
                               else Qt.CheckState.Unchecked)
        self._list.blockSignals(False)

    def _on_search(self, text: str) -> None:
        wanted = text.casefold()
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setHidden(bool(wanted) and wanted not in item.text().casefold())


class FiltersPanel(QWidget):
    """La ruota, le liste e gli intervalli; `kept(frame)` applica la regola."""

    changed = Signal()

    def __init__(self, parent=None, presets: Presets | None = None) -> None:
        super().__init__(parent)
        self._keys: list[str] = []
        self._presets = presets or Presets()
        # Quello che la pagina affida al preset oltre ai filtri: chi lo
        # legge e chi lo rimette. Senza `bind_extras` il preset è filtri
        # e basta.
        self._extras_get: Callable[[], dict] = dict
        self._extras_set: Callable[[dict], None] = lambda _: None

        # Un gesto sui filtri ridisegna la nuvola intera: mezzo secondo che
        # non va pagato a ogni lettera scritta o casella spuntata di fila.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(350)
        self._debounce.timeout.connect(self.changed.emit)

        # I preset: il menu li applica, «Save» ne fa uno da com'è ora.
        self._preset = QComboBox()
        self._preset.setToolTip(theme.hint(
            "A saved way of looking at the library: every filter here — "
            "keys, genres, moods, the ranges — plus the three weights of "
            "the transition cost. Pick one and everything is set at once; "
            "touch anything afterwards and you are simply off the preset."))
        self._preset.currentTextChanged.connect(self._on_preset_pick)
        self._preset_save = QPushButton("💾 Save preset…")
        self._preset_save.setToolTip("Saves the filters and the weights as "
                                     "they are now, under a name of yours — "
                                     "house_intro, funky_climax…")
        self._preset_save.clicked.connect(self._on_preset_save)
        self._preset_delete = QPushButton("✕")
        self._preset_delete.setToolTip("Deletes the chosen preset.")
        self._preset_delete.clicked.connect(self._on_preset_delete)
        preset_row = QHBoxLayout()
        preset_row.setContentsMargins(0, 0, 0, 0)
        preset_row.addWidget(self._preset, stretch=1)
        preset_row.addWidget(self._preset_save)
        preset_row.addWidget(self._preset_delete)
        self._list_presets()

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
        # (BPM a un decimale, danceability a tre): una maniglia che
        # arrotondasse di più taglierebbe fuori i brani sul bordo — a corsa
        # tutta aperta ne sparivano due su 87mila, che è il modo subdolo di
        # sbagliare.
        self._bpm = RangeSlider(decimals=1)
        self._groove = RangeSlider(decimals=3)
        # Energia e mood sono RANGHI sulla libreria: 0..1, e «0.25» vuol
        # dire "il quarto più calmo che possiedi", non un numero assoluto.
        self._energy = RangeSlider(decimals=2)
        self._energy.setToolTip(theme.hint(
            "How hard the track pushes, as a rank across your library: 0 "
            "is the calmest tenth you own, 1 the hardest. An intro lives "
            "low, a climax high."))
        self._valence = RangeSlider(decimals=2)
        self._valence.setToolTip(theme.hint(
            "How BRIGHT the track reads, as a rank across your library: 0 "
            "its darkest tenth — Dark, Deep, Heavy — and 1 its brightest — "
            "Happy, Party, Summer."))
        for slider in (self._bpm, self._groove, self._energy, self._valence):
            slider.valuesChanged.connect(lambda *_: self._debounce.start())

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
        grid.addWidget(self._bpm, 0, 1)
        grid.addWidget(QLabel("Groove"), 1, 0)
        grid.addWidget(self._groove, 1, 1)
        grid.addWidget(QLabel("Energy"), 2, 0)
        grid.addWidget(self._energy, 2, 1)
        grid.addWidget(QLabel("Mood"), 3, 0)
        grid.addWidget(self._valence, 3, 1)

        box = QVBoxLayout(self)
        box.addLayout(preset_row)
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

        self._bpm.set_span(*span(frame, "bpm", 60.0, 200.0))
        self._groove.set_span(*span(frame, "danceability", 0.0, 1.0))
        # Una mappa fatta prima dei ranghi non ha le colonne: lo slider
        # resta, spento, e la regola non li guarda.
        for slider, column in ((self._energy, "energy"),
                               (self._valence, "valence_rank")):
            slider.setEnabled(column in frame)
            slider.set_span(*(span(frame, column, 0.0, 1.0)
                              if column in frame else (0.0, 1.0)))

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
            self._bpm.values(), self._groove.values(),
            genre_depth=self.genre_depth(),
            energy=self._energy.values() if self._energy.isEnabled() else None,
            valence=(self._valence.values() if self._valence.isEnabled()
                     else None))
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
        for slider in (self._bpm, self._groove, self._energy, self._valence):
            slider.reset()
        self._debounce.start()

    # --- lo stato, per i preset ---
    def bind_extras(self, get: Callable[[], dict],
                    set_: Callable[[dict], None]) -> None:
        """Quello che la pagina vuole nel preset oltre ai filtri (i pesi
        del costo): `get` lo legge quando si salva, `set_` lo rimette
        quando si applica."""
        self._extras_get, self._extras_set = get, set_

    def state(self) -> dict:
        return {"keys": list(self._keys),
                "macros": self._macros.checked(),
                "genres": self._genres.checked(),
                "moods": self._moods.checked(),
                "depth": self.genre_depth(),
                "bpm": list(self._bpm.values()),
                "groove": list(self._groove.values()),
                "energy": list(self._energy.values()),
                "valence": list(self._valence.values())}

    def restore(self, saved: dict) -> None:
        """Tutto com'era, in un giro solo e con un solo `changed`. I macro
        prima delle foglie, perché sono i macro a decidere quali foglie
        esistono; un intervallo fuori dalla corsa di questa libreria si
        stringe ai suoi bordi."""
        self._keys = [k for k in saved.get("keys", [])]
        self._wheel.set_keys(self._keys)
        self._macros.set_checked(saved.get("macros", []))
        self._genres.set_options(self._under_macros())
        self._genres.set_checked(saved.get("genres", []))
        self._moods.set_checked(saved.get("moods", []))
        depths = [d for _, d in GENRE_DEPTHS]
        depth = saved.get("depth")
        self._depth.blockSignals(True)
        self._depth.setCurrentIndex(depths.index(depth) if depth in depths
                                    else len(GENRE_DEPTHS) - 1)
        self._depth.blockSignals(False)
        for slider, name in ((self._bpm, "bpm"), (self._groove, "groove"),
                             (self._energy, "energy"),
                             (self._valence, "valence")):
            values = saved.get(name)
            if values:
                slider.set_values(*values)
            else:
                slider.reset()
        self._debounce.start()

    def _list_presets(self) -> None:
        self._preset.blockSignals(True)
        self._preset.clear()
        self._preset.addItem(NO_PRESET)
        self._preset.addItems(self._presets.names())
        self._preset.blockSignals(False)
        self._preset_delete.setEnabled(False)

    def _on_preset_pick(self, name: str) -> None:
        self._preset_delete.setEnabled(name != NO_PRESET)
        saved = self._presets.read(name) if name != NO_PRESET else None
        if saved is None:
            return
        self.restore(saved)
        self._extras_set(saved)

    def _on_preset_save(self) -> None:
        current = self._preset.currentText()
        name, ok = QInputDialog.getText(
            self, "Save the preset", "Name:",
            text=current if current != NO_PRESET else "")
        name = name.strip()
        if not ok or not name or name == NO_PRESET:
            return
        if name in self._presets.names() and name != current:
            answer = QMessageBox.question(
                self, "Save the preset", f"Overwrite «{name}»?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes)
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._presets.write(name, {**self.state(), **self._extras_get()})
        self._list_presets()
        self._preset.blockSignals(True)
        self._preset.setCurrentText(name)
        self._preset.blockSignals(False)
        self._preset_delete.setEnabled(True)

    def _on_preset_delete(self) -> None:
        name = self._preset.currentText()
        if name == NO_PRESET:
            return
        answer = QMessageBox.question(
            self, "Delete the preset", f"Delete «{name}»?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._presets.delete(name)
        self._list_presets()
