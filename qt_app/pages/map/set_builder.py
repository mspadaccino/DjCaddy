"""Build a set: i tre modi di passare dalla mappa a una scaletta.

Quick List (cosa ci si mixa sopra), Chain Maker (un brano alla volta) e
Radio Mix (una playlist da un GRUPPO: i preferiti o il lazo), sopra un pannello
unico di pesi e "quanti elencare", che sono gli stessi filtri di partenza
per tutte. C'era una quarta scheda, Sounds like it, che rispondeva "cosa
gli somiglia" sui 1280 numeri dell'embedding: da quando il termine sound
del costo misura lì e non più sulla mappa, è Quick List coi pesi 1, 0, 0,
e una scheda che si ottiene girando tre manopole non serve.

Le regole vengono tutte da core: `nearest`, `magic_sort`, `sorted_after`,
`suggestions`, `chain_table`, `roster_table`, `radio.tune`.
Qui ci sono i widget e la disciplina delle liste: una lista si apre quando
la si chiede ("Make the list"), resta viva finché il seme è quello — si
ricalcola con i pesi e con i filtri — e si richiude da sé quando il seme
cambia.

Ogni scelta fatta qui — il brano preso dalla rosa, quello tolto dalla
catena, la lista mandata avanti — finisce nel `Journal`: non cambia niente
oggi, ma è la materia con cui domani i pesi si regoleranno da soli.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel,
                               QSlider,
                               QLineEdit, QListWidget, QListWidgetItem,
                               QPushButton, QSpinBox, QStackedWidget,
                               QTabWidget, QVBoxLayout, QWidget)

from core.analysis import arc, journey, mood_scale, radio
from core.analysis.duplicates import folded, normalized_name, song_key
from core.analysis.graph_playlist import GraphPlaylist, auto_chain, suggestions
from core.analysis.journal import Journal, facts
from core.analysis.mixing import magic_sort, nearest, sorted_after
from core.viz.board import _label, chain_table, roster_table
from core.viz.track_columns import READING_ORDER, genre_colors, reading
from qt_app import theme
from qt_app.state import AppState
from qt_app.widgets.track_table import TrackTable

from .library import Library
from .playlist_panel import double_marks

# Quanti candidati proporre, e a passi di quanto: gli stessi numeri della
# pagina Streamlit, per le stesse ragioni (una lista più lunga di cento non
# è più una rosa, è la libreria).
SUGGESTION_DEFAULT = 20
SUGGESTION_MAX = 100
SUGGESTION_STEP = 5

# Quanti candidati a ogni passo della catena.
FRONTIER_SIZE = 9

# Oltre questi risultati la ricerca per nome chiede una parola in più.
SEARCH_MAX = 200

# Lo stesso avviso della Playlist, sulle liste che qui non hanno un
# «Remove ticked»: dice cosa vestono le tinte, non cosa fare — quello resta
# a chi guarda (di solito, lasciare la spunta spenta prima di mandare la
# lista avanti).
DOUBLES_HINT = theme.hint(
    "Orange rows: the same song is already elsewhere in this list — same "
    "name once numbering and (mix) notes are stripped. Violet rows: they "
    "sound nearly identical to another row here, whatever the names. Only "
    "the extra copies are tinted, the first occurrence stays clean. Still "
    "a question, not a verdict — hover a tinted row for its partner and "
    "listen before deciding.")

WAITING_FOR_THE_BUTTON = ("Nothing built yet — press the button above. The "
                          "list does not open by itself: most clicks on the "
                          "map are looking around, not choosing what comes "
                          "next.")

# I titoli delle schede, senza conteggio: il conteggio ce lo appende
# `_retitle` quando c'è qualcosa da contare.
TAB_TITLES = ("✨ Quick List", "🔗 Chain Maker", "🧭 Journey", "📻 Radio Mix")
TAB_QUICK, TAB_CHAIN, TAB_JOURNEY, TAB_RADIO = range(len(TAB_TITLES))

# Quanti brani chiede un Journey, di default e al massimo. Sotto i tre non
# c'è niente in mezzo da cercare.
JOURNEY_DEFAULT = 12
JOURNEY_MIN, JOURNEY_MAX = 3, 60

# Quanti anelli aggiunge l'Auto chain in un colpo, e al massimo.
AUTO_STEPS_DEFAULT = 8
AUTO_STEPS_MAX = 50

# Cosa chiede ogni scheda, e come risponde — nello stesso ordine.
TAB_HINTS = (
    "<b>What mixes out of this one?</b><br>One seed, ranked against the "
    "whole library by the transition cost — sound, tempo and key with the "
    "weights above. A list of options, judged one by one: they may well "
    "sound alike. With a group selected instead, this tab holds the group "
    "and magic sort.",
    "<b>What comes next?</b><br>One track at a time: a roster of nine "
    "mixes out of the last one, you take one, the roster is made again. "
    "Trend looks a step ahead; Auto chain takes the top of the roster for "
    "you, N times. The order you build is the order that comes out.",
    "<b>How do I get from here to there?</b><br>A start, an optional "
    "track to land on, and how many tracks in between: the cheapest run "
    "of transitions through the library that joins the two — same cost, "
    "same weights — while the energy follows the arc of a set, Intro to "
    "Release, as much as the Arc knob asks. A draft in order, not a "
    "ranking: send it on, then reorder by hand.",
    "<b>More like these.</b><br>A playlist from a group — your favourites "
    "or the map selection — not from one seed. Sound only, against the "
    "group's taste; each pick is penalised for resembling the ones before, "
    "so it covers the group without repeating. Magic-sorted at the end.",
)

# Da dove parte il Journey: la scelta del menu, nell'ordine del menu.
JOURNEY_SOURCES = ("The seed", "Last of the chain", "Last of the playlist")

# Da dove parte la Radio: la scelta del menu, nell'ordine del menu.
RADIO_SOURCES = ("Favourites", "Map selection", "Playlist")
RADIO_FAVOURITES, RADIO_MAP, RADIO_PLAYLIST = range(len(RADIO_SOURCES))


def _dim(text: str, wrap: bool = True) -> QLabel:
    label = QLabel(text)
    label.setObjectName("dim")
    label.setWordWrap(wrap)
    return label


def _knob(into: QHBoxLayout, name: str, value: float, low: str, high: str,
          why: str) -> QDoubleSpinBox:
    """Una manopola 0..1 con la sua etichetta. Il tooltip — sull'una e
    sull'altra — apre con cosa vale lo zero e cosa vale l'uno, poi spiega."""
    told = theme.hint(f"<b>0 = {low} · 1 = {high}</b><br>{why}")
    label = QLabel(name)
    label.setToolTip(told)
    into.addWidget(label)
    spin = QDoubleSpinBox()
    spin.setRange(0.0, 1.0)
    spin.setSingleStep(0.1)
    spin.setValue(value)
    spin.setToolTip(told)
    into.addWidget(spin)
    return spin


def numbered_rows(frame: pd.DataFrame, indices, common: dict) -> pd.DataFrame:
    """Le righe scelte con il numero d'ordine davanti, come `selection_rows`
    di Streamlit: l'ordine è quello che arriva e non si tocca."""
    listed = [{"#": n + 1, **reading(frame.loc[i], common),
               "_path": frame.at[i, "path"]}
              for n, i in enumerate(indices)]
    return pd.DataFrame(listed, columns=["#", *READING_ORDER, "_path"])


class SearchPicker(QWidget):
    """Cerca un brano per nome dentro un insieme, e dillo a chi ascolta.

    Sta al posto dei selectbox di Streamlit: su una libreria vera il menu
    dei nomi non si apre più in fretta, quindi la ricerca non è un ripiego,
    è la via normale. `picked` porta l'INDICE di libreria del brano scelto
    (doppio clic, o Invio sul primo risultato).
    """

    picked = Signal(int)

    def __init__(self, placeholder: str, parent=None) -> None:
        super().__init__(parent)
        self._frame: pd.DataFrame | None = None
        self._options: list[int] = []
        self._search = QLineEdit()
        self._search.setPlaceholderText(placeholder)
        self._search.setClearButtonEnabled(True)
        self._list = QListWidget()
        self._list.setUniformItemSizes(True)
        self._list.setMaximumHeight(140)
        # La lista compare solo quando ha risultati: da ferma è un riquadro
        # vuoto che ruba altezza alle tabelle — visto nel parallel run, nel
        # Chain Maker si mangiava lo spazio dei candidati.
        self._list.setVisible(False)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)
        box.addWidget(self._search)
        box.addWidget(self._list)
        self._search.textChanged.connect(self._refresh)
        self._search.returnPressed.connect(self._first)
        self._list.itemActivated.connect(
            lambda item: self.picked.emit(item.data(Qt.ItemDataRole.UserRole)))

    def set_universe(self, frame: pd.DataFrame, options: list[int]) -> None:
        self._frame, self._options = frame, list(options)
        self._refresh(self._search.text())

    def _refresh(self, text: str) -> None:
        self._list.clear()
        wanted = folded(text.strip())
        if self._frame is not None and wanted:
            shown = 0
            for i in self._options:
                if wanted in folded(str(self._frame.at[i, "name"])):
                    item = QListWidgetItem(str(self._frame.at[i, "name"]))
                    item.setData(Qt.ItemDataRole.UserRole, int(i))
                    self._list.addItem(item)
                    shown += 1
                    if shown >= SEARCH_MAX:
                        break
        self._list.setVisible(self._list.count() > 0)

    def _first(self) -> None:
        if self._list.count():
            self.picked.emit(self._list.item(0).data(Qt.ItemDataRole.UserRole))

    def clear(self) -> None:
        self._search.clear()


class WeightSlider(QWidget):
    """Un peso 0..2 a passi di un decimo: lo slider, col nome davanti e il
    numero dietro. Uno slider e non una casella perché il peso si ASSAGGIA
    — si trascina guardando la lista cambiare — e la casella chiedeva un
    clic per ogni decimo."""

    valueChanged = Signal(float)

    def __init__(self, name: str, why: str, parent=None) -> None:
        super().__init__(parent)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 20)
        self._slider.setValue(10)
        self._slider.setMinimumWidth(90)
        self._told = QLabel("1.0")
        self._told.setFixedWidth(24)
        self.setToolTip(why)
        self._slider.valueChanged.connect(self._on_moved)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(QLabel(f"w·{name}"))
        row.addWidget(self._slider)
        row.addWidget(self._told)

    def value(self) -> float:
        return self._slider.value() / 10

    def setValue(self, value: float) -> None:
        self._slider.setValue(round(value * 10))

    def _on_moved(self, raw: int) -> None:
        self._told.setText(f"{raw / 10:.1f}")
        self.valueChanged.emit(raw / 10)


class SetBuilderPanel(QWidget):
    """Il pannello: pesi e conto sopra, le tre schede sotto.

    Parla col resto della pagina a segnali: `append_playlist` e
    `replace_playlist` portano INDICI di libreria; `suggestions_changed` e
    `chain_changed` dicono cosa cerchiare sulla mappa (la Quick List, la
    catena). La scelta corrente — seme o gruppo, con la precedenza alla
    spunta in playlist — arriva da fuori con `set_choice`: la regola di chi
    comanda sta nella pagina, qui si lavora su quello che comanda.
    """

    append_playlist = Signal(list)

    # I tre pesi sono cambiati: la playlist rilegge i suoi costi.

    weights_changed = Signal()
    replace_playlist = Signal(list)
    suggestions_changed = Signal(list)
    chain_changed = Signal(list)

    def __init__(self, state: AppState, wire_table,
                 journal: Journal | None = None, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._wire = wire_table
        self._journal = journal or Journal()
        self._lib: Library | None = None
        self._pool: np.ndarray = np.empty(0, dtype=int)
        self._seed: int | None = None
        self._selected: list[int] = []
        self._group_shown: list[int] | None = None
        self._candidates: list[int] = []
        self._asked_mixes: str | None = None
        self._graph = GraphPlaylist()
        self._source: str | None = None
        self._roster_picks: list = []
        # La radio: da cosa è stata sintonizzata (sorgente e semi), cosa ha
        # proposto, e i no raccolti finché i semi sono quelli.
        self._radio_key: tuple | None = None
        self._radio_shown: list[int] = []
        self._radio_negatives: list[int] = []
        # Il Journey: l'arrivo scelto per nome, da cosa è stato pianificato
        # e cosa ha proposto. Le quattro misure dell'arco sulla libreria si
        # fanno una volta per libreria, alla prima richiesta.
        self._journey_end: int | None = None
        self._journey_key: tuple | None = None
        self._journey_shown: list[int] = []
        self._arc_values: np.ndarray | None = None
        self._build()
        state.favourites_changed.connect(lambda _: self._refresh_radio())
        # La playlist è una delle partenze del Journey e una delle sorgenti
        # della Radio: quando cambia, tutti e due lo devono sapere.
        state.playlist_changed.connect(lambda _: self._refresh_journey())
        state.playlist_changed.connect(lambda _: self._refresh_radio())

    # ------------------------------------------------------------------
    # costruzione
    # ------------------------------------------------------------------
    def _build(self) -> None:
        self._seed_told = _dim("")

        weights = QHBoxLayout()
        self._w_sound = self._weight(weights, "sound",
                                     "How much the acoustic distance counts "
                                     "— cosine in the 1280 dimensions of "
                                     "the embedding, not on the flattened "
                                     "map. Alone, with BPM and key at 0, "
                                     "the list is «what sounds like it».")
        self._w_bpm = self._weight(weights, "BPM",
                                   "How much the tempo gap counts. Beyond "
                                   "±6% the cost climbs fast.")
        self._w_key = self._weight(weights, "key",
                                   "How much harmonic distance counts. "
                                   "Adjacent or relative keys cost nothing.")
        weights.addSpacing(12)
        weights.addWidget(QLabel("List"))
        self._count = QSpinBox()
        self._count.setRange(SUGGESTION_STEP, SUGGESTION_MAX)
        self._count.setSingleStep(SUGGESTION_STEP)
        self._count.setValue(SUGGESTION_DEFAULT)
        self._count.setToolTip("How many to list — Quick List and "
                               "Radio Mix.")
        self._count.valueChanged.connect(lambda _: self._on_knobs())
        weights.addWidget(self._count)
        weights.addStretch(1)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_quicklist(), TAB_TITLES[TAB_QUICK])
        self._tabs.addTab(self._build_chain(), TAB_TITLES[TAB_CHAIN])
        self._tabs.addTab(self._build_journey(), TAB_TITLES[TAB_JOURNEY])
        self._tabs.addTab(self._build_radio(), TAB_TITLES[TAB_RADIO])
        # La domanda di ogni scheda, sulla linguetta: si legge prima di
        # aprirla, che è quando serve sapere quale aprire.
        for tab, told in enumerate(TAB_HINTS):
            self._tabs.setTabToolTip(tab, theme.hint(told))

        box = QVBoxLayout(self)
        box.addWidget(self._seed_told)
        box.addLayout(weights)
        box.addWidget(self._tabs, stretch=1)

    def _weight(self, into: QHBoxLayout, name: str, why: str) -> WeightSlider:
        slider = WeightSlider(name, why)
        slider.valueChanged.connect(lambda _: self._on_knobs())
        into.addWidget(slider)
        return slider

    def _pick_row(self, table: TrackTable, reset=None) -> QWidget:
        """La riga che governa la lista: scelta in blocco e, dove serve, il
        ritorno alla schermata che la crea.

        Le liste qui arrivano a venti righe e più: prenderle tutte, o
        ripulire per ricominciare, non è un gesto da fare riga per riga.
        Riga a sé e non in fondo insieme agli altri bottoni, perché quelli
        AGISCONO sulla lista (aggiungi, manda alla playlist) mentre questi
        la governano — e perché cinque bottoni in fila non ci stanno nella
        colonna di destra.
        """
        pick_all = QPushButton("Select all")
        pick_all.clicked.connect(lambda: table.set_all_picked(True))
        pick_none = QPushButton("Select none")
        pick_none.clicked.connect(lambda: table.set_all_picked(False))
        row = QWidget()
        box = QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(pick_all)
        box.addWidget(pick_none)
        box.addStretch(1)
        if reset is not None:
            # In fondo a destra, lontano dai due che scelgono: è il gesto
            # che butta via la lista, non uno che la tocca.
            back = QPushButton("✖ Reset")
            back.setToolTip(theme.hint(
                "Back to the button that makes the list: this one is "
                "dropped, ticks included, and the rings come off the map. "
                "The settings above — the weights and how many to list — "
                "stay as they are, ready for the next run."))
            back.clicked.connect(reset)
            box.addWidget(back)
        return row

    def _build_quicklist(self) -> QWidget:
        self._quick = QStackedWidget()

        idle = QWidget()
        QVBoxLayout(idle).addWidget(_dim(
            "Nothing selected yet. Click a point on the map to make it the "
            "seed, or drag the lasso or the box around a group."))

        # Il gruppo: la tabella di quello che si è preso, e magic sort.
        group = QWidget()
        gbox = QVBoxLayout(group)
        self._group_told = QLabel("")
        gbox.addWidget(self._group_told)
        self._group_table = TrackTable(checkable=True)
        self._wire(self._group_table)
        # I bottoni seguono le spunte: se ne togli una, lavorano su meno.
        self._group_table.selection_paths_changed.connect(
            lambda _: self._refresh_group_buttons())
        gbox.addWidget(self._pick_row(self._group_table))
        gbox.addWidget(self._group_table, stretch=1)
        row = QHBoxLayout()
        self._sort_append = QPushButton("✨ Magic sort and append")
        # Il come e il perché stanno sul bottone che li fa: scritti sotto
        # il titolo erano tre righe tolte alla tabella.
        self._sort_append.setToolTip(theme.hint(
            "Magic sort walks all of them once, in the order that keeps "
            "every transition cheap — the travelling-salesman path over "
            "the cost. Sorted among themselves, then added after what the "
            "playlist already holds."))
        self._sort_append.clicked.connect(self._on_sort_append)
        self._plain_append = QPushButton("➕ Append them, unsorted")
        self._plain_append.clicked.connect(self._on_plain_append)
        self._sort_new = QPushButton("↺ Sort as a new playlist")
        self._sort_new.setToolTip("Starts over: what is in the playlist now "
                                  "is dropped.")
        self._sort_new.clicked.connect(self._on_sort_new)
        clear = QPushButton("✖ Clear the selection")
        clear.clicked.connect(self._state.clear_selection)
        for button in (self._sort_append, self._plain_append,
                       self._sort_new, clear):
            row.addWidget(button)
        gbox.addLayout(row)

        # Il seme: la Quick List vera e propria.
        seed = QWidget()
        sbox = QVBoxLayout(seed)
        mixes_why = theme.hint(
            "Ranked by the transition cost — sound, tempo and key "
            "together, with the weights above. Sound is measured in the "
            "1280 dimensions of the embedding: with BPM and key at 0 this "
            "is «what sounds like it», tempo and key aside. Only tracks "
            "that pass the filters are considered. The first row is the "
            "seed itself. A ranking, not a set: the twenty are judged one "
            "by one against the seed, and may well sound alike — Radio Mix "
            "is the one that builds a list where they do not.")
        self._mixes_ask = QPushButton("✨ Make the list")
        self._mixes_ask.setToolTip(mixes_why)
        self._mixes_ask.clicked.connect(self._on_ask_mixes)
        sbox.addWidget(self._mixes_ask)
        self._mixes_wait = _dim(WAITING_FOR_THE_BUTTON)
        sbox.addWidget(self._mixes_wait)
        self._mixes_table = TrackTable(checkable=True, favouritable=True)
        self._mixes_table.setToolTip(mixes_why)
        self._wire(self._mixes_table)
        sbox.addWidget(self._pick_row(self._mixes_table,
                                      reset=self._on_reset_mixes))
        sbox.addWidget(self._mixes_table, stretch=1)
        self._mixes_doubles = _dim("")
        self._mixes_doubles.setToolTip(DOUBLES_HINT)
        self._mixes_doubles.setVisible(False)
        sbox.addWidget(self._mixes_doubles)
        self._mixes_add = QPushButton("➕ Add selected to the playlist")
        self._mixes_add.clicked.connect(
            lambda: self._add_rows(self._mixes_table))
        self._mixes_send = QPushButton("↺ Send as a new playlist")
        self._mixes_send.setToolTip("Starts over: what is in the playlist "
                                    "now is dropped.")
        self._mixes_send.clicked.connect(
            lambda: self._send_rows(self._mixes_table))
        mixes_row = QHBoxLayout()
        mixes_row.addWidget(self._mixes_add)
        mixes_row.addWidget(self._mixes_send)
        sbox.addLayout(mixes_row)

        for page in (idle, group, seed):
            self._quick.addWidget(page)
        return self._quick

    def _build_chain(self) -> QWidget:
        self._chain = QStackedWidget()

        # Da fermo: si comincia dalla scelta sulla mappa, o per nome.
        start = QWidget()
        tbox = QVBoxLayout(start)
        tbox.addWidget(QLabel("<b>Start the chain with a track.</b>"))
        tbox.addWidget(_dim("Everything else grows off it, one suggestion "
                            "at a time."))
        self._start_told = QLabel("")
        self._start_told.setWordWrap(True)
        tbox.addWidget(self._start_told)
        self._start_from = QPushButton("▶ Start from the selection")
        self._start_from.clicked.connect(self._on_start_from_choice)
        tbox.addWidget(self._start_from)
        tbox.addWidget(_dim("…or pick one by name:"))
        self._start_search = SearchPicker("type part of a name")
        self._start_search.picked.connect(self._on_start_by_name)
        tbox.addWidget(self._start_search)
        tbox.addStretch(1)

        # In piedi: la catena sopra, la rosa sotto.
        going = QWidget()
        gbox = QVBoxLayout(going)
        self._chain_told = QLabel("")
        gbox.addWidget(self._chain_told)
        self._chain_table = TrackTable(reorderable=True, checkable=True,
                                       favouritable=True)
        self._wire(self._chain_table)
        self._chain_table.model_.order_changed.connect(self._on_chain_reorder)
        gbox.addWidget(self._chain_table, stretch=3)
        self._chain_doubles = _dim("")
        self._chain_doubles.setToolTip(DOUBLES_HINT)
        self._chain_doubles.setVisible(False)
        gbox.addWidget(self._chain_doubles)

        branch = QHBoxLayout()
        branch.addWidget(QLabel("Branch from"))
        self._branch = QComboBox()
        self._branch.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self._branch.currentIndexChanged.connect(self._on_branch)
        branch.addWidget(self._branch, stretch=1)
        self._unchain = QPushButton("🗑 Remove it from the chain")
        self._unchain.clicked.connect(self._on_unchain)
        branch.addWidget(self._unchain)
        self._trend = _knob(
            branch, "Trend", 0.0, "around the source", "a step ahead",
            "Where the chain is GOING, not just where it is. At 0 the roster "
            "sits around the source track, as always. Above 0 it looks one "
            "step ahead along the line from the previous track to the "
            "source — on the map and in tempo — and proposes what lies "
            "there: a rising set keeps rising. Needs a track before the "
            "source; on the first track it does nothing.")
        self._trend.valueChanged.connect(lambda _: self._refresh_roster())
        gbox.addLayout(branch)

        self._roster_told = QLabel("")
        gbox.addWidget(self._roster_told)
        self._roster_table = TrackTable(checkable=True, favouritable=True)
        self._wire(self._roster_table)
        gbox.addWidget(self._roster_table, stretch=3)
        self._roster_add = QPushButton("➕ Add selected to the chain")
        self._roster_add.setToolTip("One after the other: ticking three "
                                    "means 'then these three'.")
        self._roster_add.clicked.connect(self._on_roster_add)
        grow = QHBoxLayout()
        grow.addWidget(self._roster_add, stretch=1)
        self._auto = QPushButton("⚡ Auto chain")
        auto_why = theme.hint(
            "The chain grows on its own: the top of the roster is taken, "
            "becomes the source, the roster is made again, and so on for "
            "as many steps as the number here. Same cost, same weights, "
            "same rules on copies as picking by hand — and Trend counts, "
            "so a rising chain keeps rising. It starts from the track in "
            "«Branch from». Unlike Radio Mix it judges tempo and key too, "
            "keeps the order it chose, and does not care whether the fifth "
            "track sounds like the first.")
        self._auto.setToolTip(auto_why)
        self._auto.clicked.connect(self._on_auto_chain)
        grow.addWidget(self._auto)
        self._auto_steps = QSpinBox()
        self._auto_steps.setRange(1, AUTO_STEPS_MAX)
        self._auto_steps.setValue(AUTO_STEPS_DEFAULT)
        self._auto_steps.setToolTip(auto_why)
        grow.addWidget(self._auto_steps)
        gbox.addLayout(grow)

        by_name = QHBoxLayout()
        self._byhand_search = SearchPicker(
            "add a track by name — outside the roster")
        self._byhand_search.picked.connect(self._on_attach_by_name)
        by_name.addWidget(self._byhand_search, stretch=1)
        gbox.addLayout(by_name)

        row = QHBoxLayout()
        restart = QPushButton("↺ Start over")
        restart.setToolTip("Empties the chain. The playlist is not touched.")
        restart.clicked.connect(self._on_chain_restart)
        # Lo stesso verbo delle altre due schede: cambia solo il soggetto —
        # qui va la catena intera, non le righe spuntate.
        to_playlist = QPushButton("➕ Add to the playlist")
        to_playlist.setToolTip("The chain goes after what the playlist "
                               "already holds.")
        to_playlist.clicked.connect(self._on_chain_append)
        as_new = QPushButton("↺ Send as a new playlist")
        as_new.setToolTip("Starts over: what is in the playlist now is "
                          "dropped.")
        as_new.clicked.connect(self._on_chain_send)
        for button in (restart, to_playlist, as_new):
            row.addWidget(button)
        gbox.addLayout(row)

        self._chain.addWidget(start)
        self._chain.addWidget(going)
        return self._chain

    def _build_journey(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        journey_why = theme.hint(
            "The Journey: from one track to another in N steps. The start "
            "is the seed, the last of the chain or the last of the playlist; "
            "the end is optional — pick it by name, or leave it open and the "
            "set goes where the transitions lead. Every hop is judged by the "
            "transition cost with the weights above, on a corridor of the "
            "tracks that pass the filters and sit between the two; the Arc "
            "knob asks each position to sit in its chapter — Intro, Buildup, "
            "Tension, Climax, Release — as the chapters read them. No track "
            "twice, no twins back to back, copies of the same song once.")

        ends = QHBoxLayout()
        ends.addWidget(QLabel("From"))
        self._journey_from = QComboBox()
        self._journey_from.addItems(JOURNEY_SOURCES)
        self._journey_from.setToolTip(theme.hint(
            "Where the Journey starts. The seed: what is clicked on the map "
            "or ticked in the playlist. Last of the chain, last of the "
            "playlist: to continue what is already built."))
        self._journey_from.currentIndexChanged.connect(
            lambda _: self._refresh_journey())
        ends.addWidget(self._journey_from)
        ends.addSpacing(12)
        self._journey_count = QSpinBox()
        self._journey_count.setRange(JOURNEY_MIN, JOURNEY_MAX)
        self._journey_count.setValue(JOURNEY_DEFAULT)
        self._journey_count.setToolTip("How many tracks, the two ends "
                                       "included.")
        self._journey_count.valueChanged.connect(lambda _: self._replan())
        ends.addWidget(QLabel("Tracks"))
        ends.addWidget(self._journey_count)
        ends.addSpacing(12)
        self._journey_arc = _knob(
            ends, "Arc", 0.5, "transitions only", "the shape of a set",
            "How much each position is asked to sit in its chapter of the "
            "arc — tempo, energy, mood and groove in the chapter's band, "
            "on the scale of your library — against how cheap the "
            "transition is. At 0 the Journey is the smoothest run of "
            "transitions and nothing else.")
        self._journey_arc.valueChanged.connect(lambda _: self._replan())
        ends.addStretch(1)
        box.addLayout(ends)

        to = QHBoxLayout()
        to.addWidget(QLabel("To"))
        self._journey_end_told = QLabel("")
        self._journey_end_told.setToolTip(theme.hint(
            "The track the Journey lands on. Optional: without it the last "
            "track is wherever the cheapest run ends."))
        to.addWidget(self._journey_end_told, stretch=1)
        self._journey_end_clear = QPushButton("✕")
        self._journey_end_clear.setFixedWidth(44)
        self._journey_end_clear.setToolTip("Leave the end open.")
        self._journey_end_clear.clicked.connect(self._on_journey_end_clear)
        to.addWidget(self._journey_end_clear)
        box.addLayout(to)
        self._journey_end_search = SearchPicker(
            "land on a track — type part of its name, or leave it open")
        self._journey_end_search.picked.connect(self._on_journey_end)
        box.addWidget(self._journey_end_search)

        self._journey_told = QLabel("")
        self._journey_told.setWordWrap(True)
        box.addWidget(self._journey_told)
        self._journey_ask = QPushButton("🧭 Plan the journey")
        self._journey_ask.setToolTip(journey_why)
        self._journey_ask.clicked.connect(self._on_ask_journey)
        box.addWidget(self._journey_ask)
        self._journey_wait = _dim(WAITING_FOR_THE_BUTTON)
        box.addWidget(self._journey_wait)
        self._journey_table = TrackTable(checkable=True, favouritable=True)
        self._journey_table.setToolTip(journey_why)
        self._wire(self._journey_table)
        box.addWidget(self._pick_row(self._journey_table,
                                     reset=self._on_reset_journey))
        box.addWidget(self._journey_table, stretch=1)
        self._journey_doubles = _dim("")
        self._journey_doubles.setToolTip(DOUBLES_HINT)
        self._journey_doubles.setVisible(False)
        box.addWidget(self._journey_doubles)
        self._journey_add = QPushButton("➕ Add selected to the playlist")
        self._journey_add.clicked.connect(lambda: self._send_journey("append"))
        self._journey_send = QPushButton("↺ Send as a new playlist")
        self._journey_send.setToolTip("Starts over: what is in the playlist "
                                      "now is dropped.")
        self._journey_send.clicked.connect(
            lambda: self._send_journey("replace"))
        row = QHBoxLayout()
        row.addWidget(self._journey_add)
        row.addWidget(self._journey_send)
        box.addLayout(row)
        return page

    def _build_radio(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        radio_why = theme.hint(
            "Radio Mix: a playlist from a GROUP, not from one seed. The group's taste "
            "is the centre of its embeddings — split in two or three if the "
            "group has two or three souls, each served in turn. Every pick "
            "must be close to that taste and unlike what is already picked, "
            "so no twenty versions of the same track; near-identical twins "
            "stay out altogether. Acoustic only: tempo and key are not "
            "judged here — the list comes out magic-sorted, so it mixes. "
            "Only tracks that pass the filters are considered.")

        knobs = QHBoxLayout()
        knobs.addWidget(QLabel("From"))
        self._radio_from = QComboBox()
        self._radio_from.addItems(RADIO_SOURCES)
        self._radio_from.setToolTip(theme.hint(
            "Where Radio Mix starts from. Favourites: every starred track "
            "is a seed. Map selection: the lasso or box on the map — or the "
            "single seed, if that is all there is. Playlist: every track in "
            "it, so the list proposes what goes with the set as it stands."))
        self._radio_from.currentIndexChanged.connect(
            lambda _: self._refresh_radio())
        knobs.addWidget(self._radio_from)
        knobs.addSpacing(12)
        self._variety = _knob(
            knobs, "Variety", 0.5, "close, doubles allowed", "spread out",
            "How much a candidate pays for sounding like what is already "
            "picked. At 0 it is pure closeness to the group's taste — "
            "expect near-doubles. Higher spreads the list out.")
        self._variety.valueChanged.connect(lambda _: self._retune())
        self._drift = _knob(
            knobs, "Drift", 0.0, "stays around the group", "wanders off",
            "After each pick the taste moves a little towards it. At 0 the "
            "list stays around the group; higher and it becomes a journey "
            "that drifts away from where it started.")
        self._drift.valueChanged.connect(lambda _: self._retune())
        knobs.addStretch(1)
        box.addLayout(knobs)

        self._radio_told = QLabel("")
        self._radio_told.setWordWrap(True)
        box.addWidget(self._radio_told)
        self._radio_ask = QPushButton("📻 Tune in")
        self._radio_ask.setToolTip(radio_why)
        self._radio_ask.clicked.connect(self._on_ask_radio)
        box.addWidget(self._radio_ask)
        self._radio_wait = _dim(WAITING_FOR_THE_BUTTON)
        box.addWidget(self._radio_wait)
        self._radio_table = TrackTable(checkable=True, favouritable=True)
        self._radio_table.setToolTip(radio_why)
        self._wire(self._radio_table)
        box.addWidget(self._pick_row(self._radio_table,
                                     reset=self._on_reset_radio))
        box.addWidget(self._radio_table, stretch=1)
        self._radio_doubles = _dim("")
        self._radio_doubles.setToolTip(DOUBLES_HINT)
        self._radio_doubles.setVisible(False)
        box.addWidget(self._radio_doubles)
        self._radio_again = QPushButton("🔄 Again, minus the unticked")
        self._radio_again.setToolTip(theme.hint(
            "The unticked rows become no's: they are dropped, the taste "
            "moves away from them, and the list is made again. The no's "
            "are remembered until the group changes."))
        self._radio_again.clicked.connect(self._on_radio_again)
        self._radio_add = QPushButton("➕ Add selected to the playlist")
        self._radio_add.clicked.connect(lambda: self._send_radio("append"))
        self._radio_send = QPushButton("↺ Send as a new playlist")
        self._radio_send.setToolTip("Starts over: what is in the playlist "
                                    "now is dropped.")
        self._radio_send.clicked.connect(lambda: self._send_radio("replace"))
        row = QHBoxLayout()
        for button in (self._radio_again, self._radio_add, self._radio_send):
            row.addWidget(button)
        box.addLayout(row)
        return page

    # ------------------------------------------------------------------
    # il contesto: libreria, filtri, scelta
    # ------------------------------------------------------------------
    def set_library(self, lib: Library) -> None:
        """La libreria nuova (o ricaricata). La catena tiene i percorsi,
        quindi sopravvive da sé: i brani spariti cadono fuori dal walk."""
        self._lib = lib
        self._group_shown = None    # il frame è nuovo: la tabella va rifatta
        self._arc_values = None     # e i ranghi dell'arco con lui
        self._journey_end = None
        self._apply_weights()
        self._refresh_all()

    def set_pool(self, pool: np.ndarray) -> None:
        """I brani che passano i filtri: restringono rosa e proposte."""
        self._pool = pool
        if self._lib is not None:
            options = pool.tolist()
            self._start_search.set_universe(self._lib.frame, options)
            self._byhand_search.set_universe(self._lib.frame, options)
            self._journey_end_search.set_universe(self._lib.frame, options)
        self._refresh_all()

    def set_choice(self, seed: int | None, selected: list[int],
                   candidates: list[int]) -> None:
        """Su cosa lavorano le tre schede. `candidates` sono i brani da cui
        la catena può partire (spunta in playlist > gruppo > seme)."""
        self._seed, self._selected = seed, list(selected)
        self._candidates = list(candidates)
        self._refresh_all()

    def weights(self) -> tuple[float, float, float]:
        return (self._w_sound.value(), self._w_bpm.value(),
                self._w_key.value())

    # ------------------------------------------------------------------
    # aggiornamenti
    # ------------------------------------------------------------------
    def _apply_weights(self) -> None:
        if self._lib is not None:
            cost = self._lib.cost
            cost.w_sound, cost.w_bpm, cost.w_key = self.weights()

    def _on_knobs(self) -> None:
        self._apply_weights()
        self.weights_changed.emit()
        self._refresh_quick()
        self._refresh_roster()
        self._replan()
        self._retune()

    def _refresh_all(self) -> None:
        self._refresh_seed_told()
        self._refresh_quick()
        self._refresh_chain()
        self._refresh_journey()
        self._refresh_radio()

    def _refresh_seed_told(self) -> None:
        if self._lib is None or self._seed is None:
            self._seed_told.setVisible(False)
            return
        row = self._lib.frame.iloc[self._seed]
        groove = (f" · groove {row['danceability']:.2f}"
                  if pd.notna(row["danceability"]) else "")
        self._seed_told.setText(
            f"<b>Seed — {row['name']}</b><br>"
            f"{row['bpm'] or '?'} BPM · {row['camelot'] or '?'}{groove} · "
            f"{row['genres']}<br>"
            f"{mood_scale.summary(row['moods'], self._lib.common)}")
        self._seed_told.setVisible(True)

    # --- Quick List ---
    def _retitle(self, tab: int, count: int) -> None:
        """Il conteggio sulla linguetta — quante righe lavora quella scheda:
        il gruppo preso, le proposte aperte, la catena. Si legge da fuori,
        senza aprire."""
        base = TAB_TITLES[tab]
        self._tabs.setTabText(tab, f"{base} ({count})" if count else base)

    def _refresh_quick(self) -> None:
        if self._lib is None:
            return
        if self._selected:
            self._quick.setCurrentIndex(1)
            self._retitle(TAB_QUICK, len(self._selected))
            # La tabella si rifà solo quando il GRUPPO cambia: rifarla a
            # ogni giro (un peso toccato, una scelta altrove) rimetterebbe
            # la spunta alle righe che l'utente ha appena tolto.
            if self._selected != self._group_shown:
                frame, common = self._lib.frame, self._lib.common
                shown = numbered_rows(frame, self._selected, common)
                self._group_table.set_tracks(
                    shown, genre_colors(frame, shown["genres"], dark=theme.DARK))
                # Tutte spuntate in partenza: il lasso È già una scelta —
                # da qui si toglie chi non convince, non si rimette tutto.
                self._group_table.set_all_picked(True)
                self._group_shown = list(self._selected)
            self._refresh_group_buttons()
            self._tell_rings()
            return
        if self._seed is None:
            self._quick.setCurrentIndex(0)
            self._retitle(TAB_QUICK, 0)
            self._tell_rings()
            return
        self._quick.setCurrentIndex(2)
        path = self._lib.frame.at[self._seed, "path"]
        if self._asked_mixes == path:
            self._show_mixes()
        else:
            self._retitle(TAB_QUICK, 0)
            self._mixes_ask.setVisible(True)
            self._mixes_wait.setVisible(True)
            self._mixes_table.setVisible(False)
            self._mixes_add.setVisible(False)
            self._mixes_send.setVisible(False)
            self._tell_rings()

    def _on_ask_mixes(self) -> None:
        if self._lib is not None and self._seed is not None:
            self._asked_mixes = self._lib.frame.at[self._seed, "path"]
            self._show_mixes()

    def _on_reset_mixes(self) -> None:
        """Si riparte da capo: la lista si chiude e torna il bottone che la
        fa. Le spunte vanno tolte a mano, o rientrerebbero dalla finestra —
        `set_tracks` conserva i presi che il frame nuovo porta ancora, e il
        frame nuovo è quasi sempre lo stesso."""
        self._asked_mixes = None
        self._mixes_table.clear_picks()
        self._refresh_quick()

    def _show_mixes(self) -> None:
        frame, cost, common = (self._lib.frame, self._lib.cost,
                               self._lib.common)
        picks = ([(self._seed, 0.0)]
                 + nearest(cost, self._seed, k=self._count.value(),
                           pool=self._pool))
        listed = []
        for n, (i, value) in enumerate(picks):
            parts = cost.parts(self._seed, i)
            # Il numero in testa, seme compreso: le tinte dei doppioni
            # dicono «copy of #3», e senza la colonna il 3 non si trova.
            listed.append({
                "#": n + 1,
                "cost": round(value, 3),
                **reading(frame.loc[i], common),
                "sound": round(parts["sound"], 3),
                "bpm cost": round(parts["bpm"], 2),
                "key cost": round(parts["key"], 2),
                "_path": frame.at[i, "path"],
            })
        shown = pd.DataFrame(listed, columns=[
            "#", "cost", "file", "BPM", "key", "energy", "groove", "emotion",
            "sound", "bpm cost", "key cost", "mood", "genres", "folder",
            "_path"])
        self._mixes_table.set_tracks(
            shown, genre_colors(frame, shown["genres"], dark=theme.DARK))
        marks, told = double_marks(
            list(shown["_path"]),
            self._vectors_for([i for i, _ in picks]))
        self._mixes_table.set_marks(marks)
        self._mixes_doubles.setText(told or "")
        self._mixes_doubles.setVisible(told is not None)
        self._mixes_ask.setVisible(False)
        self._mixes_wait.setVisible(False)
        self._mixes_table.setVisible(True)
        self._mixes_add.setVisible(True)
        self._mixes_send.setVisible(True)
        # Le proposte, senza il seme in testa: il conteggio dice quante
        # scelte offre la lista, non quante righe scrive.
        self._retitle(TAB_QUICK, len(picks) - 1)
        self._tell_rings(mixes=[i for i, _ in picks[1:]])

    def _tell_rings(self, mixes: list[int] | None = None) -> None:
        """Gli anelli delle proposte sulla mappa: SOLO la lista aperta per il
        seme corrente, come `suggested()` di là — anelli attorno a una lista
        che nessuno ha visto direbbero che una scelta è stata fatta."""
        if self._lib is None or self._seed is None:
            self.suggestions_changed.emit([])
            return
        path = self._lib.frame.at[self._seed, "path"]
        if mixes is None and self._asked_mixes == path:
            mixes = [i for i, _ in nearest(self._lib.cost, self._seed,
                                           k=self._count.value(),
                                           pool=self._pool)]
        self.suggestions_changed.emit(mixes or [])

    def _vectors_for(self, indices: list[int]) -> np.ndarray | None:
        """Gli embedding di questi indici, o None se la libreria non porta
        uno store — capita nei test, con una libreria fatta a mano."""
        store = self._lib.store
        return store.embeddings[indices] if store is not None else None

    # --- i gesti del gruppo ---
    def _playlist_indices(self) -> list[int]:
        at_path = self._lib.at_path
        return [at_path[p] for p in self._state.playlist if p in at_path]

    def _group_picks(self) -> list[int]:
        """Le righe SPUNTATE del gruppo, nell'ordine della tabella: sono le
        sole che i tre bottoni mandano — prima partiva il gruppo intero,
        spunte o no, e la colonna ✓ era un ornamento."""
        at_path = self._lib.at_path
        return [at_path[p] for p in self._group_table.selected_paths()
                if p in at_path]

    def _refresh_group_buttons(self) -> None:
        if self._lib is None:
            return
        ticked = len(self._group_table.selected_paths())
        self._group_told.setText(
            f"<b>{len(self._selected)} track(s)</b> selected — "
            f"{ticked} ticked.")
        self._plain_append.setDisabled(ticked < 1)
        self._sort_append.setDisabled(ticked < 2)
        self._sort_new.setDisabled(ticked < 2 or not self._state.playlist)

    def _on_sort_append(self) -> None:
        wanted = self._group_picks()
        if len(wanted) >= 2:
            self.append_playlist.emit(sorted_after(
                self._lib.cost, self._playlist_indices(), wanted))

    def _on_plain_append(self) -> None:
        wanted = self._group_picks()
        if wanted:
            self.append_playlist.emit(wanted)

    def _on_sort_new(self) -> None:
        wanted = self._group_picks()
        if len(wanted) >= 2:
            self.replace_playlist.emit(magic_sort(self._lib.cost, wanted))

    def _add_rows(self, table: TrackTable) -> None:
        at_path = self._lib.at_path
        wanted = [at_path[p] for p in table.selected_paths() if p in at_path]
        if wanted:
            self.append_playlist.emit(wanted)

    def _send_rows(self, table: TrackTable) -> None:
        at_path = self._lib.at_path
        wanted = [at_path[p] for p in table.selected_paths() if p in at_path]
        if wanted:
            self.replace_playlist.emit(wanted)

    # ------------------------------------------------------------------
    # Chain Maker
    # ------------------------------------------------------------------
    def _walk(self) -> list[str]:
        return self._graph.walk()

    def _refresh_chain(self) -> None:
        if self._lib is None:
            return
        self._retitle(TAB_CHAIN, len(self._graph))
        if not len(self._graph):
            self._chain.setCurrentIndex(0)
            if self._candidates:
                frame = self._lib.frame
                names = ", ".join(_label(frame.at[i, "name"])
                                  for i in self._candidates[:3])
                if len(self._candidates) > 3:
                    names += f", and {len(self._candidates) - 3} more"
                self._start_told.setText(f"Selected on the map: <b>{names}</b>")
            self._start_told.setVisible(bool(self._candidates))
            self._start_from.setVisible(bool(self._candidates))
            return
        self._chain.setCurrentIndex(1)
        frame, common, at_path = (self._lib.frame, self._lib.common,
                                  self._lib.at_path)
        walk = self._walk()
        self._chain_told.setText(f"<b>The chain — {len(walk)} track(s)</b>")
        table = chain_table(frame, at_path, walk, common)
        order = ["#", "file", "BPM", "key", "energy", "groove", "emotion",
                 "Δbpm", "Δkey", "Δenergy", "Δgroove", "mood", "genres",
                 "folder", "_path"]
        table = table[[c for c in order if c in table.columns]]
        self._chain_table.set_tracks(
            table, genre_colors(frame, table["genres"], dark=theme.DARK))
        chain_paths = list(table["_path"])
        marks, told = double_marks(
            chain_paths,
            self._vectors_for([at_path[p] for p in chain_paths]))
        self._chain_table.set_marks(marks)
        self._chain_doubles.setText(told or "")
        self._chain_doubles.setVisible(told is not None)

        # Il menu della sorgente: l'ultimo arrivato di default, che è da
        # dove si continua nove volte su dieci; cambiarlo serve a ramificare.
        if self._source not in walk:
            self._source = walk[-1] if walk else None
        self._branch.blockSignals(True)
        self._branch.clear()
        for path in walk:
            name = (frame.at[at_path[path], "name"] if path in at_path
                    else Path(path).stem)
            self._branch.addItem(str(name), path)
        self._branch.setCurrentIndex(walk.index(self._source)
                                     if self._source in walk else -1)
        self._branch.blockSignals(False)
        self._unchain.setDisabled(len(walk) < 2)
        self._refresh_roster()

    def _refresh_roster(self) -> None:
        if (self._lib is None or not len(self._graph)
                or self._source not in self._graph):
            return
        frame, cost, common, at_path = (self._lib.frame, self._lib.cost,
                                        self._lib.common, self._lib.at_path)
        source_idx = at_path.get(self._source)
        if source_idx is None:
            self._roster_told.setText("The source track is not on the map "
                                      "any more.")
            self._roster_table.set_tracks(pd.DataFrame(columns=READING_ORDER))
            return
        source = frame.iloc[source_idx]
        ahead = self._ahead(source_idx)
        self._roster_told.setText(
            f"<b>Mixes out of — {_label(str(source['name']))}</b>"
            + (" · looking ahead" if ahead is not None else ""))
        taken = {at_path[p] for p in self._graph.tracks if p in at_path}
        picks = suggestions(
            cost, source_idx, taken, k=FRONTIER_SIZE, pool=self._pool,
            key_of=lambda i: normalized_name(Path(frame.at[i, "path"])),
            song_of=lambda i: song_key(Path(frame.at[i, "path"])),
            ahead=ahead)
        self._roster_picks = picks
        table = roster_table(frame, picks, source, common)
        if not len(table):
            self._roster_table.set_tracks(pd.DataFrame(columns=READING_ORDER))
            self._roster_told.setText("No candidate left that passes the "
                                      "filters.")
            return
        order = ["file", "cost", "BPM", "key", "energy", "groove", "emotion",
                 "Δbpm", "Δkey", "Δenergy", "Δgroove", "copies", "mood",
                 "genres", "folder", "_path"]
        table = table[[c for c in order if c in table.columns]]
        self._roster_table.set_tracks(
            table, genre_colors(frame, table["genres"], dark=theme.DARK))

    def _ahead(self, source_idx: int) -> tuple | None:
        """Il punto un passo avanti alla sorgente, se c'è una tendenza da
        seguire: serve un brano PRIMA della sorgente nella scaletta."""
        trend = self._trend.value()
        walk = self._walk()
        at = walk.index(self._source) if self._source in walk else 0
        if trend <= 0 or at == 0:
            return None
        previous = self._lib.at_path.get(walk[at - 1])
        if previous is None:
            return None
        return self._lib.cost.ahead(previous, source_idx, trend)

    # --- i gesti della catena ---
    def _chained(self, graph: GraphPlaylist, source: str | None) -> None:
        self._graph = graph
        self._source = source
        self._refresh_chain()
        self._refresh_journey()
        self.chain_changed.emit(self._walk())

    def _on_start_from_choice(self) -> None:
        frame = self._lib.frame
        tracks = [frame.at[i, "path"] for i in self._candidates]
        if tracks:
            self._chained(GraphPlaylist().start(*tracks), tracks[-1])

    def _on_start_by_name(self, index: int) -> None:
        path = self._lib.frame.at[index, "path"]
        self._start_search.clear()
        self._chained(GraphPlaylist().start(path), path)

    def _on_chain_reorder(self, paths: list[str]) -> None:
        # Ricostruire invece di ricucire i collegamenti: una sequenza
        # scritta a mano È una fila, come per la colonna "#" di là.
        order = [p for p in paths if p]
        if order and order != self._walk():
            self._chained(GraphPlaylist().start(*order), order[-1])

    def _on_branch(self, at: int) -> None:
        path = self._branch.itemData(at)
        if path:
            self._source = path
            self._refresh_roster()

    def _on_unchain(self) -> None:
        if self._source in self._graph and len(self._graph) > 1:
            self._journal.record(
                "unchain", path=self._source,
                neighbours=self._graph.neighbours(self._source))
            self._graph.remove(self._source)
            tracks = self._graph.tracks
            self._chained(self._graph, tracks[-1] if tracks else None)

    def _on_roster_add(self) -> None:
        frame, at_path = self._lib.frame, self._lib.at_path
        wanted = [p for p in self._roster_table.selected_paths()]
        if not wanted:
            return
        self._note_pick(wanted)
        # In fila uno dietro l'altro: spuntarne tre vuol dire "poi questi
        # tre", non tre rami dalla stessa sorgente.
        previous = self._source
        for path in wanted:
            self._graph.add(previous, path)
            previous = path
        self._chained(self._graph, previous)

    def _note_pick(self, chosen: list[str]) -> None:
        """L'appunto della scelta: la rosa com'era, con i numeri di ogni
        candidato e il suo posto in lista, e chi è stato preso. Il posto
        conta perché si tende a prendere il primo: chi imparerà da qui deve
        poterlo scontare."""
        frame, at_path = self._lib.frame, self._lib.at_path
        source_idx = at_path.get(self._source)
        if source_idx is None:
            return
        self._journal.record(
            "pick", tab="chain",
            source=facts(frame.iloc[source_idx]),
            weights=list(self.weights()), trend=self._trend.value(),
            shown=[{**facts(frame.iloc[i]), "rank": n,
                    "cost": round(float(value), 4), "copies": len(copies)}
                   for n, (i, value, copies) in enumerate(self._roster_picks)],
            chosen=list(chosen))

    def _on_auto_chain(self) -> None:
        frame, at_path = self._lib.frame, self._lib.at_path
        source_idx = at_path.get(self._source)
        if source_idx is None:
            return
        walk = self._walk()
        at = walk.index(self._source) if self._source in walk else 0
        previous = at_path.get(walk[at - 1]) if at > 0 else None
        added = auto_chain(
            self._lib.cost,
            [at_path[p] for p in self._graph.tracks if p in at_path],
            source_idx, self._auto_steps.value(), previous=previous,
            pool=self._pool,
            key_of=lambda i: normalized_name(Path(frame.at[i, "path"])),
            song_of=lambda i: song_key(Path(frame.at[i, "path"])),
            trend=self._trend.value())
        if not added:
            return
        paths = [frame.at[i, "path"] for i in added]
        # Non è una scelta: è la macchina che prende il primo. Va nel
        # quaderno con un nome suo, o chi imparerà dai «pick» imparerebbe
        # che il primo della rosa è sempre quello giusto.
        self._journal.record(
            "auto_chain", source=self._source, steps=self._auto_steps.value(),
            added=paths, trend=self._trend.value(),
            weights=list(self.weights()))
        previous_path = self._source
        for path in paths:
            self._graph.add(previous_path, path)
            previous_path = path
        self._chained(self._graph, previous_path)

    def _on_attach_by_name(self, index: int) -> None:
        if self._source is None or self._source not in self._graph:
            return
        path = self._lib.frame.at[index, "path"]
        self._byhand_search.clear()
        self._graph.add(self._source, path)
        self._chained(self._graph, path)

    def _on_chain_restart(self) -> None:
        self._chained(GraphPlaylist(), None)

    def _on_chain_append(self) -> None:
        at_path = self._lib.at_path
        self._journal.record("chain_sent", how="append", walk=self._walk())
        self.append_playlist.emit(
            [at_path[p] for p in self._walk() if p in at_path])

    def _on_chain_send(self) -> None:
        at_path = self._lib.at_path
        self._journal.record("chain_sent", how="replace", walk=self._walk())
        self.replace_playlist.emit(
            [at_path[p] for p in self._walk() if p in at_path])

    # ------------------------------------------------------------------
    # Journey
    # ------------------------------------------------------------------
    def _journey_start(self) -> int | None:
        """La partenza secondo il menu: il seme (o il primo del gruppo),
        l'ultimo della catena, l'ultimo della playlist."""
        which = self._journey_from.currentIndex()
        if which == 0:
            if self._seed is not None:
                return self._seed
            return self._candidates[0] if self._candidates else None
        at_path = self._lib.at_path
        if which == 1:
            walk = self._walk()
            return at_path.get(walk[-1]) if walk else None
        on_list = self._playlist_indices()
        return on_list[-1] if on_list else None

    def _journey_names(self, start: int | None,
                       end: int | None) -> tuple[str, str]:
        frame = self._lib.frame
        here = (_label(str(frame.at[start, "name"])) if start is not None
                else "nothing to start from")
        there = (_label(str(frame.at[end, "name"])) if end is not None
                 else "open")
        return here, there

    def _refresh_journey(self) -> None:
        if self._lib is None:
            return
        start = self._journey_start()
        end = self._journey_end
        if end is not None and end == start:
            end = None
        frame = self._lib.frame
        key = (frame.at[start, "path"] if start is not None else None,
               frame.at[end, "path"] if end is not None else None)
        if key != self._journey_key:
            # Estremi nuovi: la fila di prima parlava di un altro viaggio.
            self._journey_key = None
            self._journey_shown = []
            self._journey_table.clear_picks()
        here, there = self._journey_names(start, end)
        self._journey_end_told.setText(
            f"<b>{there}</b>" if end is not None
            else "<i>open — wherever the cheapest run ends</i>")
        self._journey_end_clear.setVisible(self._journey_end is not None)
        can = start is not None and self._lib.store is not None
        if start is None:
            self._journey_told.setText(
                "Nothing to start from — click a seed on the map, or pick "
                "another start in the menu.")
        elif self._lib.store is None:
            self._journey_told.setText("No embeddings to travel on.")
        else:
            self._journey_told.setText(f"<b>From {here}</b> · to {there}.")
        self._journey_ask.setVisible(self._journey_key is None)
        self._journey_ask.setEnabled(can)
        self._journey_wait.setVisible(self._journey_key is None)
        for widget in (self._journey_table, self._journey_add,
                       self._journey_send):
            widget.setVisible(self._journey_key is not None)
        if self._journey_key is None:
            self._journey_doubles.setVisible(False)
            self._retitle(TAB_JOURNEY, 0)

    def _on_journey_end(self, index: int) -> None:
        self._journey_end = int(index)
        self._journey_end_search.clear()
        self._refresh_journey()

    def _on_journey_end_clear(self) -> None:
        self._journey_end = None
        self._refresh_journey()

    def _on_ask_journey(self) -> None:
        if self._lib is None or self._lib.store is None:
            return
        start = self._journey_start()
        if start is None:
            return
        end = self._journey_end if self._journey_end != start else None
        frame = self._lib.frame
        self._journey_key = (frame.at[start, "path"],
                             frame.at[end, "path"] if end is not None else None)
        self._show_journey()

    def _on_reset_journey(self) -> None:
        self._journey_key = None
        self._journey_table.clear_picks()
        self._refresh_journey()

    def _replan(self) -> None:
        """Una manopola toccata a viaggio aperto: si rifà con gli stessi
        estremi."""
        if self._journey_key is not None and self._lib is not None:
            self._show_journey()

    def _arc_of_library(self) -> np.ndarray:
        if self._arc_values is None:
            frame = self._lib.frame
            self._arc_values = arc.measures(
                frame["bpm"].tolist(), frame["energy"].tolist(),
                frame["valence_rank"].tolist(),
                frame["danceability"].tolist())
        return self._arc_values

    def _show_journey(self) -> None:
        frame, common, at_path = (self._lib.frame, self._lib.common,
                                  self._lib.at_path)
        start = at_path.get(self._journey_key[0])
        end = at_path.get(self._journey_key[1]) \
            if self._journey_key[1] is not None else None
        if start is None:
            self._on_reset_journey()
            return
        w_arc = self._journey_arc.value()
        path = journey.plan(
            self._lib.cost, start, self._journey_count.value(), end=end,
            pool=self._pool, arc_values=self._arc_of_library(), w_arc=w_arc,
            song_of=lambda i: song_key(Path(frame.at[i, "path"])))
        self._journey_shown = path
        shown = numbered_rows(frame, path, common)
        if w_arc > 0:
            # Il capitolo di ogni posizione, come lo chiede l'arco: la
            # stessa pastiglia della playlist, perché è la stessa cosa.
            names = [arc.CHAPTERS[c]["name"]
                     for c in arc.chapters_along(len(path))]
            shown.insert(1, "chapter", [[name] for name in names])
        self._journey_table.set_tracks(
            shown, genre_colors(frame, shown["genres"], dark=theme.DARK))
        # Tutte spuntate: la fila È la proposta, si toglie chi non convince.
        self._journey_table.set_all_picked(True)
        marks, told = double_marks(list(shown["_path"]),
                                   self._vectors_for(path))
        self._journey_table.set_marks(marks)
        self._journey_doubles.setText(told or "")
        self._journey_doubles.setVisible(told is not None)
        short = len(path) < self._journey_count.value()
        here, there = self._journey_names(start, end)
        self._journey_told.setText(
            f"<b>From {here}</b> · to {there}."
            + (f" Only {len(path)} track(s) could be joined: the filters "
               "leave too few on the way." if short else ""))
        self._journey_ask.setVisible(False)
        self._journey_wait.setVisible(False)
        for widget in (self._journey_table, self._journey_add,
                       self._journey_send):
            widget.setVisible(True)
        self._retitle(TAB_JOURNEY, len(path))

    def _send_journey(self, how: str) -> None:
        at_path, frame = self._lib.at_path, self._lib.frame
        wanted = [at_path[p] for p in self._journey_table.selected_paths()
                  if p in at_path]
        if not wanted:
            return
        self._journal.record(
            "journey_sent", how=how, start=self._journey_key[0],
            end=self._journey_key[1], count=self._journey_count.value(),
            arc=self._journey_arc.value(), weights=list(self.weights()),
            shown=[frame.at[i, "path"] for i in self._journey_shown],
            ticked=[frame.at[i, "path"] for i in wanted])
        (self.append_playlist if how == "append"
         else self.replace_playlist).emit(wanted)

    # ------------------------------------------------------------------
    # Radio
    # ------------------------------------------------------------------
    def _radio_seeds(self) -> list[int]:
        """I semi secondo il menu: i preferiti, la playlist, o quello che
        c'è sulla mappa — il gruppo, o il seme solo se è tutto quello che
        c'è."""
        at_path = self._lib.at_path
        which = self._radio_from.currentIndex()
        if which == RADIO_FAVOURITES:
            return [at_path[p] for p in self._state.favourites if p in at_path]
        if which == RADIO_PLAYLIST:
            return self._playlist_indices()
        if self._selected:
            return list(self._selected)
        return [self._seed] if self._seed is not None else []

    def _refresh_radio(self) -> None:
        if self._lib is None:
            return
        seeds = self._radio_seeds()
        source = RADIO_SOURCES[self._radio_from.currentIndex()]
        key = (source, tuple(self._lib.frame.at[i, "path"] for i in seeds))
        if key != self._radio_key:
            # Semi nuovi: la lista di prima parlava di un altro gruppo, e i
            # suoi no con lei.
            self._radio_key = None
            self._radio_shown, self._radio_negatives = [], []
            self._radio_table.clear_picks()
        can = bool(seeds) and self._lib.store is not None
        if not seeds:
            which = self._radio_from.currentIndex()
            self._radio_told.setText(
                "No favourites yet — star some tracks first."
                if which == RADIO_FAVOURITES else
                "The playlist is empty — put some tracks in it first."
                if which == RADIO_PLAYLIST else
                "Nothing on the map — click a seed, or drag the lasso or "
                "the box around a group.")
        elif self._lib.store is None:
            self._radio_told.setText("No embeddings to tune from.")
        else:
            self._radio_told.setText(
                f"<b>Tuned from {len(seeds)} track(s)</b> — {source}.")
        self._radio_ask.setVisible(self._radio_key is None)
        self._radio_ask.setEnabled(can)
        self._radio_wait.setVisible(self._radio_key is None)
        for widget in (self._radio_table, self._radio_again,
                       self._radio_add, self._radio_send):
            widget.setVisible(self._radio_key is not None)
        if self._radio_key is None:
            self._radio_doubles.setVisible(False)
            self._retitle(TAB_RADIO, 0)

    def _on_ask_radio(self) -> None:
        if self._lib is None or self._lib.store is None:
            return
        seeds = self._radio_seeds()
        if not seeds:
            return
        source = RADIO_SOURCES[self._radio_from.currentIndex()]
        self._radio_key = (source,
                           tuple(self._lib.frame.at[i, "path"] for i in seeds))
        self._show_radio()

    def _on_reset_radio(self) -> None:
        self._radio_key = None
        self._radio_negatives = []
        self._radio_table.clear_picks()
        self._refresh_radio()

    def _retune(self) -> None:
        """Una manopola toccata a lista aperta: si rifà con gli stessi no."""
        if self._radio_key is not None and self._lib is not None:
            self._show_radio()

    def _show_radio(self) -> None:
        frame, common, store = self._lib.frame, self._lib.common, self._lib.store
        seeds = self._radio_seeds()
        picks = radio.tune(store.embeddings[:len(frame)], seeds,
                           pool=self._pool, k=self._count.value(),
                           variety=self._variety.value(),
                           drift=self._drift.value(),
                           negatives=self._radio_negatives,
                           song_of=lambda i: song_key(Path(frame.at[i, "path"])))
        ordered = magic_sort(self._lib.cost, picks)
        self._radio_shown = ordered
        shown = numbered_rows(frame, ordered, common)
        self._radio_table.set_tracks(
            shown, genre_colors(frame, shown["genres"], dark=theme.DARK))
        # Tutte spuntate: la lista È la proposta, si toglie chi non convince.
        self._radio_table.set_all_picked(True)
        marks, told = double_marks(list(shown["_path"]),
                                   self._vectors_for(ordered))
        self._radio_table.set_marks(marks)
        self._radio_doubles.setText(told or "")
        self._radio_doubles.setVisible(told is not None)
        self._radio_ask.setVisible(False)
        self._radio_wait.setVisible(False)
        for widget in (self._radio_table, self._radio_again,
                       self._radio_add, self._radio_send):
            widget.setVisible(True)
        self._radio_again.setEnabled(bool(ordered))
        self._retitle(TAB_RADIO, len(ordered))

    def _radio_unticked(self) -> list[int]:
        ticked = set(self._radio_table.selected_paths())
        frame = self._lib.frame
        return [i for i in self._radio_shown
                if frame.at[i, "path"] not in ticked]

    def _on_radio_again(self) -> None:
        if self._radio_key is None:
            return
        fresh = self._radio_unticked()
        self._radio_negatives = list(dict.fromkeys(
            self._radio_negatives + fresh))
        frame = self._lib.frame
        self._journal.record(
            "radio_again", source=self._radio_key[0],
            seeds=list(self._radio_key[1]),
            negatives=[frame.at[i, "path"] for i in self._radio_negatives],
            variety=self._variety.value(), drift=self._drift.value())
        self._show_radio()

    def _send_radio(self, how: str) -> None:
        at_path, frame = self._lib.at_path, self._lib.frame
        wanted = [at_path[p] for p in self._radio_table.selected_paths()
                  if p in at_path]
        if not wanted:
            return
        self._journal.record(
            "radio_sent", how=how, source=self._radio_key[0],
            seeds=list(self._radio_key[1]),
            shown=[frame.at[i, "path"] for i in self._radio_shown],
            ticked=[frame.at[i, "path"] for i in wanted],
            negatives=[frame.at[i, "path"] for i in self._radio_negatives],
            variety=self._variety.value(), drift=self._drift.value())
        (self.append_playlist if how == "append"
         else self.replace_playlist).emit(wanted)
