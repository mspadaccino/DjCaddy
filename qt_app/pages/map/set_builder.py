"""Build a set: i tre modi di passare dalla mappa a una scaletta.

Le stesse tre schede della pagina Streamlit — Quick List (cosa ci si mixa
sopra), Sounds like it (cosa gli somiglia), Chain Maker (un brano alla
volta) — sopra un pannello unico di pesi e "quanti elencare", che sono gli
stessi filtri di partenza per tutte e tre.

Le regole vengono tutte da core: `nearest`, `magic_sort`, `sorted_after`,
`store.similar`, `suggestions`, `chain_table`, `roster_table`. Qui ci sono i
widget e la disciplina delle liste: una lista si apre quando la si chiede
("Make the list"), resta viva finché il seme è quello — si ricalcola con i
pesi e con i filtri — e si richiude da sé quando il seme cambia.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem,
                               QPushButton, QSpinBox, QStackedWidget,
                               QTabWidget, QVBoxLayout, QWidget)

from core.analysis import mood_scale
from core.analysis.duplicates import folded, normalized_name, song_key
from core.analysis.graph_playlist import GraphPlaylist, suggestions
from core.analysis.mixing import magic_sort, nearest, sorted_after
from core.viz.board import _label, chain_table, roster_table
from core.viz.track_columns import READING_ORDER, genre_colors, reading
from qt_app.state import AppState
from qt_app.widgets.track_table import TrackTable

from .library import Library

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

WAITING_FOR_THE_BUTTON = ("Nothing built yet — press the button above. The "
                          "list does not open by itself: most clicks on the "
                          "map are looking around, not choosing what comes "
                          "next.")


def _dim(text: str, wrap: bool = True) -> QLabel:
    label = QLabel(text)
    label.setObjectName("dim")
    label.setWordWrap(wrap)
    return label


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


class SetBuilderPanel(QWidget):
    """Il pannello: pesi e conto sopra, le tre schede sotto.

    Parla col resto della pagina a segnali: `append_playlist` e
    `replace_playlist` portano INDICI di libreria; `suggestions_changed` e
    `chain_changed` dicono cosa cerchiare sulla mappa (le proposte, la
    catena). La scelta corrente — seme o gruppo, con la precedenza alla
    spunta in playlist — arriva da fuori con `set_choice`: la regola di chi
    comanda sta nella pagina, qui si lavora su quello che comanda.
    """

    append_playlist = Signal(list)
    replace_playlist = Signal(list)
    suggestions_changed = Signal(list, list)
    chain_changed = Signal(list)

    def __init__(self, state: AppState, wire_table, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._wire = wire_table
        self._lib: Library | None = None
        self._pool: np.ndarray = np.empty(0, dtype=int)
        self._seed: int | None = None
        self._selected: list[int] = []
        self._candidates: list[int] = []
        self._asked_mixes: str | None = None
        self._asked_alike: str | None = None
        self._graph = GraphPlaylist()
        self._source: str | None = None
        self._build()

    # ------------------------------------------------------------------
    # costruzione
    # ------------------------------------------------------------------
    def _build(self) -> None:
        self._seed_told = _dim("")

        weights = QHBoxLayout()
        self._w_sound = self._weight(weights, "sound",
                                     "How much the distance on the map "
                                     "counts: the acoustic affinity.")
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
                               "Sounds like it.")
        self._count.valueChanged.connect(lambda _: self._on_knobs())
        weights.addWidget(self._count)
        weights.addStretch(1)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_quicklist(), "✨ Quick List")
        self._tabs.addTab(self._build_alike(), "🎯 Sounds like it")
        self._tabs.addTab(self._build_chain(), "🔗 Chain Maker")

        box = QVBoxLayout(self)
        box.addWidget(self._seed_told)
        box.addLayout(weights)
        box.addWidget(self._tabs, stretch=1)

    def _weight(self, into: QHBoxLayout, name: str, why: str) -> QDoubleSpinBox:
        into.addWidget(QLabel(f"w·{name}"))
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 2.0)
        spin.setSingleStep(0.1)
        spin.setValue(1.0)
        spin.setToolTip(why)
        spin.valueChanged.connect(lambda _: self._on_knobs())
        into.addWidget(spin)
        return spin

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
        gbox.addWidget(_dim(
            "Magic sort walks all of them once, in the order that keeps "
            "every transition cheap — the travelling-salesman path over the "
            "cost. It is the answer to a folder of tracks in no order."))
        self._group_table = TrackTable()
        self._wire(self._group_table)
        gbox.addWidget(self._group_table, stretch=1)
        row = QHBoxLayout()
        self._sort_append = QPushButton("✨ Magic sort and append")
        self._sort_append.setToolTip("Sorted among themselves, then added "
                                     "after what the playlist already holds.")
        self._sort_append.clicked.connect(self._on_sort_append)
        self._plain_append = QPushButton("➕ Append them, unsorted")
        self._plain_append.clicked.connect(
            lambda: self.append_playlist.emit(list(self._selected)))
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
        sbox.addWidget(_dim(
            "Ranked by the transition cost — sound, tempo and key together, "
            "with the weights above. Only tracks that pass the filters are "
            "considered. The first row is the seed itself."))
        self._mixes_ask = QPushButton("✨ Make the list")
        self._mixes_ask.setToolTip("Builds the list of what mixes out of "
                                   "this seed.")
        self._mixes_ask.clicked.connect(self._on_ask_mixes)
        sbox.addWidget(self._mixes_ask)
        self._mixes_wait = _dim(WAITING_FOR_THE_BUTTON)
        sbox.addWidget(self._mixes_wait)
        self._mixes_table = TrackTable(checkable=True)
        self._wire(self._mixes_table)
        sbox.addWidget(self._mixes_table, stretch=1)
        self._mixes_add = QPushButton("➕ Add selected to the playlist")
        self._mixes_add.clicked.connect(
            lambda: self._add_rows(self._mixes_table))
        sbox.addWidget(self._mixes_add)

        for page in (idle, group, seed):
            self._quick.addWidget(page)
        return self._quick

    def _build_alike(self) -> QWidget:
        self._alike = QStackedWidget()

        idle = QWidget()
        QVBoxLayout(idle).addWidget(_dim(
            "Pick a single seed on the map — not a group — to see what "
            "sounds like it."))

        seed = QWidget()
        sbox = QVBoxLayout(seed)
        sbox.addWidget(_dim(
            "Pure acoustic closeness, measured in the 1280 dimensions of "
            "the embedding — not on the flattened map, and with no regard "
            "for tempo or key. A different question from 'what mixes out "
            "of this'. The first row is the seed itself, here too."))
        self._alike_ask = QPushButton("✨ Make the list")
        self._alike_ask.setToolTip("Builds the list of what sounds like "
                                   "this seed.")
        self._alike_ask.clicked.connect(self._on_ask_alike)
        sbox.addWidget(self._alike_ask)
        self._alike_wait = _dim(WAITING_FOR_THE_BUTTON)
        sbox.addWidget(self._alike_wait)
        self._alike_table = TrackTable(checkable=True)
        self._wire(self._alike_table)
        sbox.addWidget(self._alike_table, stretch=1)
        self._alike_add = QPushButton("➕ Add selected to the playlist")
        self._alike_add.clicked.connect(
            lambda: self._add_rows(self._alike_table))
        sbox.addWidget(self._alike_add)

        self._alike.addWidget(idle)
        self._alike.addWidget(seed)
        return self._alike

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
        self._chain_table = TrackTable(reorderable=True)
        self._wire(self._chain_table)
        self._chain_table.model_.order_changed.connect(self._on_chain_reorder)
        gbox.addWidget(self._chain_table, stretch=3)

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
        gbox.addLayout(branch)

        self._roster_told = QLabel("")
        gbox.addWidget(self._roster_told)
        self._roster_table = TrackTable(checkable=True)
        self._wire(self._roster_table)
        gbox.addWidget(self._roster_table, stretch=3)
        self._roster_add = QPushButton("➕ Add selected to the chain")
        self._roster_add.setToolTip("One after the other: ticking three "
                                    "means 'then these three'.")
        self._roster_add.clicked.connect(self._on_roster_add)
        gbox.addWidget(self._roster_add)

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
        to_playlist = QPushButton("➡️ Append to playlist")
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

    # ------------------------------------------------------------------
    # il contesto: libreria, filtri, scelta
    # ------------------------------------------------------------------
    def set_library(self, lib: Library) -> None:
        """La libreria nuova (o ricaricata). La catena tiene i percorsi,
        quindi sopravvive da sé: i brani spariti cadono fuori dal walk."""
        self._lib = lib
        self._apply_weights()
        self._refresh_all()

    def set_pool(self, pool: np.ndarray) -> None:
        """I brani che passano i filtri: restringono rosa e proposte."""
        self._pool = pool
        if self._lib is not None:
            options = pool.tolist()
            self._start_search.set_universe(self._lib.frame, options)
            self._byhand_search.set_universe(self._lib.frame, options)
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
            cost.w_map, cost.w_bpm, cost.w_key = self.weights()

    def _on_knobs(self) -> None:
        self._apply_weights()
        self._refresh_quick()
        self._refresh_alike()
        self._refresh_roster()

    def _refresh_all(self) -> None:
        self._refresh_seed_told()
        self._refresh_quick()
        self._refresh_alike()
        self._refresh_chain()

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
    def _refresh_quick(self) -> None:
        if self._lib is None:
            return
        if self._selected:
            self._quick.setCurrentIndex(1)
            frame, common = self._lib.frame, self._lib.common
            self._group_told.setText(
                f"<b>{len(self._selected)} track(s)</b> selected.")
            shown = numbered_rows(frame, self._selected, common)
            self._group_table.set_tracks(
                shown, genre_colors(frame, shown["genres"], dark=True))
            few = len(self._selected) < 2
            self._sort_append.setDisabled(few)
            self._sort_new.setDisabled(few or not self._state.playlist)
            self._tell_rings()
            return
        if self._seed is None:
            self._quick.setCurrentIndex(0)
            self._tell_rings()
            return
        self._quick.setCurrentIndex(2)
        path = self._lib.frame.at[self._seed, "path"]
        if self._asked_mixes == path:
            self._show_mixes()
        else:
            self._mixes_ask.setVisible(True)
            self._mixes_wait.setVisible(True)
            self._mixes_table.setVisible(False)
            self._mixes_add.setVisible(False)
            self._tell_rings()

    def _on_ask_mixes(self) -> None:
        if self._lib is not None and self._seed is not None:
            self._asked_mixes = self._lib.frame.at[self._seed, "path"]
            self._show_mixes()

    def _show_mixes(self) -> None:
        frame, cost, common = (self._lib.frame, self._lib.cost,
                               self._lib.common)
        picks = ([(self._seed, 0.0)]
                 + nearest(cost, self._seed, k=self._count.value(),
                           pool=self._pool))
        listed = []
        for i, value in picks:
            parts = cost.parts(self._seed, i)
            listed.append({
                "cost": round(value, 3),
                **reading(frame.loc[i], common),
                "sound": round(parts["map"], 3),
                "bpm cost": round(parts["bpm"], 2),
                "key cost": round(parts["key"], 2),
                "_path": frame.at[i, "path"],
            })
        shown = pd.DataFrame(listed, columns=[
            "cost", "file", "BPM", "key", "energy", "groove", "emotion",
            "sound", "bpm cost", "key cost", "mood", "genres", "folder",
            "_path"])
        self._mixes_table.set_tracks(
            shown, genre_colors(frame, shown["genres"], dark=True))
        self._mixes_ask.setVisible(False)
        self._mixes_wait.setVisible(False)
        self._mixes_table.setVisible(True)
        self._mixes_add.setVisible(True)
        self._tell_rings(mixes=[i for i, _ in picks[1:]])

    # --- Sounds like it ---
    def _refresh_alike(self) -> None:
        if self._lib is None:
            return
        if self._seed is None:
            self._alike.setCurrentIndex(0)
            self._tell_rings()
            return
        self._alike.setCurrentIndex(1)
        path = self._lib.frame.at[self._seed, "path"]
        if self._asked_alike == path:
            self._show_alike()
        else:
            self._alike_ask.setVisible(True)
            self._alike_wait.setVisible(True)
            self._alike_table.setVisible(False)
            self._alike_add.setVisible(False)
            self._tell_rings()

    def _on_ask_alike(self) -> None:
        if self._lib is not None and self._seed is not None:
            self._asked_alike = self._lib.frame.at[self._seed, "path"]
            self._show_alike()

    def _show_alike(self) -> None:
        frame, common = self._lib.frame, self._lib.common
        rows = ([(self._seed, 1.0)]
                + self._lib.store.similar(self._seed, k=self._count.value(),
                                          limit=len(frame)))
        listed = [{"similarity": round(score, 3),
                   **reading(frame.loc[i], common),
                   "_path": frame.at[i, "path"]}
                  for i, score in rows]
        shown = pd.DataFrame(listed,
                             columns=["similarity", *READING_ORDER, "_path"])
        self._alike_table.set_tracks(
            shown, genre_colors(frame, shown["genres"], dark=True))
        self._alike_ask.setVisible(False)
        self._alike_wait.setVisible(False)
        self._alike_table.setVisible(True)
        self._alike_add.setVisible(True)
        self._tell_rings(alike=[i for i, _ in rows[1:]])

    def _tell_rings(self, mixes: list[int] | None = None,
                    alike: list[int] | None = None) -> None:
        """Gli anelli delle proposte sulla mappa: SOLO le liste aperte per il
        seme corrente, come `suggested()` di là — anelli attorno a una lista
        che nessuno ha visto direbbero che una scelta è stata fatta."""
        if self._lib is None or self._seed is None:
            self.suggestions_changed.emit([], [])
            return
        path = self._lib.frame.at[self._seed, "path"]
        if mixes is None and self._asked_mixes == path:
            mixes = [i for i, _ in nearest(self._lib.cost, self._seed,
                                           k=self._count.value(),
                                           pool=self._pool)]
        if alike is None and self._asked_alike == path:
            alike = [i for i, _ in
                     self._lib.store.similar(self._seed,
                                             k=self._count.value(),
                                             limit=self._lib.placed)]
        self.suggestions_changed.emit(mixes or [], alike or [])

    # --- i gesti del gruppo ---
    def _playlist_indices(self) -> list[int]:
        at_path = self._lib.at_path
        return [at_path[p] for p in self._state.playlist if p in at_path]

    def _on_sort_append(self) -> None:
        order = sorted_after(self._lib.cost, self._playlist_indices(),
                             self._selected)
        self.append_playlist.emit(order)

    def _on_sort_new(self) -> None:
        self.replace_playlist.emit(magic_sort(self._lib.cost, self._selected))

    def _add_rows(self, table: TrackTable) -> None:
        at_path = self._lib.at_path
        wanted = [at_path[p] for p in table.selected_paths() if p in at_path]
        if wanted:
            self.append_playlist.emit(wanted)

    # ------------------------------------------------------------------
    # Chain Maker
    # ------------------------------------------------------------------
    def _walk(self) -> list[str]:
        return self._graph.walk()

    def _refresh_chain(self) -> None:
        if self._lib is None:
            return
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
            table, genre_colors(frame, table["genres"], dark=True))

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
        self._roster_told.setText(
            f"<b>Mixes out of — {_label(str(source['name']))}</b>")
        taken = {at_path[p] for p in self._graph.tracks if p in at_path}
        picks = suggestions(
            cost, source_idx, taken, k=FRONTIER_SIZE, pool=self._pool,
            key_of=lambda i: normalized_name(Path(frame.at[i, "path"])),
            song_of=lambda i: song_key(Path(frame.at[i, "path"])))
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
            table, genre_colors(frame, table["genres"], dark=True))

    # --- i gesti della catena ---
    def _chained(self, graph: GraphPlaylist, source: str | None) -> None:
        self._graph = graph
        self._source = source
        self._refresh_chain()
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
            self._graph.remove(self._source)
            tracks = self._graph.tracks
            self._chained(self._graph, tracks[-1] if tracks else None)

    def _on_roster_add(self) -> None:
        frame, at_path = self._lib.frame, self._lib.at_path
        wanted = [p for p in self._roster_table.selected_paths()]
        if not wanted:
            return
        # In fila uno dietro l'altro: spuntarne tre vuol dire "poi questi
        # tre", non tre rami dalla stessa sorgente.
        previous = self._source
        for path in wanted:
            self._graph.add(previous, path)
            previous = path
        self._chained(self._graph, previous)

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
        self.append_playlist.emit(
            [at_path[p] for p in self._walk() if p in at_path])

    def _on_chain_send(self) -> None:
        at_path = self._lib.at_path
        self.replace_playlist.emit(
            [at_path[p] for p in self._walk() if p in at_path])
