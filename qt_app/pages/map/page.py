"""La pagina Map: la mappa, i quadranti, e i pannelli che ci lavorano sopra.

La regola del disegno viene dallo spike della Fase 2: la nuvola si manda al
grafico solo quando cambia davvero (filtri, livello dei generi, diametro),
mentre i gesti — seme, gruppo, playlist, catena, proposte, chi suona —
aggiornano i soli tracciati di contorno via `PlotlyView.set_overlays`, che
costano millisecondi invece del secondo e mezzo della figura piena.

Chi comanda le tre schede di Build a set segue la regola della pagina
Streamlit: una spunta nella playlist viene prima del seme del riquadro in
alto — è il gesto più recente — e un clic sulla mappa gliela toglie di
mano. Il seme in alto resta quello che era, col suo cerchio bianco.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QSplitter,
                               QTabWidget, QVBoxLayout, QWidget)

from core.analysis.dj_export import playlist_positions
from core.analysis.map_job import load_map_state
from core.analysis.map_store import default_store_dir
from core.viz.filters import matching_tracks
from core.viz.map_figure import (AXIS_FIELDS, AXIS_HELP, COLORED_GENRES,
                                 DEFAULT_AXES, EMPTY_CLOUD, GENRE_LEVELS,
                                 MAX_POINTS, SIZE_FIELDS, axis_guide,
                                 build_figure, genre_level, guide_caption,
                                 marker_sizes, overlay_figure)
from core.viz.track_columns import genre_colors
from qt_app import theme
from qt_app.pages.common import reveal_in_files
from qt_app.state import AppState
from qt_app.widgets.plotly_view import PlotlyView
from qt_app.widgets.track_table import TrackTable, track_frame
from qt_app.workers import run_in_pool

from .filters import FiltersPanel
from .library import Library, load_library
from .playlist_panel import PlaylistPanel
from .set_builder import SetBuilderPanel
from .settings import SettingsDialog

# Quanti risultati di ricerca elencare: oltre, la risposta giusta è una
# parola in più, non una lista più lunga.
SEED_MATCHES_MAX = 50


def _stamp(directory: Path) -> tuple:
    """Lo stato dei file della mappa su disco: cambia quando il job scrive."""
    out = []
    for name in ("tracks.jsonl", "coords.npy"):
        try:
            stat = (directory / name).stat()
            out.append((stat.st_mtime, stat.st_size))
        except OSError:
            out.append(None)
    return tuple(out)


def _dim(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setObjectName("dim")
    label.setWordWrap(True)
    return label


class QuadrantPane(QWidget):
    """Gli stessi brani su due misure a scelta, invece che sulla proiezione.

    Stesso campione, stessi colori, stessi anelli della mappa: cambiano solo
    le coordinate. La croce si spiega sotto al disegno, perché una riga
    tratteggiata a metà sembra un centro assoluto e quasi mai lo è.
    """

    def __init__(self, view: PlotlyView, parent=None) -> None:
        super().__init__(parent)
        self._view = view
        self._drawn = None
        self._top: list[str] = []
        self._frame = None
        self._visible = None
        self._marks: dict | None = None
        self._places = None
        self._ok = False
        self._labels = True

        names = list(AXIS_FIELDS)
        self._by_x, self._by_y = QComboBox(), QComboBox()
        for combo, at in ((self._by_x, DEFAULT_AXES[0]),
                          (self._by_y, DEFAULT_AXES[1])):
            combo.addItems(names)
            combo.setCurrentText(at)
            combo.currentTextChanged.connect(lambda _: self._redraw())
        axes = QHBoxLayout()
        axes.addWidget(QLabel("Across"))
        axes.addWidget(self._by_x, stretch=1)
        axes.addWidget(QLabel("Up"))
        axes.addWidget(self._by_y, stretch=1)

        self._info = _dim("")
        self._info.setVisible(False)

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)
        box.addLayout(axes)
        box.addWidget(self._info)
        box.addWidget(self._view, stretch=1)

    def set_cloud(self, drawn, top: list[str], frame, visible,
                  labels: bool = True) -> None:
        self._drawn, self._top = drawn, top
        self._frame, self._visible = frame, visible
        self._labels = labels
        self._redraw()

    def update_overlays(self, marks: dict) -> None:
        self._marks = marks
        if self._ok and self._places is not None:
            self._view.set_overlays(overlay_figure(self._places, marks, dark=True))

    def _redraw(self) -> None:
        if self._frame is None or self._drawn is None:
            return
        names = (self._by_x.currentText(), self._by_y.currentText())
        columns = (AXIS_FIELDS[names[0]], AXIS_FIELDS[names[1]])
        for column, name in zip(columns, names):
            if column not in self._frame or not pd.to_numeric(
                    self._frame[column], errors="coerce").notna().any():
                self._info.setText(
                    f"No track carries <b>{name}</b> yet. The energy fields "
                    "arrive with the backfill; everything else is measured "
                    "when a track goes on the map.")
                self._info.setVisible(True)
                self._view.setVisible(False)
                self._ok = False
                return
        self._info.setVisible(False)
        self._view.setVisible(True)
        self._ok = True

        # Gli anelli si disegnano per INDICE, su tutta la libreria piazzata:
        # un brano cerchiato può non essere nel campione.
        self._places = np.column_stack([
            pd.to_numeric(self._frame[column], errors="coerce")
            .to_numpy(dtype=float) for column in columns])
        guides = (axis_guide(self._visible[columns[0]], columns[0]),
                  axis_guide(self._visible[columns[1]], columns[1]))
        self._view.set_figure(build_figure(
            self._drawn, self._top, self._places, playlist=[], seed=None,
            axes=columns, titles=names, guides=guides, dark=True,
            labels=self._labels))
        if self._marks is not None:
            self._view.set_overlays(overlay_figure(self._places, self._marks, dark=True))

        # Cosa dice ogni asse — e dove passa la croce — si legge fermandosi
        # col mouse sulla sua manopola: scritto sotto al disegno erano
        # quindici righe, e il disegno è quello che qui deve avere spazio.
        cross = guide_caption(guides, columns, names).replace("**", "")
        for combo, name in ((self._by_x, names[0]), (self._by_y, names[1])):
            told = AXIS_HELP.get(name, "")
            combo.setToolTip(theme.hint(told + ("\n\n" + cross
                                                if cross else "")))


class MapPage(QWidget):
    """La pagina intera: mappa e quadranti a sinistra, i pannelli a destra."""

    # Emesso una volta sola, al primo caricamento (riuscito o no): lo splash
    # d'avvio lo aspetta per chiudersi.
    loaded_once = Signal()

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._lib: Library | None = None
        self._visible = None
        self._drawn = None
        self._pool = np.empty(0, dtype=int)
        self._top: list[str] = []
        self._sampled = False
        self._quad_dirty = True
        self._pl_selection: list[str] = []
        self._mixes: list[int] = []
        self._alike: list[int] = []
        self._chain: list[str] = []
        self._overlay_queued = False
        self._choice_queued = False
        self._reloading = False
        self._job_was_running = False
        self._store_stamp: tuple | None = None
        self._ever_loaded = False

        self._build()

        state.seed_changed.connect(lambda _: self._schedule_choice())
        state.selection_changed.connect(lambda _: self._schedule_choice())
        state.playlist_changed.connect(lambda _: self._schedule_overlays())
        state.now_playing_changed.connect(lambda _: self._schedule_overlays())

        self._job_timer = QTimer(self)
        self._job_timer.setInterval(2000)
        self._job_timer.timeout.connect(self._on_job_tick)
        self._job_timer.start()

        self._reload()

    # ------------------------------------------------------------------
    # costruzione
    # ------------------------------------------------------------------
    def _build(self) -> None:
        # La riga in alto: le due manopole del disegno, il job, i settaggi.
        self._level = QComboBox()
        self._level.addItems(list(GENRE_LEVELS))
        self._level.setToolTip(theme.hint(
            "Discogs labels are already two-level. The macro genre leaves "
            "almost nothing grey; the detailed one separates the house from "
            "the disco, at the cost of a larger 'other'."))
        self._level.currentTextChanged.connect(lambda _: self._rebuild_cloud())
        self._size_by = QComboBox()
        self._size_by.addItems(list(SIZE_FIELDS))
        self._size_by.setToolTip(theme.hint(
            "What the diameter of a point says (scaled 5th–95th "
            "percentile). The position already says how a track sounds; "
            "this is room for a number you can read. Tracks missing that "
            "number stay at the smallest size."))
        self._size_by.currentTextChanged.connect(
            lambda _: self._rebuild_cloud())
        self._labels = QCheckBox("Labels")
        self._labels.setChecked(True)
        self._labels.setToolTip(theme.hint(
            "The genre names written over their clusters (Electronic, "
            "Reggae, …). Turn them off where the groups overlap and the "
            "words cover the points."))
        self._labels.toggled.connect(lambda _: self._rebuild_cloud())
        self._job_told = _dim("")
        self._job_told.setVisible(False)
        settings = QPushButton("⚙️ Map settings")
        settings.clicked.connect(self._on_settings)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Colour by"))
        bar.addWidget(self._level)
        bar.addSpacing(8)
        bar.addWidget(QLabel("Point size"))
        bar.addWidget(self._size_by)
        bar.addSpacing(8)
        bar.addWidget(self._labels)
        bar.addStretch(1)
        bar.addWidget(self._job_told)
        bar.addWidget(settings)

        # Le due viste sugli stessi brani.
        self._map = PlotlyView(background=theme.BACKGROUND)
        self._map.point_clicked.connect(self._on_click)
        self._map.points_selected.connect(self._on_selected)
        self._map.deselected.connect(self._on_deselected)

        quad_view = PlotlyView(background=theme.BACKGROUND)
        quad_view.point_clicked.connect(self._on_click)
        quad_view.points_selected.connect(self._on_selected)
        quad_view.deselected.connect(self._on_deselected)
        self._quad = QuadrantPane(quad_view)

        self._views = QTabWidget()
        self._views.addTab(self._map, "🗺️ Map")
        self._views.addTab(self._quad, "⊞ Quadrants")
        self._views.currentChanged.connect(self._on_view_changed)

        # La didascalia porta i NUMERI (quanti brani, quanti in attesa); il
        # come si usa sta nel suo tooltip — lo spazio è della mappa.
        self._caption = _dim("Opening the map…")
        self._caption.setToolTip(theme.hint(
            "Click a point to make it the seed. In the toolbar above the "
            "chart, the lasso and the box grab the group they enclose — one "
            "track is a seed, two or more are a selection; double-click "
            "clears it. Scroll to zoom. Above "
            f"{MAX_POINTS:,} tracks a stable random sample is drawn; the "
            "suggestions still consider every one."))

        # La fila del seme: ascolta, aggiungi, cerca, togli, sfoglia.
        self._listen = QPushButton("▶")
        self._listen.setFixedWidth(44)
        self._listen.setToolTip("Hear the seed, in the player at the bottom.")
        self._listen.clicked.connect(self._on_listen_seed)
        self._add_seed = QPushButton("➕")
        self._add_seed.setFixedWidth(44)
        self._add_seed.setToolTip("Add the seed to the playlist.")
        self._add_seed.clicked.connect(self._on_add_seed)
        self._search = QLineEdit()
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(lambda _: self._search_debounce.start())
        self._clear = QPushButton("✕")
        self._clear.setFixedWidth(44)
        self._clear.setToolTip("Clear the seed and the search.")
        self._clear.clicked.connect(self._on_clear_seed)
        browse = QPushButton("🎵 Browse…")
        browse.setToolTip("Pick the seed's file from the disk.")
        browse.clicked.connect(self._on_browse_seed)
        seed_row = QHBoxLayout()
        for widget in (self._listen, self._add_seed):
            seed_row.addWidget(widget)
        seed_row.addWidget(self._search, stretch=1)
        seed_row.addWidget(self._clear)
        seed_row.addWidget(browse)

        self._trouble = _dim("")
        self._trouble.setVisible(False)
        self._matches_told = _dim("")
        self._matches_told.setVisible(False)
        self._matches_told.setToolTip(theme.hint(
            "▶ plays a track; double-click a row makes it the seed."))
        # Fra i risultati si sceglie con l'orecchio: il ▶ di riga suona il
        # brano, il doppio clic resta il gesto che lo fa seme. Niente
        # colonna di spunta: qui non si sceglie un gruppo, si sceglie UN
        # seme, e il doppio clic basta.
        self._matches = TrackTable()
        self._wire(self._matches, activate_plays=False)
        self._matches.row_activated.connect(self._on_match_picked)
        self._matches.setVisible(False)
        self._matches.setMaximumHeight(240)

        self._search_debounce = QTimer(self)
        self._search_debounce.setSingleShot(True)
        self._search_debounce.setInterval(250)
        self._search_debounce.timeout.connect(self._on_search)

        left_box = QWidget()
        left = QVBoxLayout(left_box)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(6)
        left.addWidget(self._views, stretch=1)
        left.addWidget(self._caption)
        left.addLayout(seed_row)
        left.addWidget(self._trouble)
        left.addWidget(self._matches_told)
        left.addWidget(self._matches)

        # I pannelli a destra.
        self._filters = FiltersPanel()
        self._filters.changed.connect(self._rebuild_cloud)
        self._builder = SetBuilderPanel(self._state, self._wire)
        self._builder.append_playlist.connect(self._on_builder_append)
        self._builder.replace_playlist.connect(self._on_builder_replace)
        self._builder.suggestions_changed.connect(self._on_suggestions)
        self._builder.chain_changed.connect(self._on_chain)
        self._playlist = PlaylistPanel(self._state, self._wire)
        self._playlist.picked_changed.connect(self._on_playlist_picks)

        self._panels = QTabWidget()
        self._panels.addTab(self._filters, "🔎 Filters")
        self._panels.addTab(self._builder, "🎛️ Build a set")
        self._panels.addTab(self._playlist, "🎵 Playlist")
        self._panels.setCurrentWidget(self._builder)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left_box)
        split.addWidget(self._panels)
        split.setSizes([880, 620])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addLayout(bar)
        layout.addWidget(split, stretch=1)

        self._settings = SettingsDialog(self)
        self._settings.library_changed.connect(self._reload)

    def _wire(self, table: TrackTable, activate_plays: bool = True) -> None:
        """Le stesse quattro voci su ogni tabella della pagina — e il giallo
        del brano in ascolto, che segue il lettore ovunque la riga stia."""
        if activate_plays:
            table.row_activated.connect(self._state.play)
        table.play_requested.connect(self._state.play)
        table.seed_requested.connect(self._seed_by_path)
        table.add_requested.connect(self._add_paths)
        table.reveal_requested.connect(reveal_in_files)
        table.set_playing(self._state.now_playing)
        self._state.now_playing_changed.connect(table.set_playing)

    # ------------------------------------------------------------------
    # caricamento e ricarica
    # ------------------------------------------------------------------
    def _reload(self) -> None:
        if self._reloading:
            return
        self._reloading = True
        self._store_stamp = _stamp(default_store_dir())
        run_in_pool(load_library, self._on_loaded, self._on_load_failed)

    def _signal_first_load(self) -> None:
        if not self._ever_loaded:
            self._ever_loaded = True
            self.loaded_once.emit()

    def _on_load_failed(self, trouble: Exception) -> None:
        self._reloading = False
        self._caption.setText(f"The map could not be opened: {trouble}")
        self._signal_first_load()

    def _on_loaded(self, result) -> None:
        self._reloading = False
        self._signal_first_load()
        store, lib = result
        self._settings.set_store(store)
        if lib is None:
            self._panels.setEnabled(False)
            self._caption.setText(
                "The map is empty or not projected yet — open ⚙️ Map "
                "settings, point it at a folder, and recompute the "
                "projection." if len(store) else
                "The map is empty. Open ⚙️ Map settings and point it at a "
                "folder of tracks.")
            return
        self._panels.setEnabled(True)
        self._lib = lib
        self._filters.set_frame(lib.frame)
        self._builder.set_library(lib)
        self._playlist.set_library(lib)
        self._rebuild_cloud()
        self._schedule_choice()

    # ------------------------------------------------------------------
    # la nuvola e i contorni
    # ------------------------------------------------------------------
    def _marks(self) -> dict:
        at_path = self._lib.at_path
        state = self._state
        selected = [at_path[p] for p in state.selection if p in at_path]
        seed = at_path.get(state.seed) if not selected else None
        return {
            "playlist": self._playlist.indices(),
            "seed": seed,
            "seed_name": (self._lib.frame.at[seed, "name"]
                          if seed is not None else None),
            "selected": selected,
            "chained": [at_path[p] for p in self._chain if p in at_path],
            "mixes": list(self._mixes),
            "alike": list(self._alike),
            "pl_selection": [at_path[p] for p in self._pl_selection
                             if p in at_path],
            "playing": at_path.get(state.now_playing),
        }

    def _marked_indices(self, marks: dict) -> list[int]:
        out = []
        for value in marks.values():
            if isinstance(value, list):
                out.extend(v for v in value if isinstance(v, (int, np.integer)))
            elif isinstance(value, (int, np.integer)):
                out.append(int(value))
        return out

    def _rebuild_cloud(self) -> None:
        if self._lib is None:
            return
        frame = self._lib.frame
        visible = self._filters.kept(frame)
        self._visible = visible
        self._pool = visible["index"].to_numpy()
        self._builder.set_pool(self._pool)
        if not len(visible):
            self._drawn = None
            self._map.set_figure(build_figure(
                EMPTY_CLOUD, [], self._lib.store.coords[:self._lib.placed],
                playlist=[], seed=None, dark=True))
            self._caption.setText("No track matches these filters.")
            self._quad_dirty = True
            return

        visible = visible.assign(
            _size=marker_sizes(visible,
                               SIZE_FIELDS[self._size_by.currentText()]))
        self._visible = visible
        drawn = visible
        self._sampled = len(drawn) > MAX_POINTS
        if self._sampled:
            drawn = drawn.sample(MAX_POINTS, random_state=0)

        # Chi è indicato torna dentro comunque: un cerchio attorno al nulla
        # non è un dettaglio estetico, è la mappa che dice il falso.
        marks = self._marks()
        if self._sampled:
            pointed = [i for i in self._marked_indices(marks)
                       if i in visible.index and i not in drawn.index]
            if pointed:
                drawn = pd.concat([drawn, visible.loc[pointed]])

        level = GENRE_LEVELS[self._level.currentText()]
        drawn = drawn.assign(
            genre_key=drawn["top_genre"].map(lambda g: genre_level(g, level)))
        self._drawn = drawn
        ranked = Counter(g for g in drawn["genre_key"] if g)
        self._top = [g for g, _ in ranked.most_common(COLORED_GENRES)]

        coords = self._lib.store.coords[:self._lib.placed]
        self._map.set_figure(build_figure(drawn, self._top, coords,
                                          playlist=[], seed=None, dark=True,
                                          labels=self._labels.isChecked()))
        self._map.set_overlays(overlay_figure(coords, marks, dark=True))
        self._refresh_caption()

        if self._views.currentWidget() is self._quad:
            self._quad.set_cloud(drawn, self._top, frame, visible,
                                 labels=self._labels.isChecked())
            self._quad.update_overlays(marks)
            self._quad_dirty = False
        else:
            self._quad_dirty = True

    def _refresh_caption(self) -> None:
        store, visible = self._lib.store, self._visible
        waiting = len(store) - self._lib.placed
        self._caption.setText(
            f"{len(visible):,} track(s) on the map"
            + (f" — {MAX_POINTS:,} drawn (sample)" if self._sampled else "")
            + (f" · ➕ {waiting:,} not placed yet: recompute the projection "
               "in ⚙️ Map settings" if waiting else "")
            + " · ⓘ")

    def _schedule_overlays(self) -> None:
        if not self._overlay_queued:
            self._overlay_queued = True
            QTimer.singleShot(0, self._push_overlays)

    def _push_overlays(self) -> None:
        self._overlay_queued = False
        if self._lib is None or self._drawn is None:
            return
        marks = self._marks()
        if self._sampled:
            missing = [i for i in self._marked_indices(marks)
                       if i in self._visible.index
                       and i not in self._drawn.index]
            if missing:
                # Un segno fuori dal campione: la nuvola va rifatta per
                # riportarcelo dentro.
                self._rebuild_cloud()
                return
        coords = self._lib.store.coords[:self._lib.placed]
        self._map.set_overlays(overlay_figure(coords, marks, dark=True))
        if not self._quad_dirty:
            self._quad.update_overlays(marks)

    # ------------------------------------------------------------------
    # la scelta: seme, gruppo, spunta in playlist
    # ------------------------------------------------------------------
    def _schedule_choice(self) -> None:
        if not self._choice_queued:
            self._choice_queued = True
            QTimer.singleShot(0, self._refresh_choice)

    def _op_choice(self) -> tuple[int | None, list[int]]:
        """Chi comanda le tre schede: la spunta in playlist se c'è — è il
        gesto più recente — altrimenti il seme o il gruppo della mappa."""
        at_path = self._lib.at_path
        pl = [at_path[p] for p in self._pl_selection if p in at_path]
        if pl:
            return (pl[0], []) if len(pl) == 1 else (None, pl)
        selected = [at_path[p] for p in self._state.selection if p in at_path]
        if selected:
            return None, selected
        return at_path.get(self._state.seed), []

    def _refresh_choice(self) -> None:
        self._choice_queued = False
        if self._lib is None:
            return
        at_path, frame = self._lib.at_path, self._lib.frame
        seed = at_path.get(self._state.seed)
        self._listen.setEnabled(seed is not None)
        self._add_seed.setEnabled(seed is not None)
        self._search.setPlaceholderText(
            "🔍 " + str(frame.at[seed, "name"]) + "  ·  "
            + Path(str(frame.at[seed, "folder"])).name
            if seed is not None else
            "🔍 type a few words — artist, title, remix — to find a track")
        self._clear.setEnabled(seed is not None
                               or bool(self._state.selection)
                               or bool(self._search.text()))

        op_seed, op_selected = self._op_choice()
        candidates = ([at_path[p] for p in self._pl_selection if p in at_path]
                      or op_selected
                      or ([op_seed] if op_seed is not None else []))
        self._builder.set_choice(op_seed, op_selected, candidates)
        self._schedule_overlays()

    # ------------------------------------------------------------------
    # i gesti sulla mappa
    # ------------------------------------------------------------------
    def _on_click(self, index: int) -> None:
        if self._lib is None or index >= self._lib.placed:
            return
        # Un clic sulla mappa è più recente di qualunque spunta in playlist.
        self._drop_playlist_picks()
        self._state.set_seed(str(self._lib.frame.at[index, "path"]))
        self._schedule_choice()

    def _on_selected(self, indices: list[int]) -> None:
        if self._lib is None:
            return
        picked = [i for i in dict.fromkeys(indices) if i < self._lib.placed]
        if not picked:
            return
        if len(picked) == 1:
            self._on_click(picked[0])
            return
        self._drop_playlist_picks()
        frame = self._lib.frame
        self._state.set_selection([str(frame.at[i, "path"]) for i in picked])
        self._schedule_choice()

    def _on_deselected(self) -> None:
        self._drop_playlist_picks()
        self._state.clear_selection()
        self._schedule_choice()

    def _drop_playlist_picks(self) -> None:
        if self._pl_selection:
            self._pl_selection = []
            self._playlist.clear_picks()

    def _on_playlist_picks(self, paths: list[str]) -> None:
        if paths == self._pl_selection:
            return
        self._pl_selection = list(paths)
        self._schedule_choice()

    def _on_builder_append(self, indices: list[int]) -> None:
        self._playlist.append(indices)
        # Il risultato sta di là: mandare brani alla playlist e restare a
        # guardare la scheda da cui sono partiti lasciava il dubbio che non
        # fosse successo niente.
        self._panels.setCurrentWidget(self._playlist)

    def _on_builder_replace(self, indices: list[int]) -> None:
        self._playlist.replace(indices)
        self._panels.setCurrentWidget(self._playlist)

    def _on_suggestions(self, mixes: list[int], alike: list[int]) -> None:
        if (mixes, alike) != (self._mixes, self._alike):
            self._mixes, self._alike = list(mixes), list(alike)
            self._schedule_overlays()

    def _on_chain(self, walk: list[str]) -> None:
        if walk != self._chain:
            self._chain = list(walk)
            self._schedule_overlays()

    def _on_view_changed(self, _index: int) -> None:
        if (self._views.currentWidget() is self._quad and self._quad_dirty
                and self._lib is not None and self._drawn is not None):
            self._quad.set_cloud(self._drawn, self._top, self._lib.frame,
                                 self._visible,
                                 labels=self._labels.isChecked())
            self._quad.update_overlays(self._marks())
            self._quad_dirty = False

    # ------------------------------------------------------------------
    # la fila del seme
    # ------------------------------------------------------------------
    def _seed_by_path(self, path: str) -> None:
        self._drop_playlist_picks()
        self._trouble.setVisible(False)
        self._state.set_seed(path)
        self._schedule_choice()

    def _add_paths(self, paths: list[str]) -> None:
        at_path = self._lib.at_path if self._lib else {}
        self._playlist.append([at_path[p] for p in paths if p in at_path])

    def _on_listen_seed(self) -> None:
        if self._state.seed is not None:
            self._state.play(self._state.seed)

    def _on_add_seed(self) -> None:
        if self._state.seed is not None:
            self._add_paths([self._state.seed])

    def _on_clear_seed(self) -> None:
        self._search.clear()
        self._trouble.setVisible(False)
        self._drop_playlist_picks()
        self._state.set_seed(None)
        self._state.clear_selection()
        self._schedule_choice()

    def _on_browse_seed(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Choose the seed track", "",
            "Audio (*.mp3 *.flac *.wav *.m4a *.aiff *.aif *.ogg);;"
            "All files (*)")
        if not chosen or self._lib is None:
            return
        found, _missing = playlist_positions([chosen], self._lib.at_path)
        if found:
            self._seed_by_path(str(self._lib.frame.at[found[0], "path"]))
        else:
            self._trouble.setText(
                f"{Path(chosen).name} is not on the map: add its folder "
                "under ⚙️ Map settings, or pick another file.")
            self._trouble.setVisible(True)

    def _on_search(self) -> None:
        if self._lib is None:
            return
        words = [w for w in self._search.text().casefold().split() if w]
        if not words:
            self._matches.setVisible(False)
            self._matches_told.setVisible(False)
            return
        frame = self._lib.frame
        found = matching_tracks(frame, self._pool, words)
        self._matches_told.setText(
            f"{len(found):,} match"
            + (f" — showing the first {SEED_MATCHES_MAX}"
               if len(found) > SEED_MATCHES_MAX else "") + " · ⓘ")
        self._matches_told.setVisible(True)
        rows = frame.loc[found[:SEED_MATCHES_MAX]]
        shown = track_frame(rows, self._lib.common)
        self._matches.set_tracks(
            shown, genre_colors(frame, shown["genres"], dark=True))
        self._matches.setVisible(bool(found))

    def _on_match_picked(self, path: str) -> None:
        self._seed_by_path(path)
        self._search.clear()
        self._matches.setVisible(False)
        self._matches_told.setVisible(False)

    # ------------------------------------------------------------------
    # il job e i settaggi
    # ------------------------------------------------------------------
    def _on_settings(self) -> None:
        self._settings.show()
        self._settings.raise_()

    def _on_job_tick(self) -> None:
        state = load_map_state()
        running = state is not None and state.running
        self._state.set_job_running("map", running)
        if running:
            self._job_told.setText(
                f"🛠 building the map: {state.done:,}/{state.total:,} — "
                "⚙️ Map settings to control it")
        self._job_told.setVisible(running)
        finished_now = self._job_was_running and not running
        self._job_was_running = running
        if finished_now:
            self._reload()
            return
        # Anche un job partito da terminale scrive gli stessi file: quando
        # l'impronta cambia a job fermo, la mappa nuova si prende da sé.
        if not running and not self._reloading:
            stamp = _stamp(default_store_dir())
            if self._store_stamp is not None and stamp != self._store_stamp:
                self._reload()
