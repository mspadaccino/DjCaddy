"""La pagina Map della Fase 2: lo spike, non ancora la pagina vera.

Qui si decide il go/no-go del piano: la mappa VERA — lo store reale, decine
di migliaia di punti — dentro PlotlyView, con accanto i pezzi che la Fase 3
comporrà per davvero. Il giro completo che questa pagina dimostra:

- lasso o riquadro sulla mappa → i percorsi scelti finiscono nel log e le
  righe nella tabella (con le pastiglie di core/viz);
- clic su un punto → seme, e la figura si aggiorna via Plotly.react col suo
  cerchio: è la misura di quanto costa un aggiornamento a mappa piena;
- doppio clic su una riga → il brano suona nel dock;
- la selezione compare anche sulla lavagna — il frontend HTML riusato con
  lo shim — e i trascinamenti delle schede tornano nel log.

I tempi di ogni passo si scrivono nel log e sono i numeri dello spike. La
composizione vera della pagina (filtri, playlist, Chain Maker) è Fase 3:
qui ogni pezzo deve solo dimostrare di reggere il carico e di parlare con
gli altri.
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QLabel, QPlainTextEdit, QSplitter,
                               QVBoxLayout, QWidget)

from core.analysis import energy, mood_scale
from core.analysis.map_store import MapStore
from core.viz.board import DEFAULT_HEIGHT, board_payload
from core.viz.map_figure import (COLORED_GENRES, MAX_POINTS, build_figure,
                                 genre_level, marker_sizes)
from core.viz.track_columns import genre_colors
from qt_app import theme
from qt_app.state import AppState
from qt_app.widgets.board_view import BoardView
from qt_app.widgets.plotly_view import PlotlyView
from qt_app.widgets.track_table import TrackTable, track_frame
from qt_app.workers import run_in_pool


def _library(store: MapStore) -> pd.DataFrame:
    """I brani piazzati con le misure derivate: il `library_frame` di
    `streamlit_app.views.map_analysis`, senza la cache di sessione attorno.

    Energia e valence sono RANGHI sulla libreria intera, non numeri per
    brano: si calcolano su tutte le righe e si tagliano ai piazzati, come
    di là.
    """
    placed = store.placed
    frame = pd.DataFrame(store.rows[:placed])
    frame["index"] = np.arange(len(frame))
    frame["energy"] = energy.from_rows(store.rows)[:placed]
    valence = np.asarray(mood_scale.from_rows(store.rows), dtype=float)
    frame["valence_rank"] = energy.ranks(valence)[:placed]
    frame["x"], frame["y"] = store.coords[:, 0], store.coords[:, 1]
    return frame


class MapPage(QWidget):
    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._store: MapStore | None = None
        self._frame: pd.DataFrame | None = None
        self._drawn: pd.DataFrame | None = None
        self._top_genres: list[str] = []
        self._common: dict[str, int] = {}
        self._at_path: dict[str, int] = {}
        self._build()
        run_in_pool(self._load, self._on_loaded, self._on_failed)

    # --- costruzione ---
    def _build(self) -> None:
        self._map = PlotlyView(background=theme.BACKGROUND)
        self._caption = QLabel("Opening the map…")
        self._caption.setObjectName("dim")

        left_box = QWidget()
        left = QVBoxLayout(left_box)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(6)
        left.addWidget(self._map, stretch=1)
        left.addWidget(self._caption)

        self._table = TrackTable(reorderable=True)
        self._table.row_activated.connect(self._state.play)
        self._table.model_.order_changed.connect(self._on_reordered)

        self._board = BoardView()
        self._board.value_changed.connect(self._on_board_event)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(500)

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self._table)
        right.addWidget(self._board)
        right.addWidget(self._log)
        right.setSizes([380, 320, 160])

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left_box)
        split.addWidget(right)
        split.setSizes([900, 560])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(split)

        self._map.point_clicked.connect(self._on_click)
        self._map.points_selected.connect(self._on_selected)
        self._map.deselected.connect(self._on_deselected)
        self._map.rendered.connect(
            lambda ms: self._say(f"Plotly.react: {ms:.0f} ms"))

    def _say(self, line: str) -> None:
        self._log.appendPlainText(line)

    # --- caricamento (nel pool: 87k righe non si leggono sul filo della UI) ---
    def _load(self):
        began = time.perf_counter()
        store = MapStore.load()
        loaded = time.perf_counter()
        if not store.placed:
            return store, None, {}, (loaded - began, 0.0)
        frame = _library(store)
        common = (mood_scale.popularity(list(frame["moods"]))
                  if "moods" in frame else {})
        return store, frame, common, (loaded - began,
                                      time.perf_counter() - loaded)

    def _on_failed(self, trouble: Exception) -> None:
        self._caption.setText(f"The map could not be opened: {trouble}")

    def _on_loaded(self, result) -> None:
        store, frame, common, (store_s, frame_s) = result
        if frame is None:
            self._caption.setText(
                "The map is empty or not projected yet — build it from the "
                "Streamlit app, then reopen this one.")
            return
        self._store, self._frame, self._common = store, frame, common
        self._at_path = dict(zip(frame["path"], frame["index"]))

        began = time.perf_counter()
        drawn = frame
        if len(drawn) > MAX_POINTS:
            drawn = drawn.sample(MAX_POINTS, random_state=0)
        # Livello "parent" come il default della pagina Streamlit, e
        # diametro piatto: le due manopole arrivano con i filtri in Fase 3.
        self._drawn = drawn.assign(
            _size=marker_sizes(drawn, None),
            genre_key=drawn["top_genre"].map(
                lambda g: genre_level(g, "parent")))
        ranked = Counter(g for g in self._drawn["genre_key"] if g)
        self._top_genres = [g for g, _ in ranked.most_common(COLORED_GENRES)]
        spec = self._figure_json()
        built_s = time.perf_counter() - began

        self._map.set_figure(spec)
        self._caption.setText(
            f"{len(frame):,} tracks on the map · click = seed, lasso/box = "
            "selection, double-click a row = play")
        self._say(f"store: {store_s:.2f} s · frame: {frame_s:.2f} s · "
                  f"figure+json: {built_s:.2f} s "
                  f"({len(spec) / 1e6:.1f} MB, {len(frame):,} points)")

    # --- la figura (e i suoi aggiornamenti via Plotly.react) ---
    def _figure_json(self, seed: int | None = None,
                     selected: list[int] | None = None) -> str:
        figure = build_figure(
            self._drawn, self._top_genres, self._store.coords,
            playlist=[], seed=seed,
            seed_name=(self._frame.at[seed, "name"]
                       if seed is not None else None),
            selected=selected, dark=True)
        return figure.to_json()

    def _redraw(self, seed: int | None = None,
                selected: list[int] | None = None) -> None:
        began = time.perf_counter()
        spec = self._figure_json(seed, selected)
        self._say(f"figure+json: {(time.perf_counter() - began) * 1000:.0f} ms")
        self._map.set_figure(spec)

    # --- i gesti sulla mappa ---
    def _on_click(self, index: int) -> None:
        path = str(self._frame.at[index, "path"])
        self._state.set_seed(path)
        self._say(f"seed → {path}")
        self._redraw(seed=index)

    def _on_selected(self, indices: list[int]) -> None:
        if not indices:
            return
        # Un punto solo è un seme comunque lo si sia preso (stessa regola
        # della pagina Streamlit).
        if len(indices) == 1:
            self._on_click(indices[0])
            return
        rows = self._frame.iloc[indices]
        paths = [str(p) for p in rows["path"]]
        self._state.set_selection(paths)

        self._say(f"selected: {len(paths)} track(s)")
        for path in paths[:20]:
            self._say(f"  {path}")
        if len(paths) > 20:
            self._say(f"  … and {len(paths) - 20} more (all printed to stdout)")
        print(f"[map] selected {len(paths)} track(s):")
        for path in paths:
            print(f"  {path}")

        shown = track_frame(rows, self._common)
        colors = genre_colors(self._frame, shown["genres"], dark=True)
        self._table.set_tracks(shown, colors)
        # Gli args del componente sono il payload PIÙ le chiavi che
        # l'adapter Streamlit aggiunge a parte (selected, chapters, dark):
        # senza `dark` la lavagna si disegna a tema chiaro.
        self._board.set_payload({
            **board_payload(self._frame, self._at_path, paths,
                            DEFAULT_HEIGHT, self._common, dark=True),
            "selected": None, "chapters": [], "dark": True})
        self._redraw(selected=list(indices))

    def _on_deselected(self) -> None:
        self._state.clear_selection()
        self._say("deselected")
        self._redraw()

    # --- il resto della pagina ---
    def _on_reordered(self, paths: list[str]) -> None:
        self._say(f"table reordered: {len(paths)} row(s), first is "
                  f"{paths[0].rsplit('/', 1)[-1] if paths else '—'}")

    def _on_board_event(self, value: dict) -> None:
        told = {k: v for k, v in value.items() if k != "at"}
        self._say(f"board → {told}")
