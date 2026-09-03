"""La scheda Embeddings: una riga per brano, il suo vettore a colori.

La terza vista sugli stessi brani, accanto a mappa e quadranti. Quelle
mostrano due numeri per brano — la proiezione, o due misure scelte — questa
mostra il vettore intero da cui la proiezione è stata schiacciata. Si guarda
per bande: righe che vanno tutte dalla stessa parte sulle stesse colonne
sono brani imparentati, ed è la stessa parentela che la mappa disegna come
vicinanza.

I gesti sono quelli della mappa, e non per simmetria: passare col mouse dice
di che brano è la riga, un clic lo manda in seme, il lazo ne prende un
gruppo. La colonna a sinistra è la distanza dal seme nelle 1280 dimensioni —
quella vera, non l'ombra sul disegno.
"""

from __future__ import annotations

import numpy as np

from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from core.viz.embedding_figure import (build_fingerprint_figure, columns_for,
                                       cosine_distances, distance_overlay,
                                       fingerprint, fingerprint_source,
                                       rows_budget, unit_norms)
from qt_app import theme
from qt_app.widgets.plotly_view import PlotlyView


def _dim(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setObjectName("dim")
    label.setWordWrap(True)
    return label


class EmbeddingPane(QWidget):
    """L'impronta degli embedding come widget, coi suoi due comandi."""

    def __init__(self, view: PlotlyView, parent=None) -> None:
        super().__init__(parent)
        self._view = view
        self._frame = None
        self._embeddings = None
        self._norms = None
        self._rows = None
        self._columns = 0
        self._marks: dict | None = None
        self._ok = False

        self._every = QCheckBox("Every dimension")
        self._every.setToolTip(theme.hint(
            "Off, each column is the average of ten consecutive dimensions "
            "— 128 columns, the shape of the vector at a glance. On, one "
            "column per dimension, all 1280 of them: the picture is ten "
            "times wider and, at the same pixel budget, ten times fewer "
            "tracks fit in it. The count below says how many are drawn."))
        self._every.toggled.connect(lambda _: self._redraw())

        top = QHBoxLayout()
        top.addWidget(self._every)
        top.addStretch(1)

        self._info = _dim("")
        self._info.setVisible(False)
        # I NUMERI nella didascalia, il come si legge nel suo tooltip: la
        # stessa divisione della mappa, dove lo spazio è del disegno.
        self._told = _dim("")
        self._told.setToolTip(theme.hint(
            "One row per track, one column per dimension of its embedding. "
            "The colour is how far that track sits from the others on that "
            "dimension — blue below the middle, red above, the background "
            "colour for 'like everyone else'. Hover a row to see whose it "
            "is, click it to make it the seed, lasso a band to select the "
            "group. The column on the left is the cosine distance from the "
            "seed across all 1280 dimensions: that is the real distance, of "
            "which the map is the flattened shadow."))

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)
        box.addLayout(top)
        box.addWidget(self._info)
        box.addWidget(self._view, stretch=1)
        box.addWidget(self._told)

    # ------------------------------------------------------------------
    def set_cloud(self, frame, embeddings) -> None:
        """I brani da disegnare (già filtrati) e la matrice degli embedding."""
        if embeddings is not self._embeddings:
            self._norms = None
        self._frame, self._embeddings = frame, embeddings
        self._redraw()

    def update_overlays(self, marks: dict) -> None:
        self._marks = marks
        if self._ok:
            self._push_distance()

    # ------------------------------------------------------------------
    def _redraw(self) -> None:
        if self._frame is None or self._embeddings is None:
            return
        matrix = np.asarray(self._embeddings)
        wanted = self._frame["index"].to_numpy()
        if not matrix.size or not len(wanted) or matrix.shape[0] <= wanted.max():
            self._info.setText(
                "This map carries no embeddings on disk yet — the vectors "
                "arrive with the tracks the map job analyses. The map and "
                "the quadrants work without them; this view does not.")
            self._info.setVisible(True)
            self._view.setVisible(False)
            self._told.setText("")
            self._ok = False
            return
        self._info.setVisible(False)
        self._view.setVisible(True)

        every = self._every.isChecked()
        columns = columns_for(matrix.shape[1], every)
        budget = rows_budget(columns)
        # Sopra il budget si disegna un campione stabile, come fa la mappa
        # sopra `MAX_POINTS`: `sort_index` lo rimette nell'ordine della
        # libreria, o le righe uscirebbero rimescolate a ogni ridisegno.
        sampled = len(self._frame) > budget
        rows = (self._frame.sample(budget, random_state=0).sort_index()
                if sampled else self._frame)
        self._rows, self._columns, self._ok = rows, columns, True

        quadro = fingerprint(matrix[rows["index"].to_numpy()], every)
        self._view.set_figure(build_fingerprint_figure(
            rows, fingerprint_source(quadro, theme.DARK), columns,
            dark=theme.DARK))
        self._told.setText(
            f"{len(rows):,} track(s) drawn"
            + (f" of {len(self._frame):,} — a stable sample: the picture is "
               f"capped at {budget:,} rows with {columns} columns"
               if sampled else "")
            + f" · {columns} column(s) of {matrix.shape[1]} dimensions · ⓘ")
        if self._marks is not None:
            self._push_distance()

    def _push_distance(self) -> None:
        """La colonna della distanza: solo lei si rimanda a ogni gesto."""
        seed = (self._marks or {}).get("seed")
        matrix = np.asarray(self._embeddings)
        if seed is None or seed >= len(matrix):
            self._view.set_overlays(distance_overlay(None, self._columns))
            return
        if self._norms is None:
            self._norms = unit_norms(matrix)
        places = self._rows["index"].to_numpy()
        away = cosine_distances(matrix, self._norms, matrix[seed])
        self._view.set_overlays(distance_overlay(
            away[places], self._columns, places=places, dark=theme.DARK))
