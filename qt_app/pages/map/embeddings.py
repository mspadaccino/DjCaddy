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

Le due manopole in alto costano cose diverse. Accorpare o no le dimensioni
rifà l'immagine e basta; ordinare per distanza la lega invece al seme, e da
lì in poi ogni seme nuovo vuole l'impronta rifatta — mezzo secondo, contro i
millesimi della sola colonna. Per questo l'ordine di libreria resta il
default: è quello che tiene il disegno fermo mentre si sceglie.
"""

from __future__ import annotations

import numpy as np

from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel,
                               QVBoxLayout, QWidget)

from core.viz.embedding_figure import (CELL_BUDGETS, build_fingerprint_figure,
                                       columns_for, cosine_distances,
                                       distance_overlay, fingerprint,
                                       fingerprint_source, rows_budget,
                                       unit_norms)
from qt_app import theme
from qt_app.widgets.plotly_view import PlotlyView


# Le due pile possibili, e cosa vogliono dire. L'ordine di libreria è quello
# in cui i brani sono andati sulla mappa; per distanza è il seme in cima.
ORDERS = {"library order": "library", "distance from the seed": "distance"}


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
        self._away_from: int | None = None
        self._away = None
        self._ok = False

        self._group = QCheckBox("Group nearest dimensions")
        self._group.setChecked(True)
        self._group.setToolTip(theme.hint(
            "On, each column is the average of ten consecutive dimensions — "
            "128 columns, the shape of the vector at a glance. Off, one "
            "column per dimension, all 1280 of them: the picture is ten "
            "times wider (scroll sideways to walk it) and, at the same pixel "
            "budget, ten times fewer tracks fit in it. The count below says "
            "how many are drawn."))
        self._group.toggled.connect(lambda _: self._redraw())

        self._sort = QComboBox()
        self._sort.addItems(list(ORDERS))
        self._sort.setToolTip(theme.hint(
            "Library order is the order the tracks went on the map: the "
            "picture stays put when the seed changes, and only the distance "
            "column is redrawn. By distance puts the seed's nearest "
            "neighbours at the top and the strangers at the bottom — the "
            "distance column becomes a readable gradient and the bands sort "
            "themselves — at the price of redrawing the whole picture every "
            "time the seed moves."))
        self._sort.currentTextChanged.connect(lambda _: self._redraw())

        self._budget = QComboBox()
        self._budget.addItems(list(CELL_BUDGETS))
        self._budget.setToolTip(theme.hint(
            "How many pixels the picture may cost, which is how many tracks "
            "fit in it: above the budget a stable random sample of what the "
            "filters leave is drawn, and the count below says how many of "
            "how many. Light redraws in about half a second; full holds a "
            "library of ninety thousand tracks whole, at a second and a half "
            "a redraw — the same price the map pays for one of its own. "
            "Every drawn row is a real track either way: it is the rest of "
            "the library that is missing, never the truth about a row."))
        self._budget.currentTextChanged.connect(lambda _: self._redraw())

        top = QHBoxLayout()
        top.addWidget(self._group)
        top.addSpacing(12)
        top.addWidget(QLabel("Sort by"))
        top.addWidget(self._sort)
        top.addSpacing(12)
        top.addWidget(QLabel("Picture"))
        top.addWidget(self._budget)
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
            self._norms, self._away, self._away_from = None, None, None
        self._frame, self._embeddings = frame, embeddings
        self._redraw()

    def update_overlays(self, marks: dict) -> None:
        moved = self._seed_of(marks) != self._seed_of(self._marks)
        self._marks = marks
        if not self._ok:
            return
        # Ordinate per distanza, le righe DIPENDONO dal seme: un seme nuovo
        # vuole l'impronta rifatta, non solo la sua colonna.
        if moved and self._by_distance():
            self._redraw()
        else:
            self._push_distance()

    # ------------------------------------------------------------------
    def _by_distance(self) -> bool:
        return ORDERS[self._sort.currentText()] == "distance"

    def _seed_of(self, marks: dict | None) -> int | None:
        """Il seme di questi segni, se è un brano che abbiamo in matrice."""
        seed = (marks or {}).get("seed")
        if seed is None or self._embeddings is None:
            return None
        return int(seed) if seed < len(np.asarray(self._embeddings)) else None

    def _distances(self, seed: int):
        """Le distanze dal seme, tenute da parte: il disegno e la colonna le
        chiedono tutte e due, e la matrice è mezzo giga da rileggere."""
        if self._away_from != seed:
            matrix = np.asarray(self._embeddings)
            if self._norms is None:
                self._norms = unit_norms(matrix)
            self._away = cosine_distances(matrix, self._norms, matrix[seed])
            self._away_from = seed
        return self._away

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

        every = not self._group.isChecked()
        columns = columns_for(matrix.shape[1], every)
        budget = rows_budget(columns, CELL_BUDGETS[self._budget.currentText()])
        # Sopra il budget si disegna un campione stabile, come fa la mappa
        # sopra `MAX_POINTS`: `sort_index` lo rimette nell'ordine della
        # libreria, o le righe uscirebbero rimescolate a ogni ridisegno.
        sampled = len(self._frame) > budget
        rows = (self._frame.sample(budget, random_state=0).sort_index()
                if sampled else self._frame)
        # Lo stesso campione, in un altro ordine: quali brani si vedono non
        # dipende dal seme — solo dove finiscono nella pila.
        seed = self._seed_of(self._marks) if self._by_distance() else None
        if seed is not None:
            wanted = rows["index"].to_numpy()
            rows = rows.iloc[np.argsort(self._distances(seed)[wanted],
                                        kind="stable")]
        self._rows, self._columns, self._ok = rows, columns, True

        # Le righe si prendono DENTRO `fingerprint`, a blocchi: indicizzare
        # qui vorrebbe dire copiarsi accanto mezzo giga di matrice.
        quadro = fingerprint(matrix, every, take=rows["index"].to_numpy())
        self._view.set_figure(build_fingerprint_figure(
            rows, fingerprint_source(quadro, theme.DARK), columns,
            dark=theme.DARK, room=self._view.width()))
        self._told.setText(
            f"{len(rows):,} track(s) drawn"
            + (f" of {len(self._frame):,} — a stable sample: the picture is "
               f"capped at {budget:,} rows with {columns} columns"
               if sampled else "")
            + f" · {columns} column(s) of {matrix.shape[1]} dimensions"
            + (" · nearest to the seed on top" if seed is not None
               else " · no seed yet: rows stay in library order"
               if self._by_distance() else "")
            + " · ⓘ")
        if self._marks is not None:
            self._push_distance()

    def _push_distance(self) -> None:
        """La colonna della distanza: solo lei si rimanda a ogni gesto."""
        seed = self._seed_of(self._marks)
        if seed is None:
            self._view.set_overlays(distance_overlay(None, self._columns))
            return
        places = self._rows["index"].to_numpy()
        self._view.set_overlays(distance_overlay(
            self._distances(seed)[places], self._columns, places=places,
            dark=theme.DARK))
