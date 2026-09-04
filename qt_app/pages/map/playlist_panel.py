"""La playlist: la tabella, i capitoli, la lavagna, l'andata e il ritorno.

Il risultato della pagina sta qui — da qualunque scheda sia arrivato — e da
qui si porta via (M3U8, rekordbox XML) o si riprende (M3U8, file dal disco).
La lavagna disegna QUESTA playlist, con le aree colorate dei capitoli
quando ci sono; il riordino è il trascinamento delle righe, che è il motivo
per cui la tabella è nativa.

Gli scarti "from previous" si leggono a pesi fermi (1,1,1), con un costo
tutto di questa sezione: è la stessa scelta della pagina Streamlit, dove i
pesi degli slider governano le proposte e non il resoconto.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox,
                               QFileDialog, QHBoxLayout, QLabel,
                               QListWidget, QMessageBox, QPushButton,
                               QSlider, QSplitter, QVBoxLayout, QWidget)

from core.analysis.dj_export import (build_m3u8, build_rekordbox_xml,
                                     playlist_positions, read_m3u8,
                                     read_title_artist)
from core.analysis.duplicates import song_key
from core.analysis.journal import Journal
from core.analysis.mixing import TransitionCost, magic_sort
from core.viz.board import (DEFAULT_HEIGHT, HEIGHT_FIELDS, HEIGHT_MEANING,
                            board_payload, reordered)
from core.viz.chapters import (CHAPTERS, assign_chapters,
                               board_chapter_regions)
from core.viz.track_columns import genre_colors, reading
from qt_app import theme
from qt_app.state import AppState
from qt_app.widgets.board_view import BoardView
from qt_app.widgets.track_table import TrackTable

from .library import Library

AUDIO_FILTER = "Audio (*.mp3 *.flac *.wav *.m4a *.aiff *.aif *.ogg);;All files (*)"


def _dim(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("dim")
    label.setWordWrap(True)
    return label


def playlist_rows(frame: pd.DataFrame, cost: TransitionCost,
                  playlist: list[int], common: dict,
                  ch_lookup: dict[int, str] | None) -> pd.DataFrame:
    """Le righe della tabella: numero, lettura comune, costo dal precedente,
    e il capitolo quando i capitoli ci sono."""
    steps = [None] + [cost.between(a, b)
                      for a, b in zip(playlist, playlist[1:])]
    listed = []
    for position, (i, step) in enumerate(zip(playlist, steps)):
        row = {"#": position + 1, **reading(frame.loc[i], common),
               "from previous": round(step, 3) if step is not None else None,
               "_path": frame.at[i, "path"]}
        if ch_lookup is not None:
            row["chapter"] = ([ch_lookup[i]] if i in ch_lookup else [])
        listed.append(row)
    order = ["#"] + (["chapter"] if ch_lookup is not None else []) + \
        ["file", "BPM", "key", "energy", "groove", "emotion",
         "from previous", "mood", "genres", "folder", "_path"]
    return pd.DataFrame(listed, columns=order)


# Quante righe del resoconto dei saltati stanno nella finestra: oltre, la
# lista intera passa dietro «Show Details…» — un dialogo alto uno schermo
# non è più un resoconto.
SKIPPED_SHOWN = 12


def appended(current: list[str], incoming: list[str]
             ) -> tuple[list[str], list[str]]:
    """La coda dopo l'aggiunta, e i saltati perché già presenti.

    La playlist non tiene lo stesso file due volte: chi arriva e c'è già —
    da prima, o perché la stessa mandata lo porta due volte — non entra di
    nuovo e finisce nel resoconto, una volta sola, nell'ordine d'arrivo.
    Con `current` vuoto è la dedupe di una lista intera: così la usa
    `replace`, per gli M3U8 che elencano lo stesso file due volte.
    """
    merged = list(current)
    skipped: list[str] = []
    for path in incoming:
        if path not in merged:
            merged.append(path)
        elif path not in skipped:
            skipped.append(path)
    return merged, skipped


# Sopra questa somiglianza (coseno sugli embedding, la stessa misura di
# w·sound di Quick List) due brani della playlist si segnalano come lo stesso
# pezzo sotto nomi diversi. Alta apposta: verso 0.9 si pescano i vicini di
# genere, qui si vuole lo stesso audio — rip, edit, radio cut. È solo una
# tinta informativa: sbagliare per eccesso costa un'occhiata, non un brano.
SOUND_TWIN_MIN = 0.97


def playlist_doubles(paths: list[str], vectors: np.ndarray | None,
                     twin_min: float = SOUND_TWIN_MIN
                     ) -> tuple[list[list[int]], list[tuple[int, int, float]]]:
    """I sospetti doppioni della scaletta: (gruppi per nome, coppie per
    suono), come posizioni 0-based nella playlist.

    Per nome comanda `song_key`, la chiave larga fatta apposta per le
    scalette: numero di traccia e parentesi del mix non distinguono, perché
    due edit dello stesso pezzo in serata sono un doppione. Per suono il
    coseno sugli embedding (`vectors`, una riga per brano della playlist)
    sopra `twin_min`; le coppie che stanno già in un gruppo per nome non si
    ripetono — il nome è il segnale più forte, il suono aggiunge solo i
    gemelli che si chiamano in modo diverso.
    """
    by_song: dict[str, list[int]] = {}
    for n, path in enumerate(paths):
        by_song.setdefault(song_key(Path(path)), []).append(n)
    groups = [g for g in by_song.values() if len(g) > 1]
    group_of = {n: k for k, g in enumerate(groups) for n in g}

    pairs: list[tuple[int, int, float]] = []
    if vectors is not None and len(paths) > 1:
        unit = vectors / np.maximum(
            np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
        scores = unit @ unit.T
        # Il triangolo alto: ogni coppia una volta, mai (n, n). Gli zeri
        # che triu lascia sotto non passano la soglia, che è ben sopra.
        for a, b in np.argwhere(np.triu(scores, k=1) >= twin_min):
            a, b = int(a), int(b)
            same_name = (a in group_of and b in group_of
                         and group_of[a] == group_of[b])
            if not same_name:
                pairs.append((a, b, float(scores[a, b])))
    return groups, pairs


def double_marks(paths: list[str], vectors: np.ndarray | None,
                 twin_min: float = SOUND_TWIN_MIN
                 ) -> tuple[dict[str, tuple[QColor, str]], str | None]:
    """Le tinte dei sospetti per `TrackTable.set_marks`, e il resoconto.

    path -> (tinta, tooltip), SOLO per le copie in eccesso: la prima
    occorrenza di ogni gruppo resta pulita, così «via tutto il tinto» è un
    gesto sicuro — di ogni pezzo ne resta una. Arancio per lo stesso pezzo
    sotto altro nome, viola per chi suona quasi identico a una riga
    precedente; dove valgono entrambi veste il nome, che è il segnale più
    forte. Il resoconto è la riga sotto la tabella, None quando tace.
    """
    groups, pairs = playlist_doubles(paths, vectors, twin_min)
    noted: dict[int, tuple[QColor, list[str]]] = {}

    def note(n: int, tint: QColor, line: str) -> None:
        noted.setdefault(n, (tint, []))[1].append(line)

    for g in groups:
        # Il path identico due volte qui non arriva: append e replace
        # dedupicano entrambi, quindi le copie sono sempre file diversi.
        keeper = g[0]
        for n in g[1:]:
            note(n, theme.TWIN_NAME_ROW,
                 f"Copy of #{keeper + 1} — same song name once track "
                 "numbers and (mix) notes are stripped.")
    for a, b, score in pairs:
        note(b, theme.TWIN_SOUND_ROW,
             f"Sounds nearly identical to #{a + 1} "
             f"(similarity {score:.3f}).")
    if not noted:
        return {}, None
    marks = {paths[n]: (tint, theme.hint("<br>".join(lines)))
             for n, (tint, lines) in noted.items()}
    named = sum(len(g) - 1 for g in groups)
    parts = ([f"{named} repeat a song name from an earlier row"]
             if named else []) + \
        ([f"{len(noted) - named} sound nearly identical to an earlier row"]
         if len(noted) > named else [])
    told = ("🎭 Possible doubles — " + ", ".join(parts)
            + ". Tinted = the extra copy, the first take stays clean; "
            "hover one for its partner.")
    return marks, told


# Il capo lasco dello slider di pulizia: sotto questo punto la maglia
# comincia a pescare vicini di genere invece di doppioni veri (vedi il
# commento su SOUND_TWIN_MIN), ma qui è una scelta esplicita di chi tira lo
# slider, non un default silenzioso.
REMOVE_SIMILAR_FLOOR = 0.90


class SimilarityCleanupDialog(QDialog):
    """Lo slider di soglia per ripulire la playlist dai doppioni in blocco.

    Da sinistra a destra la maglia si stringe: a sinistra soglia 1.0, solo
    gli identici; a destra `REMOVE_SIMILAR_FLOOR`, dove contano come
    doppioni anche i vicini di suono più larghi. I nomi ripetuti (stesso
    `song_key`) contano sempre, a ogni posizione dello slider — sono la
    stessa canzone per costruzione, non serve una soglia per dirlo. Il
    conteggio sotto lo slider si ricalcola a ogni tacca, PRIMA che l'OK
    tocchi la playlist: `removable_paths()` è letto solo dopo l'accept.
    """

    def __init__(self, paths: list[str], vectors: np.ndarray | None,
                parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Remove similar tracks")
        self.setModal(True)
        self._paths = paths
        self._vectors = vectors
        self._losers: set[int] = set()
        self._build()
        self._on_slider(self._slider.value())

    def _build(self) -> None:
        intro = _dim(
            "Same detection as the tinted rows in the playlist, but with a "
            "threshold you control: everything it catches leaves the "
            "playlist at once — the audio files on disk are untouched. "
            "Exact repeats of the same song name always count; drag "
            "toward 'Loosely similar' to also catch tracks that merely "
            "sound alike.")
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 100)
        self._slider.setValue(round(
            (1.0 - SOUND_TWIN_MIN) / (1.0 - REMOVE_SIMILAR_FLOOR) * 100))
        self._slider.valueChanged.connect(self._on_slider)
        self._threshold_told = _dim("")
        ends = QHBoxLayout()
        ends.addWidget(_dim("Identical only"))
        ends.addWidget(self._slider, stretch=1)
        ends.addWidget(_dim("Loosely similar"))
        self._count = QLabel("")
        self._count.setWordWrap(True)
        self._list = QListWidget()
        self._list.setMaximumHeight(180)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        box = QVBoxLayout(self)
        box.addWidget(intro)
        box.addLayout(ends)
        box.addWidget(self._threshold_told)
        box.addWidget(self._count)
        box.addWidget(self._list, stretch=1)
        box.addWidget(buttons)

    def _threshold(self, value: int) -> float:
        return 1.0 - value / 100 * (1.0 - REMOVE_SIMILAR_FLOOR)

    def _on_slider(self, value: int) -> None:
        threshold = self._threshold(value)
        self._threshold_told.setText(
            f"Similarity threshold: {threshold:.3f}")
        groups, pairs = playlist_doubles(self._paths, self._vectors,
                                         threshold)
        self._losers = {n for g in groups for n in g[1:]}
        self._losers.update(b for _, b, _ in pairs)
        n = len(self._losers)
        self._count.setText(
            f"<b>{n} track(s)</b> would be removed at this threshold."
            if n else "No tracks would be removed at this threshold.")
        self._list.clear()
        self._list.addItems(
            f"#{i + 1} · {Path(self._paths[i]).name}"
            for i in sorted(self._losers))

    def removable_paths(self) -> list[str]:
        return [self._paths[n] for n in sorted(self._losers)]


class PlaylistPanel(QWidget):
    """La sezione playlist, collegata allo stato: mostra `state.playlist`.

    Ogni mutazione passa da `state.set_playlist`, così la linea sulla mappa
    e questa tabella raccontano sempre la stessa fila. La selezione delle
    righe esce come `picked_changed`: è il canale playlist→seme — quello che
    si evidenzia qui diventa il punto di partenza delle proposte, senza
    toccare il seme del riquadro sopra la mappa.
    """

    picked_changed = Signal(list)

    def __init__(self, state: AppState, wire_table,
                 journal: Journal | None = None, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._journal = journal or Journal()
        self._lib: Library | None = None
        self._cost: TransitionCost | None = None    # pesi fermi (1,1,1)
        self._chapters: list[list[int]] | None = None
        self._keep_chapters_once = False
        self._picked: str | None = None             # la scheda evidenziata
        # Dove e come è andata l'ultima «Save as…»: il «Save» riscrive lì,
        # nello stesso formato, senza ripassare dal dialogo.
        self._saved_to: tuple[Path, Callable[[list[dict]], str]] | None = None
        # C'è qualcosa di nuovo da scrivere da quando si è salvato l'ultima
        # volta? Comanda «Save», che si spegne appena scrive — come in un
        # foglio Excel — e si riaccende alla prima mutazione vera.
        self._dirty = False
        self._board_seen_at = None
        self._build(wire_table)
        state.playlist_changed.connect(self._on_playlist_changed)

    # ------------------------------------------------------------------
    def _build(self, wire_table) -> None:
        self._title = QLabel("<b>Playlist</b>")
        self._play_all = QPushButton("▶ Play all")
        self._play_all.setToolTip("Plays the whole playlist in order, one "
                                  "track after another.")
        self._play_all.clicked.connect(self._on_play_all)
        self._sort = QPushButton("✨ Magic sort")
        self._sort.setToolTip("Reorders the whole playlist so every "
                              "transition stays cheap. Starts from the "
                              "first track.")
        self._sort.clicked.connect(self._on_magic_sort)
        self._drop = QPushButton("🗑 Remove ticked")
        self._drop.clicked.connect(self._on_drop)
        self._remove_similar = QPushButton("🎯 Remove similar…")
        self._remove_similar.setToolTip(
            "Opens a dialog with a similarity threshold and a live count, "
            "then removes every track it catches from the playlist — the "
            "audio files on disk are untouched.")
        self._remove_similar.clicked.connect(self._on_remove_similar)
        self._reset = QPushButton("🗑 Clear")
        self._reset.setToolTip("Clear the entire playlist.")
        self._reset.clicked.connect(lambda: self._push([], False))
        header = QHBoxLayout()
        header.addWidget(self._title, stretch=1)
        for button in (self._play_all, self._sort, self._drop,
                      self._remove_similar, self._reset):
            header.addWidget(button)

        self._empty = _dim(
            "Nothing in it yet: pick tracks in Build a set, take them from "
            "the disk, or load an existing playlist and keep adding to it.")

        self._table = TrackTable(reorderable=True, checkable=True,
                                 favouritable=True)
        wire_table(self._table)
        self._table.model_.order_changed.connect(
            lambda paths: self._push(list(paths), False))
        self._table.selection_paths_changed.connect(self.picked_changed.emit)

        # Il numero vivo in pagina, il come e il perché nel tooltip: lo
        # spazio qui è della tabella e della lavagna.
        self._worst = _dim("")
        self._worst.setToolTip(theme.hint(
            "The transition cost from each track to the next: 0 is "
            "seamless, 1 as far as this library goes. Magic sort is what "
            "brings the worst one down. Drag rows to reorder; the ✓ ticks "
            "are what Quick List and the Chain Maker start from."))

        # I possibili doppioni: la riga compare solo quando ce ne sono, il
        # resto del racconto sta nelle tinte delle righe e nei loro tooltip.
        self._doubles = _dim("")
        self._doubles.setToolTip(theme.hint(
            "Orange rows: the same song is already in the playlist at an "
            "earlier row — same name once numbering and (mix) notes are "
            "stripped. Violet rows: they sound nearly identical to an "
            "earlier row, whatever the names. Only the extra copies are "
            "tinted, the first occurrence stays clean: dropping every "
            "tinted row keeps one take of each. Still a question, not a "
            "verdict — hover a tinted row for its partner, listen, then "
            "tick what goes and 🗑 Remove ticked. If the better file is "
            "the tinted one, drop the clean row by hand instead."))
        self._doubles.setVisible(False)

        # Il Chapter Builder: creare, applicare, rifare.
        chapters_why = theme.hint(
            "Distribute the playlist across five emotional chapters of a "
            "DJ set: Intro, Buildup, Tension, Climax, Release. The shading "
            "on the board shows them; drag a card across a boundary to "
            "move it to another chapter.")
        self._ch_create = QPushButton("📖 Create chapters")
        self._ch_create.setToolTip(chapters_why)
        self._ch_create.clicked.connect(self._on_chapters_create)
        self._ch_apply = QPushButton("📖 Apply chapter order to playlist")
        self._ch_apply.clicked.connect(self._on_chapters_apply)
        self._ch_again = QPushButton("🔄 Re-assign chapters")
        self._ch_again.setToolTip(chapters_why)
        self._ch_again.clicked.connect(self._on_chapters_create)
        chapters_row = QHBoxLayout()
        for button in (self._ch_create, self._ch_apply, self._ch_again):
            chapters_row.addWidget(button)

        # La lavagna, con la misura dell'altezza. Cosa significhi l'altezza
        # scelta lo dice il tooltip della manopola; come si usa la lavagna,
        # quello dell'etichetta accanto.
        self._axis = QComboBox()
        self._axis.addItems(list(HEIGHT_FIELDS))
        self._axis.setCurrentText(DEFAULT_HEIGHT)
        self._axis.setToolTip(theme.hint(HEIGHT_MEANING[DEFAULT_HEIGHT]))
        self._axis.currentTextChanged.connect(lambda _: self._refresh_board())
        height_label = QLabel("Height means")
        height_label.setToolTip(theme.hint(
            "Left to right the set plays; how high a card sits is the "
            "measure picked here. Hover a point for its numbers, click to "
            "pick it — underneath, ▶ listens and the bin takes it out of "
            "the playlist. Drag a point sideways to move it in the set."))
        axis_row = QHBoxLayout()
        axis_row.addWidget(height_label)
        axis_row.addWidget(self._axis)
        axis_row.addStretch(1)

        self._board = BoardView()
        self._board.value_changed.connect(self._on_board_event)

        # La lavagna sta SOTTO la tabella, nella stessa scheda: è la vista
        # della playlist, e in una scheda a parte sembrava un accessorio dei
        # capitoli. I capitoli le stanno sopra, nella stessa riga della
        # misura dell'altezza, perché sono ciò che la lavagna colora.
        chapters_row.addLayout(axis_row)
        board_box = QWidget()
        bbox = QVBoxLayout(board_box)
        bbox.setContentsMargins(0, 0, 0, 0)
        bbox.setSpacing(6)
        bbox.addLayout(chapters_row)
        bbox.addWidget(self._board, stretch=1)

        table_box = QWidget()
        tbox = QVBoxLayout(table_box)
        tbox.setContentsMargins(0, 0, 0, 0)
        tbox.setSpacing(4)
        tbox.addWidget(self._table, stretch=1)
        tbox.addWidget(self._worst)
        tbox.addWidget(self._doubles)

        # Lo splitter lascia decidere quanta altezza dare alla lavagna: chi
        # riordina a mano vuole la tabella, chi guarda la forma la lavagna.
        self._playlist_controls = QSplitter(Qt.Orientation.Vertical)
        self._playlist_controls.addWidget(table_box)
        self._playlist_controls.addWidget(board_box)
        self._playlist_controls.setStretchFactor(0, 3)
        self._playlist_controls.setStretchFactor(1, 2)
        self._playlist_controls.setCollapsible(0, False)

        adding = QPushButton("🎵 Add tracks…")
        adding.setToolTip("Pick files from the disk: they go in after what "
                          "the playlist already holds. Only tracks already "
                          "on the map can join.")
        adding.clicked.connect(self._on_add_files)
        loading = QPushButton("📂 Load playlist…")
        loading.setToolTip("The .m3u8 this page exports, or one saved by "
                           "rekordbox, Serato, Traktor… Only the "
                           "track order is read.")
        loading.clicked.connect(self._on_load)
        self._save_again = QPushButton("💾 Save")
        self._save_again.clicked.connect(self._on_save_again)
        self._save_m3u8 = QPushButton("⬇ Save as playlist (M3U8)")
        self._save_m3u8.setToolTip("What rekordbox's Import Playlist "
                                   "accepts. Order and files only — no "
                                   "BPM, no cues.")
        self._save_m3u8.clicked.connect(self._on_save_m3u8)
        self._save_xml = QPushButton("⬇ Save as library (rekordbox XML)")
        self._save_xml.setToolTip("A library, not a playlist file: load it "
                                  "under Preferences ▸ Advanced ▸ Database "
                                  "▸ rekordbox xml. Carries the BPM.")
        self._save_xml.clicked.connect(self._on_save_xml)
        files_row = QHBoxLayout()
        for button in (adding, loading, self._save_again,
                       self._save_m3u8, self._save_xml):
            files_row.addWidget(button)
        self._refresh_save_again(has=False)

        box = QVBoxLayout(self)
        box.addLayout(header)
        box.addWidget(self._empty)
        box.addWidget(self._playlist_controls, stretch=1)
        box.addLayout(files_row)

    # ------------------------------------------------------------------
    # lo stato in mano
    # ------------------------------------------------------------------
    def set_library(self, lib: Library) -> None:
        self._lib = lib
        placed = lib.placed
        self._cost = TransitionCost(lib.store.embeddings[:placed],
                                    lib.frame["bpm"].tolist(),
                                    lib.frame["camelot"].tolist())
        # Gli indici dei capitoli appartenevano al frame di prima: se dopo
        # la ricarica non descrivono più la playlist, cadono.
        self._refresh()

    def indices(self) -> list[int]:
        """La playlist come posizioni sulla mappa, i dispersi fuori."""
        if self._lib is None:
            return []
        at_path = self._lib.at_path
        return [at_path[p] for p in self._state.playlist if p in at_path]

    def clear_picks(self) -> None:
        """Toglie le spunte senza rimandarle indietro come gesto: serve
        quando un clic sulla mappa — più recente — prende il comando."""
        self._table.clear_picks()

    def append(self, indices: list[int]) -> None:
        """In coda a quello che c'è già, saltando chi c'è già — e i saltati
        si raccontano: un'aggiunta assorbita in silenzio lascia il dubbio
        che non sia successo niente, o peggio che sia entrato un doppione."""
        frame = self._lib.frame
        merged, skipped = appended(list(self._state.playlist),
                                   [frame.at[i, "path"] for i in indices])
        self._push(merged, False)
        if skipped:
            self._tell_skipped(
                skipped, "Already in the playlist",
                f"{len(skipped)} track(s) already in the playlist — "
                "not added again.")

    def replace(self, indices: list[int]) -> None:
        """La lista nuova al posto della vecchia — ripulita dei percorsi
        ripetuti. Le liste di Build a set sono uniche per costruzione, ma
        un M3U8 salvato altrove può portare lo stesso file due volte (o due
        righe che il ripiego per nome risolve sullo stesso brano), e una
        playlist col path doppio qui non si governa: la spunta toglie tutte
        le sue righe, l'export lo riscriverebbe doppio."""
        frame = self._lib.frame
        merged, repeated = appended([],
                                    [frame.at[i, "path"] for i in indices])
        self._push(merged, False)
        if repeated:
            self._tell_skipped(
                repeated, "Listed more than once",
                f"{len(repeated)} track(s) listed more than once — kept "
                "once, at their first spot.")

    def _tell_skipped(self, skipped: list[str], title: str,
                      text: str) -> None:
        """Il resoconto dei file tenuti una volta sola: quali sono e a che
        riga stanno. DOPO il push apposta, così i numeri detti sono quelli
        che la tabella mostra dietro la finestra."""
        paths = [self._lib.frame.at[i, "path"] for i in self.indices()]
        lines = [(f"#{paths.index(p) + 1} · " if p in paths else "")
                 + Path(p).name for p in skipped]
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setText(text)
        shown = lines[:SKIPPED_SHOWN]
        if len(lines) > SKIPPED_SHOWN:
            shown.append(f"…and {len(lines) - SKIPPED_SHOWN} more — see "
                         "the details below.")
            box.setDetailedText("\n".join(lines))
        box.setInformativeText("\n".join(shown))
        box.exec()

    def _push(self, paths: list[str], keep_chapters: bool) -> None:
        before = list(self._state.playlist)
        self._keep_chapters_once = keep_chapters
        self._state.set_playlist(paths)
        if self._state.playlist == before:
            self._keep_chapters_once = False
            self._refresh()

    def _on_playlist_changed(self, _paths: list[str]) -> None:
        if not self._keep_chapters_once:
            # Una playlist riscritta non è più quella che i capitoli
            # descrivono — tranne quando è la loro stessa applicazione.
            self._chapters = None
        self._keep_chapters_once = False
        self._dirty = True
        self._refresh()

    # ------------------------------------------------------------------
    # il disegno
    # ------------------------------------------------------------------
    def _chapter_lookup(self, playlist: list[int]) -> dict[int, str] | None:
        if self._chapters is None:
            return None
        lookup = {}
        for ch, tracks in zip(CHAPTERS, self._chapters):
            for i in tracks:
                lookup[i] = ch["name"]
        return lookup if set(sum(self._chapters, [])) == set(playlist) \
            else None

    def _refresh(self) -> None:
        if self._lib is None:
            return
        playlist = self.indices()
        has = bool(playlist)
        self._empty.setVisible(not has)
        self._playlist_controls.setVisible(has)
        for button in (self._play_all, self._sort, self._drop,
                       self._remove_similar, self._reset,
                       self._save_m3u8, self._save_xml):
            button.setDisabled(not has)
        self._refresh_save_again(has)
        self._title.setText(
            f"<b>Playlist — {len(playlist)} track(s)</b>" if has
            else "<b>Playlist</b>")
        if not has:
            return

        frame, common = self._lib.frame, self._lib.common
        ch_lookup = self._chapter_lookup(playlist)
        table = playlist_rows(frame, self._cost, playlist, common, ch_lookup)
        self._table.set_tracks(
            table, genre_colors(frame, table["genres"], dark=theme.DARK))
        self._sort.setDisabled(len(playlist) < 3)

        # I possibili doppioni, ricalcolati a ogni giro: la playlist è corta
        # e il conto è banale, mentre una tinta rimasta da un giro prima
        # mentirebbe. Gli embedding coprono ogni riga per costruzione dello
        # store (righe e vettori si tengono al minimo comune in load).
        marks, doubled = double_marks(
            list(table["_path"]), self._lib.store.embeddings[playlist])
        self._table.set_marks(marks)
        self._doubles.setVisible(doubled is not None)
        self._doubles.setText(doubled or "")

        steps = [self._cost.between(a, b)
                 for a, b in zip(playlist, playlist[1:])]
        worst = max(steps, default=0)
        self._worst.setText(f"Roughest transition: <b>{worst:.3f}</b> · ⓘ")

        fresh = ch_lookup is None
        self._ch_create.setVisible(fresh)
        self._ch_create.setDisabled(len(playlist) < 5)
        self._ch_apply.setVisible(not fresh)
        self._ch_again.setVisible(not fresh)
        self._refresh_board()

    def _refresh_board(self) -> None:
        if self._lib is None:
            return
        playlist = self.indices()
        if not playlist:
            return
        frame, at_path, common = (self._lib.frame, self._lib.at_path,
                                  self._lib.common)
        axis = self._axis.currentText()
        self._axis.setToolTip(theme.hint(HEIGHT_MEANING[axis]))
        paths = [frame.at[i, "path"] for i in playlist]
        regions = board_chapter_regions(self._chapter_lookup(playlist),
                                        playlist)
        self._board.set_payload({
            **board_payload(frame, at_path, paths, axis, common, dark=theme.DARK),
            "selected": self._picked if self._picked in paths else None,
            "chapters": regions, "dark": theme.DARK})

    # ------------------------------------------------------------------
    # i gesti
    # ------------------------------------------------------------------
    def _on_play_all(self) -> None:
        frame = self._lib.frame
        paths = [frame.at[i, "path"] for i in self.indices()]
        if paths:
            self._state.play_queue(paths)

    def _on_magic_sort(self) -> None:
        playlist = self.indices()
        if len(playlist) >= 3:
            self.replace(magic_sort(self._cost, playlist,
                                    start=playlist[0]))

    def _on_drop(self) -> None:
        doomed = set(self._table.selected_paths())
        if doomed:
            self._push([p for p in self._state.playlist if p not in doomed],
                       False)

    def _on_remove_similar(self) -> None:
        playlist = self.indices()
        if len(playlist) < 2:
            return
        paths = [self._lib.frame.at[i, "path"] for i in playlist]
        dialog = SimilarityCleanupDialog(
            paths, self._lib.store.embeddings[playlist], self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            doomed = set(dialog.removable_paths())
            if doomed:
                self._push([p for p in self._state.playlist
                           if p not in doomed], False)

    def _on_board_event(self, value: dict) -> None:
        if value.get("at") == self._board_seen_at:
            return
        self._board_seen_at = value.get("at")
        kind, who = value.get("type"), value.get("id")
        playlist = self.indices()
        frame, at_path = self._lib.frame, self._lib.at_path
        paths = [frame.at[i, "path"] for i in playlist]
        if kind == "click" and who in paths:
            self._picked = who
            self._refresh_board()
        elif kind == "play" and who in paths:
            self._state.play(who)
        elif kind == "remove" and who in at_path:
            self._push([p for p in self._state.playlist if p != who], False)
        elif kind == "chapter_move" and who in at_path:
            self._on_chapter_move(at_path[who], value.get("from_chapter"),
                                  value.get("to_chapter"))
        elif kind == "move" and who in paths:
            where = value.get("to")
            if isinstance(where, int) and 0 <= where < len(paths):
                order = reordered(paths, {paths.index(who): where + 1})
                if order != paths:
                    self._push(order, False)

    # --- i capitoli ---
    def _on_chapters_create(self) -> None:
        playlist = self.indices()
        if len(playlist) >= 5:
            self._chapters = assign_chapters(self._lib.frame, playlist)
            self._refresh()

    def _on_chapters_apply(self) -> None:
        if self._chapters is None:
            return
        frame = self._lib.frame
        ordered = sum(self._chapters, [])
        # L'ordine appena scritto È i capitoli srotolati: le aree colorate
        # non devono sparire proprio quando l'accordo è più vero che mai.
        self._push([frame.at[i, "path"] for i in ordered], True)

    def _on_chapter_move(self, track: int, src: str | None,
                         dst: str | None) -> None:
        if self._chapters is None:
            return
        names = [ch["name"] for ch in CHAPTERS]
        if src not in names or dst not in names:
            return
        src_i, dst_i = names.index(src), names.index(dst)
        if track in self._chapters[src_i]:
            self._chapters[src_i].remove(track)
            self._chapters[dst_i].append(track)
            frame = self._lib.frame
            ordered = sum(self._chapters, [])
            self._push([frame.at[i, "path"] for i in ordered], True)

    # --- i file ---
    def _on_add_files(self) -> None:
        chosen, _ = QFileDialog.getOpenFileNames(
            self, "Choose tracks for the playlist", "", AUDIO_FILTER)
        if not chosen:
            return
        found, missing = playlist_positions(chosen, self._lib.at_path)
        if found:
            self.append(found)
        if missing:
            names = ", ".join(Path(p).name for p in missing)
            QMessageBox.warning(
                self, "Not on the map",
                f"Not on the map, so not in the playlist: {names}.\n"
                "Add their folder under Map settings, then try again.")

    def _on_load(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Load a playlist", "",
            "Playlists (*.m3u8 *.m3u);;All files (*)")
        if not chosen:
            return
        try:
            text = Path(chosen).read_text("utf-8", errors="replace")
        except OSError as trouble:
            QMessageBox.warning(self, "Unreadable file", str(trouble))
            return
        paths = read_m3u8(text)
        if not paths:
            QMessageBox.warning(self, "Empty playlist",
                                "No tracks in that file.")
            return
        found, missing = playlist_positions(paths, self._lib.at_path)
        box = QMessageBox(self)
        box.setWindowTitle("Load the playlist")
        box.setText(f"{len(found)} of {len(paths)} track(s) are on the map.")
        if missing:
            box.setInformativeText(
                f"{len(missing)} track(s) are not on the map and cannot "
                "join — add their folder under Map settings. Details below.")
            box.setDetailedText("\n".join(missing))
        replace = box.addButton("Load as new playlist",
                                QMessageBox.ButtonRole.AcceptRole)
        append = box.addButton("Append to playlist",
                               QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        if not found:
            return
        if box.clickedButton() is replace:
            self.replace(found)
        elif box.clickedButton() is append:
            self.append(found)

    def _tracks_for_export(self) -> list[dict]:
        frame = self._lib.frame
        tracks = []
        for i in self.indices():
            path = Path(frame.at[i, "path"])
            title, artist = read_title_artist(path)
            tracks.append({"path": path, "name": title, "artist": artist,
                           "bpm": frame.at[i, "bpm"],
                           "duration": frame.at[i, "duration"],
                           "genre": frame.at[i, "top_genre"], "cues": []})
        return tracks

    def _save_as(self, build: Callable[[list[dict]], str],
                 default_name: str, title: str, wanted: str) -> None:
        chosen, _ = QFileDialog.getSaveFileName(self, title, default_name,
                                                wanted)
        if chosen:
            self._saved_to = (Path(chosen), build)
            self._write_out()

    def _write_out(self) -> None:
        path, build = self._saved_to
        path.write_text(build(self._tracks_for_export()), "utf-8")
        # Una playlist salvata è una sequenza voluta: l'appunto che vale di
        # più per chi vorrà imparare "cosa viene dopo".
        self._journal.record("playlist_saved", file=str(path),
                             format=path.suffix.lstrip("."),
                             paths=list(self._state.playlist))
        # Come in un foglio Excel: appena scritto non c'è più niente di
        # nuovo da riscrivere, e il bottone lo dice spegnendosi. Torna vivo
        # alla prima mutazione vera della playlist — vedi `_on_playlist_changed`.
        self._dirty = False
        self._refresh_save_again(has=bool(self.indices()))

    def _refresh_save_again(self, has: bool) -> None:
        known = self._saved_to is not None
        self._save_again.setEnabled(has and known and self._dirty)
        if not known:
            told = ("Enabled once the playlist has been saved with one of "
                    "the «Save as…» buttons: then it rewrites that same "
                    "file.")
        elif self._dirty:
            told = (f"Writes the playlist again to {self._saved_to[0]}, in "
                    "the same format, without asking.")
        else:
            told = f"Nothing changed since the last save to {self._saved_to[0]}."
        self._save_again.setToolTip(told)

    def _on_save_again(self) -> None:
        self._write_out()

    def _on_save_m3u8(self) -> None:
        self._save_as(build_m3u8, "djcaddy_playlist.m3u8",
                      "Save the playlist", "Playlist (*.m3u8)")

    def _on_save_xml(self) -> None:
        self._save_as(build_rekordbox_xml, "djcaddy_library.xml",
                      "Save the rekordbox library", "rekordbox XML (*.xml)")
