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

from pathlib import Path

import pandas as pd

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel,
                               QMessageBox, QPushButton, QSplitter,
                               QVBoxLayout, QWidget)

from core.analysis.dj_export import (build_m3u8, build_rekordbox_xml,
                                     playlist_positions, read_m3u8,
                                     read_title_artist)
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


class PlaylistPanel(QWidget):
    """La sezione playlist, collegata allo stato: mostra `state.playlist`.

    Ogni mutazione passa da `state.set_playlist`, così la linea sulla mappa
    e questa tabella raccontano sempre la stessa fila. La selezione delle
    righe esce come `picked_changed`: è il canale playlist→seme — quello che
    si evidenzia qui diventa il punto di partenza delle proposte, senza
    toccare il seme del riquadro sopra la mappa.
    """

    picked_changed = Signal(list)

    def __init__(self, state: AppState, wire_table, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._lib: Library | None = None
        self._cost: TransitionCost | None = None    # pesi fermi (1,1,1)
        self._chapters: list[list[int]] | None = None
        self._keep_chapters_once = False
        self._picked: str | None = None             # la scheda evidenziata
        self._board_seen_at = None
        self._build(wire_table)
        state.playlist_changed.connect(self._on_playlist_changed)

    # ------------------------------------------------------------------
    def _build(self, wire_table) -> None:
        self._title = QLabel("<b>Playlist</b>")
        self._sort = QPushButton("✨ Magic sort")
        self._sort.setToolTip("Reorders the whole playlist so every "
                              "transition stays cheap. Starts from the "
                              "first track.")
        self._sort.clicked.connect(self._on_magic_sort)
        self._drop = QPushButton("🗑 Remove ticked")
        self._drop.clicked.connect(self._on_drop)
        self._reset = QPushButton("🗑 Clear")
        self._reset.setToolTip("Clear the entire playlist.")
        self._reset.clicked.connect(lambda: self._push([], False))
        header = QHBoxLayout()
        header.addWidget(self._title, stretch=1)
        for button in (self._sort, self._drop, self._reset):
            header.addWidget(button)

        self._empty = _dim(
            "Nothing in it yet: pick tracks in Build a set, take them from "
            "the disk, or load an existing playlist and keep adding to it.")

        self._table = TrackTable(reorderable=True, checkable=True)
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
        chapters_row.addStretch(1)

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
        board_box = QWidget()
        bbox = QVBoxLayout(board_box)
        bbox.setContentsMargins(0, 0, 0, 0)
        bbox.setSpacing(4)
        bbox.addLayout(axis_row)
        bbox.addWidget(self._board, stretch=1)

        self._split = QSplitter(Qt.Orientation.Vertical)
        table_box = QWidget()
        tbox = QVBoxLayout(table_box)
        tbox.setContentsMargins(0, 0, 0, 0)
        tbox.setSpacing(4)
        tbox.addWidget(self._table, stretch=1)
        tbox.addWidget(self._worst)
        tbox.addLayout(chapters_row)
        self._split.addWidget(table_box)
        self._split.addWidget(board_box)
        self._split.setSizes([420, 340])

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
        for button in (adding, loading, self._save_m3u8, self._save_xml):
            files_row.addWidget(button)

        box = QVBoxLayout(self)
        box.addLayout(header)
        box.addWidget(self._empty)
        box.addWidget(self._split, stretch=1)
        box.addLayout(files_row)

    # ------------------------------------------------------------------
    # lo stato in mano
    # ------------------------------------------------------------------
    def set_library(self, lib: Library) -> None:
        self._lib = lib
        placed = lib.placed
        self._cost = TransitionCost(lib.store.coords[:placed],
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
        """In coda a quello che c'è già, saltando chi c'è già."""
        frame = self._lib.frame
        current = list(self._state.playlist)
        for i in indices:
            path = frame.at[i, "path"]
            if path not in current:
                current.append(path)
        self._push(current, False)

    def replace(self, indices: list[int]) -> None:
        frame = self._lib.frame
        self._push([frame.at[i, "path"] for i in indices], False)

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
        self._split.setVisible(has)
        for button in (self._sort, self._drop, self._reset,
                       self._save_m3u8, self._save_xml):
            button.setDisabled(not has)
        self._title.setText(
            f"<b>Playlist — {len(playlist)} track(s)</b>" if has
            else "<b>Playlist</b>")
        if not has:
            return

        frame, common = self._lib.frame, self._lib.common
        ch_lookup = self._chapter_lookup(playlist)
        table = playlist_rows(frame, self._cost, playlist, common, ch_lookup)
        self._table.set_tracks(
            table, genre_colors(frame, table["genres"], dark=True))
        self._sort.setDisabled(len(playlist) < 3)

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
            **board_payload(frame, at_path, paths, axis, common, dark=True),
            "selected": self._picked if self._picked in paths else None,
            "chapters": regions, "dark": True})

    # ------------------------------------------------------------------
    # i gesti
    # ------------------------------------------------------------------
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
        replace = box.addButton("Send to playlist",
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

    def _save(self, data: str, default_name: str, title: str,
              wanted: str) -> None:
        chosen, _ = QFileDialog.getSaveFileName(self, title, default_name,
                                                wanted)
        if chosen:
            Path(chosen).write_text(data, "utf-8")

    def _on_save_m3u8(self) -> None:
        self._save(build_m3u8(self._tracks_for_export()),
                   "wavecut_playlist.m3u8", "Save the playlist",
                   "Playlist (*.m3u8)")

    def _on_save_xml(self) -> None:
        self._save(build_rekordbox_xml(self._tracks_for_export()),
                   "wavecut_library.xml", "Save the rekordbox library",
                   "rekordbox XML (*.xml)")
