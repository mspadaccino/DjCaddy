"""La pagina Cue analysis: un brano, la sua onda, le sue frasi da rivedere.

Stessa sostanza della pagina Streamlit — analisi (o cache) del singolo
brano, onda a bande di frequenza con marker e regioni cantate, tabella cue
correggibile, scrittura dei cue nel brano — con l'audio nel lettore in
fondo alla finestra invece che in un tag <audio> dentro la pagina: l'onda
qui sopra e le barre del dock raccontano lo stesso ascolto.

Scarti deliberati dalla lettera di Streamlit, come in Fase 3:
- il cambio di soglia vocale rigenera daccapo start e tag delle righe
  vocali (di là un edit sopravviveva al cambio soglia su un id riciclato,
  cioè finiva su una regione diversa);
- niente export CSV né rekordbox XML: una collection di UN brano non serve
  a nessuno, e i cue si scrivono direttamente nel brano.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QSlider, QVBoxLayout,
                               QWidget)

from core.analysis.audio_features import ANALYSIS_SR, load_audio
from core.analysis.cue_export import (RB_HOT_CUES, build_cue_rows,
                                      is_vocal_row, marker_color,
                                      plan_rekordbox_markers)
from core.analysis.engine import AUDIO_EXTENSIONS, analyze_file, load_analysis
from core.analysis.rekordbox_write import available as rekordbox_available
from core.analysis.models import format_elapsed
from core.analysis.vocals import VOCAL_FLOOR
from core.analysis.vocals import available as vocals_available
from core.analysis.vocals import vocal_regions
from core.analysis.waveform import compute_frequency_waveform
from qt_app import theme
from qt_app.pages.common import dim
from qt_app.state import AppState
from qt_app.widgets.player_dock import PlayerDock
from qt_app.widgets.wave_review import WaveReview
from qt_app.workers import run_in_pool

from .cue_table import CueTable, view_rows

# Quante onde a frequenza tenere pronte: sono ~40 KB l'una e la pagina
# lavora su un brano per volta — bastano gli ultimi visti.
CACHED_WAVES = 8

_OK = "#3fbf7f"
_WARN = "#ffb454"


class WavePage(QWidget):
    """Revisione di frasi, vocali e hot cue di un singolo brano."""

    def __init__(self, state: AppState, player: PlayerDock,
                 parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._player = player
        self._track: dict | None = None
        self._floor = VOCAL_FLOOR
        self._deleted_rows: set[str] = set()
        self._deleted_vocal: dict[str, set[str]] = {}
        self._starts: dict[str, float] = {}
        self._labels: dict[str, str] = {}
        self._analysis_end: dict[str, float] = {}
        self._shown: list[dict] = []
        self._waves: dict[tuple, tuple] = {}
        self._preview: dict | None = None

        self._build()
        player.position_changed.connect(self._on_position)

    # ------------------------------------------------------------------
    # costruzione
    # ------------------------------------------------------------------
    def _build(self) -> None:
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Track path (mp3/flac)")
        self._path_edit.returnPressed.connect(self._on_analyze)
        browse = QPushButton("🎵 Browse…")
        browse.clicked.connect(self._on_browse)
        path_row = QHBoxLayout()
        path_row.addWidget(self._path_edit, stretch=1)
        path_row.addWidget(browse)

        self._analyze = QPushButton("Analyze")
        self._analyze.setStyleSheet(
            theme.primary_button())
        self._analyze.clicked.connect(self._on_analyze)
        self._force = QCheckBox("Force analysis if exists")
        self._force.setToolTip("Re-analyze even if a <name>_analysis.json "
                               "file already exists")
        self._vocals = QCheckBox("Detect vocals (Demucs, slow)")
        self._vocals.setChecked(vocals_available())
        self._vocals.setEnabled(vocals_available())
        if not vocals_available():
            self._vocals.setToolTip("Demucs not installed: vocal detection "
                                    "disabled. Install it with `poetry "
                                    "install`.")
        run_row = QHBoxLayout()
        run_row.addWidget(self._analyze)
        run_row.addWidget(self._force)
        run_row.addWidget(self._vocals)
        run_row.addStretch(1)

        self._status = dim("Pick a track and press “Analyze” to start. "
                           "Analysis reuses the same cache as the CLI.")
        self._told = QLabel("")
        self._told.setTextFormat(Qt.TextFormat.RichText)
        self._told.setVisible(False)

        self._floor_label = QLabel("")
        self._floor_slider = QSlider(Qt.Orientation.Horizontal)
        self._floor_slider.setRange(0, 100)
        self._floor_slider.setValue(round(VOCAL_FLOOR * 100))
        self._floor_slider.setToolTip(
            "Vocal threshold (voice/mix dominance). Higher = fewer sung "
            "regions. Recomputed instantly, no Demucs.")
        self._floor_slider.valueChanged.connect(self._on_floor)
        self._floor_row = QWidget()
        floor_row = QHBoxLayout(self._floor_row)
        floor_row.setContentsMargins(0, 0, 0, 0)
        floor_row.addWidget(self._floor_label)
        floor_row.addWidget(self._floor_slider, stretch=1)
        self._floor_row.setVisible(False)

        self._wave = WaveReview()
        self._wave.seek_requested.connect(self._on_seek)
        self._wave.setVisible(False)
        self._caption = dim("Click the waveform to jump to a point, or ▶ a "
                            "row below · yellow line = playhead, pink bands "
                            "= vocal regions.")
        self._caption.setVisible(False)
        self._clock = QLabel("")
        self._clock.setStyleSheet(f"color: {theme.FADED}; font-size: 11px;")
        caption_row = QHBoxLayout()
        caption_row.addWidget(self._caption, stretch=1)
        caption_row.addWidget(self._clock)

        self._table = CueTable()
        self._table.play_clicked.connect(self._on_play_row)
        self._table.delete_clicked.connect(self._on_delete_row)
        self._table.tag_edited.connect(self._on_tag_edit)
        self._table.start_edited.connect(self._on_start_edit)
        self._table.start_rejected.connect(self._on_bad_start)
        self._table.setVisible(False)

        # Le due spunte di cosa scrivere: vivono nel blocco della scrittura,
        # perché adesso quello è il solo posto dove le righe vanno a finire.
        self._vocals_only = QCheckBox("Save only vocal tags")
        self._vocals_only.setToolTip(
            "If checked, only the vocal start/end rows are written, "
            "skipping the phrase tags.")
        self._vocals_only.toggled.connect(lambda _: self._refresh_cues_out())

        box = QVBoxLayout(self)
        box.setContentsMargins(8, 8, 8, 8)
        box.setSpacing(6)
        box.addLayout(path_row)
        box.addLayout(run_row)
        box.addWidget(self._status)
        box.addWidget(self._told)
        box.addWidget(self._floor_row)
        box.addWidget(self._wave)
        box.addLayout(caption_row)
        box.addWidget(self._table, stretch=1)
        box.addWidget(self._build_write())

    def _build_write(self) -> QWidget:
        """Il blocco che porta i cue nella libreria di rekordbox.

        Non è più legato a macOS come il vecchio blocco djay: rekordbox sta
        anche su Windows, e `rekordbox_write.available()` dice da sé se su
        questo computer si può scrivere e altrimenti perché no.
        """
        self._replace = QCheckBox("Replace existing cues")
        self._replace.setToolTip(
            "Applied by Write cues to track: the cues and loops rekordbox "
            "already has on this song are removed and the rows above take "
            "their place — pads included. Off, the new ones join what is "
            "there, taking only the free pads. Changing it asks for a "
            "fresh preview, because the preview is computed with it.")
        self._replace.toggled.connect(lambda _: self._invalidate_preview())
        self._preview_btn = QPushButton("Preview cues")
        self._preview_btn.setToolTip(
            f"Phrase starts become hot cues on the {RB_HOT_CUES} pads (A-H) "
            "in time order, and memory cues once the pads run out — so "
            "nothing is dropped. Each vocal region becomes a saved loop. "
            "The slot column shows where every row lands.")
        self._preview_btn.clicked.connect(self._on_preview)
        self._preview_btn.setEnabled(False)
        self._write_btn = QPushButton("Write cues to track")
        self._write_btn.setStyleSheet(theme.primary_button())
        self._write_btn.clicked.connect(self._on_write)
        self._write_btn.setVisible(False)
        self._write_told = dim("")
        self._write_told.setVisible(False)

        block = QWidget()
        box = QVBoxLayout(block)
        box.setContentsMargins(0, 6, 0, 0)
        box.setSpacing(6)
        title = QLabel("<b>Write the cues into your rekordbox library</b>")
        row = QHBoxLayout()
        row.addWidget(self._vocals_only)
        row.addWidget(self._replace, stretch=1)
        row.addWidget(self._preview_btn)
        row.addWidget(self._write_btn)
        box.addWidget(title)
        box.addLayout(row)
        box.addWidget(self._write_told)

        can, why = rekordbox_available()
        if not can:
            # Meglio dirlo da fermi che scoprirlo al primo clic.
            self._preview_btn.setEnabled(False)
            self._preview_btn.setToolTip(why)
            self._say_write(f"Cues cannot be written here: {why}",
                            theme.FADED)
        self._writable = can
        return block

    # ------------------------------------------------------------------
    # dire le cose
    # ------------------------------------------------------------------
    def _say(self, text: str, color: str | None = None) -> None:
        self._status.setText(text)
        self._status.setStyleSheet(
            f"color: {color};" if color else f"color: {theme.FADED};")

    # ------------------------------------------------------------------
    # analisi
    # ------------------------------------------------------------------
    def _on_browse(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Choose a track", "", "Audio (*.mp3 *.flac);;All files (*)")
        if chosen:
            self._path_edit.setText(chosen)

    def _on_analyze(self) -> None:
        text = self._path_edit.text().strip()
        src = Path(text).expanduser()
        if not text or not src.is_file():
            self._say("Invalid track: please provide the path to a file.",
                      theme.PRIMARY)
            return
        if src.suffix.lower() not in AUDIO_EXTENSIONS:
            self._say(f"Unsupported format ({src.suffix}). Use mp3 or flac.",
                      theme.PRIMARY)
            return
        force = self._force.isChecked()
        vocals = self._vocals.isChecked()
        self._analyze.setEnabled(False)
        self._say(f"Analyzing {src.name}… (the first run with vocals can "
                  "take a few minutes)")

        def _job(src=src, force=force, vocals=vocals):
            existing = None if force else load_analysis(src)
            if existing is not None:
                return existing, True
            return analyze_file(src, use_cache=not force,
                                detect_vocals=vocals), False

        run_in_pool(_job, self._on_analyzed, self._on_analyze_failed)

    def _on_analyze_failed(self, trouble: Exception) -> None:
        self._analyze.setEnabled(True)
        self._say(f"The analysis failed: {trouble}", theme.PRIMARY)

    def _on_analyzed(self, result) -> None:
        self._analyze.setEnabled(True)
        t, from_file = result
        self._track = {
            "path": str(t.path), "name": t.path.name, "genre": t.genre,
            "vibe": t.vibe, "bpm": t.bpm, "duration": t.duration,
            "error": t.error, "sections": [s.to_dict() for s in t.sections],
            "vocal_regions": [list(r) for r in t.vocal_regions],
            "vocal_ratio": t.vocal_ratio, "vocal_fps": t.vocal_fps,
        }
        # Il brano nuovo riparte pulito: cancellazioni ed edit erano suoi.
        self._deleted_rows = set()
        self._deleted_vocal = {}
        self._starts, self._labels = {}, {}
        self._invalidate_preview()

        self._say(f"Loaded from {Path(self._track['name']).stem}"
                  f"_analysis.json: {self._track['name']}" if from_file
                  else f"Analyzed: {self._track['name']}", _OK)
        bpm = self._track["bpm"]
        bpm_txt = f"{bpm:.0f}" if bpm is not None else "N/A"
        self._told.setText(
            f"<b>{self._track['genre']}</b> — {self._track['vibe']} — "
            f"BPM {bpm_txt}"
            + (f"<br><span style='color:{_WARN}'>Analysis warning: "
               f"{self._track['error']}</span>" if self._track["error"]
               else ""))
        self._told.setVisible(True)

        has_env = bool(self._track.get("vocal_ratio"))
        self._floor_row.setVisible(has_env)
        if has_env:
            self._floor_slider.blockSignals(True)
            self._floor_slider.setValue(round(VOCAL_FLOOR * 100))
            self._floor_slider.blockSignals(False)
        self._floor = VOCAL_FLOOR
        self._show_floor()

        showable = bool(self._track["duration"])
        for widget in (self._wave, self._caption, self._table):
            widget.setVisible(showable)
        if not showable:
            self._say("Duration unavailable: player can't be shown.", _WARN)
            return
        self._wave.set_wave([], [], float(self._track["duration"]))
        self._wave.set_position(0.0)
        self._clock.setText("")
        self._load_wave()
        self._refresh_cues()

    # ------------------------------------------------------------------
    # l'onda a frequenza
    # ------------------------------------------------------------------
    def _load_wave(self) -> None:
        path = self._track["path"]
        try:
            stat = Path(path).stat()
            key = (path, stat.st_mtime, stat.st_size)
        except OSError:
            key = (path, 0, 0)
        if key in self._waves:
            self._apply_wave(path, self._waves[key])
            return

        def _job(path=path, key=key):
            y, sr = load_audio(Path(path), sr=ANALYSIS_SR, mono=True)
            _, amp, colors = compute_frequency_waveform(y, sr, 1600)
            return key, path, [round(float(a), 3) for a in amp], list(colors)

        run_in_pool(_job, self._on_wave_ready,
                    lambda t: self._say(f"The waveform could not be drawn: "
                                        f"{t}", _WARN))

    def _on_wave_ready(self, result) -> None:
        key, path, amp, colors = result
        self._waves[key] = (amp, colors)
        while len(self._waves) > CACHED_WAVES:
            self._waves.pop(next(iter(self._waves)))
        self._apply_wave(path, (amp, colors))

    def _apply_wave(self, path: str, wave: tuple) -> None:
        if self._track is None or path != self._track["path"]:
            return                      # nel frattempo si guarda altro
        amp, colors = wave
        self._wave.set_wave(amp, colors, float(self._track["duration"]))
        self._wave.set_regions(self._regions_live())

    # ------------------------------------------------------------------
    # righe cue
    # ------------------------------------------------------------------
    def _regions_live(self) -> list[tuple[float, float]]:
        """Le regioni cantate alla soglia corrente, dall'inviluppo in cache."""
        ratio = self._track.get("vocal_ratio") or []
        fps = self._track.get("vocal_fps")
        if ratio and fps:
            times = np.arange(len(ratio)) / fps
            return vocal_regions((times, np.asarray(ratio, dtype=float)),
                                 floor=self._floor)
        return [tuple(r) for r in self._track.get("vocal_regions", [])]

    def _base_rows(self) -> list[dict]:
        regions = self._regions_live()
        default_rows = build_cue_rows(self._track["sections"], regions,
                                      self._track["bpm"])
        deleted_vocal = self._deleted_vocal.setdefault(
            f"{self._floor:.2f}", set())
        self._analysis_end = {}
        rows = []
        for r in default_rows:
            rid = r["id"]
            if is_vocal_row(r):
                if rid in deleted_vocal:
                    continue
            elif rid in self._deleted_rows:
                continue
            self._starts.setdefault(rid, r["start"])
            self._labels.setdefault(rid, r["label"])
            if r["end"] is not None:
                self._analysis_end[rid] = r["end"]
            rows.append({"id": rid, "kind": r["kind"],
                         "tag": self._labels[rid],
                         "start": self._starts[rid]})
        return rows

    def _refresh_cues(self) -> None:
        self._shown = view_rows(self._base_rows(), self._track["bpm"],
                                self._analysis_end)
        self._table.set_rows(self._shown)
        self._wave.set_markers([
            {"t": r["start"], "label": r["tag"],
             "color": marker_color(r["kind"], r["tag"])}
            for r in self._shown])
        self._wave.set_regions(self._regions_live())
        self._invalidate_preview()
        self._refresh_cues_out()

    def _on_floor(self, value: int) -> None:
        self._floor = value / 100.0
        self._show_floor()
        # La soglia governa le regioni vocali: toccarla le ricalcola daccapo,
        # edit compresi — un id riciclato (`vs0`) indica una regione diversa.
        for rid in [r for r in self._starts if r.startswith(("vs", "ve"))]:
            del self._starts[rid]
        for rid in [r for r in self._labels if r.startswith(("vs", "ve"))]:
            del self._labels[rid]
        if self._track is not None and self._track.get("duration"):
            self._refresh_cues()

    def _show_floor(self) -> None:
        self._floor_label.setText(
            f"Vocal threshold (voice/mix dominance): {self._floor:.2f}")

    def _on_tag_edit(self, rid: str, label: str) -> None:
        self._labels[rid] = label
        self._refresh_cues()

    def _on_start_edit(self, rid: str, seconds: float) -> None:
        self._starts[rid] = seconds
        self._refresh_cues()

    def _on_bad_start(self) -> None:
        self._say("Formato non valido (atteso mm:ss): valore precedente "
                  "mantenuto.", _WARN)
        self._refresh_cues()

    def _on_delete_row(self, rid: str) -> None:
        if rid.startswith(("vs", "ve")):
            self._deleted_vocal.setdefault(
                f"{self._floor:.2f}", set()).add(rid)
        else:
            self._deleted_rows.add(rid)
        self._refresh_cues()

    # ------------------------------------------------------------------
    # ascolto
    # ------------------------------------------------------------------
    def _on_play_row(self, rid: str) -> None:
        if self._track is not None:
            self._player.play_at(self._track["path"],
                                 self._starts.get(rid, 0.0))

    def _on_seek(self, seconds: float) -> None:
        if self._track is not None:
            self._player.play_at(self._track["path"], seconds)

    def _on_position(self, seconds: float) -> None:
        if self._track is None or self._player.path != self._track["path"]:
            return
        self._wave.set_position(seconds)
        duration = float(self._track.get("duration") or 0.0)
        self._clock.setText(f"{format_elapsed(seconds)} / "
                            f"{format_elapsed(duration)}")

    # ------------------------------------------------------------------
    # cosa esce dalla tabella
    # ------------------------------------------------------------------
    def _export_rows(self) -> list[dict]:
        if self._vocals_only.isChecked():
            return [r for r in self._shown
                    if r["kind"] in ("vocal_start", "vocal_end")]
        return list(self._shown)

    def _refresh_cues_out(self) -> None:
        """Senza righe non c'è niente da scrivere: il preview si spegne."""
        if hasattr(self, "_preview_btn"):
            self._preview_btn.setEnabled(
                bool(self._export_rows()) and self._writable)

    # ------------------------------------------------------------------
    # la scrittura nella libreria di rekordbox
    # ------------------------------------------------------------------
    def _invalidate_preview(self) -> None:
        """Ogni edit cambia il piano: il preview mostrato non vale più."""
        if self._preview is None:
            return
        self._preview = None
        self._write_btn.setVisible(False)
        self._write_told.setVisible(False)

    def _marker_plan(self):
        return plan_rekordbox_markers([
            {"id": r["id"], "kind": r["kind"], "start": float(r["start"]),
             "label": r["tag"]}
            for r in self._export_rows()])

    def _on_preview(self) -> None:
        if self._track is None:
            return
        from core.analysis.rekordbox_write import preview_write
        plan = self._marker_plan()
        replace = self._replace.isChecked()
        path = Path(self._track["path"])
        self._preview_btn.setEnabled(False)

        def _job():
            return preview_write(path, plan.markers, replace=replace)

        def _done(result) -> None:
            self._preview_btn.setEnabled(True)
            self._preview = {"result": result, "markers": plan.markers,
                             "replace": replace}
            self._show_preview(plan, result)

        def _failed(trouble: Exception) -> None:
            self._preview_btn.setEnabled(True)
            self._say_write(str(trouble), theme.PRIMARY)

        run_in_pool(_job, _done, _failed)

    def _say_write(self, text: str, color: str | None = None) -> None:
        self._write_told.setText(text)
        self._write_told.setStyleSheet(f"color: {color};" if color else "")
        self._write_told.setVisible(True)

    def _show_preview(self, plan, res) -> None:
        from core.analysis.rekordbox_write import is_rekordbox_running
        verb = "replace" if self._preview["replace"] else "join"
        told = [f"«{res.title}» is in your rekordbox library "
                f"({res.cues_before} cue(s), {res.loops_before} loop(s) "
                f"already there). {res.written} marker(s) would {verb} "
                f"them: {res.hot_cues} hot cue(s), {res.memory_cues} "
                f"memory cue(s), {res.loops} loop(s)."]
        if res.pads_taken and res.memory_cues:
            told.append(
                f"Pads {', '.join(chr(ord('A') + p - 1) for p in res.pads_taken)} "
                "are already used by rekordbox, so what did not fit on a "
                "free pad becomes a memory cue — nothing is dropped. Tick "
                "Replace existing cues to take the pads back.")
        if plan.unpaired:
            told.append(f"⚠ {len(plan.unpaired)} vocal marker(s) have no "
                        "matching start/end and can't become a loop.")
        if is_rekordbox_running():
            told.append("⚠ rekordbox is running — quit it before writing, "
                        "or its own save will overwrite this.")
        self._say_write("\n".join(told))
        self._write_btn.setVisible(True)

    def _on_write(self) -> None:
        if self._preview is None or self._track is None:
            return
        from core.analysis.rekordbox_write import write_cues
        preview, self._preview = self._preview, None
        path = Path(self._track["path"])
        self._write_btn.setEnabled(False)

        def _job():
            return write_cues(path, preview["markers"],
                              replace=preview["replace"])

        def _done(result) -> None:
            self._write_btn.setEnabled(True)
            self._write_btn.setVisible(False)
            self._say_write(f"{result.message} Backup of the library: "
                            f"{result.backup_path}", _OK)

        def _failed(trouble: Exception) -> None:
            self._write_btn.setEnabled(True)
            self._say_write(f"Write failed, nothing was changed: {trouble}",
                            theme.PRIMARY)

        run_in_pool(_job, _done, _failed)
