"""La pagina Wave analysis: un brano, la sua onda, le sue frasi da rivedere.

Stessa sostanza della pagina Streamlit — analisi (o cache) del singolo
brano, onda a bande di frequenza con marker e regioni cantate, tabella cue
correggibile, export CSV/rekordbox, scrittura in djay Pro — con l'audio nel
lettore in fondo alla finestra invece che in un tag <audio> dentro la
pagina: l'onda qui sopra e le barre del dock raccontano lo stesso ascolto.

Scarti deliberati dalla lettera di Streamlit, come in Fase 3:
- il cambio di soglia vocale rigenera daccapo start e tag delle righe
  vocali (di là un edit sopravviveva al cambio soglia su un id riciclato,
  cioè finiva su una regione diversa);
- niente download-button: CSV e XML chiedono dove salvare col dialogo di
  sistema.
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QSlider, QVBoxLayout,
                               QWidget)

from core.analysis.audio_features import ANALYSIS_SR, load_audio
from core.analysis.cue_export import (DJAY_SLOTS, build_cue_rows,
                                      is_vocal_row, marker_color,
                                      plan_djay_markers)
from core.analysis.dj_export import build_rekordbox_xml, read_title_artist
from core.analysis.engine import AUDIO_EXTENSIONS, analyze_file, load_analysis
from core.analysis.models import format_elapsed, format_remaining
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
            f"QPushButton {{ background: {theme.PRIMARY}; color: white; }}")
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

        self._vocals_only = QCheckBox("Save only vocal tags")
        self._vocals_only.setToolTip(
            "If checked, CSV/XML export and the djay Pro write only include "
            "the vocal start/end rows, skipping the phrase tags.")
        self._vocals_only.toggled.connect(lambda _: self._refresh_export())
        self._csv = QPushButton("⬇ Save cues (CSV)")
        self._csv.clicked.connect(self._on_save_csv)
        self._xml = QPushButton("⬇ Export rekordbox XML")
        self._xml.setToolTip(
            "Import directly into rekordbox, or convert to Serato/Traktor/"
            "djay Pro with a third-party tool (DJ Conversion Utility, MIXO, "
            "Lexicon).")
        self._xml.clicked.connect(self._on_save_xml)
        self._export_row = QWidget()
        export_row = QHBoxLayout(self._export_row)
        export_row.setContentsMargins(0, 0, 0, 0)
        export_row.addWidget(self._vocals_only, stretch=1)
        export_row.addWidget(self._csv)
        export_row.addWidget(self._xml)
        self._export_row.setVisible(False)

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
        box.addWidget(self._export_row)
        if sys.platform == "darwin":
            box.addWidget(self._build_djay())

    def _build_djay(self) -> QWidget:
        """La scrittura diretta in djay Pro: solo dove djay Pro vive."""
        self._overwrite = QCheckBox("Overwrite tags for analyzed songs")
        self._overwrite.setToolTip(
            "If checked, existing cues/loops on this track in djay Pro are "
            "replaced by the rows above instead of adding to them. Removing "
            "is best-effort — a full backup is taken, check the preview.")
        self._overwrite.toggled.connect(lambda _: self._invalidate_preview())
        self._preview_btn = QPushButton("Preview djay Pro update")
        self._preview_btn.setToolTip(
            f"Phrase starts become hot cues (pad position sets the colour), "
            f"each vocal region becomes one saved loop. Two banks of "
            f"{DJAY_SLOTS}, handed out in time order — the djay slot column "
            "shows where each row lands.")
        self._preview_btn.clicked.connect(self._on_djay_preview)
        self._preview_btn.setEnabled(False)
        self._write_btn = QPushButton("Write to djay Pro now")
        self._write_btn.setStyleSheet(
            f"QPushButton {{ background: {theme.PRIMARY}; color: white; }}")
        self._write_btn.clicked.connect(self._on_djay_write)
        self._write_btn.setVisible(False)
        self._djay_told = dim("")
        self._djay_told.setVisible(False)

        block = QWidget()
        box = QVBoxLayout(block)
        box.setContentsMargins(0, 6, 0, 0)
        box.setSpacing(6)
        title = QLabel("<b>Update your djay Pro library</b>")
        row = QHBoxLayout()
        row.addWidget(self._overwrite, stretch=1)
        row.addWidget(self._preview_btn)
        row.addWidget(self._write_btn)
        box.addWidget(title)
        box.addLayout(row)
        box.addWidget(self._djay_told)
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
        for widget in (self._wave, self._caption, self._table,
                       self._export_row):
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
        self._refresh_export()

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
    # export
    # ------------------------------------------------------------------
    def _export_rows(self) -> list[dict]:
        if self._vocals_only.isChecked():
            return [r for r in self._shown
                    if r["kind"] in ("vocal_start", "vocal_end")]
        return list(self._shown)

    def _refresh_export(self) -> None:
        rows = self._export_rows()
        self._csv.setEnabled(bool(rows))
        self._xml.setEnabled(bool(rows))
        if hasattr(self, "_preview_btn"):
            self._preview_btn.setEnabled(bool(rows))

    def _on_save_csv(self) -> None:
        rows = self._export_rows()
        if not rows:
            return
        duration = self._track.get("duration")
        stem = Path(self._track["name"]).stem
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Save cues (CSV)", str(Path.home() / f"{stem}_cues.csv"),
            "CSV (*.csv)")
        if not chosen:
            return
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["Tag", "Start", "Beats", "from_start", "remaining"])
        for r in rows:
            writer.writerow([
                r["tag"], r["start"],
                "" if r["beats"] is None else r["beats"],
                format_elapsed(r["start"]),
                format_remaining(r["start"], duration)])
        try:
            Path(chosen).write_text(out.getvalue())
            self._say(f"Cues saved to {chosen}.", _OK)
        except OSError as trouble:
            self._say(f"Could not save the CSV: {trouble}", theme.PRIMARY)

    def _on_save_xml(self) -> None:
        rows = self._export_rows()
        if not rows:
            return
        stem = Path(self._track["name"]).stem
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Export cues to rekordbox XML",
            str(Path.home() / f"{stem}_rekordbox.xml"), "XML (*.xml)")
        if not chosen:
            return
        path = Path(self._track["path"])
        title, artist = read_title_artist(path)
        xml = build_rekordbox_xml([{
            "path": path, "name": title, "artist": artist,
            "genre": self._track["genre"], "bpm": self._track["bpm"],
            "duration": self._track.get("duration"),
            "cues": [{"name": r["tag"], "start": float(r["start"]),
                      "color": marker_color(r["kind"], r["tag"])}
                     for r in rows],
        }])
        try:
            Path(chosen).write_text(xml, encoding="utf-8")
            self._say(f"Rekordbox XML saved to {chosen}.", _OK)
        except OSError as trouble:
            self._say(f"Could not save the XML: {trouble}", theme.PRIMARY)

    # ------------------------------------------------------------------
    # djay Pro (solo macOS: i bottoni esistono solo lì)
    # ------------------------------------------------------------------
    def _invalidate_preview(self) -> None:
        """Ogni edit cambia il piano: il preview mostrato non vale più."""
        if self._preview is None:
            return
        self._preview = None
        self._write_btn.setVisible(False)
        self._djay_told.setVisible(False)

    def _djay_plan(self):
        from core.analysis.djay_write import CuePoint, LoopRegion
        plan = plan_djay_markers([
            {"id": r["id"], "kind": r["kind"], "start": float(r["start"])}
            for r in self._export_rows()])
        cues = [CuePoint(time=start, pad=pad) for _, pad, start in plan.cues]
        loops = [LoopRegion(start=start, end=end, slot=slot)
                 for _, slot, start, end in plan.loops]
        return plan, cues, loops

    def _on_djay_preview(self) -> None:
        if self._track is None:
            return
        from core.analysis.djay_write import preview_write
        plan, cues, loops = self._djay_plan()
        overwrite = self._overwrite.isChecked()
        path = Path(self._track["path"])
        self._preview_btn.setEnabled(False)

        def _job():
            return preview_write(path, cues, overwrite=overwrite,
                                 new_loops=loops)

        def _done(result) -> None:
            self._preview_btn.setEnabled(True)
            self._preview = {"result": result, "cues": cues, "loops": loops,
                             "overwrite": overwrite}
            self._show_preview(plan)

        def _failed(trouble: Exception) -> None:
            self._preview_btn.setEnabled(True)
            self._djay_told.setText(str(trouble))
            self._djay_told.setStyleSheet(f"color: {theme.PRIMARY};")
            self._djay_told.setVisible(True)

        run_in_pool(_job, _done, _failed)

    def _show_preview(self, plan) -> None:
        from core.analysis.djay_write import is_djay_running
        preview = self._preview
        res = preview["result"]
        cues, loops = preview["cues"], preview["loops"]
        verb = ("would replace" if preview["overwrite"]
                else "would be added to")
        told = [f"Track found in djay Pro ({len(res.cues_before)} existing "
                f"cue(s), {len(res.loops_before)} existing loop(s)). "
                f"{len(cues)} hot cue(s) and {len(loops)} loop(s) {verb} "
                "them."]
        told.extend(f"Cue {c.pad + 1} → {format_elapsed(c.time)}"
                    for c in cues)
        told.extend(f"Loop {lr.slot + 1} → {format_elapsed(lr.start)}–"
                    f"{format_elapsed(lr.end)}" for lr in loops)
        if plan.dropped:
            told.append(f"⚠ {len(plan.dropped)} row(s) don't fit: djay Pro "
                        f"only has {DJAY_SLOTS} hot-cue pads and "
                        f"{DJAY_SLOTS} loop slots.")
        if plan.unpaired:
            told.append(f"⚠ {len(plan.unpaired)} vocal marker(s) have no "
                        "matching start/end and can't become a loop.")
        if preview["overwrite"]:
            told.append("⚠ Overwrite is unverified for the removing case — "
                        "a full backup is still taken, check this preview "
                        "carefully.")
        if is_djay_running():
            told.append("⚠ djay Pro appears to be running — quit it before "
                        "writing, to avoid conflicts with its own "
                        "auto-save.")
        self._djay_told.setText("\n".join(told))
        self._djay_told.setStyleSheet("")
        self._djay_told.setVisible(True)
        self._write_btn.setVisible(True)

    def _on_djay_write(self) -> None:
        if self._preview is None or self._track is None:
            return
        from core.analysis.djay_write import write_new_cues
        preview, self._preview = self._preview, None
        path = Path(self._track["path"])
        self._write_btn.setEnabled(False)

        def _job():
            return write_new_cues(path, preview["cues"],
                                  overwrite=preview["overwrite"],
                                  new_loops=preview["loops"])

        def _done(result) -> None:
            self._write_btn.setEnabled(True)
            self._write_btn.setVisible(False)
            self._djay_told.setText(f"{result.message} Backup saved to: "
                                    f"{result.backup_path}")
            self._djay_told.setStyleSheet(f"color: {_OK};")

        def _failed(trouble: Exception) -> None:
            self._write_btn.setEnabled(True)
            self._djay_told.setText(f"Write failed, nothing was changed: "
                                    f"{trouble}")
            self._djay_told.setStyleSheet(f"color: {theme.PRIMARY};")

        run_in_pool(_job, _done, _failed)
