"""Il lettore in fondo alla finestra: l'equivalente del dock di st.bottom.

Stessa fila del dock Streamlit — il tondo di play, il nome sopra l'onda,
l'orologio, la ✕ — e stessa onda (i peaks arrivano da
`core.analysis.waveform.envelope`, che è il conto condiviso). Cambia solo il
motore: QMediaPlayer legge mp3 e flac direttamente da file, quindi
spariscono i data-URI e i transcode che si facevano per il browser.

Il profilo dell'onda si calcola in un thread del pool: ffmpeg ci mette
qualche decimo di secondo, e il brano deve partire subito — l'onda compare
quando è pronta, e se ffmpeg manca non compare affatto, senza impedire
l'ascolto (stessa scelta del dock Streamlit).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
                               QWidget)

from core.analysis import waveform
from qt_app import theme
from qt_app.state import AppState
from qt_app.widgets.waveform import WaveformBar
from qt_app.workers import run_in_pool

PLAY_GLYPH = "▶"
PAUSE_GLYPH = "❙❙"

# Quanti profili d'onda tenere in memoria: come la cache del dock Streamlit.
CACHED_WAVES = 64


def clock_text(seconds: float) -> str:
    """`m:ss`, come lo scrive il dock. Un numero non finito legge 0:00."""
    if not (seconds == seconds and seconds != float("inf")):  # NaN o inf
        seconds = 0
    minutes = int(seconds // 60)
    return f"{minutes}:{int(seconds % 60):02d}"


class PlayerDock(QWidget):
    """La fila del lettore, collegata allo stato: suona `now_playing`.

    Vive sotto le tab, fuori da tutte: è così che resta su ogni pagina.
    Quando non c'è niente in ascolto si nasconde, come il dock che sparisce
    con la ✕.
    """

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._path: str | None = None
        self._duration = 0.0            # dall'onda; il player fa da riserva
        self._waves: dict[tuple, tuple] = {}

        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._player.setAudioOutput(self._audio)

        self._build()
        self.hide()

        state.now_playing_changed.connect(self._on_now_playing)
        self._player.positionChanged.connect(self._on_position)
        self._player.playbackStateChanged.connect(self._on_playback_state)
        self._player.durationChanged.connect(self._on_duration)

    def _build(self) -> None:
        self.setObjectName("player")
        self.setStyleSheet(
            f"QWidget#player {{ background: {theme.RAISED};"
            f" border-radius: 8px; }}")

        self._toggle = QPushButton(PLAY_GLYPH)
        self._toggle.setFixedSize(36, 36)
        self._toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle.setStyleSheet(
            f"QPushButton {{ background: {theme.PRIMARY}; color: white;"
            f" border-radius: 18px; font-size: 13px; padding: 0; }}")
        self._toggle.clicked.connect(self._on_toggle)

        # Il nome sta DENTRO il riquadro dell'onda, non come riga a parte:
        # in fondo allo schermo ogni riga in più è spazio tolto alla pagina.
        self._name = QLabel("")
        self._name.setObjectName("dim")
        self._name.setStyleSheet(f"color: {theme.FADED}; font-size: 11px;")

        self._wave = WaveformBar()
        self._wave.seek_requested.connect(self._on_seek)

        wavebox = QVBoxLayout()
        wavebox.setContentsMargins(0, 0, 0, 0)
        wavebox.setSpacing(3)
        wavebox.addWidget(self._name)
        wavebox.addWidget(self._wave)

        self._clock = QLabel("0:00 / 0:00")
        self._clock.setStyleSheet(f"color: {theme.FADED}; font-size: 11px;")
        self._clock.setMinimumWidth(84)
        self._clock.setAlignment(Qt.AlignmentFlag.AlignRight
                                 | Qt.AlignmentFlag.AlignVCenter)

        close = QPushButton("✕")
        close.setFlat(True)
        close.setCursor(Qt.CursorShape.PointingHandCursor)
        close.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {theme.FADED};"
            f" border: none; }} QPushButton:hover {{ color: {theme.INK}; }}")
        close.clicked.connect(self._state.stop)

        row = QHBoxLayout(self)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(12)
        row.addWidget(self._toggle)
        row.addLayout(wavebox, stretch=1)
        row.addWidget(self._clock)
        row.addWidget(close)

    # --- lo stato comanda ---
    def _on_now_playing(self, path: str | None) -> None:
        if path is None:
            self._player.stop()
            self._path = None
            self.hide()
            return
        if path == self._path:
            # Il ▶ della riga già in ascolto non fa ripartire niente: come
            # nel dock, dove lo stesso indirizzo non si ricarica.
            self.show()
            return
        self._path = path
        track = Path(path)
        self.show()
        if not track.exists():
            self._player.stop()
            self._wave.clear()
            self._name.setText(f"Not there any more: {track.name}")
            self._clock.setText("0:00 / 0:00")
            return
        self._name.setText(track.name)
        self._duration = 0.0
        self._wave.clear()
        self._player.setSource(QUrl.fromLocalFile(path))
        self._player.play()
        self._load_wave(track)

    def _load_wave(self, track: Path) -> None:
        try:
            stat = track.stat()
            key = (str(track), stat.st_mtime, stat.st_size)
        except OSError:
            return
        if key in self._waves:
            self._on_wave(str(track), self._waves[key])
            return

        def _job(path=str(track)):
            return path, waveform.envelope(path)

        run_in_pool(_job, self._on_wave_ready)

    def _on_wave_ready(self, result) -> None:
        path, wave = result
        if wave is None:
            return                      # niente ffmpeg o file illeggibile
        try:
            stat = Path(path).stat()
            self._waves[(path, stat.st_mtime, stat.st_size)] = wave
            while len(self._waves) > CACHED_WAVES:
                self._waves.pop(next(iter(self._waves)))
        except OSError:
            pass
        self._on_wave(path, wave)

    def _on_wave(self, path: str, wave: tuple) -> None:
        if path != self._path:
            return                      # nel frattempo si ascolta altro
        peaks, seconds = wave
        self._duration = seconds
        self._wave.set_wave(peaks, seconds)
        self._on_position(self._player.position())

    # --- il player racconta ---
    def _total(self) -> float:
        """La durata per orologio e onda: quella di ffmpeg quando c'è —
        è quella su cui sono tagliati i peaks — il player da riserva."""
        return self._duration or self._player.duration() / 1000.0

    def _on_position(self, milliseconds: int) -> None:
        seconds = milliseconds / 1000.0
        self._wave.set_position(seconds)
        self._clock.setText(
            f"{clock_text(seconds)} / {clock_text(self._total())}")

    def _on_duration(self, milliseconds: int) -> None:
        self._on_position(self._player.position())

    def _on_playback_state(self, playing) -> None:
        paused = playing != QMediaPlayer.PlaybackState.PlayingState
        self._toggle.setText(PLAY_GLYPH if paused else PAUSE_GLYPH)

    # --- i gesti ---
    def _on_toggle(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _on_seek(self, seconds: float) -> None:
        self._player.setPosition(int(seconds * 1000))
