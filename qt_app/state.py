"""Lo stato condiviso dell'app: quello che in Streamlit era session_state.

Un oggetto solo con dei segnali, ed è il motivo strutturale per cui Qt sarà
più reattivo: un clic sulla mappa aggiorna in-process i due widget collegati
al segnale, invece di rieseguire l'intero script e riserializzare tabelle e
figure sul websocket come fa Streamlit.

I brani si ricordano per PERCORSO, non per indice: è la stessa scelta delle
chiavi di sessione Streamlit, e per la stessa ragione — la mappa si
ricostruisce, si filtra, si riproietta, e un indice ricordato da un giro
all'altro indicherebbe presto un brano diverso. Il percorso resta quello.
Gli indici si ricavano al momento, da chi ha in mano il frame.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class AppState(QObject):
    """Seed, selezione, playlist, brano in ascolto — con un segnale a testa.

    I setter emettono solo se il valore cambia davvero: i widget si
    ridisegnano quando c'è qualcosa di nuovo da disegnare, non a ogni
    gesto che conferma quello che già mostrano.
    """

    seed_changed = Signal(object)          # str | None
    selection_changed = Signal(list)       # list[str]
    playlist_changed = Signal(list)        # list[str]
    now_playing_changed = Signal(object)   # str | None
    analysis_running_changed = Signal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.seed: str | None = None
        self.selection: list[str] = []
        self.playlist: list[str] = []
        self.now_playing: str | None = None
        self._running_jobs: set[str] = set()
        self._queue: list[str] = []   # quello che resta da suonare in fila

    # --- seed e selezione ---
    def set_seed(self, path: str | None) -> None:
        """Un brano indicato. Seed e gruppo si escludono, come sulla mappa
        Streamlit: un cerchio che sopravvive al gesto successivo indica una
        scelta che non si sta più facendo."""
        if self.selection:
            self.selection = []
            self.selection_changed.emit([])
        if path != self.seed:
            self.seed = path
            self.seed_changed.emit(path)

    def set_selection(self, paths: list[str]) -> None:
        """Un gruppo preso con lasso o riquadro. Da due in su è un gruppo e
        il seme se ne va; un punto solo è un seme comunque lo si sia preso."""
        if len(paths) == 1:
            self.set_seed(paths[0])
            return
        if self.seed is not None:
            self.seed = None
            self.seed_changed.emit(None)
        if paths != self.selection:
            self.selection = list(paths)
            self.selection_changed.emit(self.selection)

    def clear_selection(self) -> None:
        if self.selection:
            self.selection = []
            self.selection_changed.emit([])

    # --- playlist ---
    def set_playlist(self, paths: list[str]) -> None:
        if paths != self.playlist:
            self.playlist = list(paths)
            self.playlist_changed.emit(self.playlist)

    # --- ascolto ---
    def play(self, path: str) -> None:
        self._queue = []           # un ▶ singolo interrompe la fila in corso
        if path != self.now_playing:
            self.now_playing = path
            self.now_playing_changed.emit(path)

    def stop(self) -> None:
        self._queue = []
        if self.now_playing is not None:
            self.now_playing = None
            self.now_playing_changed.emit(None)

    def play_queue(self, paths: list[str]) -> None:
        """Suona `paths` in fila: uno dopo l'altro, nell'ordine dato."""
        if not paths:
            return
        self._queue = list(paths[1:])
        first = paths[0]
        if first != self.now_playing:
            self.now_playing = first
            self.now_playing_changed.emit(first)
        else:
            self.advance()          # stesso brano in testa: passa al successivo

    def advance(self) -> None:
        """Il brano in ascolto è finito: il prossimo della fila, o lo stop."""
        if self._queue:
            self.now_playing = self._queue.pop(0)
            self.now_playing_changed.emit(self.now_playing)
        else:
            self.stop()

    # --- job in corso (mappa, tag): fa pulsare il marchio in home ---
    @property
    def analysis_running(self) -> bool:
        return bool(self._running_jobs)

    def set_job_running(self, name: str, running: bool) -> None:
        was_running = self.analysis_running
        if running:
            self._running_jobs.add(name)
        else:
            self._running_jobs.discard(name)
        if self.analysis_running != was_running:
            self.analysis_running_changed.emit(self.analysis_running)
