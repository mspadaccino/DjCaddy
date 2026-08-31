"""Il lavoro di tagging come job lungo, staccato dall'interfaccia.

A 4 secondi per brano una libreria intera sono giorni: non può dipendere da
una scheda del browser aperta. Qui il lavoro sta in una funzione sola, che
scrive man mano DUE cose su disco:

- i tag nei file, uno per uno, senza aspettare la fine;
- uno stato in JSON, che chiunque può leggere per sapere a che punto è.

Da questo discende che il job è interrompibile in qualsiasi momento: quello
che è stato scritto resta scritto, il registro dei fatti si aggiorna a ogni
brano, e la ripartenza salta ciò che risulta già fatto.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .essentia_tags import (
    TagSettings,
    analyze_many,
    find_taggable,
    scan_coverage,
    write_tags,
)
from .tag_tracking import ProcessedTracker

DEFAULT_STATE_FILE = Path(__file__).resolve().parent.parent.parent / ".djcaddy_tag_job.json"
# Il CLI che fa girare il job, centralizzato come MAP_CLI_PATH in map_job:
# lo lanciano sia la pagina Streamlit sia quella Qt, da cartelle diverse.
TAG_CLI_PATH = Path(__file__).resolve().parent.parent.parent / "tag_cli.py"

# Fra un brano e il successivo non passa un minuto nemmeno sul file più
# ostico: se è passato, la macchina dormiva o il job era in pausa, e quel
# tempo non va contato come lavoro.
MAX_TRACK_SECONDS = 60.0
# Su quanti minuti di lavoro si fa la media che stima quanto manca.
RECENT_MINUTES = 30


@dataclass
class JobState:
    """Quello che un altro processo può sapere del job leggendo un file."""
    pid: int = 0
    folder: str = ""
    started_at: float = 0.0
    updated_at: float = 0.0
    finished_at: float | None = None
    total: int = 0
    done: int = 0
    written: int = 0
    failed: int = 0
    skipped: int = 0
    current: str = ""
    errors: list[dict] = field(default_factory=list)
    stopped_reason: str | None = None
    # Il ritmo recente in secchielli da un minuto: [minuto, brani, secondi].
    recent: list[list[float]] = field(default_factory=list)

    @property
    def running(self) -> bool:
        """Vivo davvero, non solo "non ha ancora scritto la fine".

        Un job ucciso non fa in tempo a segnalarlo: senza questo controllo
        l'interfaccia mostrerebbe per sempre un lavoro fermo come attivo.
        """
        if self.finished_at is not None or not self.pid:
            return False
        try:
            os.kill(self.pid, 0)
        except (ProcessLookupError, PermissionError):
            return False
        return True

    def tick(self, seconds: float) -> None:
        """Un brano è finito, ed è costato `seconds`.

        Serve perché la media dall'inizio non è una misura del ritmo: conta
        anche le ore in cui il job era congelato. Su una ricostruzione della
        mappa dava 4,73 s a brano contro gli 0,85 reali — il Mac aveva
        dormito, e l'attesa dichiarata era di 32 ore invece di 6.
        """
        if seconds > MAX_TRACK_SECONDS:
            return                          # dormiva: non è tempo di lavoro
        minute = int(time.time() // 60)
        if self.recent and self.recent[-1][0] == minute:
            self.recent[-1][1] += 1
            self.recent[-1][2] += seconds
        else:
            self.recent.append([minute, 1, seconds])
        # Gli ultimi trenta minuti IN CUI HA LAVORATO, non gli ultimi trenta
        # dell'orologio: dopo una notte di sonno il ritmo di ieri sera è
        # ancora la stima migliore che abbiamo.
        del self.recent[:-RECENT_MINUTES]

    @property
    def seconds_each(self) -> float:
        """Secondi per brano sul lavoro recente.

        Si ripiega sulla media dall'inizio solo finché non c'è abbastanza
        storia — o quando si legge lo stato di un job vecchio, scritto prima
        che questo esistesse.
        """
        tracks = sum(r[1] for r in self.recent)
        if tracks:
            return sum(r[2] for r in self.recent) / tracks
        elapsed = (self.finished_at or self.updated_at) - self.started_at
        return elapsed / self.done if self.done else 0.0

    @property
    def eta_seconds(self) -> float:
        return max(0, self.total - self.done) * self.seconds_each

    def save(self, path: Path) -> None:
        self.updated_at = time.time()
        # Si scrive di fianco e poi si rinomina: chi legge nello stesso
        # istante trova il file vecchio intero, mai mezzo nuovo.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=1))
        tmp.replace(path)


def load_state(path: Path = DEFAULT_STATE_FILE) -> JobState | None:
    """Lo stato dell'ultimo job, o None se non ne è mai partito uno."""
    try:
        return JobState(**json.loads(path.read_text()))
    except (FileNotFoundError, ValueError, TypeError):
        return None


def build_queue(folder: Path, settings: TagSettings, only_missing: bool = True,
                skip_done: bool = True, tracker: ProcessedTracker | None = None,
                progress=None) -> list[Path]:
    """I brani su cui il job lavorerà.

    Due filtri distinti e volutamente separati: `only_missing` guarda i FILE
    (chi ha già genere e commento non serve rifarlo), `skip_done` guarda il
    REGISTRO di cosa è stato tentato. Il primo è la verità, il secondo è la
    memoria — e non coincidono: di 400 percorsi dati per fatti, 30 non
    esistevano più.
    """
    files = find_taggable(folder)
    if only_missing:
        report = scan_coverage(files, progress=progress)
        files = [c.path for c in report.missing(
            genre=settings.genres, comment=settings.moods)]
    if skip_done:
        files = (tracker or ProcessedTracker()).pending(files)
    return files


def run_job(folder: Path, settings: TagSettings, workers: int = 1,
            only_missing: bool = True, skip_done: bool = True,
            state_file: Path = DEFAULT_STATE_FILE, limit: int = 0,
            on_progress=None) -> JobState:
    """Analizza e SCRIVE i tag, aggiornando lo stato a ogni brano.

    Qui si scrive subito, senza l'anteprima che ha la pagina: su migliaia di
    brani nessuno rilegge una tabella prima di salvare, e tenere tutto in
    memoria in attesa di un clic vanificherebbe il fatto di essere staccati.
    """
    tracker = ProcessedTracker()
    state = JobState(pid=os.getpid(), folder=str(folder), started_at=time.time())
    state.save(state_file)

    queue = build_queue(folder, settings, only_missing, skip_done, tracker)
    if limit:
        queue = queue[:limit]
    state.total = len(queue)
    state.save(state_file)

    last = time.time()
    for path, tags, error in analyze_many(queue, settings, workers=workers):
        now = time.time()
        state.tick(now - last)
        last = now
        state.done += 1
        state.current = path.name
        if error is None:
            try:
                if write_tags(path, tags, settings):
                    state.written += 1
                else:
                    state.skipped += 1
                tracker.mark(path)
            except Exception as e:                      # noqa: BLE001
                error = f"scrittura: {type(e).__name__}: {e}"
        if error is not None:
            state.failed += 1
            # Si tengono i primi cento: servono a capire CHE COSA fallisce,
            # e un elenco che cresce senza fine renderebbe illeggibile lo
            # stato e pesante ogni riscrittura.
            if len(state.errors) < 100:
                state.errors.append({"file": str(path), "error": error})
        state.save(state_file)
        if on_progress is not None:
            on_progress(state)

    state.finished_at = time.time()
    state.current = ""
    state.save(state_file)
    return state
