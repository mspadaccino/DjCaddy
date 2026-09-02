"""Costruire la mappa come job lungo, staccato dall'interfaccia.

Stessa forma del job di tagging, e per lo stesso motivo: a qualche secondo
per brano una libreria intera sono giorni, e non possono dipendere da una
scheda del browser aperta. Il lavoro scrive man mano nella cartella della
mappa, quindi si può fermare quando si vuole e ripartire da lì.

Lo STATO è quello del tagging, riusato tale e quale (`JobState`): le
domande sono le stesse — a che punto è, quanti ne mancano, quanto ci
vuole — e averne due versioni vorrebbe dire correggerle due volte.

La proiezione UMAP NON fa parte del job: è un calcolo sulla libreria
intera, dura minuti, e va rifatto una volta sola alla fine invece che a
ogni brano.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ..bundle import resources, state_dir
from .essentia_tags import find_taggable
from .map_profile import ProfileSettings, profile_many
from .map_store import MapStore, default_store_dir
from .tag_job import JobState, load_state  # noqa: F401  (riesportato per la pagina)

# Radice del repo (dentro il bundle: la cartella dei dati inclusi).
PROJECT_ROOT = resources()

# Lo stato si SCRIVE, quindi non sta con il codice ma dove si puo' scrivere:
# accanto ad esso nel repo, in ~/.cache/djcaddy dentro il bundle.
DEFAULT_MAP_STATE_FILE = state_dir() / ".djcaddy_map_job.json"

# Percorso di map_cli.py, che resta in root: unica definizione, così chi
# lancia il job (la pagina Map) non se lo ricalcola per conto suo.
MAP_CLI_PATH = PROJECT_ROOT / "map_cli.py"

# Dove finisce quello che il job stampa. Una sola definizione, perché la usano
# in due: chi lancia il job e chi ci apre sopra un terminale.
DEFAULT_MAP_LOG = Path(tempfile.gettempdir()) / "djcaddy_map_job.log"


def build_queue(folder: Path, store: MapStore) -> list[Path]:
    """I brani della cartella che non sono ancora sulla mappa."""
    return store.pending(find_taggable(folder))


def run_job(folder: Path, settings: ProfileSettings | None = None, workers: int = 1,
            store_dir: Path | None = None, state_file: Path = DEFAULT_MAP_STATE_FILE,
            limit: int = 0, on_progress=None) -> JobState:
    """Profila la cartella e scrive nella mappa, un brano per volta."""
    store = MapStore.load(store_dir or default_store_dir())
    state = JobState(pid=os.getpid(), folder=str(folder), started_at=time.time())
    state.save(state_file)

    queue = build_queue(folder, store)
    if limit:
        queue = queue[:limit]
    state.total = len(queue)
    state.save(state_file)

    last = time.time()
    for profile in profile_many(queue, settings, workers=workers):
        now = time.time()
        state.tick(now - last)
        last = now
        state.done += 1
        state.current = profile.path.name
        if profile.error is None:
            store.append([profile])
            state.written += 1
        else:
            state.failed += 1
            if len(state.errors) < 100:
                state.errors.append({"file": str(profile.path),
                                     "error": profile.error})
        state.save(state_file)
        if on_progress is not None:
            on_progress(state)

    state.finished_at = time.time()
    state.current = ""
    state.save(state_file)
    return state


def load_map_state(path: Path = DEFAULT_MAP_STATE_FILE) -> JobState | None:
    return load_state(path)


# --------------------------------------------------------------------------
# Governare il job da fuori
# --------------------------------------------------------------------------

def process_state(pid: int) -> str:
    """"running", "paused" o "gone".

    Fermo non vuol dire morto: un processo che ha ricevuto SIGSTOP è ancora
    lì con tutta la sua memoria (i modelli sono gigabyte: ricaricarli costa),
    semplicemente non gira. `ps` lo segna con la lettera T.
    """
    if not pid:
        return "gone"
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return "gone"
    try:
        out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return "running"
    return "paused" if out.stdout.strip().startswith("T") else "running"


def _signal(pid: int, sig) -> bool:
    """Manda un segnale al job. False se non c'era più nessuno da colpire.

    Si colpisce il GRUPPO, non il singolo processo: l'analisi gira su più
    figli, e fermare solo il padre li lascerebbe a macinare a vuoto. Ma solo
    se il pid è capofila del suo gruppo — cioè se il job è stato staccato da
    noi. Un job lanciato a mano dentro una shell sta nel gruppo di QUELLA
    shell, e colpire il gruppo vorrebbe dire fermare anche il terminale di
    chi l'ha lanciato.
    """
    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, sig)
        else:
            os.kill(pid, sig)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def pause_job(pid: int) -> bool:
    return _signal(pid, signal.SIGSTOP)


def resume_job(pid: int) -> bool:
    return _signal(pid, signal.SIGCONT)


def stop_job(pid: int) -> bool:
    """Ferma il job per sempre. Quello che è sulla mappa ci resta, e la volta
    dopo si riparte da lì: è tutto il senso di scrivere man mano."""
    return _signal(pid, signal.SIGTERM)


def caffeinated(command: list[str]) -> list[str]:
    """Il comando, con la macchina tenuta sveglia per tutta la sua durata.

    Un Mac che si addormenta CONGELA il job: resta vivo, non lavora, e al
    risveglio riprende come se niente fosse. Misurato su una ricostruzione
    della mappa: quindici ore di vita per meno di tre di lavoro.

    `caffeinate` prende un comando come argomento e lascia cadere la sveglia
    da solo appena quello finisce — non resta niente da spegnere a mano né
    da sorvegliare. `-i` tiene a bada il sonno da inattività; il coperchio
    chiuso dorme lo stesso, e non c'è modo di impedirlo.
    """
    if sys.platform != "darwin":
        return command
    return ["caffeinate", "-i", *command]


@contextlib.contextmanager
def awake():
    """Tiene la macchina sveglia finché dura il blocco, non un secondo di più.

    Stessa sveglia di `caffeinated` e stessa ragione — un Mac addormentato
    congela il lavoro invece di rallentarlo — ma per chi il lavoro lo fa
    dentro di sé invece di lanciarlo come sottoprocesso. `caffeinate -w`
    aspetta un PID e se ne va da solo quando quello sparisce: anche se il job
    venisse ucciso di brutto, non resta una sveglia accesa a impedire alla
    macchina di dormire per sempre.

    Il rischio che copre non è teorico. Il backfill dell'energia e' nato
    senza, e su una libreria vera l'attesa era passata da otto ore a sedici
    per la sola ragione che nessuno toccava la tastiera.

    Se `caffeinate` non c'è — un altro sistema operativo, o un Mac che non lo
    espone — non succede niente: il lavoro parte lo stesso, solo senza
    sveglia.
    """
    if sys.platform != "darwin":
        yield
        return
    try:
        keeper = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    except OSError:
        yield
        return
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            keeper.terminate()


def open_monitor(log: Path = DEFAULT_MAP_LOG) -> None:
    """Apre Terminal.app su `tail -f` del log del job.

    Non c'è un modo per mostrare un terminale dentro una pagina web, e non
    serve: l'app gira in locale, il Terminale è lì. Stessa scorciatoia del
    selettore di cartelle, che chiede al Finder invece di inventarsi un
    browser di file.
    """
    command = f"tail -f {shlex.quote(str(log))}"
    # Staccato e senza aspettare l'esito: la prima volta macOS chiede il
    # permesso di comandare il Terminale, e quel dialogo resta lì finché non
    # lo si guarda. Aspettarlo vorrebbe dire piantare la pagina, o dichiarare
    # fallito un comando che partirà appena si dice di sì.
    subprocess.Popen(
        ["osascript",
         "-e", f'tell application "Terminal" to do script "{command}"',
         "-e", 'tell application "Terminal" to activate'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
