"""Dove stanno le cose quando l'app gira impacchettata.

Fuori dal bundle non cambia niente: `frozen()` è falso e ogni funzione qui
restituisce esattamente il percorso di prima — la radice del repo, i modelli
in `~/essentia_models`, `ffmpeg` preso dal PATH di sistema. È la condizione
perché i test e lo sviluppo quotidiano non si accorgano di questo modulo.

Dentro il bundle (PyInstaller) valgono tre regole:

1. i DATI di sola lettura — modelli Essentia, checkpoint Demucs, ffmpeg,
   plotly.min.js, i frontend HTML — stanno dentro l'app, sotto `_MEIPASS`,
   e non si scaricano mai dalla rete al primo avvio;
2. i DATI dell'utente e tutto ciò che si SCRIVE — stato dei job, elenco dei
   brani già taggati — vanno in `~/.cache/djcaddy/`, perché dentro l'app non
   si scrive (e perché devono sopravvivere all'aggiornamento dell'app);
3. l'eseguibile è UNO: i job lunghi non lanciano `python map_cli.py`, che nel
   bundle non esiste, ma se stessi con `--job map` (vedi `packaging/entry.py`).

Il PATH e `TORCH_HOME` li sistema `install()`, chiamata una volta all'avvio:
così i consumatori indiretti di ffmpeg (audioread dentro librosa, `shutil.which`
in folder_scan) trovano quello del bundle senza che nessuno li tocchi.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Radice del sorgente: core/bundle.py -> core -> radice.
_SOURCE_ROOT = Path(__file__).resolve().parent.parent


def frozen() -> bool:
    """True quando giriamo dentro il bundle PyInstaller."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def resources() -> Path:
    """La cartella dei dati di sola lettura: `_MEIPASS`, o la radice del repo."""
    return Path(sys._MEIPASS) if frozen() else _SOURCE_ROOT


def state_dir() -> Path:
    """Dove si scrive lo stato dei job.

    Nel repo restano accanto al codice, dov'erano da sempre (e dove il
    `.gitignore` li aspetta); nel bundle scendono in `~/.cache/djcaddy/`,
    accanto alla mappa e alla cache dell'analisi, perché dentro l'app il
    filesystem è di sola lettura.
    """
    if not frozen():
        return _SOURCE_ROOT
    path = Path.home() / ".cache" / "djcaddy"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_dir() -> Path:
    """La cartella dei modelli Essentia: quella dentro il bundle se c'è."""
    bundled = resources() / "essentia_models"
    if frozen() and bundled.is_dir():
        return bundled
    return Path(os.path.expanduser("~/essentia_models"))


def child_command(cli: Path) -> list[str]:
    """Come si lancia il CLI `cli` come processo figlio.

    Fuori dal bundle è l'interprete più lo script, com'è sempre stato. Dentro
    il bundle lo script non esiste: l'eseguibile è uno solo e sa fare anche i
    job, chiamato con `--job <nome>`, dove il nome è quello del CLI senza il
    suffisso (`map_cli.py` → `map`).
    """
    if frozen():
        return [sys.executable, "--job", cli.stem.removesuffix("_cli")]
    return [sys.executable, str(cli)]


def child_cwd() -> Path:
    """La cartella da cui parte il figlio. Nel bundle quella dei CLI non c'è."""
    return state_dir()


def install() -> None:
    """Fa trovare ffmpeg e il checkpoint Demucs dentro il bundle.

    Sul PATH invece che call-site per call-site: di ffmpeg e ffprobe non si
    servono solo i nostri `subprocess.run`, ma anche audioread (dentro
    `librosa.load`) e `shutil.which` in folder_scan, che non passano da qui.

    `TORCH_HOME` con `setdefault`: se l'utente ne ha già uno suo, quello vince.
    """
    if not frozen():
        return

    binaries = resources() / "bin"
    if binaries.is_dir():
        os.environ["PATH"] = os.pathsep.join(
            [str(binaries), os.environ.get("PATH", "")]).rstrip(os.pathsep)

    torch_home = resources() / "torch"
    if (torch_home / "hub" / "checkpoints").is_dir():
        os.environ.setdefault("TORCH_HOME", str(torch_home))
