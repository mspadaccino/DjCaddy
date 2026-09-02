"""L'unico eseguibile del bundle: la app, uno dei due job lunghi, o il selftest.

Impacchettata, l'app non ha accanto un interprete Python con cui lanciare
`map_cli.py`: `sys.executable` È l'app. Quindi l'eseguibile sa fare anche i
job, e `core.bundle.child_command` lo richiama con `--job map` / `--job tag`
passandogli gli argomenti che prima andavano allo script. Il flusso resta
quello di sempre — processo staccato, stato su file, log su file — e fuori
dal bundle non cambia niente: lì i CLI sono ancora script.

`--selftest` è la verifica di autonomia della Fase 5, e sta qui dentro
perché è dentro che va fatta: si lancia sulla macchina pulita, a rete
staccata, e dice pezzo per pezzo se ciò che serve è nel bundle.

    /Applications/DjCaddy.app/Contents/MacOS/DjCaddy --selftest
"""

from __future__ import annotations

import importlib
import multiprocessing
import subprocess
import sys
from pathlib import Path

# Lanciato come script, sul path c'è solo `packaging/`: senza la radice non
# si importano né `core` né `qt_app`. Nel bundle è già tutto in _MEIPASS, ma
# aggiungerla non disturba.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

JOBS = {"map": "map_cli", "tag": "tag_cli"}


# --------------------------------------------------------------------------
# La verifica di autonomia
# --------------------------------------------------------------------------

def _check_ffmpeg() -> str:
    """ffmpeg e ffprobe: quelli del bundle, e funzionanti."""
    import shutil

    from core import bundle

    told = []
    for tool in ("ffmpeg", "ffprobe"):
        found = shutil.which(tool)
        if not found:
            raise AssertionError(f"{tool} non è sul PATH")
        if bundle.frozen() and not found.startswith(str(bundle.resources())):
            raise AssertionError(f"{tool} viene da fuori: {found}")
        out = subprocess.run([found, "-version"], capture_output=True, text=True,
                             timeout=30)
        if out.returncode != 0:
            raise AssertionError(f"{tool} non parte: {out.stderr.strip()[:80]}")
        told.append(out.stdout.split("\n")[0][:40])
    return " · ".join(told)


def _check_models() -> str:
    """I modelli Essentia: tutti e cinque, dentro il bundle."""
    from core.analysis.essentia_tags import MODEL_DIR, MODELS, missing_models

    missing = missing_models()
    if missing:
        raise AssertionError(f"mancano {', '.join(missing)} da {MODEL_DIR}")
    return f"{len(MODELS)} file in {MODEL_DIR}"


def _check_demucs_checkpoint() -> str:
    """Il checkpoint di Demucs già scaricato: niente rete al primo avvio."""
    import torch

    found = sorted((Path(torch.hub.get_dir()) / "checkpoints").glob("*.th"))
    if not found:
        raise AssertionError(f"nessun checkpoint in {torch.hub.get_dir()}")
    return f"{len(found)} in {torch.hub.get_dir()}"


def _check_plotly() -> str:
    """plotly.min.js: la mappa lo carica da lì, non da un CDN."""
    from qt_app.widgets.plotly_view import plotly_package_data

    path = plotly_package_data() / "plotly.min.js"
    if not path.exists():
        raise AssertionError(f"manca {path}")
    return f"{path.stat().st_size // 1024} KB"


def _check_frontends() -> str:
    """La lavagna e la ruota Camelot: gli HTML riusati sono dentro."""
    from core.viz import frontend_dir

    for name in ("graph_board", "camelot_wheel"):
        page = frontend_dir(name) / "index.html"
        if not page.exists():
            raise AssertionError(f"manca {page}")
    return "graph_board · camelot_wheel"


def _check_libraries() -> str:
    """Le librerie pesanti si importano davvero, non solo si copiano."""
    told = []
    for name in ("torch", "demucs.apply", "librosa", "umap", "essentia.standard",
                 "pyrekordbox", "PySide6.QtWebEngineWidgets"):
        try:
            importlib.import_module(name)
            told.append(name.split(".")[0])
        except Exception as error:
            # Su Windows essentia non esiste, ed è previsto: si dice e si tira
            # dritto, come fanno le pagine che la usano.
            if name.startswith("essentia") and sys.platform == "win32":
                told.append("essentia (assente: è Windows)")
                continue
            raise AssertionError(f"{name}: {type(error).__name__} {error}") from error
    return " · ".join(told)


def _check_writable_state() -> str:
    """Dove si scrive dev'essere FUORI dall'app, che è di sola lettura."""
    from core import bundle
    from core.analysis.map_job import DEFAULT_MAP_STATE_FILE

    where = bundle.state_dir()
    if bundle.frozen() and str(where).startswith(str(bundle.resources())):
        raise AssertionError(f"lo stato finirebbe dentro l'app: {where}")
    probe = where / ".djcaddy_selftest"
    probe.write_text("ok")
    probe.unlink()
    return f"{DEFAULT_MAP_STATE_FILE.parent}"


def _check_jobs() -> str:
    """I job lunghi: l'app sa richiamare se stessa come li lancia la pagina."""
    from core import bundle
    from core.analysis.map_job import MAP_CLI_PATH

    command = [*bundle.child_command(MAP_CLI_PATH), "--help"]
    out = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if out.returncode != 0:
        raise AssertionError(f"{' '.join(command[1:])} → {out.stderr.strip()[:120]}")
    return " ".join(command[1:])


CHECKS = [
    ("ffmpeg", _check_ffmpeg),
    ("modelli Essentia", _check_models),
    ("checkpoint Demucs", _check_demucs_checkpoint),
    ("plotly.min.js", _check_plotly),
    ("frontend HTML", _check_frontends),
    ("librerie", _check_libraries),
    ("stato scrivibile", _check_writable_state),
    ("job in background", _check_jobs),
]


def selftest() -> int:
    from core import bundle

    print(f"DjCaddy — verifica di autonomia "
          f"({'bundle' if bundle.frozen() else 'sorgente'}: {bundle.resources()})\n")
    bad = 0
    for name, check in CHECKS:
        try:
            print(f"  ok   {name}: {check()}")
        except Exception as error:
            bad += 1
            print(f"  NO   {name}: {error}")
    print(f"\n{len(CHECKS) - bad}/{len(CHECKS)}")
    return 1 if bad else 0


# --------------------------------------------------------------------------

def main() -> int:
    # Prima di tutto il resto: senza, ogni processo figlio avviato da
    # multiprocessing (torch lo fa) rieseguirebbe l'app da capo.
    multiprocessing.freeze_support()

    from core import bundle
    bundle.install()

    argv = sys.argv[1:]
    if argv[:1] == ["--selftest"]:
        return selftest()

    if argv[:1] == ["--job"]:
        name = argv[1] if len(argv) > 1 else ""
        if name not in JOBS:
            print(f"--job vuole uno fra {', '.join(JOBS)}", file=sys.stderr)
            return 2
        module = importlib.import_module(JOBS[name])
        # Il CLI legge `sys.argv` con argparse e chiude con `sys.exit`.
        sys.argv = [JOBS[name], *argv[2:]]
        module.main()
        return 0

    from qt_app.main import main as run_app
    return run_app()


if __name__ == "__main__":
    raise SystemExit(main())
