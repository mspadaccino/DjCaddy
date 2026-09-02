# -*- mode: python ; coding: utf-8 -*-
"""Il bundle di DjCaddy: uno solo, e autonomo in tutto.

Dentro ci va TUTTO ciò che serve a lavorare senza rete e senza Python
installato: le librerie pesanti (torch/demucs, essentia-tensorflow,
librosa/numba, umap, pyrekordbox), i modelli Essentia, il checkpoint Demucs
già scaricato, ffmpeg e ffprobe, plotly.min.js e i frontend HTML. Non ci va
niente dell'UTENTE: la mappa, la cache e i sidecar restano in
`~/.cache/djcaddy/` e accanto ai brani, e sopravvivono agli aggiornamenti.

Il profilo "viewer" alleggerito non esiste: il prezzo è la taglia (3–4 GB) ed
è il costo dell'autonomia.

    poetry run pyinstaller packaging/djcaddy.spec --noconfirm

Su Windows essentia non ha wheel: si impacchetta con l'ambiente installato
`--without essentia` e le pagine che ne dipendono lo dicono da sole.
"""

import os
import shutil
import sys
import tomllib
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).parent
MACOS = sys.platform == "darwin"

# La versione sta scritta in un posto solo, e quel posto e' pyproject.toml:
# la leggono anche `build_macos.sh` (per il nome del DMG) e l'installer.
VERSION = tomllib.loads(
    (ROOT / "pyproject.toml").read_text())["tool"]["poetry"]["version"]


def _required(path: Path, what: str, how: str) -> Path:
    """Un pezzo senza il quale il bundle non sarebbe autonomo: o c'è, o si
    ferma qui — meglio adesso che davanti all'utente a rete staccata."""
    if not path.exists():
        raise SystemExit(f"\nManca {what}: {path}\n  {how}\n")
    return path


# --------------------------------------------------------------------------
# I dati inclusi
# --------------------------------------------------------------------------

datas = [
    # I due frontend HTML riusati (lavagna, ruota Camelot) e l'icona.
    (str(ROOT / "core" / "viz" / "frontend"), "core/viz/frontend"),
    (str(ROOT / "qt_app" / "assets"), "qt_app/assets"),
    # Il README: è la guida che l'app mostra in Help, filtrata da
    # `core.guide`. Sta nel bundle perché la finestra la legge da lì.
    (str(ROOT / "README.md"), "."),
]

# I modelli Essentia: embedding Discogs-EffNet e le teste genere/mood, più
# quelle che usano i CLI (danceability, mood_*). Sono 24 MB: si prendono tutti.
models = _required(
    Path(os.environ.get("DJCADDY_MODEL_DIR", Path.home() / "essentia_models")),
    "la cartella dei modelli Essentia",
    "scaricali con `poetry run python zoo_cli.py --help`, o punta "
    "DJCADDY_MODEL_DIR dove li tieni.",
)
datas.append((str(models), "essentia_models"))

# Il checkpoint di Demucs, pre-scaricato: dentro il bundle niente rete al
# primo avvio. `core.bundle.install()` punta qui TORCH_HOME, e torch.hub
# trova il file già in cache invece di andarselo a prendere.
checkpoints = _required(
    Path.home() / ".cache" / "torch" / "hub" / "checkpoints",
    "il checkpoint di Demucs",
    'scaricalo una volta con `poetry run python -c "from demucs.pretrained '
    "import get_model; get_model('htdemucs')\"`.",
)
datas.append((str(checkpoints), "torch/hub/checkpoints"))

# --------------------------------------------------------------------------
# I binari inclusi: ffmpeg e ffprobe, con le loro librerie
# --------------------------------------------------------------------------
# PyInstaller ne segue le dipendenze e le porta dentro riscrivendo i percorsi,
# quindi bastano i due eseguibili. Finiscono in `bin/`, che `install()`
# mette in testa al PATH: così li trovano anche i consumatori indiretti —
# audioread dentro `librosa.load`, `shutil.which` in folder_scan.

binaries = []
for tool in ("ffmpeg", "ffprobe"):
    found = shutil.which(tool)
    _required(Path(found) if found else Path(tool),
              f"l'eseguibile {tool}",
              "installalo (macOS: `brew install ffmpeg`) prima di impacchettare.")
    binaries.append((str(Path(found).resolve()), "bin"))

# --------------------------------------------------------------------------
# I pacchetti che non si lasciano trovare da soli
# --------------------------------------------------------------------------
# Dati e sottomoduli caricati per nome a runtime: essentia costruisce gli
# algoritmi dinamicamente, demucs legge i suoi .yaml, plotly il suo
# plotly.min.js, umap e pynndescent si chiamano fra loro attraverso numba,
# pyrekordbox porta le chiavi del database.
#
# Qui NON ci sono torch, librosa, sklearn, soundfile e scipy: per loro
# PyInstaller ha già i suoi hook, e un `collect_all` in più si tirerebbe
# dentro anche le loro suite di test (sklearn da sola sono migliaia di
# moduli, e pytest appresso).

hiddenimports = ["map_cli", "tag_cli"]
for package in ("plotly", "essentia", "demucs", "umap", "pynndescent",
                "pyrekordbox"):
    try:
        found_datas, found_binaries, found_hidden = collect_all(package)
    except Exception:
        continue                      # gruppo non installato (es. essentia su Windows)
    datas += found_datas
    binaries += found_binaries
    hiddenimports += [name for name in found_hidden if ".tests" not in name]


a = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "pytest", "PyInstaller"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DjCaddy",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(ROOT / "packaging" / "djcaddy.icns") if MACOS
    else str(ROOT / "packaging" / "djcaddy.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="DjCaddy",
)

if MACOS:
    app = BUNDLE(
        coll,
        name="DjCaddy.app",
        icon=str(ROOT / "packaging" / "djcaddy.icns"),
        bundle_identifier="com.mspadaccino.djcaddy",
        info_plist={
            "CFBundleName": "DjCaddy",
            "CFBundleDisplayName": "DjCaddy",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "NSHighResolutionCapable": True,
            # I brani stanno fuori dalle cartelle dell'app: senza questo,
            # su macOS recenti aprire una libreria dal volume esterno
            # chiederebbe un permesso che l'app non ha dichiarato.
            "NSDesktopFolderUsageDescription":
                "DjCaddy legge i brani della tua libreria.",
            "NSDocumentsFolderUsageDescription":
                "DjCaddy legge i brani della tua libreria.",
            "NSDownloadsFolderUsageDescription":
                "DjCaddy legge i brani della tua libreria.",
            "NSRemovableVolumesUsageDescription":
                "DjCaddy legge i brani della tua libreria dai volumi esterni.",
        },
    )
