"""`core.bundle`: dove stanno le cose dentro e fuori dal bundle.

Due sono le proprietà che contano. Fuori dal bundle NIENTE deve cambiare —
lo stato dei job resta accanto al codice, i modelli in ~/essentia_models, i
job si lanciano con l'interprete e lo script: è la ragione per cui questo
modulo è invisibile allo sviluppo di ogni giorno. Dentro il bundle, invece,
ciò che si scrive deve scendere in ~/.cache/djcaddy (l'app è di sola
lettura) e i dati inclusi devono essere trovati senza rete.
"""

import importlib.util
import sys
from pathlib import Path

from core import bundle

REPO = Path(__file__).resolve().parent.parent


def freeze(monkeypatch, meipass: Path, home: Path) -> None:
    """Fa credere a `bundle` di girare impacchettato, con casa altrove."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(meipass), raising=False)
    monkeypatch.setenv("HOME", str(home))


# --------------------------------------------------------------------------
# Fuori dal bundle: tutto com'era
# --------------------------------------------------------------------------

def test_outside_the_bundle_nothing_moves():
    assert not bundle.frozen()
    assert bundle.resources() == REPO
    assert bundle.state_dir() == REPO
    assert bundle.model_dir() == Path.home() / "essentia_models"


def test_outside_the_bundle_the_job_is_the_script():
    cli = REPO / "map_cli.py"
    assert bundle.child_command(cli) == [sys.executable, str(cli)]


def test_install_is_a_no_op_outside_the_bundle(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("TORCH_HOME", raising=False)
    bundle.install()
    assert bundle.os.environ["PATH"] == "/usr/bin"
    assert "TORCH_HOME" not in bundle.os.environ


# --------------------------------------------------------------------------
# Dentro il bundle
# --------------------------------------------------------------------------

def test_what_is_written_leaves_the_read_only_app(tmp_path, monkeypatch):
    """Lo stato dei job non può stare dentro l'app: solo lettura."""
    meipass, home = tmp_path / "app", tmp_path / "home"
    freeze(monkeypatch, meipass, home)
    assert bundle.state_dir() == home / ".cache" / "djcaddy"
    assert bundle.state_dir().is_dir()          # la crea lei, al primo giro


def test_models_come_from_the_bundle_when_they_are_there(tmp_path, monkeypatch):
    meipass, home = tmp_path / "app", tmp_path / "home"
    freeze(monkeypatch, meipass, home)
    assert bundle.model_dir() == home / "essentia_models"   # non inclusi
    (meipass / "essentia_models").mkdir(parents=True)
    assert bundle.model_dir() == meipass / "essentia_models"


def test_the_job_is_the_app_itself(tmp_path, monkeypatch):
    """Nel bundle non c'è un interprete con cui lanciare map_cli.py."""
    freeze(monkeypatch, tmp_path / "app", tmp_path / "home")
    assert bundle.child_command(REPO / "map_cli.py") == [
        sys.executable, "--job", "map"]
    assert bundle.child_command(REPO / "tag_cli.py") == [
        sys.executable, "--job", "tag"]


def test_the_names_of_the_jobs_are_the_ones_entry_answers_to():
    """Il contratto fra chi lancia (`child_command`) e chi risponde (entry)."""
    spec = importlib.util.spec_from_file_location(
        "djcaddy_entry", REPO / "packaging" / "entry.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    for name, cli in entry.JOBS.items():
        assert Path(cli).stem.removesuffix("_cli") == name


def test_install_puts_the_bundled_ffmpeg_first(tmp_path, monkeypatch):
    """Sul PATH, non call-site per call-site: di ffmpeg si serve anche chi
    non passa da noi (audioread dentro librosa, `shutil.which`)."""
    meipass, home = tmp_path / "app", tmp_path / "home"
    freeze(monkeypatch, meipass, home)
    (meipass / "bin").mkdir(parents=True)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("TORCH_HOME", raising=False)

    bundle.install()
    assert bundle.os.environ["PATH"].split(":")[0] == str(meipass / "bin")


def test_install_points_torch_at_the_included_checkpoint(tmp_path, monkeypatch):
    """Il checkpoint di Demucs è già dentro: niente rete al primo avvio."""
    meipass, home = tmp_path / "app", tmp_path / "home"
    freeze(monkeypatch, meipass, home)
    (meipass / "torch" / "hub" / "checkpoints").mkdir(parents=True)
    monkeypatch.delenv("TORCH_HOME", raising=False)

    bundle.install()
    assert bundle.os.environ["TORCH_HOME"] == str(meipass / "torch")


def test_a_torch_home_of_your_own_wins(tmp_path, monkeypatch):
    meipass, home = tmp_path / "app", tmp_path / "home"
    freeze(monkeypatch, meipass, home)
    (meipass / "torch" / "hub" / "checkpoints").mkdir(parents=True)
    monkeypatch.setenv("TORCH_HOME", "/altrove")

    bundle.install()
    assert bundle.os.environ["TORCH_HOME"] == "/altrove"
