"""map_cli.py --project: lo stato resta "in corso" anche durante la
riproiezione UMAP, non solo durante il profiling brano per brano.

Sono due fasi dello stesso processo, ma la pagina Map si ricarica un colpo
solo quando vede `state.running` diventare False (vedi render_progress in
map_analysis.py): se il file di stato dicesse "finito" già alla fine del
profiling, quel colpo solo arriverebbe prima che `coords.npy` abbia le
posizioni dei brani appena aggiunti, e nessuno gliene manderebbe un secondo.
"""

from __future__ import annotations

import os
import sys

import pytest

from core.analysis.map_job import load_map_state
from core.analysis.tag_job import JobState


def test_state_stays_running_through_reprojection(tmp_path, monkeypatch):
    import map_cli

    folder = tmp_path / "brani"
    folder.mkdir()
    state_file = tmp_path / "stato.json"

    fake_state = JobState(pid=os.getpid(), folder=str(folder), total=1,
                          done=1, written=1, started_at=100.0)

    def fake_run_job(*args, **kwargs):
        # Come farebbe davvero run_job: segna la fine del profiling e salva.
        fake_state.finished_at = 105.0
        fake_state.save(state_file)
        return fake_state

    seen_while_reprojecting = []

    def fake_reproject(store_dir, settings):
        seen_while_reprojecting.append(load_map_state(state_file).running)

    monkeypatch.setattr(map_cli, "run_job", fake_run_job)
    monkeypatch.setattr(map_cli, "reproject", fake_reproject)
    monkeypatch.setattr(map_cli, "available", lambda: True)
    monkeypatch.setattr(map_cli, "missing_models", lambda: [])
    monkeypatch.setattr(sys, "argv", [
        "map_cli.py", str(folder), "--project",
        "--state-file", str(state_file)])

    with pytest.raises(SystemExit) as exc:
        map_cli.main()
    assert exc.value.code == 0

    assert seen_while_reprojecting == [True]
    assert load_map_state(state_file).running is False


def test_prune_takes_out_only_what_is_gone(tmp_path, monkeypatch, capsys):
    """La potatura toglie i brani spariti e lascia stare gli altri."""
    import map_cli
    from core.analysis.map_profile import EMBEDDING_DIM, TrackProfile
    from core.analysis.map_store import MapStore
    import numpy as np

    library = tmp_path / "libreria"
    library.mkdir()
    resta = library / "c'e.mp3"
    sparito = library / "non c'e.ogg"
    for track in (resta, sparito):
        track.write_bytes(b"x")

    store_dir = tmp_path / "map"
    store = MapStore.load(store_dir)
    store.append([
        TrackProfile(path=track, duration=300.0, bpm=128.0, camelot="8A",
                     embedding=np.full(EMBEDDING_DIM, n, dtype=np.float32))
        for n, track in enumerate((resta, sparito), start=1)])
    sparito.unlink()

    monkeypatch.setattr("builtins.input", lambda _: "s")
    monkeypatch.setattr(sys, "argv", [
        "map_cli.py", "--prune", str(library), "--store", str(store_dir)])
    map_cli.main()

    again = MapStore.load(store_dir)
    assert [row["path"] for row in again.rows] == [str(resta)]
    assert again.embeddings.shape == (1, EMBEDDING_DIM)
    assert again.embeddings[0][0] == 1.0   # il vettore giusto, non quello del vicino


def test_prune_refuses_when_the_library_is_not_mounted(tmp_path, monkeypatch):
    """Radice irraggiungibile: si ferma invece di svuotare la mappa."""
    import map_cli
    from core.analysis.map_profile import EMBEDDING_DIM, TrackProfile
    from core.analysis.map_store import MapStore
    import numpy as np

    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    store_dir = tmp_path / "map"
    MapStore.load(store_dir).append([
        TrackProfile(path=audio, duration=300.0, bpm=128.0, camelot="8A",
                     embedding=np.full(EMBEDDING_DIM, 1.0, dtype=np.float32))])

    monkeypatch.setattr(sys, "argv", [
        "map_cli.py", "--prune", str(tmp_path / "disco-staccato"),
        "--store", str(store_dir)])
    with pytest.raises(SystemExit) as exc:
        map_cli.main()
    assert exc.value.code == 2

    assert len(MapStore.load(store_dir)) == 1
