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
