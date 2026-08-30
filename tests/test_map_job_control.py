"""Fermare, riprendere e chiudere il job dall'esterno.

Si prova su un processo qualunque (`sleep`), non sull'analisi vera: quello
che c'è da verificare è il segnale, non cosa stava facendo chi lo riceve.
"""

import os
import subprocess
import time

from core.analysis.map_job import (caffeinated, pause_job, process_state,
                              resume_job, stop_job)


def _sleeper() -> subprocess.Popen:
    # start_new_session: come il job vero, capofila del suo gruppo — è la
    # condizione che permette di colpire tutti i figli insieme.
    return subprocess.Popen(["sleep", "30"], start_new_session=True)


def _settle(pid: int, wanted: str, tries: int = 40) -> str:
    """I segnali non sono istantanei: si guarda finché non cambia."""
    for _ in range(tries):
        how = process_state(pid)
        if how == wanted:
            return how
        time.sleep(0.05)
    return process_state(pid)


def test_a_job_can_be_paused_and_resumed():
    job = _sleeper()
    try:
        assert process_state(job.pid) == "running"

        assert pause_job(job.pid)
        assert _settle(job.pid, "paused") == "paused"

        # Fermo non vuol dire morto: il processo è ancora lì.
        assert job.poll() is None

        assert resume_job(job.pid)
        assert _settle(job.pid, "running") == "running"
    finally:
        stop_job(job.pid)
        job.wait(timeout=5)


def test_stopping_a_job_ends_it():
    job = _sleeper()
    assert stop_job(job.pid)
    job.wait(timeout=5)
    assert _settle(job.pid, "gone") == "gone"


def test_a_pid_that_is_not_there_any_more():
    job = _sleeper()
    stop_job(job.pid)
    job.wait(timeout=5)
    assert process_state(job.pid) == "gone"
    assert not stop_job(job.pid)      # niente da fermare, e nessuna eccezione
    assert process_state(0) == "gone"


def test_the_job_holds_sleep_off_for_as_long_as_it_lasts(monkeypatch):
    """Un Mac che si addormenta congela il job: resta vivo e non lavora.

    `caffeinate` vuole il comando come argomento, non un pid da sorvegliare:
    così la sveglia cade da sola quando il job finisce, comunque finisca.
    """
    import core.analysis.map_job as map_job

    monkeypatch.setattr(map_job.sys, "platform", "darwin")
    assert caffeinated(["python", "map_cli.py", "/Music"]) == [
        "caffeinate", "-i", "python", "map_cli.py", "/Music"]


def test_elsewhere_the_command_is_left_alone(monkeypatch):
    import core.analysis.map_job as map_job

    monkeypatch.setattr(map_job.sys, "platform", "linux")
    assert caffeinated(["python", "map_cli.py"]) == ["python", "map_cli.py"]


def test_the_machine_is_kept_awake_only_while_the_work_lasts(monkeypatch):
    """Un Mac addormentato non rallenta il lavoro: lo CONGELA. E una sveglia
    dimenticata accesa e' l'errore opposto, altrettanto scortese."""
    import subprocess as sp

    from core.analysis import map_job

    started, killed = [], []

    class Keeper:
        def terminate(self):
            killed.append(True)

    monkeypatch.setattr(map_job.sys, "platform", "darwin")
    monkeypatch.setattr(sp, "Popen", lambda cmd, *a, **k: (started.append(cmd), Keeper())[1])

    with map_job.awake():
        assert started and started[0][0] == "caffeinate"
        # Aspetta il NOSTRO pid: se il job muore di brutto, la sveglia se ne
        # va da sola invece di restare accesa per sempre.
        assert started[0][-1] == str(os.getpid())
        assert not killed
    assert killed


def test_without_caffeinate_the_work_still_runs(monkeypatch):
    import subprocess as sp

    from core.analysis import map_job

    monkeypatch.setattr(map_job.sys, "platform", "darwin")
    monkeypatch.setattr(sp, "Popen", lambda *a, **k: (_ for _ in ()).throw(OSError))
    with map_job.awake():
        pass          # niente sveglia, ma il blocco gira lo stesso


def test_off_a_mac_there_is_nothing_to_keep_awake(monkeypatch):
    from core.analysis import map_job

    monkeypatch.setattr(map_job.sys, "platform", "linux")
    with map_job.awake():
        pass
