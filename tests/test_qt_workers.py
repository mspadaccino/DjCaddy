"""`run_in_pool`: la consegna deve sopravvivere al garbage collector.

Il pool tiene il QRunnable solo fino alla fine di run(): poi l'autoDelete
lo distrugge, e senza un riferimento Python i suoi segnali morivano PRIMA
che l'emissione accodata raggiungesse il main thread — misurato con
`truncation.inspect`, che perdeva il risultato in modo riproducibile. Il
set `_ALIVE` è la cura: qui si prova che il task ci sta finché serve e
che i due esiti arrivano davvero, anche sotto un GC aggressivo.
"""

import gc
import os
import time

import pytest

pytest.importorskip("PySide6", reason="gruppo poetry `qt` non installato")
pytest.importorskip("pytestqt", reason="gruppo poetry `qt` non installato")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication

from qt_app import workers


def _pump_until(cond, timeout=10.0) -> bool:
    began = time.time()
    while time.time() - began < timeout:
        QCoreApplication.processEvents()
        gc.collect()                    # il GC più cattivo del reale
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_result_is_delivered_and_task_released(qapp):
    got = {}
    workers.run_in_pool(lambda: 41 + 1, lambda r: got.update(result=r))
    assert len(workers._ALIVE) >= 1     # il riferimento che tiene in vita
    assert _pump_until(lambda: got)
    assert got["result"] == 42
    # A consegna avvenuta il set si svuota: niente task accumulati.
    assert _pump_until(lambda: not workers._ALIVE)


def test_failure_is_delivered_and_task_released(qapp):
    got = {}

    def _boom():
        raise RuntimeError("kaputt")

    workers.run_in_pool(_boom, lambda r: got.update(result=r),
                        lambda t: got.update(trouble=t))
    assert _pump_until(lambda: got)
    assert isinstance(got["trouble"], RuntimeError)
    assert "result" not in got
    assert _pump_until(lambda: not workers._ALIVE)


def test_failure_without_handler_still_releases_the_task(qapp):
    def _boom():
        raise RuntimeError("kaputt")

    workers.run_in_pool(_boom, lambda r: None)
    assert _pump_until(lambda: not workers._ALIVE)


def test_a_slow_job_survives_the_collector(qapp):
    """Il caso che perdeva il risultato: un job che dura abbastanza da far
    girare il GC fra l'avvio e la consegna."""
    got = {}
    workers.run_in_pool(lambda: time.sleep(0.3) or "made it",
                        lambda r: got.update(result=r))
    for _ in range(10):
        gc.collect()
    assert _pump_until(lambda: got)
    assert got["result"] == "made it"
