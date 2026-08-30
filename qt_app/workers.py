"""Lavori fuori dal filo dell'interfaccia, col risultato riportato dentro.

Il piano mette scansioni e conti lunghi in QThreadPool con segnali di
progresso; questo è il pezzo minimo che serve adesso — una funzione, il suo
risultato (o la sua eccezione) consegnato al filo giusto. L'emissione da un
thread del pool arriva al ricevente come connessione accodata, quindi lo
slot gira nel main thread senza che nessuno debba pensarci.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class _Signals(QObject):
    done = Signal(object)      # il risultato della funzione
    failed = Signal(object)    # l'eccezione, se è andata male


class _Task(QRunnable):
    def __init__(self, job) -> None:
        super().__init__()
        self.signals = _Signals()
        self._job = job

    def run(self) -> None:  # nel thread del pool
        try:
            result = self._job()
        except Exception as trouble:
            self.signals.failed.emit(trouble)
            return
        self.signals.done.emit(result)


def run_in_pool(job, on_done, on_failed=None) -> None:
    """Esegue `job()` nel pool e consegna il risultato a `on_done`.

    I `signals` del task vivono finché il task vive: il riferimento se lo
    tiene il pool, quindi qui non serve conservare niente.
    """
    task = _Task(job)
    task.signals.done.connect(on_done)
    if on_failed is not None:
        task.signals.failed.connect(on_failed)
    QThreadPool.globalInstance().start(task)
