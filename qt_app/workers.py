"""Lavori fuori dal filo dell'interfaccia, col risultato riportato dentro.

Il piano mette scansioni e conti lunghi in QThreadPool con segnali di
progresso; questo è il pezzo minimo che serve adesso — una funzione, il suo
risultato (o la sua eccezione) consegnato al filo giusto. L'emissione da un
thread del pool arriva al ricevente come connessione accodata, quindi lo
slot gira nel main thread senza che nessuno debba pensarci.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class Progress(QObject):
    """Il filo del progresso di un lavoro nel pool.

    Si crea nel main thread, si passa la sua `emit` come callback al job:
    l'emissione dal thread del pool arriva agli slot come connessione
    accodata, quindi la barra si muove nel filo giusto senza altro codice.
    """

    count = Signal(int, int)   # fatti, totale
    text = Signal(str)         # per i lavori che raccontano invece di contare


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
            self._tell(self.signals.failed, trouble)
            return
        self._tell(self.signals.done, result)

    @staticmethod
    def _tell(signal, payload) -> None:
        """Un'app in chiusura smonta i segnali sotto i piedi del worker:
        a quel punto non c'è più nessuno da avvertire, e va bene così."""
        try:
            signal.emit(payload)
        except RuntimeError:
            pass


# I task in volo, per riferimento: il pool tiene il QRunnable solo fino
# alla fine di run() — poi l'autoDelete lo distrugge e il garbage collector
# porta via i suoi segnali PRIMA che l'emissione accodata raggiunga il main
# thread. Senza questo set la consegna era una lotteria: quasi sempre vinta,
# ma un job perdeva il suo risultato in modo riproducibile (misurato con
# `truncation.inspect`: mai consegnato senza riferimento, 0,1 s con).
_ALIVE: set = set()


def run_in_pool(job, on_done, on_failed=None) -> None:
    """Esegue `job()` nel pool e consegna il risultato a `on_done`.

    Il task resta nel set dei vivi finché uno dei due segnali è stato
    consegnato: è quello che tiene in vita la connessione fino in fondo.
    """
    task = _Task(job)
    _ALIVE.add(task)

    def _deliver(result) -> None:
        _ALIVE.discard(task)
        on_done(result)

    def _trouble(trouble) -> None:
        _ALIVE.discard(task)
        if on_failed is not None:
            on_failed(trouble)

    task.signals.done.connect(_deliver)
    task.signals.failed.connect(_trouble)
    QThreadPool.globalInstance().start(task)
