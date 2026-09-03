"""Gli appunti: cosa è stato proposto, cosa si è scelto, cosa si è mandato.

Nessuna delle tre schede di Build a set impara da chi la usa: la rosa del
Chain Maker si calcola con tre pesi messi a mano, e ogni scelta fra i nove
proposti — uno preso, otto scartati — si perde nel momento in cui viene
fatta. Questo modulo la trattiene. Non decide niente: scrive una riga per
gesto, e chi vorrà imparare i pesi, o "cosa viene dopo", troverà qui la
materia prima. Senza appunti presi da subito, quel giorno non arriva.

Una riga JSON per gesto, in coda a un file accanto ai preferiti — non
dentro la cache della mappa, che si può cancellare e rifare. Ogni riga
porta l'ora, il tipo di gesto e quello che il gesto ha visto: per i brani
i NUMERI (BPM, energia, valence, groove), non solo i percorsi, perché
l'energia è un rango sulla libreria e domani, con altri brani, lo stesso
file avrà un numero diverso.

Un appunto che non si riesce a scrivere non ferma il gesto: qui si prende
nota, non si governa.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from core.analysis.map_store import default_store_dir

# Quello che si annota di un brano, quando lo si annota. Chiavi del frame
# della pagina Map; quelle che mancano restano `None`.
FACTS = ("path", "bpm", "camelot", "energy", "valence", "danceability",
         "genres", "moods")


def default_journal_path() -> Path:
    return default_store_dir().parent / "choices.jsonl"


def _plain(value):
    """Un valore come JSON lo può scrivere: i numeri numpy diventano numeri
    di Python, i NaN diventano `None` — JSON non ha un NaN, e `json.dumps`
    ne scriverebbe uno che poi nessuno rilegge."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def facts(row) -> dict:
    """I numeri di un brano, da una riga del frame (o da qualunque mapping)."""
    noted = {}
    for key in FACTS:
        try:
            noted[key] = _plain(row[key])
        except (KeyError, IndexError):
            noted[key] = None
    return noted


class Journal:
    """Il quaderno: `record` accoda, `read` rilegge. Un percorso diverso da
    quello di default serve ai test e a chi vuole tenerne più d'uno."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else default_journal_path()

    def record(self, kind: str, **fields) -> None:
        line = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind,
                **fields}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as out:
                out.write(json.dumps(line, ensure_ascii=False,
                                     default=_plain) + "\n")
        except OSError:
            pass

    def read(self) -> list[dict]:
        """Le righe scritte finora, nell'ordine. Una riga rotta — un crash a
        metà scrittura — si salta, non si fa cadere il resto."""
        try:
            text = self.path.read_text("utf-8")
        except OSError:
            return []
        lines = []
        for raw in text.splitlines():
            try:
                lines.append(json.loads(raw))
            except ValueError:
                continue
        return lines
