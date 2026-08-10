"""Cache dei risultati per file, per non rianalizzare a ogni run.

Chiave = path assoluto del file. La voce è valida solo se `mtime` e `size` del
file coincidono con quelli memorizzati (e la versione dello schema è quella
corrente): così solo i file nuovi o modificati vengono rianalizzati.

La cache memorizza la parte per-file dell'analisi (genere, BPM, RMS,
boundaries). NON memorizza la vibe, che dipende dall'intera libreria.
"""

from __future__ import annotations

import json
from pathlib import Path

CACHE_VERSION = 2  # v2: aggiunta la durata (per il tempo rimanente)


def default_cache_path() -> Path:
    return Path.home() / ".cache" / "dj-library-tools" / "analysis.json"


def _stat(filepath: Path) -> tuple[float, int]:
    st = filepath.stat()
    return st.st_mtime, st.st_size


class AnalysisCache:
    """Cache su singolo file JSON, keyed per path assoluto."""

    def __init__(self, path: Path, entries: dict | None = None):
        self.path = path
        self._entries: dict[str, dict] = entries or {}

    @classmethod
    def load(cls, path: Path | None = None) -> "AnalysisCache":
        path = path or default_cache_path()
        if path.exists():
            try:
                raw = json.loads(path.read_text())
                if raw.get("version") == CACHE_VERSION:
                    return cls(path, raw.get("entries", {}))
            except Exception:
                pass  # cache corrotta: si riparte da zero
        return cls(path, {})

    def _key(self, filepath: Path) -> str:
        return str(filepath.resolve())

    def get(self, filepath: Path) -> dict | None:
        """Ritorna il payload cacheato se il file è invariato, altrimenti None."""
        entry = self._entries.get(self._key(filepath))
        if not entry:
            return None
        try:
            mtime, size = _stat(filepath)
        except OSError:
            return None
        if entry.get("mtime") == mtime and entry.get("size") == size:
            return entry.get("data")
        return None

    def put(self, filepath: Path, data: dict) -> None:
        try:
            mtime, size = _stat(filepath)
        except OSError:
            return
        self._entries[self._key(filepath)] = {"mtime": mtime, "size": size, "data": data}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": CACHE_VERSION, "entries": self._entries}
        self.path.write_text(json.dumps(payload))
