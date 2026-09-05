"""I preset dei filtri: un nome per un modo di guardare la libreria.

«house_intro» non è solo un ritaglio dell'universo — generi, tonalità,
BPM, energia — è anche COME si misura la vicinanza dentro il ritaglio: i
tre pesi del costo di transizione. Un preset porta tutto quello che la
pagina Map imposta per rispondere a una domanda, così la domanda si rifà
con un gesto invece che con dieci.

Un file JSON solo, `{nome: {…}}`, in `user_dir()` accanto alle scalette:
cosa ci sia dentro lo decide chi salva (il pannello dei filtri), qui si
tengono solo i nomi e i dizionari. Un file rotto o assente legge come
vuoto: un preset perso si rifà, un'app che non parte no.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.analysis.user_files import user_dir


def default_presets_path() -> Path:
    return user_dir() / "presets.json"


class Presets:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else default_presets_path()

    def _all(self) -> dict[str, dict]:
        try:
            data = json.loads(self.path.read_text("utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self, data: dict[str, dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                             "utf-8")

    def names(self) -> list[str]:
        return sorted(self._all(), key=str.casefold)

    def read(self, name: str) -> dict | None:
        return self._all().get(name)

    def write(self, name: str, data: dict) -> None:
        everything = self._all()
        everything[name] = data
        self._save(everything)

    def delete(self, name: str) -> None:
        everything = self._all()
        if name in everything:
            del everything[name]
            self._save(everything)
