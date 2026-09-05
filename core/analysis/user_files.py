"""Dove stanno le cose che l'utente ha FATTO, non quelle che l'app ha
calcolato.

La cache (`~/.cache/djcaddy`) tiene la mappa: ore di lavoro, ma
ricalcolabile, e una cartella nascosta che un "pulisci cache" può
spazzare. Le scalette e i preset dei filtri sono un'altra cosa — sono il
lavoro del DJ, come i crate — e vanno dove si vedono, si copiano e si
salvano: `~/Documents/DjCaddy`. Su Windows lo stesso, in `Documents`.

Un solo posto scritto qui, perché ogni pezzo che lo scegliesse da sé
finirebbe per scegliere diverso.
"""

from __future__ import annotations

from pathlib import Path


def user_dir() -> Path:
    documents = Path.home() / "Documents"
    root = documents if documents.is_dir() else Path.home()
    return root / "DjCaddy"
