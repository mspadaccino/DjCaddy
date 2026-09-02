"""La guida dentro l'app: lo stesso README, meno i capitoli del codice.

Una sorgente sola, e il README è già quella: dice cosa fa ogni scheda e cosa
vuol dire ogni numero. Riscriverne una copia per la finestra di Help
vorrebbe dire che fra un mese le due si contraddicono, e che nessuna delle
due è più quella giusta.

Quindi la finestra legge il README, e qui si toglie ciò che a chi USA l'app
non serve: come si installa dal sorgente, i CLI, come si impacchetta, com'è
fatto il repository. Restano i capitoli che parlano dell'app.

I riferimenti che puntavano dentro i capitoli tolti non diventano link
morti: tornano testo semplice. È il prezzo di filtrare invece di riscrivere,
e si paga qui una volta sola invece che a ogni modifica del README.
"""

from __future__ import annotations

import re
from pathlib import Path

from .bundle import resources

# I capitoli (##) e le sezioni (###) che la guida non mostra. Sono scritti
# per titolo e non per numero di riga apposta: il README si riordina senza
# che questo file se ne accorga, e un titolo che cambia lo dice il test.
DROPPED_CHAPTERS = (
    "Contents",                      # la finestra ha il suo indice, a lato
    "Command line",
    "Building the standalone app",
    "How the code is laid out",
)
DROPPED_SECTIONS = (
    "What you need",                 # installare dal sorgente non riguarda
    "Install",                       #   chi ha l'app già in mano
    "Run it",
    "Building the map from the terminal",
)

_HEADING = re.compile(r"^(#{1,3}) (.+)$")
_FENCE = re.compile(r"^\s*```")
_LINK = re.compile(r"\[([^\]]+)\]\(#([^)]+)\)")


def readme() -> Path:
    """Il README, nel repo come dentro il bundle (dove è un dato incluso)."""
    return resources() / "README.md"


def anchor(title: str) -> str:
    """Lo slug di un titolo, con la regola di GitHub: minuscolo, senza
    punteggiatura, e OGNI spazio un trattino — non i gruppi di spazi, che
    è ciò che rende `groove--read-this-one-carefully` un anchor giusto."""
    return re.sub(r"\s", "-", re.sub(r"[^\w\s-]", "", title.lower()).strip())


def _headings(text: str):
    """(livello, titolo) di ogni titolo, saltando i blocchi di codice.

    I blocchi contano: dentro ci sono righe che cominciano per `#`, e sono
    commenti di shell, non titoli.
    """
    fenced = False
    for line in text.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
            continue
        if fenced:
            continue
        found = _HEADING.match(line)
        if found:
            yield len(found.group(1)), found.group(2)


def contents(text: str) -> list[tuple[int, str]]:
    """L'indice da mostrare a lato: capitoli e sezioni, in ordine di lettura."""
    return [(level, title) for level, title in _headings(text) if level > 1]


def guide(text: str | None = None) -> str:
    """Il README ridotto a guida dell'app."""
    if text is None:
        text = readme().read_text()

    kept: list[str] = []
    fenced = chapter_gone = section_gone = False
    for line in text.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
        found = None if fenced else _HEADING.match(line)
        if found:
            level, title = len(found.group(1)), found.group(2)
            if level <= 2:
                # Un capitolo nuovo decide di sé e azzera la sezione: una
                # sezione tolta non deve portarsi via quelle che la seguono.
                chapter_gone = level == 2 and title in DROPPED_CHAPTERS
                section_gone = False
            else:
                section_gone = title in DROPPED_SECTIONS
        if not (chapter_gone or section_gone):
            kept.append(line)

    return _relink("\n".join(kept).strip() + "\n")


def _relink(text: str) -> str:
    """Chi punta a un capitolo che non c'è più smette di essere un link."""
    alive = {anchor(title) for _, title in _headings(text)}
    return _LINK.sub(
        lambda m: m.group(0) if m.group(2) in alive else m.group(1), text)
