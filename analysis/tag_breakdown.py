"""Scompone i tag in elementi singoli, per contarli e cercarli.

Un tag come "Electronic - House; Electronic - Tech House" e' tre cose
insieme: due generi, ciascuno su due livelli. Qui si separano, si contano e
si torna indietro — da un elemento ai brani che lo portano.

Che cosa e' separatore e che cosa no e' stato misurato su un campione di
600 brani della libreria, non deciso a tavolino:

    " - "   98% dei GENRE   separa il livello padre dal figlio
    ";"     81% dei GENRE, 94% dei COMMENT   separa elementi diversi
    "/"     15% dei GENRE   NON separa: sta dentro "Funk / Soul"
    "&"      4% dei GENRE   NON separa: sta dentro "Brass & Military"

Le ultime due sono la ragione per cui questo modulo esiste invece di una
`split()` sul posto: spezzare su "/" spaccherebbe in due un genere Discogs
valido, e nessuno se ne accorgerebbe guardando i conteggi.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Fra elementi diversi. Il punto e virgola e' quello che scriviamo noi; la
# virgola e' ammessa perche' ricorre nei tag di provenienza esterna.
SEPARATORI = (";", ",")

# Fra livello padre e figlio, SEMPRE spaziato: senza spazi si porterebbe via
# "Hi-NRG", "Hip-Hop" e "Drum-n-Bass", che sono nomi, non gerarchie.
GERARCHIA = " - "


def split_tag_value(value: str | None, split_hierarchy: bool = True) -> list[str]:
    """Gli elementi contenuti in un valore di tag, senza doppioni.

    Con `split_hierarchy` "Electronic - House" vale sia come "Electronic"
    sia come "House": sono due domande diverse (quanto elettronico c'e' in
    tutto, e quanta house in particolare) e si vogliono entrambe.
    """
    if not value or not value.strip():
        return []
    pezzi = [value]
    for sep in SEPARATORI:
        pezzi = [p for pezzo in pezzi for p in pezzo.split(sep)]
    if split_hierarchy:
        pezzi = [p for pezzo in pezzi for p in pezzo.split(GERARCHIA)]

    visti, fuori = set(), []
    for p in pezzi:
        p = p.strip()
        chiave = p.casefold()
        if p and chiave not in visti:
            visti.add(chiave)
            fuori.append(p)
    return fuori


@dataclass
class Breakdown:
    """I tipi trovati, quante volte, e su quali brani."""
    counts: Counter = field(default_factory=Counter)
    paths: dict[str, list[Path]] = field(default_factory=lambda: defaultdict(list))
    tracks_with_any: int = 0
    tracks_without: int = 0

    def rows(self) -> list[dict]:
        return [{"Type": tipo, "Tracks": n}
                for tipo, n in self.counts.most_common()]

    def tracks(self, tipo: str) -> list[Path]:
        return self.paths.get(tipo, [])

    def tracks_of(self, tipi) -> list[Path]:
        """I brani che portano ALMENO UNO dei tipi indicati.

        Unione, non intersezione: scegliere "House" e "Tech House" vuol dire
        quasi sempre "fammi vedere tutte e due", non "i brani che sono
        entrambe le cose" — che per due generi sorelle sarebbe spesso vuoto.

        Senza doppioni, perche' un brano che porta tutti e due i tipi
        comparirebbe due volte, e l'elenco esportato conterebbe piu' righe
        dei brani che contiene.
        """
        visti, fuori = set(), []
        for t in tipi:
            for percorso in self.paths.get(t, []):
                if percorso not in visti:
                    visti.add(percorso)
                    fuori.append(percorso)
        return sorted(fuori)


def build_breakdown(items, field_name: str,
                    split_hierarchy: bool = True) -> Breakdown:
    """Scompone `field_name` ("genre" o "comment") di ogni voce di copertura.

    I tipi si contano ignorando maiuscole e minuscole ma si mostrano nella
    forma incontrata per prima: "House" e "house" sono la stessa cosa, e
    tenerli separati darebbe due righe che dicono la stessa cosa a meta'.
    """
    b = Breakdown()
    forma = {}
    for item in items:
        elementi = split_tag_value(getattr(item, field_name, None), split_hierarchy)
        if not elementi:
            b.tracks_without += 1
            continue
        b.tracks_with_any += 1
        for e in elementi:
            chiave = e.casefold()
            canonico = forma.setdefault(chiave, e)
            b.counts[canonico] += 1
            b.paths[canonico].append(Path(item.path))
    return b


def as_text(paths, header: str = "") -> str:
    """L'elenco da salvare: un percorso per riga, che e' il formato che ogni
    altro strumento sa rileggere."""
    righe = [f"# {header}", f"# {len(paths)} tracks", ""] if header else []
    righe += [str(p) for p in paths]
    return "\n".join(righe) + "\n"
