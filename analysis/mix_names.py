"""Riconoscere dal nome i file che sono più brani mixati insieme.

Serve a ripulire una libreria da medley, megamix e mashup tenendo i brani
singoli — comprese le versioni extended, che sono quelle che si vogliono.

La durata da sola non basta, e le misure sulla libreria reale (88.570 file)
dicono perché:

* i **mashup durano quanto un brano normale**: mediana 4,2 minuti contro i
  4,7 della libreria, nessuno oltre i 10. Nessuna soglia di durata li vedrà
  mai — il nome è l'unico indizio che hanno.
* i **megamix** invece durano: mediana 10,2 minuti, uno su due oltre i 10.
* le **versioni extended**, quelle da tenere, hanno mediana 5,5 minuti e solo
  l'1% supera i 10. Una soglia a 10 minuti costa quindi pochissimi falsi
  positivi: è utile, ma va affiancata al nome, non sostituita da esso.

Il nome del FILE è l'unico testo che serve. Su 900 file presi a caso il tag
titolo aggiunge un solo caso che il nome non vedeva già; il tag album ne
aggiungerebbe 17, ma guardandoli sono quasi tutti falsi positivi — i brani di
"Mastermix Extended Floorfillers" sono singoli da 5 minuti, la raccolta è
mixata, il brano no. Per questo qui si guarda il nome e basta.
"""

from __future__ import annotations

import re
from functools import lru_cache

# Parole accese di default: misurate, e tutte con durate da raccolta mixata
# (mediane fra 7 e 10 minuti) oppure — i mashup — inequivocabili nel nome.
DEFAULT_KEYWORDS = (
    "megamix", "mega mix", "medley", "mashup", "mash up", "mash-up",
    "mixed by", "in the mix", "party mix", "supermix", "mastermix",
    "mini mix", "minimix", "non-stop", "nonstop", "continuous",
    "dj set", "live set", "yearmix", "year mix",
)

# Parole che sembrano indizi e NON lo sono: misurate sulla libreria, durano
# quanto un brano qualunque. Restano fuori dai default perché accenderle
# sposterebbe in quarantena migliaia di brani singoli perfettamente buoni.
# Il valore è la spiegazione, che l'interfaccia mostra.
NOT_BY_DEFAULT = {
    "bootleg": "510 file, mediana 4,0 min — è un remix non ufficiale, "
               "un brano singolo, non un mix",
    " vs ": "1.358 file, mediana 4,7 min — di solito una collaborazione "
            "o un remix, non due brani mixati",
    "mixtape": "128 file, mediana 3,7 min — qui indica lo stile, non un mix",
    "best of": "76 file: metà supera i 10 minuti, l'altra metà è un brano "
               "singolo da una raccolta. Accendila e controlla a mano",
    "compilation": "di solito è il nome della RACCOLTA da cui viene il "
                   "brano, che resta singolo",
}


# Due "vs" nello stesso nome: due artisti E due titoli, cioè un mashup.
# Misurato: un solo "vs" prende 2.312 file, per lo più collaborazioni vere;
# due ne prendono 679, e su dodici presi a caso erano dodici mashup
# ("david guetta vs. alice deejay - play hard vs. better off alone").
_VS = re.compile(r"\bvs\b")


def parse_keywords(text: str) -> list[str]:
    """Dalla casella di testo alla lista di parole, separate da virgola."""
    return [w.strip().lower() for w in text.split(",") if w.strip()]


@lru_cache(maxsize=32)
def _matcher(keywords: tuple[str, ...]):
    """Un'unica espressione per tutte le parole: una passata per nome invece
    di una per parola, che su 90.000 file è la differenza fra un istante e
    qualche secondo. Le più lunghe vanno prima, così "mash-up" vince su
    "mash" quando ci sono entrambe."""
    if not keywords:
        return None
    joined = "|".join(sorted((re.escape(k) for k in keywords),
                             key=len, reverse=True))
    return re.compile(rf"\b(?:{joined})\b")


def matching_words(name: str, keywords) -> list[str]:
    """Quali parole compaiono nel nome del file.

    Il confronto è a parola intera, e non è un dettaglio: "mixed by" preso
    come semplice sottostringa becca anche "**Re**mixed By", che nelle
    raccolte DMC indica il remixer di un brano SINGOLO. Sono 138 file nella
    libreria — brani da tenere, che finivano fra quelli da spostare.
    """
    matcher = _matcher(tuple(keywords))
    if matcher is None:
        return []
    return sorted(set(matcher.findall(name.lower())))


def looks_like_a_mashup(name: str) -> bool:
    """Vero se il nome ha due "vs": "A vs B - primo brano vs secondo brano"."""
    return len(_VS.findall(name.lower())) >= 2
