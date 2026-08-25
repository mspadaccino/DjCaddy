"""Il mood come una scala sola: da buio a chiaro.

Il mood sulla mappa c'è già — `map_profile` salva fino a quattro etichette
per brano, ordinate per attivazione — ma è un elenco di parole, e la lavagna
della playlist vuole un'altezza. Qui le parole diventano un numero.

**Perché buio→chiaro e non calmo→intenso.** Sono le due scale che si
possono tendere su queste etichette: l'arco emotivo e l'arco di energia. Il
secondo, misurato sulla libreria (87.010 brani), è monco — *Energetic* sta
sull'89% dei brani e niente è davvero calmo, quindi il decile inferiore è
incollato allo zero e la curva sa solo salire. Il primo si muove: decili
−0,12 e +0,60, e dentro venti brani della stessa cartella la deviazione
standard resta 0,235 contro 0,289 dell'intera libreria — cioè l'escursione
sopravvive DENTRO un set, che è l'unico posto in cui la si guarda.

E soprattutto non ripete niente di quello che la lavagna già mostra: la
correlazione con il groove è +0,00, con il BPM +0,13. Un asse che ridicesse
la curva del tempo non varrebbe la manopola per sceglierlo.

**Tre gruppi e non una tabella di valori.** Dare a ognuna delle 56 etichette
un valore suo vuol dire inventare 56 numeri; scuro, neutro e chiaro sono
invece una lettura che si discute un'etichetta alla volta. Misurate una
contro l'altra sulla libreria le due danno la stessa curva (correlazione
0,88), e quella a tre gruppi è pure più larga: ampiezza dei decili 0,72
contro 0,40.
"""

from __future__ import annotations

from collections import Counter

# Le etichette del modello MTG-Jamendo che tirano in basso e quelle che
# tirano in alto, come le scrive `format_mood_tag` (iniziale maiuscola).
# Tutte le altre — Energetic, Melodic, Retro, Relaxing, Epic, i temi come
# Film o Corporate — restano al centro: dicono altro, non un colore.
DARK = frozenset({"Dark", "Deep", "Drama", "Dramatic", "Emotional", "Heavy",
                  "Melancholic", "Sad"})
BRIGHT = frozenset({"Fun", "Funny", "Happy", "Hopeful", "Inspiring", "Love",
                    "Motivational", "Party", "Positive", "Romantic", "Summer",
                    "Upbeat", "Uplifting"})


def split(moods) -> list[str]:
    """Le etichette di un brano, dalla più forte alla più debole.

    Accetta la stringa come sta sulla riga della mappa ("Deep; Summer;
    Energetic") o una lista già divisa: la riga arriva dal disco, la lista
    da chi ha appena analizzato il brano.
    """
    if moods is None:
        return []
    if isinstance(moods, str):
        return [m.strip() for m in moods.split(";") if m.strip()]
    return [str(m).strip() for m in moods if str(m).strip()]


def valence(moods) -> float | None:
    """Da −1 (buio) a +1 (chiaro). `None` se di etichette non ce n'è nessuna.

    Media pesata per il rango — il primo mood pesa 1, il secondo 1/2, il
    terzo 1/3 — perché l'ordine È la confidenza del modello, e un'etichetta
    appena sopra soglia non deve contare quanto quella che il brano porta
    davvero.

    Le etichette neutre pesano nel denominatore e non nel numeratore: un
    brano *Dark; Energetic; Melodic* è meno buio di uno che porta *Dark* e
    basta, ed è giusto che la scala lo dica.
    """
    labels = split(moods)
    if not labels:
        return None
    total = weight = 0.0
    for rank, label in enumerate(labels, start=1):
        value = -1.0 if label in DARK else (1.0 if label in BRIGHT else 0.0)
        total += value / rank
        weight += 1 / rank
    return total / weight


def popularity(values) -> dict[str, int]:
    """Quante volte ogni etichetta compare, su una colonna di mood."""
    return Counter(label for value in values for label in split(value))


def distinctive(moods, common: dict[str, int]) -> str | None:
    """Delle etichette di un brano, quella che lo separa dagli altri.

    È la più rara nella libreria, non la più forte: *Energetic* sta
    sull'89% dei brani e *Happy* sul 57%, quindi la prima etichetta è quasi
    sempre la stessa per tutti e non distingue niente. Fra due ugualmente
    rare vince quella che il modello dà per prima.
    """
    labels = split(moods)
    if not labels:
        return None
    return min(enumerate(labels),
               key=lambda pair: (common.get(pair[1], 0), pair[0]))[1]


def summary(moods, common: dict[str, int]) -> str:
    """Il mood come si legge in tabella: prima quello che distingue, poi il
    resto nell'ordine del modello."""
    labels = split(moods)
    if not labels:
        return ""
    rare = distinctive(labels, common)
    rest = [label for label in labels if label != rare]
    return f"{rare} · {'; '.join(rest)}" if rest else rare
