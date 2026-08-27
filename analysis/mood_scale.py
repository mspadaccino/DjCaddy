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


# --------------------------------------------------------------------------
# La stessa scala, letta sui pesi veri del modello
# --------------------------------------------------------------------------
#
# `valence` qui sopra legge le ETICHETTE: quelle sopra soglia, al massimo
# quattro, e al posto della confidenza usa il rango (1, 1/2, 1/3, 1/4). È
# un'approssimazione, e butta via due cose.
#
# La prima è la forza. Un brano che porta *Dark* a 0,62 e uno che lo porta a
# 0,06 leggono tutti e due −1,00: la sola etichetta che hanno è quella, e il
# rango non sa dire che il primo è buio e il secondo lo sfiora appena.
#
# La seconda sono le altre cinquantadue. Un brano con *Sad* 0,049,
# *Melancholic* 0,045 e *Dark* 0,041 non passa la soglia di 0,05 con
# nessuna: di etichette non ne ha, `valence` risponde `None`, e in tabella
# non compare nessuna freccia — mentre di prove di buio ne ha tre.
#
# Il modello quelle attivazioni le calcola tutte e cinquantasei, sempre. È
# solo la riga su disco che le perde.

def weights(text) -> dict[str, float]:
    """Le attivazioni come stanno sulla riga: "Dark:0.62; Deep:0.41".

    È lo stesso formato con cui la riga porta già la confidenza dei generi,
    e per lo stesso motivo: leggibile aprendo il file, e senza una struttura
    annidata dentro un formato che è una riga per brano.
    """
    if isinstance(text, dict):
        return {str(k): float(v) for k, v in text.items()}
    out: dict[str, float] = {}
    for piece in split(text):
        label, _, value = piece.rpartition(":")
        try:
            out[label.strip()] = float(value)
        except ValueError:
            continue
    return out


def spell_weights(pairs) -> str:
    """Le attivazioni come si scrivono sulla riga, dalla più forte."""
    ordered = sorted(pairs.items() if isinstance(pairs, dict) else pairs,
                     key=lambda pair: -pair[1])
    return "; ".join(f"{label}:{value:.3f}" for label, value in ordered)


def valence_of(activations, dilute: bool = False) -> float | None:
    """Da −1 (buio) a +1 (chiaro), sui pesi veri. `None` se non c'è colore.

    È lo stesso asse di `valence` — la valence in senso proprio, l'arco
    piacevole/spiacevole — misurata meglio: non l'ordine delle etichette ma
    quanto il modello ci crede.

    A contare sono SOLO le etichette che un colore ce l'hanno:

        (somma delle chiare − somma delle buie) / (somma di tutte e due)

    Le neutre restano fuori anche dal denominatore, ed è un cambio rispetto
    a `valence`, dove diluivano. Il motivo è che con i pesi veri diluire non
    è più neutro: *Energetic* sta sull'89% della libreria e ci sta forte, e
    lasciata nel denominatore schiaccerebbe tutti verso lo zero per una
    quantità quasi uguale su ogni brano — cioè toglierebbe escursione senza
    aggiungere lettura. Quanto un brano sia POCO colorato si legge meglio da
    `evidence`, che è un numero a parte e dice quanto colore c'è in tutto.

    `None` quando di prove non ce n'è nessuna: uno zero direbbe "in mezzo
    fra buio e chiaro", che è un'altra cosa da "di questo non si sa".

    `dilute` rimette le neutre nel denominatore, cioè fa quello che fa
    `valence` per rango. NON è un'opzione da usare: serve a `mood_cli
    --check` per separare due cambiamenti che altrimenti si guardano
    insieme e non si capisce quale dei due ha mosso cosa. Passare dai
    ranghi ai pesi veri e togliere le neutre dal denominatore sono due
    decisioni diverse, e vanno misurate una alla volta.
    """
    dark = bright = plain = 0.0
    for label, value in weights(activations).items():
        if not value > 0:
            continue
        if label in DARK:
            dark += value
        elif label in BRIGHT:
            bright += value
        else:
            plain += value
    coloured = dark + bright
    if not coloured > 0:
        return None
    return (bright - dark) / (coloured + plain if dilute else coloured)


def evidence(activations) -> float:
    """Quanto colore porta un brano in tutto, comunque orientato.

    È il denominatore di `valence_of`, tenuto a parte perché dice una cosa
    che la direzione non dice: un brano a +0,90 su 0,05 di prove e uno a
    +0,90 su 1,40 sono chiari tutti e due, ma il primo lo è per un soffio
    di attivazione e il secondo lo grida. Serve a decidere quando la
    freccia in tabella si disegna e quando no.
    """
    return sum(v for label, v in weights(activations).items()
               if v > 0 and (label in DARK or label in BRIGHT))
