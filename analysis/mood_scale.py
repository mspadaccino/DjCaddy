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


# Quante etichette per lista, che NON sono lo stesso numero: 13 chiare
# contro 8 buie. Con una testa multi-label ogni etichetta porta una
# attivazione di fondo anche su un brano che non è quella cosa — è il rumore
# della sigmoide, non una prova — e sommando le due liste il fondo entra 13
# volte da una parte e 8 dall'altra. Il risultato è una somma chiara più
# grande su OGNI brano, per un motivo che con la musica non c'entra niente.
#
# Misurato sulla libreria vera (2.000 brani): con le somme crude i nove
# decili della valence erano +0,31 +0,40 +0,47 +0,53 +0,58 +0,62 +0,67 +0,71
# +0,76. Tutti positivi: lo zero non era il mezzo di niente, aveva il 90%
# della libreria da una parte sola.
SIDES = (len(DARK), len(BRIGHT))

# Le due scelte da cui dipende la forma della scala, tenute qui e non
# sparse. Misurate su 2.000 brani veri, ed è così che sono finite dove sono:
#
# `balanced` divide ogni lato per quante etichette ha. Serve, e non basta:
# porta i brani sotto zero dall'1,1% al 6,3%. Il resto dello sbilanciamento
# non sono le liste, è il prior del modello, e nessuna correzione qui lo
# tocca — per quello si legge il RANGO e non il numero (vedi
# `views.map_analysis._valence_rank`).
#
# `floor` resta a zero, e non per pigrizia. Alzarlo non toglie il rumore in
# modo neutro: toglie per prima l'evidenza del lato PERDENTE, che è la più
# debole, e quindi toglie la sfumatura lasciando solo il vincitore. A 0,02
# un decimo della libreria è già inchiodato a +1,00; a 0,05 lo sono quattro
# decili su nove, più il 14% che resta senza nessuna lettura. Il costo è
# enorme e il guadagno sul centraggio è di pochi punti che il rango dà
# gratis.
#
# C'era una domanda aperta sul 12% di brani il cui colore viene SOLO da
# attivazioni sotto la soglia delle parole, ed è chiusa — ma non dai numeri.
#
# Le statistiche ci hanno provato tre volte e hanno sbagliato tre volte. La
# lettura forte contro la debole dà -0,387, e sembrava una condanna; è
# invece viziata per costruzione, perché l'etichetta che vince viene
# promossa sopra soglia e sotto resta il perdente. Il controllo per togliere
# quel vizio — rimescolare le etichette dello stesso brano — dà -0,273, e
# sembrava una condanna peggiore; ma il rimescolamento distrugge anche il
# prior per etichetta, che è proprio il meccanismo che nel dato vero produce
# l'anticorrelazione in più. Ogni modo di isolare le attivazioni deboli le
# altera.
#
# A rispondere sono stati venti brani messi in fila dal più buio al più
# chiaro (`mood_cli --faint-sample`) e guardati da chi la libreria ce l'ha:
# in fondo Abba, Tiffany, Taylor Dayne, Baltimora; in cima hardstyle, 50
# Cent, Lil Scrappy, un Master Beat del '92. Nessuno fuori posto su venti,
# compresi i due che a me sembravano sbagliati leggendo i titoli — un remix
# techno pesante e un reggaeton in minore, tutti e due bui davvero.
#
# Le prove deboli sono segnale. `floor` resta a zero.
BALANCED = True
FLOOR = 0.0


def _sides(activations, floor: float, balanced: bool,
           ceiling: float | None = None) -> tuple[float, float, float]:
    """Quanto pesa il buio, quanto il chiaro, quanto il resto.

    `ceiling` tiene solo le attivazioni SOTTO un valore, cioè l'esatto
    contrario di `floor`. Non serve a nessuna lettura: serve a domandare se
    le prove deboli — quelle che non diventano mai parole — dicano la stessa
    cosa di quelle forti o siano rumore. Vedi `mood_cli.check`.
    """
    dark = bright = plain = 0.0
    for label, value in weights(activations).items():
        if value <= floor or (ceiling is not None and value > ceiling):
            continue
        if label in DARK:
            dark += value
        elif label in BRIGHT:
            bright += value
        else:
            plain += value
    if balanced:
        dark, bright = dark / SIDES[0], bright / SIDES[1]
    return dark, bright, plain


def valence_of(activations, dilute: bool = False,
               floor: float | None = None,
               balanced: bool | None = None,
               ceiling: float | None = None) -> float | None:
    """Da −1 (buio) a +1 (chiaro), sui pesi veri. `None` se non c'è colore.

    È lo stesso asse di `valence` — la valence in senso proprio, l'arco
    piacevole/spiacevole — misurata meglio: non l'ordine delle etichette ma
    quanto il modello ci crede.

    A contare sono SOLO le etichette che un colore ce l'hanno:

        (chiare − buie) / (chiare + buie)

    dove ogni lato è la MEDIA delle sue etichette e non la somma, per il
    motivo scritto sopra `SIDES`. Le neutre restano fuori anche dal
    denominatore, ed è un cambio rispetto a `valence`, dove diluivano: con i
    pesi veri diluire non è più neutro, perché *Energetic* sta sull'89%
    della libreria e ci sta forte, e nel denominatore schiaccerebbe tutti
    verso lo zero per una quantità quasi uguale su ogni brano. Quanto un
    brano sia POCO colorato si legge da `evidence`, che è un numero a parte.

    `None` quando di prove non ce n'è nessuna: uno zero direbbe "in mezzo
    fra buio e chiaro", che è un'altra cosa da "di questo non si sa".

    I tre parametri non sono opzioni da usare: servono a `mood_cli --check`
    per misurare una accanto all'altra le letture fra cui si sceglie, e per
    separare cambiamenti che altrimenti si guardano tutti insieme e non si
    capisce quale ha mosso cosa. Lasciati a `None` prendono la scelta fatta,
    che è quella qui sopra.
    """
    dark, bright, plain = _sides(
        activations,
        FLOOR if floor is None else floor,
        BALANCED if balanced is None else balanced,
        ceiling)
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
    dark, bright, _ = _sides(activations, FLOOR, BALANCED)
    return dark + bright


def from_rows(rows) -> list[float]:
    """La valence di ogni riga della mappa, `nan` dove non si sa.

    Due sorgenti, e la seconda è quella che tiene in piedi tutto finché il
    backfill non è passato: se la riga porta `valence` — scritta sui pesi
    veri di tutte e 56 le etichette — si legge quella; altrimenti si ricava
    dalle parole, che è la vecchia lettura per rango. La prima è migliore, la
    seconda c'è su tutta la libreria da sempre, e sono lo stesso asse.

    Qui e non nella pagina che la disegna perché a chiederla sono in due —
    la mappa e il rapporto sullo stato della libreria — e due copie della
    stessa regola di ripiego un giorno direbbero due cose diverse.
    """
    out = []
    for row in rows:
        value = row.get("valence")
        if value is None:
            value = valence(row.get("moods", ""))
        out.append(float("nan") if value is None else float(value))
    return out
