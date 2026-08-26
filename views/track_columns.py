"""Come si legge un brano in tabella: le colonne, e i colori che portano.

Le tabelle della sezione Map sono sei, in due moduli, e mostrano tutte lo
stesso brano: la rosa del Chain Maker propone quello che un istante dopo sta
nella catena, la playlist elenca quello che la mappa ha appena selezionato.
Le colonne stanno qui una volta sola perché sono le stesse — scritte a mano
in sei posti avrebbero sei versioni e cinque da aggiornare. È già successo
due volte in piccolo: la tavolozza dei generi era duplicata fra i due moduli
per non chiudere un giro di import, e `_read_only` era gemello in entrambi.
Questo modulo non importa nessuno dei due, quindi il giro non si chiude e i
doppioni tornano uno.

**Pastiglie e non numeri.** Una tabella di venti righe per dieci colonne di
cifre si legge una cella alla volta; la stessa tabella con la tonalità, la
regolarità e il genere colorati si legge di traverso — si vede DOVE cambia
il colore senza leggere niente, che è come si cerca il prossimo brano. È
quello che fa djoid, ed è il motivo per cui lo fa.

I colori non sono nuovi: la tonalità porta esattamente le tinte della ruota
Camelot qui accanto (e dei lettori per DJ), i generi quelli dei punti sulla
mappa — stessa tavolozza assegnata con la stessa regola, i più frequenti
della libreria per primi. Sulla mappa il conto si rifà a ogni disegno su
quello che è rimasto dopo i filtri e al livello di dettaglio scelto lì,
quindi le due assegnazioni coincidono quando la mappa colora le etichette
foglia e i filtri sono larghi, non per costruzione. A restare vero sempre è
il resto: due generi diversi non condividono mai un colore, e la coda lunga
è grigia di qua come di là.

**Perché una tinta sola per colonna.** Il groove ha il suo verde e dentro il
verde il valore muove solo l'intensità. Dieci colori diversi per dieci
gradini darebbero un arcobaleno in cui il 3 e il 7 non si distinguono per
quantità ma per tinta, cioè un numero da rileggere ogni volta invece di una
scala che si guarda.
"""

from __future__ import annotations

import colorsys

import pandas as pd
import streamlit as st

from analysis import mood_scale

# I colori dei generi sulla mappa e sulle schede della lavagna. Diciotto:
# con le etichette foglia — che sono 258, di cui le prime dodici coprono il
# 64% — dodici colori lasciavano un terzo della libreria in grigio. Le tinte
# continuano a girare attorno alla ruota invece di scurirsi, o due gruppi
# vicini diventerebbero indistinguibili su punti da sette pixel.
PALETTE = ["#e0503b", "#3d9be0", "#3fbf7f", "#f2a33c", "#a06fd6", "#e06fa8",
           "#4dd0c4", "#c9b037", "#6f8fd6", "#d66f6f", "#7fbf3f", "#bf7fd6",
           "#3fb0bf", "#d68f3f", "#8fd63f", "#d63f8f", "#5f6fd6", "#bf5f3f"]

# Chi non entra fra i colorati. Due grigi, uno per tema: su fondo bianco
# serve scuro, su fondo nero serve chiaro, e in tutti e due i casi deve
# sparire rispetto ai diciotto.
OTHER_COLOR = {"light": "#9aa4b0", "dark": "#6b7684"}

# I gradini su cui si scrive una misura continua in tabella. Dieci come in
# djoid: un numero da leggere dentro una pastiglia larga venti pixel sta
# comodo in una cifra.
#
# Il groove però NON li usa, ed è una scelta di chi legge queste tabelle: la
# scheda del brano e la lavagna scrivono `0,61`, e vedere `6` in tabella e
# `0,61` sulla scheda costringe a ricordarsi che sono lo stesso numero. La
# pastiglia si allarga di due caratteri e la coerenza vale il prezzo.
LEVELS = 10


def dark() -> bool:
    """Se il tema in uso è quello scuro. Al primo giro il tema può non essere
    ancora arrivato dal browser: in quel caso si disegna chiaro."""
    theme = getattr(getattr(st, "context", None), "theme", None)
    return getattr(theme, "type", None) == "dark"


def camelot_color(camelot: str | None) -> str:
    """Il colore della ruota Camelot per una tonalità.

    È la stessa codifica dei lettori per DJ (e di djoid): il numero dà la
    tinta, la lettera dice se maggiore o minore. Serve perché due tonalità
    che si mixano stanno vicine sulla ruota, e vicine sulla ruota vuol dire
    tinte vicine — la compatibilità si vede senza leggere la sigla.
    """
    text = (camelot or "").strip().upper()
    if len(text) < 2 or not text[:-1].isdigit():
        return "#c7ccd4"
    number = int(text[:-1])
    if not 1 <= number <= 12:
        return "#c7ccd4"
    major = text[-1] == "B"
    hue = ((190 - 30 * number) % 360) / 360
    r, g, b = colorsys.hls_to_rgb(hue, 0.72 if major else 0.62,
                                  0.65 if major else 0.55)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


KEY_OPTIONS = [f"{number}{mode}" for number in range(1, 13) for mode in "AB"]
KEY_COLORS = {key: camelot_color(key) for key in KEY_OPTIONS}

LEVEL_OPTIONS = [str(n) for n in range(1, LEVELS + 1)]

# Il groove come lo scrivono la scheda e la lavagna: due decimali, da 0,00 a
# 1,00. Sono centouno pastiglie possibili, e il colore va agganciato a
# ognuna perché la tinta segue la posizione nell'elenco delle opzioni — non
# c'è modo di dire "colora per valore" e scrivere altro.
GROOVE_OPTIONS = [f"{n / 100:.2f}" for n in range(101)]


def _ramp(hue: float, steps: int = LEVELS) -> list[str]:
    """Una scala di `steps` colori di tinta fissa, dal pallido al pieno.

    A muoversi sono saturazione e luminosità insieme: la sola saturazione su
    fondo scuro non si vede — un grigio e un verde spento hanno la stessa
    luce — e la sola luminosità sbianca. Il fondo però non scende sotto la
    metà per quanto il valore salga: la pastiglia porta il numero scritto in
    scuro, e un verde pieno da 0,4 di luminosità se lo mangerebbe. È il
    motivo per cui la scala va dal pallido al pieno e non dal chiaro al buio.
    """
    return ["#%02x%02x%02x" % tuple(
        round(c * 255) for c in colorsys.hls_to_rgb(
            hue / 360,
            0.80 - 0.30 * n / (steps - 1),
            0.30 + 0.60 * n / (steps - 1)))
        for n in range(steps)]


# Il verde del groove, che è il colore con cui djoid scrive la danceability.
# Centouno gradini invece di dieci, uno per valore: la scala percorre le
# stesse due estremità di prima, solo più fitta, quindi da lontano la colonna
# si legge di traverso esattamente come prima — anzi meglio, perché il
# passaggio da un brano all'altro è continuo e non a scalini.
GROOVE_COLORS = _ramp(145, len(GROOVE_OPTIONS))

# Il verso dell'emotion, non la sua misura: la freccia dice da che parte
# guarda il brano, e la colonna del mood di fianco dice quali parole ce lo
# mandano. L'ambra è quella con cui la lavagna segna ciò che sale.
EMOTION_OPTIONS = ["↑", "↓"]
EMOTION_COLORS = ["#e0a260", "#6f8fd6"]

# Sotto questo scarto dal centro la freccia non si disegna. Senza una zona
# morta ogni brano ne avrebbe una, comprese le migliaia che portano un solo
# mood neutro e che di verso non ne hanno nessuno.
EMOTION_DEADZONE = 0.10


def read_only(*columns: str) -> dict:
    """Colonne che si guardano e basta. In una tabella con una casella da
    spuntare tutto il resto va bloccato a mano, o si finisce a correggere i
    BPM di un brano credendo di sceglierlo."""
    return {name: st.column_config.Column(disabled=True) for name in columns}


def key_column(label: str = "key"):
    """La tonalità, nel colore della sua fetta di ruota."""
    return st.column_config.MultiselectColumn(
        label, disabled=True, width="small",
        options=KEY_OPTIONS, color=[KEY_COLORS[k] for k in KEY_OPTIONS],
        help="The Camelot key, in the colour it has on the wheel: two keys "
             "that mix sit next to each other, so they carry neighbouring "
             "tints. A run of compatible tracks is visible without reading a "
             "single code.")


def groove_column(label: str = "groove"):
    """La danceability com'è, in verde: due decimali da 0,00 a 1,00."""
    return st.column_config.MultiselectColumn(
        label, disabled=True, width="small",
        options=GROOVE_OPTIONS, color=GROOVE_COLORS,
        help="The danceability from 0.00 to 1.00: regularity of the onsets, "
             "low is loose and high is a straight kick. The same number the "
             "track card, the board and the filters all call groove, written "
             "the same way everywhere so there is nothing to convert.")


def emotion_column(label: str = "emotion"):
    """Il verso del mood: su chiaro, giù buio."""
    return st.column_config.MultiselectColumn(
        label, disabled=True, width="small",
        options=EMOTION_OPTIONS, color=EMOTION_COLORS,
        help="Which way the track looks: up is bright, down is dark, nothing "
             "means it says neither. It is the same dark→bright reading of "
             "the moods that the board uses for height. Which moods those "
             "are is in the next column.")


def mood_column(label: str = "mood"):
    """La colonna del mood, con la sua spiegazione. Una sola perché le
    tabelle che la portano sono sei, in due moduli: scritta a mano ogni
    volta, la spiegazione avrebbe sei versioni e cinque da aggiornare."""
    return st.column_config.Column(
        label, disabled=True,
        help="Before the dot, the rarest of this track's moods across your "
             "library — the one that tells it apart. Energetic sits on 89% "
             "of the library and Happy on 57%, so the strongest mood is "
             "almost the same for everyone. After the dot, the others, "
             "strongest first.")


def genre_column(colors: dict[str, str], label: str = "genres"):
    """I generi come pastiglie, negli stessi colori dei punti sulla mappa.

    `colors` è il vocabolario della libreria: ogni genere che compare, col
    colore che ha sulla mappa — i più frequenti una tinta a testa, la coda
    lunga il grigio dell'"altro". Ci devono stare TUTTI e non solo i
    colorati: un genere che il vocabolario non nomina non viene disegnato
    come etichetta, e ricompare per esteso in mezzo alle pastiglie.

    Il nome si accorcia all'ultimo pezzo — "Electronic - House" diventa
    "House" — perché le etichette Discogs sono gerarchiche e il padre si
    ripete su quasi tutte le righe: occupa mezza colonna per dire ogni volta
    la stessa cosa. A stringersi è l'etichetta, il valore resta intero.
    """
    known = list(colors)
    return st.column_config.MultiselectColumn(
        label, disabled=True, width="medium",
        options=known, color=[colors[g] for g in known],
        format_func=lambda genre: str(genre).split(" - ")[-1],
        help="The genres the model reads on this track, strongest first, in "
             "the colour they have on the map. Multi-label on purpose: club "
             "music is hybrid, and 'Minimal' and 'Deep House' can both be "
             "true of the same track. Only the coloured groups of the map "
             "get a tint here too; the long tail is grey, as it is there.")


def _value(row, column: str):
    """Il valore, o `None` se manca davvero.

    Serve perché un campo vuoto arriva qui come NaN, e NaN è vero: scritto in
    una pastiglia diventa la parola "nan", che sembra un dato invece che
    l'assenza di un dato.
    """
    if row is None or column not in row:
        return None
    value = row[column]
    return value if pd.notna(value) and value != "" else None


def _pill(value) -> list[str]:
    """Una pastiglia sola, o nessuna. Le colonne colorate vogliono una lista:
    è il tipo con cui Streamlit disegna le etichette, anche dove di etichette
    per riga ce n'è al massimo una."""
    return [] if value is None else [str(value)]


def groove_pill(danceability) -> str | None:
    """La regolarità degli onset come si scrive nella pastiglia: "0.61".

    Bloccata fra 0 e 1 e a due decimali perché deve cadere ESATTAMENTE su
    una delle opzioni della colonna: una stringa fuori elenco non è un
    errore, è una pastiglia che non si colora e nessuno capisce perché.
    """
    if danceability is None or pd.isna(danceability):
        return None
    return f"{min(1.0, max(0.0, float(danceability))):.2f}"


def emotion_arrow(moods) -> str | None:
    """Il verso del mood, o niente se il brano non ne ha uno."""
    value = mood_scale.valence(moods)
    if value is None:
        return None
    if value > EMOTION_DEADZONE:
        return "↑"
    return "↓" if value < -EMOTION_DEADZONE else None


def reading(row, common: dict[str, int]) -> dict:
    """Il brano come lo scrivono tutte le tabelle della pagina.

    Gli stessi campi con gli stessi nomi ovunque, perché il brano che si
    guarda nella rosa è quello che un momento dopo sta nella catena, e
    cambiargli le colonne nel passaggio costringerebbe a ritrovarlo.
    """
    bpm = _value(row, "bpm")
    return {
        "file": row["name"],
        "BPM": round(bpm) if bpm is not None else None,
        "key": _pill(_value(row, "camelot")),
        "groove": _pill(groove_pill(_value(row, "danceability"))),
        "emotion": _pill(emotion_arrow(row["moods"] if "moods" in row else "")),
        "mood": mood_scale.summary(row["moods"] if "moods" in row else "",
                                   common),
        "genres": [g for g in str(_value(row, "genres") or "").split("; ") if g],
        # Da dove viene il file. Due brani con lo stesso nome esistono, e
        # senza la cartella non c'è modo di dire quale dei due si sta
        # guardando.
        "folder": row["folder"],
    }


# I nomi delle colonne di `reading`, nell'ordine in cui si leggono: chi è il
# brano, come suona, da dove viene. Le tabelle che ci aggiungono qualcosa di
# proprio — un costo, uno scarto, un numero d'ordine — se lo infilano dove
# serve invece di riscrivere tutta la fila.
READING_ORDER = ["file", "BPM", "key", "groove", "emotion", "mood", "genres",
                 "folder"]


def reading_config(frame: pd.DataFrame, table: pd.DataFrame) -> dict:
    """La configurazione delle colonne di `reading`: colorate e bloccate.

    Vuole tutte e due le tabelle perché i generi si colorano guardando la
    libreria e si nominano guardando le righe: vedi `genre_colors`.
    """
    return {"key": key_column(), "groove": groove_column(),
            "emotion": emotion_column(), "mood": mood_column(),
            "genres": genre_column(genre_colors(frame, table["genres"])),
            **read_only("file", "BPM", "folder")}


def genre_colors(frame: pd.DataFrame, shown) -> dict[str, str]:
    """I colori delle pastiglie: la libreria decide, la tabella chiede.

    `frame` è la libreria e serve solo a ORDINARE — i generi più frequenti
    prendono una tinta a testa, e sono gli stessi che la mappa colora.
    `shown` sono le liste di generi delle righe da disegnare, e servono a
    dire quali nomi vanno nominati: un genere che il vocabolario non nomina
    non viene disegnato come pastiglia, ricompare per esteso in mezzo alle
    altre. Prendere il vocabolario da qui e non da tutta la libreria vale
    qualche decina di millisecondi per tabella, sei volte a giro di pagina,
    su novantamila righe.

    Chi non entra fra i colorati non resta fuori dal dizionario: prende il
    grigio dell'"altro", che è la stessa sorte che ha sulla mappa.
    """
    ranked = (frame["top_genre"].value_counts().index.tolist()
              if "top_genre" in frame else [])
    colors = dict(zip(ranked[:len(PALETTE)], PALETTE))
    grey = OTHER_COLOR["dark" if dark() else "light"]
    for tags in shown:
        for genre in tags:
            colors.setdefault(genre, grey)
    return colors
