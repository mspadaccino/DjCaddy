"""Le colonne dei brani vestite da widget Streamlit.

La sostanza — tavolozze, regole colore, la lettura della riga — sta in
`core.viz.track_columns`, condivisa con l'app Qt: qui restano solo il tema
(che arriva dal browser, quindi è affare di Streamlit) e i wrapper
`st.column_config` che dicono a Streamlit come disegnare quelle colonne.
I nomi puri si re-importano qui sotto perché le altre viste e i test li
hanno sempre presi da questo modulo.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.viz.track_columns import (  # noqa: F401 — re-export per viste e test
    COLUMN_HELP, EMOTION_COLORS, EMOTION_OPTIONS, ENERGY_COLORS,
    GROOVE_COLORS, GROOVE_OPTIONS, KEY_COLORS, KEY_OPTIONS, LEVEL_OPTIONS,
    OTHER_COLOR, PALETTE, READING_ORDER, camelot_color, emotion_arrow,
    energy_level, groove_pill, reading)
from core.viz.track_columns import genre_colors as _genre_colors


def dark() -> bool:
    """Se il tema in uso è quello scuro. Al primo giro il tema può non essere
    ancora arrivato dal browser: in quel caso si disegna chiaro."""
    theme = getattr(getattr(st, "context", None), "theme", None)
    return getattr(theme, "type", None) == "dark"


def genre_colors(frame: pd.DataFrame, shown) -> dict[str, str]:
    """I colori delle pastiglie, col grigio del tema in uso.

    La regola sta in `core.viz.track_columns.genre_colors`; qui si aggiunge
    solo il tema, perché come lo si scopre è affare dell'app.
    """
    return _genre_colors(frame, shown, dark())


def read_only(*columns: str) -> dict:
    """Colonne che si guardano e basta, con la loro spiegazione.

    In una tabella con una casella da spuntare tutto il resto va bloccato a
    mano, o si finisce a correggere i BPM di un brano credendo di
    sceglierlo. E gia' che si passa di qui si attacca il punto interrogativo
    con quello che la colonna misura: una colonna che si chiama "sound" o
    "Δkey" non si spiega da sola, e chi la legge non ha nessun posto dove
    andare a chiedere.
    """
    return {name: st.column_config.Column(disabled=True,
                                          help=COLUMN_HELP.get(name))
            for name in columns}


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
        help="How UNIFORM the spacing between attacks is, 0.00 to 1.00 — "
             "not groove in the musical sense. A metronome reads 1.00, and "
             "so does an unbroken run of sixteenths: what lowers it is a "
             "rhythmic figure, some hits close and some far apart. So a "
             "track with a real groove tends to read LOW. Measured on one "
             "30-second window at the middle of the track. Same number as "
             "on the card and the board.")


def energy_column(label: str = "energy"):
    """L'energia come la scrive DJoid: un intero da 1 a 10, in rosso."""
    return st.column_config.MultiselectColumn(
        label, disabled=True, width="small",
        options=LEVEL_OPTIONS, color=ENERGY_COLORS,
        help="How hard the track pushes, 1 to 10. Four things measured at "
             "once — how many attacks per beat, how much power sits under "
             "200 Hz, where the spectral centre lies, and how much of the "
             "bass lands ON the beat — each turned into its rank across "
             "your library, then averaged. Loudness is NOT in it: that says "
             "how hard the master was pushed, not the track. The scale is "
             "deciles of YOUR library, so 10 means the top tenth of what "
             "you have, not an absolute level. Empty means the track has "
             "not been measured yet.")


def emotion_column(label: str = "emotion"):
    """Il verso del mood: su chiaro, giù buio."""
    return st.column_config.MultiselectColumn(
        label, disabled=True, width="small",
        options=EMOTION_OPTIONS, color=EMOTION_COLORS,
        help="Which way the track looks compared with the rest of YOUR "
             "library: up is among the brighter third, down among the "
             "darker third, nothing means it sits in the middle. Relative "
             "and not absolute on purpose — the model reads almost "
             "everything as bright in absolute terms, because it learned on "
             "a world where 'happy' is a far more common tag than 'sad'. "
             "Which moods put it there is in the next column.")


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


def reading_config(frame: pd.DataFrame, table: pd.DataFrame) -> dict:
    """La configurazione delle colonne di `reading`: colorate e bloccate.

    Vuole tutte e due le tabelle perché i generi si colorano guardando la
    libreria e si nominano guardando le righe: vedi `genre_colors`.
    """
    return {"key": key_column(), "energy": energy_column(),
            "groove": groove_column(),
            "emotion": emotion_column(), "mood": mood_column(),
            "genres": genre_column(genre_colors(frame, table["genres"])),
            **read_only("file", "BPM", "folder")}
