"""Il Chain Maker e la lavagna: le due metà di come si mette insieme un set.

`analysis.graph_playlist.GraphPlaylist` tiene la logica — brani, posizioni,
collegamenti. Questo modulo è lo strato sopra, e fa due cose che stanno in
due posti diversi della pagina.

**Si comanda dalle tabelle.** A sinistra la catena com'è finora, a destra i
candidati che escono dal brano su cui si sta: si spunta, si aggiunge, e la
catena cresce. Prima la rosa era una griglia di schede da cliccare, e non
funzionava — la stessa informazione che in colonna si legge e si ordina, lì
stava in riquadri da cento pixel, e ogni scelta passava per un componente
disegnato a mano.

**E si guarda la lavagna**, che sta in fondo, nella sezione della playlist, e
disegna la PLAYLIST: la forma di un set è la forma del set intero — quello
aperto da un M3U8, quello ordinato dal magic sort e la catena che ci si
appende — non del solo pezzo che si sta scrivendo in questo momento. Il
mestiere che sa fare è mostrare quella forma: l'ordine, i colori dei generi,
gli scarti fra un brano e il precedente. Ci si trascinano le schede per
disporle e c'è il cestino per toglierne una, ma niente di ciò che ci sta
dentro nasce lì.
"""

from __future__ import annotations

import colorsys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from analysis import mood_scale
from analysis.duplicates import normalized_name, song_key
from analysis.graph_playlist import GraphPlaylist, suggestions
from analysis.mixing import (BPM_TOLERANCE, TransitionCost, bpm_shift,
                             camelot_shift)
from views.components import NOW_PLAYING, fill_dock, play_table

_FRONTEND_DIR = Path(__file__).parent / "graph_board_frontend"
_graph_board = components.declare_component("graph_board", path=str(_FRONTEND_DIR))

_WHEEL_DIR = Path(__file__).parent / "camelot_wheel_frontend"
_camelot_wheel = components.declare_component("camelot_wheel", path=str(_WHEEL_DIR))

# Tavolozza duplicata da `views.map_analysis` apposta: importarla da lì
# creerebbe un giro (quel modulo importa questa sezione), e sono dodici
# colori — il doppione costa meno del giro.
PALETTE = ["#e0503b", "#3d9be0", "#3fbf7f", "#f2a33c", "#a06fd6", "#e06fa8",
           "#4dd0c4", "#c9b037", "#6f8fd6", "#d66f6f", "#7fbf3f", "#bf7fd6"]
OTHER_COLOR = {"light": "#9aa4b0", "dark": "#6b7684"}

# I brani spuntati in questo momento, in una tabella qualunque della pagina.
# La mappa li cerchia di giallo. Sta qui e non in `views.map_analysis` perché
# quel modulo importa questo: definirla di là chiuderebbe il giro.
TICKED = "map::ticked"

GRAPH_STATE = "map::graph"
GRAPH_SOURCE = "map::graph_source"
GRAPH_KEYS = "map::graph_keys"
GRAPH_KEYS_EVENT = "map::graph_keys_event"

# --- la lavagna, che ora disegna la playlist e non più la sola catena ------
# L'ultimo gesto già eseguito. Vedi `sendValue` nel frontend: il componente
# ridà lo stesso valore a ogni rerun, e senza ricordarsene un click sulla
# lavagna si rieseguirebbe all'infinito.
BOARD_EVENT = "map::board_event"
BOARD_PICKED = "map::board_picked"
# La misura scelta per l'altezza, tenuta in una chiave NORMALE e non solo in
# quella del radio. Un gesto sulla lavagna — riordinare, togliere un brano —
# esce da `render_board` prima che il radio esista, e Streamlit butta via lo
# stato di un widget che quel giro non ha disegnato: alla ripartenza si
# tornava a guardare la prima misura della lista. Riordinare un set non deve
# cambiare cosa si sta guardando del set.
BOARD_AXIS = "map::board_axis"

# Sopra questa quantità di brani il menu per nome non si apre più in fretta:
# si cerca prima, si sceglie dopo.
START_PICKER_MAX = 2000

# Quanti candidati proporre a ogni passo. Nove bastano a una scelta vera e
# stanno in una tabella senza doverla scorrere.
FRONTIER_SIZE = 9


def _camelot_color(camelot: str | None) -> str:
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


def _dark() -> bool:
    theme = getattr(getattr(st, "context", None), "theme", None)
    return getattr(theme, "type", None) == "dark"


def start_board(*tracks: str) -> None:
    """Comincia la lavagna dai brani scelti — per nome, o sulla mappa sopra.

    Sta qui e non nella mappa perché le chiavi di sessione della lavagna
    sono di questo modulo: chi la avvia deve poter dire quali brani e basta,
    senza sapere dove finiscono scritti.
    """
    _save(GraphPlaylist().start(*tracks))
    # La sorgente è l'ultimo: è quello appena messo, ed è da lì che si
    # continua — come dopo ogni altra aggiunta.
    st.session_state[GRAPH_SOURCE] = tracks[-1]


def _graph() -> GraphPlaylist:
    return GraphPlaylist.from_state(st.session_state.get(GRAPH_STATE))


def _save(graph: GraphPlaylist) -> None:
    st.session_state[GRAPH_STATE] = graph.to_state()


def _color_map(frame: pd.DataFrame) -> dict[str, str]:
    top = frame["top_genre"].value_counts().head(len(PALETTE)).index.tolist()
    return dict(zip(top, PALETTE))


def _some(row, column: str):
    """Il valore, o `None` se manca davvero.

    Serve perché un campo vuoto arriva qui come NaN, e NaN è vero: scritto
    su una scheda con `or` diventa la parola "nan" sotto al titolo, che
    sembra un dato invece che l'assenza di un dato.
    """
    if row is None:
        return None
    value = row[column]
    return value if pd.notna(value) and value != "" else None


# Cosa può dire l'ALTEZZA di una scheda sulla lavagna. L'asse x è già preso
# dall'ordine della scaletta, che non è negoziabile; l'altezza invece è libera
# e può portare la misura che in quel momento racconta il set.
HEIGHT_FIELDS = {"BPM": "bpm", "key": "camelot", "groove": "danceability",
                 "mood": "moods"}
# Quella che si apre da sé è il groove, non il BPM: un set si costruisce fra
# brani di tempo vicino — è il senso del costo di transizione — quindi la
# linea dei BPM nasce quasi piatta e non ha molto da dire, mentre la
# regolarità del ritmo sale e scende per tutta la serata. L'ordine delle voci
# resta quello: si cambia cosa si guarda per primo, non dove si clicca.
DEFAULT_HEIGHT = "groove"


def _measured(frame: pd.DataFrame, at_path: dict[str, int],
              tracks: list[str], axis: str) -> dict[str, float]:
    """Il valore grezzo della misura scelta, per i brani che ce l'hanno."""
    column = HEIGHT_FIELDS[axis]
    out: dict[str, float] = {}
    for path in tracks:
        value = _some(frame.iloc[at_path[path]] if path in at_path else None,
                      column)
        if value is None:
            continue
        if column == "moods":
            # Delle parole non si sa quanto sono alte: glielo dice
            # `mood_scale`, da −1 (buio) a +1 (chiaro).
            colour = mood_scale.valence(value)
            if colour is not None:
                out[path] = colour
        elif column == "camelot":
            # Il numero della ruota, non la lettera: è quello che dice di
            # quanto ci si sposta armonicamente.
            code = str(value).strip().upper()[:-1]
            if code.isdigit():
                out[path] = float(code)
        else:
            out[path] = float(value)
    return out


def _span_of(axis: str, values: dict[str, float],
             frame: pd.DataFrame) -> tuple[float, float]:
    """Fra quali due valori tendere l'altezza, per la misura scelta.

    NON sugli estremi della catena. Una catena di otto brani, su questa
    libreria, copre un BPM scarso — è il costo di transizione che fa il suo
    mestiere — e tenderla su sé stessa trasforma un ottavo di BPM in mezza
    lavagna: una salita che non esiste. Peggio, quando i valori sono tutti
    uguali non c'è nessuna scala da tendere e viene fuori una riga piatta
    senza spiegazione.

    Ogni misura ha invece una scala sua, e sempre la stessa: così due catene
    si confrontano, e piatto vuol dire davvero "non si muove".
    """
    if axis == "key":
        return (1.0, 12.0)                      # la ruota, tutta
    if axis == "groove":
        return _drive_span(frame)               # i decili della libreria
    if axis == "mood":
        return _mood_span(frame)                # i decili anche qui
    # Il tempo attorno a dove sta la catena, largo quanto il pitch fader:
    # oltre il ±6% la transizione costa comunque troppo per capitare.
    middle = sorted(values.values())[len(values) // 2] if values else 120.0
    return (middle * (1 - BPM_TOLERANCE), middle * (1 + BPM_TOLERANCE))


def _heights(frame: pd.DataFrame, at_path: dict[str, int],
             tracks: list[str], axis: str) -> dict[str, float]:
    """Per ogni brano, quanto in alto va: da 0 (in basso) a 1 (in cima)."""
    values = _measured(frame, at_path, tracks, axis)
    if not values:
        return {}
    low, high = _span_of(axis, values, frame)
    if high <= low:
        return {path: 0.5 for path in values}
    return {path: min(1.0, max(0.0, (value - low) / (high - low)))
            for path, value in values.items()}


def reordered(walk: list, moves: dict[int, float]) -> list:
    """L'ordine di una fila dopo che una riga ha cambiato numero.

    Pubblica perche' la usa anche la tabella della playlist, in
    `views.map_analysis`: e' lo stesso gesto — si riscrive il numero di una
    riga e la riga va li' — e due copie della stessa regola scivolerebbero.

    Si toglie il brano da dov'è e lo si rimette dove è stato chiesto, come
    una carta sfilata dal mazzo e reinfilata: gli altri scorrono e nessuno
    sparisce. Scambiare i due brani invece che spostarne uno sarebbe più
    facile da scrivere e sbagliato da usare — chi scrive 1 sull'ultima riga
    vuole quel brano in apertura, non l'apertura in fondo.
    """
    order = list(walk)
    for row, place in sorted(moves.items()):
        if not 0 <= row < len(walk):
            continue
        track = walk[row]
        target = max(1, min(len(order), int(place))) - 1
        order.remove(track)
        order.insert(target, track)
    return order


def mood_column():
    """La colonna del mood, con la sua spiegazione. Pubblica e una sola
    perché le tabelle che la portano sono sei, in due moduli: scritta a mano
    ogni volta, la spiegazione avrebbe sei versioni e cinque da aggiornare."""
    return st.column_config.Column(
        "mood", disabled=True,
        help="Before the dot, the rarest of this track's moods across your "
             "library — the one that tells it apart. Energetic sits on 89% "
             "of the library and Happy on 57%, so the strongest mood is "
             "almost the same for everyone. After the dot, the others, "
             "strongest first.")


def _read_only(*columns: str) -> dict:
    """Colonne che si guardano e basta.

    Gemella di quella in `views.map_analysis`, e duplicata per la stessa
    ragione della tavolozza: quel modulo importa questo, e importarlo di
    rimando chiuderebbe il giro. È una riga.
    """
    return {name: st.column_config.Column(disabled=True) for name in columns}


def _render_filters(frame: pd.DataFrame, pool) -> "np.ndarray | list":
    """I filtri della lavagna, e i brani che li passano.

    Sono suoi e non quelli della mappa qui sopra: la lavagna è un secondo
    modo di scegliere, non un'estensione del primo. Restringono la rosa e i
    due brani di partenza — cioè tutto quello che la lavagna propone — ma
    non toccano i brani che ci sono già finiti sopra: filtrare via un nodo
    già posato spezzerebbe una scaletta che qualcuno ha costruito.
    """
    keys = st.session_state.get(GRAPH_KEYS) or []
    kept = frame.loc[list(pool)] if len(pool) != len(frame) else frame

    # Scegliere una tonalità sulla ruota rilancia la pagina, e un pannello
    # che torna al suo stato di riposo si richiuderebbe sotto le dita al
    # primo click. Resta aperto finché la ruota è stata toccata almeno una
    # volta, anche dopo aver tolto l'ultima tonalità — chi sta filtrando non
    # ha finito solo perché ha svuotato la scelta.
    touched = GRAPH_KEYS_EVENT in st.session_state
    with st.expander(f"Filters — they narrow the roster"
                     f"{f' · {len(keys)} key(s)' if keys else ''}",
                     expanded=bool(keys or touched)):
        wheel, rest = st.columns([2, 3])

        with wheel:
            st.caption("Pick the keys you want to land on. Nothing picked "
                       "means every key is welcome.")
            event = _camelot_wheel(
                selected=keys, colors=_CAMELOT_COLORS, dark=_dark(),
                key="graph_wheel", default=None)
            if event and event.get("at") != st.session_state.get(GRAPH_KEYS_EVENT):
                st.session_state[GRAPH_KEYS_EVENT] = event.get("at")
                code = event.get("code")
                st.session_state[GRAPH_KEYS] = (
                    [k for k in keys if k != code] if code in keys
                    else keys + [code])
                st.rerun(scope="fragment")

        with rest:
            genres = Counter(g for tags in
                             frame["genres"].fillna("").str.split("; ")
                             for g in tags if g)
            chosen = st.multiselect(
                "Genres", [g for g, _ in genres.most_common()],
                key="graph_genres",
                help="A track carrying any of the chosen genres stays.")
            tempo = _range_of(frame, "bpm", 60.0, 200.0)
            bpm = st.slider("BPM", tempo[0], tempo[1], tempo, key="graph_bpm")
            swing = _range_of(frame, "danceability", 0.0, 1.0)
            dance = st.slider("Danceability", swing[0], swing[1], swing,
                              step=0.01, key="graph_dance",
                              help="Regularity of the onsets: low is loose, "
                                   "high is a straight kick.")
            if st.button("↺ Reset the filters", width="stretch"):
                _reset_filters()
                st.rerun(scope="fragment")

        if chosen:
            wanted = set(chosen)
            kept = kept[kept["genres"].fillna("").str.split("; ").map(
                lambda tags: bool(wanted & set(tags)))]
        if keys:
            kept = kept[kept["camelot"].isin(keys)]
        # Un brano senza BPM o senza danceability non viene escluso da un
        # intervallo su quel valore: non sappiamo dove cade, e farlo sparire
        # sarebbe rispondere "no" a una domanda che non è stata posta.
        kept = kept[kept["bpm"].isna() | kept["bpm"].between(*bpm)]
        kept = kept[kept["danceability"].isna()
                    | kept["danceability"].between(*dance)]
        st.caption(f"**{len(kept):,}** of {len(frame):,} tracks pass — "
                   "the roster and the two starting tracks come from these.")

    return kept.index.to_numpy()


def _range_of(frame: pd.DataFrame, column: str,
              fallback_low: float, fallback_high: float) -> tuple[float, float]:
    """Gli estremi veri di una colonna, per non offrire una corsa vuota.

    Uno slider 0..200 su una libreria che sta fra 110 e 130 è quasi tutto
    corsa morta. I due estremi devono comunque restare diversi fra loro,
    anche quando la colonna è vuota o porta un valore solo: uno slider che
    parte e finisce nello stesso punto non si disegna.
    """
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if not len(values):
        return (fallback_low, fallback_high)
    low, high = float(values.min()), float(values.max())
    return (low, high) if high > low else (low, low + 1.0)


def _reset_filters() -> None:
    for key in (GRAPH_KEYS, "graph_genres", "graph_bpm", "graph_dance"):
        st.session_state.pop(key, None)


_CAMELOT_COLORS = {f"{n}{mode}": _camelot_color(f"{n}{mode}")
                   for n in range(1, 13) for mode in "AB"}


def _drive_span(frame: pd.DataFrame) -> tuple[float, float]:
    """Fra quali due valori di danceability tendere la scala del colore.

    Non fra 0 e 1: la misura è la regolarità degli attacchi, e in una
    libreria vera si stringe attorno al mezzo — nella mia, metà dei brani
    sta fra 0.54 e 0.66. Tesa su 0..1 la scala dipingerebbe tutte le schede
    dello stesso grigio. Si tara sui decili di QUESTA libreria, come il lazo
    si tara sulla diagonale della mappa invece che su un raggio assoluto.
    """
    values = pd.to_numeric(frame["danceability"], errors="coerce").dropna()
    if len(values) < 20:
        return (0.0, 1.0)
    low, high = float(values.quantile(0.1)), float(values.quantile(0.9))
    return (low, high) if high > low else (0.0, 1.0)


def _drive(value, span: tuple[float, float]) -> float | None:
    """Dove cade questo brano nella scala, da 0 a 1. Fuori scala si appiattisce
    agli estremi: un valore raro non deve allargare la scala per tutti."""
    if value is None:
        return None
    low, high = span
    return min(1.0, max(0.0, (value - low) / (high - low)))


def _mood_stamp(frame: pd.DataFrame) -> tuple:
    """Un'impronta della libreria che costa niente da calcolare.

    La mappa si APPENDE — righe nuove in coda, mai in mezzo — quindi quante
    sono e qual è l'ultima bastano a dire che è la stessa libreria di prima.
    """
    return (len(frame), frame.at[len(frame) - 1, "path"]) if len(frame) else (0, "")


@st.cache_data(show_spinner=False)
def _mood_facts(_frame: pd.DataFrame, stamp: tuple) -> tuple[dict, float, float]:
    """Quanto è comune ogni mood, e fra quali valori tendere la sua scala.

    Le due cose insieme perché costano la stessa scorsa — quarantatré
    millisecondi su ottantasettemila righe — e perché scadono insieme:
    cambiano quando la mappa cresce, non quando si clicca una scheda. Senza
    tenerle da parte si rifarebbero cinque volte per giro di pagina, una per
    ogni tabella che porta la colonna del mood.

    La scala non si tende fra −1 e +1: il 90% della libreria sta fra −0,12 e
    +0,60, e sulla scala piena la curva sarebbe una riga piatta poco sopra il
    mezzo. I decili di QUESTA libreria, come per il groove.
    """
    moods = list(_frame["moods"]) if "moods" in _frame else []
    values = pd.Series([mood_scale.valence(m) for m in moods]).dropna()
    low, high = (float(values.quantile(0.1)), float(values.quantile(0.9))) \
        if len(values) >= 20 else (-1.0, 1.0)
    if high <= low:
        low, high = -1.0, 1.0
    return mood_scale.popularity(moods), low, high


def mood_popularity(frame: pd.DataFrame) -> dict[str, int]:
    """Quante volte ogni mood compare nella libreria. Pubblica perché la
    stessa colonna la scrivono anche le tabelle di `views.map_analysis`."""
    return _mood_facts(frame, _mood_stamp(frame))[0]


def _mood_span(frame: pd.DataFrame) -> tuple[float, float]:
    return _mood_facts(frame, _mood_stamp(frame))[1:]


def _way(value) -> int:
    """Il verso di uno scarto: +1 sale, -1 scende, 0 sta fermo.

    I confronti si convertono a mano: questi numeri arrivano dal frame, cioè
    da numpy, e sottrarre due suoi booleani è un errore invece che 1 o 0.
    """
    return int(value > 0) - int(value < 0)


def _gaps(source, row) -> dict:
    """Di quanto si muove `row` rispetto a `source`, misura per misura.

    Solo i numeri, senza deciderne la forma: la lavagna e la rosa li scrivono
    in due modi diversi — larghezze diverse, letture diverse — ma non devono
    calcolarli due volte, o prima o poi diranno due cose diverse.

    Una misura che manca da una delle due parti non compare affatto: non c'è
    scarto fra un numero e il nulla.
    """
    if source is None:
        return {}
    out = {}
    tempo = bpm_shift(_some(source, "bpm"), _some(row, "bpm"))
    if tempo is not None:
        out["bpm"] = round(tempo)
    wheel = camelot_shift(_some(source, "camelot"), _some(row, "camelot"))
    if wheel is not None:
        out["key"] = wheel
    here, there = _some(source, "danceability"), _some(row, "danceability")
    if here is not None and there is not None:
        out["dance"] = round(there - here, 2)
    return out


def _card_shifts(source, row) -> dict[str, tuple[str, int]]:
    """Gli scarti in forma corta, una cella per colonna della scheda.

    Scritti di seguito non ci starebbero, e abbreviarli in "+0 · -1 · +.05"
    su una riga a sé sarebbe un rebus. Incolonnati sotto ai valori che
    commentano diventano invece la seconda riga della stessa tabella, e le
    unità di misura le presta la riga sopra. Se un valore manca, manca la
    colonna: le due righe restano allineate perché le costruisce lo stesso
    giro.
    """
    gaps = _gaps(source, row)
    out = {}
    if "bpm" in gaps:
        out["bpm"] = (f"{gaps['bpm']:+d}", _way(gaps["bpm"]))
    if "key" in gaps:
        # Zero passi con la lettera cambiata è il relativo maggiore o minore:
        # non sale né scende, cambia colore al brano restando dov'è, e dargli
        # un verso direbbe una cosa falsa.
        steps, mode = gaps["key"]
        out["key"] = ((f"{steps:+d}", _way(steps)) if steps
                      else (("rel", 0) if mode else ("=", 0)))
    if "dance" in gaps:
        # Senza lo zero davanti: sotto una colonna di trentotto pixel "+0.05"
        # e "+.05" dicono la stessa cosa e solo uno dei due ci sta.
        out["dance"] = (f"{gaps['dance']:+.2f}".replace("0.", "."),
                        _way(gaps["dance"]))
    return out


def _label(name: str) -> str:
    name = Path(name).stem if "/" in name or "\\" in name else name
    return name if len(name) <= 22 else name[:21] + "…"


def render_chain_maker(frame: pd.DataFrame, cost: TransitionCost, pool,
                       at_path: dict[str, int], chosen: list[int],
                       set_playlist, add_to_playlist) -> None:
    """Una catena che cresce un brano alla volta, in due tabelle.

    `set_playlist` prende una lista di indici (nello stesso `frame`) e la
    rende la playlist della pagina — lo stesso canale che usa il resto della
    mappa, così "manda alla playlist" qui sotto finisce nello stesso posto
    del disegno a lazo qui sopra. `add_to_playlist` la aggiunge in coda
    invece di sostituirla: questa è un modo di far crescere un set, e un set
    cominciato altrove — caricato da un M3U8, o messo insieme qui sopra — non
    deve sparire perché gli si manda una catena.

    La lavagna non sta più qui: disegna la playlist, in fondo alla pagina.
    Guardare la forma di ciò che si sta costruendo ha senso su TUTTO il set —
    quello caricato da un file, quello ordinato dal magic sort, e la catena
    che ci si appende — non sul solo pezzo che si sta scrivendo adesso.
    """
    st.caption(
        "Grow a set one track at a time: on the left the chain as it stands, "
        "on the right what mixes out of the track you are standing on. Tick, "
        "add, repeat — and send it to the playlist, where the board shows the "
        "shape the whole set is taking.")

    graph = _graph()
    pool = _render_filters(frame, pool)

    if not len(graph):
        _render_start(frame, pool, chosen)
        return

    # Ogni brano si confronta con quello che lo precede NELLA SCALETTA, non
    # con quello da cui è stato scelto: è l'ordine in cui il set uscirà, e
    # quindi l'unico rispetto a cui "sale" o "scende" vuol dire qualcosa.
    walk = graph.walk()
    before = {track: walk[n - 1] for n, track in enumerate(walk) if n}

    _render_tables(frame, cost, pool, at_path, graph, walk, before)

    c1, c2, c3 = st.columns(3)
    if c1.button("↺ Start over", width="stretch",
                 help="Empties the chain. The playlist is not touched."):
        st.session_state[GRAPH_STATE] = None
        st.session_state[GRAPH_SOURCE] = None
        st.rerun(scope="fragment")
    # I due escono dalla sezione: la playlist si disegna fuori, e un rerun
    # del solo frammento la lascerebbe indietro di una mossa.
    if c2.button("➡️ Append to playlist", type="primary", width="stretch",
                 help="The chain goes after what the playlist already holds."):
        add_to_playlist([at_path[p] for p in walk if p in at_path])
        st.rerun()
    if c3.button("↺ Send as a new playlist", width="stretch",
                 help="Starts over: what is in the playlist now is dropped."):
        set_playlist([at_path[p] for p in walk if p in at_path])
        st.rerun()

    _render_by_hand(frame, pool, at_path, graph)


def _spelled(row, source, common: dict[str, int]) -> dict:
    """Le colonne comuni alle due tabelle: quelle che stanno sulle schede.

    Le stesse voci e con gli stessi nomi da una parte e dall'altra, perché il
    brano che si guarda a destra è quello che comparirà a sinistra, e
    cambiargli le colonne nel passaggio costringerebbe a ritrovarlo.
    """
    bpm, dance = _some(row, "bpm"), _some(row, "danceability")
    gaps = _gaps(source, row)
    steps = gaps.get("key")
    return {
        "file": row["name"],
        "BPM": round(bpm) if bpm is not None else None,
        "key": _some(row, "camelot") or "",
        "groove": round(dance, 2) if dance is not None else None,
        "Δbpm": gaps.get("bpm"),
        "Δkey": (steps[0] if steps[0] else ("rel" if steps[1] else "="))
        if steps is not None else None,
        "Δgroove": gaps.get("dance"),
        "mood": mood_scale.summary(row["moods"], common),
        "genres": row["genres"],
        # Da dove viene il file. Due brani con lo stesso nome esistono, e
        # senza la cartella non c'è modo di dire quale dei due si sta
        # guardando.
        "folder": row["folder"],
    }


def _ticks(axis: str, values: dict[str, float],
           frame: pd.DataFrame) -> list[dict]:
    """Le tacche della scala verticale, dal basso in alto.

    Senza, l'altezza è una forma senza unità: si vede che sale, non da dove a
    dove. Tre bastano — fondo, mezzo, cima — e il numero è quello vero della
    misura, non una percentuale.
    """
    if not values:
        return []
    if axis == "mood":
        # Qui la tacca dice una parola: "+0.28" non vuol dire niente a
        # nessuno, e la scala è una lettura del mood, non la sua misura.
        return [{"at": at, "label": name}
                for at, name in zip((0.0, 0.5, 1.0), ("dark", "mid", "bright"))]
    low, high = _span_of(axis, values, frame)
    if high <= low:
        return []
    digits = 0 if axis in ("key", "BPM") else 2
    return [{"at": at, "label": f"{low + (high - low) * at:.{digits}f}"}
            for at in (0.0, 0.5, 1.0)]


@st.fragment
def render_board(frame: pd.DataFrame, at_path: dict[str, int],
                 playlist: list[int], drop, move) -> None:
    """La playlist come lavagna: una scheda per brano, in fila come suonerà.

    Prima disegnava la sola catena del Chain Maker, ed era troppo poco: un
    set si mette insieme da più parti — un M3U8 aperto, un lazo sulla mappa,
    una catena costruita a mano — e la forma che conta è quella del set
    intero. Qui la lavagna guarda la playlist, quindi mostra tutto ciò che ci
    è finito dentro, da qualunque parte sia arrivato.

    **Un punto si trascina, ma solo lungo la fila.** La regola resta quella
    per cui una lavagna esiste — ordine in orizzontale, misura in verticale —
    e proprio per questo la x si può tirare: la x È l'ordine, quindi
    trascinarla vuol dire riordinare, e `move` porta il nuovo ordine alla
    playlist. La y no, e non per prudenza: la si sposterebbe senza aver
    cambiato niente di ciò che misura, e la scala a sinistra direbbe il falso.
    Era questo il guasto del trascinamento libero di prima, non il
    trascinamento in sé.

    È un frammento perché un click su una scheda fa ripartire lo script, e
    ripartire per intero vuol dire ridisegnare la mappa da ottantamila punti.
    """
    paths = [frame.at[i, "path"] for i in playlist]
    if not paths:
        return

    # Il gesto si applica PRIMA di disegnare qualsiasi cosa. Il valore del
    # componente sta in sessione sotto la sua chiave, quindi si può leggere
    # in cima al giro, invece di scoprirlo dopo aver già disegnato la scena
    # com'era prima.
    event = st.session_state.get("playlist_board_widget")
    if event and event.get("at") != st.session_state.get(BOARD_EVENT):
        st.session_state[BOARD_EVENT] = event.get("at")
        who, kind = event.get("id"), event.get("type")
        if kind == "click" and who in paths:
            st.session_state[BOARD_PICKED] = who
        elif kind == "play" and who in paths:
            st.session_state[NOW_PLAYING] = who
        elif kind == "remove" and who in at_path:
            # Togliere una scheda toglie il brano dalla PLAYLIST: la lavagna
            # non ha più una copia sua da cui cancellarlo.
            drop(at_path[who])
            return
        elif kind == "move" and who in paths:
            # Stessa regola della colonna "#" nelle tabelle, e apposta la
            # stessa funzione: il brano si sfila da dov'era e si reinfila
            # dove è stato lasciato, gli altri scorrono.
            where = event.get("to")
            if isinstance(where, int) and 0 <= where < len(paths):
                order = reordered(paths, {paths.index(who): where + 1})
                if order != paths:
                    move([at_path[p] for p in order if p in at_path])
                    return

    axis = st.radio("Height means", list(HEIGHT_FIELDS), horizontal=True,
                    index=list(HEIGHT_FIELDS).index(
                        st.session_state.get(BOARD_AXIS, DEFAULT_HEIGHT)),
                    key="board_axis_pick",
                    help="Left to right is always the playlist order. This "
                         "picks what the vertical axis says.")
    # Ricordata fuori dal widget: vedi BOARD_AXIS.
    st.session_state[BOARD_AXIS] = axis

    values = _measured(frame, at_path, paths, axis)
    heights = _heights(frame, at_path, paths, axis)
    color_of = _color_map(frame)
    other = OTHER_COLOR["dark" if _dark() else "light"]
    span = _drive_span(frame)
    before = {path: paths[n - 1] for n, path in enumerate(paths) if n}

    nodes = []
    for position, path in enumerate(paths, start=1):
        row = frame.iloc[at_path[path]] if path in at_path else None
        previous = at_path.get(before.get(path))
        came_from = frame.iloc[previous] if previous is not None else None
        name = row["name"] if row is not None else Path(path).stem
        genre = row["top_genre"] if row is not None else None
        camelot = _some(row, "camelot")
        bpm, dance = _some(row, "bpm"), _some(row, "danceability")
        nodes.append({
            "id": path, "n": position, "height": heights.get(path, 0.5),
            "label": _label(name),
            "color": color_of.get(genre, other),
            "bpm": f"{bpm:.0f}" if bpm is not None else "",
            "camelot": camelot or "",
            "keyColor": _camelot_color(camelot),
            "dance": f"{dance:.2f}" if dance is not None else "",
            "drive": _drive(dance, span),
            "genre": _label(genre) if genre else "",
            "shift": _card_shifts(came_from, row) if row is not None else {},
        })

    _graph_board(nodes=nodes, ticks=_ticks(axis, values, frame),
                 selected=st.session_state.get(BOARD_PICKED),
                 dark=_dark(), key="playlist_board_widget", default=None)

    st.caption("Left to right the set plays; how high a track sits is the "
               "measure above, on the scale at the left. **Hover** a point "
               "for its numbers, **click** it to pick it — underneath, **▶** "
               "listens to it and the **bin** takes it out of the playlist. "
               "**Drag** a point sideways to move it in the set: the others "
               "slide and the wave re-forms as you cross a lane. "
               "**Scroll** zooms, dragging the background moves, and **⛶** "
               "goes full screen.")

    # Il lettore in fondo alla pagina se lo ridisegna questa sezione. Un ▶
    # qui dentro fa ripartire il solo frammento: `app.py` non viene
    # rieseguito, e il lettore resterebbe sul brano di prima. Va chiamata
    # anche nel giro intero, o Streamlit non riserva il posto per le
    # ripartenze.
    fill_dock("board")


def _render_tables(frame: pd.DataFrame, cost: TransitionCost, pool,
                   at_path: dict[str, int], graph: GraphPlaylist,
                   walk: list[str], before: dict[str, str]) -> None:
    """La catena a sinistra, i candidati a destra, e il grafo che ne segue.

    Le tabelle sono il comando e la lavagna è il quadro. Il contrario — la
    rosa disegnata come schede da cliccare — costringeva a scegliere fra
    riquadri di cento pixel dove la stessa informazione, in colonna, si legge
    e si ordina; e ogni scelta passava per un componente disegnato a mano,
    con tutto quello che comporta un gesto che deve sopravvivere a un giro di
    pagina.
    """
    common = mood_popularity(frame)
    chain, roster = st.columns(2)

    with chain:
        st.markdown(f"**The chain — {len(walk)} track(s)**")
        table = pd.DataFrame([
            {"#": n + 1,
             **_spelled(frame.iloc[at_path[path]],
                        frame.iloc[at_path[before[path]]]
                        if path in before and before[path] in at_path else None,
                        common),
             "_path": path}
            for n, path in enumerate(walk) if path in at_path])
        # La firma della catena: chi porta questa nella propria chiave
        # rinasce appena l'ordine cambia. Per la tabella è l'unico modo di
        # dimenticare la riga appena spostata, che altrimenti resterebbe
        # scritta nello stato del widget e si riapplicherebbe a ogni giro.
        signature = "|".join(walk)
        chain_key = f"graph_chain_editor::{signature}"
        play_table("graph_chain", table,
                   ["#", "BPM", "key", "groove",
                    "Δbpm", "Δkey", "Δgroove", "file", "mood", "genres",
                    "folder"],
                   {"#": st.column_config.NumberColumn(
                       "#", min_value=1, max_value=max(len(walk), 1), step=1,
                       help="Write the position you want this track in: the "
                            "row moves there and the others slide."),
                    "mood": mood_column(),
                    **_read_only("file", "BPM", "key", "groove",
                                 "Δbpm", "Δkey", "Δgroove", "genres",
                                 "folder")},
                   editor_key=chain_key)

        moves = {int(row): values["#"]
                 for row, values in st.session_state.get(chain_key, {})
                 .get("edited_rows", {}).items() if "#" in values}
        order = reordered(walk, moves) if moves else walk
        if order != walk:
            # Ricostruire invece di ricucire i collegamenti: una sequenza
            # scritta a mano È una fila, e un grafo ramificato non
            # sopravviverebbe comunque a un ordine che lo appiattisce.
            graph = GraphPlaylist().start(*order)
            _save(graph)
            st.session_state[GRAPH_SOURCE] = order[-1]
            st.rerun(scope="fragment")
        # La sorgente di default è l'ultimo arrivato, che è da dove si
        # continua nove volte su dieci; cambiarla serve a ramificare.
        # La chiave porta dentro la catena intera: finché non cambia, la
        # scelta fatta a mano resta; appena cambia, il menu è un altro menu e
        # riparte dal fondo — che è dove si è appena arrivati. Ci vuole
        # l'ordine e non solo la lunghezza: riordinando le righe la lunghezza
        # resta quella, e il menu continuava a nominare un brano mentre la
        # rosa ne lavorava un altro.
        # Riscrivere il valore di un widget già creato Streamlit lo vieta, e
        # cancellarne la chiave a metà pagina lasciava il menu a indicare un
        # brano diverso da quello su cui la rosa stava lavorando.
        here = st.selectbox(
            "Branch from", walk, index=len(walk) - 1,
            format_func=lambda p: frame.at[at_path[p], "name"]
            if p in at_path else Path(p).stem,
            key=f"graph_branch_from::{signature}")
        if st.button("🗑 Remove it from the chain", width="stretch",
                     disabled=len(walk) < 2):
            graph.remove(here)
            _save(graph)
            st.session_state[GRAPH_SOURCE] = graph.tracks[-1] if graph.tracks else None
            st.rerun(scope="fragment")

    st.session_state[GRAPH_SOURCE] = here
    with roster:
        _render_roster(frame, cost, pool, at_path, graph, here)


def _render_roster(frame: pd.DataFrame, cost: TransitionCost, pool,
                   at_path: dict[str, int], graph: GraphPlaylist,
                   source_path: str) -> None:
    """I candidati che escono dal brano scelto, da spuntare e aggiungere."""
    source_idx = at_path.get(source_path)
    if source_idx is None:
        return
    source = frame.iloc[source_idx]
    common = mood_popularity(frame)
    st.markdown(f"**Mixes out of — {_label(source['name'])}**")

    taken = {at_path[p] for p in graph.tracks if p in at_path}
    picks = suggestions(cost, source_idx, taken, k=FRONTIER_SIZE, pool=pool,
                        key_of=lambda i: normalized_name(
                            Path(frame.at[i, "path"])),
                        song_of=lambda i: song_key(Path(frame.at[i, "path"])))
    if not picks:
        st.info("No candidate left that passes the filters.")
        return

    table = pd.DataFrame([
        {"Add": False, "cost": round(value, 3),
         **_spelled(frame.iloc[i], source, common),
         # Le copie dello stesso pezzo restano una voce sola. Il numero dice
         # quante ce ne sono: si aggiunge la più economica, e se ne serve
         # un'altra precisa c'è "Add a track by name" qui sotto.
         "copies": len(copies) if len(copies) > 1 else None,
         "_path": frame.at[i, "path"], "_row": i}
        for i, value, copies in picks])

    edited = play_table(
        "graph_roster", table,
        ["Add", "cost", "BPM", "key", "groove",
         "Δbpm", "Δkey", "Δgroove", "file", "copies", "mood", "genres",
         "folder"],
        {"Add": st.column_config.CheckboxColumn(
            "Add", help="Tick what you want next, then the button below."),
         "mood": mood_column(),
         **_read_only("cost", "file", "BPM", "key", "groove",
                      "Δbpm", "Δkey", "Δgroove", "copies", "genres",
                      "folder")},
        # Come per il menu: cambiata la sorgente o cresciuta la catena, le
        # righe sotto sono altre e le spunte di prima indicherebbero brani
        # che nessuno ha scelto.
        editor_key=f"graph_roster_editor::{source_path}::{len(graph)}")

    wanted = [int(i) for i in edited.loc[edited["Add"], "_row"]]
    # La mappa sta più in alto e si disegna prima di questa tabella: la
    # spunta si annota qui e viene cerchiata al giro successivo, che è quello
    # che parte da sola appena si spunta.
    st.session_state[TICKED] = wanted
    if st.button(f"➕ Add {len(wanted)} to the chain", type="primary",
                 width="stretch", disabled=not wanted):
        # In fila uno dietro l'altro: spuntarne tre vuol dire "poi questi
        # tre", e attaccarli tutti alla stessa sorgente farebbe tre rami
        # invece di un seguito.
        previous = source_path
        for i in wanted:
            graph.add(previous, frame.at[i, "path"])
            previous = frame.at[i, "path"]
        _save(graph)
        st.session_state[GRAPH_SOURCE] = previous
        st.rerun(scope="fragment")


def _render_by_hand(frame: pd.DataFrame, pool, at_path: dict[str, int],
                    graph: GraphPlaylist) -> None:
    """Attaccare un brano scelto per nome, fuori dalla rosa.

    La rosa risponde a "cosa ci mixa dietro"; questo risponde a "voglio
    QUESTO". Sono due domande diverse e la seconda capita davvero: un brano
    che si è deciso di suonare esiste prima del grafo, e senza questa via
    andrebbe cercato spostando la sorgente finché la rosa non lo tira fuori
    — cioè piegando lo strumento invece di usarlo.

    Il collegamento resta quello di sempre: nasce attaccato alla sorgente,
    perché anche una scelta a mano viene DA qualche parte nella scaletta.
    """
    source_path = st.session_state.get(GRAPH_SOURCE)
    if source_path is None or source_path not in graph:
        return
    here = {at_path[p] for p in graph.tracks if p in at_path}
    options = [i for i in pool.tolist() if i not in here]
    if not options:
        return

    with st.expander("Add a track by name — outside the roster"):
        options = _narrowed(frame, options, "graph_by_hand_search")
        if options is None:
            return
        chosen = st.selectbox(
            "Track", options, index=None, key="graph_by_hand",
            format_func=lambda i: frame.at[i, "name"],
            placeholder="type part of a name")
        if st.button("➕ Attach it to the current source", type="primary",
                     disabled=chosen is None):
            graph.add(source_path, frame.at[chosen, "path"])
            _save(graph)
            st.session_state[GRAPH_SOURCE] = frame.at[chosen, "path"]
            st.rerun(scope="fragment")


def _narrowed(frame: pd.DataFrame, options: list[int], key: str) -> list[int] | None:
    """Le voci fra cui scegliere per nome, o `None` se sono ancora troppe.

    Sopra qualche migliaio il menu dei nomi smette di aprirsi in fretta: si
    cerca prima e si sceglie dopo. Vale ovunque si scelga un brano scrivendo
    il nome, e su una libreria vera scatta sempre — quindi la ricerca non è
    un ripiego, è la via normale.
    """
    if len(options) <= START_PICKER_MAX:
        return options
    search = st.text_input("Name contains", key=key,
                           placeholder="too many tracks — search by name first")
    if not search.strip():
        st.caption("Type part of a name to search the library.")
        return None
    wanted = search.strip().lower()
    found = [i for i in options if wanted in frame.at[i, "name"].lower()]
    if len(found) > START_PICKER_MAX:
        st.caption(f"{len(found):,} match — narrow the search further.")
        return None
    if not found:
        st.caption("Nothing matches that.")
        return None
    return found


def _render_start(frame: pd.DataFrame, pool, chosen: list[int]) -> None:
    st.markdown("**Start the board with a track.** Everything else grows off "
               "it, one suggestion at a time.")

    # Quello che è selezionato sulla mappa viene per primo: è già stato
    # scelto, e ricercarlo per nome in un menu sarebbe farlo scegliere due
    # volte. La ricerca resta sotto, per quando la mappa non c'entra.
    if chosen:
        names = ", ".join(_label(frame.at[i, "name"]) for i in chosen[:3])
        if len(chosen) > 3:
            names += f", and {len(chosen) - 3} more"
        picked, rest = st.columns([5, 2])
        picked.markdown(f"Selected on the map: **{names}**")
        rest.markdown("<div style='height:.2em'></div>", unsafe_allow_html=True)
        if rest.button("▶ Start from the selection", type="primary",
                       width="stretch"):
            start_board(*[frame.at[i, "path"] for i in chosen])
            st.rerun(scope="fragment")
        st.caption("…or pick a different one by name:")

    options = _narrowed(frame, pool.tolist(), "graph_start_search")
    if options is None:
        return

    c1, c2 = st.columns([5, 2])
    first = c1.selectbox("Track", options, index=None,
                         format_func=lambda i: frame.at[i, "name"],
                         key="graph_start_first",
                         placeholder="type part of a name")
    c2.markdown("<div style='height:1.8em'></div>", unsafe_allow_html=True)
    if c2.button("▶ Start the board", type="primary", width="stretch",
                disabled=first is None):
        start_board(frame.at[first, "path"])
        st.rerun(scope="fragment")
