"""La lavagna: un set che cresce un brano alla volta.

`analysis.graph_playlist.GraphPlaylist` tiene la logica — brani, posizioni,
collegamenti. Questo modulo è lo strato sopra.

**Si comanda dalle tabelle e si guarda la lavagna.** A sinistra la catena
com'è finora, a destra i candidati che escono dal brano su cui si sta: si
spunta, si aggiunge, e il disegno sotto si aggiorna. Prima la rosa era una
griglia di schede da cliccare, e non funzionava — la stessa informazione che
in colonna si legge e si ordina, lì stava in riquadri da cento pixel, e ogni
scelta passava per un componente disegnato a mano.

Alla lavagna resta il mestiere che sa fare, che è mostrare la forma: la
catena, i colori dei generi, gli scarti fra un brano e il precedente. Ci si
trascinano le schede per disporle e c'è il cestino per toglierne una, ma
niente di ciò che vi si costruisce nasce lì.
"""

from __future__ import annotations

import colorsys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from analysis.duplicates import normalized_name
from analysis.graph_playlist import GraphPlaylist, suggestions
from analysis.mixing import TransitionCost, bpm_shift, camelot_shift
from views.components import play_table

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

GRAPH_STATE = "map::graph"
GRAPH_SOURCE = "map::graph_source"
GRAPH_KEYS = "map::graph_keys"
GRAPH_KEYS_EVENT = "map::graph_keys_event"
# L'ultimo gesto già eseguito. Vedi `sendValue` nel frontend: il componente
# ridà lo stesso valore a ogni rerun, e senza ricordarsene un click sulla
# lavagna si rieseguirebbe all'infinito.
GRAPH_EVENT = "map::graph_event"

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
HEIGHT_FIELDS = {"BPM": "bpm", "key": "camelot", "groove": "danceability"}
GRAPH_AXIS = "map::graph_axis"


def _heights(frame: pd.DataFrame, at_path: dict[str, int],
             tracks: list[str], axis: str) -> dict[str, float]:
    """Per ogni brano, quanto in alto va: da 0 (in basso) a 1 (in cima).

    La scala si tende sui brani CHE CI SONO, non sulla libreria: una catena
    che vive fra 118 e 124 BPM, misurata sull'intera libreria, sarebbe una
    riga piatta — e la riga piatta è proprio ciò che l'asse dovrebbe smentire
    o confermare.
    """
    column = HEIGHT_FIELDS[axis]
    raw: dict[str, float] = {}
    for path in tracks:
        row = frame.iloc[at_path[path]] if path in at_path else None
        value = _some(row, column)
        if value is None:
            continue
        if column == "camelot":
            # Il numero della ruota, non la lettera: è quello che dice di
            # quanto ci si sposta armonicamente.
            code = str(value).strip().upper()
            raw[path] = int(code[:-1]) if code[:-1].isdigit() else None
            if raw[path] is None:
                raw.pop(path)
        else:
            raw[path] = float(value)
    if not raw:
        return {}
    low, high = min(raw.values()), max(raw.values())
    if high <= low:
        return {path: 0.5 for path in raw}
    return {path: (value - low) / (high - low) for path, value in raw.items()}


def _place_on_axis(graph: GraphPlaylist, frame: pd.DataFrame,
                   at_path: dict[str, int], only=None) -> None:
    """Mette i brani al posto che la regola in vigore assegna loro.

    Con `only` tocca solo quelli, e il resto della lavagna resta com'è: chi
    l'ha disposta a mano non se la vede disfare per aver aggiunto un brano.
    """
    axis = st.session_state.get(GRAPH_AXIS)
    if axis not in HEIGHT_FIELDS:
        return
    spread = GraphPlaylist(places=dict(graph.places), links=list(graph.links),
                           order=list(graph.order))
    spread.arrange(_heights(frame, at_path, graph.walk(), axis))
    for track in (graph.walk() if only is None else only):
        if track in spread.places:
            graph.places[track] = spread.places[track]


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


def render_graph_builder(frame: pd.DataFrame, cost: TransitionCost, pool,
                         at_path: dict[str, int], chosen: list[int],
                         set_playlist) -> None:
    """La sezione lavagna: parte da due brani, poi cresce un passo alla volta.

    `set_playlist` prende una lista di indici (nello stesso `frame`) e la
    rende la playlist della pagina — lo stesso canale che usa il resto della
    mappa, così "manda alla playlist" qui sotto finisce nello stesso posto
    del disegno a lazo qui sopra.
    """
    st.caption(
        "Put one track on the board and grow the set a step at a time: pick "
        "a card, look at what mixes out of it, take one. Cards can be "
        "dragged anywhere; the line between two of them records which "
        "suggestion came from where, and is not decoration.")

    graph = _graph()
    dark = _dark()
    pool = _render_filters(frame, pool)

    if not len(graph):
        _render_start(frame, pool, chosen)
        return

    # Il gesto della lavagna si applica PRIMA di disegnare qualsiasi cosa.
    # Il valore del componente sta in sessione sotto la sua chiave, quindi si
    # può leggere in cima al giro: chiederlo al componente vorrebbe dire
    # averlo già disegnato, e disegnarlo con le posizioni di prima del gesto
    # significa rimandargli indietro la scheda dove stava — che è come si
    # perde uno spostamento appena fatto.
    event = st.session_state.get("graph_board_widget")
    if event and event.get("at") != st.session_state.get(GRAPH_EVENT):
        st.session_state[GRAPH_EVENT] = event.get("at")
        moved = event.get("id")
        if event.get("type") == "move" and moved in graph:
            graph.move(moved, event["x"], event["y"])
            _save(graph)
        elif event.get("type") == "click" and moved in graph:
            st.session_state[GRAPH_SOURCE] = moved
        elif event.get("type") == "remove" and moved in graph:
            graph.remove(moved)
            _save(graph)
            if st.session_state.get(GRAPH_SOURCE) == moved:
                st.session_state[GRAPH_SOURCE] = (graph.tracks[-1]
                                                  if graph.tracks else None)

    color_of = _color_map(frame)
    other = OTHER_COLOR["dark" if dark else "light"]
    selected = st.session_state.get(GRAPH_SOURCE)

    # Ogni scheda si confronta con quella che la precede NELLA SCALETTA, non
    # con quella da cui è stata scelta: è l'ordine in cui il set uscirà, e
    # quindi l'unico rispetto a cui "sale" o "scende" vuol dire qualcosa.
    walk = graph.walk()
    before = {track: walk[n - 1] for n, track in enumerate(walk) if n}
    span = _drive_span(frame)

    _render_tables(frame, cost, pool, at_path, graph, walk, before)

    axis = st.radio("Height means", list(HEIGHT_FIELDS), horizontal=True,
                    key="graph_axis_pick",
                    help="Left to right is always the playlist order. This "
                         "picks what the vertical axis says.")
    # Si ridispone quando si sceglie una misura diversa, non a ogni giro:
    # altrimenti uno spostamento a mano durerebbe fino al primo click su
    # qualunque cosa, che è come non poterlo fare.
    if axis != st.session_state.get(GRAPH_AXIS):
        st.session_state[GRAPH_AXIS] = axis
        _place_on_axis(graph, frame, at_path)
        _save(graph)

    nodes = []
    for path in graph.tracks:
        idx = at_path.get(path)
        row = frame.iloc[idx] if idx is not None else None
        previous = at_path.get(before.get(path))
        came_from = frame.iloc[previous] if previous is not None else None
        name = row["name"] if row is not None else Path(path).stem
        genre = row["top_genre"] if row is not None else None
        camelot = _some(row, "camelot")
        bpm, dance = _some(row, "bpm"), _some(row, "danceability")
        x, y = graph.places[path]
        nodes.append({
            "id": path, "x": x, "y": y, "label": _label(name),
            "color": color_of.get(genre, other),
            "bpm": f"{bpm:.0f}" if bpm is not None else "",
            "camelot": camelot or "",
            "keyColor": _camelot_color(camelot),
            "dance": f"{dance:.2f}" if dance is not None else "",
            "drive": _drive(dance, span),
            "genre": _label(genre) if genre else "",
            "shift": _card_shifts(came_from, row) if row is not None else {},
        })
    links = [{"a": a, "b": b} for a, b in graph.links]

    _graph_board(nodes=nodes, links=links, selected=selected,
                 dark=dark, key="graph_board_widget", default=None)

    st.caption("Left to right the cards follow the playlist; how high they "
               "sit is the measure below. **Drag** one to break the rule "
               "where it suits you — picking a measure again puts them all "
               "back on it. The **bin** under the selected card removes it, "
               "**scroll** zooms, and **⛶** goes full screen.")

    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    if c1.button("↺ Restart the board", width="stretch"):
        st.session_state[GRAPH_STATE] = None
        st.session_state[GRAPH_SOURCE] = None
        st.rerun(scope="fragment")
    if c2.button("⇥ Straighten", width="stretch",
                 help="Line the cards up in the order the playlist will read."):
        graph.straighten()
        _save(graph)
        st.rerun(scope="fragment")
    if c3.button("➡️ Send to playlist", type="primary", width="stretch"):
        order = [at_path[p] for p in graph.walk() if p in at_path]
        set_playlist(order)
        # L'unico che esce dalla sezione: la playlist si disegna fuori, e un
        # rerun del solo frammento la lascerebbe indietro di una mossa.
        st.rerun()
    c4.caption(f"{len(graph)} track(s) on the board.")

    _render_by_hand(frame, pool, at_path, graph)


def _spelled(row, source) -> dict:
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
        "genres": row["genres"],
    }


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
    chain, roster = st.columns(2)

    with chain:
        st.markdown(f"**The chain — {len(walk)} track(s)**")
        table = pd.DataFrame([
            {"#": n + 1,
             **_spelled(frame.iloc[at_path[path]],
                        frame.iloc[at_path[before[path]]]
                        if path in before and before[path] in at_path else None),
             "_path": path}
            for n, path in enumerate(walk) if path in at_path])
        play_table("graph_chain", table,
                   ["#", "file", "BPM", "key", "groove",
                    "Δbpm", "Δkey", "Δgroove", "genres"],
                   _read_only("#", "file", "BPM", "key", "groove",
                              "Δbpm", "Δkey", "Δgroove", "genres"),
                   editable=False, editor_key="graph_chain_editor")
        # La sorgente di default è l'ultimo arrivato, che è da dove si
        # continua nove volte su dieci; cambiarla serve a ramificare.
        # La chiave porta dentro la lunghezza della catena: finché non
        # cambia, la scelta fatta a mano resta; appena cresce, il menu è un
        # altro menu e riparte dal fondo — che è dove si è appena arrivati.
        # Riscrivere il valore di un widget già creato Streamlit lo vieta, e
        # cancellarne la chiave a metà pagina lasciava il menu a indicare un
        # brano diverso da quello su cui la rosa stava lavorando.
        here = st.selectbox(
            "Branch from", walk, index=len(walk) - 1,
            format_func=lambda p: frame.at[at_path[p], "name"]
            if p in at_path else Path(p).stem,
            key=f"graph_branch_from::{len(walk)}")
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
    st.markdown(f"**Mixes out of — {_label(source['name'])}**")

    taken = {at_path[p] for p in graph.tracks if p in at_path}
    picks = suggestions(cost, source_idx, taken, k=FRONTIER_SIZE, pool=pool,
                        key_of=lambda i: normalized_name(
                            Path(frame.at[i, "path"])))
    if not picks:
        st.info("No candidate left that passes the filters.")
        return

    table = pd.DataFrame([
        {"Add": False, "cost": round(value, 3),
         **_spelled(frame.iloc[i], source),
         # Le copie dello stesso pezzo restano una voce sola. Il numero dice
         # quante ce ne sono: si aggiunge la più economica, e se ne serve
         # un'altra precisa c'è "Add a track by name" qui sotto.
         "copies": len(copies) if len(copies) > 1 else None,
         "_path": frame.at[i, "path"], "_row": i}
        for i, value, copies in picks])

    edited = play_table(
        "graph_roster", table,
        ["Add", "cost", "file", "BPM", "key", "groove",
         "Δbpm", "Δkey", "Δgroove", "copies", "genres"],
        {"Add": st.column_config.CheckboxColumn(
            "Add", help="Tick what you want next, then the button below."),
         **_read_only("cost", "file", "BPM", "key", "groove",
                      "Δbpm", "Δkey", "Δgroove", "copies", "genres")},
        # Come per il menu: cambiata la sorgente o cresciuta la catena, le
        # righe sotto sono altre e le spunte di prima indicherebbero brani
        # che nessuno ha scelto.
        editor_key=f"graph_roster_editor::{source_path}::{len(graph)}")

    wanted = [int(i) for i in edited.loc[edited["Add"], "_row"]]
    if st.button(f"➕ Add {len(wanted)} to the chain", type="primary",
                 width="stretch", disabled=not wanted):
        # In fila uno dietro l'altro: spuntarne tre vuol dire "poi questi
        # tre", e attaccarli tutti alla stessa sorgente farebbe tre rami
        # invece di un seguito.
        previous, added = source_path, []
        for i in wanted:
            graph.add(previous, frame.at[i, "path"])
            added.append(frame.at[i, "path"])
            previous = frame.at[i, "path"]
        # Le nuove nascono già al loro posto sulla regola in vigore. Le
        # vecchie no: chi le avesse spostate a mano non se le vedrebbe
        # rimettere in riga per aver scelto un brano.
        _place_on_axis(graph, frame, at_path, only=added)
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
