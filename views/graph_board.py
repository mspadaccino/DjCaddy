"""La lavagna: un brano alla volta, come fa djoid nel suo "graph mode".

`analysis.graph_playlist.GraphPlaylist` tiene la logica — brani, posizioni,
collegamenti, sublist di adiacenti. Questo modulo è lo strato sopra che sa
di pixel: un componente Streamlit fatto a mano (SVG puro, niente build npm)
per il trascinare i nodi, e i controlli attorno per scegliere le due tracce
di partenza e il prossimo passo dalla rosa di suggeriti.

Il componente non porta la sua logica: manda solo "questo nodo si è mosso
qui", "questo nodo è stato cliccato", "questo nodo va tolto". La scelta di
COSA succede dopo — aggiungere al grafo, cambiare la sorgente dei
suggerimenti — resta in Python, dove sta anche il resto della pagina Map.
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
from views.components import NOW_PLAYING, render_player

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

# La rosa sta su tre colonne, come in djoid: nove candidati sono abbastanza
# da avere una scelta vera e pochi abbastanza da guardarli tutti in una volta
# senza scorrere.
FRONTIER_COLUMNS = 3
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


def start_board(first: str, second: str) -> None:
    """Comincia la lavagna da due brani scelti altrove — la mappa qui sopra.

    Sta qui e non nella mappa perché le chiavi di sessione della lavagna
    sono di questo modulo: chi la avvia deve poter dire quali due brani e
    basta, senza sapere dove finiscono scritti.
    """
    _save(GraphPlaylist().start(first, second))
    # La sorgente è il secondo: è quello appena messo, ed è da lì che si
    # continua — come dopo ogni altra aggiunta.
    st.session_state[GRAPH_SOURCE] = second


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


def _shifts(source, row) -> list[str]:
    """Come cambia il brano rispetto a quello da cui lo si sta scegliendo.

    Il costo dice quanto due brani sono lontani; questi dicono da che parte,
    e sono la cosa che il costo non può dire perché non ha segno. Restano
    fuori dall'ordinamento apposta: un set sale, tiene e lascia cadere, e
    ordinare per direzione vorrebbe dire scegliere quale al posto del DJ.
    """
    out = []
    tempo = bpm_shift(_some(source, "bpm"), _some(row, "bpm"))
    if tempo is not None:
        out.append(f"{round(tempo):+d} BPM")
    wheel = camelot_shift(_some(source, "camelot"), _some(row, "camelot"))
    if wheel is not None:
        steps, mode = wheel
        # Zero passi con la lettera cambiata non è "niente": è il relativo
        # maggiore o minore, che è una mossa e va detta.
        out.append(f"{steps:+d} wheel" if steps
                   else ("relative" if mode else "same key"))
    here, there = _some(source, "danceability"), _some(row, "danceability")
    if here is not None and there is not None:
        out.append(f"{there - here:+.2f} dance")
    return out


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
                st.rerun()

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
                st.rerun()

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


def _card_shifts(source, row) -> dict[str, tuple[str, int]]:
    """Gli stessi scarti di `_shifts`, una cella per colonna della scheda.

    Scritti di seguito non ci starebbero, e abbreviarli in "+0 · -1 · +.05"
    su una riga a sé sarebbe un rebus. Incolonnati sotto ai valori che
    commentano diventano invece la seconda riga della stessa tabella, e le
    unità di misura le presta la riga sopra. Se un valore manca, manca la
    colonna: le due righe restano allineate perché le costruisce lo stesso
    giro.
    """
    if source is None:
        return {}
    out = {}
    tempo = bpm_shift(_some(source, "bpm"), _some(row, "bpm"))
    if tempo is not None:
        out["bpm"] = (f"{round(tempo):+d}", _way(round(tempo)))
    wheel = camelot_shift(_some(source, "camelot"), _some(row, "camelot"))
    if wheel is not None:
        steps, mode = wheel
        # Il relativo maggiore o minore non sale né scende: cambia colore al
        # brano restando dov'è, e tingerlo di verso direbbe una cosa falsa.
        out["key"] = ((f"{steps:+d}", _way(steps)) if steps
                      else (("rel", 0) if mode else ("=", 0)))
    here, there = _some(source, "danceability"), _some(row, "danceability")
    if here is not None and there is not None:
        # Senza lo zero davanti: sotto una colonna di trentotto pixel "+0.05"
        # e "+.05" dicono la stessa cosa e solo uno dei due ci sta.
        gap = round(there - here, 2)
        out["dance"] = (f"{gap:+.2f}".replace("0.", "."), _way(gap))
    return out


def _label(name: str) -> str:
    name = Path(name).stem if "/" in name or "\\" in name else name
    return name if len(name) <= 22 else name[:21] + "…"


def render_graph_builder(frame: pd.DataFrame, cost: TransitionCost, pool,
                         at_path: dict[str, int], set_playlist) -> None:
    """La sezione lavagna: parte da due brani, poi cresce un passo alla volta.

    `set_playlist` prende una lista di indici (nello stesso `frame`) e la
    rende la playlist della pagina — lo stesso canale che usa il resto della
    mappa, così "manda alla playlist" qui sotto finisce nello stesso posto
    del disegno a lazo qui sopra.
    """
    st.caption(
        "In stile djoid: si parte da due brani sulla lavagna, collegati da "
        "una linea. Da lì si cresce un passo alla volta — si sceglie una "
        "sorgente, si vede la rosa di ciò che ci mixa dietro, se ne prende "
        "uno. Ogni nodo si trascina col mouse; il collegamento dice da dove "
        "è arrivato il suggerimento, non è decorazione.")

    graph = _graph()
    dark = _dark()
    pool = _render_filters(frame, pool)

    if not len(graph):
        _render_start(frame, pool)
        return

    color_of = _color_map(frame)
    other = OTHER_COLOR["dark" if dark else "light"]
    selected = st.session_state.get(GRAPH_SOURCE)

    # Ogni scheda si confronta con quella che la precede NELLA SCALETTA, non
    # con quella da cui è stata scelta: è l'ordine in cui il set uscirà, e
    # quindi l'unico rispetto a cui "sale" o "scende" vuol dire qualcosa.
    walk = graph.walk()
    before = {track: walk[n - 1] for n, track in enumerate(walk) if n}
    span = _drive_span(frame)

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

    event = _graph_board(nodes=nodes, links=links, selected=selected,
                         dark=dark, key="graph_board_widget", default=None)

    if event and event.get("at") != st.session_state.get(GRAPH_EVENT):
        st.session_state[GRAPH_EVENT] = event.get("at")
        kind = event.get("type")
        node_id = event.get("id")
        if kind == "move" and node_id in graph:
            graph.move(node_id, event["x"], event["y"])
            _save(graph)
        elif kind == "click" and node_id in graph:
            st.session_state[GRAPH_SOURCE] = node_id
            st.rerun()
        elif kind == "remove" and node_id in graph:
            graph.remove(node_id)
            _save(graph)
            if st.session_state.get(GRAPH_SOURCE) == node_id:
                st.session_state[GRAPH_SOURCE] = graph.tracks[-1] if graph.tracks else None
            st.rerun()

    st.caption("**Drag** a card to move it · **click** a card to branch new "
               "suggestions from it · the **bin** under the selected card "
               "removes it and reconnects its neighbours · **scroll** to "
               "zoom, drag the background to pan.")

    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    if c1.button("↺ Restart the board", width="stretch"):
        st.session_state[GRAPH_STATE] = None
        st.session_state[GRAPH_SOURCE] = None
        st.rerun()
    if c2.button("⇥ Straighten", width="stretch",
                 help="Line the cards up in the order the playlist will read."):
        graph.straighten()
        _save(graph)
        st.rerun()
    if c3.button("➡️ Send to playlist", type="primary", width="stretch"):
        order = [at_path[p] for p in graph.walk() if p in at_path]
        set_playlist(order)
        st.rerun()
    c4.caption(f"{len(graph)} track(s) on the board.")

    _render_frontier(frame, cost, pool, at_path, graph)


def _render_start(frame: pd.DataFrame, pool) -> None:
    st.markdown("**Start the board with two tracks.** A single track says "
               "nothing about direction — a pair does.")
    options = pool.tolist()
    if len(options) > START_PICKER_MAX:
        search = st.text_input("Name contains", key="graph_start_search",
                               placeholder="too many tracks — search by name first")
        options = [i for i in options
                  if search.strip().lower() in frame.at[i, "name"].lower()] \
            if search.strip() else []
        if len(options) > START_PICKER_MAX:
            st.caption(f"{len(options):,} match — narrow the search further.")
            return
        if not options:
            st.caption("Type part of a name to search the library.")
            return

    c1, c2, c3 = st.columns([3, 3, 2])
    first = c1.selectbox("First track", options, index=None,
                         format_func=lambda i: frame.at[i, "name"],
                         key="graph_start_first")
    second = c2.selectbox("Second track", options, index=None,
                          format_func=lambda i: frame.at[i, "name"],
                          key="graph_start_second")
    c3.markdown("<div style='height:1.8em'></div>", unsafe_allow_html=True)
    if c3.button("▶ Start the board", type="primary", width="stretch",
                disabled=first is None or second is None or first == second):
        start_board(frame.at[first, "path"], frame.at[second, "path"])
        st.rerun()


def _render_frontier(frame: pd.DataFrame, cost: TransitionCost, pool,
                     at_path: dict[str, int], graph: GraphPlaylist) -> None:
    source_path = st.session_state.get(GRAPH_SOURCE)
    if source_path is None or source_path not in graph:
        st.info("Click a track on the board to see what mixes out of it.")
        return
    source_idx = at_path.get(source_path)
    if source_idx is None:
        return

    source = frame.iloc[source_idx]
    bpm = _some(source, "bpm")
    st.markdown(f"**Branching from — {source['name']}**  \n"
               f"{f'{bpm:.0f}' if bpm is not None else '?'} BPM · "
               f"{_some(source, 'camelot') or '?'} · {source['genres']}")

    taken = {at_path[p] for p in graph.tracks if p in at_path}
    picks = suggestions(cost, source_idx, taken, k=FRONTIER_SIZE, pool=pool,
                        key_of=lambda i: normalized_name(
                            Path(frame.at[i, "path"])))
    if not picks:
        st.info("No candidate left that passes the filters.")
        return

    color_of = _color_map(frame)
    other = OTHER_COLOR["dark" if _dark() else "light"]
    for start in range(0, len(picks), FRONTIER_COLUMNS):
        row = picks[start:start + FRONTIER_COLUMNS]
        # Le colonne si chiedono sempre tutte e tre, anche per una riga
        # spaiata: altrimenti l'ultima scheda si allargherebbe a tutta la
        # pagina e sembrerebbe importante più delle altre.
        for col, voice in zip(st.columns(FRONTIER_COLUMNS), row):
            with col.container(border=True):
                _render_candidate(frame, voice, color_of, other,
                                  graph, source_path, source)

    render_player({frame.at[c, "path"]
                   for _, _, copies in picks for c in copies})


def _render_candidate(frame: pd.DataFrame, voice: tuple, color_of: dict[str, str],
                      other: str, graph: GraphPlaylist, source_path: str,
                      source) -> None:
    """Una scheda della rosa, disegnata come i nodi sulla lavagna.

    Stessa faccia apposta: quello che si sceglie qui è quello che comparirà
    là, e vederlo cambiare aspetto nel passaggio costringerebbe a ritrovarlo.

    Una voce può essere più copie dello stesso pezzo. In quel caso la scheda
    resta una — occuparne tre con lo stesso brano è il motivo per cui si
    raggruppa — e porta sotto un menu per dire QUALE copia. La scelta si
    legge prima di disegnare, perché il menu sta in fondo alla scheda ma
    decide i numeri che stanno in cima.
    """
    i, value, copies = voice
    chosen = st.session_state.get(f"graph_copy_{i}", i)
    if chosen not in copies:
        chosen = i
    row = frame.iloc[chosen]
    swatch = color_of.get(row["top_genre"], other)
    camelot = _some(row, "camelot")
    bpm, dance = _some(row, "bpm"), _some(row, "danceability")
    # Due righe e non una: sopra che brano è, sotto come si muove rispetto a
    # dove sei. Mescolate fra parentesi si leggono male entrambe.
    st.markdown(
        f"<div style='display:flex;gap:.5em;align-items:center'>"
        f"<span style='width:1.4em;height:1.4em;border-radius:.3em;"
        f"background:{swatch};flex:none'></span>"
        f"<b>{_label(row['name'])}</b>"
        f"{f'<span style=\"opacity:.55\">×{len(copies)}</span>' if len(copies) > 1 else ''}"
        f"</div>"
        f"<div style='margin:.4em 0 .1em;font-size:.8em'>"
        f"{f'{bpm:.0f} BPM · ' if bpm is not None else ''}"
        f"<span style='background:{_camelot_color(camelot)};color:#1b1f27;"
        f"padding:.1em .4em;border-radius:.3em'>{camelot or '?'}</span>"
        f"{f' · {dance:.2f} dance' if dance is not None else ''}</div>"
        f"<div style='margin:0 0 .2em;font-size:.78em;opacity:.65'>"
        f"{' · '.join(_shifts(source, row))} · cost {value:.3f}</div>",
        unsafe_allow_html=True)

    if len(copies) > 1:
        # La cartella è ciò che distingue una copia dall'altra: il nome, per
        # definizione del raggruppamento, è lo stesso.
        st.selectbox(f"{len(copies)} copies of this — which one",
                     copies, key=f"graph_copy_{i}",
                     format_func=lambda c: Path(frame.at[c, "path"]).parent.name)

    hear, take = st.columns([1, 2])
    if hear.button("▶", key=f"graph_hear_{i}", width="stretch",
                   help="Hear it before you commit to it."):
        st.session_state[NOW_PLAYING] = row["path"]
        st.rerun()
    if take.button("➕ Add", key=f"graph_pick_{i}", width="stretch"):
        graph.add(source_path, row["path"])
        _save(graph)
        st.session_state[GRAPH_SOURCE] = row["path"]
        st.rerun()
