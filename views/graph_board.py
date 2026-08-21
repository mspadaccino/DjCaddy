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
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from analysis.graph_playlist import GraphPlaylist, suggestions
from analysis.mixing import TransitionCost
from views.components import NOW_PLAYING, render_player

_FRONTEND_DIR = Path(__file__).parent / "graph_board_frontend"
_graph_board = components.declare_component("graph_board", path=str(_FRONTEND_DIR))

# Tavolozza duplicata da `views.map_analysis` apposta: importarla da lì
# creerebbe un giro (quel modulo importa questa sezione), e sono dodici
# colori — il doppione costa meno del giro.
PALETTE = ["#e0503b", "#3d9be0", "#3fbf7f", "#f2a33c", "#a06fd6", "#e06fa8",
           "#4dd0c4", "#c9b037", "#6f8fd6", "#d66f6f", "#7fbf3f", "#bf7fd6"]
OTHER_COLOR = {"light": "#9aa4b0", "dark": "#6b7684"}

GRAPH_STATE = "map::graph"
GRAPH_SOURCE = "map::graph_source"
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

    if not len(graph):
        _render_start(frame, pool)
        return

    color_of = _color_map(frame)
    other = OTHER_COLOR["dark" if dark else "light"]
    selected = st.session_state.get(GRAPH_SOURCE)

    nodes = []
    for path in graph.tracks:
        idx = at_path.get(path)
        row = frame.iloc[idx] if idx is not None else None
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
            "genre": _label(genre) if genre else "",
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
        graph = GraphPlaylist().start(frame.at[first, "path"],
                                      frame.at[second, "path"])
        _save(graph)
        st.session_state[GRAPH_SOURCE] = frame.at[second, "path"]
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

    row = frame.iloc[source_idx]
    bpm = _some(row, "bpm")
    st.markdown(f"**Branching from — {row['name']}**  \n"
               f"{f'{bpm:.0f}' if bpm is not None else '?'} BPM · "
               f"{_some(row, 'camelot') or '?'} · {row['genres']}")

    taken = {at_path[p] for p in graph.tracks if p in at_path}
    picks = suggestions(cost, source_idx, taken, k=FRONTIER_SIZE, pool=pool)
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
        for col, (i, value) in zip(st.columns(FRONTIER_COLUMNS), row):
            with col.container(border=True):
                _render_candidate(frame, i, value, color_of, other,
                                  graph, source_path)

    render_player({frame.at[i, "path"] for i, _ in picks})


def _render_candidate(frame: pd.DataFrame, i: int, value: float,
                      color_of: dict[str, str], other: str,
                      graph: GraphPlaylist, source_path: str) -> None:
    """Una scheda della rosa, disegnata come i nodi sulla lavagna.

    Stessa faccia apposta: quello che si sceglie qui è quello che comparirà
    là, e vederlo cambiare aspetto nel passaggio costringerebbe a ritrovarlo.
    """
    row = frame.iloc[i]
    swatch = color_of.get(row["top_genre"], other)
    camelot = _some(row, "camelot")
    bpm = _some(row, "bpm")
    st.markdown(
        f"<div style='display:flex;gap:.5em;align-items:center'>"
        f"<span style='width:1.4em;height:1.4em;border-radius:.3em;"
        f"background:{swatch};flex:none'></span>"
        f"<b>{_label(row['name'])}</b></div>"
        f"<div style='margin:.4em 0 .2em;font-size:.8em'>"
        f"{f'{bpm:.0f} BPM · ' if bpm is not None else ''}"
        f"<span style='background:{_camelot_color(camelot)};color:#1b1f27;"
        f"padding:.1em .4em;border-radius:.3em'>{camelot or '?'}</span>"
        f" · cost {value:.3f}</div>",
        unsafe_allow_html=True)

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
