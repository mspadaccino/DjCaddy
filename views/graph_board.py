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

from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from analysis.graph_playlist import GraphPlaylist, suggestions
from analysis.mixing import TransitionCost

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

# Sopra questa quantità di brani il menu per nome non si apre più in fretta:
# si cerca prima, si sceglie dopo.
START_PICKER_MAX = 2000


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
        name = frame.at[idx, "name"] if idx is not None else Path(path).stem
        genre = frame.at[idx, "top_genre"] if idx is not None else None
        x, y = graph.places[path]
        nodes.append({
            "id": path, "x": x, "y": y, "label": _label(name),
            "color": color_of.get(genre, other),
        })
    links = [{"a": a, "b": b} for a, b in graph.links]

    event = _graph_board(nodes=nodes, links=links, selected=selected,
                         dark=dark, key="graph_board_widget", default=None)

    if event:
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

    st.caption("**Drag** a node to move it · **click** a node to branch new "
               "suggestions from it · the **×** on the selected node removes "
               "it and reconnects its neighbours.")

    c1, c2, c3 = st.columns([2, 2, 3])
    if c1.button("↺ Restart the board", width="stretch"):
        st.session_state[GRAPH_STATE] = None
        st.session_state[GRAPH_SOURCE] = None
        st.rerun()
    if c2.button("➡️ Send to playlist", type="primary", width="stretch"):
        order = [at_path[p] for p in graph.walk() if p in at_path]
        set_playlist(order)
        st.rerun()
    c3.caption(f"{len(graph)} track(s) on the board.")

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
    st.markdown(f"**Branching from — {row['name']}**  \n"
               f"{row['bpm'] or '?'} BPM · {row['camelot'] or '?'} · "
               f"{row['genres']}")

    taken = {at_path[p] for p in graph.tracks if p in at_path}
    picks = suggestions(cost, source_idx, taken, k=6, pool=pool)
    if not picks:
        st.info("No candidate left that passes the filters.")
        return

    cols = st.columns(len(picks))
    for col, (i, value) in zip(cols, picks):
        with col:
            st.markdown(f"**{frame.at[i, 'name'][:26]}**")
            st.caption(f"{frame.at[i, 'bpm'] or '?'} BPM · "
                      f"{frame.at[i, 'camelot'] or '?'} · cost {value:.3f}")
            if st.button("➕ Add", key=f"graph_pick_{i}", width="stretch"):
                new_path = frame.at[i, "path"]
                graph.add(source_path, new_path)
                _save(graph)
                st.session_state[GRAPH_SOURCE] = new_path
                st.rerun()
