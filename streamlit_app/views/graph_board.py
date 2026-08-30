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

from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from core.analysis import mood_scale
from core.analysis.duplicates import folded, normalized_name, song_key
from core.analysis.graph_playlist import GraphPlaylist, suggestions
from core.analysis.mixing import TransitionCost
from core.viz import frontend_dir
# Gli helper con l'underscore si re-importano per i test, che li hanno
# sempre letti da questo modulo; la sostanza sta in `core.viz.board`.
from core.viz.board import (  # noqa: F401
    DEFAULT_HEIGHT, HEIGHT_FIELDS, HEIGHT_MEANING, _card_shifts, _gaps,
    _heights, _label, _measured, _some, _ticks, board_payload, chain_table,
    reordered, roster_table, wheel_payload)
from streamlit_app.views.components import NOW_PLAYING, fill_dock, play_table
from streamlit_app.views.track_columns import dark, read_only, reading_config

_graph_board = components.declare_component(
    "graph_board", path=str(frontend_dir("graph_board")))

_camelot_wheel = components.declare_component(
    "camelot_wheel", path=str(frontend_dir("camelot_wheel")))

# I brani spuntati in questo momento, in una tabella qualunque della pagina.
# La mappa li cerchia di giallo. Sta qui e non in `views.map_analysis` perché
# quel modulo importa questo: definirla di là chiuderebbe il giro.

GRAPH_STATE = "map::graph"
GRAPH_SOURCE = "map::graph_source"

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


def camelot_picker(selected: list[str], widget_key: str,
                   event_key: str) -> list[str] | None:
    """La ruota Camelot come filtro. Torna la scelta nuova, o None se nessuno
    l'ha toccata in questo giro.

    Sta in questo modulo e non fra i filtri che la usano perché qui c'è la
    cartella del suo frontend e qui c'è il colore delle tonalità: la stessa
    ragione per cui `start_board` sta qui e non nella mappa. Lo STATO invece
    è di chi chiama — la ruota non sa di che filtro fa parte, e chi la usa
    decide dove scriverne il risultato e con che portata ripartire.

    Il valore di un componente torna identico a ogni rerun, quindi il click
    si riconosce dal suo istante: senza, una tonalità scelta si aggiungerebbe
    e toglierebbe all'infinito.
    """
    event = _camelot_wheel(**wheel_payload(selected, dark()),
                           key=widget_key, default=None)
    if not event or event.get("at") == st.session_state.get(event_key):
        return None
    st.session_state[event_key] = event.get("at")
    code = event.get("code")
    return ([k for k in selected if k != code] if code in selected
            else selected + [code])


def _mood_stamp(frame: pd.DataFrame) -> tuple:
    """Un'impronta della libreria che costa niente da calcolare.

    La mappa si APPENDE — righe nuove in coda, mai in mezzo — quindi quante
    sono e qual è l'ultima bastano a dire che è la stessa libreria di prima.
    """
    return (len(frame), frame.at[len(frame) - 1, "path"]) if len(frame) else (0, "")


@st.cache_data(show_spinner=False)
def _mood_popularity(_frame: pd.DataFrame, stamp: tuple) -> dict:
    """Quanto è comune ogni mood. Tenuto da parte perché costa una scorsa —
    quarantatré millisecondi su ottantasettemila righe — e perché scade
    quando la mappa cresce, non quando si clicca una scheda: senza, si
    rifarebbe cinque volte per giro di pagina, una per ogni tabella che
    porta la colonna del mood.

    Qui c'erano anche i due estremi fra cui tendere l'altezza del mood sulla
    lavagna, ed erano i decili della valence letta DALLE PAROLE. Non ci sono
    più: l'altezza ora legge `valence_rank`, che è già un rango sulla
    libreria e quindi porta la sua scala con sé — e soprattutto è lo stesso
    numero che leggono la freccia in tabella e l'asse dei quadranti, invece
    di essere lo stesso conto rifatto in un altro posto su un altro numero.
    """
    moods = list(_frame["moods"]) if "moods" in _frame else []
    return mood_scale.popularity(moods)


def mood_popularity(frame: pd.DataFrame) -> dict[str, int]:
    """Quante volte ogni mood compare nella libreria. Pubblica perché la
    stessa colonna la scrivono anche le tabelle di `views.map_analysis`."""
    return _mood_popularity(frame, _mood_stamp(frame))


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

    E nemmeno i filtri stanno più qui: `pool` arriva già ristretto da quelli
    della mappa. Ne aveva di suoi, con la stessa ruota e le stesse manopole,
    e voleva dire due pannelli da tenere d'accordo per una domanda sola —
    quali brani sto guardando. Restringono la rosa e i brani di partenza, non
    la catena già costruita: togliere un nodo posato spezzerebbe una scaletta
    che qualcuno ha messo insieme.
    """
    st.caption(
        "Grow a set one track at a time: on the left the chain as it stands, "
        "on the right what mixes out of the track you are standing on. Tick, "
        "add, repeat — and send it to the playlist, where the board shows the "
        "shape the whole set is taking. The roster comes from whatever passes "
        "the filters at the top of the page.")

    graph = _graph()
    if not len(graph):
        _render_start(frame, pool, chosen)
        return

    walk = graph.walk()
    _render_tables(frame, cost, pool, at_path, graph, walk)

    c1, c2, c3 = st.columns(3)
    # Chi TOCCA la catena riparte per intero, non solo da questo frammento.
    # Il frammento esiste per non ridisegnare ottantamila punti a ogni casella
    # spuntata, ed è un risparmio che qui non si può prendere: da quando la
    # mappa cerchia di giallo i brani in catena, un rerun del solo frammento
    # lascerebbe quegli anelli addosso a una catena che non c'è più. Un
    # cerchio attorno al nulla non è un dettaglio estetico, è la mappa che
    # dice il falso.
    #
    # Il riordino delle righe è l'unico che se lo tiene: cambia l'ORDINE dei
    # brani in catena, non quali sono, e gli anelli sono gli stessi.
    if c1.button("↺ Start over", width="stretch",
                 help="Empties the chain. The playlist is not touched."):
        st.session_state[GRAPH_STATE] = None
        st.session_state[GRAPH_SOURCE] = None
        st.rerun()
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


@st.fragment
def render_board(frame: pd.DataFrame, at_path: dict[str, int],
                 playlist: list[int], drop, move,
                 chapters: list[dict] | None = None,
                 chapter_move=None) -> None:
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
        elif kind == "chapter_move" and who in at_path and chapter_move:
            chapter_move(at_path[who],
                         event.get("from_chapter"), event.get("to_chapter"))
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

    st.caption(HEIGHT_MEANING[axis])

    payload = board_payload(frame, at_path, paths, axis,
                            common=mood_popularity(frame), dark=dark())
    _graph_board(**payload,
                 selected=st.session_state.get(BOARD_PICKED),
                 chapters=chapters or [],
                 dark=dark(), key="playlist_board_widget", default=None)

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
                   walk: list[str]) -> None:
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
        table = chain_table(frame, at_path, walk, common)
        # La firma della catena: chi porta questa nella propria chiave
        # rinasce appena l'ordine cambia. Per la tabella è l'unico modo di
        # dimenticare la riga appena spostata, che altrimenti resterebbe
        # scritta nello stato del widget e si riapplicherebbe a ogni giro.
        signature = "|".join(walk)
        chain_key = f"graph_chain_editor::{signature}"
        play_table("graph_chain", table,
                   # Il nome subito dopo il numero d'ordine: e' quello che
                   # si cerca per primo, e in fondo alla fila di misure
                   # costringeva a scorrere per sapere di che brano si sta
                   # leggendo il BPM.
                   ["#", "file", "BPM", "key", "energy", "groove", "emotion",
                    "Δbpm", "Δkey", "Δenergy", "Δgroove", "mood",
                    "genres", "folder"],
                   {"#": st.column_config.NumberColumn(
                       "#", min_value=1, max_value=max(len(walk), 1), step=1,
                       help="Write the position you want this track in: the "
                            "row moves there and the others slide."),
                    **reading_config(frame, table),
                    **read_only("Δbpm", "Δkey", "Δenergy", "Δgroove")},
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
            st.rerun()

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

    table = roster_table(frame, picks, source, common)

    edited = play_table(
        "graph_roster", table,
        ["Add", "file", "cost", "BPM", "key", "energy", "groove", "emotion",
         "Δbpm", "Δkey", "Δenergy", "Δgroove", "copies", "mood",
         "genres", "folder"],
        {"Add": st.column_config.CheckboxColumn(
            "Add", help="Tick what you want next, then the button below."),
         **reading_config(frame, table),
         **read_only("cost", "Δbpm", "Δkey", "Δenergy", "Δgroove", "copies")},
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
        previous = source_path
        for i in wanted:
            graph.add(previous, frame.at[i, "path"])
            previous = frame.at[i, "path"]
        _save(graph)
        st.session_state[GRAPH_SOURCE] = previous
        st.rerun()


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
            st.rerun()


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
    # Senza accenti da entrambe le parti: un nome che arriva dal disco di un
    # Mac porta la tilde staccata dalla lettera e non combacia con quella che
    # si digita, e comunque nessuno vuole cercare l'accento sulla tastiera.
    wanted = folded(search.strip())
    found = [i for i in options if wanted in folded(frame.at[i, "name"])]
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
            st.rerun()
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
        st.rerun()
