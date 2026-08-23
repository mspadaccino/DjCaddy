"""Sezione "Map": la libreria come mappa, e le playlist come percorsi.

Il problema che risolve non è trovare un brano — per quello basta il nome —
ma trovare il PROSSIMO brano fra novantamila, che è la domanda a cui una
cartella non sa rispondere. Ogni brano diventa un punto: vicini quelli che
suonano vicini, secondo l'embedding della rete e non secondo l'etichetta
che qualcuno gli ha scritto sopra.

Da lì in poi si lavora sulla mappa:

- si clicca un brano e si chiede cosa ci va dietro (costo di transizione:
  distanza sulla mappa + scarto di BPM + distanza sulla ruota Camelot);
- si DISEGNA una linea attraverso i gruppi e la si trasforma in playlist,
  che è il modo di pianificare un arco narrativo invece di una scaletta;
- si prende un mucchio di brani disordinati e li si fa ordinare in modo
  che ognuno si fonda col successivo.

**L'ordine della pagina** segue quanto spesso si usa una cosa, non l'ordine
in cui le cose accadono la prima volta: la mappa e le playlist stanno in
cima perché è lì che si passa il tempo; costruire la mappa si fa una volta
e poi la si lascia lavorare, quindi sta in fondo, chiusa.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis.dj_export import build_m3u8, build_rekordbox_xml, read_title_artist
from analysis.essentia_tags import MODEL_DIR, available, find_taggable, missing_models
from analysis.map_job import (DEFAULT_MAP_LOG, load_map_state, open_monitor,
                              pause_job, process_state, resume_job, stop_job)
from analysis.map_profile import ProfileSettings, default_workers, profile_many
from analysis.map_projection import ProjectionSettings
from analysis.map_projection import available as umap_available
from analysis.map_projection import project
from analysis.map_store import MapStore, default_store_dir
from analysis.mixing import (TransitionCost, along_path, closed_shape,
                             magic_sort, nearest)
from views.components import pick_folder, play_table, tick_all
from views.graph_board import render_graph_builder

# Oltre questo numero di punti si disegna un campione. Non è la RAM a cedere
# ma il browser: WebGL regge il milione di punti in teoria, e nella pratica
# una mappa da centomila punti si trascina a ogni zoom. Il campione è casuale
# ma stabile (seme fisso), così la mappa non si rimescola a ogni rerun.
MAX_POINTS = 20000

# Quanti gruppi ricevono un colore proprio nella legenda. Il modello conosce
# 400 etichette: colorarle tutte darebbe una legenda illeggibile e una
# tavolozza in cui due tinte vicine non vogliono dire niente.
COLORED_GENRES = 18

# Le etichette Discogs sono già gerarchiche — "Electronic - House",
# "Funk / Soul - Disco" — quindi il macro genere non va inventato: sta nella
# stringa, prima del trattino. La differenza sulla libreria è netta: 258
# etichette foglia, di cui le prime dodici coprono il 64% e il resto finisce
# in un grigio indistinto; contro 15 padri, di cui i primi dodici coprono il
# 99,98%. Restano offerti tutti e due perché rispondono a domande diverse:
# il padre dice di che musica è fatta la serata, la foglia dice quale house.
GENRE_LEVELS = {"macro genre": "parent", "genre": "leaf"}


def genre_level(genre: str, level: str) -> str:
    """L'etichetta al livello scelto. Senza trattino i due coincidono."""
    text = str(genre or "")
    return text.split(" - ")[0] if level == "parent" else text

# Oltre questi brani in playlist i numeri d'ordine sulla mappa diventano una
# macchia: la linea basta a raccontare il percorso.
NUMBERED_UP_TO = 40

# Cosa può dire la DIMENSIONE del punto. La posizione la decide l'embedding e
# non vuol dire niente di preciso — è affinità, non una grandezza. Il diametro
# invece può portare un numero che si legge: quanto va veloce, quanto è dritto,
# quanto spinge. Si legge senza ruotare niente, che è il motivo per cui questa
# è la terza dimensione e non un terzo asse.
SIZE_FIELDS = {
    "same size": None,
    "BPM": "bpm",
    "groove": "danceability",
    "energy": "lufs",
}
FLAT_SIZE = 7.0
MIN_SIZE, MAX_SIZE = 4.0, 15.0

PALETTE = ["#e0503b", "#3d9be0", "#3fbf7f", "#f2a33c", "#a06fd6", "#e06fa8",
           "#4dd0c4", "#c9b037", "#6f8fd6", "#d66f6f", "#7fbf3f", "#bf7fd6",
           # Sei in più: con le etichette foglia dodici colori lasciavano un
           # terzo della libreria in grigio. Le tinte continuano a girare
           # attorno alla ruota invece di scurirsi, o due gruppi vicini
           # diventerebbero indistinguibili su punti da sette pixel.
           "#3fb0bf", "#d68f3f", "#8fd63f", "#d63f8f", "#5f6fd6", "#bf5f3f"]

# Due fondi e due inchiostri, uno per tema. Il fondo della mappa è staccato
# di poco da quello della pagina: quel poco basta a dire dove finisce il
# testo e comincia il territorio, senza farne un riquadro.
SKIN = {
    "light": {"paper": "#ffffff", "plot": "#f4f6f9", "ink": "#1b1f27",
              "other": "#9aa4b0", "label": "rgba(27,31,39,0.82)",
              "halo": "rgba(255,255,255,0.75)"},
    "dark": {"paper": "#0e1117", "plot": "#161a22", "ink": "#eef1f6",
             "other": "#6b7684", "label": "rgba(238,241,246,0.88)",
             "halo": "rgba(14,17,23,0.75)"},
}

# In sessione si tengono i PERCORSI, non le posizioni nella libreria. Una
# posizione vale finché la mappa non cambia: basta togliere un brano e la 200
# è un altro brano: la playlist resterebbe in piedi indicando le tracce
# sbagliate, che è peggio di un errore. Il percorso invece è il brano.
SEED = "map::seed"
PICKED = "map::seedpick_applied"
LASSO_MODE = "map::lassomode"
LASSO_KIND = "map::lassokind"
PLAYLIST = "map::playlist"

# Oltre queste voci il menu dei nomi smette di essere comodo (e di aprirsi
# in fretta): sopra, si restringe coi filtri.
SEED_PICKER_MAX = 2000

# Quanti risultati di ricerca elencare. Oltre, non si sta più scegliendo fra
# candidati: si sta scorrendo la libreria con parole troppo generiche, e la
# risposta giusta è una parola in più, non una lista più lunga.
SEED_MATCHES_MAX = 50

# Quanti candidati proporre attorno a un brano. Venti è la partenza: bastano a
# scegliere senza dover scorrere. Il tetto non è una soglia tecnica — cento
# righe si disegnano in un lampo — ma il punto oltre il quale una lista smette
# di essere una rosa e torna a essere la libreria, che è ciò da cui si stava
# scappando.
SUGGESTION_DEFAULT = 20
SUGGESTION_MAX = 100
SUGGESTION_STEP = 5


def _spelled(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} hours"


def _skin() -> dict:
    """I colori del tema in uso. Il tema può non essere ancora arrivato dal
    browser al primo giro: in quel caso si disegna chiaro."""
    theme = getattr(getattr(st, "context", None), "theme", None)
    return SKIN["dark" if getattr(theme, "type", None) == "dark" else "light"]


@st.cache_resource(show_spinner="Opening the map…")
def _open_store(directory: str, stamp: tuple) -> MapStore:
    """La mappa in memoria. `stamp` è lo stato dei file su disco: cambia
    quando il job aggiunge brani, e solo allora si rilegge."""
    return MapStore.load(Path(directory))


def _stamp(directory: Path) -> tuple:
    out = []
    for name in ("tracks.jsonl", "coords.npy"):
        try:
            stat = (directory / name).stat()
            out.append((stat.st_mtime, stat.st_size))
        except OSError:
            out.append(None)
    return tuple(out)


def _read_only(*columns: str) -> dict:
    """Colonne che si guardano e basta. In una tabella con una casella da
    spuntare tutto il resto va bloccato a mano, o si finisce a correggere i
    BPM di un brano credendo di sceglierlo."""
    return {name: st.column_config.Column(disabled=True) for name in columns}


def remember_playlist(frame: pd.DataFrame, indices) -> None:
    st.session_state[PLAYLIST] = [frame.at[i, "path"] for i in indices]


def remember_seed(frame: pd.DataFrame, index: int) -> None:
    st.session_state[SEED] = frame.at[index, "path"]


def read_selection() -> tuple[list[int], list]:
    """Cosa è selezionato sul grafico: gli indici dei brani e il tratto.

    Lo stato del grafico sta in sessione sotto la sua chiave. I punti dei
    tracciati di servizio (il percorso della playlist, il cerchio del seme)
    non portano `customdata` e vengono scartati: sono disegno, non brani.
    """
    state = st.session_state.get("map::chart")
    selection = state.get("selection") if state else None
    if not selection:
        return [], []
    picked = [int(p["customdata"][0]) for p in selection.get("points", [])
              if p.get("customdata")]
    return picked, list(selection.get("lasso") or [])


def matching_tracks(frame: pd.DataFrame, pool, words: list[str]) -> list[int]:
    """Le posizioni che contengono TUTTE le parole, nel nome o nella cartella.

    A parole sparse e non a sottostringa: "madonna lucky" deve trovare
    "Madonna - Lucky Star (Extended Dance Remix)", che una ricerca contigua
    non trova. L'ordine non conta — chi cerca ricorda i pezzi di un titolo,
    non come sono disposti.

    Si guarda nel nome del file e nella cartella perché è lì che stanno
    artista e titolo: la mappa non conserva i tag, e in una libreria da DJ il
    nome del file li porta quasi sempre entrambi.
    """
    inside = frame.loc[list(pool)]
    hay = (inside["name"].fillna("") + " "
           + inside["folder"].fillna("")).str.casefold()
    keep = pd.Series(True, index=hay.index)
    for word in words:
        # Anche le parole cercate, non solo il testo in cui si cerca: farlo
        # fare a chi chiama vuol dire che la funzione è giusta solo finché
        # tutti si ricordano di farlo.
        keep &= hay.str.contains(word.casefold(), regex=False)
    return keep[keep].index.tolist()


def _path_points(shape: dict) -> list[tuple[float, float]]:
    xs, ys = shape.get("x") or [], shape.get("y") or []
    return list(zip(xs, ys))


# --------------------------------------------------------------------------
# Il disegno
# --------------------------------------------------------------------------

def marker_sizes(frame: pd.DataFrame, column: str | None):
    """Il diametro dei punti a partire da una colonna. Un numero se sono
    tutti uguali, una serie allineata a `frame` altrimenti.

    Si scala sui percentili 5–95 e non su minimo e massimo: un brano a 200
    BPM in mezzo a una libreria che sta a 120 schiaccerebbe tutti gli altri
    sullo stesso diametro. Chi quel numero non ce l'ha resta al minimo —
    meglio un punto piccolo che un punto finto medio.
    """
    if column is None or column not in frame:
        return FLAT_SIZE
    values = pd.to_numeric(frame[column], errors="coerce")
    known = values.dropna()
    if len(known) < 2:
        return FLAT_SIZE
    low, high = np.percentile(known, [5, 95])
    if high <= low:
        return FLAT_SIZE
    share = ((values - low) / (high - low)).clip(0, 1).fillna(0.0)
    return MIN_SIZE + share * (MAX_SIZE - MIN_SIZE)


def build_figure(drawn: pd.DataFrame, top_genres: list[str], coords,
                 playlist: list[int], seed: int | None) -> go.Figure:
    """La mappa: un tracciato per genere, più il percorso e il seme sopra.

    Un tracciato per genere e non uno solo con i colori dentro, perché così
    la legenda esiste e ci si può cliccare per spegnere un genere. L'indice
    del brano nella libreria viaggia in `customdata`: è come si risale dal
    punto cliccato alla riga, senza dipendere dall'ordine dei tracciati.
    """
    skin = _skin()
    color_of = dict(zip(top_genres, PALETTE))
    figure = go.Figure()

    for genre in top_genres + ["other"]:
        part = (drawn[~drawn["genre_key"].isin(top_genres)] if genre == "other"
                else drawn[drawn["genre_key"] == genre])
        if not len(part):
            continue
        figure.add_trace(go.Scattergl(
            x=part["x"], y=part["y"], mode="markers", name=genre[:28],
            customdata=part[["index", "name", "bpm", "camelot", "genres"]].to_numpy(),
            marker={
                "size": part["_size"], "opacity": 0.85,
                "color": color_of.get(genre, skin["other"]),
                # Un filo di bordo del colore del fondo: dove i punti si
                # accavallano si continua a contarli invece di vedere una
                # macchia unica.
                "line": {"width": 0.5, "color": skin["plot"]},
            },
            hovertemplate="<b>%{customdata[1]}</b><br>%{customdata[2]} BPM · "
                          "%{customdata[3]}<br>%{customdata[4]}<extra></extra>",
        ))

    # Il nome del genere scritto in mezzo al suo gruppo: la legenda dice quale
    # colore è cosa, questo dice dove andare a cercarlo. Mediana e non media,
    # perché un brano isolato dall'altra parte della mappa non deve spostare
    # l'etichetta in mezzo al nulla.
    for genre in top_genres:
        part = drawn[drawn["genre_key"] == genre]
        if len(part) < 3:
            continue
        # Piu' grande, piu' opaca e su un fondo suo: prima era al 45% di
        # opacita' e spariva dentro il colore dei punti proprio dove i punti
        # sono piu' fitti, cioe' dove l'etichetta serve.
        figure.add_annotation(
            x=float(part["x"].median()), y=float(part["y"].median()),
            text=f"<b>{genre.split(' - ')[-1][:22]}</b>", showarrow=False,
            font={"size": 14, "color": skin["label"]},
            bgcolor=skin["halo"], borderpad=3)

    if playlist:
        line = coords[playlist]
        numbered = len(playlist) <= NUMBERED_UP_TO
        figure.add_trace(go.Scattergl(
            x=line[:, 0], y=line[:, 1], name="playlist",
            mode="lines+markers+text" if numbered else "lines+markers",
            text=[str(i) for i in range(1, len(playlist) + 1)] if numbered else None,
            textposition="top center",
            textfont={"size": 9, "color": skin["ink"]},
            line={"color": skin["ink"], "width": 1.5},
            marker={"size": 9, "color": skin["paper"],
                    "line": {"width": 1.5, "color": skin["ink"]}},
            hoverinfo="skip"))

    if seed is not None and seed < len(coords):
        figure.add_trace(go.Scattergl(
            x=[coords[seed][0]], y=[coords[seed][1]], mode="markers",
            name="seed", showlegend=False, hoverinfo="skip",
            marker={"size": 20, "color": "rgba(0,0,0,0)",
                    "line": {"width": 2, "color": skin["ink"]}}))

    figure.update_layout(
        height=640, margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor=skin["paper"], plot_bgcolor=skin["plot"],
        # `dragmode` NON si imposta qui. Streamlit lo sceglie da sé in base ai
        # modi di selezione richiesti, e con il lazo imposto a mano spegne il
        # clic sul singolo punto (`clickmode` torna a "event"): si potrebbe
        # disegnare ma non scegliere un brano. Lo strumento lazo resta nella
        # barra del grafico, a un clic di distanza.
        showlegend=True, hovermode="closest",
        hoverlabel={"align": "left", "font": {"size": 11}},
        legend={"orientation": "h", "y": -0.02, "x": 0,
                "font": {"size": 10, "color": skin["ink"]},
                "bgcolor": "rgba(0,0,0,0)", "itemsizing": "constant"},
        xaxis={"visible": False, "showgrid": False, "zeroline": False},
        yaxis={"visible": False, "showgrid": False, "zeroline": False,
               "scaleanchor": "x"},
    )
    return figure


# --------------------------------------------------------------------------
# Le tre parti della pagina
# --------------------------------------------------------------------------

def render_infos(store: MapStore) -> None:
    """Quanto è grande la mappa, dove sta, e la proiezione."""
    c1, c2, c3 = st.columns(3)
    c1.metric("Tracks analyzed", f"{len(store):,}")
    c2.metric("Placed on the map", f"{store.placed:,}",
              delta=(f"{len(store) - store.placed:,} waiting"
                     if not store.projected else None),
              delta_color="off",
              help="A track gets its place from the projection. The ones "
                   "analyzed since the last one wait for the next.")
    c3.metric("Embedding", f"{store.embeddings.shape[1] if len(store) else 0}-D")
    st.caption(
        f"Stored in `{store.directory}` — `tracks.jsonl` (one JSON line per "
        "track), `embeddings.f32` (1280 raw float32 per track, same order), "
        "`coords.npy` (the X/Y of the projection). The first two are "
        "append-only: that is what makes the build interruptible.")

    if not available():
        st.warning("`essentia` is not importable here, so no track can be "
                   "analyzed. The map itself still opens.")
    if missing_models():
        st.warning(f"Model files missing from `{MODEL_DIR}` — see Tag analysis.")
    if not umap_available():
        st.warning("`umap-learn` is not importable here, so the projection "
                   "cannot be recomputed. `poetry install` brings it in.")

    if not len(store) or not umap_available():
        return

    st.caption("The projection is what turns 1280 dimensions into a picture. "
               "Recompute it whenever new tracks come in — and after changing "
               "either knob below.")
    col_n, col_d, col_go = st.columns([2, 2, 2])
    neighbors = col_n.slider(
        "Neighbours", 5, 100, ProjectionSettings.n_neighbors,
        help="How many tracks define what 'nearby' means. Low: many small "
             "recognisable clusters. High: one continent with soft edges.")
    min_dist = col_d.slider(
        "Min distance", 0.0, 0.9, ProjectionSettings.min_dist, 0.05,
        help="How tightly points may pack. Low: dense separated clumps, "
             "easier to aim at with the mouse.")
    col_go.markdown("<div style='height:1.8em'></div>", unsafe_allow_html=True)
    if col_go.button("↻ Recompute the projection", width="stretch"):
        # Su decine di migliaia di brani è un'attesa di minuti, e uno spinner
        # muto non distingue "sta lavorando" da "si è piantato". Le fasi sono
        # tre e si annunciano; la percentuale no, perché UMAP non dice a che
        # punto è e inventarla sarebbe peggio.
        started = time.perf_counter()
        with st.status(f"Projecting {len(store):,} tracks…",
                       expanded=True) as status:
            def announce(label: str, _clock=[started]) -> None:
                status.write(f"{label} — {_spelled(time.perf_counter() - _clock[0])} so far")

            coords = project(store.embeddings,
                             ProjectionSettings(n_neighbors=neighbors,
                                                min_dist=min_dist),
                             on_step=announce)
            announce("Saving the coordinates")
            store.set_coords(coords)
            status.update(
                label=f"{len(coords):,} tracks placed in "
                      f"{_spelled(time.perf_counter() - started)}",
                state="complete", expanded=False)
        _open_store.clear()
        st.rerun()


def render_map(store: MapStore) -> None:
    """La mappa, la selezione e la playlist: il cuore della pagina."""
    if not len(store):
        st.info("The map is empty. Open **Add tracks to the map** at the "
                "bottom of this page and point it at a folder.")
        return
    placed = store.placed
    if not placed:
        st.warning(
            f"{len(store):,} tracks are analyzed but none has a place yet — "
            "the projection has never been computed (or was dropped). "
            "**Map infos** above has the button.")
        return

    # Si lavora sui brani PIAZZATI, che sono i primi `placed`. Gli altri sono
    # arrivati dopo l'ultima proiezione — tipicamente perché un job sta
    # lavorando proprio adesso — e aspettano la prossima: la mappa resta
    # usabile nel frattempo, che è il punto.
    frame = pd.DataFrame(store.rows[:placed])
    frame["x"], frame["y"] = store.coords[:, 0], store.coords[:, 1]
    frame["index"] = np.arange(len(frame))
    frame["genre_list"] = frame["genres"].fillna("").str.split("; ")

    genre_counts = Counter(g for tags in frame["genre_list"] for g in tags if g)
    key_counts = Counter(k for k in frame["camelot"] if k)

    with st.expander("Filters — they narrow the map *and* the suggestions"):
        f1, f2 = st.columns([3, 2])
        chosen_genres = f1.multiselect(
            "Genres", [g for g, _ in genre_counts.most_common()],
            help="A track carrying any of the chosen genres stays. Tracks are "
                 "multi-label on purpose: 'Minimal' and 'Deep House' can both "
                 "be true of the same track.")
        chosen_keys = f2.multiselect("Camelot keys", sorted(key_counts))

        bpm_values = frame["bpm"].dropna()
        low, high = (float(bpm_values.min()), float(bpm_values.max())) \
            if len(bpm_values) else (60.0, 200.0)
        b1, b2 = st.columns([3, 2])
        bpm_range = b1.slider("BPM", low, max(high, low + 1),
                              (low, max(high, low + 1)))
        search = b2.text_input("Name contains",
                               placeholder="part of a file name")

    visible = frame
    if chosen_genres:
        wanted = set(chosen_genres)
        visible = visible[visible["genre_list"].map(
            lambda tags: bool(wanted & set(tags)))]
    if chosen_keys:
        visible = visible[visible["camelot"].isin(chosen_keys)]
    visible = visible[visible["bpm"].isna() |
                      visible["bpm"].between(bpm_range[0], bpm_range[1])]
    if search.strip():
        visible = visible[visible["name"].str.contains(search.strip(),
                                                       case=False, regex=False)]
    if not len(visible):
        st.info("No track matches these filters.")
        return

    pool = visible["index"].to_numpy()

    by_level, by_size = st.columns(2)
    level = GENRE_LEVELS[by_level.radio(
        "Colour by", list(GENRE_LEVELS), horizontal=True,
        help="Discogs labels are already two-level. The macro genre leaves "
             "almost nothing grey; the detailed one separates the house from "
             "the disco, at the cost of a larger 'other'.")]
    size_by = by_size.radio(
        "Point size", list(SIZE_FIELDS), horizontal=True,
        help="What the diameter of a point says. The position already says "
             "how a track sounds; this is room for a number you can read — "
             "how fast, how straight, how loud. Tracks missing that number "
             "stay at the smallest size.")
    # La scala si calcola su TUTTI i brani filtrati, non sul campione
    # disegnato: altrimenti il significato di "punto grande" cambierebbe a
    # ogni ricampionamento.
    visible = visible.assign(_size=marker_sizes(visible, SIZE_FIELDS[size_by]))

    drawn = visible
    sampled = len(drawn) > MAX_POINTS
    if sampled:
        drawn = drawn.sample(MAX_POINTS, random_state=0)

    cost = TransitionCost(store.coords, frame["bpm"].tolist(),
                          frame["camelot"].tolist())
    # Da percorso a posizione: i brani spariti dalla mappa (tolti, o non più
    # piazzati) semplicemente non si ritrovano, e cadono fuori da soli.
    at_path = {row["path"]: i for i, row in enumerate(store.rows[:placed])}
    playlist = [at_path[p] for p in st.session_state.get(PLAYLIST, [])
                if p in at_path]

    # La selezione si legge PRIMA di ridisegnare, non dal valore restituito
    # dal grafico. Streamlit tiene lo stato della selezione in sessione, già
    # aggiornato all'inizio del giro: leggerlo qui vuol dire che il cerchio
    # del seme è addosso al punto appena cliccato, invece che al punto di
    # prima fino al clic successivo.
    picked, lasso = read_selection()
    picked = [i for i in picked if i < placed]
    if len(picked) == 1 and not lasso:
        remember_seed(frame, picked[0])
    by_name = st.session_state.get("map::seedpick")
    if by_name is not None and by_name < placed:
        if frame.at[by_name, "path"] != st.session_state.get(PICKED):
            st.session_state[PICKED] = frame.at[by_name, "path"]
            remember_seed(frame, by_name)
    seed = at_path.get(st.session_state.get(SEED))

    drawn = drawn.assign(
        genre_key=drawn["top_genre"].map(lambda g: genre_level(g, level)))
    ranked = Counter(g for g in drawn["genre_key"] if g)
    top_genres = [g for g, _ in ranked.most_common(COLORED_GENRES)]
    st.plotly_chart(
        build_figure(drawn, top_genres, store.coords, playlist, seed),
        key="map::chart", on_select="rerun",
        selection_mode=("points", "box", "lasso"),
        config={"displaylogo": False, "scrollZoom": True})

    waiting = len(store) - placed
    if waiting:
        st.caption(
            f"➕ **{waiting:,} track(s)** analyzed since the last projection "
            "are not placed yet. Recompute it in **Map infos** to bring them "
            "in — no need to wait for the job to end.")

    st.caption(
        (f"Point size: **{size_by}** (scaled 5th–95th percentile) · "
         if SIZE_FIELDS[size_by] else "")
        + f"**{len(visible):,} track(s)** on the map"
        + (f" — {MAX_POINTS:,} of them drawn, a stable random sample; the "
           "suggestions still consider every one." if sampled else "")
        + " · **Click** a point to make it the seed. In the toolbar above, "
          "the **lasso** draws a path through the clusters and the **box** "
          "grabs a group to be sorted. Scroll to zoom.")

    # Il seme si sceglie cliccando un punto, ma non solo: quando la serata
    # parte da un brano deciso prima, cercarlo per nome è più veloce che
    # trovarne il puntino. Sopra qualche migliaio di voci l'elenco non si apre
    # più in fretta, e allora si chiede di restringere i filtri.
    query = st.text_input(
        "Find a track", key="map::seedsearch",
        placeholder="a few words in any order — artist, title, remix…",
        help="Every word has to appear, in any order, in the file name or "
             "the folder. Picking one makes it the seed, exactly as "
             "clicking its point does.")
    words = [word for word in query.casefold().split() if word]

    if words:
        found = matching_tracks(frame, pool, words)
        options = found[:SEED_MATCHES_MAX]
        st.caption(f"**{len(found):,}** match"
                   + (f" — the first {len(options)} are listed."
                      if len(found) > len(options) else "."))
    elif len(pool) <= SEED_PICKER_MAX:
        # Pochi brani: l'elenco intero si apre in fretta e cercare è inutile.
        options = pool.tolist()
    else:
        options = []
        st.caption(f"{len(visible):,} tracks on the map — type a few words "
                   "above to pick one by name, or click its point.")

    # Una scelta di prima che i filtri, una ricerca nuova o una rimozione
    # hanno fatto sparire dalle opzioni va tolta PRIMA che il menu esista, o
    # si troverebbe addosso un valore che non può mostrare.
    if st.session_state.get("map::seedpick") not in set(options):
        st.session_state.pop("map::seedpick", None)
    if options:
        st.selectbox("Seed track", options=options, index=None,
                     format_func=lambda i: f"{frame.at[i, 'name']}  ·  "
                                           f"{Path(frame.at[i, 'folder']).name}",
                     key="map::seedpick", placeholder="…pick one")


    st.divider()

    # La sezione sta in un blocco a scomparsa suo, separata dalla lavagna:
    # sono due modi diversi di costruire un set e mescolarli in una colonna
    # sola voleva dire non vedere mai per intero ne' l'uno ne' l'altro. Si
    # apre da se' quando c'e' qualcosa da farci — una selezione o una
    # playlist gia' iniziata — perche' chiusa e vuota non direbbe nulla.
    _has_choice = bool(lasso or len(picked) or seed is not None)
    with st.expander(
            "✨ Magic Playlist — turn a selection into an ordered set"
            + (f" · {len(playlist)} track(s) so far" if playlist else ""),
            expanded=_has_choice or bool(playlist)):
        render_magic_playlist(frame, cost, pool, store, seed, picked,
                              lasso, playlist)


def selection_rows(frame: pd.DataFrame, indices) -> pd.DataFrame:
    """Le righe da mostrare per i brani selezionati, nell'ordine dato.

    L'ordine e' quello che arriva e non si tocca: da un lasso aperto e'
    l'ordine in cui la linea incontra i brani, che e' gia' la scaletta.
    """
    return pd.DataFrame([{
        "#": position + 1,
        "file": frame.at[i, "name"],
        "BPM": frame.at[i, "bpm"],
        "key": frame.at[i, "camelot"],
        "groove": frame.at[i, "danceability"],
        "genres": frame.at[i, "genres"],
        "folder": frame.at[i, "folder"],
        "_path": frame.at[i, "path"],
    } for position, i in enumerate(indices)])


def _selection_table(frame: pd.DataFrame, indices, key: str) -> None:
    """I brani selezionati, elencati e ascoltabili.

    Il conteggio da solo non bastava: prima di ordinare venti brani in un
    set si vuole vedere QUALI sono, e poterne sentire uno.
    """
    if not len(indices):
        return
    table = selection_rows(frame, indices)
    play_table(
        f"map_selection::{key}", table,
        ["#", "file", "BPM", "key", "groove", "genres", "folder"],
        _read_only("#", "file", "BPM", "key", "groove", "genres", "folder"),
        editable=False, editor_key=f"map_selection_editor::{key}")


def render_magic_playlist(frame: pd.DataFrame, cost: TransitionCost, pool,
                          store: MapStore, seed, picked, lasso,
                          playlist: list[int]) -> None:
    """Dalla selezione sulla mappa a una playlist ordinata."""

    if lasso:
        # Lo stesso gesto fa due domande diverse, e quale delle due si stia
        # facendo lo dice la forma: un tratto che si richiude è un recinto
        # ("prendi quello che c'è dentro"), uno aperto è un percorso
        # ("prendi quello che tocco, nell'ordine in cui lo tocco"). Si
        # indovina dalla forma e si lascia correggere: indovinare è comodo
        # finché non sbaglia, e quando sbaglia deve bastare un clic.
        stroke = _path_points(lasso[0])
        guess = "everything inside" if closed_shape(stroke) else "the line itself"
        # La forma decide da capo ogni volta che CAMBIA natura: disegnare un
        # recinto dopo una linea torna a "quello che c'è dentro". Finché si
        # continua a disegnare la stessa cosa, invece, una correzione a mano
        # resta: l'ipotesi è comoda, ma l'ultima parola è di chi disegna.
        if st.session_state.get(LASSO_KIND) != guess:
            st.session_state[LASSO_KIND] = guess
            st.session_state[LASSO_MODE] = guess
        mode = st.radio(
            "From the lasso, take", ["everything inside", "the line itself"],
            key=LASSO_MODE, horizontal=True,
            help="A stroke that comes back where it started is read as a "
                 "fence and takes what it encloses; an open one is read as a "
                 "path and takes what it passes near, in that order.")

        if mode.startswith("everything"):
            chosen = picked
            st.markdown(f"**{len(chosen)} track(s)** inside the shape you "
                        "drew — in no order of their own, which is what "
                        "magic sort is for.")
            _selection_table(frame, chosen, "inside")
            c1, c2 = st.columns(2)
            if c1.button("✨ Magic sort them into the playlist", type="primary",
                         width="stretch", disabled=len(chosen) < 2):
                with st.spinner(f"Sorting {len(chosen)} tracks…"):
                    remember_playlist(frame, magic_sort(cost, chosen))
                st.rerun()
            if c2.button("➕ Append them, unsorted", width="stretch",
                         disabled=not chosen):
                remember_playlist(frame, playlist + [i for i in chosen
                                                     if i not in playlist])
                st.rerun()
        else:
            span = store.coords.max(axis=0) - store.coords.min(axis=0)
            diagonal = float(np.hypot(*span)) or 1.0
            radius_pct = st.slider(
                "How far from the line a track can be", 0.5, 10.0, 2.0, 0.5,
                format="%.1f%%",
                help="As a share of the map's diagonal. Wider takes in more "
                     "of the cluster the line crosses.")
            ordered = along_path(store.coords, stroke,
                                 radius=diagonal * radius_pct / 100, pool=pool)
            st.markdown(f"**{len(ordered)} track(s)** under the line you "
                        "drew, in the order the line meets them.")
            _selection_table(frame, ordered, "line")
            c1, c2 = st.columns(2)
            if c1.button("➕ Use it as the playlist", type="primary",
                         width="stretch", disabled=not ordered):
                remember_playlist(frame, ordered)
                st.rerun()
            if c2.button("➕ Append it to the playlist", width="stretch",
                         disabled=not ordered):
                remember_playlist(frame, playlist + [i for i in ordered
                                                     if i not in playlist])
                st.rerun()

    elif len(picked) > 1:
        st.markdown(f"**{len(picked)} track(s)** selected.")
        st.caption(
            "Magic sort walks all of them once, in the order that keeps every "
            "transition cheap — the travelling-salesman path over the cost "
            "below. It is the answer to a folder of tracks in no order.")
        _selection_table(frame, picked, "picked")
        c1, c2 = st.columns(2)
        if c1.button("✨ Magic sort them into the playlist",
                     type="primary", width="stretch"):
            with st.spinner(f"Sorting {len(picked)} tracks…"):
                remember_playlist(frame, magic_sort(cost, picked))
            st.rerun()
        if c2.button("➕ Append them, unsorted", width="stretch"):
            remember_playlist(frame, playlist + [i for i in picked
                                                 if i not in playlist])
            st.rerun()

    if seed is not None and not lasso and len(picked) <= 1:
        render_seed(frame, cost, pool, store, seed, playlist)
    elif not lasso and not picked:
        st.info("Nothing selected yet. Click a point on the map, or draw a "
                "lasso through it.")


def render_playlist_section(store: MapStore) -> None:
    """La playlist, sotto le due sezioni che la riempiono.

    Stava in fondo a Magic Playlist, ed era il posto sbagliato: anche la
    lavagna ci scrive, e la lavagna sta più giù. Chi mandava una catena da
    laggiù non vedeva succedere niente — il risultato compariva sopra, in
    un'altra sezione, che poteva benissimo essere chiusa. Una cosa scritta da
    due posti si mostra dopo entrambi.
    """
    if not len(store) or not store.placed:
        return
    placed = store.placed
    frame = pd.DataFrame(store.rows[:placed])
    cost = TransitionCost(store.coords[:placed], frame["bpm"].tolist(),
                          frame["camelot"].tolist())
    at_path = {row["path"]: i for i, row in enumerate(store.rows[:placed])}
    playlist = [at_path[p] for p in st.session_state.get(PLAYLIST, [])
                if p in at_path]
    render_playlist(frame, cost, playlist)


def render_seed(frame: pd.DataFrame, cost: TransitionCost, pool, store: MapStore,
                seed: int, playlist: list[int]) -> None:
    """Il brano scelto e cosa ci va dietro."""
    row = frame.iloc[seed]
    # La danceability è la regolarità degli attacchi: 1 = cassa dritta,
    # verso lo 0 il ritmo è sincopato (breakbeat, funk, roba non lineare).
    groove = f" · groove {row['danceability']:.2f}" \
        if row["danceability"] is not None and not pd.isna(row["danceability"]) else ""
    st.markdown(f"**Seed — {row['name']}**  \n"
                f"{row['bpm'] or '?'} BPM · {row['camelot'] or '?'}{groove} · "
                f"{row['genres']}")
    # anche quando e' uno solo: la riga si ascolta come le altre
    _selection_table(frame, [seed], "seed")

    w1, w2, w3 = st.columns(3)
    cost.w_map = w1.slider("Weight — sound", 0.0, 2.0, 1.0, 0.1,
                           help="How much the distance on the map counts: "
                                "the acoustic affinity of the two tracks.")
    cost.w_bpm = w2.slider("Weight — BPM", 0.0, 2.0, 1.0, 0.1,
                           help="How much the tempo gap counts. Beyond ±6% "
                                "the cost climbs fast.")
    cost.w_key = w3.slider("Weight — key", 0.0, 2.0, 1.0, 0.1,
                           help="How much harmonic distance counts. Adjacent "
                                "or relative keys (8A→9A, 8A→8B) cost nothing.")

    # Quanti candidati elencare. Uno solo per entrambe le schede: è la stessa
    # domanda — quanti me ne fai vedere — posta su due criteri diversi, e due
    # manopole scollegate darebbero due liste lunghe diverse senza motivo.
    # Il tetto è la libreria stessa, perché su una mappa appena nata chiedere
    # venti vicini a chi ne ha tre non ha senso.
    room = min(SUGGESTION_MAX, max(1, len(frame) - 1))
    shown = st.slider("How many to list", SUGGESTION_STEP, room,
                      min(SUGGESTION_DEFAULT, room), SUGGESTION_STEP,
                      key="map_suggestion_count",
                      help="Applies to both tabs below.") \
        if room > SUGGESTION_STEP else room

    mix_tab, sound_tab = st.tabs(["Mixes out of it", "Sounds like it"])

    with mix_tab:
        st.caption("Ranked by the transition cost — sound, tempo and key "
                   "together, with the weights above. Only tracks that pass "
                   "the filters are considered.")
        suggestions = nearest(cost, seed, k=shown, pool=pool)
        table = pd.DataFrame([{
            "Add": False,
            "cost": round(value, 3),
            "file": frame.at[i, "name"],
            "BPM": frame.at[i, "bpm"],
            "key": frame.at[i, "camelot"],
            "groove": frame.at[i, "danceability"],
            # Le tre parti del costo, non tre scarti: dicono QUANTO due brani
            # sono lontani su ciascun asse, da 0 a 1, non da che parte. Il
            # nome "Δ" prometteva un segno che qui non c'è — e da quando la
            # lavagna mostra scarti veri, prometterlo confondeva le due cose.
            "sound": round(cost.parts(seed, i)["map"], 3),
            "bpm cost": round(cost.parts(seed, i)["bpm"], 2),
            "key cost": round(cost.parts(seed, i)["key"], 2),
            "genres": frame.at[i, "genres"],
            "folder": frame.at[i, "folder"],
            "_path": frame.at[i, "path"],
            "_row": i,
        } for i, value in suggestions])
        if not len(table):
            st.info("No candidate passes the filters.")
        else:
            # Spente di default: qui si scelgono pochi brani fra i venti
            # proposti, non si prende tutto — il contrario di Tag analysis,
            # dove la coda e' gia' quella su cui si vuole lavorare.
            add_all, mix_key = tick_all("map_suggestions", default=False)
            table["Add"] = add_all
            # Ascoltare e scegliere sulla STESSA riga. Prima erano due
            # tabelle, una per spuntare e una per sentire: gli stessi venti
            # brani scritti due volte, e la decisione presa su una riga
            # mentre l'orecchio stava sull'altra.
            edited = play_table(
                "map_suggestions", table,
                ["Add", "cost", "file", "BPM", "key", "groove", "sound",
                 "bpm cost", "key cost", "genres", "folder"],
                {"Add": st.column_config.CheckboxColumn(
                    "Add", help="Tick what you want in the playlist, then "
                                "the button below."),
                 **_read_only("cost", "file", "BPM", "key", "groove", "sound",
                              "bpm cost", "key cost", "genres", "folder")},
                editor_key=mix_key)
            wanted = [int(i) for i in edited.loc[edited["Add"], "_row"]]
            if st.button(f"➕ Add {len(wanted)} to the playlist",
                         disabled=not wanted, type="primary"):
                remember_playlist(frame, playlist + [i for i in wanted
                                                     if i not in playlist])
                st.rerun()

    with sound_tab:
        st.caption(
            "Pure acoustic closeness, measured in the 1280 dimensions of the "
            "embedding — not on the flattened map, and with no regard for "
            "tempo or key. This is 'what else sounds like this', which is a "
            "different question from 'what mixes out of this'.")
        neighbours = pd.DataFrame([{
            "Add": False,
            "similarity": round(score, 3),
            "file": frame.at[i, "name"],
            "BPM": frame.at[i, "bpm"],
            "key": frame.at[i, "camelot"],
            "genres": frame.at[i, "genres"],
            "folder": frame.at[i, "folder"],
            "_path": frame.at[i, "path"],
            "_row": i,
        } for i, score in store.similar(seed, k=shown, limit=len(frame))])
        if not len(neighbours):
            st.info("Nothing to compare this one with yet.")
        else:
            # Si sceglie anche da qui, e non solo si ascolta: un brano che
            # somiglia al seme e' un candidato quanto uno che ci si mixa —
            # trovarlo e non poterlo prendere voleva dire cercarselo a mano
            # nell'altra scheda, dove magari non compare nemmeno.
            near_all, near_key = tick_all("map_neighbours", default=False)
            neighbours["Add"] = near_all
            picked_near = play_table(
                "map_neighbours", neighbours,
                ["Add", "similarity", "file", "BPM", "key", "genres", "folder"],
                {"Add": st.column_config.CheckboxColumn(
                    "Add", help="Tick what you want in the playlist, then "
                                "the button below."),
                 **_read_only("similarity", "file", "BPM", "key", "genres",
                              "folder")},
                editor_key=near_key)
            near_wanted = [int(i) for i in
                           picked_near.loc[picked_near["Add"], "_row"]]
            if st.button(f"➕ Add {len(near_wanted)} to the playlist",
                         disabled=not near_wanted, type="primary",
                         key="map_neighbours_add"):
                remember_playlist(frame, playlist + [i for i in near_wanted
                                                     if i not in playlist])
                st.rerun()


def render_playlist(frame: pd.DataFrame, cost: TransitionCost,
                    playlist: list[int]) -> None:
    """La playlist come sta, e come portarsela via.

    Le posizioni arrivano già risolte e già ripulite da chi ha disegnato la
    mappa: rileggerle qui dalla sessione vorrebbe dire rileggerle grezze, e
    puntare a brani che sulla mappa non ci sono più.
    """
    if not playlist:
        return

    st.divider()
    st.subheader(f"Playlist — {len(playlist)} track(s)")

    costs = [None] + [cost.between(a, b) for a, b in zip(playlist, playlist[1:])]
    table = pd.DataFrame([{
        "#": position + 1,
        "file": frame.at[i, "name"],
        "BPM": frame.at[i, "bpm"],
        "key": frame.at[i, "camelot"],
        "from previous": round(step, 3) if step is not None else None,
        "genres": frame.at[i, "genres"],
        "folder": frame.at[i, "folder"],
        "_path": frame.at[i, "path"],
    } for position, (i, step) in enumerate(zip(playlist, costs))])

    play_table("map_playlist", table,
               ["#", "file", "BPM", "key", "from previous", "genres", "folder"],
               {"file": st.column_config.TextColumn(disabled=True),
                "folder": st.column_config.Column(disabled=True),
                "from previous": st.column_config.NumberColumn(
                    "from previous", disabled=True,
                    help="The transition cost from the track above: 0 is "
                         "seamless, 1 is as far as this library goes.")},
               editable=False, editor_key="map_playlist_editor")

    worst = max((c for c in costs if c is not None), default=0)
    st.caption(f"Roughest transition: **{worst:.3f}**. Magic sort is what "
               "brings that number down.")

    p1, p2, p3, p4 = st.columns(4)
    if p1.button("✨ Magic sort", width="stretch", disabled=len(playlist) < 3):
        with st.spinner(f"Sorting {len(playlist)} tracks…"):
            remember_playlist(frame, magic_sort(cost, playlist,
                                                start=playlist[0]))
        st.rerun()
    if p2.button("🗑 Clear", width="stretch"):
        st.session_state[PLAYLIST] = []
        st.rerun()

    tracks = []
    for i in playlist:
        path = Path(frame.at[i, "path"])
        title, artist = read_title_artist(path)
        tracks.append({"path": path, "name": title, "artist": artist,
                       "bpm": frame.at[i, "bpm"],
                       "duration": frame.at[i, "duration"],
                       "genre": frame.at[i, "top_genre"], "cues": []})

    # I due pulsanti dicono COSA sono, non solo in che formato: è la
    # distinzione su cui rekordbox si impunta, e leggerla sul pulsante evita
    # di scoprirla dal selettore dei file che rifiuta l'estensione.
    p3.download_button("⬇ Export as playlist (M3U8)", build_m3u8(tracks),
                       "wavecut_playlist.m3u8", "audio/x-mpegurl",
                       width="stretch",
                       help="What rekordbox's Import Playlist accepts. "
                            "Order and files only — no BPM, no cues.")
    p4.download_button("⬇ Export as library (rekordbox XML)",
                       build_rekordbox_xml(tracks), "wavecut_library.xml",
                       "application/xml", width="stretch",
                       help="A library, not a playlist file: load it under "
                            "Preferences ▸ Advanced ▸ Database ▸ rekordbox "
                            "xml. Carries the BPM and the cues.")
    # Il punto pratico, non la differenza di formato: rekordbox importa le
    # playlist da M3U8 e la libreria da XML, e il suo "Import Playlist" non
    # apre proprio i file .xml — nel selettore risultano non selezionabili,
    # che sembra un file rotto e non lo è.
    st.caption(
        "**M3U8** is what `File ▸ Import ▸ Import Playlist` takes — "
        "rekordbox's playlist import does not read XML, so an .xml will not "
        "even be selectable there. The **XML** goes in as a library instead: "
        "`Preferences ▸ Advanced ▸ Database ▸ rekordbox xml`, point *Imported "
        "Library* at the file, and the **Wavecut** playlist appears under the "
        "`rekordbox xml` tree in the sidebar, ready to drag into your "
        "collection. Only the XML carries the BPM and the cues, and it is "
        "what the third-party converters read.")


def graph_seeds(at_path: dict[str, int]) -> list[int]:
    """I brani selezionati sulla mappa, come candidati ad aprire la lavagna.

    La lavagna sta in fondo alla pagina e chiusa: senza portarle la selezione
    qui sopra, sceglierne il primo brano vuol dire cercarlo per nome davanti a
    una figura che lo sta già mostrando.

    Il seme viaggia come percorso e non come posizione — vale per lui la
    stessa ragione del resto della sessione — quindi va ritradotto qui.
    """
    picked, _ = read_selection()
    if picked:
        return picked
    seed = at_path.get(st.session_state.get(SEED))
    return [] if seed is None else [seed]


@st.fragment
def render_graph_section(store: MapStore) -> None:
    """La lavagna: costruire un percorso un brano alla volta.

    È un frammento perché ogni gesto sulla lavagna — spostare una scheda,
    prenderne una dalla rosa — fa ripartire lo script, e ripartire per intero
    vuol dire ridisegnare la mappa da ventimila punti per aver mosso una
    scheda di dieci pixel. Da frammento si ridisegna solo questa sezione, e il
    gesto smette di aspettare la mappa.

    Lavora sui brani piazzati per intero, non su quelli filtrati dalla
    sezione sopra: la lavagna è un secondo modo di scegliere, non un'
    estensione del primo, e i suoi filtri sono suoi.
    """
    if not len(store) or not store.placed:
        return
    placed = store.placed
    frame = pd.DataFrame(store.rows[:placed])
    frame["index"] = np.arange(len(frame))
    cost = TransitionCost(store.coords[:placed], frame["bpm"].tolist(),
                          frame["camelot"].tolist())
    at_path = {row["path"]: i for i, row in enumerate(store.rows[:placed])}
    pool = frame["index"].to_numpy()
    chosen = [i for i in graph_seeds(at_path) if i < placed]
    render_graph_builder(frame, cost, pool, at_path, chosen,
                         set_playlist=lambda idxs: remember_playlist(frame, idxs))


@st.fragment(run_every=2)
def render_progress() -> None:
    """L'avanzamento del job, che si aggiorna da solo ogni due secondi.

    È un frammento: ridisegna sé stesso senza rieseguire la pagina. La
    differenza non è estetica — rieseguire tutto vorrebbe dire rileggere la
    mappa e ridisegnare ventimila punti ogni due secondi, mentre il job sta
    già usando tutti i core che ha.

    Lo stato lo scrive il job su file a ogni brano, quindi qui basta
    rileggerlo: nessun canale fra i due processi, e funziona anche se il job
    è stato lanciato da terminale.
    """
    state = load_map_state()
    if state is None or not state.running:
        # Finito mentre lo si guardava: adesso la pagina intera ha qualcosa
        # da dire (brani nuovi sulla mappa), e conviene ricaricarla.
        _open_store.clear()
        st.rerun()

    how = process_state(state.pid)
    st.progress(state.done / max(1, state.total),
                text=(f"⏸ paused at {state.done:,}/{state.total:,}" if how == "paused"
                      else f"{state.done:,}/{state.total:,} · {state.current[:50]}"))
    j1, j2, j3 = st.columns(3)
    j1.metric("On the map", f"{state.written:,}")
    j2.metric("Failed", f"{state.failed:,}")
    j3.metric("Left", "—" if how == "paused" else _spelled(state.eta_seconds))

    b1, b2, b3 = st.columns(3)
    if b1.button("🖥 Terminal monitor", width="stretch",
                 disabled=not DEFAULT_MAP_LOG.exists(),
                 help=f"Opens Terminal on `tail -f {DEFAULT_MAP_LOG}` — what "
                      "the job is printing, live. Closing that window does "
                      "not touch the job. The first time, macOS asks for "
                      "permission to control Terminal: say yes and click "
                      "again."):
        open_monitor()

    if how == "paused":
        if b2.button("▶ Resume", type="primary", width="stretch"):
            resume_job(state.pid)
            st.rerun()
    elif b2.button("⏸ Pause", width="stretch",
                   help="Stops the analysis where it is without losing it: "
                        "the processes stay in memory with the models "
                        "loaded, so resuming costs nothing."):
        pause_job(state.pid)
        st.rerun()

    if b3.button("⏹ Stop", width="stretch",
                 help="Ends the job. What is already on the map stays there, "
                      "and starting again picks up from the next track."):
        stop_job(state.pid)
        time.sleep(0.5)
        _open_store.clear()
        st.rerun()

    st.caption(
        f"{'Paused' if how == 'paused' else 'Running'} as process "
        f"{state.pid} on `{state.folder}` — closing this tab does not stop "
        "it, and this counter keeps up on its own. The map takes the new "
        "tracks in when the job ends.")


def render_add(store: MapStore, state) -> None:
    """Mettere brani sulla mappa: la parte lunga, che si fa una volta."""
    st.caption(
        "About 5 seconds per track on one process, 2–3 with several — twelve "
        "10-second windows analyzed instead of the whole file. A whole "
        "library is hours, which is what the background job is for: it "
        "survives closing this tab and picks up where it left off.")

    root = pick_folder("map_analysis::path", "Folder")
    queue: list[Path] = []
    if root is not None:
        key = f"mapqueue::{root}"
        if key not in st.session_state:
            with st.spinner("Listing audio files…"):
                st.session_state[key] = find_taggable(root)
        found = st.session_state[key]
        queue = store.pending(found)
        st.success(f"**{len(found):,} track(s)** under `{root.name}` — "
                   f"{len(queue):,} not on the map yet.")

    workers = st.slider("Analyses in parallel", 1, 12, default_workers(),
                        help="Each process holds its own copy of the models, "
                             "about 1.3 GB. Half the cores is the sweet spot.")

    if state is not None and state.running:
        render_progress()
        return

    if state is not None and state.total:
        (st.warning if state.failed else st.success)(
            f"Last job: {state.written:,} added, {state.failed:,} failed out "
            f"of {state.total:,}.")
        if state.errors:
            with st.expander(f"{len(state.errors)} error(s) kept"):
                st.dataframe(pd.DataFrame(state.errors), width="stretch",
                             hide_index=True)

    if root is None:
        return
    if not queue:
        st.info("Every track in this folder is already on the map.")
        return

    blocked = not available() or bool(missing_models())
    col_job, col_now = st.columns(2)
    if col_job.button(f"▶ Add all {len(queue):,} in the background",
                      type="primary", width="stretch", disabled=blocked):
        log = DEFAULT_MAP_LOG
        cmd = [sys.executable,
               str(Path(__file__).resolve().parent.parent / "map_cli.py"),
               str(root), "--workers", str(workers), "--project"]
        with open(log, "w") as out:
            subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                             start_new_session=True,
                             cwd=Path(__file__).resolve().parent.parent)
        st.success(f"Started. Output in `{log}`.")
        time.sleep(1.5)
        st.rerun()

    batch = int(col_now.number_input("Or analyze now, this many", 1,
                                     len(queue), min(20, len(queue))))
    if col_now.button(f"Analyze {batch} now", width="stretch", disabled=blocked):
        bar = st.progress(0.0, text="Loading models…")
        failures = []
        for i, profile in enumerate(
                profile_many(queue[:batch], ProfileSettings(),
                             workers=workers), 1):
            bar.progress(i / batch, text=f"{i}/{batch} · {profile.path.name[:60]}")
            if profile.error is None:
                store.append([profile])
            else:
                failures.append({"file": profile.path.name,
                                 "error": profile.error})
        bar.empty()
        if failures:
            st.warning(f"{len(failures)} track(s) could not be analyzed.")
            st.dataframe(pd.DataFrame(failures), width="stretch",
                         hide_index=True)
        _open_store.clear()
        st.rerun()


# --------------------------------------------------------------------------
# La pagina
# --------------------------------------------------------------------------

st.title("🗺️ Map")
st.caption(
    "The whole library as one picture: every track is a point, and points "
    "that sound alike sit together. Click one to see what mixes out of it, "
    "or draw a line across the clusters to turn a path into a playlist."
)

store_dir = default_store_dir()
store = _open_store(str(store_dir), _stamp(store_dir))
job = load_map_state()

with st.expander("Map infos", expanded=not store.placed):
    render_infos(store)

# Dentro a un contenitore, e non sciolto nella pagina: il frammento che
# aggiorna il job qui sotto ridisegna solo sé stesso, e per farlo si ricorda
# in che posizione della pagina sta. La mappa invece emette un numero di
# blocchi che cambia — compare il seme, compare la playlist — e ogni volta
# che cambiava, la posizione del frammento slittava e il browser si trovava
# un aggiornamento indirizzato a un punto che non esisteva più: pagina
# bianca ("Bad delta path index"). Il contenitore fa sì che tutta quella
# variabilità stia DENTRO un blocco solo, e la pagina, vista da fuori, abbia
# sempre la stessa forma.
with st.container():
    render_map(store)

st.divider()

# La lavagna sta chiusa e in fondo: se c'è una selezione sulla mappa e nessuna
# lavagna ancora, si apre da sé e lo dice nel titolo. Altrimenti selezionare un
# punto qui sopra non produce niente di visibile là sotto, e il collegamento
# fra le due sezioni resta un segreto.
_running_board = bool(st.session_state.get("map::graph"))
_waiting = [] if _running_board else graph_seeds(
    {row["path"]: i for i, row in enumerate(store.rows[:store.placed])})
with st.expander(
        "🕸️ Graph builder — grow a set one track at a time"
        + (f" · start from the {len(_waiting)} track(s) selected above"
           if _waiting else ""),
        expanded=_running_board or bool(_waiting)):
    render_graph_section(store)

# Dopo entrambe le sezioni che la riempiono, e fuori da tutte e due: la
# playlist è il risultato, e il risultato non sta dentro uno dei due modi di
# ottenerlo. Si disegna da sé solo quando c'è qualcosa dentro.
render_playlist_section(store)

st.divider()

# In fondo e chiusa: si apre da sé finché la mappa è vuota (senza, non c'è
# nient'altro da fare in questa pagina) e mentre un job sta lavorando, perché
# allora c'è qualcosa da guardare.
running = job is not None and job.running
label = ("Add tracks to the map — ▶ job running"
         if running else "Add tracks to the map")
with st.expander(label, expanded=not len(store) or running):
    render_add(store, job)
