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

import os
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis import mood_scale
from analysis.duplicates import folded
from analysis.dj_export import (build_m3u8, build_rekordbox_xml, read_m3u8,
                                read_title_artist)
from analysis.essentia_tags import MODEL_DIR, available, find_taggable, missing_models
from analysis.map_job import (DEFAULT_MAP_LOG, caffeinated, load_map_state,
                              open_monitor, pause_job, process_state,
                              resume_job, stop_job)
from analysis.map_profile import ProfileSettings, default_workers, profile_many
from analysis.map_projection import ProjectionSettings
from analysis.map_projection import available as umap_available
from analysis.map_projection import project
from analysis.map_store import MapStore, default_store_dir
from analysis.mixing import TransitionCost, magic_sort, nearest
from views.components import (NOW_PLAYING, fill_dock, pick_file, pick_files,
                              pick_folder, play_table, save_as, tick_all)
from views.graph_board import (TICKED, camelot_picker, mood_popularity,
                               render_board, render_chain_maker, reordered)
from views.track_columns import (PALETTE, READING_ORDER, read_only, reading,
                                 reading_config)

# Oltre questo numero di punti si disegna un campione. Non è la RAM a cedere
# ma il browser: WebGL regge il milione di punti in teoria, e nella pratica
# una mappa troppo fitta si trascina a ogni zoom. Il campione è casuale ma
# stabile (seme fisso), così la mappa non si rimescola a ogni rerun.
#
# Era ventimila, ed era troppo prudente: su una libreria da quarantacinquemila
# significava non disegnarne più della metà, e una mappa che mostra metà dei
# brani non è la mappa della libreria. La soglia adesso è oltre le librerie
# vere; resta perché a un certo punto il campione è meglio di una pagina che
# non si muove, e se lo zoom diventasse pesante è questa la manopola.
MAX_POINTS = 120000

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

# Due fondi e due inchiostri, uno per tema. Il fondo della mappa è staccato
# di poco da quello della pagina: quel poco basta a dire dove finisce il
# testo e comincia il territorio, senza farne un riquadro.
SKIN = {
    "light": {"paper": "#ffffff", "plot": "#f4f6f9", "ink": "#1b1f27",
              "other": "#9aa4b0", "label": "rgba(27,31,39,0.82)",
              "halo": "rgba(255,255,255,0.75)", "pin": "#1f6fd0",
              "ticked": "#e8a300", "kept": "#1f9d55"},
    "dark": {"paper": "#0e1117", "plot": "#161a22", "ink": "#eef1f6",
             "other": "#6b7684", "label": "rgba(238,241,246,0.88)",
             "halo": "rgba(14,17,23,0.75)", "pin": "#6fb4ff",
             "ticked": "#ffc233", "kept": "#3ddc84"},
}

# In sessione si tengono i PERCORSI, non le posizioni nella libreria. Una
# posizione vale finché la mappa non cambia: basta togliere un brano e la 200
# è un altro brano: la playlist resterebbe in piedi indicando le tracce
# sbagliate, che è peggio di un errore. Il percorso invece è il brano.
SEED = "map::seed"
PICKED = "map::seedpick_applied"
# L'ultimo file passato dal Finder su cui si è già agito. Senza, il campo
# resterebbe a comandare: cliccato un punto sulla mappa, al giro dopo quel
# percorso rimetterebbe il seme dov'era e il clic non varrebbe niente.
PICKED_FILE = "map::seedfile_applied"
# I brani scelti dal Finder che sulla mappa non c'erano. Passano di qui
# perché l'avviso va scritto DOPO la ripartenza che aggiunge gli altri:
# scritto prima, sparirebbe insieme alla pagina che lo mostra.
FINDER_MISSING = "map::finder_missing"
PLAYLIST = "map::playlist"

# La selezione fatta col lazo o col riquadro si tiene qui, e non la si chiede
# al grafico quando serve. Non è comodità: Streamlit calcola l'identità del
# grafico ANCHE sulla figura che gli si passa (`plotly_spec` entra nell'id
# dell'elemento), quindi cerchiare i brani appena selezionati cambia la
# figura, cambia l'id, e lo stato del widget — cioè la selezione — riparte
# vuoto al giro dopo. Il lazo diventava inservibile: si disegnava, e un
# istante dopo restava il seme di prima col suo cerchio, l'unica cosa che
# qualcuno si era ricordato di salvare. Letta una volta e messa qui, la
# selezione sopravvive a qualunque ridisegno.
SELECTION = "map::selection"

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


def remember_playlist(frame: pd.DataFrame, indices) -> None:
    st.session_state[PLAYLIST] = [frame.at[i, "path"] for i in indices]


def append_playlist(frame: pd.DataFrame, indices) -> None:
    """In coda a quello che c'è già, saltando chi c'è già.

    Si lavora sui percorsi e non sulle posizioni perché in sessione ci stanno
    i percorsi: tradurre avanti e indietro per aggiungere due brani vorrebbe
    dire che ogni chiamante deve prima ricostruirsi la playlist com'è adesso,
    ed è esattamente il passaggio che qualcuno dimentica — la playlist
    caricata da file spariva così, sostituita da quello che le si voleva
    aggiungere.
    """
    current = list(st.session_state.get(PLAYLIST, []))
    for i in indices:
        path = frame.at[i, "path"]
        if path not in current:
            current.append(path)
    st.session_state[PLAYLIST] = current


def sorted_after(cost: TransitionCost, playlist: list[int],
                 group: list[int]) -> list[int]:
    """Il gruppo ordinato per attaccarsi a quello che c'è già.

    Magic sort da solo sceglie da dove partire, e va bene finché la playlist
    comincia lì. In coda a una playlist esistente no: il primo del gruppo
    finisce dietro all'ultimo di prima, e se lo si lascia scegliere alla
    cieca quella giuntura è l'unico salto della serata. Si parte dal brano
    del gruppo che costa meno raggiungere da lì.
    """
    if not playlist:
        return magic_sort(cost, group)
    tail = playlist[-1]
    return magic_sort(cost, group,
                      start=min(group, key=lambda i: cost.between(tail, i)))


def _play(path: str | None) -> None:
    """Manda un brano al lettore in fondo alla pagina.

    Serve dove il ▶ non sta dentro una tabella e quindi non ce l'ha già
    `play_table`: il canale è lo stesso, una riga in sessione che il dock
    legge quando si disegna.
    """
    if path is not None:
        st.session_state[NOW_PLAYING] = path


def remember_seed(frame: pd.DataFrame, index: int) -> None:
    st.session_state[SEED] = frame.at[index, "path"]


def read_selection() -> list[int]:
    """I brani selezionati sul grafico in QUESTO giro, se ce ne sono.

    Lo stato del grafico sta in sessione sotto la sua chiave. I punti dei
    tracciati di servizio (il percorso della playlist, i cerchi) non portano
    `customdata` e vengono scartati: sono disegno, non brani.

    Vuoto non vuol dire "niente selezionato": vuol dire "nessun gesto in
    questo giro", perché al primo ridisegno il widget è un altro (vedi
    `SELECTION`). Chi vuole sapere cosa è scelto lo chiede alla sessione.
    """
    state = st.session_state.get("map::chart")
    selection = state.get("selection") if state else None
    if not selection:
        return []
    return [int(p["customdata"][0]) for p in selection.get("points", [])
            if p.get("customdata")]


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
           + inside["folder"].fillna("")).map(folded)
    keep = pd.Series(True, index=hay.index)
    for word in words:
        # Anche le parole cercate, non solo il testo in cui si cerca: farlo
        # fare a chi chiama vuol dire che la funzione è giusta solo finché
        # tutti si ricordano di farlo. Vale per le maiuscole e vale per gli
        # accenti — un nome che arriva dal disco di un Mac ha la tilde
        # staccata dalla lettera e non combacia con quella che si digita.
        keep &= hay.str.contains(folded(word), regex=False)
    return keep[keep].index.tolist()


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
                 playlist: list[int], seed: int | None,
                 seed_name: str | None = None,
                 ticked: list[int] | None = None,
                 selected: list[int] | None = None) -> go.Figure:
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

    # Verde per quello che è già in playlist, giallo per quello che si sta
    # spuntando adesso, inchiostro per il gruppo appena preso dalla mappa:
    # sulla nuvola la differenza fra "l'ho preso", "lo sto guardando" e "sto
    # lavorando su questi" è proprio quella che serve mentre si sceglie. Il
    # verde non dipende da nessuna selezione: la playlist si vede sempre, che
    # è il modo di sapere dove si è già stati.
    # Diametri diversi, e non per gusto: un brano può stare in due insiemi
    # insieme — lo si è appena selezionato ED è già in playlist — e con lo
    # stesso diametro l'anello disegnato per ultimo coprirebbe l'altro
    # esattamente. Concentrici, si vedono tutti e due.
    for name, marks, color, size in (
            ("in the playlist", playlist, skin["kept"], 15),
            ("being picked", ticked or [], skin["ticked"], 19),
            ("selected", selected or [], skin["ink"], 23)):
        spots = [i for i in marks if i is not None and i < len(coords)]
        if not spots:
            continue
        figure.add_trace(go.Scattergl(
            x=coords[spots][:, 0], y=coords[spots][:, 1], mode="markers",
            name=name, showlegend=False, hoverinfo="skip",
            marker={"size": size, "color": "rgba(0,0,0,0)",
                    "line": {"width": 2.5, "color": color}}))

    if seed is not None and seed < len(coords):
        figure.add_trace(go.Scattergl(
            x=[coords[seed][0]], y=[coords[seed][1]], mode="markers",
            name="seed", showlegend=False, hoverinfo="skip",
            # Lo stesso diametro del gruppo selezionato: seme e gruppo sono
            # la stessa cosa detta al singolare e al plurale, e si escludono.
            marker={"size": 23, "color": "rgba(0,0,0,0)",
                    "line": {"width": 2, "color": skin["ink"]}}))
        # Il nome accanto al cerchio: da solo, il cerchio dice DOVE ma non
        # CHE COSA, e dopo una ricerca per nome è proprio il "che cosa" che
        # si sta verificando. Solo col seme singolo — su una selezione
        # multipla non c'è un nome da scrivere.
        if seed_name:
            figure.add_annotation(
                x=float(coords[seed][0]), y=float(coords[seed][1]),
                text=f"<b>{seed_name[:46]}</b>", showarrow=False,
                yshift=18, font={"size": 12, "color": skin["pin"]},
                bgcolor=skin["halo"], borderpad=3)

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


# I filtri sono UNO SOLO per tutta la pagina, e le loro chiavi stanno qui.
# Ne esisteva un secondo dentro al Chain Maker, con la stessa ruota e le
# stesse manopole: due pannelli da tenere d'accordo per una domanda sola —
# quali brani sto guardando — e chi ne restringeva uno si stupiva che l'altro
# proponesse ancora tutto.
FILTER_KEYS = "map::flt_keys"
FILTER_KEYS_EVENT = "map::flt_keys_at"
FILTER_WIDGETS = ("map::flt_genres", "map::flt_moods", "map::flt_bpm",
                  "map::flt_groove")


def _span(frame: pd.DataFrame, column: str,
          floor: float, ceiling: float) -> tuple[float, float]:
    """Gli estremi veri di una colonna, per non offrire una corsa vuota.

    Uno slider 0..200 su una libreria che sta fra 110 e 130 è quasi tutto
    corsa morta. I due estremi devono comunque restare diversi fra loro,
    anche quando la colonna è vuota o porta un valore solo: uno slider che
    parte e finisce nello stesso punto non si disegna.
    """
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if not len(values):
        return (floor, ceiling)
    low, high = float(values.min()), float(values.max())
    return (low, high) if high > low else (low, low + 1.0)


def render_filters(frame: pd.DataFrame) -> pd.DataFrame:
    """Il pannello dei filtri della pagina, e i brani che li passano.

    Restringono TUTTO quello che la pagina propone: i punti sulla mappa, i
    candidati di Magic Playlist e la rosa del Chain Maker. Erano due pannelli
    e adesso è uno, perché la domanda era già una.

    La ruota Camelot al posto dell'elenco a tendina delle tonalità: due
    tonalità che si mixano stanno vicine sulla ruota, e su una ruota la cosa
    si vede — in un elenco alfabetico 8A e 9A sono due righe qualunque. Era
    già così nel Chain Maker; qui arriva insieme al resto.

    Il campo "Name contains" non c'è più: cercare un brano per nome è una
    cosa diversa dal restringere la libreria — si vuole QUEL brano, non i
    brani che gli somigliano — e "Find a track", sotto la mappa, lo fa già e
    lo fa meglio, perché il nome trovato diventa il seme.
    """
    keys = st.session_state.get(FILTER_KEYS) or []
    genre_counts = Counter(g for tags in frame["genre_list"] for g in tags if g)
    mood_counts = Counter(m for tags in frame["mood_list"] for m in tags if m)

    # Scegliere una tonalità sulla ruota rilancia la pagina, e un pannello
    # che torna al suo stato di riposo si richiuderebbe sotto le dita al
    # primo click. Resta aperto finché la ruota è stata toccata almeno una
    # volta, anche dopo aver tolto l'ultima tonalità — chi sta filtrando non
    # ha finito solo perché ha svuotato la scelta.
    touched = FILTER_KEYS_EVENT in st.session_state
    narrowed = any(st.session_state.get(k) for k in FILTER_WIDGETS[:2])
    with st.expander(
            "🔎 Filters — they narrow the map, the suggestions and the roster"
            + (f" · {len(keys)} key(s)" if keys else ""),
            expanded=bool(keys or touched or narrowed)):
        wheel, rest = st.columns([2, 3])

        with wheel:
            st.caption("Pick the keys you want to land on. Nothing picked "
                       "means every key is welcome.")
            picked = camelot_picker(keys, "map::flt_wheel", FILTER_KEYS_EVENT)
            if picked is not None:
                st.session_state[FILTER_KEYS] = picked
                st.rerun()

        with rest:
            chosen_genres = st.multiselect(
                "Genres", [g for g, _ in genre_counts.most_common()],
                key="map::flt_genres",
                help="A track carrying any of the chosen genres stays. Tracks "
                     "are multi-label on purpose: 'Minimal' and 'Deep House' "
                     "can both be true of the same track.")
            chosen_moods = st.multiselect(
                "Moods", [m for m, _ in mood_counts.most_common()],
                key="map::flt_moods",
                help="Same rule as the genres: a track carrying any of the "
                     "chosen moods stays. Up to four are recorded per track, "
                     "strongest first.")
            tempo = _span(frame, "bpm", 60.0, 200.0)
            bpm = st.slider("BPM", tempo[0], tempo[1], tempo,
                            key="map::flt_bpm")
            swing = _span(frame, "danceability", 0.0, 1.0)
            groove = st.slider("Groove", swing[0], swing[1], swing, step=0.01,
                               key="map::flt_groove",
                               help="The danceability: regularity of the "
                                    "onsets, low is loose and high is a "
                                    "straight kick. It is the same number "
                                    "the tables and the board call groove.")
            if st.button("↺ Reset the filters", width="stretch",
                         key="map::flt_reset"):
                st.session_state.pop(FILTER_KEYS, None)
                for widget in FILTER_WIDGETS:
                    st.session_state.pop(widget, None)
                st.rerun()

        kept = frame
        if chosen_genres:
            wanted = set(chosen_genres)
            kept = kept[kept["genre_list"].map(
                lambda tags: bool(wanted & set(tags)))]
        if chosen_moods:
            wanted = set(chosen_moods)
            kept = kept[kept["mood_list"].map(
                lambda tags: bool(wanted & set(tags)))]
        if keys:
            kept = kept[kept["camelot"].isin(keys)]
        # Un brano senza BPM o senza groove non viene escluso da un intervallo
        # su quel valore: non sappiamo dove cade, e farlo sparire sarebbe
        # rispondere "no" a una domanda che non è stata posta.
        kept = kept[kept["bpm"].isna() | kept["bpm"].between(*bpm)]
        kept = kept[kept["danceability"].isna()
                    | kept["danceability"].between(*groove)]
        st.caption(f"**{len(kept):,}** of {len(frame):,} tracks pass — the "
                   "map, the suggestions and the roster all come from these.")
    return kept


def render_map(store: MapStore) -> tuple | None:
    """La mappa e la scelta: il cuore della pagina.

    Torna quello che serve a Magic Playlist per lavorare sulla scelta appena
    fatta — `(frame, cost, pool, store, seed, selected, playlist)`, gli
    argomenti di `render_magic_playlist` — o `None` se non c'e' mappa da
    guardare. La sezione che li usa si disegna fuori di qui, accanto al Chain
    Maker: sono due tab della stessa cosa, e devono uscire dallo stesso posto.
    """
    if not len(store):
        st.info("The map is empty. Open **Map settings** above and point "
                "*Add tracks to the map* at a folder.")
        return
    placed = store.placed
    if not placed:
        st.warning(
            f"{len(store):,} tracks are analyzed but none has a place yet — "
            "the projection has never been computed (or was dropped). "
            "**Map settings** above has the button.")
        return

    # Si lavora sui brani PIAZZATI, che sono i primi `placed`. Gli altri sono
    # arrivati dopo l'ultima proiezione — tipicamente perché un job sta
    # lavorando proprio adesso — e aspettano la prossima: la mappa resta
    # usabile nel frattempo, che è il punto.
    frame = pd.DataFrame(store.rows[:placed])
    frame["x"], frame["y"] = store.coords[:, 0], store.coords[:, 1]
    frame["index"] = np.arange(len(frame))
    frame["genre_list"] = frame["genres"].fillna("").str.split("; ")
    # Una mappa fatta prima che il mood si registrasse non ha la colonna:
    # meglio un filtro che non propone niente di un errore a metà pagina.
    moods = frame["moods"] if "moods" in frame \
        else pd.Series("", index=frame.index)
    frame["mood_list"] = moods.fillna("").str.split("; ")

    visible = render_filters(frame)
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
    picked = [i for i in read_selection() if i < placed]
    if picked:
        # Un punto solo è un seme comunque lo si sia preso: un riquadro
        # attorno a un brano solo è un brano, non un gruppo. Da due in su è
        # un gruppo, e allora il seme di prima se ne va — tenerlo vorrebbe
        # dire un cerchio che sopravvive al gesto successivo e indica una
        # scelta che non si sta più facendo. Le due cose si escludono: sulla
        # mappa c'è o un brano indicato, o un gruppo.
        if len(picked) == 1:
            remember_seed(frame, picked[0])
            st.session_state[SELECTION] = []
        else:
            st.session_state[SELECTION] = [frame.at[i, "path"] for i in picked]
            st.session_state.pop(SEED, None)
        # Le spunte erano quelle della scelta di prima: cerchiarle di giallo
        # attorno a un seme che non c'è più vuol dire indicare il nulla.
        st.session_state[TICKED] = []
    by_name = st.session_state.get("map::seedpick")
    if by_name is not None and by_name < placed:
        if frame.at[by_name, "path"] != st.session_state.get(PICKED):
            st.session_state[PICKED] = frame.at[by_name, "path"]
            remember_seed(frame, by_name)
            st.session_state[SELECTION] = []
    selected = [at_path[p] for p in st.session_state.get(SELECTION, [])
                if p in at_path]
    seed = None if selected else at_path.get(st.session_state.get(SEED))

    # Sopra ventimila brani se ne disegna un campione, e il campione può non
    # contenere proprio quello che la pagina sta indicando: il cerchio del
    # seme finiva su una zona vuota, giusto di coordinate e senza il suo
    # punto sotto. Chi è indicato torna dentro comunque — un cerchio attorno
    # al nulla non è un dettaglio estetico, è la mappa che dice il falso.
    if sampled:
        pointed = [i for i in ([seed] if seed is not None else [])
                   + selected + playlist
                   if i in visible.index and i not in drawn.index]
        if pointed:
            drawn = pd.concat([drawn, visible.loc[pointed]])

    drawn = drawn.assign(
        genre_key=drawn["top_genre"].map(lambda g: genre_level(g, level)))
    ranked = Counter(g for g in drawn["genre_key"] if g)
    top_genres = [g for g, _ in ranked.most_common(COLORED_GENRES)]
    st.plotly_chart(
        build_figure(drawn, top_genres, store.coords, playlist, seed,
                     seed_name=(frame.at[seed, "name"]
                                if seed is not None else None),
                     ticked=st.session_state.get(TICKED) or [],
                     selected=selected),
        key="map::chart", on_select="rerun",
        selection_mode=("points", "box", "lasso"),
        config={"displaylogo": False, "scrollZoom": True})

    waiting = len(store) - placed
    if waiting:
        st.caption(
            f"➕ **{waiting:,} track(s)** analyzed since the last projection "
            "are not placed yet. Recompute it in **Map settings** to bring them "
            "in — no need to wait for the job to end.")

    st.caption(
        (f"Point size: **{size_by}** (scaled 5th–95th percentile) · "
         if SIZE_FIELDS[size_by] else "")
        + f"**{len(visible):,} track(s)** on the map"
        + (f" — {MAX_POINTS:,} of them drawn, a stable random sample; the "
           "suggestions still consider every one." if sampled else "")
        + " · **Click** a point to make it the seed. In the toolbar above, "
          "the **lasso** and the **box** grab the group they enclose, to be "
          "sorted into a set. Scroll to zoom.")

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
        # Il ▶ accanto al menu, non sotto: si sceglie un brano per nome senza
        # averlo mai sentito, e la prova sta nell'ascoltarlo. È lo stesso ▶
        # delle tabelle — il brano finisce nel lettore in fondo alla pagina —
        # e per la stessa ragione passa da un `on_click`: il lettore si
        # disegna prima di questa riga, e un valore scritto adesso lo
        # troverebbe già disegnato sul brano di prima.
        picking, listening = st.columns([6, 1], vertical_alignment="bottom")
        picking.selectbox("Seed track", options=options, index=None,
                          format_func=lambda i: f"{frame.at[i, 'name']}  ·  "
                                                f"{Path(frame.at[i, 'folder']).name}",
                          key="map::seedpick", placeholder="…pick one")
        chosen = st.session_state.get("map::seedpick")
        listening.button(
            "▶", key="map::seedplay", width="stretch",
            disabled=chosen is None, on_click=_play,
            args=(frame.at[chosen, "path"] if chosen is not None else None,),
            help="Hear the track picked here, in the player at the bottom "
                 "of the page.")

    # …oppure si dà il file. Il nome che uno ricorda non è quasi mai il nome
    # del file — "Blue Monday" contro "04. Blue Monday - New Order (12''
    # mix).mp3" — mentre dentro la cartella quel brano lo si riconosce a
    # colpo d'occhio, ed è lì che si va a finire quando la serata parte da un
    # pezzo deciso prima.
    chosen_file = pick_file(
        "map::seedfile", "…or the seed's file, straight from the Finder",
        placeholder="a track already on the map",
        prompt="Choose the seed track")
    if chosen_file is not None and str(chosen_file) != st.session_state.get(PICKED_FILE):
        st.session_state[PICKED_FILE] = str(chosen_file)
        # Stesse regole con cui si ritrova una playlist caricata da file:
        # il percorso, e se non basta il nome — un brano scelto dal disco
        # sbagliato è lo stesso brano.
        on_map, _ = playlist_positions([str(chosen_file)], at_path)
        if not on_map:
            st.warning(f"`{chosen_file.name}` is not on the map: add its "
                       "folder under **Map settings** at the top, or pick "
                       "another file.")
        else:
            remember_seed(frame, on_map[0])
            st.session_state[SELECTION] = []
            # Il seme di questo giro è già stato deciso più in su: senza
            # ripartire, il cerchio resterebbe sul brano di prima fino al
            # gesto successivo.
            st.rerun()

    return frame, cost, pool, store, seed, selected, playlist


def selection_rows(frame: pd.DataFrame, indices) -> pd.DataFrame:
    """Le righe da mostrare per i brani selezionati, nell'ordine dato.

    L'ordine e' quello che arriva e non si tocca: da una selezione sulla
    mappa e' l'ordine in cui il grafico riporta i punti, che non e' una
    scaletta e non deve fingere di esserlo — a metterli in fila e' magic sort.
    """
    common = mood_popularity(frame)
    return pd.DataFrame([{
        "#": position + 1,
        **reading(frame.loc[i], common),
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
        f"map_selection::{key}", table, ["#", *READING_ORDER],
        {**read_only("#"), **reading_config(frame, table)},
        editable=False, editor_key=f"map_selection_editor::{key}")


def render_magic_playlist(frame: pd.DataFrame, cost: TransitionCost, pool,
                          store: MapStore, seed, selected: list[int],
                          playlist: list[int]) -> None:
    """Dalla selezione sulla mappa a una playlist ordinata.

    Un gesto, un significato: lazo e riquadro prendono quello che chiudono
    dentro. C'era anche il modo "la linea stessa", che prendeva i brani vicini
    al tratto nell'ordine in cui li toccava; costava un radio da leggere e uno
    slider da tarare prima di ogni disegno, e la stessa scaletta la produce
    magic sort su quello che si è recintato, senza chiedere niente.
    """
    if selected:
        st.markdown(f"**{len(selected)} track(s)** selected.")
        st.caption(
            "Magic sort walks all of them once, in the order that keeps every "
            "transition cheap — the travelling-salesman path over the cost "
            "below. It is the answer to a folder of tracks in no order.")
        _selection_table(frame, selected, "picked")
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("✨ Magic sort and append", type="primary",
                     width="stretch", disabled=len(selected) < 2,
                     help="Sorted among themselves, then added after what "
                          "the playlist already holds."):
            with st.spinner(f"Sorting {len(selected)} tracks…"):
                append_playlist(frame, sorted_after(cost, playlist, selected))
            st.rerun()
        if c2.button("➕ Append them, unsorted", width="stretch"):
            append_playlist(frame, selected)
            st.rerun()
        # Sostituire resta possibile, ma va chiesto: era il comportamento del
        # pulsante primario, e una playlist caricata da file spariva al primo
        # gruppo che le si mandava.
        if c3.button("↺ Sort as a new playlist", width="stretch",
                     disabled=len(selected) < 2 or not playlist,
                     help="Starts over: what is in the playlist now is "
                          "dropped."):
            with st.spinner(f"Sorting {len(selected)} tracks…"):
                remember_playlist(frame, magic_sort(cost, selected))
            st.rerun()
        # Il grafico non può più dire "non ho più niente selezionato" — la
        # selezione vive in sessione apposta per sopravvivergli — quindi
        # lasciarla andare è un gesto che va offerto, o i cerchi restano
        # addosso a un gruppo di cui non ci si occupa più.
        if c4.button("✖ Clear the selection", width="stretch"):
            st.session_state[SELECTION] = []
            st.rerun()

    elif seed is not None:
        render_seed(frame, cost, pool, store, seed, playlist)
    else:
        st.info("Nothing selected yet. Click a point on the map to make it "
                "the seed, or drag the lasso or the box around a group.")


def render_set_builder(store: MapStore, context: tuple | None) -> None:
    """I due modi di costruire un set, in due tab della stessa sezione.

    Erano due blocchi a scomparsa, uno sotto l'altro: aperti tutti e due
    facevano una colonna in cui né l'uno né l'altro si vedeva per intero,
    chiusi tutti e due non dicevano che erano la stessa domanda — come si
    passa dalla mappa a una scaletta — con due risposte. In due tab se ne
    vede uno alla volta e per intero, e restano affiancati.

    Il Chain Maker c'è comunque, anche quando `context` è `None` — cioè
    quando i filtri non lasciano passare niente — ma con la rosa vuota: una
    catena già in piedi resta da guardare e da mandare alla playlist, e i
    filtri non hanno mai toccato i brani che ci sono già sopra.
    """
    if not len(store) or not store.placed:
        return
    if context is None:
        pool, seed, selected, playlist = [], None, [], []
    else:
        _, _, pool, _, seed, selected, playlist = context

    # Quale dei due si apre per primo: la scelta appena fatta sulla mappa
    # comanda, perché è il gesto più recente; se non ce n'è nessuna e una
    # catena è già in piedi, si apre quella. È la stessa regola con cui i due
    # blocchi si aprivano da sé, detta una volta sola.
    chain_tab = "🔗 Chain Maker"
    magic_tab = "✨ Magic Playlist"
    running_chain = bool(st.session_state.get("map::graph"))
    first = chain_tab if running_chain and not (selected or seed is not None) \
        else magic_tab

    with st.expander(
            "🎛️ Build a set — from the map to an ordered playlist"
            + (f" · {len(playlist)} track(s) so far" if playlist else ""),
            expanded=bool(selected or seed is not None or playlist
                          or running_chain)):
        magic, chain = st.tabs([magic_tab, chain_tab], default=first)
        with magic:
            if context is None:
                st.info("No track matches the map filters above — widen them "
                        "to pick a seed or a group to sort.")
            else:
                render_magic_playlist(*context)
        with chain:
            render_chain_section(store, pool)


def render_playlist_section(store: MapStore) -> None:
    """La playlist, sotto la sezione che la riempie.

    Stava in fondo a Magic Playlist, ed era il posto sbagliato: anche il
    Chain Maker ci scrive, e i due sono tab diversi. Chi mandava una catena
    non vedeva succedere niente — il risultato compariva in un tab che in
    quel momento non era quello aperto. Una cosa scritta da due posti si
    mostra dopo entrambi, e fuori da tutti e due.
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
    render_playlist(frame, cost, playlist, at_path)


def render_seed(frame: pd.DataFrame, cost: TransitionCost, pool, store: MapStore,
                seed: int, playlist: list[int]) -> None:
    """Il brano scelto e cosa ci va dietro."""
    row = frame.iloc[seed]
    # La danceability è la regolarità degli attacchi: 1 = cassa dritta,
    # verso lo 0 il ritmo è sincopato (breakbeat, funk, roba non lineare).
    groove = f" · groove {row['danceability']:.2f}" \
        if row["danceability"] is not None and not pd.isna(row["danceability"]) else ""
    common = mood_popularity(frame)
    st.markdown(f"**Seed — {row['name']}**  \n"
                f"{row['bpm'] or '?'} BPM · {row['camelot'] or '?'}{groove} · "
                f"{row['genres']}  \n"
                f"{mood_scale.summary(row['moods'], common)}")

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
                   "the filters are considered. **The first row is the seed "
                   "itself**: where a set starts belongs in it like anything "
                   "else, and from here it goes in with one tick.")
        # In testa e a costo zero, che è la verità: da sé a sé non c'è
        # transizione. Prima il seme non compariva in nessuna delle due liste
        # e la playlist si popolava solo dei suoi simili — si partiva da un
        # brano che poi nel set non c'era.
        suggestions = [(seed, 0.0)] + nearest(cost, seed, k=shown, pool=pool)
        table = pd.DataFrame([{
            "Add": False,
            "cost": round(value, 3),
            **reading(frame.loc[i], common),
            # Le tre parti del costo, non tre scarti: dicono QUANTO due brani
            # sono lontani su ciascun asse, da 0 a 1, non da che parte. Il
            # nome "Δ" prometteva un segno che qui non c'è — e da quando la
            # Chain Maker mostra scarti veri, prometterlo confondeva le due cose.
            "sound": round(cost.parts(seed, i)["map"], 3),
            "bpm cost": round(cost.parts(seed, i)["bpm"], 2),
            "key cost": round(cost.parts(seed, i)["key"], 2),
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
                ["Add", "cost", "file", "BPM", "key", "groove", "emotion",
                 "sound", "bpm cost", "key cost", "mood", "genres", "folder"],
                {"Add": st.column_config.CheckboxColumn(
                    "Add", help="Tick what you want in the playlist, then "
                                "the button below."),
                 **reading_config(frame, table),
                 **read_only("cost", "sound", "bpm cost", "key cost")},
                editor_key=mix_key)
            wanted = [int(i) for i in edited.loc[edited["Add"], "_row"]]
            st.session_state[TICKED] = wanted
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
            "different question from 'what mixes out of this'. The first row "
            "is the seed itself, here too.")
        neighbours = pd.DataFrame([{
            "Add": False,
            "similarity": round(score, 3),
            **reading(frame.loc[i], common),
            "_path": frame.at[i, "path"],
            "_row": i,
        } for i, score in [(seed, 1.0)]
            + store.similar(seed, k=shown, limit=len(frame))])
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
                ["Add", "similarity", *READING_ORDER],
                {"Add": st.column_config.CheckboxColumn(
                    "Add", help="Tick what you want in the playlist, then "
                                "the button below."),
                 **reading_config(frame, neighbours),
                 **read_only("similarity")},
                editor_key=near_key)
            near_wanted = [int(i) for i in
                           picked_near.loc[picked_near["Add"], "_row"]]
            if st.button(f"➕ Add {len(near_wanted)} to the playlist",
                         disabled=not near_wanted, type="primary",
                         key="map_neighbours_add"):
                remember_playlist(frame, playlist + [i for i in near_wanted
                                                     if i not in playlist])
                st.rerun()


def _composed(text: str) -> str:
    """Il testo con gli accenti in un carattere solo (NFC), per confrontarlo."""
    return unicodedata.normalize("NFC", text)


def add_from_finder(frame: pd.DataFrame, at_path: dict[str, int],
                    key: str) -> None:
    """Mettere in scaletta dei brani presi dal disco, anche più d'uno.

    La mappa e la catena rispondono a "cosa ci sta bene dietro"; questo
    risponde a "questi li suono e basta" — i pezzi decisi prima della
    serata, che esistono nella testa del DJ prima che in qualunque grafico.
    Senza, andavano cercati uno per uno per nome, che è la stessa cosa fatta
    dieci volte.

    Solo brani già sulla mappa: la playlist ne porta le posizioni, e un file
    che non è stato analizzato non ne ha una. Chi resta fuori viene detto —
    con il nome, non con il percorso intero, perché la domanda è "quale" e
    non "dove".
    """
    missing = st.session_state.pop(FINDER_MISSING, [])
    if missing:
        st.warning("Not on the map, so not in the playlist: "
                   + ", ".join(f"`{name}`" for name in missing)
                   + ". Add their folder under **Map settings** at the top.")
    if not st.button("🎵 Add tracks from the Finder", width="stretch", key=key,
                     help="Pick one file or several — ⌘-click or shift-click "
                          "in the panel. They go in after what the playlist "
                          "already holds, in the order the panel returns."):
        return
    chosen = pick_files("Choose tracks for the playlist")
    if not chosen:
        return          # pannello annullato: non è successo niente
    found, absent = playlist_positions([str(p) for p in chosen], at_path)
    st.session_state[FINDER_MISSING] = [Path(p).name for p in absent]
    if found:
        append_playlist(frame, found)
    st.rerun()


def playlist_positions(paths, at_path: dict[str, int]) -> tuple[list[int], list[str]]:
    """Da una playlist letta da file alle posizioni sulla mappa.

    Il percorso scritto nel file e quello registrato sulla mappa possono non
    coincidere pur essendo lo stesso brano — un disco montato con un'altra
    lettera, la libreria spostata, una playlist salvata da un altro programma
    con percorsi relativi — quindi dopo il percorso si prova il nome del file.
    È il ripiego che salva il caso normale (la libreria è una sola, i nomi
    dentro sono unici) senza pretendere di indovinare: se due cartelle
    contengono lo stesso nome, vince la prima, e resta un brano da spostare a
    mano invece di una playlist che non si carica.

    Chi non si trova torna indietro per nome: sono i brani che sulla mappa non
    ci sono ancora, e la playlist non può indicarli perché una posizione che
    non esiste non è un brano.

    **Gli accenti si confrontano composti.** macOS scrive i nomi dei file
    decomposti — "Hervé" è "Herve" più il segno di accento, due caratteri —
    mentre chi riscrive la playlist di solito li ricompone: rekordbox lo fa.
    Sono la stessa parola sullo schermo e due stringhe diverse per il
    programma, quindi il percorso non combaciava e nemmeno il ripiego sul
    nome. Non è un caso di confine: 4.067 brani su 87.010 di questa libreria
    (il 4,7%) hanno un nome decomposto, e bastava un artista accentato perché
    una scaletta tornata da rekordbox arrivasse monca — con l'aggravante che
    il messaggio mandava ad aggiungere alla mappa una cartella che c'era già
    tutta. Si confronta allora una forma sola, e la mappa continua a
    conservare il percorso VERO, che è quello che poi riapre il file.
    """
    by_path: dict[str, int] = {}
    by_name: dict[str, int] = {}
    for path, i in at_path.items():
        by_path.setdefault(_composed(path), i)
        by_name.setdefault(_composed(os.path.basename(path)), i)

    found: list[int] = []
    missing: list[str] = []
    for path in paths:
        i = by_path.get(_composed(os.path.abspath(path)))
        if i is None:
            i = by_name.get(_composed(os.path.basename(path)))
        if i is None:
            missing.append(path)
        elif i not in found:
            found.append(i)
    return found, missing


def render_playlist_loader(frame: pd.DataFrame, at_path: dict[str, int],
                           playlist: list[int]) -> None:
    """Riprendere in mano una playlist già fatta.

    Senza questo, la pagina sa solo cominciare da zero: una scaletta iniziata
    ieri — o esportata da qui la settimana scorsa — andrebbe ricostruita brano
    per brano prima di poterle aggiungere il primo brano nuovo. Caricata,
    invece, diventa la playlist su cui lavorano tutte e due le sezioni qui
    sopra.

    Si legge il CONTENUTO del file, non il file: della playlist servono i
    percorsi che porta dentro, e quelli valgono da dovunque arrivi.
    """
    uploaded = st.file_uploader(
        "Playlist file", type=["m3u8", "m3u"], key="map::playlist_file",
        help="The .m3u8 this page exports, or one saved by rekordbox, "
             "Serato, Traktor, djay… Only the track order is read.")
    if uploaded is None:
        return

    paths = read_m3u8(uploaded.getvalue().decode("utf-8", "replace"))
    found, missing = playlist_positions(paths, at_path)
    if not paths:
        st.error("No tracks in that file.")
        return

    st.caption(f"**{len(found)}** of {len(paths)} track(s) are on the map.")
    if missing:
        # Non è un errore del file: è la mappa che non li ha ancora. Detto
        # così, la mossa successiva è chiara — e sta in fondo a questa pagina.
        with st.expander(f"⚠️ {len(missing)} track(s) not on the map — "
                         "they cannot go in the playlist"):
            st.caption("Add their folder under *Map settings* at the top of "
                       "this page, then load the playlist again.")
            st.dataframe(pd.DataFrame({"file": missing}), width="stretch",
                         hide_index=True)

    c1, c2 = st.columns(2)
    if c1.button("➕ Send to playlist", type="primary", width="stretch",
                 disabled=not found, key="map::playlist_load"):
        remember_playlist(frame, found)
        st.rerun()
    if c2.button("➕ Append it to the playlist", width="stretch",
                 disabled=not found or not playlist,
                 key="map::playlist_append"):
        remember_playlist(frame, playlist + [i for i in found
                                             if i not in playlist])
        st.rerun()


def _drop_from_playlist(frame: pd.DataFrame, playlist: list[int],
                        doomed: int) -> None:
    """Toglie un brano dalla playlist e ridisegna la pagina.

    Il giro intero e non il solo frammento: la tabella qui sopra e la linea
    sulla mappa mostrano la stessa playlist, e ridisegnare la sola lavagna le
    lascerebbe indietro di un brano.
    """
    remember_playlist(frame, [i for i in playlist if i != doomed])
    st.rerun()


def _reorder_playlist(frame: pd.DataFrame, order: list[int]) -> None:
    """Rimette la playlist nell'ordine deciso trascinando sulla lavagna.

    Giro intero come per la rimozione, e per la stessa ragione: la tabella
    qui sopra e la linea sulla mappa raccontano questa stessa fila.
    """
    remember_playlist(frame, order)
    st.rerun()


def render_playlist(frame: pd.DataFrame, cost: TransitionCost,
                    playlist: list[int], at_path: dict[str, int]) -> None:
    """La playlist come sta, come portarsela via e come riprenderne una.

    Le posizioni arrivano già risolte e già ripulite da chi ha disegnato la
    mappa: rileggerle qui dalla sessione vorrebbe dire rileggerle grezze, e
    puntare a brani che sulla mappa non ci sono più.

    Vuota, la sezione non spariva soltanto: portava via con sé l'unico posto
    da cui si carica una playlist fatta prima, e chi voleva continuarne una
    trovava una pagina che sa solo cominciare. Vuota adesso resta, e mostra
    proprio quello.
    """
    st.divider()
    st.subheader(f"Playlist — {len(playlist)} track(s)" if playlist
                 else "Playlist")

    if not playlist:
        st.caption("Nothing in it yet: pick tracks in either tab of "
                   "**Build a set** above, take them from the Finder, or "
                   "load an existing playlist and keep adding to it.")
        add_from_finder(frame, at_path, "map::playlist_finder_empty")
        render_playlist_loader(frame, at_path, playlist)
        return

    costs = [None] + [cost.between(a, b) for a, b in zip(playlist, playlist[1:])]
    common = mood_popularity(frame)
    table = pd.DataFrame([{
        "#": position + 1,
        "Drop": False,
        **reading(frame.loc[i], common),
        "from previous": round(step, 3) if step is not None else None,
        "_path": frame.at[i, "path"],
        "_row": i,
    } for position, (i, step) in enumerate(zip(playlist, costs))])

    # La firma della playlist sta nella chiave, come per la catena: appena
    # l'ordine cambia la tabella rinasce, e il numero appena riscritto non
    # resta nello stato del widget a riapplicarsi a ogni giro. `tick_all` ci
    # aggiunge il suo contatore, che è l'unico modo di svuotare davvero le
    # spunte, e porta i due pulsanti sopra la tabella: sfoltire una scaletta
    # lunga vuol dire quasi sempre spuntare tutto e ridare la spunta ai
    # pochi che restano.
    drop_all, editor_key = tick_all(
        "map_playlist_editor::" + "|".join(table["_path"]), default=False)
    table["Drop"] = drop_all
    edited = play_table(
        "map_playlist", table,
        ["#", "Drop", "file", "BPM", "key", "groove", "emotion",
         "from previous", "mood", "genres", "folder"],
        {"#": st.column_config.NumberColumn(
            "#", min_value=1, max_value=len(playlist), step=1,
            help="Write the position you want this track in: the row moves "
                 "there and the others slide."),
         "Drop": st.column_config.CheckboxColumn(
             "Drop", help="Tick what you want out, then the button below."),
         "from previous": st.column_config.NumberColumn(
             "from previous", disabled=True,
             help="The transition cost from the track above: 0 is "
                  "seamless, 1 is as far as this library goes."),
         **reading_config(frame, table)},
        editor_key=editor_key)

    # Riscrivere un numero sposta la riga. Si legge dallo stato del widget e
    # non dalla tabella restituita: quello che serve è QUALE riga è stata
    # toccata, e il valore da solo non lo dice.
    moves = {int(row): values["#"]
             for row, values in st.session_state.get(editor_key, {})
             .get("edited_rows", {}).items() if "#" in values}
    order = reordered(playlist, moves) if moves else playlist
    if order != playlist:
        remember_playlist(frame, order)
        st.rerun()

    doomed = {int(i) for i in edited.loc[edited["Drop"], "_row"]}
    # I due modi di correggere una scaletta a mano, uno di fianco all'altro:
    # quello che manca si prende dal disco, quello che avanza si spunta.
    adding, dropping = st.columns(2)
    with adding:
        add_from_finder(frame, at_path, "map::playlist_finder")
    if dropping.button(f"🗑 Remove the {len(doomed)} ticked track(s)"
                       if doomed else "🗑 Remove the ticked tracks",
                       width="stretch", disabled=not doomed,
                       key="map::playlist_drop"):
        remember_playlist(frame, [i for i in playlist if i not in doomed])
        st.rerun()

    worst = max((c for c in costs if c is not None), default=0)
    st.caption(f"Roughest transition: **{worst:.3f}**. Magic sort is what "
               "brings that number down.")

    # La lavagna guarda la playlist intera, da dovunque i brani siano
    # arrivati: è qui che si vede la forma del set, e non nella sezione che
    # ne scrive un pezzo.
    render_board(frame, at_path, playlist,
                 drop=lambda i: _drop_from_playlist(frame, playlist, i),
                 move=lambda order: _reorder_playlist(frame, order))

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
    #
    # Salvare e non scaricare: il download del browser mette il file nei
    # Download, e una playlist che deve finire sulla chiavetta o nella
    # cartella che rekordbox guarda andava poi spostata a mano. Il pannello
    # del Finder chiede nome e destinazione una volta sola. Il file si
    # costruisce solo a pulsante premuto, che è anche il momento in cui
    # serve.
    if p3.button("⬇ Save as playlist (M3U8)", width="stretch",
                 key="map::save_m3u8",
                 help="What rekordbox's Import Playlist accepts. "
                      "Order and files only — no BPM, no cues."):
        written = save_as(build_m3u8(tracks), "wavecut_playlist.m3u8",
                          "Save the playlist")
        if written:
            st.success(f"Saved: `{written}`")
    if p4.button("⬇ Save as library (rekordbox XML)", width="stretch",
                 key="map::save_xml",
                 help="A library, not a playlist file: load it under "
                      "Preferences ▸ Advanced ▸ Database ▸ rekordbox "
                      "xml. Carries the BPM and the cues."):
        written = save_as(build_rekordbox_xml(tracks), "wavecut_library.xml",
                          "Save the rekordbox library")
        if written:
            st.success(f"Saved: `{written}`")
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

    # Accanto a chi la porta via: sono lo stesso gesto in due direzioni, e la
    # playlist che si ricarica qui è, il più delle volte, l'M3U8 uscito dal
    # pulsante qui sopra. Chiuso, perché con una playlist in piedi caricarne
    # un'altra è l'eccezione.
    with st.expander("📂 Load existing playlist"):
        render_playlist_loader(frame, at_path, playlist)


def graph_seeds(at_path: dict[str, int]) -> list[int]:
    """I brani selezionati sulla mappa, come candidati ad aprire la catena.

    Il Chain Maker sta in fondo alla pagina e chiuso: senza portargli la selezione
    qui sopra, sceglierne il primo brano vuol dire cercarlo per nome davanti a
    una figura che lo sta già mostrando.

    Il seme e il gruppo viaggiano come percorsi e non come posizioni — vale
    per loro la stessa ragione del resto della sessione — quindi vanno
    ritradotti qui. Si leggono dalla sessione e non dal grafico: interrogare
    il grafico da quaggiù vorrebbe dire chiedergli una selezione che il
    ridisegno gli ha già portato via (vedi `SELECTION`).
    """
    selected = [at_path[p] for p in st.session_state.get(SELECTION, [])
                if p in at_path]
    if selected:
        return selected
    seed = at_path.get(st.session_state.get(SEED))
    return [] if seed is None else [seed]


@st.fragment
def render_chain_section(store: MapStore, pool) -> None:
    """Il Chain Maker: costruire un percorso un brano alla volta.

    È un frammento perché ogni gesto — spuntare un candidato, riordinare una
    riga — fa ripartire lo script, e ripartire per intero vuol dire
    ridisegnare la mappa da ottantamila punti per aver spuntato una casella.
    Da frammento si ridisegna solo questa sezione, e il gesto smette di
    aspettare la mappa.

    `pool` arriva dai filtri della pagina: è un altro modo di scegliere fra
    gli stessi brani, non un'altra libreria. Ne aveva di suoi, e voleva dire
    restringere due volte la stessa cosa in due posti diversi.
    """
    if not len(store) or not store.placed:
        return
    placed = store.placed
    frame = pd.DataFrame(store.rows[:placed])
    frame["index"] = np.arange(len(frame))
    cost = TransitionCost(store.coords[:placed], frame["bpm"].tolist(),
                          frame["camelot"].tolist())
    at_path = {row["path"]: i for i, row in enumerate(store.rows[:placed])}
    chosen = [i for i in graph_seeds(at_path) if i < placed]
    render_chain_maker(
        frame, cost, pool, at_path, chosen,
        set_playlist=lambda idxs: remember_playlist(frame, idxs),
        add_to_playlist=lambda idxs: append_playlist(frame, idxs))
    # Il lettore in fondo se lo ridisegna questa sezione, non app.py: un ▶ nelle
    # sue tabelle fa ripartire solo questo frammento, e il dock disegnato
    # fuori resterebbe sul brano di prima. Va chiamata anche nel giro intero,
    # o Streamlit non riserva il posto per le ripartenze.
    fill_dock("graph")


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
                   help="Freezes the job where it is, models still loaded, "
                        "so resuming is instant — but the processes keep "
                        "holding their memory. For a break, not for the "
                        "night."):
        pause_job(state.pid)
        st.rerun()

    if b3.button("⏹ Stop", width="stretch",
                 help="Ends the job and frees the memory. What is already "
                      "on the map stays there, but starting again re-lists "
                      "the folder and reloads the models."):
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
    awake = st.checkbox(
        "Keep the Mac awake until the job is done", value=True,
        help="A sleeping Mac freezes the job: it stays alive without "
             "working. One rebuild lived fifteen hours and analyzed for "
             "three. Sleep is held off only while the job runs. Closing "
             "the lid still sleeps.")
    col_job, col_now = st.columns(2)
    if col_job.button(f"▶ Add all {len(queue):,} in the background",
                      type="primary", width="stretch", disabled=blocked):
        log = DEFAULT_MAP_LOG
        cmd = [sys.executable,
               str(Path(__file__).resolve().parent.parent / "map_cli.py"),
               str(root), "--workers", str(workers), "--project"]
        if awake:
            cmd = caffeinated(cmd)
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

# Tutto quello che riguarda la mappa COME OGGETTO — quanto è grande, come si
# proietta, come la si allarga — sta in un blocco solo. Erano due, uno in
# cima e uno in fondo, e la distanza fra i due non corrispondeva a niente:
# aggiungere brani e riproiettarli sono i due tempi dello stesso gesto, e chi
# aveva appena analizzato una cartella doveva risalire tutta la pagina per
# premere il bottone che rende visibile ciò che aveva appena fatto.
running = job is not None and job.running
with st.expander("⚙️ Map settings" + (" — ▶ job running" if running else ""),
                 expanded=not store.placed or running):
    render_infos(store)
    st.divider()
    st.markdown("#### Add tracks to the map")
    render_add(store, job)

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
    map_context = render_map(store)

st.divider()

# Magic Playlist e Chain Maker: due modi di arrivare a una scaletta, due tab
# di una sezione sola. La scelta fatta sulla mappa qui sopra arriva al primo
# dei due attraverso `map_context`, che è quello che la mappa ha appena
# calcolato — filtri compresi.
render_set_builder(store, map_context)

# Dopo la sezione che la riempie, e fuori da entrambi i tab: la playlist è il
# risultato, e il risultato non sta dentro uno dei due modi di ottenerlo. Si
# disegna da sé solo quando c'è qualcosa dentro.
render_playlist_section(store)


