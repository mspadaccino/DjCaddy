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
import streamlit as st

from core.analysis import energy, mood_scale
from core.analysis.duplicates import folded
from core.analysis.dj_export import (build_m3u8, build_rekordbox_xml, read_m3u8,
                                read_title_artist)
from core.analysis.essentia_tags import MODEL_DIR, available, find_taggable, missing_models
from core.analysis.graph_playlist import GraphPlaylist
from core.analysis.map_job import (DEFAULT_MAP_LOG, MAP_CLI_PATH, caffeinated,
                              load_map_state, open_monitor, pause_job,
                              process_state, resume_job, stop_job)
from core.analysis.map_profile import ProfileSettings, default_workers, profile_many
from core.analysis.map_projection import ProjectionSettings
from core.analysis.map_projection import available as umap_available
from core.analysis.map_projection import project
from core.analysis.map_store import MapStore, default_store_dir
from core.analysis.mixing import TransitionCost, magic_sort, nearest
from core.viz.chapters import (CHAPTERS, CHAPTER_COLORS, assign_chapters,
                               board_chapter_regions)
from core.viz.filters import filter_tracks, span
# SKIN, AXIS_CENTRES e FLAT_SIZE non servono più a questa pagina, ma i test
# li hanno sempre letti da qui: restano importabili.
from core.viz.map_figure import (  # noqa: F401
    AXIS_CENTRES, AXIS_FIELDS, AXIS_HELP, COLORED_GENRES, DEFAULT_AXES,
    FLAT_SIZE, GENRE_LEVELS, MAX_POINTS, SIZE_FIELDS, SKIN, axis_guide,
    build_figure, genre_level, guide_caption, marker_sizes)
from streamlit_app.views.components import (NOW_PLAYING, ask_for_file, fill_dock, pick_files,
                              pick_folder, play_table, save_as, tick_all)
from streamlit_app.views.graph_board import (GRAPH_STATE, camelot_picker,
                               mood_popularity, render_board,
                               render_chain_maker, reordered)
from streamlit_app.views.track_columns import (READING_ORDER, dark, read_only,
                                 reading, reading_config)

# In sessione si tengono i PERCORSI, non le posizioni nella libreria. Una
# posizione vale finché la mappa non cambia: basta togliere un brano e la 200
# è un altro brano: la playlist resterebbe in piedi indicando le tracce
# sbagliate, che è peggio di un errore. Il percorso invece è il brano.
SEED = "map::seed"
# Il campo del seme: uno solo, e porta due cose diverse a seconda di cosa ci
# si e' fatto dentro. Un indice se si e' scelto un brano dall'elenco, il testo
# digitato se si sta ancora cercando — `accept_new_options` manda al server
# quello che si scrive, ed e' il tipo del valore a dire quale delle due e'.
SEED_FIELD = "map::seedpick"
SEED_QUERY = "map::seedquery"
# Il file scelto dal Finder ma che sulla mappa non c'è: si ricorda per poterlo
# dire, e si spegne alla scelta buona dopo.
SEED_TROUBLE = "map::seedfile_missing"
PICKED = "map::seedpick_applied"
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

# L'ultimo insieme di spunte sulla colonna Drop della playlist che ha già
# aggiornato il seme: serve a distinguere "ho appena spuntato una riga
# nuova" da "questo giro è ripartito per un altro motivo", o ogni ridisegno
# della pagina rimanderebbe il seme alla playlist invece di lasciarlo a chi
# lo ha scelto per ultimo (mappa, ricerca o Finder).
PLAYLIST_DROP_SEEDSYNC = "map::playlist_drop_seedsync"

# I brani spuntati nella colonna Drop della playlist, come punto di partenza
# per Quick List, Sounds like it e Chain Maker — un canale suo, separato dal
# seme del riquadro in alto (SEED/SELECTION). Il seme in alto è una scelta
# esplicita fatta lì, col suo cerchio bianco; una spunta nella playlist non
# deve spostarla né toccarne il cerchio, altrimenti chi sta guardando "il
# seme" nel riquadro se lo vede cambiare da un gesto fatto altrove.
PLAYLIST_SELECTION = "map::playlist_selection"

# Le chiavi dei due grafici. Sono due viste sugli stessi brani — si sceglie
# da tutte e due, e la scelta è una sola.
MAP_CHART = "map::chart"
QUAD_CHART = "map::quadrants"

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
# Le due schede delle proposte non si aprono da sole: ognuna aspetta il suo
# bottone. In sessione si tiene il PERCORSO del seme per cui la lista e'
# stata chiesta, non un si/no — cosi' cambiando brano la scheda torna
# chiusa da se', senza che chi cambia il seme debba ricordarsi di spegnerla.
# E' la stessa ragione per cui la playlist tiene percorsi e non posizioni.
ASKED_MIXES = "map::asked_mixes"
ASKED_ALIKE = "map::asked_alike"
WAITING_FOR_THE_BUTTON = ("Nothing built yet — press the button above. The "
                          "list does not open by itself: most clicks on the "
                          "map are looking around, not choosing what comes "
                          "next.")

SUGGESTION_DEFAULT = 20
SUGGESTION_MAX = 100
SUGGESTION_STEP = 5


def _spelled(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} hours"


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


@st.cache_data(show_spinner=False)
def _energy_of(_store: MapStore, stamp: tuple) -> np.ndarray:
    """L'energia di tutta la libreria, tenuta da parte fra un gesto e l'altro.

    Non e' un numero per brano: e' il RANGO di ogni brano fra tutti gli
    altri, quindi va rifatto su tutta la libreria ogni volta che la libreria
    cambia — e mai quando si clicca un punto. `stamp` e' lo stato dei file su
    disco, lo stesso con cui si decide di rileggere la mappa: cambia quando
    il job aggiunge brani o quando il backfill scrive le misure, e solo
    allora i ranghi si rifanno.
    """
    return energy.from_rows(_store.rows)


@st.cache_data(show_spinner=False)
def _valence_of(_store: MapStore, stamp: tuple) -> np.ndarray:
    """Il mood come numero per ogni brano, da −1 (buio) a +1 (chiaro).

    La regola — il numero scritto se c'è, le parole se manca — sta in
    `mood_scale.from_rows`: la chiede anche il rapporto sullo stato della
    libreria, e due copie della stessa regola di ripiego un giorno direbbero
    due cose diverse.
    """
    return np.asarray(mood_scale.from_rows(_store.rows), dtype=float)


@st.cache_data(show_spinner=False)
def _valence_rank(_store: MapStore, stamp: tuple) -> np.ndarray:
    """La valence come RANGO sulla libreria, da 0 (la più buia) a 1.

    Misurata sui pesi veri, la valence grezza non è centrata sullo zero e non
    lo sarà mai: sulla libreria vera (2.000 brani) sta fra +0,07 e +0,64 per
    l'80% di mezzo, cioè il 94% dei brani legge "chiaro". Non è la musica —
    è che il modello ha imparato su un mondo in cui *happy*, *positive* e
    *upbeat* sono etichette molto più frequenti di *sad* e *melancholic*, e
    quella frequenza gli è rimasta addosso come prior. Bilanciare le due
    liste sul numero di parole toglie una parte del difetto (l'1,1% sotto
    zero diventa il 6,3%) e non lo toglie tutto, perché il resto non è una
    questione di quante parole ci sono ma di quanto il modello ci creda.

    Il numero grezzo resta quello che sta sulla riga — è la misura, e si
    rifà da `embeddings.f32` quando si vuole. Ma dove la valence fa da
    POSIZIONE — l'asse dei quadranti, l'altezza sulla lavagna, la freccia in
    tabella — si legge il rango, per lo stesso motivo per cui l'energia si
    legge in decili: "più buio del 70% di quello che hai" è una frase vera,
    "valence +0,31" non lo è.
    """
    return energy.ranks(_valence_of(_store, stamp))


def mood_valence(store: MapStore, placed: int) -> np.ndarray:
    """La valence dei brani piazzati, allineata alle righe del frame."""
    return _valence_of(store, _stamp(store.directory))[:placed]


def valence_ranks(store: MapStore, placed: int) -> np.ndarray:
    """Il rango della valence, allineato alle righe del frame."""
    return _valence_rank(store, _stamp(store.directory))[:placed]


def library_frame(store: MapStore, placed: int) -> pd.DataFrame:
    """I brani piazzati come tabella, con addosso le misure derivate.

    Le sezioni della pagina che leggono la libreria sono TRE — la mappa, il
    Chain Maker e la playlist — e ognuna si costruiva il frame per conto suo.
    Finché le colonne erano quelle su disco andava bene; da quando ce ne sono
    di calcolate al momento (l'energia e la valence sono ranghi sulla
    libreria intera, non numeri per brano) voleva dire ricordarsi di
    aggiungerle in tre posti. Ne ho aggiornati due su tre, e la terza sezione
    è andata in errore appena qualcuno ha salvato una playlist.

    Una funzione sola, quindi: chi vuole la libreria la chiede qui e la
    riceve completa. Le colonne che servono a una sezione sola — le
    coordinate della mappa, gli elenchi per i filtri — restano a carico di
    chi le usa.
    """
    frame = pd.DataFrame(store.rows[:placed])
    frame["index"] = np.arange(len(frame))
    frame["energy"] = energy_ranks(store, placed)
    frame["valence_rank"] = valence_ranks(store, placed)
    return frame


def energy_ranks(store: MapStore, placed: int) -> np.ndarray:
    """L'energia dei brani piazzati, allineata alle righe del frame.

    Le due sezioni della pagina che la mostrano — la mappa e il Chain Maker —
    costruiscono ognuna il suo frame dalle stesse righe, e devono ricevere lo
    stesso numero: passa di qui tutte e due le volte.
    """
    return _energy_of(store, _stamp(store.directory))[:placed]


def remember_playlist(frame: pd.DataFrame, indices) -> None:
    st.session_state[PLAYLIST] = [frame.at[i, "path"] for i in indices]
    st.session_state.pop(CHAPTER_STATE, None)


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
    """Il seme, e il campo che lo mostra: i due insieme, sempre.

    Il campo non è solo il posto da cui si sceglie un brano per nome: è anche
    quello dove si LEGGE il seme, comunque sia arrivato. Scriverne uno senza
    l'altro lascia il campo sulla scelta di prima, con il brano nuovo che
    compare solo dentro l'elenco a discesa — che è esattamente cosa
    succedeva a chi cliccava un punto dopo aver scelto per nome.

    La chiave di un widget si può scrivere solo PRIMA che il widget esista.
    Tutte le strade che chiamano questa funzione stanno più in alto del
    campo, e devono restarci.
    """
    st.session_state[SEED] = frame.at[index, "path"]
    st.session_state[SEED_FIELD] = int(index)
    st.session_state["map::livesearch"] = ""


def forget_seed() -> None:
    """Via il seme e via quello che il campo mostrava.

    Serve quando dalla mappa arriva un GRUPPO: seme e gruppo si escludono, e
    un campo rimasto acceso sul brano di prima direbbe che c'è ancora una
    scelta singola quando non c'è più.
    """
    st.session_state.pop(SEED, None)
    st.session_state[SEED_FIELD] = None
    st.session_state["map::livesearch"] = ""


def shared_weights() -> tuple[float, float, float]:
    """I tre pesi di transizione scelti nei settaggi di Quick List.

    Comuni a Quick List, Sounds like it e Chain Maker: sono gli stessi
    slider, letti dalla sessione invece che passati a mano — ogni sezione
    costruisce il proprio `TransitionCost`, ma tutte partono dagli stessi
    pesi.
    """
    return (st.session_state.get("map::w_sound", 1.0),
            st.session_state.get("map::w_bpm", 1.0),
            st.session_state.get("map::w_key", 1.0))


def _picked_on(key: str) -> list[int]:
    """I brani presi su UN grafico, se ce n'è stato uno."""
    state = st.session_state.get(key)
    selection = state.get("selection") if state else None
    if not selection:
        return []
    return [int(p["customdata"][0]) for p in selection.get("points", [])
            if p.get("customdata")]


def read_selection() -> list[int]:
    """I brani selezionati sui grafici in QUESTO giro, se ce ne sono.

    Lo stato di un grafico sta in sessione sotto la sua chiave. I punti dei
    tracciati di servizio (il percorso della playlist, i cerchi) non portano
    `customdata` e vengono scartati: sono disegno, non brani.

    Vuoto non vuol dire "niente selezionato": vuol dire "nessun gesto in
    questo giro", perché al primo ridisegno il widget è un altro (vedi
    `SELECTION`). Chi vuole sapere cosa è scelto lo chiede alla sessione.

    I grafici sono due — la mappa e i quadranti — e si legge da tutti e due.
    Non è un'unione di due scelte diverse: è la stessa identità di widget che
    cambia a ogni ridisegno a garantire che di gesto ce ne sia al massimo uno
    per giro. Il doppione si toglie comunque, perché due volte lo stesso
    brano vorrebbe dire un GRUPPO di due invece di un seme, e la differenza
    fra le due cose decide se il cerchio del seme resta acceso.
    """
    seen: set[int] = set()
    out = []
    for key in (MAP_CHART, QUAD_CHART):
        for index in _picked_on(key):
            if index not in seen:
                seen.add(index)
                out.append(index)
    return out


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
FILTER_GEN = "map::flt_gen"


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
    # Il contatore nella chiave fa rinascere i widget da capo al reset, come
    # tick_all fa con le spunte: togliere il valore dalla sessione non basta,
    # il frontend tiene la sua copia e la rimanda indietro.
    gen = st.session_state.get(FILTER_GEN, 0)
    def _fk(base: str) -> str:
        return f"{base}::{gen}"
    narrowed = any(st.session_state.get(_fk(k)) for k in FILTER_WIDGETS[:2])
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
                key=_fk("map::flt_genres"),
                help="A track carrying any of the chosen genres stays. Tracks "
                     "are multi-label on purpose: 'Minimal' and 'Deep House' "
                     "can both be true of the same track.")
            chosen_moods = st.multiselect(
                "Moods", [m for m, _ in mood_counts.most_common()],
                key=_fk("map::flt_moods"),
                help="Same rule as the genres: a track carrying any of the "
                     "chosen moods stays. Up to four are recorded per track, "
                     "strongest first.")
            tempo = span(frame, "bpm", 60.0, 200.0)
            bpm = st.slider("BPM", tempo[0], tempo[1], tempo,
                            key=_fk("map::flt_bpm"))
            swing = span(frame, "danceability", 0.0, 1.0)
            groove = st.slider("Groove", swing[0], swing[1], swing, step=0.01,
                               key=_fk("map::flt_groove"),
                               help="The danceability: regularity of the "
                                    "onsets, low is loose and high is a "
                                    "straight kick. It is the same number "
                                    "the tables and the board call groove.")
            if st.button("↺ Reset the filters", width="stretch",
                         key="map::flt_reset"):
                st.session_state.pop(FILTER_KEYS, None)
                st.session_state[FILTER_GEN] = gen + 1
                st.rerun()

        kept = filter_tracks(frame, chosen_genres, chosen_moods, keys,
                             bpm, groove)
        st.caption(f"**{len(kept):,}** of {len(frame):,} tracks pass — the "
                   "map, the suggestions and the roster all come from these.")
    return kept


def render_quadrants(frame: pd.DataFrame, drawn: pd.DataFrame,
                     visible: pd.DataFrame, top_genres: list[str],
                     playlist: list[int], seed: int | None,
                     marks: dict, placed: int) -> None:
    """Gli stessi brani su due misure a scelta, invece che sulla proiezione.

    La mappa risponde a "che cosa somiglia a che cosa" e per farlo schiaccia
    milleduecentottanta numeri in due, che poi non vogliono dire niente presi
    uno per uno. Questo risponde a un'altra domanda — "dove sta questo brano
    fra il buio e il chiaro, fra il calmo e lo spinto" — e per farlo prende
    due misure vere e le mette sugli assi. Sono i due assi di Russell,
    valence e arousal, che è il modo in cui l'emozione di un brano si
    descrive da cinquant'anni; l'energia qui è l'arousal, misurata sul
    segnale invece che sulle parole.

    Si disegna lo STESSO campione della mappa: i punti sono gli stessi brani,
    con lo stesso colore e lo stesso diametro, e i cerchi dicono le stesse
    cose. Cambiano solo le due coordinate.
    """
    by_x, by_y = st.columns(2)
    names = list(AXIS_FIELDS)
    x_name = by_x.selectbox(
        "Across", names, index=names.index(DEFAULT_AXES[0]), key="quad::x",
        help="What the horizontal axis measures. Most of these are RANKS "
             "across your library, not absolute values: what they say is "
             "where a track sits among the ones you own. The line under the "
             "chart spells out the two you pick.")
    y_name = by_y.selectbox(
        "Up", names, index=names.index(DEFAULT_AXES[1]), key="quad::y",
        help="What the vertical axis measures. Same reading as the other "
             "axis: mostly a rank across your library.")
    xcol, ycol = AXIS_FIELDS[x_name], AXIS_FIELDS[y_name]

    for column, name in ((xcol, x_name), (ycol, y_name)):
        if column not in frame or not pd.to_numeric(
                frame[column], errors="coerce").notna().any():
            st.info(f"No track carries **{name}** yet. The energy fields "
                    "arrive with the backfill; everything else is measured "
                    "when a track goes on the map.")
            return

    # Gli anelli si disegnano per INDICE, su tutta la libreria piazzata: un
    # brano cerchiato può non essere nel campione, e per la mappa lo stesso
    # servizio lo fa `store.coords`.
    places = np.column_stack([
        pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        for column in (xcol, ycol)])
    guides = (axis_guide(visible[xcol], xcol), axis_guide(visible[ycol], ycol))

    st.plotly_chart(
        build_figure(drawn, top_genres, places, playlist, seed, **marks,
                     axes=(xcol, ycol), titles=(x_name, y_name),
                     guides=guides, dark=dark()),
        key=QUAD_CHART, on_select="rerun",
        selection_mode=("points", "box", "lasso"),
        config={"displaylogo": False, "scrollZoom": True})

    # Prima cosa vogliono dire i due assi, poi dove passa la croce: si legge
    # nell'ordine in cui servono, e senza il primo il secondo non ha senso.
    for name in (x_name, y_name):
        if name in AXIS_HELP:
            st.caption(f"**{name}** — {AXIS_HELP[name]}")
    told = guide_caption(guides, (xcol, ycol), (x_name, y_name))
    if told:
        st.caption(told)


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
    frame = library_frame(store, placed)
    frame["x"], frame["y"] = store.coords[:, 0], store.coords[:, 1]
    # Il numero firmato lo guarda solo la mappa, fra le voci degli assi: le
    # altre due sezioni leggono il rango, che e' quello che si usa come
    # posizione.
    frame["valence"] = mood_valence(store, placed)
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

    w_map, w_bpm, w_key = shared_weights()
    cost = TransitionCost(store.coords, frame["bpm"].tolist(),
                          frame["camelot"].tolist(),
                          w_map=w_map, w_bpm=w_bpm, w_key=w_key)
    # Da percorso a posizione: i brani spariti dalla mappa (tolti, o non più
    # piazzati) semplicemente non si ritrovano, e cadono fuori da soli.
    at_path = {row["path"]: i for i, row in enumerate(store.rows[:placed])}
    playlist = [at_path[p] for p in st.session_state.get(PLAYLIST, [])
                if p in at_path]
    chained = chain_places(at_path)

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
            forget_seed()
        # Un clic sulla mappa è un gesto più recente di qualunque spunta
        # fatta prima nella playlist: quella smette di comandare Quick List,
        # Sounds like it e Chain Maker, e lascia il posto a questo.
        st.session_state.pop(PLAYLIST_SELECTION, None)
        st.session_state.pop(PLAYLIST_DROP_SEEDSYNC, None)
    selected = [at_path[p] for p in st.session_state.get(SELECTION, [])
                if p in at_path]
    seed = None if selected else at_path.get(st.session_state.get(SEED))

    # Quello che si è spuntato nella playlist: un canale a parte da SEED e
    # SELECTION, che non tocca il seme del riquadro in alto (vedi
    # PLAYLIST_SELECTION) ma è quello che conta per le tre schede sotto — se
    # c'è, viene prima del seme in alto, perché è il gesto più recente.
    pl_selected = [at_path[p] for p in st.session_state.get(PLAYLIST_SELECTION, [])
                   if p in at_path]
    if pl_selected:
        op_seed = pl_selected[0] if len(pl_selected) == 1 else None
        op_selected = [] if len(pl_selected) == 1 else pl_selected
    else:
        op_seed, op_selected = seed, selected

    mixes, alike = suggested(store, cost, pool, op_seed, placed)
    playing = at_path.get(st.session_state.get(NOW_PLAYING))

    # Sopra ventimila brani se ne disegna un campione, e il campione può non
    # contenere proprio quello che la pagina sta indicando: il cerchio del
    # seme finiva su una zona vuota, giusto di coordinate e senza il suo
    # punto sotto. Chi è indicato torna dentro comunque — un cerchio attorno
    # al nulla non è un dettaglio estetico, è la mappa che dice il falso.
    if sampled:
        pointed = [i for i in ([seed] if seed is not None else [])
                   + ([playing] if playing is not None else [])
                   + selected + playlist + mixes + alike + pl_selected
                   if i in visible.index and i not in drawn.index]
        if pointed:
            drawn = pd.concat([drawn, visible.loc[pointed]])

    drawn = drawn.assign(
        genre_key=drawn["top_genre"].map(lambda g: genre_level(g, level)))
    ranked = Counter(g for g in drawn["genre_key"] if g)
    top_genres = [g for g, _ in ranked.most_common(COLORED_GENRES)]
    marks = {"seed_name": (frame.at[seed, "name"] if seed is not None else None),
             "selected": selected, "chained": chained,
             "mixes": mixes, "alike": alike, "playing": playing,
             "pl_selection": pl_selected}

    # Due viste sugli stessi brani, non due schermi. La mappa dice come un
    # brano SUONA — è il vicinato a portare il significato, e gli assi non ne
    # hanno uno; i quadranti dicono dove sta su due misure che si scelgono, e
    # lì il significato sta proprio nei numeri. Il seme, il gruppo, la
    # catena, la playlist e chi sta suonando sono gli stessi da tutte e due
    # le parti, e si sceglie indifferentemente di qua o di là.
    on_map, on_axes = st.tabs(["🗺️ Map", "⊞ Quadrants"])
    with on_map:
        st.plotly_chart(
            build_figure(drawn, top_genres, store.coords, playlist, seed,
                         **marks, dark=dark()),
            key=MAP_CHART, on_select="rerun",
            selection_mode=("points", "box", "lasso"),
            config={"displaylogo": False, "scrollZoom": True})
    with on_axes:
        render_quadrants(frame, drawn, visible, top_genres, playlist, seed,
                         marks, placed)

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

    def _browse_seed() -> None:
        chosen = ask_for_file("Choose the seed track")
        if chosen is None:
            return
        on_map, _ = playlist_positions([str(chosen)], at_path)
        if on_map:
            remember_seed(frame, int(on_map[0]))
            st.session_state["map::livesearch"] = ""
            st.session_state[SEED_TROUBLE] = ""
        else:
            st.session_state[SEED_TROUBLE] = chosen.name

    def _clear_seed() -> None:
        st.session_state["map::livesearch"] = ""
        st.session_state.pop(SEED, None)
        st.session_state.pop(PICKED, None)
        st.session_state[SELECTION] = []
        st.session_state.pop(SEED_TROUBLE, None)

    def _add_seed_to_playlist() -> None:
        s = st.session_state.get(SEED)
        if s is not None:
            current = list(st.session_state.get(PLAYLIST, []))
            if s not in current:
                current.append(s)
                st.session_state[PLAYLIST] = current

    live_q = st.session_state.get("map::livesearch", "")
    has_input = seed is not None or bool(live_q)

    listening, adding_seed, searching, clearing, browsing = st.columns(
        [1, 1, 10, 1, 2], vertical_alignment="bottom")
    listening.button(
        "▶", key="map::seedplay", width="stretch", disabled=seed is None,
        on_click=_play,
        args=(frame.at[seed, "path"] if seed is not None else None,),
        help="Hear the seed, in the player at the bottom of the page.")
    adding_seed.button(
        "➕", key="map::seedadd", width="stretch", disabled=seed is None,
        on_click=_add_seed_to_playlist,
        help="Add the seed to the playlist.")
    searching.text_input(
        "Find a track", key="map::livesearch",
        placeholder=("🔍 " + frame.at[seed, "name"]
                     + "  ·  " + Path(frame.at[seed, "folder"]).name
                     if seed is not None
                     else "🔍 type a few words — artist, title, remix — "
                          "and press Enter"),
        label_visibility="collapsed")
    clearing.button(
        "✕", key="map::seedclear", width="stretch", disabled=not has_input,
        on_click=_clear_seed,
        help="Clear the seed and the search.")
    browsing.button("🎵 Browse…", on_click=_browse_seed, width="stretch",
                    key="map::seedbrowse",
                    help="Pick the seed's file from the Finder.")

    live_words = [w for w in live_q.casefold().split() if w] if live_q else []
    if live_words:
        live_found = matching_tracks(frame, pool, live_words)
        st.caption(f"**{len(live_found):,}** match"
                   + (f" — showing the first {SEED_MATCHES_MAX}."
                      if len(live_found) > SEED_MATCHES_MAX else "."))
        if live_found:
            common = mood_popularity(frame)
            live_table = pd.DataFrame([{
                "Pick": "☞",
                **reading(frame.loc[i], common),
                "_path": frame.at[i, "path"],
                "_row": i,
            } for i in live_found[:SEED_MATCHES_MAX]])

            def _on_pick_live() -> None:
                click = st.session_state.get("click::map_livesearch_pick")
                order = st.session_state.get("order::map_livesearch", [])
                if click and 0 <= click.get("row", -1) < len(order):
                    path = order[click["row"]]
                    idx = at_path.get(path)
                    if idx is not None:
                        remember_seed(frame, idx)
                        st.session_state["map::livesearch"] = ""

            play_table(
                "map_livesearch", live_table,
                ["Pick", "file", "BPM", "key", "energy", "groove",
                 "emotion", "folder"],
                {"Pick": st.column_config.ButtonColumn(
                    "☞", on_click=_on_pick_live,
                    key="click::map_livesearch_pick", width="small",
                    help="Use this track as the seed."),
                 **reading_config(frame, live_table)},
                editable=False, editor_key="map_livesearch_editor")
    elif not live_q:
        st.caption(f"{len(visible):,} tracks on the map — type a few words "
                   "above to pick one by name, or click its point.")
    missing_file = st.session_state.get(SEED_TROUBLE)
    if missing_file:
        st.warning(f"`{missing_file}` is not on the map: add its folder under "
                   "**Map settings** at the top, or pick another file.")

    return frame, cost, pool, store, op_seed, op_selected, playlist


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
                          playlist: list[int], shown: int) -> None:
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
        render_mixes_list(frame, cost, pool, seed, playlist, shown)
    else:
        st.info("Nothing selected yet. Click a point on the map to make it "
                "the seed, or drag the lasso or the box around a group.")


def render_set_builder(store: MapStore, context: tuple | None) -> None:
    """I tre modi di costruire un set, in tre tab della stessa sezione.

    Erano due blocchi a scomparsa, uno sotto l'altro: aperti tutti e due
    facevano una colonna in cui né l'uno né l'altro si vedeva per intero,
    chiusi tutti e due non dicevano che erano la stessa domanda — come si
    passa dalla mappa a una scaletta — con più risposte. In tab separate se
    ne vede una alla volta e per intero, e restano affiancate.

    "Sounds like it" era una seconda scheda dentro Quick List: una domanda
    diversa da "cosa ci mixo sopra" (non guarda tempo né tonalità, solo
    affinità acustica) merita il suo posto, non un angolo di un'altra
    modalità. I pesi e "quanti elencare" restano un pannello solo, sopra le
    tre tab: sono gli stessi filtri di partenza per tutte e tre.

    Il Chain Maker c'è comunque, anche quando `context` è `None` — cioè
    quando i filtri non lasciano passare niente — ma con la rosa vuota: una
    catena già in piedi resta da guardare e da mandare alla playlist, e i
    filtri non hanno mai toccato i brani che ci sono già sopra.
    """
    if not len(store) or not store.placed:
        return
    if context is None:
        frame, cost, pool, seed, selected, playlist = None, None, [], None, [], []
    else:
        frame, cost, pool, _, seed, selected, playlist = context

    # Quale delle tre si apre per prima: la scelta appena fatta sulla mappa
    # comanda, perché è il gesto più recente; se non ce n'è nessuna e una
    # catena è già in piedi, si apre quella. È la stessa regola con cui i due
    # blocchi si aprivano da sé, detta una volta sola.
    quicklist_tab = "✨ Quick List"
    alike_tab = "🎯 Sounds like it"
    chain_tab = "🔗 Chain Maker"
    running_chain = bool(st.session_state.get("map::graph"))
    first = chain_tab if running_chain and not (selected or seed is not None) \
        else quicklist_tab

    with st.expander(
            "🎛️ Build a set — from the map to an ordered playlist"
            + (f" · {len(playlist)} track(s) so far" if playlist else ""),
            expanded=bool(selected or seed is not None or playlist
                          or running_chain)):
        shown = SUGGESTION_DEFAULT
        if context is not None:
            shown = render_settings(frame, cost, seed)

        quicklist, alike, chain = st.tabs(
            [quicklist_tab, alike_tab, chain_tab], default=first)
        with quicklist:
            if context is None:
                st.info("No track matches the map filters above — widen them "
                        "to pick a seed or a group to sort.")
            else:
                render_magic_playlist(frame, cost, pool, store, seed,
                                      selected, playlist, shown)
        with alike:
            if context is None or seed is None:
                st.info("Pick a single seed on the map — not a group — to "
                        "see what sounds like it.")
            else:
                render_sounds_alike(frame, store, seed, playlist, shown)
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
    frame = library_frame(store, placed)
    cost = TransitionCost(store.coords[:placed], frame["bpm"].tolist(),
                          frame["camelot"].tolist())
    at_path = {row["path"]: i for i, row in enumerate(store.rows[:placed])}
    playlist = [at_path[p] for p in st.session_state.get(PLAYLIST, [])
                if p in at_path]
    render_playlist(frame, cost, playlist, at_path)


def asked_for(key: str, path: str, label: str, why: str) -> bool:
    """Se la lista di questa scheda e' stata chiesta per QUESTO seme.

    Il bottone sparisce una volta premuto, ed e' voluto: quello che chiedeva
    e' li' sotto, e un bottone che resta acceso davanti a cio' che ha appena
    prodotto invita a premerlo di nuovo per niente. Torna da se' al seme
    dopo.

    La lista, una volta aperta, resta VIVA: si ricalcola a ogni giro insieme
    ai pesi e a quanti brani elencare. Congelarla al momento del clic
    avrebbe voluto dire mostrare le proposte di pesi che non sono piu'
    quelli sugli slider — una lista vecchia scritta con sicurezza, che e'
    peggio di una lista che non c'e'.
    """
    if st.session_state.get(key) == path:
        return True
    if st.button(label, key=f"{key}::ask", type="primary", help=why):
        st.session_state[key] = path
        # E si riparte, invece di disegnare la lista qui sotto e basta. Il
        # bottone sta PIÙ IN BASSO della mappa: nel giro in cui lo si preme
        # la mappa è già disegnata, e i suoi anelli — quelli che cerchiano
        # proprio la lista appena chiesta — arriverebbero al gesto dopo. Una
        # lista sotto e nessun anello sopra è la stessa incoerenza di prima
        # rovesciata, e costa più di un ridisegno.
        st.rerun()
    return False


def render_settings(frame: pd.DataFrame, cost: TransitionCost, seed: int | None) -> int:
    """I pesi di transizione e "quanti elencare" — comuni a Quick List,
    Sounds like it e Chain Maker.

    Erano dentro al seme, e le tre schede se li costruivano ciascuna per
    conto proprio: stessi slider, tre posti diversi. Un pannello solo, sopra
    le tab, e i pesi restano gli stessi filtri di partenza per tutte e tre.
    """
    if seed is not None:
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
    cost.w_map = w1.slider("Weight — sound", 0.0, 2.0, 1.0, 0.1, key="map::w_sound",
                           help="How much the distance on the map counts: "
                                "the acoustic affinity of the two tracks.")
    cost.w_bpm = w2.slider("Weight — BPM", 0.0, 2.0, 1.0, 0.1, key="map::w_bpm",
                           help="How much the tempo gap counts. Beyond ±6% "
                                "the cost climbs fast.")
    cost.w_key = w3.slider("Weight — key", 0.0, 2.0, 1.0, 0.1, key="map::w_key",
                           help="How much harmonic distance counts. Adjacent "
                                "or relative keys (8A→9A, 8A→8B) cost nothing.")

    # Quanti candidati elencare. Uno solo per Quick List e Sounds like it: è
    # la stessa domanda — quanti me ne fai vedere — posta su due criteri
    # diversi, e due manopole scollegate darebbero due liste lunghe diverse
    # senza motivo. Il tetto è la libreria stessa, perché su una mappa appena
    # nata chiedere venti vicini a chi ne ha tre non ha senso.
    room = min(SUGGESTION_MAX, max(1, len(frame) - 1))
    return st.slider("How many to list", SUGGESTION_STEP, room,
                     min(SUGGESTION_DEFAULT, room), SUGGESTION_STEP,
                     key="map_suggestion_count",
                     help="Applies to Quick List and Sounds like it.") \
        if room > SUGGESTION_STEP else room


def render_mixes_list(frame: pd.DataFrame, cost: TransitionCost, pool,
                      seed: int, playlist: list[int], shown: int) -> None:
    """Quick List: cosa ci si mixa sopra questo seme."""
    common = mood_popularity(frame)
    st.caption("Ranked by the transition cost — sound, tempo and key "
               "together, with the weights above. Only tracks that pass "
               "the filters are considered. **The first row is the seed "
               "itself**: where a set starts belongs in it like anything "
               "else, and from here it goes in with one tick.")
    if asked_for(ASKED_MIXES, frame.at[seed, "path"], "✨ Make the list",
                 "Builds the list of what mixes out of this seed."):
        # Il seme in testa e a costo zero, che è la verità: da sé a sé
        # non c'è transizione. Prima non compariva in nessuna delle due
        # liste e la playlist si popolava solo dei suoi simili — si
        # partiva da un brano che poi nel set non c'era.
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
                ["Add", "cost", "file", "BPM", "key", "energy", "groove",
                 "emotion", "sound", "bpm cost", "key cost", "mood", "genres",
                 "folder"],
                {"Add": st.column_config.CheckboxColumn(
                    "Add", help="Tick what you want in the playlist, then "
                                "the button below."),
                 **reading_config(frame, table),
                 **read_only("cost", "sound", "bpm cost", "key cost")},
                editor_key=mix_key)
            wanted = [int(i) for i in edited.loc[edited["Add"], "_row"]]
            if st.button(f"➕ Add {len(wanted)} to the playlist",
                         disabled=not wanted, type="primary"):
                remember_playlist(frame, playlist + [i for i in wanted
                                                     if i not in playlist])
                st.rerun()
    else:
        st.caption(WAITING_FOR_THE_BUTTON)


def render_sounds_alike(frame: pd.DataFrame, store: MapStore, seed: int,
                        playlist: list[int], shown: int) -> None:
    """Sounds like it: pura affinità acustica, non mixabilità."""
    common = mood_popularity(frame)
    st.caption(
        "Pure acoustic closeness, measured in the 1280 dimensions of the "
        "embedding — not on the flattened map, and with no regard for "
        "tempo or key. This is 'what else sounds like this', which is a "
        "different question from 'what mixes out of this'. The first row "
        "is the seed itself, here too.")
    if asked_for(ASKED_ALIKE, frame.at[seed, "path"], "✨ Make the list",
                 "Builds the list of what sounds like this seed."):
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
    else:
        st.caption(WAITING_FOR_THE_BUTTON)


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


# ---------------------------------------------------------------------------
# Chapter Builder
# ---------------------------------------------------------------------------

CHAPTER_STATE = "map::chapters"

def _chapter_of(playlist: list[int]) -> dict[int, str] | None:
    """Map track index → chapter name from session state, or None."""
    chapters = st.session_state.get(CHAPTER_STATE)
    if chapters is None:
        return None
    lookup = {}
    for ch, ch_tracks in zip(CHAPTERS, chapters):
        for i in ch_tracks:
            lookup[i] = ch["name"]
    return lookup if set(sum(chapters, [])) == set(playlist) else None


def _move_between_chapters(frame: pd.DataFrame, track_idx: int,
                           src_name: str, dst_name: str) -> None:
    """Move a track from one chapter to another via board drag."""
    chapters = st.session_state.get(CHAPTER_STATE)
    if chapters is None:
        return
    src_ci = next((i for i, ch in enumerate(CHAPTERS)
                   if ch["name"] == src_name), None)
    dst_ci = next((i for i, ch in enumerate(CHAPTERS)
                   if ch["name"] == dst_name), None)
    if src_ci is None or dst_ci is None:
        return
    if track_idx in chapters[src_ci]:
        chapters[src_ci].remove(track_idx)
        chapters[dst_ci].append(track_idx)
        ordered = sum(chapters, [])
        remember_playlist(frame, ordered)
        st.session_state[CHAPTER_STATE] = chapters
    st.rerun()


def render_chapter_builder(frame: pd.DataFrame, playlist: list[int]) -> None:
    if len(playlist) < 5:
        st.caption("Need at least 5 tracks in the playlist to build chapters.")
        return

    playlist_set = set(playlist)
    cached = st.session_state.get(CHAPTER_STATE)
    has_chapters = cached is not None and set(sum(cached, [])) == playlist_set

    if not has_chapters:
        st.caption("Distribute the playlist across five emotional chapters "
                   "of a DJ set: Intro, Buildup, Tension, Climax, Release.")
        if st.button("📖 Create chapters", type="primary",
                     key="map::chapter_create"):
            st.session_state[CHAPTER_STATE] = assign_chapters(frame, playlist)
            st.rerun()
        return

    chapters = cached
    changed = False
    for ci, (ch, ch_tracks) in enumerate(zip(CHAPTERS, chapters)):
        label = f"{ch['icon']} {ch['name']} ({len(ch_tracks)})"
        st.markdown(f"**{label}**")
        if not ch_tracks:
            st.caption("No tracks assigned.")
            continue

        common = mood_popularity(frame)
        table = pd.DataFrame([{
            "#": pos + 1,
            **reading(frame.loc[i], common),
            "_path": frame.at[i, "path"],
            "_row": i,
        } for pos, i in enumerate(ch_tracks)])

        editor_key = f"map_chapter_{ci}::" + "|".join(
            str(i) for i in ch_tracks)
        play_table(
            f"map_chapter_{ci}", table,
            ["#", "file", "BPM", "key", "energy", "groove", "emotion"],
            {"#": st.column_config.NumberColumn(
                "#", min_value=1, max_value=len(ch_tracks), step=1,
                help="Write the position you want this track in: the row "
                     "moves there and the others slide."),
             **reading_config(frame, table)},
            editor_key=editor_key)

        moves = {int(row): values["#"]
                 for row, values in st.session_state.get(editor_key, {})
                 .get("edited_rows", {}).items() if "#" in values}
        if moves:
            new_order = reordered(ch_tracks, moves)
            if new_order != ch_tracks:
                chapters[ci] = new_order
                changed = True

    if changed:
        st.session_state[CHAPTER_STATE] = chapters
        st.rerun()

    c1, c2 = st.columns(2)
    if c1.button("📖 Apply chapter order to playlist", type="primary",
                 key="map::chapter_apply"):
        ordered = sum(chapters, [])
        remember_playlist(frame, ordered)
        # remember_playlist svuota sempre CHAPTER_STATE, perché di norma una
        # playlist riscritta non è più quella che i capitoli descrivono. Qui
        # lo è: l'ordine appena scritto è `chapters` stesso, srotolato — le
        # aree colorate sulla lavagna sparirebbero altrimenti proprio nel
        # momento in cui l'accordo fra playlist e capitoli è più vero che mai.
        st.session_state[CHAPTER_STATE] = chapters
        st.rerun()
    if c2.button("🔄 Re-assign chapters", key="map::chapter_reassign"):
        st.session_state[CHAPTER_STATE] = assign_chapters(frame, playlist)
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
    ch_lookup = _chapter_of(playlist)
    table = pd.DataFrame([{
        "#": position + 1,
        "Drop": False,
        **reading(frame.loc[i], common),
        "from previous": round(step, 3) if step is not None else None,
        **({"chapter": ch_lookup[i]} if ch_lookup and i in ch_lookup else {}),
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
    tick_left, tick_right = st.columns([4, 2])
    drop_all, editor_key = tick_all(
        "map_playlist_editor::" + "|".join(table["_path"]), default=False,
        into=tick_left)
    if tick_right.button("🗑 Reset playlist", width="stretch",
                         key="map::playlist_reset",
                         help="Clear the entire playlist."):
        st.session_state[PLAYLIST] = []
        st.session_state.pop(CHAPTER_STATE, None)
        st.rerun()
    table["Drop"] = drop_all
    col_order = ["#", "Drop", "file", "BPM", "key", "energy", "groove",
                  "emotion", "from previous", "mood", "genres", "folder"]
    col_config: dict = {
        "#": st.column_config.NumberColumn(
            "#", min_value=1, max_value=len(playlist), step=1,
            help="Write the position you want this track in: the row moves "
                 "there and the others slide."),
        "Drop": st.column_config.CheckboxColumn(
            "Drop", help="Tick what you want out, then the button below."),
        "from previous": st.column_config.NumberColumn(
            "from previous", disabled=True,
            help="The transition cost from the track above: 0 is "
                 "seamless, 1 is as far as this library goes."),
        **reading_config(frame, table),
    }
    if ch_lookup:
        col_order.insert(2, "chapter")
        ch_names = [ch["name"] for ch in CHAPTERS]
        ch_colors = [CHAPTER_COLORS[n] for n in ch_names]
        col_config["chapter"] = st.column_config.MultiselectColumn(
            "chapter", disabled=True, width="small",
            options=ch_names, color=ch_colors,
            help="Which chapter of the DJ set this track belongs to.")
        table["chapter"] = table["chapter"].apply(
            lambda v: [v] if pd.notna(v) else [])
    edited = play_table(
        "map_playlist", table, col_order, col_config,
        editor_key=editor_key)

    # La colonna Drop non serve solo a togliere: quello che ci si spunta
    # diventa anche il punto di partenza per Chain Maker e Quick List, come
    # la stessa scelta fatta sulla mappa — ma per un canale suo
    # (PLAYLIST_SELECTION), non il seme del riquadro in alto: quello resta
    # quello che era, col suo cerchio bianco, e qui il brano si cerchia di un
    # altro colore ("current PL selection"). Solo sui cambiamenti veri:
    # rifarlo a ogni giro rimanderebbe indietro la scelta fatta nel frattempo
    # altrove.
    ticked_paths = tuple(sorted(edited.loc[edited["Drop"], "_path"]))
    if ticked_paths and ticked_paths != st.session_state.get(PLAYLIST_DROP_SEEDSYNC):
        st.session_state[PLAYLIST_DROP_SEEDSYNC] = ticked_paths
        st.session_state[PLAYLIST_SELECTION] = list(ticked_paths)
        st.rerun()
    elif not ticked_paths:
        st.session_state.pop(PLAYLIST_DROP_SEEDSYNC, None)
        st.session_state.pop(PLAYLIST_SELECTION, None)

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

    with st.expander("📖 Chapter Builder — reorganise this playlist into "
                      "a five-chapter DJ set arc"):
        render_chapter_builder(frame, playlist)

    # La lavagna guarda la playlist intera, da dovunque i brani siano
    # arrivati: è qui che si vede la forma del set, e non nella sezione che
    # ne scrive un pezzo.
    board_chapters = board_chapter_regions(ch_lookup, playlist)
    render_board(frame, at_path, playlist,
                 drop=lambda i: _drop_from_playlist(frame, playlist, i),
                 move=lambda order: _reorder_playlist(frame, order),
                 chapters=board_chapters,
                 chapter_move=lambda idx, src, dst:
                     _move_between_chapters(frame, idx, src, dst))

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


def suggested(store: MapStore, cost: TransitionCost, pool, seed: int | None,
              placed: int) -> tuple[list[int], list[int]]:
    """Le due liste di proposte del seme, per cerchiarle sulla mappa — se
    sono state chieste.

    Si calcolano QUI, prima del disegno, e non nel pannello che poi le
    elenca: la mappa sta più in alto nella pagina, e prendere le liste dal
    giro precedente vorrebbe dire cerchiare le proposte del seme di PRIMA.
    Non un ritardo — un'informazione sbagliata, e proprio nel momento in cui
    si è appena cambiato brano.

    Si pagano due volte, una qui e una nel pannello. Rifarle costa una scorsa
    sulla libreria; passarsele avrebbe voluto dire allungare la tupla che
    questa pagina si porta dietro fin dentro le sue schede, per risparmiare
    qualche decina di millisecondi su un gesto che ne dura centinaia.

    I pesi si leggono dalla sessione e non dagli slider: quelli stanno nel
    pannello, cioè più in basso del disegno. Hanno una chiave apposta.
    """
    if seed is None:
        return [], []
    # Cerchiate solo le liste che sono state CHIESTE. Il bottone che le apre
    # sta piu' in basso nella pagina; finche' non lo si preme la scheda dice
    # "premi il bottone", e degli anelli attorno a venti punti dicevano che
    # una scelta era stata fatta mentre sotto non c'era niente. Peggio: erano
    # gli anelli di una lista che nessuno aveva visto.
    #
    # Cosi' le due scorse della libreria si pagano quando servono e non a
    # ogni clic sulla mappa, che era il motivo per cui il bottone esiste.
    path = store.rows[seed]["path"]
    if st.session_state.get(ASKED_MIXES) != path \
            and st.session_state.get(ASKED_ALIKE) != path:
        return [], []
    cost.w_map = st.session_state.get("map::w_sound", 1.0)
    cost.w_bpm = st.session_state.get("map::w_bpm", 1.0)
    cost.w_key = st.session_state.get("map::w_key", 1.0)
    shown = st.session_state.get("map_suggestion_count", SUGGESTION_DEFAULT)
    return ([i for i, _ in nearest(cost, seed, k=shown, pool=pool)]
            if st.session_state.get(ASKED_MIXES) == path else [],
            [i for i, _ in store.similar(seed, k=shown, limit=placed)]
            if st.session_state.get(ASKED_ALIKE) == path else [])


def chain_places(at_path: dict[str, int]) -> list[int]:
    """I brani già nella catena, come posizioni sulla mappa.

    La catena si costruiva alla cieca: il Chain Maker sta sotto la mappa e
    dice quali brani ne fanno parte, ma sulla nuvola quei brani non si
    distinguevano da tutti gli altri — e la nuvola è il posto in cui si
    guarda per decidere il prossimo. Adesso portano il loro anello, come la
    playlist porta il suo.

    Nell'ordine della scaletta e non in quello di inserimento, per la stessa
    ragione per cui gli anelli hanno diametri diversi: se un giorno la catena
    si disegnasse anche come linea, la linea è quella.

    I percorsi che sulla mappa non ci sono più — un brano tolto, o non ancora
    piazzato — cadono fuori da soli invece di far esplodere il disegno.
    """
    graph = GraphPlaylist.from_state(st.session_state.get(GRAPH_STATE))
    return [at_path[path] for path in graph.walk() if path in at_path]


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

    Una spunta appena fatta nella playlist viene prima di tutto il resto: è
    il gesto più recente, e non passa dal seme del riquadro in alto — quello
    resta quello che era.
    """
    pl_selected = [at_path[p] for p in st.session_state.get(PLAYLIST_SELECTION, [])
                   if p in at_path]
    if pl_selected:
        return pl_selected
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
    frame = library_frame(store, placed)
    w_map, w_bpm, w_key = shared_weights()
    cost = TransitionCost(store.coords[:placed], frame["bpm"].tolist(),
                          frame["camelot"].tolist(),
                          w_map=w_map, w_bpm=w_bpm, w_key=w_key)
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
        cmd = [sys.executable, str(MAP_CLI_PATH),
               str(root), "--workers", str(workers), "--project"]
        if awake:
            cmd = caffeinated(cmd)
        with open(log, "w") as out:
            subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                             start_new_session=True,
                             cwd=MAP_CLI_PATH.parent)
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
_has_map = bool(store.placed)
_has_seed = SEED in st.session_state
_has_playlist = bool(st.session_state.get(PLAYLIST))
with st.expander(
        "🗺️ Map"
        + (" · seed selected" if _has_seed else "")
        + (f" · {len(st.session_state.get(PLAYLIST, []))} in playlist"
           if _has_playlist else ""),
        expanded=True):
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


