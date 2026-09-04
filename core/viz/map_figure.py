"""La figura della mappa (e dei quadranti): punti, anelli, legenda, croce.

È la parte disegnabile della sezione Map, senza la sezione: funzioni che
prendono dataframe e stato e restituiscono una Figure Plotly — la stessa per
Streamlit, che la mette in `st.plotly_chart`, e per Qt, che la renderizzerà
in un QWebEngineView. Qui non si legge nessuna sessione e nessun tema: chi
chiama dice cosa cerchiare e se il tema è scuro.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.viz.track_columns import PALETTE

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
#
# E c'è "none": nessuna chiave per nessuno, quindi nessun genere in
# classifica e tutta la nuvola nel grigio sfumato dell'"altro" — la mappa
# nuda, per quando i colori distraggono da quello che si sta cerchiando.
GENRE_LEVELS = {"macro genre": "parent", "genre": "leaf", "none": "none"}


def genre_level(genre: str, level: str) -> str:
    """L'etichetta al livello scelto. Senza trattino padre e foglia
    coincidono; "none" non ne dà nessuna, ed è come tutto diventa grigio."""
    if level == "none":
        return ""
    text = str(genre or "")
    return text.split(" - ")[0] if level == "parent" else text

# Oltre questi brani in playlist i numeri d'ordine sulla mappa diventano una
# macchia: la linea basta a raccontare il percorso.
NUMBERED_UP_TO = 40

# Cosa può dire la DIMENSIONE del punto. La posizione la decide l'embedding e
# non vuol dire niente di preciso — è affinità, non una grandezza. Il diametro
# invece può portare un numero che si legge: quanto va veloce, quanto è
# uniforme il ritmo, quanto è stato spinto il master. Si legge senza ruotare
# niente, che è il motivo per cui questa è la terza dimensione e non un terzo
# asse.
#
# L'ultima si chiamava "energy" e mostrava `lufs`, che è un'altra cosa: la
# loudness dice quanto ha spinto chi ha masterizzato, non quanto spinge il
# brano — al punto che la pipeline la normalizza via a −14 LUFS prima
# dell'inferenza, proprio per non farsi influenzare. Il nome prometteva la
# misura che stiamo costruendo altrove e ne mostrava una che le assomiglia
# solo nel titolo.
SIZE_FIELDS = {
    "same size": None,
    "BPM": "bpm",
    # L'energia e' gia' un rango sulla libreria, cioe' proprio la scala che
    # `marker_sizes` si costruirebbe da se': i suoi percentili 5-95 sono
    # 0,05 e 0,95, e i diametri escono distribuiti invece che ammassati.
    "energy": "energy",
    "groove": "danceability",
    "loudness": "lufs",
}
# Cosa si puo' mettere sui due assi del grafico a quadranti. Tutto quello
# che e' un numero per brano e che significa qualcosa da solo: la mappa dice
# come un brano SUONA, i quadranti dicono dove sta su due misure scelte.
#
# I quattro grezzi dell'energia ci sono uno per uno oltre che nel voto: sono
# quelli che spiegano PERCHE' un brano legge 8, e guardarli separati e' come
# si scopre che una deep roller ha piu' basso di una peak-time.
AXIS_FIELDS = {
    "energy": "energy",
    # Il RANGO della valence, non il numero firmato: vedi
    # `streamlit_app.views.map_analysis._valence_rank`. Il firmato resta
    # disponibile in fondo, per chi vuole vedere la misura com'e' invece di
    # dov'e'.
    "valence (mood)": "valence_rank",
    "BPM": "bpm",
    "groove": "danceability",
    "loudness": "lufs",
    "length": "duration",
    "mood evidence": "mood_evidence",
    "energy · density": "energy_density",
    "energy · bass": "energy_bass",
    "energy · brightness": "energy_bright",
    "energy · pulse": "energy_pulse",
    "valence · signed": "valence",
}
DEFAULT_AXES = ("valence (mood)", "energy")

# Cosa vuol dire ogni asse, scritto sotto al disegno. Serve perche' un asse
# che si chiama "valence" e va da 0 a 1 non si spiega da se': non dice ne'
# in che unita' sia, ne' — che e' quello che conta — che i due estremi sono
# la TUA libreria e non una scala assoluta.
AXIS_HELP = {
    "energy": "How hard the track pushes, as a rank across your library: "
              "0 is the calmest tenth you own, 1 the hardest. Four measures "
              "in one — attacks per beat, power under 200 Hz, spectral "
              "centre, and how much of the bass lands on the beat. Loudness "
              "is deliberately not in it.",
    "valence (mood)": "How BRIGHT the track reads, as a rank across your "
                      "library: 0 is its darkest tenth — Dark, Deep, Heavy, "
                      "Sad — and 1 its brightest — Happy, Party, Summer, "
                      "Love. A rank and not an absolute value because the "
                      "model has no absolute to give: it learned on a world "
                      "where 'happy' is a far commoner tag than 'sad', so "
                      "it reads 94% of any library as bright. What survives "
                      "that bias is the ORDER, and the rank is the order.",
    "BPM": "Tempo, from the file's tags where it has them.",
    "groove": "How UNIFORM the spacing between attacks is, 0 to 1 — a "
              "metronome reads 1.00, a syncopated figure reads low. Not "
              "groove in the musical sense.",
    "loudness": "Integrated loudness in LUFS: how hard the master was "
                "pushed. It is the control, not a measure of the track — "
                "the pipeline normalises it away before anything else.",
    "length": "Track length in seconds.",
    "mood evidence": "How much colour the model reads at all, whichever way "
                     "it points. Two tracks can both sit at the bright end "
                     "with one shouting it and the other barely whispering "
                     "it: this is what tells them apart.",
    "energy · density": "Attacks per beat — how thick the rhythmic weave "
                        "is. Raw, not ranked.",
    "energy · bass": "What share of the power sits below 200 Hz. Raw.",
    "energy · brightness": "Where the spectral centroid sits, in Hz: closed "
                           "and dark, or open with hats on top. Raw.",
    "energy · pulse": "How deeply the low end pulses ON the beat — a "
                      "straight kick against a syncopated 808. Raw.",
    "valence · signed": "The same dark→bright reading as 'valence (mood)' "
                        "but as the raw signed number, −1 to +1. Useful to "
                        "see the measure as it is; misleading as an axis, "
                        "because its zero is not a middle — 94% of any "
                        "library sits above it.",
}

# Dove passa la croce che divide i quadranti, per le misure che un centro
# vero ce l'hanno. Ce n'e' UNA: l'energia e' un rango sulla libreria, quindi
# il suo mezzo E' la mediana per costruzione e lo sara' sempre.
#
# La valence GREZZA non c'e', e la sua assenza e' una correzione: lo zero di
# una scala firmata sembra un centro e non lo e'. Misurata sulla libreria
# vera, la valence ha il 94% dei brani sopra lo zero — con la croce li' i due
# quadranti bui restavano vuoti, e il grafico diceva che la libreria e' tutta
# allegra. Il suo RANGO invece un mezzo ce l'ha per costruzione, esattamente
# come l'energia, ed e' quello che va sugli assi.
AXIS_CENTRES = {"energy": 0.5, "valence_rank": 0.5}

FLAT_SIZE = 7.0
MIN_SIZE, MAX_SIZE = 4.0, 15.0

# Due fondi e due inchiostri, uno per tema. Il fondo della mappa è LO STESSO
# della pagina, in tutti e due: il riquadro staccato di poco si leggeva come
# una finestra dentro la finestra, e il territorio non ha bordi.
SKIN = {
    "light": {"paper": "#ffffff", "plot": "#ffffff", "ink": "#1b1f27",
              "other": "#9aa4b0", "label": "rgba(27,31,39,0.82)",
              "halo": "rgba(255,255,255,0.75)", "pin": "#1f6fd0",
              "chained": "#f2cc0c", "mixes": "#1f9dd0",
              "alike": "#8a4fd6", "playing": "#d92b2b",
              "pl_selection": "#ff8a1e"},
    "dark": {"paper": "#0e1117", "plot": "#0e1117", "ink": "#eef1f6",
             "other": "#6b7684", "label": "rgba(238,241,246,0.88)",
             "halo": "rgba(14,17,23,0.75)", "pin": "#6fb4ff",
             "chained": "#ffe94d", "mixes": "#5fd0f5",
             "alike": "#c08cff", "playing": "#ff5c5c",
             "pl_selection": "#ffb454"},
}


def axis_guide(values, column: str) -> float | None:
    """Dove passa la riga che divide i quadranti, su un asse.

    Dove la misura un centro vero ce l'ha si mette lì: lo zero della valence
    vuol dire "né buia né chiara", e il mezzo dell'energia È la mediana della
    libreria perché l'energia è un rango. Dove non c'è — i BPM, la loudness,
    la durata — si mette alla mediana di quello che si sta guardando, che è
    l'unica riga che significhi qualcosa: metà dei brani di qua, metà di là.
    Cambia con i filtri, ed è giusto che cambi.
    """
    if column in AXIS_CENTRES:
        return AXIS_CENTRES[column]
    numbers = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(numbers.median()) if len(numbers) else None


def guide_caption(guides: tuple, columns: tuple[str, str],
                  names: tuple[str, str]) -> str:
    """Cosa dice la croce, scritto sotto al disegno.

    Va scritto perché la croce è la sola parte del grafico che non si spiega
    da sé: una riga tratteggiata a metà del disegno sembra un centro
    assoluto, e su quasi tutte le misure è invece la mediana di ciò che i
    filtri lasciano — cioè si sposta appena si tocca un filtro. Un quadrante
    letto come "questi sono i brani veloci" quando dice "questi sono i più
    veloci della metà che stai guardando" è una conclusione sbagliata presa
    con fiducia.
    """
    if any(at is None for at in guides):
        return ""
    where = (f"The cross sits at **{guides[0]:.2f}** across and "
             f"**{guides[1]:.2f}** up.")
    return where + "".join(
        (f" On **{name}** that is the middle of the measure itself."
         if column in AXIS_CENTRES else
         f" On **{name}** it is the median of what the filters leave: half "
         "the tracks on each side, and it moves when they do.")
        for column, name in zip(columns, names))


# Un frame senza righe con la sola colonna che `build_figure` tocca sempre:
# è la "nuvola vuota" con cui si costruisce la figura dei soli contorni.
EMPTY_CLOUD = pd.DataFrame({"genre_key": pd.Series(dtype=object)})


def overlay_figure(coords, marks: dict, dark: bool = False) -> go.Figure:
    """La figura dei SOLI tracciati di contorno: anelli, percorso, seme.

    Serve all'app Qt, che manda la nuvola una volta e a ogni gesto incolla
    questi tracciati in coda (`PlotlyView.set_overlays`): il costo del gesto
    diventa qualche chilobyte invece dei megabyte della figura piena.
    `coords` è dove sta OGNI brano piazzato — gli anelli si disegnano per
    indice, non dal campione — e `marks` sono gli argomenti di contorno di
    `build_figure` (playlist, seed, selected, chained, mixes, alike,
    pl_selection, playing, seed_name).
    """
    return build_figure(EMPTY_CLOUD, [], coords, dark=dark, **marks)


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
                 selected: list[int] | None = None,
                 chained: list[int] | None = None,
                 mixes: list[int] | None = None,
                 alike: list[int] | None = None,
                 pl_selection: list[int] | None = None,
                 playing: int | None = None,
                 axes: tuple[str, str] = ("x", "y"),
                 titles: tuple[str, str] | None = None,
                 guides: tuple[float | None, float | None] | None = None,
                 dark: bool = False,
                 labels: bool = True,
                 ) -> go.Figure:
    """La mappa: un tracciato per genere, più il percorso e il seme sopra.

    Un tracciato per genere e non uno solo con i colori dentro, perché così
    la legenda esiste e ci si può cliccare per spegnere un genere. L'indice
    del brano nella libreria viaggia in `customdata`: è come si risale dal
    punto cliccato alla riga, senza dipendere dall'ordine dei tracciati.

    **Due disegni, una funzione.** `axes` dice quali colonne di `drawn`
    fanno da coordinate e `coords` dove sta OGNI brano della libreria — che
    non è la stessa cosa: i punti si disegnano da `drawn`, che è il campione
    filtrato, mentre gli anelli e il percorso si disegnano per indice, e un
    brano cerchiato può benissimo non essere nel campione. Per la mappa le
    due cose sono la proiezione; per i quadranti sono le due misure scelte.
    Il resto — anelli, percorso, seme, etichette dei generi, la X di chi
    suona — non sa su che assi sta e non deve saperlo: è quello che rende i
    due disegni due modi di guardare la stessa scelta invece che due schermi
    che non si parlano.

    `titles` sono i nomi degli assi (nessuno sulla mappa: le due dimensioni
    della proiezione non hanno un nome che voglia dire qualcosa) e `guides`
    dove passa la croce dei quadranti. `dark` sceglie la pelle: il tema lo
    sa l'app, non questa funzione.
    """
    xcol, ycol = axes
    skin = SKIN["dark" if dark else "light"]
    color_of = dict(zip(top_genres, PALETTE))
    figure = go.Figure()

    for genre in top_genres + ["other"]:
        part = (drawn[~drawn["genre_key"].isin(top_genres)] if genre == "other"
                else drawn[drawn["genre_key"] == genre])
        if not len(part):
            continue
        figure.add_trace(go.Scattergl(
            x=part[xcol], y=part[ycol], mode="markers",
            # Quando i colori sono spenti "other" sarebbe una bugia in
            # legenda: non c'è un "principale" da cui distinguersi.
            name=("tracks" if genre == "other" and not top_genres
                  else genre[:28]),
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
    # l'etichetta in mezzo al nulla. `labels=False` le spegne tutte: dove i
    # gruppi si accavallano le scritte coprono i punti, e chi la mappa la
    # conosce già può volerla nuda.
    for genre in (top_genres if labels else []):
        part = drawn[drawn["genre_key"] == genre]
        if len(part) < 3:
            continue
        # Piu' grande, piu' opaca e su un fondo suo: prima era al 45% di
        # opacita' e spariva dentro il colore dei punti proprio dove i punti
        # sono piu' fitti, cioe' dove l'etichetta serve.
        figure.add_annotation(
            x=float(part[xcol].median()), y=float(part[ycol].median()),
            text=f"<b>{genre.split(' - ')[-1][:22]}</b>", showarrow=False,
            font={"size": 14, "color": skin["label"]},
            bgcolor=skin["halo"], borderpad=3)

    # La croce, prima di tutto il resto perché ci sta SOTTO: una riga che
    # passasse sopra i punti dividerebbe il disegno invece di misurarlo.
    for at, direction in zip(guides or (), ("v", "h")):
        if at is None:
            continue
        figure.add_shape(
            type="line", **({"x0": at, "x1": at, "y0": 0, "y1": 1,
                             "yref": "paper"} if direction == "v"
                            else {"y0": at, "y1": at, "x0": 0, "x1": 1,
                                  "xref": "paper"}),
            line={"color": skin["other"], "width": 1, "dash": "dot"},
            layer="below")

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

    # Giallo per la catena che si sta costruendo, verde per quello che è già
    # in playlist, inchiostro per il gruppo appena preso dalla mappa:
    #
    # C'era anche un anello per le caselle appena spuntate. È stato tolto: la
    # spunta dura il tempo di premere il pulsante accanto, e per cerchiarla in
    # tempo la mappa avrebbe dovuto ridisegnare ottantamila punti a ogni
    # casella — mentre quello che la spunta diventerà, la catena o la
    # playlist, il suo segno ce l'ha già.
    # E c'era un anello verde attorno a ogni brano in playlist. Tolto anche
    # lui: marcava ESATTAMENTE l'insieme che il percorso qui sopra già marca
    # — punti bianchi, linea, numeri — e due segni per lo stesso insieme non
    # dicono di più, affollano. Il doppione è saltato fuori appena l'anello
    # ha avuto la sua voce in legenda, nel parallel run: due voci, un
    # insieme solo. È rimasto il segno che porta più informazione (l'ORDINE,
    # che l'anello non sapeva dire).
    # sulla nuvola la differenza fra "l'ho preso", "lo sto guardando" e "sto
    # lavorando su questi" è proprio quella che serve mentre si sceglie.
    # Diametri diversi, e non per gusto: un brano può stare in due insiemi
    # insieme — appena selezionato E spuntato in playlist — e con lo
    # stesso diametro l'anello disegnato per ultimo coprirebbe l'altro
    # esattamente. Concentrici, si vedono tutti e due.
    # In legenda ci vanno TUTTI: senza, restano cerchi di colori diversi e
    # nessun posto dove chiedere cosa vogliano dire.
    #
    # Compaiono solo quando ci sono davvero: una voce per un insieme vuoto
    # sarebbe una legenda che promette un colore introvabile sul disegno.
    for name, marks, color, size, listed in (
            # Il più stretto per primo, e non è l'ordine dell'elenco: un
            # anello dentro l'altro si vede mentre due sovrapposti no, e il
            # brano in catena è quello che più spesso porta anche un altro
            # segno addosso.
            ("in the chain", chained or [], skin["chained"], 11, True),
            # I due elenchi di proposte stanno FUORI da tutto: sono molti
            # punti, e un alone largo attorno al seme si legge come "guarda
            # da queste parti" invece di litigare con gli anelli di dentro,
            # che dicono cosa un brano E'.
            ("mixes out of it", mixes or [], skin["mixes"], 27, True),
            ("sounds like it", alike or [], skin["alike"], 31, True),
            ("selected", selected or [], skin["ink"], 23, True),
            # Il colore suo, diverso da "selected": quello è la scelta fatta
            # SULLA mappa (lasso, riquadro), questo è la scelta fatta nella
            # tabella della playlist. Il cerchio bianco del seme in alto resta
            # a parte — non è in questo elenco — proprio perché quella scelta
            # non deve confondersi con nessuna delle due.
            ("current PL selection", pl_selection or [],
             skin["pl_selection"], 19, True)):
        spots = [i for i in marks if i is not None and i < len(coords)]
        if not spots:
            continue
        figure.add_trace(go.Scattergl(
            x=coords[spots][:, 0], y=coords[spots][:, 1], mode="markers",
            name=name, showlegend=listed, hoverinfo="skip",
            marker={"size": size, "color": "rgba(0,0,0,0)",
                    "line": {"width": 2.5, "color": color}}))

    # Il brano che sta suonando: un anello che PULSA, non un segno fermo,
    # perché non dice cosa quel brano È — come lo dicono gli altri sei
    # segni — ma cosa sta succedendo adesso, e in mezzo a un nodo di cerchi
    # concentrici è il movimento a distinguerlo. È l'unico tracciato SVG
    # della figura (Scatter, non Scattergl): la pulsazione è un'animazione
    # CSS della pagina (`PlotlyView`), che trova questo punto proprio
    # perché è il solo nel livello SVG — e non costa un ridisegno della
    # nuvola, come costerebbe animare un tracciato gl con un restyle.
    if playing is not None and playing < len(coords):
        figure.add_trace(go.Scatter(
            x=[coords[playing][0]], y=[coords[playing][1]], mode="markers",
            name="playing", showlegend=True, hoverinfo="skip",
            # Sui simboli «-open» il bordo prende `color`, non `line.color`.
            marker={"symbol": "circle-open", "size": 18,
                    "color": skin["playing"], "line": {"width": 2.5}}))

    if seed is not None and seed < len(coords):
        figure.add_trace(go.Scattergl(
            x=[coords[seed][0]], y=[coords[seed][1]], mode="markers",
            # In legenda anche lui, e non fa doppione con "selected": il
            # commento qui sotto dice che i due si escludono, quindi sul
            # disegno ce n'è sempre al massimo uno.
            name="seed", showlegend=True, hoverinfo="skip",
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
        # Sotto al disegno quando sotto non c'è nient'altro, sopra quando
        # gli assi hanno un nome: là sotto ci stanno già i numeri della scala
        # e il nome dell'asse, e la legenda ci finiva addosso — le voci dei
        # generi scritte sopra i "-1" e gli "0,5", con nessuno dei due
        # leggibile.
        legend={"orientation": "h", "x": 0,
                "y": 1.04 if titles else -0.02,
                "yanchor": "bottom" if titles else "auto",
                "font": {"size": 10, "color": skin["ink"]},
                "bgcolor": "rgba(0,0,0,0)", "itemsizing": "constant"},
        # Sulla mappa gli assi non si disegnano: le due dimensioni della
        # proiezione non sono misure, sono il risultato di uno schiacciamento
        # e un numero su di esse non vuol dire niente. Sui quadranti invece
        # gli assi SONO il disegno.
        #
        # E `scaleanchor` solo sulla mappa: là le due dimensioni hanno la
        # stessa unità e stirarne una falserebbe le distanze, cioè l'unica
        # cosa che la mappa dice. Sui quadranti gli assi portano due misure
        # diverse — dei BPM e un rango — e legarli schiaccerebbe il disegno
        # in una riga.
        #
        # `automargin` va acceso insieme ai nomi, o il margine a zero qui
        # sopra taglia via il nome dell'asse orizzontale: il disegno arriva
        # fino al bordo e sotto non resta niente in cui scriverlo. Sulla
        # mappa il margine a zero e' invece giusto — non c'e' niente da
        # scrivere fuori dal riquadro, e ogni pixel e' territorio.
        xaxis={"visible": bool(titles), "title": titles[0] if titles else None,
               "automargin": bool(titles),
               "showgrid": False, "zeroline": False},
        yaxis={"visible": bool(titles), "title": titles[1] if titles else None,
               "automargin": bool(titles),
               "showgrid": False, "zeroline": False,
               "scaleanchor": None if titles else "x"},
    )
    return figure
