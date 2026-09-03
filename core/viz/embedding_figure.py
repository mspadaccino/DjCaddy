"""L'impronta degli embedding: una riga per brano, i 1280 numeri a colori.

La mappa e i quadranti mostrano DUE numeri per brano — una proiezione o due
misure scelte. Qui si guarda il vettore intero, quello da cui la mappa è
stata schiacciata: ogni riga è un brano, ogni colonna una dimensione (o un
gruppo di dieci), e il colore è quanto quel brano si stacca dagli altri su
quella dimensione. Non si legge un brano alla volta: si leggono le BANDE —
brani che sulla stessa colonna vanno tutti dalla stessa parte suonano
imparentati, ed è la stessa parentela che la mappa disegna come vicinanza.

**Perché un'immagine e non un heatmap.** Le righe sono tutti i brani che
passano i filtri: a ottantasettemila, un heatmap Plotly vorrebbe dire
mandare al browser undici milioni di celle in JSON. L'impronta si disegna
allora in numpy, si codifica in un PNG a tavolozza (un byte per pixel) e
viaggia come `go.Image`: qualche megabyte invece di qualche centinaio. Sopra
ci sta un tracciato di punti trasparenti, uno per riga, che è ciò che rende
la figura interrogabile — passare il mouse a qualunque altezza risponde col
brano di quella riga, e un clic lì lo manda in seme, esattamente come un clic
sulla mappa.

La colonna della distanza sta invece nei CONTORNI (`distance_overlay`): il
seme cambia a ogni clic e l'impronta no, quindi il PNG si rifà solo quando
cambiano i filtri, le colonne o il tema, mentre la colonna si rimanda da
sola in qualche decina di chilobyte. È la stessa divisione fra i due canali
di `PlotlyView` che regge la mappa.
"""

from __future__ import annotations

import base64
import struct
import zlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.viz.map_figure import SKIN

# Quanti PIXEL può valere l'impronta. Oltre, si disegna un campione stabile
# dei brani filtrati — la stessa regola di `MAX_POINTS` sulla mappa, e per lo
# stesso motivo: non è la RAM a cedere, è il PNG che a ottantasettemila righe
# per 1280 colonne sarebbe da trecento megabyte. Il numero è un budget e non
# un numero di righe apposta: chiedere tutte le dimensioni costa righe, e
# così il costo si vede invece di essere una regola nascosta.
MAX_CELLS = 3_000_000

# Quante dimensioni finiscono in una colonna quando si accorpa. Dieci è ciò
# che porta le 1280 di questo modello a 128 colonne esatte: una larghezza che
# sta su uno schermo senza che il browser debba schiacciarla.
GROUP = 10

# Quanto è larga la colonna della distanza, in colonne dell'impronta: una
# frazione e non un numero fisso, o a 1280 colonne sparirebbe.
STRIP_SHARE = 0.05
GAP_SHARE = 0.012

# Il margine a destra, tenuto libero per la barra dei colori della distanza.
RIGHT_MARGIN = 52


def columns_for(dimensions: int, every: bool) -> int:
    """Quante colonne avrà l'impronta: tutte le dimensioni, o a gruppi."""
    return int(dimensions) if every else int(dimensions) // GROUP


def rows_budget(columns: int) -> int:
    """Quante righe si possono disegnare con quelle colonne."""
    return max(1, MAX_CELLS // max(1, int(columns)))


def fingerprint(vectors, every: bool = False) -> np.ndarray:
    """I vettori ridotti a un quadro di valori fra −1 e +1.

    Ogni colonna si centra sulla PROPRIA mediana e si scala sul proprio
    scarto: le 1280 dimensioni hanno medie e ampiezze molto diverse fra loro,
    e disegnarle con una scala sola darebbe un quadro quasi piatto in cui
    tre o quattro dimensioni grandi decidono tutti i colori. Centrata per
    colonna, l'immagine dice l'unica cosa che qui significhi qualcosa: dove
    un brano si stacca dagli altri che stai guardando.

    Lo scarto è il 90° percentile degli scostamenti, non il massimo: un solo
    brano lontanissimo su una dimensione schiaccerebbe tutti gli altri sul
    grigio di mezzo. Chi esce dalla scala si ferma agli estremi.
    """
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or not matrix.size:
        return np.zeros((0, 0), dtype=np.float32)
    if not every:
        columns = matrix.shape[1] // GROUP
        # Le dimensioni che avanzano si lasciano fuori: una colonna fatta di
        # due dimensioni accanto a colonne da dieci non è confrontabile.
        matrix = matrix[:, :columns * GROUP].reshape(
            len(matrix), columns, GROUP).mean(axis=2)
    centre = np.median(matrix, axis=0)
    spread = np.percentile(np.abs(matrix - centre), 90, axis=0)
    return np.clip((matrix - centre) / np.maximum(spread, 1e-9),
                   -1.0, 1.0).astype(np.float32)


def diverging(dark: bool) -> np.ndarray:
    """I 256 colori dell'impronta: blu sotto, il fondo in mezzo, rosso sopra.

    Divergente e non a scala unica perché lo zero qui è un posto vero — è la
    mediana della colonna — e i due lati sono due cose diverse: sotto la
    media e sopra la media. Il colore di mezzo è il fondo della pagina, così
    "come tutti gli altri" si legge come assenza invece che come una tinta.
    """
    low = (58, 122, 196) if dark else (33, 102, 172)
    middle = (14, 17, 23) if dark else (247, 247, 247)
    high = (222, 88, 74) if dark else (196, 46, 42)
    steps = np.linspace(0.0, 1.0, 128)[:, None]
    below = np.asarray(low) + (np.asarray(middle) - np.asarray(low)) * steps
    above = np.asarray(middle) + (np.asarray(high) - np.asarray(middle)) * steps
    return np.vstack([below, above]).round().astype(np.uint8)


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))


def png_bytes(levels: np.ndarray, colors: np.ndarray) -> bytes:
    """Un PNG a tavolozza dai livelli 0–255 e dai loro 256 colori.

    Scritto a mano con `zlib` e `struct` — sono venti righe — invece di
    tirarsi dietro un encoder: l'app si impacchetta con PyInstaller, e una
    dipendenza in più per scrivere l'unico formato che il browser legge
    sempre non si ripaga. A tavolozza e non RGB perché è un byte per pixel
    invece di tre: a parità di peso, tre volte le righe.
    """
    levels = np.ascontiguousarray(levels, dtype=np.uint8)
    height, width = levels.shape
    # Ogni riga di un PNG comincia col suo byte di filtro; 0 vuol dire
    # "nessun filtro", che su un'immagine di rumore è anche il migliore.
    raw = np.hstack([np.zeros((height, 1), dtype=np.uint8), levels]).tobytes()
    return (b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height,
                                          8, 3, 0, 0, 0))
            + _chunk(b"PLTE", np.ascontiguousarray(
                colors, dtype=np.uint8).tobytes())
            + _chunk(b"IDAT", zlib.compress(raw, 6))
            + _chunk(b"IEND", b""))


def fingerprint_source(quadro: np.ndarray, dark: bool) -> str:
    """L'impronta come `data:` URI, pronta per `go.Image(source=…)`."""
    colors = diverging(dark)
    levels = np.rint((np.clip(quadro, -1.0, 1.0) + 1.0) * 127.5)
    return "data:image/png;base64," + base64.b64encode(
        png_bytes(levels, colors)).decode("ascii")


def unit_norms(embeddings) -> np.ndarray:
    """La lunghezza di ogni vettore. Si calcola una volta e si tiene: è
    l'unica parte del coseno che non dipende dal seme."""
    matrix = np.asarray(embeddings, dtype=np.float32)
    if not matrix.size:
        return np.zeros(len(matrix), dtype=np.float32)
    return np.maximum(np.linalg.norm(matrix, axis=1), 1e-9)


def cosine_distances(embeddings, norms, seed_vector) -> np.ndarray:
    """1 − coseno fra ogni riga e il seme: 0 è lo stesso suono, 1 è ortogonale.

    È la distanza VERA nelle 1280 dimensioni, la stessa di `MapStore.similar`
    e delle proposte «sounds like it» — non la distanza sulla mappa, che di
    questa è l'ombra schiacciata. Due brani vicini qui possono finire
    lontani sul disegno, e la colonna serve proprio a vederlo.
    """
    matrix = np.asarray(embeddings, dtype=np.float32)
    seed = np.asarray(seed_vector, dtype=np.float32)
    length = float(np.linalg.norm(seed))
    if not matrix.size or length <= 0:
        return np.zeros(len(matrix), dtype=np.float32)
    # Tagliata a zero: il coseno di un brano con se stesso esce a
    # 0,9999999 e la distanza a −1e−7, che sulla barra dei colori si
    # leggerebbe come un numero negativo.
    return np.clip(1.0 - (matrix @ (seed / length)) / np.asarray(norms),
                   0.0, 2.0).astype(np.float32)


def strip_geometry(columns: int) -> tuple[float, float]:
    """Larghezza della colonna della distanza e dove ne cade il centro."""
    width = max(4.0, columns * STRIP_SHARE)
    gap = max(1.0, columns * GAP_SHARE)
    return width, -(gap + width / 2.0)


def picture_width(columns: int) -> int:
    """Quanto vorrebbe essere larga l'impronta: un pixel per colonna.

    È la larghezza a cui il PNG si vede com'è — un pixel dell'immagine sopra
    un pixel dello schermo, senza che il browser ne schiacci dieci in uno.
    A 128 colonne ci sta in qualunque riquadro; a 1280 no, ed è lì che serve
    poter scorrere di lato invece di guardare 1280 dimensioni spalmate su
    ottocento pixel, dove ogni colonna è mezza colonna di qualcun altro.
    """
    width, centre = strip_geometry(columns)
    return int(round(columns - (centre - width))) + RIGHT_MARGIN


def _hover_points(rows: int, at: float) -> dict:
    """Un punto per riga, all'ascissa `at` e trasparente.

    L'immagine non si può interrogare — un `go.Image` sa dire il pixel, non
    il brano — quindi la riga la si rende cliccabile con dei punti veri che
    portano il loro `customdata`, come ogni punto della mappa. Trasparenti
    perché il disegno è già tutto lì sotto: questi servono al mouse.

    Dove stanno decide CHI risponde: `hoverdistance = −1` fa sì che Plotly
    trovi sempre il punto più vicino, e siccome tutti i punti di una fila
    condividono l'ascissa, il più vicino è quello della riga sotto il mouse.
    Sull'impronta rispondono i punti dell'impronta, sulla colonna della
    distanza quelli della colonna: ogni zona dice la sua cosa.
    """
    return {"x": np.full(rows, at, dtype=np.float32),
            "y": np.arange(rows, dtype=np.float32),
            "mode": "markers",
            "marker": {"size": 6, "color": "rgba(0,0,0,0)"},
            "showlegend": False}


def build_fingerprint_figure(rows: pd.DataFrame, source: str, columns: int,
                             dark: bool = False,
                             room: int | None = None) -> go.Figure:
    """L'impronta più i punti che la rendono interrogabile.

    `rows` sono i brani disegnati NELL'ORDINE delle righe dell'immagine, e
    l'indice di libreria viaggia in `customdata[0]` come su ogni altro
    grafico della pagina: è così che il ponte JS risale dal punto cliccato
    al brano, senza sapere niente di questa figura.

    `room` sono i pixel che il riquadro ha davvero. Quando l'impronta non ci
    sta, la figura si dichiara larga quanto le serve e la pagina scorre di
    lato: schiacciare 1280 colonne in ottocento pixel non è mostrarle, è
    mescolarle. Quando ci sta — 128 colonne ci stanno sempre — la larghezza
    non si scrive, e il disegno resta elastico come gli altri due.
    """
    skin = SKIN["dark" if dark else "light"]
    width, centre = strip_geometry(columns)
    natural = picture_width(columns)
    figure = go.Figure()
    if source:
        figure.add_trace(go.Image(source=source, hoverinfo="skip"))
    figure.add_trace(go.Scattergl(
        customdata=rows[["index", "name", "bpm", "camelot",
                         "genres"]].to_numpy(),
        hovertemplate="<b>%{customdata[1]}</b><br>%{customdata[2]} BPM · "
                      "%{customdata[3]}<br>%{customdata[4]}<extra></extra>",
        name="", **_hover_points(len(rows), columns / 2.0)))
    figure.update_layout(
        # Il margine a destra è tenuto libero SEMPRE, anche senza seme: ci va
        # la barra dei colori della distanza, che arriva coi contorni e non
        # può allargare il riquadro da sé — e un disegno che si stringe al
        # primo clic sarebbe peggio di quaranta pixel non usati.
        height=640, margin={"l": 0, "r": RIGHT_MARGIN, "t": 0, "b": 26},
        width=natural if room is not None and natural > room else None,
        paper_bgcolor=skin["paper"], plot_bgcolor=skin["plot"],
        showlegend=False,
        # Il mouse deve rispondere col brano della RIGA su cui sta, a
        # qualunque altezza dell'immagine. Ci arriva `hoverdistance = −1`,
        # che toglie il raggio entro cui cercare: il punto più vicino c'è
        # sempre, e con una fila di punti tutti alla stessa ascissa il più
        # vicino è quello della riga. Non `hovermode="y"`, che pure lo
        # farebbe: quello è il modo "confronta", e con due tracciati
        # scrive le etichette RUOTATE lungo il fianco del disegno.
        hovermode="closest", hoverdistance=-1,
        hoverlabel={"align": "left", "font": {"size": 11}},
        # Il riquadro non si stira col contenuto: senza un intervallo fisso
        # l'apparire della colonna della distanza sposterebbe l'impronta.
        xaxis={"range": [centre - width, columns], "showgrid": False,
               "zeroline": False, "showspikes": False,
               "tickvals": [centre, columns / 2.0],
               "ticktext": ["distance", "embedding dimensions"],
               "tickfont": {"size": 10, "color": skin["other"]},
               "ticks": "", "showline": False},
        # I pixel dell'immagine NON sono quadrati qui: una riga è un brano e
        # una colonna una dimensione, due unità che non si confrontano. Il
        # vincolo va spento con un `False` esplicito — un `None` Plotly lo
        # legge come "non detto" e ci rimette il suo, che per un'immagine è
        # il lato uguale: l'impronta diventerebbe un filo verticale.
        #
        # E l'intervallo si scrive invece di lasciarlo calcolare: la prima
        # riga in alto, l'ultima in basso, e nessun margine vuoto sopra o
        # sotto il disegno.
        #
        # La riga sotto il mouse si segna da parte a parte: è come si legge
        # la colonna della distanza di QUEL brano, che sta a due palmi di
        # distanza dall'impronta. "across" e non "toaxis": la riga sì, ma
        # senza il numero d'ordine scritto sull'asse, che non vuol dire
        # niente per nessuno.
        yaxis={"visible": False, "scaleanchor": False,
               "range": [len(rows) - 0.5, -0.5],
               "showspikes": True, "spikemode": "across", "spikedash": "dot",
               "spikethickness": 1, "spikecolor": skin["other"]},
    )
    return figure


def distance_overlay(distances, columns: int, places=None,
                     dark: bool = False) -> go.Figure:
    """La colonna della distanza dal seme, da incollare sopra l'impronta.

    Chiara vicino al seme, scura lontano, con la scala scritta accanto: senza
    la barra dei colori una colonna sfumata direbbe l'ORDINE dei brani ma non
    quanto sono distanti, e su una libreria dove tutto sta fra 0,3 e 0,5 è
    proprio quel numero che cambia la lettura.
    """
    figure = go.Figure()
    values = np.asarray(distances, dtype=np.float32) if distances is not None \
        else np.zeros(0, dtype=np.float32)
    if not values.size:
        return figure
    width, centre = strip_geometry(columns)
    figure.add_trace(go.Heatmap(
        z=values.reshape(-1, 1), x0=centre, dx=width, y0=0.0, dy=1.0,
        colorscale="Cividis", reversescale=True, hoverinfo="skip",
        showlegend=False,
        colorbar={"thickness": 9, "len": 0.4, "x": 1.0, "xanchor": "left",
                  "tickfont": {"size": 9},
                  "title": {"text": "dist", "font": {"size": 9}}}))
    # I punti della colonna portano l'INDICE come primi, esattamente come
    # quelli dell'impronta: un clic sulla colonna della distanza fa seme il
    # brano di quella riga, come un clic qualunque sul disegno. Il ponte JS
    # legge `customdata[0]` e non sa — né deve sapere — su quale dei due
    # tracciati ha cliccato.
    places = np.arange(len(values)) if places is None \
        else np.asarray(places, dtype=int)
    figure.add_trace(go.Scattergl(
        customdata=np.column_stack([places, values]),
        hovertemplate="cosine distance from the seed: "
                      "%{customdata[1]:.3f}<extra></extra>",
        name="", **_hover_points(len(values), centre)))
    return figure
