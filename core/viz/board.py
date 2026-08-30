"""La lavagna e le tabelle del Chain Maker: i dati, senza i widget.

Il frontend della lavagna (`core/viz/frontend/graph_board`) riceve un JSON —
le schede con altezza, colori e scarti, le tacche della scala — e quel JSON
si costruisce qui, uguale per le due app: Streamlit lo passa al componente,
Qt lo passerà allo stesso HTML dentro un QWebEngineView. Stessa cosa per le
due tabelle del Chain Maker (pandas puro) e per il payload della ruota
Camelot.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.analysis import energy, mood_scale
from core.analysis.mixing import BPM_TOLERANCE, bpm_shift, camelot_shift
from core.viz.track_columns import (KEY_COLORS, OTHER_COLOR, PALETTE,
                                    camelot_color, reading)

# Cosa può dire l'ALTEZZA di una scheda sulla lavagna. L'asse x è già preso
# dall'ordine della scaletta, che non è negoziabile; l'altezza invece è libera
# e può portare la misura che in quel momento racconta il set.
HEIGHT_FIELDS = {"BPM": "bpm", "energy": "energy", "key": "camelot",
                 "groove": "danceability", "mood": "valence_rank"}
# Quella che si apre da sé è il groove, non il BPM: un set si costruisce fra
# brani di tempo vicino — è il senso del costo di transizione — quindi la
# linea dei BPM nasce quasi piatta e non ha molto da dire, mentre la
# regolarità del ritmo sale e scende per tutta la serata. L'ordine delle voci
# resta quello: si cambia cosa si guarda per primo, non dove si clicca.
DEFAULT_HEIGHT = "groove"

# Cosa dice ognuna, in una riga sotto la manopola. Il nome sulla voce dice
# QUALE misura, non cosa significhi alta o bassa — e senza quello la curva è
# una forma senza senso: chi guarda "key" non ha modo di sapere che
# l'altezza è il numero della ruota e non quanto stona.
HEIGHT_MEANING = {
    "BPM": "How fast the track runs. The scale spans ±6% around the middle "
           "of the set — as far as the pitch fader stretches before the "
           "transition costs too much anyway.",
    "energy": "How hard the track pushes: at the top the tenth of your "
              "library that pushes hardest, at the bottom the tenth that "
              "pushes least. It is already a rank, so the scale is the "
              "whole library and nothing has to be stretched.",
    "key": "Where the track sits on the Camelot wheel, 1 to 12. Major or "
           "minor is not in the height — it is in the colour of the key on "
           "the card.",
    "groove": "How regular the beat is: at the top a straight kick, towards "
              "the bottom a syncopated rhythm — breakbeat, funk, anything "
              "not linear. Read on the deciles of your library.",
    "mood": "How dark or bright the track reads COMPARED WITH THE REST of "
            "your library: at the bottom its darkest tenth, at the top its "
            "brightest. Relative because absolute does not work here — the "
            "model reads almost everything as bright, having learned on a "
            "world where 'happy' is a far more common tag than 'sad'.",
}


def _some(row, column: str):
    """Il valore, o `None` se manca davvero.

    Serve perché un campo vuoto arriva qui come NaN, e NaN è vero: scritto
    su una scheda con `or` diventa la parola "nan" sotto al titolo, che
    sembra un dato invece che l'assenza di un dato.

    E una colonna che non c'è affatto è anch'essa un dato che manca, non un
    errore: prima faceva saltare la pagina intera con un `KeyError`, ed è
    successo davvero — una delle tre sezioni costruiva il frame senza le
    colonne calcolate, e salvare una playlist rompeva la lavagna. La causa
    l'ha tolta `map_analysis.library_frame`; questo è il paracadute, perché
    una colonna in meno vale una scheda senza quel numero, non una pagina
    bianca.
    """
    if row is None or column not in row:
        return None
    value = row[column]
    return value if pd.notna(value) and value != "" else None


def _color_map(frame: pd.DataFrame) -> dict[str, str]:
    top = frame["top_genre"].value_counts().head(len(PALETTE)).index.tolist()
    return dict(zip(top, PALETTE))


def _measured(frame: pd.DataFrame, at_path: dict[str, int],
              tracks: list[str], axis: str) -> dict[str, float]:
    """Il valore grezzo della misura scelta, per i brani che ce l'hanno."""
    column = HEIGHT_FIELDS[axis]
    out: dict[str, float] = {}
    for path in tracks:
        value = _some(frame.iloc[at_path[path]] if path in at_path else None,
                      column)
        if value is None:
            continue
        if column == "camelot":
            # Il numero della ruota, non la lettera: è quello che dice di
            # quanto ci si sposta armonicamente.
            code = str(value).strip().upper()[:-1]
            if code.isdigit():
                out[path] = float(code)
        else:
            out[path] = float(value)
    return out


def _span_of(axis: str, values: dict[str, float],
             frame: pd.DataFrame) -> tuple[float, float]:
    """Fra quali due valori tendere l'altezza, per la misura scelta.

    NON sugli estremi della catena. Una catena di otto brani, su questa
    libreria, copre un BPM scarso — è il costo di transizione che fa il suo
    mestiere — e tenderla su sé stessa trasforma un ottavo di BPM in mezza
    lavagna: una salita che non esiste. Peggio, quando i valori sono tutti
    uguali non c'è nessuna scala da tendere e viene fuori una riga piatta
    senza spiegazione.

    Ogni misura ha invece una scala sua, e sempre la stessa: così due catene
    si confrontano, e piatto vuol dire davvero "non si muove".
    """
    if axis in ("energy", "mood"):
        # Gia' ranghi sulla libreria tutti e due: i decili SONO la scala, e
        # tenderla una seconda volta vorrebbe dire prendere il rango di un
        # rango. Per il mood e' un cambio — prima si tendeva sui decili qui,
        # che era lo stesso conto fatto in un altro posto e su un altro
        # numero (la valence dalle parole invece che dai pesi).
        return (0.0, 1.0)
    if axis == "key":
        return (1.0, 12.0)                      # la ruota, tutta
    if axis == "groove":
        return _drive_span(frame)               # i decili della libreria
    # Il tempo attorno a dove sta la catena, largo quanto il pitch fader:
    # oltre il ±6% la transizione costa comunque troppo per capitare.
    middle = sorted(values.values())[len(values) // 2] if values else 120.0
    return (middle * (1 - BPM_TOLERANCE), middle * (1 + BPM_TOLERANCE))


def _heights(frame: pd.DataFrame, at_path: dict[str, int],
             tracks: list[str], axis: str) -> dict[str, float]:
    """Per ogni brano, quanto in alto va: da 0 (in basso) a 1 (in cima)."""
    values = _measured(frame, at_path, tracks, axis)
    if not values:
        return {}
    low, high = _span_of(axis, values, frame)
    if high <= low:
        return {path: 0.5 for path in values}
    return {path: min(1.0, max(0.0, (value - low) / (high - low)))
            for path, value in values.items()}


def reordered(walk: list, moves: dict[int, float]) -> list:
    """L'ordine di una fila dopo che una riga ha cambiato numero.

    Pubblica perche' la usano la lavagna e le tabelle con la colonna "#", in
    tutte e due le app: e' lo stesso gesto — si riscrive il numero di una
    riga e la riga va li' — e due copie della stessa regola scivolerebbero.

    Si toglie il brano da dov'è e lo si rimette dove è stato chiesto, come
    una carta sfilata dal mazzo e reinfilata: gli altri scorrono e nessuno
    sparisce. Scambiare i due brani invece che spostarne uno sarebbe più
    facile da scrivere e sbagliato da usare — chi scrive 1 sull'ultima riga
    vuole quel brano in apertura, non l'apertura in fondo.
    """
    order = list(walk)
    for row, place in sorted(moves.items()):
        if not 0 <= row < len(walk):
            continue
        track = walk[row]
        target = max(1, min(len(order), int(place))) - 1
        order.remove(track)
        order.insert(target, track)
    return order


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
    # L'energia si confronta in GRADINI, non in ranghi: "+2" vuol dire due
    # decili piu' su, che e' come si decide se il set sale. Lo scarto fra
    # 0,71 e 0,89 non dice niente a nessuno.
    here, there = (energy.decile(_some(source, "energy")),
                   energy.decile(_some(row, "energy")))
    if here is not None and there is not None:
        out["energy"] = there - here
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
    if "energy" in gaps:
        out["energy"] = (f"{gaps['energy']:+d}", _way(gaps["energy"]))
    if "dance" in gaps:
        # Senza lo zero davanti: sotto una colonna di trentotto pixel "+0.05"
        # e "+.05" dicono la stessa cosa e solo uno dei due ci sta.
        out["dance"] = (f"{gaps['dance']:+.2f}".replace("0.", "."),
                        _way(gaps["dance"]))
    return out


def _label(name: str) -> str:
    name = Path(name).stem if "/" in name or "\\" in name else name
    return name if len(name) <= 22 else name[:21] + "…"


def _spelled(row, source, common: dict[str, int]) -> dict:
    """Il brano come lo scrivono le due tabelle del Chain Maker: la lettura
    comune a tutta la pagina (`core.viz.track_columns.reading`) più gli
    scarti dalla sorgente.

    Gli scarti sono ciò che queste due tabelle hanno in più delle altre:
    dicono di quanto ci si sposta rispetto al brano da cui si esce, che è la
    domanda del Chain Maker e di nessun altro posto.
    """
    gaps = _gaps(source, row)
    steps = gaps.get("key")
    return {
        **reading(row, common),
        "Δbpm": gaps.get("bpm"),
        "Δenergy": gaps.get("energy"),
        "Δkey": (steps[0] if steps[0] else ("rel" if steps[1] else "="))
        if steps is not None else None,
        "Δgroove": gaps.get("dance"),
    }


def _ticks(axis: str, values: dict[str, float],
           frame: pd.DataFrame) -> list[dict]:
    """Le tacche della scala verticale, dal basso in alto.

    Senza, l'altezza è una forma senza unità: si vede che sale, non da dove a
    dove. Tre bastano — fondo, mezzo, cima — e il numero è quello vero della
    misura, non una percentuale.
    """
    if not values:
        return []
    if axis == "mood":
        # Qui la tacca dice una parola: "+0.28" non vuol dire niente a
        # nessuno, e la scala è una lettura del mood, non la sua misura.
        return [{"at": at, "label": name}
                for at, name in zip((0.0, 0.5, 1.0), ("dark", "mid", "bright"))]
    low, high = _span_of(axis, values, frame)
    if high <= low:
        return []
    digits = 0 if axis in ("key", "BPM") else 2
    return [{"at": at, "label": f"{low + (high - low) * at:.{digits}f}"}
            for at in (0.0, 0.5, 1.0)]


def board_payload(frame: pd.DataFrame, at_path: dict[str, int],
                  paths: list[str], axis: str, common: dict[str, int],
                  dark: bool) -> dict:
    """Il JSON che la lavagna disegna: le schede e le tacche della scala.

    Una scheda per brano, nell'ordine di `paths` (che È la scaletta), con
    l'altezza sulla misura scelta, il colore del genere, i numeri che la
    scheda scrive e gli scarti dal brano precedente NELLA SCALETTA — non da
    quello da cui è stato scelto: è l'ordine in cui il set uscirà, e quindi
    l'unico rispetto a cui "sale" o "scende" vuol dire qualcosa.
    """
    values = _measured(frame, at_path, paths, axis)
    heights = _heights(frame, at_path, paths, axis)
    color_of = _color_map(frame)
    other = OTHER_COLOR["dark" if dark else "light"]
    span = _drive_span(frame)
    before = {path: paths[n - 1] for n, path in enumerate(paths) if n}

    nodes = []
    for position, path in enumerate(paths, start=1):
        row = frame.iloc[at_path[path]] if path in at_path else None
        previous = at_path.get(before.get(path))
        came_from = frame.iloc[previous] if previous is not None else None
        name = row["name"] if row is not None else Path(path).stem
        genre = row["top_genre"] if row is not None else None
        camelot = _some(row, "camelot")
        bpm, dance = _some(row, "bpm"), _some(row, "danceability")
        nodes.append({
            "id": path, "n": position, "height": heights.get(path, 0.5),
            "label": _label(name),
            "color": color_of.get(genre, other),
            "bpm": f"{bpm:.0f}" if bpm is not None else "",
            "camelot": camelot or "",
            "keyColor": camelot_color(camelot),
            "dance": f"{dance:.2f}" if dance is not None else "",
            "drive": _drive(dance, span),
            # I tag per intero, non il solo genere principale: il colore del
            # punto quello lo dice già, e un brano di club è ibrido — sapere
            # che è house E acid house è metà del motivo per cui lo si guarda.
            "genres": row["genres"] if row is not None else "",
            "mood": mood_scale.summary(row["moods"], common)
            if row is not None else "",
            "shift": _card_shifts(came_from, row) if row is not None else {},
        })
    return {"nodes": nodes, "ticks": _ticks(axis, values, frame)}


def chain_table(frame: pd.DataFrame, at_path: dict[str, int],
                walk: list[str], common: dict[str, int]) -> pd.DataFrame:
    """La catena come tabella: numero d'ordine, lettura comune, scarti.

    Ogni brano si confronta con quello che lo precede NELLA SCALETTA, non
    con quello da cui è stato scelto — la stessa regola della lavagna, per
    la stessa ragione.
    """
    before = {track: walk[n - 1] for n, track in enumerate(walk) if n}
    return pd.DataFrame([
        {"#": n + 1,
         **_spelled(frame.iloc[at_path[path]],
                    frame.iloc[at_path[before[path]]]
                    if path in before and before[path] in at_path else None,
                    common),
         "_path": path}
        for n, path in enumerate(walk) if path in at_path])


def roster_table(frame: pd.DataFrame, picks, source,
                 common: dict[str, int]) -> pd.DataFrame:
    """I candidati che escono dalla sorgente, come tabella da spuntare.

    `picks` è quello che torna da `core.analysis.graph_playlist.suggestions`:
    posizione, costo, e le copie dello stesso pezzo — che restano una voce
    sola. Il numero dice quante ce ne sono: si aggiunge la più economica, e
    se ne serve un'altra precisa c'è "Add a track by name".
    """
    return pd.DataFrame([
        {"Add": False, "cost": round(value, 3),
         **_spelled(frame.iloc[i], source, common),
         "copies": len(copies) if len(copies) > 1 else None,
         "_path": frame.at[i, "path"], "_row": i}
        for i, value, copies in picks])


def wheel_payload(selected: list[str], dark: bool) -> dict:
    """Quello che il frontend della ruota Camelot riceve: le tonalità scelte,
    il colore di ognuna (le stesse tinte delle pastiglie), il tema."""
    return {"selected": selected, "colors": KEY_COLORS, "dark": dark}
