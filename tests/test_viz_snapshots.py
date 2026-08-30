"""La rete di sicurezza della Fase 1: figure e payload, fotografati.

Prima dell'estrazione di `core/viz` questi stessi conti li faceva il codice
dentro `streamlit_app/views`; su questo dataset fisso i suoi risultati sono
stati salvati in `tests/snapshots/` (JSON), da uno script una tantum eseguito
sul codice di PRIMA del refactor. Le funzioni estratte devono riprodurli
identici: qualunque scarto è una regressione dell'estrazione, non una scelta.

Se un giorno una figura cambia APPOSTA, lo snapshot si rigenera a mano
chiamando le funzioni di `core.viz` e salvando il nuovo JSON — insieme alla
modifica che lo cambia, mai per farla passare.

Gli import di `core.viz` stanno dentro ai test e non in cima al modulo: i
costruttori del dataset qui sopra servono anche allo script che ha scritto
gli snapshot, e quello girava quando `core.viz` non esisteva ancora.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.analysis import mood_scale

SNAPSHOTS = Path(__file__).parent / "snapshots"

# --------------------------------------------------------------------------
# Il dataset fisso: nessun caso, nessun file — solo formule deterministiche.
# --------------------------------------------------------------------------

GENRES = ["Electronic - House", "Electronic - Techno", "Funk / Soul - Disco"]
RARE = ["Rock - Prog", "Pop - Italo", "Jazz - Fusion"]
MOODS = ["Energetic; Happy", "Dark; Deep", "Energetic; Party",
         "Sad; Melodic", "Happy; Summer", "Deep; Chill"]
KEYS = [f"{n}{mode}" for n in range(1, 13) for mode in "AB"]


def library(n: int = 24) -> pd.DataFrame:
    """Ventiquattro brani finti con tutte le colonne che le viste leggono.

    Tre generi da sette brani (abbastanza per l'etichetta sulla mappa, che
    vuole almeno tre punti) più tre rari che finiscono nell'"altro".
    """
    rows = []
    for i in range(n):
        top = GENRES[i % 3] if i < n - 3 else RARE[i - (n - 3)]
        rows.append({
            "index": i,
            "name": f"Track {i:02d}.mp3",
            "path": f"/lib/Track {i:02d}.mp3",
            "folder": "/lib/crate",
            "bpm": 110.0 + i,
            "camelot": KEYS[(i * 5) % 24],
            "top_genre": top,
            "genres": top if i % 2 else f"{top}; Electronic - Minimal",
            "moods": MOODS[i % 6],
            "danceability": round(0.35 + 0.025 * i, 3),
            "energy": round(i / (n - 1), 4),
            "valence_rank": round((n - 1 - i) / (n - 1), 4),
            "duration": 180.0 + 5 * i,
        })
    frame = pd.DataFrame(rows)
    frame["x"] = [((i * 7) % n) / 3.0 for i in range(n)]
    frame["y"] = [((i * 11) % n) / 3.0 for i in range(n)]
    return frame


def drawn_library() -> pd.DataFrame:
    """La stessa libreria come la disegna la mappa: genere-chiave e diametro."""
    frame = library()
    return frame.assign(genre_key=frame["top_genre"], _size=7.0)


def library_coords(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([frame["x"].to_numpy(dtype=float),
                            frame["y"].to_numpy(dtype=float)])


def mood_common(frame: pd.DataFrame) -> dict:
    """Quanto è comune ogni mood, calcolato come lo calcola la pagina."""
    return mood_scale.popularity(list(frame["moods"]))


def at_path_of(frame: pd.DataFrame) -> dict[str, int]:
    return {frame.at[i, "path"]: i for i in range(len(frame))}


# I segni sulla mappa: uno per ogni tipo di anello, per esercitarli tutti.
MARKS = {"seed_name": "Track 05.mp3", "selected": [7, 9], "chained": [4, 10],
         "mixes": [11, 13], "alike": [12, 14], "pl_selection": [3],
         "playing": 2}
RICH_PLAYLIST = [0, 4, 8, 2]

# Coppie per gli scarti fra due brani (la seconda riga della scheda).
GAP_ROWS = {
    "a": {"bpm": 120.0, "camelot": "8A", "danceability": 0.52, "energy": 0.25},
    "b": {"bpm": 124.0, "camelot": "9A", "danceability": 0.61, "energy": 0.85},
    "rel": {"bpm": 120.0, "camelot": "8B", "danceability": 0.52,
            "energy": 0.25},
    "poor": {"bpm": 118.0},
}


def as_json(value):
    """Il valore come lo scrive lo snapshot: tuple in liste, numpy in numeri."""
    return json.loads(json.dumps(value, default=float))


def figure_json(figure) -> dict:
    """La figura come JSON, senza `layout.template`.

    Il template non lo mette `build_figure`: glielo appiccica `to_json`
    leggendo il default GLOBALE di plotly, che cambia con l'ambiente —
    importare Streamlit lo sostituisce col suo. Non è opera nostra, e
    tenerlo nel confronto renderebbe l'esito dipendente dall'ordine dei
    test invece che dal codice.
    """
    out = json.loads(figure.to_json())
    out.get("layout", {}).pop("template", None)
    return out


def snapshot(name: str):
    path = SNAPSHOTS / f"{name}.json"
    assert path.exists(), (
        f"Manca {path}: gli snapshot si generano dal codice PRIMA del "
        "refactor (vedi la docstring del modulo).")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# Le figure
# --------------------------------------------------------------------------

def test_the_rich_map_figure_is_identical_to_the_snapshot():
    from core.viz.map_figure import build_figure

    frame = drawn_library()
    figure = build_figure(frame, GENRES, library_coords(frame),
                          playlist=RICH_PLAYLIST, seed=5, **MARKS)
    assert figure_json(figure) == snapshot("map_figure_rich")


def test_the_bare_map_figure_is_identical_to_the_snapshot():
    from core.viz.map_figure import build_figure

    frame = drawn_library()
    figure = build_figure(frame, GENRES, library_coords(frame),
                          playlist=[], seed=None)
    assert figure_json(figure) == snapshot("map_figure_bare")


def test_the_quadrant_figure_is_identical_to_the_snapshot():
    from core.viz.map_figure import build_figure

    frame = drawn_library()
    places = np.column_stack([frame["energy"].to_numpy(dtype=float),
                              frame["valence_rank"].to_numpy(dtype=float)])
    figure = build_figure(frame, GENRES, places, playlist=RICH_PLAYLIST,
                          seed=5, **MARKS,
                          axes=("energy", "valence_rank"),
                          titles=("energy", "valence (mood)"),
                          guides=(0.5, 0.5))
    assert figure_json(figure) == snapshot("quad_figure")


def test_the_marker_sizes_are_identical_to_the_snapshot():
    from core.viz.map_figure import FLAT_SIZE, marker_sizes

    frame = library()
    got = {"bpm": list(marker_sizes(frame, "bpm")),
           "missing": marker_sizes(frame, "nowhere"),
           "flat": marker_sizes(frame, None)}
    assert got["missing"] == got["flat"] == FLAT_SIZE
    assert as_json(got) == snapshot("marker_sizes")


# --------------------------------------------------------------------------
# La lettura del brano e i colori
# --------------------------------------------------------------------------

def test_the_reading_and_the_colours_are_identical_to_the_snapshot():
    from core.viz.track_columns import camelot_color, genre_colors, reading

    frame = library()
    common = mood_common(frame)
    bare = pd.Series({"name": "bare.mp3", "folder": "/lib", "bpm": float("nan"),
                      "camelot": "", "danceability": None, "moods": "",
                      "genres": "", "valence_rank": None})
    got = {
        "rich": reading(frame.iloc[1], common),
        "bright": reading(frame.iloc[4], common),
        "bare": reading(bare, common),
        "genre_colors": genre_colors(
            frame, list(frame["genres"].str.split("; ")), dark=False),
        "genre_colors_dark": genre_colors(
            frame, list(frame["genres"].str.split("; ")), dark=True),
        "camelot_junk": [camelot_color(x) for x in ("", None, "13A", "0B",
                                                    "8", "xx")],
    }
    assert as_json(got) == snapshot("reading")


# --------------------------------------------------------------------------
# Il payload della lavagna, pezzo per pezzo
# --------------------------------------------------------------------------

def test_the_board_pieces_are_identical_to_the_snapshot():
    from core.viz.board import (HEIGHT_FIELDS, _card_shifts, _color_map,
                                _drive, _drive_span, _gaps, _heights, _label,
                                _measured, _span_of, _spelled, _ticks)

    frame = library()
    at_path = at_path_of(frame)
    paths = list(frame["path"])
    common = mood_common(frame)
    rows = {name: pd.Series(values) for name, values in GAP_ROWS.items()}

    axes = {}
    for axis in HEIGHT_FIELDS:
        values = _measured(frame, at_path, paths, axis)
        axes[axis] = {"measured": values,
                      "heights": _heights(frame, at_path, paths, axis),
                      "span": _span_of(axis, values, frame),
                      "ticks": _ticks(axis, values, frame)}

    span = _drive_span(frame)
    got = {
        "axes": axes,
        "drive_span": span,
        "drive": [_drive(v, span) for v in (0.2, 0.5, 0.61, 0.9)],
        "color_map": _color_map(frame),
        "labels": [_label(n) for n in
                   ("short.mp3", "/lib/with a path and a long name.mp3",
                    "A name that runs longer than the card is wide.mp3")],
        "gaps": {name: _gaps(rows["a"], rows[name])
                 for name in ("b", "rel", "a", "poor")},
        "gaps_from_poor": _gaps(rows["poor"], rows["b"]),
        "shifts": {name: _card_shifts(rows["a"], rows[name])
                   for name in ("b", "rel", "a", "poor")},
        "shifts_from_nothing": _card_shifts(None, rows["b"]),
        "spelled": [_spelled(frame.iloc[1], frame.iloc[0], common),
                    _spelled(frame.iloc[0], None, common)],
    }
    assert as_json(got) == snapshot("board_pieces")


# --------------------------------------------------------------------------
# I capitoli
# --------------------------------------------------------------------------

def test_the_chapters_are_identical_to_the_snapshot():
    from core.viz.chapters import (CHAPTERS, assign_chapters,
                                   board_chapter_regions)

    frame = library()
    playlist = list(range(len(frame)))
    assigned = assign_chapters(frame, playlist)

    lookup = {i: ch["name"] for ch, tracks in zip(CHAPTERS, assigned)
              for i in tracks}
    ordered = [i for tracks in assigned for i in tracks]
    # E un buco nel lookup, per fotografare anche la regione che si spezza.
    holed = {i: name for i, name in lookup.items() if i != ordered[2]}

    got = {"assigned": assigned,
           "regions": board_chapter_regions(lookup, ordered),
           "holed_regions": board_chapter_regions(holed, ordered),
           "no_regions": board_chapter_regions(None, ordered)}
    assert as_json(got) == snapshot("chapters")


# --------------------------------------------------------------------------
# Le costanti condivise: tavolozze, assi, testi
# --------------------------------------------------------------------------

def test_the_shared_constants_are_identical_to_the_snapshot():
    from core.viz import board, chapters, map_figure, track_columns

    frame = library()
    got = {
        "PALETTE": track_columns.PALETTE,
        "OTHER_COLOR": track_columns.OTHER_COLOR,
        "LEVELS": track_columns.LEVELS,
        "KEY_OPTIONS": track_columns.KEY_OPTIONS,
        "KEY_COLORS": track_columns.KEY_COLORS,
        "LEVEL_OPTIONS": track_columns.LEVEL_OPTIONS,
        "GROOVE_OPTIONS": track_columns.GROOVE_OPTIONS,
        "GROOVE_COLORS": track_columns.GROOVE_COLORS,
        "ENERGY_COLORS": track_columns.ENERGY_COLORS,
        "EMOTION_OPTIONS": track_columns.EMOTION_OPTIONS,
        "EMOTION_COLORS": track_columns.EMOTION_COLORS,
        "EMOTION_DEADZONE": track_columns.EMOTION_DEADZONE,
        "COLUMN_HELP": track_columns.COLUMN_HELP,
        "READING_ORDER": track_columns.READING_ORDER,
        "SKIN": map_figure.SKIN,
        "SIZE_FIELDS": map_figure.SIZE_FIELDS,
        "AXIS_FIELDS": map_figure.AXIS_FIELDS,
        "AXIS_HELP": map_figure.AXIS_HELP,
        "AXIS_CENTRES": map_figure.AXIS_CENTRES,
        "DEFAULT_AXES": map_figure.DEFAULT_AXES,
        "GENRE_LEVELS": map_figure.GENRE_LEVELS,
        "MAX_POINTS": map_figure.MAX_POINTS,
        "COLORED_GENRES": map_figure.COLORED_GENRES,
        "NUMBERED_UP_TO": map_figure.NUMBERED_UP_TO,
        "SIZES": [map_figure.FLAT_SIZE, map_figure.MIN_SIZE,
                  map_figure.MAX_SIZE],
        "genre_level": [map_figure.genre_level(g, level)
                        for g in ("Electronic - House", "NoDash", None)
                        for level in ("parent", "leaf")],
        "axis_guide": [map_figure.axis_guide(list(frame["bpm"]), "bpm"),
                       map_figure.axis_guide([0.1, 0.2], "energy"),
                       map_figure.axis_guide([], "bpm")],
        "guide_caption": map_figure.guide_caption(
            (124.0, 0.5), ("bpm", "energy"), ("BPM", "energy")),
        "HEIGHT_FIELDS": board.HEIGHT_FIELDS,
        "DEFAULT_HEIGHT": board.DEFAULT_HEIGHT,
        "HEIGHT_MEANING": board.HEIGHT_MEANING,
        "CHAPTERS": chapters.CHAPTERS,
        "CHAPTER_COLORS": chapters.CHAPTER_COLORS,
    }
    assert as_json(got) == snapshot("shared_constants")
