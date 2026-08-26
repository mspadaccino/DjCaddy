import pandas as pd

from views.track_columns import (PALETTE, camelot_color, emotion_arrow,
                                 genre_colors, groove_level, reading)


def _track(**changes) -> pd.Series:
    row = {"name": "A.mp3", "bpm": 126.4, "camelot": "8A", "danceability": 0.83,
           "moods": "Deep; Energetic", "genres": "Electronic - House",
           "folder": "/DJSet", "top_genre": "Electronic - House"}
    return pd.Series({**row, **changes})


def test_the_key_carries_the_colour_it_has_on_the_wheel():
    # Maggiore e minore dello stesso numero sono la stessa tinta a due
    # luminosita': e' cosi' che si vede a colpo d'occhio che si mixano.
    assert camelot_color("8A") != camelot_color("8B")
    assert camelot_color("8A") == camelot_color("8a")
    # Quello che tonalita' non e' non prende un colore a caso.
    assert camelot_color("") == camelot_color(None) == camelot_color("13A")


def test_the_groove_lands_on_ten_steps_and_never_on_a_zero():
    assert groove_level(0.83) == 8
    assert groove_level(1.0) == 10
    # Uno a fondo scala e non zero: la pastiglia porta un numero da leggere,
    # e uno zero si scambia per un dato che manca.
    assert groove_level(0.0) == 1
    assert groove_level(None) is None
    assert groove_level(float("nan")) is None


def test_the_emotion_is_an_arrow_only_when_the_track_has_a_way_to_look():
    assert emotion_arrow("Happy") == "↑"
    assert emotion_arrow("Dark") == "↓"
    # Un mood che di colore non dice niente non muove nessuna freccia, e
    # nemmeno l'assenza di mood.
    assert emotion_arrow("Energetic") is None
    assert emotion_arrow("") is None
    assert emotion_arrow(None) is None


def test_a_track_reads_as_pills_and_what_is_missing_stays_empty():
    got = reading(_track(), {"Deep": 3, "Energetic": 900})
    assert got["key"] == ["8A"]
    assert got["groove"] == ["8"]
    assert got["emotion"] == ["↓"]
    assert got["genres"] == ["Electronic - House"]
    assert got["BPM"] == 126
    # Il mood distintivo davanti: Energetic sta quasi su tutti e non separa.
    assert got["mood"] == "Deep · Energetic"

    bare = reading(_track(camelot="", danceability=None, moods="", genres=""),
                   {})
    # Nessuna pastiglia, e non una pastiglia vuota: una lista vuota non si
    # disegna, una lista con dentro il nulla scriverebbe "None" o "nan".
    assert bare["key"] == bare["groove"] == bare["emotion"] == []
    assert bare["genres"] == []


def test_the_frequent_genres_get_a_colour_and_the_long_tail_gets_grey():
    frame = pd.DataFrame([{"genres": "Electronic - House", "top_genre": "Electronic - House"}] * 3
                         + [{"genres": "Funk / Soul - Disco", "top_genre": "Funk / Soul - Disco"}])
    colors = genre_colors(frame, [["Electronic - House"], ["Funk / Soul - Disco"]])
    assert colors["Electronic - House"] == PALETTE[0]
    assert colors["Funk / Soul - Disco"] == PALETTE[1]

    # Oltre la tavolozza si finisce nel grigio dell'"altro", che e' la stessa
    # sorte che si ha sulla mappa.
    crowded = pd.DataFrame([{"genres": f"G{n}", "top_genre": f"G{n}"}
                            for n in range(len(PALETTE) + 3)])
    many = genre_colors(crowded, [[f"G{n}"] for n in range(len(PALETTE) + 3)])
    assert many["G0"] in PALETTE
    assert many[f"G{len(PALETTE) + 2}"] not in PALETTE


def test_a_genre_that_only_the_shown_rows_carry_is_still_named():
    # Chi non entra nel vocabolario non viene disegnato come etichetta: la
    # pastiglia sparisce e il nome ricompare per esteso in mezzo alle altre.
    frame = pd.DataFrame([{"genres": "Electronic - House",
                           "top_genre": "Electronic - House"}])
    colors = genre_colors(frame, [["Electronic - House", "Rock - Prog"]])
    assert "Rock - Prog" in colors
