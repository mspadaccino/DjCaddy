import pandas as pd

from streamlit_app.views.track_columns import (ENERGY_COLORS, GROOVE_OPTIONS, LEVEL_OPTIONS,
                                 PALETTE, READING_ORDER, camelot_color,
                                 emotion_arrow, energy_level, genre_colors,
                                 groove_pill, reading)


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


def test_the_groove_is_written_the_way_the_card_writes_it():
    # Due decimali come sulla scheda del brano e sulla lavagna: chi legge
    # non deve convertire niente fra una vista e l'altra.
    assert groove_pill(0.83) == "0.83"
    assert groove_pill(0.613) == "0.61"
    assert groove_pill(0.0) == "0.00"
    assert groove_pill(1.0) == "1.00"


def test_the_groove_always_lands_on_one_of_the_column_options():
    # Una stringa fuori elenco non e' un errore: e' una pastiglia che non si
    # colora, e nessuno capisce perche'.
    for value in (0.0, 0.004, 0.615, 0.999, 1.0, 1.2, -0.3):
        assert groove_pill(value) in GROOVE_OPTIONS


def test_a_track_without_a_groove_has_no_pill():
    assert groove_pill(None) is None
    assert groove_pill(float("nan")) is None


def test_the_emotion_is_an_arrow_only_when_the_track_leaves_the_middle():
    # Sul RANGO, dove 0,5 e' la mediana della libreria: sul numero firmato
    # non si poteva, perche' il 94% dei brani sta sopra lo zero e la freccia
    # sarebbe stata in su per quasi tutti.
    assert emotion_arrow(0.9) == "↑"
    assert emotion_arrow(0.1) == "↓"
    # Chi sta in mezzo al mucchio non guarda da nessuna parte, e chi la
    # misura non ce l'ha nemmeno.
    assert emotion_arrow(0.5) is None
    assert emotion_arrow(None) is None
    assert emotion_arrow(float("nan")) is None


def test_a_track_reads_as_pills_and_what_is_missing_stays_empty():
    got = reading(_track(valence_rank=0.12), {"Deep": 3, "Energetic": 900})
    assert got["key"] == ["8A"]
    assert got["groove"] == ["0.83"]
    assert got["emotion"] == ["↓"]
    assert got["genres"] == ["Electronic - House"]
    assert got["BPM"] == 126
    # Il mood distintivo davanti: Energetic sta quasi su tutti e non separa.
    assert got["mood"] == "Deep · Energetic"

    bare = reading(_track(camelot="", danceability=None, moods="", genres="",
                          valence_rank=None), {})
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


def test_every_plain_column_carries_its_explanation():
    """Una colonna che si chiama "sound" o "Δkey" non si spiega da sola, e
    chi la legge non ha nessun posto dove andare a chiedere."""
    from streamlit_app.views.track_columns import COLUMN_HELP, read_only

    for name in COLUMN_HELP:
        assert read_only(name)[name]["help"] == COLUMN_HELP[name]


def test_the_columns_the_tables_actually_ask_for_are_all_covered():
    # I nomi sono quelli che map_analysis e graph_board passano a read_only:
    # se se ne aggiunge uno senza spiegazione, questo test lo trova.
    from streamlit_app.views.track_columns import COLUMN_HELP

    asked = {"#", "file", "BPM", "folder", "cost", "sound", "bpm cost",
             "key cost", "similarity", "copies", "Δbpm", "Δkey", "Δgroove"}
    assert asked <= set(COLUMN_HELP)


# --- l'energia -------------------------------------------------------------

def test_the_energy_pill_is_a_decile_not_a_decimal():
    # Arriva come rango 0..1 e esce da 1 a 10: sono i decili della libreria,
    # e due decimali fingerebbero una precisione che il rango non ha.
    assert energy_level(0.0) == "1"
    assert energy_level(0.55) == "6"
    assert energy_level(1.0) == "10"


def test_a_track_the_backfill_has_not_reached_has_no_energy_pill():
    assert energy_level(None) is None
    assert energy_level(float("nan")) is None


def test_every_energy_pill_falls_on_an_option_of_its_column():
    # Una stringa fuori elenco non e' un errore: e' una pastiglia che non si
    # colora e nessuno capisce perche'.
    written = {energy_level(n / 100) for n in range(101)}
    assert written <= set(LEVEL_OPTIONS)
    assert len(ENERGY_COLORS) == len(LEVEL_OPTIONS)


def test_the_energy_sits_next_to_the_bpm_and_before_the_groove():
    # Quanto va veloce, quanto spinge, quanto e' dritto: si leggono insieme.
    order = READING_ORDER
    assert order.index("BPM") < order.index("energy") < order.index("groove")


def test_the_reading_carries_the_energy_when_the_row_has_it():
    assert reading(_track(energy=0.72), {})["energy"] == ["8"]


def test_a_map_made_before_the_energy_existed_does_not_break_the_reading():
    assert reading(_track(), {})["energy"] == []
