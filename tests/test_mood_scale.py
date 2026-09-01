import pandas as pd

from core.analysis import mood_scale


def test_a_dark_mood_sits_at_the_bottom_and_a_bright_one_at_the_top():
    assert mood_scale.valence("Dark") == -1.0
    assert mood_scale.valence("Happy") == 1.0


def test_a_mood_that_says_nothing_about_colour_sits_in_the_middle():
    assert mood_scale.valence("Energetic") == 0.0


def test_no_mood_at_all_is_not_a_zero():
    assert mood_scale.valence("") is None
    assert mood_scale.valence(None) is None


def test_the_first_mood_weighs_more_than_the_second():
    # Dark davanti tira in basso, Happy davanti tira in alto: sono le stesse
    # due etichette, e l'ordine è la confidenza del modello.
    assert mood_scale.valence("Dark; Happy") < 0
    assert mood_scale.valence("Happy; Dark") > 0


def test_a_neutral_mood_dilutes_the_colour():
    assert mood_scale.valence("Dark; Energetic") > mood_scale.valence("Dark")


def test_the_labels_arrive_either_as_the_stored_line_or_already_split():
    assert mood_scale.split("Deep; Summer") == ["Deep", "Summer"]
    assert mood_scale.split(["Deep", "Summer"]) == ["Deep", "Summer"]


def test_the_distinctive_mood_is_the_rarest_one_not_the_strongest():
    common = mood_scale.popularity(["Energetic; Dark", "Energetic", "Energetic"])
    assert mood_scale.distinctive("Energetic; Dark", common) == "Dark"


def test_two_moods_equally_rare_are_broken_by_the_model_order():
    common = mood_scale.popularity(["Deep; Summer"])
    assert mood_scale.distinctive("Deep; Summer", common) == "Deep"


def test_the_summary_puts_the_distinctive_mood_first():
    common = mood_scale.popularity(["Energetic; Deep", "Energetic", "Energetic"])
    assert mood_scale.summary("Energetic; Deep", common) == "Deep · Energetic"


def test_a_single_mood_needs_no_dot():
    assert mood_scale.summary("Deep", {"Deep": 1}) == "Deep"


# --- la lavagna ------------------------------------------------------------

def _board():
    # La lavagna appende le schede al RANGO della valence sulla libreria, non
    # al numero firmato: `views.map_analysis` lo mette sul frame accanto alle
    # parole, che restano quelle che si leggono in tabella.
    return pd.DataFrame([
        {"path": "/a.mp3", "moods": "Dark; Deep", "valence_rank": 0.0},
        {"path": "/b.mp3", "moods": "Energetic; Melodic", "valence_rank": 0.5},
        {"path": "/c.mp3", "moods": "Happy; Party", "valence_rank": 1.0},
    ])


def test_the_board_can_measure_the_mood():
    from core.viz.board import HEIGHT_FIELDS, _measured

    frame = _board()
    at_path = {row["path"]: i for i, row in enumerate(frame.to_dict("records"))}
    assert HEIGHT_FIELDS["mood"] == "valence_rank"
    measured = _measured(frame, at_path, list(at_path), "mood")
    assert measured["/a.mp3"] < measured["/b.mp3"] < measured["/c.mp3"]


def test_the_darkest_track_sits_lowest_and_the_brightest_highest():
    from core.viz.board import _heights

    frame = _board()
    at_path = {row["path"]: i for i, row in enumerate(frame.to_dict("records"))}
    heights = _heights(frame, at_path, list(at_path), "mood")
    assert heights["/a.mp3"] == 0.0 and heights["/c.mp3"] == 1.0


def test_the_mood_scale_is_read_in_words_not_in_numbers():
    from core.viz.board import _ticks

    frame = _board()
    ticks = _ticks("mood", {"/a.mp3": -1.0}, frame)
    assert [t["label"] for t in ticks] == ["dark", "mid", "bright"]


# --- la stessa scala, sui pesi veri del modello ----------------------------

def test_the_weighted_valence_reads_the_activations_not_the_order():
    # Le stesse due etichette nello stesso ordine: a decidere e' quanto il
    # modello ci crede, non chi viene prima.
    assert mood_scale.valence_of({"Dark": 0.60, "Happy": 0.10}) < 0
    assert mood_scale.valence_of({"Dark": 0.10, "Happy": 0.60}) > 0


def test_a_faint_mood_and_a_strong_one_no_longer_read_the_same():
    # E' il difetto che questa funzione esiste per togliere: con le sole
    # etichette i due brani leggevano tutti e due -1,00.
    faint = {"Dark": 0.06, "Energetic": 0.80}
    strong = {"Dark": 0.62, "Energetic": 0.80}
    assert mood_scale.valence_of(faint) == mood_scale.valence_of(strong) == -1.0
    assert mood_scale.evidence(faint) < mood_scale.evidence(strong)


def test_evidence_below_every_threshold_still_counts():
    # Tre prove di buio da 0,04: nessuna etichetta passa la soglia di 0,05 e
    # `valence` non risponde, ma il brano buio lo e'.
    faint = {"Sad": 0.049, "Melancholic": 0.045, "Dark": 0.041}
    assert mood_scale.valence("") is None
    assert mood_scale.valence_of(faint) == -1.0


def test_a_track_with_no_colour_at_all_is_not_a_zero():
    assert mood_scale.valence_of({"Energetic": 0.9, "Melodic": 0.4}) is None
    assert mood_scale.evidence({"Energetic": 0.9}) == 0.0


def test_the_neutral_labels_do_not_move_the_direction():
    # Cambio rispetto a `valence`: con i pesi veri le neutre restano fuori
    # anche dal denominatore. Quanto un brano sia poco colorato lo dice
    # `evidence`, che e' un numero a parte.
    plain = {"Dark": 0.5}
    crowded = {"Dark": 0.5, "Energetic": 0.9, "Melodic": 0.7}
    assert mood_scale.valence_of(plain) == mood_scale.valence_of(crowded)
    assert mood_scale.evidence(plain) == mood_scale.evidence(crowded)


def test_the_activations_survive_a_trip_through_the_stored_line():
    line = mood_scale.spell_weights({"Dark": 0.62, "Happy": 0.05})
    assert line == "Dark:0.620; Happy:0.050"
    assert mood_scale.weights(line) == {"Dark": 0.62, "Happy": 0.05}


def test_a_line_written_by_hand_does_not_break_the_reading():
    assert mood_scale.weights("") == {}
    assert mood_scale.weights("Dark; Happy:0.4") == {"Happy": 0.4}
