import pandas as pd

from analysis import mood_scale


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
    return pd.DataFrame([
        {"path": "/a.mp3", "moods": "Dark; Deep"},
        {"path": "/b.mp3", "moods": "Energetic; Melodic"},
        {"path": "/c.mp3", "moods": "Happy; Party"},
    ])


def test_the_board_can_measure_the_mood():
    from views.graph_board import HEIGHT_FIELDS, _measured

    frame = _board()
    at_path = {row["path"]: i for i, row in enumerate(frame.to_dict("records"))}
    assert "mood" in HEIGHT_FIELDS
    measured = _measured(frame, at_path, list(at_path), "mood")
    assert measured["/a.mp3"] < measured["/b.mp3"] < measured["/c.mp3"]


def test_the_darkest_track_sits_lowest_and_the_brightest_highest():
    from views.graph_board import _heights

    frame = _board()
    at_path = {row["path"]: i for i, row in enumerate(frame.to_dict("records"))}
    heights = _heights(frame, at_path, list(at_path), "mood")
    assert heights["/a.mp3"] == 0.0 and heights["/c.mp3"] == 1.0


def test_the_mood_scale_is_read_in_words_not_in_numbers():
    from views.graph_board import _ticks

    frame = _board()
    ticks = _ticks("mood", {"/a.mp3": -1.0}, frame)
    assert [t["label"] for t in ticks] == ["dark", "mid", "bright"]
