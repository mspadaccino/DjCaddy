"""La vista dello scaffale: una riga per playlist, i numeri della serata."""

import numpy as np
import pandas as pd

from core.viz.shelf_view import (energy_bar, length_told, shared_tracks,
                                 shelf_rows, shelf_summary)


def _frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"name": "a.mp3", "path": "/x/a.mp3", "bpm": 124.0, "camelot": "8A",
         "energy": 0.2, "duration": 300.0},
        {"name": "b.mp3", "path": "/x/b.mp3", "bpm": 118.0, "camelot": "1B",
         "energy": 0.6, "duration": 240.0},
        {"name": "c.mp3", "path": "/x/c.mp3", "bpm": np.nan, "camelot": None,
         "energy": np.nan, "duration": 180.0},
    ])


def _at_path(frame) -> dict[str, int]:
    return {frame.at[i, "path"]: i for i in range(len(frame))}


def test_energy_bar_and_length_read_like_a_night():
    assert energy_bar(0.52) == "0.52 ▅"
    assert energy_bar(1.0) == "1.00 █"
    assert energy_bar(None) == "—"
    assert length_told(38 * 60) == "38m"
    assert length_told(72 * 60 + 20) == "1h 12m"


def test_shared_tracks_names_every_home_of_a_track_in_two_playlists():
    shared = shared_tracks({"intro": ["/x/a.mp3", "/x/b.mp3"],
                            "climax": ["/x/b.mp3", "/x/b.mp3"],
                            "release": ["/x/c.mp3"]})
    assert shared == {"/x/b.mp3": ["intro", "climax"]}


def test_shelf_rows_measure_only_what_the_map_knows():
    frame = _frame()
    rows = shelf_rows({"intro": ["/x/a.mp3", "/x/b.mp3", "/x/c.mp3",
                                 "/nowhere/ghost.mp3"],
                       "empty": []}, frame, _at_path(frame))
    intro, empty = rows.iloc[0], rows.iloc[1]
    assert intro["tracks"] == 4                    # il fantasma conta
    assert intro["BPM"] == "118–124"               # ma non misura
    assert intro["energy"] == "0.40 ▄"
    assert intro["keys"] == "1B 8A"                # lungo la ruota
    assert intro["length"] == "12m"
    assert intro["shared"] == 0
    assert empty["BPM"] == "—" and empty["energy"] == "—"
    assert empty["keys"] == "—" and empty["length"] == "0m"


def test_shelf_rows_count_the_shared_and_tell_where():
    frame = _frame()
    rows = shelf_rows({"intro": ["/x/a.mp3", "/x/b.mp3"],
                       "climax": ["/x/b.mp3"]}, frame, _at_path(frame))
    assert list(rows["shared"]) == [1, 1]
    assert rows.iloc[0]["_shared_told"] == "b.mp3 — also in climax"
    assert rows.iloc[1]["_shared_told"] == "b.mp3 — also in intro"


def test_shelf_summary_adds_the_night_up():
    frame = _frame()
    told = shelf_summary({"intro": ["/x/a.mp3", "/x/b.mp3"],
                          "climax": ["/x/b.mp3", "/x/c.mp3"]},
                         frame, _at_path(frame))
    assert told == ("2 playlist(s) · 4 track(s) · 16m · 1 track(s) in more "
                    "than one playlist")
