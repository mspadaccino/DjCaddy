"""Il payload della lavagna e le tabelle del Chain Maker (`core.viz.board`).

I pezzi numerici — altezze, tacche, scarti — hanno il loro snapshot in
`test_viz_snapshots`; qui si prova l'ASSEMBLAGGIO, che prima del refactor
viveva inline fra i widget di `render_board` e delle due tabelle e quindi
non era fotografabile: che ogni scheda porti i suoi campi, che gli scarti
guardino il brano precedente NELLA SCALETTA, che un brano sparito dalla
mappa faccia una scheda muta e non una pagina rotta.
"""

import pandas as pd

from core.viz.board import (_heights, board_payload, chain_table,
                            roster_table, wheel_payload)
from core.viz.track_columns import KEY_COLORS, OTHER_COLOR
from test_viz_snapshots import at_path_of, library, mood_common


def _payload(paths=None, axis="groove", dark=False):
    frame = library()
    at_path = at_path_of(frame)
    chosen = paths if paths is not None else list(frame["path"])[:6]
    return frame, at_path, board_payload(frame, at_path, chosen, axis,
                                         mood_common(frame), dark)


def test_one_card_per_track_in_playlist_order():
    frame, _, payload = _payload()
    nodes = payload["nodes"]
    assert [n["n"] for n in nodes] == [1, 2, 3, 4, 5, 6]
    assert [n["id"] for n in nodes] == list(frame["path"])[:6]
    assert payload["ticks"], "la scala verticale porta le sue tacche"


def test_the_heights_are_the_same_the_pieces_compute():
    frame, at_path, payload = _payload(axis="energy")
    heights = _heights(frame, at_path, list(frame["path"])[:6], "energy")
    for node in payload["nodes"]:
        assert node["height"] == heights[node["id"]]


def test_the_shifts_look_at_the_previous_track_in_the_playlist():
    """Non il brano da cui è stato scelto: l'ordine in cui il set uscirà è
    l'unico rispetto a cui "sale" o "scende" vuol dire qualcosa."""
    _, _, payload = _payload()
    nodes = payload["nodes"]
    assert nodes[0]["shift"] == {}          # il primo non viene da nessuno
    assert nodes[1]["shift"]["bpm"] == ("+1", 1)   # 110 → 111 nella libreria


def test_a_track_gone_from_the_map_makes_a_mute_card_not_a_crash():
    known = "/lib/Track 00.mp3"
    _, _, payload = _payload(paths=[known, "/altrove/sparito.mp3"])
    ghost = payload["nodes"][1]
    assert ghost["label"] == "sparito"
    assert ghost["height"] == 0.5
    assert ghost["bpm"] == "" and ghost["camelot"] == ""
    assert ghost["shift"] == {}
    assert payload["nodes"][0]["id"] == known


def test_the_unknown_genre_grey_follows_the_theme():
    _, _, light = _payload(paths=["/altrove/sparito.mp3"])
    _, _, dark = _payload(paths=["/altrove/sparito.mp3"], dark=True)
    assert light["nodes"][0]["color"] == OTHER_COLOR["light"]
    assert dark["nodes"][0]["color"] == OTHER_COLOR["dark"]


def test_the_chain_table_numbers_the_walk_and_skips_what_is_not_on_the_map():
    frame = library()
    at_path = at_path_of(frame)
    walk = [frame.at[2, "path"], "/altrove/sparito.mp3", frame.at[5, "path"]]
    table = chain_table(frame, at_path, walk, mood_common(frame))
    assert list(table["#"]) == [1, 3]       # il numero è la posizione nel set
    assert list(table["_path"]) == [walk[0], walk[2]]
    # Il primo della catena non viene da nessuno: nessuno scarto.
    assert table.iloc[0]["Δbpm"] is None


def test_the_roster_table_keeps_one_line_per_song_and_counts_the_copies():
    frame = library()
    picks = [(3, 0.1234, [3]), (7, 0.5678, [7, 9])]
    table = roster_table(frame, picks, frame.iloc[0], mood_common(frame))
    assert list(table["cost"]) == [0.123, 0.568]
    assert not table["Add"].any()           # si parte senza spunte
    assert pd.isna(table.iloc[0]["copies"])  # una copia sola non si conta
    assert table.iloc[1]["copies"] == 2
    assert list(table["_row"]) == [3, 7]


def test_the_wheel_payload_carries_the_choice_the_colours_and_the_theme():
    assert wheel_payload(["8A"], True) == {
        "selected": ["8A"], "colors": KEY_COLORS, "dark": True}
