import numpy as np
import pytest

from analysis.graph_playlist import GraphPlaylist, suggestions
from analysis.mixing import TransitionCost


def test_start_places_two_tracks_symmetrically():
    graph = GraphPlaylist().start("a", "b")
    assert graph.tracks == ["a", "b"]
    assert graph.linked("a", "b")
    ax, _ = graph.places["a"]
    bx, _ = graph.places["b"]
    assert ax < 0.5 < bx


def test_start_refuses_the_same_track_twice():
    with pytest.raises(ValueError):
        GraphPlaylist().start("a", "a")


def test_add_hangs_a_track_off_its_source():
    graph = GraphPlaylist().start("a", "b").add("b", "c")
    assert graph.tracks == ["a", "b", "c"]
    assert graph.linked("b", "c")
    assert not graph.linked("a", "c")
    assert graph.neighbours("b") == ["a", "c"]


def test_add_an_already_placed_track_only_connects_it():
    graph = GraphPlaylist().start("a", "b").add("a", "c").add("b", "c")
    assert graph.tracks == ["a", "b", "c"]        # non duplicato
    assert graph.linked("a", "c") and graph.linked("b", "c")


def test_add_from_a_source_not_on_the_board_fails():
    graph = GraphPlaylist().start("a", "b")
    with pytest.raises(KeyError):
        graph.add("z", "c")


def test_move_repositions_a_placed_track():
    graph = GraphPlaylist().start("a", "b")
    graph.move("a", 0.1, 0.9)
    assert graph.places["a"] == (0.1, 0.9)


def test_move_a_track_not_on_the_board_is_a_no_op():
    graph = GraphPlaylist()
    graph.move("ghost", 0.5, 0.5)
    assert "ghost" not in graph


def test_remove_from_the_middle_of_a_chain_reconnects_it():
    graph = GraphPlaylist().start("a", "b").add("b", "c")
    graph.remove("b")
    assert graph.tracks == ["a", "c"]
    assert graph.linked("a", "c")


def test_remove_a_junction_leaves_the_branches_split():
    # "a" con tre vicini: toglierlo non ricuce, o si inventerebbero due
    # collegamenti che nessuno ha scelto.
    graph = GraphPlaylist().start("a", "b").add("a", "c").add("a", "d")
    graph.remove("a")
    assert graph.tracks == ["b", "c", "d"]
    assert not graph.linked("b", "c")
    assert not graph.linked("b", "d")
    assert not graph.linked("c", "d")


def test_ends_are_the_free_tips_of_the_chain():
    graph = GraphPlaylist().start("a", "b").add("b", "c")
    assert graph.ends() == ["a", "c"]


def test_walk_reads_a_chain_in_order():
    graph = GraphPlaylist().start("a", "b").add("b", "c").add("c", "d")
    assert graph.walk() == ["a", "b", "c", "d"]


def test_walk_goes_depth_first_down_a_branch_before_the_next():
    graph = GraphPlaylist().start("a", "b").add("b", "c").add("b", "d")
    walked = graph.walk()
    assert walked[0] == "a" and walked[1] == "b"
    assert set(walked[2:]) == {"c", "d"}


def test_state_roundtrip_keeps_everything():
    graph = GraphPlaylist().start("a", "b").add("b", "c")
    restored = GraphPlaylist.from_state(graph.to_state())
    assert restored.tracks == graph.tracks
    assert restored.places == graph.places
    assert restored.links == graph.links


def test_from_state_drops_a_link_pointing_nowhere():
    state = {"places": {"a": [0.5, 0.5]}, "order": ["a"],
             "links": [["a", "ghost"]]}
    graph = GraphPlaylist.from_state(state)
    assert graph.tracks == ["a"]
    assert graph.links == []


def test_from_state_of_nothing_is_an_empty_board():
    assert GraphPlaylist.from_state(None).tracks == []


def test_straighten_puts_a_chain_left_to_right_in_reading_order():
    graph = GraphPlaylist().start("a", "b").add("b", "c").add("c", "d")
    graph.move("a", 0.9, 0.9)
    graph.straighten()
    xs = [graph.places[t][0] for t in ["a", "b", "c", "d"]]
    assert xs == sorted(xs)
    assert len({graph.places[t][1] for t in graph.tracks}) == 1


def test_straighten_alternates_the_direction_of_each_row():
    graph = GraphPlaylist().start("a", "b")
    for previous, track in zip("bcd", "cde"):
        graph.add(previous, track)
    graph.straighten(per_row=2)
    # Riga 1 va a destra, riga 2 torna indietro: "b" e "c" restano vicini.
    assert graph.places["a"][0] < graph.places["b"][0]
    assert graph.places["d"][0] < graph.places["c"][0]
    assert graph.places["a"][1] < graph.places["c"][1]


def test_straighten_of_an_empty_board_does_nothing():
    assert GraphPlaylist().straighten().places == {}


def _library():
    coords = np.array([[0, 0], [1, 0], [2, 0], [10, 0]], dtype=np.float32)
    return TransitionCost(coords, [128, 128, 128, 128], ["8A", "8A", "8A", "8A"])


def test_suggestions_exclude_what_is_already_on_the_board():
    cost = _library()
    found = suggestions(cost, seed=0, taken={0, 1}, k=2)
    assert [i for i, _, _ in found] == [2, 3]


def test_suggestions_respect_a_pool():
    cost = _library()
    found = suggestions(cost, seed=0, taken=set(), k=5, pool=[0, 2])
    assert [i for i, _, _ in found] == [2]


def test_suggestions_give_every_track_its_own_voice_without_a_key():
    cost = _library()
    found = suggestions(cost, seed=0, taken=set(), k=3)
    assert [copies for _, _, copies in found] == [[1], [2], [3]]


# I brani 1 e 2 sono due copie della stessa musica in cartelle diverse: hanno
# gli stessi BPM e la stessa tonalità, quindi lo stesso costo da qualunque
# sorgente, ed è per questo che si presentano in fila.
_COPIES = {0: "a", 1: "b", 2: "b", 3: "c"}


def test_suggestions_gather_the_copies_of_one_track_into_one_voice():
    cost = _library()
    found = suggestions(cost, seed=0, taken=set(), k=3, key_of=_COPIES.get)
    assert [i for i, _, _ in found] == [1, 3]
    assert found[0][2] == [1, 2]        # la copia viaggia con la voce


def test_suggestions_drop_every_copy_of_what_is_already_on_the_board():
    cost = _library()
    found = suggestions(cost, seed=0, taken={1}, k=3, key_of=_COPIES.get)
    # La 2 è l'altra copia della 1, che sta già sulla lavagna: proporla
    # significherebbe mettere lo stesso brano due volte nello stesso set.
    assert [i for i, _, _ in found] == [3]


def test_suggestions_keep_collecting_copies_once_the_roster_is_full():
    cost = _library()
    found = suggestions(cost, seed=0, taken=set(), k=1, key_of=_COPIES.get)
    assert [i for i, _, _ in found] == [1]
    assert found[0][2] == [1, 2]
