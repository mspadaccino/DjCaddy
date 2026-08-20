import numpy as np

from analysis.mixing import (
    TransitionCost,
    along_path,
    bpm_distance,
    camelot_distance,
    magic_sort,
    nearest,
    resample_path,
    to_camelot,
)


def test_camelot_from_every_way_of_writing_a_key():
    assert to_camelot("A minor") == "8A"
    assert to_camelot("C") == "8B"          # maggiore sottinteso
    assert to_camelot("C major") == "8B"
    assert to_camelot("F# minor") == "11A"
    assert to_camelot("Gbm") == "11A"       # stessa chiave, altra grafia
    assert to_camelot("Bb major") == "6B"
    assert to_camelot("8A") == "8A"         # già in Camelot: si lascia stare
    assert to_camelot("E", "minor") == "9A"
    assert to_camelot("") is None
    assert to_camelot("qualcosa") is None


def test_camelot_distance_free_moves():
    assert camelot_distance("8A", "8A") == 0.0      # stessa chiave
    assert camelot_distance("8A", "9A") == 0.0      # adiacente
    assert camelot_distance("8A", "7A") == 0.0
    assert camelot_distance("8A", "8B") == 0.0      # relativa maggiore
    assert camelot_distance("8A", "10A") > 0.0      # due passi: si sente
    assert camelot_distance("8A", "2A") == 1.0      # dall'altra parte della ruota
    assert camelot_distance("1A", "12A") == 0.0     # la ruota si chiude


def test_camelot_distance_without_a_key():
    assert camelot_distance(None, "8A") == 0.5
    assert camelot_distance("8A", "non è un codice") == 0.5


def test_bpm_distance_folds_octaves():
    assert bpm_distance(128, 128) == 0.0
    assert bpm_distance(128, 64) == 0.0          # half-time: stessa griglia
    assert bpm_distance(128, 256) == 0.0
    assert bpm_distance(128, 132) < 0.5          # dentro il ±6%
    assert bpm_distance(128, 145) > 0.5          # fuori: si paga
    assert bpm_distance(None, 128) == 0.5


def _library():
    coords = np.array([[0, 0], [1, 0], [10, 0], [11, 0]], dtype=np.float32)
    return TransitionCost(coords, [128, 128, 128, 128], ["8A", "8A", "8A", "8A"])


def test_cost_is_normalized_and_symmetric_here():
    cost = _library()
    assert cost.between(0, 0) == 0.0
    assert 0.0 < cost.between(0, 1) < cost.between(0, 2)
    assert cost.between(0, 2) == cost.between(2, 0)


def test_weights_change_the_ranking():
    coords = np.array([[0, 0], [0.1, 0], [5, 0]], dtype=np.float32)
    # Il vicino sulla mappa ha il tempo sbagliato, il lontano ce l'ha giusto.
    cost = TransitionCost(coords, [128, 150, 128], ["8A", "8A", "8A"])
    cost.w_map, cost.w_bpm, cost.w_key = 1.0, 0.0, 0.0
    assert nearest(cost, 0, k=1)[0][0] == 1
    cost.w_map, cost.w_bpm, cost.w_key = 0.0, 1.0, 0.0
    assert nearest(cost, 0, k=1)[0][0] == 2


def test_nearest_stays_inside_the_pool():
    cost = _library()
    assert [i for i, _ in nearest(cost, 0, k=3, pool=[2, 3])] == [2, 3]


def test_resample_path_is_evenly_spaced():
    points = resample_path([(0, 0), (10, 0)], step=2.0)
    assert len(points) == 6
    assert np.allclose(np.diff(points[:, 0]), 2.0)


def test_along_path_orders_by_the_line_and_never_repeats():
    coords = np.array([[0, 0], [5, 0.1], [10, 0], [5, 50]], dtype=np.float32)
    taken = along_path(coords, [(0, 0), (10, 0)], radius=1.0)
    assert taken == [0, 1, 2]        # nell'ordine del tratto, il lontano fuori
    assert len(taken) == len(set(taken))


def test_along_path_honours_the_pool():
    coords = np.array([[0, 0], [5, 0], [10, 0]], dtype=np.float32)
    assert along_path(coords, [(0, 0), (10, 0)], radius=1.0, pool=[0, 2]) == [0, 2]


def test_magic_sort_beats_the_order_it_was_given():
    # Quattro brani in fila sulla mappa, dati in ordine sparso.
    coords = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float32)
    cost = TransitionCost(coords, [128] * 4, ["8A"] * 4)
    given = [0, 3, 1, 2]

    def total(order):
        return sum(cost.between(a, b) for a, b in zip(order, order[1:]))

    sorted_order = magic_sort(cost, given, start=0)
    assert set(sorted_order) == set(given)
    assert sorted_order[0] == 0                 # il primo brano resta il primo
    assert total(sorted_order) < total(given)
    assert sorted_order == [0, 1, 2, 3]


def test_magic_sort_with_too_few_tracks_to_sort():
    cost = _library()
    assert magic_sort(cost, [2]) == [2]
    assert magic_sort(cost, [2, 1]) == [2, 1]
