import numpy as np

from core.analysis.mixing import (
    TransitionCost,
    along_path,
    closed_shape,
    bpm_distance,
    bpm_shift,
    camelot_distance,
    camelot_shift,
    magic_sort,
    nearest,
    resample_path,
    to_camelot,
)


def _fan(*degrees) -> np.ndarray:
    """Vettori unitari a ventaglio: la distanza di suono cresce con
    l'angolo, che è quello che le rette di prima facevano con la x."""
    angles = np.radians(degrees)
    return np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)


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


def test_bpm_shift_keeps_the_sign():
    assert bpm_shift(118, 122) == 4
    assert bpm_shift(122, 118) == -4
    assert bpm_shift(120, 120) == 0


def test_bpm_shift_folds_octaves_like_the_distance():
    # Half-time non è una frenata di 64 BPM: è lo stesso passo.
    assert bpm_shift(128, 64) == 0
    assert bpm_shift(128, 66) == 4
    assert bpm_shift(128, 260) == 2


def test_bpm_shift_without_a_tempo():
    assert bpm_shift(None, 128) is None
    assert bpm_shift(128, 0) is None


def test_camelot_shift_takes_the_short_way_round():
    assert camelot_shift("8A", "9A") == (1, False)
    assert camelot_shift("8A", "7A") == (-1, False)
    assert camelot_shift("12A", "1A") == (1, False)     # la ruota si chiude
    assert camelot_shift("1A", "12A") == (-1, False)


def test_camelot_shift_reports_a_change_of_mode():
    assert camelot_shift("8A", "8B") == (0, True)       # il relativo maggiore
    assert camelot_shift("8A", "9B") == (1, True)


def test_camelot_shift_without_a_key():
    assert camelot_shift(None, "8A") is None
    assert camelot_shift("8A", "non è un codice") is None


def test_bpm_distance_folds_octaves():
    assert bpm_distance(128, 128) == 0.0
    assert bpm_distance(128, 64) == 0.0          # half-time: stessa griglia
    assert bpm_distance(128, 256) == 0.0
    assert bpm_distance(128, 132) < 0.5          # dentro il ±6%
    assert bpm_distance(128, 145) > 0.5          # fuori: si paga
    assert bpm_distance(None, 128) == 0.5


def _library():
    return TransitionCost(_fan(0, 15, 60, 75), [128, 128, 128, 128],
                          ["8A", "8A", "8A", "8A"])


def test_cost_is_normalized_and_symmetric_here():
    cost = _library()
    assert cost.between(0, 0) == 0.0
    assert 0.0 < cost.between(0, 1) < cost.between(0, 2)
    assert cost.between(0, 2) == cost.between(2, 0)


def test_weights_change_the_ranking():
    # Il vicino di suono ha il tempo sbagliato, il lontano ce l'ha giusto.
    cost = TransitionCost(_fan(0, 2, 75), [128, 150, 128], ["8A", "8A", "8A"])
    cost.w_sound, cost.w_bpm, cost.w_key = 1.0, 0.0, 0.0
    assert nearest(cost, 0, k=1)[0][0] == 1
    cost.w_sound, cost.w_bpm, cost.w_key = 0.0, 1.0, 0.0
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
    # Quattro brani in fila nel suono, dati in ordine sparso.
    cost = TransitionCost(_fan(0, 15, 30, 45), [128] * 4, ["8A"] * 4)
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


def test_a_stroke_that_comes_back_is_a_fence():
    circle = [(np.cos(a), np.sin(a)) for a in np.linspace(0, 2 * np.pi, 24)]
    assert closed_shape(circle)
    # Un cerchio lasciato aperto di poco è ancora un cerchio.
    assert closed_shape(circle[:-3])


def test_a_stroke_that_goes_somewhere_is_a_path():
    assert not closed_shape([(0, 0), (1, 0), (2, 0), (3, 0)])
    assert not closed_shape([(0, 0), (1, 1), (2, 0), (3, 1)])
    assert not closed_shape([(0, 0), (1, 0)])       # due punti non recintano


# --- la tendenza: cercare da dove la catena sta andando ---

def _line():
    # In fila sul ventaglio: la 0 sta un po' più indietro della 2 rispetto
    # alla 1, così da 1 la rosa ha un ordine e non un pareggio.
    return TransitionCost(_fan(-5, 15, 30, 45, -20),
                          [120, 124, 128, 132, 116],
                          ["8A", "8A", "8A", "8A", "8A"])


def test_from_point_at_a_track_is_the_cost_from_that_track():
    cost = _line()
    point = (cost.vectors[1], cost.bpm[1], cost.camelot[1])
    assert np.allclose(cost.from_point(point, [0, 2, 3]), cost.to(1, [0, 2, 3]))


def test_ahead_continues_the_step_in_sound_and_in_tempo():
    cost = _line()
    vector, bpm, key = cost.ahead(previous=0, last=1, trend=1.0)
    assert bpm == 128 and key == "8A"
    # Un passo avanti da 1 nella direzione 0→1 sta più vicino alla 2 che
    # alla 1 stessa.
    at = (vector, 128, "8A")
    assert cost.from_point(at, [2])[0] < cost.from_point(at, [1])[0]
    vector, bpm, _ = cost.ahead(previous=0, last=1, trend=0.0)
    assert np.allclose(vector, cost.vectors[1]) and bpm == 124


def test_ahead_keeps_the_last_tempo_when_one_is_unknown():
    cost = TransitionCost(_fan(0, 15), [None, 124], ["8A", "8A"])
    assert cost.ahead(0, 1, 1.0)[1] == 124


def test_nearest_looks_ahead_when_told_to():
    cost = _line()
    # Da 1, la 2 poi la 0. Guardando avanti di un passo pieno il punto sta
    # sulla 2, e la 3 batte la 0.
    assert [i for i, _ in nearest(cost, 1, k=2)] == [2, 0]
    ahead = cost.ahead(0, 1, 1.0)
    assert [i for i, _ in nearest(cost, 1, k=2, ahead=ahead)] == [2, 3]
