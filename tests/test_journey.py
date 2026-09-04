"""Il Journey: da un brano a un altro in n passi, sull'arco.

I brani sono raggi in un piano: la distanza di suono cresce con l'angolo,
e 1 − cos è convessa sotto i 90 gradi, quindi la strada più economica fra
due raggi è a passi uguali — che è quello che questi test verificano."""

import numpy as np

from core.analysis import arc
from core.analysis.journey import corridor, plan
from core.analysis.mixing import TransitionCost


def _fan(*degrees) -> np.ndarray:
    angles = np.radians(degrees)
    return np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)


def _cost(degrees, bpm=None, keys=None) -> TransitionCost:
    n = len(degrees)
    return TransitionCost(_fan(*degrees), bpm or [128.0] * n,
                          keys or ["8A"] * n)


def _hops(cost, order):
    return [cost.between(a, b) for a, b in zip(order, order[1:])]


def test_from_a_to_b_in_n_steps_takes_even_steps():
    # Nove raggi a venti gradi l'uno dall'altro: da 0 a 8 in cinque brani
    # la strada dritta è ogni due.
    cost = _cost(range(0, 180, 20))
    assert plan(cost, 0, 5, end=8) == [0, 2, 4, 6, 8]
    assert plan(cost, 8, 5, end=0) == [8, 6, 4, 2, 0]


def test_the_ends_stay_where_they_were_asked_and_nothing_repeats():
    cost = _cost(range(0, 180, 20))
    for n in (3, 4, 6, 9):
        path = plan(cost, 0, n, end=8)
        assert len(path) == n
        assert path[0] == 0 and path[-1] == 8
        assert len(set(path)) == n


def test_without_an_end_the_journey_still_has_n_distinct_tracks():
    cost = _cost(range(0, 180, 20))
    path = plan(cost, 4, 5)
    assert path[0] == 4 and len(path) == 5 and len(set(path)) == 5
    # Ogni passo costa poco: niente salti dall'altra parte del ventaglio.
    assert max(_hops(cost, path)) <= cost.between(0, 2) + 1e-6


def test_short_journeys_and_a_start_that_is_also_the_end():
    cost = _cost(range(0, 180, 20))
    assert plan(cost, 3, 1, end=8) == [3]
    assert plan(cost, 3, 0) == [3]
    assert plan(cost, 3, 2, end=8) == [3, 8]
    same = plan(cost, 3, 2, end=3)
    assert len(same) == 2 and same[0] == 3 and same[1] != 3
    assert len(plan(cost, 3, 4, end=3)) == 4         # l'arrivo uguale non conta


def test_the_pool_limits_the_middle_but_never_the_ends():
    cost = _cost(range(0, 180, 20))
    path = plan(cost, 0, 4, end=8, pool=[1, 3, 5, 7])
    assert path[0] == 0 and path[-1] == 8
    assert set(path[1:-1]) <= {1, 3, 5, 7}


def test_too_narrow_a_corridor_gives_what_there_is():
    cost = _cost(range(0, 180, 20))
    path = plan(cost, 0, 6, end=8, pool=[4])
    assert path == [0, 4, 8]


def test_a_walk_that_would_come_back_is_straightened():
    # A(0°) X(5°) Y(60°) Z(65°) B(10°): da A a B in cinque, la strada più
    # economica passa da X, va a Y e TORNA su X. Un set non prende lo stesso
    # disco due volte: X resta dove compare la prima volta e al posto del
    # ritorno entra Z.
    cost = _cost([0, 5, 60, 65, 10])
    path = plan(cost, 0, 5, end=4, twin=0.0)
    assert path == [0, 1, 2, 3, 4]
    assert len(set(path)) == 5


def test_twins_are_never_a_transition():
    # 1 e 2 sono lo stesso master: fra loro non si passa. Da 0 a 3 in
    # quattro si deve passare per uno dei due e poi per qualcos'altro.
    cost = _cost([0, 30, 30, 90, 60])
    path = plan(cost, 0, 4, end=3, twin=0.03)
    assert path[0] == 0 and path[-1] == 3
    assert not ({1, 2} <= set(path[1:3]))
    assert set(path[1:3]) & {1, 2} and 4 in path


def test_copies_of_the_same_song_enter_once():
    # 1 e 3 sono la stessa canzone in due cartelle; 0 e 5 pure, e 0 è la
    # partenza: la copia della partenza non entra mai.
    cost = _cost(range(0, 120, 20))
    songs = {0: "a", 5: "a", 1: "b", 3: "b", 2: "c", 4: "d"}
    inner = corridor(cost, 0, None, reach=10, song_of=songs.get)
    assert 5 not in inner and len({songs[i] for i in inner}) == len(inner)
    path = plan(cost, 0, 4, song_of=songs.get, twin=0.0)
    assert len({songs[i] for i in path}) == 4


def test_the_corridor_is_the_ellipse_between_the_ends():
    cost = _cost([0, 20, 40, 60, 80, 100, 170])
    # Fra 0 e 60 chi sta in mezzo viene prima di chi sta oltre.
    assert corridor(cost, 0, 3, reach=2) == [1, 2]
    # Senza arrivo è la palla attorno alla partenza.
    assert corridor(cost, 0, None, reach=2) == [1, 2]
    assert corridor(cost, 0, 3, pool=[0, 3]) == []


def test_the_arc_pulls_the_climax_to_the_middle_of_the_set():
    # Dieci raggi in fila, energie tiepide, tranne il 6: il più energico di
    # tutti, ma un BPM fuori griglia lo mette un po' fuori strada. Senza
    # arco la strada dritta lo salta; con l'arco il Climax, che vuole
    # energia alta verso i tre quarti, lo porta dentro.
    degrees = [0, 20, 40, 60, 80, 100, 130, 120, 140, 160]
    bpm = [128.0] * 10
    bpm[6] = 129.0
    cost = _cost(degrees, bpm=bpm)
    energy = [0.05, 0.15, 0.3, 0.45, 0.5, 0.55, 0.98, 0.55, 0.5, 0.35]
    values = arc.measures(bpm=bpm, energy=energy,
                          valence_rank=[0.5] * 10,
                          danceability=[0.5] * 10)
    flat = plan(cost, 0, 6, end=9, arc_values=values, w_arc=0.0)
    shaped = plan(cost, 0, 6, end=9, arc_values=values, w_arc=1.0)
    assert 6 not in flat
    assert 6 in shaped
    at = shaped.index(6)
    assert arc.CHAPTERS[arc.chapters_along(6)[at]]["name"] == "Climax"
    assert shaped[0] == 0 and shaped[-1] == 9 and len(set(shaped)) == 6
