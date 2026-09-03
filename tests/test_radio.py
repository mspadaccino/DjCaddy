import numpy as np

from core.analysis import radio
from core.analysis.radio import split, unit


def tune(*args, **kwargs):
    # I ventagli qui sotto stanno a dieci gradi l'uno dall'altro, che in
    # coseno è 0.985: sopra la soglia dei gemelli vera. Qui si prova la
    # scelta, non la soglia — che ha il suo test.
    kwargs.setdefault("twin_min", 0.999)
    return radio.tune(*args, **kwargs)


def _ray(degrees: float) -> list[float]:
    angle = np.radians(degrees)
    return [float(np.cos(angle)), float(np.sin(angle))]


def _fan(*degrees) -> np.ndarray:
    return np.array([_ray(d) for d in degrees], dtype=np.float32)


def test_unit_makes_every_row_length_one():
    rows = unit([[3.0, 4.0], [0.0, 2.0]])
    assert np.allclose(np.linalg.norm(rows, axis=1), 1.0)


def test_split_leaves_one_tight_group_alone():
    assert split(_fan(0, 3, 6, 2, 4)) == [[0, 1, 2, 3, 4]]


def test_split_finds_the_two_souls_of_a_mixed_group():
    parts = split(_fan(0, 5, 90, 95, 2))
    assert sorted(sorted(p) for p in parts) == [[0, 1, 4], [2, 3]]


def test_split_stops_at_max_parts():
    parts = split(_fan(0, 90, 180, 270), max_parts=2)
    assert len(parts) == 2


def test_tune_never_returns_a_seed_or_a_negative_and_respects_the_pool():
    vectors = _fan(0, 10, 20, 30, 40)
    picks = tune(vectors, seeds=[0], negatives=[1], k=10, pool=[0, 1, 2, 3])
    assert picks == [2, 3]


def test_tune_of_nothing_is_nothing():
    assert tune(_fan(0, 10), seeds=[], k=5) == []
    assert tune(_fan(0, 10), seeds=[0], k=0) == []


def test_a_twin_of_a_seed_or_of_a_pick_stays_out():
    vectors = _fan(0, 0.5, 20, 20.5, 40)
    # La 1 è gemella del seme, la 3 è gemella della 2: entrano solo 2 e 4.
    assert tune(vectors, seeds=[0], k=5, variety=0.0, twin_min=0.97) == [2, 4]


def test_variety_pushes_the_second_pick_away_from_the_first():
    vectors = _fan(0, 10, 15, -25)
    # Senza varietà si va per vicinanza pura: 10, poi la 15 che le sta
    # accanto. Con varietà, la 15 paga la somiglianza alla 10 e passa la -25.
    assert tune(vectors, seeds=[0], k=2, variety=0.0) == [1, 2]
    assert tune(vectors, seeds=[0], k=2, variety=1.0) == [1, 3]


def test_drift_turns_the_walk_into_a_journey():
    vectors = _fan(0, 10, -10, 20, -20, 30, -30)
    # Ferma, la radio resta attorno al seme e alterna i due lati. In deriva
    # piena il profilo È l'ultimo preso, e si continua da quella parte.
    assert tune(vectors, seeds=[0], k=3, variety=0.0, drift=0.0) == [1, 2, 3]
    assert tune(vectors, seeds=[0], k=3, variety=0.0, drift=1.0) == [1, 3, 5]


def test_negatives_pull_the_profile_the_other_way():
    vectors = _fan(0, 10, -10, 25)
    assert tune(vectors, seeds=[0], k=1, variety=0.0) == [1]
    assert tune(vectors, seeds=[0], k=1, variety=0.0, negatives=[3]) == [2]


def test_a_two_souled_group_gets_picks_from_both():
    vectors = _fan(0, 90, 5, 10, 85, 80)
    picks = tune(vectors, seeds=[0, 1], k=4, variety=0.0)
    assert sorted(picks) == [2, 3, 4, 5]
    assert picks[0] in (2, 3) and picks[1] in (4, 5)   # a turno
