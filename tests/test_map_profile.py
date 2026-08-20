import numpy as np

from analysis.map_profile import (
    ProfileSettings,
    gain_for_target,
    onset_regularity,
    segment_offsets,
    select_labels,
)

SETTINGS = ProfileSettings()


def test_three_windows_on_a_normal_track():
    starts = segment_offsets(400, SETTINGS)
    assert len(starts) == 3
    # Al 25/50/75% della durata, centrate sulla finestra da 30 s.
    assert starts == [85.0, 185.0, 285.0]
    assert all(start + SETTINGS.segment_seconds <= 400 for start in starts)


def test_windows_never_hang_off_the_end():
    starts = segment_offsets(70, SETTINGS)
    assert all(0 <= s <= 40 for s in starts)


def test_a_track_shorter_than_one_window_is_analyzed_once():
    assert segment_offsets(20, SETTINGS) == [0.0]
    assert segment_offsets(30, SETTINGS) == [0.0]


def test_overlapping_windows_are_not_analyzed_twice():
    # Su un brano corto le tre posizioni cadono quasi nello stesso punto:
    # analizzarlo tre volte darebbe tre volte lo stesso vettore.
    starts = segment_offsets(35, SETTINGS)
    assert len(starts) < 3


def test_gain_brings_a_quiet_track_up_and_a_loud_one_down():
    assert gain_for_target(-20.0, -14.0) > 1.0
    assert gain_for_target(-14.0, -14.0) == 1.0
    assert gain_for_target(-8.0, -14.0) < 1.0


def test_gain_does_not_amplify_silence_into_noise():
    assert gain_for_target(-80.0) == 1.0        # praticamente muto: si lascia
    assert gain_for_target(None) == 1.0
    assert gain_for_target(-60.0) <= 10 ** (12 / 20)


def test_multi_label_keeps_everything_over_the_threshold():
    labels = ["Tech House", "Deep House", "Techno"]
    chosen = select_labels([0.85, 0.42, 0.04], labels, threshold=0.40, limit=4)
    assert chosen == [("Tech House", 0.85), ("Deep House", 0.42)]


def test_below_the_threshold_the_best_guess_survives():
    # Senza nemmeno un'etichetta il brano sparirebbe da ogni filtro: un
    # genere ce l'ha sempre, anche se il modello non è convinto.
    chosen = select_labels([0.03, 0.09], ["A", "B"], threshold=0.40, limit=4)
    assert chosen == [("B", 0.09)]


def test_labels_are_capped():
    activations = [0.9, 0.8, 0.7, 0.6, 0.5]
    chosen = select_labels(activations, list("abcde"), threshold=0.1, limit=2)
    assert [label for label, _ in chosen] == ["a", "b"]


def test_a_straight_kick_is_more_danceable_than_a_scattered_one():
    steady = onset_regularity(np.arange(0, 20, 0.5))
    scattered = onset_regularity([0, 0.1, 1.4, 1.5, 4.0, 7.9, 8.0, 8.05, 13.0, 20.0])
    assert steady == 1.0
    assert scattered < steady


def test_too_few_onsets_is_answered_with_i_do_not_know():
    assert onset_regularity([1.0, 2.0, 3.0]) is None
    assert onset_regularity([]) is None
