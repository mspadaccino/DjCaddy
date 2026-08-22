import numpy as np
import pytest

from analysis.map_profile import (
    ProfileSettings,
    gain_for_target,
    onset_regularity,
    rhythm_offset,
    segment_offsets,
    select_labels,
)

SETTINGS = ProfileSettings()


def test_twelve_windows_spread_over_the_whole_track():
    starts = segment_offsets(400, SETTINGS)
    assert len(starts) == 12
    # Centrate a 1/24, 3/24, 5/24... della durata: la prima sta nell'intro,
    # l'ultima nella coda, che con tre finestre ai quarti non si vedevano.
    assert starts[0] == pytest.approx(400 * 0.5 / 12 - 5)
    assert starts[-1] == pytest.approx(400 * 11.5 / 12 - 5)
    assert all(start + SETTINGS.segment_seconds <= 400 for start in starts)
    gaps = {round(b - a, 4) for a, b in zip(starts, starts[1:])}
    assert len(gaps) == 1                       # equidistanti


def test_the_rhythm_window_sits_in_the_middle_and_is_longer():
    # Non è una delle finestre dell'embedding: quelle sono da 10 s, e un
    # rilevatore di tempo su 10 s non trova abbastanza battute.
    assert SETTINGS.rhythm_seconds > SETTINGS.segment_seconds
    assert rhythm_offset(400, SETTINGS) == 185.0
    assert rhythm_offset(400, SETTINGS) + SETTINGS.rhythm_seconds <= 400


def test_a_track_shorter_than_the_rhythm_window_is_taken_whole():
    assert rhythm_offset(20, SETTINGS) == 0.0


def test_windows_never_hang_off_the_end():
    starts = segment_offsets(70, SETTINGS)
    assert all(0 <= s <= 70 - SETTINGS.segment_seconds for s in starts)


def test_a_track_shorter_than_one_window_is_analyzed_once():
    assert segment_offsets(8, SETTINGS) == [0.0]
    assert segment_offsets(10, SETTINGS) == [0.0]


def test_overlapping_windows_are_not_analyzed_twice():
    # Su un brano corto le posizioni si accavallano: analizzare due volte lo
    # stesso pezzo lo peserebbe il doppio nella media.
    starts = segment_offsets(35, SETTINGS)
    assert len(starts) < 12
    assert all(b - a > SETTINGS.segment_seconds / 4
               for a, b in zip(starts, starts[1:]))


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
