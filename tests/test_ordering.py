"""Gli ordinamenti semplici: stabili, i senza-misura in coda, la tonalità
lungo la ruota."""

import numpy as np
import pandas as pd

from core.analysis.ordering import camelot_rank, order_by


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "bpm": [124.0, 118.0, 124.0, np.nan, 130.0],
        "energy": [0.9, 0.2, 0.4, 0.5, 0.4],
        "camelot": ["8A", "1B", "12A", "8B", None],
    })


def test_camelot_rank_walks_the_wheel():
    assert camelot_rank("1A") == 2
    assert camelot_rank("1B") == 3
    assert camelot_rank("2A") == 4
    assert camelot_rank("12B") == 25
    assert camelot_rank(None) is None
    assert camelot_rank("garbage") is None


def test_order_by_bpm_is_stable_and_puts_the_unknown_last():
    assert order_by(_frame(), [0, 1, 2, 3, 4], "bpm") == [1, 0, 2, 4, 3]
    assert order_by(_frame(), [2, 1, 0, 3, 4], "bpm") == [1, 2, 0, 4, 3]


def test_order_by_energy_descending_still_leaves_the_unknown_last():
    frame = _frame()
    frame.loc[4, "energy"] = np.nan
    assert order_by(frame, [0, 1, 2, 3, 4], "energy", descending=True) \
        == [0, 3, 2, 1, 4]


def test_order_by_key_follows_the_wheel():
    assert order_by(_frame(), [0, 1, 2, 3, 4], "key") == [1, 0, 3, 2, 4]


def test_sorts_compose_the_later_one_wins_and_the_earlier_breaks_ties():
    frame = _frame()
    by_energy = order_by(frame, [0, 1, 2, 3, 4], "energy", descending=True)
    then_bpm = order_by(frame, by_energy, "bpm")
    # A 124 BPM stanno 0 (energia 0.9) e 2 (0.4): l'energia decide.
    assert then_bpm == [1, 0, 2, 4, 3]
