from core.analysis.models import Section
from core.analysis.sections import _merge_consecutive


def _labels(sections):
    return [s.label for s in sections]


def test_merge_collapses_consecutive_duplicates():
    raw = [
        Section(0, 10, "Intro", 0.2),
        Section(10, 20, "Groove", 0.5),
        Section(20, 30, "Groove", 0.7),   # stesso tipo -> unito al precedente
        Section(30, 40, "Drop/Chorus", 1.0),
        Section(40, 50, "Drop/Chorus", 0.9),  # unito
        Section(50, 60, "Breakdown", 0.3),
        Section(60, 70, "Drop/Chorus", 0.95),  # NON consecutivo al primo drop
    ]
    merged = _merge_consecutive(raw)
    assert _labels(merged) == ["Intro", "Groove", "Drop/Chorus", "Breakdown", "Drop/Chorus"]
    groove = merged[1]
    assert (groove.start, groove.end) == (10, 30)   # estende fino alla fine del run
    assert groove.energy == 0.7                      # max del run


def test_merge_empty():
    assert _merge_consecutive([]) == []
