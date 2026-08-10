import numpy as np

from analysis.models import Section
from analysis.sections import annotate_vocals
from analysis.vocals import vocal_regions


def _envelope():
    times = np.round(np.arange(0, 10, 0.1), 3)
    rms = np.zeros_like(times)
    rms[20:40] = 1.0    # t 2.0..3.9
    rms[45:50] = 1.0    # t 4.5..4.9 (pausa 0.6s -> unita alla precedente)
    rms[80:83] = 1.0    # t 8.0..8.2 (troppo breve -> scartata)
    return times, rms


def test_vocal_regions_merge_and_min_length():
    regions = vocal_regions(_envelope())
    assert len(regions) == 1
    st, en = regions[0]
    assert abs(st - 2.0) < 0.1
    assert abs(en - 5.0) < 0.15


def test_vocal_regions_empty():
    assert vocal_regions((np.array([]), np.array([]))) == []


def test_annotate_vocals_coverage():
    sections = [Section(0, 4, "Intro"), Section(4, 8, "Drop/Chorus"),
                Section(8, 12, "Outro")]
    annotate_vocals(sections, [(2.0, 5.0)])
    assert sections[0].vocal and sections[0].vocal_score == 0.5   # 2s su 4
    assert sections[1].vocal                                       # 1s su 4 = 0.25
    assert not sections[2].vocal                                   # nessuna copertura
