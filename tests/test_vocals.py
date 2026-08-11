import numpy as np

from analysis.models import Section
from analysis.sections import annotate_vocals
from analysis.vocals import vocal_regions


def _envelope():
    # ratio = dominanza voce/mix; 0.6 = voce, 0.1 = solo bleed
    times = np.round(np.arange(0, 10, 0.1), 3)
    ratio = np.full_like(times, 0.1)
    ratio[20:40] = 0.6    # t 2.0..3.9
    ratio[45:50] = 0.6    # t 4.5..4.9 (pausa 0.6s -> unita alla precedente)
    ratio[80:83] = 0.6    # t 8.0..8.2 (troppo breve -> scartata)
    return times, ratio


def test_vocal_regions_merge_and_min_length():
    # Soglia esplicita: il test verifica la logica di merge/durata minima,
    # indipendentemente da quale sia il default prodotto del momento.
    regions = vocal_regions(_envelope(), floor=0.30)
    assert len(regions) == 1
    st, en = regions[0]
    assert abs(st - 2.0) < 0.1
    assert abs(en - 5.0) < 0.15


def test_vocal_regions_instrumental_no_regions():
    # Strumentale: solo bleed sotto soglia -> nessuna regione (caso WTP).
    # Soglia esplicita, non legata al default prodotto corrente.
    times = np.round(np.arange(0, 10, 0.1), 3)
    ratio = np.full_like(times, 0.12)
    assert vocal_regions((times, ratio), floor=0.30) == []


def test_vocal_regions_empty():
    assert vocal_regions((np.array([]), np.array([]))) == []


def test_annotate_vocals_coverage():
    sections = [Section(0, 4, "Intro"), Section(4, 8, "Drop/Chorus"),
                Section(8, 12, "Outro")]
    annotate_vocals(sections, [(2.0, 5.0)])
    assert sections[0].vocal and sections[0].vocal_score == 0.5   # 2s su 4
    assert sections[1].vocal                                       # 1s su 4 = 0.25
    assert not sections[2].vocal                                   # nessuna copertura
