import numpy as np

from analysis.models import BREAKDOWN, BUILDUP, DROP, GROOVE, INTRO, OUTRO
from analysis.sections import _label


def test_label_edm_arc():
    # Arco tipico: intro, build, drop, breakdown, secondo drop, outro
    energy = np.array([0.20, 0.50, 1.00, 0.30, 0.95, 0.15])
    bass = np.array([0.30, 0.40, 0.60, 0.20, 0.60, 0.30])
    assert _label(energy, bass) == [INTRO, BUILDUP, DROP, BREAKDOWN, DROP, OUTRO]


def test_label_single_section_is_groove():
    assert _label(np.array([0.5]), np.array([0.5])) == [GROOVE]


def test_label_two_sections_are_intro_outro():
    assert _label(np.array([0.3, 0.9]), np.array([0.3, 0.6])) == [INTRO, OUTRO]
