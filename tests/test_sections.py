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


# --- riconoscimento dei build-up dalla pendenza interna --------------------

def test_buildup_recognized_from_internal_rise_not_mean_energy():
    """Il caso che la sola energia media non può cogliere: una sezione che
    per media sembra un groove qualunque ma dentro di sé sale verso il drop."""
    energy = np.array([0.20, 0.60, 1.00, 0.15])
    bass = np.array([0.30, 0.30, 0.60, 0.30])
    flat = np.array([0.0, 0.0, 0.0, 0.0])
    rising = np.array([0.0, 0.9, 0.0, 0.0])

    # senza pendenza la sezione 1 viene presa lo stesso (sta prima del drop)
    assert _label(energy, bass, flat)[1] == BUILDUP
    assert _label(energy, bass, rising)[1] == BUILDUP


def test_buildup_found_even_when_not_followed_by_a_drop():
    """Salita verso una sezione più intensa che non arriva alla soglia di
    drop: prima restava un Groove anonimo."""
    energy = np.array([0.20, 0.45, 0.60, 0.15])
    bass = np.array([0.30, 0.30, 0.50, 0.30])
    slope = np.array([0.0, 0.8, 0.0, 0.0])
    assert _label(energy, bass, slope)[1] == BUILDUP
    assert _label(energy, bass)[1] != BUILDUP     # senza pendenza, non c'è indizio


def test_rising_section_is_not_a_drop_even_if_loud():
    """La coda di un build-up arriva forte e col basso: senza la pendenza
    verrebbe scambiata per un drop."""
    energy = np.array([0.20, 0.90, 1.00, 0.15])
    bass = np.array([0.30, 0.60, 0.60, 0.30])
    slope = np.array([0.0, 0.9, 0.0, 0.0])
    labels = _label(energy, bass, slope)
    assert labels[1] == BUILDUP and labels[2] == DROP


def test_breakdown_is_a_fall_not_just_a_low_section():
    """Un breakdown è un CALO dopo qualcosa di più grosso. La stessa sezione
    bassa, se arriva salendo dall'intro, è la rampa verso il drop."""
    bass = np.array([0.30, 0.20, 0.60, 0.30])
    caduta = np.array([0.90, 0.30, 1.00, 0.15])
    salita = np.array([0.20, 0.30, 1.00, 0.15])
    assert _label(caduta, bass)[1] == BREAKDOWN
    assert _label(salita, bass)[1] == BUILDUP


def test_breakdown_found_when_only_the_bass_leaves():
    """Il caso che la vecchia regola perdeva: nella house/disco il breakdown
    tiene su archi e voce (energia ancora 0.61) e cala solo il basso.
    Misurato su un brano reale."""
    energy = np.array([0.47, 0.92, 0.61, 1.00, 0.04])
    bass = np.array([0.20, 1.00, 0.07, 0.99, 0.01])
    assert _label(energy, bass) == [INTRO, DROP, BREAKDOWN, DROP, OUTRO]


def test_drop_survives_realistic_bass_levels():
    """La regressione vera: con la vecchia feature (quota di spettro sotto i
    200 Hz) nessuna sezione raggiungeva mai la soglia e non c'era MAI un
    Drop. Con il basso rapportato al massimo del brano, sì."""
    energy = np.array([0.17, 0.58, 0.94, 0.30, 1.00, 0.16])
    bass = np.array([0.04, 1.00, 0.92, 0.11, 0.95, 0.00])
    assert _label(energy, bass) == [INTRO, BUILDUP, DROP, BREAKDOWN, DROP, OUTRO]


def test_edges_stay_intro_outro_but_not_over_a_drop():
    """Brano che chiude in pieno: chiamare Outro l'ultimo drop sarebbe
    fuorviante in console."""
    energy = np.array([0.20, 0.55, 0.95])
    bass = np.array([0.30, 0.50, 0.60])
    assert _label(energy, bass) == [INTRO, BUILDUP, DROP]

    # e viceversa: se l'ultima sezione non è un drop resta l'Outro
    energy = np.array([0.20, 0.95, 0.30])
    bass = np.array([0.30, 0.60, 0.30])
    assert _label(energy, bass) == [INTRO, DROP, OUTRO]


def test_opening_drop_is_not_called_intro():
    energy = np.array([0.95, 0.40, 0.20])
    bass = np.array([0.60, 0.30, 0.30])
    assert _label(energy, bass)[0] == DROP


def test_label_accepts_missing_slope():
    energy = np.array([0.2, 0.5, 1.0, 0.15])
    bass = np.array([0.3, 0.4, 0.6, 0.3])
    assert _label(energy, bass) == _label(energy, bass, np.zeros(4))
