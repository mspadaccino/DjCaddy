import numpy as np

from analysis.structure import (
    EDGE_GAP_BEATS,
    MIN_GAP_BEATS,
    _checkerboard_kernel,
    _fold,
    _keep_away_from_edges,
    _novelty,
    snap_to_local_beat,
)


def test_checkerboard_kernel_shape_and_antisymmetry():
    k = _checkerboard_kernel(4)
    assert k.shape == (9, 9)
    # Quadranti opposti hanno segno opposto (scacchiera)
    assert k[0, 0] > 0          # (-,-) -> segno +
    assert k[0, -1] < 0         # (-,+) -> segno -
    assert abs(k[4, 4]) < 1e-9  # centro (sign 0)


def test_novelty_peaks_at_block_change():
    # SSM a due blocchi: alta similarità dentro ogni metà, bassa fra le due.
    n = 40
    ssm = np.zeros((n, n))
    ssm[: n // 2, : n // 2] = 1.0
    ssm[n // 2:, n // 2:] = 1.0
    nov = _novelty(ssm, half=6)
    assert nov.shape == (n,)
    assert nov.max() == 1.0                 # normalizzata
    # Il picco di novelty cade vicino al confine fra i due blocchi
    assert abs(int(np.argmax(nov)) - n // 2) <= 2


# --- confini troppo vicini ai bordi del brano ------------------------------

def test_keep_away_from_edges_drops_boundary_at_track_start():
    """Il caso reale: il frame sintetico a 0 produceva un boundary a
    0.046 s, cioè una sezione di 46 ms — e uno degli 8 pad hot-cue sprecato
    su un doppione del cue a inizio brano."""
    assert _keep_away_from_edges([0, 1, 40, 100], last_beat_idx=200) == [40, 100]


def test_keep_away_from_edges_drops_sliver_at_track_end():
    assert _keep_away_from_edges([40, 199, 200], last_beat_idx=200) == [40]


def test_keep_away_from_edges_keeps_short_outro():
    """La soglia sui bordi è UNA battuta, non le 4 richieste fra due
    boundary: l'outro di un brano da club dura spesso pochi secondi e con
    la soglia piena sparirebbe — insieme al punto di mix-out."""
    assert EDGE_GAP_BEATS < MIN_GAP_BEATS
    kept = _keep_away_from_edges([4, 195], last_beat_idx=200)
    assert kept == [4, 195]


def test_keep_away_from_edges_on_very_short_track_can_drop_everything():
    assert _keep_away_from_edges([1, 6], last_beat_idx=7) == []


# --- aggancio al battito ---------------------------------------------------

def test_fold_finds_the_pulse_phase():
    """Un inviluppo con un colpo ogni mezzo secondo, sfasato di 0.1 s: il
    ripiegamento deve ritrovare periodo e fase, e segnalare contrasto alto."""
    times = np.arange(0, 60, 0.01)
    env = np.zeros_like(times)
    env[np.abs((times - 0.1) % 0.5) < 0.011] = 1.0
    contrast, phase = _fold(env, times, 0.5)
    assert abs(phase - 0.1) < 0.02
    assert contrast > 10


def test_fold_reports_low_contrast_without_a_pulse():
    """Senza pulsazione il ripiegamento è piatto: è questo che tiene fuori
    dall'aggancio i brani a tempo suonato. (Densità come nell'uso reale:
    con pochi campioni per bin anche il rumore fa picco.)"""
    times = np.arange(0, 300, 0.0116)          # ~5 min all'hop reale
    rng = np.random.default_rng(0)
    contrast, _ = _fold(rng.random(times.size), times, 0.5)
    assert contrast < 3.0


def test_snap_to_local_beat_pulls_onto_the_pulse():
    times = np.arange(0, 60, 0.005)
    env = np.zeros_like(times)
    env[np.abs((times - 0.1) % 0.5) < 0.006] = 1.0
    # 30.08 sta 20 ms dopo un battito (30.10 e' sulla griglia 0.1 + k*0.5)
    assert abs(snap_to_local_beat(30.08, 0.5, env, times) - 30.10) < 0.01


def test_snap_to_local_beat_keeps_the_nearest_beat_not_the_first():
    times = np.arange(0, 60, 0.005)
    env = np.zeros_like(times)
    env[np.abs((times - 0.1) % 0.5) < 0.006] = 1.0
    assert abs(snap_to_local_beat(10.62, 0.5, env, times) - 10.60) < 0.01


def test_snap_to_local_beat_is_a_no_op_without_data():
    assert snap_to_local_beat(5.0, 0.0, np.array([]), np.array([])) == 5.0
    assert snap_to_local_beat(5.0, 0.5, np.array([]), np.array([])) == 5.0
