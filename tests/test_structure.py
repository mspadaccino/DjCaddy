import numpy as np

from analysis.structure import _checkerboard_kernel, _novelty


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
