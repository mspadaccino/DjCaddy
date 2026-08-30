import numpy as np

from core.analysis.waveform import _build, _colors, _scale


def test_scale_clips_to_unit_range():
    out = _scale(np.array([0.0, 1.0, 2.0, 100.0]))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_colors_map_bands_to_rgb():
    # 3 bin: solo bassi, solo medi, solo alti
    lo = np.array([1.0, 0.0, 0.0])
    mid = np.array([0.0, 1.0, 0.0])
    hi = np.array([0.0, 0.0, 1.0])
    assert _colors(lo, mid, hi) == ["#ff0000", "#00ff00", "#0000ff"]


def test_build_bass_vs_treble():
    # righe = frequenze; bassi(50,100) medi(500,1000) alti(3000,5000)
    freqs = np.array([50.0, 100.0, 500.0, 1000.0, 3000.0, 5000.0])
    T = 10
    S = np.zeros((6, T))
    S[0:2, :5] = 1.0    # prima metà: energia nei bassi
    S[4:6, 5:] = 1.0    # seconda metà: energia negli alti
    times = np.arange(T, dtype=float)

    t, amp, colors = _build(S, freqs, times, points=T)

    assert len(colors) == T
    assert np.allclose(amp, 1.0)          # ampiezza costante -> normalizzata a 1
    assert colors[0] == "#ff0000"         # bassi -> rosso
    assert colors[-1] == "#0000ff"        # alti -> blu
