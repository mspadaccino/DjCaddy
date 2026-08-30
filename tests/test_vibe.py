from core.analysis.vibe import (
    bpm_to_tempo_bucket,
    build_energy_labeler,
    compute_vibe,
)


def test_tempo_buckets_boundaries():
    assert bpm_to_tempo_bucket(90) == "Warm-Up"
    assert bpm_to_tempo_bucket(100) == "Groove"     # estremo incluso a sinistra
    assert bpm_to_tempo_bucket(117.9) == "Groove"
    assert bpm_to_tempo_bucket(118) == "Peak-Time"
    assert bpm_to_tempo_bucket(127) == "Peak-Time"
    assert bpm_to_tempo_bucket(128) == "High-Energy-Tempo"
    assert bpm_to_tempo_bucket(None) == "Unknown-Tempo"


def test_energy_labeler_percentiles():
    rms = [float(x) for x in range(1, 101)]  # 1..100
    labeler = build_energy_labeler(rms)
    assert labeler(5) == "Low"       # sotto il 33° percentile
    assert labeler(50) == "Mid"
    assert labeler(95) == "High"     # sopra il 66° percentile
    assert labeler(None) == "Unknown-Energy"


def test_energy_labeler_empty():
    labeler = build_energy_labeler([])
    assert labeler(1.0) == "Unknown-Energy"


def test_compute_vibe():
    assert compute_vibe("Peak-Time", "High") == "Peak-Time-High"
