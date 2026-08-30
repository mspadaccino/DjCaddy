from core.analysis.engine import load_analysis, save_analysis, sidecar_path
from core.analysis.models import Section, TrackAnalysis


def test_sidecar_roundtrip(tmp_path):
    audio = tmp_path / "song.flac"
    audio.write_bytes(b"x")
    t = TrackAnalysis(
        path=audio, genre="Dance", bpm=124.0, rms=0.1, duration=200.0,
        sections=[Section(0, 32, "Intro", 0.2, 16.0)],
        vocal_regions=[(10.0, 20.0)],
    )
    p = save_analysis(t)
    assert p == sidecar_path(audio)
    assert p.name == "song_analysis.json"

    loaded = load_analysis(audio)
    assert loaded is not None
    assert loaded.bpm == 124.0
    assert loaded.vibe == "Peak-Time"          # bucket ricalcolato al load
    assert [s.label for s in loaded.sections] == ["Intro"]
    assert loaded.vocal_regions == [(10.0, 20.0)]


def test_load_missing(tmp_path):
    assert load_analysis(tmp_path / "nope.flac") is None
