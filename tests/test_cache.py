from analysis.cache import AnalysisCache


def _sample_data():
    return {
        "genre": "House",
        "bpm": 124.0,
        "rms": 0.12,
        "boundaries": [{"time": 32.0, "confidence": 0.8, "label": "Rise"}],
        "error": None,
    }


def test_cache_roundtrip_and_invalidation(tmp_path):
    audio = tmp_path / "track.mp3"
    audio.write_bytes(b"fake-audio")
    cache_path = tmp_path / "cache.json"

    cache = AnalysisCache.load(cache_path)
    assert cache.get(audio) is None          # miss iniziale

    cache.put(audio, _sample_data())
    cache.save()

    reloaded = AnalysisCache.load(cache_path)
    assert reloaded.get(audio) == _sample_data()   # hit dopo persistenza

    # Modifica del file -> invalidazione (mtime/size cambiano)
    audio.write_bytes(b"fake-audio-modificato-piu-lungo")
    assert reloaded.get(audio) is None


def test_cache_missing_file(tmp_path):
    cache = AnalysisCache.load(tmp_path / "cache.json")
    assert cache.get(tmp_path / "inesistente.mp3") is None
