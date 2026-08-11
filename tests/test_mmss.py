from analysis.models import format_elapsed, parse_mmss


def test_parse_mmss_roundtrip():
    assert parse_mmss("01:07.3") == 67.3
    assert parse_mmss("1:07.3") == 67.3
    assert parse_mmss("0:00.0") == 0.0
    assert abs(parse_mmss(format_elapsed(70.291)) - 70.3) < 0.05


def test_parse_mmss_invalid():
    assert parse_mmss("not a time") is None
    assert parse_mmss("1:75") is None      # secondi >= 60
    assert parse_mmss("") is None
