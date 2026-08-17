"""Test della parte di tagging che NON richiede Essentia installato.

Scelta delle etichette, formattazione e scrittura nei tag si provano da
sole: i modelli servono solo a produrre le attivazioni, che qui sono
semplici liste di numeri.
"""

import shutil
from pathlib import Path

import pytest

from analysis.essentia_tags import (
    Prediction,
    TagSettings,
    TrackTags,
    build_tag_values,
    format_genre_tag,
    format_mood_tag,
    select_genres,
    select_moods,
    write_tags,
)

TEST_AUDIO = Path(__file__).resolve().parent.parent / "test_mp3"


# --- scelta delle etichette ------------------------------------------------

def test_select_genres_takes_the_strongest_above_threshold():
    labels = ["a", "b", "c", "d"]
    activations = [0.05, 0.60, 0.30, 0.20]
    chosen = select_genres(activations, labels, TagSettings(top_genres=2, genre_threshold=0.15))
    assert [(p.label, round(p.confidence, 2)) for p in chosen] == [("b", 0.6), ("c", 0.3)]


def test_select_genres_falls_back_to_the_best_when_none_pass():
    """Un brano un genere ce l'ha sempre: meglio il candidato migliore con la
    sua confidenza accanto che un campo vuoto."""
    chosen = select_genres([0.01, 0.04, 0.02], ["a", "b", "c"],
                           TagSettings(genre_threshold=0.9))
    assert len(chosen) == 1 and chosen[0].label == "b"


def test_select_genres_respects_top_n():
    activations = [0.9, 0.8, 0.7, 0.6]
    chosen = select_genres(activations, list("abcd"),
                           TagSettings(top_genres=2, genre_threshold=0.0))
    assert len(chosen) == 2


def test_select_moods_has_no_fallback():
    """Al contrario del genere: un brano può non avere un mood riconoscibile,
    e inventarne uno sarebbe peggio che lasciarlo vuoto."""
    assert select_moods([0.001, 0.002], ["happy", "sad"],
                        TagSettings(mood_threshold=0.5)) == []


def test_select_moods_sorted_and_capped():
    activations = [0.10, 0.90, 0.50, 0.70]
    chosen = select_moods(activations, list("abcd"),
                          TagSettings(mood_threshold=0.05, max_moods=3))
    assert [p.label for p in chosen] == ["b", "d", "c"]


# --- formattazione ---------------------------------------------------------

@pytest.mark.parametrize("style,expected", [
    ("parent_child", "Rock - Alternative Rock"),
    ("child_parent", "Alternative Rock - Rock"),
    ("child_only", "Alternative Rock"),
    ("raw", "Rock---Alternative Rock"),
])
def test_format_genre_tag(style, expected):
    assert format_genre_tag("Rock---Alternative Rock", style) == expected


def test_format_genre_tag_without_separator():
    assert format_genre_tag("Electronic", "child_only") == "Electronic"


def test_format_mood_tag_capitalises():
    assert format_mood_tag("happy dance") == "Happy Dance"


# --- cosa finisce nei tag --------------------------------------------------

def _tags():
    return TrackTags(
        genres=[Prediction("Rock---Alternative Rock", 0.42), Prediction("Pop", 0.21)],
        moods=[Prediction("happy", 0.8), Prediction("energetic", 0.6),
               Prediction("dark", 0.4), Prediction("calm", 0.2)],
    )


def test_build_tag_values_joins_and_formats():
    values = build_tag_values(_tags(), TagSettings())
    assert values.genre == "Rock - Alternative Rock; Pop"
    assert values.mood == "Happy; Energetic; Dark"        # moods_in_tag = 3
    assert "42.00%" in values.genre_confidence
    assert "happy: 80.00%" in values.mood_confidence


def test_build_tag_values_can_disable_confidence():
    values = build_tag_values(_tags(), TagSettings(confidence_tags=False))
    assert values.genre and values.genre_confidence is None
    assert values.mood and values.mood_confidence is None


def test_build_tag_values_respects_the_switches():
    values = build_tag_values(_tags(), TagSettings(genres=False))
    assert values.genre is None and values.mood is not None
    values = build_tag_values(_tags(), TagSettings(moods=False))
    assert values.mood is None and values.genre is not None


def test_build_tag_values_empty_when_nothing_predicted():
    values = build_tag_values(TrackTags(), TagSettings())
    assert values.genre is None and values.mood is None


# --- scrittura su file veri ------------------------------------------------

pytestmark_audio = pytest.mark.skipif(
    not TEST_AUDIO.is_dir() or not any(TEST_AUDIO.glob("*.mp3")),
    reason="serve un file audio in test_mp3/ (non versionato: ha il copyright)",
)


@pytestmark_audio
def test_write_tags_mp3_puts_mood_in_the_default_comment(tmp_path):
    """Il punto di tutto: djay Pro mostra SOLO il commento predefinito, non
    i frame COMM con descrizione."""
    from mutagen.id3 import ID3

    src = next(TEST_AUDIO.glob("*.mp3"))
    target = tmp_path / "track.mp3"
    shutil.copy(src, target)

    written = write_tags(target, _tags(), TagSettings(overwrite=True))
    tags = ID3(target)

    assert tags.getall("TCON")[0].text[0] == "Rock - Alternative Rock; Pop"
    default_comment = tags.getall("COMM:'':eng") or [
        f for f in tags.getall("COMM") if f.desc == ""]
    assert default_comment[0].text[0] == "Happy; Energetic; Dark"
    assert any("COMM(default)" in w for w in written)


@pytestmark_audio
def test_write_tags_mp3_keeps_an_existing_genre_unless_overwriting(tmp_path):
    from mutagen.id3 import ID3, TCON

    src = next(TEST_AUDIO.glob("*.mp3"))
    target = tmp_path / "track.mp3"
    shutil.copy(src, target)
    tags = ID3(target)
    tags.delall("TCON")
    tags.add(TCON(encoding=3, text="Il mio genere"))
    tags.save(target)

    write_tags(target, _tags(), TagSettings(overwrite=False))
    assert ID3(target).getall("TCON")[0].text[0] == "Il mio genere"

    write_tags(target, _tags(), TagSettings(overwrite=True))
    assert ID3(target).getall("TCON")[0].text[0] != "Il mio genere"


@pytestmark_audio
def test_write_tags_flac(tmp_path):
    import mutagen

    src = next(TEST_AUDIO.glob("*.flac"))
    target = tmp_path / "track.flac"
    shutil.copy(src, target)

    write_tags(target, _tags(), TagSettings(overwrite=True))
    audio = mutagen.File(target)
    assert audio["GENRE"][0] == "Rock - Alternative Rock; Pop"
    assert audio["MOOD"][0] == "Happy; Energetic; Dark"
    assert audio["COMMENT"][0] == "Happy; Energetic; Dark"


def test_write_tags_nothing_to_write_is_a_no_op(tmp_path):
    target = tmp_path / "track.mp3"
    target.write_bytes(b"non verra' aperto")
    assert write_tags(target, TrackTags(), TagSettings()) == []


def test_write_tags_refuses_an_unknown_format(tmp_path):
    with pytest.raises(ValueError):
        write_tags(tmp_path / "track.xyz", _tags(), TagSettings())
