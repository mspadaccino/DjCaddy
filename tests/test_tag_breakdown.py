"""Scomposizione dei tag: conta cosa separa e cosa no."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from analysis.tag_breakdown import as_text, build_breakdown, split_tag_value


@dataclass
class Voce:
    path: Path
    genre: str | None = None
    comment: str | None = None


def test_slash_and_ampersand_are_part_of_the_name_not_separators():
    """Il caso che rompe uno splitter scritto a intuito.

    Su 600 brani di libreria "/" compare nel 15% dei generi e "&" nel 4%,
    ma in "Funk / Soul" e "Brass & Military" — che sono nomi Discogs interi.
    Spezzarli inventerebbe generi inesistenti, e nei conteggi non si
    vedrebbe: uscirebbero solo "Funk" e "Soul" con numeri plausibili.
    """
    assert split_tag_value("Funk / Soul - Contemporary R&B") == [
        "Funk / Soul", "Contemporary R&B"]
    assert split_tag_value("Brass & Military - Military") == [
        "Brass & Military", "Military"]


def test_the_hierarchy_dash_must_be_spaced():
    """Senza spazi il trattino sta dentro al nome, non fra due livelli."""
    assert split_tag_value("Electronic - Hi-NRG") == ["Electronic", "Hi-NRG"]
    assert split_tag_value("Drum-n-Bass") == ["Drum-n-Bass"]


def test_one_tag_can_hold_several_genres_on_two_levels():
    assert split_tag_value("Electronic - House; Electronic - Tech House") == [
        "Electronic", "House", "Tech House"]          # "Electronic" una volta sola
    assert split_tag_value("Electronic - House", split_hierarchy=False) == [
        "Electronic - House"]


def test_empty_values_produce_nothing():
    assert split_tag_value(None) == []
    assert split_tag_value("   ") == []
    assert split_tag_value("; ;") == []


def test_counting_is_case_insensitive_but_keeps_the_first_spelling():
    """"House" e "house" sono la stessa cosa: due righe direbbero la stessa
    cosa a meta', e nessuna delle due darebbe il totale giusto."""
    voci = [Voce(Path("/a.mp3"), genre="Electronic - House"),
            Voce(Path("/b.mp3"), genre="electronic - HOUSE"),
            Voce(Path("/c.mp3"), genre=None)]

    b = build_breakdown(voci, "genre")

    assert b.counts["Electronic"] == 2 and b.counts["House"] == 2
    assert "house" not in b.counts and "HOUSE" not in b.counts
    assert b.tracks_with_any == 2 and b.tracks_without == 1
    assert b.tracks("House") == [Path("/a.mp3"), Path("/b.mp3")]
    assert b.rows()[0]["Tracks"] == 2


def test_the_export_is_one_path_per_line():
    testo = as_text([Path("/x/uno.mp3"), Path("/y/due.mp3")], "GENRE = House")
    righe = testo.splitlines()
    assert righe[0].startswith("#") and "House" in righe[0]
    assert righe[-2:] == ["/x/uno.mp3", "/y/due.mp3"]
