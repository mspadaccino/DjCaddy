"""Canzone interrotta o file corto voluto: il verdetto e le sue prove."""

from __future__ import annotations

from pathlib import Path

from analysis.truncation import ShortFileVerdict


def _v(**kw) -> ShortFileVerdict:
    return ShortFileVerdict(path=Path("/x.mp3"), **kw)


def test_a_sharp_ending_alone_is_not_enough():
    """Un estratto tagliato di netto finisce come un download interrotto.

    Misurato: troncata 1,25 contro taglio netto a 30s 1,16 — la coda da
    sola non li separa, ed e' il motivo per cui serve il secondo segnale.
    """
    netto = _v(ending=1.16)
    assert netto.ends_abruptly
    assert not netto.claims_to_be_a_song
    assert not netto.truncated
    assert netto.verdict == "ends sharp, untagged"


def test_sharp_ending_plus_full_song_tags_means_cut_off():
    """I tag ID3v2 stanno in testa al file, quindi un download interrotto
    conserva artista e titolo: dichiara di essere un brano intero pur non
    durandone la meta'."""
    v = _v(ending=1.25, artist="Bananarama", title="Venus", album="Deep Sea Skiving")
    assert v.truncated and v.verdict == "cut off"


def test_a_file_that_dies_away_is_finished_whatever_its_tags():
    """Chi si spegne e' finito apposta: nessun tag lo rende un troncamento."""
    for tags in ({}, {"artist": "Tizio", "title": "Caio"}):
        v = _v(ending=0.07, **tags)
        assert not v.ends_abruptly
        assert not v.truncated
        assert v.verdict == "finished"


def test_half_a_title_is_not_a_song():
    """Serve artista E titolo: uno solo lo lasciano dietro anche i sample."""
    assert not _v(ending=1.2, artist="Tizio").claims_to_be_a_song
    assert not _v(ending=1.2, title="Caio").claims_to_be_a_song
    assert _v(ending=1.2, artist="Tizio", title="Caio").claims_to_be_a_song


def test_what_could_not_be_read_is_never_called_truncated():
    v = _v(ending=None, artist="Tizio", title="Caio", error="audio: boom")
    assert not v.ends_abruptly and not v.truncated
    assert v.verdict == "unreadable"
