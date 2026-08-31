"""Le parti pure della pagina Tag Qt: coda filtrata, righe, stime.

Come gli altri test Qt: girano solo col gruppo `qt` installato, offscreen,
e provano funzioni — i widget qui sono composizione già coperta altrove.
"""

import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6", reason="gruppo poetry `qt` non installato")
pytest.importorskip("pytestqt", reason="gruppo poetry `qt` non installato")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.analysis.essentia_tags import (CoverageReport, Prediction,
                                         TagCoverage, TagSettings, TrackTags)
from qt_app.pages.tag.page import filtered_coverage, queue_rows
from qt_app.pages.tag.run_panel import proposed_rows, seconds_each


def coverage() -> CoverageReport:
    return CoverageReport(items=[
        TagCoverage(Path("/x/full.mp3"), genre="House", comment="Happy"),
        TagCoverage(Path("/x/no_genre.mp3"), genre=None, comment="Dark"),
        TagCoverage(Path("/x/no_comment.mp3"), genre="Techno", comment=None),
        TagCoverage(Path("/x/bare.mp3"), genre=None, comment=None),
        TagCoverage(Path("/x/broken.mp3"), error="no reader opens it"),
    ])


# --- il filtro della coda ---------------------------------------------------

def test_filtered_coverage_follows_the_choice():
    report = coverage()
    names = lambda listed: [c.path.name for c in listed]     # noqa: E731

    assert names(filtered_coverage(report, "genre or comment")) == \
        ["no_genre.mp3", "no_comment.mp3", "bare.mp3"]
    assert names(filtered_coverage(report, "genre")) == \
        ["no_genre.mp3", "bare.mp3"]
    assert names(filtered_coverage(report, "comment")) == \
        ["no_comment.mp3", "bare.mp3"]
    assert names(filtered_coverage(report, "both")) == ["bare.mp3"]
    # "everything" elenca i leggibili, MAI i file coi tag illeggibili.
    assert names(filtered_coverage(report, "everything (no filter)")) == \
        ["full.mp3", "no_genre.mp3", "no_comment.mp3", "bare.mp3"]


def test_queue_rows_marks_what_is_missing():
    table = queue_rows(filtered_coverage(coverage(), "genre or comment"))
    assert list(table.columns) == ["file", "GENRE", "COMMENT", "folder",
                                   "_path"]
    assert table.at[0, "GENRE"] == "❌ missing"
    assert table.at[0, "COMMENT"] == "Dark"
    assert table.at[2, "GENRE"] == table.at[2, "COMMENT"] == "❌ missing"
    assert table.at[0, "_path"] == str(Path("/x/no_genre.mp3"))


def test_queue_rows_empty_keeps_columns():
    table = queue_rows([])
    assert list(table.columns) == ["file", "GENRE", "COMMENT", "folder",
                                   "_path"]
    assert len(table) == 0


# --- le righe proposte e le stime -------------------------------------------

def test_proposed_rows_follow_current_settings():
    tags = TrackTags(
        genres=[Prediction("Electronic---House", 0.82)],
        moods=[Prediction("happy", 0.44), Prediction("summer", 0.31)])
    analyzed = [(Path("/x/one.mp3"), tags)]

    rows = proposed_rows(analyzed, TagSettings())
    assert rows[0]["GENRE"] == "Electronic - House"
    assert rows[0]["COMMENT"] == "Happy; Summer"
    assert rows[0]["_path"] == str(Path("/x/one.mp3"))

    # Cambiare il formato riformatta SENZA rianalizzare: stessa analisi.
    rows = proposed_rows(analyzed, TagSettings(genre_format="child_only"))
    assert rows[0]["GENRE"] == "House"

    rows = proposed_rows(analyzed, TagSettings(moods=False))
    assert rows[0]["COMMENT"] == "—"


def test_seconds_each_snaps_to_the_nearest_measure():
    assert seconds_each(1) == 8.2
    assert seconds_each(5) == 4.1
    assert seconds_each(20) == 3.7      # oltre l'ultimo misurato: quello
