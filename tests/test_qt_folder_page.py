"""Le parti pure della pagina Folder Qt: righe dei duplicati e filtro.

Come gli altri test Qt: solo col gruppo `qt` installato, offscreen, e su
funzioni — la meccanica di scansioni e quarantena è già coperta dai test
di core (`test_duplicates`, `test_folder_scan`).
"""

import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6", reason="gruppo poetry `qt` non installato")
pytest.importorskip("pytestqt", reason="gruppo poetry `qt` non installato")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.analysis.duplicates import LEVEL_SAME_FOLDER, DuplicateGroup
from core.analysis.folder_scan import ScannedFile, TrackDuration
from qt_app.pages.folder.contents import listed_rows
from qt_app.pages.folder.duplicates_panel import duplicate_rows
from qt_app.pages.folder.filtering_panel import filter_reasons


def group(keep: str, duplicates: list[str],
          size: int = 1000) -> DuplicateGroup:
    return DuplicateGroup(level=LEVEL_SAME_FOLDER, size=size,
                          md5="abcdef0123456789", keep=Path(keep),
                          duplicates=[Path(d) for d in duplicates])


# --- le righe dei duplicati -------------------------------------------------

def test_duplicate_rows_one_per_extra_copy():
    table = duplicate_rows([group("/x/a.mp3", ["/x/a copy.mp3",
                                               "/x/a(2).mp3"])])
    assert list(table.columns[:3]) == ["folder", "keep", "duplicate"]
    assert len(table) == 2                      # un gruppo, DUE file di troppo
    assert set(table["duplicate"]) == {"a copy.mp3", "a(2).mp3"}
    assert (table["keep"] == "a.mp3").all()
    assert (table["copies"] == 3).all()
    assert (table["md5"] == "abcdef012345").all()      # tagliato a 12


def test_duplicate_rows_full_paths_for_other_folders():
    """Nei livelli B e C le copie stanno in cartelle DIVERSE: servono i
    percorsi interi, non un nome più una cartella sola."""
    table = duplicate_rows([group("/x/a.mp3", ["/y/a.mp3"])],
                           full_paths=True)
    assert list(table.columns[:2]) == ["stays", "moves if ticked"]
    assert table.at[0, "stays"] == str(Path("/x/a.mp3"))
    assert table.at[0, "moves if ticked"] == str(Path("/y/a.mp3"))
    assert table.at[0, "_path"] == str(Path("/y/a.mp3"))


def test_duplicate_rows_empty_keeps_columns():
    assert len(duplicate_rows([])) == 0
    assert "duplicate" in duplicate_rows([]).columns


# --- il filtro della libreria -----------------------------------------------

def tracks():
    return [
        TrackDuration(Path("/x/megamix in the mix.mp3"), 100, 15 * 60.0),
        TrackDuration(Path("/x/long plain set.mp3"), 90, 12 * 60.0),
        TrackDuration(Path("/x/ordinary song.mp3"), 10, 4 * 60.0),
        TrackDuration(Path("/x/a vs b - one vs two.mp3"), 10, 4 * 60.0),
        TrackDuration(Path("/x/fragment.mp3"), 1, 30.0),
    ]


def test_filter_reasons_both_signals():
    reasons, by_length, by_name = filter_reasons(
        tracks(), 10, 30, ["megamix"], vs_rule=True,
        rule="both signals fire")
    assert set(reasons) == {Path("/x/megamix in the mix.mp3")}
    assert reasons[Path("/x/megamix in the mix.mp3")] == \
        ["10–30 min", "name: megamix"]
    assert by_length == 2                       # i due file lunghi
    assert by_name == 2                         # megamix + il doppio "vs"


def test_filter_reasons_either_reaches_the_mashup():
    reasons, _, _ = filter_reasons(
        tracks(), 10, 30, ["megamix"], vs_rule=True,
        rule="either one fires")
    assert Path("/x/a vs b - one vs two.mp3") in reasons
    assert reasons[Path("/x/a vs b - one vs two.mp3")] == ["two “vs”"]
    assert Path("/x/long plain set.mp3") in reasons     # lungo, senza nome


def test_filter_reasons_duration_alone_reaches_the_short_end():
    reasons, _, _ = filter_reasons(
        tracks(), 0, 1, ["megamix"], vs_rule=True,
        rule="the duration alone")
    assert set(reasons) == {Path("/x/fragment.mp3")}
    assert reasons[Path("/x/fragment.mp3")] == ["0–1 min"]


# --- l'elenco per estensione ------------------------------------------------

def test_listed_rows_columns_and_sizes():
    table = listed_rows([
        ScannedFile(Path("/x/cover.jpg"), 2048, "OTHER", mtime=0.0)])
    assert table.at[0, "file"] == "cover.jpg"
    assert table.at[0, "size"] == "2.0 KB"
    assert table.at[0, "modified"] == "—"
    assert table.at[0, "_bytes"] == 2048
