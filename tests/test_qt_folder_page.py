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
from qt_app import theme
from qt_app.pages.common import ConfirmBar
from qt_app.pages.folder.duplicates_panel import _Section, duplicate_rows
from qt_app.state import AppState
from qt_app.widgets.track_table import CHECK_A_COLUMN, CHECK_B_COLUMN
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
    percorsi interi, non un nome più una cartella sola — e una casella per
    file, perché `keep` è una proposta e non una decisione."""
    table = duplicate_rows([group("/x/a.mp3", ["/y/a.mp3"])],
                           full_paths=True)
    assert list(table.columns[:4]) == [CHECK_A_COLUMN, "file A",
                                       CHECK_B_COLUMN, "file B"]
    assert table.at[0, "file A"] == str(Path("/x/a.mp3"))
    assert table.at[0, "file B"] == str(Path("/y/a.mp3"))
    assert table.at[0, "_path2"] == str(Path("/x/a.mp3"))
    assert table.at[0, "_path"] == str(Path("/y/a.mp3"))


def test_duplicate_rows_one_file_per_row_has_no_second_path():
    """Nel livello A i due file stanno nella stessa cartella e la scelta è
    una sola: niente `_path2`, niente seconda casella."""
    table = duplicate_rows([group("/x/a.mp3", ["/x/a copy.mp3"])])
    assert "_path2" not in table.columns
    assert CHECK_A_COLUMN not in table.columns


def test_either_file_of_a_pair_can_be_ticked(qtbot):
    """Il gesto che la sezione B e C devono permettere: mandare via
    l'originale invece della copia."""
    section = _Section(AppState(), "nota")
    qtbot.addWidget(section)
    section.set_rows(duplicate_rows([group("/x/a.mp3", ["/y/a.mp3"])],
                                    full_paths=True), preselect=False)
    assert section.chosen() == ([], 0)

    section.table.toggle_pick(0, "_path2")          # l'originale
    assert section.chosen() == ([Path("/x/a.mp3")], 1000)
    section.table.toggle_pick(0, "_path")           # e anche la copia
    assert section.chosen() == ([Path("/x/a.mp3"), Path("/y/a.mp3")], 2000)


def test_a_file_on_two_rows_is_counted_once(qtbot):
    """Un gruppo di tre copie mette l'originale su DUE righe: spuntarlo una
    volta non deve valere il doppio dello spazio liberato."""
    section = _Section(AppState(), "nota")
    qtbot.addWidget(section)
    section.set_rows(duplicate_rows([group("/x/a.mp3", ["/y/a.mp3",
                                                        "/z/a.mp3"])],
                                    full_paths=True), preselect=False)
    section.table.toggle_pick(0, "_path2")
    assert section.chosen() == ([Path("/x/a.mp3")], 1000)


def test_select_all_takes_the_copies_not_the_originals(qtbot):
    """Select all su una tabella a due file vuol dire "tutte le copie": se
    prendesse entrambe le colonne porterebbe via anche gli originali."""
    section = _Section(AppState(), "nota")
    qtbot.addWidget(section)
    section.set_rows(duplicate_rows([group("/x/a.mp3", ["/y/a.mp3"])],
                                    full_paths=True), preselect=False)
    section.table.set_all_picked(True)
    assert section.chosen() == ([Path("/y/a.mp3")], 1000)


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


# --- la conferma delle azioni che pesano ------------------------------------

def test_confirm_button_is_inert_until_the_box_is_ticked(qtbot):
    """Move to quarantine e Delete them non partono senza la spunta: è per
    questo che sembravano rotti — si cliccavano a vuoto."""
    bar = ConfirmBar("Move to quarantine", primary=True)
    qtbot.addWidget(bar)
    fired = []
    bar.activated.connect(lambda: fired.append(1))

    bar._button.click()
    assert fired == []
    bar._check.setChecked(True)
    bar._button.click()
    assert fired == [1]
    assert not bar._check.isChecked()      # la conferma vale per UN gesto


def test_a_disabled_action_button_looks_disabled():
    """Il foglio in linea del rosso vince su quello dell'app: senza la sua
    riga `:disabled` un bottone spento resta rosso pieno, e un clic che non
    fa niente sembra un bottone rotto."""
    assert "QPushButton:disabled" in theme.primary_button()
