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
from core.analysis.folder_scan import ScannedFile, TrackDuration, human_size
from qt_app.pages.folder.contents import listed_rows
from qt_app import theme
from qt_app.pages.common import ConfirmBar
from qt_app.pages.folder.duplicates_panel import (_Section, duplicate_rows,
                                                   files_and_bytes)
from qt_app.state import AppState
from qt_app.widgets.track_table import (CHECK_A_COLUMN, CHECK_B_COLUMN,
                                        PLAY_A_COLUMN, PLAY_B_COLUMN)
from qt_app.pages.folder.filtering_panel import filter_reasons


def group(keep: str, duplicates: list[str], size: int = 1000,
          file_sizes: dict[str, int] | None = None,
          file_hashes: dict[str, str | None] | None = None) -> DuplicateGroup:
    return DuplicateGroup(
        level=LEVEL_SAME_FOLDER, size=size, md5="abcdef0123456789",
        keep=Path(keep), duplicates=[Path(d) for d in duplicates],
        file_sizes={Path(k): v for k, v in (file_sizes or {}).items()},
        file_hashes={Path(k): v for k, v in (file_hashes or {}).items()})


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
    assert list(table.columns[:6]) == [
        CHECK_A_COLUMN, PLAY_A_COLUMN, "file A",
        CHECK_B_COLUMN, PLAY_B_COLUMN, "file B"]
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


def test_duplicate_rows_compare_columns_come_last():
    """Le colonne del confronto — livello C, dove i due file NON sono
    byte-identici — stanno in fondo, size A e size B vicine per il
    paragone a vista."""
    table = duplicate_rows(
        [group("/x/a.mp3", ["/y/a.mp3"],
              file_sizes={"/x/a.mp3": 1000, "/y/a.mp3": 2000},
              file_hashes={"/x/a.mp3": "aaa", "/y/a.mp3": "bbb"})],
        full_paths=True, compare=True)
    visible = [c for c in table.columns if not c.startswith("_")]
    assert visible[-5:] == ["size A", "size B", "same size", "same name",
                            "same hash"]


def test_duplicate_rows_compare_drops_the_redundant_group_size():
    """"size" è quella (unica) del gruppo — con size A/size B per file non
    aggiunge niente, e nei gruppi disomogenei è pure fuorviante. Sparisce
    solo con `compare`: nei livelli A e B resta l'unica size che c'è."""
    table = duplicate_rows(
        [group("/x/a.mp3", ["/y/a.mp3"],
              file_sizes={"/x/a.mp3": 1000, "/y/a.mp3": 2000},
              file_hashes={"/x/a.mp3": "aaa", "/y/a.mp3": "bbb"})],
        full_paths=True, compare=True)
    assert "size" not in table.columns

    level_b = duplicate_rows([group("/x/a.mp3", ["/y/a.mp3"])],
                             full_paths=True)
    assert "size" in level_b.columns


def test_duplicate_rows_compare_different_sizes_skips_the_hash():
    """Il caso vero del livello C: stesso nome, contenuto diverso. Le
    dimensioni bastano a dire che non sono lo stesso file — l'hash non va
    nemmeno guardato."""
    table = duplicate_rows(
        [group("/x/a.mp3", ["/y/a.mp3"],
              file_sizes={"/x/a.mp3": 1000, "/y/a.mp3": 2000},
              file_hashes={"/x/a.mp3": "aaa", "/y/a.mp3": "bbb"})],
        full_paths=True, compare=True)
    row = table.iloc[0]
    assert row["size A"] == human_size(1000)
    assert row["size B"] == human_size(2000)
    assert not row["same size"]
    assert not row["same hash"]
    assert row["same name"]                     # stesso nome "a.mp3"


def test_duplicate_rows_compare_same_size_different_hash():
    """Stessa dimensione ma contenuto diverso: decide l'hash, non la size —
    è esattamente il caso dei due "ditty - paperboy" con lo stesso nome."""
    table = duplicate_rows(
        [group("/x/a.mp3", ["/y/a.mp3"],
              file_sizes={"/x/a.mp3": 1000, "/y/a.mp3": 1000},
              file_hashes={"/x/a.mp3": "aaa", "/y/a.mp3": "bbb"})],
        full_paths=True, compare=True)
    row = table.iloc[0]
    assert row["same size"]
    assert not row["same hash"]


def test_duplicate_rows_compare_missing_hash_is_not_claimed_equal():
    """Se l'hash di un file non è mai stato calcolato (dimensione unica nel
    resto della libreria), "same hash" non deve dire True per mancanza di
    prova contraria."""
    table = duplicate_rows(
        [group("/x/a.mp3", ["/y/a.mp3"],
              file_sizes={"/x/a.mp3": 1000, "/y/a.mp3": 1000},
              file_hashes={"/x/a.mp3": None, "/y/a.mp3": None})],
        full_paths=True, compare=True)
    assert not table.iloc[0]["same hash"]


def test_duplicate_rows_without_compare_has_no_extra_columns():
    """Il livello B non ha chiesto queste colonne: `compare` di default
    resta spento e non le aggiunge."""
    table = duplicate_rows([group("/x/a.mp3", ["/y/a.mp3"])],
                           full_paths=True)
    assert "size A" not in table.columns
    assert "same hash" not in table.columns


def test_duplicate_rows_compare_shows_every_pair_in_a_trio():
    """Un gruppo di TRE file dallo stesso nome (non solo due): la tabella
    deve portare ogni coppia, non solo ciascuno contro il presunto
    originale — altrimenti "y" uguale a "z" non si vedrebbe mai, solo che
    entrambi sono diversi da "x"."""
    table = duplicate_rows(
        [group("/x/a.mp3", ["/y/a.mp3", "/z/a.mp3"],
              file_sizes={"/x/a.mp3": 1000, "/y/a.mp3": 1000,
                         "/z/a.mp3": 2000},
              file_hashes={"/x/a.mp3": "aaa", "/y/a.mp3": "aaa",
                          "/z/a.mp3": "ccc"})],
        full_paths=True, compare=True)
    assert len(table) == 3                        # le 3 coppie di un trio
    pairs = set(zip(table["file A"], table["file B"]))
    assert pairs == {
        (str(Path("/x/a.mp3")), str(Path("/y/a.mp3"))),
        (str(Path("/x/a.mp3")), str(Path("/z/a.mp3"))),
        (str(Path("/y/a.mp3")), str(Path("/z/a.mp3"))),
    }
    xy = table[(table["file A"] == str(Path("/x/a.mp3")))
              & (table["file B"] == str(Path("/y/a.mp3")))].iloc[0]
    assert xy["same hash"]                        # x e y sono lo stesso file
    xz = table[(table["file A"] == str(Path("/x/a.mp3")))
              & (table["file B"] == str(Path("/z/a.mp3")))].iloc[0]
    assert not xz["same size"]                    # z è più grande


def test_files_and_bytes_does_not_inflate_the_smaller_file():
    """Il livello C può accostare due file di size diversa nella stessa
    riga: prima del fix entrambi pesavano come il più grande del gruppo
    (`g.size`), sovrastimando lo spazio liberato dal più piccolo."""
    table = duplicate_rows(
        [group("/x/a.mp3", ["/y/a.mp3"],
              file_sizes={"/x/a.mp3": 1000, "/y/a.mp3": 5000},
              file_hashes={"/x/a.mp3": "aaa", "/y/a.mp3": "bbb"})],
        full_paths=True, compare=True)
    _, total = files_and_bytes(table)
    assert total == 1000 + 5000                  # non 2 * 5000 (il max)


def test_files_and_bytes_gives_each_file_of_a_trio_its_own_size():
    """Stesso bug, versione a tre file: x e y pesano 1000, z pesa 2000 —
    prima del fix tutti e tre avrebbero pesato 2000 (il max del gruppo),
    sovrastimando il totale di 2000 byte."""
    table = duplicate_rows(
        [group("/x/a.mp3", ["/y/a.mp3", "/z/a.mp3"],
              file_sizes={"/x/a.mp3": 1000, "/y/a.mp3": 1000,
                         "/z/a.mp3": 2000},
              file_hashes={"/x/a.mp3": "aaa", "/y/a.mp3": "aaa",
                          "/z/a.mp3": "ccc"})],
        full_paths=True, compare=True)
    listed, total = files_and_bytes(table)
    assert set(listed) == {str(Path("/x/a.mp3")), str(Path("/y/a.mp3")),
                           str(Path("/z/a.mp3"))}
    assert total == 1000 + 1000 + 2000


def test_files_and_bytes_unaffected_for_byte_identical_levels():
    """Nei livelli A e B i due file di una riga sono byte-identici per
    costruzione: `_bytes2` non aggiunge nulla di nuovo lì, resta `g.size`
    come prima del fix."""
    table = duplicate_rows([group("/x/a.mp3", ["/y/a.mp3"], size=4000)],
                           full_paths=True)          # niente compare: livello B
    _, total = files_and_bytes(table)
    assert total == 4000 + 4000


def test_select_matching_button_only_shows_where_it_means_something(qtbot):
    """"Select same size + name" ha senso solo dove ci sono quelle colonne
    — il livello Similar. Su A e B, senza `compare`, deve restare nascosto:
    `isHidden()` legge lo stato esplicito del widget, non serve mostrare
    la finestra per verificarlo offscreen."""
    section = _Section(AppState(), "nota")
    qtbot.addWidget(section)
    assert section._match.isHidden()             # niente righe caricate

    section.set_rows(duplicate_rows([group("/x/a.mp3", ["/y/a.mp3"])],
                                    full_paths=True), preselect=False)
    assert section._match.isHidden()              # livello B: niente compare

    section.set_rows(duplicate_rows(
        [group("/x/a.mp3", ["/y/a.mp3"],
              file_sizes={"/x/a.mp3": 1000, "/y/a.mp3": 1000},
              file_hashes={"/x/a.mp3": "aaa", "/y/a.mp3": "bbb"})],
        full_paths=True, compare=True), preselect=False)
    assert not section._match.isHidden()           # livello Similar: c'è


def test_select_matching_picks_only_same_size_and_name_pairs(qtbot):
    """Su tre coppie — uguali per size e nome, uguali solo per nome,
    diverse — il bottone deve prendere SOLO la prima."""
    section = _Section(AppState(), "nota")
    qtbot.addWidget(section)
    groups = [
        group("/x/a.mp3", ["/y/a.mp3"],           # stessa size, stesso nome
              file_sizes={"/x/a.mp3": 1000, "/y/a.mp3": 1000},
              file_hashes={"/x/a.mp3": "aaa", "/y/a.mp3": "bbb"}),
        group("/p/b.mp3", ["/q/b (2024 remaster).mp3"],   # size diversa
              file_sizes={"/p/b.mp3": 1000, "/q/b (2024 remaster).mp3": 2000},
              file_hashes={"/p/b.mp3": "ccc",
                          "/q/b (2024 remaster).mp3": "ddd"}),
    ]
    section.set_rows(duplicate_rows(groups, full_paths=True, compare=True),
                     preselect=False)
    section._select_matching()
    assert section.chosen() == ([Path("/y/a.mp3")], 1000)


def test_select_matching_replaces_the_previous_pick(qtbot):
    """Come Select all/none: rimpiazza la scelta di prima, non la somma."""
    section = _Section(AppState(), "nota")
    qtbot.addWidget(section)
    section.set_rows(duplicate_rows(
        [group("/x/a.mp3", ["/y/a.mp3"],
              file_sizes={"/x/a.mp3": 1000, "/y/a.mp3": 1000},
              file_hashes={"/x/a.mp3": "aaa", "/y/a.mp3": "bbb"})],
        full_paths=True, compare=True), preselect=False)
    section.table.toggle_pick(0, "_path2")        # sceglie "a mano" file A
    assert section.chosen() == ([Path("/x/a.mp3")], 1000)
    section._select_matching()
    assert section.chosen() == ([Path("/y/a.mp3")], 1000)   # non più A + B


def test_select_matching_picks_a_match_even_if_a_third_file_does_not_match(qtbot):
    """"Stessa size e nome" è un'uguaglianza, quindi transitiva: se x e y
    combaciano, un terzo file z diverso da entrambi non invalida la coppia
    x/y — resta lo stesso file, va comunque scelto. (Bug reale: un fix
    precedente escludeva l'intero trio in questo caso, lasciando fuori una
    copia che l'utente si aspettava spuntata.)"""
    section = _Section(AppState(), "nota")
    qtbot.addWidget(section)
    trio = group("/x/song.mp3", ["/y/song.mp3", "/z/song (remaster).mp3"],
                file_sizes={"/x/song.mp3": 1000, "/y/song.mp3": 1000,
                           "/z/song (remaster).mp3": 2000},
                file_hashes={"/x/song.mp3": "aaa", "/y/song.mp3": "aaa",
                            "/z/song (remaster).mp3": "ccc"})
    section.set_rows(duplicate_rows([trio], full_paths=True, compare=True),
                     preselect=False)
    section._select_matching()
    assert section.chosen() == ([Path("/y/song.mp3")], 1000)


def test_select_matching_picks_a_fully_consistent_trio(qtbot):
    """Quando OGNI coppia del trio soddisfa il criterio (stesso nome, stessa
    size su tutte e tre), niente da escludere: entrambe le copie in più si
    prendono, come farebbe l'utente a mano."""
    section = _Section(AppState(), "nota")
    qtbot.addWidget(section)
    trio = group("/x/song.mp3", ["/y/song.mp3", "/z/song.mp3"],
                file_sizes={"/x/song.mp3": 1000, "/y/song.mp3": 1000,
                           "/z/song.mp3": 1000},
                file_hashes={"/x/song.mp3": "aaa", "/y/song.mp3": "bbb",
                            "/z/song.mp3": "ccc"})
    section.set_rows(duplicate_rows([trio], full_paths=True, compare=True),
                     preselect=False)
    section._select_matching()
    chosen, freed = section.chosen()
    assert set(chosen) == {Path("/y/song.mp3"), Path("/z/song.mp3")}
    assert freed == 2000


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
