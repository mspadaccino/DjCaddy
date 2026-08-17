import csv

import pytest

from analysis.duplicates import (
    LEVEL_OTHER_FOLDER,
    LEVEL_SAME_FOLDER,
    LEVEL_SIMILAR_NAME,
    QUARANTINE_DIRNAME,
    apply_quarantine_plan,
    build_quarantine_plan,
    duplicates_of,
    choose_keeper,
    find_duplicates,
    name_quality,
    normalized_name,
    write_csv,
)
from analysis.folder_scan import format_of, human_size, scan_folder


def _write(path, content=b"audio"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _scan_audio(root):
    return scan_folder(root, audio_only=True).audio


# --- scansione -------------------------------------------------------------

def test_scan_counts_by_format_and_size(tmp_path):
    _write(tmp_path / "a.mp3", b"1234")
    _write(tmp_path / "sub" / "b.flac", b"12345")
    _write(tmp_path / "cover.jpg", b"1")
    scan = scan_folder(tmp_path)

    assert len(scan.files) == 3
    assert len(scan.audio) == 2
    assert scan.counts_by_format()["MP3"] == 1
    assert scan.counts_by_format()["OTHER"] == 1
    assert scan.total_size() == 10


def test_scan_audio_only_skips_other_files(tmp_path):
    _write(tmp_path / "a.mp3")
    _write(tmp_path / "notes.txt")
    assert len(scan_folder(tmp_path, audio_only=True).files) == 1


def test_format_of_is_case_insensitive():
    from pathlib import Path
    assert format_of(Path("x.MP3")) == "MP3"
    assert format_of(Path("x.zip")) == "OTHER"


def test_human_size():
    assert human_size(512) == "512 B"
    assert human_size(1024 * 1024 * 8.4).startswith("8.4 MB")


# --- normalizzazione e scelta del file da tenere ---------------------------

def test_normalized_name_ignores_punctuation(tmp_path):
    """Il caso reale: apostrofo vero contro trattino basso che lo sostituisce."""
    a = tmp_path / "O'NEIL& Dj Quba - What Is Love.mp3"
    b = tmp_path / "O_NEIL& Dj Quba - What Is Love.mp3"
    assert normalized_name(a) == normalized_name(b) == "o neil dj quba what is love"


def test_choose_keeper_prefers_the_real_apostrophe(tmp_path):
    clean = tmp_path / "O'NEIL& Dj Quba - What Is Love.mp3"
    ugly = tmp_path / "O_NEIL& Dj Quba - What Is Love.mp3"
    keep, dups = choose_keeper([ugly, clean])
    assert keep == clean and dups == [ugly]


def test_choose_keeper_avoids_copy_markers(tmp_path):
    original = tmp_path / "Track.mp3"
    copy = tmp_path / "Track (1).mp3"
    assert choose_keeper([copy, original])[0] == original


def test_choose_keeper_is_deterministic(tmp_path):
    """A parità di qualità l'esito non deve dipendere dall'ordine in cui il
    filesystem ha restituito i file."""
    a, b = tmp_path / "aaa.mp3", tmp_path / "bbb.mp3"
    assert choose_keeper([a, b]) == choose_keeper([b, a])


def test_name_quality_prefers_shallower_path(tmp_path):
    shallow = tmp_path / "x.mp3"
    deep = tmp_path / "a" / "b" / "x.mp3"
    assert name_quality(shallow) < name_quality(deep)


# --- i tre livelli ---------------------------------------------------------

def test_level_a_same_folder_identical_files(tmp_path):
    _write(tmp_path / "f" / "Track.mp3", b"same")
    _write(tmp_path / "f" / "Track (1).mp3", b"same")
    report = find_duplicates(_scan_audio(tmp_path))

    assert len(report.same_folder) == 1
    group = report.same_folder[0]
    assert group.level == LEVEL_SAME_FOLDER
    assert group.keep.name == "Track.mp3"
    assert [d.name for d in group.duplicates] == ["Track (1).mp3"]
    assert group.copies == 2 and group.wasted_bytes == 4
    assert report.other_folder == []


def test_level_b_other_folders_is_only_a_candidate(tmp_path):
    """Lo stesso brano in cartelle diverse è spesso voluto: deve finire in B,
    mai in A."""
    _write(tmp_path / "80s" / "Track.mp3", b"same")
    _write(tmp_path / "Workout" / "Track.mp3", b"same")
    report = find_duplicates(_scan_audio(tmp_path))

    assert report.same_folder == []
    assert len(report.other_folder) == 1
    assert report.other_folder[0].level == LEVEL_OTHER_FOLDER
    assert report.other_folder[0].copies == 2


def test_level_a_and_b_together(tmp_path):
    _write(tmp_path / "80s" / "Track.mp3", b"same")
    _write(tmp_path / "80s" / "Track copy.mp3", b"same")
    _write(tmp_path / "Workout" / "Track.mp3", b"same")
    report = find_duplicates(_scan_audio(tmp_path))

    assert len(report.same_folder) == 1          # la coppia dentro 80s/
    assert len(report.other_folder) == 1         # 80s/ contro Workout/
    # B ragiona per cartella, non per file: un rappresentante ciascuna
    assert report.other_folder[0].copies == 2


def test_level_c_similar_name_but_different_content(tmp_path):
    _write(tmp_path / "Track (Radio Edit).mp3", b"one")
    _write(tmp_path / "Track [Radio Edit].mp3", b"two-different")
    report = find_duplicates(_scan_audio(tmp_path))

    assert report.same_folder == [] and report.other_folder == []
    assert len(report.similar_name) == 1
    assert report.similar_name[0].level == LEVEL_SIMILAR_NAME
    assert report.similar_name[0].md5 is None


def test_identical_files_do_not_also_appear_as_similar_name(tmp_path):
    _write(tmp_path / "f" / "Track.mp3", b"same")
    _write(tmp_path / "f" / "Track (1).mp3", b"same")
    report = find_duplicates(_scan_audio(tmp_path))
    assert report.similar_name == []


def test_same_size_different_content_is_not_a_duplicate(tmp_path):
    _write(tmp_path / "a.mp3", b"aaaa")
    _write(tmp_path / "b.mp3", b"bbbb")
    report = find_duplicates(_scan_audio(tmp_path))
    assert report.same_folder == [] and report.other_folder == []


def test_only_same_size_files_get_hashed(tmp_path):
    """Due file di dimensione diversa non possono essere identici: leggerli
    per intero su una libreria vera costerebbe decine di gigabyte."""
    _write(tmp_path / "a.mp3", b"1")
    _write(tmp_path / "b.mp3", b"22")
    _write(tmp_path / "c.mp3", b"33")
    hashed = []

    def _spy(path):
        hashed.append(path)
        return "digest"

    find_duplicates(_scan_audio(tmp_path), hasher=_spy)
    assert sorted(p.name for p in hashed) == ["b.mp3", "c.mp3"]


def test_unreadable_file_is_reported_not_raised(tmp_path):
    _write(tmp_path / "a.mp3", b"same")
    _write(tmp_path / "b.mp3", b"same")

    def _boom(path):
        raise OSError("permesso negato")

    report = find_duplicates(_scan_audio(tmp_path), hasher=_boom)
    assert len(report.unreadable) == 2
    assert report.same_folder == []


# --- report CSV ------------------------------------------------------------

def test_csv_marks_only_same_folder_as_delete(tmp_path):
    _write(tmp_path / "f" / "Track.mp3", b"same")
    _write(tmp_path / "f" / "Track (1).mp3", b"same")
    _write(tmp_path / "other" / "Track.mp3", b"same")
    report = find_duplicates(_scan_audio(tmp_path))

    out = write_csv(report.all_groups(), tmp_path / "report.csv")
    rows = list(csv.DictReader(out.open(encoding="utf-8")))

    actions = {r["level"]: r["action"] for r in rows}
    assert actions[LEVEL_SAME_FOLDER] == "DELETE"
    assert actions[LEVEL_OTHER_FOLDER] == "REVIEW"
    row = next(r for r in rows if r["level"] == LEVEL_SAME_FOLDER)
    assert row["keep"].endswith("Track.mp3")
    assert row["delete"].endswith("Track (1).mp3")
    assert row["md5"] and row["size_bytes"] == "4"


# --- quarantena ------------------------------------------------------------

def test_quarantine_plan_mirrors_the_original_structure(tmp_path):
    _write(tmp_path / "DANCE RETRO" / "Track.mp3", b"same")
    _write(tmp_path / "DANCE RETRO" / "Track (1).mp3", b"same")
    report = find_duplicates(_scan_audio(tmp_path))

    plan = build_quarantine_plan(duplicates_of(report.same_folder), tmp_path)
    assert len(plan) == 1
    src, dest = plan[0]
    assert src.name == "Track (1).mp3"
    assert dest == tmp_path / QUARANTINE_DIRNAME / "DANCE RETRO" / "Track (1).mp3"


def test_quarantine_plan_disambiguates_equal_names(tmp_path):
    """Cartelle diverse possono avere lo stesso nome file: in quarantena non
    devono sovrascriversi a vicenda."""
    for folder in ("a", "b"):
        _write(tmp_path / folder / "Track.mp3", b"same")
        _write(tmp_path / folder / "Track copy.mp3", b"same")
    report = find_duplicates(_scan_audio(tmp_path))
    plan = build_quarantine_plan(duplicates_of(report.same_folder), tmp_path)

    assert len(plan) == 2
    assert len({dest for _, dest in plan}) == 2


def test_quarantine_moves_and_never_deletes(tmp_path):
    _write(tmp_path / "f" / "Track.mp3", b"same")
    dup = _write(tmp_path / "f" / "Track (1).mp3", b"same")
    report = find_duplicates(_scan_audio(tmp_path))
    plan = build_quarantine_plan(duplicates_of(report.same_folder), tmp_path)

    moved, errors = apply_quarantine_plan(plan, dry_run=False)
    assert moved == 1 and errors == []
    assert not dup.exists()                                  # spostato...
    assert plan[0][1].exists()                               # ...non cancellato
    assert (tmp_path / "f" / "Track.mp3").exists()            # l'originale resta


def test_quarantine_dry_run_touches_nothing(tmp_path):
    _write(tmp_path / "f" / "Track.mp3", b"same")
    dup = _write(tmp_path / "f" / "Track (1).mp3", b"same")
    report = find_duplicates(_scan_audio(tmp_path))
    plan = build_quarantine_plan(duplicates_of(report.same_folder), tmp_path)

    moved, errors = apply_quarantine_plan(plan)               # dry_run di default
    assert moved == 1 and errors == []
    assert dup.exists() and not plan[0][1].exists()


def test_quarantine_plan_skips_what_is_already_quarantined(tmp_path):
    q = tmp_path / QUARANTINE_DIRNAME / "f"
    _write(q / "Track.mp3", b"same")
    _write(q / "Track (1).mp3", b"same")
    report = find_duplicates(_scan_audio(tmp_path))
    assert build_quarantine_plan(duplicates_of(report.same_folder), tmp_path) == []


def test_similar_name_survives_when_one_file_is_a_copy_of_a_third(tmp_path):
    """Trovato provando su dati veri: due nomi simili con contenuto diverso
    sparivano dal livello C solo perché uno dei due era identico a un terzo
    file in un'altra cartella. La somiglianza dentro la coppia resta
    un'informazione valida."""
    _write(tmp_path / "80s" / "Some Track (Radio Edit).mp3", b"one")
    _write(tmp_path / "80s" / "Some Track [Radio Edit].mp3", b"different")
    _write(tmp_path / "Workout" / "Some Track (Radio Edit).mp3", b"one")
    report = find_duplicates(_scan_audio(tmp_path))

    assert len(report.other_folder) == 1          # la copia fra cartelle c'e'
    assert len(report.similar_name) == 1          # ...e la coppia simile pure
    names = {report.similar_name[0].keep.name,
             *(d.name for d in report.similar_name[0].duplicates)}
    assert names == {"Some Track (Radio Edit).mp3", "Some Track [Radio Edit].mp3"}


def test_duplicates_of_counts_files_not_groups(tmp_path):
    """Il fraintendimento da cui e' nata la modifica: un gruppo con tre copie
    e' UN gruppo ma DUE file da spostare, per cui i due totali non
    coincidono e vanno mostrati distinti."""
    for name in ("Track.mp3", "Track (1).mp3", "Track (2).mp3"):
        _write(tmp_path / "f" / name, b"same")
    _write(tmp_path / "g" / "Other.mp3", b"pair")
    _write(tmp_path / "g" / "Other copy.mp3", b"pair")
    report = find_duplicates(_scan_audio(tmp_path))

    assert len(report.same_folder) == 2                  # gruppi
    assert len(duplicates_of(report.same_folder)) == 3   # file da spostare


def test_quarantine_plan_takes_only_the_selected_paths(tmp_path):
    """La selezione la fa l'utente riga per riga: il piano deve limitarsi a
    quello che gli viene passato."""
    for name in ("Track.mp3", "Track (1).mp3", "Track (2).mp3"):
        _write(tmp_path / "f" / name, b"same")
    report = find_duplicates(_scan_audio(tmp_path))
    everything = duplicates_of(report.same_folder)

    plan = build_quarantine_plan(everything[:1], tmp_path)
    assert len(plan) == 1 and plan[0][0] == everything[0]


# --- file AppleDouble di macOS ---------------------------------------------

def test_appledouble_files_are_not_audio(tmp_path):
    """Su exFAT il Finder affianca a ogni brano un "._<nome>" da 4 KB con gli
    attributi estesi. Ha l'estensione del brano ma non e' audio."""
    from analysis.folder_scan import APPLEDOUBLE, format_of

    assert format_of(tmp_path / "._Track.mp3") == APPLEDOUBLE
    assert format_of(tmp_path / "Track.mp3") == "MP3"

    _write(tmp_path / "Track.mp3", b"vero")
    _write(tmp_path / "._Track.mp3", b"\x00\x05\x16\x07Mac OS X")
    scan = scan_folder(tmp_path)
    assert len(scan.files) == 2 and len(scan.audio) == 1
    assert scan.counts_by_format()[APPLEDOUBLE] == 1


def test_appledouble_files_never_become_duplicates(tmp_path):
    """Sono tutti della stessa dimensione e spesso identici: senza escluderli
    inonderebbero il report di duplicati falsi."""
    sidecar = b"\x00\x05\x16\x07Mac OS X" + b"\x00" * 100
    for folder in ("a", "b"):
        _write(tmp_path / folder / "Track.mp3", f"brano {folder}".encode())
        _write(tmp_path / folder / "._Track.mp3", sidecar)

    report = find_duplicates(_scan_audio(tmp_path))
    assert report.same_folder == [] and report.other_folder == []
    assert all("._" not in g.keep.name for g in report.all_groups())


def test_appledouble_excluded_from_audio_only_scan(tmp_path):
    _write(tmp_path / "Track.mp3", b"vero")
    _write(tmp_path / "._Track.mp3", b"sidecar")
    assert len(scan_folder(tmp_path, audio_only=True).files) == 1
