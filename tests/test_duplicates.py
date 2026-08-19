import csv
from pathlib import Path

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


def _find(root, **kw):
    """Duplicati con la verifica audio SPENTA: questi test guardano il
    raggruppamento, e i loro contenuti finti non sono audio vero. La verifica
    ha i suoi test dedicati più sotto."""
    kw.setdefault("verify_audio", False)
    return find_duplicates(_scan_audio(root), **kw)


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
    report = _find(tmp_path)

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
    report = _find(tmp_path)

    assert report.same_folder == []
    assert len(report.other_folder) == 1
    assert report.other_folder[0].level == LEVEL_OTHER_FOLDER
    assert report.other_folder[0].copies == 2


def test_level_a_and_b_together(tmp_path):
    _write(tmp_path / "80s" / "Track.mp3", b"same")
    _write(tmp_path / "80s" / "Track copy.mp3", b"same")
    _write(tmp_path / "Workout" / "Track.mp3", b"same")
    report = _find(tmp_path)

    assert len(report.same_folder) == 1          # la coppia dentro 80s/
    assert len(report.other_folder) == 1         # 80s/ contro Workout/
    # B ragiona per cartella, non per file: un rappresentante ciascuna
    assert report.other_folder[0].copies == 2


def test_level_c_similar_name_but_different_content(tmp_path):
    _write(tmp_path / "Track (Radio Edit).mp3", b"one")
    _write(tmp_path / "Track [Radio Edit].mp3", b"two-different")
    report = _find(tmp_path)

    assert report.same_folder == [] and report.other_folder == []
    assert len(report.similar_name) == 1
    assert report.similar_name[0].level == LEVEL_SIMILAR_NAME
    assert report.similar_name[0].md5 is None


def test_identical_files_do_not_also_appear_as_similar_name(tmp_path):
    _write(tmp_path / "f" / "Track.mp3", b"same")
    _write(tmp_path / "f" / "Track (1).mp3", b"same")
    report = _find(tmp_path)
    assert report.similar_name == []


def test_same_size_different_content_is_not_a_duplicate(tmp_path):
    _write(tmp_path / "a.mp3", b"aaaa")
    _write(tmp_path / "b.mp3", b"bbbb")
    report = _find(tmp_path)
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

    _find(tmp_path, hasher=_spy)
    assert sorted(p.name for p in hashed) == ["b.mp3", "c.mp3"]


def test_unreadable_file_is_reported_not_raised(tmp_path):
    _write(tmp_path / "a.mp3", b"same")
    _write(tmp_path / "b.mp3", b"same")

    def _boom(path):
        raise OSError("permesso negato")

    report = _find(tmp_path, hasher=_boom)
    assert len(report.unreadable) == 2
    assert report.same_folder == []


# --- report CSV ------------------------------------------------------------

def test_csv_marks_only_same_folder_as_delete(tmp_path):
    _write(tmp_path / "f" / "Track.mp3", b"same")
    _write(tmp_path / "f" / "Track (1).mp3", b"same")
    _write(tmp_path / "other" / "Track.mp3", b"same")
    report = _find(tmp_path)

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
    report = _find(tmp_path)

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
    report = _find(tmp_path)
    plan = build_quarantine_plan(duplicates_of(report.same_folder), tmp_path)

    assert len(plan) == 2
    assert len({dest for _, dest in plan}) == 2


def test_quarantine_moves_and_never_deletes(tmp_path):
    _write(tmp_path / "f" / "Track.mp3", b"same")
    dup = _write(tmp_path / "f" / "Track (1).mp3", b"same")
    report = _find(tmp_path)
    plan = build_quarantine_plan(duplicates_of(report.same_folder), tmp_path)

    moved, errors = apply_quarantine_plan(plan, dry_run=False)
    assert moved == 1 and errors == []
    assert not dup.exists()                                  # spostato...
    assert plan[0][1].exists()                               # ...non cancellato
    assert (tmp_path / "f" / "Track.mp3").exists()            # l'originale resta


def test_quarantine_dry_run_touches_nothing(tmp_path):
    _write(tmp_path / "f" / "Track.mp3", b"same")
    dup = _write(tmp_path / "f" / "Track (1).mp3", b"same")
    report = _find(tmp_path)
    plan = build_quarantine_plan(duplicates_of(report.same_folder), tmp_path)

    moved, errors = apply_quarantine_plan(plan)               # dry_run di default
    assert moved == 1 and errors == []
    assert dup.exists() and not plan[0][1].exists()


def test_quarantine_plan_skips_what_is_already_quarantined(tmp_path):
    q = tmp_path / QUARANTINE_DIRNAME / "f"
    _write(q / "Track.mp3", b"same")
    _write(q / "Track (1).mp3", b"same")
    report = _find(tmp_path)
    assert build_quarantine_plan(duplicates_of(report.same_folder), tmp_path) == []


def test_similar_name_survives_when_one_file_is_a_copy_of_a_third(tmp_path):
    """Trovato provando su dati veri: due nomi simili con contenuto diverso
    sparivano dal livello C solo perché uno dei due era identico a un terzo
    file in un'altra cartella. La somiglianza dentro la coppia resta
    un'informazione valida."""
    _write(tmp_path / "80s" / "Some Track (Radio Edit).mp3", b"one")
    _write(tmp_path / "80s" / "Some Track [Radio Edit].mp3", b"different")
    _write(tmp_path / "Workout" / "Some Track (Radio Edit).mp3", b"one")
    report = _find(tmp_path)

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
    report = _find(tmp_path)

    assert len(report.same_folder) == 2                  # gruppi
    assert len(duplicates_of(report.same_folder)) == 3   # file da spostare


def test_quarantine_plan_takes_only_the_selected_paths(tmp_path):
    """La selezione la fa l'utente riga per riga: il piano deve limitarsi a
    quello che gli viene passato."""
    for name in ("Track.mp3", "Track (1).mp3", "Track (2).mp3"):
        _write(tmp_path / "f" / name, b"same")
    report = _find(tmp_path)
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

    report = _find(tmp_path)
    assert report.same_folder == [] and report.other_folder == []
    assert all("._" not in g.keep.name for g in report.all_groups())


def test_appledouble_excluded_from_audio_only_scan(tmp_path):
    _write(tmp_path / "Track.mp3", b"vero")
    _write(tmp_path / "._Track.mp3", b"sidecar")
    assert len(scan_folder(tmp_path, audio_only=True).files) == 1


# --- rimozione dei sidecar AppleDouble --------------------------------------

APPLEDOUBLE = b"\x00\x05\x16\x07" + b"Mac OS X" + b"\x00" * 100


def test_find_sidecars_confirms_by_content_not_by_name(tmp_path):
    """Il nome da solo non basta a cancellare: si guarda dentro."""
    from analysis.folder_scan import find_sidecars

    _write(tmp_path / "._Track.mp3", APPLEDOUBLE)
    _write(tmp_path / "._Impostore.mp3", b"ID3\x04 sono un mp3 vero")
    _write(tmp_path / "Track.mp3", b"brano")

    report = find_sidecars(tmp_path)
    assert [p.name for p in report.confirmed] == ["._Track.mp3"]
    assert [p.name for p in report.unverified] == ["._Impostore.mp3"]
    assert report.freed_bytes == len(APPLEDOUBLE)


def test_find_sidecars_includes_folder_sidecars(tmp_path):
    """macOS ne crea anche per le cartelle: stessa spazzatura, niente
    estensione."""
    from analysis.folder_scan import find_sidecars

    _write(tmp_path / "._FLASHBACK", APPLEDOUBLE)
    _write(tmp_path / "sub" / "._Track.flac", APPLEDOUBLE)
    assert len(find_sidecars(tmp_path).confirmed) == 2


def test_delete_sidecars_removes_only_confirmed_ones(tmp_path):
    from analysis.folder_scan import delete_sidecars

    good = _write(tmp_path / "._Track.mp3", APPLEDOUBLE)
    impostor = _write(tmp_path / "._Impostore.mp3", b"ID3\x04 vero mp3")

    removed, freed, errors = delete_sidecars([good, impostor], dry_run=False)
    assert removed == 1 and freed == len(APPLEDOUBLE)
    assert not good.exists()
    assert impostor.exists()                  # il contenuto lo salva
    assert len(errors) == 1 and "AppleDouble" in errors[0][1]


def test_delete_sidecars_dry_run_touches_nothing(tmp_path):
    from analysis.folder_scan import delete_sidecars

    target = _write(tmp_path / "._Track.mp3", APPLEDOUBLE)
    removed, freed, errors = delete_sidecars([target])       # dry_run di default
    assert removed == 1 and freed == len(APPLEDOUBLE) and errors == []
    assert target.exists()


def test_delete_sidecars_never_touches_the_real_track(tmp_path):
    from analysis.folder_scan import delete_sidecars, find_sidecars

    real = _write(tmp_path / "Track.mp3", b"ID3\x04 brano vero")
    _write(tmp_path / "._Track.mp3", APPLEDOUBLE)

    report = find_sidecars(tmp_path)
    delete_sidecars(report.confirmed, dry_run=False)
    assert real.exists() and real.read_bytes() == b"ID3\x04 brano vero"


# --- file audio illeggibili -------------------------------------------------

def test_check_readable_catches_the_obvious_junk(tmp_path):
    from analysis.folder_scan import check_readable

    assert check_readable(_write(tmp_path / "vuoto.mp3", b"")) == "file vuoto"
    assert "intestazione" in check_readable(
        _write(tmp_path / "pagina.mp3", b"<!DOCTYPE html>404 Not Found"))
    assert "intestazione" in check_readable(
        _write(tmp_path / "troncato.mp3", b"\x00\x05\x16\x07"))


def test_check_readable_shallow_can_miss_a_broken_stream(tmp_path, monkeypatch):
    """Il limite da conoscere: intestazione valida e stream rotto passa il
    controllo superficiale. Misurato su un brano reale, dove mutagen legge
    219 secondi di durata e il decoder si rifiuta di aprirlo."""
    import analysis.folder_scan as fs

    target = _write(tmp_path / "sano_in_apparenza.mp3", b"x")
    monkeypatch.setattr(fs, "_decoder_error", lambda p, **k: "Invalid data found")
    fake = type("A", (), {"info": type("I", (), {"length": 219.0})()})()
    monkeypatch.setattr("mutagen.File", lambda *a, **k: fake)

    assert fs.check_readable(target) is None                      # superficiale
    assert fs.check_readable(target, deep=True) == "Invalid data found"   # profondo


def test_check_integrity_separates_bad_from_missing(tmp_path):
    """Rotto e sparito richiedono rimedi diversi — riscaricare, oppure
    niente — quindi non vanno mescolati."""
    from analysis.folder_scan import check_integrity

    bad = _write(tmp_path / "vuoto.mp3", b"")
    gone = tmp_path / "sparito.mp3"

    report = check_integrity([bad, gone])
    assert report.checked == 2
    assert [b.path for b in report.bad] == [bad]
    assert report.bad[0].reason == "file vuoto"
    assert report.missing == [gone]


@pytest.mark.skipif(
    not (Path(__file__).resolve().parent.parent / "test_mp3").is_dir()
    or not any((Path(__file__).resolve().parent.parent / "test_mp3").glob("*.mp3")),
    reason="serve un file audio vero in test_mp3/",
)
def test_check_integrity_passes_a_real_track(tmp_path):
    """Nessun falso positivo su un brano vero: il decoder viene sempre interpellato."""
    from analysis.folder_scan import check_integrity

    src = next((Path(__file__).resolve().parent.parent / "test_mp3").glob("*.mp3"))
    report = check_integrity([src])
    assert report.bad == [] and report.missing == []


def test_decoder_error_stays_silent_without_ffprobe(tmp_path, monkeypatch):
    """Senza ffprobe non si può dire nulla: meglio tacere che accusare un
    file di essere rotto."""
    import analysis.folder_scan as fs

    def _boom(*a, **k):
        raise FileNotFoundError("ffprobe")

    monkeypatch.setattr(fs.subprocess, "run", _boom)
    assert fs._decoder_error(tmp_path / "x.mp3") is None


# --- identici perché rotti, non perché la stessa musica ---------------------

def test_zero_byte_files_are_not_duplicates_of_each_other(tmp_path):
    """Il caso che ha fatto scattare l'allarme: nella libreria reale ci sono
    49 file da zero byte. Hanno tutti lo stesso MD5 — quello del vuoto — e
    venivano proposti come duplicati fra loro, pur essendo BRANI DIVERSI.
    Cancellarne uno vorrebbe dire perdere l'unica traccia di un brano che
    manca."""
    for name in ("Os Serranos - Baile da Mariquinha.mp3",
                 "JOCA MARTINS _ Baile do Sapucay.mp3",
                 "Tche Garotos - Ajoelha e chora.mp3"):
        _write(tmp_path / "GAUCHAS" / name, b"")
    report = find_duplicates(_scan_audio(tmp_path))

    assert report.same_folder == [] and report.other_folder == []
    assert len(report.broken) == 1
    assert report.broken[0].reason == "file vuoto"
    assert len(report.broken[0].paths) == 3


def test_tag_only_stubs_are_not_duplicates(tmp_path):
    """Stessa famiglia: stub da 2311 byte con dentro solo un tag ID3 e
    nessun audio, identici fra loro ma relativi a brani diversi."""
    stub = b"ID3\x04" + b"\x00" * 40 + b"WXXXwww.soundarea.org"
    _write(tmp_path / "07. Tony Dize - El Doctorado.mp3", stub)
    _write(tmp_path / "10. Prince Royce - Stand By Me.mp3", stub)
    report = find_duplicates(_scan_audio(tmp_path))

    assert report.same_folder == []
    assert len(report.broken) == 1 and len(report.broken[0].paths) == 2


def test_broken_groups_never_reach_the_quarantine_plan(tmp_path):
    for name in ("uno.mp3", "due.mp3"):
        _write(tmp_path / name, b"")
    report = find_duplicates(_scan_audio(tmp_path))
    assert build_quarantine_plan(duplicates_of(report.all_groups()), tmp_path) == []


def test_verify_audio_can_be_turned_off(tmp_path):
    _write(tmp_path / "uno.mp3", b"")
    _write(tmp_path / "due.mp3", b"")
    report = find_duplicates(_scan_audio(tmp_path), verify_audio=False)
    assert len(report.same_folder) == 1 and report.broken == []


# --- la quarantena non va rianalizzata --------------------------------------

def test_scan_skips_the_quarantine_folder(tmp_path):
    """Sta dentro la libreria: senza escluderla, la seconda analisi ritrova
    quello che la prima aveva gia' messo da parte."""
    from analysis.folder_scan import QUARANTINE_DIRNAME, scan_folder

    _write(tmp_path / "Track.mp3", b"brano")
    _write(tmp_path / QUARANTINE_DIRNAME / "library" / "Vecchio.mp3", b"gia' spostato")

    scan = scan_folder(tmp_path)
    assert [f.path.name for f in scan.files] == ["Track.mp3"]
    assert len(scan_folder(tmp_path, skip_dirs=()).files) == 2


def test_quarantined_duplicates_are_not_found_again(tmp_path):
    """Scenario reale: prima pulizia fatta, seconda analisi lanciata."""
    from analysis.folder_scan import QUARANTINE_DIRNAME

    _write(tmp_path / "DANCE RETRO" / "Track.mp3", b"stesso contenuto")
    _write(tmp_path / QUARANTINE_DIRNAME / "DANCE RETRO" / "Track (1).mp3",
           b"stesso contenuto")

    report = find_duplicates(_scan_audio(tmp_path))
    assert report.same_folder == [] and report.other_folder == []


# --- durate: filtro per stanare i medley ------------------------------------

def test_human_duration_always_shows_hours():
    """Le ore ci sono sempre: la colonna è testo e viene ordinata
    alfabeticamente, quindi "15:00" finirebbe dopo "1:10:57"."""
    from analysis.folder_scan import human_duration

    assert human_duration(95) == "00:01:35"
    assert human_duration(20 * 60) == "00:20:00"
    assert human_duration(3661) == "01:01:01"


def test_human_duration_sorts_alphabetically_as_it_does_in_time():
    """La proprietà che serve davvero alla tabella."""
    from analysis.folder_scan import human_duration

    seconds = [180, 900, 4257, 1500, 6300, 36000, 32400]
    formatted = [human_duration(s) for s in seconds]
    assert sorted(formatted) == [human_duration(s) for s in sorted(seconds)]


def test_longer_than_filters_and_sorts(tmp_path):
    from analysis.folder_scan import DurationReport, TrackDuration

    report = DurationReport(tracks=[
        TrackDuration(tmp_path / "corto.mp3", 1, 180),
        TrackDuration(tmp_path / "medley.mp3", 2, 45 * 60),
        TrackDuration(tmp_path / "lungo.mp3", 3, 25 * 60),
    ])
    got = report.longer_than(20)
    assert [t.path.name for t in got] == ["medley.mp3", "lungo.mp3"]   # dal più lungo
    assert report.longer_than(60) == []
    assert round(report.longest_minutes) == 45


def test_read_durations_keeps_unknown_apart(tmp_path, monkeypatch):
    """Una durata non leggibile non deve diventare zero: sparirebbe da ogni
    filtro "più lungo di", proprio mentre si propone di spostare dei file."""
    import analysis.folder_scan as fs

    good = _write(tmp_path / "ok.mp3", b"x")
    bad = _write(tmp_path / "muto.mp3", b"y")

    def _fake(path, *a, **k):
        if Path(path).name == "ok.mp3":
            return type("A", (), {"info": type("I", (), {"length": 1200.0})()})()
        return None

    monkeypatch.setattr("mutagen.File", _fake)
    report = fs.read_durations([good, bad])
    assert [t.path for t in report.tracks] == [good]
    assert report.unknown == [bad]


def test_quarantine_plan_moves_a_file_only_once(tmp_path):
    """Lo stesso file puo' essere spuntato in due sezioni diverse: il piano
    deve contenerlo una volta sola, altrimenti il secondo spostamento
    fallisce perche' non e' piu' dov'era."""
    target = _write(tmp_path / "f" / "Track.mp3", b"x")
    plan = build_quarantine_plan([target, target], tmp_path)
    assert len(plan) == 1


def test_scanned_items_are_resolved_by_attribute_not_by_class(tmp_path):
    """Il bug visto in esecuzione: Streamlit ricarica i moduli, ma gli oggetti
    rimasti in sessione portano la classe VECCHIA, e `isinstance` contro
    quella nuova risponde False. Si guarda l'attributo, non il tipo."""
    from dataclasses import dataclass
    from analysis.folder_scan import check_integrity, read_durations

    @dataclass
    class ScannedFileFromAnOlderImport:      # stesso ruolo, classe diversa
        path: Path
        size: int
        fmt: str

    stale = ScannedFileFromAnOlderImport(path=_write(tmp_path / "vuoto.mp3", b""),
                                         size=0, fmt="MP3")

    report = check_integrity([stale])
    assert report.checked == 1 and report.bad[0].reason == "file vuoto"

    durations = read_durations([stale])
    assert durations.unknown == [stale.path]


def test_scanned_items_also_accept_plain_paths_and_strings(tmp_path):
    from analysis.folder_scan import check_integrity

    target = _write(tmp_path / "vuoto.mp3", b"")
    assert check_integrity([target]).checked == 1
    assert check_integrity([str(target)]).checked == 1


# --- Riconoscere mix e mashup dal nome -------------------------------------

def test_default_keywords_catch_mixes_and_spare_extended_versions():
    """I casi veri presi dalla libreria: i mix passano, gli extended no."""
    from analysis.mix_names import DEFAULT_KEYWORDS, matching_words

    caught = [
        "The Outhere Brothers - The Outhere Brothers Megamix (Revibes).wav",
        "[02] Celebrate The Nun - Medley (Razormaid Mix).flac",
        "03 Awsome Party Mix Vol. 2 (Late 70's European Mix).flac",
        "Select Essentials 125th Anniversary Mini Mix.mp3",
        "Some Guy - Whatever (Mash-Up).mp3",
    ]
    spared = [
        "01. Mina - Ancora, ancora, ancora (Extended Version).m4a",
        "ANOTR, 54 Ultra - Talk To You (Dj Dark Extended Remix).mp3",
        "0409 - DJ Gregory - Elle (Main Version).flac",
        # "bootleg" e " vs " NON sono fra i default: misurati, durano quanto
        # un brano qualunque e sono strumenti da dj, non raccolte mixate
        "Artist A vs Artist B - Track (Bootleg).mp3",
    ]
    for name in caught:
        assert matching_words(name, DEFAULT_KEYWORDS), name
    for name in spared:
        assert not matching_words(name, DEFAULT_KEYWORDS), name


def test_keyword_matching_ignores_case_and_parses_the_box():
    from analysis.mix_names import matching_words, parse_keywords

    assert parse_keywords(" Megamix , medley ,, ") == ["megamix", "medley"]
    assert matching_words("X - MEGAMIX.mp3", ["megamix"]) == ["megamix"]
    assert matching_words("X.mp3", ["megamix"]) == []


def test_mixed_by_does_not_swallow_remixed_by():
    """Il caso vero che ha fatto cambiare il confronto a parola intera.

    Nelle raccolte DMC "Remixed By" indica il remixer di un brano singolo:
    sono 138 file nella libreria, tutti da tenere, e come sottostringa
    finivano fra i mix da spostare.
    """
    from analysis.mix_names import DEFAULT_KEYWORDS, matching_words

    remix = "Barry White - Everything (DMC RKL Remix) (Remixed By Rod Layman).mp3"
    megamix = "Duke Dumont - Duke Dumont Megamix (Mixed By Kevin Sweeney).mp3"

    assert matching_words(remix, DEFAULT_KEYWORDS) == []
    assert "mixed by" in matching_words(megamix, DEFAULT_KEYWORDS)


def test_two_vs_mean_a_mashup_but_one_does_not():
    from analysis.mix_names import looks_like_a_mashup

    assert looks_like_a_mashup(
        "david guetta vs. alice deejay - play hard vs. better off alone.mp3")
    assert looks_like_a_mashup("03 beyonce vs destiny's child - 7-11 vs independent woman.mp3")
    # una collaborazione, o un remix: resta dov'e'
    assert not looks_like_a_mashup("Artist A vs Artist B - Some Track (Extended).mp3")
    assert not looks_like_a_mashup("Elvis - Suspicious Minds.mp3")


def test_duration_band_includes_the_top_but_not_the_bottom():
    from analysis.folder_scan import DurationReport, TrackDuration

    t = lambda name, mins: TrackDuration(path=Path(name), size=0, seconds=mins * 60)
    report = DurationReport(tracks=[
        t("corto.mp3", 4), t("dieci.mp3", 10), t("dodici.mp3", 12),
        t("trenta.mp3", 30), t("lunghissimo.mp3", 45),
    ])

    got = [p.path.name for p in report.between(10, 30)]
    # 10 escluso (lo escludeva gia' "piu' lungo di 10"), 30 compreso
    assert got == ["trenta.mp3", "dodici.mp3"]
    assert [p.path.name for p in report.between(1, 45)] == [
        "lunghissimo.mp3", "trenta.mp3", "dodici.mp3", "dieci.mp3", "corto.mp3"]


def test_a_band_starting_at_zero_reaches_the_short_files():
    """La fascia bassa serve a trovare frammenti e anteprime.

    `between` esclude l'estremo basso di proposito, cosi' "da 10 a 30" non
    ripesca chi dura esattamente 10 minuti. A zero quell'esclusione non
    toglie niente: i file a durata nulla o illeggibile non entrano affatto
    fra i `tracks` (finiscono negli `unknown`), quindi 0-1 vale davvero
    "fino a un minuto".
    """
    from pathlib import Path

    from analysis.folder_scan import DurationReport, TrackDuration

    r = DurationReport(tracks=[
        TrackDuration(Path("/frammento.mp3"), seconds=12, size=100),
        TrackDuration(Path("/anteprima.mp3"), seconds=60, size=200),
        TrackDuration(Path("/brano.mp3"), seconds=240, size=300),
        TrackDuration(Path("/megamix.mp3"), seconds=1800, size=400),
    ])

    corti = [t.path.name for t in r.between(0, 1)]
    assert corti == ["anteprima.mp3", "frammento.mp3"]      # 60s compreso
    assert "brano.mp3" not in corti

    # e la fascia alta continua a comportarsi come prima
    assert [t.path.name for t in r.between(10, 30)] == ["megamix.mp3"]
