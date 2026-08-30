from pathlib import Path

from core.analysis.tag_tracking import ProcessedTracker


def _tracker(tmp_path, name="progress.txt"):
    return ProcessedTracker(tracking_file=tmp_path / name)


def test_empty_when_the_file_does_not_exist(tmp_path):
    tracker = _tracker(tmp_path)
    assert len(tracker) == 0 and not tracker.existed
    assert tracker.pending([Path("/a.mp3")]) == [Path("/a.mp3")]


def test_reads_an_existing_file(tmp_path):
    (tmp_path / "progress.txt").write_text("/music/a.mp3\n/music/b.mp3\n", encoding="utf-8")
    tracker = _tracker(tmp_path)

    assert len(tracker) == 2 and tracker.existed
    assert Path("/music/a.mp3") in tracker
    assert tracker.pending([Path("/music/a.mp3"), Path("/music/c.mp3")]) == [Path("/music/c.mp3")]


def test_duplicate_lines_are_absorbed(tmp_path):
    """Il file dello script originale ha circa un quinto di righe ripetute:
    113.000 righe per 90.000 percorsi distinti. Copiarlo qui deve bastare."""
    (tmp_path / "progress.txt").write_text(
        "/music/a.mp3\n/music/a.mp3\n/music/b.mp3\n", encoding="utf-8")
    tracker = _tracker(tmp_path)

    assert len(tracker) == 2
    assert tracker.lines_read == 3 and tracker.duplicate_lines == 1


def test_blank_lines_are_ignored(tmp_path):
    (tmp_path / "progress.txt").write_text(
        "/music/a.mp3\n\n   \n/music/b.mp3\n", encoding="utf-8")
    assert len(_tracker(tmp_path)) == 2


def test_mark_appends_and_is_remembered(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.mark(Path("/music/new.mp3"))

    assert (tmp_path / "progress.txt").read_text(encoding="utf-8") == "/music/new.mp3\n"
    assert Path("/music/new.mp3") in tracker


def test_mark_is_idempotent(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.mark(Path("/music/a.mp3"))
    tracker.mark(Path("/music/a.mp3"))
    assert (tmp_path / "progress.txt").read_text(encoding="utf-8").count("\n") == 1


def test_mark_does_not_rewrite_what_the_file_already_had(tmp_path):
    path = tmp_path / "progress.txt"
    path.write_text("/music/a.mp3\n", encoding="utf-8")
    tracker = _tracker(tmp_path)
    tracker.mark(Path("/music/a.mp3"))
    assert path.read_text(encoding="utf-8") == "/music/a.mp3\n"


def test_mark_creates_the_parent_directory(tmp_path):
    tracker = ProcessedTracker(tracking_file=tmp_path / "nested" / "deep" / "t.txt")
    tracker.mark(Path("/music/a.mp3"))
    assert (tmp_path / "nested" / "deep" / "t.txt").exists()


def test_a_truncated_last_line_does_not_raise(tmp_path):
    """Il file copiato può essere stato preso mentre l'altro script scriveva:
    al peggio un brano viene rianalizzato, non deve saltare la lettura."""
    (tmp_path / "progress.txt").write_bytes(b"/music/a.mp3\n/music/partial")
    tracker = _tracker(tmp_path)
    assert Path("/music/a.mp3") in tracker and len(tracker) == 2
