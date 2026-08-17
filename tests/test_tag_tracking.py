from pathlib import Path

from analysis.tag_tracking import ProcessedTracker


def _tracker(tmp_path, sources=(), name="wavecut.txt"):
    return ProcessedTracker(tracking_file=tmp_path / name, sources=list(sources))


def test_empty_when_nothing_exists(tmp_path):
    tracker = _tracker(tmp_path)
    assert len(tracker) == 0
    assert tracker.pending([Path("/a.mp3")]) == [Path("/a.mp3")]


def test_reads_an_existing_source(tmp_path):
    source = tmp_path / "legacy.txt"
    source.write_text("/music/a.mp3\n/music/b.mp3\n", encoding="utf-8")
    tracker = _tracker(tmp_path, [source])

    assert len(tracker) == 2
    assert Path("/music/a.mp3") in tracker
    assert tracker.pending([Path("/music/a.mp3"), Path("/music/c.mp3")]) == [Path("/music/c.mp3")]


def test_union_of_sources_and_own_tracking(tmp_path):
    """Il punto dell'unione: il lavoro dello script esterno e quello fatto
    qui contano entrambi, senza dover fondere i file."""
    legacy = tmp_path / "legacy.txt"
    legacy.write_text("/music/a.mp3\n", encoding="utf-8")
    (tmp_path / "wavecut.txt").write_text("/music/b.mp3\n", encoding="utf-8")

    tracker = _tracker(tmp_path, [legacy])
    assert len(tracker) == 2
    assert Path("/music/a.mp3") in tracker and Path("/music/b.mp3") in tracker


def test_duplicate_lines_are_absorbed(tmp_path):
    """Il file originale ha 113.204 righe per 90.404 percorsi distinti."""
    source = tmp_path / "legacy.txt"
    source.write_text("/music/a.mp3\n/music/a.mp3\n/music/b.mp3\n", encoding="utf-8")
    tracker = _tracker(tmp_path, [source])

    assert len(tracker) == 2
    assert tracker.total_lines == 3


def test_blank_lines_are_ignored(tmp_path):
    source = tmp_path / "legacy.txt"
    source.write_text("/music/a.mp3\n\n   \n/music/b.mp3\n", encoding="utf-8")
    assert len(_tracker(tmp_path, [source])) == 2


def test_mark_appends_only_to_our_own_file(tmp_path):
    """La fonte esterna non va toccata: può essere aperta in append da uno
    script in esecuzione."""
    legacy = tmp_path / "legacy.txt"
    legacy.write_text("/music/a.mp3\n", encoding="utf-8")
    before = legacy.read_bytes()

    tracker = _tracker(tmp_path, [legacy])
    tracker.mark(Path("/music/new.mp3"))

    assert legacy.read_bytes() == before
    assert (tmp_path / "wavecut.txt").read_text(encoding="utf-8") == "/music/new.mp3\n"
    assert Path("/music/new.mp3") in tracker


def test_mark_is_idempotent(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.mark(Path("/music/a.mp3"))
    tracker.mark(Path("/music/a.mp3"))
    assert (tmp_path / "wavecut.txt").read_text(encoding="utf-8").count("\n") == 1


def test_mark_does_not_rewrite_what_a_source_already_had(tmp_path):
    legacy = tmp_path / "legacy.txt"
    legacy.write_text("/music/a.mp3\n", encoding="utf-8")
    tracker = _tracker(tmp_path, [legacy])
    tracker.mark(Path("/music/a.mp3"))
    assert not (tmp_path / "wavecut.txt").exists()


def test_mark_creates_the_parent_directory(tmp_path):
    tracker = ProcessedTracker(tracking_file=tmp_path / "nested" / "deep" / "t.txt",
                               sources=[])
    tracker.mark(Path("/music/a.mp3"))
    assert (tmp_path / "nested" / "deep" / "t.txt").exists()


def test_a_truncated_last_line_does_not_raise(tmp_path):
    """Il file cresce mentre lo leggiamo: al peggio un brano viene
    rianalizzato, non deve saltare la lettura."""
    source = tmp_path / "legacy.txt"
    source.write_bytes(b"/music/a.mp3\n/music/partial")
    tracker = _tracker(tmp_path, [source])
    assert Path("/music/a.mp3") in tracker
    assert len(tracker) == 2


def test_stats_report_each_source(tmp_path):
    legacy = tmp_path / "legacy.txt"
    legacy.write_text("/music/a.mp3\n", encoding="utf-8")
    tracker = _tracker(tmp_path, [legacy])

    reported = {s.path: s for s in tracker.stats}
    assert reported[legacy].exists and reported[legacy].lines == 1
    assert not reported[tmp_path / "wavecut.txt"].exists


def test_missing_source_is_reported_not_fatal(tmp_path):
    tracker = _tracker(tmp_path, [tmp_path / "nope.txt"])
    assert len(tracker) == 0
    assert all(not s.exists for s in tracker.stats)
