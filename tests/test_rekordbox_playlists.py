"""Le playlist dello scaffale verso rekordbox: la parte che decide, senza
database — cosa si trova, cosa manca, cosa si rifà."""

from pathlib import Path

from core.analysis.rekordbox_playlists import FOLDER, plan, summary


def _find(path: Path):
    """rekordbox conosce tutto tranne i fantasmi."""
    return None if "ghost" in path.name else f"row:{path.name}"


def test_plan_sorts_each_track_into_found_or_missing_keeping_the_order():
    plans = plan([("house_intro", [Path("/x/a.mp3"), Path("/x/ghost.mp3"),
                                   Path("/x/b.mp3")])], _find)
    assert [p.name for p in plans] == ["house_intro"]
    assert plans[0].found == ["row:a.mp3", "row:b.mp3"]
    assert plans[0].missing == [Path("/x/ghost.mp3")]
    assert not plans[0].replaces


def test_plan_marks_the_playlists_already_in_the_folder():
    plans = plan([("a", []), ("b", [])], _find, existing={"b"})
    assert [p.replaces for p in plans] == [False, True]


def test_summary_names_the_folder_the_missing_and_the_rebuilt():
    plans = plan([("house_intro", [Path("/x/a.mp3"), Path("/x/ghost.mp3")]),
                  ("house_climax", [Path("/x/b.mp3")])],
                 _find, existing={"house_climax"})
    told = summary(plans)
    assert f"«{FOLDER}»" in told
    assert "2 playlist(s)" in told and "2 track(s) rekordbox knows" in told
    assert "1 track(s) are not in rekordbox" in told
    assert "rebuilt as on the shelf: house_climax" in told
