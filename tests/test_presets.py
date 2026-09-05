"""I preset dei filtri: un JSON solo, nomi e dizionari."""

from core.analysis.presets import Presets


def test_write_read_and_names_in_order(tmp_path):
    presets = Presets(tmp_path / "presets.json")
    presets.write("house_intro", {"bpm": [118, 124]})
    presets.write("Dance_climax", {"bpm": [126, 130]})
    assert presets.names() == ["Dance_climax", "house_intro"]
    assert presets.read("house_intro") == {"bpm": [118, 124]}
    assert presets.read("ghost") is None


def test_a_missing_or_broken_file_reads_as_empty(tmp_path):
    assert Presets(tmp_path / "nowhere" / "p.json").names() == []
    broken = tmp_path / "p.json"
    broken.write_text("{not json", "utf-8")
    assert Presets(broken).names() == []


def test_delete_removes_one_and_leaves_the_rest(tmp_path):
    presets = Presets(tmp_path / "p.json")
    presets.write("a", {})
    presets.write("b", {})
    presets.delete("a")
    presets.delete("never there")
    assert presets.names() == ["b"]
