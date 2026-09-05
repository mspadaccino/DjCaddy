"""Lo scaffale delle playlist: una cartella di .m3u8, un nome per file."""

from core.analysis.shelf import Shelf, valid_name


def test_names_are_the_file_stems_in_order(tmp_path):
    shelf = Shelf(tmp_path)
    shelf.write("house_intro", ["/x/one.mp3"])
    shelf.write("Dance_climax", [])
    assert shelf.names() == ["Dance_climax", "house_intro"]


def test_write_and_read_round_trip_the_paths(tmp_path):
    shelf = Shelf(tmp_path)
    shelf.write("set", ["/Music/one.mp3", "/Music/My Track.mp3"])
    assert shelf.read("set") == ["/Music/one.mp3", "/Music/My Track.mp3"]
    assert (tmp_path / "set.m3u8").read_text().startswith("#EXTM3U")


def test_a_missing_folder_or_file_reads_as_nothing(tmp_path):
    shelf = Shelf(tmp_path / "nowhere")
    assert shelf.names() == []
    assert shelf.read("ghost") == []
    assert shelf.active() is None


def test_rename_moves_the_file_and_follows_the_active_one(tmp_path):
    shelf = Shelf(tmp_path)
    shelf.write("a", ["/x/one.mp3"])
    shelf.set_active("a")
    shelf.rename("a", "b")
    assert shelf.names() == ["b"]
    assert shelf.read("b") == ["/x/one.mp3"]
    assert shelf.active() == "b"


def test_active_is_forgotten_when_its_file_is_gone(tmp_path):
    shelf = Shelf(tmp_path)
    shelf.write("a", [])
    shelf.set_active("a")
    shelf.delete("a")
    assert shelf.active() is None


def test_free_name_counts_up_past_the_taken_ones(tmp_path):
    shelf = Shelf(tmp_path)
    assert shelf.free_name("night") == "night"
    shelf.write("night", [])
    shelf.write("night 2", [])
    assert shelf.free_name("night") == "night 3"


def test_valid_name_is_a_file_name():
    assert valid_name("house_intro")
    assert valid_name("Funk / Soul".replace("/", "-"))
    assert not valid_name("")
    assert not valid_name("   ")
    assert not valid_name(".hidden")
    assert not valid_name("a/b")
    assert not valid_name("a\\b")
