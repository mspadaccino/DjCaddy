from pathlib import Path
from xml.etree import ElementTree as ET

from core.analysis.dj_export import (
    _hex_to_rgb,
    build_m3u8,
    build_rekordbox_xml,
    file_uri,
    read_m3u8,
    section_cues,
)
from core.analysis.models import Section


def test_hex_to_rgb():
    assert _hex_to_rgb("#e0503b") == (224, 80, 59)
    assert _hex_to_rgb("ffffff") == (255, 255, 255)


def test_file_uri_encodes_spaces():
    uri = file_uri(Path("/Users/dj/Music/My Track.mp3"))
    assert uri.startswith("file://localhost/")
    assert "%20" in uri
    assert " " not in uri


def test_section_cues_marks_vocal_and_uses_palette():
    sections = [
        Section(0, 30, "Intro", 0.2),
        Section(30, 90, "Drop/Chorus", 1.0, vocal=True),
    ]
    cues = section_cues(sections)
    assert cues[0]["name"] == "Intro"
    assert cues[1]["name"] == "VOCAL Drop/Chorus"
    assert cues[1]["color"] == "#e0503b"


def test_build_rekordbox_xml_structure_and_escaping():
    tracks = [{
        "path": Path("/Users/dj/Music/Weird & Wild.mp3"),
        "name": 'Weird & "Wild"',
        "artist": "Some <Artist>",
        "genre": "Dance",
        "bpm": 124.0,
        "duration": 200.0,
        "cues": [
            {"name": "Intro", "start": 0.0, "color": "#8e9aa6"},
            {"name": "Drop/Chorus", "start": 32.5, "color": "#e0503b"},
        ],
    }]
    xml_str = build_rekordbox_xml(tracks)

    root = ET.fromstring(xml_str)  # deve essere XML valido (verifica l'escaping)
    assert root.tag == "DJ_PLAYLISTS"

    track = root.find(".//TRACK[@TrackID='1']")
    assert track is not None
    assert track.get("Name") == 'Weird & "Wild"'
    assert track.get("Artist") == "Some <Artist>"
    assert track.get("AverageBpm") == "124.00"
    assert track.get("Location").startswith("file://localhost/")

    marks = track.findall("POSITION_MARK")
    assert len(marks) == 2
    assert marks[0].get("Num") == "0"          # primo cue -> hot cue slot 0
    assert marks[1].get("Start") == "32.500"
    assert marks[1].get("Red") == "224"        # colore del Drop tradotto in RGB

    playlist_tracks = root.findall(".//PLAYLISTS//NODE[@Name='DjCaddy']/TRACK")
    assert [t.get("Key") for t in playlist_tracks] == ["1"]


def test_build_rekordbox_xml_more_than_eight_cues_become_memory_cues():
    tracks = [{
        "path": Path("/x/track.mp3"), "name": "t", "artist": "", "genre": "",
        "bpm": None, "duration": None,
        "cues": [{"name": f"C{i}", "start": float(i), "color": "#ffffff"} for i in range(10)],
    }]
    root = ET.fromstring(build_rekordbox_xml(tracks))
    marks = root.find(".//TRACK").findall("POSITION_MARK")
    nums = [m.get("Num") for m in marks]
    assert nums[:8] == [str(i) for i in range(8)]
    assert nums[8:] == ["-1", "-1"]


def test_build_rekordbox_xml_empty_tracks():
    xml_str = build_rekordbox_xml([])
    root = ET.fromstring(xml_str)
    assert root.find("COLLECTION").get("Entries") == "0"


def test_read_m3u8_keeps_the_order_and_drops_the_directives():
    text = "#EXTM3U\n#EXTINF:213,A - One\n/Music/one.mp3\n" \
           "#EXTINF:198,B - Two\n/Music/two.mp3\n"
    assert read_m3u8(text) == ["/Music/one.mp3", "/Music/two.mp3"]


def test_read_m3u8_reads_a_playlist_this_app_wrote():
    written = build_m3u8([
        {"path": Path("/Music/one.mp3"), "name": "One", "artist": "A",
         "duration": 213.0},
        {"path": Path("/Music/two.mp3"), "name": "Two", "artist": "",
         "duration": None},
    ])
    assert read_m3u8(written) == ["/Music/one.mp3", "/Music/two.mp3"]


def test_read_m3u8_turns_file_uris_back_into_paths():
    text = "#EXTM3U\nfile://localhost/Music/My%20Track.mp3\n"
    assert read_m3u8(text) == ["/Music/My Track.mp3"]


def test_read_m3u8_ignores_blank_lines():
    assert read_m3u8("\n#EXTM3U\n\n  /Music/one.mp3  \n\n") == ["/Music/one.mp3"]
