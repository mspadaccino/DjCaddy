import json

import numpy as np

from core.analysis.journal import Journal, facts


def test_record_appends_one_json_line_per_gesture(tmp_path):
    journal = Journal(tmp_path / "deep" / "choices.jsonl")
    journal.record("pick", chosen=["/a.mp3"], shown=[{"path": "/a.mp3"}])
    journal.record("unchain", path="/b.mp3")
    lines = journal.read()
    assert [line["kind"] for line in lines] == ["pick", "unchain"]
    assert lines[0]["chosen"] == ["/a.mp3"]
    assert lines[0]["at"]                      # l'ora c'è sempre


def test_read_of_nothing_is_empty_and_a_broken_line_is_skipped(tmp_path):
    journal = Journal(tmp_path / "choices.jsonl")
    assert journal.read() == []
    journal.record("pick", chosen=[])
    with journal.path.open("a") as out:
        out.write('{"kind": "pi')                # crash a metà riga
    assert [line["kind"] for line in journal.read()] == ["pick"]


def test_facts_turn_numpy_and_nan_into_plain_json(tmp_path):
    row = {"path": "/a.mp3", "bpm": np.float32(128.0), "camelot": "8A",
           "energy": float("nan"), "valence": np.float64(0.3),
           "danceability": 0.7, "genres": "House"}
    noted = facts(row)
    assert noted["bpm"] == 128.0 and isinstance(noted["bpm"], float)
    assert noted["energy"] is None and noted["moods"] is None
    journal = Journal(tmp_path / "choices.jsonl")
    journal.record("pick", source=noted)
    text = journal.path.read_text("utf-8")
    assert "NaN" not in text and json.loads(text)["source"]["valence"] == 0.3
