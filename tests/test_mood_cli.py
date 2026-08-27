"""Il backfill del mood: quello che rifà i numeri senza riaprire i file."""

import numpy as np

import mood_cli
from analysis import mood_scale
from analysis.map_profile import EMBEDDING_DIM, ProfileSettings, TrackProfile
from analysis.map_store import MapStore

SETTINGS = ProfileSettings()

# Una testa finta con tre etichette: una buia, una chiara, una che non dice
# niente. Prevede leggendo i primi tre numeri del vettore, il che rende ogni
# brano un caso costruito a mano invece che un risultato da indovinare.
LABELS = ["Dark", "Happy", "Energetic"]


def _predict(vectors):
    return np.asarray(vectors, dtype=float)[:, :3]


def _profile(path, dark, bright, plain):
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[:3] = (dark, bright, plain)
    return TrackProfile(path=path, duration=300.0, bpm=128.0, camelot="8A",
                        embedding=vector)


def _store(tmp_path, *tracks):
    store = MapStore.load(tmp_path / "map")
    profiles = []
    for n, (dark, bright, plain) in enumerate(tracks):
        audio = tmp_path / f"{n}.mp3"
        audio.write_bytes(b"x")
        profiles.append(_profile(audio, dark, bright, plain))
    store.append(profiles)
    return store


def test_the_three_fields_come_out_of_the_activations(tmp_path):
    written = mood_cli.written([0.60, 0.20, 0.90], LABELS, SETTINGS)
    assert written["valence"] == -0.5           # (0,20 − 0,60) / 0,80
    assert written["mood_evidence"] == 0.8      # la neutra non conta
    assert written["mood_conf"] == "Energetic:0.900; Dark:0.600; Happy:0.200"


def test_a_colourless_track_gets_no_valence_but_still_gets_the_weights():
    written = mood_cli.written([0.0, 0.0, 0.90], LABELS, SETTINGS)
    assert written["valence"] is None
    assert written["mood_evidence"] == 0.0
    assert written["mood_conf"] == "Energetic:0.900"


def test_the_vectors_go_through_the_head_in_batches():
    vectors = np.arange(30, dtype=np.float32).reshape(10, 3)
    rows = [row for chunk in mood_cli.scored(vectors, _predict, batch=3)
            for row in chunk]
    assert len(rows) == 10                      # nessuno perso ai bordi
    assert rows[7].tolist() == [21.0, 22.0, 23.0]


def test_the_backfill_writes_the_numbers_on_every_row(tmp_path, monkeypatch):
    store = _store(tmp_path, (0.60, 0.20, 0.90), (0.10, 0.70, 0.30))
    monkeypatch.setattr(mood_cli, "_head", lambda: (_predict, LABELS))
    assert mood_cli.backfill(store, SETTINGS) == 2

    again = MapStore.load(tmp_path / "map")
    assert [row["valence"] for row in again.rows] == [-0.5, 0.75]
    # E i vettori restano quelli, allineati riga per riga: il backfill
    # riscrive le righe, non la mappa.
    assert again.embeddings.shape == (2, EMBEDDING_DIM)


def test_the_backfill_skips_what_it_has_already_done(tmp_path, monkeypatch):
    store = _store(tmp_path, (0.60, 0.20, 0.90))
    monkeypatch.setattr(mood_cli, "_head", lambda: (_predict, LABELS))
    assert mood_cli.backfill(store, SETTINGS) == 1
    assert mood_cli.backfill(store, SETTINGS) == 0


def test_a_map_with_fewer_vectors_than_rows_is_refused(tmp_path, monkeypatch):
    store = _store(tmp_path, (0.60, 0.20, 0.90))
    store.rows.append(dict(store.rows[0], path="/altro.mp3"))
    monkeypatch.setattr(mood_cli, "_head", lambda: (_predict, LABELS))
    try:
        mood_cli.backfill(store, SETTINGS)
    except ValueError as error:
        assert "non è allineata" in str(error)
    else:
        raise AssertionError("una mappa disallineata non va riscritta")


def test_the_check_measures_the_top_label_not_just_the_set():
    # Le etichette salvate dicono Dark, la riprevisione pure: la prima tiene.
    same, share = mood_cli.agreement([0.60, 0.20, 0.90], LABELS,
                                     "Energetic; Dark", SETTINGS)
    assert same                                 # Energetic davanti come prima
    assert share == 2 / 3                       # ma la riprevisione trova Happy


def test_the_check_says_when_a_track_lost_its_labels():
    same, share = mood_cli.agreement([0.0, 0.0, 0.0], LABELS, "Dark", SETTINGS)
    assert not same and share == 0.0


def test_missing_asks_the_one_field_that_is_always_written():
    # Non `valence`, che sui brani senza colore resta `None` per sempre.
    rows = [{"valence": -0.5, "mood_evidence": 0.8, "mood_conf": "Dark:0.600"},
            {"valence": None, "mood_evidence": 0.0, "mood_conf": "Fun:0.030"},
            {}]
    assert mood_cli.missing(rows) == [2]


def test_a_track_with_no_colour_is_not_backfilled_forever(tmp_path, monkeypatch):
    # `valence` resta `None` perche' il brano colore non ne ha: se `missing`
    # guardasse solo quello, il backfill lo rifarebbe a ogni giro.
    store = _store(tmp_path, (0.0, 0.0, 0.90))
    monkeypatch.setattr(mood_cli, "_head", lambda: (_predict, LABELS))
    assert mood_cli.backfill(store, SETTINGS) == 1
    assert store.rows[0]["valence"] is None
    assert mood_cli.backfill(store, SETTINGS) == 0


def test_the_written_fields_are_the_same_the_analyzer_writes():
    # Se qui si ricopiasse la formula invece di chiamarla, i brani vecchi e i
    # nuovi finirebbero su due scale diverse senza che nessuno se ne accorga.
    scores = [0.60, 0.20, 0.90]
    whole = dict(zip(LABELS, scores))
    fresh = TrackProfile(path=__import__("pathlib").Path("/x.mp3"),
                         valence=mood_scale.valence_of(whole),
                         mood_evidence=mood_scale.evidence(whole),
                         mood_weights=[("Energetic", 0.9), ("Dark", 0.6),
                                       ("Happy", 0.2)]).to_row()
    backfilled = mood_cli.written(scores, LABELS, SETTINGS)
    assert all(fresh[name] == backfilled[name] for name in mood_cli.FIELDS)


def test_the_check_reports_without_writing_anything(tmp_path, monkeypatch):
    """E' il primo comando che si lancia: dice quanto la riprevisione dagli
    embedding somiglia alle etichette salvate, prima che qualcuno riscriva
    ottantasettemila righe."""
    store = _store(tmp_path, (0.60, 0.20, 0.90), (0.10, 0.70, 0.30))
    store.rows[0]["moods"] = "Energetic; Dark"      # come la riprevede
    store.rows[1]["moods"] = "Dark"                 # come NON la riprevede
    monkeypatch.setattr(mood_cli, "_head", lambda: (_predict, LABELS))

    report = mood_cli.check(store, 10, SETTINGS)
    assert report["tracks"] == 2
    assert report["top label kept"] == 0.5
    # E le righe restano come stavano: `--check` non scrive.
    assert mood_cli.missing(store.rows) == [0, 1]


def test_the_check_on_an_empty_map_says_nothing_instead_of_dividing_by_zero(
        tmp_path, monkeypatch):
    store = MapStore.load(tmp_path / "map")
    monkeypatch.setattr(mood_cli, "_head", lambda: (_predict, LABELS))
    assert mood_cli.check(store, 10, SETTINGS) == {"tracks": 0}
