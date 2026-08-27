"""Le altre quattro teste del model zoo, provate prima di adottarle."""

import json

import numpy as np
import pytest

import zoo_cli
from analysis.map_profile import EMBEDDING_DIM, TrackProfile
from analysis.map_store import MapStore


def _profile(path, a, b, c, d):
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[:4] = (a, b, c, d)
    return TrackProfile(path=path, duration=300.0, bpm=128.0, camelot="8A",
                        danceability=0.5, embedding=vector,
                        energy={"energy_density": float(a),
                                "energy_bass": float(a),
                                "energy_bright": float(a),
                                "energy_pulse": float(a)})


def _store(tmp_path, *tracks):
    store = MapStore.load(tmp_path / "map")
    profiles = []
    for n, values in enumerate(tracks):
        audio = tmp_path / f"{n}.mp3"
        audio.write_bytes(b"x")
        profiles.append(_profile(audio, *values))
    store.append(profiles)
    for row in store.rows:
        row["moods"] = "Dark"
    return store


def _heads():
    """Quattro teste finte: ognuna legge un numero diverso del vettore e ne
    fa due probabilità che sommano a uno, come farebbe una softmax."""
    def head(at):
        def predict(vectors):
            v = np.asarray(vectors, dtype=float)[:, at]
            return np.column_stack([v, 1.0 - v])
        return predict
    return {name: (head(i), 0) for i, name in enumerate(zoo_cli.HEADS)}


def test_the_positive_class_is_the_one_that_is_not_a_negation():
    # L'ordine non e' garantito e il prefisso cambia da modello a modello:
    # fidarsi della posizione avrebbe dato, sul modello sbagliato, un numero
    # perfettamente plausibile e rovesciato.
    assert zoo_cli.positive_class(["aggressive", "non_aggressive"]) == 0
    assert zoo_cli.positive_class(["non_aggressive", "aggressive"]) == 1
    assert zoo_cli.positive_class(["not_danceable", "danceable"]) == 1
    # Se non si capisce, si tiene la prima invece di indovinare.
    assert zoo_cli.positive_class(["one", "other"]) == 0


def test_the_node_names_come_from_the_model_and_not_from_a_guess():
    metadata = {"schema": {"inputs": [{"name": "serving/in"}],
                           "outputs": [{"name": "serving/out"}]}}
    assert zoo_cli.nodes(metadata) == ("serving/in", "serving/out")
    # Un JSON che non li porta non fa saltare niente: si ripiega.
    assert zoo_cli.nodes({}) == (zoo_cli.FALLBACK_IN, zoo_cli.FALLBACK_OUT)


def test_every_head_needs_both_its_graph_and_its_classes(tmp_path):
    assert len(zoo_cli.missing(tmp_path)) == 2 * len(zoo_cli.HEADS)
    for stem in zoo_cli.HEADS.values():
        (tmp_path / f"{stem}.pb").write_bytes(b"x")
        (tmp_path / f"{stem}.json").write_text(json.dumps({"classes": []}))
    assert zoo_cli.missing(tmp_path) == []


def test_the_four_answers_come_out_one_column_each(tmp_path):
    store = _store(tmp_path, (0.9, 0.1, 0.8, 0.2), (0.1, 0.9, 0.2, 0.8))
    fresh = zoo_cli.scored(store.embeddings, _heads())
    assert set(fresh) == set(zoo_cli.HEADS)
    # `pytest.approx` e non l'uguaglianza: gli embedding stanno su disco in
    # float32, e 0,9 tornera' sempre 0,8999999.
    assert fresh["aggressive"] == pytest.approx([0.9, 0.1], abs=1e-6)
    assert fresh["relaxed"] == pytest.approx([0.1, 0.9], abs=1e-6)


def test_the_vectors_go_through_in_batches_without_losing_anyone(tmp_path):
    store = _store(tmp_path, *[(0.5, 0.5, 0.5, 0.5)] * 10)
    fresh = zoo_cli.scored(store.embeddings, _heads(), batch=3)
    assert len(fresh["party"]) == 10


def test_a_head_that_answers_the_same_thing_every_time_has_no_spread():
    flat = zoo_cli.spread(np.full(100, 0.9))
    assert flat["above 0.5"] == 1.0
    assert set(flat["deciles"]) == {0.9}          # una costante, non una misura
    moving = zoo_cli.spread(np.linspace(0.0, 1.0, 100))
    assert moving["deciles"][0] < 0.2 < 0.8 < moving["deciles"][-1]


def test_a_head_that_repeats_something_we_have_says_so():
    values = np.linspace(0.0, 1.0, 50)
    others = {"energy": values.copy(),             # lo stesso numero
              "groove": values[::-1].copy(),       # lo stesso rovesciato
              "BPM": np.full(50, 120.0)}           # una costante
    seen = zoo_cli.against(values, others)
    assert seen["energy"] > 0.99
    assert seen["groove"] < -0.99
    # Contro una costante la correlazione non esiste: la voce non compare
    # invece di uscire `nan`.
    assert "BPM" not in seen or not np.isfinite(seen["BPM"])


def test_the_report_measures_each_head_against_everything_else(tmp_path):
    store = _store(tmp_path, *[(i / 20, 1 - i / 20, 0.5, 0.5)
                               for i in range(20)])
    facts = zoo_cli.report(store, 20, _heads())
    assert set(facts) == set(zoo_cli.HEADS)
    # Le due teste finte sono l'una l'opposto dell'altra, e si vede.
    assert facts["aggressive"]["repeats"]["relaxed"] < -0.99
    # E ognuna si confronta anche con quello che la mappa ha gia'.
    assert {"energy", "valence", "groove", "BPM"} >= set(zoo_cli.KNOWN)
    assert "energy" in facts["aggressive"]["repeats"]


def test_the_listening_sample_walks_the_head_you_chose(tmp_path):
    store = _store(tmp_path, *[(i / 20, 0.5, 0.5, 0.5) for i in range(20)])
    table = zoo_cli.listing(store, 4, _heads(), "aggressive")
    values = [row["aggressive"] for row in table]
    assert values == sorted(values)               # dal meno al piu'
    assert "path" in table[0] and "moods" in table[0]


def test_nothing_is_written_anywhere(tmp_path):
    """La ragione per cui questo comando esiste: decidere PRIMA. Non tocca
    i tag dei file e non tocca la mappa."""
    store = _store(tmp_path, (0.9, 0.1, 0.8, 0.2))
    before = (tmp_path / "map" / "tracks.jsonl").read_bytes()
    zoo_cli.report(store, 10, _heads())
    zoo_cli.listing(store, 1, _heads(), "party")
    assert (tmp_path / "map" / "tracks.jsonl").read_bytes() == before
    import inspect
    source = inspect.getsource(zoo_cli)
    assert "rewrite" not in source and "write_tags" not in source
