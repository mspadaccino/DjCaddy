from pathlib import Path

import numpy as np

from analysis.map_profile import EMBEDDING_DIM, TrackProfile
from analysis.map_store import MapStore


def _profile(path, vector, bpm=128.0):
    return TrackProfile(path=path, duration=300.0, bpm=bpm, camelot="8A",
                        embedding=np.full(EMBEDDING_DIM, vector, dtype=np.float32))


def test_appending_survives_a_reload(tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"not really audio")
    store = MapStore.load(tmp_path / "map")
    assert store.append([_profile(audio, 1.0)]) == 1

    again = MapStore.load(tmp_path / "map")
    assert len(again) == 1
    assert again.rows[0]["path"] == str(audio)
    assert again.embeddings.shape == (1, EMBEDDING_DIM)


def test_a_track_that_failed_does_not_go_on_the_map(tmp_path):
    store = MapStore.load(tmp_path / "map")
    broken = TrackProfile(path=tmp_path / "b.mp3", error="MonoLoader: no")
    assert store.append([broken]) == 0
    assert len(store) == 0


def test_new_tracks_invalidate_the_projection(tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    store = MapStore.load(tmp_path / "map")
    store.append([_profile(audio, 1.0)])
    store.set_coords(np.array([[0.0, 0.0]]))
    assert store.projected

    other = tmp_path / "b.mp3"
    other.write_bytes(b"y")
    store.append([_profile(other, 2.0)])
    # Le coordinate di prima valgono per una libreria che non esiste più.
    assert not store.projected


def test_coordinates_must_cover_every_track(tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    store = MapStore.load(tmp_path / "map")
    store.append([_profile(audio, 1.0)])
    try:
        store.set_coords(np.array([[0.0, 0.0], [1.0, 1.0]]))
    except ValueError:
        return
    raise AssertionError("due coordinate per un brano solo sono passate")


def test_pending_skips_what_is_already_there_and_catches_what_changed(tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    other = tmp_path / "b.mp3"
    other.write_bytes(b"y")

    store = MapStore.load(tmp_path / "map")
    store.append([_profile(audio, 1.0)])
    assert store.pending([audio, other]) == [other]

    audio.write_bytes(b"a different track, same name")
    assert store.pending([audio, other]) == [audio, other]


def test_a_job_killed_mid_write_leaves_the_two_files_agreeing(tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    store = MapStore.load(tmp_path / "map")
    store.append([_profile(audio, 1.0)])

    # Il job è stato ucciso dopo aver scritto il vettore e prima della riga.
    with store.embeddings_file.open("ab") as fh:
        fh.write(np.ones(EMBEDDING_DIM, dtype=np.float32).tobytes())

    again = MapStore.load(tmp_path / "map")
    assert len(again) == 1
    assert len(again.embeddings) == 1


def test_similar_finds_the_closest_sound(tmp_path):
    store = MapStore.load(tmp_path / "map")
    vectors = [np.zeros(EMBEDDING_DIM, dtype=np.float32) for _ in range(3)]
    vectors[0][0] = 1.0
    vectors[1][0], vectors[1][1] = 1.0, 0.05     # quasi identico al primo
    vectors[2][1] = 1.0                          # ortogonale
    for i, vector in enumerate(vectors):
        path = tmp_path / f"{i}.mp3"
        path.write_bytes(b"x")
        profile = TrackProfile(path=path, embedding=vector)
        store.append([profile])

    best = store.similar(0, k=2)
    assert [i for i, _ in best] == [1, 2]
    assert best[0][1] > best[1][1]


def test_the_same_folder_twice_does_not_analyze_anything_again(tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    store = MapStore.load(tmp_path / "map")
    store.append([_profile(audio, 1.0)])

    # Riselezionare la stessa cartella: la coda esce vuota.
    assert store.pending([audio]) == []
    assert MapStore.load(tmp_path / "map").pending([audio]) == []


def test_the_same_track_spelled_two_ways_is_one_track(tmp_path, monkeypatch):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    store = MapStore.load(tmp_path / "map")
    store.append([_profile(audio, 1.0)])

    # Lo stesso file indicato in modo relativo, come farebbe la riga di
    # comando lanciata da dentro la cartella.
    monkeypatch.chdir(tmp_path)
    from pathlib import Path as P
    assert store.pending([P("a.mp3")]) == []


def test_a_track_analyzed_again_replaces_itself(tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    store = MapStore.load(tmp_path / "map")
    store.append([_profile(audio, 1.0, bpm=120.0)])

    # Il file cambia (i tag riscritti bastano: mtime e dimensione si
    # muovono), quindi torna in coda e viene rianalizzato.
    audio.write_bytes(b"the same track, new tags")
    assert store.pending([audio]) == [audio]
    store.append([_profile(audio, 2.0, bpm=128.0)])

    # Sulla mappa resta UN punto solo, quello dell'ultima analisi.
    again = MapStore.load(tmp_path / "map")
    assert len(again) == 1
    assert again.rows[0]["bpm"] == 128.0
    assert len(again.embeddings) == 1
    assert again.embeddings[0][0] == 2.0


def test_the_map_stays_usable_while_a_job_adds_tracks(tmp_path):
    store = MapStore.load(tmp_path / "map")
    for i in range(2):
        path = tmp_path / f"{i}.mp3"
        path.write_bytes(b"x")
        store.append([_profile(path, float(i))])
    store.set_coords(np.array([[0.0, 0.0], [1.0, 1.0]]))

    # Il job aggiunge un brano: le coordinate di prima restano buone per i
    # due di prima, e il terzo aspetta la prossima proiezione.
    third = tmp_path / "2.mp3"
    third.write_bytes(b"x")
    store.append([_profile(third, 2.0)])

    again = MapStore.load(tmp_path / "map")
    assert len(again) == 3
    assert again.placed == 2          # la mappa continua a mostrarne due
    assert not again.projected        # ma non è completa


def test_coordinates_are_dropped_when_a_duplicate_shifts_the_order(tmp_path):
    store = MapStore.load(tmp_path / "map")
    for i in range(2):
        path = tmp_path / f"{i}.mp3"
        path.write_bytes(b"x")
        store.append([_profile(path, float(i))])
    store.set_coords(np.array([[0.0, 0.0], [1.0, 1.0]]))

    # Il primo brano cambia e viene rianalizzato: la sua riga vecchia sparisce
    # dal mezzo della fila, quindi le coordinate non corrispondono più.
    first = tmp_path / "0.mp3"
    first.write_bytes(b"different")
    store.append([_profile(first, 9.0)])

    again = MapStore.load(tmp_path / "map")
    assert len(again) == 2
    assert again.placed == 0          # meglio niente che il posto di un altro


def test_removing_a_track_takes_its_row_vector_and_place_with_it(tmp_path):
    store = MapStore.load(tmp_path / "map")
    for i in range(3):
        path = tmp_path / f"{i}.mp3"
        path.write_bytes(b"x")
        store.append([_profile(path, float(i))])
    store.set_coords(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))

    assert store.remove([tmp_path / "1.mp3"]) == 1

    again = MapStore.load(tmp_path / "map")
    assert [Path(r["path"]).name for r in again.rows] == ["0.mp3", "2.mp3"]
    assert again.embeddings.shape == (2, EMBEDDING_DIM)
    assert again.embeddings[1][0] == 2.0            # il vettore giusto è restato
    # I due che restano tengono il posto che avevano: toglierne uno non
    # sposta gli altri, e non obbliga a riproiettare.
    assert again.projected
    assert again.coords.tolist() == [[0.0, 0.0], [2.0, 2.0]]


def test_removing_something_that_is_not_there_changes_nothing(tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    store = MapStore.load(tmp_path / "map")
    store.append([_profile(audio, 1.0)])
    assert store.remove([tmp_path / "never-seen.mp3"]) == 0
    assert len(MapStore.load(tmp_path / "map")) == 1
