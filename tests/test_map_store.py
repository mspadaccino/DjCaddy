import json
from pathlib import Path

import numpy as np

from core.analysis.map_profile import EMBEDDING_DIM, TrackProfile
from core.analysis.map_store import MapStore


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


def test_a_duplicate_after_the_coordinates_leaves_them_alone(tmp_path):
    """Il caso che spegneva la mappa.

    Le coordinate coprono un prefisso; se il duplicato riguarda una riga che
    sta OLTRE quel prefisso, la fila coperta non si è mossa e le coordinate
    valgono ancora. Prima bastava l'esistenza di un duplicato qualunque per
    buttarle via, e da lì non si tornava indietro: ricalcolare le riscriveva,
    il caricamento dopo le scartava di nuovo.
    """
    store = MapStore.load(tmp_path / "map")
    for i in range(3):
        path = tmp_path / f"{i}.mp3"
        path.write_bytes(b"x")
        store.append([_profile(path, float(i))])
    store.set_coords(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))

    # L'ULTIMO brano cambia e viene rianalizzato: la fila non si accorcia nel
    # mezzo, quindi ogni coordinata indica ancora il file per cui è stata
    # calcolata. Quella del terzo è vecchia di un'analisi — il brano si è
    # spostato un po' — ma è il brano giusto, e una posizione vecchia si
    # aggiusta alla prossima proiezione. Una mappa spenta no.
    third = tmp_path / "2.mp3"
    third.write_bytes(b"different")
    store.append([_profile(third, 9.0)])

    again = MapStore.load(tmp_path / "map")
    assert len(again) == 3
    assert again.placed == 3
    assert again.coords.tolist() == [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]


def test_recomputing_after_a_duplicate_takes_effect(tmp_path):
    """Ricalcolare deve poter rimettere in piedi una mappa spenta."""
    store = MapStore.load(tmp_path / "map")
    for i in range(2):
        path = tmp_path / f"{i}.mp3"
        path.write_bytes(b"x")
        store.append([_profile(path, float(i))])
    store.set_coords(np.array([[0.0, 0.0], [1.0, 1.0]]))
    first = tmp_path / "0.mp3"
    first.write_bytes(b"different")
    store.append([_profile(first, 9.0)])
    assert MapStore.load(tmp_path / "map").placed == 0     # giustamente spenta

    fresh = MapStore.load(tmp_path / "map")
    fresh.set_coords(np.array([[5.0, 5.0], [6.0, 6.0]]))
    assert MapStore.load(tmp_path / "map").placed == 2


def test_a_job_appending_does_not_undo_a_projection_made_meanwhile(tmp_path):
    """Due processi sullo stesso `meta.json`.

    Il job appende brani per ore e l'app ricalcola la proiezione nel mentre.
    Il job si porta dietro dall'avvio la sua copia dei metadati: se
    riscrivesse anche il segno delle coordinate, cancellerebbe quello del
    ricalcolo e la mappa si spegnerebbe al brano successivo.
    """
    def _add(store, name, content=b"x"):
        path = tmp_path / name
        path.write_bytes(content)
        store.append([_profile(path, 1.0)])

    job = MapStore.load(tmp_path / "map")          # il job, aperto prima
    _add(job, "0.mp3")
    _add(job, "1.mp3")

    app = MapStore.load(tmp_path / "map")          # l'app, aperta dopo
    app.set_coords(np.array([[0.0, 0.0], [1.0, 1.0]]))
    assert MapStore.load(tmp_path / "map").placed == 2

    _add(job, "2.mp3")                             # il job continua
    assert MapStore.load(tmp_path / "map").placed == 2


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


def test_a_library_that_changed_disk_is_still_the_same_library(tmp_path):
    old_root = tmp_path / "old"
    (old_root / "house").mkdir(parents=True)
    audio = old_root / "house" / "a.mp3"
    audio.write_bytes(b"x")

    store = MapStore.load(tmp_path / "map")
    store.append([_profile(audio, 1.0)])
    store.set_coords(np.array([[0.0, 0.0]]))

    # Stessi file, stessa struttura, disco nuovo.
    new_root = tmp_path / "new"
    (new_root / "house").mkdir(parents=True)
    moved = new_root / "house" / "a.mp3"
    moved.write_bytes(b"x")
    audio.unlink()

    assert store.relocate(old_root, new_root) == (1, 0)

    again = MapStore.load(tmp_path / "map")
    assert again.rows[0]["path"] == str(moved)
    assert again.rows[0]["folder"] == str(moved.parent)
    # L'analisi non si rifà: il brano è riconosciuto al nuovo indirizzo.
    assert again.pending([moved]) == []
    # E il suo posto sulla mappa è rimasto dov'era.
    assert again.projected
    assert again.embeddings[0][0] == 1.0


def test_relocating_stops_at_the_folder_boundary(tmp_path):
    store = MapStore.load(tmp_path / "map")
    for name in ("disk/a.mp3", "disk backup/b.mp3"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        store.append([_profile(path, 1.0)])

    # "disk" non deve tirarsi dietro "disk backup".
    moved, missing = store.relocate(tmp_path / "disk", tmp_path / "elsewhere")
    assert (moved, missing) == (1, 1)   # spostata una riga, e là non c'è
    kept = [r["path"] for r in MapStore.load(tmp_path / "map").rows]
    assert str(tmp_path / "elsewhere" / "a.mp3") in kept
    assert str(tmp_path / "disk backup" / "b.mp3") in kept


def test_rewriting_the_rows_keeps_their_order_and_the_vectors(tmp_path):
    """Aggiungere un campo non deve scomporre la fila: e' l'ordine l'unica
    cosa che tiene allineati metadati, embedding e coordinate."""
    import numpy as np

    from core.analysis.map_store import MapStore

    store = MapStore.load(tmp_path)
    store.rows = [{"path": f"/lib/{i}.flac", "bpm": 120 + i} for i in range(5)]
    store.embeddings = np.zeros((5, 1280), dtype=np.float32)
    store.embeddings_file.write_bytes(store.embeddings.tobytes())
    for i, row in enumerate(store.rows):
        row["energy_pulse"] = i / 10
    assert store.rewrite() == 5

    again = MapStore.load(tmp_path)
    assert [r["path"] for r in again.rows] == [r["path"] for r in store.rows]
    assert [r["energy_pulse"] for r in again.rows] == [0.0, 0.1, 0.2, 0.3, 0.4]


def test_a_rewrite_that_dies_halfway_leaves_the_old_file_alone(tmp_path):
    import numpy as np

    from core.analysis.map_store import MapStore

    store = MapStore.load(tmp_path)
    store.rows = [{"path": "/lib/a.flac"}]
    store.embeddings_file.write_bytes(
        np.zeros((1, 1280), dtype=np.float32).tobytes())
    store.rewrite()
    store.rows = [{"path": "/lib/a.flac"}, {"path": object()}]   # non serializzabile
    store.embeddings = np.zeros((2, 1280), dtype=np.float32)
    try:
        store.rewrite()
    except TypeError:
        pass
    assert [r["path"] for r in MapStore.load(tmp_path).rows] == ["/lib/a.flac"]


def test_rewriting_fewer_rows_than_vectors_is_refused(tmp_path):
    """Il silenzio sarebbe peggio: `load` terrebbe solo il prefisso comune e
    meta' mappa sparirebbe senza dire niente."""
    import numpy as np
    import pytest

    from core.analysis.map_store import MapStore

    store = MapStore.load(tmp_path)
    store.rows = [{"path": "/lib/a.flac"}]
    store.embeddings = np.zeros((2, 1280), dtype=np.float32)
    with pytest.raises(ValueError):
        store.rewrite()


def test_a_rewrite_after_a_duplicate_was_absorbed_realigns_the_vectors(tmp_path):
    """Il caso che non si vede in memoria e che rovinerebbe la mappa per sempre.

    `load` assorbe i duplicati e compatta righe E vettori, quindi in memoria
    tornano pari; sul disco pero' gli embedding sono ancora quelli lunghi.
    Riscrivendo le sole righe, il caricamento dopo prenderebbe i PRIMI vettori
    invece di quelli scelti, e da li' in poi ogni brano avrebbe il vettore del
    vicino — senza un errore.
    """
    import numpy as np

    from core.analysis.map_store import EMBEDDING_DIM, MapStore

    store = MapStore.load(tmp_path)
    # b compare due volte: vale l'ultima, e la fila si accorcia NEL MEZZO.
    paths = ["/lib/a.flac", "/lib/b.flac", "/lib/b.flac", "/lib/c.flac"]
    store.rows_file.write_text(
        "".join(json.dumps({"path": p, "n": i}) + "\n"
                for i, p in enumerate(paths)), encoding="utf-8")
    vectors = np.zeros((4, EMBEDDING_DIM), dtype=np.float32)
    for i in range(4):
        vectors[i, 0] = i                      # ogni vettore si riconosce
    store.embeddings_file.write_bytes(vectors.tobytes())

    loaded = MapStore.load(tmp_path)
    assert [r["path"] for r in loaded.rows] == ["/lib/a.flac", "/lib/b.flac",
                                                "/lib/c.flac"]
    assert list(loaded.embeddings[:, 0]) == [0.0, 2.0, 3.0]

    for row in loaded.rows:                    # il backfill aggiunge un campo
        row["energy_pulse"] = 0.5
    loaded.rewrite()

    again = MapStore.load(tmp_path)
    assert [r["path"] for r in again.rows] == ["/lib/a.flac", "/lib/b.flac",
                                               "/lib/c.flac"]
    # Senza riscrivere anche i vettori qui uscirebbe [0, 1, 2]: /lib/b.flac si
    # ritroverebbe il vettore della sua copia scartata e /lib/c.flac quello di b.
    assert list(again.embeddings[:, 0]) == [0.0, 2.0, 3.0]


# --- il contratto della riga ----------------------------------------------

# Cosa la pagina si aspetta di trovare su una riga della mappa. Non e' un
# elenco decorativo: e' l'unico punto in cui il lato che SCRIVE le righe e il
# lato che le LEGGE si guardano in faccia. Un campo aggiunto a una colonna
# nuova e dimenticato in `to_row` non si vede finche' qualcuno non riapre la
# libreria settimane dopo e trova una colonna vuota senza spiegazione.
READ_BY_THE_PAGE = (
    "path", "name", "folder", "duration", "bpm", "camelot", "key", "lufs",
    "danceability", "genres", "top_genre", "moods", "confidence",
    # I quattro grezzi dell'energia: il voto da 1 a 10 e' un rango sulla
    # libreria intera e si calcola a ogni apertura, questi si salvano.
    "energy_density", "energy_bass", "energy_bright", "energy_pulse",
    # Il mood come numero, sui pesi veri di tutte e 56 le etichette.
    "valence", "mood_evidence", "mood_conf",
)


def test_a_track_analyzed_today_carries_everything_the_page_reads(tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"not really audio")
    store = MapStore.load(tmp_path / "map")
    store.append([_profile(audio, 1.0)])

    written = MapStore.load(tmp_path / "map").rows[0]
    assert not [name for name in READ_BY_THE_PAGE if name not in written]


def test_a_row_written_before_a_field_existed_still_loads(tmp_path):
    """La libreria ha ottantasettemila righe scritte prima che questi campi
    ci fossero: leggerle non deve rompersi, e il backfill le raggiunge."""
    import json

    directory = tmp_path / "map"
    directory.mkdir()
    (directory / "tracks.jsonl").write_text(
        json.dumps({"path": "/x/a.mp3", "name": "a.mp3", "folder": "/x",
                    "bpm": 128.0, "moods": "Dark"}) + "\n")
    np.full(EMBEDDING_DIM, 1.0, dtype=np.float32).tofile(
        directory / "embeddings.f32")
    row = MapStore.load(directory).rows[0]
    assert row.get("valence") is None and row.get("energy_bass") is None


def test_missing_under_names_the_tracks_gone_from_the_disk(tmp_path):
    kept, gone = tmp_path / "lib" / "a.mp3", tmp_path / "lib" / "b.mp3"
    kept.parent.mkdir()
    kept.write_bytes(b"x")
    gone.write_bytes(b"y")
    store = MapStore.load(tmp_path / "map")
    store.append([_profile(kept, 1.0), _profile(gone, 2.0)])
    gone.unlink()
    assert store.missing_under(tmp_path / "lib") == [str(gone)]
    # Fuori dalla radice non si guarda: "lib" non è "lib2".
    (tmp_path / "lib2").mkdir()
    assert store.missing_under(tmp_path / "lib2") == []
