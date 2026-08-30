from pathlib import Path

import numpy as np
import pytest

from core.analysis import mood_scale
from core.analysis.map_profile import (
    MOOD_FIELDS,
    ProfileSettings,
    TrackProfile,
    mood_numbers,
    pooled_mood,
    gain_for_target,
    onset_regularity,
    rhythm_offset,
    segment_offsets,
    select_labels,
)

SETTINGS = ProfileSettings()


def test_twelve_windows_spread_over_the_whole_track():
    starts = segment_offsets(400, SETTINGS)
    assert len(starts) == 12
    # Centrate a 1/24, 3/24, 5/24... della durata: la prima sta nell'intro,
    # l'ultima nella coda, che con tre finestre ai quarti non si vedevano.
    assert starts[0] == pytest.approx(400 * 0.5 / 12 - 5)
    assert starts[-1] == pytest.approx(400 * 11.5 / 12 - 5)
    assert all(start + SETTINGS.segment_seconds <= 400 for start in starts)
    gaps = {round(b - a, 4) for a, b in zip(starts, starts[1:])}
    assert len(gaps) == 1                       # equidistanti


def test_the_rhythm_window_sits_in_the_middle_and_is_longer():
    # Non è una delle finestre dell'embedding: quelle sono da 10 s, e un
    # rilevatore di tempo su 10 s non trova abbastanza battute.
    assert SETTINGS.rhythm_seconds > SETTINGS.segment_seconds
    assert rhythm_offset(400, SETTINGS) == 185.0
    assert rhythm_offset(400, SETTINGS) + SETTINGS.rhythm_seconds <= 400


def test_a_track_shorter_than_the_rhythm_window_is_taken_whole():
    assert rhythm_offset(20, SETTINGS) == 0.0


def test_windows_never_hang_off_the_end():
    starts = segment_offsets(70, SETTINGS)
    assert all(0 <= s <= 70 - SETTINGS.segment_seconds for s in starts)


def test_a_track_shorter_than_one_window_is_analyzed_once():
    assert segment_offsets(8, SETTINGS) == [0.0]
    assert segment_offsets(10, SETTINGS) == [0.0]


def test_overlapping_windows_are_not_analyzed_twice():
    # Su un brano corto le posizioni si accavallano: analizzare due volte lo
    # stesso pezzo lo peserebbe il doppio nella media.
    starts = segment_offsets(35, SETTINGS)
    assert len(starts) < 12
    assert all(b - a > SETTINGS.segment_seconds / 4
               for a, b in zip(starts, starts[1:]))


def test_gain_brings_a_quiet_track_up_and_a_loud_one_down():
    assert gain_for_target(-20.0, -14.0) > 1.0
    assert gain_for_target(-14.0, -14.0) == 1.0
    assert gain_for_target(-8.0, -14.0) < 1.0


def test_gain_does_not_amplify_silence_into_noise():
    assert gain_for_target(-80.0) == 1.0        # praticamente muto: si lascia
    assert gain_for_target(None) == 1.0
    assert gain_for_target(-60.0) <= 10 ** (12 / 20)


def test_multi_label_keeps_everything_over_the_threshold():
    labels = ["Tech House", "Deep House", "Techno"]
    chosen = select_labels([0.85, 0.42, 0.04], labels, threshold=0.40, limit=4)
    assert chosen == [("Tech House", 0.85), ("Deep House", 0.42)]


def test_below_the_threshold_the_best_guess_survives():
    # Senza nemmeno un'etichetta il brano sparirebbe da ogni filtro: un
    # genere ce l'ha sempre, anche se il modello non è convinto.
    chosen = select_labels([0.03, 0.09], ["A", "B"], threshold=0.40, limit=4)
    assert chosen == [("B", 0.09)]


def test_labels_are_capped():
    activations = [0.9, 0.8, 0.7, 0.6, 0.5]
    chosen = select_labels(activations, list("abcde"), threshold=0.1, limit=2)
    assert [label for label, _ in chosen] == ["a", "b"]


def test_a_straight_kick_is_more_danceable_than_a_scattered_one():
    steady = onset_regularity(np.arange(0, 20, 0.5))
    scattered = onset_regularity([0, 0.1, 1.4, 1.5, 4.0, 7.9, 8.0, 8.05, 13.0, 20.0])
    assert steady == 1.0
    assert scattered < steady


def test_too_few_onsets_is_answered_with_i_do_not_know():
    assert onset_regularity([1.0, 2.0, 3.0]) is None
    assert onset_regularity([]) is None


def test_a_dead_worker_does_not_take_the_whole_queue_with_it(monkeypatch):
    """Essentia ogni tanto libera un puntatore che non ha allocato e il
    processo figlio muore sul colpo: non è un'eccezione, quindi dentro al
    figlio non c'è niente da intercettare e il pool resta inservibile.
    Prima che questo fosse gestito, un mp3 così buttava giù un job da
    cinquanta ore dopo sedicimila brani analizzati bene.
    """
    import concurrent.futures
    from concurrent.futures.process import BrokenProcessPool

    from core.analysis import map_profile

    pools = []

    class FakePool:
        """Il primo pool ne consegna tre e poi muore; il secondo lavora."""
        def __init__(self, **kwargs):
            pools.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, fn, paths):
            paths = list(paths)
            if len(pools) == 1:
                for path in paths[:3]:
                    yield map_profile.TrackProfile(path=path)
                raise BrokenProcessPool("figlio morto in codice nativo")
            for path in paths:
                yield map_profile.TrackProfile(path=path)

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", FakePool)
    queue = [Path(f"/x/{i}.mp3") for i in range(10)]
    out = list(map_profile.profile_many(queue, workers=2))

    # Nessun brano perso per strada, e nell'ordine di partenza.
    assert [p.path for p in out] == queue
    # I due che erano in volo quando il pool è morto: segnati falliti, non
    # scritti sulla mappa, quindi il rilancio li rimette in coda da solo.
    assert [p.path.name for p in out if p.error] == ["3.mp3", "4.mp3"]
    assert len(pools) == 2                      # ne ha aperto uno nuovo


# --- il mood come numero, sulla riga ---------------------------------------

LABELS = ["Dark", "Happy", "Energetic"]


def test_the_three_numbers_come_out_of_the_activations():
    # Ogni lato e' la MEDIA delle sue etichette, non la somma: le liste vere
    # sono da 8 e da 13, e sommarle farebbe entrare il fondo della sigmoide
    # 13 volte da una parte e 8 dall'altra. Vedi `mood_scale.SIDES`.
    buie, chiare = mood_scale.SIDES
    dark, bright = 0.60 / buie, 0.20 / chiare
    numbers = mood_numbers([0.60, 0.20, 0.90], LABELS, SETTINGS)
    assert numbers["valence"] == pytest.approx((bright - dark) / (bright + dark),
                                               abs=1e-4)
    assert numbers["valence"] < 0                       # il buio vince
    assert numbers["mood_evidence"] == round(dark + bright, 3)
    # I pesi si scrivono dal piu' forte, come la confidenza dei generi.
    assert numbers["mood_conf"] == "Energetic:0.900; Dark:0.600; Happy:0.200"


def test_a_track_the_model_reads_as_colourless_gets_no_valence():
    # `None` e non zero: zero direbbe "in mezzo fra buio e chiaro", che e'
    # un'altra cosa da "di questo non si sa".
    numbers = mood_numbers([0.0, 0.0, 0.90], LABELS, SETTINGS)
    assert numbers["valence"] is None
    assert numbers["mood_evidence"] == 0.0
    assert numbers["mood_conf"] == "Energetic:0.900"


def test_the_row_carries_the_three_numbers_next_to_the_words():
    profile = TrackProfile(path=Path("/x/a.mp3"),
                           moods=[("Dark", 0.62), ("Deep", 0.41)],
                           mood_numbers=mood_numbers([0.60, 0.20, 0.90],
                                                     LABELS, SETTINGS))
    row = profile.to_row()
    assert row["moods"] == "Dark; Deep"
    assert row["valence"] == mood_numbers([0.60, 0.20, 0.90],
                                          LABELS, SETTINGS)["valence"]
    assert row["mood_conf"].startswith("Energetic:0.900")


def test_a_row_from_before_the_numbers_existed_writes_them_empty():
    row = TrackProfile(path=Path("/x/a.mp3"), moods=[("Dark", 0.6)]).to_row()
    assert all(row[name] is None for name in MOOD_FIELDS)


def test_the_numbers_are_read_from_the_vector_that_goes_on_disk():
    """È la scelta da cui dipende che i brani vecchi e i nuovi stiano sulla
    stessa scala: il backfill ha solo l'embedding, quindi anche l'analisi di
    un brano nuovo deve partire da lì."""
    seen = []

    def head(vectors):
        seen.append(np.asarray(vectors))
        return np.asarray(vectors)[:, :3]

    stored = np.arange(1280, dtype=np.float32)
    assert pooled_mood(head, stored).tolist() == [0.0, 1.0, 2.0]
    # Una riga sola, il vettore intero: è quello che sta in `embeddings.f32`.
    assert seen[0].shape == (1, 1280)


def test_the_two_poolings_really_do_disagree():
    """Media-poi-testa e testa-poi-media non sono la stessa cosa, e questo è
    il motivo per cui la scelta esiste. Su una testa lineare coinciderebbero
    e non ci sarebbe niente da decidere; la testa vera ha una sigmoide."""
    def head(vectors):
        return 1 / (1 + np.exp(-np.asarray(vectors, dtype=float)))

    # Due fettine molto diverse: una scurissima, una per niente.
    slices = np.array([[6.0], [-6.0]])
    mean_of_predictions = head(slices).mean(axis=0)[0]
    prediction_of_mean = head(slices.mean(axis=0)[None, :])[0][0]
    assert mean_of_predictions == pytest.approx(0.5, abs=1e-3)
    assert prediction_of_mean == pytest.approx(0.5, abs=1e-3)

    # Sul caso simmetrico coincidono; basta sbilanciarle e si separano.
    slices = np.array([[6.0], [0.0]])
    assert head(slices).mean(axis=0)[0] == pytest.approx(0.7487, abs=1e-3)
    assert head(slices.mean(axis=0)[None, :])[0][0] == pytest.approx(0.9526,
                                                                     abs=1e-3)
