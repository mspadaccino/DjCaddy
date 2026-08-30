"""Il backfill del mood: quello che rifà i numeri senza riaprire i file."""

import numpy as np
import pytest

import mood_cli
from core.analysis import mood_scale
from core.analysis.map_profile import EMBEDDING_DIM, ProfileSettings, TrackProfile
from core.analysis.map_store import MapStore

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


# I tre campi in sé sono provati in `test_map_profile`, dove sta la funzione
# che li scrive: qui `written` è la stessa, e riprovarla vorrebbe dire due
# copie dello stesso test che un giorno diranno due cose diverse.


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
    written = [row["valence"] for row in again.rows]
    assert written[0] < 0 < written[1]                   # buio, poi chiaro
    assert written == [mood_cli.written(a, LABELS, SETTINGS)["valence"]
                       for a in ([0.60, 0.20, 0.90], [0.10, 0.70, 0.30])]
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


def test_the_backfill_and_the_analyzer_are_the_same_function():
    """Non "danno lo stesso risultato": SONO la stessa funzione.

    Il test di prima confrontava il risultato, e passava mentre le due strade
    calcolavano il numero da due pooling diversi — i brani vecchi dalla media
    dei vettori, i nuovi dalla media delle previsioni. Confrontare i risultati
    su un vettore di attivazioni non poteva vederlo, perche' il vettore gliela
    passavo io identico a tutte e due.
    """
    from core.analysis.map_profile import mood_numbers

    assert mood_cli.written is mood_numbers
    assert mood_cli.FIELDS == ("valence", "mood_evidence", "mood_conf")


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


def test_the_check_counts_the_tracks_that_had_no_arrow_before(tmp_path,
                                                              monkeypatch):
    """Il secondo guadagno, e va contato a parte: brani a cui nessuna
    etichetta passava la soglia e che quindi in tabella non avevano nessuna
    freccia, mentre di prove di colore ne avevano."""
    store = _store(tmp_path, (0.04, 0.0, 0.0), (0.60, 0.20, 0.90))
    store.rows[0]["moods"] = ""                  # sotto soglia: niente parole
    store.rows[1]["moods"] = "Energetic; Dark"
    monkeypatch.setattr(mood_cli, "_head", lambda: (_predict, LABELS))

    report = mood_cli.check(store, 10, SETTINGS)
    assert report["newly measured"] == 0.5
    # Un brano solo con tutte e due le letture: la correlazione non si fa su
    # uno, e la voce non compare invece di uscire `nan`.
    assert "agrees with the old reading" not in report


def test_a_track_whose_words_were_all_neutral_counts_as_newly_measured(
        tmp_path, monkeypatch):
    """Il caso che la prima versione di questo conto non vedeva: le parole
    c'erano, ma erano tutte neutre, quindi la valence vecchia leggeva 0,00 —
    che sembra un numero e non lo e'. Le prove sotto soglia un verso ce
    l'hanno, e questo e' il brano a cui la lettura nuova serve davvero."""
    store = _store(tmp_path, (0.03, 0.0, 0.95))
    store.rows[0]["moods"] = "Energetic"         # neutra, e sola
    monkeypatch.setattr(mood_cli, "_head", lambda: (_predict, LABELS))

    assert mood_scale.valence("Energetic") == 0.0
    assert mood_cli.check(store, 10, SETTINGS)["newly measured"] == 1.0


def test_the_check_separates_the_three_changes(tmp_path, monkeypatch):
    """Fra la valence vecchia e la nuova ci sono tre cambiamenti
    sovrapposti, e guardarli insieme non dice quale ha mosso cosa."""
    store = _store(tmp_path, (0.60, 0.20, 0.90), (0.10, 0.70, 0.30),
                   (0.40, 0.40, 0.10))
    for row, words in zip(store.rows, ("Dark; Energetic", "Happy", "Dark; Happy")):
        row["moods"] = words
    monkeypatch.setattr(mood_cli, "_head", lambda: (_predict, LABELS))

    report = mood_cli.check(store, 10, SETTINGS)
    assert {"weights vs ranks", "dropping the neutrals",
            "old vs new, both changes"} <= set(report)
    # E ogni lettura candidata porta i suoi decili e da che parte cade.
    for name, _, _ in mood_cli.CANDIDATES:
        assert len(report[f"{name} · deciles"]) == 9
        assert 0.0 <= report[f"{name} · below zero"] <= 1.0


def test_the_diluted_reading_is_the_old_one_with_the_new_weights():
    """E' il gradino di mezzo: pesi veri, ma neutre ancora nel
    denominatore. Serve solo a separare i due cambiamenti."""
    whole = {"Dark": 0.5, "Energetic": 0.9}
    dark = 0.5 / mood_scale.SIDES[0]
    assert mood_scale.valence_of(whole) == -1.0
    assert mood_scale.valence_of(whole, dilute=True) == pytest.approx(
        -dark / (dark + 0.9), abs=1e-4)
    # Senza colore non risponde in nessuno dei due modi.
    assert mood_scale.valence_of({"Energetic": 0.9}, dilute=True) is None


def test_the_check_asks_whether_the_faint_evidence_says_the_same_thing(
        tmp_path, monkeypatch):
    """La domanda che decide se le attivazioni sotto soglia sono un segnale
    debole o rumore: la valence sulle sole forti contro quella sulle sole
    deboli, sui brani che hanno tutte e due."""
    store = _store(tmp_path, (0.60, 0.02, 0.0), (0.02, 0.60, 0.0),
                   (0.60, 0.0, 0.0), (0.0, 0.60, 0.0))
    monkeypatch.setattr(mood_cli, "_head", lambda: (_predict, LABELS))

    report = mood_cli.check(store, 10, SETTINGS)
    # Solo i primi due hanno prove da tutte e due le parti.
    assert report["...measured on"] == 0.5
    assert "faint vs strong" in report


def test_the_faint_reading_is_anti_correlated_even_on_pure_noise():
    """Il motivo per cui quel numero da solo non si legge: l'etichetta che
    VINCE viene promossa sopra soglia, quindi quello che resta sotto e' fatto
    soprattutto del lato che ha perso. Su attivazioni completamente casuali,
    senza nessuna musica dentro, la correlazione esce comunque nettamente
    negativa — ed e' contro QUELLA che va confrontato il dato vero."""
    rng = np.random.default_rng(0)
    words = sorted(mood_scale.DARK | mood_scale.BRIGHT)
    pairs = []
    for _ in range(800):
        whole = {w: float(rng.beta(1.2, 12)) for w in words}
        above = mood_scale.valence_of(whole, floor=0.05)
        below = mood_scale.valence_of(whole, ceiling=0.05)
        if above is not None and below is not None:
            pairs.append((above, below))
    assert np.corrcoef(*np.array(pairs).T)[0, 1] < -0.2


def test_the_check_says_what_a_threshold_costs(tmp_path, monkeypatch):
    """Alzare la soglia non e' gratis: i brani le cui uniche prove di colore
    stanno sotto restano senza nessuna lettura."""
    store = _store(tmp_path, (0.03, 0.0, 0.90), (0.60, 0.20, 0.90))
    monkeypatch.setattr(mood_cli, "_head", lambda: (_predict, LABELS))

    report = mood_cli.check(store, 10, SETTINGS)
    assert report["balanced · no reading"] == 0.0
    assert report["floor 0.05 · no reading"] == 0.5


def test_the_faint_sample_picks_only_the_tracks_under_discussion(tmp_path,
                                                                 monkeypatch):
    """Brani il cui colore viene SOLO da prove sotto soglia: sono gli unici
    su cui nessuna misura interna puo' decidere, perche' di forte non hanno
    niente con cui confrontarsi."""
    store = _store(tmp_path,
                   (0.03, 0.0, 0.90),      # solo debole: e' dei nostri
                   (0.60, 0.0, 0.90),      # forte: no
                   (0.0, 0.02, 0.90),      # solo debole: e' dei nostri
                   (0.0, 0.0, 0.90))       # nessun colore: no
    monkeypatch.setattr(mood_cli, "_head", lambda: (_predict, LABELS))

    picked = mood_cli.faint_sample(store, 10, SETTINGS)
    assert [row["name"] for row in picked] == ["0.mp3", "2.mp3"]
    # Dal piu' buio al piu' chiaro: e' un ordine da ascoltare in fila.
    assert picked[0]["valence"] < picked[1]["valence"]


def test_a_library_where_nobody_needs_the_faint_evidence_gives_no_sample(
        tmp_path, monkeypatch):
    store = _store(tmp_path, (0.60, 0.0, 0.90), (0.0, 0.70, 0.90))
    monkeypatch.setattr(mood_cli, "_head", lambda: (_predict, LABELS))
    assert mood_cli.faint_sample(store, 10, SETTINGS) == []


def test_the_spread_sample_walks_the_whole_scale(tmp_path):
    """A backfill fatto il numero sta sulla riga: niente modello, niente
    audio. Si prende a passo costante sul RANGO, o venti brani finirebbero
    ammassati dove la libreria e' fitta."""
    store = _store(tmp_path, *[(0.0, 0.0, 0.0)] * 9)
    # Una valence ammassata in basso e una sola in cima: sul valore grezzo
    # il passo costante non se ne accorgerebbe.
    for row, value in zip(store.rows, [0.01, 0.02, 0.03, 0.04, 0.05,
                                       0.06, 0.07, 0.08, 0.99]):
        row["valence"] = value

    picked = mood_cli.spread_sample(store, 3)
    assert [row["valence"] for row in picked] == [0.01, 0.04, 0.07]
    assert [row["rank"] for row in picked] == [0.0, 0.375, 0.75]


def test_the_spread_sample_skips_what_has_no_reading(tmp_path):
    store = _store(tmp_path, *[(0.0, 0.0, 0.0)] * 3)
    store.rows[0]["valence"] = -0.5
    store.rows[1]["valence"] = 0.5
    store.rows[2]["moods"] = ""          # ne' numero ne' parole
    assert len(mood_cli.spread_sample(store, 10)) == 2
