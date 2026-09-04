"""La logica della vista Map, dov'è finita dopo la Fase 1: `core.viz` per le
figure e i filtri, `core.analysis` per playlist e ordinamenti. I casi legati
allo stato Streamlit (seed in sessione, liste chieste, selezione dei due
grafici) se ne sono andati con quella app: il lato Qt ha i suoi, in
`test_qt_map_panels`."""

import numpy as np
import pandas as pd

from core.analysis.dj_export import playlist_positions
from core.analysis.mixing import TransitionCost, sorted_after
from core.viz.filters import matching_tracks
from core.viz.map_figure import (AXIS_CENTRES, AXIS_FIELDS, AXIS_HELP,
                                 DEFAULT_AXES, FLAT_SIZE, SIZE_FIELDS, SKIN,
                                 axis_guide, build_figure, guide_caption,
                                 marker_sizes)


def _library() -> pd.DataFrame:
    return pd.DataFrame([
        {"name": "Madonna - Lucky Star (Extended Dance Remix).mp3",
         "folder": "/DJSet/80s Extended"},
        {"name": "Bananarama - Venus (Extended Mix).mp3",
         "folder": "/DJSet/80s Extended"},
        {"name": "Corona - Rhythm Of The Night - Optical Disco Remix.mp3",
         "folder": "/DJSet/90s"},
        {"name": "untitled.mp3", "folder": "/DJSet/Madonna B-sides"},
    ])


def _all(frame):
    return np.arange(len(frame))


def test_words_may_arrive_in_any_order():
    frame = _library()
    assert matching_tracks(frame, _all(frame), ["madonna", "lucky"]) == [0]
    assert matching_tracks(frame, _all(frame), ["lucky", "madonna"]) == [0]


def test_words_need_not_be_next_to_each_other():
    # "night remix" sta agli estremi del titolo, con altre parole in mezzo:
    # una ricerca per sottostringa contigua non lo troverebbe.
    frame = _library()
    assert matching_tracks(frame, _all(frame), ["night", "remix"]) == [2]


def test_every_word_has_to_appear():
    frame = _library()
    assert matching_tracks(frame, _all(frame), ["madonna", "venus"]) == []


def test_the_folder_counts_as_well_as_the_name():
    # L'artista a volte è solo nella cartella, e cercarlo deve trovarlo.
    frame = _library()
    assert matching_tracks(frame, _all(frame), ["madonna"]) == [0, 3]


def test_case_does_not_matter():
    frame = _library()
    assert matching_tracks(frame, _all(frame), ["BANANARAMA"]) == [1]


def test_the_search_stays_inside_the_pool():
    """I filtri della pagina restringono già l'universo: la ricerca non deve
    ripescare un brano che quelli hanno escluso."""
    frame = _library()
    assert matching_tracks(frame, [1, 2], ["madonna"]) == []


def _map() -> dict:
    return {"/DJSet/80s/lucky star.mp3": 0,
            "/DJSet/80s/venus.mp3": 1,
            "/DJSet/90s/rhythm.mp3": 2}


def test_a_playlist_comes_back_in_its_own_order():
    found, missing = playlist_positions(
        ["/DJSet/90s/rhythm.mp3", "/DJSet/80s/lucky star.mp3"], _map())
    assert found == [2, 0]
    assert missing == []


def test_a_track_the_map_does_not_have_is_reported_not_dropped_silently():
    found, missing = playlist_positions(
        ["/DJSet/80s/venus.mp3", "/Elsewhere/unknown.mp3"], _map())
    assert found == [1]
    assert missing == ["/Elsewhere/unknown.mp3"]


def test_the_same_track_twice_lands_in_the_playlist_once():
    found, missing = playlist_positions(
        ["/DJSet/80s/venus.mp3", "/DJSet/80s/venus.mp3"], _map())
    assert found == [1]
    assert missing == []


def test_a_moved_library_is_matched_by_file_name():
    """La playlist di ieri punta al disco di ieri: il brano è lo stesso."""
    found, missing = playlist_positions(
        ["/Volumes/OldDrive/80s/venus.mp3", "../90s/rhythm.mp3"], _map())
    assert found == [1, 2]
    assert missing == []


def test_the_path_wins_over_the_name():
    """Due cartelle con lo stesso nome dentro: chi ha il percorso giusto va
    al suo posto, non al primo omonimo."""
    twins = {"/DJSet/a/venus.mp3": 0, "/DJSet/b/venus.mp3": 1}
    found, _ = playlist_positions(["/DJSet/b/venus.mp3"], twins)
    assert found == [1]


def test_an_accent_written_the_other_way_is_the_same_track():
    """macOS scrive "Hervé" decomposto, rekordbox lo ricompone: stessa
    parola, due stringhe, e il brano spariva dalla playlist."""
    import unicodedata
    decomposto = unicodedata.normalize("NFD", "/DJSet/80s/Hervé.mp3")
    composto = unicodedata.normalize("NFC", "/DJSet/80s/Hervé.mp3")
    assert decomposto != composto
    found, missing = playlist_positions([composto], {decomposto: 7})
    assert found == [7]
    assert missing == []


def test_the_accent_is_matched_by_name_too_when_the_library_has_moved():
    import unicodedata
    decomposto = unicodedata.normalize("NFD", "/DJSet/80s/Hervé.mp3")
    composto = unicodedata.normalize("NFC", "/Volumes/OldDrive/80s/Hervé.mp3")
    found, missing = playlist_positions([composto], {decomposto: 7})
    assert found == [7]
    assert missing == []


def _drawn(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        "x": np.arange(n, dtype=float), "y": np.arange(n, dtype=float),
        "index": np.arange(n), "name": [f"{i}.flac" for i in range(n)],
        "bpm": [120] * n, "camelot": ["8A"] * n, "genres": ["House"] * n,
        "top_genre": ["House"] * n, "genre_key": ["House"] * n,
        "_size": [7.0] * n})


def _ring(figure, name):
    """Il tracciato di un anello, se disegnato."""
    return next((t for t in figure.data if t.name == name), None)


def test_the_tick_no_longer_draws_a_ring():
    """La spunta dura il tempo di premere il pulsante accanto, e per
    cerchiarla in tempo la mappa avrebbe dovuto ridisegnare ottantamila punti
    a ogni casella. Quello che la spunta diventera' — la catena o la
    playlist — il suo anello ce l'ha gia'."""
    coords = np.column_stack([np.arange(4.0), np.zeros(4)])
    figure = build_figure(_drawn(), ["House"], coords, playlist=[], seed=None)
    assert _ring(figure, "being picked") is None


def test_playlist_tracks_are_marked_by_the_path_alone():
    """Il segno della playlist e' il percorso — punti bianchi, linea,
    numeri. L'anello verde marcava lo stesso identico insieme e non sapeva
    dire l'ordine: tolto come doppione, appena la voce in legenda l'ha reso
    visibile."""
    coords = np.column_stack([np.arange(4.0), np.zeros(4)])
    figure = build_figure(_drawn(), ["House"], coords, playlist=[0, 2],
                          seed=None)
    path = _ring(figure, "playlist")
    assert list(path.x) == [0.0, 2.0]
    assert _ring(figure, "in the playlist") is None


def test_no_ring_when_there_is_nothing_to_ring():
    coords = np.column_stack([np.arange(4.0), np.zeros(4)])
    figure = build_figure(_drawn(), ["House"], coords, playlist=[], seed=None)
    assert _ring(figure, "playlist") is None
    assert _ring(figure, "selected") is None


def test_the_selected_group_gets_the_ink_ring():
    """Lazo e riquadro cerchiano quello che hanno preso, col nero del seme:
    è la stessa cosa detta al plurale — "sto lavorando su questi"."""
    coords = np.column_stack([np.arange(4.0), np.zeros(4)])
    figure = build_figure(_drawn(), ["House"], coords, playlist=[], seed=None,
                          selected=[0, 3])
    ring = _ring(figure, "selected")
    assert list(ring.x) == [0.0, 3.0]
    assert ring.marker.line.color == SKIN["light"]["ink"]


def test_the_playlist_path_does_not_need_a_selection():
    """Il principio di fondo: quello che è in playlist si vede sempre."""
    coords = np.column_stack([np.arange(4.0), np.zeros(4)])
    figure = build_figure(_drawn(), ["House"], coords, playlist=[1],
                          seed=None, selected=[])
    assert list(_ring(figure, "playlist").x) == [1.0]
    assert _ring(figure, "selected") is None


def _line_cost():
    """Quattro brani in fila nel suono: il costo cresce con l'angolo."""
    angles = np.radians([0, 15, 30, 45])
    vectors = np.column_stack([np.cos(angles), np.sin(angles)])
    return TransitionCost(vectors, [120] * 4, ["8A"] * 4)


def test_a_group_appended_starts_from_what_the_tail_reaches_cheapest():
    """La giuntura con quello che c'è già non si lascia al caso: il primo del
    gruppo è quello che costa meno raggiungere dall'ultimo della playlist."""
    cost = _line_cost()
    assert sorted_after(cost, [0], [3, 1, 2]) == [1, 2, 3]


def test_a_group_sorted_onto_nothing_picks_its_own_start():
    """Senza niente prima, non c'è una giuntura da rispettare: resta magic
    sort, che sceglie da dove partire e lascia una catena senza salti."""
    cost = _line_cost()
    order = sorted_after(cost, [], [2, 0, 1])
    assert sorted(order) == [0, 1, 2]
    assert all(abs(b - a) == 1 for a, b in zip(order, order[1:]))


def _legend_of(**kwargs) -> list[str]:
    figure = build_figure(_drawn(), ["House"], np.column_stack(
        [np.arange(4.0), np.arange(4.0)]), **kwargs)
    # `showlegend` non impostato vuol dire "si'" per default, non "no": e' il
    # False esplicito che nasconde.
    return [t.name for t in figure.data if t.showlegend is not False]


def test_the_selection_rings_say_what_they_are_in_the_legend():
    """Senza, restano cerchi di colori diversi e nessun posto dove chiedere
    cosa vogliano dire."""
    names = _legend_of(playlist=[], seed=None, chained=[0], selected=[2])
    assert {"in the chain", "selected"} <= set(names)


def test_an_empty_ring_promises_no_colour():
    # Una voce per un insieme vuoto sarebbe una legenda che promette un
    # colore introvabile sul disegno.
    names = _legend_of(playlist=[], seed=None, chained=[], selected=[])
    assert "in the chain" not in names
    assert "selected" not in names


def test_the_playlist_has_one_sign_and_one_name():
    """Il percorso — punti bianchi, linea, numeri — E' il segno della
    playlist. L'anello verde marcava lo stesso identico insieme, e il
    doppione e' saltato fuori appena ha avuto la sua voce in legenda (due
    voci, un insieme solo): tolto l'anello, resta il segno che porta piu'
    informazione, cioe' l'ordine."""
    names = _legend_of(playlist=[0, 1], seed=None)
    assert names.count("playlist") == 1
    assert "in the playlist" not in names


def test_the_seed_has_its_own_entry():
    names = _legend_of(playlist=[], seed=3)
    assert "seed" in names


def test_no_size_option_promises_a_measure_it_does_not_show():
    """La voce si chiamava "energy" e mostrava `lufs`: la loudness dice quanto
    ha spinto chi ha masterizzato, non quanto spinge il brano."""
    assert SIZE_FIELDS.get("loudness") == "lufs"
    # Ora l'energia c'e', ed e' la sua: quattro misure sue, non la loudness.
    assert SIZE_FIELDS.get("energy") == "energy"


def test_the_energy_of_a_library_nobody_has_measured_is_no_size_at_all():
    """Finche' il backfill non e' passato la colonna e' vuota, e i punti
    devono restare tutti uguali invece di ammassarsi al diametro minimo."""
    frame = pd.DataFrame({"energy": [np.nan, np.nan, np.nan]})
    assert marker_sizes(frame, "energy") == FLAT_SIZE


def test_the_two_suggestion_lists_get_their_own_rings():
    """Le proposte del seme non erano cerchiate: la mappa e' proprio il posto
    in cui si guarda per decidere il prossimo brano."""
    names = _legend_of(playlist=[], seed=0, mixes=[1, 2], alike=[3])
    assert "mixes out of it" in names
    assert "sounds like it" in names


def test_the_track_playing_gets_a_pulsing_svg_ring():
    """Un anello SVG e non gl: gli altri segni dicono cosa un brano E',
    questo dice cosa sta succedendo adesso, e a dirlo e' il pulsare — che
    e' un'animazione CSS della pagina, e trova il punto perche' e' il solo
    nel livello SVG. Se qualcun altro finisse in quel livello pulserebbe
    anche lui."""
    figure = build_figure(_drawn(), ["House"],
                          np.column_stack([np.arange(4.0), np.arange(4.0)]),
                          playlist=[], seed=None, playing=2)
    ring = [t for t in figure.data if t.name == "playing"]
    assert len(ring) == 1
    assert ring[0].type == "scatter"
    assert ring[0].marker.symbol == "circle-open"
    assert ring[0].showlegend is True
    assert [t.type for t in figure.data if t.type == "scatter"] == ["scatter"]


def test_with_nothing_playing_there_is_no_cross():
    names = _legend_of(playlist=[], seed=None, playing=None)
    assert "playing" not in names


def test_no_two_rings_share_a_colour():
    """Il rosa era ambra, e l'ambra accanto al giallo della catena erano due
    gialli: si distinguevano per diametro, cioe' bisognava misurarli."""
    # «playing» non è in lista: si distingue perché batte, non per il
    # colore — è l'inchiostro del tema, e sta accanto al seme senza
    # confondersi proprio perché si muove.
    for theme in ("light", "dark"):
        rings = [SKIN[theme][k] for k in
                 ("chained", "ink", "mixes", "alike", "pl_selection")]
        assert len(set(rings)) == len(rings), theme


# --- i quadranti: gli stessi brani su due misure a scelta -----------------

def _measured():
    """Quattro brani con due misure vere addosso, oltre alle coordinate."""
    frame = _drawn()
    frame["valence"] = [-0.8, -0.2, 0.3, 0.9]
    frame["energy"] = [0.1, 0.9, 0.4, 0.6]
    frame["bpm"] = [96.0, 120.0, 128.0, 140.0]
    return frame


def test_the_quadrants_draw_the_same_tracks_on_two_chosen_measures():
    frame = _measured()
    places = frame[["valence", "energy"]].to_numpy()
    figure = build_figure(frame, ["House"], places, playlist=[], seed=None,
                          axes=("valence", "energy"),
                          titles=("valence (mood)", "energy"))
    points = next(t for t in figure.data if t.name == "House")
    assert list(points.x) == [-0.8, -0.2, 0.3, 0.9]
    assert list(points.y) == [0.1, 0.9, 0.4, 0.6]
    assert figure.layout.xaxis.title.text == "valence (mood)"
    assert figure.layout.yaxis.title.text == "energy"


def test_the_rings_follow_the_tracks_onto_the_new_axes():
    """E' il punto di avere una funzione sola: il seme, la catena e le
    proposte dicono le stesse cose di qua e di la', invece di essere due
    schermi che non si parlano."""
    frame = _measured()
    places = frame[["valence", "energy"]].to_numpy()
    figure = build_figure(frame, ["House"], places, playlist=[1], seed=3,
                          axes=("valence", "energy"), titles=("v", "e"))
    path = _ring(figure, "playlist")
    assert list(path.x) == [-0.2] and list(path.y) == [0.9]
    seed = _ring(figure, "seed")
    assert list(seed.x) == [0.9] and list(seed.y) == [0.6]


def test_the_map_keeps_its_axes_hidden_and_its_square_scale():
    """Le due dimensioni della proiezione non sono misure: un numero su di
    esse non vuol dire niente, e stirarne una falserebbe le distanze."""
    coords = np.column_stack([np.arange(4.0), np.zeros(4)])
    figure = build_figure(_drawn(), ["House"], coords, playlist=[], seed=None)
    assert figure.layout.xaxis.visible is False
    assert figure.layout.yaxis.scaleanchor == "x"


def test_the_quadrant_axes_are_not_tied_to_each_other():
    """Portano due misure diverse — dei BPM e un rango — e legarle
    schiaccerebbe il disegno in una riga."""
    frame = _measured()
    figure = build_figure(frame, ["House"],
                          frame[["bpm", "energy"]].to_numpy(),
                          playlist=[], seed=None, axes=("bpm", "energy"),
                          titles=("BPM", "energy"))
    assert figure.layout.yaxis.scaleanchor is None
    assert figure.layout.xaxis.visible is True


def test_the_cross_sits_where_the_measure_has_its_own_middle():
    # Una misura sola ce l'ha: l'energia e' un rango sulla libreria, quindi
    # il suo mezzo E' la mediana per costruzione.
    assert axis_guide([0.1, 0.2, 0.3], "energy") == 0.5


def test_the_valence_does_not_get_to_call_its_zero_a_centre():
    """Misurata sulla libreria vera i nove decili erano tutti positivi: con
    la croce sullo zero i due quadranti bui sarebbero rimasti vuoti, e il
    grafico avrebbe detto che la libreria e' tutta allegra — che e' una
    proprieta' di come sono fatte le due liste di parole, non della musica."""
    assert "valence" not in AXIS_CENTRES
    assert axis_guide([0.3, 0.5, 0.7], "valence") == 0.5


def test_the_cross_falls_back_to_the_median_of_what_is_on_screen():
    assert axis_guide([96.0, 120.0, 128.0, 140.0], "bpm") == 124.0
    assert axis_guide([], "bpm") is None


def test_the_caption_says_which_of_the_two_kinds_of_middle_it_is():
    """Una riga tratteggiata a meta' del disegno sembra un centro assoluto,
    e su quasi tutte le misure e' invece la mediana di cio' che i filtri
    lasciano — cioe' si sposta appena si tocca un filtro."""
    told = guide_caption((124.0, 0.5), ("bpm", "energy"), ("BPM", "energy"))
    assert "median of what the filters leave" in told
    assert "middle of the measure itself" in told
    assert guide_caption((None, 0.5), ("bpm", "energy"), ("BPM", "e")) == ""


def test_the_default_axes_are_the_two_that_answer_the_question():
    """Valence e arousal, i due assi di Russell: dove sta questo brano fra il
    buio e il chiaro, fra il calmo e lo spinto."""
    assert all(name in AXIS_FIELDS for name in DEFAULT_AXES)
    # Il RANGO della valence: il numero firmato non e' centrato sullo zero e
    # non lo sara' mai, e su un asse conta dove sta un brano rispetto agli
    # altri, non un valore assoluto che il modello non sa dare.
    assert AXIS_FIELDS[DEFAULT_AXES[0]] == "valence_rank"
    assert AXIS_FIELDS[DEFAULT_AXES[1]] == "energy"


def test_the_valence_goes_on_the_axis_as_a_rank_not_as_a_signed_number():
    """Misurata sui pesi veri, la valence grezza ha il 94% della libreria
    sopra lo zero, e nessun rimedio sulle due liste di parole la centra: il
    modello ha imparato su un mondo dove 'happy' e' un'etichetta molto piu'
    frequente di 'sad'. Il rango un mezzo ce l'ha per costruzione."""
    from core.analysis import energy

    skewed = [0.07, 0.18, 0.26, 0.33, 0.39, 0.45, 0.51, 0.57, 0.64]
    assert min(skewed) > 0                       # nessuno sotto lo zero
    ranked = energy.ranks(skewed)
    assert float(np.median(ranked)) == 0.5       # il rango invece si centra

    assert AXIS_CENTRES["valence_rank"] == 0.5
    assert "valence" not in AXIS_CENTRES
    assert axis_guide(skewed, "valence_rank") == 0.5
    # Il numero firmato resta disponibile, per vedere la misura com'e'
    # invece di dov'e': ma non e' quello che si apre da se'.
    assert AXIS_FIELDS["valence · signed"] == "valence"


def test_every_axis_says_what_it_means():
    """Un asse che si chiama "valence" e va da 0 a 1 non si spiega da se':
    non dice in che unita' sia, ne' — che e' quello che conta — che i due
    estremi sono la TUA libreria e non una scala assoluta."""
    assert not [name for name in AXIS_FIELDS if name not in AXIS_HELP]
    # E le due che sono ranghi lo dicono, perche' e' l'equivoco possibile.
    for name in ("energy", "valence (mood)"):
        assert "rank" in AXIS_HELP[name]
