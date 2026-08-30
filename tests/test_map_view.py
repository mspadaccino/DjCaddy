from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from streamlit_app.views.map_analysis import (SKIN, build_figure, matching_tracks,
                                playlist_positions, sorted_after)


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


def _drawn():
    return pd.DataFrame([
        {"index": i, "name": f"t{i}.mp3", "bpm": 120, "camelot": "8A",
         "genres": "House", "genre_key": "House", "x": float(i), "y": 0.0,
         "_size": 7.0}
        for i in range(4)])


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


def test_playlist_tracks_get_a_green_ring():
    coords = np.column_stack([np.arange(4.0), np.zeros(4)])
    figure = build_figure(_drawn(), ["House"], coords, playlist=[0, 2],
                          seed=None)
    ring = _ring(figure, "in the playlist")
    assert list(ring.x) == [0.0, 2.0]
    assert ring.marker.line.color == SKIN["light"]["kept"]


def test_no_ring_when_there_is_nothing_to_ring():
    coords = np.column_stack([np.arange(4.0), np.zeros(4)])
    figure = build_figure(_drawn(), ["House"], coords, playlist=[], seed=None)
    assert _ring(figure, "in the playlist") is None
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


def test_the_playlist_ring_does_not_need_a_selection():
    """Il principio di fondo: quello che è in playlist si vede sempre."""
    coords = np.column_stack([np.arange(4.0), np.zeros(4)])
    figure = build_figure(_drawn(), ["House"], coords, playlist=[1],
                          seed=None, selected=[])
    assert list(_ring(figure, "in the playlist").x) == [1.0]
    assert _ring(figure, "selected") is None


def _line_cost():
    """Quattro brani in fila su una retta: il costo è la distanza."""
    from core.analysis.mixing import TransitionCost
    coords = np.column_stack([np.arange(4.0), np.zeros(4)])
    return TransitionCost(coords, [120] * 4, ["8A"] * 4)


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


def test_the_chain_wears_its_own_ring_on_the_map(monkeypatch):
    """La catena si costruiva alla cieca: il Chain Maker sta sotto la mappa,
    ma sulla nuvola i suoi brani non si distinguevano dagli altri."""
    import streamlit as st

    from core.analysis.graph_playlist import GraphPlaylist
    from streamlit_app.views.graph_board import GRAPH_STATE
    from streamlit_app.views.map_analysis import chain_places

    graph = GraphPlaylist().start("/lib/a.flac")
    graph.add("/lib/a.flac", "/lib/b.flac")
    graph.add("/lib/b.flac", "/lib/c.flac")
    monkeypatch.setitem(st.session_state, GRAPH_STATE, graph.to_state())

    at_path = {"/lib/a.flac": 5, "/lib/b.flac": 2, "/lib/c.flac": 9}
    assert sorted(chain_places(at_path)) == [2, 5, 9]


def test_a_track_no_longer_on_the_map_drops_out_by_itself(monkeypatch):
    import streamlit as st

    from core.analysis.graph_playlist import GraphPlaylist
    from streamlit_app.views.graph_board import GRAPH_STATE
    from streamlit_app.views.map_analysis import chain_places

    graph = GraphPlaylist().start("/lib/a.flac")
    graph.add("/lib/a.flac", "/lib/sparito.flac")
    monkeypatch.setitem(st.session_state, GRAPH_STATE, graph.to_state())
    assert chain_places({"/lib/a.flac": 0}) == [0]


def test_with_no_chain_there_are_no_rings(monkeypatch):
    import streamlit as st

    from streamlit_app.views.graph_board import GRAPH_STATE
    from streamlit_app.views.map_analysis import chain_places

    monkeypatch.setitem(st.session_state, GRAPH_STATE, None)
    assert chain_places({"/lib/a.flac": 0}) == []


def _four_tracks() -> "pd.DataFrame":
    import pandas as pd
    return pd.DataFrame([{"path": f"/lib/{i}.flac"} for i in range(4)])


def test_choosing_a_seed_fills_the_field_that_shows_it(monkeypatch):
    """Il campo non e' solo da dove si sceglie: e' anche dove si LEGGE il
    seme, comunque sia arrivato."""
    import streamlit as st

    from streamlit_app.views.map_analysis import SEED, SEED_FIELD, remember_seed

    monkeypatch.setattr(st, "session_state", {})
    remember_seed(_four_tracks(), 2)
    assert st.session_state[SEED] == "/lib/2.flac"
    assert st.session_state[SEED_FIELD] == 2


def test_a_new_pick_replaces_the_one_the_field_was_showing(monkeypatch):
    # Il caso rotto: cliccato un punto DOPO aver scelto per nome, il brano
    # nuovo compariva solo dentro l'elenco a discesa e il campo restava sul
    # vecchio.
    import streamlit as st

    from streamlit_app.views.map_analysis import SEED_FIELD, remember_seed

    monkeypatch.setattr(st, "session_state", {})
    frame = _four_tracks()
    remember_seed(frame, 1)
    remember_seed(frame, 3)
    assert st.session_state[SEED_FIELD] == 3


def test_a_group_from_the_map_leaves_the_field_empty(monkeypatch):
    """Seme e gruppo si escludono: un campo acceso sul brano di prima direbbe
    che c'e' ancora una scelta singola quando non c'e' piu'."""
    import streamlit as st

    from streamlit_app.views.map_analysis import SEED, SEED_FIELD, forget_seed, remember_seed

    monkeypatch.setattr(st, "session_state", {})
    remember_seed(_four_tracks(), 0)
    forget_seed()
    assert SEED not in st.session_state
    assert st.session_state[SEED_FIELD] is None


def _drawn(n: int = 4) -> "pd.DataFrame":
    import numpy as np
    import pandas as pd
    return pd.DataFrame({
        "x": np.arange(n, dtype=float), "y": np.arange(n, dtype=float),
        "index": np.arange(n), "name": [f"{i}.flac" for i in range(n)],
        "bpm": [120] * n, "camelot": ["8A"] * n, "genres": ["House"] * n,
        "top_genre": ["House"] * n, "genre_key": ["House"] * n,
        "_size": [7.0] * n})


def _legend_of(**kwargs) -> list[str]:
    import numpy as np

    from streamlit_app.views.map_analysis import build_figure

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


def test_the_playlist_is_named_once_and_not_twice():
    """Il percorso e' gia' in legenda: due voci per lo stesso insieme di brani
    direbbero che sono due cose."""
    names = _legend_of(playlist=[0, 1], seed=None)
    assert names.count("playlist") == 1
    assert "in the playlist" not in names


def test_the_seed_has_its_own_entry():
    names = _legend_of(playlist=[], seed=3)
    assert "seed" in names


def test_no_size_option_promises_a_measure_it_does_not_show():
    """La voce si chiamava "energy" e mostrava `lufs`: la loudness dice quanto
    ha spinto chi ha masterizzato, non quanto spinge il brano."""
    from streamlit_app.views.map_analysis import SIZE_FIELDS

    assert SIZE_FIELDS.get("loudness") == "lufs"
    # Ora l'energia c'e', ed e' la sua: quattro misure sue, non la loudness.
    assert SIZE_FIELDS.get("energy") == "energy"


def test_the_energy_of_a_library_nobody_has_measured_is_no_size_at_all():
    """Finche' il backfill non e' passato la colonna e' vuota, e i punti
    devono restare tutti uguali invece di ammassarsi al diametro minimo."""
    from streamlit_app.views.map_analysis import FLAT_SIZE, marker_sizes

    frame = pd.DataFrame({"energy": [np.nan, np.nan, np.nan]})
    assert marker_sizes(frame, "energy") == FLAT_SIZE


def test_the_two_suggestion_lists_get_their_own_rings():
    """Le proposte del seme non erano cerchiate: la mappa e' proprio il posto
    in cui si guarda per decidere il prossimo brano."""
    names = _legend_of(playlist=[], seed=0, mixes=[1, 2], alike=[3])
    assert "mixes out of it" in names
    assert "sounds like it" in names


def test_without_a_seed_there_is_nothing_to_propose(monkeypatch):
    import streamlit as st

    from core.analysis.mixing import TransitionCost
    from streamlit_app.views.map_analysis import suggested

    monkeypatch.setattr(st, "session_state", {})
    cost = TransitionCost(_np().zeros((3, 2)), [120, 121, 122], ["8A"] * 3)
    assert suggested(None, cost, _np().arange(3), None, 3) == ([], [])


def _np():
    import numpy as np
    return np


def test_the_weights_come_from_the_session_not_from_the_sliders(monkeypatch):
    """Gli slider dei pesi stanno nel pannello, cioe' PIU' IN BASSO del
    disegno: senza chiave il loro valore non si potrebbe leggere in tempo."""
    import streamlit as st

    from core.analysis.mixing import TransitionCost
    from streamlit_app.views.map_analysis import suggested

    from streamlit_app.views.map_analysis import ASKED_ALIKE, ASKED_MIXES

    monkeypatch.setattr(st, "session_state",
                        {"map::w_sound": 0.0, "map::w_bpm": 2.0,
                         "map::w_key": 0.5, "map_suggestion_count": 2,
                         # Le liste si cerchiano solo se chieste, e questo
                         # test guarda i pesi: le si chiede tutte e due.
                         ASKED_MIXES: "/x/a.mp3", ASKED_ALIKE: "/x/a.mp3"})
    np = _np()

    class Store:
        rows = [{"path": "/x/a.mp3"}, {"path": "/x/b.mp3"}, {"path": "/x/c.mp3"}]

        def similar(self, index, k, limit):
            return [(1, 0.9), (2, 0.8)][:k]

    cost = TransitionCost(np.zeros((3, 2)), [120, 121, 122], ["8A"] * 3)
    mixes, alike = suggested(Store(), cost, np.arange(3), 0, 3)
    assert (cost.w_map, cost.w_bpm, cost.w_key) == (0.0, 2.0, 0.5)
    assert len(mixes) <= 2 and alike == [1, 2]


def test_the_track_playing_gets_a_red_cross():
    """Una X e non un anello: gli altri segni dicono cosa un brano E', questo
    dice cosa sta succedendo adesso."""
    from streamlit_app.views.map_analysis import build_figure

    np = _np()
    figure = build_figure(_drawn(), ["House"],
                          np.column_stack([np.arange(4.0), np.arange(4.0)]),
                          playlist=[], seed=None, playing=2)
    cross = [t for t in figure.data if t.name == "playing"]
    assert len(cross) == 1
    assert cross[0].marker.symbol == "x-thin"
    assert cross[0].showlegend is True


def test_with_nothing_playing_there_is_no_cross():
    names = _legend_of(playlist=[], seed=None, playing=None)
    assert "playing" not in names


def test_no_two_rings_share_a_colour():
    """Il rosa era ambra, e l'ambra accanto al giallo della catena erano due
    gialli: si distinguevano per diametro, cioe' bisognava misurarli."""
    from streamlit_app.views.map_analysis import SKIN

    for theme in ("light", "dark"):
        rings = [SKIN[theme][k] for k in
                 ("chained", "kept", "ink", "mixes", "alike", "playing")]
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
    from streamlit_app.views.map_analysis import build_figure

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
    from streamlit_app.views.map_analysis import build_figure

    frame = _measured()
    places = frame[["valence", "energy"]].to_numpy()
    figure = build_figure(frame, ["House"], places, playlist=[1], seed=3,
                          axes=("valence", "energy"), titles=("v", "e"))
    kept = _ring(figure, "in the playlist")
    assert list(kept.x) == [-0.2] and list(kept.y) == [0.9]
    seed = _ring(figure, "seed")
    assert list(seed.x) == [0.9] and list(seed.y) == [0.6]


def test_the_map_keeps_its_axes_hidden_and_its_square_scale():
    """Le due dimensioni della proiezione non sono misure: un numero su di
    esse non vuol dire niente, e stirarne una falserebbe le distanze."""
    from streamlit_app.views.map_analysis import build_figure

    coords = np.column_stack([np.arange(4.0), np.zeros(4)])
    figure = build_figure(_drawn(), ["House"], coords, playlist=[], seed=None)
    assert figure.layout.xaxis.visible is False
    assert figure.layout.yaxis.scaleanchor == "x"


def test_the_quadrant_axes_are_not_tied_to_each_other():
    """Portano due misure diverse — dei BPM e un rango — e legarle
    schiaccerebbe il disegno in una riga."""
    from streamlit_app.views.map_analysis import build_figure

    frame = _measured()
    figure = build_figure(frame, ["House"],
                          frame[["bpm", "energy"]].to_numpy(),
                          playlist=[], seed=None, axes=("bpm", "energy"),
                          titles=("BPM", "energy"))
    assert figure.layout.yaxis.scaleanchor is None
    assert figure.layout.xaxis.visible is True


def test_the_cross_sits_where_the_measure_has_its_own_middle():
    from streamlit_app.views.map_analysis import axis_guide

    # Una misura sola ce l'ha: l'energia e' un rango sulla libreria, quindi
    # il suo mezzo E' la mediana per costruzione.
    assert axis_guide([0.1, 0.2, 0.3], "energy") == 0.5


def test_the_valence_does_not_get_to_call_its_zero_a_centre():
    """Misurata sulla libreria vera i nove decili erano tutti positivi: con
    la croce sullo zero i due quadranti bui sarebbero rimasti vuoti, e il
    grafico avrebbe detto che la libreria e' tutta allegra — che e' una
    proprieta' di come sono fatte le due liste di parole, non della musica."""
    from streamlit_app.views.map_analysis import AXIS_CENTRES, axis_guide

    assert "valence" not in AXIS_CENTRES
    assert axis_guide([0.3, 0.5, 0.7], "valence") == 0.5


def test_the_cross_falls_back_to_the_median_of_what_is_on_screen():
    from streamlit_app.views.map_analysis import axis_guide

    assert axis_guide([96.0, 120.0, 128.0, 140.0], "bpm") == 124.0
    assert axis_guide([], "bpm") is None


def test_the_caption_says_which_of_the_two_kinds_of_middle_it_is():
    """Una riga tratteggiata a meta' del disegno sembra un centro assoluto,
    e su quasi tutte le misure e' invece la mediana di cio' che i filtri
    lasciano — cioe' si sposta appena si tocca un filtro."""
    from streamlit_app.views.map_analysis import guide_caption

    told = guide_caption((124.0, 0.5), ("bpm", "energy"), ("BPM", "energy"))
    assert "median of what the filters leave" in told
    assert "middle of the measure itself" in told
    assert guide_caption((None, 0.5), ("bpm", "energy"), ("BPM", "e")) == ""


def test_both_charts_feed_the_same_choice():
    """Si sceglie indifferentemente sulla mappa o sui quadranti, e la scelta
    e' una sola. Il doppione si toglie: due volte lo stesso brano vorrebbe
    dire un GRUPPO di due invece di un seme."""
    from streamlit_app.views.map_analysis import MAP_CHART, QUAD_CHART, read_selection

    def state(points):
        return {"selection": {"points": [{"customdata": [i]} for i in points]}}

    st.session_state[MAP_CHART] = state([7])
    st.session_state[QUAD_CHART] = state([7])
    try:
        assert read_selection() == [7]
        st.session_state[QUAD_CHART] = state([3])
        assert read_selection() == [7, 3]
    finally:
        del st.session_state[MAP_CHART], st.session_state[QUAD_CHART]


def test_the_default_axes_are_the_two_that_answer_the_question():
    """Valence e arousal, i due assi di Russell: dove sta questo brano fra il
    buio e il chiaro, fra il calmo e lo spinto."""
    from streamlit_app.views.map_analysis import AXIS_FIELDS, DEFAULT_AXES

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
    from streamlit_app.views.map_analysis import AXIS_CENTRES, AXIS_FIELDS, axis_guide

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
    from streamlit_app.views.map_analysis import AXIS_FIELDS, AXIS_HELP

    assert not [name for name in AXIS_FIELDS if name not in AXIS_HELP]
    # E le due che sono ranghi lo dicono, perche' e' l'equivoco possibile.
    for name in ("energy", "valence (mood)"):
        assert "rank" in AXIS_HELP[name]


# --- le due liste del seme non si aprono da sole --------------------------

def test_the_list_waits_to_be_asked_for_this_very_seed():
    """In sessione si tiene il PERCORSO del seme, non un si'/no: cosi'
    cambiando brano la scheda torna chiusa da se', senza che chi cambia il
    seme debba ricordarsi di spegnerla."""
    from streamlit_app.views.map_analysis import ASKED_MIXES, asked_for

    st.session_state[ASKED_MIXES] = "/DJSet/a.mp3"
    try:
        assert asked_for(ASKED_MIXES, "/DJSet/a.mp3", "x", "y")
        # Un altro seme: la lista di prima non vale per questo.
        assert not asked_for(ASKED_MIXES, "/DJSet/b.mp3", "x", "y")
    finally:
        del st.session_state[ASKED_MIXES]


def test_the_two_tabs_ask_separately():
    """Due chiavi e non una: si puo' volere le proposte di mix senza volere
    anche i simili, che sono due domande diverse sullo stesso brano."""
    from streamlit_app.views.map_analysis import ASKED_ALIKE, ASKED_MIXES, asked_for

    st.session_state[ASKED_MIXES] = "/DJSet/a.mp3"
    try:
        assert ASKED_MIXES != ASKED_ALIKE
        assert asked_for(ASKED_MIXES, "/DJSet/a.mp3", "x", "y")
        assert not asked_for(ASKED_ALIKE, "/DJSet/a.mp3", "x", "y")
    finally:
        del st.session_state[ASKED_MIXES]


def test_the_file_name_comes_right_after_the_column_you_act_on():
    """Nelle due tabelle del Chain Maker il nome era in fondo alla fila di
    misure, e costringeva a scorrere per sapere di che brano si sta leggendo
    il BPM."""
    import inspect

    from streamlit_app.views import graph_board

    source = inspect.getsource(graph_board)
    assert '["#", "file", "BPM"' in source          # la catena
    assert '["Add", "file", "cost"' in source       # la rosa


# --- una libreria sola per tutte le sezioni -------------------------------

def test_the_library_frame_carries_the_measures_that_are_not_on_disk():
    """Energia e valence sono ranghi sulla LIBRERIA, non numeri per brano:
    non stanno sulla riga, si calcolano aprendo la mappa. Chi costruisce il
    frame deve riceverli senza doverseli ricordare."""
    from core.analysis.map_store import MapStore
    from streamlit_app.views.map_analysis import library_frame

    rows = [{"path": f"/x/{i}.mp3", "name": f"{i}.mp3", "bpm": 120.0,
             "camelot": "8A", "moods": "Dark" if i else "Happy",
             "energy_density": float(i), "energy_bass": float(i),
             "energy_bright": float(i), "energy_pulse": float(i)}
            for i in range(4)]
    store = MapStore(directory=Path("/tmp/none"), rows=rows,
                     embeddings=np.zeros((0, 1280), dtype=np.float32))
    frame = library_frame(store, 4)
    assert list(frame["index"]) == [0, 1, 2, 3]
    assert frame["energy"].iloc[0] == 0.0 and frame["energy"].iloc[3] == 1.0
    # E la valence come rango, che e' quella che si legge come posizione.
    assert frame["valence_rank"].iloc[0] == 1.0      # l'unico Happy


def test_no_section_builds_the_library_frame_on_its_own():
    """La causa vera del KeyError: tre sezioni si costruivano il frame per
    conto loro, e le colonne calcolate andavano aggiunte in tre posti. Ne ho
    aggiornati due su tre, e salvare una playlist rompeva la lavagna."""
    import inspect

    from streamlit_app.views import map_analysis

    body = inspect.getsource(map_analysis)
    built = body.count("pd.DataFrame(store.rows[:placed])")
    assert built == 1, "solo `library_frame` costruisce la libreria"


def test_the_rings_show_only_the_lists_that_were_asked_for():
    """Gli anelli attorno a venti punti dicevano che una scelta era stata
    fatta mentre sotto la scheda diceva "premi il bottone". Peggio: erano
    gli anelli di una lista che nessuno aveva visto."""
    from core.analysis.map_store import MapStore
    from core.analysis.mixing import TransitionCost
    from streamlit_app.views.map_analysis import ASKED_ALIKE, ASKED_MIXES, suggested

    rows = [{"path": f"/x/{i}.mp3", "name": f"{i}.mp3", "bpm": 120.0,
             "camelot": "8A"} for i in range(6)]
    coords = np.column_stack([np.arange(6.0), np.zeros(6)])
    store = MapStore(directory=Path("/tmp/none"), rows=rows,
                     embeddings=np.eye(6, 1280, dtype=np.float32),
                     coords=coords)
    cost = TransitionCost(coords, [120.0] * 6, ["8A"] * 6)
    pool = np.arange(6)

    for key in (ASKED_MIXES, ASKED_ALIKE):
        st.session_state.pop(key, None)
    try:
        assert suggested(store, cost, pool, 0, 6) == ([], [])
        # Chiesta una sola: si cerchia una sola.
        st.session_state[ASKED_MIXES] = "/x/0.mp3"
        mixes, alike = suggested(store, cost, pool, 0, 6)
        assert mixes and not alike
        # E per un ALTRO seme la richiesta di prima non vale.
        assert suggested(store, cost, pool, 3, 6) == ([], [])
    finally:
        for key in (ASKED_MIXES, ASKED_ALIKE):
            st.session_state.pop(key, None)
