import numpy as np
import pandas as pd

from views.map_analysis import (SKIN, build_figure, matching_tracks,
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
    from analysis.mixing import TransitionCost
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

    from analysis.graph_playlist import GraphPlaylist
    from views.graph_board import GRAPH_STATE
    from views.map_analysis import chain_places

    graph = GraphPlaylist().start("/lib/a.flac")
    graph.add("/lib/a.flac", "/lib/b.flac")
    graph.add("/lib/b.flac", "/lib/c.flac")
    monkeypatch.setitem(st.session_state, GRAPH_STATE, graph.to_state())

    at_path = {"/lib/a.flac": 5, "/lib/b.flac": 2, "/lib/c.flac": 9}
    assert sorted(chain_places(at_path)) == [2, 5, 9]


def test_a_track_no_longer_on_the_map_drops_out_by_itself(monkeypatch):
    import streamlit as st

    from analysis.graph_playlist import GraphPlaylist
    from views.graph_board import GRAPH_STATE
    from views.map_analysis import chain_places

    graph = GraphPlaylist().start("/lib/a.flac")
    graph.add("/lib/a.flac", "/lib/sparito.flac")
    monkeypatch.setitem(st.session_state, GRAPH_STATE, graph.to_state())
    assert chain_places({"/lib/a.flac": 0}) == [0]


def test_with_no_chain_there_are_no_rings(monkeypatch):
    import streamlit as st

    from views.graph_board import GRAPH_STATE
    from views.map_analysis import chain_places

    monkeypatch.setitem(st.session_state, GRAPH_STATE, None)
    assert chain_places({"/lib/a.flac": 0}) == []


def _four_tracks() -> "pd.DataFrame":
    import pandas as pd
    return pd.DataFrame([{"path": f"/lib/{i}.flac"} for i in range(4)])


def test_choosing_a_seed_fills_the_field_that_shows_it(monkeypatch):
    """Il campo non e' solo da dove si sceglie: e' anche dove si LEGGE il
    seme, comunque sia arrivato."""
    import streamlit as st

    from views.map_analysis import SEED, SEED_FIELD, remember_seed

    monkeypatch.setattr(st, "session_state", {})
    remember_seed(_four_tracks(), 2)
    assert st.session_state[SEED] == "/lib/2.flac"
    assert st.session_state[SEED_FIELD] == 2


def test_a_new_pick_replaces_the_one_the_field_was_showing(monkeypatch):
    # Il caso rotto: cliccato un punto DOPO aver scelto per nome, il brano
    # nuovo compariva solo dentro l'elenco a discesa e il campo restava sul
    # vecchio.
    import streamlit as st

    from views.map_analysis import SEED_FIELD, remember_seed

    monkeypatch.setattr(st, "session_state", {})
    frame = _four_tracks()
    remember_seed(frame, 1)
    remember_seed(frame, 3)
    assert st.session_state[SEED_FIELD] == 3


def test_a_group_from_the_map_leaves_the_field_empty(monkeypatch):
    """Seme e gruppo si escludono: un campo acceso sul brano di prima direbbe
    che c'e' ancora una scelta singola quando non c'e' piu'."""
    import streamlit as st

    from views.map_analysis import SEED, SEED_FIELD, forget_seed, remember_seed

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

    from views.map_analysis import build_figure

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
    from views.map_analysis import SIZE_FIELDS

    assert SIZE_FIELDS.get("loudness") == "lufs"
    # L'energia vera ha quattro campi suoi, e finche' non sono su tutta la
    # libreria questa voce non deve esistere: mezza libreria senza valore
    # starebbe tutta al diametro minimo, che e' peggio di non offrirla.
    assert "energy" not in SIZE_FIELDS


def test_the_two_suggestion_lists_get_their_own_rings():
    """Le proposte del seme non erano cerchiate: la mappa e' proprio il posto
    in cui si guarda per decidere il prossimo brano."""
    names = _legend_of(playlist=[], seed=0, mixes=[1, 2], alike=[3])
    assert "mixes out of it" in names
    assert "sounds like it" in names


def test_without_a_seed_there_is_nothing_to_propose(monkeypatch):
    import streamlit as st

    from analysis.mixing import TransitionCost
    from views.map_analysis import suggested

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

    from analysis.mixing import TransitionCost
    from views.map_analysis import suggested

    monkeypatch.setattr(st, "session_state",
                        {"map::w_sound": 0.0, "map::w_bpm": 2.0,
                         "map::w_key": 0.5, "map_suggestion_count": 2})
    np = _np()

    class Store:
        def similar(self, index, k, limit):
            return [(1, 0.9), (2, 0.8)][:k]

    cost = TransitionCost(np.zeros((3, 2)), [120, 121, 122], ["8A"] * 3)
    mixes, alike = suggested(Store(), cost, np.arange(3), 0, 3)
    assert (cost.w_map, cost.w_bpm, cost.w_key) == (0.0, 2.0, 0.5)
    assert len(mixes) <= 2 and alike == [1, 2]


def test_the_track_playing_gets_a_red_cross():
    """Una X e non un anello: gli altri segni dicono cosa un brano E', questo
    dice cosa sta succedendo adesso."""
    from views.map_analysis import build_figure

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
    from views.map_analysis import SKIN

    for theme in ("light", "dark"):
        rings = [SKIN[theme][k] for k in
                 ("chained", "kept", "ink", "mixes", "alike", "playing")]
        assert len(set(rings)) == len(rings), theme
