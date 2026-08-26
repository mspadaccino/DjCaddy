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


def test_ticked_tracks_get_a_yellow_ring():
    coords = np.column_stack([np.arange(4.0), np.zeros(4)])
    figure = build_figure(_drawn(), ["House"], coords, playlist=[],
                          seed=None, ticked=[1, 3])
    ring = _ring(figure, "being picked")
    assert list(ring.x) == [1.0, 3.0]
    assert ring.marker.line.color == SKIN["light"]["ticked"]


def test_playlist_tracks_get_a_green_ring():
    coords = np.column_stack([np.arange(4.0), np.zeros(4)])
    figure = build_figure(_drawn(), ["House"], coords, playlist=[0, 2],
                          seed=None)
    ring = _ring(figure, "in the playlist")
    assert list(ring.x) == [0.0, 2.0]
    assert ring.marker.line.color == SKIN["light"]["kept"]


def test_no_ring_when_nothing_is_ticked():
    coords = np.column_stack([np.arange(4.0), np.zeros(4)])
    figure = build_figure(_drawn(), ["House"], coords, playlist=[], seed=None)
    assert _ring(figure, "being picked") is None
    assert _ring(figure, "in the playlist") is None


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
                          seed=None, selected=[], ticked=[])
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
