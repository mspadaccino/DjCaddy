import numpy as np
import pandas as pd

from views.map_analysis import (SKIN, build_figure, matching_tracks,
                                playlist_positions)


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
