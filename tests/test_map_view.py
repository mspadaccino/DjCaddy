import numpy as np
import pandas as pd

from views.map_analysis import matching_tracks


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
