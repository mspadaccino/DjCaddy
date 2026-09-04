"""L'arco di un set, letto dal codice e non dalla lavagna: quote, fasce, e
le misure sulla scala della libreria che il Journey insegue."""

import numpy as np

from core.analysis.arc import (CHAPTERS, MEASURES, arc_costs, chapter_at,
                               chapter_score, chapters_along, measures)


def test_the_quotas_cover_the_set_exactly_once():
    assert abs(sum(ch["quota"] for ch in CHAPTERS) - 1.0) < 1e-9


def test_chapter_at_reads_the_quotas_in_order():
    assert chapter_at(0.0) == 0                     # l'apertura è Intro
    assert chapter_at(0.14) == 0
    assert chapter_at(0.15) == 1                    # il primo 15% è finito
    assert chapter_at(0.5) == 2
    assert chapter_at(0.7) == 3
    assert chapter_at(0.99) == 4
    assert chapter_at(1.0) == 4                     # la chiusura è Release


def test_chapters_along_a_set_start_in_intro_and_end_in_release():
    along = chapters_along(10)
    assert along[0] == 0 and along[-1] == 4
    assert along == sorted(along)                   # l'arco non torna indietro
    assert set(along) == {0, 1, 2, 3, 4}            # dieci bastano per tutti
    assert chapters_along(1) == [0]
    assert chapters_along(0) == []


def test_chapter_score_is_zero_inside_and_grows_outside():
    climax = CHAPTERS[3]
    assert chapter_score(0.9, 0.9, 0.9, 0.5, climax) == 0.0
    # Il tempo dell'Intro dentro al Climax: fuori di 0.7 su una misura sola.
    assert abs(chapter_score(0.0, 0.9, 0.9, 0.5, climax) - 0.7) < 1e-9


def test_measures_rank_tempo_and_groove_and_pass_the_ranks_through():
    got = measures(bpm=[120, 130, 140], energy=[0.2, 0.5, 0.8],
                   valence_rank=[0.9, 0.5, 0.1],
                   danceability=[0.6, 0.5, 0.7])
    assert got.shape == (3, len(MEASURES))
    assert list(got[:, 0]) == [0.0, 0.5, 1.0]       # il tempo, in rango
    assert list(got[:, 1]) == [0.2, 0.5, 0.8]       # l'energia com'è
    assert list(got[:, 2]) == [0.9, 0.5, 0.1]
    assert list(got[:, 3]) == [0.5, 0.0, 1.0]       # il groove, in rango


def test_a_missing_measure_sits_in_the_middle():
    got = measures(bpm=[120, None], energy=[np.nan, 0.5],
                   valence_rank=[0.9, 0.5], danceability=[0.6, 0.5])
    assert got[1, 0] == 0.5 and got[0, 1] == 0.5
    assert measures([], [], [], []).shape == (0, 4)


def test_arc_costs_match_the_chapter_score_scaled_to_one():
    values = measures(bpm=[120, 130, 140], energy=[0.1, 0.5, 0.95],
                      valence_rank=[0.4, 0.5, 0.9],
                      danceability=[0.6, 0.5, 0.7])
    for chapter in range(len(CHAPTERS)):
        got = arc_costs(values, chapter)
        for row in range(3):
            expected = chapter_score(*values[row], CHAPTERS[chapter]) / 4
            assert abs(got[row] - expected) < 1e-6
    # La 0 è un intro: il capitolo che le costa meno è l'Intro, e il Climax
    # la respinge.
    by_chapter = [float(arc_costs(values, ch)[0]) for ch in range(5)]
    assert by_chapter.index(min(by_chapter)) == 0
    assert by_chapter[3] > 0.3
