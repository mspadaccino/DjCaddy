import pandas as pd

from views.components import HEARING_GLYPH, PLAY_GLYPH, play_marks


def test_the_row_you_are_listening_to_carries_a_speaker():
    paths = ["/lib/a.flac", "/lib/b.flac", "/lib/c.flac"]
    assert play_marks(paths, "/lib/b.flac") == [PLAY_GLYPH, HEARING_GLYPH,
                                                PLAY_GLYPH]


def test_with_nothing_playing_every_row_is_an_arrow():
    paths = ["/lib/a.flac", "/lib/b.flac"]
    assert play_marks(paths, None) == [PLAY_GLYPH, PLAY_GLYPH]


def test_a_track_that_is_not_in_this_table_lights_nothing_up():
    assert play_marks(["/lib/a.flac"], "/lib/altrove.flac") == [PLAY_GLYPH]


def test_the_same_track_lights_up_in_every_table_that_shows_it():
    # Il confronto e' per percorso e non per numero di riga: la rosa e la
    # catena mostrano lo stesso brano in posizioni diverse, e deve accendersi
    # in tutte e due.
    rosa = ["/lib/x.flac", "/lib/y.flac"]
    catena = ["/lib/z.flac", "/lib/y.flac", "/lib/x.flac"]
    assert play_marks(rosa, "/lib/y.flac").count(HEARING_GLYPH) == 1
    assert play_marks(catena, "/lib/y.flac").count(HEARING_GLYPH) == 1


def test_it_reads_a_pandas_column_the_way_the_table_hands_it_over():
    # `play_table` passa table["_path"], non una lista: se l'indice non parte
    # da zero — una tabella filtrata — il segno deve restare sulla riga giusta.
    column = pd.Series(["/lib/a.flac", "/lib/b.flac"], index=[7, 12])
    assert play_marks(column, "/lib/b.flac") == [PLAY_GLYPH, HEARING_GLYPH]
