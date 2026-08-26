import pandas as pd

from views.components import HEARING_GLYPH, PLAY_GLYPH, play_marks


def test_the_row_you_are_listening_to_carries_a_pause():
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


def test_the_finder_panel_can_be_asked_for_on_its_own(monkeypatch):
    """Serve dove il campo dove scrivere il percorso c'e' gia' altrove: il
    seme della mappa ha un menu solo, e accanto vuole un pulsante, non un
    secondo campo."""
    import subprocess as sp
    from pathlib import Path

    from views import components

    class Out:
        stdout = "/Volumes/X/DJSet/a.flac\n"

    monkeypatch.setattr(sp, "run", lambda *a, **k: Out())
    assert components.ask_for_file("scegli") == Path("/Volumes/X/DJSet/a.flac")


def test_a_cancelled_panel_chooses_nothing(monkeypatch):
    # In AppleScript annullare e' un ERRORE, non una risposta vuota.
    import subprocess as sp

    from views import components

    monkeypatch.setattr(sp, "run", lambda *a, **k: (_ for _ in ()).throw(OSError))
    assert components.ask_for_file("scegli") is None


def test_pressing_the_row_that_is_playing_turns_it_off():
    """E' il gesto che il segno ⏸ promette: prometterlo senza farlo era
    peggio che non mostrarlo."""
    from views.components import next_playing

    assert next_playing("/lib/a.flac", "/lib/a.flac") is None


def test_pressing_another_row_moves_the_player_there():
    from views.components import next_playing

    assert next_playing("/lib/b.flac", "/lib/a.flac") == "/lib/b.flac"
    assert next_playing("/lib/b.flac", None) == "/lib/b.flac"


def test_a_click_that_chose_nothing_leaves_things_as_they_are():
    # Puo' succedere se la tabella si e' riordinata sotto le dita: spegnere
    # sarebbe una risposta a un gesto che nessuno ha fatto.
    from views.components import next_playing

    assert next_playing(None, "/lib/a.flac") == "/lib/a.flac"
    assert next_playing(None, None) is None
