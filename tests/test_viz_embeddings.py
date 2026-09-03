"""L'impronta degli embedding: il quadro dei colori, il PNG, la distanza.

Sono le parti pure della scheda Embeddings — niente Qt, come per la mappa e
i quadranti: qui si prova cosa la figura DICE, non come il widget la mostra.
"""

import base64
import struct

import numpy as np
import pandas as pd

from core.viz.embedding_figure import (CELL_BUDGETS, GROUP, MAX_CELLS,
                                       build_fingerprint_figure, columns_for,
                                       cosine_distances, fingerprint,
                                       fingerprint_source, hover_overlay,
                                       picture_width, png_bytes, rows_budget,
                                       unit_norms)


def _rows(count: int) -> pd.DataFrame:
    return pd.DataFrame({
        "index": np.arange(count),
        "name": [f"track {i}.mp3" for i in range(count)],
        "bpm": np.full(count, 124.0),
        "camelot": ["8A"] * count,
        "genres": ["Electronic - House"] * count,
    })


# --- il quadro dei colori ---

def test_a_dimension_everyone_shares_reads_neutral():
    # Una colonna in cui tutti stanno allo stesso valore non dice niente su
    # nessuno: deve uscire del colore di mezzo, non di un estremo.
    vectors = np.tile(np.arange(GROUP * 3, dtype=np.float32), (12, 1))
    assert np.allclose(fingerprint(vectors), 0.0)


def test_the_track_that_stands_out_goes_to_the_extreme():
    vectors = np.zeros((12, GROUP * 2), dtype=np.float32)
    vectors[4, :GROUP] = 9.0
    vectors[7, :GROUP] = -9.0
    quadro = fingerprint(vectors)
    assert quadro[4, 0] == 1.0 and quadro[7, 0] == -1.0
    # Gli altri restano dov'erano: sotto la mediana, ma non agli estremi.
    assert set(np.unique(quadro[:, 1])) == {0.0}


def test_ten_dimensions_make_one_column_unless_every_is_asked():
    vectors = np.random.default_rng(0).random((6, 1280), dtype=np.float32)
    assert fingerprint(vectors).shape == (6, 128)
    assert fingerprint(vectors, every=True).shape == (6, 1280)


def test_the_leftover_dimensions_stay_out_of_the_picture():
    # 25 dimensioni sono due colonne da dieci e cinque che avanzano: una
    # colonna da cinque non sarebbe confrontabile con le altre.
    vectors = np.random.default_rng(1).random((4, 25), dtype=np.float32)
    assert fingerprint(vectors).shape == (4, 2)


def test_asking_for_every_dimension_costs_rows():
    assert columns_for(1280, every=False) == 128
    assert columns_for(1280, every=True) == 1280
    assert rows_budget(1280) == rows_budget(128) // GROUP
    assert rows_budget(128) * 128 <= MAX_CELLS


def test_take_picks_the_rows_and_keeps_their_order():
    # Prendere le righe dentro `fingerprint` o fuori deve dare lo stesso
    # quadro: dentro si fa a blocchi, ed è solo per non copiare mezzo giga.
    vectors = np.random.default_rng(4).random((8, GROUP * 3), dtype=np.float32)
    wanted = [5, 0, 3]
    assert np.array_equal(fingerprint(vectors, take=wanted),
                          fingerprint(vectors[wanted]))


def test_the_bigger_budget_holds_a_whole_library():
    # 86.921 brani a 128 colonne sono 11,1 milioni di celle: il budget
    # piccolo li campiona, quello grande li tiene tutti.
    library = 86_921
    assert rows_budget(128, CELL_BUDGETS["light · 3M pixels"]) < library
    assert rows_budget(128, CELL_BUDGETS["full · 12M pixels"]) > library
    # A 1280 colonne non ci sta nemmeno il grande, e va detto invece che
    # promesso: dieci volte le colonne, un decimo delle righe.
    assert rows_budget(1280, CELL_BUDGETS["full · 12M pixels"]) < library


# --- il PNG scritto a mano ---

def test_the_png_carries_the_size_of_the_picture():
    quadro = fingerprint(np.random.default_rng(2).random((30, 40),
                                                         dtype=np.float32),
                         every=True)
    source = fingerprint_source(quadro, dark=True)
    assert source.startswith("data:image/png;base64,")
    raw = base64.b64decode(source.split(",", 1)[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    width, height, depth, kind = struct.unpack(">IIBB", raw[16:26])
    assert (width, height) == (40, 30)
    assert (depth, kind) == (8, 3)      # 8 bit a tavolozza


def test_every_row_of_the_png_is_one_track():
    quadro = np.zeros((7, 5), dtype=np.float32)
    raw = png_bytes(np.zeros((7, 5), dtype=np.uint8),
                    np.zeros((256, 3), dtype=np.uint8))
    assert struct.unpack(">I", raw[20:24])[0] == len(quadro)


# --- la distanza dal seme ---

def test_the_seed_sits_at_zero_from_itself():
    vectors = np.random.default_rng(3).random((20, 64), dtype=np.float32)
    away = cosine_distances(vectors, unit_norms(vectors), vectors[5])
    assert away[5] == 0.0
    assert (away >= 0.0).all()


def test_length_does_not_count_only_direction_does():
    # Il coseno guarda la direzione: lo stesso vettore raddoppiato è lo
    # stesso suono, e deve leggere distanza zero.
    vectors = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    away = cosine_distances(vectors, unit_norms(vectors), vectors[0])
    assert away[1] == 0.0
    assert away[2] == 1.0


def test_a_seed_without_a_vector_asks_for_no_distance():
    vectors = np.zeros((4, 8), dtype=np.float32)
    away = cosine_distances(vectors, unit_norms(vectors), np.zeros(8))
    assert (away == 0.0).all()


# --- le figure ---

def test_the_index_travels_in_customdata_as_on_the_map():
    rows = _rows(5)
    hover = hover_overlay(rows, columns=128, dark=True).data[-1]
    assert list(hover.customdata[:, 0]) == list(rows["index"])
    assert len(hover.y) == len(rows)


def test_the_picture_is_a_row_per_track_not_a_square_of_pixels():
    figure = build_fingerprint_figure(_rows(3), "data:image/png;base64,x",
                                      columns=128, dark=False)
    # Trascinare scorre il disegno: il riquadro e il lazo stanno nella barra.
    assert figure.layout.dragmode == "pan"
    # `False` esplicito: un `None` Plotly lo leggerebbe come "non detto" e
    # ci rimetterebbe il lato uguale delle immagini.
    assert figure.layout.yaxis.scaleanchor is False
    # La prima riga in alto, l'ultima in basso, senza margini vuoti.
    assert figure.layout.yaxis.range == (2.5, -0.5)


def test_the_distance_column_stands_beside_the_picture():
    away = np.linspace(0.0, 0.5, 6, dtype=np.float32)
    strip = hover_overlay(_rows(6), columns=128, distances=away,
                          dark=True).data[0]
    assert strip.z.shape == (6, 1)
    assert strip.x0 < 0            # a sinistra dell'impronta, che parte da 0
    assert list(strip.z[:, 0]) == list(away)


def test_no_seed_means_no_column_at_all():
    assert len(hover_overlay(_rows(0), columns=128).data) == 0
    assert not [t for t in hover_overlay(_rows(3), columns=128).data
                if t.type == "heatmap"]


def test_a_click_anywhere_on_the_row_finds_the_same_track():
    # Il ponte JS legge `customdata[0]`: un clic sulla colonna della distanza
    # e uno sull'impronta cadono sullo stesso tracciato, quindi sullo stesso
    # brano — l'indice di libreria sta in testa in tutte e due i casi.
    rows = _rows(3).assign(index=[4, 9, 30])
    away = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    overlay = hover_overlay(rows, columns=128, distances=away)
    assert list(overlay.data[-1].customdata[:, 0]) == [4, 9, 30]


def test_the_row_number_rides_along_with_the_track():
    # In testa resta l'indice di libreria — il ponte JS legge quello — e il
    # numero di riga va in coda: sotto l'ordine per distanza è una classifica.
    hover = hover_overlay(_rows(4), columns=128).data[-1]
    assert list(hover.customdata[:, 5]) == [1, 2, 3, 4]


def test_one_label_says_both_the_track_and_its_distance():
    # Due tracciati che rispondono al mouse si contendevano il disegno, e a
    # sinistra vinceva la distanza: il brano non si vedeva più. Uno solo, e
    # l'etichetta le dice tutte e due.
    away = np.array([0.0, 0.4, 0.5, 0.6], dtype=np.float32)
    overlay = hover_overlay(_rows(4), columns=128, distances=away)
    assert [t.type for t in overlay.data] == ["heatmap", "scattergl"]
    hover = overlay.data[-1]
    assert list(hover.customdata[:, 6]) == list(away)
    assert "from the seed" in hover.hovertemplate
    assert overlay.data[0].hoverinfo == "skip"
    # Senza seme resta il solo tracciato del mouse, e senza la distanza.
    bare = hover_overlay(_rows(4), columns=128)
    assert [t.type for t in bare.data] == ["scattergl"]
    assert "from the seed" not in bare.data[0].hovertemplate


def test_the_picture_asks_for_a_pixel_a_column():
    # Un pixel dell'immagine sopra un pixel dello schermo: a 128 colonne ci
    # sta ovunque, a 1280 no — ed è lì che il disegno va fatto scorrere.
    assert picture_width(128) < 400
    assert picture_width(1280) > 1400


def test_the_wide_picture_declares_its_width_and_the_narrow_one_does_not():
    rows = _rows(4)
    tight = build_fingerprint_figure(rows, "", columns=1280, room=900)
    assert tight.layout.width == picture_width(1280)
    roomy = build_fingerprint_figure(rows, "", columns=128, room=900)
    # Senza larghezza scritta il disegno resta elastico, come gli altri due.
    assert roomy.layout.width is None


def test_the_hovered_row_is_marked_across_the_whole_drawing():
    figure = build_fingerprint_figure(_rows(3), "", columns=128)
    assert figure.layout.yaxis.spikemode == "across"
    assert figure.layout.hoverdistance == -1
