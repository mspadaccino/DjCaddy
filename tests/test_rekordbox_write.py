"""Le parti PURE della scrittura in rekordbox: piano, campi, pad liberi.

Come per `test_djay_write`, qui non si apre nessun database: le funzioni che
decidono qualcosa non ne hanno bisogno, e un `master.db` vero è cifrato,
pesa un giga ed è la libreria di qualcuno. Il giro completo (preview →
write → rilettura) è stato fatto a mano su una COPIA della libreria reale,
e i numeri che ne sono usciti sono quelli fissati qui: `InFrame` a 150 fps
e i pad occupati che spingono i marcatori in memory cue.
"""

import pytest

from core.analysis.cue_export import (RB_HOT_CUES, RekordboxMarker,
                                      build_cue_rows, plan_rekordbox_markers)
from core.analysis.rekordbox_write import (MEMORY_CUE, RekordboxWriteError,
                                           check_markers, cue_row_values,
                                           fit_to_free_pads)


def plan_of(n_phrases: int, regions=(), hot=None):
    """Il piano per n frasi e le regioni date. `hot` sono le scelte
    ESPLICITE per riga, come le tiene la pagina: {row_id: True/False}."""
    sections = [{"label": f"Phrase {i}", "start": i * 30.0,
                 "end": (i + 1) * 30.0} for i in range(n_phrases)]
    rows = build_cue_rows(sections, regions, bpm=120.0)
    return plan_rekordbox_markers([
        {"id": r["id"], "kind": r["kind"], "start": r["start"],
         "label": r["label"], "hot": (hot or {}).get(r["id"])}
        for r in rows])


# --- il piano ---------------------------------------------------------------

def test_first_eight_phrases_take_the_pads():
    plan = plan_of(8)
    assert [m.pad for m in plan.markers] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert plan.slot_label["sec0"] == "Hot cue A"
    assert plan.slot_label["sec7"] == "Hot cue H"


def test_beyond_the_pads_nothing_is_dropped():
    """È il guadagno vero rispetto a djay: i memory cue non hanno un numero
    da esaurire, quindi la nona frase non si perde."""
    plan = plan_of(11)
    pads = [m.pad for m in plan.markers]
    assert pads == [1, 2, 3, 4, 5, 6, 7, 8, None, None, None]
    assert plan.slot_label["sec8"] == "Memory cue"
    assert all(label for label in plan.slot_label.values())


def test_a_vocal_region_becomes_one_memory_loop():
    plan = plan_of(2, [(45.0, 61.5)])
    loops = [m for m in plan.markers if m.end is not None]
    assert len(loops) == 1
    assert (loops[0].start, loops[0].end) == (45.0, 61.5)
    assert loops[0].pad is None       # i loop non rubano un pad alle frasi
    assert plan.slot_label["vs0"] == plan.slot_label["ve0"] == "Memory loop"


def test_a_vocal_marker_without_its_twin_is_unpaired():
    plan = plan_rekordbox_markers([
        {"id": "vs0", "kind": "vocal_start", "start": 10.0, "label": "V"}])
    assert plan.unpaired == ["vs0"]
    assert plan.markers == []


def test_markers_come_out_in_time_order():
    plan = plan_of(3, [(45.0, 50.0)])
    assert [m.start for m in plan.markers] == sorted(
        m.start for m in plan.markers)


# --- pad o memory, riga per riga --------------------------------------------

def test_a_phrase_can_be_pushed_down_to_memory():
    """La colonna Hot spenta: la frase non vuole un pad, e quel pad resta
    per una riga più avanti nel brano."""
    plan = plan_of(3, hot={"sec0": False})
    by_id = {m.row_id: m for m in plan.markers}
    assert by_id["sec0"].pad is None
    assert plan.slot_label["sec0"] == "Memory cue"
    assert [by_id["sec1"].pad, by_id["sec2"].pad] == [1, 2]


def test_a_vocal_region_can_be_lifted_onto_a_pad():
    """Un loop salvato su pad: rekordbox lo sa fare, e la spunta lo chiede."""
    plan = plan_of(1, [(45.0, 61.5)], hot={"vs0": True})
    loop = [m for m in plan.markers if m.end is not None][0]
    assert loop.pad == 2                      # dopo sec0, che è a 0 s
    assert plan.slot_label["vs0"] == "Hot loop B"
    assert plan.slot_label["ve0"] == "Hot loop B"


def test_the_ninth_asked_pad_falls_back_to_memory():
    """La spunta è una richiesta, non una promessa: i pad sono otto."""
    plan = plan_of(9)
    by_id = {m.row_id: m for m in plan.markers}
    assert by_id["sec7"].pad == RB_HOT_CUES
    assert by_id["sec8"].pad is None
    assert plan.slot_label["sec8"] == "Memory cue"


def test_the_pads_go_in_time_order_not_in_row_order():
    plan = plan_of(3, hot={"sec1": False})
    by_id = {m.row_id: m for m in plan.markers}
    assert [by_id["sec0"].pad, by_id["sec1"].pad, by_id["sec2"].pad] == \
        [1, None, 2]


# --- i campi di djmdCue -----------------------------------------------------

def test_frames_are_a_hundred_and_fifty_per_second():
    """Misurato sui cue scritti da rekordbox stesso, ed è un TRONCAMENTO,
    non un arrotondamento: 1051 ms → 157 e non 158, 147491 → 22123 e non
    22124. Il test è nato sbagliato con round() e i numeri veri lo hanno
    smentito."""
    for ms, frame in ((34, 5), (1051, 157), (17322, 2598), (147491, 22123)):
        values = cue_row_values(
            RekordboxMarker("x", ms / 1000, None, "", None))
        assert values["InMsec"] == ms
        assert values["InFrame"] == frame


def test_a_cue_declares_no_out_point():
    values = cue_row_values(RekordboxMarker("x", 12.0, None, "Drop", 3))
    assert values["OutMsec"] == -1
    assert values["Kind"] == 3               # il pad C
    assert values["Comment"] == "Drop"


def test_a_loop_carries_its_out_point():
    values = cue_row_values(RekordboxMarker("x", 12.0, 20.0, "Vocal", None))
    assert values["Kind"] == MEMORY_CUE
    assert (values["OutMsec"], values["OutFrame"]) == (20000, 3000)


def test_a_loop_out_frame_is_truncated_too():
    """Da un loop vero di rekordbox: 19384 ms → 2907 frame, non 2908."""
    values = cue_row_values(RekordboxMarker("x", 1.0, 19.384, "", None))
    assert values["OutFrame"] == 2907


def test_a_label_longer_than_the_column_is_cut():
    values = cue_row_values(RekordboxMarker("x", 0.0, None, "z" * 400, None))
    assert len(values["Comment"]) == 255


# --- i pad già occupati -----------------------------------------------------

def test_only_the_free_pads_are_used():
    """Due cue sullo stesso Kind sono un pad solo sul controller: chi non
    trova posto scende a memory cue invece di sovrapporsi."""
    markers = plan_of(4).markers
    fitted = fit_to_free_pads(markers, taken={1, 2, 3, 5, 6})
    assert [m.pad for m in fitted] == [4, 7, 8, None]
    assert [m.row_id for m in fitted] == ["sec0", "sec1", "sec2", "sec3"]


def test_with_every_pad_taken_everything_becomes_memory():
    fitted = fit_to_free_pads(plan_of(3).markers,
                              taken=set(range(1, RB_HOT_CUES + 1)))
    assert [m.pad for m in fitted] == [None, None, None]


def test_loops_are_left_where_they_are():
    plan = plan_of(1, [(45.0, 50.0)])
    fitted = fit_to_free_pads(plan.markers, taken=set())
    loop = [m for m in fitted if m.end is not None][0]
    assert loop.pad is None and loop.start == 45.0


# --- i rifiuti --------------------------------------------------------------

def test_refuses_an_empty_plan():
    with pytest.raises(RekordboxWriteError):
        check_markers([])


def test_refuses_two_markers_on_the_same_pad():
    with pytest.raises(RekordboxWriteError):
        check_markers([RekordboxMarker("a", 1.0, None, "", 2),
                       RekordboxMarker("b", 2.0, None, "", 2)])


def test_refuses_a_pad_out_of_range():
    with pytest.raises(RekordboxWriteError):
        check_markers([RekordboxMarker("a", 1.0, None, "", RB_HOT_CUES + 1)])


def test_refuses_a_loop_that_ends_before_it_starts():
    with pytest.raises(RekordboxWriteError):
        check_markers([RekordboxMarker("a", 10.0, 4.0, "", None)])


def test_accepts_the_plan_the_page_builds():
    check_markers(plan_of(11, [(45.0, 61.5), (120.0, 138.0)]).markers)
