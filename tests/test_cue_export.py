from core.analysis.cue_export import (
    DJAY_SLOTS,
    build_cue_rows,
    is_vocal_row,
    plan_djay_markers,
)


def test_build_cue_rows_one_row_per_phrase():
    """Una frase è UNA riga: la sua fine è un'informazione a corredo, non un
    secondo marcatore (cadrebbe sull'inizio della frase successiva)."""
    sections = [{"start": 0.0, "end": 32.0, "label": "Intro"},
                {"start": 32.0, "end": 96.0, "label": "Drop/Chorus"}]
    rows = build_cue_rows(sections, [], bpm=120.0)

    assert len(rows) == 2
    intro, drop = rows
    assert intro == {"id": "sec0", "kind": "phrase_start", "label": "Intro",
                     "start": 0.0, "end": 32.0, "beats": 64.0}
    assert drop == {"id": "sec1", "kind": "phrase_start", "label": "Drop/Chorus",
                    "start": 32.0, "end": 96.0, "beats": 128.0}


def test_build_cue_rows_vocal_pairs():
    rows = build_cue_rows([], [(10.0, 15.0)], bpm=None)
    assert len(rows) == 2
    assert rows[0]["kind"] == "vocal_start" and rows[0]["start"] == 10.0
    assert rows[1]["kind"] == "vocal_end" and rows[1]["start"] == 15.0
    assert all(is_vocal_row(r) for r in rows)


def test_build_cue_rows_no_bpm_no_beats():
    rows = build_cue_rows([{"start": 0.0, "end": 10.0, "label": "Intro"}], [], bpm=None)
    assert rows[0]["beats"] is None


def test_is_vocal_row():
    assert is_vocal_row({"kind": "vocal_start"})
    assert not is_vocal_row({"kind": "phrase_start"})


# --- assegnazione degli slot djay Pro ---------------------------------------

def test_plan_phrases_become_hot_cues_in_time_order():
    rows = build_cue_rows([{"start": 30.0, "end": 60.0, "label": "Drop/Chorus"},
                           {"start": 0.0, "end": 30.0, "label": "Intro"}], [])
    plan = plan_djay_markers(rows)

    assert [(pad, start) for _, pad, start in plan.cues] == [(0, 0.0), (1, 30.0)]
    assert plan.loops == []
    assert plan.slot_label["sec1"] == "Cue 1"    # Intro, il più presto
    assert not plan.dropped and not plan.unpaired


def test_plan_ignores_a_phrase_end_row_if_it_gets_one():
    """Le fini di frase non vengono più nemmeno costruite, ma se una arriva
    (righe salvate da una versione precedente) non deve produrre un cue
    sovrapposto - né essere contata fra quelle che "non ci stanno"."""
    rows = build_cue_rows([{"start": 0.0, "end": 30.0, "label": "Intro"}], [])
    rows.append({"id": "sec0e", "kind": "phrase_end", "label": "Intro end",
                 "start": 30.0, "end": None, "beats": None})
    plan = plan_djay_markers(rows)

    assert [start for _, _, start in plan.cues] == [0.0]
    assert plan.slot_label["sec0e"] == ""
    assert plan.dropped == []


def test_plan_keeps_both_ends_of_a_vocal_region():
    """Le vocali restano l'eccezione: un loop ha bisogno di inizio E fine,
    ma li tiene in un solo slot."""
    plan = plan_djay_markers(build_cue_rows([], [(10.0, 15.0)]))
    assert [(s, e) for _, _, s, e in plan.loops] == [(10.0, 15.0)]
    assert plan.slot_label["vs0"] == plan.slot_label["ve0"] == "Loop 1"


def test_plan_each_vocal_region_is_one_loop():
    """Il punto del banco loop: una regione vocale occupa UN solo slot,
    non due pad, e i due banchi non si contendono gli slot."""
    rows = build_cue_rows([{"start": 0.0, "end": 30.0, "label": "Intro"}],
                          [(10.0, 15.0), (40.0, 50.0)])
    plan = plan_djay_markers(rows)

    assert [(slot, s, e) for _, slot, s, e in plan.loops] == [
        (0, 10.0, 15.0), (1, 40.0, 50.0),
    ]
    assert len(plan.cues) == 1                      # solo l'inizio della frase
    assert plan.slot_label["vs0"] == plan.slot_label["ve0"] == "Loop 1"
    assert plan.slot_label["vs1"] == plan.slot_label["ve1"] == "Loop 2"


def test_plan_truncates_each_bank_at_eight_keeping_earliest():
    sections = [{"start": float(i * 10), "end": float(i * 10 + 5), "label": "Intro"}
                for i in range(12)]          # 12 inizi di frase
    vocals = [(float(i * 10 + 200), float(i * 10 + 205)) for i in range(10)]
    plan = plan_djay_markers(build_cue_rows(sections, vocals))

    assert len(plan.cues) == DJAY_SLOTS
    assert len(plan.loops) == DJAY_SLOTS
    assert [start for _, _, start in plan.cues] == sorted(start for _, _, start in plan.cues)
    assert plan.cues[0][2] == 0.0                   # tiene le più presto
    # 4 inizi di frase in eccesso + 2 regioni vocali (2 righe ciascuna)
    assert len(plan.dropped) == (12 - DJAY_SLOTS) + (10 - DJAY_SLOTS) * 2


def test_plan_no_duplicate_slots_within_a_bank():
    """La regressione da cui è nato tutto: prima ogni marcatore vocale
    finiva sullo stesso pad e sopravviveva solo l'ultimo."""
    plan = plan_djay_markers(build_cue_rows(
        [{"start": float(i), "end": float(i) + 0.5, "label": "Intro"} for i in range(4)],
        [(float(i) + 20, float(i) + 21) for i in range(4)],
    ))
    assert len({pad for _, pad, _ in plan.cues}) == len(plan.cues)
    assert len({slot for _, slot, _, _ in plan.loops}) == len(plan.loops)


def test_plan_reports_unpaired_vocal_rows():
    rows = [r for r in build_cue_rows([], [(10.0, 15.0)]) if r["id"] != "ve0"]
    plan = plan_djay_markers(rows)
    assert plan.loops == []
    assert plan.unpaired == ["vs0"]
    assert plan.slot_label["vs0"] == ""


def test_plan_ignores_vocal_region_with_non_positive_length():
    rows = build_cue_rows([], [(10.0, 10.0)])
    plan = plan_djay_markers(rows)
    assert plan.loops == []
    assert plan.unpaired == ["vs0"]


def test_plan_empty():
    plan = plan_djay_markers([])
    assert plan.cues == [] and plan.loops == [] and plan.slot_label == {}
