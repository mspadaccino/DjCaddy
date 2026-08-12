import struct

import pytest

from analysis.djay_write import (
    CuePoint,
    DjayWriteError,
    PAD_COLORS,
    _encode,
    _Parser,
    _parse,
    _read_all_slots,
    read_cue_points,
    verify_write,
    LoopRegion,
    read_loop_regions,
    write_markers,
)

# Blob reale catturato da djay Pro (zztop_1cue_orange.bin), usato come
# verita' di riferimento sul formato.
REAL_1CUE_ORANGE = bytes.fromhex(
    "54534146030003000900000000000000140000002b084144434d656469614974656d55736572"
    "4461746100083435313462383338623861343031306265656664636231666336326561366132"
    "000875756964000b0400000008636f6c6f72496e6465780008637565506f696e747300087469"
    "746c654944730008706c6179436f756e740008757365724368616e676564436c6f75644b6579"
    "73000b00010000002b084144434d656469614974656d5469746c654944000501050208536861"
    "72702044726573736564204d616e202831322727204d6178692900087469746c6500085a5a20"
    "546f70000861727469737400130000002a86ae43086475726174696f6e000005050a00000400"
    "00002b08414443437565506f696e74001300000080bf0874696d65001300000080bf08656e64"
    "54696d65002d086e756d62657200002b050e13003961eb42050f1300000080bf051008000863"
    "6f6d6d656e74000f020511002b050e130000000080bf050f1300000080bf05100f030511002b"
    "050e1300000080bf050f1300000080bf05100f0405110005042d050600"
)
# Blob reale catturato da djay Pro (zztop_8cues.bin), usato come
# verita' di riferimento sul formato.
REAL_8CUES = bytes.fromhex(
    "54534146030003000d00000000000000140000002b084144434d656469614974656d55736572"
    "4461746100083435313462383338623861343031306265656664636231666336326561366132"
    "000875756964000b0400000008636f6c6f72496e6465780008637565506f696e747300087469"
    "746c654944730008706c6179436f756e740008757365724368616e676564436c6f75644b6579"
    "73000b00010000002b084144434d656469614974656d5469746c654944000501050208536861"
    "72702044726573736564204d616e202831322727204d6178692900087469746c6500085a5a20"
    "546f70000861727469737400130000002a86ae43086475726174696f6e000005050a00000800"
    "00002b08414443437565506f696e74001300dc9a41430874696d65001300000080bf08656e64"
    "54696d6500080008636f6d6d656e74002d086e756d62657200002b050e13000089a94743050f"
    "1300000080bf0510051105120f020513002b050e130074f53d43050f1300000080bf05100511"
    "05120f030513002b050e130077774d43050f1300000080bf0510051105120f040513002b050e"
    "13004d115143050f1300000080bf05100f050513002b050e1300e32b5643050f1300000080bf"
    "05100f060513002b050e1300b0f85a43050f1300000080bf05100f070513002b050e1300e297"
    "5d43050f1300000080bf05100f0805130005040f02050600"
)


REAL_BLOBS = [REAL_1CUE_ORANGE, REAL_8CUES]


def _root_fields(blob: bytes) -> dict:
    return dict(_parse(blob)[2])


def _cue_numbers(blob: bytes) -> list[int]:
    """Il campo `number` di ogni slot, nell'ordine dell'array."""
    out = []
    for entry in _root_fields(blob)[b"cuePoints"][2]:
        node = dict(entry[2])[b"number"]
        out.append(1 if node[1] == 0x2D else node[2])
    return out


def _blob_no_cues() -> bytes:
    """Lo stesso blob reale, ma con il campo cuePoints (e la sua
    dichiarazione in userChangedCloudKeys) rimossi: è la forma che ha una
    traccia mai avuta cue point in djay Pro."""
    root = _parse(REAL_1CUE_ORANGE)
    root[2][:] = [(n, v) for n, v in root[2] if n != b"cuePoints"]
    keys = dict(root[2])[b"userChangedCloudKeys"]
    keys[2][:] = [k for k in keys[2] if k != ("str", b"cuePoints")]
    return _encode(REAL_1CUE_ORANGE, root)


def _blob_with_slots(slots: list[tuple[float, float]]) -> bytes:
    """Blob reale con l'array cuePoints sostituito dagli slot richiesti."""
    base = _blob_no_cues()
    cues = [CuePoint(time=t, pad=i, end_time=et) for i, (t, et) in enumerate(slots)
            if t != -1.0]
    blob = write_markers(base, cues or [CuePoint(time=0.0, pad=len(slots) - 1)])
    assert len(_read_all_slots(blob)) == len(slots)
    return blob


# --- il formato: round-trip contro i dati reali -----------------------------

@pytest.mark.parametrize("blob", REAL_BLOBS)
def test_reencode_of_real_blob_is_byte_identical(blob):
    """La riserializzazione rigenera da zero tabella delle stringhe,
    allineamenti a 4 byte e contatori di testa: se il modello del formato è
    giusto, deve restituire esattamente i byte scritti da djay Pro."""
    assert _encode(blob, _parse(blob)) == blob


@pytest.mark.parametrize("blob", REAL_BLOBS)
def test_real_blob_header_counts_match_content(blob):
    root = _parse(blob)
    rebuilt = _encode(blob, root)
    assert struct.unpack("<Q", rebuilt[8:16])[0] == struct.unpack("<Q", blob[8:16])[0]
    assert struct.unpack("<I", rebuilt[16:20])[0] == struct.unpack("<I", blob[16:20])[0]


def test_parse_refuses_broken_4_byte_alignment():
    """Il bug che faceva scartare i cue a djay Pro: un byte in più prima
    dell'array manda fuori posto tutti gli allineamenti successivi."""
    idx = REAL_1CUE_ORANGE.find(b"\x08cuePoints\x00")
    corrupted = REAL_1CUE_ORANGE[:idx] + b"\x08x\x00" + REAL_1CUE_ORANGE[idx:]
    with pytest.raises(DjayWriteError):
        _parse(corrupted)


def test_parse_refuses_trailing_garbage():
    with pytest.raises(DjayWriteError):
        _parse(REAL_1CUE_ORANGE + b"\x00\x00\x00\x00")


def test_parse_refuses_wrong_magic():
    with pytest.raises(DjayWriteError):
        _parse(b"NOTA MAGIC HERE")


# --- lettura ---------------------------------------------------------------

def test_read_all_slots_of_real_blob():
    slots = _read_all_slots(REAL_1CUE_ORANGE)
    assert len(slots) == 4
    assert slots[0] == (-1.0, -1.0)
    assert round(slots[1][0], 2) == 117.69
    assert slots[2] == (-1.0, -1.0) and slots[3] == (-1.0, -1.0)


def test_read_cue_points_filters_placeholders():
    """Nel blob reale c'è un solo cue vero, sul pad 2 (posizione 1, arancio)."""
    cues = read_cue_points(REAL_1CUE_ORANGE)
    assert len(cues) == 1
    assert cues[0].pad == 1
    assert PAD_COLORS[cues[0].pad] == "orange"


def test_read_cue_points_all_eight_pads():
    cues = read_cue_points(REAL_8CUES)
    assert [c.pad for c in cues] == list(range(8))


def test_read_all_slots_no_array():
    assert _read_all_slots(_blob_no_cues()) == []


def test_read_cue_points_refuses_wrong_magic():
    with pytest.raises(DjayWriteError):
        read_cue_points(b"NOTA MAGIC HERE")


def test_pad_colors_has_eight_entries():
    assert len(PAD_COLORS) == 8
    assert PAD_COLORS[0] == "red"


# --- scrittura -------------------------------------------------------------

def test_write_markers_output_is_always_parsable():
    """Ogni blob prodotto deve essere riletto per intero: è la proprietà
    che il vecchio writer violava e che faceva rigenerare il record a
    djay Pro."""
    for blob in (_blob_no_cues(), REAL_1CUE_ORANGE, REAL_8CUES):
        updated = write_markers(blob, [CuePoint(time=42.0, pad=5)])
        assert _encode(updated, _parse(updated)) == updated


def test_write_markers_create_from_scratch_pads_earlier_slots():
    # traccia senza array cuePoints: scrivere direttamente al pad 4 (indice)
    # deve creare placeholder per i pad 0-3, come fa djay Pro stesso.
    updated = write_markers(_blob_no_cues(), [CuePoint(time=99.0, pad=4)])
    assert _read_all_slots(updated) == [(-1.0, -1.0)] * 4 + [(99.0, -1.0)]
    assert read_cue_points(updated) == [CuePoint(time=99.0, pad=4, end_time=-1.0)]


def test_write_markers_create_declares_cloud_key():
    """djay Pro aggiunge `cuePoints` a userChangedCloudKeys quando nasce il
    primo cue di una traccia."""
    base = _blob_no_cues()
    keys = _root_fields(base)[b"userChangedCloudKeys"][2]
    assert ("str", b"cuePoints") not in keys

    updated = write_markers(base, [CuePoint(time=5.0, pad=0)])
    keys = _root_fields(updated)[b"userChangedCloudKeys"][2]
    assert ("str", b"cuePoints") in keys


def test_write_markers_create_places_field_after_title_ids():
    """Posizione del campo verificata sul passaggio reale
    userdata_before.bin -> userdata_after.bin fatto da djay Pro."""
    updated = write_markers(_blob_no_cues(), [CuePoint(time=5.0, pad=0)])
    names = [n for n, _ in _parse(updated)[2]]
    assert names.index(b"cuePoints") == names.index(b"titleIDs") + 1


def test_write_markers_numbers_are_one_based_positions():
    """djay Pro scrive `number` = posizione+1 e usa il tag breve 0x2d per 1."""
    updated = write_markers(_blob_no_cues(), [CuePoint(time=1.0, pad=3)])
    assert _cue_numbers(updated) == [1, 2, 3, 4]


def test_write_markers_append_preserves_other_pads():
    blob = _blob_with_slots([(1.0, -1.0), (2.0, -1.0), (3.0, -1.0)])
    updated = write_markers(blob, [CuePoint(time=42.0, pad=1)], overwrite=False)
    assert _read_all_slots(updated) == [(1.0, -1.0), (42.0, -1.0), (3.0, -1.0)]


def test_write_markers_append_keeps_untouched_entries_untouched():
    """Gli slot non toccati vanno riusati tali e quali, `comment` incluso:
    nel blob reale a 8 cue i primi quattro ce l'hanno e gli altri no."""
    before = _root_fields(REAL_8CUES)[b"cuePoints"][2]
    after = _root_fields(write_markers(REAL_8CUES, [CuePoint(time=7.0, pad=0)]))[b"cuePoints"][2]
    assert after[1:] == before[1:]


def test_write_markers_append_extends_array_with_placeholders():
    blob = _blob_with_slots([(1.0, -1.0), (2.0, -1.0)])
    updated = write_markers(blob, [CuePoint(time=42.0, pad=4)], overwrite=False)
    assert _read_all_slots(updated) == [
        (1.0, -1.0), (2.0, -1.0), (-1.0, -1.0), (-1.0, -1.0), (42.0, -1.0),
    ]


def test_write_markers_overwrite_ignores_existing():
    blob = _blob_with_slots([(1.0, -1.0), (2.0, -1.0), (3.0, -1.0)])
    updated = write_markers(blob, [CuePoint(time=42.0, pad=1)], overwrite=True)
    # overwrite=True: tutto il resto diventa placeholder, non viene preservato
    assert _read_all_slots(updated) == [(-1.0, -1.0), (42.0, -1.0)]


def test_write_markers_leaves_everything_else_alone():
    """Nessun campo diverso dai cue point deve cambiare: impronta audio,
    titleIDs, playCount, timestampIdentifier."""
    before = _root_fields(REAL_8CUES)
    after = _root_fields(write_markers(REAL_8CUES, [CuePoint(time=7.0, pad=0)]))
    for name in (b"uuid", b"titleIDs", b"playCount"):
        assert before[name] == after[name]


def test_write_markers_refuses_pad_out_of_range():
    with pytest.raises(DjayWriteError):
        write_markers(REAL_1CUE_ORANGE, [CuePoint(time=1.0, pad=8)])
    with pytest.raises(DjayWriteError):
        write_markers(REAL_1CUE_ORANGE, [CuePoint(time=1.0, pad=-1)])


def test_write_markers_refuses_empty_list():
    with pytest.raises(DjayWriteError):
        write_markers(REAL_1CUE_ORANGE, [])


def test_write_markers_refuses_malformed_blob():
    with pytest.raises(DjayWriteError):
        write_markers(b"TSAF" + b"\x00" * 50, [CuePoint(time=1.0, pad=0)])


# --- verifica --------------------------------------------------------------

def test_verify_write_catches_tampered_slot():
    updated = write_markers(REAL_1CUE_ORANGE, [CuePoint(time=9.0, pad=0)])
    expected = _read_all_slots(updated)
    with pytest.raises(DjayWriteError):
        verify_write(REAL_1CUE_ORANGE, updated, [(999.0, -1.0)] + expected[1:])


def test_verify_write_catches_slot_count_mismatch():
    updated = write_markers(REAL_1CUE_ORANGE, [CuePoint(time=9.0, pad=0)])
    with pytest.raises(DjayWriteError):
        verify_write(REAL_1CUE_ORANGE, updated, [(9.0, -1.0)])


def test_verify_write_catches_other_fields_changed():
    """Se la scrittura tocca qualcosa che non sia i cue point, deve fermarsi."""
    root = _parse(REAL_1CUE_ORANGE)
    fields = dict(root[2])
    fields[b"titleIDs"][2][0][2][1] = (b"title", ("str", b"altro titolo"))
    tampered = _encode(REAL_1CUE_ORANGE, root)
    with pytest.raises(DjayWriteError):
        verify_write(REAL_1CUE_ORANGE, tampered, _read_all_slots(tampered))


def test_read_tolerates_cue_without_end_time():
    """I cue arrivati da un import (es. XML rekordbox) non hanno il campo
    `endTime` e ne hanno altri in più (`comment` col nome, `colorIndex`):
    vanno letti lo stesso, e una scrittura su un altro pad li deve lasciare
    intatti. Forma osservata su una traccia reale della libreria."""
    root = _parse(REAL_1CUE_ORANGE)
    imported = (
        "obj", ("str", b"ADCCuePoint"),
        [(b"time", ("f32", 12.5)),
         (b"comment", ("str", b"Intro")),
         (b"number", ("int", 0x2D, 0)),
         (b"colorIndex", ("int", 0x0F, 0))],
    )
    dict(root[2])[b"cuePoints"][2][:] = [imported]
    blob = _encode(REAL_1CUE_ORANGE, root)

    assert _read_all_slots(blob) == [(12.5, -1.0)]
    assert read_cue_points(blob) == [CuePoint(time=12.5, pad=0, end_time=-1.0)]

    updated = write_markers(blob, [CuePoint(time=99.0, pad=1)])
    assert _root_fields(updated)[b"cuePoints"][2][0] == imported
    assert _encode(updated, _parse(updated)) == updated


def test_roundtrip_of_anonymous_struct_field():
    """Il tag 0x18 (struttura anonima a campi contati, usata dentro
    `beatGridEdits`) non ha byte di chiusura: va riscritto identico."""
    root = _parse(REAL_1CUE_ORANGE)
    root[2].append((b"beatGridEdits", ("coll", 0x0A, [
        ("struct", [(b"position", ("f32", 136.75)),
                    (b"numberOfBars", ("int", 0x0F, 255))]),
    ])))
    blob = _encode(REAL_1CUE_ORANGE, root)
    assert _encode(blob, _parse(blob)) == blob
    assert _parse(blob)[2][-1] == root[2][-1]


# --- loop salvati (banco separato dagli hot cue) ---------------------------

def _loops(blob):
    return [(lr.slot, round(lr.start, 2), round(lr.end, 2))
            for lr in read_loop_regions(blob)]


def test_read_loop_regions_empty_when_absent():
    assert read_loop_regions(REAL_1CUE_ORANGE) == []


def test_write_loops_creates_dense_array_keyed_by_number():
    """A differenza di cuePoints l'array loopRegions è DENSO: saltare gli
    slot 1-3 non deve creare placeholder, lo slot sta nel campo `number`."""
    updated = write_markers(REAL_1CUE_ORANGE, loops=[LoopRegion(start=10.0, end=20.0, slot=3)])
    assert _loops(updated) == [(3, 10.0, 20.0)]

    entries = _root_fields(updated)[b"loopRegions"][2]
    assert len(entries) == 1
    assert entries[0][1][1] == b"ADCLoopRegion"
    assert [n for n, _ in entries[0][2]] == [b"startTime", b"endTime", b"number"]
    assert _cue_numbers(updated) == [1, 2, 3, 4]  # i cue point restano intatti


def test_write_loops_output_is_parsable_and_declares_cloud_key():
    updated = write_markers(REAL_8CUES, loops=[LoopRegion(start=1.0, end=2.0, slot=0)])
    assert _encode(updated, _parse(updated)) == updated
    keys = _root_fields(updated)[b"userChangedCloudKeys"][2]
    assert ("str", b"loopRegions") in keys


def test_write_loops_field_goes_after_cue_points():
    """Ordine dei campi osservato su tutte le tracce reali con loop."""
    updated = write_markers(REAL_8CUES, loops=[LoopRegion(start=1.0, end=2.0, slot=0)])
    names = [n for n, _ in _parse(updated)[2]]
    assert names.index(b"loopRegions") == names.index(b"cuePoints") + 1


def test_write_loops_sorted_by_slot_and_preserves_existing():
    blob = write_markers(REAL_1CUE_ORANGE, loops=[LoopRegion(start=5.0, end=6.0, slot=5)])
    updated = write_markers(blob, loops=[LoopRegion(start=1.0, end=2.0, slot=1)])
    assert _loops(updated) == [(1, 1.0, 2.0), (5, 5.0, 6.0)]


def test_write_loops_overwrite_drops_existing():
    blob = write_markers(REAL_1CUE_ORANGE, loops=[LoopRegion(start=5.0, end=6.0, slot=5)])
    updated = write_markers(blob, loops=[LoopRegion(start=1.0, end=2.0, slot=1)], overwrite=True)
    assert _loops(updated) == [(1, 1.0, 2.0)]


def test_write_cues_and_loops_together():
    updated = write_markers(
        REAL_1CUE_ORANGE,
        cues=[CuePoint(time=30.0, pad=0)],
        loops=[LoopRegion(start=10.0, end=20.0, slot=0)],
    )
    assert _encode(updated, _parse(updated)) == updated
    assert _loops(updated) == [(0, 10.0, 20.0)]
    assert read_cue_points(updated)[0] == CuePoint(time=30.0, pad=0, end_time=-1.0)


def test_write_markers_refuses_duplicate_slots():
    """Il bug che faceva sparire tutti i marcatori tranne l'ultimo: due
    righe sullo stesso pad devono essere un errore, non un sorpasso muto."""
    with pytest.raises(DjayWriteError, match="due volte"):
        write_markers(REAL_1CUE_ORANGE,
                      [CuePoint(time=1.0, pad=2), CuePoint(time=2.0, pad=2)])
    with pytest.raises(DjayWriteError, match="due volte"):
        write_markers(REAL_1CUE_ORANGE, loops=[LoopRegion(start=1.0, end=2.0, slot=0),
                                               LoopRegion(start=3.0, end=4.0, slot=0)])


def test_write_markers_refuses_bad_loop_range():
    with pytest.raises(DjayWriteError):
        write_markers(REAL_1CUE_ORANGE, loops=[LoopRegion(start=9.0, end=9.0, slot=0)])
    with pytest.raises(DjayWriteError):
        write_markers(REAL_1CUE_ORANGE, loops=[LoopRegion(start=1.0, end=2.0, slot=8)])


def test_write_markers_refuses_nothing_to_write():
    with pytest.raises(DjayWriteError):
        write_markers(REAL_1CUE_ORANGE)


def test_write_loops_leaves_everything_else_alone():
    before = _root_fields(REAL_8CUES)
    after = _root_fields(write_markers(REAL_8CUES, loops=[LoopRegion(start=1.0, end=2.0, slot=0)]))
    for name in (b"uuid", b"titleIDs", b"playCount", b"cuePoints"):
        assert before[name] == after[name]


def test_plan_djay_markers_output_is_writable():
    """Integrazione: quello che il planner produce dalla tabella deve essere
    accettato dal writer senza collisioni di slot."""
    from analysis.cue_export import build_cue_rows, plan_djay_markers

    rows = build_cue_rows(
        [{"start": float(i * 20), "end": float(i * 20 + 10), "label": "Intro"}
         for i in range(12)],
        [(float(i * 15 + 500), float(i * 15 + 507)) for i in range(10)],
        bpm=120.0,
    )
    plan = plan_djay_markers(rows)
    updated = write_markers(
        REAL_1CUE_ORANGE,
        cues=[CuePoint(time=start, pad=pad) for _, pad, start in plan.cues],
        loops=[LoopRegion(start=s, end=e, slot=slot) for _, slot, s, e in plan.loops],
    )
    assert _encode(updated, _parse(updated)) == updated
    assert len(read_cue_points(updated)) == 8
    assert [lr.slot for lr in read_loop_regions(updated)] == list(range(8))


# --- riferimenti fra oggetti (\x02) ----------------------------------------

def _blob_with_fingerprint_ref():
    """Blob con un oggetto impronta audio e un campo che lo REFERENZIA, come
    su 910 delle 927 tracce reali della libreria."""
    root = _parse(REAL_1CUE_ORANGE)
    fingerprint = ("obj", ("str", b"ADCAudioAlignmentFingerprint"),
                   [(b"timestampIdentifierData", ("data", b"x" * 12)),
                    (b"version", ("int", 0x2E, 0))])
    root[2].append((b"audioAlignmentFingerprint", fingerprint))
    root[2].append((b"timestampIdentifier", ("objref", fingerprint)))
    return _encode(REAL_1CUE_ORANGE, root)


def _ref_target(blob):
    """(classe, indice nella tabella degli oggetti) del bersaglio del
    riferimento, riletti dal blob."""
    parser = _Parser(blob)
    root = parser.value()
    ref = dict(root[2])[b"timestampIdentifier"]
    assert ref[0] == "objref"
    idx = next(i for i, n in enumerate(parser.objects) if n is ref[1])
    return ref[1][1][1], idx


def test_object_reference_round_trips():
    blob = _blob_with_fingerprint_ref()
    assert _encode(blob, _parse(blob)) == blob
    assert _ref_target(blob)[0] == b"ADCAudioAlignmentFingerprint"


def test_object_reference_is_renumbered_when_markers_are_added():
    """La regressione che impediva a djay Pro di caricare il brano:
    `-[ADCCuePoint fingerprintFloatsArray]: unrecognized selector`. Il campo
    timestampIdentifier è un PUNTATORE all'impronta audio; aggiungere cue e
    loop sposta l'impronta più avanti nella tabella degli oggetti, e un
    puntatore non aggiornato finisce su un ADCCuePoint."""
    blob = _blob_with_fingerprint_ref()
    _, before = _ref_target(blob)

    updated = write_markers(
        blob,
        cues=[CuePoint(time=float(i), pad=i) for i in range(8)],
        loops=[LoopRegion(start=float(i) + 100, end=float(i) + 105, slot=i)
               for i in range(8)],
    )
    cls, after = _ref_target(updated)

    assert after > before, "l'indice non è stato ricalcolato"
    assert cls == b"ADCAudioAlignmentFingerprint"
    assert _encode(updated, _parse(updated)) == updated


def test_parse_refuses_forward_object_reference():
    """Un riferimento a un oggetto non ancora serializzato non si saprebbe
    riscrivere: meglio fermarsi che tirare a indovinare."""
    blob = _blob_with_fingerprint_ref()
    tag = blob.rindex(b"\x02" + bytes([_ref_target(blob)[1]]))
    corrupted = blob[:tag + 1] + b"\xfe" + blob[tag + 2:]
    with pytest.raises(DjayWriteError):
        _parse(corrupted)
