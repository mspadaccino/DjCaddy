import struct

import pytest

from analysis.djay_write import (
    CuePoint,
    DjayWriteError,
    append_cues,
    read_cue_points,
    verify_roundtrip,
)


def _verbose_cue(time: float, end_time: float, number: int) -> bytes:
    """Costruisce un cue "verboso" (nomi dei campi in chiaro), come il primo
    cue point scritto da djay Pro in un blob mediaItemUserData."""
    return (
        b"ADCCuePoint\x00"
        + b"\x13\x00" + struct.pack("<f", time)
        + b"\x08time\x00"
        + b"\x13\x00" + struct.pack("<f", end_time)
        + b"\x08endTime\x00"
        + struct.pack("<B", number)
        + b"\x08number\x00"
    )


def _compact_cue(time: float, end_time: float, number: int) -> bytes:
    return (
        b"+\x05\x0f\x13\x00" + struct.pack("<f", time)
        + b"\x05\x10\x13\x00" + struct.pack("<f", end_time)
        + b"\x05\x11\x0f" + struct.pack("<B", number)
        + b"\x05\x12\x00"
    )


def _synthetic_blob(n_cues: int) -> bytes:
    """Blob minimale ma strutturalmente fedele: firma TSAF, header a offset 8
    (valore arbitrario ma coerente), array cuePoints con `n_cues` elementi
    (un verboso + il resto compatti), e il marcatore dell'impronta audio con
    un po' di coda per verificare che venga preservata.
    """
    assert 1 <= n_cues <= 3
    cues = _verbose_cue(1.0, -1.0, 0)
    for i in range(1, n_cues):
        cues += _compact_cue(10.0 * i, -1.0, i)

    header = b"TSAF" + b"\x00\x00\x00\x00" + struct.pack("<I", 6)  # offset 8 = 6 (valore fittizio)
    array_header = b"\x05\x06\n\x00\x00" + struct.pack("<I", n_cues)
    tail = b"+\x08ADCAudioAlignmentFingerprint\x00restodellaccoda"
    return header + array_header + cues + tail


def test_read_cue_points_verbose_only():
    blob = _synthetic_blob(1)
    cues = read_cue_points(blob)
    assert cues == [CuePoint(time=1.0, number=0, end_time=-1.0)]


def test_read_cue_points_mixed():
    blob = _synthetic_blob(3)
    cues = read_cue_points(blob)
    assert cues == [
        CuePoint(time=1.0, number=0, end_time=-1.0),
        CuePoint(time=10.0, number=1, end_time=-1.0),
        CuePoint(time=20.0, number=2, end_time=-1.0),
    ]


def test_append_cues_and_roundtrip():
    blob = _synthetic_blob(2)
    new_cues = [CuePoint(time=99.0, number=6), CuePoint(time=150.0, number=7)]
    updated = append_cues(blob, new_cues)

    verify_roundtrip(blob, updated, new_cues)  # non deve sollevare

    after = read_cue_points(updated)
    assert len(after) == 4
    assert after[2] == CuePoint(time=99.0, number=6, end_time=-1.0)
    assert after[3] == CuePoint(time=150.0, number=7, end_time=-1.0)

    # il contenuto dopo il marcatore dell'impronta audio resta intatto
    assert updated.endswith(b"+\x08ADCAudioAlignmentFingerprint\x00restodellaccoda")


def test_append_cues_updates_counters():
    blob = _synthetic_blob(2)
    before_total = struct.unpack("<I", blob[8:12])[0]
    updated = append_cues(blob, [CuePoint(time=5.0, number=1)])
    after_total = struct.unpack("<I", updated[8:12])[0]
    assert after_total == before_total + 1  # contatore totale oggetti


def test_append_cues_refuses_empty_array():
    # array cuePoints presente ma con conteggio 0 (mai popolato)
    header = b"TSAF" + b"\x00\x00\x00\x00" + struct.pack("<I", 0)
    array_header = b"\x05\x06\n\x00\x00" + struct.pack("<I", 0)
    tail = b"+\x08ADCAudioAlignmentFingerprint\x00coda"
    blob = header + array_header + tail
    with pytest.raises(DjayWriteError):
        append_cues(blob, [CuePoint(time=1.0, number=0)])


def test_append_cues_refuses_malformed_blob():
    with pytest.raises(DjayWriteError):
        append_cues(b"TSAF" + b"\x00" * 50, [CuePoint(time=1.0, number=0)])


def test_read_cue_points_refuses_wrong_magic():
    with pytest.raises(DjayWriteError):
        read_cue_points(b"NOTA MAGIC HERE")


def test_verify_roundtrip_catches_tampered_existing_cue():
    blob = _synthetic_blob(2)
    new_cues = [CuePoint(time=99.0, number=6)]
    updated = append_cues(blob, new_cues)
    # manomette artificialmente un cue esistente nel blob "aggiornato"
    tampered = bytearray(updated)
    time_offset = updated.find(_compact_cue(10.0, -1.0, 1)) + 5
    tampered[time_offset:time_offset + 4] = struct.pack("<f", 999.0)
    with pytest.raises(DjayWriteError):
        verify_roundtrip(blob, bytes(tampered), new_cues)
