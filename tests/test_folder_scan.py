

def test_decoder_retries_with_the_demuxer_the_extension_suggests(monkeypatch, tmp_path):
    """I pool per DJ distribuiscono MP3 dentro un contenitore WAV.

    ffmpeg salta il tag ID3 per riconoscere il formato, vede RIFF e sceglie
    il demuxer wav; quello rilegge dall'offset 0, trova "ID3" e fallisce. Il
    file è sano — misurati 110 casi veri in una libreria — quindi un rifiuto
    del riconoscimento automatico non basta a condannarlo.
    """
    from analysis import folder_scan as fs

    track = tmp_path / "pool.mp3"
    track.write_bytes(b"ID3\x03\x00\x00\x00\x00\x1fvRIFF")
    tentativi = []

    def finto(path, demuxer, timeout):
        tentativi.append(demuxer)
        if demuxer is None:
            return False, "invalid start code ID3[3] in RIFF header"
        return True, ""

    monkeypatch.setattr(fs, "_probe", finto)

    assert fs._decoder_error(track) is None
    assert tentativi == [None, "mp3"]


def test_decoder_still_condemns_what_fails_twice(monkeypatch, tmp_path):
    from analysis import folder_scan as fs

    track = tmp_path / "rotto.mp3"
    track.write_bytes(b"spazzatura")
    monkeypatch.setattr(
        fs, "_probe", lambda path, demuxer, timeout: (False, "Invalid data found"))

    assert fs._decoder_error(track) == "Invalid data found"


def test_deep_check_lets_the_decoder_overrule_mutagen(monkeypatch, tmp_path):
    """Mutagen legge i TAG: un file che non gli torna può benissimo suonare."""
    from analysis import folder_scan as fs

    track = tmp_path / "strano.mp3"
    track.write_bytes(b"qualcosa di lungo abbastanza")
    monkeypatch.setattr(fs, "_header_error",
                        lambda path: "intestazione illeggibile: HeaderNotFoundError")
    monkeypatch.setattr(fs, "_has_ffprobe", lambda: True)
    monkeypatch.setattr(fs, "_decoder_error", lambda path: None)

    assert fs.check_readable(track, deep=True) is None
    # senza decoder interpellato, invece, il dubbio di mutagen resta
    assert fs.check_readable(track, deep=False) is not None


def test_empty_file_is_condemned_before_anything_else(tmp_path):
    from analysis.folder_scan import check_readable

    vuoto = tmp_path / "vuoto.mp3"
    vuoto.touch()

    assert check_readable(vuoto, deep=True) == "file vuoto"
