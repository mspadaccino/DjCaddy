"""Il job lungo: quello che deve reggere e' l'interruzione."""

from __future__ import annotations

import json
import os
from pathlib import Path

from analysis.essentia_tags import Prediction, TagSettings, TrackTags
from analysis.tag_job import JobState, load_state, run_job


def test_a_dead_job_does_not_look_alive():
    """Un job ucciso non fa in tempo a scrivere che e' finito.

    Senza il controllo sul pid l'interfaccia mostrerebbe per sempre un
    lavoro fermo come ancora in corso.
    """
    assert JobState(pid=os.getpid(), started_at=0).running
    assert not JobState(pid=999_999, started_at=0).running
    assert not JobState(pid=os.getpid(), started_at=0, finished_at=1).running
    assert not JobState().running


def test_state_survives_being_read_mid_write(tmp_path):
    stato = JobState(pid=1, total=9, done=4, written=3, started_at=100.0)
    percorso = tmp_path / "stato.json"
    stato.save(percorso)

    riletto = load_state(percorso)
    assert (riletto.total, riletto.done, riletto.written) == (9, 4, 3)
    assert load_state(tmp_path / "mai-esistito.json") is None
    (tmp_path / "rotto.json").write_text("{ meta")
    assert load_state(tmp_path / "rotto.json") is None


def test_the_job_writes_as_it_goes_and_keeps_going_after_a_failure(
        tmp_path, monkeypatch):
    """Un brano che fallisce non deve fermare il job ne' sparire dal conto.

    E lo stato su disco deve essere aggiornato PRIMA della fine: e' l'unico
    modo che ha l'app di sapere a che punto e', ed e' cio' che rende il
    lavoro non perso se il job viene interrotto.
    """
    from analysis import tag_job

    brani = [tmp_path / f"{n}.mp3" for n in ("uno", "due", "tre")]
    for b in brani:
        b.write_bytes(b"finto")

    monkeypatch.setattr(tag_job, "find_taggable", lambda root: list(brani))
    monkeypatch.setattr(
        tag_job, "analyze_many",
        lambda paths, settings, workers=1: (
            (p, None, "boom") if p.name == "due.mp3"
            else (p, TrackTags(genres=[Prediction("House", 0.9)]), None)
            for p in paths))

    visti = []
    stato_file = tmp_path / "stato.json"

    def scrivi(path, tags, settings):
        # a meta' corsa lo stato su disco deve gia' raccontare i fatti
        visti.append(json.loads(stato_file.read_text())["done"])
        return ["TCON=House"]

    monkeypatch.setattr(tag_job, "write_tags", scrivi)

    stato = run_job(tmp_path, TagSettings(), only_missing=False,
                    skip_done=False, state_file=stato_file)

    assert (stato.total, stato.done) == (3, 3)
    assert stato.written == 2 and stato.failed == 1
    assert stato.errors[0]["error"] == "boom"
    assert "due.mp3" in stato.errors[0]["file"]
    # Lo stato su disco e' gia' avanzato mentre il job e' a meta': quando si
    # scrive il terzo brano, il file racconta gia' i due precedenti. Resta
    # indietro del brano in volo, che e' corretto — si salva DOPO aver
    # scritto il tag, cosi' non dichiara mai fatto qualcosa che non lo e'.
    assert visti == [0, 2]
    assert stato.finished_at is not None


def test_the_queue_can_ignore_the_progress_file(tmp_path, monkeypatch):
    """I due filtri sono distinti: i FILE e il REGISTRO di cosa e' stato
    tentato. Non coincidono, e si deve poter lavorare senza il secondo."""
    from analysis import tag_job

    brani = [tmp_path / "a.mp3", tmp_path / "b.mp3"]
    monkeypatch.setattr(tag_job, "find_taggable", lambda root: list(brani))

    class RegistroCheSaTutto:
        def pending(self, paths):
            return []

    coda = tag_job.build_queue(tmp_path, TagSettings(), only_missing=False,
                               skip_done=True, tracker=RegistroCheSaTutto())
    assert coda == []

    coda = tag_job.build_queue(tmp_path, TagSettings(), only_missing=False,
                               skip_done=False)
    assert coda == brani


def test_the_eta_measures_recent_work_not_wall_clock(monkeypatch):
    """Un job congelato — Mac addormentato, o Pause — resta vivo per ore
    senza lavorare. Dividendo il tempo trascorso per i brani fatti l'attesa
    dichiarata esplode: misurato su una ricostruzione della mappa, 4,73 s a
    brano contro gli 0,85 veri, cioè 32 ore invece di 6.
    """
    import analysis.tag_job as tag_job

    minuto = [1_000_000]
    monkeypatch.setattr(tag_job.time, "time", lambda: minuto[0] * 60.0)

    # Dodici ore a muro, ma solo dieci brani da un secondo l'uno.
    stato = JobState(pid=1, total=110, done=10, started_at=0.0,
                     updated_at=12 * 3600.0)
    for _ in range(10):
        stato.tick(1.0)

    assert stato.seconds_each == 1.0
    assert stato.eta_seconds == 100.0           # non 12 ore per 100 brani


def test_the_hours_asleep_are_not_mistaken_for_a_slow_track(monkeypatch):
    import analysis.tag_job as tag_job

    monkeypatch.setattr(tag_job.time, "time", lambda: 60_000_000.0)
    stato = JobState(pid=1, total=10, done=3)
    stato.tick(2.0)
    stato.tick(9 * 3600.0)          # qui in mezzo la macchina dormiva
    stato.tick(2.0)

    assert stato.seconds_each == 2.0


def test_only_the_last_half_hour_of_work_counts(monkeypatch):
    """Se il ritmo cambia, la stima deve seguirlo invece di ricordarsi per
    sempre com'era all'inizio."""
    import analysis.tag_job as tag_job

    minuto = [1_000_000]
    monkeypatch.setattr(tag_job.time, "time", lambda: minuto[0] * 60.0)

    stato = JobState(pid=1, total=100, done=0)
    for _ in range(40):             # quaranta minuti lenti, uno per minuto
        stato.tick(10.0)
        minuto[0] += 1
    assert len(stato.recent) == tag_job.RECENT_MINUTES
    for _ in range(30):             # poi mezz'ora veloce
        stato.tick(1.0)
        minuto[0] += 1

    assert stato.seconds_each == 1.0        # dei dieci secondi non resta nulla


def test_an_old_state_file_still_says_how_long_it_took():
    """Chi legge lo stato di un job partito prima che il ritmo si misurasse
    non deve trovarsi un'attesa di zero."""
    stato = JobState(pid=1, total=10, done=4, started_at=100.0,
                     updated_at=140.0)
    assert stato.recent == []
    assert stato.seconds_each == 10.0
    assert stato.eta_seconds == 60.0
