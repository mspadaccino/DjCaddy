import numpy as np

from analysis import energy


def _tone(hz, seconds=1.0, sr=44100):
    t = np.arange(int(seconds * sr)) / sr
    return np.sin(2 * np.pi * hz * t).astype(np.float32)


# --- le tre misure --------------------------------------------------------

def test_a_sub_tone_is_all_bass_and_a_hiss_is_none():
    low = energy.spectrum(_tone(60), 44100)
    high = energy.spectrum(_tone(9000), 44100)
    assert energy.bass_share(*low) > 0.95
    assert energy.bass_share(*high) < 0.05


def test_brightness_lands_on_the_tone_it_is_given():
    freqs, power = energy.spectrum(_tone(4000), 44100)
    assert 3800 < energy.brightness(freqs, power) < 4200


def test_a_dark_track_reads_lower_than_a_bright_one():
    dark = energy.spectrum(_tone(80) + 0.5 * _tone(300), 44100)
    bright = energy.spectrum(_tone(80) + 0.5 * _tone(6000), 44100)
    assert energy.brightness(*dark) < energy.brightness(*bright)


def test_a_direct_current_offset_is_not_bass():
    # Un file registrato male ha una continua: senza il taglio a 20 Hz
    # passerebbe per il brano piu' bassoso della libreria.
    freqs, power = energy.spectrum(_tone(9000) + 1.0, 44100)
    assert energy.bass_share(freqs, power) < 0.05


def test_a_window_shorter_than_the_fft_says_it_does_not_know():
    assert energy.spectrum(np.zeros(100, dtype=np.float32), 44100) == (None, None)
    assert energy.bass_share(None, None) is None
    assert energy.brightness(None, None) is None


def test_silence_is_not_a_zero():
    freqs, power = energy.spectrum(np.zeros(44100, dtype=np.float32), 44100)
    assert energy.bass_share(freqs, power) is None
    assert energy.brightness(freqs, power) is None


# --- la densita' ----------------------------------------------------------

def test_onsets_become_a_count_per_beat():
    assert energy.per_beat(4.0, 120.0) == 2.0        # 4/s a 2 battiti/s


def test_the_same_onsets_are_denser_at_a_slower_tempo():
    assert energy.per_beat(8.0, 90.0) > energy.per_beat(8.0, 128.0)


def test_without_a_tempo_there_is_no_density():
    assert energy.per_beat(4.0, None) is None
    assert energy.per_beat(None, 120.0) is None
    assert energy.per_beat(4.0, 0.0) is None


# --- il rango -------------------------------------------------------------

def test_ranks_stretch_from_zero_to_one():
    assert list(energy.ranks([10.0, 20.0, 30.0])) == [0.0, 0.5, 1.0]


def test_equal_values_share_the_same_rank():
    r = energy.ranks([5.0, 5.0, 9.0])
    assert r[0] == r[1] < r[2]


def test_a_missing_value_stays_missing_and_does_not_shift_the_others():
    r = energy.ranks([10.0, np.nan, 30.0])
    assert np.isnan(r[1])
    assert (r[0], r[2]) == (0.0, 1.0)


def test_an_empty_column_gives_nothing_back():
    assert np.isnan(energy.ranks([np.nan, np.nan])).all()


# --- la scala -------------------------------------------------------------

def _library(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(4, 1, n), rng.uniform(0.2, 0.8, n), rng.normal(2000, 600, n))


def test_the_track_that_leads_on_all_three_ends_on_top():
    density, bass, bright = _library()
    density[0], bass[0], bright[0] = density.max() + 1, 1.0, bright.max() + 1
    density[1], bass[1], bright[1] = density.min() - 1, 0.0, bright.min() - 1
    level = energy.levels(density, bass, bright)
    assert level[0] == 10 and level[1] == 1


def test_the_ten_levels_are_all_populated():
    # E' la promessa della scala: l'asse energia dal mood era stato scartato
    # perche' il decile inferiore restava vuoto.
    level = energy.levels(*_library(2000))
    assert set(np.unique(level[np.isfinite(level)])) == set(range(1, 11))


def test_the_spread_uses_the_whole_height():
    # La media di tre ranghi si stringe su 0,5: e' il secondo rango che
    # riapre la scala, e senza la lavagna disegnerebbe una riga piatta.
    density, bass, bright = _library(2000)
    narrow = energy.mix(density, bass, bright)
    wide = energy.spread(density, bass, bright)
    assert narrow.max() - narrow.min() < 0.95
    assert wide.max() - wide.min() > 0.99


def test_a_missing_measure_leaves_the_others_deciding():
    density, bass, bright = _library()
    density[0] = np.nan
    assert np.isfinite(energy.levels(density, bass, bright)[0])


def test_a_track_with_nothing_measured_has_no_energy():
    nan = [np.nan] * 3
    assert np.isnan(energy.levels(nan, nan, nan)).all()


def test_a_measure_that_is_missing_does_not_count_as_a_zero():
    # Un brano senza BPM non deve finire in fondo alla scala per un dato
    # assente: senza density vale quello che dicono le altre due.
    density, bass, bright = _library()
    density[0], bass[0], bright[0] = np.nan, 1.0, bright.max() + 1
    assert energy.levels(density, bass, bright)[0] == 10


def test_the_weights_can_move_the_reading():
    density, bass, bright = _library()
    density[0], bass[0], bright[0] = density.max() + 1, 0.0, bright.min() - 1
    on_density = energy.levels(density, bass, bright, weights=(3.0, 1.0, 1.0))
    off_density = energy.levels(density, bass, bright, weights=(0.0, 1.0, 1.0))
    assert on_density[0] > off_density[0]


def test_measure_reads_a_window_into_the_three_fields():
    out = energy.measure(_tone(60, seconds=2.0), 44100, onset_rate=4.0, bpm=120.0)
    assert set(out) == set(energy.INGREDIENTS)
    assert out["energy_density"] == 2.0
    assert out["energy_bass"] > 0.95


# --- il tempo piegato ------------------------------------------------------

def test_a_half_time_tag_reads_like_the_tempo_it_really_is():
    # "60 bpm - Nicki Minaj" e' un 120 contato a meta': diviso per 60 la sua
    # densita' verrebbe il doppio del vero.
    assert energy.per_beat(6.0, 60.0) == energy.per_beat(6.0, 120.0)
    assert energy.fold_tempo(60.0) == 120.0


def test_folding_never_slows_a_tempo_down():
    # I due errori peggiori del campione erano i due brani piu' veloci, e
    # venivano da qui: piegare 151,6 a 75,8 ne raddoppiava la densita'.
    for bpm in (142.2, 149.0, 151.6, 172.3, 184.6):
        assert energy.fold_tempo(bpm) == bpm


def test_folding_lands_at_or_above_the_floor():
    for bpm in (40.0, 63.0, 86.0, 125.0, 300.0):
        assert energy.fold_tempo(bpm) >= 70.0


def test_folding_leaves_a_tempo_that_is_already_home_alone():
    assert energy.fold_tempo(125.0) == 125.0


def test_a_tempo_that_is_not_a_tempo_folds_to_nothing():
    assert energy.fold_tempo(None) is None
    assert energy.fold_tempo(0.0) is None
    assert energy.fold_tempo(float("nan")) is None


# --- la finestra muta ------------------------------------------------------

def test_a_silent_window_is_not_measured():
    assert not energy.usable(np.zeros(44100, dtype=np.float32))
    out = energy.measure(np.zeros(44100, dtype=np.float32), 44100, 4.0, 120.0)
    assert all(v is None for v in out.values())


def test_a_window_with_music_in_it_is_measured():
    assert energy.usable(_tone(200, seconds=1.0))


def test_a_window_shorter_than_the_fft_is_not_measured():
    assert not energy.usable(_tone(200, seconds=0.01))


# --- chi entra nella scala -------------------------------------------------

def test_a_drop_too_short_to_be_a_track_stays_out():
    import energy_cli
    rows = [{"duration": 12.0}, {"duration": 240.0}, {"duration": 61.0}]
    assert energy_cli.playable(rows) == [{"duration": 240.0}, {"duration": 61.0}]


def test_a_missing_duration_is_not_a_short_track():
    import energy_cli
    assert energy_cli.playable([{"duration": 0.0}, {}]) == [{"duration": 0.0}, {}]


# --- il basso che batte in tempo -------------------------------------------

def _kicks(at_beats, bpm=120.0, seconds=8.0, sr=44100, hz=60.0):
    """Un basso che colpisce nei punti dati, misurati in battiti."""
    out = np.zeros(int(seconds * sr), dtype=np.float32)
    body = np.arange(int(0.15 * sr)) / sr
    hit = (np.sin(2 * np.pi * hz * body) * np.exp(-body * 25)).astype(np.float32)
    for beat in at_beats:
        start = int(beat * 60.0 / bpm * sr)
        if start + len(hit) <= len(out):
            out[start:start + len(hit)] += hit
    return out


def test_a_straight_kick_pulses_on_every_beat():
    floor = _kicks(np.arange(16))                     # una cassa per battito
    assert energy.pulse(floor, 44100, 120.0) > 0.5


def test_a_syncopated_bass_does_not_pulse_on_the_beat():
    # Gli STESSI colpi, spostati fuori dal battito: un 808 che cade in punti
    # diversi di ogni battuta ha la stessa densita' e non spinge.
    floor = _kicks(np.arange(16))
    off = _kicks(np.arange(16) * 0.75)
    assert energy.pulse(off, 44100, 120.0) < energy.pulse(floor, 44100, 120.0)


def test_a_bass_that_never_stops_does_not_pulse():
    # Un sub tenuto e' tanto basso quanto vuoi ma non batte: e' la differenza
    # fra "quanto fondo c'e'" (energy_bass) e "il fondo va a tempo" (questa).
    t = np.arange(int(8.0 * 44100)) / 44100
    drone = np.sin(2 * np.pi * 60 * t).astype(np.float32)
    assert energy.pulse(drone, 44100, 120.0) < 0.2


def test_without_a_tempo_there_is_no_pulse():
    assert energy.pulse(_kicks(np.arange(16)), 44100, None) is None


def test_a_half_time_tag_still_finds_the_beat():
    floor = _kicks(np.arange(16))
    assert energy.pulse(floor, 44100, 60.0) > 0.5      # 60 si piega a 120


def test_measure_now_reads_four_things():
    out = energy.measure(_kicks(np.arange(16)), 44100, onset_rate=2.0, bpm=120.0)
    assert set(out) == set(energy.INGREDIENTS)
    assert out["energy_pulse"] > 0.5
