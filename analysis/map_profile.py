"""Il profilo acustico di un brano: embedding, etichette, tempo, chiave.

È il primo stadio della sezione Map. Da un file audio tira fuori le due
cose che servono a metterlo su una mappa:

1. l'EMBEDDING — il vettore a 1280 dimensioni del penultimo strato di
   Discogs-EffNet, cioè l'identità acustica del brano prima che venga
   schiacciata su un elenco di nomi di genere;
2. le ETICHETTE — le stesse attivazioni, ma lette dalla testa di
   classificazione: più generi e più mood per brano, ognuno con la sua
   confidenza.

Sono lo stesso calcolo, intercettato in due punti: il ramo di
classificazione riusa gli embedding del primo, quindi le etichette non
costano niente in più. Questo è il motivo per cui un modello solo basta.

**Perché non si analizza tutto il brano.** Si prendono dodici finestre da
10 secondi distribuite su tutta la durata e si fa la media dei risultati
(temporal average pooling). Su una libreria da 90.000 brani la differenza
fra "tre minuti e mezzo per brano" e "un minuto e mezzo" sono giorni.

**Perché dodici corte e non tre lunghe.** Per una media conta il numero di
campioni INDIPENDENTI, e trenta secondi consecutivi sono quasi sempre una
sezione sola: le loro fettine si somigliano fra loro, quindi tre finestre
valgono tre campioni per quanto siano lunghe, e i primi novanta secondi di
un brano da sette minuti non li guarda nessuno. Misurato su 300 brani della
libreria, contro l'embedding del brano intero preso come verità: tre
finestre da 30 s azzeccano il brano più simile nel 58% dei casi, dodici da
10 s nell'84%.

La parte pura (dove tagliare, quali etichette tenere, quanto è regolare un
ritmo) sta in cima e si prova senza Essentia installata.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .essentia_tags import (
    GENRE_METADATA,
    GENRE_MODEL,
    MODEL_DIR,
    MOOD_METADATA,
    MOOD_MODEL,
    EMBEDDING_MODEL,
    _labels,
    format_genre_tag,
    format_mood_tag,
    missing_models,
)
from .mixing import to_camelot

# Il modello vuole 16 kHz, ma il file si legge a 44100: a 16 kHz il tempo e
# la tonalità escono sbagliati (misurato su un disco a 124 BPM: 86,5 BPM e
# la chiave sbagliata di una quinta), e per giunta il decoder ci mette di
# più, perché deve ricampionare tutto il brano invece che i 90 secondi che
# servono davvero (0,95 s contro 5,4 s su un brano di sette minuti).
ANALYSIS_RATE = 44100
MODEL_RATE = 16000

EMBEDDING_DIM = 1280

# Livello a cui si porta ogni brano prima dell'inferenza (EBU R128). Serve a
# non far dipendere il risultato da quanto è stato spinto il mastering.
TARGET_LUFS = -14.0


@dataclass
class ProfileSettings:
    """Come guardare un brano."""

    # Le finestre dell'embedding, come frazioni della durata totale. La
    # specifica ne diceva tre da 30 s ai quarti; a parità di audio e di
    # costo (3x40 s contro 12x10 s) dodici corte e sparse portano la
    # fedeltà al brano intero da 0,9859 a 0,9974, e il caso peggiore — il
    # brano su cui le finestre cascano male — da 0,9562 a 0,9937.
    segment_positions: tuple[float, ...] = tuple((i + 0.5) / 12 for i in range(12))
    segment_seconds: float = 10.0
    # Il ritmo NON si misura su quelle: dieci secondi sono una manciata di
    # battute e RhythmExtractor2013 ci si perde. Se ne prende una dedicata.
    rhythm_seconds: float = 30.0
    target_lufs: float = TARGET_LUFS

    # Soglia di confidenza per tenere un'etichetta (il T della specifica).
    #
    # La specifica dice 0,40 per entrambi i rami. Misurato su questi modelli
    # non regge: il genere è un softmax su 400 classi, e su un brano
    # riconoscibilissimo il primo genere arriva a 0,404 — a 0,40 la quasi
    # totalità della libreria resterebbe senza un'etichetta. Il mood è un
    # sigmoid multi-label e sullo stesso brano non passa 0,20. I valori qui
    # sotto sono quelli che questa repo già usa nel tagging e che producono
    # tre generi e cinque mood su quel brano; la soglia resta un parametro,
    # e chi vuole i 0,40 della specifica li mette dalla pagina.
    genre_threshold: float = 0.15
    mood_threshold: float = 0.05
    max_genres: int = 4
    max_moods: int = 4

    # Tempo e tonalità: se il file li porta già nei tag (una libreria da DJ
    # di solito sì) si prendono da lì, che è esatto e gratis.
    trust_tags: bool = True


@dataclass
class TrackProfile:
    """Cosa sappiamo di un brano dopo averlo guardato."""

    path: Path
    duration: float = 0.0
    bpm: float | None = None
    camelot: str | None = None
    key: str | None = None
    lufs: float | None = None            # loudness integrata PRIMA della normalizzazione
    danceability: float | None = None    # 0..1, regolarità degli onset
    genres: list[tuple[str, float]] = field(default_factory=list)
    moods: list[tuple[str, float]] = field(default_factory=list)
    embedding: np.ndarray | None = None
    error: str | None = None

    @property
    def top_genre(self) -> str:
        return self.genres[0][0] if self.genres else "—"

    def to_row(self) -> dict:
        """La riga da mettere in tabella. L'embedding NO: quello vive nella
        sua matrice, di fianco, perché sono 1280 numeri per brano."""
        return {
            "path": str(self.path),
            "name": self.path.name,
            "folder": str(self.path.parent),
            "duration": round(self.duration, 1),
            "bpm": round(self.bpm, 1) if self.bpm else None,
            "camelot": self.camelot or "",
            "key": self.key or "",
            "lufs": round(self.lufs, 1) if self.lufs is not None else None,
            "danceability": round(self.danceability, 3)
            if self.danceability is not None else None,
            "genres": "; ".join(g for g, _ in self.genres),
            "top_genre": self.top_genre,
            "moods": "; ".join(m for m, _ in self.moods),
            "confidence": "; ".join(f"{g}:{c:.2f}" for g, c in self.genres),
        }


# --------------------------------------------------------------------------
# Parte pura
# --------------------------------------------------------------------------

def segment_offsets(duration: float, settings: ProfileSettings) -> list[float]:
    """Dove cominciano le finestre da analizzare, in secondi.

    Le posizioni sono frazioni della durata, ma una finestra non può
    sporgere dalla fine del brano: si arretra. Su un brano più corto della
    finestra stessa resta un solo pezzo, che è tutto il brano — analizzarlo
    tre volte darebbe tre volte lo stesso vettore.
    """
    window = settings.segment_seconds
    if duration <= window:
        return [0.0]
    starts = []
    for fraction in settings.segment_positions:
        start = min(max(0.0, duration * fraction - window / 2), duration - window)
        if all(abs(start - s) > window / 4 for s in starts):
            starts.append(start)
    return starts or [0.0]


def rhythm_offset(duration: float, settings: ProfileSettings) -> float:
    """Dove comincia la finestra su cui si misurano tempo, tonalità e onset.

    Al centro del brano e lunga per conto suo: le finestre dell'embedding
    sono troppo corte perché un rilevatore di tempo ci trovi delle battute.
    Su un brano più corto della finestra resta tutto il brano.
    """
    return max(0.0, (duration - settings.rhythm_seconds) / 2)


def gain_for_target(lufs: float | None, target: float = TARGET_LUFS) -> float:
    """Il fattore per cui moltiplicare il segnale per portarlo a `target`.

    Il limite a 12 dB serve ai brani quasi muti (registrazioni sbagliate,
    tracce di soli effetti): senza, si moltiplicherebbe per mille il rumore
    di fondo e il modello classificherebbe quello.
    """
    if lufs is None or not np.isfinite(lufs) or lufs < -70:
        return 1.0
    return float(min(10 ** ((target - lufs) / 20), 10 ** (12 / 20)))


def select_labels(activations, labels: list[str], threshold: float,
                  limit: int) -> list[tuple[str, float]]:
    """Le etichette sopra soglia, dalla più forte, al massimo `limit`.

    Multi-label per scelta: la musica da club è ibrida, e un brano che è
    per metà minimal e per metà deep house va trovato cercando l'una o
    l'altra. Se nessuna passa la soglia si tiene comunque la prima — un
    genere un brano ce l'ha sempre, e senza almeno un'etichetta il brano
    sparirebbe da ogni filtro.
    """
    activations = np.asarray(activations, dtype=float)
    order = np.argsort(activations)[::-1][:limit]
    chosen = [(labels[i], float(activations[i])) for i in order
              if activations[i] >= threshold]
    if not chosen and len(order):
        best = int(order[0])
        chosen = [(labels[best], float(activations[best]))]
    return chosen


def onset_regularity(onsets) -> float | None:
    """Quanto è regolare la spaziatura degli attacchi: 0..1.

    È il coefficiente di danceability della specifica. Si misura la
    dispersione degli intervalli fra un onset e il successivo: una cassa
    dritta li tiene tutti uguali (dispersione ~0, valore ~1), un breakbeat
    sincopato li sparpaglia. Serve meno di una decina di onset per dire
    qualcosa: sotto, si risponde "non lo so" invece di inventare.
    """
    times = np.asarray(onsets, dtype=float)
    if len(times) < 8:
        return None
    gaps = np.diff(np.sort(times))
    gaps = gaps[gaps > 0]
    if len(gaps) < 4 or gaps.mean() <= 0:
        return None
    variation = gaps.std() / gaps.mean()      # coefficiente di variazione
    return float(max(0.0, 1.0 - min(1.0, variation)))


# --------------------------------------------------------------------------
# Tempo e tonalità dai tag
# --------------------------------------------------------------------------

def read_tag_tempo_key(path: Path) -> tuple[float | None, str | None]:
    """BPM e tonalità come li porta il file, se li porta.

    Una libreria già passata per rekordbox o djay li ha quasi sempre, e sono
    quelli che il DJ vede sui deck: ricalcolarli significherebbe mostrare due
    numeri diversi per la stessa cosa.
    """
    import mutagen

    try:
        audio = mutagen.File(path)
    except Exception:                                   # noqa: BLE001
        return None, None
    if audio is None or audio.tags is None:
        return None, None
    tags = audio.tags

    bpm = key = None
    if hasattr(tags, "getall"):                         # ID3
        bpm = _tag_first([f.text for f in tags.getall("TBPM")])
        key = _tag_first([f.text for f in tags.getall("TKEY")])
    elif type(tags).__name__ == "MP4Tags":
        tempo = tags.get("tmpo")
        bpm = str(tempo[0]) if tempo else None
        key = _tag_first(tags.get("----:com.apple.iTunes:initialkey"))
    else:                                               # Vorbis, APEv2, ASF
        for name in ("BPM", "bpm", "TBPM", "WM/BeatsPerMinute"):
            bpm = bpm or _tag_first(tags.get(name))
        for name in ("INITIALKEY", "initialkey", "KEY", "key", "WM/InitialKey"):
            key = key or _tag_first(tags.get(name))

    try:
        tempo = float(str(bpm).strip()) if bpm else None
    except ValueError:
        tempo = None
    return (tempo if tempo and 40 <= tempo <= 220 else None), (str(key) if key else None)


def _tag_first(values) -> str | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        values = [values]
    for v in values:
        if isinstance(v, (list, tuple)):
            found = _tag_first(v)
            if found:
                return found
            continue
        text = v.decode("utf-8", "ignore") if isinstance(v, bytes) else str(v)
        if text.strip():
            return text.strip()
    return None


# --------------------------------------------------------------------------
# L'analizzatore (l'unica parte che richiede Essentia)
# --------------------------------------------------------------------------

class ProfileAnalyzer:
    """Carica i modelli una volta e profila i brani uno dopo l'altro."""

    def __init__(self, settings: ProfileSettings | None = None,
                 model_dir: Path = MODEL_DIR) -> None:
        self.settings = settings or ProfileSettings()
        self.model_dir = Path(model_dir)
        self._loaded = False
        self._embedding = self._genre = self._mood = None
        self._resample = None
        self.genre_labels: list[str] = []
        self.mood_labels: list[str] = []

    def load(self) -> None:
        if self._loaded:
            return
        missing = missing_models(self.model_dir)
        if missing:
            raise FileNotFoundError(
                f"Modelli mancanti in {self.model_dir}: {', '.join(missing)}")

        import essentia
        essentia.log.warningActive = False
        from essentia.standard import (Resample, TensorflowPredict2D,
                                       TensorflowPredictEffnetDiscogs)

        self._embedding = TensorflowPredictEffnetDiscogs(
            graphFilename=str(self.model_dir / EMBEDDING_MODEL),
            output="PartitionedCall:1")
        self._genre = TensorflowPredict2D(
            graphFilename=str(self.model_dir / GENRE_MODEL),
            input="serving_default_model_Placeholder", output="PartitionedCall")
        self._mood = TensorflowPredict2D(
            graphFilename=str(self.model_dir / MOOD_MODEL),
            input="model/Placeholder", output="model/Sigmoid")
        self.genre_labels = _labels(self.model_dir / GENRE_METADATA)
        self.mood_labels = _labels(self.model_dir / MOOD_METADATA)
        self._resample = Resample(inputSampleRate=ANALYSIS_RATE,
                                  outputSampleRate=MODEL_RATE, quality=1)
        self._loaded = True

    def analyze(self, path: Path) -> TrackProfile:
        from essentia.standard import (KeyExtractor, LoudnessEBUR128, MonoLoader,
                                       OnsetRate, RhythmExtractor2013)

        self.load()
        settings = self.settings
        path = Path(path)

        audio = MonoLoader(filename=str(path), sampleRate=ANALYSIS_RATE,
                           resampleQuality=1)()
        duration = len(audio) / ANALYSIS_RATE
        if duration < 1.0:
            return TrackProfile(path=path, duration=duration,
                                error="troppo corto per essere analizzato")

        window = int(settings.segment_seconds * ANALYSIS_RATE)
        segments = [audio[int(start * ANALYSIS_RATE):int(start * ANALYSIS_RATE) + window]
                    for start in segment_offsets(duration, settings)]
        joined = np.concatenate(segments)

        # Loudness integrata sulle finestre, non sul brano intero: è la stessa
        # misura sul materiale che poi si analizza, e costa mezzo secondo
        # invece di tre. Il segnale è mono, l'algoritmo vuole stereo.
        stereo = np.column_stack([joined, joined]).astype(np.float32)
        _, _, integrated, _ = LoudnessEBUR128(sampleRate=ANALYSIS_RATE)(stereo)
        gain = gain_for_target(float(integrated), settings.target_lufs)
        segments = [(s * gain).astype(np.float32) for s in segments]

        profile = TrackProfile(path=path, duration=duration,
                               lufs=float(integrated))

        # Tempo e tonalità: dal tag se c'è, altrimenti da una finestra presa
        # al centro apposta per loro — sono proprietà del brano intero,
        # misurarle su ognuna delle dodici finestre non le migliora e costa
        # mezzo secondo a giro.
        if settings.trust_tags:
            profile.bpm, profile.key = read_tag_tempo_key(path)
        rhythm_start = int(rhythm_offset(duration, settings) * ANALYSIS_RATE)
        middle = (audio[rhythm_start:
                        rhythm_start + int(settings.rhythm_seconds * ANALYSIS_RATE)]
                  * gain).astype(np.float32)
        if profile.bpm is None:
            bpm, _, _, _, _ = RhythmExtractor2013(method="multifeature")(middle)
            profile.bpm = float(bpm)
        if profile.key is None:
            note, scale, _ = KeyExtractor()(middle)
            profile.key = f"{note} {scale}"
        profile.camelot = to_camelot(profile.key)

        onsets = OnsetRate()(middle)[0]
        profile.danceability = onset_regularity(onsets)

        # Ramo embedding + ramo classificazione: un'inferenza sola, letta in
        # due punti. Le finestre si incollano PRIMA di entrare nel modello,
        # che lavora a lotti fissi da 64 fettine e ne paga uno intero anche
        # per le nove che escono da una finestra da dieci secondi: dodici
        # chiamate sarebbero dodici lotti, incollate stanno in due. Misurato,
        # 5,1 s contro 2,3 s a brano; il vettore si sposta di 0,0006.
        frames = np.asarray(self._embedding(np.concatenate(
            [self._resample(s) for s in segments]).astype(np.float32)))
        profile.embedding = frames.mean(axis=0).astype(np.float32)
        profile.genres = select_labels(
            np.mean(self._genre(frames), axis=0),
            [format_genre_tag(g, "parent_child") for g in self.genre_labels],
            settings.genre_threshold, settings.max_genres)
        profile.moods = select_labels(
            np.mean(self._mood(frames), axis=0),
            [format_mood_tag(m) for m in self.mood_labels],
            settings.mood_threshold, settings.max_moods)
        return profile


# --- Analisi in parallelo (stesso schema del tagging) ----------------------

def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) // 2)


_POOL_ANALYZER: ProfileAnalyzer | None = None


def _pool_init(settings: ProfileSettings, model_dir: Path) -> None:
    global _POOL_ANALYZER
    _POOL_ANALYZER = ProfileAnalyzer(settings, model_dir)
    _POOL_ANALYZER.load()


def _pool_analyze(path: Path) -> TrackProfile:
    try:
        return _POOL_ANALYZER.analyze(path)
    except Exception as e:                              # noqa: BLE001
        return TrackProfile(path=Path(path), error=f"{type(e).__name__}: {e}")


def profile_many(paths, settings: ProfileSettings | None = None, workers: int = 1,
                 model_dir: Path = MODEL_DIR):
    """Profila `paths`, restituendo un `TrackProfile` per volta.

    L'errore viaggia dentro al profilo, non come eccezione: un brano
    illeggibile su cinquemila non deve far cadere la coda.
    """
    settings = settings or ProfileSettings()
    paths = [Path(p) for p in paths]
    if workers <= 1 or len(paths) <= 1:
        analyzer = ProfileAnalyzer(settings, model_dir)
        for path in paths:
            try:
                yield analyzer.analyze(path)
            except Exception as e:                      # noqa: BLE001
                yield TrackProfile(path=path, error=f"{type(e).__name__}: {e}")
        return

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(
        max_workers=min(workers, len(paths)),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_pool_init, initargs=(settings, model_dir),
    ) as pool:
        yield from pool.map(_pool_analyze, paths)
