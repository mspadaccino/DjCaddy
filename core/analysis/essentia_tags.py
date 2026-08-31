"""Genere e mood dai modelli Essentia, scritti nei tag del file.

Porta dentro Wavecut quello che faceva `tag_music.py` del progetto
Essentia-to-Metadata: embedding Discogs-EffNet, poi due classificatori
(genere a 400 classi, mood MTG-Jamendo), poi scrittura nei tag.

Il modulo è diviso in tre parti che si possono provare separatamente:

1. SCELTA (`select_genres`, `select_moods`) — dalle attivazioni dei modelli
   alle etichette da scrivere. Pure: si provano con array finti, senza
   Essentia installato.
2. FORMATO (`build_tag_values`) — quali stringhe finiscono nei tag. Pura, e
   soprattutto UNA sola: nell'originale la stessa logica era ricopiata in
   sei writer quasi identici, che è anche il motivo per cui in quello WMA
   era rimasto un `formatted_genresf` che lo faceva fallire su ogni file.
3. SCRITTURA (`write_tags`) — mappa quelle stringhe sui campi del formato.

`essentia` serve solo alla classe `EssentiaAnalyzer`: tutto il resto (e
tutti i test) gira senza.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .folder_scan import is_metadata_sidecar

MODEL_DIR = Path(os.path.expanduser("~/essentia_models"))

EMBEDDING_MODEL = "discogs-effnet-bs64-1.pb"
GENRE_MODEL = "genre_discogs400-discogs-effnet-1.pb"
GENRE_METADATA = "genre_discogs400-discogs-effnet-1.json"
MOOD_MODEL = "mtg_jamendo_moodtheme-discogs-effnet-1.pb"
MOOD_METADATA = "mtg_jamendo_moodtheme-discogs-effnet-1.json"

MODELS = {
    EMBEDDING_MODEL: "embedding (serve sempre)",
    GENRE_MODEL: "genere: 400 classi Discogs",
    GENRE_METADATA: "genere: elenco delle classi",
    MOOD_MODEL: "mood: MTG-Jamendo",
    MOOD_METADATA: "mood: elenco delle classi",
}

# Formati su cui sappiamo scrivere, raggruppati per famiglia di tag.
_VORBIS = {".flac", ".ogg", ".oga", ".opus"}
_ID3 = {".mp3", ".aiff", ".aif", ".wav", ".dsf"}
_MP4 = {".m4a", ".m4b", ".mp4", ".aac"}
_APEV2 = {".wv", ".ape", ".mpc", ".mp+"}
_ASF = {".wma"}
AUDIO_EXTENSIONS = _VORBIS | _ID3 | _MP4 | _APEV2 | _ASF

GENRE_FORMATS = ("parent_child", "child_parent", "child_only", "raw")

# Il modello di embedding vuole 16 kHz.
SAMPLE_RATE = 16000


def find_taggable(root: Path) -> list[Path]:
    """I file su cui ha senso lavorare sotto `root`.

    Scarta i sidecar "._<nome>" di macOS: portano l'estensione del brano ma
    sono metadati da 4 KB. Su una libreria reale su exFAT sono 88.115 file
    su 116.381 — i tre quarti della coda — e lo script originale li
    incontrava uno per uno, falliva, e non potendoli segnare come fatti li
    ritentava a ogni esecuzione. Meglio non metterli mai in coda.
    """
    return sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in AUDIO_EXTENSIONS
        and not is_metadata_sidecar(p)
    )


def available() -> bool:
    """True se il pacchetto Essentia è importabile in questo ambiente."""
    try:
        import essentia  # noqa: F401
    except Exception:
        return False
    return True


def missing_models(model_dir: Path = MODEL_DIR) -> list[str]:
    """Quali file di modello mancano. Lista vuota = tutto a posto."""
    return [name for name in MODELS if not (model_dir / name).exists()]


# --------------------------------------------------------------------------
# Impostazioni e risultati
# --------------------------------------------------------------------------

@dataclass
class TagSettings:
    """Le stesse opzioni che lo script chiedeva a riga di comando."""
    genres: bool = True
    moods: bool = True
    top_genres: int = 3
    genre_threshold: float = 0.15      # attivazione minima (0-1)
    genre_format: str = "parent_child"
    mood_threshold: float = 0.05       # attivazione minima (0-1)
    max_moods: int = 5                 # quanti mood tenere in tutto
    moods_in_tag: int = 3              # quanti finiscono nel tag/commento
    confidence_tags: bool = True
    # Se le percentuali entrano anche nel COMMENTO predefinito, quello che
    # djay Pro mostra: "Happy 87%; Deep 62%" invece di "Happy; Deep". Spenta
    # di default — il commento e' l'unica riga che si legge mentre si suona,
    # e allungarla e' una scelta, non un miglioramento ovvio. Il campo MOOD
    # dedicato resta pulito in ogni caso.
    confidence_in_comment: bool = False
    overwrite: bool = False            # riscrive anche se il tag c'è già
    # Quanti secondi di audio dare ai modelli, presi dall'INIZIO (0 = tutto).
    # Non abbassarlo sotto i 300 senza motivo: misurato su un disco-house con
    # un minuto di intro, a 120 s usciva "Ambient / Space, Dark, Relaxing",
    # a 300 s e a brano intero "Nu-Disco, House / Summer, Deep, Happy". Il
    # risparmio e' comunque modesto, 12 s contro 14 s su un brano di 4 minuti.
    max_seconds: int = 300


@dataclass
class Prediction:
    label: str
    confidence: float


@dataclass
class TrackTags:
    """Esito dell'analisi di un brano."""
    genres: list[Prediction] = field(default_factory=list)
    moods: list[Prediction] = field(default_factory=list)

    def formatted_genres(self, style: str = "parent_child") -> list[str]:
        return [format_genre_tag(g.label, style) for g in self.genres]

    def formatted_moods(self) -> list[str]:
        return [format_mood_tag(m.label) for m in self.moods]


# --------------------------------------------------------------------------
# 1. Scelta delle etichette
# --------------------------------------------------------------------------

def select_genres(activations, labels: list[str], settings: TagSettings) -> list[Prediction]:
    """I generi più forti che superano la soglia.

    Se nessuno la supera si tiene comunque il primo: un brano un genere ce
    l'ha sempre, e lasciare il campo vuoto è meno utile che scriverci il
    candidato migliore con la sua confidenza accanto.
    """
    order = sorted(range(len(labels)), key=lambda i: activations[i], reverse=True)
    chosen = [Prediction(labels[i], float(activations[i]))
              for i in order[:settings.top_genres]
              if activations[i] >= settings.genre_threshold]
    if not chosen and order:
        best = order[0]
        chosen = [Prediction(labels[best], float(activations[best]))]
    return chosen


def select_moods(activations, labels: list[str], settings: TagSettings) -> list[Prediction]:
    """I mood sopra soglia, dal più forte. Qui NON si ripiega sul migliore:
    un brano può benissimo non avere un mood riconoscibile."""
    chosen = [Prediction(labels[i], float(a)) for i, a in enumerate(activations)
              if a >= settings.mood_threshold]
    chosen.sort(key=lambda p: p.confidence, reverse=True)
    return chosen[:settings.max_moods]


def format_genre_tag(raw_genre: str, style: str = "parent_child") -> str:
    """Da "Rock---Alternative Rock" a qualcosa di leggibile."""
    if style == "raw" or "---" not in raw_genre:
        return raw_genre
    parent, _, child = raw_genre.partition("---")
    parent, child = parent.strip(), child.strip()
    if not child:
        return parent
    if style == "child_parent":
        return f"{child} - {parent}"
    if style == "child_only":
        return child
    return f"{parent} - {child}"


def format_mood_tag(raw_mood: str) -> str:
    return raw_mood.title()


# --------------------------------------------------------------------------
# 2. Cosa finisce nei tag
# --------------------------------------------------------------------------

@dataclass
class TagValues:
    """Le stringhe da scrivere, indipendenti dal formato del file."""
    genre: str | None = None
    genre_confidence: str | None = None
    mood: str | None = None
    mood_confidence: str | None = None
    comment: str | None = None         # cosa va nel commento PREDEFINITO


def build_tag_values(tags: TrackTags, settings: TagSettings) -> TagValues:
    values = TagValues()
    if settings.genres and tags.genres:
        values.genre = "; ".join(tags.formatted_genres(settings.genre_format))
        if settings.confidence_tags:
            values.genre_confidence = ", ".join(
                f"{g.label}: {g.confidence:.2%}" for g in tags.genres)
    if settings.moods and tags.moods:
        picked = tags.moods[:settings.moods_in_tag]
        values.mood = "; ".join(format_mood_tag(m.label) for m in picked)
        # Percentuali intere: il commento si legge di sfuggita mentre si
        # suona, i due decimali dei tag di servizio qui sono rumore.
        values.comment = "; ".join(
            f"{format_mood_tag(m.label)} {m.confidence:.0%}" for m in picked
        ) if settings.confidence_in_comment else values.mood
        if settings.confidence_tags:
            values.mood_confidence = ", ".join(
                f"{m.label}: {m.confidence:.2%}" for m in picked)
    return values


# --------------------------------------------------------------------------
# 3. Scrittura nei tag, per famiglia di formato
# --------------------------------------------------------------------------

def write_tags(filepath: Path, tags: TrackTags, settings: TagSettings) -> list[str]:
    """Scrive genere e mood nel file. Ritorna l'elenco di cosa è stato
    scritto (vuoto se non c'era niente da scrivere, o se i tag c'erano già
    e `overwrite` è spento).

    Il mood finisce anche nel campo commento PREDEFINITO, non solo in un
    campo "mood" dedicato: djay Pro mostra solo quello, ed è il motivo per
    cui questa funzione esiste.
    """
    import mutagen

    values = build_tag_values(tags, settings)
    if values.genre is None and values.mood is None:
        return []

    suffix = filepath.suffix.lower()
    if suffix in _VORBIS:
        return _write_vorbis(filepath, values, settings)
    if suffix in _ID3:
        return _write_id3(filepath, values, settings)
    if suffix in _MP4:
        return _write_mp4(filepath, values, settings)
    if suffix in _ASF:
        return _write_asf(filepath, values, settings)
    if suffix in _APEV2:
        return _write_apev2(filepath, values, settings)
    raise ValueError(f"Formato non gestito: {suffix}")


def _skip_genre(values, settings: TagSettings) -> bool:
    """Se il campo del file va lasciato in pace.

    `values` è il contenuto corrente del campo, così com'è nel file: "già
    scritto" vale solo se contiene del TESTO — `_first`, la stessa regola
    con cui `read_coverage` decide cosa manca. Un frame o una chiave
    presenti ma vuoti sono la versione taggata del nulla: la copertura li
    dà per mancanti, e se qui contassero come scritti il brano tornerebbe
    in coda a ogni giro senza mai ricevere il genere (successo davvero: un
    TCON con text=[] teneva un file in coda per sempre).
    """
    return _first(values) is not None and not settings.overwrite


def _write_vorbis(filepath: Path, values: TagValues, settings: TagSettings) -> list[str]:
    import mutagen

    audio = mutagen.File(filepath)
    written = []
    if values.genre and not _skip_genre(audio.get("GENRE"), settings):
        audio["GENRE"] = values.genre
        written.append(f"GENRE={values.genre}")
        if values.genre_confidence:
            audio["ESSENTIA_GENRE"] = values.genre_confidence
    if values.mood and not _skip_genre(audio.get("MOOD"), settings):
        audio["MOOD"] = values.mood
        written.append(f"MOOD={values.mood}")
        if values.mood_confidence:
            audio["ESSENTIA_MOOD"] = values.mood_confidence
        # Il campo che djay Pro mostra davvero.
        audio["COMMENT"] = values.comment
        written.append(f"COMMENT={values.comment}")
    if written:
        audio.save()
    return written


def _write_id3(filepath: Path, values: TagValues, settings: TagSettings) -> list[str]:
    from mutagen.id3 import COMM, ID3, TCON, ID3NoHeaderError

    try:
        tags = ID3(filepath)
    except ID3NoHeaderError:
        tags = ID3()
    written = []
    if values.genre and not _skip_genre(
            [f.text for f in tags.getall("TCON")], settings):
        tags.delall("TCON")
        tags.add(TCON(encoding=3, text=values.genre))
        written.append(f"TCON={values.genre}")
        if values.genre_confidence:
            tags.setall("COMM:Essentia Genre:eng", [COMM(
                encoding=3, lang="eng", desc="Essentia Genre",
                text=values.genre_confidence)])
    if values.mood:
        tags.setall("COMM:Essentia Mood:eng", [COMM(
            encoding=3, lang="eng", desc="Essentia Mood", text=values.mood)])
        written.append(f"COMM(Essentia Mood)={values.mood}")
        if values.mood_confidence:
            tags.setall("COMM:Essentia Mood Confidence:eng", [COMM(
                encoding=3, lang="eng", desc="Essentia Mood Confidence",
                text=values.mood_confidence)])
        # I frame COMM sono indicizzati per (lingua, descrizione): quello qui
        # sopra è invisibile a chi mostra solo il commento predefinito, djay
        # Pro compreso. Questo lo rende visibile lì.
        tags.setall("COMM::eng", [COMM(
            encoding=3, lang="eng", desc="", text=values.comment)])
        written.append(f"COMM(default)={values.comment}")
    if written:
        tags.save(filepath)
    return written


def _write_mp4(filepath: Path, values: TagValues, settings: TagSettings) -> list[str]:
    from mutagen.mp4 import MP4, AtomDataType, MP4FreeForm

    audio = MP4(filepath)
    if audio.tags is None:
        audio.add_tags()
    freeform = lambda s: [MP4FreeForm(s.encode("utf-8"), dataformat=AtomDataType.UTF8)]
    written = []
    if values.genre and not _skip_genre(audio.tags.get("\xa9gen"), settings):
        audio["\xa9gen"] = [values.genre]
        written.append(f"genre={values.genre}")
        if values.genre_confidence:
            audio["----:com.apple.iTunes:ESSENTIA_GENRE"] = freeform(values.genre_confidence)
    if values.mood:
        audio["----:com.apple.iTunes:MOOD"] = freeform(values.mood)
        written.append(f"MOOD={values.mood}")
        if values.mood_confidence:
            audio["----:com.apple.iTunes:ESSENTIA_MOOD"] = freeform(values.mood_confidence)
        audio["\xa9cmt"] = [values.comment]
        written.append(f"comment={values.comment}")
    if written:
        audio.save()
    return written


def _write_asf(filepath: Path, values: TagValues, settings: TagSettings) -> list[str]:
    from mutagen.asf import ASF

    audio = ASF(filepath)
    written = []
    if values.genre and not _skip_genre(audio.get("WM/Genre"), settings):
        audio["WM/Genre"] = values.genre
        written.append(f"WM/Genre={values.genre}")
        if values.genre_confidence:
            audio["ESSENTIA_GENRE"] = values.genre_confidence
    if values.mood:
        audio["WM/Mood"] = values.mood
        written.append(f"WM/Mood={values.mood}")
        if values.mood_confidence:
            audio["ESSENTIA_MOOD"] = values.mood_confidence
        audio["Description"] = values.comment
        written.append(f"Description={values.comment}")
    if written:
        audio.save()
    return written


def _write_apev2(filepath: Path, values: TagValues, settings: TagSettings) -> list[str]:
    import mutagen

    audio = mutagen.File(filepath)
    if audio is None:
        raise ValueError(f"Impossibile aprire per il tagging: {filepath}")
    if audio.tags is None:
        audio.add_tags()
    written = []
    if values.genre and not _skip_genre(audio.tags.get("Genre"), settings):
        audio.tags["Genre"] = values.genre
        written.append(f"Genre={values.genre}")
        if values.genre_confidence:
            audio.tags["Essentia Genre"] = values.genre_confidence
    if values.mood:
        audio.tags["Mood"] = values.mood
        written.append(f"Mood={values.mood}")
        if values.mood_confidence:
            audio.tags["Essentia Mood"] = values.mood_confidence
        audio.tags["Comment"] = values.comment
        written.append(f"Comment={values.mood}")
    if written:
        audio.save()
    return written


# --------------------------------------------------------------------------
# L'analizzatore vero e proprio (l'unica parte che richiede Essentia)
# --------------------------------------------------------------------------

class EssentiaAnalyzer:
    """Carica i modelli una volta sola e analizza i brani.

    I modelli ci mettono qualche secondo a caricarsi e occupano memoria, per
    cui vengono caricati alla prima analisi e non nel costruttore: così si
    può creare l'oggetto in una UI senza pagare nulla finché non si parte.
    """

    def __init__(self, settings: TagSettings, model_dir: Path = MODEL_DIR) -> None:
        self.settings = settings
        self.model_dir = Path(model_dir)
        self._loaded = False
        self._embedding = None
        self._genre = None
        self._mood = None
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
        from essentia.standard import TensorflowPredict2D, TensorflowPredictEffnetDiscogs

        self._embedding = TensorflowPredictEffnetDiscogs(
            graphFilename=str(self.model_dir / EMBEDDING_MODEL),
            output="PartitionedCall:1")
        if self.settings.genres:
            self._genre = TensorflowPredict2D(
                graphFilename=str(self.model_dir / GENRE_MODEL),
                input="serving_default_model_Placeholder", output="PartitionedCall")
            self.genre_labels = _labels(self.model_dir / GENRE_METADATA)
        if self.settings.moods:
            self._mood = TensorflowPredict2D(
                graphFilename=str(self.model_dir / MOOD_MODEL),
                input="model/Placeholder", output="model/Sigmoid")
            self.mood_labels = _labels(self.model_dir / MOOD_METADATA)
        self._loaded = True

    def analyze(self, filepath: Path) -> TrackTags:
        import numpy as np
        from essentia.standard import MonoLoader

        self.load()
        audio = MonoLoader(filename=str(filepath), sampleRate=SAMPLE_RATE,
                           resampleQuality=1)()
        if self.settings.max_seconds:
            audio = audio[:int(self.settings.max_seconds * SAMPLE_RATE)]
        embeddings = self._embedding(audio)

        result = TrackTags()
        if self._genre is not None:
            activations = np.mean(self._genre(embeddings), axis=0)
            result.genres = select_genres(activations, self.genre_labels, self.settings)
        if self._mood is not None:
            activations = np.mean(self._mood(embeddings), axis=0)
            result.moods = select_moods(activations, self.mood_labels, self.settings)
        return result


def _labels(metadata_file: Path) -> list[str]:
    with metadata_file.open(encoding="utf-8") as fh:
        return json.load(fh)["classes"]


# --------------------------------------------------------------------------
# Cosa è già stato scritto: copertura dei tag
# --------------------------------------------------------------------------

@dataclass
class TagCoverage:
    """Quali tag ha già un file. Letto dal FILE, non dal registro di ciò che
    è stato tentato: sono due cose diverse, e l'unica che conta davvero è
    cosa c'è dentro al brano adesso."""
    path: Path
    genre: str | None = None
    comment: str | None = None
    error: str | None = None

    @property
    def has_genre(self) -> bool:
        return bool(self.genre)

    @property
    def has_comment(self) -> bool:
        return bool(self.comment)

    @property
    def complete(self) -> bool:
        return self.has_genre and self.has_comment


def _first(values) -> str | None:
    """Primo valore non vuoto, appiattendo le liste: mutagen restituisce ora
    una stringa, ora una lista, ora un frame che contiene una lista."""
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        values = [values]
    for v in values:
        if isinstance(v, (list, tuple)):
            found = _first(v)
            if found:
                return found
            continue
        text = str(v).strip()
        if text:
            return text
    return None


def read_coverage(path: Path) -> TagCoverage:
    """Legge genere e commento predefinito da un file, qualunque sia il
    formato. Il commento è quello a descrizione VUOTA: è l'unico che djay
    Pro mostra, ed è dove finiscono i mood."""
    import mutagen

    try:
        audio = mutagen.File(path)
    except Exception as e:
        return TagCoverage(path=path, error=f"{type(e).__name__}")
    if audio is None:
        return TagCoverage(path=path, error="nessuna intestazione audio")

    tags = audio.tags
    if tags is None:
        return TagCoverage(path=path)

    # Si sceglie in base al TIPO dei tag, non provando le chiavi a tentoni:
    # i tag Vorbis sollevano ValueError se interrogati con una chiave non
    # ASCII come quelle di MP4.
    genre = comment = None
    if hasattr(tags, "getall"):                     # ID3: mp3, aiff, wav, dsf
        genre = _first([f.text for f in tags.getall("TCON")])
        comment = _first([f.text for f in tags.getall("COMM") if f.desc == ""])
    elif type(tags).__name__ == "MP4Tags":
        genre = _first(tags.get("\xa9gen"))
        comment = _first(tags.get("\xa9cmt"))
    else:                                           # Vorbis, APEv2, ASF
        for key in ("GENRE", "genre", "WM/Genre", "Genre"):
            genre = genre or _first(tags.get(key))
        for key in ("COMMENT", "comment", "Description", "Comment"):
            comment = comment or _first(tags.get(key))
    return TagCoverage(path=path, genre=genre, comment=comment)


@dataclass
class CoverageReport:
    """Copertura dei tag su un insieme di file."""
    items: list[TagCoverage] = field(default_factory=list)

    @property
    def readable(self) -> list[TagCoverage]:
        return [c for c in self.items if c.error is None]

    @property
    def unreadable(self) -> list[TagCoverage]:
        return [c for c in self.items if c.error is not None]

    def missing(self, genre: bool = True, comment: bool = True,
                require_both: bool = False) -> list[TagCoverage]:
        """Chi ha bisogno di essere (ri)analizzato.

        `require_both` chiede i file a cui mancano ENTRAMBI i tag, invece di
        quelli a cui ne manca almeno uno.
        """
        out = []
        for c in self.readable:
            lacks = []
            if genre:
                lacks.append(not c.has_genre)
            if comment:
                lacks.append(not c.has_comment)
            if not lacks:
                continue
            if all(lacks) if require_both else any(lacks):
                out.append(c)
        return out


def scan_coverage(files, progress=None) -> CoverageReport:
    """Legge la copertura dei tag di ogni file. Circa 12 ms l'uno."""
    report = CoverageReport()
    total = len(files)
    for i, path in enumerate(files, 1):
        report.items.append(read_coverage(Path(path)))
        if progress is not None and (i % 100 == 0 or i == total):
            progress(i, total)
    return report


# --- Analisi in parallelo ---------------------------------------------------

def default_workers() -> int:
    """Quanti processi conviene usare qui.

    Metà dei core. Misurato su M5 (10 core) con 24 analisi: 1 processo 8,2 s
    a brano, 2 -> 5,7 s, 3 -> 5,0 s, 5 -> 4,1 s, 8 -> 3,7 s. Il guadagno si
    ferma perché anche con 8 processi i core davvero occupati restano 3,4:
    TensorFlow è già multi-thread al suo interno e i processi si contendono
    la stessa banda di memoria. Da 5 a 8 si guadagna l'8% e si spendono 2,6 GB
    in più — su una macchina da 16 GB condivisi col browser non vale.
    """
    return max(1, (os.cpu_count() or 2) // 2)


# Ogni processo carica i modelli una volta sola (2 s) e poi li riusa per tutta
# la sua parte di coda: sono ~1,3 GB a processo, il costo vero della scelta.
_POOL_ANALYZER: "EssentiaAnalyzer | None" = None


def _pool_init(settings: TagSettings, model_dir: Path) -> None:
    global _POOL_ANALYZER
    _POOL_ANALYZER = EssentiaAnalyzer(settings, model_dir)
    _POOL_ANALYZER.load()


def _pool_analyze(path: Path) -> tuple[Path, TrackTags | None, str | None]:
    """L'errore torna come valore, non come eccezione.

    Un'eccezione che attraversa il pool fa fallire l'intera chiamata: un
    brano illeggibile su cinquemila butterebbe via tutto il resto.
    """
    try:
        return path, _POOL_ANALYZER.analyze(path), None
    except Exception as e:                              # noqa: BLE001
        return path, None, f"{type(e).__name__}: {e}"


def analyze_many(paths, settings: TagSettings, workers: int = 1,
                 model_dir: Path = MODEL_DIR):
    """Analizza `paths`, restituendo `(percorso, tag, errore)` uno per volta.

    Con `workers` a 1 resta tutto nel processo corrente: sotto una manciata
    di brani i 2 s di caricamento modelli per processo costerebbero più di
    quanto rendano.
    """
    paths = list(paths)
    if workers <= 1 or len(paths) <= 1:
        analyzer = EssentiaAnalyzer(settings, model_dir)
        for path in paths:
            try:
                yield path, analyzer.analyze(path), None
            except Exception as e:                      # noqa: BLE001
                yield path, None, f"{type(e).__name__}: {e}"
        return

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    # "spawn" e non "fork": la fork di un processo che ha già caricato
    # TensorFlow duplica uno stato di thread che nel figlio non riparte.
    with ProcessPoolExecutor(
        max_workers=min(workers, len(paths)),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_pool_init, initargs=(settings, model_dir),
    ) as pool:
        yield from pool.map(_pool_analyze, paths)
