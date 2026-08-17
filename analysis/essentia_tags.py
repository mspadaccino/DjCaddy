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
    mood_threshold: float = 0.005      # attivazione minima (0-1)
    max_moods: int = 5                 # quanti mood tenere in tutto
    moods_in_tag: int = 3              # quanti finiscono nel tag/commento
    confidence_tags: bool = True
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


def _skip_genre(existing: bool, settings: TagSettings) -> bool:
    return existing and not settings.overwrite


def _write_vorbis(filepath: Path, values: TagValues, settings: TagSettings) -> list[str]:
    import mutagen

    audio = mutagen.File(filepath)
    written = []
    if values.genre and not _skip_genre("GENRE" in audio, settings):
        audio["GENRE"] = values.genre
        written.append(f"GENRE={values.genre}")
        if values.genre_confidence:
            audio["ESSENTIA_GENRE"] = values.genre_confidence
    if values.mood and not _skip_genre("MOOD" in audio, settings):
        audio["MOOD"] = values.mood
        written.append(f"MOOD={values.mood}")
        if values.mood_confidence:
            audio["ESSENTIA_MOOD"] = values.mood_confidence
        # Il campo che djay Pro mostra davvero.
        audio["COMMENT"] = values.mood
        written.append(f"COMMENT={values.mood}")
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
    if values.genre and not _skip_genre(bool(tags.getall("TCON")), settings):
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
        # I frame COMM sono indicizzati per (lingua, descrizione): quello qui
        # sopra è invisibile a chi mostra solo il commento predefinito, djay
        # Pro compreso. Questo lo rende visibile lì.
        tags.setall("COMM::eng", [COMM(
            encoding=3, lang="eng", desc="", text=values.mood)])
        written.append(f"COMM(default)={values.mood}")
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
    if values.genre and not _skip_genre("\xa9gen" in audio.tags, settings):
        audio["\xa9gen"] = [values.genre]
        written.append(f"genre={values.genre}")
        if values.genre_confidence:
            audio["----:com.apple.iTunes:ESSENTIA_GENRE"] = freeform(values.genre_confidence)
    if values.mood:
        audio["----:com.apple.iTunes:MOOD"] = freeform(values.mood)
        written.append(f"MOOD={values.mood}")
        if values.mood_confidence:
            audio["----:com.apple.iTunes:ESSENTIA_MOOD"] = freeform(values.mood_confidence)
        audio["\xa9cmt"] = [values.mood]
        written.append(f"comment={values.mood}")
    if written:
        audio.save()
    return written


def _write_asf(filepath: Path, values: TagValues, settings: TagSettings) -> list[str]:
    from mutagen.asf import ASF

    audio = ASF(filepath)
    written = []
    if values.genre and not _skip_genre("WM/Genre" in audio, settings):
        audio["WM/Genre"] = values.genre
        written.append(f"WM/Genre={values.genre}")
        if values.genre_confidence:
            audio["ESSENTIA_GENRE"] = values.genre_confidence
    if values.mood:
        audio["WM/Mood"] = values.mood
        written.append(f"WM/Mood={values.mood}")
        if values.mood_confidence:
            audio["ESSENTIA_MOOD"] = values.mood_confidence
        audio["Description"] = values.mood
        written.append(f"Description={values.mood}")
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
    if values.genre and not _skip_genre("Genre" in audio.tags, settings):
        audio.tags["Genre"] = values.genre
        written.append(f"Genre={values.genre}")
        if values.genre_confidence:
            audio.tags["Essentia Genre"] = values.genre_confidence
    if values.mood:
        audio.tags["Mood"] = values.mood
        written.append(f"Mood={values.mood}")
        if values.mood_confidence:
            audio.tags["Essentia Mood"] = values.mood_confidence
        audio.tags["Comment"] = values.mood
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
