"""Tagging di genere e mood con i modelli Essentia (TensorFlow).

Porta dentro Wavecut quello che faceva `tag_music.py` del progetto
Essentia-to-Metadata. Per ora c'è solo la parte di disponibilità: serve a far
degradare con grazia la sezione "Tag analysis" quando la dipendenza pesante
non c'è, esattamente come `vocals.available()` fa con Demucs.

`essentia-tensorflow` non è una dipendenza di base: pesa 424 MB (TensorFlow
è incluso nel pacchetto) e su PyPI esiste solo come pre-release, quindi va
chiesta esplicitamente:

    poetry install --with essentia
"""

from __future__ import annotations

import os
from pathlib import Path

MODEL_DIR = Path(os.path.expanduser("~/essentia_models"))

# Nome file -> a cosa serve, per poter dire all'utente cosa manca e perché.
MODELS = {
    "discogs-effnet-bs64-1.pb": "embedding (serve sempre)",
    "genre_discogs400-discogs-effnet-1.pb": "genere: 400 classi Discogs",
    "genre_discogs400-discogs-effnet-1.json": "genere: elenco delle classi",
    "mtg_jamendo_moodtheme-discogs-effnet-1.pb": "mood: MTG-Jamendo",
    "mtg_jamendo_moodtheme-discogs-effnet-1.json": "mood: elenco delle classi",
}


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
