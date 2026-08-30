"""Lettura del tag genere via mutagen, indipendente dal formato.

`mutagen.File(..., easy=True)` gestisce in modo uniforme sia gli ID3 degli mp3
sia i Vorbis comment dei flac, esponendo il genere sotto la chiave 'genre'.
"""

from __future__ import annotations

from pathlib import Path

from mutagen import File as MutagenFile

UNCLASSIFIED = "Unclassified"


def get_genre(filepath: Path) -> str:
    """Legge il tag genere (ID3 per mp3, Vorbis comment per flac).

    Ritorna 'Unclassified' se il tag è assente o il file è illeggibile.
    Prende il primo genere e normalizza in Title Case, come nella prima
    versione dello strumento.
    """
    try:
        audio = MutagenFile(filepath, easy=True)
        if audio is not None:
            genres = audio.get("genre")
            if genres:
                raw = genres[0].split(",")[0].split("/")[0].strip()
                return raw.title() if raw else UNCLASSIFIED
    except Exception:
        pass
    return UNCLASSIFIED
