"""Lettura dei tag ID3 (genere) via mutagen."""

from __future__ import annotations

from pathlib import Path

from mutagen.easyid3 import EasyID3

UNCLASSIFIED = "Unclassified"


def get_genre(filepath: Path) -> str:
    """Legge il tag genere ID3.

    Ritorna 'Unclassified' se il tag è assente o il file è illeggibile.
    Prende il primo genere e normalizza in Title Case, come nella prima
    versione dello strumento.
    """
    try:
        tags = EasyID3(filepath)
        genres = tags.get("genre")
        if genres:
            raw = genres[0].split(",")[0].split("/")[0].strip()
            return raw.title() if raw else UNCLASSIFIED
    except Exception:
        pass
    return UNCLASSIFIED
