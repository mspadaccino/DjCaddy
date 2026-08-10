"""Orchestrazione dell'analisi: cache-aware, two-pass, piano di organizzazione.

È il punto d'ingresso del motore condiviso usato sia dal CLI sia da Streamlit.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from .audio_features import ANALYSIS_DURATION_SECONDS, compute_bpm_rms, load_audio
from .cache import AnalysisCache, default_cache_path
from .models import TrackAnalysis
from .structure import detect_boundaries
from .tags import get_genre
from .vibe import bpm_to_tempo_bucket, build_energy_labeler, compute_vibe

AUDIO_EXTENSIONS = {".mp3", ".flac"}

ProgressFn = Callable[[int, int, Path], None]


def discover_tracks(source: Path) -> list[Path]:
    """Elenca ricorsivamente i file audio supportati nella cartella sorgente."""
    return sorted(
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def analyze_track(filepath: Path, cache: AnalysisCache | None = None) -> TrackAnalysis:
    """Analizza una singola traccia (feature per-file). Cache-aware.

    Ogni fase è isolata: un file corrotto o un tag mancante non fa cadere il
    batch, ma produce un record con `error` valorizzato.
    """
    if cache is not None:
        cached = cache.get(filepath)
        if cached is not None:
            return TrackAnalysis.from_dict(filepath, cached)

    genre = get_genre(filepath)
    bpm: float | None = None
    rms: float | None = None
    duration: float | None = None
    boundaries = []
    error: str | None = None

    try:
        y, sr = load_audio(filepath)
        duration = float(len(y) / sr) if sr else None
        try:
            window = y[: int(ANALYSIS_DURATION_SECONDS * sr)]
            bpm, rms = compute_bpm_rms(window, sr)
        except Exception as e:
            error = f"features: {e}"
        try:
            boundaries = detect_boundaries(y, sr)
        except Exception as e:
            error = f"{error}; structure: {e}" if error else f"structure: {e}"
    except Exception as e:
        error = f"load: {e}"

    track = TrackAnalysis(
        path=filepath, genre=genre, bpm=bpm, rms=rms, duration=duration,
        boundaries=boundaries, error=error,
    )
    if cache is not None:
        cache.put(filepath, track.to_dict())
    return track


def analyze_library(
    source: Path,
    use_cache: bool = True,
    cache: AnalysisCache | None = None,
    progress: ProgressFn | None = None,
) -> list[TrackAnalysis]:
    """Analizza l'intera libreria in due passaggi.

    Pass 1: feature per-file (cache-aware). Pass 2: soglie di energia
    library-relative e assegnazione della vibe a ogni traccia.
    """
    paths = discover_tracks(source)

    if use_cache and cache is None:
        cache = AnalysisCache.load(default_cache_path())
    active_cache = cache if use_cache else None

    # Pass 1
    tracks: list[TrackAnalysis] = []
    total = len(paths)
    for i, p in enumerate(paths, start=1):
        tracks.append(analyze_track(p, active_cache))
        if progress is not None:
            progress(i, total, p)
    if active_cache is not None:
        active_cache.save()

    # Pass 2: energia relativa alla libreria -> vibe
    valid_rms = [t.rms for t in tracks if t.rms is not None]
    labeler = build_energy_labeler(valid_rms)
    for t in tracks:
        t.vibe = compute_vibe(bpm_to_tempo_bucket(t.bpm), labeler(t.rms))

    return tracks


def build_organize_plan(tracks: list[TrackAnalysis], dest: Path) -> list[tuple[Path, Path]]:
    """Piano (sorgente -> destinazione) per copiare i file in `Genere/Vibe`."""
    plan: list[tuple[Path, Path]] = []
    for t in tracks:
        target_dir = dest / t.genre / (t.vibe or "Unknown")
        plan.append((t.path, target_dir / t.path.name))
    return plan


def organize(plan: list[tuple[Path, Path]], dry_run: bool = False) -> tuple[int, int]:
    """Copia (non sposta) i file secondo il piano, senza sovrascrivere.

    Ritorna (copiati, saltati). In dry-run non tocca il filesystem.
    """
    copied = skipped = 0
    for src, target in plan:
        if target.exists():
            skipped += 1
            continue
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
        copied += 1
    return copied, skipped
