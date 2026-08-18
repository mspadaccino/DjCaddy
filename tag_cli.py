#!/usr/bin/env python3
"""Entry point 3 — il tagging come job lungo, fuori dall'app.

Analizza una cartella con i modelli Essentia e SCRIVE genere e mood nei tag,
senza dipendere da una scheda del browser: a qualche secondo per brano una
libreria intera sono giorni.

    poetry run python tag_cli.py CARTELLA [opzioni]

Esempi:
    # tutto quello che manca, con il parallelismo di default
    poetry run python tag_cli.py "/Volumes/Crucial X9/DJSet"

    # una prova da cinquanta brani prima di lanciarlo sul serio
    poetry run python tag_cli.py ~/Music --limit 50

    # staccato dal terminale: si chiude tutto e lui continua
    nohup poetry run python tag_cli.py ~/Music > tagjob.log 2>&1 &

Si può fermare quando si vuole: i tag scritti restano scritti e la volta
dopo riparte da dove era, saltando quelli già fatti.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from analysis.essentia_tags import (
    GENRE_FORMATS,
    TagSettings,
    available,
    default_workers,
    missing_models,
)
from analysis.tag_job import DEFAULT_STATE_FILE, run_job


def _human(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def _report(state) -> None:
    fine = f" · ~{_human(state.eta_seconds)} alla fine" if state.done > 2 else ""
    sys.stdout.write(
        f"\r  {state.done:,}/{state.total:,} · scritti {state.written:,} "
        f"· saltati {state.skipped:,} · falliti {state.failed:,}{fine}"
        f" · {state.current[:40]:40s}")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrive genere e mood nei tag, in background.")
    parser.add_argument("folder", type=Path, help="Cartella da analizzare (ricorsiva)")
    parser.add_argument("--workers", type=int, default=default_workers(),
                        help="Analisi in parallelo (default: metà dei core)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Fermati dopo N brani (0 = tutti)")
    parser.add_argument("--all", action="store_true",
                        help="Anche i brani che i tag già ce l'hanno")
    parser.add_argument("--ignore-progress", action="store_true",
                        help="Non saltare quelli già nel registro")
    parser.add_argument("--overwrite", action="store_true",
                        help="Riscrivi i tag esistenti invece di rispettarli")
    parser.add_argument("--no-genres", action="store_true")
    parser.add_argument("--no-moods", action="store_true")
    parser.add_argument("--genre-format", choices=GENRE_FORMATS, default="parent_child")
    parser.add_argument("--confidence-in-comment", action="store_true",
                        help="Percentuali anche nel commento che djay mostra")
    parser.add_argument("--max-seconds", type=int, default=300,
                        help="Secondi di audio analizzati (0 = brano intero)")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE,
                        help="Dove scrivere lo stato leggibile dall'app")
    args = parser.parse_args()

    if not args.folder.is_dir():
        parser.error(f"Non è una cartella: {args.folder}")
    if not available():
        parser.error("essentia non è importabile in questo ambiente")
    if missing_models():
        parser.error(f"Modelli mancanti: {', '.join(missing_models())}")

    settings = TagSettings(
        genres=not args.no_genres, moods=not args.no_moods,
        genre_format=args.genre_format, overwrite=args.overwrite,
        confidence_in_comment=args.confidence_in_comment,
        max_seconds=args.max_seconds,
    )

    print(f"Cartella: {args.folder}")
    print(f"Parallelismo: {args.workers} · stato in {args.state_file}")
    print("Costruisco la coda…", flush=True)

    t0 = time.time()
    state = run_job(
        args.folder, settings, workers=args.workers,
        only_missing=not args.all, skip_done=not args.ignore_progress,
        state_file=args.state_file, limit=args.limit, on_progress=_report,
    )

    print(f"\n\nFinito in {_human(time.time() - t0)}.")
    print(f"  scritti {state.written:,} · saltati {state.skipped:,} "
          f"· falliti {state.failed:,} su {state.total:,}")
    if state.errors:
        print(f"\n  primi errori ({len(state.errors)} tenuti):")
        for e in state.errors[:5]:
            print(f"    {Path(e['file']).name[:60]} — {e['error'][:70]}")
    sys.exit(1 if state.failed and not state.written else 0)


if __name__ == "__main__":
    main()
