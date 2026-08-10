#!/usr/bin/env python3
"""Entry point 1 — CLI batch.

Analizza ricorsivamente una cartella di mp3, produce un report per libreria
(CSV o JSON) con path/genere/BPM/vibe/phrase boundary, e opzionalmente
organizza i file in `Genere/Vibe` (copia, non sposta, senza sovrascrivere).

    poetry run python cli.py SORGENTE [opzioni]

Esempi:
    # solo report a video
    poetry run python cli.py ~/Music/dj

    # report su file + dry-run dell'organizzazione
    poetry run python cli.py ~/Music/dj --dest ~/Music/master --report report.csv --dry-run

    # organizza davvero (copia in Genere/Vibe)
    poetry run python cli.py ~/Music/dj --dest ~/Music/master

I file mp3 richiedono ffmpeg installato a livello di sistema (brew install ffmpeg).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from analysis.engine import (
    analyze_library,
    build_organize_plan,
    discover_tracks,
    organize,
)
from analysis.models import TrackAnalysis

REPORT_COLUMNS = ["path", "genre", "bpm", "vibe", "boundaries", "error"]


def _write_report(tracks: list[TrackAnalysis], out: Path, fmt: str) -> None:
    if fmt == "json":
        payload = [
            {**t.to_row(), "boundaries_detail": [b.to_dict() for b in t.boundaries]}
            for t in tracks
        ]
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    else:  # csv
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
            writer.writeheader()
            for t in tracks:
                writer.writerow(t.to_row())


def _progress(i: int, total: int, path: Path) -> None:
    print(f"  [{i}/{total}] {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analizza e classifica una libreria mp3 (genere/vibe + phrase boundary)."
    )
    parser.add_argument("source", type=Path, help="Cartella sorgente con i file mp3")
    parser.add_argument("--dest", type=Path, default=None,
                        help="Cartella master per l'organizzazione in Genere/Vibe")
    parser.add_argument("--report", type=Path, default=None,
                        help="File di output del report (estensione .csv o .json)")
    parser.add_argument("--format", choices=["csv", "json"], default=None,
                        help="Formato del report (default: dedotto dall'estensione, altrimenti csv)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mostra il piano di organizzazione senza copiare i file")
    parser.add_argument("--no-cache", action="store_true",
                        help="Non usa la cache: rianalizza tutti i file")
    args = parser.parse_args()

    if not args.source.is_dir():
        sys.exit(f"Cartella sorgente non trovata: {args.source}")

    n = len(discover_tracks(args.source))
    if n == 0:
        sys.exit("Nessun file mp3 trovato nella cartella sorgente.")
    print(f"Trovati {n} file. Analisi in corso...\n")

    tracks = analyze_library(args.source, use_cache=not args.no_cache, progress=_progress)

    print("\nRisultati:")
    errors = 0
    for t in tracks:
        bpm = f"{t.bpm:.0f}" if t.bpm is not None else "N/D"
        nb = len(t.boundaries)
        print(f"  {t.path.name}  ->  {t.genre}/{t.vibe}  (BPM: {bpm}, boundary: {nb})")
        if t.error:
            errors += 1
            print(f"      [WARN] {t.error}")

    if args.report is not None:
        fmt = args.format or ("json" if args.report.suffix.lower() == ".json" else "csv")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        _write_report(tracks, args.report, fmt)
        print(f"\nReport ({fmt}) scritto in: {args.report}")

    if args.dest is not None:
        plan = build_organize_plan(tracks, args.dest)
        copied, skipped = organize(plan, dry_run=args.dry_run)
        if args.dry_run:
            print(f"\n[DRY RUN] {copied} file da copiare, {skipped} già presenti. "
                  "Rilancia senza --dry-run per eseguire.")
        else:
            print(f"\nOrganizzazione completata. Copiati: {copied}, saltati: {skipped}.")

    if errors:
        print(f"\n{errors} file con avvisi durante l'analisi (vedi [WARN] sopra).")


if __name__ == "__main__":
    main()
