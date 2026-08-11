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

from analysis.dj_export import read_title_artist, section_cues, write_rekordbox_xml
from analysis.engine import (
    analyze_library,
    build_organize_plan,
    discover_tracks,
    organize,
)
from analysis.models import TrackAnalysis

REPORT_COLUMNS = ["path", "genre", "bpm", "vibe", "sections", "boundaries", "error"]


def _write_report(tracks: list[TrackAnalysis], out: Path, fmt: str) -> None:
    if fmt == "json":
        payload = [
            {
                **t.to_row(),
                "sections_detail": [s.to_dict() for s in t.sections],
                "boundaries_detail": [b.to_dict() for b in t.boundaries],
            }
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
        description="Analyze and classify an mp3 library (genre/vibe + phrase boundaries)."
    )
    parser.add_argument("source", type=Path, help="Source folder with the mp3 files")
    parser.add_argument("--dest", type=Path, default=None,
                        help="Master folder to organize into Genre/Vibe")
    parser.add_argument("--report", type=Path, default=None,
                        help="Report output file (.csv or .json extension)")
    parser.add_argument("--format", choices=["csv", "json"], default=None,
                        help="Report format (default: inferred from extension, otherwise csv)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show the organize plan without copying files")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore the cache: re-analyze every file")
    parser.add_argument("--no-vocals", action="store_true",
                        help="Skip vocal detection (Demucs), much faster")
    parser.add_argument("--rekordbox-xml", type=Path, default=None,
                        help="Write a rekordbox-compatible XML collection with section "
                             "tags as cue points (import into rekordbox directly, or "
                             "convert to Serato/Traktor/djay Pro with a third-party tool "
                             "such as DJ Conversion Utility or MIXO)")
    args = parser.parse_args()

    if not args.source.is_dir():
        sys.exit(f"Source folder not found: {args.source}")

    n = len(discover_tracks(args.source))
    if n == 0:
        sys.exit("No audio files (mp3/flac) found in the source folder.")
    print(f"Found {n} files. Analyzing...\n")

    tracks = analyze_library(args.source, use_cache=not args.no_cache,
                             progress=_progress, detect_vocals=not args.no_vocals)

    print("\nResults:")
    errors = 0
    for t in tracks:
        bpm = f"{t.bpm:.0f}" if t.bpm is not None else "N/A"
        secs = " | ".join(s.label for s in t.sections) or "-"
        print(f"  {t.path.name}  ->  {t.genre}/{t.vibe}  (BPM: {bpm})")
        print(f"      sections: {secs}")
        if t.error:
            errors += 1
            print(f"      [WARN] {t.error}")

    if args.report is not None:
        fmt = args.format or ("json" if args.report.suffix.lower() == ".json" else "csv")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        _write_report(tracks, args.report, fmt)
        print(f"\nReport ({fmt}) written to: {args.report}")

    if args.rekordbox_xml is not None:
        entries = []
        for t in tracks:
            title, artist = read_title_artist(t.path)
            entries.append({
                "path": t.path, "name": title, "artist": artist, "genre": t.genre,
                "bpm": t.bpm, "duration": t.duration, "cues": section_cues(t.sections),
            })
        args.rekordbox_xml.parent.mkdir(parents=True, exist_ok=True)
        write_rekordbox_xml(entries, args.rekordbox_xml)
        print(f"\nrekordbox XML written to: {args.rekordbox_xml}")
        print("Import it in rekordbox directly, or convert it to Serato/Traktor/djay Pro "
              "with a third-party tool (DJ Conversion Utility, MIXO, Lexicon).")

    if args.dest is not None:
        plan = build_organize_plan(tracks, args.dest)
        copied, skipped = organize(plan, dry_run=args.dry_run)
        if args.dry_run:
            print(f"\n[DRY RUN] {copied} files to copy, {skipped} already present. "
                  "Re-run without --dry-run to execute.")
        else:
            print(f"\nOrganize complete. Copied: {copied}, skipped: {skipped}.")

    if errors:
        print(f"\n{errors} files with warnings during analysis (see [WARN] above).")


if __name__ == "__main__":
    main()
