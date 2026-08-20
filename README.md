# dj-library-tools

Local tool (macOS) to analyze an mp3/flac library and prepare DJ work in
**djay Pro**:

- automatic classification by **genre/vibe**,
- organization into `Genre/Vibe` folders,
- identification of suggested **phrase boundaries** for manually placing hot cues.

> **Note on cue points.** djay Pro stores cue points and loops in its own
> internal database, not in the file tags, and does not expose an import for
> external cues. So this tool **does not write hot cues into djay Pro**: it
> produces *suggested* timestamps that you confirm by ear in the review app and
> then place manually in djay Pro — or export them as **rekordbox XML** (see
> below) and use a third-party converter to reach djay Pro, Serato or Traktor.

## Architecture

A **shared analysis engine** (`analysis/`, pure Python module) imported by both
the batch CLI and the Streamlit app — no duplicated logic.

| Module | Responsibility |
| --- | --- |
| `analysis/tags.py` | read the genre tag via mutagen (ID3 for mp3, Vorbis comment for flac) |
| `analysis/audio_features.py` | audio loading (librosa) + BPM and RMS in a single pass |
| `analysis/vibe.py` | tempo buckets + percentile energy (two-pass) → vibe |
| `analysis/structure.py` | structural segmentation (Foote novelty over self-similarity) → phrase boundaries |
| `analysis/sections.py` | section classification (Intro/Build-up/Drop/Breakdown/Outro) from energy arc and bass presence |
| `analysis/vocals.py` | vocal detection via source separation (Demucs): sung regions + 🎤 flag per section |
| `analysis/waveform.py` | frequency-band colored waveform (djay Pro style) |
| `analysis/dj_export.py` | export section cues to rekordbox XML (the hub format for third-party DJ software converters) |
| `analysis/cache.py` | per-file cache (key = path, valid by mtime+size) |
| `analysis/engine.py` | orchestration: two-pass, cache, organize plan |
| `analysis/map_profile.py` | acoustic profile of a track: Discogs-EffNet embedding (1280-D) + multi-label genre/mood, over three 30 s windows |
| `analysis/map_projection.py` | UMAP projection of the embeddings to the 2D map |
| `analysis/map_store.py` | the map on disk: append-only metadata + embeddings, plus the projected X/Y |
| `analysis/mixing.py` | Camelot wheel, transition cost, path-drawn playlists, magic sort |
| `analysis/map_job.py` | the map build as a long, resumable background job |
| `cli.py` | entry point 1 — batch CLI |
| `app.py` | entry point 2 — Wavecut review app (Streamlit) |
| `map_cli.py` | build the map in the background (long job, resumable) |

**Audio loading.** Each file is loaded **once** (22050 Hz, mono); BPM/RMS are
computed on the first 60 s of that signal, segmentation over the whole track.

**Vibe.** BPM → tempo bucket (`Warm-Up`/`Groove`/`Peak-Time`/
`High-Energy-Tempo`); RMS → 33/66 percentiles relative to the library →
`Low`/`Mid`/`High`. Final vibe e.g. `Peak-Time-High`. Buckets and percentiles
are configured in `analysis/vibe.py`.

## Setup

**mp3** files require **ffmpeg** at the system level (librosa decodes them via
audioread); **flac** files are read natively by soundfile and don't need it:

```bash
brew install ffmpeg
```

Python dependencies with Poetry (Python ^3.11):

```bash
poetry install
```

## Usage

### Batch CLI

```bash
# report to stdout only
poetry run python cli.py ~/Music/dj

# report to file + dry-run of the organization (copy, not move)
poetry run python cli.py ~/Music/dj --dest ~/Music/master --report report.csv --dry-run

# actually organize into Genre/Vibe (without overwriting existing files)
poetry run python cli.py ~/Music/dj --dest ~/Music/master
```

The report (CSV or JSON) contains, per track: path, genre, BPM, vibe and the
suggested phrase-boundary timestamps. The cache avoids re-analyzing files that
were already processed: use `--no-cache` to force re-analysis.

### Streamlit app — Wavecut (review)

```bash
poetry run streamlit run app.py
```

Pick a **single track** (path field or “Browse…”, the files are already on
disk), run the analysis. Each analysis saves a **sidecar** `<name>_analysis.json`
next to the track: on a later reload, if that file exists and *Force analysis if
exists* is off, the results are **loaded from it** without re-analyzing (no
Demucs). Then review the **frequency-band colored waveform** (djay Pro style:
red = lows, green = mids, blue = highs) with the **section tags** overlaid
(Intro/Build-up/Drop/Breakdown/Outro). For each tag a **slider** moves the
section start and a menu changes its label; the waveform and the downloadable
report update live. A synced player lets you scrub the waveform and listen from
any point to confirm by ear.

> Section classification is **heuristic** (rules on energy and bass, thresholds
> in `analysis/sections.py`): meant as a starting point to correct by ear, not
> as ground truth. Ambiguous sections are labeled `Groove`. Consecutive sections
> of the same type are **merged**: each tag marks a **phrase change**, so you can
> anticipate in djay Pro what's coming next.

**Vocal detection** uses Demucs (source separation) to isolate the vocal stem:
the **sung regions** appear as pink bands on the waveform (the parts not to
overlap with other vocals while mixing) and sections with vocals get the 🎤 flag.
It is accurate but **heavy**: it downloads a model on first run and runs a neural
network on each track. It is optional — skip it with `--no-vocals` (CLI) or by
unchecking the box in the app; if Demucs is not installed the flag stays manual.

### Streamlit app — Map

The library as one picture: every track is a point, and points that sound
alike sit together. It answers the question a folder cannot — *what do I play
next, out of ninety thousand tracks?*

**How a track becomes a point.** The Essentia Discogs-EffNet model is read at
two places at once: the penultimate layer gives a **1280-dimension embedding**
(the acoustic identity of the track, before it is flattened into genre names),
and the classification heads give **several genres and moods with their
confidence** — a track can be Minimal *and* Deep House, and both are kept.
Only **three 30-second windows** are analyzed, at 25%, 50% and 75% of the
track, and their results are averaged: about 8 seconds per track instead of
half a minute, which on a whole library is the difference between hours and
days. Before inference each window is brought to **-14 LUFS** (EBU R128) so
that a loud master does not read as a different genre. BPM and key come from
the file tags when they are there, and are measured otherwise; the key is
converted to its **Camelot** code.

**The map itself** is a UMAP projection of the embeddings to two dimensions —
UMAP rather than t-SNE because the distance *between* clusters keeps its
meaning, and that distance is exactly what a line drawn across the map uses.

**What you do with it**

- **click a point** (or pick a track by name) to make it the seed, and see
  what mixes out of it, ranked by the transition cost
  `w1·distance on the map + w2·BPM gap + w3·Camelot distance`, with the three
  weights on sliders;
- **draw a lasso**, which does one of two things depending on the shape: a
  **line** through the clusters takes the tracks it passes near, in the order
  it meets them — a way to plan an arc (start in ambient, cross deep house,
  peak in tech house) by drawing it — while a **loop** that comes back where
  it started takes everything it encloses, like the box. The guess follows
  the shape and a radio lets you overrule it;
- **box-select a group** and let **magic sort** order it: the cheapest path
  that visits every track once (an open travelling-salesman problem, solved
  nearest-neighbour then 2-opt), so each track melts into the next;
- export the result as **M3U8** or **rekordbox XML**.

The **size of a point** carries a number you choose — BPM, groove (how
regular the onsets are) or energy (integrated loudness) — scaled between the
5th and 95th percentile so one outlier does not flatten everything. The
position already says how a track sounds; the diameter is room for a quantity
you can actually read, which is why the map stays in two dimensions: a third
axis would cost the lasso and the box (Plotly has neither in 3D) and would
have to be read by rotating the scene.

While the background job runs, the page shows its progress **live** (a
fragment that re-reads the job's state file every two seconds, without
redrawing the map) and can **pause**, **resume** or **stop** it — signals go
to the whole process group, so the parallel workers stop too — or open a
**Terminal** on `tail -f` of its log.

```bash
# build the map for a folder, then project it (hours on a whole library,
# resumable: stop it whenever, it picks up where it left off)
poetry run python map_cli.py "/Volumes/Crucial X9/DJSet" --project

# only recompute the projection, on a map that is already built
poetry run python map_cli.py --project-only

# the library moved to another disk: update the paths instead of
# re-analyzing 90,000 tracks from scratch
poetry run python map_cli.py --relocate "/Volumes/Old/DJSet" "/Volumes/New/DJSet"
```

A track is recognised by its absolute path, so moving the library to another
volume makes it a **different** library as far as the map is concerned —
hence `--relocate`, which rewrites the paths and leaves the embeddings and
the projection alone. It also takes the modification date from the file at
its new address when the size still matches, because copying without
preserving dates (plain `cp` does that) would otherwise make every track
look changed and send it back to the queue.

The map lives in `~/.cache/dj-library-tools/map/`: one JSON line and one
1280-float block per track (both append-only, which is what makes the job
interruptible), plus the X/Y of the projection.

> **On the confidence threshold.** The spec this section follows puts the
> multi-label threshold at 0.40. Measured on these models it is too high: the
> genre head is a softmax over 400 classes, and on an unmistakable track the
> top genre reaches 0.404 — at 0.40 almost the whole library would come back
> with a single label. The defaults are 0.15 for genres and 0.05 for moods
> (the same values the tagging already uses); both are settings.

## Exporting cues to other DJ software

Section tags can be exported as a **rekordbox XML** collection — the format
most DJ software converters accept as input:

```bash
# whole library, one collection.xml with all tracks' cues
poetry run python cli.py ~/Music/dj --rekordbox-xml collection.xml
```

In the app, a single track's **confirmed** sections (after you've adjusted the
sliders/labels) can be downloaded the same way from the "Export cues to
rekordbox XML" button.

- **rekordbox**: import the XML directly (File → Import Collection).
- **Serato / Traktor / djay Pro**: rekordbox XML is the format that
  third-party converters — [DJ Conversion Utility](https://atgr-production-team.sellfy.store/p/emuy/),
  [MIXO](https://www.mixo.dj/), [Lexicon](https://www.lexicondj.com/) — accept
  to produce a library those apps can import. This tool does not talk to those
  converters directly; you run them yourself on the exported XML.

There is no verified way to write cues straight into Serato's file tags or into
djay Pro's own database from here — see the note on cue points above.

## Tests

```bash
poetry run pytest
```

The tests cover the pure logic (buckets/percentiles, cache, novelty, sections,
vocal regions). The end-to-end audio analysis should be tried on a **small
subset** of files before running on the whole library.
