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
| `cli.py` | entry point 1 — batch CLI |
| `app.py` | entry point 2 — Wavecut review app (Streamlit) |

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
