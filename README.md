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
| `analysis/map_profile.py` | acoustic profile of a track: Discogs-EffNet embedding (1280-D) feeding the genre/mood heads, over three 30 s windows; BPM and key from tags or Essentia; groove from onset regularity |
| `analysis/map_projection.py` | PCA to 64-D, then UMAP projection of the embeddings to the 2D map |
| `analysis/map_store.py` | the map on disk: `tracks.jsonl` + `embeddings.f32` appended, `coords.npy` rewritten; cosine nearest-neighbours on the raw embeddings |
| `analysis/mixing.py` | Camelot wheel, transition cost, signed tempo/key shifts, path-drawn playlists, magic sort |
| `analysis/graph_playlist.py` | the chain as a graph: tracks, links, layout on the board, and the roster of what comes next |
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

**How a track becomes a point.** Two stages, one inference. The Essentia
**Discogs-EffNet** model produces a **1280-dimension embedding** — the
acoustic identity of the track, before it is flattened into genre names — and
that same vector is then fed to two classification heads, one for **400
Discogs genres** and one for **moods**, which return several labels with
their confidence: a track can be Minimal *and* Deep House, and both are kept.
The embedding is not a by-product read off the side of a classifier; it is the
first model's output, and the heads are consumers of it. The frames are
computed once and read at both places.

Only **three 30-second windows** are analyzed, at 25%, 50% and 75% of the
track, and their frames are stacked and averaged — temporal average pooling,
one vector per track. About 8 seconds per track instead of half a minute,
which on a whole library is the difference between hours and days. Before
inference each window is brought to **-14 LUFS** (EBU R128) so that a loud
master does not read as a different genre.

**What the model does not decide.** BPM, key and groove never touch the
network. BPM and key are read from the **file's own tags** when they are there
(and a BPM outside 40–220 is refused), and measured with Essentia otherwise —
`RhythmExtractor2013` and `KeyExtractor`, on the middle window only, since
both are properties of the whole track and measuring them three times does not
improve them. The key is converted to its **Camelot** code. Groove is not a
model output either: it is `1 − (spread of the gaps between onsets)`, a
hand-computed statistic. It measures **rhythmic regularity** — a straight kick
scores high, a syncopated breakbeat low — which correlates with danceability
without being it.

**Where it is kept.** Three files in one folder, and the split follows how
they are written: `tracks.jsonl` (one JSON line per track, **appended**),
`embeddings.f32` (the raw float32 vectors end to end, **appended** — line *n*
of the first is block *n* of this one), and `coords.npy` (the two UMAP
columns, **rewritten whole** on each projection, because a projection is a
fact about the entire library rather than about one track). Appending instead
of rewriting is what makes the job interruptible: at 90,000 tracks the
embedding matrix is half a gigabyte.

**The map itself** is a UMAP projection of the embeddings to two dimensions —
UMAP rather than t-SNE because the distance *between* clusters keeps its
meaning, and that distance is exactly what a line drawn across the map uses.
A PCA to 64 dimensions runs first; it does not change the map (these
embeddings keep nearly all their variance there) but it removes most of
UMAP's neighbour search, which is almost all of its time.

**Two kinds of nearness, and they are not the same.** *Sounds like it* is the
cosine over the full **1280 dimensions** — the real one. *Mixes out of it*
ranks by the transition cost, which uses the **two projected coordinates**
plus tempo and key. The 2D map is a shadow of the embedding: convenient to
look at, and flattened. A track can sound close and mix badly, or the reverse,
and the two tabs under a seed exist to let you ask both questions.

**Genre and mood never enter proximity.** The transition cost takes exactly
three things — position on the map, BPM gap, Camelot distance — and no label.
Genres are a **filter**: they narrow the universe before the question is
asked, and say nothing about how close two tracks are once it has been. This
is deliberate: the whole point of the embedding is that it hears things a
genre name has already thrown away.

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
- **grow a set one track at a time** in the graph builder, below;
- export the result as **M3U8** or **rekordbox XML**.

Both the selection and the graph builder write to the same **playlist**,
which is why it has a section of its own below them rather than living inside
either.

#### The graph builder

Magic sort answers *put these in the best order*. This answers the other
question — *what comes next?* — one track at a time, which is how a set is
actually decided.

**Two tables give the orders, the board shows the result.** On the left the
chain as it stands; on the right the candidates that mix out of whichever
track you are standing on, ranked by the same transition cost. Tick one or
several, add them, and they go on **one behind the other** — ticking three
means "then these three", not three branches off the same track. Both tables
carry the same columns: BPM, key, groove, the folder the file came from, and
the **signed shift** against the previous track.

That shift is the thing the cost cannot tell you. A cost is a distance and
has no sign: from a track at 118 BPM, one at 122 and one at 114 score the
same. `+4 BPM · +1 wheel · +0.09 groove` says which way the set is moving,
warm for rising and cool for falling. It is deliberately **not** in the
ranking — a set climbs, holds and lets go, and sorting by direction would be
choosing which of the three on the DJ's behalf.

**The board is a picture, not a control.** Left to right the cards follow the
playlist order; how high a card sits is a measure you pick with a radio —
tempo, key or groove — so a set that climbs looks like a climb. Each scale is
fixed rather than stretched over the chain: the wheel for keys, the library's
deciles for groove, and for tempo the pitch fader's ±6% around where the
chain sits. Stretching a chain over its own range turns one BPM of drift into
half the board, and this cost proposes tracks at the same tempo — a chain of
eight here often spans about a single BPM. With fixed scales two chains can
be compared, and a flat row honestly means the measure does not move. Cards
can be dragged off the rule and stay off it; picking a measure again puts
everything back.

**Copies are one entry.** A track filed in four folders has the same tempo
and key in all four, so it has the same cost from anywhere and would take
four of the nine slots. They are gathered under one row marked `×4`, and
putting one down blocks the rest — a set should not take the same record
twice. Which copy is a real question, so the roster names them by folder and
lets you choose rather than picking for you.

The builder has **its own filters**, not the map's: a clickable **Camelot
wheel** (two rings, major outside and minor inside, the way the players draw
it, because harmonic mixing is a question about neighbours and a list of
twenty-four codes hides exactly the adjacency that matters), plus genres, BPM
and groove ranges. Tracks already on the board are never filtered away —
a filter is about what to propose next, not about breaking a chain someone
has built.

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
