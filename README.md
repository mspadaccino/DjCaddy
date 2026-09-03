# DjCaddy

A desktop app that reads your mp3/flac library and helps you build sets from
it. Everything runs locally: no account, no upload, no service.

It does four things, and they are the four tabs of the app:

| tab | what it is for |
|---|---|
| 🧭 **Navigator** | the whole library as one picture — points that sound alike sit together — with playlists drawn as paths across it |
| 🌊 **Cue Finder** | one track at a time: check the suggested phrase boundaries by ear, then write them as hot/memory cues **straight into the rekordbox library** |
| 🏷️ **Tag Maker** | genre and mood inferred by the Essentia models and written into the files' own tags |
| 📁 **File Analysis** | what is in a folder, what repeats, what is unreadable — and a reviewable quarantine plan |

Under all four there is one persistent player: waveform preview with the
played portion coloured, click anywhere on it to seek.

**This guide is also inside the app**: the 📖 button in the header opens it,
without the chapters below about the command line, the build and the code.

---

## Contents

- [Before you start](#before-you-start)
- [Getting started: from a folder of tracks to a set](#getting-started-from-a-folder-of-tracks-to-a-set)
- [Navigator](#navigator)
- [Cue Finder](#cue-finder)
- [Tag Maker](#tag-maker)
- [File Analysis](#file-analysis)
- [Reference: what every number means](#reference-what-every-number-means)
- [Command line](#command-line)
- [Getting cues into other DJ software](#getting-cues-into-other-dj-software)
- [Building the standalone app](#building-the-standalone-app)
- [How the code is laid out](#how-the-code-is-laid-out)

---

## Before you start

DjCaddy targets **macOS** and **Windows 11**. One asymmetry is structural:
**Essentia publishes no Windows wheels**, so *building* the map and *tagging*
run on macOS only. *Using* an existing map — playlists, set building, the
board, the player, the cues, File Analysis — works on both. In practice the
Mac is the analysis machine and Windows is a consumption machine; the pages
that need Essentia say so themselves rather than breaking.

### What you need

**ffmpeg**, at the system level. mp3 files are decoded through it; flac files
are read natively by soundfile and do not need it.

```bash
brew install ffmpeg
```

**Python 3.11 or newer**, and Poetry.

If you would rather not have any of that, skip to
[Building the standalone app](#building-the-standalone-app): the packaged
build carries its own Python, its own ffmpeg and its own models, and needs
nothing installed.

### Install

```bash
poetry install
```

That installs everything. The two optional groups are exits, not choices you
have to make up front:

| group | drop it with | what you lose |
|---|---|---|
| `essentia` | `--without essentia` | map building and tagging — use this if there is no wheel for your Python, and the rest stays alive |
| `rekordbox` | `--without rekordbox` | writing cues into rekordbox's database; the Cue Finder page says so instead of breaking |

The Essentia models are a separate download, and you need them before the map
or the tagging can run. They belong in `~/essentia_models`.

### Run it

```bash
poetry run python -m qt_app.main
```

```bash
poetry run pytest
```

The tests cover the pure logic (buckets and percentiles, cache, novelty,
sections, vocal regions, the stores and jobs), the shared figures — snapshot
tests, so the extracted builders must keep producing the same figure JSON —
and the Qt pages via pytest-qt.

---

## Getting started: from a folder of tracks to a set

Read this once and the rest of the guide becomes reference.

### 1 · Point it at your library and build the map

Open **Navigator**, then **⚙️ Map settings**, and give it the folder your
tracks live in. Building the map runs every track through a neural network,
so it is a **long background job** — hours on a whole library — but it is
*resumable*: stop it whenever you like and it picks up where it left off. The
page shows its progress live, and you can pause, resume or stop it without
closing the app.

You do not have to wait for all of it. The map is usable as soon as a few
thousand tracks are in and the projection has run once.

> Try the end-to-end analysis on a **small subset** of files before setting it
> loose on ninety thousand.

### 2 · Find your way around

Every track is a point, and points that sound alike sit together. **Click a
point** to make it the *seed*: the tables below fill with what sounds like it
and what mixes out of it. Narrow the picture with the filters — genres, BPM,
groove, energy — and use the point-size control to read a second quantity off
the same picture.

### 3 · Build a set

Two ways, and they answer different questions:

- *Put these in the best order.* **Box-select** a group and run **magic
  sort**, or **draw a lasso** across the clusters to plan an arc.
- *What comes next?* Go to **Build a set** and grow the chain one track at a
  time from the ranked candidates.

Both write into the same **playlist**, which is what the **board** draws as
cards. Export it as **M3U8** or **rekordbox XML**.

### 4 · Cue a track

Open **Cue Finder** on a track from the set, run the analysis, and check the
suggested phrase boundaries on the waveform by ear. Tick which of them become
**hot cues** (pads A–H) and which become **memory cues**, then write them
**straight into rekordbox's library** — no XML round-trip, no converter.

### 5 · Keep the library honest

**Tag Maker** fills in the genre and mood tags that are missing. **File
Analysis** finds the duplicates, the junk and the files that will not open,
and moves them into a quarantine folder according to a plan you can review
before it runs.

---

## Navigator

The library as one picture. It answers the question a folder cannot: *what do
I play next, out of ninety thousand tracks?*

### What you do here

- **Click a point** (or pick a track by name) to make it the seed, and see
  what mixes out of it, ranked by the transition cost
  `w1·distance on the map + w2·BPM gap + w3·Camelot distance`, with the three
  weights on sliders.
- **Draw a lasso**, which does one of two things depending on the shape you
  draw. A **line** through the clusters takes the tracks it passes near, in
  the order it meets them — a way to plan an arc (start in ambient, cross
  deep house, peak in tech house) by drawing it. A **loop** that comes back
  where it started takes everything it encloses, like the box. The app
  guesses from the shape, and a radio button lets you overrule it.
- **Box-select a group** and let **magic sort** order it: the cheapest path
  that visits every track once — an open travelling-salesman problem, solved
  nearest-neighbour then 2-opt — so each track melts into the next.
- **Grow a set one track at a time** in [Build a set](#building-a-set).
- **Export** the result as M3U8 or rekordbox XML.

Both the selection and the set builder write to the same **playlist**, which
is why the playlist has a section of its own rather than living inside
either — and why the board that draws a set as cards lives there too.

### How a track becomes a point

Two stages, one inference. The Essentia **Discogs-EffNet** model produces a
**1280-dimension embedding** — the acoustic identity of the track, before it
is flattened into genre names — and that same vector is then fed to two
classification heads, one for **400 Discogs genres** and one for **moods**,
which return several labels with their confidence: a track can be Minimal
*and* Deep House, and both are kept. The embedding is not a by-product read
off the side of a classifier; it is the first model's output, and the heads
are consumers of it. The frames are computed once and read at both places.

Only **twelve 10-second windows** are analyzed, spread evenly from the intro
to the outro, and their frames are averaged — temporal average pooling, one
vector per track. About 5 seconds per track instead of half a minute, which
on a whole library is the difference between hours and days. Before inference
each window is brought to **−14 LUFS** (EBU R128) so that a loud master does
not read as a different genre.

Many short windows rather than few long ones, because an average is only as
good as the number of *independent* samples in it, and thirty consecutive
seconds are almost always a single section of the track. Measured on 300
tracks of a real library against the embedding of the whole track: three 30 s
windows name the same nearest neighbour 58% of the time, twelve 10 s windows
84%. The windows are also concatenated into **one** call to the model, which
runs in fixed batches of 64 patches and would otherwise pay a whole batch for
the nine patches of a 10-second window.

### What the model does not decide

BPM, key and groove never touch the network. BPM and key are read from the
**file's own tags** when they are there (and a BPM outside 40–220 is
refused), and measured with Essentia otherwise — `RhythmExtractor2013` and
`KeyExtractor`, on a **30-second window of their own** at the centre of the
track, because the embedding windows are too short for a tempo detector to
find bars in. Both are properties of the whole track, and measuring them
twelve times does not improve them. The key is converted to its **Camelot**
code.

Groove is not a model output either: it is `1 − (spread of the gaps between
onsets)`, a hand-computed statistic. Read
[Groove](#groove--read-this-one-carefully) before trusting its name — it
measures how *uniform* the spacing of attacks is, which is not the same
thing, and on produced music it behaves in ways the label does not suggest.

### Where the map is kept

`~/.cache/djcaddy/map/`, in three files, and the split follows how they are
written:

| file | how it is written |
|---|---|
| `tracks.jsonl` | one JSON line per track, **appended** |
| `embeddings.f32` | the raw float32 vectors end to end, **appended** — line *n* of the first is block *n* of this one |
| `coords.npy` | the two UMAP columns, **rewritten whole** on each projection, because a projection is a fact about the entire library rather than about one track |

Appending instead of rewriting is what makes the job interruptible: at 90,000
tracks the embedding matrix is half a gigabyte.

**The map itself** is a UMAP projection of the embeddings to two dimensions —
UMAP rather than t-SNE because the distance *between* clusters keeps its
meaning, and that distance is exactly what a line drawn across the map uses.
A PCA to 64 dimensions runs first; it does not change the map (these
embeddings keep nearly all their variance there) but it removes most of
UMAP's neighbour search, which is almost all of its time.

A track is recognised by its **absolute path**, so moving the library to
another volume makes it a *different* library as far as the map is concerned.
That is what `--relocate` is for — see
[Building the map from the terminal](#building-the-map-from-the-terminal).

### Two kinds of nearness, and they are not the same

*Sounds like it* is the cosine over the full **1280 dimensions** — the real
one. *Mixes out of it* ranks by the transition cost, which uses the **two
projected coordinates** plus tempo and key. The 2D map is a shadow of the
embedding: convenient to look at, and flattened. A track can sound close and
mix badly, or the reverse, and the two tabs under a seed exist to let you ask
both questions.

**Genre and mood never enter proximity.** The transition cost takes exactly
three things — position on the map, BPM gap, Camelot distance — and no label.
Genres are a **filter**: they narrow the universe before the question is
asked, and say nothing about how close two tracks are once it has been. This
is deliberate: the whole point of the embedding is that it hears things a
genre name has already thrown away.

### Building a set

Magic sort answers *put these in the best order*. The set builder answers the
other question — *what comes next?* — one track at a time, which is how a set
is actually decided.

**Two tables give the orders.** On the left the chain as it stands; on the
right the candidates that mix out of whichever track you are standing on,
ranked by the same transition cost. Tick one or several, add them, and they
go on **one behind the other** — ticking three means "then these three", not
three branches off the same track. Both tables carry the same columns: BPM,
key, groove, the folder the file came from, and the **signed shift** against
the previous track.

That shift is the thing the cost cannot tell you. A cost is a distance and
has no sign: from a track at 118 BPM, one at 122 and one at 114 score the
same. `+4 BPM · +1 wheel · +0.09 groove` says which way the set is moving,
warm for rising and cool for falling. It is deliberately **not** in the
ranking — a set climbs, holds and lets go, and sorting by direction would be
choosing which of the three on the DJ's behalf.

The set builder has **its own filters**, not the map's: a clickable **Camelot
wheel** (two rings, major outside and minor inside, the way the players draw
it, because harmonic mixing is a question about neighbours and a list of
twenty-four codes hides exactly the adjacency that matters), plus genres, BPM
and groove ranges. Tracks already in the chain are never filtered away — a
filter is about what to propose next, not about breaking a chain someone has
built.

**Copies are one entry.** A track filed in four folders has the same tempo
and key in all four, so it has the same cost from anywhere and would take
four of the nine slots. They are gathered under one row marked `×4`, and
putting one down blocks the rest — a set should not take the same record
twice. Which copy is a real question, so the roster names them by folder and
lets you choose rather than picking for you.

### The board

**The board draws the playlist, not the chain**, and it sits in the playlist
section for that reason: a set is assembled from several places — an M3U8
opened, a lasso on the map, a chain built here — and the shape worth seeing
is the shape of the whole thing.

Left to right the cards follow the playlist order; how high a card sits is a
measure you pick with a radio — tempo, key or groove — so a set that climbs
looks like a climb. Each scale is fixed rather than stretched over the set:
the wheel for keys, the library's deciles for groove, and for tempo the pitch
fader's ±6% around where the set sits. Stretching a set over its own range
turns one BPM of drift into half the board, and this cost proposes tracks at
the same tempo — a chain of eight often spans about a single BPM. With fixed
scales two sets can be compared, and a flat row honestly means the measure
does not move.

Cards can be dragged off the rule and stay off it; picking a measure again
puts everything back, and the bin under a selected card takes that track out
of the playlist.

### Point size, and the running job

The **size of a point** carries a number you choose — BPM, groove or loudness
(integrated LUFS) — scaled between the 5th and 95th percentile so one outlier
does not flatten everything. The position already says how a track sounds;
the diameter is room for a quantity you can actually read. That is also why
the map stays in two dimensions: a third axis would cost the lasso and the
box (Plotly has neither in 3D) and would have to be read by rotating the
scene.

While the background job runs, the page shows its progress **live** — the app
re-reads the job's state file every couple of seconds without redrawing the
map — and can **pause**, **resume** or **stop** it (signals go to the whole
process group, so the parallel workers stop too), or open a window on the
tail of its log.

> **On the confidence threshold.** The spec this follows puts the multi-label
> threshold at 0.40. Measured on these models it is too high: the genre head
> is a softmax over 400 classes, and on an unmistakable track the top genre
> reaches 0.404 — at 0.40 almost the whole library would come back with a
> single label. The defaults are 0.15 for genres and 0.05 for moods (the same
> values the tagging already uses); both are settings.

### The quadrant chart

A second tab next to the map, on the same tracks. The map answers *what
sounds like what*, and to do it flattens 1280 numbers into two that mean
nothing on their own; the quadrants answer *where does this track sit between
dark and bright, between calm and driving*, and put two real measures on the
axes. Both charts feed the same seed and the same selection: click a point in
either one.

Either axis can be any of eleven measures, the four raw energy ingredients
included — those are what explain *why* a track reads 8.

The cross sits at the measure's own middle where it has one (valence's zero,
energy's half — energy is a rank, so its half *is* the median by
construction) and at the median of what the filters leave where it does not.
The caption under the chart says which of the two you are looking at, because
reading a quadrant as "these are the fast ones" when it says "these are the
faster half of what you are currently looking at" is a wrong conclusion drawn
confidently.

### The embedding fingerprint

A third tab on the same tracks, and the one that shows what the other two
hide. The map flattens 1280 numbers into two; this chart draws the 1280. One
row per track, one column per dimension — or per ten of them — and the colour
is how far that track sits from the others on that dimension: blue below the
middle, red above, the page's own background for *like everyone else*. The
centring is per column and not one scale for the whole picture, because the
dimensions have wildly different means: on a single scale three or four large
ones would decide every colour and the rest would be flat.

It is not read one track at a time, it is read in **bands**. Rows that go the
same way on the same columns are related, and that relatedness is exactly
what the map draws as nearness — here you see it before it is flattened.

The gestures are the map's: hovering a row says whose it is, clicking makes
it the seed, the lasso takes a band as a selection. All three charts feed the
same seed and the same selection. Dragging scrolls the picture rather than
cutting a box out of it — on a chart that can be wider than the pane that is
the gesture you want — and the box and the lasso are a click away in the
toolbar, which stays in the corner of the window even when the picture is
scrolled sideways.

The column on the left is the **cosine distance from the seed**, across all
1280 dimensions — the real distance, of which the map is the flattened
shadow. Its number is in the same tooltip as everything else: one label per
row, whether you point at the column or at the fingerprint, saying who the
track is, which row it is, and how far it sits from the seed. Clicking the
column seeds that row exactly like clicking the picture.

**Group nearest dimensions** is on by default: each column is then the
average of ten consecutive dimensions, 128 in all. Turn it off and every one
of the 1280 gets a column of its own — the picture is then wider than the
pane and **scrolls sideways**, one screen pixel per dimension, because
squeezing 1280 columns into eight hundred pixels is not showing them. It also
costs rows, because the picture has a pixel budget.

**Picture** is that budget: *light* is three million pixels — 23,437 rows at
128 columns, redrawn in about half a second — and *full* is twelve million,
enough to hold a library of ninety thousand tracks whole at a second and a
half a redraw, which is what a redraw of the map costs too. Above whatever
the budget allows, a stable random sample of what the filters leave is drawn
and the caption says how many of how many — the same rule the map follows
above 120,000 points. Every drawn row is a real track either way: what a
sample costs you is the rest of the library, never the truth about a row.
(Both budgets sample at 1280 columns: the whole library there would be a
hundred and ten million pixels.)

**Sort by** chooses the pile. *Library order* is the order the tracks went on
the map: the picture stays put when the seed changes, and only the distance
column is redrawn. *Distance from the seed* puts the seed's nearest
neighbours at the top and the strangers at the bottom — the distance column
becomes a readable gradient and the bands sort themselves — at the price of
redrawing the whole picture, about half a second, every time the seed moves.
Either way the tooltip ends with the row's number out of the total, which
under that second order is a ranking: row 12 is the twelfth nearest thing you
own to the seed.

### Building the map from the terminal

The same job the page launches, if you would rather drive it yourself:

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

`--relocate` rewrites the paths and leaves the embeddings and the projection
alone. It also takes the modification date from the file at its new address
when the size still matches, because copying without preserving dates (plain
`cp` does that) would otherwise make every track look changed and send it
back to the queue.

---

## Cue Finder

One track at a time: validate the suggested phrase boundaries by ear, then
turn them into cues.

**Analyse.** Pick a track (path field or file browser) and run it. Each
analysis saves a **sidecar** `<name>_analysis.json` next to the track: on a
later load, if that file is there and *Force analysis* is off, the results
come **from it** without re-analyzing — and without Demucs.

**Review.** The waveform is coloured by frequency band (red = lows,
green = mids, blue = highs) with the **section tags** overlaid
(Intro/Build-up/Drop/Breakdown/Outro). Move a section start, change its
label, and the waveform updates live. The player scrubs from any point so you
can confirm by ear.

> Section classification is **heuristic** — rules on energy and bass, with
> the thresholds in `core/analysis/sections.py`. It is a starting point to
> correct by ear, not ground truth. Ambiguous sections are labeled `Groove`;
> consecutive sections of the same type are **merged**, so each tag marks a
> **phrase change**.

**Vocals.** Demucs (source separation) isolates the vocal stem: the **sung
regions** appear as bands on the waveform — the parts not to overlap with
other vocals while mixing — and sections with vocals get the 🎤 flag. It is
accurate but **heavy**: a neural network per track, and a model download on
first run unless you are on the packaged build, which carries it. It is
optional — uncheck the box in the app, or `--no-vocals` on the CLI.

**From sections to cues.** Every phrase boundary and every vocal region
start/end becomes a row in the **cue table**. Nothing is assigned on its own:
you tick, row by row, which cues become **hot cues** (pads A–H, eight in
rekordbox) and which become **memory cues** — what does not fit on a pad is
not lost, it goes alongside as a memory cue.

**Writing into rekordbox.** The confirmed cues are written **directly into
rekordbox 6/7's library** — `master.db`, encrypted with SQLCipher, opened by
`pyrekordbox`, which also refuses to commit while rekordbox is running and
keeps the sync sequence numbers honest. A backup of the database is taken
before writing, one per day. No XML round-trip, no converter: the cues appear
on the waveform in rekordbox with their labels.
[The XML export](#getting-cues-into-other-dj-software) remains for everything
else.

---

## Tag Maker

Genre and mood written into the files' own tags, with the Essentia models.

The starting point is not a registry of what was attempted but the **files**:
the app reads what they contain now, and whoever has incomplete tags becomes
the work queue. On 400 tracks that a registry gave as done, 30 were no longer
at that path.

The batch job runs with progress, pause and resume, and a breakdown shows
what the library carries. `tag_cli.py` does the same from the terminal.

On Windows this page says up front that it needs the Mac — see
[Before you start](#before-you-start).

---

## File Analysis

What is in a folder, which files repeat, and which ones will not open.

The scan and the duplicate hunt are **two separate operations**, and not by
accident: the scan is fast and can be redone at will, while the duplicate
hunt reads the candidate files in full and takes minutes on a real library,
so it only runs when you ask.

Duplicates are moved into a **quarantine** folder according to a reviewable
plan, rather than deleted. Anything that deletes or moves sits behind a
confirmation tick, and the scan restarts by itself afterwards so what you are
looking at is never stale.

---

## Reference: what every number means

Every measure a track carries, what it is computed from, and what it does
**not** say. The app shows the same explanations on each column's tooltip.

### BPM and key

| | range | where from |
|---|---|---|
| **BPM** | 40–220 | the file's own tag when it has one, `RhythmExtractor2013` otherwise |
| **key / Camelot** | 1A–12B | the file's own tag, `KeyExtractor` otherwise |

Both are properties of the whole track, so they are measured once on a
dedicated **30-second window at the centre**, not on the twelve embedding
windows. A DJ library usually carries both already, and recomputing them
would show two different numbers for the same thing.

### Groove — read this one carefully

`1 − (standard deviation ÷ mean of the gaps between onsets)`, clamped to
0–1, on the same central 30-second window.

It measures **how uniform the spacing between attacks is**. That is not what
its name suggests, and the difference is not academic:

| pattern | groove |
|---|---|
| a bare straight kick | **1.00** |
| an unbroken run of sixteenths (240 attacks) | **1.00** |
| the same kick with 60 attacks on the sixteenth grid | 0.46 |
| the same kick with a vocal line over it | 0.30 |

Density does not lower it — a full grid still reads 1.00. What lowers it is a
**rhythmic figure**: some hits close together, some far apart. Which means a
track with a real groove tends to score **low**, and the most metronomic
material scores highest.

Two more things worth knowing. **`0.00` is a floor, not a measurement**: it
is what comes out whenever the spread of the gaps reaches their mean, so a
merely irregular track and a wildly irregular one both read 0.00. And an
**empty cell is different** — it means fewer than 8 onsets were found and the
statistic was refused rather than invented.

### Energy — four measures and one scale

How much the track pushes. Perceived energy has no single correlate in the
signal, so it is built from four, all on the same central 30-second window:

| ingredient | what it asks |
|---|---|
| `energy_density` | how many attacks per beat — how thick the rhythmic weave is |
| `energy_bass` | what share of the power sits below 200 Hz |
| `energy_bright` | where the spectral centroid sits — closed and dark, or open with hats on top |
| `energy_pulse` | how deeply the low end pulses **at the beat** — a straight kick against a syncopated 808 |

The four have incompatible units, so each is converted to its **percentile
rank across your library** before they are averaged, and the average is
ranked once more so the ten levels are evenly populated by construction. The
result is an integer **1–10**, read as deciles: a 10 is "the top tenth of
*what you own*", not an absolute level.

The raw four are what is stored; the 1–10 is derived at read time, so the
scale re-tunes itself as the library grows and the weights can change without
re-analysing anything.

**Loudness is deliberately not one of them.** LUFS measures how hard the
master was pushed, not how hard the track pushes — and the pipeline already
normalises it away at −14 LUFS before inference precisely because it is a
nuisance variable. Two of the four ingredients are scale-invariant by
construction: the same track at −26 dB gives identical values to the sixth
decimal. `lufs` is kept as the control instead: if energy correlated with it,
the measure would have rebuilt loudness by accident.

Validated on 27 tracks judged by ear across the whole range: mean error 0.33
levels, 25 of 27 within one level, r = +0.96. Removing `energy_pulse` triples
the error.

Where you see it: an `energy` column in every table (red, 1–10, between the
BPM and the groove), a `Δenergy` in the set builder's two tables written in
**steps** rather than ranks (`+2` is two deciles up, which is how you decide
whether the set is lifting), one of the point-size options on the map, one of
the board's height axes, and the default vertical axis of the quadrant chart.
An empty cell means the track has not been measured yet.

### Mood, and the emotion arrow

**`moods`** are words, not a number: up to four labels from the MTG-Jamendo
model's fixed vocabulary of 56, ordered by confidence, kept when they pass
0.05. The vocabulary mixes feelings (`Dark`, `Happy`) with themes (`Film`,
`Christmas`, `Retro`) — the thematic half cannot be projected onto any axis,
which is why the words are kept alongside the numbers rather than replaced by
them. In tables the **rarest** of a track's moods is printed first, because
the strongest one is usually the one nearly everybody shares.

**Valence** — the `emotion` arrow, the height of the board's mood axis, and
the horizontal axis of the quadrant chart — is a projection of those words
onto one dark→bright axis, −1 to +1. "Valence" is the proper name for it: it
is one of the two axes of Russell's circumplex, and Energy above is the other
one (arousal) measured from the signal instead of from words. There is no
third indicator to invent — the two together are the model.

The sign of each word comes from **two lists written by hand**: 8 words pull
dark, 13 pull bright, the other 35 are neutral. That part stays hand-made,
and it matters: reclassifying a single word can move a track across the axis
— the same track reads −0.27 with `Deep` counted dark and +0.27 with it
counted neutral.

The *weight* of each word is measured two different ways, and which one you
get depends on what the row carries:

| field on the row | how it weighs the words | what it misses |
|---|---|---|
| `moods` only | the label's **position** (1, ½, ⅓) | the strength, and the 52 labels under the threshold |
| `valence` | the model's **real activation**, over all 56 | nothing |

The first is what the library carried until the mood backfill; the second is
what `mood_cli` writes. The difference is not cosmetic. Under the first, a
track with `Dark` at 0.62 and one with `Dark` at 0.06 both read −1.00; and a
track with `Sad` 0.049, `Melancholic` 0.045 and `Dark` 0.041 passes no
threshold at all, carries no label, and gets no arrow — while having three
pieces of evidence for dark.

The two also treat neutral words differently, on purpose. By position,
neutral words count in the denominator: a track that is `Dark` *and also*
energetic and melodic reads less dark than one that is only `Dark`. By real
activation they are left out of both sides, because `Energetic` sits on 89%
of the library **and sits strongly**, so keeping it in the denominator would
push every track towards zero by nearly the same amount — losing range
without adding reading. How little colour a track carries is said instead by
`mood_evidence`, which is kept as its own number.

**Each side is the mean of its words, not their sum.** The two hand-written
lists are not the same size — 13 bright words against 8 dark ones — and a
multi-label head gives every one of the 56 labels a small baseline activation
even on a track that is not that thing. Summing therefore lets the sigmoid's
noise floor in 13 times on one side and 8 on the other. Taking the mean of
each side cancels it.

**And the number is read as a rank, never as an absolute.** This is the part
that measurement settled, after two guesses did not. On 2,000 real tracks:

| reading | share below zero | deciles |
|---|---|---|
| sums | 1.1% | +0.31 … +0.76 |
| means (shipped) | 6.3% | +0.07 … +0.64 |
| means, floor 0.02 | 8.9% | +0.03 … +1.00, saturating |

Balancing the lists helped and did not fix it: 94% of the library still reads
bright. The remaining skew is not the lists and not the music — it is the
model's prior. MTG-Jamendo learned on a corpus where `happy`, `positive` and
`upbeat` are far commoner tags than `sad` and `melancholic`, and that
frequency stayed in its head. No hand adjustment to the two word lists can
undo a bias that lives in the weights.

So the signed number stays on the row — it is the measurement, and it can be
rebuilt from `embeddings.f32` whenever — but everywhere valence acts as a
**position** (the quadrant axis, the board's height, the `emotion` arrow) it
is the **percentile rank across your library**, for the same reason energy is
read in deciles: *"darker than 70% of what you own"* is a true sentence,
*"valence +0.31"* is not. The rank has a real middle by construction, which
the raw number never had.

The arrow's dead zone is ±0.15 around that middle: the middle 30% of the
library gets no arrow, 35% points up, 35% points down.

**Sub-threshold activations are kept**, and that decision was settled by ear
rather than by statistics. 12% of the library takes its direction only from
activations below the 0.05 label threshold — words like `Dark` never appear
on those rows, yet the evidence for dark does. Three different internal
measures suggested that evidence was noise, and all three turned out to be
confounded: any way of isolating the faint activations alters them, because
the winning label is the one promoted above the threshold, so what remains
below is mostly the losing side. What answered the question was 20 of those
tracks laid out darkest-to-brightest (`mood_cli --faint-sample`): hardstyle,
50 Cent and a 1992 techno remix at one end, Abba, Tiffany and Baltimora at
the other. Nothing out of place. The faint evidence is signal.

`mood_conf` on the row is the top few activations written out
(`Dark:0.620; Deep:0.410; …`), the same way genre confidences are written. It
is there to be read, and to check the number against.

**Which pooling the number uses, and why it matters.** The model does not
read a track in one go: it cuts it into ~2 s slices and reads each one. There
are then two ways to get one answer out of many slices, and they do not agree
because the head is not linear:

| | how | what it preserves |
|---|---|---|
| mean of predictions | read every slice, average the 56 answers | a dark breakdown inside a bright track stays visible |
| prediction of the mean | average the slice vectors, read once | only what the track is *on average* |

The **words** (`moods`) come from the first — that is how all 87k rows were
already written, and it is the better reading. The three **numbers**
(`valence`, `mood_evidence`, `mood_conf`) come from the second, for one
reason: the mean of predictions was never saved, so tracks already on the map
cannot have it without re-reading every file. Using the better reading for
new tracks and the only available one for old tracks would leave the library
with two scales mixed and an invisible step in the middle of every
comparison. One scale is worth more than a slightly better one. `analyze()`
and `mood_cli` therefore call the same function on the same input, and the
number of any track can be rebuilt from `embeddings.f32` alone.

> `mood_cli --check N` measures how much that choice costs, before anything
> gets rewritten. **`top label kept`** is the number to read: both sides use
> the same threshold and the same selection rule, so the only difference
> between them is the pooling. `agrees with the old reading` is *not* that
> measurement — it also contains the switch from ranks to real weights, which
> is the improvement being sought, so a value below 1 there is expected.

### Distances, in the tables

| | what it measures |
|---|---|
| `similarity` | cosine between the two **1280-dimension** fingerprints — the real nearness |
| `sound` | distance on the **2-D projection**, which is that nearness flattened for drawing |
| `cost` | the transition cost: `sound`, `bpm cost` and `key cost` in one number |

`similarity` and `sound` answer the same question at two resolutions, and
they can disagree: the map is a shadow, and shadows lose a dimension.

### Vibe, sections, loudness

**Vibe** (`Warm-Up-Low`, `Peak-Time-High`, …) is a name, not a measure:
tempo bucket + RMS energy split at the 33rd and 66th percentiles of the
library. It predates the Energy above and is coarser.

**Sections** (`Intro`, `Build-up`, `Drop`, `Breakdown`, `Outro`) are a
heuristic on the energy arc and the presence of bass, with thresholds
relative to the track itself — not machine learning, and meant to be checked
by ear in the app.

**`lufs`** is the integrated loudness of the analysed windows *before*
normalisation. It describes the master, not the music.

---

## Command line

Everything the app does in the background is also a CLI, and the app launches
exactly these.

**Batch analysis and organisation** — `cli.py`:

```bash
# report to stdout only
poetry run python cli.py ~/Music/dj

# report to file + dry-run of the organization (copy, not move)
poetry run python cli.py ~/Music/dj --dest ~/Music/master --report report.csv --dry-run

# actually organize into Genre/Vibe (without overwriting existing files)
poetry run python cli.py ~/Music/dj --dest ~/Music/master
```

The report (CSV or JSON) contains, per track: path, genre, BPM, vibe and the
suggested phrase-boundary timestamps. The cache avoids re-analyzing files
that were already processed; `--no-cache` forces re-analysis.

**The rest**, all working on tracks already in the map store:

| CLI | what it does |
|---|---|
| `map_cli.py` | builds the map, projects it, relocates it — see [above](#building-the-map-from-the-terminal) |
| `tag_cli.py` | the tagging job, same as the Tag Maker page |
| `energy_cli.py` | measures the four energy fields — re-reads the audio, resumable |
| `mood_cli.py` | re-scores valence from the stored embeddings: no audio, minutes instead of hours |
| `zoo_cli.py` | tries the model zoo's other Discogs-EffNet heads (aggressive, relaxed, party, danceability) — reports only, writes nothing |

---

## Getting cues into other DJ software

The native path is the one in [Cue Finder](#cue-finder): cues go **directly
into the rekordbox library**. For everything else, section tags can be
exported as a **rekordbox XML** collection — the format most DJ software
converters accept as input:

```bash
# whole library, one collection.xml with all tracks' cues
poetry run python cli.py ~/Music/dj --rekordbox-xml collection.xml
```

- **rekordbox**: the XML is a *library*, not a playlist file, and
  `File ▸ Import ▸ Import Playlist` will not even let you select an `.xml`.
  Load it under `Preferences ▸ Advanced ▸ Database ▸ rekordbox xml` by
  pointing **Imported Library** at the file; the tracks and playlists then
  appear under the `rekordbox xml` tree in the sidebar, to drag into your own
  collection. For a playlist alone, with no BPM or cues, the **M3U8** export
  is what Import Playlist accepts.
- **Serato / Traktor**: rekordbox XML is the format that third-party
  converters — [MIXO](https://www.mixo.dj/),
  [Lexicon](https://www.lexicondj.com/) — accept to produce a library those
  apps can import.

---

## Building the standalone app

One bundle, and it depends on nothing: no Python, no Poetry, no first-run
download, no system ffmpeg. Everything is inside — torch/demucs,
essentia-tensorflow, librosa, umap, pyrekordbox, the Essentia models, the
Demucs checkpoint, ffmpeg and ffprobe, `plotly.min.js`, the two HTML
frontends. Only your DATA stays outside (`~/.cache/djcaddy/`, the sidecars
next to your tracks), so it survives every update of the app.

### Once, before the first build

The spec refuses to build rather than ship a bundle that would need the
network later, and it tells you which of these is missing:

| what | how |
|---|---|
| ffmpeg + ffprobe on the PATH | `brew install ffmpeg` |
| the Essentia models in `~/essentia_models` | download them there, or point `DJCADDY_MODEL_DIR` at wherever you keep them |
| the Demucs checkpoint in the torch cache | `poetry run python -c "from demucs.pretrained import get_model; get_model('htdemucs')"` — downloads it once |

Plus the full environment: `poetry install`, the `essentia` group included —
without it the bundle still builds, but map building and tagging will not be
in it.

### Every time

```bash
./packaging/build_macos.sh
```

That is the whole loop. It rasterizes the icon from the app's own SVG, runs
PyInstaller, ad-hoc signs the `.app`, and makes the DMG:

```
dist/DjCaddy.app          ~2 GB
dist/DjCaddy-0.1.0.dmg    ~780 MB
```

PyInstaller takes about **three and a half minutes** on a quiet machine once
`build/` is warm; the DMG step is the slow one, because it compresses two
gigabytes. The version in the DMG name and in the app's Info.plist both come
from `pyproject.toml` — that is the only place it is written.

The signature is **ad-hoc, not notarised**: enough to run it here, and on
another machine if you open it with right-click ▸ Open the first time.
Shipping it to strangers would need a Developer ID and `xcrun notarytool`.

### Checking what you built

Run the bundle's own autonomy check. It answers eight questions: is ffmpeg
the bundled one, are the models inside, is the Demucs checkpoint there, does
`plotly.min.js` resolve, are the HTML frontends in, do the heavy libraries
actually import, does what gets WRITTEN land outside the read-only app, and
can the app still launch its own background jobs.

```bash
./dist/DjCaddy.app/Contents/MacOS/DjCaddy --selftest
```

Better still, run it from the mounted DMG — that is the state your other
machine will be in, read-only volume and all:

```bash
hdiutil attach dist/DjCaddy-0.1.0.dmg -nobrowse -readonly
/Volumes/DjCaddy/DjCaddy.app/Contents/MacOS/DjCaddy --selftest
hdiutil detach /Volumes/DjCaddy
```

It works from the source tree too (`poetry run python packaging/entry.py
--selftest`), asking the same eight questions of your working copy.

### When you change something, what do you have to touch?

Most of the time: nothing. Rebuild and you are done.

| you changed | what the bundle needs |
|---|---|
| Python code, the HTML frontends, the theme, the icon SVG | nothing — just rebuild |
| added a dependency with `poetry add` | usually nothing: PyInstaller follows the imports. Add it to the `collect_all` list in the spec only if it loads modules **by name** or ships **data files** of its own |
| a new file the app reads at runtime (a model, another frontend) | add it to `datas` in `packaging/djcaddy.spec` |
| a new external binary the code shells out to | add it to `binaries` in the spec; it lands in `bin/`, which is on the PATH inside the bundle |
| a new CLI meant to run as a background job | add it to `JOBS` in `packaging/entry.py` and to `hiddenimports` — inside the bundle `sys.executable` is the app, so jobs are the app calling itself with `--job <name>` |
| the version | `pyproject.toml`, and nowhere else |

The two failures worth recognising, because they only ever happen inside the
bundle:

- **`ModuleNotFoundError` for something that imports fine from source** — the
  module is reached by name, so the analysis never saw it. Add it to
  `hiddenimports`.
- **`FileNotFoundError` on a data file** — it was not collected. Add it to
  `datas`, and read it through a path anchored on `core.bundle.resources()`
  rather than on the current directory.

`core/bundle.py` is the only place that knows where things live, and outside
the bundle it changes nothing at all — the tests in `tests/test_bundle.py`
pin that path by path. So when a new piece of the app needs a file, ask it
where the file is; do not compute it twice.

### Keeping the loop short

Do not debug in the bundle. Run from source (`poetry run python -m
qt_app.main`, `poetry run pytest`) until the behaviour is right — the bundle
tests **packaging**, not behaviour, and it costs minutes where source costs
seconds.

Rebuilds reuse `build/pyinstaller`, so the dependency analysis is cached
between runs. Delete `build/` when you want a clean slate: after editing the
spec's own logic, or when a stale collected file is the suspect.

### Windows

The spec is platform-agnostic, but PyInstaller cannot cross-compile: the
Win11 bundle has to be built on Win11.

```powershell
poetry install --without essentia
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
```

Essentia has no Windows wheel, so that bundle is complete except for what
depends on it — Tag Maker and *building* the map. Everything else works, and
the pages concerned say so themselves. The last step needs Inno Setup 6
(`iscc`) on the PATH; without it you still get `dist\DjCaddy\`, just no
installer.

---

## How the code is laid out

```
core/
├── analysis/        # the engine: pure Python, no UI — imported by the app and the CLIs
├── viz/             # presentation logic: Plotly figure builders, palettes,
│   │                #   table columns, board/wheel payloads — functions that
│   │                #   return data, never widgets
│   └── frontend/    # the two HTML components (playlist board, Camelot wheel)
└── bundle.py        # where things live when the app is packaged
qt_app/              # the desktop app: pages, widgets, AppState (Qt signals),
│                    #   background workers
packaging/           # the standalone bundle: PyInstaller spec, the single
│                    #   entry point, icon, build scripts, Windows installer
tests/               # pytest; Qt pages under pytest-qt; figure snapshot tests
cli.py, map_cli.py, energy_cli.py, mood_cli.py, tag_cli.py, zoo_cli.py
```

The rule that holds `core/viz` together: its functions take dataframes and
state and return **data** — a Plotly figure, a dataframe, a payload dict —
never a widget. That is the contract that keeps a chart the same chart
wherever it is drawn.

Key engine modules (`core/analysis/`):

| Module | Responsibility |
| --- | --- |
| `engine.py` | orchestration: two-pass analysis, cache, organize plan |
| `audio_features.py` | audio loading (librosa) + BPM and RMS in a single pass |
| `structure.py` | structural segmentation (Foote novelty over self-similarity) → phrase boundaries |
| `sections.py` | section classification (Intro/Build-up/Drop/Breakdown/Outro) from energy arc and bass presence |
| `vocals.py` | vocal detection via source separation (Demucs): sung regions + 🎤 flag per section |
| `waveform.py` | frequency-band colored waveform (red = lows, green = mids, blue = highs) |
| `vibe.py`, `tags.py`, `cache.py` | tempo/energy buckets → vibe; tag reading via mutagen; per-file cache |
| `map_profile.py` | acoustic profile of a track: Discogs-EffNet embedding (1280-D) feeding the genre/mood heads; BPM and key from tags or Essentia; groove from onset regularity |
| `map_projection.py` | PCA to 64-D, then UMAP projection of the embeddings to the 2D map |
| `map_store.py` | the map on disk: `tracks.jsonl` + `embeddings.f32` appended, `coords.npy` rewritten; cosine nearest-neighbours on the raw embeddings |
| `map_job.py` | the map build as a long, resumable background job |
| `mixing.py` | Camelot wheel, transition cost, signed tempo/key shifts, path-drawn playlists, magic sort |
| `graph_playlist.py` | the chain as a graph: tracks, links, layout on the board, the roster of what comes next |
| `energy.py` | the four raw energy measures, and the library-wide ranking that turns them into a 1–10 |
| `mood_scale.py` | the mood words onto one dark→bright axis (valence), by rank or by the model's real weights |
| `essentia_tags.py`, `tag_job.py` | genre/mood inference and the batch tagging job |
| `duplicates.py`, `folder_scan.py` | duplicate hunting, quarantine plan, folder contents |
| `cue_export.py` | phrase sections and vocal regions → cue rows, and their mapping onto pads |
| `rekordbox_write.py` | hot/memory cues written into rekordbox 6/7's encrypted `master.db` (via pyrekordbox) |
| `dj_export.py` | export to rekordbox XML and M3U8 |

The design decisions and their trade-offs — why Plotly inside a
QWebEngineView, why the HTML components were reused rather than rewritten,
why the long jobs are separate processes — are recorded phase by phase in
[docs/piano-qt.md](docs/piano-qt.md).
