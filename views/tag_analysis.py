"""Sezione "Tag analysis": genere e mood nei tag, con i modelli Essentia.

Il punto di partenza non è il registro di cosa è stato tentato ma i FILE:
si legge cosa contengono adesso, e chi ha i tag incompleti diventa la coda
di lavoro. Sono due cose diverse — su 400 brani che il registro dava per
fatti, 30 non erano più a quel percorso.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analysis.essentia_tags import (
    GENRE_FORMATS,
    MODEL_DIR,
    MODELS,
    EssentiaAnalyzer,
    TagSettings,
    available,
    build_tag_values,
    find_taggable,
    missing_models,
    scan_coverage,
    write_tags,
)
from analysis.tag_tracking import DEFAULT_TRACKING_FILE, ProcessedTracker
from views.components import pick_folder, play_table

st.title("🏷️ Tag analysis")
st.caption(
    "Analyze tracks with the Essentia models and write **genre** and **mood** "
    "into the file tags — the mood goes into the default comment field, "
    "which is the one djay Pro actually displays."
)

# --- Ambiente --------------------------------------------------------------

with st.expander("Environment", expanded=not available() or bool(missing_models())):
    if available():
        st.success("`essentia` is importable in this environment.")
    else:
        st.warning(
            "`essentia` is not importable here. It comes with a plain "
            "`poetry install` — it lives in its own group only so that "
            "`poetry install --without essentia` stays available if the wheel "
            "is ever missing for the Python in use."
        )
    missing = missing_models()
    if missing:
        st.warning(f"{len(missing)} model file(s) missing from `{MODEL_DIR}`.")
    else:
        st.success(f"All {len(MODELS)} model files found in `{MODEL_DIR}`.")
    st.dataframe(
        [{"file": name, "purpose": purpose, "present": name not in missing}
         for name, purpose in MODELS.items()],
        width="stretch", hide_index=True,
    )

    tracker = ProcessedTracker()
    if tracker.existed:
        st.caption(
            f"Progress file: {len(tracker):,} tracks recorded "
            f"({tracker.duplicate_lines:,} repeated lines absorbed) · "
            f"`{DEFAULT_TRACKING_FILE}`")
    else:
        st.caption(
            "No progress file yet. Copy the one from the standalone script "
            f"over `{DEFAULT_TRACKING_FILE}` — same format, one absolute path "
            "per line, repeats are harmless.")

# --- Cosa c'è già ----------------------------------------------------------

st.divider()
st.subheader("What is already tagged")
st.caption(
    "Reads the files themselves rather than the progress log: the log says "
    "what was *attempted*, this says what is *in the file now*. Roughly 12 ms "
    "per track, so a whole library takes a while — but only once, after which "
    "the filters below are instant."
)

root = pick_folder("tag_analysis::path", "Folder to scan")
if root is None:
    st.info("Choose a folder to start.")
    st.stop()

scan_key = f"tagscan::{root}"
if st.button("Scan tags", type="primary"):
    with st.spinner("Listing audio files…"):
        files = find_taggable(root)
    if not files:
        st.warning("No taggable audio files here.")
    else:
        bar = st.progress(0.0, text="Reading tags…")
        st.session_state[scan_key] = scan_coverage(
            files, progress=lambda done, total: bar.progress(
                done / total if total else 1.0,
                text=f"Read {done:,}/{total:,} — about "
                     f"{max(0, total - done) * 0.012 / 60:.0f} min left"))
        bar.empty()

coverage = st.session_state.get(scan_key)
if coverage is None:
    st.info("Press **Scan tags** to see what is missing.")
    st.stop()

readable = coverage.readable
if not readable:
    st.warning("Nothing readable was found here.")
    st.stop()

with_genre = sum(c.has_genre for c in readable)
with_comment = sum(c.has_comment for c in readable)
complete = sum(c.complete for c in readable)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Tracks", f"{len(readable):,}")
m2.metric("With genre", f"{with_genre:,}",
          delta=f"{with_genre / len(readable):.0%}", delta_color="off")
m3.metric("With comment", f"{with_comment:,}",
          delta=f"{with_comment / len(readable):.0%}", delta_color="off",
          help="Only the default comment field counts — the one djay Pro shows.")
m4.metric("Complete", f"{complete:,}",
          delta=f"{complete / len(readable):.0%}", delta_color="off")

if coverage.unreadable:
    with st.expander(f"⚠️ {len(coverage.unreadable)} files whose tags could not be read"):
        st.caption(
            "Not a tagging problem — these are files no reader opens. Folder "
            "analysis → **Unreadable files** is where they get dealt with.")
        st.dataframe(
            pd.DataFrame([{"file": c.path.name, "folder": str(c.path.parent),
                           "why": c.error} for c in coverage.unreadable]),
            width="stretch", hide_index=True)

# --- Cosa manca ------------------------------------------------------------

st.divider()
st.subheader("What still needs doing")

choice = st.radio(
    "Show tracks that are missing…",
    ["genre or comment", "genre", "comment", "both", "nothing (already complete)"],
    horizontal=True,
)

if choice == "nothing (already complete)":
    selected = [c for c in readable if c.complete]
    note = ("Already carrying both tags. Re-running these would only change "
            "anything with **overwrite** turned on.")
else:
    selected = coverage.missing(
        genre=choice in ("genre or comment", "genre", "both"),
        comment=choice in ("genre or comment", "comment", "both"),
        require_both=choice == "both",
    )
    note = ("This is the work queue: exactly the tracks whose tags are not "
            "there, read from the files rather than assumed from a log.")

st.caption(note)
st.metric("Matching tracks", f"{len(selected):,}",
          delta=f"{len(selected) / len(readable):.0%} of the folder",
          delta_color="off")

if not selected:
    st.success("Nothing matches — there is nothing to do for this filter.")
else:
    table = pd.DataFrame([
        {"file": c.path.name, "folder": str(c.path.parent),
         "genre": c.genre or "—", "comment": c.comment or "—",
         "_path": str(c.path)}
        for c in selected[:1000]
    ])
    play_table(
        f"cov::{root}", table, ["file", "folder", "genre", "comment"],
        {c: st.column_config.TextColumn(disabled=True)
         for c in ("file", "folder", "genre", "comment")},
        editable=False, editor_key=f"cov_editor::{root}::{choice}")
    if len(selected) > 1000:
        st.caption(f"Showing the first 1.000 of {len(selected):,}.")

    st.session_state["tag_analysis::queue"] = [c.path for c in selected]

# --- Impostazioni ----------------------------------------------------------

queue = st.session_state.get("tag_analysis::queue", [])
if not queue:
    st.stop()

st.divider()
st.subheader("Settings")
st.caption("The same options the terminal script asks for, as controls.")

col_g, col_m = st.columns(2)
with col_g:
    do_genres = st.checkbox("Write genre", value=True)
    top_genres = st.number_input("How many genres", 1, 10, 3, disabled=not do_genres)
    genre_threshold = st.slider(
        "Genre threshold", 0.0, 1.0, 0.15, 0.01, disabled=not do_genres,
        help="Minimum activation. If nothing clears it the single best label "
             "is written anyway — a track always has a genre.")
    genre_format = st.selectbox(
        "Genre format", GENRE_FORMATS, disabled=not do_genres,
        help='How "Rock---Alternative Rock" is written out.')
with col_m:
    do_moods = st.checkbox("Write mood", value=True)
    moods_in_tag = st.number_input("How many moods in the comment", 1, 5, 3,
                                   disabled=not do_moods)
    mood_threshold = st.slider(
        "Mood threshold", 0.0, 1.0, 0.005, 0.005, disabled=not do_moods,
        help="Lower than the genre one on purpose: mood activations are much "
             "smaller. Nothing is invented if none clear it.")
    confidence_tags = st.checkbox(
        "Also write confidence tags", value=True,
        help="Percentages in a separate field, next to the tag itself.")

col_o, col_s = st.columns(2)
overwrite = col_o.checkbox(
    "Overwrite tags that are already there", value=False,
    help="Off: a track that already has a genre keeps it. The queue above "
         "mostly contains tracks missing something, so this rarely matters.")
max_seconds = col_s.number_input(
    "Seconds of audio to analyze", 0, 1200, 300, 30,
    help="0 = the whole track. Do not go below 300 without reason: measured "
         "on a disco-house track with a one-minute intro, 120s gave "
         '"Ambient / Space, Dark, Relaxing" where 300s gave the correct '
         '"Nu-Disco, House / Summer, Deep, Happy".')

settings = TagSettings(
    genres=do_genres, moods=do_moods, top_genres=int(top_genres),
    genre_threshold=genre_threshold, genre_format=genre_format,
    mood_threshold=mood_threshold, moods_in_tag=int(moods_in_tag),
    confidence_tags=confidence_tags, overwrite=overwrite,
    max_seconds=int(max_seconds),
)

# --- Esecuzione ------------------------------------------------------------

st.divider()
st.subheader("Run")

if not available():
    st.error("`essentia` is not importable, so nothing can be analyzed here.")
    st.stop()
if missing_models():
    st.error("Model files are missing — see Environment above.")
    st.stop()

st.caption(
    f"About 10-14 seconds per track, plus a one-off 2.6s to load the models. "
    f"The whole queue of {len(queue):,} would take roughly "
    f"{len(queue) * 12 / 3600:.1f} hours, so it runs in batches: each pass "
    "writes as it goes and records what it finished, and pressing the button "
    "again carries on from there.")

col_n, col_dry = st.columns([1, 2])
batch = col_n.number_input("Tracks this pass", 1, 500, min(10, len(queue)))
dry_run = col_dry.checkbox(
    "Dry run — analyze but write nothing", value=True,
    help="On by default. Turn it off when the results look right.")

if st.button(f"{'Analyze' if dry_run else 'Analyze and write'} "
             f"{int(batch)} of {len(queue):,}",
             type="primary"):
    todo = queue[:int(batch)]
    analyzer = EssentiaAnalyzer(settings)
    tracker = ProcessedTracker()
    bar = st.progress(0.0, text="Loading models…")
    rows, failures = [], []

    for i, path in enumerate(todo, 1):
        bar.progress((i - 1) / len(todo), text=f"{i}/{len(todo)} · {path.name[:60]}")
        try:
            tags = analyzer.analyze(path)
            values = build_tag_values(tags, settings)
            written = [] if dry_run else write_tags(path, tags, settings)
            rows.append({
                "file": path.name, "genre": values.genre or "—",
                "mood": values.mood or "—",
                "written": "dry run" if dry_run else (", ".join(written) or "nothing"),
                "_path": str(path),
            })
            if not dry_run:
                tracker.mark(path)
        except Exception as e:
            failures.append({"file": path.name, "folder": str(path.parent),
                             "error": f"{type(e).__name__}: {e}"})
    bar.progress(1.0, text="Done")
    st.session_state["tag_analysis::results"] = (rows, failures, dry_run)

results = st.session_state.get("tag_analysis::results")
if results:
    rows, failures, was_dry = results
    if rows:
        st.success(f"{len(rows)} track(s) analyzed"
                   + (" — nothing written (dry run)." if was_dry else " and written."))
        play_table(
            "run", pd.DataFrame(rows), ["file", "genre", "mood", "written"],
            {c: st.column_config.TextColumn(disabled=True)
             for c in ("file", "genre", "mood", "written")},
            editable=False, editor_key="run_editor")
    if failures:
        st.warning(f"{len(failures)} track(s) could not be analyzed.")
        st.dataframe(pd.DataFrame(failures), width="stretch", hide_index=True)
    if not was_dry:
        st.caption("Re-run **Scan tags** above to refresh the queue.")
