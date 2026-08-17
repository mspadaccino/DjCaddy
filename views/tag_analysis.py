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
    MODEL_DIR,
    MODELS,
    available,
    find_taggable,
    missing_models,
    scan_coverage,
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
    st.info(
        f"**{len(selected):,} tracks** are queued. Running them is the next "
        "piece of this section — the analyzer and the tag writer are already "
        "in place and verified against the real models.")
