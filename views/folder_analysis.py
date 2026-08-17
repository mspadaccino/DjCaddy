"""Sezione "Folder analysis": cosa c'è in una cartella e quali file si ripetono.

Due operazioni separate, e non per caso: la scansione è veloce e si può
rifare a piacere, la ricerca duplicati legge per intero i file candidati e su
una libreria vera dura minuti. La seconda si lancia solo quando serve.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from analysis.duplicates import (
    LEVEL_OTHER_FOLDER,
    LEVEL_SAME_FOLDER,
    LEVEL_SIMILAR_NAME,
    QUARANTINE_DIRNAME,
    apply_quarantine_plan,
    build_quarantine_plan,
    find_duplicates,
    write_csv,
)
from analysis.folder_scan import human_size, scan_folder

st.title("📁 Folder analysis")
st.caption(
    "Count what a folder actually contains, then look for duplicates. "
    "Nothing here ever deletes a file: duplicates are reported, and can be "
    f"**moved** to a `{QUARANTINE_DIRNAME}/` quarantine folder that you empty "
    "yourself once djay Pro still looks right."
)

folder_input = st.text_input(
    "Folder to analyze",
    value=st.session_state.get("folder_analysis::path", ""),
    placeholder="/Volumes/Crucial X9/DJSet",
)
audio_only = st.checkbox(
    "Audio files only", value=False,
    help="Off: counts every file, so you also see artwork, .DS_Store, cue sheets "
         "and the rest. Duplicate search always considers audio files only.",
)

root = Path(folder_input).expanduser() if folder_input.strip() else None
if root is None:
    st.info("Enter a folder path to start.")
    st.stop()
if not root.is_dir():
    st.error(f"Not a folder: {root}")
    st.stop()
st.session_state["folder_analysis::path"] = str(root)

scan_key = f"scan::{root}::{audio_only}"

if st.button("Scan folder", type="primary"):
    bar = st.progress(0.0, text="Scanning…")
    seen = st.empty()

    def _progress(n: int) -> None:
        seen.caption(f"{n:,} files seen…")

    with st.spinner("Walking the folder tree…"):
        st.session_state[scan_key] = scan_folder(root, audio_only=audio_only,
                                                 progress=_progress)
    bar.empty()
    seen.empty()

scan = st.session_state.get(scan_key)
if scan is None:
    st.info("Press **Scan folder** to begin.")
    st.stop()

# --- Conteggi -------------------------------------------------------------

audio = scan.audio
c1, c2, c3 = st.columns(3)
c1.metric("Files", f"{len(scan.files):,}")
c2.metric("Audio files", f"{len(audio):,}")
c3.metric("Total size", human_size(scan.total_size()))

counts = scan.counts_by_format()
sizes = scan.size_by_format()
st.dataframe(
    pd.DataFrame(
        [{"format": fmt, "files": n, "size": human_size(sizes[fmt])}
         for fmt, n in counts.most_common()]
    ),
    width="stretch", hide_index=True,
)

with st.expander(f"By extension ({len(scan.counts_by_extension())} distinct)"):
    st.dataframe(
        pd.DataFrame([{"extension": e, "files": n}
                      for e, n in scan.counts_by_extension().most_common()]),
        width="stretch", hide_index=True,
    )

if scan.unreadable:
    with st.expander(f"⚠️ {len(scan.unreadable)} unreadable entries"):
        st.dataframe(pd.DataFrame([{"path": str(p), "error": e}
                                   for p, e in scan.unreadable]),
                     width="stretch", hide_index=True)

# --- Duplicati ------------------------------------------------------------

st.divider()
st.subheader("Duplicates")
st.caption(
    "Files are grouped by size first, and only same-size files get hashed — "
    "two files of different size cannot be identical. Expect the hashing pass "
    "to read those files in full."
)

dup_key = f"dups::{root}"
if st.button(f"Find duplicates among {len(audio):,} audio files"):
    bar = st.progress(0.0, text="Hashing…")

    def _progress(done: int, total: int) -> None:
        bar.progress(done / total if total else 1.0,
                     text=f"Hashing {done:,}/{total:,} candidate files…")

    st.session_state[dup_key] = find_duplicates(audio, progress=_progress)
    bar.empty()

report = st.session_state.get(dup_key)
if report is None:
    st.stop()

wasted = sum(g.wasted_bytes for g in report.same_folder)
d1, d2, d3 = st.columns(3)
d1.metric("A — same folder", f"{len(report.same_folder):,}",
          help="Certain duplicates: same folder, same size, same MD5.")
d2.metric("B — other folders", f"{len(report.other_folder):,}",
          help="Same file in different folders — often deliberate.")
d3.metric("C — similar name", f"{len(report.similar_name):,}",
          help="Names that look alike but the files differ.")
st.caption(f"{report.hashed_files:,} files hashed · "
           f"{human_size(wasted)} recoverable from level A alone")


def _table(groups) -> pd.DataFrame:
    return pd.DataFrame([
        {"folder": str(g.folder), "keep": g.keep.name, "duplicate": dup.name,
         "same folder": dup.parent == g.keep.parent,
         "size": human_size(g.size), "copies": g.copies, "md5": (g.md5 or "")[:12]}
        for g in groups for dup in g.duplicates
    ])


for title, groups, note in [
    (f"A · Certain duplicates, same folder ({len(report.same_folder)})",
     report.same_folder,
     "Safe to quarantine: byte-identical files sitting side by side."),
    (f"B · Same file in other folders ({len(report.other_folder)})",
     report.other_folder,
     "**Candidates only.** The same track under 80s/, DANCE RETRO/ and "
     "Workout/ is probably how you organised the library on purpose."),
    (f"C · Similar names, different content ({len(report.similar_name)})",
     report.similar_name,
     "Informational. Different edits, remixes or rips — never auto-removed."),
]:
    with st.expander(title, expanded=groups is report.same_folder and bool(groups)):
        st.caption(note)
        table = _table(groups)
        if table.empty:
            st.write("Nothing found.")
        else:
            st.dataframe(table, width="stretch", hide_index=True)

# --- Report e quarantena --------------------------------------------------

st.divider()
csv_name = st.text_input("Report file name", value=f"{root.name}_duplicates.csv")
if st.button("Write CSV report"):
    try:
        written = write_csv(report.all_groups(), root / csv_name)
        st.success(f"Report written: {written}")
    except OSError as e:
        st.error(f"Could not write the report: {e}")

st.subheader("Quarantine")
st.caption(
    f"Moves the level-A duplicates into `{root / QUARANTINE_DIRNAME}`, keeping "
    "their original folder structure so you can see where each came from and "
    "put it back. Nothing is deleted — check djay Pro still sees everything, "
    "then empty that folder yourself."
)

plan = build_quarantine_plan(report.same_folder, root)
if not plan:
    st.info("No level-A duplicates to quarantine.")
else:
    st.write(f"**{len(plan):,} files** would move, freeing {human_size(wasted)}.")
    st.dataframe(
        pd.DataFrame([{"from": str(s), "to": str(d)} for s, d in plan[:200]]),
        width="stretch", hide_index=True,
    )
    if len(plan) > 200:
        st.caption(f"Showing the first 200 of {len(plan):,}.")

    confirm = st.checkbox(
        f"I have read the list and want to move these {len(plan):,} files",
        key="quarantine_confirm",
    )
    if st.button("Move duplicates to quarantine", type="primary", disabled=not confirm):
        moved, errors = apply_quarantine_plan(plan, dry_run=False)
        st.success(f"{moved:,} files moved to {root / QUARANTINE_DIRNAME}")
        if errors:
            st.warning(f"{len(errors)} could not be moved.")
            st.dataframe(pd.DataFrame([{"path": str(p), "error": e}
                                       for p, e in errors]),
                         width="stretch", hide_index=True)
        st.session_state.pop(dup_key, None)
        st.session_state.pop(scan_key, None)
