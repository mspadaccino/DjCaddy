"""Sezione "Folder analysis": cosa c'è in una cartella e quali file si ripetono.

Due operazioni separate, e non per caso: la scansione è veloce e si può
rifare a piacere, la ricerca duplicati legge per intero i file candidati e su
una libreria vera dura minuti. La seconda si lancia solo quando serve.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

from analysis.duplicates import (
    QUARANTINE_DIRNAME,
    apply_quarantine_plan,
    build_quarantine_plan,
    duplicates_of,
    find_duplicates,
    write_csv,
)
from analysis.folder_scan import (
    delete_sidecars,
    find_sidecars,
    human_size,
    scan_folder,
)

st.title("📁 Folder analysis")
st.caption(
    "Count what a folder actually contains, then look for duplicates. "
    "Nothing here ever deletes a file: duplicates are reported, and the ones "
    f"you tick are **moved** to a `{QUARANTINE_DIRNAME}/` folder that you "
    "empty yourself once djay Pro still looks right."
)


# --- Scelta della cartella -------------------------------------------------

PATH_KEY = "folder_analysis::path"


def _pick_folder() -> None:
    """Apre il selettore di cartelle nativo del Mac e riempie il campo.

    Stesso meccanismo di `_pick_file` in Wave analysis: Streamlit non ha un
    dialogo di cartelle (il browser non dà accesso al filesystem), ma l'app
    gira in locale, quindi si può chiedere direttamente al Finder.
    """
    try:
        out = subprocess.run(
            ["osascript", "-e",
             'POSIX path of (choose folder with prompt "Choose a folder to analyze")'],
            capture_output=True, text=True, check=True,
        )
        chosen = out.stdout.strip()
        if chosen:
            st.session_state[PATH_KEY] = chosen.rstrip("/")
    except Exception:
        pass  # dialogo annullato o non disponibile: nessuna modifica


col_path, col_browse = st.columns([5, 1])
folder_input = col_path.text_input(
    "Folder to analyze", key=PATH_KEY,
    placeholder="/Volumes/Crucial X9/DJSet",
)
col_browse.markdown("<div style='height:1.8em'></div>", unsafe_allow_html=True)
col_browse.button("📁 Browse…", on_click=_pick_folder, width="stretch")

if not folder_input.strip():
    st.info("Choose a folder to start.")
    st.stop()
root = Path(folder_input).expanduser()
if not root.is_dir():
    st.error(f"Not a folder: {root}")
    st.stop()

audio_only = st.checkbox(
    "Audio files only", value=False,
    help="Off: counts every file, so you also see artwork, .DS_Store, cue sheets "
         "and the rest. Duplicate search always considers audio files only.",
)

scan_key = f"scan::{root}::{audio_only}"

if st.button("Scan folder", type="primary"):
    seen = st.empty()
    with st.spinner("Walking the folder tree…"):
        st.session_state[scan_key] = scan_folder(
            root, audio_only=audio_only,
            progress=lambda n: seen.caption(f"{n:,} files seen…"))
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

sizes = scan.size_by_format()
st.dataframe(
    pd.DataFrame([{"format": fmt, "files": n, "size": human_size(sizes[fmt])}
                  for fmt, n in scan.counts_by_format().most_common()]),
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
    st.session_state[dup_key] = find_duplicates(
        audio,
        progress=lambda done, total: bar.progress(
            done / total if total else 1.0,
            text=f"Hashing {done:,}/{total:,} candidate files…"))
    bar.empty()

report = st.session_state.get(dup_key)
if report is None:
    st.stop()

# Gruppi e file sono due numeri diversi: un gruppo con tre copie e' un gruppo
# solo ma DUE file da spostare. Mostrarli entrambi evita di chiedersi perche'
# la quarantena ne proponga piu' di quanti ne conta il riquadro qui sopra.
a_files = duplicates_of(report.same_folder)
b_files = duplicates_of(report.other_folder)
recoverable = sum(g.wasted_bytes for g in report.same_folder)

d1, d2, d3 = st.columns(3)
d1.metric("A — same folder", f"{len(report.same_folder):,} groups",
          delta=f"{len(a_files):,} duplicate files", delta_color="off",
          help="Certain duplicates: same folder, same size, same MD5.")
d2.metric("B — other folders", f"{len(report.other_folder):,} groups",
          delta=f"{len(b_files):,} duplicate files", delta_color="off",
          help="Same file in different folders — often deliberate.")
d3.metric("C — similar name", f"{len(report.similar_name):,} groups",
          delta="informational only", delta_color="off",
          help="Names that look alike but the files differ.")
st.caption(f"{report.hashed_files:,} files hashed · "
           f"{human_size(recoverable)} recoverable from level A alone")


def _rows(groups, preselect: bool) -> pd.DataFrame:
    return pd.DataFrame([
        {"Move": preselect, "folder": str(g.folder), "keep": g.keep.name,
         "duplicate": dup.name, "size": human_size(g.size), "copies": g.copies,
         "md5": (g.md5 or "")[:12], "_path": str(dup)}
        for g in groups for dup in g.duplicates
    ])


def _selection_table(level: str, title: str, note: str, groups,
                     preselect: bool) -> list[Path]:
    """Tabella con una spunta per riga, più i comandi per spuntarle tutte.

    Il pulsante globale cambia i valori di partenza, e perché l'editor li
    mostri davvero va ricreato da zero: da qui il contatore nella chiave.
    """
    with st.expander(title, expanded=bool(groups) and level == "A"):
        st.caption(note)
        table = _rows(groups, preselect)
        if table.empty:
            st.write("Nothing found.")
            return []

        epoch_key, force_key = f"epoch::{level}::{root}", f"force::{level}::{root}"
        epoch = st.session_state.setdefault(epoch_key, 0)

        col_all, col_none, col_count = st.columns([1, 1, 3])
        if col_all.button("Select all", key=f"all::{level}", width="stretch"):
            st.session_state[force_key] = True
            st.session_state[epoch_key] = epoch + 1
            st.rerun()
        if col_none.button("Select none", key=f"none::{level}", width="stretch"):
            st.session_state[force_key] = False
            st.session_state[epoch_key] = epoch + 1
            st.rerun()
        if force_key in st.session_state:
            table["Move"] = st.session_state[force_key]

        edited = st.data_editor(
            table, key=f"editor::{level}::{root}::{epoch}",
            width="stretch", hide_index=True,
            column_order=["Move", "folder", "keep", "duplicate", "size", "copies", "md5"],
            column_config={
                "Move": st.column_config.CheckboxColumn(
                    "Move", help="Tick the files to move into quarantine."),
                "folder": st.column_config.TextColumn(disabled=True),
                "keep": st.column_config.TextColumn("keep (stays)", disabled=True),
                "duplicate": st.column_config.TextColumn(disabled=True),
                "size": st.column_config.TextColumn(disabled=True),
                "copies": st.column_config.NumberColumn(disabled=True),
                "md5": st.column_config.TextColumn(disabled=True),
            },
        )
        chosen = [Path(p) for p in edited.loc[edited["Move"], "_path"]]
        col_count.write("")
        col_count.caption(f"**{len(chosen):,}** of {len(table):,} selected")
        return chosen


selected_a = _selection_table(
    "A", f"A · Certain duplicates, same folder "
         f"({len(report.same_folder)} groups, {len(a_files)} files)",
    "Byte-identical files sitting side by side. Ticked by default.",
    report.same_folder, preselect=True)

selected_b = _selection_table(
    "B", f"B · Same file in other folders "
         f"({len(report.other_folder)} groups, {len(b_files)} files)",
    "**Candidates only, so nothing is ticked by default.** The same track "
    "under 80s/, DANCE RETRO/ and Workout/ is probably how you organised the "
    "library on purpose.",
    report.other_folder, preselect=False)

with st.expander(f"C · Similar names, different content "
                 f"({len(report.similar_name)} groups)"):
    st.caption(
        "Informational only — no ticks here, because these are NOT the same "
        "file: different edits, remixes or rips that happen to be named alike.")
    table_c = _rows(report.similar_name, preselect=False)
    if table_c.empty:
        st.write("Nothing found.")
    else:
        st.dataframe(table_c.drop(columns=["Move", "md5", "_path"]),
                     width="stretch", hide_index=True)

# --- Report e quarantena --------------------------------------------------

st.divider()
csv_name = st.text_input("Report file name", value=f"{root.name}_duplicates.csv")
if st.button("Write CSV report"):
    try:
        st.success(f"Report written: {write_csv(report.all_groups(), root / csv_name)}")
    except OSError as e:
        st.error(f"Could not write the report: {e}")

st.subheader("Quarantine")
st.caption(
    f"Moves the ticked files into `{root / QUARANTINE_DIRNAME}`, keeping their "
    "original folder structure so you can see where each came from and put it "
    "back. Nothing is deleted — check djay Pro still sees everything, then "
    "empty that folder yourself."
)

selected = selected_a + selected_b
plan = build_quarantine_plan(selected, root)
if not plan:
    st.info("Nothing ticked yet — select the files to move in the tables above.")
else:
    freed = sum(p.stat().st_size for p, _ in plan if p.exists())
    st.write(f"**{len(plan):,} files** ticked "
             f"({len(selected_a):,} from A, {len(selected_b):,} from B), "
             f"freeing {human_size(freed)}.")
    st.dataframe(
        pd.DataFrame([{"from": str(s), "to": str(d)} for s, d in plan[:200]]),
        width="stretch", hide_index=True,
    )
    if len(plan) > 200:
        st.caption(f"Showing the first 200 of {len(plan):,}.")

    confirm = st.checkbox(
        f"I have read the list and want to move these {len(plan):,} files",
        key="quarantine_confirm")
    if st.button("Move to quarantine", type="primary", disabled=not confirm):
        moved, errors = apply_quarantine_plan(plan, dry_run=False)
        st.success(f"{moved:,} files moved to {root / QUARANTINE_DIRNAME}")
        if errors:
            st.warning(f"{len(errors)} could not be moved.")
            st.dataframe(pd.DataFrame([{"path": str(p), "error": e}
                                       for p, e in errors]),
                         width="stretch", hide_index=True)
        st.session_state.pop(dup_key, None)
        st.session_state.pop(scan_key, None)


# --- Sidecar AppleDouble ---------------------------------------------------

st.divider()
st.subheader("macOS sidecar files")
st.caption(
    "On a non-macOS volume (this drive is exFAT) the Finder cannot store a "
    "file's extended attributes inside it, so it writes them to a companion "
    "`._<name>` — one per track, plus one per folder. They carry the track's "
    "own extension but hold no audio: typically 4 KB whose only real content "
    "is the browser's download-quarantine flag. The Finder hides them, "
    "rekordbox refuses them, djay Pro ignores them."
)

sidecar_key = f"sidecars::{root}"
if st.button("Find sidecar files"):
    seen = st.empty()
    with st.spinner("Looking for sidecars…"):
        st.session_state[sidecar_key] = find_sidecars(
            root, progress=lambda n: seen.caption(f"{n:,} checked…"))
    seen.empty()

sidecars = st.session_state.get(sidecar_key)
if sidecars is not None:
    s1, s2 = st.columns(2)
    s1.metric("Sidecars found", f"{len(sidecars.confirmed):,}",
              help="Name starts with ._ AND the content is a real AppleDouble.")
    s2.metric("Space they take", human_size(sidecars.freed_bytes))

    if sidecars.unverified:
        st.warning(
            f"{len(sidecars.unverified)} file(s) are named like a sidecar but "
            "are not one. They will NOT be touched — deleting on the strength "
            "of a name is how a real file gets lost.")
        st.dataframe(pd.DataFrame([{"path": str(p)} for p in sidecars.unverified]),
                     width="stretch", hide_index=True)

    if not sidecars.confirmed:
        st.info("No sidecar files here.")
    else:
        with st.expander(f"Show the {len(sidecars.confirmed):,} files"):
            st.dataframe(
                pd.DataFrame([{"path": str(p)} for p in sidecars.confirmed[:500]]),
                width="stretch", hide_index=True)
            if len(sidecars.confirmed) > 500:
                st.caption(f"Showing the first 500 of {len(sidecars.confirmed):,}.")

        st.caption(
            "These are **deleted**, not quarantined: there is nothing inside "
            "worth keeping, and on this filesystem macOS recreates them the "
            "next time the Finder touches a folder — so it is a recurring "
            "tidy-up, not a one-off. Each file's content is re-checked "
            "immediately before removal.")
        sure = st.checkbox(
            f"Delete these {len(sidecars.confirmed):,} files "
            f"({human_size(sidecars.freed_bytes)})", key="sidecar_confirm")
        if st.button("Delete sidecar files", type="primary", disabled=not sure):
            removed, freed, errors = delete_sidecars(sidecars.confirmed, dry_run=False)
            st.success(f"{removed:,} sidecar files deleted, {human_size(freed)} freed.")
            if errors:
                st.warning(f"{len(errors)} skipped.")
                st.dataframe(pd.DataFrame([{"path": str(p), "reason": e}
                                           for p, e in errors]),
                             width="stretch", hide_index=True)
            st.session_state.pop(sidecar_key, None)
            st.session_state.pop(scan_key, None)
