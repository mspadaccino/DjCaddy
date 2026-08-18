"""Sezione "Tag analysis": genere e mood nei tag, con i modelli Essentia.

Il punto di partenza non è il registro di cosa è stato tentato ma i FILE:
si legge cosa contengono adesso, e chi ha i tag incompleti diventa la coda
di lavoro. Sono due cose diverse — su 400 brani che il registro dava per
fatti, 30 non erano più a quel percorso.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from analysis.essentia_tags import (
    GENRE_FORMATS,
    MODEL_DIR,
    MODELS,
    TagSettings,
    analyze_many,
    available,
    build_tag_values,
    default_workers,
    find_taggable,
    missing_models,
    scan_coverage,
    write_tags,
)
from analysis.tag_job import load_state
from analysis.tag_tracking import DEFAULT_TRACKING_FILE, ProcessedTracker
from views.components import pick_folder, play_table

# Secondi a brano misurati su questa macchina (M5, 10 core, 24 analisi), per
# numero di processi. Servono a dire quanto ci vorrà PRIMA di partire, che è
# l'unica informazione che cambia la scelta.
SECONDS_PER_TRACK = {1: 8.2, 2: 5.7, 3: 5.0, 5: 4.1, 8: 3.7}


def _spelled(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} hours"


def _seconds_each(n: int) -> float:
    known = min(SECONDS_PER_TRACK, key=lambda k: abs(k - n))
    return SECONDS_PER_TRACK[known]


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

# --- Cosa analizzare -------------------------------------------------------

st.divider()
st.subheader("Folder to analyze")
st.caption("Every audio track inside it, subfolders included.")

root = pick_folder("tag_analysis::path", "Folder")
if root is None:
    st.info("Choose a folder to start.")
    st.stop()

list_key = f"taglist::{root}"
if list_key not in st.session_state:
    with st.spinner("Listing audio files…"):
        st.session_state[list_key] = find_taggable(root)
scope = st.session_state[list_key]
scope_name = f"{len(scope):,} track(s) under {root.name}"

if not scope:
    st.info("Nothing selected yet.")
    st.stop()
st.success(f"**{scope_name}** ready.")

# --- Impostazioni ----------------------------------------------------------

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
        help="Percentages in a SEPARATE field (ESSENTIA_GENRE / "
             "ESSENTIA_MOOD), beside the tag itself. djay Pro does not "
             "display these — they are there for later inspection.")
    confidence_in_comment = st.checkbox(
        "Percentages in the comment too", value=False, disabled=not do_moods,
        help="Puts them in the comment djay Pro actually shows: "
             '"Happy 87%; Deep 62%" instead of "Happy; Deep". The dedicated '
             "MOOD field stays clean either way.")

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

workers = int(st.number_input(
    "Tracks at the same time", 1, max(2, (os.cpu_count() or 2)),
    default_workers(),
    help="How many analyses run in parallel, each in its own process. Half "
         "the cores is the sweet spot here: measured, 8 processes are only "
         "8% faster than 5 because the cores actually busy stop at 3.4 "
         "either way, while the memory keeps going up (~1.3 GB each)."))

settings = TagSettings(
    genres=do_genres, moods=do_moods, top_genres=int(top_genres),
    genre_threshold=genre_threshold, genre_format=genre_format,
    mood_threshold=mood_threshold, moods_in_tag=int(moods_in_tag),
    confidence_tags=confidence_tags, overwrite=overwrite,
    confidence_in_comment=confidence_in_comment,
    max_seconds=int(max_seconds),
)

# --- Job in background -----------------------------------------------------

# L'alternativa al blocco qui sopra, per quando i brani sono troppi: la
# pagina resta bloccata per tutta la durata di una corsa, e a qualche secondo
# per brano una libreria intera sono giorni. Il job e' lo STESSO lavoro
# lanciato come processo a se', che sopravvive alla chiusura del browser.

st.divider()
st.subheader("The whole folder, in the background")
st.caption(
    "Two ways to run, and this is the one for a lot of tracks: it works "
    "through the folder on its own and writes as it goes. To pick tracks by "
    "hand and see what comes back before saving, keep scrolling — the tag "
    "read and the **Analyze** button are below.")

stato = load_state()

if stato is not None and stato.running:
    fatti, totale = stato.done, max(1, stato.total)
    st.progress(fatti / totale,
                text=f"{fatti:,}/{stato.total:,} · {stato.current[:50]}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Written", f"{stato.written:,}")
    c2.metric("Failed", f"{stato.failed:,}")
    c3.metric("Per track", f"{stato.seconds_each:.1f}s")
    c4.metric("Left", _spelled(stato.eta_seconds))
    st.caption(
        f"Running as process {stato.pid} on `{stato.folder}` — closing this "
        "tab does not stop it. Unlike the run above, the job writes the tags "
        "as it goes, so what is done is already saved.")
    if st.button("↻ Refresh"):
        st.rerun()
else:
    if stato is not None:
        esito = (f"Last job: {stato.written:,} written, {stato.failed:,} "
                 f"failed out of {stato.total:,}.")
        (st.warning if stato.failed else st.success)(esito)
        if stato.errors:
            with st.expander(f"{len(stato.errors)} error(s) kept"):
                st.dataframe(pd.DataFrame(stato.errors), width="stretch",
                             hide_index=True)

    st.caption(
        f"Analyzes **{root}** and writes as it goes, skipping whatever the "
        "progress file already records — so it can be stopped and restarted "
        "without losing or repeating work. It uses the settings above.")
    if st.button(f"▶ Start the job with {workers} at a time"):
        log = Path(tempfile.gettempdir()) / "wavecut_tag_job.log"
        cmd = [sys.executable, str(Path(__file__).resolve().parent.parent / "tag_cli.py"),
               str(root), "--workers", str(workers),
               "--genre-format", settings.genre_format,
               "--max-seconds", str(settings.max_seconds)]
        if settings.overwrite:
            cmd.append("--overwrite")
        if settings.confidence_in_comment:
            cmd.append("--confidence-in-comment")
        if not settings.genres:
            cmd.append("--no-genres")
        if not settings.moods:
            cmd.append("--no-moods")
        # start_new_session: staccato da Streamlit, cosi' un riavvio della
        # pagina (o la sua chiusura) non porta giu' anche il lavoro.
        with open(log, "w") as out:
            subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                             start_new_session=True,
                             cwd=Path(__file__).resolve().parent.parent)
        st.success(f"Started. Output in `{log}`.")
        time.sleep(1.5)          # il job scrive lo stato quasi subito
        st.rerun()

# --- Cosa c'è dentro -------------------------------------------------------

# Sotto questa soglia i tag si leggono da soli appena scegli la cartella: a
# ~12 ms l'uno, duemila file sono meno di mezzo minuto. Sopra, il tempo lo si
# dichiara e lo si fa chiedere — sull'intera libreria sarebbero 18 minuti, e
# non è una cosa da far partire per sbaglio.
AUTO_READ_BELOW = 2000

st.divider()
st.subheader("What is in them")

scan_key = f"tagscan::{root}::{len(scope)}"
estimate = len(scope) * 0.012 / 60

if scan_key not in st.session_state:
    if len(scope) <= AUTO_READ_BELOW:
        bar = st.progress(0.0, text="Reading tags…")
        st.session_state[scan_key] = scan_coverage(
            scope, progress=lambda done, total: bar.progress(
                done / total if total else 1.0, text=f"Read {done:,}/{total:,}…"))
        bar.empty()
    else:
        st.info(
            f"**{len(scope):,} tracks** — reading their tags takes about "
            f"{estimate:.0f} minutes. It happens once; after that the filters "
            "are instant. Pick a smaller folder if that is too long.")
        if st.button(f"Read the tags of {len(scope):,} tracks", type="primary"):
            bar = st.progress(0.0, text="Reading tags…")
            st.session_state[scan_key] = scan_coverage(
                scope, progress=lambda done, total: bar.progress(
                    done / total if total else 1.0,
                    text=f"Read {done:,}/{total:,} — about "
                         f"{max(0, total - done) * 0.012 / 60:.0f} min left"))
            bar.empty()
            st.rerun()
        st.stop()

coverage = st.session_state[scan_key]
readable = coverage.readable

if not readable:
    st.warning("None of these files had readable tags.")
    st.stop()

with_genre = sum(c.has_genre for c in readable)
with_comment = sum(c.has_comment for c in readable)
complete = sum(c.complete for c in readable)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Tracks read", f"{len(readable):,}")
m2.metric("With genre", f"{with_genre:,}",
          delta=f"{with_genre / len(readable):.0%}", delta_color="off")
m3.metric("With comment", f"{with_comment:,}",
          delta=f"{with_comment / len(readable):.0%}", delta_color="off",
          help="Only the default comment counts — the one djay Pro shows.")
m4.metric("Complete", f"{complete:,}",
          delta=f"{complete / len(readable):.0%}", delta_color="off")

choice = st.radio(
    "Work on tracks missing…",
    ["genre or comment", "genre", "comment", "both", "everything (no filter)"],
    horizontal=True)
if choice.startswith("everything"):
    selected = readable
else:
    selected = coverage.missing(
        genre=choice in ("genre or comment", "genre", "both"),
        comment=choice in ("genre or comment", "comment", "both"),
        require_both=choice == "both")

if not selected:
    st.success(
        "**Every track here already has what this filter looks for.** "
        "That is why there is no table and no **Analyze** button below: with "
        "nothing missing, there is nothing queued. To go over these tracks "
        "anyway — different settings, a second opinion — pick "
        "**everything (no filter)** above, and turn on *Overwrite tags that "
        "are already there* in the settings if the point is to replace what "
        "they carry.")
    st.stop()

st.caption(
    f"**{len(selected):,} tracks** match — all of them listed, all ticked. "
    "Untick whatever you want left alone.")

# Si elencano TUTTI, senza tetto: la griglia disegna solo le righe a schermo,
# quindi la lunghezza dell'elenco quasi non si paga — misurate 26 ms per
# 10.000 righe e 88 ms per 90.000. Un tetto renderebbe invisibile lo stato di
# tag proprio dei file che restano da sistemare.
table = pd.DataFrame([
    {"Analyze": True, "file": c.path.name,
     "GENRE": c.genre or "❌ missing",
     "COMMENT": c.comment or "❌ missing",
     "folder": str(c.path.parent), "_path": str(c.path)}
    for c in selected])
edited_cov = play_table(
    "cov", table, ["Analyze", "file", "GENRE", "COMMENT", "folder"],
    {"Analyze": st.column_config.CheckboxColumn("Analyze"),
     "GENRE": st.column_config.TextColumn(
        "GENRE", disabled=True, width="medium",
        help="What is in the genre tag right now."),
     "COMMENT": st.column_config.TextColumn(
        "COMMENT", disabled=True, width="medium",
        help="The default comment — where the moods go, and the only one "
             "djay Pro displays."),
     "file": st.column_config.TextColumn(disabled=True),
     "folder": st.column_config.TextColumn(disabled=True)},
    editor_key=f"cov_editor::{choice}")

unticked = {Path(x) for x in edited_cov.loc[~edited_cov["Analyze"], "_path"]}
queue = [c.path for c in selected if c.path not in unticked]

if coverage.unreadable:
    with st.expander(f"⚠️ {len(coverage.unreadable)} files whose tags could not be read"):
        st.caption("Not a tagging problem — no reader opens these. Folder "
                   "analysis → **Unreadable files** deals with them.")
        st.dataframe(
            pd.DataFrame([{"file": c.path.name, "folder": str(c.path.parent),
                           "why": c.error} for c in coverage.unreadable]),
            width="stretch", hide_index=True)

st.session_state["tag_analysis::queue"] = queue

# --- Esecuzione ------------------------------------------------------------

if not queue:
    st.info(
        "Everything in the table is unticked, so nothing is queued and the "
        "**Analyze** button has nothing to run on. Tick a row to bring it "
        "back.")
    st.stop()

st.divider()
st.subheader("Analyze")

if not available():
    st.error("`essentia` is not importable, so nothing can be analyzed here.")
    st.stop()
if missing_models():
    st.error("Model files are missing — see Environment above.")
    st.stop()

st.caption(
    "Nothing is written at this stage: the results appear below and saving "
    "them is a separate click, so the analysis is never paid for twice."
)

batch = int(st.number_input(
    "How many to analyze now", 1, max(1, len(queue)), max(1, len(queue)),
    help="Defaults to the whole queue. Lower it to try a handful first and "
         "see what comes back before committing to hours."))

each = _seconds_each(workers)
eta = batch * each
spelled = _spelled(eta)
st.caption(
    f"About **{each:.0f}s per track** at {workers} at a time"
    f"{f' (vs {SECONDS_PER_TRACK[1]:.0f}s one at a time)' if workers > 1 else ''}"
    f" — roughly **{spelled}** for {batch:,}. "
    f"Each process holds its own copy of the models, about "
    f"{workers * 1.3:.1f} GB in total."
)
if eta > 3600:
    st.warning(
        f"That is about {eta / 3600:.1f} hours, and the browser tab has to "
        "stay open the whole time — closing it stops the run, though "
        "everything already saved stays saved. Long runs are what the "
        "background job is for.")

if st.button(f"Analyze {batch} of {len(queue):,}", type="primary"):
    todo = queue[:batch]
    bar = st.progress(0.0, text="Loading models…")
    done, failures = [], []
    for i, (path, tags, error) in enumerate(
            analyze_many(todo, settings, workers=workers), 1):
        bar.progress(i / len(todo),
                     text=f"{i}/{len(todo)} · {path.name[:60]}")
        if error is None:
            done.append((path, tags))
        else:
            failures.append({"file": path.name, "folder": str(path.parent),
                             "error": error})
    bar.empty()
    st.session_state["tag_analysis::analyzed"] = done
    st.session_state["tag_analysis::failed"] = failures

analyzed = st.session_state.get("tag_analysis::analyzed", [])
failures = st.session_state.get("tag_analysis::failed", [])

if failures:
    st.warning(f"{len(failures)} track(s) could not be analyzed.")
    st.dataframe(pd.DataFrame(failures), width="stretch", hide_index=True)

if not analyzed:
    st.stop()

# --- Cosa verrebbe scritto, e salvataggio ----------------------------------

st.divider()
st.subheader("What it found — save when it looks right")
st.caption(
    "Nothing has touched the files yet. The formatting below follows the "
    "settings above, so changing the genre format or how many moods go in the "
    "comment updates this without re-analyzing. The thresholds are the "
    "exception: they are applied while analyzing, so changing those needs "
    "another pass."
)

rows = []
for path, tags in analyzed:
    values = build_tag_values(tags, settings)
    rows.append({
        "Save": True, "file": path.name,
        "GENRE": values.genre or "—", "COMMENT": values.mood or "—",
        "confidence": (values.genre_confidence or "")[:60],
        "_path": str(path),
    })
edited_run = play_table(
    "run", pd.DataFrame(rows),
    ["Save", "file", "GENRE", "COMMENT", "confidence"],
    {"Save": st.column_config.CheckboxColumn("Save"),
     "GENRE": st.column_config.TextColumn("GENRE (proposed)", disabled=True,
                                          width="medium"),
     "COMMENT": st.column_config.TextColumn("COMMENT (proposed)", disabled=True,
                                            width="medium"),
     "file": st.column_config.TextColumn(disabled=True),
     "confidence": st.column_config.TextColumn(disabled=True)},
    editor_key="run_editor")

to_save = set(edited_run.loc[edited_run["Save"], "_path"])
st.caption(f"**{len(to_save)}** of {len(rows)} ticked for saving.")

if st.button(f"💾 Save tags to {len(to_save)} file(s)", type="primary",
             disabled=not to_save):
    tracker = ProcessedTracker()
    written, problems = 0, []
    pending = [(p, t) for p, t in analyzed if str(p) in to_save]
    bar = st.progress(0.0, text="Writing…")
    for i, (path, tags) in enumerate(pending, 1):
        bar.progress(i / len(pending), text=f"{i}/{len(pending)} · {path.name[:60]}")
        try:
            if write_tags(path, tags, settings):
                written += 1
            tracker.mark(path)
        except Exception as e:
            problems.append({"file": path.name, "error": f"{type(e).__name__}: {e}"})
    bar.empty()
    st.success(f"Tags written to {written} file(s).")
    if problems:
        st.warning(f"{len(problems)} could not be written.")
        st.dataframe(pd.DataFrame(problems), width="stretch", hide_index=True)
    st.session_state.pop("tag_analysis::analyzed", None)
    for key in [k for k in list(st.session_state)
                if str(k).startswith(("tagscan::", "tagpreview::"))]:
        st.session_state.pop(key, None)
