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
    QUARANTINE_DIRNAME,
    apply_quarantine_plan,
    build_quarantine_plan,
    duplicates_of,
    find_duplicates,
    write_csv,
)
from views.components import pick_folder, play_table
from analysis.mix_names import (
    DEFAULT_KEYWORDS,
    NOT_BY_DEFAULT,
    looks_like_a_mashup,
    matching_words,
    parse_keywords,
)
from analysis.truncation import inspect
from analysis.folder_scan import (
    CHECK_THREADS,
    check_integrity,
    human_duration,
    read_durations,
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

root = pick_folder("folder_analysis::path", "Folder to analyze")
if root is None:
    st.info("Choose a folder to start.")
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
st.header("Duplicates")
st.caption(
    "Files are grouped by size first, and only same-size files get hashed — "
    "two files of different size cannot be identical. Expect the hashing pass "
    "to read those files in full. Each identical set is then checked once to "
    f"confirm it is really audio, and the `{QUARANTINE_DIRNAME}/` folder is "
    "skipped so a second run does not re-find what the first set aside."
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

def _selection_table(level: str, title: str, note: str, groups,
                     preselect: bool, full_paths: bool = False) -> tuple[list[Path], int]:
    """Tabella con una spunta per riga, più i comandi per spuntarle tutte.

    Il pulsante globale cambia i valori di partenza, e perché l'editor li
    mostri davvero va ricreato da zero: da qui il contatore nella chiave.
    """
    with st.expander(title, expanded=bool(groups)):
        st.caption(note)
        table = _rows(groups, preselect, full_paths=full_paths)
        if table.empty:
            st.write("Nothing found.")
            return [], 0

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

        pair = (["stays", "moves if ticked"] if full_paths
                else ["folder", "keep", "duplicate"])
        edited = play_table(
            f"dup{level}::{root}", table,
            ["Move", *pair, "size", "copies", "md5"],
            {
                "Move": st.column_config.CheckboxColumn(
                    "Move", help="Tick the files to move into quarantine."),
                "keep": st.column_config.TextColumn("keep (stays)", disabled=True),
                "stays": st.column_config.TextColumn(
                    "stays", disabled=True, width="large"),
                "moves if ticked": st.column_config.TextColumn(
                    "moves if ticked", disabled=True, width="large"),
                **{c: st.column_config.TextColumn(disabled=True)
                   for c in ("folder", "duplicate", "size", "md5")},
                "copies": st.column_config.NumberColumn(disabled=True),
            },
            editor_key=f"editor::{level}::{root}::{epoch}")
        picked = edited.loc[edited["Move"]]
        chosen = [Path(p) for p in picked["_path"]]
        chosen_bytes = int(picked["_bytes"].sum()) if not picked.empty else 0
        col_count.write("")
        col_count.caption(
            f"**{len(chosen):,}** of {len(table):,} selected · "
            f"**{human_size(chosen_bytes)}** would be freed "
            f"(of {human_size(int(table['_bytes'].sum()))} in this section)")
        return chosen, chosen_bytes



report = st.session_state.get(dup_key)

# Se i duplicati non sono ancora stati cercati si salta SOLO questo blocco:
# fermare la pagina qui teneva in ostaggio anche le sezioni sotto, che con i
# duplicati non c'entrano, ed e' il motivo per cui sembravano parte di essi.
if report is not None:

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

    if report.broken:
        broken_files = sum(len(g.paths) for g in report.broken)
        st.error(
            f"**{broken_files:,} files in {len(report.broken):,} groups are "
            "identical because they are equally BROKEN, not because they are the "
            "same music** — and they are excluded from everything above. Empty "
            "files all share the hash of nothingness; tag-only stubs share their "
            "tag. They are different tracks, so removing one would destroy the "
            "only trace of a song you are missing.")
        with st.expander(f"Show the {broken_files:,} broken files"):
            play_table(
                f"broken::{root}",
                pd.DataFrame([{"file": p.name, "folder": str(p.parent),
                               "size": human_size(g.size), "why": g.reason,
                               "_path": str(p)}
                              for g in report.broken for p in g.paths]),
                ["file", "folder", "size", "why"],
                {c: st.column_config.TextColumn(disabled=True)
                 for c in ("file", "folder", "size", "why")},
                editable=False, editor_key=f"broken_editor::{root}")
            st.caption(
                "Use **Unreadable files** below to move these to quarantine if "
                "you want them out of the way — one by one, not as duplicates.")
    st.caption(f"{report.hashed_files:,} files hashed · "
               f"{human_size(recoverable)} recoverable from level A alone")


    def _rows(groups, preselect: bool, full_paths: bool = False) -> pd.DataFrame:
        """Righe della tabella. Con `full_paths` le due copie sono mostrate per
        intero invece che come nome più una cartella sola: nei livelli B e C
        stanno in cartelle DIVERSE, e vedere solo quella di una delle due non
        dice dove sia l'altra — che è proprio l'informazione che serve per
        decidere."""
        return pd.DataFrame([
            {"Move": preselect,
             **({"stays": str(g.keep), "moves if ticked": str(dup)} if full_paths
                else {"folder": str(g.folder), "keep": g.keep.name,
                      "duplicate": dup.name}),
             "size": human_size(g.size), "copies": g.copies,
             "md5": (g.md5 or "")[:12], "_path": str(dup), "_bytes": g.size}
            for g in groups for dup in g.duplicates
        ])


    selected_a, bytes_a = _selection_table(
        "A", f"A · Certain duplicates, same folder "
             f"({len(report.same_folder)} groups, {len(a_files)} files)",
        "Byte-identical files sitting side by side. Ticked by default.",
        report.same_folder, preselect=True)

    selected_b, bytes_b = _selection_table(
        "B", f"B · Same file in other folders "
             f"({len(report.other_folder)} groups, {len(b_files)} files)",
        "Byte-identical copies sitting in different folders. If you organise in "
        "rekordbox rather than by folder, one copy is enough — **Select all** "
        "ticks the lot in one go. Nothing is ticked to begin with, because this "
        "is the section where a folder layout you rely on would be undone. The "
        "copy kept is the one on the left.",
        report.other_folder, preselect=False, full_paths=True)

    with st.expander(f"C · Similar names, different content "
                     f"({len(report.similar_name)} groups)"):
        st.caption(
            "Informational only — no ticks here, because these are NOT the same "
            "file: different edits, remixes or rips that happen to be named "
            "alike. Both paths are shown so you can tell them apart, and ▶ plays "
            "either one.")
        table_c = _rows(report.similar_name, preselect=False, full_paths=True)
        if table_c.empty:
            st.write("Nothing found.")
        else:
            play_table(f"simil::{root}", table_c.drop(columns=["Move", "md5"]),
                        ["stays", "moves if ticked", "size", "copies"],
                        {c: st.column_config.TextColumn(disabled=True, width="large")
                         for c in ("stays", "moves if ticked")}
                        | {"size": st.column_config.TextColumn(disabled=True)},
                        editable=False, editor_key=f"simil_editor::{root}")

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
        st.write(
            f"**{len(plan):,} files** ticked, freeing "
            f"**{human_size(bytes_a + bytes_b)}** — "
            f"{len(selected_a):,} from A ({human_size(bytes_a)}), "
            f"{len(selected_b):,} from B ({human_size(bytes_b)}).")
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


# --- Filtri sulla libreria -------------------------------------------------

st.divider()
st.header("Library filtering")
st.caption(
    "Files that do not belong in a collection meant for mixing single "
    "tracks: at the long end whole sets and megamixes, at the short end "
    "fragments, previews and truncated downloads. Two signals — how long "
    "the file is and what it is called — because for the mixes neither is "
    "enough alone; for the short ones the duration is the only one that "
    "works."
)

dur_key = f"durations::{root}"
if st.button(f"Read durations of {len(audio):,} files"):
    bar = st.progress(0.0, text="Reading headers…")
    st.session_state[dur_key] = read_durations(
        audio, progress=lambda done, total: bar.progress(
            done / total if total else 1.0, text=f"Read {done:,}/{total:,}…"))
    bar.empty()

durations = st.session_state.get(dur_key)
if durations is not None and durations.tracks:
    ceiling = max(30, int(durations.longest_minutes) + 1)
    # 10 e non 20: sulla libreria reale NIENTE supera i 20 minuti, e una
    # soglia che non trova mai nulla sembra un guasto. A 10 minuti passa
    # metà dei megamix e solo l'1% delle versioni extended (misurato).
    low, high = st.slider(
        "Duration between (minutes)", min_value=0, max_value=ceiling,
        value=(min(10, ceiling), ceiling),
        help="Moves instantly: the durations are already in memory. Around "
             "10 minutes is the useful lower line for megamixes — half of "
             "them are longer, while only 1% of extended versions are. It "
             "starts at 0 for the other end of the problem: 0–1 catches the "
             "fragments, previews and truncated downloads.")

    words_text = st.text_input(
        "…or the name contains", value=", ".join(DEFAULT_KEYWORDS),
        help="Comma separated, matched anywhere in the file name, case "
             "insensitive. Add your own.")
    keywords = parse_keywords(words_text)

    vs_rule = st.checkbox(
        'Also catch "A vs B - first song vs second song"', value=True,
        help="Two 'vs' in one name means two artists and two titles — a "
             "mashup. One 'vs' alone is usually a real collaboration, so it "
             "is not enough: measured here, one catches 2,312 files, two "
             "catch 679 and twelve out of twelve sampled were mashups.")

    with st.expander("Words deliberately left out, and why"):
        st.caption(
            "Measured on this library — these look like clues but the files "
            "carrying them are ordinary length. Add them above if you "
            "disagree, but check the list before moving anything.")
        for word, why in NOT_BY_DEFAULT.items():
            st.markdown(f"- **`{word.strip()}`** — {why}")

    # Di default i due criteri si sommano in AND, che è il verso più
    # prudente e — la ragione vera — l'unico che copre tutta la scala:
    # abbassando la soglia l'AND converge su "solo il nome", mentre l'OR non
    # può mai stringere al di sotto di quello. La differenza è grossa e va
    # detta: sulla cartella DMC 416 l'AND a 10 minuti trova 5 file, l'OR 24,
    # e fra quei 19 ci sono "Pink Hitmix" e "80s Hen Party Mix" — mix veri,
    # appena sotto i 10 minuti. Con l'AND si prendono abbassando la soglia.
    regola = st.radio(
        "A file is listed when…",
        ["both signals fire", "either one fires", "the duration alone"],
        horizontal=True,
        help="Both: long enough AND named like a mix — near-certain, and you "
             "widen it by lowering the slider until only the name matters. "
             "Either: the wider net, which also shows mashups — they are of "
             "ordinary length, so no duration will ever reach them. "
             "Duration alone: the name is ignored, which is the only way to "
             "reach the short files — nothing under a minute is ever going "
             "to be called a megamix.")

    by_length = {t.path for t in durations.between(low, high)}
    by_name: dict[Path, list[str]] = {}
    for track in durations.tracks:
        hits = matching_words(track.path.name, keywords)
        if hits:
            by_name.setdefault(track.path, []).append("name: " + ", ".join(hits))
        if vs_rule and looks_like_a_mashup(track.path.name):
            by_name.setdefault(track.path, []).append("two “vs”")

    reasons: dict[Path, list[str]] = {}
    for track in durations.tracks:
        long_enough, named = track.path in by_length, track.path in by_name
        if regola == "the duration alone":
            passa = long_enough
        elif regola == "both signals fire":
            passa = long_enough and named
        else:
            passa = long_enough or named
        if not passa:
            continue
        reasons[track.path] = (
            ([f"{low}–{high} min"] if long_enough else [])
            + by_name.get(track.path, []))

    long_tracks = sorted(
        (t for t in durations.tracks if t.path in reasons),
        key=lambda t: t.seconds, reverse=True)
    total_bytes = sum(t.size for t in long_tracks)

    l1, l2, l3, l4 = st.columns(4)
    l1.metric("Matching", f"{len(long_tracks):,}")
    l2.metric(f"{low}–{high} min", f"{len(by_length):,}")
    l3.metric("By name", f"{len(by_name):,}")
    l4.metric("Space", human_size(total_bytes))

    if durations.unknown:
        st.caption(f"{len(durations.unknown):,} file(s) had no readable duration "
                   "and are left out of this filter entirely.")

    if not long_tracks:
        st.info("Nothing matches either signal.")
    else:
        # Fra i file corti convivono due cose opposte: sample voluti e
        # canzoni il cui scaricamento si e' interrotto. I metadati non le
        # separano — misurato, un mp3 tagliato a meta' dichiara la durata
        # che ha davvero e ffprobe non protesta — quindi si guarda l'audio,
        # e solo su richiesta: va decodificato un file per volta.
        verdetti = st.session_state.get(f"cut::{root}")
        if any(t.seconds <= 60 for t in long_tracks):
            st.caption(
                f"{sum(1 for t in long_tracks if t.seconds <= 60):,} of these "
                "last under a minute, where deliberate samples and "
                "interrupted downloads sit side by side. Two things tell "
                "them apart: whether the file **ends at full volume** "
                "instead of dying away, and whether it still **claims to be "
                "a whole song** — ID3 tags live at the head of the file, so "
                "an interrupted download keeps its artist and title.")
            if st.button("Listen to how the short ones end"):
                corti = [t for t in long_tracks if t.seconds <= 60]
                bar = st.progress(0.0, text="Decoding…")
                fatti = {}
                for i, t in enumerate(corti, 1):
                    bar.progress(i / len(corti), text=f"{i}/{len(corti)} · {t.path.name[:50]}")
                    fatti[t.path] = inspect(t.path, t.seconds)
                bar.empty()
                st.session_state[f"cut::{root}"] = fatti
                st.rerun()

        epoch = st.session_state.setdefault(f"len_epoch::{root}", 0)
        table = pd.DataFrame([
            {"Move": True, "duration": human_duration(t.seconds),
             "why": " + ".join(reasons[t.path]),
             "verdict": (verdetti or {}).get(t.path).verdict
                        if (verdetti or {}).get(t.path) else "—",
             "file": t.path.name, "folder": str(t.path.parent),
             "size": human_size(t.size), "_path": str(t.path), "_bytes": t.size}
            for t in long_tracks
        ])
        c_all, c_none, c_count = st.columns([1, 1, 3])
        if c_all.button("Select all", key="all::len", width="stretch"):
            st.session_state[f"len_force::{root}"] = True
            st.session_state[f"len_epoch::{root}"] = epoch + 1
            st.rerun()
        if c_none.button("Select none", key="none::len", width="stretch"):
            st.session_state[f"len_force::{root}"] = False
            st.session_state[f"len_epoch::{root}"] = epoch + 1
            st.rerun()
        if f"len_force::{root}" in st.session_state:
            table["Move"] = st.session_state[f"len_force::{root}"]

        edited = play_table(
            f"long::{root}", table,
            ["Move", "duration", "verdict", "why", "file", "folder", "size"],
            {
                "Move": st.column_config.CheckboxColumn("Move"),
                "duration": st.column_config.TextColumn(disabled=True),
                "verdict": st.column_config.TextColumn(
                    "verdict", disabled=True,
                    help="Only for files under a minute, and only after the "
                         "button above: «cut off» ends at full volume AND "
                         "still carries artist and title, so it was meant to "
                         "be a whole song; «finished» dies away; «ends "
                         "sharp, untagged» is the honest middle — a hard-cut "
                         "excerpt looks the same as an interrupted download "
                         "when nothing says whose song it was."),
                "why": st.column_config.TextColumn(
                    "why", disabled=True, width="medium",
                    help="Which signal put this row here."),
                "file": st.column_config.TextColumn(disabled=True),
                "folder": st.column_config.TextColumn(disabled=True),
                "size": st.column_config.TextColumn(disabled=True),
            },
            editor_key=f"len_editor::{root}::{low}::{high}::{len(long_tracks)}::{vs_rule}::{regola}::{epoch}")
        picked = edited.loc[edited["Move"]]
        chosen = [Path(p) for p in picked["_path"]]
        chosen_bytes = int(picked["_bytes"].sum()) if not picked.empty else 0
        c_count.write("")
        c_count.caption(
            f"**{len(chosen):,}** of {len(table):,} selected · "
            f"**{human_size(chosen_bytes)}** would be freed "
            f"(of {human_size(total_bytes)} matching)")

        if chosen:
            long_plan = build_quarantine_plan(chosen, root)
            st.caption(
                f"Moved to `{root / QUARANTINE_DIRNAME}`, not deleted — a "
                "match might be a set you actually want, and you can put it "
                "back from there.")
            go = st.checkbox(
                f"Move these {len(long_plan):,} files to quarantine "
                f"({human_size(chosen_bytes)})", key="long_confirm")
            if st.button("Quarantine these", disabled=not go):
                moved, errors = apply_quarantine_plan(long_plan, dry_run=False)
                st.success(f"{moved:,} tracks moved.")
                if errors:
                    st.warning(f"{len(errors)} could not be moved.")
                st.session_state.pop(dur_key, None)
                st.session_state.pop(scan_key, None)
# --- File che non sono musica ----------------------------------------------

st.divider()
st.header("Junk files")
st.caption(
    "Small companion files macOS leaves beside the real ones, named "
    "`._<name>`. They carry the track's extension but hold no audio, and "
    "nothing plays them. Removing them frees space and keeps them out of "
    "the way."
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

    if not sidecars.looked_properly:
        st.error(
            f"**The folder could not be read**, so this says nothing about "
            f"what is in it: `{sidecars.root_error}`. A removable volume that "
            "denies access looks exactly like an empty one to a directory "
            "walk — press the button again once it is reachable.")
    elif not sidecars.confirmed:
        st.info(
            f"No sidecar files here — {sidecars.walked:,} entries walked and "
            "none of them was one.")
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


# --- File audio illeggibili -------------------------------------------------

st.divider()
st.header("Unreadable files")
st.caption(
    "Tracks that no player will open — usually a download that arrived "
    "truncated, or an error page saved with an .mp3 name. Unlike the junk "
    "files above these were meant to be music, so they are **moved to "
    "quarantine, not deleted**: the list is what you need in order to "
    "fetch them again."
)

# Secondi a file col controllo in parallelo, misurati su questa libreria via
# USB: 5,8 s per 800 file con 8 thread, contro 40,3 s in fila.
SECONDS_PER_CHECK = 0.0073

st.caption(
    "Every file is opened with the decoder — the same one the tagging uses. "
    f"About **{len(audio) * SECONDS_PER_CHECK / 60:.0f} min** for "
    f"{len(audio):,} files, {CHECK_THREADS} at a time. There is no "
    "headers-only shortcut any more: it was quick and wrong, condemning 31 "
    "tracks here that the decoder opens without complaint. This list decides "
    "what goes to quarantine, so a guess is not good enough."
)

integrity_key = f"integrity::{root}"
if st.button(f"Check {len(audio):,} audio files"):
    bar = st.progress(0.0, text="Checking…")
    st.session_state[integrity_key] = check_integrity(
        audio,
        progress=lambda done, total: bar.progress(
            done / total if total else 1.0, text=f"Checked {done:,}/{total:,}…"))
    bar.empty()

integrity = st.session_state.get(integrity_key)
if integrity is not None:
    i1, i2, i3 = st.columns(3)
    i1.metric("Checked", f"{integrity.checked:,}")
    i2.metric("Unreadable", f"{len(integrity.bad):,}",
              delta=human_size(sum(b.size for b in integrity.bad)), delta_color="off")
    i3.metric("Vanished", f"{len(integrity.missing):,}",
              help="Listed by the scan but gone by the time we looked — "
                   "moved or renamed in the meantime. Nothing to do.")

    if integrity.missing:
        with st.expander(f"{len(integrity.missing)} vanished since the scan"):
            st.dataframe(pd.DataFrame([{"path": str(p)} for p in integrity.missing]),
                         width="stretch", hide_index=True)

    if not integrity.bad:
        st.success("Every file opened correctly.")
    else:
        play_table(
            f"bad::{root}",
            pd.DataFrame([{"file": b.path.name, "folder": str(b.path.parent),
                           "size": human_size(b.size), "why": b.reason,
                           "_path": str(b.path)}
                          for b in integrity.bad]),
            ["file", "folder", "size", "why"],
            {c: st.column_config.TextColumn(disabled=True)
             for c in ("file", "folder", "size", "why")},
            editable=False, editor_key=f"bad_editor::{root}")
        bad_plan = build_quarantine_plan([b.path for b in integrity.bad], root)
        bad_bytes = sum(b.size for b in integrity.bad)
        st.caption(
            f"Moving them to `{root / QUARANTINE_DIRNAME}` keeps the folder "
            "structure, so you can see which album each came from and "
            "re-download just those.")
        ok = st.checkbox(
            f"Move these {len(bad_plan):,} unreadable files to quarantine "
            f"({human_size(bad_bytes)})", key="bad_confirm")
        if st.button("Quarantine unreadable files", disabled=not ok):
            moved, errors = apply_quarantine_plan(bad_plan, dry_run=False)
            st.success(f"{moved:,} files moved.")
            if errors:
                st.warning(f"{len(errors)} could not be moved.")
            st.session_state.pop(integrity_key, None)


