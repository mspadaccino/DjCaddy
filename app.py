"""Entry point 2 — App Streamlit (fase di revisione).

Serve a validare a orecchio i phrase boundary proposti dalla segmentazione
strutturale (la parte meno affidabile dell'analisi). Riusa lo stesso motore e
la stessa cache del CLI.

    poetry run streamlit run app.py

I file mp3 richiedono ffmpeg installato a livello di sistema (brew install ffmpeg).
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from analysis.audio_features import ANALYSIS_SR, load_audio
from analysis.engine import analyze_library, discover_tracks
from analysis.models import SECTION_LABELS, format_remaining
from analysis.waveform import compute_frequency_waveform

# Colori dei tag di sezione (accostati ai marker triangolari di djay Pro)
SECTION_COLORS = {
    "Intro": "#8e9aa6",
    "Build-up": "#f2a33c",
    "Drop/Chorus": "#e0503b",
    "Breakdown": "#3d9be0",
    "Outro": "#8e9aa6",
    "Groove": "#3fbf7f",
}

st.set_page_config(page_title="dj-library-tools — revisione", layout="wide")
st.title("dj-library-tools — revisione delle sezioni")

st.caption(
    "I file sono già su disco: indica il percorso della cartella, non caricare nulla. "
    "L'analisi riusa la cache condivisa con il CLI."
)


@st.cache_data(show_spinner=False)
def _waveform_df(path_str: str, points: int = 1600) -> pd.DataFrame:
    """Waveform per bande di frequenza (colore) + ampiezza simmetrica."""
    y, sr = load_audio(Path(path_str), sr=ANALYSIS_SR, mono=True)
    t, amp, colors = compute_frequency_waveform(y, sr, points)
    return pd.DataFrame({"t": t, "amp": amp, "namp": -amp, "color": colors})


def _waveform_chart(path_str: str, sections, duration) -> alt.TopLevelMixin:
    """Waveform colorata + tag di sezione (linea colorata + etichetta in alto)."""
    wave_df = _waveform_df(path_str)
    wave = (
        alt.Chart(wave_df)
        .mark_rule(strokeWidth=1)
        .encode(
            x=alt.X("t:Q", title="secondi"),
            y=alt.Y("amp:Q", title=None, scale=alt.Scale(domain=[-1, 1]), axis=None),
            y2="namp:Q",
            color=alt.Color("color:N", scale=None, legend=None),
        )
    )
    layers = [wave]
    if sections:
        sdf = pd.DataFrame({
            "start": [s["start"] for s in sections],
            "restante": [format_remaining(s["start"], duration) for s in sections],
            "label": [s["label"] for s in sections],
            "color": [SECTION_COLORS.get(s["label"], "#ffffff") for s in sections],
            "ypos": [0.9] * len(sections),
        })
        rules = (
            alt.Chart(sdf)
            .mark_rule(strokeWidth=2, opacity=0.9)
            .encode(x="start:Q", color=alt.Color("color:N", scale=None, legend=None),
                    tooltip=["label:N", "restante:N"])
        )
        tags = (
            alt.Chart(sdf)
            .mark_text(align="left", dx=3, dy=0, fontSize=11, fontWeight="bold")
            .encode(
                x="start:Q",
                y=alt.Y("ypos:Q", scale=alt.Scale(domain=[-1, 1]), axis=None),
                text="label:N",
                color=alt.Color("color:N", scale=None, legend=None),
            )
        )
        layers += [rules, tags]

    return (
        alt.layer(*layers)
        .properties(height=240)
        .configure(background="#0f0f12")
        .configure_view(strokeOpacity=0)
        .configure_axis(labelColor="#bbb", titleColor="#bbb",
                        gridColor="#2a2a2a", domainColor="#2a2a2a")
    )


# --- Input cartella + analisi ---
folder = st.text_input("Percorso della cartella locale con gli mp3", value="")
col_run, col_nocache = st.columns([1, 2])
with col_run:
    run = st.button("Analizza", type="primary")
with col_nocache:
    no_cache = st.checkbox("Ignora la cache (rianalizza tutto)", value=False)

if run:
    src = Path(folder).expanduser()
    if not folder or not src.is_dir():
        st.error("Cartella non valida.")
    elif not discover_tracks(src):
        st.warning("Nessun file mp3/flac trovato nella cartella.")
    else:
        progress = st.progress(0.0, text="Analisi in corso...")

        def _cb(i: int, total: int, path: Path) -> None:
            progress.progress(i / total, text=f"[{i}/{total}] {path.name}")

        tracks = analyze_library(src, use_cache=not no_cache, progress=_cb)
        progress.empty()
        # Ripulisce gli slider/selectbox di sezione della run precedente
        for k in [k for k in st.session_state if k.startswith(("st::", "lab::"))]:
            del st.session_state[k]
        st.session_state["tracks"] = [
            {
                "path": str(t.path),
                "name": t.path.name,
                "genre": t.genre,
                "vibe": t.vibe,
                "bpm": t.bpm,
                "duration": t.duration,
                "error": t.error,
                "sections": [s.to_dict() for s in t.sections],
            }
            for t in tracks
        ]

tracks = st.session_state.get("tracks")
if not tracks:
    st.info("Indica una cartella e premi «Analizza» per iniziare.")
    st.stop()

st.success(f"{len(tracks)} tracce analizzate.")

# --- Selezione traccia ---
names = [t["name"] for t in tracks]
sel = st.selectbox("Traccia da revisionare", options=range(len(names)),
                   format_func=lambda i: names[i])
track = tracks[sel]

bpm_txt = f"{track['bpm']:.0f}" if track["bpm"] is not None else "N/D"
st.markdown(f"**{track['genre']} / {track['vibe']}** — BPM {bpm_txt}")
if track["error"]:
    st.warning(f"Avviso in analisi: {track['error']}")

duration = track.get("duration")
path = track["path"]
bar_seconds = (4 * 60.0 / track["bpm"]) if track["bpm"] else None

# La waveform viene riempita DOPO gli slider, così riflette le modifiche.
chart_slot = st.empty()

# --- Slider: sposta l'inizio di ogni tag o cambia l'etichetta ---
st.subheader("Tag delle sezioni")
st.caption(
    "Classificazione euristica da confermare a orecchio: sposta l'inizio di una "
    "sezione o cambia l'etichetta; il grafico e il report si aggiornano."
)

edited: list[dict] = []
if track["sections"] and duration:
    for i, s in enumerate(track["sections"]):
        c1, c2, c3 = st.columns([2, 4, 1])
        idx = SECTION_LABELS.index(s["label"]) if s["label"] in SECTION_LABELS else len(SECTION_LABELS) - 1
        label = c1.selectbox(f"Tag {i + 1}", SECTION_LABELS, index=idx,
                             key=f"lab::{path}::{i}")
        start_s = c2.slider(f"Inizio {i + 1} (s)", 0.0, float(duration),
                            value=float(s["start"]), step=0.5, key=f"st::{path}::{i}")
        c3.markdown(f"`{format_remaining(start_s, duration)}`")
        edited.append({"start": start_s, "label": label})

    # Riordina per posizione, ricalcola durate e battute
    edited.sort(key=lambda d: d["start"])
    starts = [d["start"] for d in edited]
    ends = starts[1:] + [duration]
    for d, end in zip(edited, ends):
        length = max(0.0, end - d["start"])
        d["bars"] = (length / bar_seconds) if bar_seconds else None
else:
    st.info("Nessuna sezione rilevata per questa traccia.")

# Riempie il grafico in cima con le sezioni (eventualmente) modificate
chart_slot.altair_chart(_waveform_chart(path, edited, duration),
                        use_container_width=True)

# --- Player dall'inizio di una sezione ---
if edited:
    opts = [f'{format_remaining(s["start"], duration)} — {s["label"]}' for s in edited]
    pick = st.selectbox("Ascolta dall'inizio sezione", options=range(len(edited)),
                        format_func=lambda i: opts[i])
    start = int(edited[pick]["start"])
else:
    start = 0

try:
    audio_path = Path(path)
    mime = "audio/flac" if audio_path.suffix.lower() == ".flac" else "audio/mp3"
    st.audio(audio_path.read_bytes(), format=mime, start_time=start)
except Exception as e:
    st.error(f"Impossibile aprire l'audio: {e}")

# --- Report rivisto delle sezioni ---
if edited:
    rows = [{
        "label": s["label"],
        "restante": format_remaining(s["start"], duration),
        "start_s": round(s["start"], 2),
        "bars": round(s["bars"], 1) if s.get("bars") else "",
    } for s in edited]
    report_df = pd.DataFrame(rows)
    st.dataframe(report_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Scarica sezioni riviste (CSV)",
        data=report_df.to_csv(index=False).encode(),
        file_name=f"{Path(track['name']).stem}_sections.csv",
        mime="text/csv",
    )
