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
import numpy as np
import pandas as pd
import streamlit as st

from analysis.audio_features import ANALYSIS_SR, load_audio
from analysis.engine import analyze_library, discover_tracks

st.set_page_config(page_title="dj-library-tools — revisione", layout="wide")
st.title("dj-library-tools — revisione dei phrase boundary")

st.caption(
    "I file sono già su disco: indica il percorso della cartella, non caricare nulla. "
    "L'analisi riusa la cache condivisa con il CLI."
)


@st.cache_data(show_spinner=False)
def _waveform_envelope(path_str: str, points: int = 2000):
    """Inviluppo di ampiezza (max assoluto per blocco) per il disegno."""
    y, sr = load_audio(Path(path_str), sr=ANALYSIS_SR, mono=True)
    if y.size == 0:
        return np.array([]), np.array([])
    hop = max(1, y.size // points)
    n = (y.size // hop) * hop
    env = np.abs(y[:n]).reshape(-1, hop).max(axis=1)
    times = (np.arange(env.size) * hop) / sr
    return times, env


def _waveform_chart(path_str: str, boundaries) -> alt.LayerChart:
    times, env = _waveform_envelope(path_str)
    wave_df = pd.DataFrame({"t": times, "amp": env})
    wave = (
        alt.Chart(wave_df)
        .mark_area(opacity=0.6)
        .encode(
            x=alt.X("t:Q", title="secondi"),
            y=alt.Y("amp:Q", title="ampiezza", scale=alt.Scale(domain=[0, 1])),
        )
    )
    if boundaries:
        b_df = pd.DataFrame(
            {"t": [b["time"] for b in boundaries],
             "label": [f'{b["label"]} ({b["confidence"]:.2f})' for b in boundaries]}
        )
        rules = (
            alt.Chart(b_df)
            .mark_rule(color="red", strokeWidth=2)
            .encode(x="t:Q", tooltip=["t:Q", "label:N"])
        )
        return (wave + rules).properties(height=220)
    return wave.properties(height=220)


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
        st.warning("Nessun file mp3 trovato nella cartella.")
    else:
        progress = st.progress(0.0, text="Analisi in corso...")

        def _cb(i: int, total: int, path: Path) -> None:
            progress.progress(i / total, text=f"[{i}/{total}] {path.name}")

        tracks = analyze_library(src, use_cache=not no_cache, progress=_cb)
        progress.empty()
        st.session_state["tracks"] = [
            {
                "path": str(t.path),
                "name": t.path.name,
                "genre": t.genre,
                "vibe": t.vibe,
                "bpm": t.bpm,
                "error": t.error,
                "boundaries": [b.to_dict() for b in t.boundaries],
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

# --- Forma d'onda con boundary sovrapposti ---
st.altair_chart(_waveform_chart(track["path"], track["boundaries"]),
                use_container_width=True)

# --- Player al punto esatto ---
boundaries = track["boundaries"]
if boundaries:
    labels = [f'{b["time"]:.1f}s — {b["label"]} ({b["confidence"]:.2f})' for b in boundaries]
    pick = st.selectbox("Ascolta dal boundary", options=range(len(boundaries)),
                        format_func=lambda i: labels[i])
    start = int(boundaries[pick]["time"])
else:
    st.info("Nessun phrase boundary rilevato per questa traccia.")
    start = 0

try:
    audio_bytes = Path(track["path"]).read_bytes()
    st.audio(audio_bytes, format="audio/mp3", start_time=start)
except Exception as e:
    st.error(f"Impossibile aprire l'audio: {e}")

# --- Correzione / conferma dei marker ---
st.subheader("Correggi o conferma i marker")
st.caption("Modifica i secondi, aggiungi o togli righe, poi scarica i marker validati.")
editor_df = pd.DataFrame(
    boundaries or [{"time": 0.0, "confidence": 0.0, "label": ""}]
)
edited = st.data_editor(editor_df, num_rows="dynamic", use_container_width=True,
                        key=f"editor_{track['path']}")
st.download_button(
    "Scarica marker (CSV)",
    data=edited.to_csv(index=False).encode(),
    file_name=f"{Path(track['name']).stem}_cues.csv",
    mime="text/csv",
)
