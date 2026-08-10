"""Entry point 2 — App Streamlit (fase di revisione).

Serve a validare a orecchio i phrase boundary proposti dalla segmentazione
strutturale (la parte meno affidabile dell'analisi). Riusa lo stesso motore e
la stessa cache del CLI.

    poetry run streamlit run app.py

I file mp3 richiedono ffmpeg installato a livello di sistema (brew install ffmpeg).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis.audio_features import ANALYSIS_SR, load_audio
from analysis.engine import AUDIO_EXTENSIONS, analyze_file
from analysis.models import SECTION_LABELS, format_elapsed, format_remaining
from analysis.sections import VOCAL_SECTION_COVER
from analysis.vocals import VOCAL_FLOOR, available as vocals_available
from analysis.vocals import vocal_regions
from analysis.waveform import compute_frequency_waveform


def _covered_fraction(start: float, end: float, regions) -> float:
    """Frazione di [start, end] coperta dalle regioni cantate."""
    length = end - start
    if length <= 0:
        return 0.0
    covered = sum(
        max(0.0, min(e, end) - max(st, start))
        for st, e in regions if e > start and st < end
    )
    return covered / length


def _live_regions(track: dict, floor: float):
    """Ricalcola le regioni cantate dalla soglia, usando l'inviluppo in cache."""
    ratio = track.get("vocal_ratio") or []
    fps = track.get("vocal_fps")
    if ratio and fps:
        times = np.arange(len(ratio)) / fps
        return vocal_regions((times, np.asarray(ratio, dtype=float)), floor=floor)
    return [tuple(r) for r in track.get("vocal_regions", [])]  # fallback: senza inviluppo

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
st.title("dj-library-tools — analisi di un brano")

st.caption(
    "Scegli un singolo brano già su disco, analizzalo e rivedi i risultati. "
    "L'analisi riusa la cache condivisa con il CLI."
)


@st.cache_data(show_spinner=False)
def _waveform_df(path_str: str, points: int = 1600) -> pd.DataFrame:
    """Waveform per bande di frequenza (colore) + ampiezza simmetrica."""
    y, sr = load_audio(Path(path_str), sr=ANALYSIS_SR, mono=True)
    t, amp, colors = compute_frequency_waveform(y, sr, points)
    return pd.DataFrame({"t": t, "amp": amp, "color": colors})


def _tag_text(label: str, start: float, duration, vocal: bool = False) -> str:
    """Etichetta del tag: tipo (con 🎤 se vocal), tempo dall'inizio e residuo."""
    mic = "🎤 " if vocal else ""
    return f"{mic}{label} {format_elapsed(start)} ({format_remaining(start, duration)})"


def _waveform_figure(path_str: str, sections, duration, vocal_regions=None) -> go.Figure:
    """Waveform colorata (Plotly) + regioni cantate + tag di sezione hot-cue."""
    df = _waveform_df(path_str)
    fig = go.Figure()

    if len(df):
        width = float(df["t"].iloc[1] - df["t"].iloc[0]) if len(df) > 1 else 0.1
        fig.add_bar(
            x=df["t"], y=(2 * df["amp"]), base=(-df["amp"]),
            marker=dict(color=list(df["color"])), width=width,
            hoverinfo="skip", showlegend=False,
        )

    xmax = duration if duration else (float(df["t"].max()) if len(df) else 1.0)

    # Regioni cantate: bande evidenziate (dove NON sovrapporre altre voci)
    for st_, en_ in (vocal_regions or []):
        fig.add_vrect(x0=st_, x1=en_, fillcolor="#ff5db1", opacity=0.18,
                      line_width=0, layer="below")

    if sections:
        xs = [s["start"] for s in sections]
        cols = [SECTION_COLORS.get(s["label"], "#ffffff") for s in sections]
        texts = [_tag_text(s["label"], s["start"], duration, s.get("vocal", False))
                 for s in sections]

        # Linee verticali tenui in corrispondenza dei tagli
        for x, c in zip(xs, cols):
            fig.add_vline(x=x, line=dict(color=c, width=1), opacity=0.35)

        # Triangoli hot-cue sull'asse orizzontale, colorati per tipo
        fig.add_scatter(
            x=xs, y=[-1.12] * len(xs), mode="markers",
            marker=dict(symbol="triangle-up", size=15, color=cols,
                        line=dict(color="#0f0f12", width=1)),
            hovertext=texts, hoverinfo="text", showlegend=False,
        )
        # Etichetta accanto a ogni triangolo (tipo + tempi), colore del tipo
        for x, c, txt in zip(xs, cols, texts):
            fig.add_annotation(
                x=x, y=-1.12, text=txt, showarrow=False,
                xanchor="left", yanchor="middle", xshift=8,
                font=dict(size=10, color=c),
            )

    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#0f0f12", plot_bgcolor="#0f0f12", bargap=0,
        xaxis=dict(title="secondi", range=[0, xmax], color="#bbb",
                   gridcolor="#2a2a2a", zeroline=False),
        yaxis=dict(visible=False, range=[-1.4, 1.05], fixedrange=True),
    )
    return fig


def _pick_file() -> None:
    """Apre il selettore di file nativo del Mac e riempie il campo del brano."""
    try:
        out = subprocess.run(
            ["osascript", "-e", 'POSIX path of (choose file with prompt "Scegli un brano")'],
            capture_output=True, text=True, check=True,
        )
        chosen = out.stdout.strip()
        if chosen:
            st.session_state["song"] = chosen
    except Exception:
        pass  # dialogo annullato o non disponibile: nessuna modifica


# --- Input brano + analisi ---
col_path, col_browse = st.columns([5, 1])
song = col_path.text_input("Percorso del brano (mp3/flac)", key="song")
col_browse.markdown("<div style='height:1.8em'></div>", unsafe_allow_html=True)
col_browse.button("🎵 Sfoglia…", on_click=_pick_file, use_container_width=True)
col_run, col_nocache, col_voc = st.columns([1, 2, 2])
with col_run:
    run = st.button("Analizza", type="primary")
with col_nocache:
    no_cache = st.checkbox("Ignora la cache (rianalizza)", value=False)
with col_voc:
    want_vocals = st.checkbox("Rileva voce (Demucs, lento)", value=vocals_available(),
                              disabled=not vocals_available())
if not vocals_available():
    st.caption("Demucs non installato: rilevamento voce disattivato. "
               "Installa con `poetry install`.")

if run:
    src = Path(song).expanduser()
    if not song or not src.is_file():
        st.error("Brano non valido: indica il percorso di un file.")
    elif src.suffix.lower() not in AUDIO_EXTENSIONS:
        st.error(f"Formato non supportato ({src.suffix}). Usa mp3 o flac.")
    else:
        with st.spinner(f"Analisi di {src.name} in corso… "
                        "(la prima volta con voce può richiedere qualche minuto)"):
            t = analyze_file(src, use_cache=not no_cache, detect_vocals=want_vocals)
        # Ripulisce gli slider di sezione della run precedente
        for k in [k for k in st.session_state if k.startswith(("st::", "lab::", "floor::"))]:
            del st.session_state[k]
        st.session_state["track"] = {
            "path": str(t.path),
            "name": t.path.name,
            "genre": t.genre,
            "vibe": t.vibe,
            "bpm": t.bpm,
            "duration": t.duration,
            "error": t.error,
            "sections": [s.to_dict() for s in t.sections],
            "vocal_regions": [list(r) for r in t.vocal_regions],
            "vocal_ratio": t.vocal_ratio,
            "vocal_fps": t.vocal_fps,
        }

track = st.session_state.get("track")
if not track:
    st.info("Scegli un brano e premi «Analizza» per iniziare.")
    st.stop()

st.success(f"Analizzato: {track['name']}")

bpm_txt = f"{track['bpm']:.0f}" if track["bpm"] is not None else "N/D"
st.markdown(f"**{track['genre']}** — {track['vibe']} — BPM {bpm_txt}")
if track["error"]:
    st.warning(f"Avviso in analisi: {track['error']}")

duration = track.get("duration")
path = track["path"]
bar_seconds = (4 * 60.0 / track["bpm"]) if track["bpm"] else None

# Ordine visivo: grafico, player+selettore, tabella cue, tabella cluster vocali.
# I controlli (soglia voce, slider sezioni) stanno sotto e riempiono questi
# contenitori, così tutto si aggiorna live.
chart_slot = st.empty()
player_container = st.container()
cues_container = st.container()
vocals_container = st.container()

# --- Soglia voce (live): ricalcola le regioni cantate senza rifare Demucs ---
has_env = bool(track.get("vocal_ratio"))
if has_env:
    vocal_floor = st.slider(
        "Soglia voce (dominanza voce/mix)", 0.0, 1.0, value=float(VOCAL_FLOOR),
        step=0.01, key=f"floor::{path}",
        help="Più alta = meno regioni cantate. Ricalcolo immediato, niente Demucs.",
    )
else:
    vocal_floor = VOCAL_FLOOR
regions_live = _live_regions(track, vocal_floor)

# --- Slider: sposta l'inizio o cambia l'etichetta di ogni tag ---
st.subheader("Tag delle sezioni")
st.caption(
    "Classificazione euristica da confermare a orecchio: sposta l'inizio o cambia "
    "l'etichetta; il 🎤 deriva dalla soglia voce qui sopra. Tutto si aggiorna live."
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
        c3.markdown(f"`{format_elapsed(start_s)} ({format_remaining(start_s, duration)})`")
        edited.append({"start": start_s, "label": label})

    # Riordina per posizione, ricalcola durate, battute e flag vocal (da soglia)
    edited.sort(key=lambda d: d["start"])
    starts = [d["start"] for d in edited]
    ends = starts[1:] + [duration]
    for d, end in zip(edited, ends):
        length = max(0.0, end - d["start"])
        d["bars"] = (length / bar_seconds) if bar_seconds else None
        d["vocal"] = _covered_fraction(d["start"], end, regions_live) >= VOCAL_SECTION_COVER
else:
    st.info("Nessuna sezione rilevata per questa traccia.")

# --- Grafico (in cima) con sezioni e regioni cantate correnti ---
chart_slot.plotly_chart(
    _waveform_figure(path, edited, duration, regions_live),
    use_container_width=True,
)

# --- Player + selettore di porzione (sezioni + parti vocali), sotto il grafico ---
cue_points = [(s["start"], f'{format_remaining(s["start"], duration)} — {s["label"]}')
              for s in edited]
cue_points += [(st_, f'{format_remaining(st_, duration)} — 🎤 Voce')
               for st_, _ in regions_live]
cue_points.sort(key=lambda c: c[0])
with player_container:
    if cue_points:
        pick = st.selectbox("Vai a…", options=range(len(cue_points)),
                            format_func=lambda i: cue_points[i][1])
        start = int(cue_points[pick][0])
    else:
        start = 0
    try:
        audio_path = Path(path)
        mime = "audio/flac" if audio_path.suffix.lower() == ".flac" else "audio/mp3"
        st.audio(audio_path.read_bytes(), format=mime, start_time=start)
    except Exception as e:
        st.error(f"Impossibile aprire l'audio: {e}")

# --- Tabella dei cue (sezioni) ---
if edited:
    rows = [{
        "tag": f'{"🎤 " if s.get("vocal") else ""}{s["label"]}',
        "dall_inizio": format_elapsed(s["start"]),
        "restante": format_remaining(s["start"], duration),
        "start_s": round(s["start"], 2),
        "bars": round(s["bars"], 1) if s.get("bars") else "",
        "vocal": "sì" if s.get("vocal") else "",
    } for s in edited]
    report_df = pd.DataFrame(rows)
    cues_container.dataframe(report_df, use_container_width=True, hide_index=True)
    cues_container.download_button(
        "Scarica sezioni (CSV)",
        data=report_df.to_csv(index=False).encode(),
        file_name=f"{Path(track['name']).stem}_sections.csv",
        mime="text/csv",
    )

# --- Tabella dei cluster vocali (sotto quella dei cue) ---
with vocals_container:
    st.markdown("**Cluster vocali** 🎤")
    if regions_live:
        vrows = [{
            "n": i + 1,
            "inizio": format_elapsed(st_),
            "fine": format_elapsed(en_),
            "durata_s": round(en_ - st_, 1),
            "restante_inizio": format_remaining(st_, duration),
        } for i, (st_, en_) in enumerate(regions_live)]
        vocals_df = pd.DataFrame(vrows)
        st.dataframe(vocals_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Scarica cluster vocali (CSV)",
            data=vocals_df.to_csv(index=False).encode(),
            file_name=f"{Path(track['name']).stem}_vocals.csv",
            mime="text/csv",
        )
    elif has_env:
        st.caption("Nessuna parte vocale rilevata a questa soglia.")
    else:
        st.caption("Rilevamento voce non eseguito (Demucs disattivato in analisi).")
