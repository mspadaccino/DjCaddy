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

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis.audio_features import ANALYSIS_SR, load_audio
from analysis.engine import analyze_library, discover_tracks
from analysis.models import SECTION_LABELS, format_elapsed, format_remaining
from analysis.vocals import available as vocals_available
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


def _pick_folder() -> None:
    """Apre il selettore di cartella nativo del Mac e riempie il campo path."""
    try:
        out = subprocess.run(
            ["osascript", "-e", "POSIX path of (choose folder)"],
            capture_output=True, text=True, check=True,
        )
        chosen = out.stdout.strip().rstrip("/")
        if chosen:
            st.session_state["folder"] = chosen
    except Exception:
        pass  # dialogo annullato o non disponibile: nessuna modifica


# --- Input cartella + analisi ---
col_path, col_browse = st.columns([5, 1])
folder = col_path.text_input("Percorso della cartella locale con gli mp3", key="folder")
col_browse.markdown("<div style='height:1.8em'></div>", unsafe_allow_html=True)
col_browse.button("📂 Sfoglia…", on_click=_pick_folder, use_container_width=True)
col_run, col_nocache, col_voc = st.columns([1, 2, 2])
with col_run:
    run = st.button("Analizza", type="primary")
with col_nocache:
    no_cache = st.checkbox("Ignora la cache (rianalizza tutto)", value=False)
with col_voc:
    want_vocals = st.checkbox("Rileva voce (Demucs, lento)", value=vocals_available(),
                              disabled=not vocals_available())
if not vocals_available():
    st.caption("Demucs non installato: rilevamento voce disattivato "
               "(puoi comunque taggare i vocal a mano). Installa con `poetry install`.")

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

        tracks = analyze_library(src, use_cache=not no_cache, progress=_cb,
                                 detect_vocals=want_vocals)
        progress.empty()
        # Ripulisce gli slider/selectbox di sezione della run precedente
        for k in [k for k in st.session_state if k.startswith(("st::", "lab::", "voc::"))]:
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
                "vocal_regions": [list(r) for r in t.vocal_regions],
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

# Grafico e tabella sono in cima ma vengono riempiti DOPO gli slider,
# così riflettono le modifiche in tempo reale.
chart_slot = st.empty()
table_container = st.container()

# --- Slider: sposta l'inizio, cambia l'etichetta, marca i vocal ---
st.subheader("Tag delle sezioni")
st.caption(
    "Classificazione euristica da confermare a orecchio: sposta l'inizio, cambia "
    "l'etichetta o spunta 🎤 se c'è voce; il grafico e la tabella si aggiornano."
)

edited: list[dict] = []
if track["sections"] and duration:
    for i, s in enumerate(track["sections"]):
        c1, c2, c3, c4 = st.columns([2, 4, 1, 1])
        idx = SECTION_LABELS.index(s["label"]) if s["label"] in SECTION_LABELS else len(SECTION_LABELS) - 1
        label = c1.selectbox(f"Tag {i + 1}", SECTION_LABELS, index=idx,
                             key=f"lab::{path}::{i}")
        start_s = c2.slider(f"Inizio {i + 1} (s)", 0.0, float(duration),
                            value=float(s["start"]), step=0.5, key=f"st::{path}::{i}")
        vocal = c3.checkbox("🎤", value=bool(s.get("vocal")), key=f"voc::{path}::{i}")
        c4.markdown(f"`{format_elapsed(start_s)} ({format_remaining(start_s, duration)})`")
        edited.append({"start": start_s, "label": label, "vocal": vocal})

    # Riordina per posizione, ricalcola durate e battute
    edited.sort(key=lambda d: d["start"])
    starts = [d["start"] for d in edited]
    ends = starts[1:] + [duration]
    for d, end in zip(edited, ends):
        length = max(0.0, end - d["start"])
        d["bars"] = (length / bar_seconds) if bar_seconds else None
else:
    st.info("Nessuna sezione rilevata per questa traccia.")

# Riempie il grafico in cima con le sezioni (eventualmente) modificate.
# Le regioni cantate (bande rosa) restano quelle rilevate: sono la parte da
# non sovrapporre ad altre voci in mixaggio.
chart_slot.plotly_chart(
    _waveform_figure(path, edited, duration, track.get("vocal_regions")),
    use_container_width=True,
)

# Tabella riepilogativa subito sotto il grafico
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
    table_container.dataframe(report_df, use_container_width=True, hide_index=True)
    table_container.download_button(
        "Scarica sezioni riviste (CSV)",
        data=report_df.to_csv(index=False).encode(),
        file_name=f"{Path(track['name']).stem}_sections.csv",
        mime="text/csv",
    )

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
