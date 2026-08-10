"""Entry point 2 — App Streamlit (fase di revisione).

Serve a validare a orecchio i phrase boundary proposti dalla segmentazione
strutturale (la parte meno affidabile dell'analisi). Riusa lo stesso motore e
la stessa cache del CLI.

    poetry run streamlit run app.py

I file mp3 richiedono ffmpeg installato a livello di sistema (brew install ffmpeg).
"""

from __future__ import annotations

import base64
import hashlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from analysis.audio_features import ANALYSIS_SR, load_audio
from analysis.engine import AUDIO_EXTENSIONS, analyze_file, load_analysis
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

st.set_page_config(page_title="Wavecut", page_icon="🌊", layout="wide")
st.title("🌊 Wavecut — analisi di un brano")

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


@st.cache_data(show_spinner=False)
def _audio_data_uri(path_str: str, mtime: float) -> str | None:
    """Prepara un mp3 compatto (ffmpeg) da incorporare nel player, in cache."""
    src = Path(path_str)
    token = hashlib.md5(f"{path_str}:{mtime}".encode()).hexdigest()
    tmp = Path(tempfile.gettempdir()) / f"wavecut_{token}.mp3"
    try:
        if not tmp.exists():
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-b:a", "96k",
                 "-f", "mp3", str(tmp)],
                check=True, capture_output=True,
            )
        return "data:audio/mpeg;base64," + base64.b64encode(tmp.read_bytes()).decode()
    except Exception:
        if src.suffix.lower() == ".mp3":   # fallback: incorpora l'mp3 originale
            try:
                return "data:audio/mpeg;base64," + base64.b64encode(src.read_bytes()).decode()
            except Exception:
                return None
        return None


# --- Componente CCv2: waveform interattiva + audio sincronizzato ---
_PLAYER_HTML = """
<div class="wc-wrap" style="width:100%;font-family:system-ui,-apple-system,sans-serif;">
  <canvas class="wc-canvas" style="width:100%;height:200px;display:block;
    border-radius:6px;cursor:pointer;background:#0f0f12;"></canvas>
  <div style="display:flex;align-items:center;gap:12px;margin-top:6px;">
    <audio class="wc-audio" controls preload="auto" style="flex:1;height:34px;"></audio>
    <span class="wc-time" style="color:#bbb;font-size:12px;
      font-variant-numeric:tabular-nums;white-space:nowrap;">00:00.0 / 00:00.0</span>
  </div>
</div>
"""

_PLAYER_JS = """
export default function (component) {
  const { data, parentElement } = component;
  const root = parentElement;

  function fmt(t) {
    if (!isFinite(t) || t < 0) t = 0;
    const m = Math.floor(t / 60);
    const s = (t - m * 60);
    return String(m).padStart(2, "0") + ":" + s.toFixed(1).padStart(4, "0");
  }

  function buildWave(ui) {
    const cv = ui.canvas, d = ui.data;
    const dpr = window.devicePixelRatio || 1;
    ui.dpr = dpr;
    const cw = cv.clientWidth || 600, ch = cv.clientHeight || 200;
    cv.width = Math.round(cw * dpr);
    cv.height = Math.round(ch * dpr);
    const off = document.createElement("canvas");
    off.width = cv.width; off.height = cv.height;
    const ctx = off.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.fillStyle = "#0f0f12"; ctx.fillRect(0, 0, cw, ch);
    const dur = d.duration || 1, mid = ch / 2;
    (d.regions || []).forEach(([s, e]) => {
      const x0 = s / dur * cw, x1 = e / dur * cw;
      ctx.fillStyle = "rgba(255,93,177,0.18)";
      ctx.fillRect(x0, 0, Math.max(1, x1 - x0), ch);
    });
    const amp = d.amp || [], col = d.colors || [], n = amp.length;
    ctx.lineWidth = Math.max(1, cw / Math.max(1, n));
    for (let i = 0; i < n; i++) {
      const x = i / n * cw, h = amp[i] * mid;
      ctx.strokeStyle = col[i] || "#888";
      ctx.beginPath(); ctx.moveTo(x, mid - h); ctx.lineTo(x, mid + h); ctx.stroke();
    }
    (d.sections || []).forEach(sec => {
      const x = sec.t / dur * cw;
      ctx.strokeStyle = sec.color; ctx.globalAlpha = 0.5;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, ch); ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = sec.color;
      ctx.beginPath(); ctx.moveTo(x - 4, ch - 1); ctx.lineTo(x + 4, ch - 1);
      ctx.lineTo(x, ch - 9); ctx.closePath(); ctx.fill();
      ctx.font = "10px system-ui"; ctx.fillText(sec.label, x + 5, 12);
    });
    ui.wave = off;
  }

  function draw(ui) {
    const cv = ui.canvas;
    if (!ui.data) return;
    if (Math.abs(cv.clientWidth * (ui.dpr || 1) - cv.width) > 1) buildWave(ui);
    const ctx = cv.getContext("2d");
    if (ui.wave) ctx.drawImage(ui.wave, 0, 0);
    const dur = ui.data.duration || 1;
    const dpr = ui.dpr || 1;
    const x = (ui.audio.currentTime / dur) * cv.clientWidth * dpr;
    ctx.save();
    ctx.strokeStyle = "#ffe14d"; ctx.lineWidth = 2 * dpr;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, cv.height); ctx.stroke();
    ctx.restore();
    ui.timeEl.textContent = fmt(ui.audio.currentTime) + " / " + fmt(dur);
  }

  let ui = root.__wc;
  if (!ui) {
    const canvas = root.querySelector(".wc-canvas");
    const audio = root.querySelector(".wc-audio");
    const timeEl = root.querySelector(".wc-time");
    ui = root.__wc = { canvas, audio, timeEl, data: null, wave: null, raf: null, dpr: 1 };
    canvas.addEventListener("click", (e) => {
      const r = canvas.getBoundingClientRect();
      if (ui.data && ui.data.duration) {
        ui.audio.currentTime = ((e.clientX - r.left) / r.width) * ui.data.duration;
      }
    });
    const loop = () => { draw(ui); ui.raf = requestAnimationFrame(loop); };
    ui.raf = requestAnimationFrame(loop);
  }
  ui.data = data;
  if (ui.audio.dataset.aid !== data.audioId) {   // cambia sorgente solo se serve
    ui.audio.src = data.audio;
    ui.audio.dataset.aid = data.audioId;
  }
  buildWave(ui);

  return () => { if (ui.raf) cancelAnimationFrame(ui.raf); };
}
"""

_WAVE_PLAYER = st.components.v2.component(
    "wavecut_player", html=_PLAYER_HTML, js=_PLAYER_JS,
)


def wave_player(path: str, sections, regions, duration, key: str) -> None:
    """Monta il player interattivo (waveform + audio sincronizzato)."""
    mtime = Path(path).stat().st_mtime
    audio = _audio_data_uri(path, mtime)
    if audio is None:
        st.warning("Player interattivo non disponibile (serve ffmpeg): uso il player base.")
        mime = "audio/flac" if path.lower().endswith(".flac") else "audio/mp3"
        st.audio(Path(path).read_bytes(), format=mime)
        return
    df = _waveform_df(path)
    sec = [{"t": float(s["start"]),
            "label": ("🎤 " if s.get("vocal") else "") + s["label"],
            "color": SECTION_COLORS.get(s["label"], "#ffffff")} for s in sections]
    _WAVE_PLAYER(key=key, data={
        "amp": [round(float(a), 3) for a in df["amp"].tolist()],
        "colors": df["color"].tolist(),
        "duration": float(duration or 0.0),
        "sections": sec,
        "regions": [[float(a), float(b)] for a, b in regions],
        "audio": audio,
        "audioId": f"{path}:{mtime}",
    })


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
    force = st.checkbox("Force analysis if exists", value=False,
                        help="Rianalizza anche se esiste già il file <nome>_analysis.json")
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
        existing = None if force else load_analysis(src)
        if existing is not None:
            t, from_file = existing, True
        else:
            with st.spinner(f"Analisi di {src.name} in corso… "
                            "(la prima volta con voce può richiedere qualche minuto)"):
                t = analyze_file(src, use_cache=not force, detect_vocals=want_vocals)
            from_file = False
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
            "from_file": from_file,
        }

track = st.session_state.get("track")
if not track:
    st.info("Scegli un brano e premi «Analizza» per iniziare.")
    st.stop()

if track.get("from_file"):
    st.success(f"Caricato da {Path(track['name']).stem}_analysis.json: {track['name']}")
else:
    st.success(f"Analizzato: {track['name']}")

bpm_txt = f"{track['bpm']:.0f}" if track["bpm"] is not None else "N/D"
st.markdown(f"**{track['genre']}** — {track['vibe']} — BPM {bpm_txt}")
if track["error"]:
    st.warning(f"Avviso in analisi: {track['error']}")

duration = track.get("duration")
path = track["path"]
bar_seconds = (4 * 60.0 / track["bpm"]) if track["bpm"] else None

# Ordine visivo: player interattivo, tabella cue, tabella cluster vocali.
# I controlli (soglia voce, slider sezioni) stanno sotto e riempiono questi
# contenitori, così tutto si aggiorna live.
player_slot = st.container()
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

# --- Player interattivo (waveform + audio sincronizzato), sotto l'header ---
with player_slot:
    if duration:
        wave_player(path, edited, regions_live, duration, key=f"player::{path}")
        st.caption("Clic sull'onda per spostarti al punto · ▶ per ascoltare · "
                   "linea gialla = testina, bande rosa = voce, triangoli = sezioni.")
    else:
        st.info("Durata non disponibile: player non mostrabile.")

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
