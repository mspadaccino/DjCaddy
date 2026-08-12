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
from analysis.cue_export import (
    DJAY_SLOTS, PHRASE_START, build_cue_rows, is_vocal_row, plan_djay_markers,
)
from analysis.dj_export import build_rekordbox_xml, read_title_artist
from analysis.djay_write import CuePoint as DjayCuePoint
from analysis.djay_write import LoopRegion as DjayLoopRegion
from analysis.djay_write import DjayWriteError, is_djay_running
from analysis.djay_write import preview_write as djay_preview_write
from analysis.djay_write import write_new_cues as djay_write_new_cues
from analysis.models import (
    SECTION_COLORS,
    SECTION_LABELS,
    VOCAL_END,
    VOCAL_MARKER_COLORS,
    VOCAL_START,
    format_elapsed,
    format_remaining,
    parse_mmss,
)
from analysis.vocals import VOCAL_FLOOR, available as vocals_available
from analysis.vocals import vocal_regions
from analysis.waveform import compute_frequency_waveform

# Una frase è una riga sola (il suo inizio); la fine è informazione nella
# colonna "End", non un tag a sé — vedi analysis.cue_export.build_cue_rows.
TAG_OPTIONS = list(SECTION_LABELS) + [VOCAL_START, VOCAL_END]


def _marker_color(kind: str, tag: str) -> str:
    if kind in ("vocal_start", "vocal_end"):
        return VOCAL_MARKER_COLORS.get(tag, "#ffffff")
    base = tag.rsplit(" ", 1)[0] if tag.endswith((" start", " end")) else tag
    return SECTION_COLORS.get(base, "#ffffff")


def _live_regions(track: dict, floor: float):
    """Ricalcola le regioni cantate dalla soglia, usando l'inviluppo in cache."""
    ratio = track.get("vocal_ratio") or []
    fps = track.get("vocal_fps")
    if ratio and fps:
        times = np.arange(len(ratio)) / fps
        return vocal_regions((times, np.asarray(ratio, dtype=float)), floor=floor)
    return [tuple(r) for r in track.get("vocal_regions", [])]  # fallback: senza inviluppo

st.set_page_config(page_title="Wavecut — Phrase analyzer", page_icon="🌊", layout="wide")
st.title("🌊 Wavecut — Phrase analyzer")

st.caption(
    "Pick a single track already on disk, analyze it and review the results. "
    "Analysis reuses the same cache as the CLI."
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
<div class="wc-wrap" style="width:100%;font-family:system-ui,-apple-system,sans-serif;position:relative;">
  <canvas class="wc-canvas" style="width:100%;height:200px;display:block;
    border-radius:6px;cursor:pointer;background:#0f0f12;"></canvas>
  <div class="wc-tooltip" style="position:absolute;top:6px;transform:translateX(-50%);
    background:rgba(15,15,18,0.92);color:#eee;font-size:11px;padding:3px 7px;
    border-radius:4px;pointer-events:none;white-space:nowrap;display:none;
    font-variant-numeric:tabular-nums;border:1px solid #333;"></div>
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

  function fmtPair(t, dur) {
    return fmt(t) + " (-" + fmt(Math.max(0, dur - t)) + ")";
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
      ctx.fillStyle = "rgba(255,93,177,0.32)";
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
    const tooltip = root.querySelector(".wc-tooltip");
    ui = root.__wc = { canvas, audio, timeEl, tooltip, data: null, wave: null, raf: null, dpr: 1 };
    canvas.addEventListener("click", (e) => {
      const r = canvas.getBoundingClientRect();
      if (ui.data && ui.data.duration) {
        ui.audio.currentTime = ((e.clientX - r.left) / r.width) * ui.data.duration;
      }
    });
    canvas.addEventListener("mousemove", (e) => {
      const r = canvas.getBoundingClientRect();
      if (!ui.data || !ui.data.duration) return;
      const frac = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
      const t = frac * ui.data.duration;
      ui.tooltip.textContent = fmtPair(t, ui.data.duration);
      ui.tooltip.style.left = (e.clientX - r.left) + "px";
      ui.tooltip.style.display = "block";
    });
    canvas.addEventListener("mouseleave", () => { ui.tooltip.style.display = "none"; });
    const loop = () => { draw(ui); ui.raf = requestAnimationFrame(loop); };
    ui.raf = requestAnimationFrame(loop);
  }
  ui.data = data;
  if (ui.audio.dataset.aid !== data.audioId) {   // cambia sorgente solo se serve
    ui.audio.src = data.audio;
    ui.audio.dataset.aid = data.audioId;
  }
  if (data.seekToken !== undefined && data.seekToken !== ui.lastSeekToken) {
    ui.lastSeekToken = data.seekToken;
    if (data.seekTo !== undefined && data.seekTo !== null) {
      ui.audio.currentTime = data.seekTo;
      ui.audio.play().catch(() => {});
    }
  }
  buildWave(ui);

  return () => { if (ui.raf) cancelAnimationFrame(ui.raf); };
}
"""

_WAVE_PLAYER = st.components.v2.component(
    "wavecut_player", html=_PLAYER_HTML, js=_PLAYER_JS,
)


def wave_player(path: str, markers, regions, duration, key: str,
                seek_to: float | None = None, seek_token: int = 0) -> None:
    """Monta il player interattivo (waveform + audio sincronizzato).

    `markers` è una lista di {"t", "label", "color"} pronta per il disegno
    (sezioni + marcatori vocali inizio/fine, dalla tabella cue unificata).
    `regions` sono le bande (span) delle parti cantate, per il colpo d'occhio.
    """
    mtime = Path(path).stat().st_mtime
    audio = _audio_data_uri(path, mtime)
    if audio is None:
        st.warning("Interactive player unavailable (ffmpeg required): falling back to basic player.")
        mime = "audio/flac" if path.lower().endswith(".flac") else "audio/mp3"
        st.audio(Path(path).read_bytes(), format=mime)
        return
    df = _waveform_df(path)
    _WAVE_PLAYER(key=key, data={
        "amp": [round(float(a), 3) for a in df["amp"].tolist()],
        "colors": df["color"].tolist(),
        "duration": float(duration or 0.0),
        "sections": markers,
        "regions": [[float(a), float(b)] for a, b in regions],
        "audio": audio,
        "audioId": f"{path}:{mtime}",
        "seekTo": seek_to,
        "seekToken": seek_token,
    })


def _pick_file() -> None:
    """Apre il selettore di file nativo del Mac e riempie il campo del brano."""
    try:
        out = subprocess.run(
            ["osascript", "-e", 'POSIX path of (choose file with prompt "Choose a track")'],
            capture_output=True, text=True, check=True,
        )
        chosen = out.stdout.strip()
        if chosen:
            st.session_state["song"] = chosen
    except Exception:
        pass  # dialogo annullato o non disponibile: nessuna modifica


# --- Track input + analysis ---
col_path, col_browse = st.columns([5, 1])
song = col_path.text_input("Track path (mp3/flac)", key="song")
col_browse.markdown("<div style='height:1.8em'></div>", unsafe_allow_html=True)
col_browse.button("🎵 Browse…", on_click=_pick_file, width="stretch")
col_run, col_nocache, col_voc = st.columns([1, 2, 2])
with col_run:
    run = st.button("Analyze", type="primary")
with col_nocache:
    force = st.checkbox("Force analysis if exists", value=False,
                        help="Re-analyze even if a <name>_analysis.json file already exists")
with col_voc:
    want_vocals = st.checkbox("Detect vocals (Demucs, slow)", value=vocals_available(),
                              disabled=not vocals_available())
if not vocals_available():
    st.caption("Demucs not installed: vocal detection disabled. "
               "Install it with `poetry install`.")

if run:
    src = Path(song).expanduser()
    if not song or not src.is_file():
        st.error("Invalid track: please provide the path to a file.")
    elif src.suffix.lower() not in AUDIO_EXTENSIONS:
        st.error(f"Unsupported format ({src.suffix}). Use mp3 or flac.")
    else:
        existing = None if force else load_analysis(src)
        if existing is not None:
            t, from_file = existing, True
        else:
            with st.spinner(f"Analyzing {src.name}… "
                            "(the first run with vocals can take a few minutes)"):
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
    st.info("Pick a track and press “Analyze” to start.")
    st.stop()

duration = track.get("duration")
path = track["path"]
bar_seconds = (4 * 60.0 / track["bpm"]) if track["bpm"] else None

# --- Vocal threshold (live), right below the Analyze controls above ---
has_env = bool(track.get("vocal_ratio"))
if has_env:
    vocal_floor = st.slider(
        "Vocal threshold (voice/mix dominance)", 0.0, 1.0, value=float(VOCAL_FLOOR),
        step=0.01, key=f"floor::{path}",
        help="Higher = fewer sung regions. Recomputed instantly, no Demucs.",
    )
else:
    vocal_floor = VOCAL_FLOOR
regions_live = _live_regions(track, vocal_floor)

if track.get("from_file"):
    st.success(f"Loaded from {Path(track['name']).stem}_analysis.json: {track['name']}")
else:
    st.success(f"Analyzed: {track['name']}")

bpm_txt = f"{track['bpm']:.0f}" if track["bpm"] is not None else "N/A"
st.markdown(f"**{track['genre']}** — {track['vibe']} — BPM {bpm_txt}")
if track["error"]:
    st.warning(f"Analysis warning: {track['error']}")

player_slot = st.container()

beat_seconds = (60.0 / track["bpm"]) if track["bpm"] else None

# --- Stato persistito: righe cancellate dalla tabella, per-traccia ---
# Frasi: id "sec{i}s"/"sec{i}e" (start/end indipendenti, persistono finché
# non si ri-analizza). Vocali: id "vs{j}"/"ve{j}", si rigenerano da zero a
# ogni cambio soglia, quindi le cancellazioni valgono solo per il set corrente.
deleted_rows = st.session_state.setdefault(f"deleted_rows::{path}", set())
deleted_vocal = st.session_state.setdefault(
    f"deleted_vocal::{path}::{round(vocal_floor, 2)}", set()
)

# --- Costruisce le righe base (default) della tabella unificata: una riga
# per frase (il suo inizio, con la fine come informazione) e due per ogni
# regione vocale, che del suo inizio E della sua fine ha bisogno davvero. ---
base_rows: list[dict] = []
analysis_end: dict[str, float] = {}   # fine rilevata, usata solo per l'ultima frase
if duration:
    default_rows = build_cue_rows(track["sections"], regions_live, track["bpm"])
    for r in default_rows:
        rid = r["id"]
        if is_vocal_row(r):
            if rid in deleted_vocal:
                continue
        elif rid in deleted_rows:
            continue
        start_key = f"start::{path}::{rid}"
        label_key = f"label::{path}::{rid}"
        st.session_state.setdefault(start_key, r["start"])
        st.session_state.setdefault(label_key, r["label"])
        if r["end"] is not None:
            analysis_end[rid] = r["end"]
        base_rows.append({
            "_id": rid, "_kind": r["kind"],
            "Tag": st.session_state[label_key],
            "Start": st.session_state[start_key],
        })


def _phrase_ends(kinds: dict, starts: dict) -> dict:
    """Fine di ogni frase: l'inizio della frase successiva, perché le sezioni
    sono contigue — così la colonna resta coerente anche dopo che l'utente ha
    spostato un inizio. Per l'ultima frase resta la fine rilevata
    dall'analisi. Le righe vocali non hanno una fine propria: la loro sta
    nella riga "Vocal end" gemella."""
    order = sorted((rid for rid, k in kinds.items() if k == PHRASE_START),
                   key=lambda rid: starts[rid])
    return {
        rid: (starts[order[i + 1]] if i + 1 < len(order) else analysis_end.get(rid))
        for i, rid in enumerate(order)
    }

# --- Ordina per tempo corrente (solo per la visualizzazione: l'id resta
# l'indice stabile, così Streamlit riaggancia gli edit alla riga giusta anche
# se la posizione visiva cambia) ---
if base_rows:
    df = pd.DataFrame(base_rows).set_index("_id").sort_values("Start")

    # Dove finirà ogni riga in djay Pro (frasi -> hot cue, regioni vocali ->
    # slot loop), in ordine di tempo. Ricalcolato a ogni run: dopo un edit
    # dei tempi l'assegnazione può cambiare.
    df["Slot"] = df.index.map(plan_djay_markers(
        [{"id": rid, "kind": row["_kind"], "start": row["Start"]}
         for rid, row in df.iterrows()]
    ).slot_label)

    df["Play"] = "▶"
    df["Del"] = "🗑"
    df.insert(0, "#", range(1, len(df) + 1))

    # Stash per i callback dei bottoni (che non vedono le variabili locali di
    # questo run): id in ordine di visualizzazione + Start corrente (secondi)
    # di ognuno — sempre numerico, indipendentemente da come viene mostrato.
    st.session_state[f"row_order::{path}"] = df.index.tolist()
    for rid, val in zip(df.index, df["Start"]):
        st.session_state[f"row_start::{path}::{rid}"] = float(val)

    # Fine della frase: solo informativa, non diventa un cue in djay Pro.
    ends = _phrase_ends(df["_kind"].to_dict(), df["Start"].to_dict())
    df["End"] = [
        format_elapsed(ends[rid]) if ends.get(rid) is not None else ""
        for rid in df.index
    ]

    # Solo per la visualizzazione/edit in tabella: mm:ss invece di secondi.
    df["Start"] = df["Start"].map(format_elapsed)

    play_key = f"play_col::{path}"
    del_key = f"del_col::{path}"

    def _on_play_click(path=path, play_key=play_key):
        click = st.session_state.get(play_key)
        if not click:
            return
        ids = st.session_state.get(f"row_order::{path}", [])
        pos = click["row"]
        if 0 <= pos < len(ids):
            rid = ids[pos]
            start_val = st.session_state.get(f"row_start::{path}::{rid}")
            if start_val is not None:
                st.session_state[f"seek_to::{path}"] = float(start_val)
                st.session_state[f"seek_token::{path}"] = (
                    st.session_state.get(f"seek_token::{path}", 0) + 1
                )

    def _on_delete_click(path=path, del_key=del_key):
        click = st.session_state.get(del_key)
        if not click:
            return
        ids = st.session_state.get(f"row_order::{path}", [])
        pos = click["row"]
        if 0 <= pos < len(ids):
            rid = ids[pos]
            if rid.startswith(("vs", "ve")):
                st.session_state[f"deleted_vocal::{path}::{round(vocal_floor, 2)}"].add(rid)
            else:
                st.session_state[f"deleted_rows::{path}"].add(rid)

    edited_df = st.data_editor(
        df,
        key=f"cue_table::{path}",
        hide_index=True,
        column_order=["#", "Play", "Tag", "Start", "End", "Beats", "Slot", "Del"],
        column_config={
            "#": st.column_config.NumberColumn(disabled=True),
            "Play": st.column_config.ButtonColumn(
                "▶", on_click=_on_play_click, key=play_key, width="small",
            ),
            "Tag": st.column_config.SelectboxColumn(options=TAG_OPTIONS, required=True),
            "Start": st.column_config.TextColumn(
                "Start (mm:ss)", help="Formato mm:ss o mm:ss.d, es. 1:07.3",
            ),
            "End": st.column_config.TextColumn(
                "End (mm:ss)", disabled=True,
                help="Dove finisce la frase, solo come riferimento: NON "
                     "diventa un cue in djay Pro. Le sezioni sono contigue, "
                     "quindi coincide con l'inizio della frase successiva. "
                     "Le righe vocali hanno la loro fine nella riga gemella.",
            ),
            "Beats": st.column_config.NumberColumn(
                disabled=True, help="Lunghezza della frase (solo sul tag start), "
                                     "ricalcolata dalla riga end gemella.",
            ),
            "Slot": st.column_config.TextColumn(
                "djay slot", disabled=True,
                help="Dove finisce questa riga in djay Pro: gli INIZI di frase "
                     "sui pad hot-cue 1-8 (la posizione decide il colore), le "
                     "regioni vocali sugli slot loop 1-8 (inizio e fine nello "
                     "stesso slot). Assegnati in ordine di tempo. Vuoto = non "
                     "viene scritta: le FINI di frase cadono sull'inizio della "
                     "frase successiva, quindi sarebbero doppioni, e oltre "
                     "l'ottavo slot non c'è più posto.",
            ),
            "Del": st.column_config.ButtonColumn(
                "🗑", on_click=_on_delete_click, key=del_key, width="small",
            ),
        },
    )

    # Riconverte Start da mm:ss a secondi; in caso di formato non valido
    # mantiene il valore precedente della riga invece di far cadere l'app.
    bad_rows = 0
    parsed_starts = {}
    for rid, row in edited_df.iterrows():
        val = parse_mmss(str(row["Start"]))
        if val is None:
            val = st.session_state.get(f"row_start::{path}::{rid}", 0.0)
            bad_rows += 1
        parsed_starts[rid] = val
    if bad_rows:
        st.warning(f"Formato non valido (atteso mm:ss) per {bad_rows} riga/e: "
                   "valore precedente mantenuto.")
    edited_df["Start"] = edited_df.index.map(parsed_starts).astype(float)

    # Beats: solo per le frasi, ricalcolato dalla fine corrente (cioè
    # dall'inizio della frase successiva, dopo gli edit di questo run).
    starts_by_id = edited_df["Start"].to_dict()
    ends_by_id = _phrase_ends(edited_df["_kind"].to_dict(), starts_by_id)

    def _beats_for(rid: str, kind: str) -> float | None:
        end = ends_by_id.get(rid)
        if kind != PHRASE_START or not beat_seconds or end is None:
            return None
        return round((end - starts_by_id[rid]) / beat_seconds, 1)

    edited_df["Beats"] = [
        _beats_for(rid, row["_kind"]) for rid, row in edited_df.iterrows()
    ]

    # Riporta gli edit nello stato persistito (id già univoco per ruolo, non
    # serve più separare sezioni/vocali qui).
    for rid, row in edited_df.iterrows():
        st.session_state[f"start::{path}::{rid}"] = float(row["Start"])
        st.session_state[f"label::{path}::{rid}"] = row["Tag"]
else:
    edited_df = pd.DataFrame(columns=["_kind", "Tag", "Start", "End", "Beats", "Slot"])
    st.info("No cues to show for this track.")

markers = [
    {"t": float(row["Start"]), "label": row["Tag"], "color": _marker_color(row["_kind"], row["Tag"])}
    for _, row in edited_df.iterrows()
]

# --- Interactive player (waveform + synced audio), sopra la tabella ---
with player_slot:
    if duration:
        wave_player(
            path, markers, regions_live, duration, key=f"player::{path}",
            seek_to=st.session_state.get(f"seek_to::{path}"),
            seek_token=st.session_state.get(f"seek_token::{path}", 0),
        )
        st.caption("Click the waveform to jump to a point, or ▶ a row below · "
                   "yellow line = playhead, pink bands = vocal regions.")
    else:
        st.info("Duration unavailable: player can't be shown.")

# --- Opzione condivisa fra export ed export/scrittura djay ---
vocals_only = st.checkbox(
    "Save only vocal tags", value=True,
    help="If checked, CSV/XML export and the djay Pro write below only "
         "include the vocal start/end rows, skipping the phrase tags.",
)
export_df = edited_df[edited_df["_kind"].isin(["vocal_start", "vocal_end"])] \
    if vocals_only else edited_df

# --- Export (CSV + rekordbox XML) dalla tabella così com'è ora ---
if not export_df.empty:
    csv_df = export_df.reset_index(drop=True)[["Tag", "Start", "Beats"]].copy()
    csv_df["from_start"] = csv_df["Start"].map(format_elapsed)
    csv_df["remaining"] = csv_df["Start"].map(lambda s: format_remaining(s, duration))
    dl_col, xml_col = st.columns(2)
    dl_col.download_button(
        "Download cues (CSV)",
        data=csv_df.to_csv(index=False).encode(),
        file_name=f"{Path(track['name']).stem}_cues.csv",
        mime="text/csv",
    )
    title, artist = read_title_artist(Path(path))
    xml_str = build_rekordbox_xml([{
        "path": Path(path), "name": title, "artist": artist, "genre": track["genre"],
        "bpm": track["bpm"], "duration": duration,
        "cues": [{"name": row["Tag"], "start": float(row["Start"]),
                  "color": _marker_color(row["_kind"], row["Tag"])}
                for _, row in export_df.iterrows()],
    }])
    xml_col.download_button(
        "Export cues to rekordbox XML",
        data=xml_str.encode("utf-8"),
        file_name=f"{Path(track['name']).stem}_rekordbox.xml",
        mime="application/xml",
        help="Import directly into rekordbox, or convert to Serato/Traktor/djay Pro "
             "with a third-party tool (DJ Conversion Utility, MIXO, Lexicon).",
    )
elif not edited_df.empty:
    st.caption("No vocal rows to export with the current threshold/edits.")

# --- Scrittura diretta nel database live di djay Pro (Mac) ---
st.divider()
st.subheader("Update your djay Pro library")
st.caption(
    f"Writes the rows above (or only the vocal ones, per the checkbox above) "
    f"straight into your djay Pro library — no XML export/import needed. "
    f"djay Pro has two independent banks of {DJAY_SLOTS}: **phrase starts "
    f"become hot cues** (pad position sets the colour), and **each vocal "
    f"region becomes one saved loop**, holding its start and end in a single "
    f"slot instead of burning two pads. Phrase *ends* aren't written — a "
    f"section ends exactly where the next one starts, so they'd just be "
    f"duplicate cues. Slots are handed out in time order; the **djay slot** "
    f"column shows where each row lands, and rows past the {DJAY_SLOTS}th are "
    f"left out."
)
overwrite_tags = st.checkbox(
    "Overwrite tags for analyzed songs", value=False,
    help="If checked, existing cues/loops on this track in djay Pro are "
         "replaced by the rows below, instead of adding to them. Unlike "
         "adding (verified against real before/after data), removing "
         "has NOT been verified the same way — treat it as best-effort.",
)
if overwrite_tags:
    st.warning(
        "Overwrite is unverified for the 'removing' case (see caption "
        "above) — a full backup is still taken automatically, but double-check "
        "the preview below carefully before writing."
    )

if not export_df.empty:
    # Il piano viene ricalcolato QUI sugli edit correnti (export_df), non
    # riusato da quello mostrato in tabella: i tempi possono essere cambiati.
    plan = plan_djay_markers(
        [{"id": rid, "kind": row["_kind"], "start": float(row["Start"])}
         for rid, row in export_df.iterrows()]
    )
    # Le frasi vanno sui pad hot-cue: in djay Pro il colore è dato dalla
    # POSIZIONE nell'array cuePoints (1° = pad 1/rosso, 2° = pad 2/arancio,
    # ...), confermato incrociando tempi reali su tutti e 8 i pad.
    djay_cues = [DjayCuePoint(time=start, pad=pad) for _, pad, start in plan.cues]
    # Ogni regione vocale è un loop salvato: inizio e fine in un solo slot.
    djay_loops = [DjayLoopRegion(start=start, end=end, slot=slot)
                  for _, slot, start, end in plan.loops]

    if plan.dropped:
        st.warning(
            f"{len(plan.dropped)} row(s) don't fit: djay Pro only has "
            f"{DJAY_SLOTS} hot-cue pads and {DJAY_SLOTS} loop slots, so the "
            "latest ones in the track are left out."
        )
    if plan.unpaired:
        st.warning(
            f"{len(plan.unpaired)} vocal marker(s) have no matching start/end "
            "and can't become a loop — a loop needs both ends."
        )

    preview_key = f"djay_preview::{path}"

    if st.button("Preview djay Pro update"):
        try:
            result = djay_preview_write(Path(path), djay_cues,
                                        overwrite=overwrite_tags,
                                        new_loops=djay_loops)
            st.session_state[preview_key] = {
                "ok": True, "result": result, "cues": djay_cues,
                "loops": djay_loops, "overwrite": overwrite_tags,
            }
        except DjayWriteError as e:
            st.session_state[preview_key] = {"ok": False, "error": str(e)}

    preview = st.session_state.get(preview_key)
    if preview:
        if not preview["ok"]:
            st.error(preview["error"])
        else:
            res = preview["result"]
            verb = "would replace" if preview["overwrite"] else "would be added to"
            st.success(
                f"Track found in djay Pro ({len(res.cues_before)} existing cue(s), "
                f"{len(res.loops_before)} existing loop(s)). "
                f"{len(djay_cues)} hot cue(s) and {len(djay_loops)} loop(s) "
                f"below {verb} them."
            )
            st.dataframe(
                pd.DataFrame(
                    [{"slot": f"Cue {c.pad + 1}", "start": format_elapsed(c.time), "end": ""}
                     for c in djay_cues]
                    + [{"slot": f"Loop {lr.slot + 1}", "start": format_elapsed(lr.start),
                        "end": format_elapsed(lr.end)} for lr in djay_loops]
                ),
                width="stretch", hide_index=True,
            )
            if is_djay_running():
                st.warning("djay Pro appears to be running — quit it before writing, "
                          "to avoid conflicts with its own auto-save.")
            if st.button("Write to djay Pro now", type="primary"):
                try:
                    write_res = djay_write_new_cues(
                        Path(path), preview["cues"], overwrite=preview["overwrite"],
                        new_loops=preview["loops"],
                    )
                    st.success(f"{write_res.message} Backup saved to: {write_res.backup_path}")
                    del st.session_state[preview_key]
                except DjayWriteError as e:
                    st.error(f"Write failed, nothing was changed: {e}")
elif not edited_df.empty:
    st.caption("No vocal rows to write with the current threshold/edits.")
