"""Pezzi di interfaccia usati da più sezioni.

Stanno qui perché la scelta di una cartella e l'ascolto di una riga servono
uguali in Folder analysis e in Tag analysis, e duplicarli significherebbe
correggerli due volte.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

NOW_PLAYING = "components::now_playing"

# Formati che il browser sa riprodurre da solo, col tipo MIME da dichiarare.
# Dichiararlo evita di affidarsi a come il browser indovina il contenuto:
# senza, Streamlit serve un mp3 con estensione .wav e funziona solo perché
# Chrome guarda dentro al file.
PLAYABLE = {
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4", ".mp4": "audio/mp4", ".aac": "audio/mp4",
    ".ogg": "audio/ogg", ".oga": "audio/ogg", ".opus": "audio/ogg",
}


def pick_folder(state_key: str, label: str = "Folder",
                placeholder: str = "/Volumes/Crucial X9/DJSet") -> Path | None:
    """Campo di testo più il selettore di cartelle nativo del Mac.

    Streamlit non ha un dialogo di cartelle — il browser non dà accesso al
    filesystem — ma l'app gira in locale, quindi si può chiedere al Finder.
    """
    def _browse() -> None:
        try:
            out = subprocess.run(
                ["osascript", "-e",
                 'POSIX path of (choose folder with prompt "Choose a folder")'],
                capture_output=True, text=True, check=True,
            )
            chosen = out.stdout.strip()
            if chosen:
                st.session_state[state_key] = chosen.rstrip("/")
        except Exception:
            pass   # dialogo annullato o non disponibile: nessuna modifica

    col_path, col_browse = st.columns([5, 1])
    typed = col_path.text_input(label, key=state_key, placeholder=placeholder)
    col_browse.markdown("<div style='height:1.8em'></div>", unsafe_allow_html=True)
    col_browse.button("📁 Browse…", on_click=_browse, width="stretch",
                      key=f"browse::{state_key}")

    if not typed.strip():
        return None
    folder = Path(typed).expanduser()
    if not folder.is_dir():
        st.error(f"Not a folder: {folder}")
        return None
    return folder


def play_table(section: str, table: pd.DataFrame, column_order: list[str],
               column_config: dict, editable: bool = True,
               editor_key: str | None = None) -> pd.DataFrame:
    """Tabella con una colonna ▶ per riga e un lettore sotto.

    Serve a sentire un file PRIMA di deciderne la sorte. Il lettore carica
    solo il file cliccato: la colonna audio nativa di Streamlit vorrebbe
    invece un URL per ogni riga, e generarli legge OGNI file per intero in
    memoria a ogni rerun — misurato, 2,5 ms e l'intero contenuto per riga,
    cioè giga di traffico su una tabella di qualche centinaio di righe.
    """
    order_key, click_key = f"order::{section}", f"click::{section}"
    st.session_state[order_key] = list(table["_path"])

    def _on_play(order_key=order_key, click_key=click_key):
        click = st.session_state.get(click_key)
        order = st.session_state.get(order_key, [])
        if click and 0 <= click.get("row", -1) < len(order):
            st.session_state[NOW_PLAYING] = order[click["row"]]

    shown = table.copy()
    shown.insert(0, "Play", "▶")
    config = {"Play": st.column_config.ButtonColumn(
        "▶", on_click=_on_play, key=click_key, width="small"), **column_config}

    edited = st.data_editor(
        shown, key=editor_key, width="stretch", hide_index=True,
        column_order=["Play", *column_order], column_config=config,
        **({} if editable else {"disabled": column_order}),
    )
    render_player(set(table["_path"]))
    return edited


# Quante colonne disegna l'onda, e a che frequenza si legge l'audio per
# ricavarla. Mille campioni al secondo sono un millesimo di quelli veri e
# bastano per il PROFILO: quello che si guarda qui e' dove il brano sale e
# dove stacca, non la forma della singola oscillazione.
WAVE_POINTS = 800
WAVE_RATE = 1000

_PLAYER_HTML = """
<div class="wp">
  <button class="pp" type="button" aria-label="Play or pause">&#9654;</button>
  <canvas class="wave"></canvas>
  <span class="clock">0:00</span>
</div>
"""

_PLAYER_CSS = """
.wp {
  display: flex; align-items: center; gap: 0.75rem;
  background: var(--st-secondary-background-color, #f0f2f6);
  border-radius: var(--st-base-radius, 0.5rem);
  padding: 0.5rem 0.75rem;
}
.wp .pp {
  flex: 0 0 auto; width: 2.25rem; height: 2.25rem;
  border: none; border-radius: 50%; cursor: pointer;
  background: var(--st-primary-color, #ff4b4b);
  color: var(--st-background-color, #fff);
  font-size: 0.9rem; line-height: 1;
}
.wp canvas { flex: 1 1 auto; height: 84px; cursor: pointer; display: block; }
.wp .clock {
  /* nowrap piu' una larghezza minima: senza, "5:33 / 7:23" andava a capo e
     si leggeva solo meta' del tempo. */
  flex: 0 0 auto; white-space: nowrap; min-width: 5.5rem; text-align: right;
  font-variant-numeric: tabular-nums;
  font-size: 0.8rem; color: var(--st-text-color, #333); opacity: 0.75;
}
"""

# L'onda E' la barra di avanzamento: la parte gia' suonata si colora, e un
# clic sposta la riproduzione in quel punto. Un lettore nativo affiancato a
# un grafico non lo permetterebbe — la sua barra non e' allineata all'asse
# dell'onda, e sovrapporli sarebbe solo decorazione.
_PLAYER_JS = """
export default function (component) {
  const { data, parentElement } = component
  const root = parentElement.querySelector(".wp")
  if (!root || !data || !data.url) return

  const canvas = root.querySelector("canvas")
  const button = root.querySelector(".pp")
  const clock = root.querySelector(".clock")
  const peaks = data.peaks || []
  const total = data.duration || 0

  if (!root._audio) root._audio = new Audio()
  const audio = root._audio

  const clock_text = (s) => {
    if (!isFinite(s)) s = 0
    const m = Math.floor(s / 60)
    return m + ":" + String(Math.floor(s % 60)).padStart(2, "0")
  }

  const draw = () => {
    const style = getComputedStyle(root)
    const done_color = style.getPropertyValue("--st-primary-color").trim() || "#ff4b4b"
    const rest_color = style.getPropertyValue("--st-text-color").trim() || "#808495"
    const ratio = window.devicePixelRatio || 1
    const width = canvas.clientWidth, height = canvas.clientHeight
    if (!width || !height) return
    canvas.width = width * ratio
    canvas.height = height * ratio
    const ctx = canvas.getContext("2d")
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0)
    ctx.clearRect(0, 0, width, height)

    const played = total ? (audio.currentTime / total) * peaks.length : 0
    const step = width / peaks.length
    const middle = height / 2
    for (let i = 0; i < peaks.length; i++) {
      const bar = Math.max(1, peaks[i] * (height - 4))
      ctx.fillStyle = i <= played ? done_color : rest_color
      ctx.globalAlpha = i <= played ? 1 : 0.35
      ctx.fillRect(i * step, middle - bar / 2, Math.max(step - 0.5, 0.5), bar)
    }
    ctx.globalAlpha = 1
    clock.textContent = clock_text(audio.currentTime) + " / " + clock_text(total)
  }

  if (root._url !== data.url) {
    root._url = data.url
    audio.src = data.url
    // il pulsante della riga E' la richiesta di ascoltare: si parte da soli
    audio.play().catch(() => {})
  }

  audio.ontimeupdate = draw
  audio.onplay = () => { button.innerHTML = "&#10073;&#10073;"; draw() }
  audio.onpause = () => { button.innerHTML = "&#9654;" }
  audio.onended = () => { button.innerHTML = "&#9654;" }
  button.onclick = () => { audio.paused ? audio.play().catch(() => {}) : audio.pause() }
  canvas.onclick = (e) => {
    const box = canvas.getBoundingClientRect()
    const where = (e.clientX - box.left) / box.width
    if (total) { audio.currentTime = Math.min(total, Math.max(0, where * total)); draw() }
  }
  if (root._watching !== true) {
    root._watching = true
    new ResizeObserver(draw).observe(canvas)
  }
  button.innerHTML = audio.paused ? "&#9654;" : "&#10073;&#10073;"
  draw()
}
"""

_wave_player = st.components.v2.component(
    "wavecut_wave_player", html=_PLAYER_HTML, css=_PLAYER_CSS, js=_PLAYER_JS)


@st.cache_data(show_spinner=False, max_entries=64)
def _envelope(path: str, mtime: float, size: int):
    """Il profilo di ampiezza del brano, o None se non si riesce a leggerlo.

    Decodifica con ffmpeg e non con librosa: misurato sullo stesso brano da
    17 MB, 0,37 s contro 1,31 s, e ffprobe e' gia' quello che usa il
    controllo di integrita'. `mtime` e `size` non si usano nel corpo — sono
    li' per la cache, che cosi' rilegge il file se cambia.

    Se ffmpeg manca o il file e' illeggibile si torna None: sopra si ricade
    sul lettore normale, perche' l'onda e' un di piu' e non deve poter
    impedire l'ascolto.
    """
    try:
        raw = subprocess.run(
            ["ffmpeg", "-v", "quiet", "-i", path, "-ac", "1",
             "-ar", str(WAVE_RATE), "-f", "s16le", "-"],
            capture_output=True, timeout=120).stdout
    except Exception:
        return None

    samples = np.frombuffer(raw, dtype="<i2")
    if samples.size < WAVE_POINTS:
        return None
    usable = samples.size - samples.size % WAVE_POINTS
    peaks = np.abs(samples[:usable].reshape(WAVE_POINTS, -1)).max(axis=1)
    loudest = peaks.max() or 1
    return (peaks / loudest).round(3).tolist(), samples.size / WAVE_RATE


def _media_url(track: Path, mime: str) -> str | None:
    """L'indirizzo con cui il browser scarica il brano.

    E' lo stesso registro che usa `st.audio` dietro le quinte, quindi non si
    paga niente in piu' del lettore normale: senza, l'unico modo di dare il
    file al componente sarebbe infilarcelo dentro codificato, e su un brano
    da 17 MB sarebbero 23 MB a ogni ridisegno.
    """
    from streamlit import runtime

    if not runtime.exists():
        return None
    try:
        return runtime.get_instance().media_file_mgr.add(
            str(track), mime, f"wave-player::{track}")
    except Exception:
        return None


def render_player(paths_here: set) -> None:
    """Il lettore del file scelto, se appartiene a questa tabella."""
    current = st.session_state.get(NOW_PLAYING)
    if current is None or current not in paths_here:
        return
    track = Path(current)
    if not track.exists():
        st.warning(f"Not there any more: {track.name}")
        return
    st.caption(f"▶ {track.name}")
    mime = PLAYABLE.get(track.suffix.lower())
    if mime is None:
        st.info(f"The browser cannot play {track.suffix} files.")
        return

    try:
        stat = track.stat()
        wave = _envelope(str(track), stat.st_mtime, stat.st_size)
    except OSError:
        wave = None
    url = _media_url(track, mime) if wave else None

    if wave and url:
        peaks, seconds = wave
        _wave_player(data={"url": url, "peaks": peaks, "duration": seconds},
                     key=f"wave::{track}", height=110)
        return

    try:
        st.audio(str(track), format=mime, autoplay=True)
    except Exception as e:
        st.warning(f"Could not play it: {e}")
