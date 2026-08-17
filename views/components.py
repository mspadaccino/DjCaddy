"""Pezzi di interfaccia usati da più sezioni.

Stanno qui perché la scelta di una cartella e l'ascolto di una riga servono
uguali in Folder analysis e in Tag analysis, e duplicarli significherebbe
correggerli due volte.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

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
        st.audio(str(track), format=mime)
    except Exception as e:
        st.warning(f"Could not play it: {e}")
