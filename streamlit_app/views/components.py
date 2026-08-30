"""Pezzi di interfaccia usati da più sezioni.

Stanno qui perché la scelta di una cartella e l'ascolto di una riga servono
uguali in Folder analysis e in Tag analysis, e duplicarli significherebbe
correggerli due volte.
"""

from __future__ import annotations

import inspect
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

from core.analysis import waveform

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


# Più file in una volta: i percorsi tornano uno per riga, perché una lista
# AppleScript letta da Python sarebbe da spacchettare a mano e un a capo in
# un nome di file non esiste.
FILES_PANEL = """on run argv
    set picks to (choose file with prompt (item 1 of argv) ¬
        with multiple selections allowed)
    set out to {}
    repeat with one in picks
        set end of out to POSIX path of one
    end repeat
    set AppleScript's text item delimiters to linefeed
    return out as text
end run"""


def pick_files(prompt: str = "Choose tracks") -> list[Path]:
    """Il selettore di file del Mac, con più brani in una volta sola.

    Non è un widget come i due qui accanto: si chiama quando serve — da un
    pulsante — e torna quello che è stato scelto, o una lista vuota se il
    pannello è stato annullato, che in AppleScript è un errore e non una
    risposta vuota.
    """
    try:
        out = subprocess.run(["osascript", "-e", FILES_PANEL, prompt],
                             capture_output=True, text=True, check=True)
    except Exception:
        return []
    return [Path(line) for line in out.stdout.splitlines() if line.strip()]


# Il prompt arriva come ARGOMENTO e non incollato dentro allo script, per
# la stessa ragione spiegata sotto al pannello di salvataggio.
FILE_PANEL = """on run argv
    return POSIX path of (choose file with prompt (item 1 of argv))
end run"""


def ask_for_file(prompt: str = "Choose a track") -> Path | None:
    """Il pannello del Finder e basta, senza il campo di testo accanto.

    Gemello di `pick_files` al singolare: si chiama da un pulsante quando
    l'unica cosa che serve e' scegliere un file, e il posto dove scriverne il
    percorso c'e' gia' altrove. `pick_file` ci si appoggia sopra invece di
    rifare la stessa chiamata, cosi' il pannello si chiede in un punto solo.
    """
    try:
        out = subprocess.run(["osascript", "-e", FILE_PANEL, prompt],
                             capture_output=True, text=True, check=True)
    except Exception:
        return None                # dialogo annullato o non disponibile
    chosen = out.stdout.strip()
    return Path(chosen) if chosen else None


def pick_file(state_key: str, label: str = "Track", placeholder: str = "",
              prompt: str = "Choose a track",
              browse: str = "🎵 Browse…") -> Path | None:
    """Campo di testo più il selettore di file nativo del Mac.

    Gemello di `pick_folder`: stessa forma e stessa ragione — l'app gira in
    locale, quindi il Finder si può chiedere — e cambia solo cosa si sceglie.
    """
    def _browse() -> None:
        chosen = ask_for_file(prompt)
        if chosen is not None:
            st.session_state[state_key] = str(chosen)

    col_path, col_browse = st.columns([5, 1])
    typed = col_path.text_input(label, key=state_key, placeholder=placeholder)
    col_browse.markdown("<div style='height:1.8em'></div>", unsafe_allow_html=True)
    # L'etichetta si accorcia dove il campo sta stretto — dentro una colonna
    # annidata "🎵 Browse…" va a capo e il pulsante diventa alto il doppio.
    col_browse.button(browse, on_click=_browse, width="stretch",
                      key=f"browsefile::{state_key}",
                      help="Choose the file from the Finder.")

    if not typed.strip():
        return None
    track = Path(typed).expanduser()
    if not track.is_file():
        st.error(f"Not a file: {track}")
        return None
    return track


# Il pannello di salvataggio si chiede in AppleScript, e prompt e nome
# arrivano come ARGOMENTI invece che incollati dentro allo script: un nome
# con una virgoletta dentro romperebbe il testo dello script, e i nomi dei
# file non li scegliamo noi.
SAVE_PANEL = """on run argv
    set target to (choose file name with prompt (item 1 of argv) ¬
        default name (item 2 of argv))
    return POSIX path of target
end run"""


def save_as(data: str | bytes, default_name: str,
            prompt: str = "Save as") -> Path | None:
    """Il pannello «Salva col nome» del Finder, e il file scritto dove dice.

    Il pulsante di download del browser scrive dove il browser ha deciso —
    tipicamente i Download — e per un file che deve finire in una cartella
    precisa vuol dire salvare e poi andarlo a spostare. Qui si sceglie nome e
    destinazione una volta sola. Come per il selettore di cartelle qui sopra,
    ci si può permettere di chiamare il Finder perché l'app gira in locale:
    il pannello si apre sulla macchina che esegue Python, che è la stessa
    davanti a cui si sta.

    Torna il percorso scritto, o `None` se il pannello è stato annullato —
    che in AppleScript è un errore, non una risposta vuota.
    """
    try:
        out = subprocess.run(["osascript", "-e", SAVE_PANEL,
                              prompt, default_name],
                             capture_output=True, text=True, check=True)
    except Exception:
        return None    # annullato, o niente Finder a cui chiedere
    chosen = out.stdout.strip()
    if not chosen:
        return None
    path = Path(chosen)
    path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    return path


def reveal_in_finder(path: Path) -> str | None:
    """Mostra il file nel Finder, già selezionato. None se è andata bene.

    `open -R` e non `open`: aprire il file lo APRE — una copertina finirebbe
    in Anteprima e un .xml in un editor — mentre qui la domanda è sempre
    "dov'è", non "cos'è". Come per il selettore di cartelle qui sopra, ci si
    può permettere di chiamare il Finder perché l'app gira in locale.
    """
    try:
        out = subprocess.run(["open", "-R", str(path)],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as e:
        return str(e)
    if out.returncode == 0:
        return None
    return out.stderr.strip() or "il Finder non ha risposto"


# Il ▶ di una riga qualunque, e quello della riga che si sta ascoltando.
# Due glifi e non un colore perche' `st.data_editor` non accetta uno Styler:
# le righe non si possono tingere, e l'unica cosa che si puo' cambiare riga
# per riga e' il contenuto delle celle. La colonna del pulsante e' gia' una
# cella per riga, quindi il segno sta li' senza aggiungere una colonna che
# farebbe ballare tutte le altre nel momento in cui si preme play.
PLAY_GLYPH = "▶"
HEARING_GLYPH = "🔊"

# La colonna del play resta ferma mentre le altre scorrono di lato: le tabelle
# della mappa sono larghe — file, BPM, tonalita', groove, mood, generi,
# cartella, piu' gli scarti — e su uno schermo di portatile il ▶ della riga
# che si vuole sentire finisce fuori proprio mentre si guarda la colonna che
# ha fatto venire voglia di sentirla.
#
# Si chiede solo se questa versione di Streamlit sa cosa farsene: `pinned` e'
# arrivato dopo il minimo che il progetto dichiara, e su una versione piu'
# vecchia passarlo sarebbe un errore invece di una colonna che scorre. Stessa
# prudenza con cui `track_columns.dark()` chiede del tema.
_PINNED = ({"pinned": True}
           if "pinned" in inspect.signature(
               st.column_config.ButtonColumn).parameters else {})


def play_marks(paths, playing: str | None) -> list[str]:
    """Il glifo del pulsante per ogni riga: l'altoparlante su quella in ascolto.

    Pura, perche' e' l'unico pezzo di `play_table` che si puo' provare senza
    un runtime di Streamlit — e perche' il confronto e' fra percorsi, che e'
    esattamente il genere di cosa che si rompe in silenzio.

    Il confronto e' per percorso e non per numero di riga: la stessa tabella
    si riordina, si filtra e si ridisegna di continuo, e un indice ricordato
    da un giro all'altro indicherebbe presto un brano diverso. Il percorso
    invece resta quello, e lo stesso brano si illumina in TUTTE le tabelle
    che lo mostrano, non solo in quella da cui si e' premuto.
    """
    return [HEARING_GLYPH if playing is not None and path == playing
            else PLAY_GLYPH for path in paths]


def play_table(section: str, table: pd.DataFrame, column_order: list[str],
               column_config: dict, editable: bool = True,
               editor_key: str | None = None,
               play: bool = True, reveal: bool = False) -> pd.DataFrame:
    """Tabella con i pulsanti di riga: ▶ per sentire, 🔍 per mostrare nel Finder.

    Serve a sentire un file PRIMA di deciderne la sorte. Il clic sceglie il
    brano e basta: a suonarlo ci pensa `render_dock`, in fondo alla pagina.
    Il lettore carica solo il file scelto — la colonna audio nativa di
    Streamlit vorrebbe invece un URL per ogni riga, e generarli legge OGNI
    file per intero in memoria a ogni rerun (misurato: 2,5 ms e l'intero
    contenuto per riga, cioè giga di traffico su qualche centinaio di righe).

    `play` si spegne dove il ▶ non vuol dire niente — un elenco di .jpg — e
    `reveal` accende il 🔍 dove serve vedere il file dov'è.
    """
    order_key, click_key = f"order::{section}", f"click::{section}"
    finder_key, trouble_key = f"finder::{section}", f"finder_error::{section}"
    st.session_state[order_key] = list(table["_path"])

    def _clicked(which: str, order_key=order_key):
        click = st.session_state.get(which)
        order = st.session_state.get(order_key, [])
        if click and 0 <= click.get("row", -1) < len(order):
            return order[click["row"]]
        return None

    def _on_play(click_key=click_key):
        chosen = _clicked(click_key)
        if chosen is not None:
            st.session_state[NOW_PLAYING] = chosen

    def _on_reveal(finder_key=finder_key, trouble_key=trouble_key):
        chosen = _clicked(finder_key)
        if chosen is not None:
            st.session_state[trouble_key] = reveal_in_finder(Path(chosen))

    shown = table.copy()
    config = dict(column_config)
    buttons = []
    if play:
        buttons.append("Play")
        shown.insert(len(buttons) - 1, "Play",
                     play_marks(table["_path"],
                                st.session_state.get(NOW_PLAYING)))
        config["Play"] = st.column_config.ButtonColumn(
            "▶", on_click=_on_play, key=click_key, width="small", **_PINNED,
            help="Hear this track. The row you are listening to keeps a "
                 "speaker instead of the arrow, in every table that shows "
                 "it. To stop, close the player at the bottom of the page.")
    if reveal:
        buttons.append("Finder")
        shown.insert(len(buttons) - 1, "Finder", "🔍")
        config["Finder"] = st.column_config.ButtonColumn(
            "🔍", on_click=_on_reveal, key=finder_key, width="small",
            help="Show this file in the Finder.")

    edited = st.data_editor(
        shown, key=editor_key, width="stretch", hide_index=True,
        column_order=[*buttons, *column_order], column_config=config,
        **({} if editable else {"disabled": column_order}),
    )
    problem = st.session_state.get(trouble_key)
    if problem:
        st.caption(f"⚠️ The Finder could not show it: {problem}")
    return edited


def tick_all(base: str, into=None, default: bool = True) -> tuple[bool, str]:
    """I pulsanti Select all / Unselect all sopra una tabella con le spunte.

    Torna due cose: il valore da mettere nella colonna delle spunte quando
    la tabella si ricostruisce, e la chiave da passarle.

    La chiave porta un CONTATORE che i due pulsanti fanno avanzare, e non e'
    un dettaglio: cambiare chiave fa nascere la tabella da capo, ed e' l'unico
    modo di svuotare davvero le spunte. Cancellare il suo stato in sessione
    non basta — le spunte cambiate a mano vivono anche nella griglia sul
    frontend, che le rimanda indietro: misurato, dopo "Unselect all"
    restavano spuntate proprio le righe toccate poco prima.
    """
    tick_key, fresh_key = f"tickall::{base}", f"tickfresh::{base}"
    fresh = st.session_state.get(fresh_key, 0)

    def _set(value: bool, fresh=fresh) -> None:
        st.session_state[tick_key] = value
        st.session_state[fresh_key] = fresh + 1

    col_all, col_none = (into or st).columns(2)
    col_all.button("Select all", on_click=_set, args=(True,),
                   width="stretch", key=f"tickyes::{base}")
    col_none.button("Unselect all", on_click=_set, args=(False,),
                    width="stretch", key=f"tickno::{base}")
    return st.session_state.get(tick_key, default), f"{base}::{fresh}"


_PLAYER_HTML = """
<div class="wp">
  <button class="pp" type="button" aria-label="Play or pause">&#9654;</button>
  <div class="wavebox">
    <div class="name"></div>
    <canvas class="wave"></canvas>
  </div>
  <span class="clock">0:00</span>
  <button class="x" type="button" aria-label="Close the player">&#10005;</button>
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
/* La larghezza la decide QUESTO riquadro, mai il canvas. Il canvas senza
   width in CSS userebbe come misura intrinseca il proprio attributo width,
   che il disegno imposta a clientWidth * devicePixelRatio: su uno schermo
   Retina ogni passata raddoppiava la larghezza (misurato: 637, 1274, 2548,
   5096, 10192) e l'onda si allargava a destra fino a esplodere. */
.wp .wavebox { flex: 1 1 0; min-width: 0; }
/* Il nome sta DENTRO il riquadro dell'onda e non come didascalia sopra il
   componente: in fondo allo schermo ogni riga in piu' e' spazio tolto alla
   pagina, e una riga di testo larga quanto l'onda non allunga il lettore. */
.wp .name {
  font-size: 0.78rem; line-height: 1.2; margin-bottom: 0.25rem;
  color: var(--st-text-color, #333); opacity: 0.7;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.wp canvas { display: block; width: 100%; height: 56px; cursor: pointer; }
.wp .clock {
  /* nowrap piu' una larghezza minima: senza, "5:33 / 7:23" andava a capo e
     si leggeva solo meta' del tempo. */
  flex: 0 0 auto; white-space: nowrap; min-width: 5.5rem; text-align: right;
  font-variant-numeric: tabular-nums;
  font-size: 0.8rem; color: var(--st-text-color, #333); opacity: 0.75;
}
.wp .x {
  flex: 0 0 auto; border: none; background: none; cursor: pointer;
  padding: 0.25rem 0; font-size: 0.85rem; line-height: 1;
  color: var(--st-text-color, #333); opacity: 0.4;
}
.wp .x:hover { opacity: 0.9; }
"""

# L'onda E' la barra di avanzamento: la parte gia' suonata si colora, e un
# clic sposta la riproduzione in quel punto. Un lettore nativo affiancato a
# un grafico non lo permetterebbe — la sua barra non e' allineata all'asse
# dell'onda, e sovrapporli sarebbe solo decorazione.
_PLAYER_JS = """
export default function (component) {
  const { data, parentElement, setStateValue } = component
  const root = parentElement.querySelector(".wp")
  if (!root || !data || !data.url) return

  const canvas = root.querySelector("canvas")
  const button = root.querySelector(".pp")
  const clock = root.querySelector(".clock")
  root.querySelector(".name").textContent = data.name || ""
  const peaks = data.peaks || []
  const total = data.duration || 0

  // UN SOLO oggetto Audio per tutta la pagina, tenuto su window e non
  // sull'elemento. Un Audio creato per un componente che poi viene
  // rimontato resta staccato dal DOM e CONTINUA A SUONARE: cambiando riga
  // partivano due brani insieme, tre al clic dopo, e cosi' via. Misurato:
  // il vecchio a 124,8 s "staccato: true, suona: true" mentre il nuovo
  // ripartiva da zero.
  if (!window.__wavecut_audio) window.__wavecut_audio = new Audio()
  const audio = window.__wavecut_audio
  root._audio = audio

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
    // solo se cambiano davvero: assegnarli azzera il disegno a ogni frame
    const pixel_w = Math.round(width * ratio), pixel_h = Math.round(height * ratio)
    if (canvas.width !== pixel_w) canvas.width = pixel_w
    if (canvas.height !== pixel_h) canvas.height = pixel_h
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

  if (window.__wavecut_url !== data.url) {
    window.__wavecut_url = data.url
    audio.pause()
    audio.src = data.url
    // il pulsante della riga E' la richiesta di ascoltare: si parte da soli
    audio.play().catch(() => {})
  }

  audio.ontimeupdate = draw
  audio.onplay = () => { button.innerHTML = "&#10073;&#10073;"; draw() }
  audio.onpause = () => { button.innerHTML = "&#9654;" }
  audio.onended = () => { button.innerHTML = "&#9654;" }
  button.onclick = () => { audio.paused ? audio.play().catch(() => {}) : audio.pause() }
  // Si ferma QUI, prima di avvisare Python: l'oggetto Audio vive su window e
  // sopravvive allo smontaggio del componente, quindi lasciar fare tutto a un
  // rerun avrebbe fatto sparire il lettore col brano che continuava a suonare.
  // Azzerare l'indirizzo fa anche ripartire lo stesso brano se lo si riclicca.
  root.querySelector(".x").onclick = () => {
    audio.pause()
    window.__wavecut_url = null
    setStateValue("closed", Date.now())
  }
  canvas.onclick = (e) => {
    const box = canvas.getBoundingClientRect()
    const where = (e.clientX - box.left) / box.width
    if (total) { audio.currentTime = Math.min(total, Math.max(0, where * total)); draw() }
  }
  if (root._watching !== true) {
    root._watching = true
    // si osserva il riquadro, non il canvas: osservare cio' che il disegno
    // ridimensiona e' il modo piu' diretto di rientrare all'infinito.
    new ResizeObserver(draw).observe(root.querySelector(".wavebox"))
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

    Il conto sta in `core.analysis.waveform.envelope`, perche' i lettori
    sono due — questo e quello dell'app Qt — e devono disegnare la stessa
    onda. Qui resta solo la cache: `mtime` e `size` non si usano nel corpo,
    sono li' perche' la cache rilegga il file se cambia.
    """
    return waveform.envelope(path)


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


def _forget() -> None:
    """Il ✕ del lettore: si dimentica il brano, e il dock sparisce."""
    st.session_state.pop(NOW_PLAYING, None)


DOCK_SLOT = "components::dock_slot"


def claim_dock() -> None:
    """Prenota il posto del lettore in fondo alla pagina, e lo riempie.

    La chiama `app.py` una volta sola, PRIMA della pagina: `st.bottom` e' un
    contenitore a se', quindi l'ordine non cambia dove appare, e cosi' il
    lettore c'e' anche nelle pagine che si fermano presto con st.stop().
    """
    st.session_state[DOCK_SLOT] = st.bottom.empty()
    fill_dock()


def fill_dock(owner: str = "app") -> None:
    """Mette nel posto gia' prenotato il brano che si sta suonando.

    La chiama anche chi vive dentro un frammento e ha tabelle con ▶ — la
    lavagna. Un clic dentro un frammento fa ripartire SOLO il frammento:
    `app.py` non viene rieseguito, e il lettore restava fermo sul brano di
    prima. Era esattamente il guasto della lavagna.

    Le regole sono di Streamlit, e sono state misurate una per una:

    - il posto va prenotato nel giro intero con `st.bottom.empty()`, o il
      frammento non ha dove scrivere e si ferma con un errore;
    - anche il frammento deve scriverci DURANTE il giro intero, altrimenti
      alla sua prima ripartenza da solo trova il posto non riservato;
    - le due scritture vogliono chiavi diverse, perche' la stessa chiave due
      volte nello stesso giro e' un errore. L'ultima vince, quindi sulla
      pagina della mappa si vede il lettore del frammento;
    - nel posto ci sta UN elemento solo: il secondo sostituisce il primo, e
      un container annidato qui rompe l'albero degli elementi sul frontend.
    """
    slot = st.session_state.get(DOCK_SLOT)
    if slot is None:
        return

    current = st.session_state.get(NOW_PLAYING)
    if current is None:
        slot.empty()             # chiuso col ✕: il posto torna a non esserci
        return

    track = Path(current)
    if not track.exists():
        slot.warning(f"Not there any more: {track.name}")
        return
    mime = PLAYABLE.get(track.suffix.lower())
    if mime is None:
        slot.info(f"The browser cannot play {track.suffix} files.")
        return

    try:
        stat = track.stat()
        wave = _envelope(str(track), stat.st_mtime, stat.st_size)
    except OSError:
        wave = None
    url = _media_url(track, mime) if wave else None

    if wave and url:
        peaks, seconds = wave
        with slot:
            _wave_player(data={"url": url, "peaks": peaks, "duration": seconds,
                               "name": track.name},
                         key=f"wave::dock::{owner}", height=96,
                         on_closed_change=_forget)
        return

    # Senza il nome del brano, che nel posto non ci sta: e' un elemento in
    # piu' e caccerebbe il lettore. Succede solo senza ffmpeg.
    try:
        slot.audio(str(track), format=mime, autoplay=True)
    except Exception as e:
        slot.warning(f"Could not play it: {e}")
