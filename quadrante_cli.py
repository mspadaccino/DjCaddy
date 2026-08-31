"""Il quadrante prima/dopo come pagina locale, con l'ascolto dentro.

La pagina dei quadranti pubblicata come artifact non può suonare niente: il
browser le vieta ogni richiesta fuori da sé, e i brani stanno su questo
disco. Questo comando serve la STESSA pagina da localhost e le affianca
l'audio: clic su un punto — o su un titolo di «Chi entra dove» — e il brano
parte, con la barra di ascolto in fondo. È il pezzo che mancava per fare la
valutazione a orecchio senza uscire dal grafico.

Due pezzi, un file:

- i DATI (`quadrante_data.json`, accanto a questo script): le due letture
  della valence su tutta la mappa, come le calcola `mood_shift_cli`, più i
  percorsi dei brani disegnati. Si rifanno con `--rebuild` (minuti: rilegge
  le attivazioni dagli embedding, l'audio non si tocca);
- il SERVER: pagina su `/`, audio su `/audio/<n>` — per indice nel
  manifest, mai per percorso, così la pagina non può chiedere file a caso.
  Range HTTP gestito, che senza non si può saltare dentro al brano.

    poetry run python quadrante_cli.py --rebuild   # la prima volta
    poetry run python quadrante_cli.py             # http://127.0.0.1:8766

Il rebuild vuole l'ambiente poetry (Essentia); il solo ascolto no.
Da cancellare, insieme a `mood_shift_cli.py`, quando la decisione è presa.
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "quadrante_data.json"
PORT = 8766
CHUNK = 512 * 1024

MIME = {".mp3": "audio/mpeg", ".flac": "audio/flac", ".m4a": "audio/mp4",
        ".mp4": "audio/mp4", ".aac": "audio/aac", ".ogg": "audio/ogg",
        ".oga": "audio/ogg", ".opus": "audio/ogg", ".wav": "audio/x-wav",
        ".aif": "audio/aiff", ".aiff": "audio/aiff", ".wma": "audio/x-ms-wma"}


# --------------------------------------------------------------------------
# I dati: le stesse due letture di mood_shift_cli, più i percorsi
# --------------------------------------------------------------------------

def build_data() -> dict:
    """Ricalcola tutto e scrive `quadrante_data.json`. Minuti, non ore."""
    import numpy as np

    from core.analysis import energy
    from core.analysis.map_store import MapStore, default_store_dir
    from mood_cli import _head
    from mood_shift_cli import (activations_matrix, calibrated_reading,
                                current_reading)

    store = MapStore.load(default_store_dir())
    n = min(len(store.rows), len(store.embeddings))
    rows = store.rows[:n]
    print(f"Mappa: {n:,} brani")

    predict, labels = _head()
    acts = activations_matrix(store.embeddings[:n], predict)
    old_raw = current_reading(acts, labels)
    new_raw = calibrated_reading(acts, labels)
    x0 = energy.ranks(old_raw)
    x1 = energy.ranks(new_raw)
    y = energy.from_rows(rows)

    mask = np.isfinite(x0) & np.isfinite(x1) & np.isfinite(y)
    idx = np.flatnonzero(mask)

    def quad(x, yv):
        return (0 if yv >= 0.5 else 2) + (1 if x >= 0.5 else 0)

    q0 = np.array([quad(x0[i], y[i]) for i in idx])
    q1 = np.array([quad(x1[i], y[i]) for i in idx])
    matrix = [[int(((q0 == a) & (q1 == b)).sum()) for b in range(4)]
              for a in range(4)]

    def clean(name, limit):
        stem = name.rsplit(".", 1)[0]
        return stem[: limit - 1] + "…" if len(stem) > limit else stem

    delta = x1 - x0

    # Il campione disegnato: un punto ogni otto. Il suo indice nel manifest
    # audio È l'indice del punto, quindi i percorsi vanno in quest'ordine.
    sub = idx[::8]
    points = [[round(float(x0[i]), 3), round(float(x1[i]), 3),
               round(float(y[i]), 3)] for i in sub]
    names = [clean(rows[i].get("name", ""), 48) for i in sub]
    moods = [(rows[i].get("moods") or "")[:40] for i in sub]
    paths = [rows[i]["path"] for i in sub]

    # Chi entra in ogni quadrante: anche loro nel manifest, in coda, così i
    # titoli della lista si ascoltano pure se il punto non è fra i disegnati.
    examples = []
    for target in range(4):
        coming = [k for k, i in enumerate(idx)
                  if q1[k] == target and q0[k] != target]
        coming.sort(key=lambda k: -abs(delta[idx[k]]))
        block = []
        for k in coming[:6]:
            i = idx[k]
            block.append({"name": clean(rows[i].get("name", ""), 46),
                          "moods": (rows[i].get("moods") or "")[:40],
                          "shift": round(float(delta[i]), 2),
                          "play": len(paths)})
            paths.append(rows[i]["path"])
        examples.append(block)

    data = {
        "stats": {"n": int(mask.sum())},
        "matrix": matrix,
        "counts0": [int((q0 == a).sum()) for a in range(4)],
        "counts1": [int((q1 == a).sum()) for a in range(4)],
        "examples": examples,
        "points": points, "names": names, "moods": moods,
        "paths": paths,
    }
    DATA_FILE.write_text(json.dumps(data, ensure_ascii=False))
    print(f"scritto {DATA_FILE.name}  "
          f"({DATA_FILE.stat().st_size / 1e6:.1f} MB, {len(points):,} punti)")
    return data


# --------------------------------------------------------------------------
# Il server: la pagina, e l'audio per indice
# --------------------------------------------------------------------------

def render_page(data: dict) -> bytes:
    """La pagina intera, con i dati dentro. I percorsi restano di qua."""
    payload = {k: v for k, v in data.items() if k != "paths"}
    body = TEMPLATE.replace(
        "__DATA__",
        json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"))
    return ("<!doctype html><html lang=\"it\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, "
            "initial-scale=1\"></head><body style=\"margin:0\">"
            + body + "</body></html>").encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    html: bytes = b""
    paths: list[str] = []

    def log_message(self, *args) -> None:  # noqa: D102 — niente riga per request
        pass

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(self.html)))
            self.end_headers()
            self.wfile.write(self.html)
            return
        m = re.fullmatch(r"/audio/(\d+)", self.path)
        if not m:
            self.send_error(404)
            return
        i = int(m.group(1))
        if i >= len(self.paths):
            self.send_error(404, "indice fuori dal manifest")
            return
        track = Path(self.paths[i])
        if not track.is_file():
            self.send_error(404, "file non trovato: il disco è montato?")
            return
        print(f"  ▶ {track.name}")
        self._stream(track)

    def _stream(self, track: Path) -> None:
        """Il file, intero o a pezzi: senza Range non si salta nel brano."""
        size = track.stat().st_size
        start, end = 0, size - 1
        wanted = self.headers.get("Range")
        if wanted:
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", wanted.strip())
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    if m.group(2):
                        end = min(int(m.group(2)), size - 1)
                else:
                    start = max(0, size - int(m.group(2)))
            if start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type",
                         MIME.get(track.suffix.lower(),
                                  "application/octet-stream"))
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with track.open("rb") as fh:
            fh.seek(start)
            left = end - start + 1
            while left > 0:
                piece = fh.read(min(CHUNK, left))
                if not piece:
                    break
                try:
                    self.wfile.write(piece)
                except (BrokenPipeError, ConnectionResetError):
                    return          # il player ha saltato altrove: normale
                left -= len(piece)


def serve(data: dict, port: int) -> None:
    Handler.html = render_page(data)
    Handler.paths = data.get("paths", [])
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Quadrante su  http://127.0.0.1:{port}  — Ctrl-C per fermare")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nfermato")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve il quadrante prima/dopo in locale, con l'ascolto. "
                    "Non scrive niente nella mappa.")
    parser.add_argument("--rebuild", action="store_true",
                        help="rifà i numeri dagli embedding (minuti)")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    if args.rebuild or not DATA_FILE.exists():
        if not DATA_FILE.exists():
            print("quadrante_data.json non c'è ancora: lo costruisco.")
        data = build_data()
    else:
        data = json.loads(DATA_FILE.read_text())
    serve(data, args.port)


# --------------------------------------------------------------------------
# La pagina. Identica all'artifact, più la barra di ascolto: il fragment
# per l'artifact è questo stesso testo, con i percorsi tolti dai dati.
# --------------------------------------------------------------------------

TEMPLATE = r"""<title>Quadrante ricalibrato</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
  :root {
    --paper: #ffffff;
    --plot: #f4f6f9;
    --ink: #1b1f27;
    --muted: #5c6672;
    --faint: #8a94a2;
    --line: rgba(27, 31, 39, 0.14);
    --hair: rgba(27, 31, 39, 0.08);
    --card: #fafbfd;
    --accent: #1f6fd0;
    --pole-dark: #7b4fbf;
    --pole-bright: #a07800;
    --pole-neutral: #8a94a2;
    --tip-bg: rgba(255, 255, 255, 0.97);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --paper: #0e1117;
      --plot: #161a22;
      --ink: #eef1f6;
      --muted: #97a1ae;
      --faint: #6b7684;
      --line: rgba(238, 241, 246, 0.16);
      --hair: rgba(238, 241, 246, 0.08);
      --card: #131720;
      --accent: #6fb4ff;
      --pole-dark: #9d68e0;
      --pole-bright: #b8860b;
      --pole-neutral: #77828f;
      --tip-bg: rgba(19, 23, 32, 0.97);
    }
  }
  :root[data-theme="dark"] {
    --paper: #0e1117;
    --plot: #161a22;
    --ink: #eef1f6;
    --muted: #97a1ae;
    --faint: #6b7684;
    --line: rgba(238, 241, 246, 0.16);
    --hair: rgba(238, 241, 246, 0.08);
    --card: #131720;
    --accent: #6fb4ff;
    --pole-dark: #9d68e0;
    --pole-bright: #b8860b;
    --pole-neutral: #77828f;
    --tip-bg: rgba(19, 23, 32, 0.97);
  }

  * { box-sizing: border-box; }
  body {
    background: var(--paper);
    color: var(--ink);
    font-family: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
    line-height: 1.55;
    margin: 0;
    padding: 0 20px 120px;
  }
  .wrap { max-width: 1060px; margin: 0 auto; }
  .mono {
    font-family: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
    font-variant-numeric: tabular-nums;
  }

  header { padding: 40px 0 8px; }
  .eyebrow {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--muted); margin: 0 0 10px;
  }
  h1 {
    font-size: clamp(26px, 4vw, 34px); font-weight: 700;
    letter-spacing: -0.015em; margin: 0 0 10px; text-wrap: balance;
  }
  .lede { max-width: 62ch; color: var(--muted); margin: 0; font-size: 15.5px; }
  .lede strong { color: var(--ink); font-weight: 600; }

  .tiles {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;
    margin: 26px 0 10px;
  }
  .tile {
    background: var(--card); border: 1px solid var(--hair); border-radius: 6px;
    padding: 12px 14px 10px;
  }
  .tile b {
    display: block; font-size: 22px; font-weight: 600; letter-spacing: -0.01em;
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-variant-numeric: tabular-nums;
  }
  .tile span { font-size: 12.5px; color: var(--muted); }

  .controls {
    display: flex; align-items: center; gap: 18px; flex-wrap: wrap;
    margin: 22px 0 12px;
  }
  .seg { display: inline-flex; border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
  .seg button {
    appearance: none; border: 0; background: transparent; color: var(--muted);
    font: 600 13.5px "IBM Plex Sans", system-ui, sans-serif;
    padding: 8px 22px; cursor: pointer;
  }
  .seg button[aria-pressed="true"] { background: var(--accent); color: var(--paper); }
  .seg button:focus-visible, .check input:focus-visible, #bar-close:focus-visible {
    outline: 2px solid var(--accent); outline-offset: 2px;
  }
  .check {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 13.5px; color: var(--muted); cursor: pointer; user-select: none;
  }
  .check input { accent-color: var(--accent); width: 15px; height: 15px; margin: 0; }

  .stage { position: relative; }
  canvas#plot {
    display: block; width: 100%; background: var(--plot);
    border: 1px solid var(--hair); border-radius: 8px; cursor: pointer;
  }
  .tip {
    position: absolute; pointer-events: none; display: none; z-index: 3;
    background: var(--tip-bg); border: 1px solid var(--line); border-radius: 6px;
    padding: 8px 11px; max-width: 320px; box-shadow: 0 4px 18px rgba(0,0,0,0.18);
  }
  .tip .t-name { font-size: 13px; font-weight: 600; line-height: 1.35; }
  .tip .t-moods { font-size: 12px; color: var(--muted); margin-top: 1px; }
  .tip .t-shift {
    font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12px;
    margin-top: 5px; color: var(--ink); font-variant-numeric: tabular-nums;
  }

  .legend { display: flex; align-items: center; gap: 12px; margin: 12px 2px 4px; flex-wrap: wrap; }
  .grad {
    height: 10px; width: 240px; border-radius: 5px; border: 1px solid var(--hair);
    background: linear-gradient(90deg, var(--pole-dark), var(--pole-neutral) 50%, var(--pole-bright));
  }
  .legend small, .note { font-size: 12.5px; color: var(--muted); }

  h2 {
    font-size: 19px; font-weight: 700; letter-spacing: -0.01em;
    margin: 44px 0 6px; text-wrap: balance;
  }
  .h2sub { color: var(--muted); font-size: 14px; margin: 0 0 16px; max-width: 66ch; }

  .caveat {
    border: 1px solid var(--line); border-left: 3px solid var(--pole-dark);
    border-radius: 6px; background: var(--card); padding: 14px 18px;
    margin: 26px 0 0; max-width: 100%;
  }
  .caveat p { margin: 6px 0; font-size: 14px; color: var(--muted); }
  .caveat p strong { color: var(--ink); }

  .quads { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .qcard {
    background: var(--card); border: 1px solid var(--hair); border-radius: 8px;
    padding: 14px 16px 12px;
  }
  .qhead { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; }
  .qname {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 11.5px; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--muted);
  }
  .qcount {
    font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 14px;
    font-variant-numeric: tabular-nums;
  }
  .qcount .after { font-weight: 600; }
  .qflow {
    font-size: 12.5px; color: var(--muted); margin: 3px 0 10px;
    font-variant-numeric: tabular-nums;
  }
  .qlist { list-style: none; margin: 0; padding: 0; border-top: 1px solid var(--hair); }
  .qlist li {
    display: flex; align-items: center; gap: 9px; padding: 6px 4px;
    border-bottom: 1px solid var(--hair); font-size: 13px; min-width: 0;
    cursor: pointer; border-radius: 4px;
  }
  .qlist li:hover { background: var(--plot); }
  .dot { width: 8px; height: 8px; border-radius: 50%; flex: 0 0 auto; }
  .qlist .nm { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1 1 auto; }
  .qlist .sh {
    font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12px;
    color: var(--muted); flex: 0 0 auto; font-variant-numeric: tabular-nums;
  }

  footer { margin-top: 40px; border-top: 1px solid var(--hair); padding-top: 14px; }
  footer p { font-size: 12.5px; color: var(--muted); margin: 4px 0; max-width: 82ch; }

  #bar {
    position: fixed; left: 0; right: 0; bottom: 0; z-index: 5;
    display: none; align-items: center; gap: 16px;
    padding: 10px 20px; background: var(--tip-bg);
    border-top: 1px solid var(--line); backdrop-filter: blur(6px);
  }
  #bar.on { display: flex; }
  #bar .b-info { min-width: 0; flex: 1 1 auto; }
  #bar .b-name {
    font-size: 13.5px; font-weight: 600;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  #bar .b-moods {
    font-size: 12px; color: var(--muted);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  #bar .b-shift {
    font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12.5px;
    color: var(--ink); flex: 0 0 auto; font-variant-numeric: tabular-nums;
  }
  #bar audio { flex: 0 0 340px; max-width: 44vw; height: 36px; }
  #bar .b-note { font-size: 12px; color: var(--muted); flex: 0 0 auto; max-width: 30ch; }
  #bar-close {
    appearance: none; border: 1px solid var(--line); background: transparent;
    color: var(--muted); border-radius: 5px; width: 28px; height: 28px;
    cursor: pointer; flex: 0 0 auto; font-size: 13px; line-height: 1;
  }
  #bar-close:hover { color: var(--ink); }

  @media (max-width: 760px) {
    .tiles { grid-template-columns: 1fr 1fr; }
    .quads { grid-template-columns: 1fr; }
    #bar { flex-wrap: wrap; gap: 8px; }
    #bar audio { flex: 1 1 100%; max-width: none; }
  }
</style>

<div class="wrap">
  <header>
    <p class="eyebrow">Wavecut · analisi del mood · 87.019 brani</p>
    <h1>Quadrante ricalibrato</h1>
    <p class="lede">Gli stessi assi dell'app — <strong>energia</strong> in verticale,
      <strong>valence</strong> in orizzontale, entrambi come rango sulla libreria —
      con la valence letta in due modi: dalle attivazioni crude com'è oggi, e
      calibrata etichetta per etichetta. L'energia non cambia, quindi ogni brano
      scorre solo in orizzontale. Il colore di un punto è il suo spostamento,
      ed è lo stesso nelle due viste.</p>
  </header>

  <div class="tiles">
    <div class="tile"><b>+0,66</b><span>correlazione fra i due ordini: un riordino vero, non un ritocco</span></div>
    <div class="tile"><b>61,5%</b><span>dei brani si sposta di almeno un decile</span></div>
    <div class="tile"><b>24,5%</b><span>cambia lato: da «più chiaro della mediana» a «più scuro», o viceversa</span></div>
    <div class="tile"><b>0,00 → +0,27</b><span>correlazione della valence con l'energia: l'indipendenza si paga</span></div>
  </div>

  <div class="controls">
    <div class="seg" role="group" aria-label="Quale lettura mostrare">
      <button id="btn-before" aria-pressed="true">Prima</button>
      <button id="btn-after" aria-pressed="false">Dopo</button>
    </div>
    <label class="check"><input type="checkbox" id="only-cross">
      metti in risalto solo chi cambia lato</label>
  </div>

  <div class="stage">
    <canvas id="plot"></canvas>
    <div class="tip" id="tip">
      <div class="t-name"></div>
      <div class="t-moods"></div>
      <div class="t-shift"></div>
    </div>
  </div>

  <div class="legend">
    <div class="grad" aria-hidden="true"></div>
    <small>va verso scuro · fermo · va verso chiaro</small>
    <small>· clicca un punto, o un titolo qui sotto, per ascoltarlo</small>
    <small>· disegnato un brano ogni otto (10.878 punti); i conteggi sono sull'intera libreria</small>
  </div>

  <div class="caveat">
    <p><strong>Da pesare all'ascolto.</strong> La lettura attuale era ortogonale
      all'energia (+0,00, una qualità dichiarata di <code>mood_scale</code>); quella
      calibrata correla +0,27 — i brani che spingono tendono a leggersi più chiari, i
      calmi più scuri. Per questo i quadranti dopo non sono più quattro quarti:
      «chiaro · spinge» e «scuro · calmo» crescono, gli altri due si assottigliano.</p>
    <p>Nei flussi si riconoscono i due pattern già visti nel CSV: verso chiaro i
      brani feelgood che oggi <em>Deep</em> tiene bassi (credibili all'ascolto);
      verso scuro il classic rock guidato da <em>Love</em> (da ascoltare con più
      scetticismo).</p>
  </div>

  <h2>Chi entra dove</h2>
  <p class="h2sub">I quattro quadranti disposti come nel grafico. Per ognuno: quanti
    brani lo abitano prima e dopo, i flussi in ingresso e in uscita, e i sei arrivi
    più spostati — clicca un titolo per sentirlo.</p>
  <div class="quads" id="quads"></div>

  <footer>
    <p>Metodo: valence attuale = <span class="mono">valence_of</span> sulle attivazioni
      crude (la stessa funzione della mappa); valence calibrata = media dei ranghi di
      libreria delle etichette chiare meno quella delle scure, poi il tutto riportato a
      rango. Attivazioni rilette dagli embedding su disco, mappa mai scritta. Croce a
      0,5 su entrambi gli assi, come nell'app.</p>
    <p>Generato da <span class="mono">quadrante_cli.py</span> · da cancellare insieme
      agli script di prova quando la decisione è presa.</p>
  </footer>
</div>

<div id="bar">
  <button id="bar-close" aria-label="chiudi l'ascolto">✕</button>
  <div class="b-info">
    <div class="b-name"></div>
    <div class="b-moods"></div>
  </div>
  <div class="b-shift"></div>
  <audio id="player" controls preload="auto"></audio>
  <span class="b-note" hidden>l'ascolto funziona nella pagina locale:
    <span class="mono">poetry run python quadrante_cli.py</span></span>
</div>

<script id="data" type="application/json">__DATA__</script>
<script>
(function () {
  "use strict";
  const DATA = JSON.parse(document.getElementById("data").textContent);
  const pts = DATA.points, names = DATA.names, moods = DATA.moods;
  const N = pts.length;
  const QNAMES = ["scuro · spinge", "chiaro · spinge", "scuro · calmo", "chiaro · calmo"];
  const it = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ".");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const LOCAL = location.hostname === "localhost" || location.hostname === "127.0.0.1";

  // --- tokens letti dal CSS, riletti a ogni cambio di tema -----------------
  let C = {};
  function readTokens() {
    const s = getComputedStyle(document.documentElement);
    const g = (k) => s.getPropertyValue(k).trim();
    C = { plot: g("--plot"), ink: g("--ink"), muted: g("--muted"),
          faint: g("--faint"), line: g("--line"), accent: g("--accent"),
          dark: g("--pole-dark"), bright: g("--pole-bright"),
          neutral: g("--pole-neutral") };
  }
  const hex = (h) => {
    h = h.replace("#", "");
    return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
  };
  const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
  // Colore dal Δ: violetto → grigio → ambra, saturo oltre |Δ| = 0,6.
  function shade(delta) {
    const t = Math.max(-1, Math.min(1, delta / 0.6));
    const neutral = hex(C.neutral);
    const pole = hex(t < 0 ? C.dark : C.bright);
    const [r, g, b] = mix(neutral, pole, Math.abs(t));
    return [r, g, b];
  }

  // --- geometria -----------------------------------------------------------
  const canvas = document.getElementById("plot");
  const ctx = canvas.getContext("2d");
  const M = { l: 46, r: 14, t: 14, b: 40 };
  let W = 0, H = 0, dpr = 1;
  function resize() {
    const w = canvas.parentElement.clientWidth;
    const h = Math.max(380, Math.min(600, Math.round(w * 0.62)));
    dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.height = h + "px";
    W = w; H = h;
    draw();
  }
  const px = (x) => M.l + x * (W - M.l - M.r);
  const py = (y) => H - M.b - y * (H - M.t - M.b);

  // Ordine di disegno: i fermi sotto, i più spostati sopra.
  const order = Array.from({ length: N }, (_, i) => i)
    .sort((a, b) => Math.abs(pts[a][1] - pts[a][0]) - Math.abs(pts[b][1] - pts[b][0]));

  // --- stato e animazione --------------------------------------------------
  let t = 0;               // 0 = prima, 1 = dopo
  let onlyCross = false;
  let hover = -1;
  let selected = -1;       // il punto in ascolto
  let raf = null;
  const ease = (u) => u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;

  function animateTo(v) {
    if (reduced.matches) { t = v; draw(); return; }
    const from = t, start = performance.now(), dur = 750;
    cancelAnimationFrame(raf);
    (function step(now) {
      const u = Math.min(1, (now - start) / dur);
      t = from + (v - from) * ease(u);
      draw();
      if (u < 1) raf = requestAnimationFrame(step);
    })(start);
  }

  function counts() {
    return DATA.counts0.map((c, q) => Math.round(c + (DATA.counts1[q] - c) * t));
  }

  // --- disegno -------------------------------------------------------------
  function draw() {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    // croce a 0,5 e bordo, come nell'app
    ctx.strokeStyle = C.muted;
    ctx.globalAlpha = 0.45;
    ctx.setLineDash([5, 5]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(px(0.5), py(0)); ctx.lineTo(px(0.5), py(1));
    ctx.moveTo(px(0), py(0.5)); ctx.lineTo(px(1), py(0.5));
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.setLineDash([]);

    // punti
    for (const i of order) {
      const [x0, x1, y] = pts[i];
      const d = x1 - x0;
      const crosses = (x0 >= 0.5) !== (x1 >= 0.5);
      const x = x0 + d * t;
      const [r, g, b] = shade(d);
      let alpha = 0.34 + 0.4 * Math.min(1, Math.abs(d) / 0.45);
      if (onlyCross && !crosses) alpha = 0.05;
      ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
      const rad = 1.7 + 0.8 * Math.min(1, Math.abs(d) / 0.5);
      ctx.beginPath();
      ctx.arc(px(x), py(y), rad, 0, 6.2832);
      ctx.fill();
    }

    // il punto in ascolto: un anello del colore dell'interfaccia
    if (selected >= 0) {
      const p = pts[selected];
      const x = p[0] + (p[1] - p[0]) * t;
      ctx.strokeStyle = C.accent;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(px(x), py(p[2]), 7, 0, 6.2832);
      ctx.stroke();
    }

    // il punto sotto il mouse
    if (hover >= 0) {
      const [x0, x1, y] = pts[hover];
      const x = x0 + (x1 - x0) * t;
      const [r, g, b] = shade(x1 - x0);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.strokeStyle = C.plot;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(px(x), py(y), 5, 0, 6.2832);
      ctx.fill(); ctx.stroke();
    }

    // etichette dei quadranti con i conteggi (libreria intera)
    const c = counts();
    ctx.font = "600 10.5px 'IBM Plex Mono', ui-monospace, monospace";
    const lab = (q, xa, ya, alignX, alignY) => {
      ctx.textAlign = alignX; ctx.textBaseline = alignY;
      ctx.lineJoin = "round";
      ctx.strokeStyle = C.plot;
      ctx.lineWidth = 4;
      const yy = py(ya) + (alignY === "top" ? 10 : -22);
      ctx.strokeText(QNAMES[q].toUpperCase(), px(xa), yy);
      ctx.fillStyle = C.muted;
      ctx.fillText(QNAMES[q].toUpperCase(), px(xa), yy);
      ctx.font = "600 12.5px 'IBM Plex Mono', ui-monospace, monospace";
      ctx.strokeText(it(c[q]), px(xa), yy + 13);
      ctx.fillStyle = C.ink;
      ctx.fillText(it(c[q]), px(xa), yy + 13);
      ctx.font = "600 10.5px 'IBM Plex Mono', ui-monospace, monospace";
    };
    lab(0, 0.02, 1, "left", "top");
    lab(1, 0.98, 1, "right", "top");
    lab(2, 0.02, 0, "left", "bottom");
    lab(3, 0.98, 0, "right", "bottom");

    // assi
    ctx.fillStyle = C.muted;
    ctx.font = "500 11px 'IBM Plex Sans', system-ui, sans-serif";
    ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
    ctx.fillText("valence — rango sulla libreria", (px(0) + px(1)) / 2, H - 12);
    ctx.textAlign = "left";
    ctx.fillText("scuro", px(0), H - 12);
    ctx.textAlign = "right";
    ctx.fillText("chiaro", px(1), H - 12);
    ctx.save();
    ctx.translate(14, (py(0) + py(1)) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("energia — calmo → spinge", 0, 0);
    ctx.restore();
  }

  // --- l'ascolto -----------------------------------------------------------
  const bar = document.getElementById("bar");
  const player = document.getElementById("player");
  const barNote = bar.querySelector(".b-note");
  player.hidden = !LOCAL;
  barNote.hidden = LOCAL;

  const fmt = (v) => v.toFixed(2).replace(".", ",");
  function openBar(name, moodText, shiftText, playIndex) {
    bar.querySelector(".b-name").textContent = name;
    bar.querySelector(".b-moods").textContent = moodText || "—";
    bar.querySelector(".b-shift").textContent = shiftText;
    bar.classList.add("on");
    if (LOCAL && playIndex != null) {
      player.src = "/audio/" + playIndex;
      player.play().catch(() => {});
    }
  }
  function closeBar() {
    bar.classList.remove("on");
    player.pause();
    player.removeAttribute("src");
    selected = -1;
    draw();
  }
  document.getElementById("bar-close").addEventListener("click", closeBar);
  addEventListener("keydown", (e) => {
    if (e.key === "Escape" && bar.classList.contains("on")) closeBar();
  });
  // I primi secondi sono quasi sempre intro: si parte da un quarto del
  // brano, che per giudicare il colore è il posto giusto. Si può sempre
  // tornare indietro con la barra.
  player.addEventListener("loadedmetadata", () => {
    if (isFinite(player.duration) && player.duration > 90) {
      player.currentTime = player.duration * 0.25;
    }
  });

  // --- interazione sul grafico ---------------------------------------------
  function nearest(e, radius) {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    let best = -1, bd = radius * radius;
    for (let i = 0; i < N; i++) {
      const p = pts[i];
      if (onlyCross && (p[0] >= 0.5) === (p[1] >= 0.5)) continue;
      const dx = px(p[0] + (p[1] - p[0]) * t) - mx;
      const dy = py(p[2]) - my;
      const d2 = dx * dx + dy * dy;
      if (d2 < bd) { bd = d2; best = i; }
    }
    return best;
  }

  const tip = document.getElementById("tip");
  canvas.addEventListener("mousemove", (e) => {
    const best = nearest(e, 10);
    if (best !== hover) {
      hover = best;
      draw();
      if (best >= 0) {
        const p = pts[best], d = p[1] - p[0];
        tip.querySelector(".t-name").textContent = names[best];
        tip.querySelector(".t-moods").textContent = moods[best] || "—";
        tip.querySelector(".t-shift").textContent =
          `rango ${fmt(p[0])} → ${fmt(p[1])}  (${d >= 0 ? "+" : "−"}${fmt(Math.abs(d))})`;
        tip.style.display = "block";
      } else tip.style.display = "none";
    }
    if (hover >= 0) {
      const p = pts[hover];
      const cx = px(p[0] + (p[1] - p[0]) * t), cy = py(p[2]);
      tip.style.left = Math.min(W - 330, Math.max(4, cx + 14)) + "px";
      tip.style.top = Math.max(4, cy - 14 - tip.offsetHeight) + "px";
    }
  });
  canvas.addEventListener("mouseleave", () => {
    hover = -1; tip.style.display = "none"; draw();
  });
  canvas.addEventListener("click", (e) => {
    const best = nearest(e, 12);
    if (best < 0) return;          // un clic a vuoto non ferma l'ascolto
    selected = best;
    draw();
    const p = pts[best], d = p[1] - p[0];
    openBar(names[best], moods[best],
            `rango ${fmt(p[0])} → ${fmt(p[1])}  (${d >= 0 ? "+" : "−"}${fmt(Math.abs(d))})`,
            best);
  });

  const bB = document.getElementById("btn-before");
  const bA = document.getElementById("btn-after");
  function setView(after) {
    bB.setAttribute("aria-pressed", String(!after));
    bA.setAttribute("aria-pressed", String(after));
    animateTo(after ? 1 : 0);
  }
  bB.addEventListener("click", () => setView(false));
  bA.addEventListener("click", () => setView(true));
  document.getElementById("only-cross").addEventListener("change", (e) => {
    onlyCross = e.target.checked; draw();
  });

  // --- pannelli dei quadranti ---------------------------------------------
  function buildQuads() {
    const host = document.getElementById("quads");
    host.textContent = "";
    for (const q of [0, 1, 2, 3]) {
      const inflow = DATA.matrix.reduce((s, row, a) => s + (a === q ? 0 : row[q]), 0);
      const outflow = DATA.matrix[q].reduce((s, v, b) => s + (b === q ? 0 : v), 0);
      const card = document.createElement("div");
      card.className = "qcard";
      const head = document.createElement("div");
      head.className = "qhead";
      const nm = document.createElement("span");
      nm.className = "qname"; nm.textContent = QNAMES[q];
      const ct = document.createElement("span");
      ct.className = "qcount";
      ct.innerHTML = `${it(DATA.counts0[q])} → <span class="after">${it(DATA.counts1[q])}</span>`;
      head.append(nm, ct);
      const flow = document.createElement("p");
      flow.className = "qflow";
      flow.textContent = `entrano ${it(inflow)} · escono ${it(outflow)}`;
      const list = document.createElement("ul");
      list.className = "qlist";
      for (const e of DATA.examples[q]) {
        const li = document.createElement("li");
        const dot = document.createElement("span");
        dot.className = "dot";
        const [r, g, b] = shade(e.shift);
        dot.style.background = `rgb(${r},${g},${b})`;
        const name = document.createElement("span");
        name.className = "nm"; name.textContent = e.name; name.title = e.name;
        const sh = document.createElement("span");
        sh.className = "sh";
        sh.textContent = (e.shift >= 0 ? "+" : "−") + fmt(Math.abs(e.shift));
        li.append(dot, name, sh);
        li.addEventListener("click", () => {
          selected = -1;
          draw();
          openBar(e.name, e.moods,
                  `shift ${(e.shift >= 0 ? "+" : "−")}${fmt(Math.abs(e.shift))}`,
                  e.play != null ? e.play : null);
        });
        list.append(li);
      }
      card.append(head, flow, list);
      host.append(card);
    }
  }

  // --- tema e avvio --------------------------------------------------------
  function retheme() { readTokens(); buildQuads(); draw(); }
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", retheme);
  new MutationObserver(retheme).observe(document.documentElement,
    { attributes: true, attributeFilter: ["data-theme"] });
  addEventListener("resize", resize);

  readTokens();
  buildQuads();
  resize();
})();
</script>
"""


if __name__ == "__main__":
    main()
