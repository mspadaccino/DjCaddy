"""La mappa Plotly dentro Qt: stessa figura, stesso motore, senza browser.

Un QWebEngineView carica una pagina locale con plotly.min.js preso dal
pacchetto Python di Plotly — niente CDN: l'app deve funzionare offline, e
nel bundle il file viaggia con il pacchetto. `set_figure` consegna il JSON
della figura a `Plotly.react`, che ridisegna la differenza invece di rifare
la pagina; gli eventi del grafico — clic su un punto, lasso, riquadro,
deselezione — tornano indietro dal ponte QWebChannel come segnali Qt con gli
INDICI di libreria dei brani (il `customdata[0]` che `core.viz.build_figure`
mette su ogni punto proprio per questo).

**Due canali, non uno.** Lo spike della Fase 2 ha misurato che il costo di
un gesto non sta nel ridisegno ma nel rifare figura+JSON in Python a mappa
piena (~1,4 s, ~15 MB): la nuvola non cambia mai a un clic, cambiano solo il
seme, gli anelli e il percorso. Quindi `set_figure` manda la NUVOLA (i
tracciati per genere e le etichette), la pagina se la tiene, e
`set_overlays` manda solo i tracciati di contorno: il JS incolla i secondi
in coda ai primi e richiama `Plotly.react`, che riconosce i tracciati di
base per identità e non li tocca. `layout.uirevision` fisso fa il resto:
zoom, pan e i generi spenti in legenda sopravvivono a ogni aggiornamento.
"""

from __future__ import annotations

from pathlib import Path

import plotly

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QColor
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView

from core.viz.map_figure import SKIN
from qt_app import theme
from qt_app.widgets.webchannel import attach_bridge


def plotly_package_data() -> Path:
    """La cartella del pacchetto Plotly con dentro plotly.min.js.

    È la `baseUrl` della pagina: lo <script src="plotly.min.js"> del
    template si risolve qui, quindi il file non si copia da nessuna parte —
    né adesso né nel bundle, dove il pacchetto c'è comunque.
    """
    return Path(plotly.__file__).parent / "package_data"


# La pagina è piccola apposta: `setHtml` accetta al massimo 2 MB, quindi
# plotly.min.js (4,6 MB) NON può stare inline — arriva dalla baseUrl.
_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<script src="plotly.min.js"></script>
<style>
  html, body { margin: 0; height: 100%; background: BACKGROUND; }
  :root { --playing: PLAYING; --ink: INK; --dim: DIM; }
  /* La barra degli strumenti — zoom, pan, riquadro, lazo, ripristina —
     sempre in vista e coi colori del tema: di suo Plotly la mostra solo
     al passaggio del mouse, in un grigio scuro semitrasparente che sul
     fondo scuro non si vede. I colori Plotly li scrive in regole sue,
     per id: da qui gli `!important`. */
  .modebar-btn .icon path { fill: var(--dim) !important; }
  .modebar-btn:hover .icon path,
  .modebar-btn.active .icon path { fill: var(--ink) !important; }
  #map { width: 100%; height: 100%; }
  /* Una figura può essere più larga del riquadro — l'impronta degli
     embedding a 1280 colonne lo è — e allora la pagina scorre di lato.
     Plotly la barra degli strumenti la incolla all'angolo in alto a destra
     DELLA FIGURA, che così finisce fuori dallo schermo: fissata alla
     finestra resta dov'è sempre stata per le figure che ci stanno, e
     raggiungibile per quelle che non ci stanno. */
  .modebar-container { position: fixed !important; }
  /* Il brano in ascolto: l'annotazione che `build_figure` chiama
     «playing», e che `tagPlaying` qui sotto marca dopo ogni disegno. Il
     suo quadrato di sfondo diventa un cerchio (rx) e batte come un cuore,
     via CSS, che alla nuvola non costa niente: due colpi ravvicinati di
     bordo e riempimento, poi la pausa. NESSUNA trasformazione: `scale`
     si comporrebbe col translate che Plotly scrive nell'attributo
     transform e il punto girerebbe per la mappa. Il fill sta nei
     keyframes perché Plotly lo scrive inline, e solo l'animazione vince
     sull'inline. Il colore è `SKIN["playing"]` del tema in corso —
     bianco sullo scuro, quasi nero sul chiaro — passato come variabile
     CSS così da seguirlo al cambio senza rifare la pagina. */
  .djcaddy-playing rect.bg {
    rx: 50%; animation: djcaddy-beat 1.2s ease-out infinite;
  }
  @keyframes djcaddy-beat {
    0%   { stroke-width: 3.5px; fill: var(--playing); fill-opacity: .25; }
    12%  { stroke-width: 9px;   fill: var(--playing); fill-opacity: .95; }
    28%  { stroke-width: 3.5px; fill: var(--playing); fill-opacity: .25; }
    40%  { stroke-width: 9px;   fill: var(--playing); fill-opacity: .95; }
    60%  { stroke-width: 3.5px; fill: var(--playing); fill-opacity: .25; }
    100% { stroke-width: 3.5px; fill: var(--playing); fill-opacity: .25; }
  }
</style>
</head><body><div id="map"></div>
<script>
(function () {
  var bridge = null;
  var config = {displaylogo: false, displayModeBar: true, scrollZoom: true,
                responsive: true};
  // La base è la nuvola dell'ultima `render`: i suoi tracciati restano gli
  // STESSI oggetti fra un gesto e l'altro, ed è per identità che react
  // capisce di non doverli ridisegnare.
  var base = null;
  // L'ultimo contorno ricevuto: se arriva prima della nuvola — o mentre
  // quella di prima è ancora per strada — si riappiccica appena c'è una
  // base sotto, invece di sparire senza dire niente.
  var pending = null;
  function tell(msg) { if (bridge) bridge.event(JSON.stringify(msg)); }

  // Dal punto disegnato all'indice di libreria: customdata[0]. I tracciati
  // di contorno (anelli, percorso, seme) non ce l'hanno, e non sono brani.
  function indices(points) {
    var out = [];
    (points || []).forEach(function (p) {
      if (p.customdata !== undefined) out.push(p.customdata[0]);
    });
    return out;
  }

  // L'annotazione del brano in ascolto: Plotly non le dà una classe sua,
  // ma scrive l'indice — e in `layout.annotations` a quell'indice c'è il
  // `name`. Va rifatto dopo OGNI ridisegno, non solo dopo i nostri: un
  // lasso o uno zoom ricreano i nodi delle annotazioni, e la classe se ne
  // andrebbe con quelli vecchi — da qui `plotly_afterplot` qui sotto.
  function tagPlaying(gd) {
    var notes = gd.layout.annotations || [];
    gd.querySelectorAll(".annotation").forEach(function (g) {
      var note = notes[+g.getAttribute("data-index")];
      g.classList.toggle("djcaddy-playing", !!(note && note.name === "playing"));
    });
  }

  function react(data, layout) {
    var began = performance.now();
    Plotly.react(document.getElementById("map"), data, layout, config)
      .then(function (gd) {
        tagPlaying(gd);
        if (!gd._djcaddy_wired) {
          // Una volta sola: il div sopravvive alle react successive, e
          // gli ascoltatori con lui.
          gd._djcaddy_wired = true;
          gd.on("plotly_afterplot", function () { tagPlaying(gd); });
          gd.on("plotly_click", function (e) {
            var hit = indices(e.points);
            if (hit.length) tell({type: "click", index: hit[0]});
          });
          gd.on("plotly_selected", function (e) {
            tell({type: "selected", indices: e ? indices(e.points) : []});
          });
          gd.on("plotly_deselect", function () {
            tell({type: "deselected"});
          });
        }
        tell({type: "rendered", ms: performance.now() - began});
      });
  }

  window.djcaddy = {
    render: function (spec) {
      // Lo zoom, il pan e le voci spente in legenda restano dove sono a
      // ogni aggiornamento: è il contratto di uirevision.
      spec.layout.uirevision = "djcaddy";
      // Una figura può chiedere di essere più larga del riquadro — è come
      // l'impronta degli embedding si fa scorrere di lato invece di
      // schiacciare 1280 colonne in ottocento pixel. Plotly la larghezza
      // la scrive nello stile del div, quindi va anche CANCELLATA quando la
      // figura dopo non la chiede: restando lì, la misura del riquadro
      // diventerebbe quella vecchia e il disegno non tornerebbe più elastico.
      var div = document.getElementById("map");
      div.style.width = spec.layout.width ? spec.layout.width + "px" : "";
      base = {data: spec.data, layout: spec.layout,
              notes: (spec.layout.annotations || [])};
      react(spec.data, spec.layout);
      if (pending) window.djcaddy.overlays(pending);
    },
    overlays: function (spec) {
      pending = spec;
      if (!base) return;   // nessuna nuvola sotto: non c'è dove appoggiarli
      var notes = ((spec.layout || {}).annotations) || [];
      // Un layout NUOVO a ogni giro: react confronta per riferimento, e un
      // oggetto mutato sul posto passerebbe per già visto.
      var layout = Object.assign({}, base.layout,
                                 {annotations: base.notes.concat(notes)});
      react(base.data.concat(spec.data), layout);
    },
  };

  new QWebChannel(qt.webChannelTransport, function (channel) {
    bridge = channel.objects.bridge;
    tell({type: "ready"});
  });
})();
</script></body></html>"""


def _playing_colour() -> str:
    """Il rosso del brano in ascolto, lo stesso del suo anello in figura."""
    return SKIN["dark" if theme.DARK else "light"]["playing"]


class PlotlyView(QWebEngineView):
    """Il grafico come widget: `set_figure(figura)` e i segnali di scelta.

    La figura si può dare da subito: finché la pagina non dice `ready`
    resta in attesa, e parte da sola al primo giro del ponte. Se ne arriva
    più d'una nel frattempo vale l'ultima — le altre non sono mai state
    sullo schermo e non devono passarci. Lo stesso per i contorni.
    """

    point_clicked = Signal(int)
    points_selected = Signal(list)
    deselected = Signal()
    rendered = Signal(float)            # ms di Plotly.react, per misurare

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # La pagina è un file locale che carica un altro file locale: il
        # permesso va detto, di suo QtWebEngine non si fida.
        self.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls,
            True)
        # Anche il fondo della pagina: senza, prima che il CSS arrivi c'è
        # un lampo bianco sul tema scuro.
        self.page().setBackgroundColor(QColor(theme.BACKGROUND))
        self._ready = False
        self._queued: str | None = None
        self._queued_overlays: str | None = None
        bridge = attach_bridge(self.page())
        bridge.received.connect(self._on_event)
        self.setHtml(_PAGE.replace("BACKGROUND", theme.BACKGROUND)
                     .replace("PLAYING", _playing_colour())
                     .replace("INK", theme.INK).replace("DIM", theme.FADED),
                     QUrl.fromLocalFile(str(plotly_package_data()) + "/"))
        theme.bus().changed.connect(self._on_theme)

    def _on_theme(self) -> None:
        """Il fondo della pagina segue il tema. La FIGURA no: i suoi colori
        sono cotti nel JSON, e a rifarla è chi l'ha costruita — qui si
        cambia solo la superficie sotto, perché fra la richiesta e la
        figura nuova non ci sia un lampo del tema di prima."""
        self.page().setBackgroundColor(QColor(theme.BACKGROUND))
        self.page().runJavaScript(
            f"document.body.style.background = '{theme.BACKGROUND}';"
            f"document.documentElement.style.setProperty('--playing', "
            f"'{_playing_colour()}');"
            f"document.documentElement.style.setProperty('--ink', "
            f"'{theme.INK}');")

    def set_figure(self, figure) -> None:
        """Mostra (o aggiorna) la figura di base — un oggetto con `to_json`,
        o direttamente la stringa JSON se chi chiama l'ha già."""
        spec = figure if isinstance(figure, str) else figure.to_json()
        if not self._ready:
            self._queued = spec
            return
        self.page().runJavaScript(f"window.djcaddy.render({spec})")

    def set_overlays(self, figure) -> None:
        """Aggiorna i soli tracciati di contorno sopra l'ultima figura di
        base: una figura Plotly SENZA nuvola — anelli, percorso, seme — i
        cui tracciati vengono incollati in coda a quelli di base."""
        spec = figure if isinstance(figure, str) else figure.to_json()
        if not self._ready:
            self._queued_overlays = spec
            return
        self.page().runJavaScript(f"window.djcaddy.overlays({spec})")

    def _on_event(self, data: dict) -> None:
        kind = data.get("type")
        if kind == "ready":
            self._ready = True
            if self._queued is not None:
                spec, self._queued = self._queued, None
                self.page().runJavaScript(f"window.djcaddy.render({spec})")
            if self._queued_overlays is not None:
                spec, self._queued_overlays = self._queued_overlays, None
                self.page().runJavaScript(f"window.djcaddy.overlays({spec})")
        elif kind == "click":
            self.point_clicked.emit(int(data["index"]))
        elif kind == "selected":
            self.points_selected.emit(
                [int(i) for i in data.get("indices", [])])
        elif kind == "deselected":
            self.deselected.emit()
        elif kind == "rendered":
            self.rendered.emit(float(data.get("ms", 0.0)))
