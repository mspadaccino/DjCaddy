"""La mappa Plotly dentro Qt: stessa figura, stesso motore, senza browser.

Un QWebEngineView carica una pagina locale con plotly.min.js preso dal
pacchetto Python di Plotly — niente CDN: l'app deve funzionare offline, e
nel bundle il file viaggia con il pacchetto. `set_figure` consegna il JSON
della figura a `Plotly.react`, che ridisegna la differenza invece di rifare
la pagina; gli eventi del grafico — clic su un punto, lasso, riquadro,
deselezione — tornano indietro dal ponte QWebChannel come segnali Qt con gli
INDICI di libreria dei brani (il `customdata[0]` che `core.viz.build_figure`
mette su ogni punto proprio per questo).
"""

from __future__ import annotations

from pathlib import Path

import plotly

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QColor
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView

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
  #map { width: 100%; height: 100%; }
</style>
</head><body><div id="map"></div>
<script>
(function () {
  var bridge = null;
  var config = {displaylogo: false, scrollZoom: true, responsive: true};
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

  window.wavecut = {
    render: function (spec) {
      var began = performance.now();
      Plotly.react(document.getElementById("map"),
                   spec.data, spec.layout, config)
        .then(function (gd) {
          if (!gd._wavecut_wired) {
            // Una volta sola: il div sopravvive alle react successive, e
            // gli ascoltatori con lui.
            gd._wavecut_wired = true;
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
    },
  };

  new QWebChannel(qt.webChannelTransport, function (channel) {
    bridge = channel.objects.bridge;
    tell({type: "ready"});
  });
})();
</script></body></html>"""


class PlotlyView(QWebEngineView):
    """Il grafico come widget: `set_figure(figura)` e i segnali di scelta.

    La figura si può dare da subito: finché la pagina non dice `ready`
    resta in attesa, e parte da sola al primo giro del ponte. Se ne arriva
    più d'una nel frattempo vale l'ultima — le altre non sono mai state
    sullo schermo e non devono passarci.
    """

    point_clicked = Signal(int)
    points_selected = Signal(list)
    deselected = Signal()
    rendered = Signal(float)            # ms di Plotly.react, per misurare

    def __init__(self, background: str = "#0e1117", parent=None) -> None:
        super().__init__(parent)
        # La pagina è un file locale che carica un altro file locale: il
        # permesso va detto, di suo QtWebEngine non si fida.
        self.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls,
            True)
        # Anche il fondo della pagina: senza, prima che il CSS arrivi c'è
        # un lampo bianco sul tema scuro.
        self.page().setBackgroundColor(QColor(background))
        self._ready = False
        self._queued: str | None = None
        bridge = attach_bridge(self.page())
        bridge.received.connect(self._on_event)
        self.setHtml(_PAGE.replace("BACKGROUND", background),
                     QUrl.fromLocalFile(str(plotly_package_data()) + "/"))

    def set_figure(self, figure) -> None:
        """Mostra (o aggiorna) una figura Plotly — un oggetto con `to_json`,
        o direttamente la stringa JSON se chi chiama l'ha già."""
        spec = figure if isinstance(figure, str) else figure.to_json()
        if not self._ready:
            self._queued = spec
            return
        self._run_render(spec)

    def _run_render(self, spec: str) -> None:
        self.page().runJavaScript(f"window.wavecut.render({spec})")

    def _on_event(self, data: dict) -> None:
        kind = data.get("type")
        if kind == "ready":
            self._ready = True
            if self._queued is not None:
                spec, self._queued = self._queued, None
                self._run_render(spec)
        elif kind == "click":
            self.point_clicked.emit(int(data["index"]))
        elif kind == "selected":
            self.points_selected.emit(
                [int(i) for i in data.get("indices", [])])
        elif kind == "deselected":
            self.deselected.emit()
        elif kind == "rendered":
            self.rendered.emit(float(data.get("ms", 0.0)))
