"""I frontend HTML riusati (lavagna, ruota) dentro Qt, con lo shim.

Le 800 righe di drag&drop della lavagna parlano l'API componenti di
Streamlit: mandano `streamlit:componentReady`, `setComponentValue` e
`setFrameHeight` a `window.parent` via postMessage, e ascoltano
`streamlit:render` con gli args dentro. Qui non c'è nessun iframe, la
pagina è caricata al livello più alto — quindi `window.parent` è la pagina
stessa, e uno script iniettato PRIMA degli script della pagina può fare da
genitore finto: ascolta quei messaggi e li gira al ponte QWebChannel,
consegna i payload come `streamlit:render`. L'HTML non si tocca di una
riga, che è tutto il punto del riuso.
"""

from __future__ import annotations

import json

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QColor
from PySide6.QtWebEngineCore import QWebEngineScript, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView

from core.viz import frontend_dir
from qt_app import theme
from qt_app.widgets.webchannel import attach_bridge, qwebchannel_source

# Lo shim vero e proprio. Tre tempi che non hanno un ordine garantito — il
# canale che si apre, la pagina che si dichiara pronta, il payload che
# arriva da Qt — quindi si tiene lo stato di tutti e tre e si consegna
# quando ci sono gli ultimi due: un render prima del componentReady andrebbe
# a una pagina che non ascolta ancora.
_SHIM = """
(function () {
  var bridge = null, component_ready = false, queued = null;
  function tell(msg) { if (bridge) bridge.event(JSON.stringify(msg)); }
  function flush() {
    if (component_ready && queued !== null) {
      var payload = queued; queued = null;
      window.postMessage({type: "streamlit:render", args: payload}, "*");
    }
  }
  window.__wavecut_render = function (payload) { queued = payload; flush(); };
  window.addEventListener("message", function (event) {
    var data = event.data || {};
    if (!data.isStreamlitMessage) return;
    if (data.type === "streamlit:componentReady") {
      component_ready = true; tell({type: "ready"}); flush();
    } else if (data.type === "streamlit:setComponentValue") {
      tell({type: "value", value: data.value});
    } else if (data.type === "streamlit:setFrameHeight") {
      tell({type: "height", height: data.height});
    }
  });
  new QWebChannel(qt.webChannelTransport, function (channel) {
    bridge = channel.objects.bridge;
    tell({type: "channel"});
  });
})();
"""


class ComponentView(QWebEngineView):
    """Un frontend di `core/viz/frontend` come widget: payload giù, valori su.

    `set_payload` prende lo stesso dict che l'adapter Streamlit passa al
    componente (gli args); `value_changed` porta quello che il frontend
    manda con setComponentValue — per la lavagna, gli spostamenti delle
    schede e il cestino. `height_suggested` è l'altezza che il frontend
    chiederebbe al suo iframe: qui è solo un suggerimento, l'altezza la
    decide il layout Qt.
    """

    value_changed = Signal(dict)
    height_suggested = Signal(int)

    def __init__(self, frontend: str, parent=None) -> None:
        super().__init__(parent)
        self.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls,
            True)
        # Il fondo della PAGINA, non del widget: dentro Streamlit l'iframe è
        # alto esattamente quanto il disegno, qui il widget può essere più
        # alto — e sotto il disegno spunterebbe il bianco di default.
        self.page().setBackgroundColor(QColor(theme.BACKGROUND))
        self._payload: dict | None = None

        bridge = attach_bridge(self.page())
        bridge.received.connect(self._on_event)

        # DocumentCreation: prima che la pagina esegua qualunque cosa, così
        # il componentReady trova l'ascoltatore già al suo posto. MainWorld
        # perché lo shim e la pagina devono vedersi (stesso window).
        shim = QWebEngineScript()
        shim.setName("wavecut-shim")
        shim.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentCreation)
        shim.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
        shim.setSourceCode(qwebchannel_source() + _SHIM)
        self.page().scripts().insert(shim)

        self.load(QUrl.fromLocalFile(
            str(frontend_dir(frontend) / "index.html")))

    def set_payload(self, payload: dict) -> None:
        """Consegna (o aggiorna) gli args del componente. Come per la mappa,
        di più payload in attesa vale l'ultimo."""
        self._payload = payload
        self.page().runJavaScript(
            f"window.__wavecut_render({json.dumps(payload)})")

    def _on_event(self, data: dict) -> None:
        kind = data.get("type")
        if kind == "ready" and self._payload is not None:
            # La pagina si è dichiarata dopo che il payload era già stato
            # mandato: lo shim l'ha in coda solo se il runJavaScript è
            # arrivato a pagina fatta; rimandarlo è innocuo (vale l'ultimo).
            self.set_payload(self._payload)
        elif kind == "value" and isinstance(data.get("value"), dict):
            self.value_changed.emit(data["value"])
        elif kind == "height":
            self.height_suggested.emit(int(data.get("height", 0)))


class BoardView(ComponentView):
    """La lavagna della playlist: il frontend `graph_board`, riusato."""

    def __init__(self, parent=None) -> None:
        super().__init__("graph_board", parent)
