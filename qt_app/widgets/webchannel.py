"""Il ponte fra una pagina in QWebEngineView e i segnali Qt.

Due widget parlano con del JavaScript — la mappa Plotly e i frontend HTML
riusati (lavagna, ruota) — e il canale è lo stesso per tutti: un oggetto
`bridge` registrato su QWebChannel, a cui il JS consegna eventi come JSON.
Una stringa JSON e non gli argomenti tipati del canale, così il contratto
resta uno solo e ogni widget decide le sue chiavi.
"""

from __future__ import annotations

import json

from PySide6.QtCore import QFile, QIODevice, QObject, Signal, Slot
from PySide6.QtWebChannel import QWebChannel


def qwebchannel_source() -> str:
    """Il sorgente di qwebchannel.js, letto dalle risorse di Qt.

    Serve a iniettarlo nelle pagine che non lo dichiarano da sé — i frontend
    HTML riusati, che di QWebChannel non sanno niente. La pagina della mappa
    invece lo carica col suo <script src="qrc:...">, che è la stessa risorsa
    per l'altra porta.
    """
    handle = QFile(":/qtwebchannel/qwebchannel.js")
    if not handle.open(QIODevice.OpenModeFlag.ReadOnly):
        raise RuntimeError("qwebchannel.js non è nelle risorse di Qt")
    try:
        return bytes(handle.readAll()).decode("utf-8")
    finally:
        handle.close()


class JsonBridge(QObject):
    """L'oggetto che il JS vede come `bridge`: riceve JSON, emette dict."""

    received = Signal(dict)

    @Slot(str)
    def event(self, payload: str) -> None:
        try:
            data = json.loads(payload)
        except ValueError:
            return                      # non è JSON: non è roba nostra
        if isinstance(data, dict):
            self.received.emit(data)


def attach_bridge(page) -> JsonBridge:
    """Registra un `bridge` sul canale della pagina e lo ritorna.

    Canale e bridge prendono la pagina come parent: devono vivere quanto
    lei, e senza parent il garbage collector di Python se li porterebbe via
    con il canale ancora aperto.
    """
    bridge = JsonBridge(page)
    channel = QWebChannel(page)
    channel.registerObject("bridge", bridge)
    page.setWebChannel(channel)
    return bridge
