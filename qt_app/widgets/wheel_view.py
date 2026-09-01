"""La ruota Camelot come widget: il frontend HTML riusato, via shim.

Due tonalità che si mixano stanno vicine sulla ruota, e su una ruota la
cosa si vede — in un elenco alfabetico 8A e 9A sono due righe qualunque.
Il disegno è lo stesso HTML di Streamlit (`core/viz/frontend/camelot_wheel`)
dentro un ComponentView; qui c'è solo la traduzione fra il suo protocollo
e un segnale Qt.
"""

from __future__ import annotations

from PySide6.QtCore import Signal

from core.viz.board import wheel_payload
from qt_app import theme
from qt_app.widgets.board_view import ComponentView


class WheelView(ComponentView):
    """`set_keys` dice quali tonalità sono scelte, `key_toggled` porta il
    codice cliccato. CHI tiene l'elenco è chi ascolta: la ruota non sa di
    che filtro fa parte, esattamente come in Streamlit.
    """

    key_toggled = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__("camelot_wheel", parent)
        self._seen_at = None
        self.value_changed.connect(self._on_value)

    def set_keys(self, selected: list[str], dark: bool | None = None) -> None:
        """Il tema è quello dell'app, se non lo si dice: al cambio la ruota
        si ridipinge da sé — `ComponentView` rimanda il payload."""
        self.set_payload(wheel_payload(
            list(selected), theme.DARK if dark is None else dark))

    def _on_value(self, value: dict) -> None:
        # Il click si riconosce dal suo istante, come fa l'adapter
        # Streamlit: un payload rimandato non è un gesto nuovo.
        if value.get("at") == self._seen_at:
            return
        self._seen_at = value.get("at")
        code = value.get("code")
        if code:
            self.key_toggled.emit(str(code))
