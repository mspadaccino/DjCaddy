"""La presentazione condivisa fra le due app: figure, tavolozze, payload.

Regola d'oro: le funzioni di questo package ricevono dataframe e stato, e
restituiscono DATI — Figure Plotly, DataFrame, payload dict — mai un `st.*`,
mai un widget Qt. Sono il contratto che garantisce "stesso grafico nelle due
app": Streamlit e Qt disegnano quello che esce da qui, ognuna a modo suo.

Anche il tema passa da fuori: dove un colore dipende dal chiaro/scuro la
funzione prende un booleano `dark`, perché come si scopre il tema è affare
dell'app (il browser per Streamlit, la palette per Qt).
"""

from __future__ import annotations

from pathlib import Path


def frontend_dir(name: str) -> Path:
    """La cartella del frontend HTML `name` (graph_board, camelot_wheel).

    Sta qui perché gli HTML sono di tutte e due le app: Streamlit li carica
    col suo adapter componenti, Qt li caricherà in un QWebEngineView con lo
    shim. Il percorso lo dà il package, così nessuna delle due se lo
    ricostruisce a mano.
    """
    return Path(__file__).parent / "frontend" / name
