"""L'app desktop Qt6 di Wavecut, in parallel run con quella Streamlit.

Le due app leggono gli stessi store e disegnano le stesse figure: la logica
sta in `core/`, qui c'è solo il vestito Qt — finestre, widget, segnali.
Si avvia con:

    poetry run python -m qt_app.main
"""
