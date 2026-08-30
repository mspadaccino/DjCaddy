"""Mette la radice del repo su sys.path quando gira pytest.

I test importano `core.*` e `streamlit_app.*` dalla radice, ma pytest da
solo aggiunge al path la cartella dei test, non la radice: senza questo file
`poetry run pytest` si ferma alla raccolta con ModuleNotFoundError e
funzionava solo con PYTHONPATH=. davanti. Un conftest alla radice È il modo
canonico di pytest per dire "il progetto parte da qui".
"""
