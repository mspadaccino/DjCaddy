"""Entry point dell'app Streamlit: sceglie fra le sezioni di Wavecut.

    poetry run streamlit run app.py

Ogni sezione vive in `views/` ed è uno script Streamlit a sé: qui si fa solo
la configurazione di pagina (che Streamlit accetta una volta sola, e deve
stare nell'entry point) e il menu di navigazione.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Wavecut", page_icon="🌊", layout="wide")

navigation = st.navigation([
    st.Page("views/wave_analysis.py", title="Wave analysis", icon="🌊", default=True),
    st.Page("views/tag_analysis.py", title="Tag analysis", icon="🏷️"),
    st.Page("views/folder_analysis.py", title="Folder analysis", icon="📁"),
])
navigation.run()
