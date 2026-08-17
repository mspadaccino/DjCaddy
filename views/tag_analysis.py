"""Sezione "Tag analysis": genere e mood nei tag, con i modelli Essentia.

In costruzione. Per ora la pagina riporta soltanto se l'ambiente è pronto,
che è l'informazione utile prima di scaricare 424 MB di dipendenza.
"""

from __future__ import annotations

import streamlit as st

from analysis.essentia_tags import MODEL_DIR, MODELS, available, missing_models

st.title("🏷️ Tag analysis")
st.caption(
    "Analyze tracks with the Essentia models and write **genre** and **mood** "
    "into the file tags — mood goes into the default comment field, which is "
    "the one djay Pro actually displays."
)

st.info(
    "Not built yet. Coming here: analysis of a single track, a selection, a "
    "folder or a whole subtree, with every option the terminal script asks "
    "for turned into a control — plus a separate command that launches the "
    "whole library as a background job."
)

st.subheader("Environment")

if available():
    st.success("`essentia` is importable in this environment.")
else:
    st.warning(
        "`essentia` is not installed here. It is not a base dependency: it "
        "weighs 424 MB (TensorFlow ships inside it) and PyPI only publishes "
        "it as a pre-release, so it has to be asked for explicitly:\n\n"
        "```bash\npoetry install --with essentia\n```"
    )

missing = missing_models()
if missing:
    st.warning(f"{len(missing)} model file(s) missing from `{MODEL_DIR}`.")
else:
    st.success(f"All {len(MODELS)} model files found in `{MODEL_DIR}`.")

st.dataframe(
    [{"file": name, "purpose": purpose, "present": name not in missing}
     for name, purpose in MODELS.items()],
    width="stretch", hide_index=True,
)
