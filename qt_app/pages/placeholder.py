"""Il segnaposto delle sezioni che arrivano con le fasi successive."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


def placeholder(title: str, phase: str) -> QWidget:
    """Una pagina che dice solo cosa sarà e quando: meglio di una tab che
    manca — le quattro sezioni esistono dal primo giorno, come di là."""
    page = QWidget()
    label = QLabel(f"{title} — arriva con la {phase}.\n"
                   "Nel parallel run, per ora, si usa la pagina Streamlit.")
    label.setObjectName("dim")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout = QVBoxLayout(page)
    layout.addWidget(label)
    return page
