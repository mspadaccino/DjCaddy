"""L'SVG del marchio, ricomposto a runtime.

Viewbox 100×100: l'header lo usa per disegnare il marchio, a riposo o che
pulsa (`pulse`). Generare la stringa e renderla con `QSvgRenderer` è il
modo più semplice per ottenere la curva "A 38 38 0 1 0" dell'anello senza
rifare a mano la trigonometria dell'arco.
"""

from __future__ import annotations

_RING_D = "M81.1 71.8 A 38 38 0 1 0 40.2 86.7"

# x, y, w, h, rx, colore — dal design handoff (geometria del marchio).
_BARS = [
    (25.5, 42, 9, 28, 4.5, "#FFB37A"),
    (35.5, 32, 9, 38, 4.5, "#FF8F4D"),
    (45.5, 24, 9, 46, 4.5, "#FF6A2B"),
    (58.5, 18, 9, 78, 4.5, "#FF6A2B"),
]
_CUE = (86.5, 35, 5, "#3EE5A0")


def build_mark_svg(size: int, *, pulse: float = 1.0) -> bytes:
    """Il marchio come bytes SVG, pronto per `QSvgRenderer`, scalato di
    `pulse` intorno al proprio centro (1.0 = a riposo)."""
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" '
        f'height="{size}" viewBox="0 0 100 100">',
        f'<g transform="translate(50 50) scale({pulse}) translate(-50 -50)">',
        f'<path d="{_RING_D}" fill="none" stroke="#9184D9" stroke-width="8" '
        f'stroke-linecap="round"/>',
    ]
    for x, y, w, h, rx, color in _BARS:
        parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{color}"/>')
    cx, cy, r, color = _CUE
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}"/>')
    parts.append("</g></svg>")
    return "".join(parts).encode("utf-8")
