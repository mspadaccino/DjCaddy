"""L'SVG del marchio, ricomposto a runtime frame per frame.

Stessa geometria di `assets/djcaddy-mark.svg` (viewBox 100×100), ma con
anello, barre e cue parametrici: header e splash lo usano per disegnare lo
stato esatto di un istante — anello che si chiude, barre che crescono, cue
che scatta — passando i numeri invece di tenere quattro `QPropertyAnimation`
sincronizzati a mano. Generare la stringa e renderla con `QSvgRenderer` è
anche il modo più semplice per riottenere l'identica curva "A 38 38 0 1 0"
dell'anello senza rifare a mano la trigonometria dell'arco.
"""

from __future__ import annotations

_RING_D = "M81.1 71.8 A 38 38 0 1 0 40.2 86.7"
_RING_LEN = 200  # stroke-dasharray del riferimento: copre l'intero anello

# x, y, w, h, rx, colore — dal design handoff (geometria del marchio).
_BARS = [
    (25.5, 42, 9, 28, 4.5, "#FFB37A"),
    (35.5, 32, 9, 38, 4.5, "#FF8F4D"),
    (45.5, 24, 9, 46, 4.5, "#FF6A2B"),
    (58.5, 18, 9, 78, 4.5, "#FF6A2B"),
]
_CUE = (86.5, 35, 5, "#3EE5A0")


def build_mark_svg(
    size: int,
    *,
    ring_dashoffset: float = 0.0,
    ring_opacity: float = 1.0,
    bar_scales: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    cue_scale: float = 1.0,
    cue_opacity: float = 1.0,
    pulse: float = 1.0,
) -> bytes:
    """Il marchio come bytes SVG, pronto per `QSvgRenderer`.

    Ogni parametro rispecchia una riga della tabella di animazione dello
    splash; l'header lo chiama sempre a "tutto rivelato" (i default),
    variando solo `pulse`.
    """
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" '
        f'height="{size}" viewBox="0 0 100 100">',
        f'<g transform="translate(50 50) scale({pulse}) translate(-50 -50)">',
        f'<path d="{_RING_D}" fill="none" stroke="#9184D9" stroke-width="8" '
        f'stroke-linecap="round" stroke-dasharray="{_RING_LEN}" '
        f'stroke-dashoffset="{ring_dashoffset}" opacity="{ring_opacity}"/>',
    ]
    for (x, y, w, h, rx, color), scale in zip(_BARS, bar_scales):
        bottom = y + h
        cx = x + w / 2
        parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{color}" transform="translate({cx} {bottom}) '
            f'scale(1 {scale}) translate({-cx} {-bottom})"/>')
    cx, cy, r, color = _CUE
    parts.append(
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}" '
        f'opacity="{cue_opacity}" transform="translate({cx} {cy}) '
        f'scale({cue_scale}) translate({-cx} {-cy})"/>')
    parts.append("</g></svg>")
    return "".join(parts).encode("utf-8")
