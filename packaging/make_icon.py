"""L'icona dell'app dai suoi SVG, senza portarsi dietro altri strumenti.

L'unico disegno è `qt_app/assets/djcaddy-icon.svg`, quello della finestra:
qui lo si rasterizza alle misure che macOS e Windows vogliono. Il rendering
lo fa Qt, che c'è già; l'`.icns` lo cuce `iconutil`, che sta in macOS.

    poetry run python packaging/make_icon.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "qt_app" / "assets" / "djcaddy-icon.svg"

# Le misure dell'iconset di macOS (ogni misura anche in @2x) e quelle che
# Windows mette in un .ico.
ICNS_SIZES = [16, 32, 128, 256, 512]
ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]


def render(renderer: QSvgRenderer, size: int) -> QImage:
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    renderer.render(painter)
    painter.end()
    return image


def main() -> int:
    QGuiApplication([])                 # serve al motore di rendering di Qt
    renderer = QSvgRenderer(str(SOURCE))
    if not renderer.isValid():
        raise SystemExit(f"SVG illeggibile: {SOURCE}")

    iconset = HERE / "djcaddy.iconset"
    shutil.rmtree(iconset, ignore_errors=True)
    iconset.mkdir()
    for size in ICNS_SIZES:
        render(renderer, size).save(str(iconset / f"icon_{size}x{size}.png"))
        render(renderer, size * 2).save(str(iconset / f"icon_{size}x{size}@2x.png"))

    if sys.platform == "darwin":
        subprocess.run(["iconutil", "-c", "icns", str(iconset),
                        "-o", str(HERE / "djcaddy.icns")], check=True)
        shutil.rmtree(iconset)
        print(f"scritto {HERE / 'djcaddy.icns'}")

    # Il .ico di Windows: Qt scrive un ICO alla volta, quindi la misura
    # grande — quella che Windows scala meglio.
    ico = HERE / "djcaddy.ico"
    render(renderer, max(ICO_SIZES)).save(str(ico))
    print(f"scritto {ico}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
