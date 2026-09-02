#!/usr/bin/env bash
#
# Il bundle di macOS: DjCaddy.app e il DMG che lo consegna.
#
#     ./packaging/build_macos.sh
#
# Vuole l'ambiente completo (`poetry install`, essentia compresa), ffmpeg
# installato, i modelli Essentia in ~/essentia_models e il checkpoint Demucs
# già in cache: la spec si ferma da sola dicendo cosa manca. Il risultato è
# 3–4 GB e ci mette parecchi minuti — è il costo dell'autonomia totale.
#
# Firma ad-hoc, non notarizzata: basta a farla girare su questa macchina e su
# chi la apre col tasto destro → Apri. Per distribuirla fuori serve un
# Developer ID e `xcrun notarytool`, che non fanno parte di questo script.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)"
DMG="dist/DjCaddy-${VERSION}.dmg"

echo "==> icona"
poetry run python packaging/make_icon.py

echo "==> bundle"
poetry run pyinstaller packaging/djcaddy.spec --noconfirm \
    --distpath dist --workpath build/pyinstaller

echo "==> firma ad-hoc"
codesign --force --sign - --timestamp=none dist/DjCaddy.app

echo "==> DMG"
rm -f "$DMG"
hdiutil create -volname "DjCaddy" -srcfolder dist/DjCaddy.app \
    -ov -format UDZO "$DMG"

echo
echo "Fatto:"
du -sh dist/DjCaddy.app "$DMG"
