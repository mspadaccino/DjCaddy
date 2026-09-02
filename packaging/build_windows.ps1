# Il bundle di Win11: DjCaddy.exe e il setup che lo consegna.
#
#     powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
#
# Da lanciare SU Windows: PyInstaller non compila per un'altra piattaforma.
# Vuole l'ambiente installato senza essentia — su Windows non esiste wheel —
# più ffmpeg e ffprobe sul PATH, i modelli Essentia in %USERPROFILE%\essentia_models
# (servono lo stesso a chi legge una mappa fatta sul Mac) e il checkpoint
# Demucs già in cache. Inno Setup 6 (`iscc`) serve solo per l'ultimo passo.
#
#     poetry install --without essentia

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "==> icona"
poetry run python packaging\make_icon.py

Write-Host "==> bundle"
poetry run pyinstaller packaging\djcaddy.spec --noconfirm `
    --distpath dist --workpath build\pyinstaller

Write-Host "==> setup"
$iscc = Get-Command iscc -ErrorAction SilentlyContinue
if ($null -eq $iscc) {
    Write-Warning "Inno Setup (iscc) non trovato: dist\DjCaddy\ e' pronto, il setup no."
    exit 0
}
& $iscc.Source packaging\djcaddy.iss
