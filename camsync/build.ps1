# build.ps1 — build PawCapture as a portable PyInstaller --onedir bundle.
#
# Usage (from the camsync directory):
#   .\build.ps1            # incremental build
#   .\build.ps1 -Clean     # wipe build/ and dist/ first
#
# Output: dist\PawCapture\  — copy or zip this whole folder to deploy.
# End users only need the folder; no Python install required.
#
# First-time setup on a clean machine:
#   1. Install Python 3.10+ (python.org installer, "Add to PATH").
#   2. Drop ffmpeg.exe into camsync\ffmpeg\  (official Windows build).
#   3. Run this script — it will install requirements.txt and build.

param(
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'

# `py -3` is the Windows Python launcher; more reliable than bare `python`,
# which on stock Win10/11 often resolves to the Microsoft Store stub.
$Py = 'py'
$PyArgs = @('-3')

if ($Clean) {
    Write-Host "Cleaning build/ and dist/..." -ForegroundColor Cyan
    Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
}

if (-not (Test-Path 'ffmpeg\ffmpeg.exe')) {
    Write-Error "Missing ffmpeg\ffmpeg.exe. Drop an official Windows ffmpeg build there before building."
    exit 1
}

Write-Host "Installing/refreshing dependencies..." -ForegroundColor Cyan
& $Py @PyArgs -m pip install --quiet --upgrade pip
& $Py @PyArgs -m pip install --quiet -r requirements.txt

Write-Host "Running PyInstaller..." -ForegroundColor Cyan
& $Py @PyArgs -m PyInstaller --noconfirm --clean PawCapture.spec

$out = "dist\PawCapture\PawCapture.exe"
if (Test-Path $out) {
    $size = "{0:N1} MB" -f ((Get-ChildItem -Recurse "dist\PawCapture" |
                              Measure-Object Length -Sum).Sum / 1MB)
    Write-Host ""
    Write-Host "Built: $out ($size total)" -ForegroundColor Green
    Write-Host "To distribute: zip the dist\PawCapture folder and ship it."
} else {
    Write-Error "Build did not produce $out — check PyInstaller output above."
    exit 1
}
