@echo off
REM build.bat — PyInstaller bundle build (no PowerShell execution-policy hassle).
REM
REM Usage:  build.bat           (incremental)
REM         build.bat --clean   (wipe build/ and dist/ first)
REM
REM First-time setup:
REM   1. Install Python 3.10+ (python.org installer, "Add to PATH").
REM   2. Drop ffmpeg.exe into pawcapture\ffmpeg\ (official Windows build).
REM   3. Run this script.

setlocal
cd /d "%~dp0"

if /i "%1"=="--clean" (
    echo Cleaning build\ and dist\...
    if exist build rmdir /s /q build
    if exist dist  rmdir /s /q dist
)

if not exist "ffmpeg\ffmpeg.exe" (
    echo ERROR: missing ffmpeg\ffmpeg.exe — drop an official Windows ffmpeg build there before building.
    exit /b 1
)

echo Installing/refreshing dependencies...
py -3 -m pip install --quiet --upgrade pip || goto :err
py -3 -m pip install --quiet -r requirements.txt || goto :err

echo Running PyInstaller...
py -3 -m PyInstaller --noconfirm --clean PawCapture.spec || goto :err

if not exist "dist\PawCapture\PawCapture.exe" (
    echo ERROR: build did not produce dist\PawCapture\PawCapture.exe — check PyInstaller output above.
    exit /b 1
)

echo.
echo Built: dist\PawCapture\PawCapture.exe
echo To distribute: zip the dist\PawCapture folder and ship it.
exit /b 0

:err
echo Build failed.
exit /b 1
