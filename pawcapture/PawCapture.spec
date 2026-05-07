# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for PawCapture (camsync_precursor.py).
#
# Produces a flat --onedir bundle at dist\PawCapture\:
#   PawCapture.exe          ← launcher (windowed, no console)
#   ffmpeg.exe              ← bundled, picked up by _find_ffmpeg()
#   *.dll, Qt plugins, etc. ← Python + PyQt5 + cv2 runtime
#
# Build:  py -3 -m PyInstaller --noconfirm --clean PawCapture.spec
# (or run build.ps1 from the pawcapture directory)

from pathlib import Path

ROOT       = Path(SPECPATH)
FFMPEG_SRC = ROOT / "ffmpeg" / "ffmpeg.exe"
if not FFMPEG_SRC.exists():
    raise SystemExit(
        f"ffmpeg.exe not found at {FFMPEG_SRC}. "
        "Drop the official Windows ffmpeg build into pawcapture\\ffmpeg\\ before building."
    )

a = Analysis(
    ['camsync_precursor.py'],
    pathex=[str(ROOT)],
    # ('source', '.') drops ffmpeg.exe directly next to PawCapture.exe in the
    # output dir, so _find_ffmpeg()'s first lookup (Path(sys.executable).parent
    # / "ffmpeg.exe") finds it without any code change.
    binaries=[(str(FFMPEG_SRC), '.')],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Heavy or never-imported PyQt5 submodules — trimming these saves
        # ~150 MB off the bundle. Add back if a future feature needs them.
        'PyQt5.QtWebEngine', 'PyQt5.QtWebEngineCore',
        'PyQt5.QtWebEngineWidgets',
        'PyQt5.QtBluetooth', 'PyQt5.QtPositioning',
        'PyQt5.QtMultimedia', 'PyQt5.QtMultimediaWidgets',
        'PyQt5.QtNfc', 'PyQt5.QtSensors', 'PyQt5.QtSerialPort',
        'PyQt5.QtRemoteObjects', 'PyQt5.QtQml', 'PyQt5.QtQuick',
        'PyQt5.QtQuickWidgets', 'PyQt5.QtSql', 'PyQt5.QtTest',
        # Stdlib UI we don't use
        'tkinter',
        # Common heavy science libs that some pip envs have installed but
        # PawCapture never imports
        'matplotlib', 'pandas', 'IPython', 'notebook', 'scipy', 'sympy',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PawCapture',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX compression often triggers Windows Defender false positives on
    # PyInstaller bundles — leave it off for distribution sanity.
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Flat layout — Python deps land alongside PawCapture.exe instead of in
    # _internal\. Keeps ffmpeg.exe visible to the user (so they can swap in
    # a newer build if they want) and matches the lookup path _find_ffmpeg
    # uses.
    contents_directory='.',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='PawCapture',
)
