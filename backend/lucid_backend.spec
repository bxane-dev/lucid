# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

ROOT = Path(SPECPATH).parent.parent
FRONTEND = ROOT / "frontend" / "out"

datas = [(str(FRONTEND), "frontend")]
binaries = []
hiddenimports = []

for package in ("mne", "sklearn", "scipy", "pandas"):
    datas += collect_data_files(package)
    hiddenimports += collect_submodules(package)

binaries += collect_dynamic_libs("torch")

a = Analysis(
    ["desktop_server.py"],
    pathex=[str(Path(SPECPATH))],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="lucid-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="lucid-backend",
)
