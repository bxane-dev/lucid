# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)


SPEC_DIR = Path(SPEC).resolve().parent
ROOT = SPEC_DIR.parent
FRONTEND = ROOT / "frontend" / "out"

datas = [(str(FRONTEND), "frontend")]
# MNE uses lazy_loader with .pyi stubs at runtime. PyInstaller treats those
# type stubs as non-code data, so collect them explicitly.
datas += collect_data_files("mne", includes=["**/*.pyi"])
binaries = collect_dynamic_libs("torch")

# Uvicorn loads its selected loop/protocol implementations by string, so
# declare the exact implementations used by desktop_server.py.
MNE_EXCLUDED_PREFIXES = (
    "mne.tests",
    "mne.viz",
    "mne.report",
    "mne.gui",
    "mne.datasets",
    "mne.commands",
    "mne.export",
)

hiddenimports = [
    name
    for name in collect_submodules("mne")
    if not name.startswith(MNE_EXCLUDED_PREFIXES)
]
hiddenimports += collect_submodules("torch.testing")
hiddenimports += [
    "edfio",
    "uvicorn.logging",
    "uvicorn.loops.asyncio",
    "uvicorn.lifespan.on",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.websockets_impl",
]

# Lucid is a headless signal-processing/training backend. Plotting, reports,
# package test suites, notebooks and docs are never used by the desktop app.
excludes = [
    "pytest",
    "numpy.tests",
    "pandas.tests",
    "scipy.tests",
    "sklearn.tests",
    "mne.tests",
    "mne.viz",
    "mne.report",
    "mne.gui",
    "mne.datasets",
    "matplotlib",
    "seaborn",
    "PIL",
    "IPython",
    "jupyter",
    "notebook",
    "sphinx",
]

a = Analysis(
    [str(SPEC_DIR / "desktop_server.py")],
    pathex=[str(SPEC_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
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
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="lucid-backend",
)
