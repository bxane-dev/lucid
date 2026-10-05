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
binaries = []
hiddenimports = []

EXCLUDED_DATA = [
    "**/tests/**",
    "**/test/**",
    "**/testing/**",
    "**/benchmarks/**",
    "**/examples/**",
]

def runtime_module(name: str) -> bool:
    lowered = name.lower()
    blocked = (
        ".tests",
        ".test.",
        ".testing",
        ".benchmarks",
        ".examples",
        ".conftest",
    )
    return not any(part in lowered for part in blocked)

for package in (
    "mne",
    "sklearn",
    "scipy",
    "pandas",
    "fastapi",
    "uvicorn",
    "pydantic",
):
    datas += collect_data_files(
        package,
        excludes=EXCLUDED_DATA,
    )
    hiddenimports += collect_submodules(
        package,
        filter=runtime_module,
    )

binaries += collect_dynamic_libs("torch")

a = Analysis(
    [str(SPEC_DIR / "desktop_server.py")],
    pathex=[str(SPEC_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest",
    ],
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
