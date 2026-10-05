from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback

# PyInstaller windowed executables on Windows may provide no console streams.
# Uvicorn/logging still expects file-like stdout/stderr, so route them to the
# null device instead of allowing startup to fail.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

import uvicorn
from fastapi.staticfiles import StaticFiles

def write_diagnostic() -> None:
    path = os.getenv("LUCID_DIAGNOSTIC_LOG")
    if not path:
        return
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(traceback.format_exc(), encoding="utf-8")
    except Exception:
        pass


def frontend_dir() -> Path:
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS"))
        return base / "frontend"
    return Path(__file__).resolve().parents[1] / "frontend" / "out"


def main() -> None:
    from app.main import app

    parser = argparse.ArgumentParser(description="Lucid desktop server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    web = frontend_dir()
    if not web.exists():
        raise SystemExit(
            f"Lucid frontend bundle was not found at {web}. "
            "Build the frontend before packaging."
        )

    app.mount(
        "/",
        StaticFiles(directory=web, html=True),
        name="desktop-ui",
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        write_diagnostic()
        raise
