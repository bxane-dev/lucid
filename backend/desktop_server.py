from __future__ import annotations

import argparse
from pathlib import Path
import sys

import uvicorn
from fastapi.staticfiles import StaticFiles

from app.main import app


def frontend_dir() -> Path:
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS"))
        return base / "frontend"
    return Path(__file__).resolve().parents[1] / "frontend" / "out"


def main() -> None:
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
    main()
