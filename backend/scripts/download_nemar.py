from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.public_data import DATASETS, download_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download real public EEG directly from Lucid's verified "
            "NEMAR registry. No generated EEG is created."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="nm000113",
    )
    parser.add_argument(
        "--subjects",
        default="all",
        help=(
            "Comma-separated BIDS participants such as "
            "sub-01,sub-02,sub-03 or 'all'."
        ),
    )
    parser.add_argument(
        "--derivatives-only",
        action="store_true",
        help=(
            "Force derivative-only download. Lucid already defaults "
            "on003626 to published derivatives."
        ),
    )
    args = parser.parse_args()

    if args.subjects.strip().lower() == "all":
        subjects = None
    else:
        subjects = [
            item.strip()
            for item in args.subjects.split(",")
            if item.strip()
        ]

    meta = DATASETS[args.dataset]
    derivatives_only = (
        True
        if args.derivatives_only
        else bool(meta["derivatives_only"])
    )

    def progress(update: dict) -> None:
        phase = update.get("phase", "working")
        done = update.get("files_done")
        total = update.get("files_total")
        message = update.get("message")
        parts = [phase]
        if done is not None and total is not None:
            parts.append(f"{done}/{total} files")
        if message:
            parts.append(str(message))
        print(" | ".join(parts))

    result = download_dataset(
        args.dataset,
        subjects=subjects,
        derivatives_only=derivatives_only,
        progress=progress,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
