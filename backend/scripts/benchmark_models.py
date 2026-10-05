from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.model import ARCHITECTURES
from app.training import benchmark_models, read_prepared_metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Lucid architectures on the same participant-held-out "
            "public EEG archive. Ranking uses validation balanced accuracy."
        )
    )
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--architectures",
        nargs="*",
        choices=ARCHITECTURES,
        default=list(ARCHITECTURES),
    )
    args = parser.parse_args()

    metadata = read_prepared_metadata(args.prepared)
    output_dir = (
        BACKEND_DIR.parent
        / "models"
        / "benchmarks"
        / metadata["dataset_id"]
        / metadata["task"]
    )
    result = benchmark_models(
        args.prepared,
        output_dir,
        architectures=args.architectures,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        progress=lambda update: print(
            json.dumps(update, sort_keys=True)
        ),
    )

    print(json.dumps(result, indent=2))
    print(f"report={result['report_path']}")


if __name__ == "__main__":
    main()
