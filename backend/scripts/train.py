from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import DEFAULT_MODEL_PATH, DEFAULT_PREPARED_PATH
from app.model import ARCHITECTURES
from app.training import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--task", choices=["words", "state"], default=None)
    parser.add_argument(
        "--architecture",
        choices=ARCHITECTURES,
        default="eegnet",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = train_model(
        args.prepared,
        args.output,
        task=args.task,
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        progress=lambda update: print(
            json.dumps(update, sort_keys=True)
        ),
    )
    print(json.dumps(result["metrics"], indent=2))
    print(f"model={result['model_path']}")
    print(f"metrics={result['metrics_path']}")


if __name__ == "__main__":
    main()
