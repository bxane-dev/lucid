from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

BACKEND_DIR = Path(
    __file__
).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(BACKEND_DIR),
    )

import numpy as np

from app.model import ARCHITECTURES


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train every Lucid architecture "
            "on the same prepared public EEG "
            "and compare held-out metrics."
        )
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--architectures",
        nargs="*",
        choices=ARCHITECTURES,
        default=list(
            ARCHITECTURES
        ),
    )
    args = parser.parse_args()

    if not args.prepared.exists():
        raise SystemExit(
            f"Prepared archive not found: "
            f"{args.prepared}"
        )

    archive = np.load(
        args.prepared,
        allow_pickle=False,
    )
    dataset_id = str(
        archive["dataset_id"]
    )
    task = (
        str(archive["task"])
        if "task" in archive.files
        else "words"
    )

    output_dir = (
        BACKEND_DIR.parent
        / "models"
        / "benchmarks"
        / dataset_id
        / task
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for architecture in args.architectures:
        output = (
            output_dir
            / f"{architecture}.pt"
        )

        command = [
            sys.executable,
            str(
                Path(__file__)
                .resolve()
                .with_name(
                    "train.py"
                )
            ),
            "--prepared",
            str(args.prepared),
            "--output",
            str(output),
            "--task",
            task,
            "--architecture",
            architecture,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
        ]

        print(
            "\n=== "
            f"{architecture} "
            "===\n"
        )
        subprocess.run(
            command,
            check=True,
        )

        metrics_path = (
            output.with_suffix(
                ".metrics.json"
            )
        )
        metrics = json.loads(
            metrics_path.read_text(
                encoding="utf-8"
            )
        )
        results.append(
            {
                "architecture": (
                    architecture
                ),
                "test_accuracy": (
                    metrics[
                        "test_accuracy"
                    ]
                ),
                "test_balanced_accuracy": (
                    metrics[
                        "test_balanced_accuracy"
                    ]
                ),
                "model_path": str(
                    output
                ),
            }
        )

    results.sort(
        key=lambda item: item[
            "test_balanced_accuracy"
        ],
        reverse=True,
    )

    report = {
        "dataset_id": dataset_id,
        "task": task,
        "seed": args.seed,
        "epochs": args.epochs,
        "ranking_metric": (
            "test_balanced_accuracy"
        ),
        "results": results,
    }

    report_path = (
        output_dir
        / "benchmark.json"
    )
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + json.dumps(
            report,
            indent=2,
        )
    )
    print(
        f"report={report_path}"
    )


if __name__ == "__main__":
    main()
