import argparse
import json
from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import PREPARED_DATA_DIR, PUBLIC_DATA_DIR
from app.inner_speech import prepare_nieto_derivatives


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare real Nieto/OpenNeuro inner-speech and baseline derivative "
            "epochs for Lucid."
        )
    )
    parser.add_argument(
        "--dataset",
        default="on003626",
        choices=["on003626"],
    )
    args = parser.parse_args()

    dataset_root = PUBLIC_DATA_DIR / args.dataset
    summary = prepare_nieto_derivatives(
        dataset_root=dataset_root,
        word_output=PREPARED_DATA_DIR / "on003626_words.npz",
        state_output=PREPARED_DATA_DIR / "on003626_state.npz",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
