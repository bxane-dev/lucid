import argparse
import json

from app.config import PREPARED_DATA_DIR, PUBLIC_DATA_DIR
from app.dataset import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="nm000113")
    args = parser.parse_args()

    root = PUBLIC_DATA_DIR / args.dataset
    output = PREPARED_DATA_DIR / f"{args.dataset}_words.npz"
    summary = prepare_dataset(root, output, args.dataset)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
