import argparse
import json
from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.dataset import extract_trials_from_recording
from app.inference import Predictor


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Lucid on one annotated public BIDS EEG recording."
        )
    )
    parser.add_argument("edf", type=Path)
    parser.add_argument(
        "--event-index",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    trials = extract_trials_from_recording(args.edf)

    if not trials:
        raise SystemExit(
            "No usable annotated trials found."
        )

    if args.event_index < 0 or args.event_index >= len(trials):
        raise SystemExit(
            f"event-index must be 0..{len(trials) - 1}"
        )

    trial = trials[args.event_index]
    result = Predictor().predict(
        trial.data,
        source_recording=str(args.edf),
    )

    output = {
        **result.model_dump(),
        "ground_truth_annotation": trial.label,
        "ground_truth_note": (
            "Dataset annotation only; this field is not "
            "the neural prediction."
        ),
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
