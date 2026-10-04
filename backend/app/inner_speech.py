from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .dataset import subject_group_split
from .preprocess import DEFAULT_PREPROCESS, normalize_trial


DIRECTION_LABELS = {
    0: "up",
    1: "down",
    2: "right",
    3: "left",
}
INNER_SPEECH_CONDITION = 1


def _subject_from_derivative(path: Path) -> str:
    for part in path.parts:
        if part.startswith("sub-"):
            return part
    raise ValueError(f"Could not determine subject from {path}")


def _load_events(path: Path) -> np.ndarray:
    try:
        events = pd.read_pickle(path)
        if hasattr(events, "to_numpy"):
            events = events.to_numpy()
        return np.asarray(events)
    except Exception:
        events = np.load(path, allow_pickle=True)
        if hasattr(events, "to_numpy"):
            events = events.to_numpy()
        return np.asarray(events)


def _load_epoch_data(path: Path, window_seconds: float = 2.0) -> np.ndarray:
    epochs = mne.read_epochs(path, preload=True, verbose="ERROR")
    epochs.pick("eeg")

    sfreq = float(epochs.info["sfreq"])
    if abs(sfreq - DEFAULT_PREPROCESS.target_sfreq) > 1e-6:
        epochs.resample(DEFAULT_PREPROCESS.target_sfreq, verbose="ERROR")

    data = epochs.get_data(copy=True).astype(np.float32)
    samples = int(round(window_seconds * DEFAULT_PREPROCESS.target_sfreq))

    if data.shape[-1] < samples:
        raise ValueError(
            f"{path} has only {data.shape[-1]} samples after resampling; "
            f"{samples} are required for a {window_seconds}s window."
        )

    data = data[..., :samples]
    return normalize_trial(data)


def prepare_nieto_derivatives(
    dataset_root: Path,
    word_output: Path,
    state_output: Path,
) -> dict:
    derivatives = dataset_root / "derivatives"
    eeg_files = sorted(derivatives.rglob("*_eeg-epo.fif"))

    if not eeg_files:
        raise FileNotFoundError(
            f"No Nieto derivative EEG epoch files found under {derivatives}. "
            "Download on003626 derivatives first."
        )

    word_x = []
    word_y = []
    word_groups = []
    word_recordings = []

    state_x = []
    state_y = []
    state_groups = []
    state_recordings = []

    expected_shape = None

    for eeg_path in eeg_files:
        prefix = eeg_path.name.replace("_eeg-epo.fif", "")
        session_dir = eeg_path.parent
        baseline_path = session_dir / f"{prefix}_baseline-epo.fif"
        events_path = session_dir / f"{prefix}_events.dat"

        if not baseline_path.exists() or not events_path.exists():
            raise FileNotFoundError(
                f"Missing companion derivative for {eeg_path}: "
                f"baseline={baseline_path.exists()} events={events_path.exists()}"
            )

        eeg = _load_epoch_data(eeg_path)
        baseline = _load_epoch_data(baseline_path)
        events = _load_events(events_path)

        if events.ndim != 2 or events.shape[1] < 2:
            raise ValueError(
                f"Unexpected event table shape for {events_path}: {events.shape}"
            )

        if len(events) != len(eeg) or len(events) != len(baseline):
            raise ValueError(
                f"Epoch/event count mismatch in {session_dir}: "
                f"events={len(events)} eeg={len(eeg)} baseline={len(baseline)}"
            )

        if eeg.shape[1:] != baseline.shape[1:]:
            raise ValueError(
                f"EEG/baseline shape mismatch in {session_dir}: "
                f"{eeg.shape[1:]} vs {baseline.shape[1:]}"
            )

        if expected_shape is None:
            expected_shape = eeg.shape[1:]
        elif eeg.shape[1:] != expected_shape:
            raise ValueError(
                f"Channel/sample shape changed across recordings: "
                f"expected {expected_shape}, got {eeg.shape[1:]} in {eeg_path}"
            )

        condition = events[:, 1].astype(int)
        direction = events[:, 0].astype(int)
        mask = condition == INNER_SPEECH_CONDITION

        if not mask.any():
            continue

        subject = _subject_from_derivative(eeg_path)
        inner_eeg = eeg[mask]
        inner_baseline = baseline[mask]
        inner_direction = direction[mask]

        for trial, direction_code in zip(inner_eeg, inner_direction):
            if int(direction_code) not in DIRECTION_LABELS:
                continue
            word_x.append(trial)
            word_y.append(DIRECTION_LABELS[int(direction_code)])
            word_groups.append(subject)
            word_recordings.append(str(eeg_path))

        for trial in inner_baseline:
            state_x.append(trial)
            state_y.append("rest")
            state_groups.append(subject)
            state_recordings.append(str(baseline_path))

        for trial in inner_eeg:
            state_x.append(trial)
            state_y.append("imagined_speech")
            state_groups.append(subject)
            state_recordings.append(str(eeg_path))

    if not word_x or not state_x:
        raise RuntimeError("No inner-speech derivative trials were prepared.")

    def save_archive(x_list, label_list, groups_list, recordings_list, output, task):
        labels = sorted(set(label_list))
        label_to_idx = {label: i for i, label in enumerate(labels)}
        x = np.stack(x_list).astype(np.float32)
        y = np.array([label_to_idx[label] for label in label_list], dtype=np.int64)
        groups = np.array(groups_list, dtype="U32")
        recordings = np.array(recordings_list, dtype="U512")
        split = subject_group_split(groups)

        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            x=x,
            y=y,
            groups=groups,
            recordings=recordings,
            split=split,
            labels=np.array(labels, dtype="U64"),
            dataset_id=np.array("on003626"),
            task=np.array(task),
            sfreq=np.array(DEFAULT_PREPROCESS.target_sfreq, dtype=np.float32),
        )

        return {
            "task": task,
            "trials": int(len(x)),
            "labels": labels,
            "channels": int(x.shape[1]),
            "samples": int(x.shape[2]),
            "subjects": sorted(set(groups.tolist())),
            "train_trials": int((split == 0).sum()),
            "val_trials": int((split == 1).sum()),
            "test_trials": int((split == 2).sum()),
            "output": str(output),
        }

    return {
        "words": save_archive(
            word_x,
            word_y,
            word_groups,
            word_recordings,
            word_output,
            "words",
        ),
        "state": save_archive(
            state_x,
            state_y,
            state_groups,
            state_recordings,
            state_output,
            "state",
        ),
    }
