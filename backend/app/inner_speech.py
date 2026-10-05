from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .dataset import subject_group_split
from .preprocess import DEFAULT_PREPROCESS, normalize_trial
from .provenance import build_source_manifest, write_manifest


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
    readers = (
        lambda: pd.read_pickle(path),
        lambda: np.load(path, allow_pickle=True),
        lambda: pd.read_csv(path, index_col=0),
    )

    last_error: Exception | None = None
    for reader in readers:
        try:
            events = reader()
            if hasattr(events, "to_numpy"):
                events = events.to_numpy()
            array = np.asarray(events)
            if array.ndim == 2:
                return array
        except Exception as exc:
            last_error = exc

    raise ValueError(
        f"Could not read event table {path}: {last_error}"
    )


def _load_task_epochs(
    path: Path,
    window_seconds: float = 2.0,
) -> tuple[np.ndarray, list[str]]:
    epochs = mne.read_epochs(path, preload=True, verbose="ERROR")
    epochs.pick("eeg")

    sfreq = float(epochs.info["sfreq"])
    if abs(sfreq - DEFAULT_PREPROCESS.target_sfreq) > 1e-6:
        epochs.resample(DEFAULT_PREPROCESS.target_sfreq, verbose="ERROR")

    samples = int(round(window_seconds * DEFAULT_PREPROCESS.target_sfreq))
    start = int(np.searchsorted(epochs.times, 0.0, side="left"))
    stop = start + samples

    data = epochs.get_data(copy=True).astype(np.float32)
    if stop > data.shape[-1]:
        raise ValueError(
            f"{path} does not contain {window_seconds}s after task onset."
        )

    return normalize_trial(data[..., start:stop]), list(epochs.ch_names)


def _load_baseline_windows(
    path: Path,
    window_seconds: float = 2.0,
) -> tuple[np.ndarray, list[str]]:
    """
    Cut non-overlapping windows from the actual recorded baseline epoch.

    The Nieto preprocessing publishes one long baseline epoch per session.
    Lucid never duplicates that epoch to imitate trial-level REST examples.
    """
    epochs = mne.read_epochs(path, preload=True, verbose="ERROR")
    epochs.pick("eeg")

    sfreq = float(epochs.info["sfreq"])
    if abs(sfreq - DEFAULT_PREPROCESS.target_sfreq) > 1e-6:
        epochs.resample(DEFAULT_PREPROCESS.target_sfreq, verbose="ERROR")

    data = epochs.get_data(copy=True).astype(np.float32)
    samples = int(round(window_seconds * DEFAULT_PREPROCESS.target_sfreq))

    windows = []
    for epoch in data:
        window_count = epoch.shape[-1] // samples
        for index in range(window_count):
            start = index * samples
            stop = start + samples
            windows.append(epoch[:, start:stop])

    if not windows:
        raise ValueError(
            f"{path} contains no complete {window_seconds}s baseline windows."
        )

    return (
        normalize_trial(np.stack(windows).astype(np.float32)),
        list(epochs.ch_names),
    )


def _align_common_channels(
    task_data: np.ndarray,
    task_names: list[str],
    baseline_data: np.ndarray,
    baseline_names: list[str],
    expected_order: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if len(set(task_names)) != len(task_names):
        raise ValueError("Task EEG contains duplicate channel names.")
    if len(set(baseline_names)) != len(baseline_names):
        raise ValueError("Baseline EEG contains duplicate channel names.")

    baseline_set = set(baseline_names)
    common = [name for name in task_names if name in baseline_set]
    if not common:
        raise ValueError("Task EEG and baseline have no common named EEG channels.")

    if expected_order is not None:
        if set(common) != set(expected_order):
            missing = sorted(set(expected_order) - set(common))
            added = sorted(set(common) - set(expected_order))
            raise ValueError(
                "Common EEG channel set changed across sessions: "
                f"missing={missing} added={added}"
            )
        common = list(expected_order)

    task_index = {name: index for index, name in enumerate(task_names)}
    baseline_index = {
        name: index for index, name in enumerate(baseline_names)
    }
    task_aligned = task_data[
        :,
        [task_index[name] for name in common],
        :,
    ]
    baseline_aligned = baseline_data[
        :,
        [baseline_index[name] for name in common],
        :,
    ]
    return task_aligned, baseline_aligned, common


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

    word_x: list[np.ndarray] = []
    word_y: list[str] = []
    word_groups: list[str] = []
    word_recordings: list[str] = []

    state_x: list[np.ndarray] = []
    state_y: list[str] = []
    state_groups: list[str] = []
    state_recordings: list[str] = []

    expected_shape: tuple[int, int] | None = None
    expected_channel_names: list[str] | None = None
    provenance_sources: list[Path] = []

    for eeg_path in eeg_files:
        prefix = eeg_path.name.replace("_eeg-epo.fif", "")
        session_dir = eeg_path.parent
        baseline_path = session_dir / f"{prefix}_baseline-epo.fif"
        events_path = session_dir / f"{prefix}_events.dat"

        if not baseline_path.exists() or not events_path.exists():
            raise FileNotFoundError(
                f"Missing companion derivative for {eeg_path}: "
                f"baseline={baseline_path.exists()} "
                f"events={events_path.exists()}"
            )

        eeg, eeg_channels = _load_task_epochs(eeg_path)
        baseline_windows, baseline_channels = _load_baseline_windows(
            baseline_path
        )
        eeg, baseline_windows, common_channels = _align_common_channels(
            eeg,
            eeg_channels,
            baseline_windows,
            baseline_channels,
            expected_channel_names,
        )
        if expected_channel_names is None:
            expected_channel_names = list(common_channels)

        events = _load_events(events_path)

        if events.ndim != 2 or events.shape[1] < 2:
            raise ValueError(
                f"Unexpected event table shape for {events_path}: {events.shape}"
            )

        if len(events) != len(eeg):
            raise ValueError(
                f"Speech epoch/event count mismatch in {session_dir}: "
                f"events={len(events)} eeg={len(eeg)}"
            )

        if expected_shape is None:
            expected_shape = (int(eeg.shape[1]), int(eeg.shape[2]))
        elif tuple(eeg.shape[1:]) != expected_shape:
            raise ValueError(
                f"Channel/sample shape changed across recordings: "
                f"expected {expected_shape}, got {tuple(eeg.shape[1:])} "
                f"in {eeg_path}"
            )

        condition = events[:, 1].astype(int)
        direction = events[:, 0].astype(int)
        mask = condition == INNER_SPEECH_CONDITION

        if not mask.any():
            continue

        provenance_sources.extend(
            [eeg_path, baseline_path, events_path]
        )
        subject = _subject_from_derivative(eeg_path)
        inner_eeg = eeg[mask]
        inner_direction = direction[mask]

        for trial, direction_code in zip(inner_eeg, inner_direction):
            code = int(direction_code)
            if code not in DIRECTION_LABELS:
                continue

            word_x.append(trial)
            word_y.append(DIRECTION_LABELS[code])
            word_groups.append(subject)
            word_recordings.append(str(eeg_path))

        # Every REST example below is a distinct, non-overlapping segment
        # of the session's published resting baseline recording.
        for window in baseline_windows:
            state_x.append(window)
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

    manifest = build_source_manifest(
        provenance_sources,
        dataset_id="on003626",
        dataset_root=dataset_root,
    )
    provenance_sha256 = manifest["manifest_sha256"]

    def save_archive(
        x_list,
        label_list,
        groups_list,
        recordings_list,
        output,
        task,
    ):
        labels = sorted(set(label_list))
        label_to_idx = {label: i for i, label in enumerate(labels)}

        x = np.stack(x_list).astype(np.float32)
        y = np.array(
            [label_to_idx[label] for label in label_list],
            dtype=np.int64,
        )
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
            provenance_sha256=np.array(
                provenance_sha256,
                dtype="U64",
            ),
            sfreq=np.array(
                DEFAULT_PREPROCESS.target_sfreq,
                dtype=np.float32,
            ),
            channel_names=np.array(
                expected_channel_names or [],
                dtype="U64",
            ),
        )

        manifest_path = write_manifest(
            manifest,
            output,
        )
        counts = {
            label: int((y == index).sum())
            for index, label in enumerate(labels)
        }

        return {
            "task": task,
            "trials": int(len(x)),
            "class_counts": counts,
            "labels": labels,
            "channels": int(x.shape[1]),
            "channel_names": list(expected_channel_names or []),
            "samples": int(x.shape[2]),
            "subjects": sorted(set(groups.tolist())),
            "train_trials": int((split == 0).sum()),
            "val_trials": int((split == 1).sum()),
            "test_trials": int((split == 2).sum()),
            "output": str(output),
            "provenance_sha256": provenance_sha256,
            "manifest": str(manifest_path),
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
