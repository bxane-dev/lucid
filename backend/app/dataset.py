from dataclasses import dataclass
from pathlib import Path
import re

import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .preprocess import DEFAULT_PREPROCESS, is_acceptable_trial, normalize_trial, preprocess_raw


@dataclass
class TrialRecord:
    data: np.ndarray
    label: str
    subject: str
    recording: str


def _subject_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("sub-"):
            return part
    match = re.search(r"(sub-[A-Za-z0-9]+)", path.name)
    return match.group(1) if match else "unknown"


def _events_path(edf_path: Path) -> Path:
    if edf_path.name.endswith("_eeg.edf"):
        return edf_path.with_name(edf_path.name.replace("_eeg.edf", "_events.tsv"))
    return edf_path.with_suffix(".tsv")


def _event_label(row: pd.Series) -> str | None:
    for key in ("trial_type", "value"):
        if key in row and pd.notna(row[key]):
            value = str(row[key]).strip()
            if value and value.lower() not in {"n/a", "nan", "none"}:
                return value.lower().replace(" ", "_")
    return None


def extract_trials_from_recording(edf_path: Path, *, window_seconds: float = 2.0) -> list[TrialRecord]:
    events_path = _events_path(edf_path)
    if not events_path.exists():
        raise FileNotFoundError(f"Missing BIDS events file for {edf_path}: {events_path}")

    raw = preprocess_raw(mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR"))
    sfreq = float(raw.info["sfreq"])
    samples = int(round(window_seconds * sfreq))
    events = pd.read_csv(events_path, sep="\t")
    subject = _subject_from_path(edf_path)
    out: list[TrialRecord] = []

    for _, row in events.iterrows():
        label = _event_label(row)
        if label is None or "onset" not in row or pd.isna(row["onset"]):
            continue
        start = int(round(float(row["onset"]) * sfreq))
        stop = start + samples
        if start < 0 or stop > raw.n_times:
            continue
        trial = raw.get_data(start=start, stop=stop)
        if trial.shape[-1] != samples or not is_acceptable_trial(trial):
            continue
        out.append(TrialRecord(normalize_trial(trial), label, subject, str(edf_path)))
    return out


def discover_edf(dataset_root: Path) -> list[Path]:
    return sorted(dataset_root.rglob("*_eeg.edf"))


def subject_group_split(groups: np.ndarray, random_state: int = 42) -> np.ndarray:
    groups = np.asarray(groups)
    if np.unique(groups).size < 3:
        raise ValueError("Subject-held-out splitting needs at least 3 participants.")
    indices = np.arange(len(groups))
    first = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=random_state)
    train_idx, remainder_idx = next(first.split(indices, groups=groups))
    remainder_groups = groups[remainder_idx]
    second = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=random_state + 1)
    val_rel, test_rel = next(second.split(np.arange(len(remainder_idx)), groups=remainder_groups))
    split = np.full(len(groups), 2, dtype=np.int8)
    split[train_idx] = 0
    split[remainder_idx[val_rel]] = 1
    split[remainder_idx[test_rel]] = 2
    return split


def prepare_dataset(dataset_root: Path, output_path: Path, dataset_id: str) -> dict:
    edf_files = discover_edf(dataset_root)
    if not edf_files:
        raise FileNotFoundError(f"No *_eeg.edf files found under {dataset_root}")
    records: list[TrialRecord] = []
    for edf in edf_files:
        records.extend(extract_trials_from_recording(edf))
    if not records:
        raise RuntimeError("No valid annotated EEG trials were extracted.")

    channel_counts = {r.data.shape[0] for r in records}
    sample_counts = {r.data.shape[1] for r in records}
    if len(channel_counts) != 1 or len(sample_counts) != 1:
        raise RuntimeError(f"Inconsistent trial shapes: channels={channel_counts}, samples={sample_counts}")

    labels = sorted({r.label for r in records})
    label_to_idx = {label: i for i, label in enumerate(labels)}
    x = np.stack([r.data for r in records]).astype(np.float32)
    y = np.array([label_to_idx[r.label] for r in records], dtype=np.int64)
    groups = np.array([r.subject for r in records], dtype="U32")
    recordings = np.array([r.recording for r in records], dtype="U512")
    split = subject_group_split(groups)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, x=x, y=y, groups=groups, recordings=recordings, split=split,
        labels=np.array(labels, dtype="U64"), dataset_id=np.array(dataset_id),
        sfreq=np.array(DEFAULT_PREPROCESS.target_sfreq, dtype=np.float32),
    )
    return {
        "trials": len(records), "channels": int(x.shape[1]), "samples": int(x.shape[2]),
        "labels": labels, "subjects": sorted(set(groups.tolist())),
        "train_trials": int((split == 0).sum()), "val_trials": int((split == 1).sum()),
        "test_trials": int((split == 2).sum()), "output": str(output_path),
    }
