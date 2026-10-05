from dataclasses import dataclass
from pathlib import Path
import re

import mne
import numpy as np
import pandas as pd

from .preprocess import (
    DEFAULT_PREPROCESS,
    is_acceptable_trial,
    normalize_trial,
    preprocess_raw,
)
from .provenance import build_source_manifest, write_manifest


@dataclass
class TrialRecord:
    data: np.ndarray
    label: str
    subject: str
    recording: str
    channel_names: tuple[str, ...]


def _subject_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("sub-"):
            return part

    match = re.search(r"(sub-[A-Za-z0-9]+)", path.name)
    return match.group(1) if match else "unknown"


def _events_path(edf_path: Path) -> Path:
    if edf_path.name.endswith("_eeg.edf"):
        return edf_path.with_name(
            edf_path.name.replace("_eeg.edf", "_events.tsv")
        )
    return edf_path.with_suffix(".tsv")


def _event_label(row: pd.Series) -> str | None:
    for key in ("trial_type", "value"):
        if key in row and pd.notna(row[key]):
            value = str(row[key]).strip()
            if value and value.lower() not in {"n/a", "nan", "none"}:
                return value.lower().replace(" ", "_")
    return None


def extract_trials_from_recording(
    edf_path: Path,
    *,
    window_seconds: float = 2.0,
) -> list[TrialRecord]:
    events_path = _events_path(edf_path)

    if not events_path.exists():
        raise FileNotFoundError(
            f"Missing BIDS events file for {edf_path}: {events_path}"
        )

    raw = preprocess_raw(
        mne.io.read_raw_edf(
            edf_path,
            preload=False,
            verbose="ERROR",
        )
    )
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

        out.append(
            TrialRecord(
                normalize_trial(trial),
                label,
                subject,
                str(edf_path),
                tuple(raw.ch_names),
            )
        )

    return out


def discover_edf(dataset_root: Path) -> list[Path]:
    return sorted(dataset_root.rglob("*_eeg.edf"))


def subject_group_split(
    groups: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    """
    Deterministically split whole participants into train/validation/test.

    For three subjects this produces exactly one subject in each split.
    Larger datasets keep approximately 70/15/15 while guaranteeing every
    split contains at least one held-out participant.
    """
    groups = np.asarray(groups)
    unique_subjects = np.array(sorted(set(groups.tolist())), dtype="U64")
    subject_count = len(unique_subjects)

    if subject_count < 3:
        raise ValueError(
            "Subject-held-out splitting needs at least 3 participants."
        )

    rng = np.random.default_rng(random_state)
    rng.shuffle(unique_subjects)

    holdout_count = max(1, int(round(subject_count * 0.15)))
    if 2 * holdout_count >= subject_count:
        holdout_count = 1

    val_subjects = set(unique_subjects[:holdout_count].tolist())
    test_subjects = set(
        unique_subjects[holdout_count : 2 * holdout_count].tolist()
    )
    train_subjects = set(
        unique_subjects[2 * holdout_count :].tolist()
    )

    if not train_subjects or not val_subjects or not test_subjects:
        raise RuntimeError(
            "Failed to create non-empty train/validation/test subject splits."
        )

    split = np.full(len(groups), -1, dtype=np.int8)

    for index, subject in enumerate(groups.tolist()):
        if subject in train_subjects:
            split[index] = 0
        elif subject in val_subjects:
            split[index] = 1
        elif subject in test_subjects:
            split[index] = 2

    if np.any(split < 0):
        raise RuntimeError("At least one trial was not assigned to a split.")

    return split


def prepare_dataset(
    dataset_root: Path,
    output_path: Path,
    dataset_id: str,
) -> dict:
    edf_files = discover_edf(dataset_root)

    if not edf_files:
        raise FileNotFoundError(
            f"No *_eeg.edf files found under {dataset_root}"
        )

    records: list[TrialRecord] = []
    provenance_sources: list[Path] = []
    for edf in edf_files:
        events = _events_path(edf)
        extracted = extract_trials_from_recording(edf)
        if extracted:
            provenance_sources.extend([edf, events])
            records.extend(extracted)

    if not records:
        raise RuntimeError("No valid annotated EEG trials were extracted.")

    channel_counts = {record.data.shape[0] for record in records}
    sample_counts = {record.data.shape[1] for record in records}
    channel_orders = {record.channel_names for record in records}

    if (
        len(channel_counts) != 1
        or len(sample_counts) != 1
        or len(channel_orders) != 1
    ):
        raise RuntimeError(
            f"Inconsistent trial shapes: "
            f"channels={channel_counts}, samples={sample_counts}, "
            f"channel_orders={len(channel_orders)}"
        )

    channel_names = list(next(iter(channel_orders)))
    labels = sorted({record.label for record in records})
    label_to_idx = {
        label: index
        for index, label in enumerate(labels)
    }

    x = np.stack([record.data for record in records]).astype(np.float32)
    y = np.array(
        [label_to_idx[record.label] for record in records],
        dtype=np.int64,
    )
    groups = np.array(
        [record.subject for record in records],
        dtype="U32",
    )
    recordings = np.array(
        [record.recording for record in records],
        dtype="U512",
    )
    split = subject_group_split(groups)
    manifest = build_source_manifest(
        provenance_sources,
        dataset_id=dataset_id,
        dataset_root=dataset_root,
    )
    provenance_sha256 = manifest["manifest_sha256"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        x=x,
        y=y,
        groups=groups,
        recordings=recordings,
        split=split,
        labels=np.array(labels, dtype="U64"),
        dataset_id=np.array(dataset_id),
        provenance_sha256=np.array(provenance_sha256, dtype="U64"),
        sfreq=np.array(
            DEFAULT_PREPROCESS.target_sfreq,
            dtype=np.float32,
        ),
        channel_names=np.array(channel_names, dtype="U64"),
    )

    manifest_path = write_manifest(manifest, output_path)

    return {
        "trials": len(records),
        "channels": int(x.shape[1]),
        "channel_names": channel_names,
        "samples": int(x.shape[2]),
        "labels": labels,
        "subjects": sorted(set(groups.tolist())),
        "train_subjects": sorted(set(groups[split == 0].tolist())),
        "val_subjects": sorted(set(groups[split == 1].tolist())),
        "test_subjects": sorted(set(groups[split == 2].tolist())),
        "train_trials": int((split == 0).sum()),
        "val_trials": int((split == 1).sum()),
        "test_trials": int((split == 2).sum()),
        "output": str(output_path),
        "provenance_sha256": provenance_sha256,
        "manifest": str(manifest_path),
    }
