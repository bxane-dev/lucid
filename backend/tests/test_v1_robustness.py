import json
from pathlib import Path

import numpy as np
import pytest

import app.model_registry as registry_module
from app.dataset import subject_group_split
from app.model_registry import ModelRegistry
from app.replay import PublicEEGReplay


def test_subject_split_is_deterministic_and_participant_held_out():
    groups = np.array(
        [
            "sub-01", "sub-01",
            "sub-02", "sub-02",
            "sub-03", "sub-03",
            "sub-04", "sub-04",
            "sub-05", "sub-05",
        ],
        dtype="U16",
    )
    first = subject_group_split(groups, random_state=42)
    second = subject_group_split(groups, random_state=42)
    assert np.array_equal(first, second)

    split_subjects = {
        split: set(groups[first == split].tolist())
        for split in (0, 1, 2)
    }
    assert split_subjects[0]
    assert split_subjects[1]
    assert split_subjects[2]
    assert split_subjects[0].isdisjoint(split_subjects[1])
    assert split_subjects[0].isdisjoint(split_subjects[2])
    assert split_subjects[1].isdisjoint(split_subjects[2])


def test_replay_fails_closed_for_corrupted_archive(tmp_path):
    archive = tmp_path / "corrupt.npz"
    archive.write_bytes(b"not an npz file")
    replay = PublicEEGReplay(archive)
    assert replay.ready is False
    with pytest.raises(Exception):
        replay.next_test_trial()


def test_replay_requires_source_provenance(tmp_path):
    archive = tmp_path / "legacy.npz"
    np.savez_compressed(
        archive,
        x=np.zeros((3, 2, 8), dtype=np.float32),
        y=np.zeros(3, dtype=np.int64),
        split=np.array([0, 1, 2], dtype=np.int8),
        labels=np.array(["yes"], dtype="U8"),
        recordings=np.array(["a", "b", "c"], dtype="U8"),
        dataset_id=np.array("nm000113"),
        sfreq=np.array(128.0, dtype=np.float32),
    )
    replay = PublicEEGReplay(archive)
    assert replay.ready is False


def test_corrupt_settings_fall_back_to_safe_defaults(tmp_path, monkeypatch):
    models = tmp_path / "models"
    prepared = tmp_path / "prepared"
    runtime = tmp_path / "runtime"
    settings = runtime / "lucid-settings.json"
    runtime.mkdir(parents=True)
    settings.write_text("{ definitely-not-json", encoding="utf-8")

    monkeypatch.setattr(registry_module, "MODELS_DIR", models)
    monkeypatch.setattr(registry_module, "PREPARED_DATA_DIR", prepared)
    monkeypatch.setattr(registry_module, "RUNTIME_DIR", runtime)
    monkeypatch.setattr(registry_module, "SETTINGS_PATH", settings)

    registry = ModelRegistry()
    assert registry.active_dataset == "nm000113"
    assert registry.list_models() == []


def test_registry_rejects_path_traversal_dataset_id(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(registry_module, "RUNTIME_DIR", runtime)
    monkeypatch.setattr(
        registry_module,
        "SETTINGS_PATH",
        runtime / "lucid-settings.json",
    )
    registry = ModelRegistry()
    with pytest.raises(ValueError):
        registry.set_active_dataset("../escape")
