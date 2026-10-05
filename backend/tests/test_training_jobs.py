from pathlib import Path

import numpy as np

import app.training_jobs as jobs_module
from app.training_jobs import TrainingJobManager


def _prepared(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.zeros((6, 2, 16), dtype=np.float32)
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    split = np.array([0, 0, 1, 1, 2, 2], dtype=np.int8)
    np.savez_compressed(
        path,
        x=x,
        y=y,
        split=split,
        labels=np.array(["a", "b"], dtype="U8"),
        groups=np.array(
            ["sub-01", "sub-01", "sub-02", "sub-02", "sub-03", "sub-03"],
            dtype="U16",
        ),
        dataset_id=np.array("nm000113"),
        task=np.array("words"),
        provenance_sha256=np.array("a" * 64, dtype="U64"),
        sfreq=np.array(128.0, dtype=np.float32),
    )


def test_training_job_requires_prepared_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(
        jobs_module,
        "prepared_path",
        lambda dataset_id, task: tmp_path / f"{dataset_id}_{task}.npz",
    )
    manager = TrainingJobManager()
    try:
        manager.start(
            dataset_id="nm000113",
            task="words",
            kind="train",
            architecture="eegnet",
            architectures=None,
            epochs=1,
            batch_size=2,
            seed=42,
            auto_activate=False,
        )
        assert False, "missing prepared archive should fail"
    except FileNotFoundError:
        pass


def test_training_job_rejects_unknown_architecture(tmp_path, monkeypatch):
    archive = tmp_path / "nm000113_words.npz"
    _prepared(archive)
    monkeypatch.setattr(
        jobs_module,
        "prepared_path",
        lambda dataset_id, task: archive,
    )
    manager = TrainingJobManager()
    try:
        manager.start(
            dataset_id="nm000113",
            task="words",
            kind="train",
            architecture="not-real",
            architectures=None,
            epochs=1,
            batch_size=2,
            seed=42,
            auto_activate=False,
        )
        assert False, "unknown architecture should fail"
    except ValueError as exc:
        assert "Unsupported" in str(exc)
