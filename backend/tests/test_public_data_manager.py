from pathlib import Path

import app.public_data as public_data
from app.dataset_jobs import DatasetJobManager


def test_registry_contains_only_supported_public_datasets():
    assert set(public_data.DATASETS) == {"nm000113", "on003626"}
    assert public_data.DATASETS["nm000113"]["derivatives_only"] is False
    assert public_data.DATASETS["on003626"]["derivatives_only"] is True


def test_local_status_requires_three_subjects_and_no_partial(
    tmp_path,
    monkeypatch,
):
    public_root = tmp_path / "public"
    prepared_root = tmp_path / "prepared"
    monkeypatch.setattr(public_data, "PUBLIC_DATA_DIR", public_root)
    monkeypatch.setattr(public_data, "PREPARED_DATA_DIR", prepared_root)

    dataset = public_root / "nm000113"
    for subject in ("sub-01", "sub-02"):
        folder = dataset / subject
        folder.mkdir(parents=True)
        (folder / "recording_eeg.edf").write_bytes(b"real eeg bytes")

    status = public_data.local_dataset_status("nm000113")
    assert status["downloaded"] is False
    assert status["local_subjects"] == ["sub-01", "sub-02"]

    third = dataset / "sub-03"
    third.mkdir(parents=True)
    (third / "recording_eeg.edf").write_bytes(b"real eeg bytes")

    status = public_data.local_dataset_status("nm000113")
    assert status["downloaded"] is True

    (third / "unfinished.edf.part").write_bytes(b"partial")
    status = public_data.local_dataset_status("nm000113")
    assert status["downloaded"] is False
    assert status["partial_files"] == 1


def test_derivative_dataset_counts_derivative_subjects(
    tmp_path,
    monkeypatch,
):
    public_root = tmp_path / "public"
    prepared_root = tmp_path / "prepared"
    monkeypatch.setattr(public_data, "PUBLIC_DATA_DIR", public_root)
    monkeypatch.setattr(public_data, "PREPARED_DATA_DIR", prepared_root)

    for subject in ("sub-01", "sub-02", "sub-03"):
        folder = public_root / "on003626" / "derivatives" / subject
        folder.mkdir(parents=True)
        (folder / "task_eeg-epo.fif").write_bytes(b"real derivative")

    status = public_data.local_dataset_status("on003626")
    assert status["downloaded"] is True
    assert status["local_subjects"] == ["sub-01", "sub-02", "sub-03"]


def test_job_manager_lists_registry_without_network(monkeypatch):
    monkeypatch.setattr(
        "app.dataset_jobs.local_dataset_status",
        lambda dataset_id: {"id": dataset_id},
    )
    manager = DatasetJobManager()
    assert manager.list_datasets() == [
        {"id": "nm000113"},
        {"id": "on003626"},
    ]
