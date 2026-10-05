import json
from pathlib import Path

import numpy as np
import torch

import app.model_registry as registry_module
from app.inference import Predictor
from app.model import build_model
from app.model_registry import ModelRegistry
from app.provenance import sha256_file


def _write_prepared(path: Path, provenance: str, task: str = "words"):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=np.zeros((3, 2, 8), dtype=np.float32),
        y=np.array([0, 0, 0], dtype=np.int64),
        split=np.array([0, 1, 2], dtype=np.int8),
        labels=np.array(["yes"], dtype="U16"),
        groups=np.array(["sub-01", "sub-02", "sub-03"], dtype="U16"),
        dataset_id=np.array("nm000113"),
        task=np.array(task),
        provenance_sha256=np.array(provenance, dtype="U64"),
        sfreq=np.array(128.0, dtype=np.float32),
    )


def _write_model(path: Path, provenance: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    model = build_model("eegnet", channels=2, classes=1)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": ["yes"],
            "channels": 2,
            "samples": 8,
            "dataset_id": "nm000113",
            "provenance_sha256": provenance,
            "task": "words",
            "architecture": "eegnet",
            "model_version": "test-model",
        },
        path,
    )
    metrics = {
        "dataset_id": "nm000113",
        "task": "words",
        "architecture": "eegnet",
        "provenance_sha256": provenance,
        "best_validation_balanced_accuracy": 0.5,
        "test_accuracy": 0.5,
        "test_balanced_accuracy": 0.5,
        "best_epoch": 1,
        "labels": ["yes"],
        "checkpoint_sha256": sha256_file(path),
    }
    path.with_suffix(".metrics.json").write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )


def test_registry_activates_only_matching_provenance(tmp_path, monkeypatch):
    models = tmp_path / "models"
    prepared = tmp_path / "prepared"
    runtime = tmp_path / "runtime"
    settings = runtime / "lucid-settings.json"

    monkeypatch.setattr(registry_module, "MODELS_DIR", models)
    monkeypatch.setattr(registry_module, "PREPARED_DATA_DIR", prepared)
    monkeypatch.setattr(registry_module, "RUNTIME_DIR", runtime)
    monkeypatch.setattr(registry_module, "SETTINGS_PATH", settings)

    provenance = "a" * 64
    _write_prepared(prepared / "nm000113_words.npz", provenance)
    model_path = models / "nm000113" / "words" / "eegnet.pt"
    _write_model(model_path, provenance)

    registry = ModelRegistry()
    listed = registry.list_models()
    assert len(listed) == 1
    activated = registry.activate(listed[0]["id"])
    assert activated["active"] is True

    with np.load(
        prepared / "nm000113_words.npz",
        allow_pickle=False,
    ) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["provenance_sha256"] = np.array("b" * 64, dtype="U64")
    np.savez_compressed(prepared / "nm000113_words.npz", **payload)

    registry = ModelRegistry()
    with pytest.raises(ValueError, match="provenance"):
        registry.activate(listed[0]["id"])


def test_predictor_rejects_untraceable_checkpoint(tmp_path):
    path = tmp_path / "model.pt"
    model = build_model("eegnet", channels=2, classes=1)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": ["yes"],
            "channels": 2,
            "samples": 8,
            "dataset_id": "nm000113",
            "task": "words",
            "architecture": "eegnet",
        },
        path,
    )
    predictor = Predictor(path)
    assert predictor.ready is False
    assert "provenance" in (predictor.load_error or "").lower()
