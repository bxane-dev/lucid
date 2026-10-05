from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading

import numpy as np

from .config import MODELS_DIR, PREPARED_DATA_DIR, RUNTIME_DIR
from .inference import Predictor
from .provenance import sha256_file


SETTINGS_PATH = RUNTIME_DIR / "lucid-settings.json"


def _model_id(path: Path) -> str:
    relative = path.resolve().relative_to(MODELS_DIR.resolve()).as_posix()
    return hashlib.sha256(relative.encode("utf-8")).hexdigest()[:24]


def prepared_path(dataset_id: str, task: str) -> Path:
    return PREPARED_DATA_DIR / f"{dataset_id}_{task}.npz"


class ModelRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._settings = self._load_settings()

    def _load_settings(self) -> dict:
        defaults = {
            "active_dataset": "nm000113",
            "active_models": {},
        }
        if not SETTINGS_PATH.exists():
            return defaults
        try:
            payload = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return defaults

        active_models = payload.get("active_models")
        if not isinstance(active_models, dict):
            active_models = {}
        active_dataset = payload.get("active_dataset")
        if not isinstance(active_dataset, str) or not active_dataset:
            active_dataset = "nm000113"
        return {
            "active_dataset": active_dataset,
            "active_models": {
                str(key): str(value)
                for key, value in active_models.items()
                if isinstance(key, str) and isinstance(value, str)
            },
        }

    def _save(self) -> None:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = SETTINGS_PATH.with_suffix(".json.part")
        tmp.write_text(
            json.dumps(self._settings, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(SETTINGS_PATH)

    @property
    def active_dataset(self) -> str:
        with self._lock:
            return str(self._settings["active_dataset"])

    def set_active_dataset(self, dataset_id: str) -> None:
        if not dataset_id or "/" in dataset_id or "\\" in dataset_id:
            raise ValueError("Invalid dataset id.")
        with self._lock:
            self._settings["active_dataset"] = dataset_id
            self._save()

    def _resolve_relative(self, relative: str) -> Path | None:
        try:
            path = (MODELS_DIR / relative).resolve()
            path.relative_to(MODELS_DIR.resolve())
        except Exception:
            return None
        return path

    def active_path(self, dataset_id: str, task: str) -> Path | None:
        key = f"{dataset_id}:{task}"
        with self._lock:
            relative = self._settings["active_models"].get(key)
        if not relative:
            return None
        path = self._resolve_relative(relative)
        return path if path and path.is_file() else None

    def _metrics_for(self, path: Path) -> dict | None:
        metrics_path = path.with_suffix(".metrics.json")
        if not metrics_path.exists():
            return None
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def list_models(self) -> list[dict]:
        models = []
        active_map = {}
        with self._lock:
            active_map = dict(self._settings["active_models"])

        for path in sorted(MODELS_DIR.rglob("*.pt")):
            if ".part" in path.name:
                continue
            metrics = self._metrics_for(path)
            if not metrics:
                continue

            dataset_id = str(metrics.get("dataset_id", ""))
            task = str(metrics.get("task", ""))
            architecture = str(metrics.get("architecture", ""))
            provenance = str(metrics.get("provenance_sha256", ""))
            if (
                not dataset_id
                or task not in {"words", "state"}
                or not architecture
                or len(provenance) != 64
            ):
                continue

            relative = path.resolve().relative_to(MODELS_DIR.resolve()).as_posix()
            key = f"{dataset_id}:{task}"
            models.append(
                {
                    "id": _model_id(path),
                    "path": relative,
                    "dataset_id": dataset_id,
                    "task": task,
                    "architecture": architecture,
                    "provenance_sha256": provenance,
                    "checkpoint_sha256": (
                        metrics.get("checkpoint_sha256")
                        or sha256_file(path)
                    ),
                    "best_validation_balanced_accuracy": metrics.get(
                        "best_validation_balanced_accuracy"
                    ),
                    "test_accuracy": metrics.get("test_accuracy"),
                    "test_balanced_accuracy": metrics.get(
                        "test_balanced_accuracy"
                    ),
                    "best_epoch": metrics.get("best_epoch"),
                    "labels": metrics.get("labels", []),
                    "active": active_map.get(key) == relative,
                }
            )
        return models

    def find(self, model_id: str) -> dict | None:
        return next(
            (model for model in self.list_models() if model["id"] == model_id),
            None,
        )

    def activate(self, model_id: str) -> dict:
        model = self.find(model_id)
        if model is None:
            raise FileNotFoundError("Model not found.")

        archive_path = prepared_path(
            model["dataset_id"],
            model["task"],
        )
        if not archive_path.exists():
            raise ValueError(
                "Matching prepared public EEG is missing. "
                "Prepare the dataset before activating this model."
            )

        with np.load(archive_path, allow_pickle=False) as archive:
            if "provenance_sha256" not in archive.files:
                raise ValueError("Prepared EEG has no provenance hash.")
            archive_provenance = str(archive["provenance_sha256"])

        if archive_provenance != model["provenance_sha256"]:
            raise ValueError(
                "Model provenance does not match the prepared public EEG."
            )

        absolute = (MODELS_DIR / model["path"]).resolve()
        predictor = Predictor(absolute)
        if not predictor.ready:
            raise ValueError(
                "Model could not be loaded: "
                + (predictor.load_error or "unknown error")
            )

        key = f"{model['dataset_id']}:{model['task']}"
        with self._lock:
            self._settings["active_models"][key] = model["path"]
            self._save()

        return {
            **model,
            "active": True,
            "model_version": predictor.loaded.version if predictor.loaded else None,
        }

    def activate_path(self, path: Path) -> dict:
        path = Path(path).resolve()
        try:
            path.relative_to(MODELS_DIR.resolve())
        except ValueError as exc:
            raise ValueError("Model must live under Lucid's models directory.") from exc

        model_id = _model_id(path)
        return self.activate(model_id)

    def activate_best(
        self,
        dataset_id: str,
        task: str,
        provenance_sha256: str | None = None,
    ) -> dict | None:
        candidates = [
            model
            for model in self.list_models()
            if model["dataset_id"] == dataset_id
            and model["task"] == task
            and (
                provenance_sha256 is None
                or model["provenance_sha256"] == provenance_sha256
            )
            and isinstance(
                model["best_validation_balanced_accuracy"],
                (int, float),
            )
        ]
        if not candidates:
            return None
        winner = max(
            candidates,
            key=lambda item: float(
                item["best_validation_balanced_accuracy"]
            ),
        )
        return self.activate(winner["id"])


model_registry = ModelRegistry()
