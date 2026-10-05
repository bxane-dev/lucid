from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import threading
import time
import uuid

from .config import MODELS_DIR
from .model import ARCHITECTURES
from .model_registry import model_registry, prepared_path
from .training import benchmark_models, read_prepared_metadata, train_model


class TrainingCancelled(RuntimeError):
    pass


class TrainingJobManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}

    def _set(self, job_id: str, **values) -> None:
        with self._lock:
            if job_id not in self._jobs:
                return
            self._jobs[job_id].update(values)
            self._jobs[job_id]["updated_at"] = time.time()

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return deepcopy(job) if job else None

    def list(self) -> list[dict]:
        with self._lock:
            return [
                deepcopy(job)
                for job in sorted(
                    self._jobs.values(),
                    key=lambda item: item["created_at"],
                    reverse=True,
                )
            ]

    def _active_job(self) -> dict | None:
        return next(
            (
                job
                for job in self._jobs.values()
                if job["status"] in {"queued", "running"}
            ),
            None,
        )

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise FileNotFoundError("Training job not found.")
            if job["status"] not in {"queued", "running"}:
                return deepcopy(job)
            job["cancel_requested"] = True
            job["updated_at"] = time.time()
            return deepcopy(job)

    def _progress(self, job_id: str, update: dict) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise TrainingCancelled("Training job no longer exists.")
            if job.get("cancel_requested"):
                raise TrainingCancelled("Training cancelled by user.")
        self._set(job_id, progress=update)

    def start(
        self,
        *,
        dataset_id: str,
        task: str,
        kind: str,
        architecture: str | None,
        architectures: list[str] | None,
        epochs: int,
        batch_size: int,
        seed: int,
        auto_activate: bool,
    ) -> dict:
        if task not in {"words", "state"}:
            raise ValueError("Task must be 'words' or 'state'.")
        if kind not in {"train", "benchmark"}:
            raise ValueError("Training kind must be 'train' or 'benchmark'.")
        if architecture is not None and architecture not in ARCHITECTURES:
            raise ValueError("Unsupported model architecture.")
        if architectures is not None:
            unknown = sorted(set(architectures) - set(ARCHITECTURES))
            if unknown:
                raise ValueError(
                    "Unsupported architectures: " + ", ".join(unknown)
                )

        archive = prepared_path(dataset_id, task)
        metadata = read_prepared_metadata(archive)
        if metadata["dataset_id"] != dataset_id or metadata["task"] != task:
            raise ValueError("Prepared archive metadata does not match request.")

        with self._lock:
            active = self._active_job()
            if active:
                raise RuntimeError(
                    f"Training job {active['id']} is already running."
                )

            job_id = uuid.uuid4().hex
            now = time.time()
            job = {
                "id": job_id,
                "kind": kind,
                "dataset_id": dataset_id,
                "task": task,
                "architecture": architecture,
                "architectures": architectures,
                "epochs": epochs,
                "batch_size": batch_size,
                "seed": seed,
                "auto_activate": auto_activate,
                "status": "queued",
                "cancel_requested": False,
                "progress": {"phase": "queued"},
                "result": None,
                "error": None,
                "created_at": now,
                "updated_at": now,
            }
            self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._worker,
            args=(job_id,),
            daemon=True,
        )
        thread.start()
        return deepcopy(job)

    def _worker(self, job_id: str) -> None:
        with self._lock:
            job = deepcopy(self._jobs[job_id])
        self._set(job_id, status="running")

        try:
            archive = prepared_path(job["dataset_id"], job["task"])
            output_dir = (
                MODELS_DIR
                / job["dataset_id"]
                / job["task"]
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            if job["kind"] == "benchmark":
                architectures = (
                    job["architectures"]
                    if job["architectures"]
                    else list(ARCHITECTURES)
                )
                result = benchmark_models(
                    archive,
                    output_dir / "benchmark",
                    architectures=architectures,
                    epochs=job["epochs"],
                    batch_size=job["batch_size"],
                    seed=job["seed"],
                    progress=lambda update: self._progress(job_id, update),
                )
                selected = None
                if job["auto_activate"]:
                    selected = model_registry.activate_path(
                        Path(result["winner_model_path"])
                    )
                result["activated_model"] = selected
            else:
                architecture = job["architecture"] or "eegnet"
                result = train_model(
                    archive,
                    output_dir / f"{architecture}.pt",
                    task=job["task"],
                    architecture=architecture,
                    epochs=job["epochs"],
                    batch_size=job["batch_size"],
                    seed=job["seed"],
                    progress=lambda update: self._progress(job_id, update),
                )
                selected = None
                if job["auto_activate"]:
                    selected = model_registry.activate_path(
                        Path(result["model_path"])
                    )
                result["activated_model"] = selected

            self._progress(job_id, {"phase": "completed"})
            self._set(
                job_id,
                status="completed",
                result=result,
            )
        except TrainingCancelled as exc:
            self._set(
                job_id,
                status="cancelled",
                error=str(exc),
                progress={"phase": "cancelled"},
            )
        except Exception as exc:
            self._set(
                job_id,
                status="failed",
                error=str(exc),
                progress={
                    "phase": "failed",
                    "message": str(exc),
                },
            )


training_jobs = TrainingJobManager()
