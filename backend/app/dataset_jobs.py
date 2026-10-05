from __future__ import annotations

from copy import deepcopy
import threading
import time
import uuid

from .config import PREPARED_DATA_DIR, PUBLIC_DATA_DIR
from .dataset import prepare_dataset
from .inner_speech import prepare_nieto_derivatives
from .public_data import DATASETS, download_dataset, local_dataset_status


class DatasetJobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}

    def _set(self, job_id: str, **values) -> None:
        with self._lock:
            self._jobs[job_id].update(values)
            self._jobs[job_id]["updated_at"] = time.time()

    def _progress(self, job_id: str, update: dict) -> None:
        self._set(job_id, progress=update)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return deepcopy(job) if job else None

    def list_datasets(self) -> list[dict]:
        return [local_dataset_status(dataset_id) for dataset_id in DATASETS]

    def start_download(
        self,
        dataset_id: str,
        *,
        subjects: list[str] | None,
        derivatives_only: bool | None,
    ) -> dict:
        job_id = uuid.uuid4().hex
        now = time.time()
        job = {
            "id": job_id,
            "kind": "download",
            "dataset_id": dataset_id,
            "status": "queued",
            "progress": {"phase": "queued"},
            "error": None,
            "result": None,
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            for existing in self._jobs.values():
                if (
                    existing["dataset_id"] == dataset_id
                    and existing["status"] in {"queued", "running"}
                ):
                    raise RuntimeError(
                        "Another operation for this dataset is already running."
                    )
            self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._download_worker,
            args=(job_id, dataset_id, subjects, derivatives_only),
            daemon=True,
        )
        thread.start()
        return deepcopy(job)

    def _download_worker(
        self,
        job_id: str,
        dataset_id: str,
        subjects: list[str] | None,
        derivatives_only: bool | None,
    ) -> None:
        self._set(job_id, status="running")
        try:
            result = download_dataset(
                dataset_id,
                subjects=subjects,
                derivatives_only=derivatives_only,
                progress=lambda update: self._progress(job_id, update),
            )
            self._set(
                job_id,
                status="completed",
                result=result,
                progress={"phase": "completed"},
            )
        except Exception as exc:
            self._set(
                job_id,
                status="failed",
                error=str(exc),
                progress={"phase": "failed", "message": str(exc)},
            )

    def start_prepare(self, dataset_id: str) -> dict:
        job_id = uuid.uuid4().hex
        now = time.time()
        job = {
            "id": job_id,
            "kind": "prepare",
            "dataset_id": dataset_id,
            "status": "queued",
            "progress": {"phase": "queued"},
            "error": None,
            "result": None,
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            for existing in self._jobs.values():
                if (
                    existing["dataset_id"] == dataset_id
                    and existing["status"] in {"queued", "running"}
                ):
                    raise RuntimeError(
                        "Another operation for this dataset is already running."
                    )
            self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._prepare_worker,
            args=(job_id, dataset_id),
            daemon=True,
        )
        thread.start()
        return deepcopy(job)

    def _prepare_worker(self, job_id: str, dataset_id: str) -> None:
        self._set(
            job_id,
            status="running",
            progress={"phase": "preparing", "message": "Reading real public EEG."},
        )
        try:
            if dataset_id == "nm000113":
                result = {
                    "words": prepare_dataset(
                        PUBLIC_DATA_DIR / dataset_id,
                        PREPARED_DATA_DIR / "nm000113_words.npz",
                        dataset_id,
                    )
                }
            elif dataset_id == "on003626":
                result = prepare_nieto_derivatives(
                    dataset_root=PUBLIC_DATA_DIR / dataset_id,
                    word_output=PREPARED_DATA_DIR / "on003626_words.npz",
                    state_output=PREPARED_DATA_DIR / "on003626_state.npz",
                )
            else:
                raise ValueError(f"Unsupported dataset: {dataset_id}")

            self._set(
                job_id,
                status="completed",
                result=result,
                progress={"phase": "completed"},
            )
        except Exception as exc:
            self._set(
                job_id,
                status="failed",
                error=str(exc),
                progress={"phase": "failed", "message": str(exc)},
            )


dataset_jobs = DatasetJobManager()
