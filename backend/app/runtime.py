from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading

from .config import MODELS_DIR
from .inference import Predictor
from .model_registry import model_registry, prepared_path
from .replay import PublicEEGReplay


@dataclass(frozen=True)
class RuntimeSnapshot:
    dataset_id: str
    replay: PublicEEGReplay
    word_predictor: Predictor
    state_predictor: Predictor


class LucidRuntime:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._snapshot = self._build(model_registry.active_dataset)

    def _missing_model_path(self, dataset_id: str, task: str) -> Path:
        return MODELS_DIR / ".unavailable" / f"{dataset_id}_{task}.pt"

    def _build(self, dataset_id: str) -> RuntimeSnapshot:
        replay = PublicEEGReplay(prepared_path(dataset_id, "words"))

        word_path = model_registry.active_path(dataset_id, "words")
        state_path = model_registry.active_path(dataset_id, "state")

        word_predictor = Predictor(
            word_path
            if word_path is not None
            else self._missing_model_path(dataset_id, "words")
        )
        state_predictor = Predictor(
            state_path
            if state_path is not None
            else self._missing_model_path(dataset_id, "state")
        )

        return RuntimeSnapshot(
            dataset_id=dataset_id,
            replay=replay,
            word_predictor=word_predictor,
            state_predictor=state_predictor,
        )

    def snapshot(self) -> RuntimeSnapshot:
        with self._lock:
            return self._snapshot

    def refresh(self) -> RuntimeSnapshot:
        with self._lock:
            self._snapshot = self._build(
                model_registry.active_dataset
            )
            return self._snapshot

    def select_dataset(self, dataset_id: str) -> RuntimeSnapshot:
        archive = prepared_path(dataset_id, "words")
        if not archive.exists():
            raise FileNotFoundError(
                "Prepared word EEG archive is missing for this dataset."
            )
        probe = PublicEEGReplay(archive)
        if not probe.ready:
            raise ValueError(
                "Prepared EEG archive is not provenance-ready."
            )

        model_registry.set_active_dataset(dataset_id)
        with self._lock:
            self._snapshot = self._build(dataset_id)
            return self._snapshot


runtime = LucidRuntime()
