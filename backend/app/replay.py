from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .config import DEFAULT_PREPARED_PATH


@dataclass
class ReplayTrial:
    data: np.ndarray
    ground_truth: str
    recording: str
    sfreq: float
    dataset_id: str
    provenance_sha256: str


class PublicEEGReplay:
    def __init__(self, path: Path = DEFAULT_PREPARED_PATH) -> None:
        self.path = Path(path)
        self._archive = None
        self._cursor = 0

    @property
    def ready(self) -> bool:
        if not self.path.exists():
            return False
        try:
            return self.provenance_sha256 is not None
        except Exception:
            return False

    @property
    def provenance_sha256(self) -> str | None:
        if not self.path.exists():
            return None
        archive = self._load()
        if "provenance_sha256" not in archive.files:
            return None
        value = str(archive["provenance_sha256"])
        return value if len(value) == 64 else None

    def reload(self) -> None:
        if self._archive is not None:
            try:
                self._archive.close()
            except Exception:
                pass
        self._archive = None
        self._cursor = 0

    def _load(self):
        if self._archive is None:
            if not self.path.exists():
                raise FileNotFoundError(f"Prepared public EEG not found at {self.path}")
            self._archive = np.load(self.path, allow_pickle=False)
        return self._archive

    def next_test_trial(self) -> ReplayTrial:
        archive = self._load()
        test_indices = np.flatnonzero(archive["split"] == 2)
        if len(test_indices) == 0:
            raise RuntimeError("Prepared dataset contains no held-out test trials.")
        idx = int(test_indices[self._cursor % len(test_indices)])
        self._cursor += 1
        if "provenance_sha256" not in archive.files:
            raise RuntimeError(
                "Prepared public EEG has no provenance hash. "
                "Re-run dataset preparation."
            )
        labels = archive["labels"].tolist()
        label_idx = int(archive["y"][idx])
        return ReplayTrial(
            archive["x"][idx].astype(np.float32, copy=False),
            str(labels[label_idx]), str(archive["recordings"][idx]),
            float(archive["sfreq"]), str(archive["dataset_id"]),
            str(archive["provenance_sha256"]),
        )
