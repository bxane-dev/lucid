from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import DEFAULT_MODEL_PATH
from .model import build_model
from .schemas import Alternative, Prediction


@dataclass
class LoadedModel:
    model: torch.nn.Module
    labels: list[str]
    channels: int
    samples: int
    dataset_id: str
    version: str
    task: str
    architecture: str
    provenance_sha256: str
    device: torch.device


@dataclass
class ClassResult:
    status: str
    label: str | None = None
    confidence: float | None = None
    probabilities: (
        list[tuple[str, float]]
        | None
    ) = None
    message: str | None = None


class Predictor:
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
    ) -> None:
        self.model_path = Path(
            model_path
        )
        self.loaded: LoadedModel | None = None
        self.load_error: str | None = None
        self.reload()

    @property
    def ready(self) -> bool:
        return self.loaded is not None

    def reload(self) -> None:
        self.loaded = None
        self.load_error = None

        if not self.model_path.exists():
            self.load_error = (
                "Model not found: "
                f"{self.model_path}"
            )
            return

        try:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
            checkpoint = torch.load(
                self.model_path,
                map_location=device,
                weights_only=False,
            )
            labels = list(
                checkpoint["labels"]
            )
            channels = int(
                checkpoint["channels"]
            )
            samples = int(
                checkpoint["samples"]
            )
            task = str(
                checkpoint.get(
                    "task",
                    "words",
                )
            )
            architecture = str(
                checkpoint.get(
                    "architecture",
                    "eegnet",
                )
            )
            provenance_sha256 = str(
                checkpoint.get(
                    "provenance_sha256",
                    "",
                )
            )
            if len(provenance_sha256) != 64:
                raise ValueError(
                    "Checkpoint has no valid source provenance. "
                    "Retrain it from a provenance-enabled prepared archive."
                )

            model = build_model(
                architecture,
                channels=channels,
                classes=len(labels),
            )
            model.load_state_dict(
                checkpoint["state_dict"]
            )
            model.to(device).eval()

            self.loaded = LoadedModel(
                model=model,
                labels=labels,
                channels=channels,
                samples=samples,
                dataset_id=str(
                    checkpoint[
                        "dataset_id"
                    ]
                ),
                version=str(
                    checkpoint.get(
                        "model_version",
                        (
                            f"{architecture}_"
                            f"{task}_v1"
                        ),
                    )
                ),
                task=task,
                architecture=architecture,
                provenance_sha256=provenance_sha256,
                device=device,
            )
        except Exception as exc:
            self.load_error = str(exc)

    def classify(
        self,
        window: np.ndarray,
        *,
        source_provenance: str | None = None,
    ) -> ClassResult:
        if self.loaded is None:
            return ClassResult(
                status=(
                    "model_unavailable"
                ),
                message=(
                    self.load_error
                    or (
                        "No trained model "
                        "is loaded."
                    )
                ),
            )

        loaded = self.loaded
        if (
            source_provenance is not None
            and source_provenance != loaded.provenance_sha256
        ):
            return ClassResult(
                status="error",
                message=(
                    "Model/source provenance mismatch. "
                    "Lucid refuses to score EEG from a different prepared source."
                ),
            )

        x = np.asarray(
            window,
            dtype=np.float32,
        )

        if x.shape != (
            loaded.channels,
            loaded.samples,
        ):
            return ClassResult(
                status="error",
                message=(
                    "Model expects "
                    f"{(loaded.channels, loaded.samples)}, "
                    "received "
                    f"{tuple(x.shape)}"
                ),
            )

        tensor = (
            torch.from_numpy(x)
            .unsqueeze(0)
            .to(loaded.device)
        )

        with torch.inference_mode():
            probs = (
                torch.softmax(
                    loaded.model(
                        tensor
                    ),
                    dim=1,
                )[0]
                .cpu()
                .numpy()
            )

        order = np.argsort(
            probs
        )[::-1]
        best = int(order[0])
        probabilities = [
            (
                loaded.labels[
                    int(index)
                ],
                float(
                    probs[int(index)]
                ),
            )
            for index in order
        ]

        return ClassResult(
            status="ok",
            label=loaded.labels[best],
            confidence=float(
                probs[best]
            ),
            probabilities=(
                probabilities
            ),
        )

    def predict(
        self,
        window: np.ndarray,
        *,
        source_recording: (
            str | None
        ) = None,
        source_provenance: (
            str | None
        ) = None,
    ) -> Prediction:
        result = self.classify(
            window,
            source_provenance=source_provenance,
        )

        if result.status != "ok":
            return Prediction(
                status=result.status,
                source_dataset=(
                    self.loaded.dataset_id
                    if self.loaded
                    else None
                ),
                model_version=(
                    self.loaded.version
                    if self.loaded
                    else None
                ),
                provenance_sha256=(
                    self.loaded.provenance_sha256
                    if self.loaded
                    else None
                ),
                message=result.message,
            )

        assert self.loaded is not None
        probabilities = (
            result.probabilities
            or []
        )
        alternatives = [
            Alternative(
                label=label,
                confidence=confidence,
            )
            for label, confidence
            in probabilities[1:]
        ]

        return Prediction(
            status="ok",
            prediction=result.label,
            prediction_confidence=(
                result.confidence
            ),
            alternatives=alternatives,
            source_dataset=(
                self.loaded.dataset_id
            ),
            source_recording=(
                source_recording
            ),
            model_version=(
                self.loaded.version
            ),
            provenance_sha256=(
                self.loaded.provenance_sha256
            ),
        )
