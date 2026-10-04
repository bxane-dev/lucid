from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from .config import DEFAULT_MODEL_PATH
from .model import EEGNet
from .schemas import Alternative, Prediction


@dataclass
class LoadedModel:
    model: EEGNet
    labels: list[str]
    channels: int
    samples: int
    dataset_id: str
    version: str
    device: torch.device


class Predictor:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
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
            self.load_error = f"Model not found: {self.model_path}"
            return
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)
            labels = list(checkpoint["labels"])
            channels = int(checkpoint["channels"])
            samples = int(checkpoint["samples"])
            model = EEGNet(channels=channels, classes=len(labels))
            model.load_state_dict(checkpoint["state_dict"])
            model.to(device).eval()
            self.loaded = LoadedModel(model, labels, channels, samples, str(checkpoint["dataset_id"]), str(checkpoint.get("model_version", "eegnet_words_v1")), device)
        except Exception as exc:
            self.load_error = str(exc)

    def predict(self, window: np.ndarray, *, source_recording: str | None = None) -> Prediction:
        if self.loaded is None:
            return Prediction(status="model_unavailable", message=self.load_error or "No trained model is loaded.")
        loaded = self.loaded
        x = np.asarray(window, dtype=np.float32)
        if x.shape != (loaded.channels, loaded.samples):
            return Prediction(status="error", source_dataset=loaded.dataset_id, model_version=loaded.version, message=f"Model expects {(loaded.channels, loaded.samples)}, received {tuple(x.shape)}")
        tensor = torch.from_numpy(x).unsqueeze(0).to(loaded.device)
        with torch.inference_mode():
            probs = torch.softmax(loaded.model(tensor), dim=1)[0].cpu().numpy()
        order = np.argsort(probs)[::-1]
        best = int(order[0])
        alternatives = [Alternative(label=loaded.labels[int(i)], confidence=float(probs[int(i)])) for i in order[1:]]
        return Prediction(
            status="ok", state="imagined_speech", state_confidence=None,
            prediction=loaded.labels[best], prediction_confidence=float(probs[best]),
            alternatives=alternatives, source_dataset=loaded.dataset_id,
            source_recording=source_recording, model_version=loaded.version,
        )
