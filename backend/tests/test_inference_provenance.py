import numpy as np
import torch

from app.inference import LoadedModel, Predictor


def test_predictor_refuses_provenance_mismatch(tmp_path):
    predictor = Predictor(tmp_path / "missing.pt")
    predictor.loaded = LoadedModel(
        model=torch.nn.Identity(),
        labels=["yes"],
        channels=2,
        samples=4,
        dataset_id="nm000113",
        version="test",
        task="words",
        architecture="eegnet",
        provenance_sha256="a" * 64,
        device=torch.device("cpu"),
    )
    predictor.load_error = None

    result = predictor.classify(
        np.zeros((2, 4), dtype=np.float32),
        source_provenance="b" * 64,
    )

    assert result.status == "error"
    assert "provenance mismatch" in (result.message or "").lower()
