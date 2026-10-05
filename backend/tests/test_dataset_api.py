from fastapi.testclient import TestClient

from app.main import app


def test_dataset_catalog_is_local_and_supported_only():
    client = TestClient(app)
    response = client.get("/api/datasets")
    assert response.status_code == 200

    payload = response.json()
    ids = {item["id"] for item in payload["datasets"]}
    assert ids == {"nm000113", "on003626"}
    assert "No synthetic EEG" in payload["rule"]


def test_unsupported_dataset_is_rejected_before_download():
    client = TestClient(app)
    response = client.post(
        "/api/datasets/not-a-dataset/download",
        json={"mode": "recommended"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Unsupported public dataset."


def test_unsupported_dataset_is_rejected_before_prepare():
    client = TestClient(app)
    response = client.post("/api/datasets/not-a-dataset/prepare")
    assert response.status_code == 404


def test_preprocessing_profile_is_reproducible():
    client = TestClient(app)
    response = client.get("/api/preprocessing")
    assert response.status_code == 200
    payload = response.json()
    assert payload["profile"] == "lucid-standard-v1"
    assert payload["immutable_for_v1"] is True
    assert payload["raw_eeg"]["bandpass_hz"] == [1.0, 40.0]
    assert payload["raw_eeg"]["target_sfreq_hz"] == 128.0
    assert payload["raw_eeg"]["window_seconds"] == 2.0


def test_nm000113_rejects_state_training_request():
    client = TestClient(app)
    response = client.post(
        "/api/training",
        json={
            "dataset_id": "nm000113",
            "task": "state",
            "kind": "train",
            "architecture": "eegnet",
            "epochs": 1,
        },
    )
    assert response.status_code == 409
    assert "does not provide" in response.json()["detail"]


def test_unknown_model_activation_is_rejected():
    client = TestClient(app)
    response = client.post("/api/models/not-a-model/activate")
    assert response.status_code == 404
