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
