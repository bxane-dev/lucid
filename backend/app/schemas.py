from typing import Literal
from pydantic import BaseModel, Field


class Alternative(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class Prediction(BaseModel):
    status: Literal[
        "ok",
        "model_unavailable",
        "data_unavailable",
        "error",
    ]
    state: str | None = None
    state_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    prediction: str | None = None
    prediction_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    alternatives: list[Alternative] = Field(
        default_factory=list
    )
    source_dataset: str | None = None
    source_recording: str | None = None
    model_version: str | None = None
    provenance_sha256: str | None = None
    message: str | None = None


class StatusResponse(BaseModel):
    data_ready: bool
    model_ready: bool
    state_model_ready: bool
    dataset_id: str
    prepared_path: str
    model_path: str
    state_model_path: str
    prepared_provenance: str | None = None
    model_provenance: str | None = None
    state_model_provenance: str | None = None
    rule: str = (
        "No synthetic EEG or fabricated predictions."
    )


class DatasetDownloadRequest(BaseModel):
    mode: Literal["recommended", "all"] = "recommended"
    subjects: list[str] | None = None


class DatasetJobResponse(BaseModel):
    id: str
    kind: Literal["download", "prepare"]
    dataset_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: dict = Field(default_factory=dict)
    error: str | None = None
    result: dict | None = None
    created_at: float
    updated_at: float


class TrainingRequest(BaseModel):
    dataset_id: str
    task: Literal["words", "state"] = "words"
    kind: Literal["train", "benchmark"] = "train"
    architecture: str | None = "eegnet"
    architectures: list[str] | None = None
    epochs: int = Field(default=25, ge=1, le=500)
    batch_size: int = Field(default=32, ge=1, le=1024)
    seed: int = 42
    auto_activate: bool = True


class TrainingJobResponse(BaseModel):
    id: str
    kind: Literal["train", "benchmark"]
    dataset_id: str
    task: Literal["words", "state"]
    architecture: str | None = None
    architectures: list[str] | None = None
    epochs: int
    batch_size: int
    seed: int
    auto_activate: bool
    status: Literal[
        "queued",
        "running",
        "completed",
        "failed",
        "cancelled",
    ]
    cancel_requested: bool = False
    progress: dict = Field(default_factory=dict)
    result: dict | None = None
    error: str | None = None
    created_at: float
    updated_at: float


class DatasetSelectRequest(BaseModel):
    dataset_id: str


class ModelActivateResponse(BaseModel):
    id: str
    path: str
    dataset_id: str
    task: str
    architecture: str
    provenance_sha256: str
    checkpoint_sha256: str
    best_validation_balanced_accuracy: float | None = None
    test_accuracy: float | None = None
    test_balanced_accuracy: float | None = None
    best_epoch: int | None = None
    labels: list[str] = Field(default_factory=list)
    active: bool
    model_version: str | None = None
