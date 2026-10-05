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
