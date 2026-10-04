from typing import Literal
from pydantic import BaseModel, Field


class Alternative(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class Prediction(BaseModel):
    status: Literal["ok", "model_unavailable", "data_unavailable", "error"]
    state: str | None = None
    state_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    prediction: str | None = None
    prediction_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    alternatives: list[Alternative] = []
    source_dataset: str | None = None
    source_recording: str | None = None
    model_version: str | None = None
    message: str | None = None


class StatusResponse(BaseModel):
    data_ready: bool
    model_ready: bool
    dataset_id: str
    prepared_path: str
    model_path: str
    rule: str = "No synthetic EEG or fabricated predictions."
