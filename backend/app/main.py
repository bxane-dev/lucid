import asyncio
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware

from .db import init_db, log_prediction
from .dataset_jobs import dataset_jobs
from .model import ARCHITECTURES
from .model_registry import model_registry, prepared_path
from .preprocess import DEFAULT_PREPROCESS
from .public_data import DATASETS
from .runtime import runtime
from .schemas import (
    DatasetDownloadRequest,
    DatasetJobResponse,
    DatasetSelectRequest,
    ModelActivateResponse,
    Prediction,
    StatusResponse,
    TrainingJobResponse,
    TrainingRequest,
)
from .training import read_prepared_metadata
from .training_jobs import training_jobs


def attach_state_prediction(
    result: Prediction,
    window,
    source_provenance: str | None,
    state_predictor,
) -> Prediction:
    state_result = state_predictor.classify(
        window,
        source_provenance=source_provenance,
    )

    if state_result.status == "ok":
        result.state = state_result.label
        result.state_confidence = state_result.confidence

    return result


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    runtime.refresh()
    yield


app = FastAPI(
    title="Lucid EEG API",
    version="1.0.0",
    description=(
        "Zero-cost constrained EEG classification research API using "
        "real public recordings."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}


@app.get(
    "/api/status",
    response_model=StatusResponse,
)
def status():
    snapshot = runtime.snapshot()
    return StatusResponse(
        data_ready=snapshot.replay.ready,
        model_ready=snapshot.word_predictor.ready,
        state_model_ready=snapshot.state_predictor.ready,
        dataset_id=snapshot.dataset_id,
        prepared_path=str(snapshot.replay.path),
        model_path=str(snapshot.word_predictor.model_path),
        state_model_path=str(snapshot.state_predictor.model_path),
        prepared_provenance=snapshot.replay.provenance_sha256,
        model_provenance=(
            snapshot.word_predictor.loaded.provenance_sha256
            if snapshot.word_predictor.loaded
            else None
        ),
        state_model_provenance=(
            snapshot.state_predictor.loaded.provenance_sha256
            if snapshot.state_predictor.loaded
            else None
        ),
    )


@app.get("/api/preprocessing")
def preprocessing_status():
    archives = []
    for dataset_id, meta in DATASETS.items():
        for task in meta["tasks"]:
            path = prepared_path(dataset_id, task)
            archive = None
            error = None
            if path.exists():
                try:
                    archive = read_prepared_metadata(path)
                except Exception as exc:
                    error = str(exc)
            archives.append(
                {
                    "dataset_id": dataset_id,
                    "task": task,
                    "prepared": archive is not None,
                    "path": str(path),
                    "metadata": archive,
                    "error": error,
                }
            )

    return {
        "profile": "lucid-standard-v1",
        "immutable_for_v1": True,
        "reason": (
            "Lucid v1 keeps one documented preprocessing profile so "
            "training, evaluation, replay, and provenance remain reproducible."
        ),
        "raw_eeg": {
            "notch": "recording metadata line frequency when available",
            "bandpass_hz": [
                DEFAULT_PREPROCESS.l_freq,
                DEFAULT_PREPROCESS.h_freq,
            ],
            "average_reference": True,
            "target_sfreq_hz": DEFAULT_PREPROCESS.target_sfreq,
            "artifact_peak_to_peak_limit_uv": (
                DEFAULT_PREPROCESS.peak_to_peak_limit_uv
            ),
            "window_seconds": 2.0,
            "normalization": "per-trial per-channel z-score",
        },
        "published_derivatives": {
            "policy": (
                "Preserve published derivative epochs; resample to "
                "the Lucid target rate and normalize windows."
            )
        },
        "archives": archives,
    }


@app.get("/api/datasets")
def datasets():
    active = model_registry.active_dataset
    entries = []
    for item in dataset_jobs.list_datasets():
        entries.append(
            {
                **item,
                "active": item["id"] == active,
            }
        )
    return {
        "datasets": entries,
        "active_dataset": active,
        "rule": (
            "Only Lucid's verified public dataset registry can be downloaded. "
            "No synthetic EEG or arbitrary data source is substituted."
        ),
    }


@app.post(
    "/api/datasets/{dataset_id}/download",
    response_model=DatasetJobResponse,
)
def download_public_dataset(
    dataset_id: str,
    request: DatasetDownloadRequest,
):
    if dataset_id not in DATASETS:
        raise HTTPException(
            status_code=404,
            detail="Unsupported public dataset.",
        )

    meta = DATASETS[dataset_id]
    subjects = request.subjects
    if subjects is None and request.mode == "recommended":
        subjects = list(meta["recommended_subjects"])
    elif subjects is None and request.mode == "all":
        subjects = None

    try:
        return dataset_jobs.start_download(
            dataset_id,
            subjects=subjects,
            derivatives_only=bool(meta["derivatives_only"]),
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc


@app.post(
    "/api/datasets/{dataset_id}/prepare",
    response_model=DatasetJobResponse,
)
def prepare_public_dataset(dataset_id: str):
    if dataset_id not in DATASETS:
        raise HTTPException(
            status_code=404,
            detail="Unsupported public dataset.",
        )
    try:
        return dataset_jobs.start_prepare(dataset_id)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc


@app.get(
    "/api/dataset-jobs/{job_id}",
    response_model=DatasetJobResponse,
)
def dataset_job(job_id: str):
    job = dataset_jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset job not found.",
        )

    if (
        job["kind"] == "prepare"
        and job["status"] == "completed"
        and job["dataset_id"] == model_registry.active_dataset
    ):
        runtime.refresh()

    return job


@app.post("/api/datasets/{dataset_id}/activate")
def activate_dataset(dataset_id: str):
    if dataset_id not in DATASETS:
        raise HTTPException(
            status_code=404,
            detail="Unsupported public dataset.",
        )
    try:
        snapshot = runtime.select_dataset(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {
        "dataset_id": snapshot.dataset_id,
        "data_ready": snapshot.replay.ready,
        "word_model_ready": snapshot.word_predictor.ready,
        "state_model_ready": snapshot.state_predictor.ready,
        "provenance_sha256": snapshot.replay.provenance_sha256,
    }


@app.get("/api/models")
def models():
    snapshot = runtime.snapshot()
    return {
        "active_dataset": snapshot.dataset_id,
        "architectures": list(ARCHITECTURES),
        "models": model_registry.list_models(),
    }


@app.post(
    "/api/models/{model_id}/activate",
    response_model=ModelActivateResponse,
)
def activate_model(model_id: str):
    try:
        model = model_registry.activate(model_id)
        runtime.refresh()
        return model
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post(
    "/api/training",
    response_model=TrainingJobResponse,
)
def start_training(request: TrainingRequest):
    if request.dataset_id not in DATASETS:
        raise HTTPException(
            status_code=404,
            detail="Unsupported public dataset.",
        )
    if request.task not in DATASETS[request.dataset_id]["tasks"]:
        raise HTTPException(
            status_code=409,
            detail=(
                f"{request.dataset_id} does not provide the "
                f"{request.task!r} Lucid task."
            ),
        )

    try:
        return training_jobs.start(
            dataset_id=request.dataset_id,
            task=request.task,
            kind=request.kind,
            architecture=request.architecture,
            architectures=request.architectures,
            epochs=request.epochs,
            batch_size=request.batch_size,
            seed=request.seed,
            auto_activate=request.auto_activate,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/training")
def list_training_jobs():
    return {"jobs": training_jobs.list()}


@app.get(
    "/api/training/{job_id}",
    response_model=TrainingJobResponse,
)
def training_job(job_id: str):
    job = training_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found.")
    if (
        job["status"] == "completed"
        and job.get("auto_activate")
    ):
        runtime.refresh()
    return job


@app.post(
    "/api/training/{job_id}/cancel",
    response_model=TrainingJobResponse,
)
def cancel_training(job_id: str):
    try:
        return training_jobs.cancel(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/reload-model")
def reload_model():
    snapshot = runtime.refresh()
    return {
        "dataset_id": snapshot.dataset_id,
        "model_ready": snapshot.word_predictor.ready,
        "model_error": snapshot.word_predictor.load_error,
        "state_model_ready": snapshot.state_predictor.ready,
        "state_model_error": snapshot.state_predictor.load_error,
    }


@app.post(
    "/api/predict/replay",
    response_model=Prediction,
)
def predict_replay():
    snapshot = runtime.snapshot()
    if not snapshot.replay.ready:
        return Prediction(
            status="data_unavailable",
            source_dataset=snapshot.dataset_id,
            message=(
                "No provenance-ready public EEG is available for the active "
                "dataset. Download and prepare public data in Dataset Manager."
            ),
        )

    trial = snapshot.replay.next_test_trial()
    result = snapshot.word_predictor.predict(
        trial.data,
        source_recording=trial.recording,
        source_provenance=trial.provenance_sha256,
    )
    result = attach_state_prediction(
        result,
        trial.data,
        trial.provenance_sha256,
        snapshot.state_predictor,
    )

    if result.status == "ok":
        log_prediction(
            source_dataset=result.source_dataset,
            source_recording=result.source_recording,
            ground_truth=trial.ground_truth,
            prediction=result.prediction,
            confidence=result.prediction_confidence,
            model_version=result.model_version,
        )

    return result


@app.websocket("/ws/live")
async def live(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            snapshot = runtime.snapshot()
            if not snapshot.replay.ready:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": (
                            "No provenance-ready public EEG is selected. "
                            "Use Dataset Manager to download, prepare, and "
                            "activate a supported dataset."
                        ),
                    }
                )
                await asyncio.sleep(2.0)
                continue

            trial = snapshot.replay.next_test_trial()

            await websocket.send_json(
                {
                    "type": "trial_start",
                    "dataset": trial.dataset_id,
                    "recording": trial.recording,
                    "ground_truth": trial.ground_truth,
                    "sfreq": trial.sfreq,
                    "channels": int(trial.data.shape[0]),
                    "samples": int(trial.data.shape[1]),
                    "provenance_sha256": trial.provenance_sha256,
                }
            )

            interval = 1.0 / trial.sfreq
            visible_channels = min(8, trial.data.shape[0])

            for sample_idx in range(trial.data.shape[1]):
                await websocket.send_json(
                    {
                        "type": "sample",
                        "index": sample_idx,
                        "values": trial.data[
                            :visible_channels,
                            sample_idx,
                        ].tolist(),
                    }
                )
                await asyncio.sleep(interval)

            result = snapshot.word_predictor.predict(
                trial.data,
                source_recording=trial.recording,
                source_provenance=trial.provenance_sha256,
            )
            result = attach_state_prediction(
                result,
                trial.data,
                trial.provenance_sha256,
                snapshot.state_predictor,
            )

            await websocket.send_json(
                {
                    "type": "prediction",
                    "ground_truth": trial.ground_truth,
                    "prediction": result.model_dump(),
                }
            )

            if result.status == "ok":
                log_prediction(
                    source_dataset=result.source_dataset,
                    source_recording=result.source_recording,
                    ground_truth=trial.ground_truth,
                    prediction=result.prediction,
                    confidence=result.prediction_confidence,
                    model_version=result.model_version,
                )

            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        return
