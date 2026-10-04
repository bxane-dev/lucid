import asyncio
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    DEFAULT_DATASET_ID,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREPARED_PATH,
    DEFAULT_STATE_MODEL_PATH,
)
from .db import init_db, log_prediction
from .inference import Predictor
from .replay import PublicEEGReplay
from .schemas import Prediction, StatusResponse


predictor = Predictor(DEFAULT_MODEL_PATH)
state_predictor = Predictor(
    DEFAULT_STATE_MODEL_PATH
)
replay = PublicEEGReplay()


def attach_state_prediction(
    result: Prediction,
    window,
) -> Prediction:
    state_result = state_predictor.classify(
        window
    )

    if state_result.status == "ok":
        result.state = state_result.label
        result.state_confidence = (
            state_result.confidence
        )

    return result


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    predictor.reload()
    state_predictor.reload()
    yield


app = FastAPI(
    title="Lucid EEG API",
    version="0.2.0",
    description=(
        "Zero-cost EEG research API using "
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
    return {"status": "ok"}


@app.get(
    "/api/status",
    response_model=StatusResponse,
)
def status():
    return StatusResponse(
        data_ready=replay.ready,
        model_ready=predictor.ready,
        state_model_ready=(
            state_predictor.ready
        ),
        dataset_id=DEFAULT_DATASET_ID,
        prepared_path=str(
            DEFAULT_PREPARED_PATH
        ),
        model_path=str(DEFAULT_MODEL_PATH),
        state_model_path=str(
            DEFAULT_STATE_MODEL_PATH
        ),
    )


@app.post("/api/reload-model")
def reload_model():
    predictor.reload()
    state_predictor.reload()

    return {
        "model_ready": predictor.ready,
        "model_error": predictor.load_error,
        "state_model_ready": (
            state_predictor.ready
        ),
        "state_model_error": (
            state_predictor.load_error
        ),
    }


@app.post(
    "/api/predict/replay",
    response_model=Prediction,
)
def predict_replay():
    if not replay.ready:
        return Prediction(
            status="data_unavailable",
            source_dataset=(
                DEFAULT_DATASET_ID
            ),
            message=(
                "No prepared public EEG is "
                "available. Run the documented "
                "public-data preparation first."
            ),
        )

    trial = replay.next_test_trial()
    result = predictor.predict(
        trial.data,
        source_recording=trial.recording,
    )
    result = attach_state_prediction(
        result,
        trial.data,
    )

    if result.status == "ok":
        log_prediction(
            source_dataset=(
                result.source_dataset
            ),
            source_recording=(
                result.source_recording
            ),
            ground_truth=(
                trial.ground_truth
            ),
            prediction=result.prediction,
            confidence=(
                result.prediction_confidence
            ),
            model_version=(
                result.model_version
            ),
        )

    return result


@app.websocket("/ws/live")
async def live(websocket: WebSocket):
    await websocket.accept()

    if not replay.ready:
        await websocket.send_json(
            {
                "type": "error",
                "message": (
                    "No prepared public EEG. "
                    "Run the NEMAR download and "
                    "preparation scripts. Lucid "
                    "will not generate substitute EEG."
                ),
            }
        )
        await websocket.close()
        return

    try:
        while True:
            trial = replay.next_test_trial()

            await websocket.send_json(
                {
                    "type": "trial_start",
                    "dataset": (
                        trial.dataset_id
                    ),
                    "recording": (
                        trial.recording
                    ),
                    "ground_truth": (
                        trial.ground_truth
                    ),
                    "sfreq": trial.sfreq,
                    "channels": int(
                        trial.data.shape[0]
                    ),
                    "samples": int(
                        trial.data.shape[1]
                    ),
                }
            )

            interval = 1.0 / trial.sfreq
            visible_channels = min(
                8,
                trial.data.shape[0],
            )

            for sample_idx in range(
                trial.data.shape[1]
            ):
                await websocket.send_json(
                    {
                        "type": "sample",
                        "index": sample_idx,
                        "values": (
                            trial.data[
                                :visible_channels,
                                sample_idx,
                            ].tolist()
                        ),
                    }
                )
                await asyncio.sleep(interval)

            result = predictor.predict(
                trial.data,
                source_recording=(
                    trial.recording
                ),
            )
            result = attach_state_prediction(
                result,
                trial.data,
            )

            await websocket.send_json(
                {
                    "type": "prediction",
                    "ground_truth": (
                        trial.ground_truth
                    ),
                    "prediction": (
                        result.model_dump()
                    ),
                }
            )

            if result.status == "ok":
                log_prediction(
                    source_dataset=(
                        result.source_dataset
                    ),
                    source_recording=(
                        result.source_recording
                    ),
                    ground_truth=(
                        trial.ground_truth
                    ),
                    prediction=(
                        result.prediction
                    ),
                    confidence=(
                        result.prediction_confidence
                    ),
                    model_version=(
                        result.model_version
                    ),
                )

            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        return
