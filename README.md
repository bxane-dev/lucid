# Lucid

Lucid is a zero-cost EEG research prototype for classifying limited brain states and EEG-derived intents from public or locally recorded datasets.

> Lucid is **not** a mind-reading system. Predictions are statistical classifications from EEG recordings and must not be presented as literal private thoughts.

## First milestone

```json
{
  "state": "imagined_speech",
  "confidence": 0.91,
  "prediction": "yes",
  "prediction_confidence": 0.74
}
```

## Stack

- Python, PyTorch, MNE-Python, NumPy, SciPy, scikit-learn
- FastAPI + WebSockets
- Next.js + React
- SQLite
- Local development / free Colab or Kaggle training

## Architecture

```text
EEG file -> preprocessing -> windowing -> model -> FastAPI
                                         |
                                         +-> WebSocket simulated live stream -> Next.js UI
```

## Repository

- `backend/` preprocessing, EEGNet, training, inference, SQLite and streaming API
- `frontend/` live Lucid dashboard
- `dataset/` common local dataset format
- `models/` locally trained model artifacts
- `docs/` research and dataset notes

## Quick start

### Backend

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_demo_data.py
uvicorn app.main:app --reload
```

Backend: http://localhost:8000

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend: http://localhost:3000

The UI connects to `ws://localhost:8000/ws/live` by default.

## Dataset format

```text
dataset/
  subject_01/
    eeg.npy
    labels.json
```

`eeg.npy` is shaped `[channels, samples]`. `labels.json` stores sampling frequency, channel names, and labeled time segments.

## Research order

1. REST vs imagined speech
2. Small fixed vocabulary such as YES / NO / LEFT / RIGHT / STOP / GO
3. Session-independent evaluation
4. Broader intent classes only after earlier stages are reliable

No paid API, cloud database, GPU, server, or EEG headset is required for the first prototype.
