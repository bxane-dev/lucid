# Lucid

Lucid is a zero-cost EEG research prototype for classifying limited brain states and imagined-speech commands from **real public EEG recordings**.

Lucid never fabricates EEG, confidence scores, predictions, or model results. If public EEG has not been downloaded, prepared, and a model has not been trained, the API and UI report that the relevant data or model is unavailable.

> Lucid is not a mind-reading system. It performs statistical classification of constrained EEG tasks from research datasets.

## Public data

Lucid currently supports two research paths.

### Lightweight word-decoding path — NEMAR `nm000113`

BIDS conversion of the 2020 BCI Competition Track 3 imagined-speech dataset:

- 15 participants
- 64 EEG channels
- 256 Hz
- commands: `Hello`, `Help me`, `Stop`, `Thank you`, `Yes`
- about 585 MB
- CC-BY-4.0
- DOI: `10.82901/nemar.nm000113`

### Full first-milestone path — `on003626` / OpenNeuro `ds003626`

Nieto et al. Inner Speech dataset:

- 10 participants
- three sessions
- inner speech, pronounced speech, and visualized conditions
- direction classes: Up, Down, Right, Left
- published derivative EEG epochs
- published derivative baseline/rest epochs
- NEMAR DOI: `10.82901/nemar.on003626`
- OpenNeuro DOI: `10.18112/openneuro.ds003626.v2.0.0`

Lucid uses the published derivative event table to select condition `1` (inner speech), direction labels for the word model, and the matching published baseline epochs for the REST class. It does **not** manufacture rest windows.

See [docs/DATASETS.md](docs/DATASETS.md).

## Architecture

```text
public EEG recordings
       |
       v
MNE loader
       |
metadata-aware notch -> band-pass where applicable -> resample -> rejection -> normalization
       |
       v
participant-held-out train / validation / test
       |
       +------ EEGNet word classifier
       |
       +------ EEGNet state classifier
       |
       v
FastAPI + sample-by-sample WebSocket replay
       |
       v
Next.js Lucid dashboard
```

The live simulator replays actual recorded EEG samples in temporal order. Dataset ground truth is displayed separately from model output.

## Setup

### 1. Backend

```bash
cd backend
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Option A — smaller real word classifier

Download all `nm000113` participants:

```bash
python scripts/download_nemar.py --dataset nm000113 --subjects all
python scripts/prepare_dataset.py --dataset nm000113
python scripts/train.py --epochs 25
```

This trains the five-class imagined-speech word model. It does not pretend to provide a separately validated REST-vs-speech state confidence.

## Option B — complete state + word milestone

Download the **published derivatives only** from the larger Nieto dataset. Raw BDF recordings are not required for this path:

```bash
python scripts/download_nemar.py --dataset on003626 --subjects all --derivatives-only
python scripts/prepare_inner_speech.py
```

Train the direction-word model:

```bash
python scripts/train.py ^
  --prepared ../dataset/prepared/on003626_words.npz ^
  --output ../models/eegnet_words.pt ^
  --task words ^
  --epochs 25
```

Train the REST-vs-IMAGINED-SPEECH state model:

```bash
python scripts/train.py ^
  --prepared ../dataset/prepared/on003626_state.npz ^
  --output ../models/eegnet_state.pt ^
  --task state ^
  --epochs 25
```

The commands above use Windows line continuation. On macOS/Linux replace `^` with `\`.

To make the API replay the matching on003626 word archive, set:

Windows PowerShell:

```powershell
$env:LUCID_DATASET_ID="on003626"
uvicorn app.main:app --reload
```

macOS/Linux:

```bash
export LUCID_DATASET_ID=on003626
uvicorn app.main:app --reload
```

When both trained checkpoints are compatible with the replay window, Lucid returns a model-derived state, state confidence, word prediction, and word confidence. The repository contains **no pre-filled result percentages**; metrics are written only after real training.

## Web UI

```bash
cd ../frontend
npm install
npm run dev
```

Open <http://localhost:3000>.

The interface shows:

- real EEG replay traces;
- source dataset and recording;
- dataset ground-truth annotation;
- neural word prediction;
- neural state prediction when a compatible state model exists;
- real model probabilities only after inference;
- clear unavailable states before training.

## Evaluation

The preparation code splits by participant, not by random trial. Subjects held out for validation and test are therefore unseen during training.

Each trained checkpoint stores:

- source dataset ID;
- task;
- labels;
- preprocessing metadata;
- test accuracy;
- balanced accuracy;
- per-class precision/recall/F1;
- confusion matrix.

Metrics are also written beside the checkpoint as `*.metrics.json`.

## Repository layout

```text
backend/
  app/
    dataset.py          nm000113 BIDS preparation
    inner_speech.py     on003626 derivative state + word preparation
    preprocess.py       signal preprocessing
    model.py            EEGNet
    inference.py        checkpoint-only inference
    replay.py           real public EEG replay
    main.py             FastAPI + WebSocket
  scripts/
    download_nemar.py
    prepare_dataset.py
    prepare_inner_speech.py
    train.py
    predict_recording.py
frontend/
  app/                  Next.js Lucid dashboard
docs/
  DATASETS.md
  METHODOLOGY.md
dataset/                local public downloads/derived arrays; gitignored
models/                 locally trained checkpoints; gitignored
runtime/                SQLite prediction log; gitignored
```

## Zero-cost rule

The prototype requires no paid API, server, GPU, cloud database, commercial dataset, or EEG headset. CPU training works locally; free Colab/Kaggle compute can be used without changing the data pipeline.

## License

Lucid source code is MIT licensed. Public EEG retains its upstream license and attribution requirements.
