# Lucid

Lucid is a zero-cost EEG research prototype for classifying limited brain states and imagined-speech commands from **real public EEG recordings**.

Lucid never fabricates EEG, confidence scores, predictions, or model results. If public EEG has not been downloaded, prepared, and a model has not been trained, the API and UI report that the model/data are unavailable.

> Lucid is not a mind-reading system. It performs statistical classification of constrained EEG tasks from research datasets. A language model must never be presented as having decoded content that the EEG model did not predict.

## Default public dataset

The first supported source is NEMAR `nm000113`, the BIDS conversion of the 2020 BCI Competition Track 3 imagined-speech dataset.

- 15 participants
- 64 EEG channels
- 256 Hz
- five recorded imagined-speech tasks: `Hello`, `Help me`, `Stop`, `Thank you`, `Yes`
- about 585 MB
- CC-BY-4.0
- DOI: `10.82901/nemar.nm000113`

A second, larger source is NEMAR/OpenNeuro `on003626` / `ds003626`, the Nieto et al. Inner Speech dataset. It has 10 participants, 136 channels, four directional command classes and three paradigms. It is optional because the full dataset is about 24.6 GB.

See [docs/DATASETS.md](docs/DATASETS.md).

## Architecture

```text
public BIDS EEG
      |
      v
MNE loader
      |
notch -> 1-40 Hz band-pass -> resample -> window rejection -> per-channel normalization
      |
      v
subject-held-out train / validation / test split
      |
      v
EEGNet (PyTorch)
      |
      +----> JSON prediction
      |
      +----> FastAPI WebSocket
                 |
                 v
          Next.js dashboard
```

The live simulator replays samples from a real downloaded EEG trial one sample at a time. Ground-truth annotations are displayed separately from neural-network output.

## Run locally

### 1. Backend

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Download public EEG

Download the default dataset from NEMAR:

```bash
python scripts/download_nemar.py --dataset nm000113 --subjects all
```

For a smaller pipeline check using one real participant:

```bash
python scripts/download_nemar.py --dataset nm000113 --subjects sub-01
```

No generated or synthetic EEG is used.

### 3. Prepare windows

```bash
python scripts/prepare_dataset.py --dataset nm000113
```

This creates a local prepared file under `dataset/prepared/`. Raw public EEG remains under `dataset/public/`. Both directories are gitignored because the research recordings are much larger than source code.

### 4. Train

```bash
python scripts/train.py --epochs 25
```

The default split holds out entire participants for validation and test. This is intentionally stricter than randomly splitting trials from the same person.

### 5. API

```bash
uvicorn app.main:app --reload
```

Open <http://localhost:8000/api/status> to see whether real data and a trained model are available.

### 6. Web UI

```bash
cd ../frontend
npm install
npm run dev
```

Open <http://localhost:3000>.

## API result

When a trained model exists, a replay prediction has this shape:

```json
{
  "status": "ok",
  "state": "imagined_speech",
  "state_confidence": null,
  "prediction": "yes",
  "prediction_confidence": 0.74,
  "alternatives": [
    {"label": "stop", "confidence": 0.11}
  ],
  "source_dataset": "nm000113",
  "model_version": "eegnet_words_v1"
}
```

`state_confidence` remains `null` until a separately trained state model exists. Lucid does not turn the dataset annotation into a fake model confidence.

## Repository layout

```text
backend/
  app/                 FastAPI, preprocessing, model, inference, streaming
  scripts/             download, prepare, train, predict
frontend/
  app/                 Next.js dashboard
docs/
  DATASETS.md           provenance and licenses
  METHODOLOGY.md        evaluation rules and limitations
dataset/                local only; public EEG downloads are not committed
models/                 local model checkpoints; not committed
```

## Zero-cost rule

The prototype requires no paid API, server, GPU, cloud database, commercial dataset, or EEG headset. CPU training works locally; Colab/Kaggle free compute may be used later.

## License

Lucid source code is MIT licensed. Public EEG keeps its original dataset license and attribution requirements.
