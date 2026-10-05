# Lucid

Lucid v1 is a zero-cost EEG research application for classifying limited brain states and imagined-speech commands from **real public EEG recordings**.

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
- published resting baseline epochs
- NEMAR DOI: `10.82901/nemar.on003626`
- OpenNeuro DOI: `10.18112/openneuro.ds003626.v2.0.0`

For REST-vs-IMAGINED-SPEECH, Lucid cuts non-overlapping 2-second windows from the actual published resting baseline recording. It does **not** duplicate baseline samples or manufacture rest windows.

See [docs/DATASETS.md](docs/DATASETS.md) and [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Architecture

```text
public EEG recordings
       |
       v
MNE loader
       |
metadata-aware preprocessing -> 2 s windows -> normalization
       |
       v
participant-held-out train / validation / test
       |
       +------ word classifier
       |
       +------ state classifier
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
  --architecture eegnet ^
  --epochs 25
```

Train the REST-vs-IMAGINED-SPEECH state model:

```bash
python scripts/train.py ^
  --prepared ../dataset/prepared/on003626_state.npz ^
  --output ../models/eegnet_state.pt ^
  --task state ^
  --architecture eegnet ^
  --epochs 25
```

The commands above use Windows line continuation. On macOS/Linux replace `^` with `\`.

To make the API replay the matching `on003626` word archive, set:

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

## Model comparison

Lucid includes six free local architectures:

- EEGNet
- 1D CNN
- CNN + LSTM
- Temporal CNN
- Transformer
- compact EEG-Conformer-style network

Run all architectures against the **same** prepared public recording split:

```bash
python scripts/benchmark_models.py ^
  --prepared ../dataset/prepared/on003626_words.npz ^
  --epochs 15
```

The benchmark ranks architectures by **validation balanced accuracy**. Test metrics are reported but are not used to choose the winner.

Use `--architectures eegnet cnn1d temporal_cnn` to benchmark a subset.

## Lucid v1 workbench

The desktop application no longer requires terminal commands for the normal
workflow. From the Lucid interface you can:

1. download a three-participant starter subset or all supported participants;
2. prepare the public recordings with the fixed `lucid-standard-v1` profile;
3. inspect trial counts, subject-held-out splits, channel/sample metadata, and
   provenance;
4. train one of six local architectures or benchmark all six;
5. activate only a checkpoint whose provenance matches the prepared EEG;
6. switch between prepared datasets;
7. replay real held-out EEG sample by sample;
8. view separate dataset ground truth, word/intent output, state output, and
   model probabilities.

For development, the Next.js interface still runs with:

```bash
cd ../frontend
npm install
npm run dev
```

Open <http://localhost:3000>.

Nothing is filled in before the corresponding real dataset/model exists.

## Provenance and anti-fabrication checks

Every prepared archive now gets a sidecar manifest such as:

```text
dataset/prepared/nm000113_words.npz
dataset/prepared/nm000113_words.npz.manifest.json
```

The manifest contains the SHA-256 and byte size of every public EEG/event
source file used to build the archive. The manifest itself has a deterministic
SHA-256 identifier. That identifier is embedded into the prepared archive,
the trained checkpoint, the metrics JSON, API responses, and the live replay.

Lucid refuses to load legacy checkpoints without provenance and refuses to
score a replay when the checkpoint hash does not match the prepared EEG hash.

The dashboard shows the abbreviated source SHA-256 and whether the loaded
word model matches it. The real-data GitHub Actions smoke test independently
checks the manifest -> prepared archive -> checkpoint -> metrics chain.

## Evaluation

The preparation code splits by participant, not by random trial. Subjects held out for validation and test are therefore unseen during training.

Training uses inverse-frequency class weights calculated only from the training split. It does not synthesize or duplicate EEG for balancing.

Each trained checkpoint stores:

- source dataset ID
- task and architecture
- labels and preprocessing metadata
- best validation balanced accuracy
- test accuracy and test balanced accuracy
- per-class precision/recall/F1
- confusion matrix

Metrics are written beside the checkpoint as `*.metrics.json`.

## Repository layout

```text
backend/
  app/
    dataset.py          nm000113 BIDS preparation
    inner_speech.py     on003626 real state + word preparation
    preprocess.py       signal preprocessing
    model.py            six PyTorch EEG architectures
    inference.py        checkpoint-only inference
    replay.py           real public EEG replay
    main.py             FastAPI + WebSocket
  scripts/
    download_nemar.py
    prepare_dataset.py
    prepare_inner_speech.py
    train.py
    benchmark_models.py
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

## Desktop installer

Lucid can be packaged as a normal desktop application. The installed app starts
its own local FastAPI/PyTorch backend and opens the Lucid interface in an
Electron window. Users do not need to run `uvicorn` or `npm run dev`.

### Windows

GitHub Actions builds:

```text
Lucid-Setup-1.0.0-x64.exe
```

The NSIS installer supports a custom install directory and creates Start Menu
and desktop shortcuts.

### Linux

The same workflow builds:

```text
Lucid-1.0.0-x64.AppImage
```

### Installed data location

The application binary is installed separately from research data. Public EEG,
prepared archives, trained models, and the local SQLite database are stored in
the operating system's per-user Lucid application-data directory. Uninstalling
Lucid does not delete research data by default.

No EEG dataset or pretrained result is silently bundled into the installer.
Lucid still follows the real-public-data and provenance rules documented below.

### Rebuild installers

Open the repository's **Actions** tab and run the `desktop-installers`
workflow, or push a change affecting `desktop/`, `backend/`, or
`frontend/`.

The workflow builds a self-contained Python backend with PyInstaller, exports
the Next.js frontend, packages them with Electron, smoke-tests the frozen
backend/UI, calculates SHA-256 checksums, and uploads the installer files.

Release builds also publish electron-builder update metadata. Installed Lucid
checks the public `bxane-dev/lucid` GitHub Releases feed for newer versions.
An update failure never blocks local EEG analysis.

> Windows packages are currently unsigned. Windows SmartScreen may therefore
> identify the publisher as unknown until a code-signing certificate is added.

## Zero-cost rule

The prototype requires no paid API, server, GPU, cloud database, commercial dataset, or EEG headset. CPU training works locally; free Colab/Kaggle compute can be used without changing the data pipeline.

## License

Lucid source code is MIT licensed. Public EEG retains its upstream license and attribution requirements.
