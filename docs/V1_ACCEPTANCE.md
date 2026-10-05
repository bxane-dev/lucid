# Lucid v1.0 acceptance criteria

Lucid v1.0 is accepted only when the following gates are satisfied.

## 1. Real public data only

- Dataset downloads originate from Lucid's fixed NEMAR registry.
- Prepared archives carry a deterministic source-manifest SHA-256.
- Incomplete `.part` downloads are never treated as ready data.
- Synthetic EEG and fabricated predictions are not used.

## 2. Reproducible preparation

- Participant-held-out train / validation / test splits are deterministic.
- No participant appears in more than one split.
- The v1 preprocessing profile is fixed and reported by the API.
- REST examples come from actual published resting-baseline recordings.

## 3. Model selection

- All six architectures can train through the same pipeline.
- Architecture ranking uses validation balanced accuracy only.
- Test metrics are reported after checkpoint selection and are not used to rank.
- Models without source provenance are rejected.
- A model cannot be activated against a prepared archive with different provenance.

## 4. Word and state paths

- `nm000113` validates the five-class imagined-speech word path.
- `on003626` validates both direction-word classification and REST vs IMAGINED_SPEECH classification from genuine derivatives.
- Word and state checkpoints must reference the same real-source provenance when they are used together during replay.

## 5. Desktop workflow

A normal installed user can:

1. download supported public EEG;
2. prepare it;
3. inspect preparation metadata and provenance;
4. train one architecture or benchmark all six;
5. automatically activate the validation-selected winner;
6. switch prepared datasets and provenance-compatible models;
7. replay held-out real EEG sample by sample;
8. see ground truth separately from neural predictions;
9. view state, word, confidence, alternatives, metrics and model provenance.

## 6. Failure handling

- Corrupted prepared archives fail closed.
- Corrupted settings fall back to safe defaults.
- Unsupported dataset IDs are rejected before download.
- Interrupted downloads remain `.part` files.
- Only one dataset/training operation is run at a time.
- Auto-update failure does not prevent local EEG analysis.

## 7. Desktop packaging

Windows and Linux builds must each:

- export the frontend;
- freeze the local Python backend;
- start the frozen backend;
- return HTTP 200 from `/health`;
- serve the bundled frontend;
- contain the expected backend executable;
- produce installer/AppImage update metadata;
- produce SHA-256 installer checksums.

## 8. Release evidence

Before publishing `v1.0.0`:

- normal CI must pass;
- desktop installer verification must pass;
- the full all-subject `nm000113` six-model benchmark must complete;
- the real `on003626` word + REST/IMAGINED_SPEECH path must complete;
- actual metrics and source provenance hashes must be recorded in `docs/V1_VALIDATION.md`.

No minimum accuracy is invented as a release criterion. Lucid reports the measured held-out results exactly as produced by real training.
