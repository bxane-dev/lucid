# Methodology

## Scope

Lucid is a constrained EEG classifier, not a private-thought decoder.

## Public-recording rule

No synthetic EEG is used for training, validation, testing, live replay, screenshots, or fallback predictions. Missing inputs produce an unavailable state.

Dataset annotations are shown separately from neural-network output and never converted into model confidence.

## Preprocessing

For the `nm000113` BIDS word path:

1. load EEG with MNE-Python;
2. keep EEG channels;
3. apply a notch filter only when recording metadata or explicit configuration supplies the line frequency;
4. band-pass 1–40 Hz;
5. average-reference;
6. resample to 128 Hz;
7. extract fixed 2-second windows from BIDS task annotations;
8. reject non-finite or extreme-amplitude windows;
9. z-normalize each channel inside each trial.

The NEMAR `nm000113` description leaves line-noise frequency unspecified, so Lucid does not silently assume 50 Hz or 60 Hz.

For the Nieto `on003626` derivative path, Lucid reads the authors' published epoched EEG and resting baseline derivatives. Speech windows begin at task time zero. REST examples are non-overlapping 2-second cuts of the actual long baseline epoch recorded in each session. Lucid does not duplicate a baseline segment to create artificial REST trials.

The first reproducible pipeline does not automatically run a new ICA stage on top of published derivatives and claim complete artifact removal.

## Evaluation

Lucid groups by participant before splitting:

- training: approximately 70% of participants;
- validation: approximately 15%;
- test: remaining participants.

The split is deterministic from random seed 42.

Architectures are selected using **validation balanced accuracy**. The test split is evaluated only after the best validation checkpoint is fixed. Model comparison reports test metrics for transparency but does not sort/select architectures using test performance.

When training classes are imbalanced, Lucid uses inverse-frequency cross-entropy weights computed from the training split. It does not invent, duplicate, or synthesize EEG to balance classes.

Reported metrics include accuracy, balanced accuracy, per-class precision/recall/F1, and a confusion matrix.

## Live replay

The WebSocket replay reads a held-out prepared trial cut from a public recording, sends samples in temporal order, and runs locally trained checkpoints on that same real EEG window.

The source recording and dataset annotation are transmitted separately.

## State classification

The `nm000113` lightweight path supplies imagined-speech word labels but no validated REST class, so Lucid leaves state confidence unavailable there.

The `on003626` state path uses the authors' published resting baseline recordings for REST and published inner-speech epochs for IMAGINED_SPEECH. A state value appears in the API only when a separately trained compatible state checkpoint actually produces it.
