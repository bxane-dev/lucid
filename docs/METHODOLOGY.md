# Methodology

## Scope

Lucid is a constrained EEG classifier, not a private-thought decoder.

## Preprocessing

Default pipeline:

1. load the EEG recording with MNE-Python;
2. keep EEG channels;
3. notch-filter 50 Hz only when the source sampling rate supports that frequency;
4. band-pass 1–40 Hz;
5. resample to 128 Hz;
6. extract fixed-length windows from BIDS event annotations;
7. reject non-finite or extreme-amplitude windows;
8. z-normalize every channel inside each trial.

Lucid does not automatically run ICA and pretend that all artifacts were removed. ICA needs dataset-specific component review or reliable auxiliary-channel criteria. The first pipeline therefore uses conservative window rejection.

## Evaluation

Lucid groups by participant before splitting:

- training: approximately 70% of participants;
- validation: approximately 15%;
- test: remaining participants.

The split is deterministic from random seed 42.

Metrics include accuracy, balanced accuracy, per-class precision/recall/F1, and confusion matrix.

## Live replay

The WebSocket replay reads a prepared trial cut from a public recording, sends its samples in temporal order, and runs the locally trained model on that same recorded trial. The event annotation is sent separately as `ground_truth`.

## State classification

The default `nm000113` pipeline trains word classes. It does not have a separate validated REST-vs-IMAGINED-SPEECH state model in Lucid v0.1.

Accordingly, `state_confidence` remains `null` unless a separately trained state model exists. This prevents an annotation from being misrepresented as model certainty.
