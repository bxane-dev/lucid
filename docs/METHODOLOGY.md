# Methodology

## Scope

Lucid is a constrained EEG classifier, not a private-thought decoder.

## Preprocessing

Default pipeline:

1. load the EEG recording with MNE-Python;
2. keep EEG channels;
3. apply a notch filter only when a power-line frequency is supplied by recording metadata or explicit configuration;
4. band-pass 1–40 Hz;
5. re-reference to the average EEG reference;
6. resample to 128 Hz;
7. extract fixed-length windows from BIDS event annotations;
8. reject non-finite or extreme-amplitude windows;
9. z-normalize every channel inside each trial.

The default NEMAR `nm000113` description reports line-noise frequency as unspecified, so Lucid does not silently assume 50 Hz or 60 Hz for that dataset.

Lucid also does not automatically run ICA and claim that all artifacts were removed. ICA requires dataset-specific component review or reliable auxiliary-channel criteria. The first pipeline therefore uses conservative window rejection.

## Evaluation

Lucid groups by participant before splitting:

- training: approximately 70% of participants;
- validation: approximately 15%;
- test: remaining participants.

The split is deterministic from random seed 42.

Metrics include:

- accuracy;
- balanced accuracy;
- per-class precision, recall, and F1;
- confusion matrix.

## Live replay

The WebSocket replay reads a prepared trial cut from a public recording, sends its samples in temporal order, and runs the locally trained model on that same recorded trial.

The event annotation is sent separately as `ground_truth`.

## State classification

The default `nm000113` pipeline trains word classes. It does not have a separate validated REST-vs-IMAGINED-SPEECH state model in Lucid v0.1.

Accordingly, `state_confidence` remains `null` unless a separately trained state model exists.

This prevents a dataset annotation from being misrepresented as model certainty.
