# Lucid v1.0 validation evidence

This file records measured results produced by Lucid's real-public-data validation workflows. No metric below is a placeholder.

## on003626 real inner-speech + resting-baseline validation

- Workflow run: `37343736417`
- State-path job: `111876992454` — **passed**
- Participants: `sub-01`, `sub-02`, `sub-03`
- Shared source provenance SHA-256: `f5235d8a8bada939eb0ae8af5832c8f3b92cebfe793321839f1e814c3f20a321`
- Word trials: **620**
- State trials: **683**
- State class counts: **620 imagined_speech**, **63 rest**
- Channels: **128**
- Window samples: **256** at Lucid's 128 Hz target rate
- State labels: `imagined_speech`, `rest`
- Word labels: `down`, `left`, `right`, `up`

### on003626 word model smoke validation

- Architecture: EEGNet
- Epochs requested: 3
- Best epoch: 3
- Best validation balanced accuracy: **0.2388888889**
- Held-out test accuracy: **0.1958333333**
- Held-out test balanced accuracy: **0.1958333333**
- Checkpoint SHA-256: `88680efc6eaf85239a59a7726814569fa280ff8022c6e4371eb8048704e868fd`

These numbers are a short three-epoch end-to-end validation of the word path, not a claim of strong decoding performance.

### on003626 REST vs IMAGINED_SPEECH state model

- Architecture: EEGNet
- Epochs requested: 3
- Best epoch: 3
- Best validation balanced accuracy: **0.4876984127**
- Held-out test accuracy: **0.7241379310**
- Held-out test balanced accuracy: **0.6979166667**
- Held-out imagined-speech recall: **0.7291666667** (240 trials)
- Held-out rest recall: **0.6666666667** (21 trials)
- Checkpoint SHA-256: `0e41da9cdc59a1a1468383988458e5b58ee2306fc6612fa8a2b170e295336bba`

The state archive uses distinct non-overlapping windows from the dataset's published resting baseline. It does not manufacture rest examples.

## nm000113 all-subject six-architecture benchmark

Pending completion of the active all-15-participant benchmark. The final winner and all six measured validation/test scores will be inserted here from the workflow output.

## Software / packaging gates

- current CI: pending final-head run
- current Windows/Linux installer verification: pending final-head run

## Selection method

- participant-held-out train / validation / test;
- architecture selection by validation balanced accuracy only;
- test metrics are reported after checkpoint selection and do not choose the winner;
- prepared EEG and checkpoints must carry matching source provenance;
- no synthetic EEG, fabricated confidence, or pre-filled metric is accepted.
