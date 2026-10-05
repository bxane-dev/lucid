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

- Workflow run: `37343081991`
- Word-benchmark job: `111874781963` — **passed**
- Participants: **15** (`sub-01` through `sub-15`)
- Prepared trials: **5,127**
- Channels: **64**
- Samples per window: **256**
- Train trials: **3,697**
- Validation trials: **762**
- Test trials: **668**
- Source provenance SHA-256: `e7b4d1fb43de4b33ce27909b391886ad87eea54b81bc3345dcd83acadf41c0dd`
- Epochs per architecture: **5**
- Selection metric: **validation balanced accuracy**
- Validation-selected winner: **EEG Conformer**

| Rank | Architecture | Best epoch | Validation balanced accuracy | Held-out test balanced accuracy |
| ---: | --- | ---: | ---: | ---: |
| 1 | EEG Conformer | 4 | **0.2229575571** | 0.1781203008 |
| 2 | Temporal CNN | 3 | 0.2179322396 | **0.2017473917** |
| 3 | CNN + LSTM | 1 | 0.2178306114 | 0.1867325290 |
| 4 | Transformer | 5 | 0.2131472273 | 0.1941761380 |
| 5 | EEGNet | 3 | 0.2054225444 | 0.1760453459 |
| 6 | 1D CNN | 4 | 0.1996824994 | 0.1793670222 |

The winner is selected strictly from validation balanced accuracy. Temporal CNN
happened to have the highest test balanced accuracy in this short run, but the
test set is intentionally not used to change the selected winner.

Winner checkpoint SHA-256:
`397df4b23bd0fd7c223203e12da2b604ccfcf59b341937050e68ab14f0eed0ff`.

The five-class chance baseline is 0.20. These short five-epoch results therefore
show that the current cross-participant word-decoding benchmark is weak and
should not be described as reliable thought or speech decoding. Lucid exposes
the measured metrics instead of concealing that limitation.

## Software / packaging gates

- Final pre-release CI run: `37349003342` — **passed**
  - backend compile/tests: passed
  - frontend production build: passed
- Final pre-release desktop installer run: `37349003257` — **passed**
  - Windows x64 frozen backend startup/UI smoke test: passed
  - Windows NSIS installer build/content/update-metadata/checksum verification: passed
  - Linux x64 frozen backend startup/UI smoke test: passed
  - Linux AppImage build/content/update-metadata/checksum verification: passed
- Windows installer artifact digest: `sha256:481b6c1b5da8c483d340e82889ac16a4c8a6bbd850778038c0cc2529899c57f8`
- Linux AppImage artifact digest: `sha256:537cbe0af3f5446c76f82ecb3dcb407efb81909b0b8a316849bccfcbff9daf32`

## Selection method

- participant-held-out train / validation / test;
- architecture selection by validation balanced accuracy only;
- test metrics are reported after checkpoint selection and do not choose the winner;
- prepared EEG and checkpoints must carry matching source provenance;
- no synthetic EEG, fabricated confidence, or pre-filled metric is accepted.
