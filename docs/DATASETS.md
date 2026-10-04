# Public EEG datasets

Lucid is designed around provenance. Every prediction must be traceable to a public recording and a locally trained model.

## nm000113 — 2020 BCI Competition Track 3

Primary dataset for the first working word classifier.

| Field | Value |
|---|---|
| Archive | NEMAR |
| Dataset ID | `nm000113` |
| Version used by default | `v1.0.0` |
| Participants | 15 |
| Channels | 64 |
| Sampling rate | 256 Hz |
| Task | imagined speech |
| Commands | Hello, Help me, Stop, Thank you, Yes |
| Approximate size | 585 MB |
| License | CC-BY-4.0 |
| DOI | 10.82901/nemar.nm000113 |

Official archive:
- https://nemar.org/dataset/nm000113
- https://data.nemar.org/nm000113/v1.0.0/

Lucid downloads bytes directly from the official NEMAR data host. It does not rehost the EEG in this repository.

## on003626 / ds003626 — Inner Speech

Optional second dataset.

| Field | Value |
|---|---|
| Archive | NEMAR mirror of OpenNeuro |
| NEMAR ID | `on003626` |
| OpenNeuro ID | `ds003626` |
| Participants | 10 |
| Channels | 136 acquisition channels |
| Sessions | 3 |
| Paradigms | inner speech, pronounced speech, visualized condition |
| Commands | Up, Down, Right, Left |
| Approximate size | 24.6 GB |
| NEMAR license | CC0 |
| NEMAR DOI | 10.82901/nemar.on003626 |
| OpenNeuro DOI | 10.18112/openneuro.ds003626.v2.0.0 |

Publication: N. Nieto et al., "Thinking out loud, an open-access EEG-based BCI dataset for inner speech recognition", Scientific Data 9, 52 (2022).

## Dataset policy

1. No synthetic EEG is used for training, evaluation, screenshots, or demo predictions.
2. Ground-truth labels are always visually distinct from model predictions.
3. Dataset labels may identify the recorded task, but must not be shown as a neural prediction.
4. Confidence values only come from a trained model's output probabilities.
5. If a model is unavailable, Lucid says so rather than producing fallback predictions.
