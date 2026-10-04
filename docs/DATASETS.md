# Public EEG datasets

Lucid is designed around provenance. Every model result must be traceable to a public recording and a locally trained checkpoint.

## nm000113 — 2020 BCI Competition Track 3

Primary lightweight dataset for the five-class imagined-speech classifier.

| Field | Value |
|---|---|
| Archive | NEMAR |
| Dataset ID | `nm000113` |
| Version used | `v1.0.0` |
| Participants | 15 |
| Channels | 64 |
| Sampling rate | 256 Hz |
| Task | imagined speech |
| Commands | Hello, Help me, Stop, Thank you, Yes |
| Approximate archive size | 585 MB |
| License | CC-BY-4.0 |
| DOI | 10.82901/nemar.nm000113 |

Official archive:

- https://nemar.org/dataset/nm000113
- https://data.nemar.org/nm000113/v1.0.0/

The NEMAR description states that the original epoched competition recordings were converted to BIDS EDF and annotated with trial onset, duration, and command labels.

Lucid downloads the bytes directly from the official NEMAR data host and does not commit or rehost them.

## on003626 / ds003626 — Inner Speech

Source for the complete state + direction-word milestone.

| Field | Value |
|---|---|
| Archive | NEMAR mirror of OpenNeuro |
| NEMAR ID | `on003626` |
| OpenNeuro ID | `ds003626` |
| Participants | 10 |
| Sessions | 3 |
| Acquisition channels | 136 |
| Paradigms | pronounced, inner speech, visualized |
| Direction classes | Up, Down, Right, Left |
| Approximate full archive size | 24.6 GB |
| NEMAR license | CC0 |
| NEMAR DOI | 10.82901/nemar.on003626 |
| OpenNeuro DOI | 10.18112/openneuro.ds003626.v2.0.0 |

Publication:

N. Nieto, V. Peterson, H. L. Rufiner, J. E. Kamienkowski, R. Spies, "Thinking out loud, an open-access EEG-based BCI dataset for inner speech recognition", Scientific Data 9, 52 (2022).

The authors' public processing repository documents derivative files for each subject/session:

- `*_eeg-epo.fif`
- `*_baseline-epo.fif`
- `*_events.dat`

Their exploration documentation defines the event columns as:

- column 1: direction class — 0 Up, 1 Down, 2 Right, 3 Left;
- column 2: condition — 0 Pronounced, 1 Inner Speech, 2 Visualized;
- column 3: session.

Lucid's `prepare_inner_speech.py` therefore uses only rows with condition `1` for the inner-speech examples and uses the **matching published baseline epochs** as REST examples. No artificial REST samples are generated.

The `--derivatives-only` downloader mode avoids downloading the raw BDF folders when only the published epoch derivatives are needed.

## Dataset policy

1. No synthetic EEG is used for training, evaluation, screenshots, or demo predictions.
2. Ground-truth labels are always visually distinct from model predictions.
3. Dataset annotations are never converted into fake model confidence.
4. Confidence values only come from a trained model's output probabilities.
5. Missing data or missing checkpoints produce an unavailable state rather than fallback predictions.
6. Participant-held-out evaluation is the default.
7. Public EEG remains outside git and keeps its upstream license/attribution.
