# Lucid v1.0.0

Lucid v1.0.0 is the first complete desktop research release.

## What is included

- Windows x64 NSIS installer.
- Linux x64 AppImage.
- In-app verified public EEG dataset manager.
- Real NEMAR download and preparation pipeline.
- Fixed reproducible Lucid v1 preprocessing profile.
- Participant-held-out train / validation / test splits.
- Six local EEG architectures: EEGNet, 1D CNN, CNN+LSTM, Temporal CNN, Transformer, and EEG Conformer.
- In-app model training and six-model benchmarking.
- REST vs IMAGINED_SPEECH classification from published resting baseline data where the dataset supports it.
- Provenance-safe model registry and activation.
- Sample-by-sample replay of real recorded EEG.
- GitHub release auto-update metadata.
- Local SQLite prediction history.
- SHA-256 installer checksums and source/model provenance.

## Scientific scope

Lucid is a constrained EEG classification research application. It is not a
mind-reading system and does not claim unrestricted private-thought decoding.
Metrics shown by Lucid are created only by actual model training and held-out
evaluation on prepared public recordings.

No synthetic EEG, fabricated confidence value, or pre-filled benchmark result
is included.

## Public datasets

Lucid currently supports NEMAR `nm000113` and `on003626`. Dataset files keep
their upstream licenses and attribution requirements.

## Windows signing

The v1 Windows installer is not Authenticode-signed. Windows SmartScreen may
therefore display an unknown-publisher warning. Installer SHA-256 checksums are
published with the release.
