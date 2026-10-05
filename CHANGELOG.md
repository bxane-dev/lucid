# Changelog

All notable Lucid desktop releases are documented here.

## v1.0.2 — 2026-10-05

### Fixed
- Prevented silent Electron exits when the bundled backend process cannot be spawned.
- Added a visible startup window while Lucid's local EEG engine initializes.
- Added immediate startup failure reporting with the diagnostics-folder path.
- Disabled GPU acceleration to avoid Electron startup failures caused by problematic graphics drivers.
- Added renderer/process crash logging.

### Validation
- Added packaged Electron launcher self-tests for Windows, Linux, and macOS.
- Release CI now requires the actual packaged desktop executable to start its bundled backend successfully.

## v1.0.1 — 2026-10-05

### Added
- Actual macOS desktop package for Intel/x64 Macs as both DMG and ZIP.
- macOS frozen-backend startup and UI smoke testing in GitHub Actions.
- macOS SHA-256 checksums and electron-builder auto-update metadata.
- Three-platform release packaging: Windows, Linux, and macOS.

### Changed
- Desktop version bumped from 1.0.0 to 1.0.1.
- Release workflow now requires verified artifacts for all three desktop platforms.
- Release notes now include platform-specific signing/runtime caveats.

### Fixed
- PyInstaller packaging now includes PyTorch runtime modules needed by the frozen backend.
- MNE lazy-loader runtime stubs are verified before installer packaging.

## v1.0.0 — 2026-10-05

### Added
- Windows x64 NSIS installer.
- Linux x64 AppImage.
- In-app public EEG dataset manager.
- Real NEMAR download and preparation workflow.
- Participant-held-out train/validation/test splitting.
- Six local model architectures: EEGNet, 1D CNN, CNN+LSTM, Temporal CNN, Transformer, and EEG Conformer.
- In-app model training, benchmarking, model activation, and real recorded EEG replay.
- REST vs IMAGINED_SPEECH path using published resting-baseline recordings.
- Source/model provenance SHA-256 chain.
- GitHub release auto-update metadata.
- Local SQLite prediction history.

### Validation
- Real-data word benchmark and state-path validation results are recorded in `docs/V1_VALIDATION.md`.
- No synthetic EEG, fabricated confidence values, or placeholder metrics are included.
