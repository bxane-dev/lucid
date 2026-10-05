# Lucid v1.0.1

Lucid v1.0.1 is the first three-platform desktop release.

## Downloads

- **Windows x64:** NSIS installer (`Lucid-Setup-1.0.1-x64.exe`)
- **Linux x64:** AppImage (`Lucid-1.0.1-x64.AppImage`)
- **macOS Intel/x64:** DMG and ZIP (`Lucid-1.0.1-macOS-x64.dmg` / `Lucid-1.0.1-macOS-x64.zip`)

The macOS build is currently Intel/x64. Apple Silicon Macs can run it using Rosetta 2. A native arm64 build is not claimed in this release.

## Changelog

### Added
- Actual macOS desktop application packaging.
- macOS frozen-backend, `/health`, and bundled-UI smoke tests.
- macOS DMG/ZIP checksums and update metadata.
- Windows + Linux + macOS assets published together from one release workflow.

### Fixed
- Bundled PyTorch runtime modules required by the frozen backend.
- MNE lazy-loader runtime packaging checks.

### Existing v1 functionality
- In-app verified public EEG dataset manager.
- Real NEMAR download and preprocessing pipeline.
- Six local EEG model architectures.
- In-app training and six-model benchmarking.
- Provenance-safe model activation.
- Real recorded EEG replay.
- REST vs IMAGINED_SPEECH classification where supported by genuine baseline data.
- Local SQLite history and release auto-update metadata.

## Scientific scope

Lucid is a constrained EEG classification research application. It is not a mind-reading system and does not claim unrestricted private-thought decoding.

All published validation metrics come from actual held-out public EEG evaluation. Lucid does not fill missing results with synthetic data or placeholder accuracy values.

## Signing

- **Windows:** unsigned; SmartScreen may show Unknown publisher.
- **macOS:** unsigned/not notarized; Gatekeeper may require the user to explicitly allow the app.
- **Linux:** AppImage is unsigned.

SHA-256 checksum files are included in the release assets.

See `CHANGELOG.md` for the full release history and `docs/V1_VALIDATION.md` for measured research validation.
