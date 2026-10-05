# Lucid v1.0.2

Lucid v1.0.2 is a desktop startup reliability hotfix.

## Fixed

- Fixed a launcher path where failure to spawn the bundled EEG backend could terminate Electron before any useful error appeared.
- Lucid now opens a visible startup window immediately while its local backend initializes.
- Backend spawn failures and early exits are logged and shown as explicit startup errors.
- Startup errors now include the local diagnostics-folder path.
- Hardware acceleration is disabled because Lucid's UI does not need it; this avoids startup failures on systems with problematic Chromium/Electron GPU drivers.
- Renderer-load and renderer-process failures are logged.

## Packaging validation

The release workflow now launches the **actual packaged Electron application** in self-test mode on Windows, Linux, and macOS. A healthy frozen backend alone is no longer sufficient for release.

## Platforms

- Windows x64 NSIS installer
- Linux x64 AppImage
- macOS Intel/x64 DMG and ZIP

The macOS build remains unsigned/not notarized and Intel/x64. Apple Silicon users may use Rosetta 2.

## Scientific behavior

No EEG/model behavior is changed by this hotfix. Lucid still uses real public EEG only, and does not fabricate signals, predictions, confidence values, or benchmark metrics.
