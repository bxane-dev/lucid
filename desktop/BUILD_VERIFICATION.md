# Desktop build verification

This file exists to document the installer acceptance gate.

A Lucid desktop package is accepted only when the pull-request installer
workflow succeeds on both Windows and Linux.

The workflow must:

- build the exported Next.js interface;
- freeze the local FastAPI/PyTorch backend with PyInstaller;
- launch the frozen backend and receive HTTP 200 from /health;
- serve the bundled Lucid interface from the frozen backend;
- package the Windows NSIS installer and Linux AppImage;
- confirm the packaged backend executable exists;
- generate a SHA-256 checksum for each distributable.

Passing this gate verifies packaging and startup only. It does not replace
Lucid's separate real-public-EEG model validation.
