# Lucid v1.0 validation evidence

This file records only measured results produced by Lucid's real-public-data validation workflows.

## Release status

Pending final workflow completion. Do not replace pending fields with estimates.

## Required evidence

- all-subject `nm000113` six-architecture benchmark: pending
- `on003626` real word model validation: pending
- `on003626` real REST vs IMAGINED_SPEECH validation: pending
- current CI: pending
- current Windows/Linux installer verification: pending

## Method

- public EEG only;
- participant-held-out train / validation / test;
- architecture selection by validation balanced accuracy;
- test metrics are not used to choose the winner;
- every prepared archive and checkpoint must carry the same source provenance SHA-256;
- no synthetic EEG, fabricated confidence, or pre-filled metric is accepted.
