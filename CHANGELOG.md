# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/) and
Semantic Versioning.

## [0.2.0] - 2026-08-10

### Changed

- Architecture and documentation now describe C4 and C8 backbones;
  `group_order` parameter selects the cyclic group (4 or 8).
- C4 backbone: 2.2 M parameters, 2.7× faster training, comparable margin
  (+0.2546 vs +0.2501 on the validated signature dataset).
- C8 backbone: 4.3 M parameters, finer 45° equivariance.
- TRAINING.md, ARCHITECTURE.md and MODEL_CARD.md updated for both variants.
- `model.py` docstring: C8 → C_N steerable encoder.
- `prepare_dataset.sh` in sigorbit-trainer now provides standalone dataset
  download, deduplication and materialization.

## [0.1.0] - 2026-08-07

### Added

- MIT-licensed `sigorbit` Python package;
- SO(2) canonicalizer and C_N-steerable signature encoder (C4 or C8 group_order);
- safe slim-checkpoint loader and versioned preprocessing;
- single and batched embedding API;
- minimal FastAPI `/health` and `/embed` example;
- bounded request/file/image decoding, optional bearer auth, inference backpressure,
  generic errors and pre-load checkpoint hash verification;
- slim-checkpoint tooling, model card, provenance manifest and release gate;
- dataset licence audit; checkpoint excluded from Git and distributions pending permission;
