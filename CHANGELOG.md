# Changelog

All notable changes to understanding-dataset-lab are recorded here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the package (`udlab-generator`) follows Semantic Versioning.

## [0.1.0] - 2026-11-03

The first public release, as a **prototype**.

### Added

- `udlab-gen`: plans multi-party scenarios, writes them with a local writer model or a CPU mock, seals the keys with a
  private seed, verifies a written set by re-rendering and diffing it, and runs out-of-fold shortcut probes.
- The Stage 5 pipeline (code moves and rewritten speech, revision 2), which passed all 7 of its pre-registered
  development gates.
- The writer prompt for `qwen3.6:35b`, the Stage 4 legend, and a Colab launcher for the writer on a rented GPU.
- 759 tests, using synthetic seeds and invented scenes only.
- README, MIT `LICENSE`, `SECURITY.md`, `SHIP_GATE.md`, CI with coverage and the Atlas map, a landing page and a
  handbook.
- `.gitattributes` keeps every file byte-exact, since the legend and several fixtures are pinned by SHA-256.

### Changed

- The Colab launcher's reconcile script has no default path (`COLAB_RECONCILE_SH` is empty unless set).
- One test sentence is now hand-written.
- Stage 5: a rewrite transport error (an Ollama timeout, for example) now aborts the run instead of being recorded
  as a failed rewrite. It is a fault in the rig, not a data outcome. The rung phase already behaved this way.
