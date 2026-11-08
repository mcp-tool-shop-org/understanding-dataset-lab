# Changelog

All notable changes to understanding-dataset-lab are recorded here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the package (`udlab-generator`) follows Semantic Versioning.

## [Unreleased]

### Added

- Stage 5 run record and status (`stage5_run_record.json`, `stage5_status.json`): per-step `model_endpoint` (`b`, `rung`, `swap`, `rewrite`), per-step `step_timings` (`start`, `end`, `seconds`, `calls`, `call_count`) preserved across `--resume`, and `PlanStatisticsResult` in `_compute_plan_statistics` / `--stage5 --dry` reporting per-20-event stream block spreads alongside global balance spreads.
- Stage 5 dual-endpoint partial-offload guard (`Stage5PartialOffloadError`, `check_endpoints_partial_offload`) reporting whether an offloaded model is the run's own model or another seat's model, and end-of-run default-endpoint (`127.0.0.1:11434`) cleanup (`unload_and_verify_default_endpoint_empty`, `default_endpoint_end`) that unloads only the run's own model tags while leaving foreign models untouched and recording `foreign_models`, `own_models_absent`, and `foreign_models_present`.
- 6 Stage 5 regression tests (`test_33`–`test_38`) and 2 JSON fixtures (`stage5_step_timings_fixture.json`, `stage5_plan_statistics_fixture.json`).

### Fixed

- Stage 5 owned-server lifecycle (`run_stage5_pipeline`): keep one owned Ollama server open across Phase 1 (`b`), Phase 2 (`rung`), and Phase 3 (`rewrite`), route every model call in every phase (including `--resume` after B and custom `popen_factory` / `server_port` setups) through the owned server's URL with zero calls to `127.0.0.1:11434`, unload only the run's own model tags on `127.0.0.1:11434` at run end without touching other seats' models, and compute `_compute_plan_statistics` on 20-event blocks along each variant's own stream (`move`, `see_enter`, `see_Placed`, `say`) to match `generate_stage5_shape_plan`.

## [0.1.0] - 2026-11-04

The first public release, as a **prototype**.

### Added

- `udlab-gen`: plans multi-party scenarios, writes them with a local writer model or a CPU mock, seals the keys with a
  private seed, verifies a written set by re-rendering and diffing it, and runs out-of-fold shortcut probes.
- The Stage 5 pipeline (code moves and rewritten speech, revision 2), which passed all 7 of its pre-registered
  development gates.
- The writer prompt for `qwen3.6:35b`, the Stage 4 legend, and a Colab launcher for the writer on a rented GPU.
- 759 tests, using synthetic seeds and invented scenes only.
- README, MIT `LICENSE`, `SECURITY.md`, `SHIP_GATE.md`, CI with coverage and the Atlas map, a landing page and a
  handbook. CI runs a fast set on pull requests (the slow Stage 5 tests listed in `tests/conftest.py` left out) and
  the full suite on every push to main. The full suite's job timeout is 90 minutes instead of 45, because it
  measured over 45 minutes on a private 2-core runner (public runners have 4).
- `.gitattributes` keeps every file byte-exact, since the legend and several fixtures are pinned by SHA-256.

### Changed

- The Colab launcher's reconcile script has no default path (`COLAB_RECONCILE_SH` is empty unless set).
- One test sentence is now hand-written.
- Stage 5: a rewrite transport error (an Ollama timeout, for example) now aborts the run instead of being recorded
  as a failed rewrite. It is a fault in the rig, not a data outcome. The rung phase already behaved this way.
