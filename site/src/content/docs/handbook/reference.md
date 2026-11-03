---
title: Command reference
description: Every udlab-gen mode and option, grouped by what it does.
sidebar:
  order: 4
---

`udlab-gen` is the only command. `uv run udlab-gen --help` prints the full list; this page groups the options.

## Generate and seal

| Option | What it does |
|---|---|
| `--count N` | Number of scenarios (default 300). |
| `--seed N` | Base deterministic generation seed. |
| `--private-seed HEX` | A 256-bit hex sealing seed. If omitted, it reads `UDLAB_PRIVATE_SEED`, or generates a fresh one. |
| `--out-dir DIR` | Where `part_public/` and `sealed_store/` go. |

## Write (put scenes into prose)

| Option | What it does |
|---|---|
| `--write` | Run the writer pass. |
| `--writer {mock,ollama}` | `mock` (CPU template, the default) or `ollama` (the pinned `qwen3.6:35b`). |
| `--num-predict N`, `--max-num-ctx N` | The writer's output budget, and its context cap (at most 9,216). |
| `--think` / `--no-think` | Thinking mode for the writer (off by default). |
| `--max-retries N`, `--retry-form`, `--multi-turn-retries` | Repair retries for refused renderings (0 by default). |
| `--vram-halt-mib N` | Stop the Ollama server udlab started if video memory goes over N MiB (default 26,000). |
| `--server-port N` | Pin that server's port. |
| `--llama-tokenize PATH`, `--writer-gguf PATH` | Count prompt tokens exactly to size the context. Required for `--writer ollama`. |
| `--grant-id ID` | A GPU grant id, required with `--writer ollama`. |
| `--resume` | Resume an interrupted build from its verified sealed checkpoint. |

## Check and verify

| Option | What it does |
|---|---|
| `--probe-only FILE` | Run the shortcut probe on a sealed JSONL, then exit. |
| `--verify-written DIR` | Post-write verification: an independent re-derivation, all recounts, the prose checks and the probe. |
| `--recheck DIR` | Re-run every prose check on saved texts, with no model and no GPU. |

## Stage options

| Option | What it does |
|---|---|
| `--stage5` | Run the Stage 5 pipeline and write `stage5_run_record.json`. |
| `--dry` | With `--stage5`: build the worlds and the plan, and print counts without a writer or files. |
| `--stage5-seed-bank FILE` | A JSON list of scenario seeds for `--stage5`. |
| `--stage5-plan-seed`, `--stage5-rung-seed` | Fixed seeds, for replay and tests only. |
| `--stage5-redo-b`, `--stage5-redo-rung`, `--stage5-s-code` | Re-run parts of Stage 5 on `--resume`, or use its fallback. |
| `--goal-mode`, `--constraint-layout`, `--accepted-forms-line`, `--checker-profile` | Writer-prompt and checker variants from earlier stages. |
| `--shape-plan`, `--shape-plan-seed` | Stage 4's shape-plan modes. |
| `--paired-seed-bank`, `--paired-compare`, `--paired-out`, `--diagnose-paired` | Paired runs on a fixed seed bank, their comparison (McNemar) and an offline diagnosis. |

## Exit codes

- **0:** done.
- **1:** a refusal (a prose check or the sealing failed).
- **2:** a usage error.
- **70:** an internal error, printed as `error: INTERNAL: …` with a hint and no traceback.
