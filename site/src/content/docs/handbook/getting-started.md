---
title: Getting started
description: Install udlab and run the whole pipeline on a CPU, with no model and no GPU.
sidebar:
  order: 1
---

## What you need

- [uv](https://docs.astral.sh/uv/). It installs Python 3.15 for you if you don't have it.
- Git.

The default writer is a **CPU mock**: a template writer that needs no model. Everything below runs on any machine.

## Run it

```bash
git clone https://github.com/mcp-tool-shop-org/understanding-dataset-lab
cd understanding-dataset-lab
uv run udlab-gen --count 3 --out-dir artifacts/demo
```

That does three things:
- it plans three scenes and writes them with the mock;
- it seals each scene's key with a fresh private seed;
- it prints a JSON summary.

Git ignores the output folder (`artifacts/`):

```
artifacts/demo/
  part_public/        the public part: items.jsonl and part.json
  sealed_store/       the keys and responses, sealed with the private seed
  build_summary.json
  checkpoint.json
  generations.jsonl
```

## Probe it

```bash
uv run udlab-gen --probe-only artifacts/demo/sealed_store/keys.sealed.jsonl
```

The shortcut probe tests whether surface features of the scenes give the answers away. If they did, a model could
score well without tracking the people.

## Test it

```bash
uv run pytest -n auto
```

753 tests, using synthetic seeds and invented scenes only. The full suite is long, so `-n auto` runs it in parallel.

## Using the real writer

`--write --writer ollama` uses the pinned writer model, `qwen3.6:35b`, through a local Ollama server. That needs a
GPU, and udlab starts and owns the server itself. You also have to pass the options that size each prompt exactly
(`--llama-tokenize` and `--writer-gguf`). See the [command reference](../reference/).
