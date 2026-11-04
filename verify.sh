#!/usr/bin/env bash
# One-command check: the test suite with its coverage floor, then a CPU smoke run of the whole pipeline and its probe.
set -euo pipefail
cd "$(dirname "$0")"
uv sync --frozen
uv run pytest -n auto --cov=udlab_generator --cov-fail-under=90
smoke="$(mktemp -d)"
trap 'rm -rf "$smoke"' EXIT
uv run udlab-gen --count 3 --out-dir "$smoke/demo" > /dev/null
uv run udlab-gen --probe-only "$smoke/demo/sealed_store/keys.sealed.jsonl" > /dev/null
echo "verify: tests, coverage floor, smoke run and probe all passed"
