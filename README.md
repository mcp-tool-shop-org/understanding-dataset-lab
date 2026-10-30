# udlab-generator

Release-1 scenario generator, writer prompt (`qwen3.6:35b`), banned-adverb render check, private-seed sealing with canonical `verify` (re-render and diff), template-diversity counter, and out-of-fold Theory-of-Mind (ToM) shortcut probes for `udlab`.

## Usage

```bash
# Generate 300 scenarios with a fresh random 256-bit private shuffle seed:
uv run udlab-gen --count 300 --out-dir artifacts/r1_300

# Re-run the writer-agnostic shortcut probe on any sealed JSONL (e.g. after qwen3.6:35b prose rendering):
uv run udlab-gen --probe-only artifacts/r1_300/sealed_store/keys.sealed.jsonl

# Run the test suite with coverage:
uv run pytest --cov=udlab_generator --cov-report=term-missing -v
```
