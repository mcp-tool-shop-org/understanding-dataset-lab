# udlab-generator

Release-1 scenario generator, writer prompt (`qwen3.6:35b`), banned-adverb render check, private-seed sealing with canonical `verify` (re-render and diff), template-diversity counter, and out-of-fold Theory-of-Mind (ToM) shortcut probes for `udlab`.

## Usage

```bash
uv run udlab-gen --count 300 --out-dir artifacts/r1_300
uv run pytest --cov=udlab_generator --cov-report=term-missing -v
```
