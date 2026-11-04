# Ship Gate

> No repo is "done" until every applicable line is checked.
> Copy this into your repo root. Check items off per-release.

**Tags:** `[all]` every repo · `[npm]` `[pypi]` `[vsix]` `[desktop]` `[container]` published artifacts · `[mcp]` MCP servers · `[cli]` CLI tools

---

## A. Security Baseline

- [x] `[all]` SECURITY.md exists (report email, supported versions, response timeline) (2026-11-03)
- [x] `[all]` README includes threat model paragraph (data touched, data NOT touched, permissions required) (2026-11-03) — README "Security and trust"; SECURITY.md "Threat model"
- [x] `[all]` No secrets, tokens, or credentials in source or diagnostics output (2026-11-03) — identity scan RESULT CLEAN over all 263 blobs in history; token-shape grep empty; Colab strings in tests are fakes; sealing seeds never appear in public outputs
- [x] `[all]` No telemetry by default (2026-11-03) — stated in README and SECURITY.md

### Default safety posture

- [ ] `[cli|mcp|desktop]` SKIP: no delete or restart action; the only kill is of the Ollama server udlab itself started, when video memory goes over `--vram-halt-mib`
- [x] `[cli|mcp|desktop]` File operations constrained to known directories (2026-11-03) — writes only under `--out-dir`; reads its own prompt and legend and the files passed to `--probe-only`, `--verify-written` and `--recheck`
- [ ] `[mcp]` SKIP: not an MCP server
- [ ] `[mcp]` SKIP: not an MCP server

## B. Error Handling

- [x] `[all]` Errors follow the Structured Error Shape (2026-11-03) — refusals print `REFUSED (CODE): message`; prose-check refusals name the check, event and sentence offset; any other error prints `error: INTERNAL: <type>: <message>` with a hint (`cli.console_main`)
- [x] `[cli]` Exit codes (2026-11-03) — 0 ok · 1 refusal (a prose check or sealing failed) · 2 usage · 70 internal error; no command reports partial success, so 3 is unused
- [x] `[cli]` No raw stack traces without `--debug` (2026-11-03) — the `udlab-gen` entry (`console_main`) prints `error: INTERNAL` and exits 70, never a traceback (test `test_console_main_prints_internal_error_without_traceback`); there is no `--debug` flag
- [ ] `[mcp]` SKIP: not an MCP server
- [ ] `[mcp]` SKIP: not an MCP server
- [ ] `[desktop]` SKIP: not a desktop app
- [ ] `[vscode]` SKIP: not a VS Code extension

## C. Operator Docs

- [x] `[all]` README is current (2026-11-03) — what it is, the prototype status, the results, what is here, running it without a GPU (Python 3.15+, uv), what stays private
- [x] `[all]` CHANGELOG.md (Keep a Changelog format) (2026-11-03)
- [x] `[all]` LICENSE file present and repo states support status (2026-11-03) — MIT; README and SECURITY.md say prototype, 0.1.x supported
- [x] `[cli]` `--help` output accurate for all commands and flags (2026-11-03) — argparse help; the handbook reference groups every option from it
- [ ] `[cli|mcp|desktop]` SKIP: no log levels; it prints a JSON summary or a refusal, and the private sealing seed never appears in either
- [ ] `[mcp]` SKIP: not an MCP server
- [ ] `[complex]` SKIP: not an operated service; the Starlight handbook covers use, the method, results, the commands and security

## D. Shipping Hygiene

- [x] `[all]` `verify` script exists (2026-11-03) — `verify.sh`: tests with the 90% coverage floor, then a CPU smoke run and its probe
- [x] `[all]` Version in manifest matches git tag (2026-11-03) — pyproject 0.1.0; tag v0.1.0 at release
- [x] `[all]` Dependency scanning runs in CI (2026-11-03) — `npm audit --audit-level=high` on the site tree in `ci.yml`; the package has no runtime dependencies
- [x] `[all]` No known high/critical vulnerabilities in any dependency tree (2026-11-03) — site `npm audit`: 0 vulnerabilities; no runtime Python dependencies
- [ ] `[all]` SKIP: org rule, no dependabot unless requested
- [ ] `[npm]` SKIP: not published to npm
- [ ] `[npm]` SKIP: not published to npm
- [ ] `[npm]` SKIP: not published to npm or PyPI (0.1.0 is a GitHub release only); `requires-python = ">=3.15"` is set
- [ ] `[npm]` SKIP: not published to a registry; `uv.lock` and `site/package-lock.json` are committed
- [ ] `[vsix]` SKIP: not a VS Code extension
- [ ] `[desktop]` SKIP: not a desktop app

## E. Identity (soft gate — does not block ship)

- [x] `[all]` Logo in README header (2026-11-04) — brand PR #114 merged; the README's raw logo URL returns 200
- [x] `[all]` Translations (polyglot-mcp, 8 languages) (2026-11-03) — README in English plus ja, zh, es, fr, hi, it and pt-BR, made locally by `translate-all.mjs` (translategemma:27b); one Stage 5 table cell each in ja, zh and fr pinned by hand in `.polyglot-pins.json` so every cell keeps its numbers
- [x] `[org]` Landing page (@mcptoolshop/site-theme) (2026-11-03) — `site/` with a six-page Starlight handbook, deployed by `pages.yml`
- [x] `[all]` GitHub repo metadata: description, homepage, topics (2026-11-04) — set at the visibility change

---

## Gate Rules

**Hard gate (A–D):** Must pass before any version is tagged or published.
If a section doesn't apply, mark `SKIP:` with justification — don't leave it unchecked.

**Soft gate (E):** Should be done. Product ships without it, but isn't "whole."

**Executed vs attested.** `shipcheck audit` only *counts these checkboxes* — it does not read your repo, so a box can be green while the fact is false. The lines that say **"executed by `npx @mcptoolshop/shipcheck <gate>`"** are backed by a command that reads the real artifact and exits 1 on the real defect. Run those gates (they are wired into shipcheck's own `verify`); don't just tick their boxes. Executed today: **A1/A2** (`security-docs`), **A3** (`secrets`), **D2/D6/D7** (`manifest`), **D3-config + OIDC/provenance** (`ci`), **real vulnerabilities + alerting** (`deps`), **D5** (`pack`), plus front-door (`front-door`) and dogfood freshness (`dogfood`). Every other line is still an attestation you are vouching for. Note the two dependency layers: `ci` proves a scanner is *configured*; `deps` proves there are *no known vulnerabilities* — a repo can pass the first while failing the second.

**Checking off:**
```
- [x] `[all]` SECURITY.md exists (2026-02-27)
```

**Skipping:**
```
- [ ] `[pypi]` SKIP: not a Python project
```
