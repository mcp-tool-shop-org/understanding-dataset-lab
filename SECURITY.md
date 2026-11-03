# Security

## Reporting a Vulnerability

Report a vulnerability privately through GitHub's security advisories on this repository, or by email to
`64996768+mcp-tool-shop@users.noreply.github.com`. Please do not open a public issue for a security vulnerability.

We aim to acknowledge a report within 48 hours, assess its impact within 7 days, and ship a fix within 30 days.

Please also report, privately in the same way, anything in this repository or its history that would let someone
work out a sealed answer key: a private seed, a sealed store, or a held-out scenario.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes, as a prototype |

## Scope

`understanding-dataset-lab` (`udlab-gen`) is a local CLI that generates, writes, seals, verifies and probes test
scenarios. It is not a network server, does not listen on any public port, and collects no telemetry.

## Data and network

- **Reads:** its own prompt and legend files, and the files you point it at (`--probe-only`, `--verify-written`,
  `--recheck`).
- **Writes:** the output directory you give it (`--out-dir`).
- **Network:** with `--writer ollama` only, an Ollama server on loopback. Any endpoint other than plain
  `http://127.0.0.1:<port>` is refused before a request is sent. In GPU mode it starts and owns that server, and stops
  it if video memory goes over `--vram-halt-mib`. The Colab launcher in `scripts/colab/` is separate, and talks to Colab
  only when you run it.

## Threat model

- **A model's output trusted as an answer key.** Handled by design: keys come from the code's plan, are sealed with a
  private seed, and every rendering is checked by code against the plan before it counts.
- **A leaked seed or sealed store,** which would give the answers away. Sealed stores and seeds are git-ignored, and the
  tests use synthetic seeds only.
- **A crafted input file.** Handled by validating it before use.
