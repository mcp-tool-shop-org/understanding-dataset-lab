---
title: Security and privacy
description: What udlab touches, what it never touches, and what stays private.
sidebar:
  order: 5
---

## What udlab touches

- **Reads:** its own prompt and legend files, and the files you point it at (`--probe-only`, `--verify-written`,
  `--recheck`).
- **Writes:** only the output directory you give it.
- **Network:** with `--writer ollama` only, an Ollama server on loopback. Any endpoint other than plain
  `http://127.0.0.1:<port>` is refused before a request is sent. The Colab launcher in `scripts/colab/` is separate,
  and talks to Colab only when you run it.
- **No telemetry.**

## What stays private

These stay private, because publishing them would give the answers away:
- the sealed test bank and the development seed banks;
- every sealed store and generation log;
- the run records with their plan seeds.

None of them is in the repository or its history. The tests use synthetic seeds and invented scenes only. Sealed
stores and seed files are git-ignored.

## Reporting a problem

Report vulnerabilities privately, through GitHub's security advisories on the repository or by email to
`64996768+mcp-tool-shop@users.noreply.github.com`. Please also report, privately, anything that would let someone
work out a sealed key. See
[SECURITY.md](https://github.com/mcp-tool-shop-org/understanding-dataset-lab/blob/main/SECURITY.md).

## Licences

The code and the repository's own text are MIT. The definition of empathy quoted on these pages is from
[Wiktionary](https://en.wiktionary.org/wiki/empathy), under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). It is not used in any dataset or prompt.
