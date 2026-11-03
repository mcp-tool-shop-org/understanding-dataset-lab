---
title: understanding-dataset-lab
description: A prototype lab for cognitive-empathy test scenarios, where every answer key comes from code.
sidebar:
  order: 0
---

> **empathy**: "Identification with or understanding of the thoughts, feelings, or emotional state of another person."
> Definition from [Wiktionary](https://en.wiktionary.org/wiki/empathy), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

**Cognitive empathy, as udlab means it:** an accurate model of what each party in a situation knows, believes, wants
and needs.

udlab (understanding-dataset-lab) builds test scenarios for that. Each one is a short scene with several people, and
the questions are about them: who saw the move, who still believes the old thing, who needs what.

## A prototype

udlab is open research in progress.
- **What it offers:** the method and the code, to read and run.
- **What its results are:** checks of how well it generates scenes, on development banks.
- **What it doesn't have yet:** a final or sealed result, or a trained model.
- **What it doesn't claim:** anything about whether a model understands anyone.

Stages can still change how scenes are made.

## In this handbook

- [Getting started](./getting-started/): run the whole pipeline on a CPU, with no model.
- [How it works](./how-it-works/): planned events, code-made keys, checked prose, sealing and probes.
- [Results so far](./results/): what each stage tried, negative results first.
- [Command reference](./reference/): every `udlab-gen` mode and option.
- [Security and privacy](./security/): what udlab touches, and what stays private.

udlab is the companion to [reasoning-dataset-lab](https://mcp-tool-shop-org.github.io/reasoning-dataset-lab/), which
builds puzzles and verification items the same way: code proves the key before any model words the item.
