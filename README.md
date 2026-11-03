<p align="center">
  <a href="README.md">English</a> | <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/understanding-dataset-lab/readme.png" alt="understanding-dataset-lab" width="400">
</p>

<p align="center">
  <b>empathy</b>: "Identification with or understanding of the thoughts, feelings, or emotional state of another person."<br>
  <sub>Definition from <a href="https://en.wiktionary.org/wiki/empathy">Wiktionary</a>, <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>.</sub><br><br>
  Cognitive empathy, as udlab means it: an accurate model of what each party in a situation knows, believes, wants and needs.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-prototype-orange" alt="Prototype">
  <a href="https://github.com/mcp-tool-shop-org/understanding-dataset-lab/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/understanding-dataset-lab/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/understanding-dataset-lab/"><img src="https://img.shields.io/badge/docs-landing_page-blue" alt="Landing page"></a>
</p>

> **Prototype.** udlab is open research in progress. The method and the code are here to read and run. Its results
> so far are checks of how well it generates scenes, on development banks. It has no final or sealed result, no
> trained model, and makes no claim about whether any AI understands anyone. Stages can still change how scenes are made.

**understanding-dataset-lab (udlab) builds test scenarios for cognitive empathy: whether a model can keep an accurate
picture of what each party in a scene knows, believes, wants and needs.** Short scenes with several people are planned
by code, then put into prose by a local writer model. The answer to every question about a scene comes from the code's
plan, never from a model.

It is the companion to [reasoning-dataset-lab](https://github.com/mcp-tool-shop-org/reasoning-dataset-lab), which does
the same for puzzles and verification items.

## Why scenes with code-made keys

In a planned scene, the right answer follows mechanically from the events: who moved where, who could see, who said
what, and who heard it. Someone who was out of sight when an object was moved still believes it is where they last
saw it. Because those rules are in code, the key is computed, not judged.

A *code-made key* means:
- the answer comes from the code's plan of the scene;
- it is sealed with a private seed;
- it is never written or graded by a model.

Every prose rendering is checked by code against the plan before it counts.

## Status: what has been tried

Results are counts on development banks of 160 renderings. The negative results come first, because they shaped the
design.

| Stage | What was tried | Result | Adopted? |
|---|---|---|---|
| 3 | A list of accepted sentence forms in the writer's prompt | 111 of 160 renderings passed, against 127 of 160 for the baseline prompt. The writer didn't use the listed forms. | **No**, by the pre-registered rule |
| 4 | Sentence shapes assigned by code | 89 of 160, against 120 of 160 for the baseline. The development gate failed, so there was no confirmatory run. 16 renderings per arm (10%) were never generated, because the prompt plus the output budget went over the writer's context limit. | **No** |
| 5 | Code moves and rewritten speech (revision 2) | The development run passed all 7 pre-registered gates. First drafts: 126 of 160, against a floor of 112; 132 of 160 after the pipeline. Code swaps kept 2,501 of 2,501. Speech rewrites kept 350 of 358. 0 renderings went over the context limit. | Passed its development gates |

**Not yet shown:**
- no sealed confirmatory run of Stage 5;
- no model trained on udlab data;
- nothing yet about whether any model understands the people in a scene.

## What is here

- **`udlab-gen`**, the generator. It plans scenarios, runs the Stage 5 pipeline, seals each set's keys with a private
  seed, and verifies a written set by re-rendering and diffing it.
- **Shortcut probes**, run out of fold. They test whether surface features of a scene give the answer away.
- **The writer prompt** (`prompts/`). The writer is `qwen3.6:35b`, run locally through Ollama.
- **A Colab launcher** (`scripts/colab/`) for running the writer on a rented GPU.
- **759 tests.**

## Try it without a GPU

The default writer is a CPU mock, so the whole pipeline runs on any machine:

```bash
git clone https://github.com/mcp-tool-shop-org/understanding-dataset-lab
cd understanding-dataset-lab
uv run udlab-gen --count 3 --out-dir artifacts/demo
uv run udlab-gen --probe-only artifacts/demo/sealed_store/keys.sealed.jsonl
uv run pytest
```

- Python 3.15 or newer; `uv` installs it if needed.
- The first command writes a public part (`part_public/`) and a sealed store (`sealed_store/`) under `artifacts/demo/`,
  a folder Git ignores.
- `uv run udlab-gen --help` lists every option.

## What stays private

These stay private, because publishing them would give the answers away:
- the sealed test bank and the development seed banks;
- every sealed store and generation log;
- the run records with their plan seeds.

None of them is in this repository or its history. The tests use synthetic seeds and invented scenes only.

## Security and trust

- **Data touched:** files under the `--out-dir` you choose, and the writer prompt. With `--writer ollama`, it also
  calls a local Ollama server.
- **Data not touched:** nothing outside the output folder. There is no telemetry, and no network calls except to the
  local Ollama server and, if you run it, the Colab launcher.
- **Permissions:** none beyond reading and writing the output folder. In GPU mode, it stops the Ollama server it
  started if video memory goes over `--vram-halt-mib`.
- Report security issues as [SECURITY.md](SECURITY.md) describes.

## Licence

The code and the repository's own text are under the [MIT licence](LICENSE). The definition quoted under the logo is
from Wiktionary and stays under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). It is not used in
any dataset or prompt.
