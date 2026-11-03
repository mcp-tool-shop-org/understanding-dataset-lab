---
title: Results so far
description: What each stage tried, with counts, and the negative results first.
sidebar:
  order: 3
---

The results so far are counts on **development banks of 160 renderings**. They measure how well udlab generates
scenes, not whether any model understands people. The negative results come first, because they shaped the design.

## Stage 3: a list of accepted forms (not adopted)

The idea was to add a list of accepted sentence forms to the writer's prompt.
- With the list: **111 of 160** renderings passed.
- With the baseline prompt: **127 of 160**.
- Not adopted, by the pre-registered rule. The writer didn't use the listed forms.

## Stage 4: shapes assigned by code (not adopted)

The idea was for code to assign each sentence's shape.
- With assigned shapes: **89 of 160**.
- With the baseline: **120 of 160**.
- The development gate failed, so there was no confirmatory run.
- A defect turned up: **16 renderings per arm (10%) were never generated**, because the prompt plus the output budget
  went over the writer's context limit.

## Stage 5: code moves and rewritten speech (passed its development gates)

Stage 5 (revision 2) passed all **7** of its pre-registered development gates:
1. **First drafts:** 126 of 160, against a floor of 112, and no rendering that passed before failed after. After the
   pipeline: 132 of 160.
2. **Code swaps:** 2,501 of 2,501 kept, 0 rolled back.
3. **Variety:** for each event kind, the most common form's share stayed under its bar.
4. **Speech rewrites:** 350 of 358 kept (97.8%).
5. **Leaks:** how much an event's form reveals about its position or witness stayed under the ceiling for every kind.
6. **Context:** 0 renderings over the context limit.
7. **Rung choice:** it matched its pre-registered rule.

## Not yet shown

- No sealed confirmatory run of Stage 5.
- No model trained on udlab data.
- Nothing yet about whether any model understands the people in a scene.
