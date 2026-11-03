---
title: How it works
description: Planned events, code-made keys, checked prose, sealing and shortcut probes.
sidebar:
  order: 2
---

## The key comes from the plan

A scene starts as a plan of events, in code:
- someone **enters** or **leaves** a room;
- someone **moves** an object;
- someone **sees** something, or is out of sight of it;
- someone **says** something, or sends a **message**.

Two rules in the code turn those events into beliefs:
- **What each person can see.** In each room, a screen stands between one station and the containers. Someone on
  the far side can't see what happens at the containers.
- **How belief persists.** Each person assumes an object stays where they last saw it in open view, or where a
  witness last said it was put.

So "where does she think the ledger is?" has one right answer, and code works it out. That answer is the scene's
**key**.

## Code-made keys

A code-made key means:
- the answer comes from the code's plan of the scene;
- it is sealed with a private seed;
- it is never written or graded by a model.

A key that a model chose would be an opinion. Because udlab computes it, it can't drift toward what a model finds
plausible.

## Prose, checked by code

A local writer model (`qwen3.6:35b`) puts each planned scene into prose. The writer can make mistakes:
- drop an event;
- reorder events;
- blur who saw what.

So **every rendering is checked by code against the plan before it counts.** A rendering that fails is refused.

## Sealing

Each set's keys are sealed with a private 256-bit seed:
- the public part holds the scenes and questions;
- the keys and the model responses sit in the sealed store;
- `--verify-written` re-derives and rechecks a written set.

## Shortcut probes

A test set is only useful if the answer can't be read off the surface. Examples are a word that always appears with
the right answer, or a sentence position that gives it away. udlab's shortcut probes run out of fold and test whether
surface features alone predict the answer.
