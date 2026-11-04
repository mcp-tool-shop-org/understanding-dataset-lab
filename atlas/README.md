# understanding-dataset-lab: how it works

Mapped at 2026-11-04 from commit 81765f0 by Atlas 1.24.0.

## What this is

understanding-dataset-lab (a prototype) plans multi-party scenes in code, puts them into prose with a local writer model, checks every rendering against the plan, and seals the code-made answer keys with a private seed. (written by a person)

7 parts, mostly Python (23 files), CSS (2), TypeScript (2), shell (2), Astro (1) and JavaScript (1). Work enters through 3 doors; the busiest is ci, which reaches 3 parts. It deploys a site to GitHub Pages. People run udlab-gen.

## What changed since 2026-11-03 (d951db7)

Nothing structural changed since 2026-11-03; 1 file added and 3 changed content.

## What comes in

1. **ci.** On a pull request; on a push to main touching 10 paths; or by hand. Runs tests/.
2. **Deploy site to GitHub Pages.** On a push to main touching 2 paths; or by hand. Runs site/astro.config.mjs and site/src/.
3. **udlab-gen** (a command people run). Runs src/udlab_generator/cli.py.

## What happens through ci

1. The workflow runs tests/ in tests.
2. That reaches scripts (1 file) and src (11 files).
3. It uploads coverage to Codecov.

## Who reads the results

ci writes nothing this map can see.

## The other doors

**Deploy site to GitHub Pages** runs site/astro.config.mjs and site/src/, and deploys the site.

**udlab-gen** (a command people run) runs src/udlab_generator/cli.py.

## What breaks what

- **src** is imported only from tests, by 1 part (tests), and sits on the path of 2 doors.
- **scripts** is imported only from tests, by 1 part (tests), and sits on the path of 1 door.

## What tends to change together

- **src/udlab_generator/render.py** and **tests/test_generator.py** changed together in 32 of 40 commits, and the tests part imports the src part.
- **src/udlab_generator/keys.py** and **src/udlab_generator/world.py** changed together in 6 of 9 commits, inside the src part.
- **src/udlab_generator/__init__.py** and **src/udlab_generator/render.py** changed together in 25 of 39 commits, inside the src part.
- **src/udlab_generator/__init__.py** and **tests/test_generator.py** changed together in 22 of 38 commits, and the tests part imports the src part.
- **src/udlab_generator/__init__.py** and **src/udlab_generator/cli.py** changed together in 16 of 30 commits, inside the src part.

Confidence is low: fewer than 25 source files reach 10 revisions in the window.

Window: 180 days; a pair counts from 3 shared commits, since 5 source files reach 10 revisions; the floor rises to 10 when 25 do.

## What no test touches

Every code part is imported by at least one test.

verify.sh runs in no workflow.

## Written but never read

No place this map can see is written, so none goes unread.

## Helpers that look duplicated

No two parts export a helper that looks alike.

## Generated, never hand-edited

Nothing in this repository writes to a tracked place this map can see.

## Hand-authored

People write .github/, prompts/, the repository root and site/. Nothing in this repository writes to them.

## Where to start

src/udlab_generator/cli.py → src/udlab_generator/render.py → src/udlab_generator/stage5_pipeline.py → src/udlab_generator/sealing.py

Read those in order to follow one run of udlab-gen end to end. This path follows udlab-gen (a command people run) from its entry, since ci runs only tests.

## What this map cannot see

- 7 reads use paths built at run time and are not named here.
- 26 writes and 56 reads go to a path their caller passes, not to this repository.
- 4 reads go to the directory the command is run in, not to this repository.
- Statistics confidence is low: fewer than 25 source files reach 10 revisions in the window.

Regenerate with `npx --yes @dogfood-lab/atlas map`.
