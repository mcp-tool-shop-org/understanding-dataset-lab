# Scorecard

**Repo:** understanding-dataset-lab
**Date:** 2026-11-04
**Release:** v0.1.0 (prototype)
**Source:** `npx @mcptoolshop/shipcheck audit` against [SHIP_GATE.md](SHIP_GATE.md); exit 0, all hard gates pass.

| Gate | Checked | Skipped (with reason) | Unchecked |
|------|---------|-----------------------|-----------|
| A. Security baseline (hard) | 5 | 3 | 0 |
| B. Error handling (hard) | 3 | 4 | 0 |
| C. Operator docs (hard) | 4 | 3 | 0 |
| D. Shipping hygiene (hard) | 4 | 7 | 0 |
| E. Identity (soft) | 4 | 0 | 0 |
| Gate rules | 1 | 1 | 0 |
| **Total** | **21** | **18** | **0** |

Each skip names its reason in SHIP_GATE.md. Most are for surfaces udlab doesn't have: no npm package, MCP server,
desktop app or operated service.
