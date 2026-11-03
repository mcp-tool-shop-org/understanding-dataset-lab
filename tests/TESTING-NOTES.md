# Testing Notes

## Tests that diff against fixed commits

1. **Pin only commits that are on `main` (squash-merge commits).** A PR-branch commit disappears from `main` after a squash merge, so any test pinned to a PR-branch SHA skips forever on a fresh clone where the unmerged branch object is absent. The U3 commit-range diff test (`tests/test_stage5_rewrite.py:1186`) was the case: its original endpoint `a7ca5831a96d2b034a9f995ac58ea0a85e73206f` was a PR-branch commit not on `main` and was re-pinned in U4 to the U3 squash-merge commit `f6277c6d6f52b6280b0f1cab518dd7d741a85c6e` on `main`.
2. **Skip with an explicit reason when a pinned commit is missing.** Each commit-range diff test checks commit reachability via `git cat-file -e <sha>^{commit}` and calls `pytest.skip(f"Commit {sha} missing from clone (shallow checkout)")` when a commit is unavailable (for example, in a shallow CI checkout).
3. **Bound both endpoints of a commit-range diff.** Tests that verify scope boundaries of a completed card must diff `<base_sha_on_main>..<merge_sha_on_main>` rather than `<base_sha>..HEAD` or the working tree so subsequent cards on `main` do not trip prior cards' file allow-lists.

### Inventory of tests running `git diff`, `git rev-parse`, or `git ls-files`

| File : Line | Test Name | Git Command / Range | Pinned Commits on `main`? | Could U4 Changes Trip It? |
| :--- | :--- | :--- | :--- | :--- |
| `tests/test_stage5_templates.py:714` | `test_edge_10_no_existing_module_changes` | `git diff --name-only 50441f46159d073eae0ea3d99c36903514b4bddf 842a2e6b6ef15c46ca92fb85d2412ec82404d187` | Yes (`50441f46` = U1 merge on `main`; `842a2e6b` = U2 merge on `main`) | No (bounded historical range on `main`) |
| `tests/test_stage5_rewrite.py:1186` | `test_case_15_u2_public_wrapper_parity_and_count_changed` | `git diff --name-only 842a2e6b6ef15c46ca92fb85d2412ec82404d187 f6277c6d6f52b6280b0f1cab518dd7d741a85c6e` | Yes after U4 re-pin (`842a2e6b` = U2 merge on `main`; `f6277c6d` = U3 squash-merge on `main`; previously pinned to PR-branch commit `a7ca5831`, which was not on `main`) | No (bounded historical range on `main`, with `tests/test_stage5_templates.py` allow-listed for U3-fix1) |
| `tests/test_generator.py:7600` | `test_80_card_tk_leak_guard_git_ls_files_and_planted_fixtures` | `git ls-files` | N/A (no fixed commit SHA; inspects tracked index files) | Yes if any tracked file introduced by U4 leaked a drive/home path, private seed, or unsealed `response_text` |
| `tests/test_generator.py:11092` | `test_104_cli_paired_seed_bank_with_fake_transport` | `git rev-parse HEAD` | N/A (no fixed commit SHA; resolves current `HEAD`) | No (asserts `out_doc["commit"] == HEAD` dynamically) |
| `tests/test_generator.py:12059` | `test_109_commit_sha_exact_40_hex_and_unknown_fallback` | `git rev-parse HEAD` | N/A (no fixed commit SHA; resolves current `HEAD`) | No (asserts `_resolve_git_commit_sha() == HEAD` dynamically) |
