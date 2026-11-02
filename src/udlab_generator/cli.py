"""CLI entrypoint for the udlab release-1 scenario generator, writer pass, sealing, and shortcut probe."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
import time
from typing import Any

from udlab_generator.diversity import count_template_diversity
from udlab_generator.probe import (
    ClassBalanceReport,
    ShortcutProbeReport,
    run_shortcut_probe,
)
from udlab_generator.render import (
    ATTEMPT_0_TEMPERATURE,
    DEFAULT_ACCEPTED_FORMS_LINE,
    DEFAULT_CHECKER_PROFILE,
    DEFAULT_CONSTRAINT_LAYOUT,
    DEFAULT_GOAL_MODE,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_RETRY_FORM,
    DEFAULT_SHAPE_PLAN,
    DEFAULT_VRAM_HALT_MIB,
    DEFAULT_WRITER_MAX_NUM_CTX,
    DEFAULT_WRITER_MODEL_PIN,
    DEFAULT_WRITER_NUM_PREDICT,
    DEFAULT_WRITER_THINK,
    DEFAULT_WRITER_THINK_NUM_PREDICT,
    GENERATIONS_FILENAME,
    NUM_CTX_MARGIN_TOKENS,
    NUM_CTX_STEP_TOKENS,
    SEALED_DIAGNOSTICS_FILENAME,
    SEALED_RESPONSES_FILENAME,
    SHAPE_PLAN_BLOCK_SIZE,
    VALID_ACCEPTED_FORMS_LINES,
    VALID_CHECKER_PROFILES,
    VALID_CONSTRAINT_LAYOUTS,
    VALID_GOAL_MODES,
    VALID_RETRY_FORMS,
    VALID_SHAPE_PLANS,
    EventDiagnosticRecord,
    GenerationCallRecord,
    LlamaTokenizeCounter,
    MaxNumCtxRefusalError,
    MockWriter,
    OllamaHttpTransport,
    OllamaWriter,
    PromptTokenCounter,
    ProseFaultRecord,
    RenderCheckError,
    RenderedScenario,
    SealedResponseEntry,
    WriterCheckpointManager,
    _resolve_failing_sentence_in_raw_response,
    append_sealed_response,
    check_rendered_prose,
    collect_prose_faults,
    compare_paired_runs,
    compute_hard_constraints_sha256,
    compute_sealed_response_digests,
    compute_shape_plan_run_telemetry,
    count_duplicate_stake_statements,
    count_prompt_tokens_exact,
    decide_num_ctx,
    derive_call_seed,
    diagnose_paired_run,
    diagnose_rendered_prose,
    estimate_prompt_tokens,
    format_non_sealed_refusal_reason,
    generate_shape_plan,
    kill_pid_tree,
    locate_failing_sentence_in_response,
    parse_generation_refusal_metadata,
    resolve_sealed_failing_sentence,
    run_paired_seed_bank,
    summarize_diagnostics_public,
    summarize_fault_vector_counts,
    write_sealed_diagnostics,
)
from udlab_generator.sealing import (
    SealingError,
    _deserialize_world,
    verify_written_dataset_recounts_and_prose,
    write_sealed_dataset,
)
from udlab_generator.world import World, generate_world, make_balanced_configs


def _summarize_d_key_patterns(sealed_rows: Sequence[dict[str, object]]) -> dict[str, object]:
    """Summarize the distribution of D key patterns and set sizes across scenarios."""
    pattern_counts: Counter[str] = Counter()
    size_counts: Counter[str] = Counter()
    total_d = 0
    for row in sealed_rows:
        if int(row.get("pass_index", 1)) != 1:  # type: ignore[arg-type]
            continue
        questions = row.get("questions", [])
        if not isinstance(questions, list):
            continue
        for q in questions:
            if isinstance(q, dict) and q.get("question_type") == "D":
                raw_key = q.get("key", [])
                if isinstance(raw_key, (list, tuple)):
                    fact_ids = tuple(sorted(str(part).strip() for part in raw_key if str(part).strip()))
                else:
                    fact_ids = tuple(sorted(part.strip() for part in str(raw_key).split(",") if part.strip()))
                pattern_label = ",".join(fact_ids) if fact_ids else "NONE"
                pattern_counts[pattern_label] += 1
                size_counts[str(len(fact_ids))] += 1
                total_d += 1
    max_pattern_count = max(pattern_counts.values(), default=0)
    max_pattern_share = round(max_pattern_count / total_d, 6) if total_d > 0 else 0.0
    return {
        "total_scenarios": total_d,
        "distinct_patterns": len(pattern_counts),
        "max_pattern_count": max_pattern_count,
        "max_pattern_share": max_pattern_share,
        "counts_by_pattern": dict(sorted(pattern_counts.items())),
        "counts_by_set_size": dict(sorted(size_counts.items())),
    }


def _empty_shortcut_probe_report() -> ShortcutProbeReport:
    """Return an empty `ShortcutProbeReport` when 0 scenarios are kept."""
    return ShortcutProbeReport(
        num_scenarios=0,
        num_renderings=0,
        probes=(),
        balance=ClassBalanceReport(
            u_share_among_open_stems=0.0,
            who_u_share=0.0,
            why_u_share=0.0,
            b1_true_belief_share=0.0,
            b2_true_belief_share=0.0,
            x_unnotified_share=0.0,
            p_yes_share=0.0,
            k_yes_share=0.0,
            balanced=False,
        ),
        passed=False,
    )


def _summarize_call_outcomes(call_records: Sequence[GenerationCallRecord]) -> dict[str, object]:
    """Summarize kept vs refused renderings and refusals per check id across `call_records`."""
    refusals_by_check_id: Counter[str] = Counter()
    calls_kept = 0
    calls_refused = 0
    final_by_rendering: dict[tuple[str, int], GenerationCallRecord] = {}
    scenarios_seen: list[str] = []
    seen_set: set[str] = set()

    for rec in call_records:
        if rec.scenario_id not in seen_set:
            seen_set.add(rec.scenario_id)
            scenarios_seen.append(rec.scenario_id)
        final_by_rendering[(rec.scenario_id, rec.rendering_index)] = rec
        if rec.checks_passed:
            calls_kept += 1
        else:
            calls_refused += 1
            for reason_str in rec.refusal_reasons:
                check_id = reason_str.split(":", 1)[0].strip() if ":" in reason_str else reason_str.strip()
                if check_id:
                    refusals_by_check_id[check_id] += 1

    renderings_kept = sum(1 for r in final_by_rendering.values() if r.checks_passed)
    renderings_refused = sum(1 for r in final_by_rendering.values() if not r.checks_passed)
    scenarios_kept = 0
    scenarios_refused = 0
    for sid in scenarios_seen:
        r0 = final_by_rendering.get((sid, 0))
        r1 = final_by_rendering.get((sid, 1))
        if r0 is not None and r1 is not None and r0.checks_passed and r1.checks_passed:
            scenarios_kept += 1
        else:
            scenarios_refused += 1

    return {
        "scenarios_kept": scenarios_kept,
        "scenarios_refused": scenarios_refused,
        "renderings_kept": renderings_kept,
        "renderings_refused": renderings_refused,
        "calls_kept": calls_kept,
        "calls_refused": calls_refused,
        "kept": renderings_kept,
        "refused": renderings_refused,
        "refusals_by_check_id": dict(sorted(refusals_by_check_id.items())),
        "fault_vector_counts_by_check_id": summarize_fault_vector_counts(
            rec.fault_check_ids for rec in call_records
        ),
    }


def recheck_saved_responses(
    output_dir: Path,
    *,
    checker_profile: str = DEFAULT_CHECKER_PROFILE,
) -> dict[str, object]:
    """Re-run every prose check on the saved texts in `<output_dir>/sealed_store/responses.sealed.jsonl` offline (no model, no GPU)."""
    if checker_profile not in VALID_CHECKER_PROFILES:
        raise ValueError(f"Invalid checker_profile {checker_profile!r}; expected one of {sorted(VALID_CHECKER_PROFILES)}")
    sealed_dir = output_dir / "sealed_store"
    responses_path = sealed_dir / SEALED_RESPONSES_FILENAME
    sealing_meta_path = sealed_dir / "sealing.sealed.json"
    generations_path = output_dir / GENERATIONS_FILENAME

    if not responses_path.is_file():
        raise FileNotFoundError(f"Saved responses file not found: {responses_path}")
    if not sealing_meta_path.is_file():
        raise FileNotFoundError(f"Sealing metadata file not found: {sealing_meta_path}")

    sealing_meta = json.loads(sealing_meta_path.read_text(encoding="utf-8"))
    raw_all_scenarios = sealing_meta.get("all_scenarios") or sealing_meta.get("scenarios") or {}
    if not isinstance(raw_all_scenarios, dict):
        raise ValueError(f"Invalid scenarios dictionary in {sealing_meta_path}.")

    worlds_by_sid: dict[str, World] = {
        str(sid): _deserialize_world(w_dict)
        for sid, w_dict in raw_all_scenarios.items()
        if isinstance(w_dict, dict)
    }

    gen_by_call_id: dict[str, dict[str, Any]] = {}
    if generations_path.is_file():
        for line in generations_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict) and row.get("call_id"):
                    gen_by_call_id[str(row["call_id"])] = row

    saved_entries = [
        json.loads(line)
        for line in responses_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    call_results: list[dict[str, object]] = []
    refusals_by_check_id: Counter[str] = Counter()
    calls_kept = 0
    calls_refused = 0
    duplicate_stake_statements = 0
    legacy_refusal_format_calls = 0
    final_by_rendering: dict[tuple[str, int], bool] = {}
    scenarios_seen: list[str] = []
    seen_set: set[str] = set()
    all_verdicts_match = True

    for entry in saved_entries:
        call_id = str(entry["call_id"])
        call_index = int(entry.get("call_index", len(call_results) + 1))
        scen_id = str(entry["scenario_id"])
        r_idx = int(entry["rendering_index"])
        focal_agent = str(entry["focal_agent"])
        attempt = int(entry.get("attempt", 0))
        seed = int(entry.get("seed", 0))
        response_text = str(entry.get("response_text") or "")
        thinking_text = str(entry.get("thinking_text") or "")
        stored_resp_sha = str(entry.get("response_sha256") or "")
        stored_think_sha = str(entry.get("thinking_sha256") or "")

        actual_resp_sha, actual_think_sha = compute_sealed_response_digests(response_text, thinking_text)
        sha_verified = (actual_resp_sha == stored_resp_sha) and (actual_think_sha == stored_think_sha)

        gen_row = gen_by_call_id.get(call_id)
        if gen_row is not None:
            if str(gen_row.get("response_sha256") or "") != actual_resp_sha:
                sha_verified = False
            if str(gen_row.get("thinking_sha256") or "") != actual_think_sha:
                sha_verified = False

        if scen_id not in worlds_by_sid:
            raise KeyError(f"Scenario '{scen_id}' (for call '{call_id}') not found in sealing.sealed.json.")
        world = worlds_by_sid[scen_id]

        if scen_id not in seen_set:
            seen_set.add(scen_id)
            scenarios_seen.append(scen_id)

        entry_goal_mode = str(
            entry.get("goal_mode")
            or (gen_row.get("goal_mode") if gen_row is not None else None)
            or DEFAULT_GOAL_MODE
        )
        try:
            check_rendered_prose(
                response_text,
                world,
                focal_agent=focal_agent,
                goal_mode=entry_goal_mode,
                checker_profile=checker_profile,
            )
            passed = True
            check_id: str | None = None
            rule_id: str | None = None
            event_index: int | None = None
            event_kind: str | None = None
            failing_sentence: str | None = None
            reason: str | None = None
            faults: tuple[ProseFaultRecord, ...] = ()
            fault_check_ids: list[str] = []
            fault_rule_ids: list[str] = []
            calls_kept += 1
            call_dup_stakes = count_duplicate_stake_statements(response_text, world)
            duplicate_stake_statements += call_dup_stakes
        except RenderCheckError as exc:
            passed = False
            check_id = exc.check_id
            rule_id = exc.rule_id
            event_index = exc.event_index
            event_kind = exc.event_kind
            failing_sentence = _resolve_failing_sentence_in_raw_response(
                response_text,
                exc.failing_sentence,
                world=world,
            )
            reason = str(exc)
            faults = collect_prose_faults(
                response_text,
                world,
                focal_agent=focal_agent,
                goal_mode=entry_goal_mode,
                checker_profile=checker_profile,
            )
            fault_check_ids = list(dict.fromkeys(f.check_id for f in faults)) or (
                [exc.check_id] if exc.check_id else []
            )
            fault_rule_ids = [f.rule_id for f in faults] or (
                [exc.rule_id] if exc.rule_id else []
            )
            calls_refused += 1
            call_dup_stakes = count_duplicate_stake_statements(response_text, world)
            refusals_by_check_id[exc.check_id] += 1

        recomputed_fail_sha, recomputed_fail_off, recomputed_fail_len = (
            locate_failing_sentence_in_response(response_text, failing_sentence)
            if not passed
            else ("", None, None)
        )
        recorded_refusal_format = "sealed_offset_v1"
        failing_sentence_sha_verified = True
        if gen_row is not None:
            parsed_meta = parse_generation_refusal_metadata(gen_row)
            recorded_refusal_format = str(parsed_meta["format"])
            if recorded_refusal_format == "legacy_inline_v0":
                legacy_refusal_format_calls += 1
                fail_sha = recomputed_fail_sha
                fail_off = recomputed_fail_off
                fail_len = recomputed_fail_len
                if fail_sha:
                    resolved_sent, slice_ok = resolve_sealed_failing_sentence(
                        response_text,
                        failing_sentence_sha256=fail_sha,
                        failing_sentence_offset=fail_off,
                        failing_sentence_length=fail_len,
                    )
                    failing_sentence_sha_verified = slice_ok
                    if slice_ok and resolved_sent is not None:
                        failing_sentence = resolved_sent
            else:
                fail_sha = str(parsed_meta["failing_sentence_sha256"] or "")
                fail_off = parsed_meta["failing_sentence_offset"]
                fail_len = parsed_meta["failing_sentence_length"]
                if not passed:
                    resolved_sent, slice_ok = resolve_sealed_failing_sentence(
                        response_text,
                        failing_sentence_sha256=fail_sha,
                        failing_sentence_offset=fail_off,
                        failing_sentence_length=fail_len,
                    )
                    if (
                        not slice_ok
                        or fail_sha != recomputed_fail_sha
                        or fail_off != recomputed_fail_off
                        or fail_len != recomputed_fail_len
                    ):
                        failing_sentence_sha_verified = False
                        sha_verified = False
                    elif resolved_sent is not None:
                        failing_sentence = resolved_sent
                elif fail_sha or fail_off is not None or fail_len is not None:
                    failing_sentence_sha_verified = False
                    sha_verified = False
        else:
            fail_sha = recomputed_fail_sha
            fail_off = recomputed_fail_off
            fail_len = recomputed_fail_len

        final_by_rendering[(scen_id, r_idx)] = passed
        recorded_passed = bool(gen_row.get("checks_passed", True)) if gen_row is not None else passed
        verdict_matches = (passed == recorded_passed) and sha_verified and failing_sentence_sha_verified
        if not verdict_matches:
            all_verdicts_match = False

        call_results.append(
            {
                "call_id": call_id,
                "call_index": call_index,
                "scenario_id": scen_id,
                "rendering_index": r_idx,
                "focal_agent": focal_agent,
                "attempt": attempt,
                "seed": seed,
                "response_sha256": actual_resp_sha,
                "thinking_sha256": actual_think_sha,
                "sha256_verified": sha_verified,
                "checks_passed": passed,
                "recorded_checks_passed": recorded_passed,
                "verdict_matches_recorded": verdict_matches,
                "recorded_refusal_format": recorded_refusal_format,
                "check_id": check_id,
                "rule_id": rule_id,
                "event_index": event_index,
                "event_kind": event_kind,
                "failing_sentence": failing_sentence,
                "failing_sentence_sha256": fail_sha,
                "failing_sentence_offset": fail_off,
                "failing_sentence_length": fail_len,
                "failing_sentence_sha256_verified": failing_sentence_sha_verified,
                "duplicate_stake_statements": call_dup_stakes,
                "reason": reason,
                "fault_check_ids": fault_check_ids,
                "fault_rule_ids": fault_rule_ids,
                "fault_vector": [f.to_dict() for f in faults],
            }
        )

    renderings_kept = sum(1 for v in final_by_rendering.values() if v)
    renderings_refused = sum(1 for v in final_by_rendering.values() if not v)
    scenarios_kept = 0
    scenarios_refused = 0
    for sid in scenarios_seen:
        if final_by_rendering.get((sid, 0), False) and final_by_rendering.get((sid, 1), False):
            scenarios_kept += 1
        else:
            scenarios_refused += 1

    rule_id_counts: Counter[str] = Counter()
    for c in call_results:
        for rid in c.get("fault_rule_ids", []):  # type: ignore[union-attr]
            rule_id_counts[str(rid)] += 1

    return {
        "output_dir": str(output_dir),
        "responses_sealed_path": str(responses_path),
        "checker_profile": str(checker_profile),
        "num_calls": len(call_results),
        "calls": call_results,
        "scenarios_kept": scenarios_kept,
        "scenarios_refused": scenarios_refused,
        "renderings_kept": renderings_kept,
        "renderings_refused": renderings_refused,
        "calls_kept": calls_kept,
        "calls_refused": calls_refused,
        "kept": renderings_kept,
        "refused": renderings_refused,
        "duplicate_stake_statements": duplicate_stake_statements,
        "refusals_by_check_id": dict(sorted(refusals_by_check_id.items())),
        "rule_id_counts": dict(sorted(rule_id_counts.items())),
        "fault_vector_counts_by_check_id": summarize_fault_vector_counts(
            c["fault_check_ids"] for c in call_results  # type: ignore[misc]
        ),
        "legacy_refusal_format_calls": legacy_refusal_format_calls,
        "matches_recorded_verdicts": all_verdicts_match,
    }


def run_probe_only(sealed_jsonl_path: Path) -> dict[str, object]:
    """Run the shortcut probe directly on any sealed JSONL file (including writer-rendered prose)."""
    sealed_rows = [
        json.loads(line)
        for line in sealed_jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    probe_report = run_shortcut_probe(sealed_rows) if sealed_rows else _empty_shortcut_probe_report()
    return {
        "sealed_jsonl_path": str(sealed_jsonl_path),
        "num_rows": len(sealed_rows),
        "d_key_pattern_distribution": _summarize_d_key_patterns(sealed_rows),
        "probe_report": asdict(probe_report),
    }


def verify_written_dataset(
    output_dir: Path,
    *,
    checker_profile: str = DEFAULT_CHECKER_PROFILE,
) -> dict[str, object]:
    """Post-write verification command (`--verify-written <dir>`):

    1. Runs canonical `verify_sealed_part`, prose checks (a)-(d), independent B2 derivation
       from the stated rule sentence alone, and all independent recounts (`P, K, B1, B2, X, D, S, U`).
    2. Runs `--probe-only` (`run_probe_only`) on the written `sealed_store/keys.sealed.jsonl`.
    3. Validates `generations.jsonl` existence, record count (`>= num_rows`), pinned `manifest_digest`,
       `done_reason == "stop"`, and `checks_passed == True` for every kept scenario rendering, and
       verifies sealed slice SHA-256 for any refused rows without writing model sentence text into
       `verify_written_report.json`.
    """
    public_dir = output_dir / "part_public"
    sealed_dir = output_dir / "sealed_store"
    keys_sealed_path = sealed_dir / "keys.sealed.jsonl"
    responses_sealed_path = sealed_dir / SEALED_RESPONSES_FILENAME
    generations_path = output_dir / GENERATIONS_FILENAME

    recount_report = verify_written_dataset_recounts_and_prose(
        public_dir,
        sealed_dir,
        checker_profile=checker_profile,
    )
    probe_only_result = run_probe_only(keys_sealed_path)
    num_rows = int(probe_only_result["num_rows"])  # type: ignore[arg-type]

    sealed_responses_by_call_id: dict[str, dict[str, Any]] = {}
    if responses_sealed_path.is_file():
        for raw_resp_line in responses_sealed_path.read_text(encoding="utf-8").splitlines():
            if raw_resp_line.strip():
                try:
                    resp_obj = json.loads(raw_resp_line)
                    if isinstance(resp_obj, dict) and resp_obj.get("call_id"):
                        sealed_responses_by_call_id[str(resp_obj["call_id"])] = resp_obj
                except Exception:
                    pass

    generations_sha256: str | None = None
    generations_count = 0
    generations_valid = False
    generations_errors: list[str] = []

    if not generations_path.is_file():
        generations_errors.append(f"Missing required {GENERATIONS_FILENAME} in {output_dir}.")
    else:
        gen_bytes = generations_path.read_bytes()
        generations_sha256 = hashlib.sha256(gen_bytes).hexdigest()
        gen_lines = [line for line in gen_bytes.decode("utf-8").splitlines() if line.strip()]
        generations_count = len(gen_lines)
        if generations_count < num_rows:
            generations_errors.append(
                f"generations_count ({generations_count}) is less than num_rows ({num_rows})."
            )
        kept_by_rendering: dict[tuple[str, int], dict[str, Any]] = {}
        for line_idx, raw_line in enumerate(gen_lines, start=1):
            try:
                rec_obj = json.loads(raw_line)
            except Exception as exc:
                generations_errors.append(f"Invalid JSON on line {line_idx} of {GENERATIONS_FILENAME}: {exc}")
                continue
            if not isinstance(rec_obj, dict):
                generations_errors.append(f"Non-object JSON on line {line_idx} of {GENERATIONS_FILENAME}.")
                continue
            if "think" not in rec_obj or not isinstance(rec_obj.get("think"), bool):
                generations_errors.append(f"Missing boolean 'think' on line {line_idx} of {GENERATIONS_FILENAME}.")
            sid = str(rec_obj.get("scenario_id") or "")
            r_idx = rec_obj.get("rendering_index")
            if not sid or not isinstance(r_idx, int) or isinstance(r_idx, bool):
                generations_errors.append(
                    f"Missing or invalid (scenario_id, rendering_index) on line {line_idx} of {GENERATIONS_FILENAME}."
                )
            else:
                kept_by_rendering[(sid, r_idx)] = rec_obj

            got_digest = str(rec_obj.get("manifest_digest") or "")
            pins_digest = str((rec_obj.get("pins") or {}).get("manifest_digest") or "")
            if got_digest != DEFAULT_WRITER_MODEL_PIN.manifest_digest or (
                pins_digest and pins_digest != DEFAULT_WRITER_MODEL_PIN.manifest_digest
            ):
                generations_errors.append(
                    f"Line {line_idx} ({sid}, r{r_idx}) has manifest_digest={got_digest!r}, expected {DEFAULT_WRITER_MODEL_PIN.manifest_digest!r}."
                )
            if rec_obj.get("done_reason") != "stop":
                generations_errors.append(
                    f"Line {line_idx} ({sid}, r{r_idx}) has done_reason={rec_obj.get('done_reason')!r}, expected 'stop'."
                )
            if rec_obj.get("checks_passed", True) is not True or rec_obj.get("refusal_reasons"):
                parsed_meta = parse_generation_refusal_metadata(rec_obj)
                cid = str(parsed_meta["check_id"] or "render_check")
                ev_idx = parsed_meta["event_index"]
                ev_kind = parsed_meta["event_kind"]
                fail_sha = str(parsed_meta["failing_sentence_sha256"] or "")
                fail_off = parsed_meta["failing_sentence_offset"]
                fail_len = parsed_meta["failing_sentence_length"]
                call_id_str = str(rec_obj.get("call_id") or "")
                sealed_resp = sealed_responses_by_call_id.get(call_id_str)
                if parsed_meta["format"] == "sealed_offset_v1" and sealed_resp is not None:
                    resp_text = str(sealed_resp.get("response_text") or "")
                    _, slice_ok = resolve_sealed_failing_sentence(
                        resp_text,
                        failing_sentence_sha256=fail_sha,
                        failing_sentence_offset=fail_off,
                        failing_sentence_length=fail_len,
                    )
                    if not slice_ok:
                        generations_errors.append(
                            f"Line {line_idx} ({sid}, r{r_idx}) failing_sentence_sha256 mismatch at offset={fail_off}, length={fail_len} in {SEALED_RESPONSES_FILENAME}."
                        )
                sanitized_refusal = format_non_sealed_refusal_reason(
                    check_id=cid,
                    event_index=ev_idx,
                    event_kind=ev_kind,
                    failing_sentence_sha256=fail_sha,
                    failing_sentence_offset=fail_off,
                    failing_sentence_length=fail_len,
                )
                generations_errors.append(
                    f"Line {line_idx} ({sid}, r{r_idx}) did not pass prose checks (checks_passed={rec_obj.get('checks_passed')!r}, refusal={sanitized_refusal!r})."
                )

        if len(kept_by_rendering) < num_rows:
            generations_errors.append(
                f"Only {len(kept_by_rendering)} distinct (scenario_id, rendering_index) calls in {GENERATIONS_FILENAME}, expected {num_rows}."
            )
        generations_valid = len(generations_errors) == 0

    result: dict[str, object] = {
        "output_dir": str(output_dir),
        "keys_sealed_path": str(keys_sealed_path),
        "generations_path": str(generations_path) if generations_path.is_file() else None,
        "generations_sha256": generations_sha256,
        "generations_count": generations_count,
        "generations_valid": generations_valid,
        "generations_errors": generations_errors,
        "recount_verification": recount_report,
        "probe_only": probe_only_result,
        "passed": bool(
            generations_valid
            and recount_report["independent_b2_verified"]
            and recount_report["independent_recounts_verified"]
            and recount_report["prose_checks_passed"]
            and probe_only_result["probe_report"]["passed"]  # type: ignore[index]
        ),
    }
    verify_report_path = output_dir / "verify_written_report.json"
    verify_report_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def format_verify_written_console_refusals(output_dir: Path) -> list[str]:
    """Resolve and format refused calls' failing sentences from `sealed_store/responses.sealed.jsonl` for console-only `--verify-written` output."""
    generations_path = output_dir / GENERATIONS_FILENAME
    responses_sealed_path = output_dir / "sealed_store" / SEALED_RESPONSES_FILENAME
    if not generations_path.is_file() or not responses_sealed_path.is_file():
        return []
    sealed_by_call_id: dict[str, dict[str, Any]] = {}
    for line in responses_sealed_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and obj.get("call_id"):
                    sealed_by_call_id[str(obj["call_id"])] = obj
            except Exception:
                pass
    lines: list[str] = []
    for raw_line in generations_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            rec_obj = json.loads(raw_line)
        except Exception:
            continue
        if not isinstance(rec_obj, dict):
            continue
        if rec_obj.get("checks_passed", True) is True and not rec_obj.get("refusal_reasons"):
            continue
        call_id = str(rec_obj.get("call_id") or "")
        parsed = parse_generation_refusal_metadata(rec_obj)
        cid = str(parsed["check_id"] or "render_check")
        ev_idx = parsed["event_index"]
        ev_kind = parsed["event_kind"]
        fail_sha = str(parsed["failing_sentence_sha256"] or "")
        fail_off = parsed["failing_sentence_offset"]
        fail_len = parsed["failing_sentence_length"]
        sealed_entry = sealed_by_call_id.get(call_id)
        resolved_sent: str | None = None
        sha_ok = False
        if sealed_entry is not None:
            resp_text = str(sealed_entry.get("response_text") or "")
            if parsed["format"] == "sealed_offset_v1":
                resolved_sent, sha_ok = resolve_sealed_failing_sentence(
                    resp_text,
                    failing_sentence_sha256=fail_sha,
                    failing_sentence_offset=fail_off,
                    failing_sentence_length=fail_len,
                )
            elif sealed_entry.get("failing_sentence"):
                resolved_sent = str(sealed_entry["failing_sentence"])
                sha_ok = True
        lines.append(
            f"[verify-written] {call_id}: FAIL check_id={cid} event_index={ev_idx} "
            f"event_kind={ev_kind} sha256_verified={sha_ok} failing_sentence={(resolved_sent or '')!r}"
        )
    return lines


# Card ul: the exact prompt-token counter's paths come from the CLI or these environment variables, never from
# the source.
LLAMA_TOKENIZE_ENV: str = "UDLAB_LLAMA_TOKENIZE"
WRITER_GGUF_ENV: str = "UDLAB_WRITER_GGUF"


def describe_prompt_token_counter(counter: PromptTokenCounter | None) -> dict[str, object] | str | None:
    """Card ul: what sized `num_ctx`: the llama-tokenize counter's description (tool and blob names, args), the
    word 'custom' for any other callable (tests), or None when the bytes/3 estimate stood in."""
    if counter is None:
        return None
    if isinstance(counter, LlamaTokenizeCounter):
        return counter.describe()
    return "custom"


def resolve_prompt_token_counter(
    tokenize_exe: Path | None,
    gguf_path: Path | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> LlamaTokenizeCounter | None:
    """Card ul: build the exact counter from `--llama-tokenize`/`--writer-gguf` or the environment. Both or
    neither; the blob's file name must carry the pinned model-layer digest. A live `--writer ollama` run
    without one is refused by `OllamaWriter` (EXACT_COUNT_REQUIRED), the way a missing grant is."""
    env: Mapping[str, str] = os.environ if environ is None else environ
    exe = tokenize_exe if tokenize_exe is not None else (Path(env[LLAMA_TOKENIZE_ENV]) if env.get(LLAMA_TOKENIZE_ENV) else None)
    gguf = gguf_path if gguf_path is not None else (Path(env[WRITER_GGUF_ENV]) if env.get(WRITER_GGUF_ENV) else None)
    if exe is None and gguf is None:
        return None
    if exe is None or gguf is None:
        raise ValueError("--llama-tokenize and --writer-gguf go together (both or neither).")
    counter = LlamaTokenizeCounter(exe, gguf)
    counter.verify_pinned_digest(DEFAULT_WRITER_MODEL_PIN.model_layer_digest)
    return counter


def build_and_probe_dataset(
    *,
    count: int = 300,
    base_seed: int = 20261030,
    private_seed_hex: str | None = None,
    output_dir: Path,
    writer_kind: str = "mock",
    vram_halt_mib: int = DEFAULT_VRAM_HALT_MIB,
    num_predict: int | None = None,
    max_num_ctx: int = DEFAULT_WRITER_MAX_NUM_CTX,
    think: bool = DEFAULT_WRITER_THINK,
    max_retries: int = 0,
    retry_form: str = DEFAULT_RETRY_FORM,
    multi_turn_retries: bool = False,
    short_circuit_dead_scenarios: bool = False,
    goal_mode: str = DEFAULT_GOAL_MODE,
    constraint_layout: str = DEFAULT_CONSTRAINT_LAYOUT,
    accepted_forms_line: str = DEFAULT_ACCEPTED_FORMS_LINE,
    checker_profile: str = DEFAULT_CHECKER_PROFILE,
    shape_plan: str = DEFAULT_SHAPE_PLAN,
    shape_plan_seed: int | None = None,
    presence_penalty: float | None = None,
    grant_id: str | None = None,
    resume: bool = False,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    server_port: int | None = None,
    start_owned_server: bool | None = None,
    http_transport: OllamaHttpTransport | None = None,
    vram_probe: Callable[[], int] | None = None,
    halt_pid: int | None = None,
    pid_killer: Callable[[int], None] = kill_pid_tree,
    popen_factory: Callable[..., Any] | None = None,
    prompt_token_counter: PromptTokenCounter | None = None,
) -> dict[str, object]:
    """Generate `count` scenarios, render via `mock` or `ollama` writer with checkpointing, seal, verify, and probe.

    Card ul: `prompt_token_counter` (llama-tokenize on CPU against the pinned blob) gives the exact prompt-token
    count that decides `num_ctx` and the cap check; without it the bytes/3 estimate stands in (CPU tests)."""
    if writer_kind not in ("mock", "ollama"):
        raise ValueError(f"writer_kind must be 'mock' or 'ollama', got {writer_kind!r}.")
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}.")
    resolved_retry_form = "multi_turn" if multi_turn_retries else str(retry_form)
    if resolved_retry_form not in VALID_RETRY_FORMS:
        raise ValueError(
            f"retry_form must be one of {sorted(VALID_RETRY_FORMS)}, got {resolved_retry_form!r}."
        )
    if goal_mode not in VALID_GOAL_MODES:
        raise ValueError(
            f"goal_mode must be one of {sorted(VALID_GOAL_MODES)}, got {goal_mode!r}."
        )
    if constraint_layout not in VALID_CONSTRAINT_LAYOUTS:
        raise ValueError(
            f"constraint_layout must be one of {sorted(VALID_CONSTRAINT_LAYOUTS)}, got {constraint_layout!r}."
        )
    if accepted_forms_line not in VALID_ACCEPTED_FORMS_LINES:
        raise ValueError(
            f"accepted_forms_line must be one of {sorted(VALID_ACCEPTED_FORMS_LINES)}, got {accepted_forms_line!r}."
        )
    if checker_profile not in VALID_CHECKER_PROFILES:
        raise ValueError(
            f"checker_profile must be one of {sorted(VALID_CHECKER_PROFILES)}, got {checker_profile!r}."
        )
    if shape_plan not in VALID_SHAPE_PLANS:
        raise ValueError(
            f"shape_plan must be one of {sorted(VALID_SHAPE_PLANS)}, got {shape_plan!r}."
        )
    resolved_goal_mode = str(goal_mode)
    resolved_constraint_layout = str(constraint_layout)
    resolved_accepted_forms_line = str(accepted_forms_line)
    resolved_checker_profile = str(checker_profile)
    resolved_shape_plan = str(shape_plan)
    resolved_shape_plan_seed = (
        int(shape_plan_seed) if shape_plan_seed is not None else int(base_seed)
    )
    resolved_presence_penalty = float(presence_penalty) if presence_penalty is not None else None
    resolved_short_circuit = bool(short_circuit_dead_scenarios)
    dead_scenario_mode = (
        "short_circuit" if resolved_short_circuit else "continue_other_rendering"
    )
    if max_num_ctx > DEFAULT_WRITER_MAX_NUM_CTX:
        raise MaxNumCtxRefusalError(
            f"REFUSED (MAX_NUM_CTX_CAP): max_num_ctx={max_num_ctx} exceeds the hard VRAM ceiling limit "
            f"{DEFAULT_WRITER_MAX_NUM_CTX}."
        )
    resolved_think = bool(think)
    resolved_num_predict = (
        int(num_predict)
        if num_predict is not None
        else (DEFAULT_WRITER_THINK_NUM_PREDICT if resolved_think else DEFAULT_WRITER_NUM_PREDICT)
    )
    effective_writer_pin = replace(
        DEFAULT_WRITER_MODEL_PIN,
        num_predict=resolved_num_predict,
        max_num_ctx=int(max_num_ctx),
        think=resolved_think,
    )

    configs = make_balanced_configs(count, base_seed=base_seed)
    worlds = [generate_world(cfg) for cfg in configs]
    plan_bundle = generate_shape_plan(
        worlds,
        seed=resolved_shape_plan_seed,
        block_size=SHAPE_PLAN_BLOCK_SIZE,
    )
    plan_by_sid: dict[str, list[dict[str, object]]] = plan_bundle["plan_by_scenario_id"]  # type: ignore[assignment]

    output_dir.mkdir(parents=True, exist_ok=True)
    public_dir = output_dir / "part_public"
    sealed_dir = output_dir / "sealed_store"
    generations_path = output_dir / GENERATIONS_FILENAME
    sealed_responses_path = sealed_dir / SEALED_RESPONSES_FILENAME

    initial_private_seed = (
        private_seed_hex.strip().lower()
        if private_seed_hex
        else secrets.token_hex(32)
    )
    checkpoint = WriterCheckpointManager(
        output_dir=output_dir,
        count=count,
        base_seed=base_seed,
        private_seed_hex=initial_private_seed,
        writer_kind=writer_kind,
        writer_pin=effective_writer_pin,
        short_circuit_dead_scenarios=resolved_short_circuit,
        retry_form=resolved_retry_form,
        goal_mode=resolved_goal_mode,
        presence_penalty=resolved_presence_penalty,
        constraint_layout=resolved_constraint_layout,
        accepted_forms_line=resolved_accepted_forms_line,
        checker_profile=resolved_checker_profile,
        shape_plan=resolved_shape_plan,
        shape_plan_seed=resolved_shape_plan_seed,
    )

    if resume:
        restored_pairs, restored_calls, resolved_private_seed = checkpoint.load_and_verify(
            worlds=worlds
        )
        if private_seed_hex and private_seed_hex.strip().lower() != resolved_private_seed:
            raise ValueError(
                "Explicit --private-seed differs from the checkpoint's sealed private_seed_hex."
            )
    else:
        restored_pairs = []
        restored_calls = []
        resolved_private_seed = initial_private_seed
        if generations_path.exists():
            generations_path.unlink()
        if sealed_responses_path.exists():
            sealed_responses_path.unlink()
        checkpoint.initialize_fresh()

    rendered_pairs: list[tuple[RenderedScenario, RenderedScenario]] = list(restored_pairs)
    completed_count = checkpoint.completed_count

    if writer_kind == "ollama":
        should_own_server = (
            start_owned_server
            if start_owned_server is not None
            else (http_transport is None)
        )
        ollama_writer = OllamaWriter(
            model_pin=effective_writer_pin,
            base_url=ollama_base_url,
            server_port=server_port,
            num_predict=resolved_num_predict,
            max_num_ctx=max_num_ctx,
            think=resolved_think,
            vram_halt_mib=vram_halt_mib,
            grant_id=grant_id,
            http_transport=http_transport,
            vram_probe=vram_probe,
            halt_pid=halt_pid,
            pid_killer=pid_killer,
            generations_path=generations_path,
            sealed_responses_path=sealed_responses_path,
            max_retries=max_retries,
            retry_form=resolved_retry_form,
            short_circuit_dead_scenarios=resolved_short_circuit,
            goal_mode=resolved_goal_mode,
            constraint_layout=resolved_constraint_layout,
            accepted_forms_line=resolved_accepted_forms_line,
            checker_profile=resolved_checker_profile,
            shape_plan=resolved_shape_plan,
            shape_plan_seed=resolved_shape_plan_seed,
            presence_penalty=resolved_presence_penalty,
            start_owned_server=should_own_server,
            popen_factory=popen_factory,
            prompt_token_counter=prompt_token_counter,
        )
        if resume:
            ollama_writer.restore_call_records(restored_calls)
        try:
            for idx in range(completed_count, len(worlds)):
                w = worlds[idx]
                prev_calls = len(ollama_writer.call_records)
                pair = ollama_writer.render_scenario_pair(
                    w,
                    allow_refusal=True,
                    short_circuit_dead_scenarios=resolved_short_circuit,
                    scenario_shape_plan=plan_by_sid.get(w.scenario_id),
                )
                checkpoint.set_modelfile_parameters(ollama_writer.modelfile_parameters)
                new_calls = ollama_writer.call_records[prev_calls:]
                if pair is not None:
                    rendered_pairs.append(pair)
                checkpoint.append_scenario(
                    scenario_index=idx,
                    pair=pair,
                    call_records=new_calls,
                    world=w,
                )
        finally:
            ollama_writer.close()
        all_call_records: Sequence[GenerationCallRecord] = ollama_writer.call_records
        peak_vram_mib = ollama_writer.peak_vram_mib
        effective_grant_id = ollama_writer.grant_id
        modelfile_parameters = ollama_writer.modelfile_parameters
        server_info = (
            ollama_writer.owned_server.to_dict()
            if ollama_writer.owned_server is not None
            else None
        )
    else:
        mock_writer = MockWriter(
            goal_mode=resolved_goal_mode,
            constraint_layout=resolved_constraint_layout,
            checker_profile=resolved_checker_profile,
            accepted_forms_line=resolved_accepted_forms_line,
            shape_plan=resolved_shape_plan,
            shape_plan_seed=resolved_shape_plan_seed,
        )
        all_mock_calls: list[GenerationCallRecord] = list(restored_calls)
        if resume and generations_path is not None:
            lines = "".join(
                json.dumps(r.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
                for r in all_mock_calls
            )
            generations_path.write_text(lines, encoding="utf-8", newline="\n")

        for idx in range(completed_count, len(worlds)):
            w = worlds[idx]
            scen_plan = plan_by_sid.get(w.scenario_id)
            t0 = time.perf_counter()
            pair = mock_writer.render_scenario_pair(
                w,
                scenario_shape_plan=scen_plan,
            )
            if resolved_presence_penalty is not None:
                pair = (
                    replace(pair[0], presence_penalty=resolved_presence_penalty),
                    replace(pair[1], presence_penalty=resolved_presence_penalty),
                )
            elapsed = max(0.0, time.perf_counter() - t0)
            scen_calls: list[GenerationCallRecord] = []
            for r_idx, rend in enumerate(pair):
                call_index = len(all_mock_calls) + 1
                call_id = f"{w.scenario_id}:r{r_idx}:call_{call_index:04d}"
                call_seed = derive_call_seed(int(w.seed), r_idx, 0)
                messages = [{"role": "user", "content": rend.writer_prompt}]
                prompt_sha = hashlib.sha256(
                    json.dumps(messages, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                raw_mock_response = mock_writer.render_raw_response(
                    w,
                    rend.focal_agent,
                )
                resp_sha, think_sha = compute_sealed_response_digests(raw_mock_response, "")
                p_est = estimate_prompt_tokens(messages)
                # Card ul: the exact count decides num_ctx when a tokenizer is configured; the estimate otherwise.
                p_exact = (
                    count_prompt_tokens_exact(messages, think=resolved_think, counter=prompt_token_counter)
                    if prompt_token_counter is not None
                    else None
                )
                n_ctx = decide_num_ctx(
                    p_exact if p_exact is not None else p_est,
                    resolved_num_predict,
                    max_num_ctx=max_num_ctx,
                    margin=NUM_CTX_MARGIN_TOKENS,
                    step=NUM_CTX_STEP_TOKENS,
                )
                rec = GenerationCallRecord(
                    call_id=call_id,
                    call_index=call_index,
                    scenario_id=w.scenario_id,
                    rendering_index=r_idx,
                    focal_agent=rend.focal_agent,
                    operation="render_scenario_mock",
                    model=effective_writer_pin.ollama_tag,
                    manifest_digest=effective_writer_pin.manifest_digest,
                    model_layer_digest=effective_writer_pin.model_layer_digest,
                    licence_id=effective_writer_pin.licence_id,
                    prompt_sha256=prompt_sha,
                    prompt_tokens_estimate=p_est,
                    prompt_tokens_exact=p_exact,
                    prompt_eval_count=p_exact if p_exact is not None else p_est,
                    eval_count=len(raw_mock_response.split()),
                    thinking_words=0,
                    num_predict=resolved_num_predict,
                    num_ctx=n_ctx,
                    max_num_ctx=max_num_ctx,
                    think=resolved_think,
                    temperature=ATTEMPT_0_TEMPERATURE,
                    min_p=None,
                    done_reason="stop",
                    seconds=round(elapsed / 2.0, 6),
                    vram_peak_mib=0,
                    vram_halt_mib=vram_halt_mib,
                    grant_id="cpu-mock",
                    checks_passed=True,
                    refusal_reasons=(),
                    attempt=0,
                    seed=call_seed,
                    response_sha256=resp_sha,
                    thinking_sha256=think_sha,
                    goal_mode=resolved_goal_mode,
                    presence_penalty=resolved_presence_penalty,
                    constraint_layout=resolved_constraint_layout,
                )
                all_mock_calls.append(rec)
                scen_calls.append(rec)
                with generations_path.open("a", encoding="utf-8", newline="\n") as fh:
                    fh.write(json.dumps(rec.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")
                append_sealed_response(
                    sealed_responses_path,
                    SealedResponseEntry(
                        call_id=call_id,
                        call_index=call_index,
                        scenario_id=w.scenario_id,
                        rendering_index=r_idx,
                        focal_agent=rend.focal_agent,
                        attempt=0,
                        seed=call_seed,
                        temperature=ATTEMPT_0_TEMPERATURE,
                        min_p=None,
                        retry_form="",
                        response_text=raw_mock_response,
                        thinking_text="",
                        response_sha256=resp_sha,
                        thinking_sha256=think_sha,
                        goal_mode=resolved_goal_mode,
                        presence_penalty=resolved_presence_penalty,
                        constraint_layout=resolved_constraint_layout,
                    ),
                )

            rendered_pairs.append(pair)
            checkpoint.append_scenario(
                scenario_index=idx,
                pair=pair,
                call_records=scen_calls,
                world=w,
            )
        all_call_records = all_mock_calls
        peak_vram_mib = 0
        effective_grant_id = "cpu-mock"
        modelfile_parameters = None
        server_info = None

    diagnostics_error: str | None = None
    diagnostics_summary: dict[str, object] | None = None
    diagnostics_sealed_sha256: str | None = None
    attempt0_responses_by_rend: dict[tuple[str, int], str] = {}
    final_responses_by_rend: dict[tuple[str, int], str] = {}
    try:
        worlds_by_sid = {w.scenario_id: w for w in worlds}
        diag_records: list[EventDiagnosticRecord] = []
        if sealed_responses_path.is_file():
            final_sealed_by_rend: dict[tuple[str, int], dict[str, Any]] = {}
            for sr_line in sealed_responses_path.read_text(encoding="utf-8").splitlines():
                if sr_line.strip():
                    sr_obj = json.loads(sr_line)
                    if isinstance(sr_obj, dict):
                        sid_key = str(sr_obj["scenario_id"])
                        r_idx_key = int(sr_obj["rendering_index"])
                        resp_txt = str(sr_obj.get("response_text") or "")
                        if int(sr_obj.get("attempt", 0)) == 0 or (sid_key, r_idx_key) not in attempt0_responses_by_rend:
                            attempt0_responses_by_rend[(sid_key, r_idx_key)] = resp_txt
                        final_responses_by_rend[(sid_key, r_idx_key)] = resp_txt
                        final_sealed_by_rend[(sid_key, r_idx_key)] = sr_obj
            for (sid, r_idx), sr_obj in final_sealed_by_rend.items():
                if sid in worlds_by_sid:
                    diag_records.extend(
                        diagnose_rendered_prose(
                            str(sr_obj.get("response_text") or ""),
                            worlds_by_sid[sid],
                            rendering_index=r_idx,
                            focal_agent=str(sr_obj.get("focal_agent") or ""),
                            goal_mode=str(sr_obj.get("goal_mode") or resolved_goal_mode),
                            checker_profile=resolved_checker_profile,
                        )
                    )
        sealed_diagnostics_path = sealed_dir / SEALED_DIAGNOSTICS_FILENAME
        write_sealed_diagnostics(sealed_diagnostics_path, diag_records)
        diagnostics_bytes = (
            sealed_diagnostics_path.read_bytes() if sealed_diagnostics_path.is_file() else b""
        )
        diagnostics_sealed_sha256 = hashlib.sha256(diagnostics_bytes).hexdigest()
        diagnostics_summary = summarize_diagnostics_public(diag_records)
    except Exception as diag_exc:
        diagnostics_error = type(diag_exc).__name__
        diagnostics_summary = None
        diagnostics_sealed_sha256 = None

    shape_plan_telemetry = compute_shape_plan_run_telemetry(
        worlds,
        shape_plan=resolved_shape_plan,
        shape_plan_seed=resolved_shape_plan_seed,
        plan_bundle=plan_bundle,
        attempt0_prose_by_rendering=attempt0_responses_by_rend,
        final_prose_by_rendering=final_responses_by_rend,
        goal_mode=resolved_goal_mode,
        checker_profile=resolved_checker_profile,
    )

    generations_bytes = generations_path.read_bytes() if generations_path.is_file() else b""
    generations_sha256 = hashlib.sha256(generations_bytes).hexdigest()
    responses_bytes = sealed_responses_path.read_bytes() if sealed_responses_path.is_file() else b""
    responses_sealed_sha256 = hashlib.sha256(responses_bytes).hexdigest()

    outcomes = _summarize_call_outcomes(all_call_records)
    hard_constraints_sha256 = compute_hard_constraints_sha256(
        goal_mode=resolved_goal_mode,
        constraint_layout=resolved_constraint_layout,
        accepted_forms_line=resolved_accepted_forms_line,
        shape_plan=resolved_shape_plan,
    )

    writer_provenance: dict[str, Any] = {
        "writer": writer_kind,
        "writer_pin": effective_writer_pin.to_dict(),
        "goal_mode": resolved_goal_mode,
        "constraint_layout": resolved_constraint_layout,
        "accepted_forms_line": resolved_accepted_forms_line,
        "checker_profile": resolved_checker_profile,
        "shape_plan": resolved_shape_plan,
        "shape_plan_seed": resolved_shape_plan_seed,
        "shape_plan_telemetry": shape_plan_telemetry,
        "hard_constraints_sha256": hard_constraints_sha256,
        "presence_penalty": resolved_presence_penalty,
        "modelfile_parameters": modelfile_parameters,
        "think": resolved_think,
        "num_predict": resolved_num_predict,
        "max_num_ctx": max_num_ctx,
        "max_retries": max_retries,
        "retry_form": resolved_retry_form,
        "short_circuit_dead_scenarios": resolved_short_circuit,
        "dead_scenario_mode": dead_scenario_mode,
        "vram_halt_mib": vram_halt_mib,
        "grant_id": effective_grant_id,
        "prompt_token_counter": describe_prompt_token_counter(prompt_token_counter),
        "generations_file": GENERATIONS_FILENAME,
        "generations_sha256": generations_sha256,
        "responses_sealed_file": SEALED_RESPONSES_FILENAME,
        "responses_sealed_sha256": responses_sealed_sha256,
        "diagnostics_sealed_file": SEALED_DIAGNOSTICS_FILENAME,
        "diagnostics_sealed_sha256": diagnostics_sealed_sha256,
        "diagnostics_error": diagnostics_error,
        "checkpoint_chain_head_sha256": checkpoint.chain_head_sha256,
    }

    receipt = write_sealed_dataset(
        rendered_pairs=rendered_pairs,
        public_dir=public_dir,
        sealed_dir=sealed_dir,
        private_seed_hex=resolved_private_seed,
        writer_provenance=writer_provenance,
        all_worlds=worlds,
        allow_empty=True,
    )
    recount_verification = verify_written_dataset_recounts_and_prose(
        public_dir,
        sealed_dir,
        checker_profile=resolved_checker_profile,
    )

    sealed_rows = [
        json.loads(line)
        for line in receipt.keys_sealed_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    probe_report = (
        run_shortcut_probe(sealed_rows)
        if sealed_rows
        else _empty_shortcut_probe_report()
    )
    diversity_report = count_template_diversity(worlds)
    d_distribution = _summarize_d_key_patterns(sealed_rows)

    summary: dict[str, object] = {
        "writer": writer_kind,
        "writer_pin": effective_writer_pin.to_dict(),
        "goal_mode": resolved_goal_mode,
        "constraint_layout": resolved_constraint_layout,
        "accepted_forms_line": resolved_accepted_forms_line,
        "checker_profile": resolved_checker_profile,
        "shape_plan": resolved_shape_plan,
        "shape_plan_seed": resolved_shape_plan_seed,
        "shape_plan_telemetry": shape_plan_telemetry,
        "two_form_switch_rate_band": shape_plan_telemetry["two_form_switch_rate_band"],
        "block_balance_max_spread": shape_plan_telemetry["block_balance_max_spread"],
        "plan_records": shape_plan_telemetry["plan_records"],
        "lag1_persistence_by_kind": shape_plan_telemetry["lag1_persistence_by_kind"],
        "compliance_by_kind": shape_plan_telemetry["compliance_by_kind"],
        "compliance_by_position_tercile": shape_plan_telemetry["compliance_by_position_tercile"],
        "confusion_matrix_by_kind": shape_plan_telemetry["confusion_matrix_by_kind"],
        "outcomes_by_assigned_form": shape_plan_telemetry["outcomes_by_assigned_form"],
        "hard_constraints_sha256": hard_constraints_sha256,
        "presence_penalty": resolved_presence_penalty,
        "modelfile_parameters": modelfile_parameters,
        "max_retries": max_retries,
        "retry_form": resolved_retry_form,
        "short_circuit_dead_scenarios": resolved_short_circuit,
        "dead_scenario_mode": dead_scenario_mode,
        "grant_id": effective_grant_id,
        "prompt_token_counter": describe_prompt_token_counter(prompt_token_counter),
        "vram_halt_mib": vram_halt_mib,
        "vram_peak_mib": peak_vram_mib,
        "ollama_server": server_info,
        "num_scenarios": receipt.num_scenarios,
        "num_items": receipt.num_items,
        "duplicate_stake_statements": int(recount_verification.get("duplicate_stake_statements", 0)),
        **outcomes,
        "items_sha256": receipt.items_sha256,
        "part_json_sha256": receipt.part_json_sha256,
        "keys_sealed_sha256": receipt.keys_sealed_sha256,
        "generations_file": GENERATIONS_FILENAME,
        "generations_sha256": generations_sha256,
        "responses_sealed_file": SEALED_RESPONSES_FILENAME,
        "responses_sealed_sha256": responses_sealed_sha256,
        "diagnostics_sealed_file": SEALED_DIAGNOSTICS_FILENAME,
        "diagnostics_sealed_sha256": diagnostics_sealed_sha256,
        "diagnostics_error": diagnostics_error,
        "diagnostics_summary": diagnostics_summary,
        "checkpoint_chain_head_sha256": checkpoint.chain_head_sha256,
        "recount_verification": recount_verification,
        "d_key_pattern_distribution": d_distribution,
        "probe_report": asdict(probe_report),
        "diversity_report": {
            "total_scenarios": diversity_report.total_scenarios,
            "distinct_event_skeletons": diversity_report.distinct_event_skeletons,
            "distinct_domain_skeletons": diversity_report.distinct_domain_skeletons,
            "max_cluster_size": diversity_report.max_cluster_size,
            "mean_cluster_size": diversity_report.mean_cluster_size,
        },
    }
    summary_path = output_dir / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(
    argv: Sequence[str] | None = None,
    *,
    http_transport: OllamaHttpTransport | None = None,
    vram_probe: Callable[[], int] | None = None,
    popen_factory: Callable[..., Any] | None = None,
) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == "diagnose-paired":
        raw_argv = ["--diagnose-paired", *raw_argv[1:]]

    parser = argparse.ArgumentParser(
        prog="udlab-gen",
        description="Generate, write, seal, verify, and probe udlab release-1 cognitive-empathy scenarios.",
    )
    parser.add_argument("--count", type=int, default=300, help="Number of scenarios to generate (default: 300).")
    parser.add_argument("--seed", type=int, default=20261030, help="Base deterministic generation seed.")
    parser.add_argument(
        "--private-seed",
        type=str,
        default=None,
        help="Optional 256-bit hex private shuffle seed (if omitted, reads UDLAB_PRIVATE_SEED or generates fresh secrets.token_hex(32)).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/r1_300"),
        help="Output directory containing part_public/ and sealed_store/.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Run the scenario writer pass (paired with --writer ollama|mock).",
    )
    parser.add_argument(
        "--writer",
        choices=("ollama", "mock"),
        default="mock",
        help="Writer adapter to use: 'ollama' (pinned qwen3.6:35b) or 'mock' (CPU template, default).",
    )
    parser.add_argument(
        "--vram-halt-mib",
        type=int,
        default=DEFAULT_VRAM_HALT_MIB,
        help=f"VRAM ceiling in MiB above which the owned Ollama server PID tree is killed (default: {DEFAULT_VRAM_HALT_MIB}).",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=None,
        help=f"Total num_predict budget per writer call (default: {DEFAULT_WRITER_NUM_PREDICT} without --think, {DEFAULT_WRITER_THINK_NUM_PREDICT} with --think).",
    )
    parser.add_argument(
        "--max-num-ctx",
        type=int,
        default=DEFAULT_WRITER_MAX_NUM_CTX,
        help=f"Maximum allowed num_ctx for writer calls (default and hard cap: {DEFAULT_WRITER_MAX_NUM_CTX}).",
    )
    parser.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_WRITER_THINK,
        help="Enable thinking mode for the writer pass (default: off / think=False).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Number of repair retries per refused rendering (default: 0).",
    )
    parser.add_argument(
        "--retry-form",
        choices=("single_turn", "multi_turn"),
        default=DEFAULT_RETRY_FORM,
        help=f"Repair retry prompt form (default: {DEFAULT_RETRY_FORM}).",
    )
    parser.add_argument(
        "--multi-turn-retries",
        action="store_true",
        default=False,
        help="Explicitly opt into multi-turn repair retries that include the refused draft (default: off).",
    )
    parser.add_argument(
        "--short-circuit-dead-scenarios",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Stop rendering a scenario as soon as one rendering has failed all its attempts "
            "(default: off / continue writing and checking the other rendering)."
        ),
    )
    parser.add_argument(
        "--goal-mode",
        choices=("copy", "splice"),
        default=DEFAULT_GOAL_MODE,
        help=f"Writer goal mode: 'copy' (default) or 'splice' ([[GOAL:<agent>]] tokens expanded before checks).",
    )
    parser.add_argument(
        "--constraint-layout",
        choices=tuple(sorted(VALID_CONSTRAINT_LAYOUTS)),
        default=DEFAULT_CONSTRAINT_LAYOUT,
        help=(
            f"Writer prompt Paragraph 2 event-order constraint layout: "
            f"'line' (default, 11 per-event-kind bullets) or 'table' (compact 8-row pipe table + 2 bullets)."
        ),
    )
    parser.add_argument(
        "--accepted-forms-line",
        choices=tuple(sorted(VALID_ACCEPTED_FORMS_LINES)),
        default=DEFAULT_ACCEPTED_FORMS_LINE,
        help=(
            f"Append four accepted-forms lines for say, move, see (occluded), and actor phrase to the writer prompt "
            f"(default: {DEFAULT_ACCEPTED_FORMS_LINE})."
        ),
    )
    parser.add_argument(
        "--checker-profile",
        choices=tuple(sorted(VALID_CHECKER_PROFILES)),
        default=DEFAULT_CHECKER_PROFILE,
        help=(
            f"Checker profile: 'c' (default, Stage 2 rule surface + Corrections 1-3) "
            f"or 'd' (Profile C + Widenings M2, M3, V2, P1)."
        ),
    )
    parser.add_argument(
        "--shape-plan",
        choices=("off", "assigned", "wildcard"),
        default=DEFAULT_SHAPE_PLAN,
        help=(
            f"Stage 4 shape-plan mode: 'off' (default), 'assigned' (balanced per-row form tags), "
            f"or 'wildcard' (token-matched wildcard tags)."
        ),
    )
    parser.add_argument(
        "--shape-plan-seed",
        type=int,
        default=None,
        help="Optional deterministic seed for Stage 4 shape-plan assignment (defaults to bank/base seed).",
    )
    parser.add_argument(
        "--grant-id",
        type=str,
        default=None,
        help="Publisher GPU grant ID required when running --writer ollama.",
    )
    parser.add_argument(
        "--llama-tokenize",
        type=Path,
        default=None,
        help=(
            f"Path to llama.cpp's llama-tokenize executable (or env {LLAMA_TOKENIZE_ENV}). With --writer-gguf it "
            "gives the exact prompt-token count (CPU, -ngl 0) that sizes num_ctx; required for --writer ollama."
        ),
    )
    parser.add_argument(
        "--writer-gguf",
        type=Path,
        default=None,
        help=(
            f"Path to the pinned writer GGUF blob, named sha256-<model_layer_digest> in the Ollama blobs dir "
            f"(or env {WRITER_GGUF_ENV})."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted writer build from the verified per-scenario sealed checkpoint.",
    )
    parser.add_argument(
        "--recheck",
        type=Path,
        default=None,
        help="Offline recheck: re-run every prose check on saved texts in <dir> with no model and no GPU.",
    )
    parser.add_argument(
        "--verify-written",
        type=Path,
        default=None,
        help="Run post-write verification (independent B2 derivation, all recounts, prose checks, and --probe-only) on a written output directory.",
    )
    parser.add_argument(
        "--probe-only",
        type=Path,
        default=None,
        help="Run the shortcut probe directly on an existing sealed JSONL file and exit.",
    )
    parser.add_argument(
        "--paired-seed-bank",
        type=Path,
        default=None,
        help="Run attempt-0 only (max_retries=0) on a fixed JSON list of scenario seeds in --goal-mode copy|splice.",
    )
    parser.add_argument(
        "--paired-compare",
        nargs=2,
        type=Path,
        default=None,
        metavar=("RUN_A", "RUN_B"),
        help="Pair two --paired-seed-bank outputs by (scenario_seed, rendering_index) and report McNemar stats.",
    )
    parser.add_argument(
        "--paired-out",
        type=Path,
        default=None,
        help="Optional output JSON path for --paired-seed-bank, --paired-compare, or --diagnose-paired.",
    )
    parser.add_argument(
        "--diagnose-paired",
        type=Path,
        default=None,
        help="Run the diagnostic alignment pass offline over a sealed paired run (CPU only, no model).",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=None,
        help="Pin the port for the owned Ollama server.",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help="Optional presence_penalty passed in Ollama request options (omitted when not set).",
    )
    args = parser.parse_args(raw_argv)

    try:
        if args.diagnose_paired is not None:
            diag_summary = diagnose_paired_run(
                args.diagnose_paired,
                public_summary_out=args.paired_out,
                checker_profile=args.checker_profile,
            )
            print(json.dumps(diag_summary, indent=2, sort_keys=True))
            return 0

        if args.paired_compare is not None:
            comp = compare_paired_runs(
                args.paired_compare[0],
                args.paired_compare[1],
                output_path=args.paired_out,
            )
            print(json.dumps(comp, indent=2, sort_keys=True))
            return 0

        if args.paired_seed_bank is not None:
            try:
                prompt_token_counter = resolve_prompt_token_counter(args.llama_tokenize, args.writer_gguf)
            except (ValueError, FileNotFoundError) as exc:
                parser.error(str(exc))
            paired_base_dir = args.paired_out.parent if args.paired_out is not None else args.out_dir
            paired_sealed_responses_path = paired_base_dir / "sealed_store" / SEALED_RESPONSES_FILENAME
            paired_res = run_paired_seed_bank(
                seed_bank=args.paired_seed_bank,
                goal_mode=args.goal_mode,
                constraint_layout=args.constraint_layout,
                accepted_forms_line=args.accepted_forms_line,
                checker_profile=args.checker_profile,
                shape_plan=args.shape_plan,
                shape_plan_seed=args.shape_plan_seed,
                presence_penalty=args.presence_penalty,
                output_path=args.paired_out,
                sealed_responses_path=paired_sealed_responses_path,
                server_port=args.server_port,
                num_predict=args.num_predict,
                max_num_ctx=args.max_num_ctx,
                think=bool(args.think),
                vram_halt_mib=args.vram_halt_mib,
                grant_id=args.grant_id,
                http_transport=http_transport,
                vram_probe=vram_probe,
                popen_factory=popen_factory,
                prompt_token_counter=prompt_token_counter,
            )
            print(json.dumps(paired_res, indent=2, sort_keys=True))
            return 0

        if args.recheck is not None:
            recheck_res = recheck_saved_responses(
                args.recheck,
                checker_profile=args.checker_profile,
            )
            for call_info in recheck_res["calls"]:  # type: ignore[union-attr]
                status = "PASS" if call_info["checks_passed"] else "FAIL"
                cid = call_info["check_id"] or "ok"
                sent = call_info["failing_sentence"] or ""
                print(
                    f"[recheck] {call_info['call_id']} (attempt={call_info['attempt']}, seed={call_info['seed']}): "
                    f"{status} check_id={cid} failing_sentence={sent!r}"
                )
            print(json.dumps(recheck_res, indent=2, sort_keys=True))
            return 0

        if args.verify_written is not None:
            verify_res = verify_written_dataset(
                args.verify_written,
                checker_profile=args.checker_profile,
            )
            for console_line in format_verify_written_console_refusals(args.verify_written):
                print(console_line)
            passed = bool(verify_res["passed"])
            print(json.dumps(verify_res, indent=2, sort_keys=True))
            return 0 if passed else 1

        if args.probe_only is not None:
            result = run_probe_only(args.probe_only)
            probe_passed = bool(result["probe_report"]["passed"])  # type: ignore[index]
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if probe_passed else 1

        resolved_private_seed = args.private_seed or os.environ.get("UDLAB_PRIVATE_SEED") or None
        try:
            prompt_token_counter = resolve_prompt_token_counter(args.llama_tokenize, args.writer_gguf)
        except (ValueError, FileNotFoundError) as exc:
            parser.error(str(exc))
        summary = build_and_probe_dataset(
            count=args.count,
            base_seed=args.seed,
            private_seed_hex=resolved_private_seed,
            output_dir=args.out_dir,
            writer_kind=args.writer,
            vram_halt_mib=args.vram_halt_mib,
            num_predict=args.num_predict,
            max_num_ctx=args.max_num_ctx,
            think=bool(args.think),
            max_retries=args.max_retries,
            retry_form=args.retry_form,
            multi_turn_retries=bool(args.multi_turn_retries),
            short_circuit_dead_scenarios=bool(args.short_circuit_dead_scenarios),
            goal_mode=args.goal_mode,
            constraint_layout=args.constraint_layout,
            accepted_forms_line=args.accepted_forms_line,
            checker_profile=args.checker_profile,
            shape_plan=args.shape_plan,
            shape_plan_seed=args.shape_plan_seed,
            presence_penalty=args.presence_penalty,
            grant_id=args.grant_id,
            resume=args.resume,
            server_port=args.server_port,
            http_transport=http_transport,
            vram_probe=vram_probe,
            popen_factory=popen_factory,
            prompt_token_counter=prompt_token_counter,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except RenderCheckError as exc:
        print(
            format_non_sealed_refusal_reason(
                check_id=exc.check_id,
                event_index=exc.event_index,
                event_kind=exc.event_kind,
                failing_sentence_sha256=exc.failing_sentence_sha256,
                failing_sentence_offset=exc.failing_sentence_offset,
                failing_sentence_length=exc.failing_sentence_length,
            ),
            file=sys.stderr,
        )
        return 1
    except SealingError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
