"""CLI entrypoint for the udlab release-1 scenario generator, writer pass, sealing, and shortcut probe."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Sequence
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
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_VRAM_HALT_MIB,
    DEFAULT_WRITER_MAX_NUM_CTX,
    DEFAULT_WRITER_MODEL_PIN,
    DEFAULT_WRITER_NUM_PREDICT,
    DEFAULT_WRITER_THINK,
    DEFAULT_WRITER_THINK_NUM_PREDICT,
    GENERATIONS_FILENAME,
    NUM_CTX_MARGIN_TOKENS,
    NUM_CTX_STEP_TOKENS,
    SEALED_RESPONSES_FILENAME,
    GenerationCallRecord,
    MaxNumCtxRefusalError,
    MockWriter,
    OllamaHttpTransport,
    OllamaWriter,
    RenderCheckError,
    RenderedScenario,
    SealedResponseEntry,
    WriterCheckpointManager,
    append_sealed_response,
    check_rendered_prose,
    compute_num_ctx,
    compute_sealed_response_digests,
    estimate_prompt_tokens,
    format_non_sealed_refusal_reason,
    kill_pid_tree,
    locate_failing_sentence_in_response,
    parse_generation_refusal_metadata,
    resolve_sealed_failing_sentence,
)
from udlab_generator.sealing import (
    SealingError,
    _deserialize_world,
    verify_sealed_part,
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
    }


def recheck_saved_responses(output_dir: Path) -> dict[str, object]:
    """Re-run every prose check on the saved texts in `<output_dir>/sealed_store/responses.sealed.jsonl` offline (no model, no GPU)."""
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

        try:
            check_rendered_prose(response_text, world, focal_agent=focal_agent)
            passed = True
            check_id: str | None = None
            event_index: int | None = None
            event_kind: str | None = None
            failing_sentence: str | None = None
            reason: str | None = None
            calls_kept += 1
        except RenderCheckError as exc:
            passed = False
            check_id = exc.check_id
            event_index = exc.event_index
            event_kind = exc.event_kind
            failing_sentence = exc.failing_sentence
            reason = str(exc)
            calls_refused += 1
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
                "event_index": event_index,
                "event_kind": event_kind,
                "failing_sentence": failing_sentence,
                "failing_sentence_sha256": fail_sha,
                "failing_sentence_offset": fail_off,
                "failing_sentence_length": fail_len,
                "failing_sentence_sha256_verified": failing_sentence_sha_verified,
                "reason": reason,
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

    return {
        "output_dir": str(output_dir),
        "responses_sealed_path": str(responses_path),
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
        "refusals_by_check_id": dict(sorted(refusals_by_check_id.items())),
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


def verify_written_dataset(output_dir: Path) -> dict[str, object]:
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

    recount_report = verify_written_dataset_recounts_and_prose(public_dir, sealed_dir)
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
    grant_id: str | None = None,
    resume: bool = False,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    start_owned_server: bool | None = None,
    http_transport: OllamaHttpTransport | None = None,
    vram_probe: Callable[[], int] | None = None,
    halt_pid: int | None = None,
    pid_killer: Callable[[int], None] = kill_pid_tree,
    popen_factory: Callable[..., Any] | None = None,
) -> dict[str, object]:
    """Generate `count` scenarios, render via `mock` or `ollama` writer with checkpointing, seal, verify, and probe."""
    if writer_kind not in ("mock", "ollama"):
        raise ValueError(f"writer_kind must be 'mock' or 'ollama', got {writer_kind!r}.")
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}.")
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
            start_owned_server=should_own_server,
            popen_factory=popen_factory,
        )
        if resume:
            ollama_writer.restore_call_records(restored_calls)
        try:
            for idx in range(completed_count, len(worlds)):
                w = worlds[idx]
                prev_calls = len(ollama_writer.call_records)
                pair = ollama_writer.render_scenario_pair(w, allow_refusal=True)
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
        server_info = (
            ollama_writer.owned_server.to_dict()
            if ollama_writer.owned_server is not None
            else None
        )
    else:
        mock_writer = MockWriter()
        all_mock_calls: list[GenerationCallRecord] = list(restored_calls)
        if resume and generations_path is not None:
            lines = "".join(
                json.dumps(r.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
                for r in all_mock_calls
            )
            generations_path.write_text(lines, encoding="utf-8", newline="\n")

        for idx in range(completed_count, len(worlds)):
            w = worlds[idx]
            t0 = time.perf_counter()
            pair = mock_writer.render_scenario_pair(w)
            elapsed = max(0.0, time.perf_counter() - t0)
            scen_calls: list[GenerationCallRecord] = []
            for r_idx, rend in enumerate(pair):
                call_index = len(all_mock_calls) + 1
                call_id = f"{w.scenario_id}:r{r_idx}:call_{call_index:04d}"
                messages = [{"role": "user", "content": rend.writer_prompt}]
                prompt_sha = hashlib.sha256(
                    json.dumps(messages, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                resp_sha, think_sha = compute_sealed_response_digests(rend.prose, "")
                p_est = estimate_prompt_tokens(messages)
                if resolved_think and p_est + resolved_num_predict <= max_num_ctx:
                    n_ctx = min(
                        max_num_ctx,
                        compute_num_ctx(
                            p_est,
                            resolved_num_predict,
                            margin=NUM_CTX_MARGIN_TOKENS,
                            step=NUM_CTX_STEP_TOKENS,
                        ),
                    )
                else:
                    n_ctx = compute_num_ctx(
                        p_est,
                        resolved_num_predict,
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
                    prompt_eval_count=p_est,
                    eval_count=len(rend.prose.split()),
                    thinking_words=0,
                    num_predict=resolved_num_predict,
                    num_ctx=n_ctx,
                    max_num_ctx=max_num_ctx,
                    think=resolved_think,
                    temperature=0.0,
                    done_reason="stop",
                    seconds=round(elapsed / 2.0, 6),
                    vram_peak_mib=0,
                    vram_halt_mib=vram_halt_mib,
                    grant_id="cpu-mock",
                    checks_passed=True,
                    refusal_reasons=(),
                    attempt=0,
                    seed=int(w.seed),
                    response_sha256=resp_sha,
                    thinking_sha256=think_sha,
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
                        seed=int(w.seed),
                        response_text=rend.prose,
                        thinking_text="",
                        response_sha256=resp_sha,
                        thinking_sha256=think_sha,
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
        server_info = None

    generations_bytes = generations_path.read_bytes() if generations_path.is_file() else b""
    generations_sha256 = hashlib.sha256(generations_bytes).hexdigest()
    responses_bytes = sealed_responses_path.read_bytes() if sealed_responses_path.is_file() else b""
    responses_sealed_sha256 = hashlib.sha256(responses_bytes).hexdigest()

    outcomes = _summarize_call_outcomes(all_call_records)

    writer_provenance: dict[str, Any] = {
        "writer": writer_kind,
        "writer_pin": effective_writer_pin.to_dict(),
        "think": resolved_think,
        "num_predict": resolved_num_predict,
        "max_num_ctx": max_num_ctx,
        "max_retries": max_retries,
        "vram_halt_mib": vram_halt_mib,
        "grant_id": effective_grant_id,
        "generations_file": GENERATIONS_FILENAME,
        "generations_sha256": generations_sha256,
        "responses_sealed_file": SEALED_RESPONSES_FILENAME,
        "responses_sealed_sha256": responses_sealed_sha256,
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
    recount_verification = verify_written_dataset_recounts_and_prose(public_dir, sealed_dir)

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
        "max_retries": max_retries,
        "grant_id": effective_grant_id,
        "vram_halt_mib": vram_halt_mib,
        "vram_peak_mib": peak_vram_mib,
        "ollama_server": server_info,
        "num_scenarios": receipt.num_scenarios,
        "num_items": receipt.num_items,
        **outcomes,
        "items_sha256": receipt.items_sha256,
        "part_json_sha256": receipt.part_json_sha256,
        "keys_sealed_sha256": receipt.keys_sealed_sha256,
        "generations_file": GENERATIONS_FILENAME,
        "generations_sha256": generations_sha256,
        "responses_sealed_file": SEALED_RESPONSES_FILENAME,
        "responses_sealed_sha256": responses_sealed_sha256,
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


def main(argv: Sequence[str] | None = None) -> int:
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
        "--grant-id",
        type=str,
        default=None,
        help="Publisher GPU grant ID required when running --writer ollama.",
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
    args = parser.parse_args(argv)

    try:
        if args.recheck is not None:
            recheck_res = recheck_saved_responses(args.recheck)
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
            verify_res = verify_written_dataset(args.verify_written)
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
            grant_id=args.grant_id,
            resume=args.resume,
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
