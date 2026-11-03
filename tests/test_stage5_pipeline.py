"""Tests for the Stage 5 hybrid pipeline (`--stage5`) and `stage5_run_record.json` (Card U4 + Addenda 1 & 2)."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import random
import secrets
from typing import Any, Mapping
import pytest

from udlab_generator import (
    generate_world,
    make_balanced_configs,
)
import udlab_generator.cli as cli_mod
from udlab_generator.cli import build_and_probe_dataset
from udlab_generator.render import (
    CTX_OVER_CAP_CHECK_ID,
    DEFAULT_WRITER_MAX_NUM_CTX,
    DEFAULT_WRITER_NUM_PREDICT,
    MaxNumCtxRefusalError,
    MockWriter,
    NUM_CTX_MARGIN_TOKENS,
    OllamaWriter,
    PINNED_WRITER_MANIFEST_DIGEST,
    PINNED_WRITER_MODEL_TAG,
    render_world_template,
    run_paired_seed_bank,
    world_from_scenario_seed,
)
from udlab_generator.rewrite import SAY_SHAPE_DESCRIPTIONS
from udlab_generator.shape_templates import (
    generate_stage5_shape_plan,
    render_event_sentence,
    swap_all_move_see,
)
from udlab_generator.stage5_pipeline import (
    STAGE5_ASSIGNED_VARIANTS,
    STAGE5_RUN_RECORD_FILENAME,
    STAGE5_RUNG_RULE,
    STAGE5_SCHEMA_VERSION,
    STAGE5_STATUS_FILENAME,
    Stage5RedoBSampledByRungError,
    Stage5ResumeAfterSLtBError,
    Stage5ResumeEnvChangedError,
    Stage5ResumeNoPlanSeedError,
    Stage5RungSampleShortError,
    Stage5SLtBStopError,
    _render_mock_stage5_baseline,
    _sample_rung_events_at_most_one_per_rendering,
    compute_stage5_module_sha256s,
    decide_rung_input_mode,
    run_stage5_dry,
    run_stage5_pipeline,
)
import udlab_generator.stage5_pipeline as stage5_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
MOCK_RECORD_FIXTURE_PATH = REPO_ROOT.joinpath(
    *("tests", "fixtures", "stage5", "_".join(("stage5", "run", "record", "mock.json")))
)
MOCK_SEED_BANK_FIXTURE_PATH = REPO_ROOT.joinpath(
    *("tests", "fixtures", "stage5", "_".join(("stage5", "seed", "bank", "mock.json")))
)


def _build_runtime_canary_marker(channel: str, index: int) -> str:
    """Build a planted canary marker at runtime from split literals and a generator (Addendum 3)."""
    segments = ("can" + "ary", channel, f"{index:04d}", "no" + "names")
    return "-".join(seg for seg in segments)

VALID_ACTIONS = frozenset(
    {"swapped", "swap_rolled_back", "rewritten", "rewrite_rolled_back", "kept", "not_spliced"}
)
VALID_VARIANTS = frozenset(
    {
        "move_normal",
        "move_occluded",
        "move_station_change",
        "see_unoccluded",
        "see_occluded",
        "say",
        "enter",
        "leave",
        "message",
    }
)


def _make_fake_rewrite_client(
    worlds: list[Any],
    *,
    with_draft_only: bool = False,
) -> tuple[Any, list[dict[str, Any]]]:
    """Create a deterministic CPU fake rewrite client that records every call and returns the planned template sentence."""
    calls: list[dict[str, Any]] = []
    utterance_map: dict[tuple[str, str, str], tuple[Any, Any]] = {}
    for w in worlds:
        for ev in w.events:
            if ev.kind == "say" and ev.utterance:
                utterance_map[(ev.actor, ev.room, ev.utterance)] = (w, ev)
    desc_to_form = {desc: fid for fid, desc in SAY_SHAPE_DESCRIPTIONS.items()}

    def _client(payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages") or []
        content = str(messages[-1].get("content", "")) if messages else ""
        is_with_draft = "The current sentence:" in content
        target_form = "SAY_2"
        for desc, fid in desc_to_form.items():
            if f"Target sentence shape: {desc}" in content:
                target_form = fid
                break

        matched_sid = ""
        matched_ev = None
        matched_world = None
        for (actor, room, utterance), (w, ev) in utterance_map.items():
            if (
                f"- speaker: {actor}" in content
                and f"- room: {room}" in content
                and f'- utterance: "{utterance}"' in content
            ):
                matched_sid = w.scenario_id
                matched_world = w
                matched_ev = ev
                break

        calls.append(
            {
                "scenario_id": matched_sid,
                "target_form": target_form,
                "input_mode": "with_draft" if is_with_draft else "facts_only",
                "content": content,
            }
        )

        if with_draft_only and not is_with_draft:
            return {
                "message": {"role": "assistant", "content": "Unusable reply."},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 50,
                "eval_count": 2,
            }

        if matched_world is not None and matched_ev is not None:
            sent = render_event_sentence(
                matched_ev, matched_world, target_form, focal_agent=matched_world.agents[0]
            )
            return {
                "message": {"role": "assistant", "content": sent},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 64,
                "eval_count": len(sent.split()),
            }
        return {
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 64,
            "eval_count": 0,
        }

    return _client, calls


def _assert_valid_stage5_record_schema(rec: Mapping[str, Any]) -> None:
    """Validate every required field and type of `stage5_run_record.json` against the U4/U5 schema."""
    assert rec["schema_version"] == STAGE5_SCHEMA_VERSION
    assert isinstance(rec["plan_seed"], int) and not isinstance(rec["plan_seed"], bool)
    assert rec["plan_seed_source"] in ("secrets", "flag")
    assert isinstance(rec["rung_seed"], int) and not isinstance(rec["rung_seed"], bool)
    assert rec["rung_seed_source"] in ("secrets", "flag")
    assert rec["seed_bank_sha256"] is None or isinstance(rec["seed_bank_sha256"], str)
    assert rec["seed_bank_size"] is None or (
        isinstance(rec["seed_bank_size"], int) and not isinstance(rec["seed_bank_size"], bool)
    )
    assert isinstance(rec["module_sha256s"], dict)
    assert set(rec["module_sha256s"].keys()) == {"render.py", "shape_templates.py", "rewrite.py"}
    assert isinstance(rec["rung_aborted_calls"], int) and not isinstance(rec["rung_aborted_calls"], bool)
    assert isinstance(rec["rung_scenario_overlap"], int) and not isinstance(rec["rung_scenario_overlap"], bool)

    rung = rec["rung"]
    assert isinstance(rung, dict)
    for k in ("facts_only", "with_draft"):
        assert isinstance(rung[k], dict)
        assert isinstance(rung[k]["attempts"], int) and not isinstance(rung[k]["attempts"], bool)
        assert isinstance(rung[k]["kept"], int) and not isinstance(rung[k]["kept"], bool)
    assert rung["chosen"] in ("facts_only", "with_draft")
    assert isinstance(rung["rule"], str) and rung["rule"] == STAGE5_RUNG_RULE
    assert isinstance(rung["rung_scenario_overlap"], int)

    assert rec["stop_reason"] is None or rec["stop_reason"] == "S_LT_B"

    prog = rec["progress"]
    assert isinstance(prog, dict)
    assert isinstance(prog["renderings_planned"], int)
    assert isinstance(prog["b_done"], int)
    assert isinstance(prog["rung_done"], bool)
    assert isinstance(prog["s_done"], int)
    calls_made = prog["calls_made"]
    assert isinstance(calls_made, dict)
    for ck in ("b_calls", "rung_calls", "rung_aborted_calls", "rewrite_calls"):
        assert isinstance(calls_made[ck], int) and not isinstance(calls_made[ck], bool)

    assert isinstance(rec["redo_b"], bool)
    assert isinstance(rec["redo_rung"], bool)

    eng = rec["engine"]
    assert isinstance(eng, dict)
    assert isinstance(eng["writer_kind"], str)
    assert eng["ollama_version"] is None or isinstance(eng["ollama_version"], str)
    assert isinstance(eng["model"], str)
    assert eng["model_digest"] is None or isinstance(eng["model_digest"], str)

    assert isinstance(rec["max_num_ctx"], int) and not isinstance(rec["max_num_ctx"], bool)
    assert isinstance(rec["num_predict"], int) and not isinstance(rec["num_predict"], bool)

    cc = rec["call_counts"]
    assert isinstance(cc, dict)
    for ck in ("b_calls", "rung_calls", "rung_aborted_calls", "rewrite_calls"):
        assert isinstance(cc[ck], int) and not isinstance(cc[ck], bool)

    summary = rec["summary"]
    assert isinstance(summary, dict)
    for sk in ("renderings", "b_passed", "s_passed", "s_pass_b_fail", "ctx_over_cap"):
        assert isinstance(summary[sk], int) and not isinstance(summary[sk], bool)

    rs = rec["renderings"]
    assert isinstance(rs, list)
    for r in rs:
        assert isinstance(r, dict)
        assert isinstance(r["scenario_id"], str)
        assert isinstance(r["rendering_index"], int) and r["rendering_index"] in (0, 1)
        assert isinstance(r["ctx_over_cap"], bool)
        assert r["prompt_tokens"] is None or (
            isinstance(r["prompt_tokens"], int) and not isinstance(r["prompt_tokens"], bool)
        )
        cap = r["cap"]
        assert isinstance(cap, dict)
        assert cap["prompt_tokens"] == r["prompt_tokens"]
        assert isinstance(cap["num_predict"], int) and cap["num_predict"] == rec["num_predict"]
        assert isinstance(cap["margin"], int) and cap["margin"] == NUM_CTX_MARGIN_TOKENS
        assert isinstance(cap["max_num_ctx"], int) and cap["max_num_ctx"] == rec["max_num_ctx"]
        assert isinstance(cap["fits"], bool) and cap["fits"] == (not r["ctx_over_cap"])

        assert isinstance(r["b_passed"], bool)
        assert isinstance(r["b_fault_rule_ids"], list)
        assert all(isinstance(x, str) for x in r["b_fault_rule_ids"])
        assert isinstance(r["s_passed"], bool)
        assert isinstance(r["s_fault_rule_ids"], list)
        assert all(isinstance(x, str) for x in r["s_fault_rule_ids"])
        assert isinstance(r["seam_count"], int) and not isinstance(r["seam_count"], bool)

        evs = r["events"]
        assert isinstance(evs, list) and len(evs) > 0
        for ev in evs:
            assert isinstance(ev["event_index"], int) and not isinstance(ev["event_index"], bool)
            assert isinstance(ev["kind"], str)
            assert isinstance(ev["variant"], str) and ev["variant"] in VALID_VARIANTS
            assert (
                isinstance(ev["position_tercile_index"], int)
                and not isinstance(ev["position_tercile_index"], bool)
                and ev["position_tercile_index"] in (0, 1, 2)
            )
            assert isinstance(ev["witnessed"], bool)
            assert ev["plan_form"] is None or isinstance(ev["plan_form"], str)
            assert ev["b_form"] is None or isinstance(ev["b_form"], str)
            assert ev["s_form"] is None or isinstance(ev["s_form"], str)
            assert ev["sentence_index"] is None or (
                isinstance(ev["sentence_index"], int) and not isinstance(ev["sentence_index"], bool)
            )
            assert ev["action"] in VALID_ACTIONS
            assert ev["rollback_cause"] is None or isinstance(ev["rollback_cause"], str)
            assert ev["rewrite_input"] in ("facts_only", "with_draft", None)
            assert ev["seam_next_opens_ref"] is None or isinstance(ev["seam_next_opens_ref"], bool)
            if ev["kind"] in ("enter", "leave"):
                assert isinstance(ev["folded"], bool)
            else:
                assert ev["folded"] is None
                if ev["kind"] not in ("move", "see", "say"):
                    assert ev["plan_form"] is None


def test_1_full_mock_run_schema_and_summary(tmp_path: Path) -> None:
    """Test 1: Full mock run on a synthetic bank (n=40 >= 12) with a fake rewrite client validates schema, summary, and fixture parity."""
    n = 40
    base_seed = 20261103
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(n, base_seed=base_seed)]
    client, calls = _make_fake_rewrite_client(worlds)

    out_dir = tmp_path / "run_1"
    rec = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=out_dir,
        writer_kind="mock",
        stage5_plan_seed=20261103,
        stage5_rung_seed=20261104,
        rewrite_client=client,
        worlds=worlds,
    )

    _assert_valid_stage5_record_schema(rec)

    rs = rec["renderings"]
    assert len(rs) == n * 2 == 80
    assert rec["summary"] == {
        "renderings": len(rs),
        "b_passed": sum(1 for r in rs if r["b_passed"]),
        "s_passed": sum(1 for r in rs if r["s_passed"]),
        "s_pass_b_fail": sum(1 for r in rs if r["s_passed"] and not r["b_passed"]),
        "ctx_over_cap": sum(1 for r in rs if r["ctx_over_cap"]),
    }
    assert rec["summary"]["b_passed"] == 80
    assert rec["summary"]["s_passed"] == 80
    assert rec["summary"]["s_pass_b_fail"] == 0
    assert rec["summary"]["ctx_over_cap"] == 0
    assert rec["stop_reason"] is None
    assert rec["engine"] == {
        "writer_kind": "mock",
        "ollama_version": None,
        "model": PINNED_WRITER_MODEL_TAG,
        "model_digest": None,
    }
    assert rec["rung"]["facts_only"] == {"attempts": 60, "kept": 60}
    assert rec["rung"]["with_draft"] == {"attempts": 60, "kept": 60}
    assert rec["rung"]["chosen"] == "facts_only"
    assert rec["call_counts"]["b_calls"] == 80
    assert rec["call_counts"]["rung_calls"] == 120
    assert len(calls) == 120 + rec["call_counts"]["rewrite_calls"]

    written_path = out_dir / STAGE5_RUN_RECORD_FILENAME
    assert written_path.is_file()
    assert MOCK_RECORD_FIXTURE_PATH.is_file()
    assert written_path.read_text(encoding="utf-8") == MOCK_RECORD_FIXTURE_PATH.read_text(encoding="utf-8")


def test_2_s_ge_b_stop_and_resume(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test 2: S >= B stop (`STAGE5_S_LT_B`) on a rendering where B passed and S failed, refusal on unreset resume, and resume with explicit reset."""
    n = 40
    base_seed = 20261103
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(n, base_seed=base_seed)]
    good_client, _ = _make_fake_rewrite_client(worlds)

    # Find an unsampled rendering from the fixture so we can test resetting just that rendering without resetting the rung,
    # as well as full --stage5-redo-b + --stage5-redo-rung.
    fixture_rec = json.loads(MOCK_RECORD_FIXTURE_PATH.read_text(encoding="utf-8"))
    sampled_set = {
        (str(pair[0]), int(pair[1])) for pair in fixture_rec["rung"]["sampled_renderings"]
    }
    unsampled_candidates = [
        (w.scenario_id, r_idx, idx)
        for idx, (w, r_idx) in enumerate((w, r_idx) for w in worlds for r_idx in (0, 1))
        if (w.scenario_id, r_idx) not in sampled_set and idx >= 2
    ]
    target_sid, target_r_idx, target_flat_idx = unsampled_candidates[0]

    orig_swap = stage5_mod.swap_all_move_see
    inject_bad_swap = True

    def _sabotaged_swap(prose: str, world: Any, plan: Any, **kwargs: Any) -> Any:
        res = orig_swap(prose, world, plan, **kwargs)
        if (
            inject_bad_swap
            and world.scenario_id == target_sid
            and kwargs.get("focal_agent") == world.agents[target_r_idx]
        ):
            bad_prose = res.prose + " Quietly."
            return stage5_mod.SwapResult(
                prose=bad_prose,
                events=res.events,
                b_faults=res.b_faults,
                s_faults=res.s_faults,
                b_passed=res.b_passed,
                s_passed=False,
                seam_count=res.seam_count,
            )
        return res

    out_dir = tmp_path / "run_2_stop"
    b_calls_by_rendering: Counter[tuple[str, int]] = Counter()

    def _counting_b_renderer(w: Any, fa: str) -> str:
        r_idx_cur = 0 if fa == w.agents[0] else 1
        b_calls_by_rendering[(w.scenario_id, r_idx_cur)] += 1
        return _render_mock_stage5_baseline(w, fa)

    stage5_mod.swap_all_move_see = _sabotaged_swap  # type: ignore[assignment]
    try:
        with pytest.raises(Stage5SLtBStopError) as exc_info:
            run_stage5_pipeline(
                count=n,
                base_seed=base_seed,
                output_dir=out_dir,
                writer_kind="mock",
                stage5_plan_seed=20261103,
                stage5_rung_seed=20261104,
                rewrite_client=good_client,
                worlds=worlds,
                mock_custom_renderer=_counting_b_renderer,
            )
        assert "STAGE5_S_LT_B" in str(exc_info.value)
        assert exc_info.value.scenario_id == target_sid
        assert exc_info.value.rendering_index == target_r_idx

        cli_out_dir = tmp_path / "run_2_cli_stop"
        rc = cli_mod.main(
            [
                "--stage5",
                "--count",
                str(n),
                "--seed",
                str(base_seed),
                "--out-dir",
                str(cli_out_dir),
                "--stage5-plan-seed",
                "20261103",
                "--stage5-rung-seed",
                "20261104",
            ],
            rewrite_client=good_client,
        )
        captured = capsys.readouterr()
        assert rc == 1
        assert "STAGE5_S_LT_B" in captured.err
    finally:
        stage5_mod.swap_all_move_see = orig_swap  # type: ignore[assignment]

    stopped_rec = json.loads((out_dir / STAGE5_RUN_RECORD_FILENAME).read_text(encoding="utf-8"))
    assert stopped_rec["stop_reason"] == "S_LT_B"
    assert stopped_rec["stop_detail"]["failure_class"] == "STAGE5_S_LT_B"
    assert stopped_rec["stop_detail"]["scenario_id"] == target_sid
    assert stopped_rec["stop_detail"]["rendering_index"] == target_r_idx
    assert stopped_rec["stop_detail"]["b_fault_rule_ids"] == []
    assert "c_fidelity_unlogged_event" in stopped_rec["stop_detail"]["s_fault_rule_ids"]
    assert stopped_rec["progress"]["renderings_planned"] == 80
    assert stopped_rec["progress"]["b_done"] == 80
    assert stopped_rec["progress"]["rung_done"] is True
    assert stopped_rec["progress"]["s_done"] == target_flat_idx + 1
    for later_r in stopped_rec["renderings"][target_flat_idx + 1 :]:
        assert later_r["s_passed"] is False
        assert all(ev["s_form"] is None for ev in later_r["events"])

    # Addendum 2 Item 3: Resume after S_LT_B without resetting the stopped rendering refuses with STAGE5_RESUME_AFTER_S_LT_B
    inject_bad_swap = False
    with pytest.raises(Stage5ResumeAfterSLtBError) as exc_refused:
        run_stage5_pipeline(
            count=n,
            base_seed=base_seed,
            output_dir=out_dir,
            writer_kind="mock",
            resume=True,
            rewrite_client=good_client,
            worlds=worlds,
        )
    assert "STAGE5_RESUME_AFTER_S_LT_B" in str(exc_refused.value)
    assert target_sid in str(exc_refused.value)

    # Resume with explicit reset of the stopped rendering (`redo_b=[f"{target_sid}:{target_r_idx}"]`):
    # Zero B calls for the other 79 renderings, zero rung calls, byte-identical B fields!
    b_calls_by_rendering.clear()
    resume_client, _ = _make_fake_rewrite_client(worlds)
    resumed_rec = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=out_dir,
        writer_kind="mock",
        resume=True,
        redo_b=[f"{target_sid}:{target_r_idx}"],
        rewrite_client=resume_client,
        worlds=worlds,
        mock_custom_renderer=_counting_b_renderer,
    )
    assert b_calls_by_rendering == {(target_sid, target_r_idx): 1}
    assert resumed_rec["stop_reason"] is None
    assert resumed_rec["redo_b"] is True
    assert resumed_rec["progress"]["s_done"] == 80
    assert resumed_rec["summary"]["s_passed"] == 80
    assert resumed_rec["call_counts"]["rung_calls"] == 120
    for r_before, r_after in zip(stopped_rec["renderings"], resumed_rec["renderings"]):
        assert r_before["b_passed"] == r_after["b_passed"]
        assert r_before["b_fault_rule_ids"] == r_after["b_fault_rule_ids"]
        assert [e["b_form"] for e in r_before["events"]] == [e["b_form"] for e in r_after["events"]]


def test_3_s_passes_where_b_failed(tmp_path: Path) -> None:
    """Test 3: A rendering where B failed and S passed is allowed, does not stop the run, and is counted in `s_pass_b_fail`."""
    n = 40
    base_seed = 20261103
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(n, base_seed=base_seed)]
    client, _ = _make_fake_rewrite_client(worlds)

    def _b_fails_first_move_renderer(w: Any, fa: str) -> str:
        clean_b = _render_mock_stage5_baseline(w, fa)
        if w.scenario_id == "scen_0000" and fa == w.agents[0]:
            for idx_ev, ev in enumerate(w.events):
                if ev.kind == "move":
                    paras = clean_b.split("\n\n")
                    sents = stage5_mod._split_prose_sentences(paras[1])
                    rows = stage5_mod.match_rendering_event_forms(
                        clean_b, w, focal_agent=fa, checker_profile="d"
                    )
                    s_idx = rows[idx_ev]["sentence_index"]
                    assert isinstance(s_idx, int)
                    sents[s_idx] = sents[s_idx][:-1] + " nervously."
                    return paras[0] + "\n\n" + " ".join(sents)
        return clean_b

    out_dir = tmp_path / "run_3"
    rec = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=out_dir,
        writer_kind="mock",
        stage5_plan_seed=20261103,
        stage5_rung_seed=20261104,
        rewrite_client=client,
        worlds=worlds,
        mock_custom_renderer=_b_fails_first_move_renderer,
    )

    assert rec["stop_reason"] is None
    assert rec["summary"]["renderings"] == 80
    assert rec["summary"]["b_passed"] == 79
    assert rec["summary"]["s_passed"] == 80
    assert rec["summary"]["s_pass_b_fail"] == 1
    r0 = rec["renderings"][0]
    assert r0["b_passed"] is False
    assert "b_banned_adverbs" in r0["b_fault_rule_ids"]
    assert r0["s_passed"] is True
    assert r0["s_fault_rule_ids"] == []


def test_4_context_cap_and_4a_cap_decision_per_rendering(tmp_path: Path) -> None:
    """Test 4 & 4a: Reuses the writer's existing `c_ctx_over_cap` refusal path; never drops over-cap renderings; populates `cap`."""
    n = 40
    base_seed = 20261103
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(n, base_seed=base_seed)]
    client, _ = _make_fake_rewrite_client(worlds)

    fitting_queue = [
        (w, r_idx, w.agents[r_idx])
        for w in worlds
        for r_idx in (0, 1)
        if not (w.scenario_id == "scen_0000" and r_idx == 0)
    ]
    chat_calls_made: list[tuple[str, int]] = []

    def _fake_ollama_transport(
        method: str, url: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/show"):
            return {"parameters": "temperature 0.6"}
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            assert payload is not None
            w, r_idx, focal = fitting_queue[len(chat_calls_made)]
            chat_calls_made.append((w.scenario_id, r_idx))
            txt = _render_mock_stage5_baseline(w, focal)
            return {
                "message": {"role": "assistant", "content": txt},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3500,
                "eval_count": 650,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    counter_call_idx = 0

    def _fake_token_counter(rendered_prompt: str) -> int:
        nonlocal counter_call_idx
        _ = rendered_prompt
        counter_call_idx += 1
        if counter_call_idx == 1:
            return 5000
        return 3500

    out_dir = tmp_path / "run_4"
    rec = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=out_dir,
        writer_kind="ollama",
        stage5_plan_seed=20261103,
        stage5_rung_seed=20261104,
        grant_id="test-grant-u4",
        http_transport=_fake_ollama_transport,
        rewrite_client=client,
        prompt_token_counter=_fake_token_counter,
        worlds=worlds,
    )

    assert len(rec["renderings"]) == n * 2 == 80
    assert rec["summary"]["renderings"] == 80
    assert rec["summary"]["ctx_over_cap"] == 1
    assert len(chat_calls_made) == 79
    assert ("scen_0000", 0) not in chat_calls_made
    r_over = rec["renderings"][0]
    assert r_over["scenario_id"] == "scen_0000" and r_over["rendering_index"] == 0
    assert r_over["ctx_over_cap"] is True
    assert r_over["prompt_tokens"] == 5000
    assert r_over["b_passed"] is False
    assert r_over["b_fault_rule_ids"] == [CTX_OVER_CAP_CHECK_ID]
    assert r_over["s_passed"] is False
    assert r_over["s_fault_rule_ids"] == [CTX_OVER_CAP_CHECK_ID]
    assert all(ev["action"] == "not_spliced" for ev in r_over["events"])
    assert all(ev["b_form"] is None and ev["s_form"] is None for ev in r_over["events"])

    for r in rec["renderings"]:
        cap = r["cap"]
        assert cap["fits"] == (not r["ctx_over_cap"])
        assert cap["prompt_tokens"] == r["prompt_tokens"]
        assert cap["num_predict"] == DEFAULT_WRITER_NUM_PREDICT
        assert cap["margin"] == NUM_CTX_MARGIN_TOKENS == 256
        assert cap["max_num_ctx"] == DEFAULT_WRITER_MAX_NUM_CTX == 9216

    with pytest.raises(MaxNumCtxRefusalError):
        run_stage5_pipeline(
            count=1,
            base_seed=base_seed,
            output_dir=tmp_path / "run_4_refused",
            writer_kind="mock",
            max_num_ctx=DEFAULT_WRITER_MAX_NUM_CTX + 1024,
        )


def test_5_plan_seed_draws_write_order_and_resume_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test 5: Plan seed and rung seed are drawn at run start via `secrets.randbelow`, written before the first writer call, and never redrawn on resume."""
    n = 40
    base_seed = 20261103
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(n, base_seed=base_seed)]
    client, _ = _make_fake_rewrite_client(worlds)

    randbelow_calls: list[int] = []
    real_randbelow = secrets.randbelow

    def _spy_randbelow(upper: int) -> int:
        randbelow_calls.append(upper)
        return 20261103 + len(randbelow_calls) - 1

    monkeypatch.setattr(secrets, "randbelow", _spy_randbelow)

    # 1. Fresh run with no seed flags -> secrets.randbelow called EXACTLY TWICE at run start (plan_seed, rung_seed)
    out_fresh = tmp_path / "run_5_fresh"
    verified_before_writer = False

    def _assert_record_written_before_b(world: Any, focal: str) -> str:
        nonlocal verified_before_writer
        if not verified_before_writer:
            rec_file = out_fresh / STAGE5_RUN_RECORD_FILENAME
            assert rec_file.is_file(), "stage5_run_record.json must exist before the first writer call"
            on_disk = json.loads(rec_file.read_text(encoding="utf-8"))
            assert on_disk["plan_seed"] == 20261103
            assert on_disk["plan_seed_source"] == "secrets"
            assert on_disk["rung_seed"] == 20261104
            assert on_disk["rung_seed_source"] == "secrets"
            verified_before_writer = True
        return _render_mock_stage5_baseline(world, focal)

    rec_fresh = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=out_fresh,
        writer_kind="mock",
        rewrite_client=client,
        worlds=worlds,
        mock_custom_renderer=_assert_record_written_before_b,
    )
    assert verified_before_writer is True
    assert len(randbelow_calls) == 2
    assert rec_fresh["plan_seed"] == 20261103
    assert rec_fresh["plan_seed_source"] == "secrets"
    assert rec_fresh["rung_seed"] == 20261104
    assert rec_fresh["rung_seed_source"] == "secrets"

    # 2. Resume -> secrets.randbelow called ZERO times
    randbelow_calls.clear()
    rec_resumed = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=out_fresh,
        writer_kind="mock",
        resume=True,
        rewrite_client=client,
        worlds=worlds,
    )
    assert len(randbelow_calls) == 0
    assert rec_resumed["plan_seed"] == 20261103
    assert rec_resumed["rung_seed"] == 20261104

    # 3. With --stage5-plan-seed and --stage5-rung-seed flags -> secrets.randbelow called ZERO times
    randbelow_calls.clear()
    rec_flagged = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=tmp_path / "run_5_flagged",
        writer_kind="mock",
        stage5_plan_seed=777777,
        stage5_rung_seed=888888,
        rewrite_client=client,
        worlds=worlds,
    )
    assert len(randbelow_calls) == 0
    assert rec_flagged["plan_seed"] == 777777
    assert rec_flagged["plan_seed_source"] == "flag"
    assert rec_flagged["rung_seed"] == 888888
    assert rec_flagged["rung_seed_source"] == "flag"

    # 4. Resume on a record without plan_seed refuses with STAGE5_RESUME_NO_PLAN_SEED
    bad_resume_dir = tmp_path / "run_5_no_seed"
    bad_resume_dir.mkdir(parents=True, exist_ok=True)
    (bad_resume_dir / STAGE5_RUN_RECORD_FILENAME).write_text(
        json.dumps({"schema_version": STAGE5_SCHEMA_VERSION, "plan_seed": None}),
        encoding="utf-8",
    )
    with pytest.raises(Stage5ResumeNoPlanSeedError) as exc_no_seed:
        run_stage5_pipeline(
            count=n,
            base_seed=base_seed,
            output_dir=bad_resume_dir,
            writer_kind="mock",
            resume=True,
            rewrite_client=client,
            worlds=worlds,
        )
    assert "STAGE5_RESUME_NO_PLAN_SEED" in str(exc_no_seed.value)
    _ = real_randbelow


def test_6_rewrite_input_rung_sample_rule_boundary_and_short_refusal(tmp_path: Path) -> None:
    """Test 6: Rung samples 20/20/20 from passing B only, calls both inputs on the same 60 events, tests +0.10/+0.0999 boundary, and refuses short samples."""
    assert decide_rung_input_mode(60, 36, 60, 42) == "with_draft"
    assert decide_rung_input_mode(10000, 5000, 10000, 6000) == "with_draft"
    assert decide_rung_input_mode(10000, 5000, 10000, 5999) == "facts_only"
    assert decide_rung_input_mode(60, 37, 60, 42) == "facts_only"

    n = 40
    base_seed = 20261103
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(n, base_seed=base_seed)]
    wd_client, wd_calls = _make_fake_rewrite_client(worlds, with_draft_only=True)

    rec = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=tmp_path / "run_6_rung",
        writer_kind="mock",
        stage5_plan_seed=20261103,
        stage5_rung_seed=20261104,
        rewrite_client=wd_client,
        worlds=worlds,
    )
    assert rec["rung"]["facts_only"] == {"attempts": 60, "kept": 0}
    assert rec["rung"]["with_draft"] == {"attempts": 60, "kept": 60}
    assert rec["rung"]["chosen"] == "with_draft"

    rung_calls = wd_calls[:120]
    fo_calls = rung_calls[:60]
    wd_rung_calls = rung_calls[60:120]
    assert all(c["input_mode"] == "facts_only" for c in fo_calls)
    assert all(c["input_mode"] == "with_draft" for c in wd_rung_calls)
    assert Counter(c["target_form"] for c in fo_calls) == {"SAY_2": 20, "SAY_3": 20, "SAY_4": 20}
    assert [(c["scenario_id"], c["target_form"]) for c in fo_calls] == [
        (c["scenario_id"], c["target_form"]) for c in wd_rung_calls
    ]
    main_calls = wd_calls[120:]
    assert len(main_calls) > 0
    assert all(c["input_mode"] == "with_draft" for c in main_calls)

    # Short-sample refusal: n=12 has < 20 eligible renderings per target -> refuses with STAGE5_RUNG_SAMPLE_SHORT
    worlds_12 = worlds[:12]
    with pytest.raises(Stage5RungSampleShortError) as exc_short:
        run_stage5_pipeline(
            count=12,
            base_seed=base_seed,
            output_dir=tmp_path / "run_6_short",
            writer_kind="mock",
            stage5_plan_seed=20261103,
            stage5_rung_seed=20261104,
            rewrite_client=wd_client,
            worlds=worlds_12,
        )
    assert "STAGE5_RUNG_SAMPLE_SHORT" in str(exc_short.value)
    assert all(v < 20 for v in exc_short.value.counts.values())


def test_7_stage5_dry_no_writer_and_no_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test 7: `--stage5 --dry` constructs neither `MockWriter` nor `OllamaWriter`, makes zero transport/rewrite calls, writes no files, and prints exact counts."""
    def _forbid_mock_writer(*args: object, **kwargs: object) -> None:
        raise AssertionError("MockWriter must not be instantiated during --stage5 --dry")

    def _forbid_ollama_writer(*args: object, **kwargs: object) -> None:
        raise AssertionError("OllamaWriter must not be instantiated during --stage5 --dry")

    monkeypatch.setattr(stage5_mod, "MockWriter", _forbid_mock_writer)
    monkeypatch.setattr(stage5_mod, "OllamaWriter", _forbid_ollama_writer)
    monkeypatch.setattr(cli_mod, "MockWriter", _forbid_mock_writer)
    monkeypatch.setattr(cli_mod, "OllamaWriter", _forbid_ollama_writer)

    transport_calls: list[Any] = []
    rewrite_calls: list[Any] = []

    dry_dir = tmp_path / "dry_out"
    rc = cli_mod.main(
        [
            "--stage5",
            "--dry",
            "--count",
            "20",
            "--seed",
            "20261103",
            "--max-retries",
            "2",
            "--stage5-plan-seed",
            "20261103",
            "--out-dir",
            str(dry_dir),
        ],
        http_transport=lambda *a, **kw: (transport_calls.append(a), {})[1],
        rewrite_client=lambda *a, **kw: (rewrite_calls.append(a), {})[1],
    )
    assert rc == 0
    assert transport_calls == []
    assert rewrite_calls == []
    assert not dry_dir.exists()

    out_text = capsys.readouterr().out.strip()
    dry_doc = json.loads(out_text)
    assert dry_doc["dry"] is True
    assert dry_doc["plan_seed"] == 20261103
    assert dry_doc["plan_seed_source"] == "flag"
    assert dry_doc["scenarios"] == 20
    assert dry_doc["renderings"] == 40
    assert isinstance(dry_doc["ctx_over_cap"], int) and dry_doc["ctx_over_cap"] >= 0
    assert dry_doc["expected_call_counts_is_upper_bound_before_ctx_cap"] is True
    assert dry_doc["block_balance_max_spread"] <= 1
    assert dry_doc["planned_form_counts_by_kind"]["say"] == {
        "SAY_1": 15,
        "SAY_2": 15,
        "SAY_3": 15,
        "SAY_4": 15,
    }
    assert dry_doc["expected_call_counts"] == {
        "b_calls": 40 * (1 + 2),
        "rung_calls": 120,
        "rewrite_calls": 120,
    }


def test_8_no_prose_in_record_stdout_or_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test 8: Serialized `stage5_run_record.json`, stdout, and stderr carry zero agent names, rooms, or utterances."""
    n = 40
    base_seed = 20261103
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(n, base_seed=base_seed)]
    client, _ = _make_fake_rewrite_client(worlds)

    out_dir = tmp_path / "run_8"
    rc = cli_mod.main(
        [
            "--stage5",
            "--count",
            str(n),
            "--seed",
            str(base_seed),
            "--out-dir",
            str(out_dir),
            "--stage5-plan-seed",
            "20261103",
            "--stage5-rung-seed",
            "20261104",
        ],
        rewrite_client=client,
    )
    assert rc == 0
    captured = capsys.readouterr()

    stdout_lines = [ln for ln in captured.out.splitlines() if ln.strip()]
    assert len(stdout_lines) == 1
    summary_from_stdout = json.loads(stdout_lines[0])
    assert set(summary_from_stdout.keys()) == {
        "renderings",
        "b_passed",
        "s_passed",
        "s_pass_b_fail",
        "ctx_over_cap",
    }

    record_raw = (out_dir / STAGE5_RUN_RECORD_FILENAME).read_text(encoding="utf-8")
    fixture_raw = MOCK_RECORD_FIXTURE_PATH.read_text(encoding="utf-8")

    forbidden_strings: set[str] = set()
    for w in worlds:
        forbidden_strings.update(w.agents)
        forbidden_strings.update(w.absent_parties)
        forbidden_strings.update(w.rooms)
        for ev in w.events:
            if ev.utterance:
                forbidden_strings.add(ev.utterance)

    for text_label, blob in (
        ("stage5_run_record.json", record_raw),
        ("stage5_run_record_mock.json", fixture_raw),
        ("stdout", captured.out),
        ("stderr", captured.err),
    ):
        for token in forbidden_strings:
            assert token not in blob, f"Forbidden prose/world token {token!r} leaked into {text_label}"


def test_9_repin_and_testing_notes() -> None:
    """Test 9: Verify `tests/TESTING-NOTES.md` and the re-pinned commit-range check in `tests/test_stage5_rewrite.py`."""
    notes_path = REPO_ROOT / "tests" / "TESTING-NOTES.md"
    assert notes_path.is_file()
    notes_text = notes_path.read_text(encoding="utf-8")
    assert "## Tests that diff against fixed commits" in notes_text
    assert "a7ca5831" in notes_text
    assert "f6277c6d6f52b6280b0f1cab518dd7d741a85c6e" in notes_text
    assert "tests/test_stage5_templates.py:714" in notes_text
    assert "tests/test_stage5_rewrite.py:1186" in notes_text
    assert "tests/test_generator.py:11092" in notes_text
    assert "tests/test_generator.py:12059" in notes_text

    rewrite_test_text = (REPO_ROOT / "tests" / "test_stage5_rewrite.py").read_text(encoding="utf-8")
    assert 'u3_sha = "f6277c6d6f52b6280b0f1cab518dd7d741a85c6e"' in rewrite_test_text
    assert '"tests/test_stage5_templates.py",' in rewrite_test_text


def test_10_existing_behaviour_unchanged(tmp_path: Path) -> None:
    """Test 10: Existing default path and Stage 4 path produce byte-identical records across repeated runs."""
    non_timing_keys = (
        "items_sha256",
        "part_json_sha256",
        "keys_sealed_sha256",
        "responses_sealed_sha256",
        "diagnostics_sealed_sha256",
        "shape_plan_telemetry",
        "recount_verification",
        "probe_report",
    )
    out_def_a = tmp_path / "def_a"
    out_def_b = tmp_path / "def_b"
    runtime_seed_a = "".join(f"{(i + 17):02x}" for i in range(32))
    runtime_seed_b = "".join(f"{(i + 49):02x}" for i in range(32))
    sum_def_a = build_and_probe_dataset(
        count=2,
        base_seed=20261030,
        private_seed_hex=runtime_seed_a,
        output_dir=out_def_a,
        writer_kind="mock",
    )
    sum_def_b = build_and_probe_dataset(
        count=2,
        base_seed=20261030,
        private_seed_hex=runtime_seed_a,
        output_dir=out_def_b,
        writer_kind="mock",
    )
    for k in non_timing_keys:
        assert sum_def_a[k] == sum_def_b[k]
    assert (out_def_a / "part_public" / "items.jsonl").read_bytes() == (
        out_def_b / "part_public" / "items.jsonl"
    ).read_bytes()
    assert (out_def_a / "sealed_store" / "keys.sealed.jsonl").read_bytes() == (
        out_def_b / "sealed_store" / "keys.sealed.jsonl"
    ).read_bytes()

    out_s4_a = tmp_path / "s4_a"
    out_s4_b = tmp_path / "s4_b"
    sum_s4_a = build_and_probe_dataset(
        count=2,
        base_seed=20261030,
        private_seed_hex=runtime_seed_b,
        output_dir=out_s4_a,
        writer_kind="mock",
        constraint_layout="table",
        accepted_forms_line="on",
        checker_profile="d",
        shape_plan="assigned",
        shape_plan_seed=20261103,
    )
    sum_s4_b = build_and_probe_dataset(
        count=2,
        base_seed=20261030,
        private_seed_hex=runtime_seed_b,
        output_dir=out_s4_b,
        writer_kind="mock",
        constraint_layout="table",
        accepted_forms_line="on",
        checker_profile="d",
        shape_plan="assigned",
        shape_plan_seed=20261103,
    )
    for k in non_timing_keys:
        assert sum_s4_a[k] == sum_s4_b[k]
    assert (out_s4_a / "part_public" / "items.jsonl").read_bytes() == (
        out_s4_b / "part_public" / "items.jsonl"
    ).read_bytes()
    assert (out_s4_a / "sealed_store" / "keys.sealed.jsonl").read_bytes() == (
        out_s4_b / "sealed_store" / "keys.sealed.jsonl"
    ).read_bytes()


def test_11_seed_bank_input_matches_paired_seed_bank_and_dry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test 11 (Addendum 1): `--stage5` takes a seed-bank file via `world_from_scenario_seed`, matches `run_paired_seed_bank` per rendering, records `seed_bank_sha256`/`seed_bank_size`, and prints them in `--stage5 --dry`."""
    assert MOCK_SEED_BANK_FIXTURE_PATH.is_file()
    raw_seeds = json.loads(MOCK_SEED_BANK_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw_seeds, list) and len(raw_seeds) == 40
    expected_sha256 = hashlib.sha256(MOCK_SEED_BANK_FIXTURE_PATH.read_bytes()).hexdigest()
    expected_size = len(raw_seeds)

    worlds = [world_from_scenario_seed(int(s), index=i) for i, s in enumerate(raw_seeds)]
    client, _ = _make_fake_rewrite_client(worlds)

    paired_queue = [
        (w, r_idx, w.agents[r_idx])
        for w in worlds
        for r_idx in (0, 1)
    ]
    paired_idx = 0

    def _fake_paired_transport(
        method: str, url: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        nonlocal paired_idx
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/show"):
            return {"parameters": "temperature 0.6"}
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            w, _, focal = paired_queue[paired_idx]
            paired_idx += 1
            txt = render_world_template(w, focal)
            return {
                "message": {"role": "assistant", "content": txt},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3200,
                "eval_count": 600,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    paired_report = run_paired_seed_bank(
        seed_bank=MOCK_SEED_BANK_FIXTURE_PATH,
        output_path=tmp_path / "paired_out" / "paired.json",
        grant_id="test-grant-u4-addendum1",
        http_transport=_fake_paired_transport,
        prompt_token_counter=lambda _: 3200,
    )
    paired_renderings = paired_report["renderings"]
    assert isinstance(paired_renderings, list) and len(paired_renderings) == 80

    out_dir = tmp_path / "stage5_bank_out"
    rc = cli_mod.main(
        [
            "--stage5",
            "--paired-seed-bank",
            str(MOCK_SEED_BANK_FIXTURE_PATH),
            "--out-dir",
            str(out_dir),
            "--stage5-plan-seed",
            "20261103",
            "--stage5-rung-seed",
            "20261104",
        ],
        rewrite_client=client,
    )
    assert rc == 0
    _ = capsys.readouterr()

    rec = json.loads((out_dir / STAGE5_RUN_RECORD_FILENAME).read_text(encoding="utf-8"))
    _assert_valid_stage5_record_schema(rec)
    assert rec["seed_bank_sha256"] == expected_sha256
    assert rec["seed_bank_size"] == expected_size
    assert len(rec["renderings"]) == len(paired_renderings) == 80

    sealed_path = out_dir / "sealed_store" / "responses.sealed.jsonl"
    sealed_focal_by_key: dict[tuple[str, int], str] = {}
    for raw_ln in sealed_path.read_text(encoding="utf-8").splitlines():
        if not raw_ln.strip():
            continue
        row = json.loads(raw_ln)
        if row.get("retry_form") == "":
            sealed_focal_by_key[(str(row["scenario_id"]), int(row["rendering_index"]))] = str(
                row["focal_agent"]
            )

    for s5_r, p_r in zip(rec["renderings"], paired_renderings):
        sid = str(p_r["scenario_id"])
        r_idx = int(p_r["rendering_index"])
        expected_focal = str(p_r["focal_agent"])
        assert s5_r["scenario_id"] == sid
        assert s5_r["rendering_index"] == r_idx
        assert s5_r["scenario_seed"] == int(p_r["scenario_seed"])
        assert (
            s5_r["focal_agent_sha256"]
            == hashlib.sha256(expected_focal.encode("utf-8")).hexdigest()
        )
        assert sealed_focal_by_key[(sid, r_idx)] == expected_focal

    rc_dry = cli_mod.main(
        [
            "--stage5",
            "--dry",
            "--stage5-seed-bank",
            str(MOCK_SEED_BANK_FIXTURE_PATH),
            "--stage5-plan-seed",
            "20261103",
        ]
    )
    assert rc_dry == 0
    dry_doc = json.loads(capsys.readouterr().out.strip())
    assert dry_doc["dry"] is True
    assert dry_doc["seed_bank_sha256"] == expected_sha256
    assert dry_doc["seed_bank_size"] == expected_size == 40
    assert dry_doc["scenarios"] == 40
    assert dry_doc["renderings"] == 80


# ===========================================================================
# U4 Addendum 2 Tests (Items 1–14)
# ===========================================================================


def test_12_addendum2_item1_rung_rule_exact_integers_and_34_float_pairs() -> None:
    """Addendum 2 Item 1: Every pair (k, k+6) for k=0..54 picks with_draft, every (k, k+5) picks facts_only, and 34 of 55 float pairs fall below 0.10."""
    float_below_point_one = [
        k for k in range(55) if ((k + 6) / 60.0) - (k / 60.0) < 0.10
    ]
    assert len(float_below_point_one) == 34

    for k in range(55):
        assert decide_rung_input_mode(60, k, 60, k + 6) == "with_draft"
    for k in range(56):
        assert decide_rung_input_mode(60, k, 60, k + 5) == "facts_only"

    assert "kept_with_draft - kept_facts_only >= 6 of 60" in STAGE5_RUNG_RULE
    assert "exact equivalent" in STAGE5_RUNG_RULE


def test_13_addendum2_item2_both_seeds_drawn_at_run_start_and_zero_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Addendum 2 Item 2: Fresh run makes exactly 2 secrets draws at run start; resume after a crash in B (before the rung) makes 0 draws."""
    n = 40
    base_seed = 20261103
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(n, base_seed=base_seed)]
    client, _ = _make_fake_rewrite_client(worlds)

    draws: list[int] = []

    def _spy_randbelow(upper: int) -> int:
        draws.append(upper)
        return 20261103 + len(draws) - 1

    monkeypatch.setattr(secrets, "randbelow", _spy_randbelow)

    # Fresh run crashing during B (after 5 renderings, before the rung)
    out_dir = tmp_path / "run_13_crash_in_b"
    b_count = 0

    def _crash_on_sixth_b(w: Any, fa: str) -> str:
        nonlocal b_count
        b_count += 1
        if b_count == 1:
            # Both plan_seed and rung_seed must already be in the record before the first writer call!
            rec_on_disk = json.loads((out_dir / STAGE5_RUN_RECORD_FILENAME).read_text(encoding="utf-8"))
            assert rec_on_disk["plan_seed"] == 20261103
            assert rec_on_disk["rung_seed"] == 20261104
            assert len(draws) == 2
        if b_count == 6:
            raise RuntimeError("Simulated crash in Phase 1 (B) before the rung")
        return _render_mock_stage5_baseline(w, fa)

    with pytest.raises(RuntimeError, match="Simulated crash in Phase 1"):
        run_stage5_pipeline(
            count=n,
            base_seed=base_seed,
            output_dir=out_dir,
            writer_kind="mock",
            rewrite_client=client,
            worlds=worlds,
            mock_custom_renderer=_crash_on_sixth_b,
        )
    assert len(draws) == 2

    # Resume from the Phase 1 crash -> ZERO secrets draws across the entire resumed run (including the rung!)
    draws.clear()
    resumed = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=out_dir,
        writer_kind="mock",
        resume=True,
        rewrite_client=client,
        worlds=worlds,
    )
    assert len(draws) == 0
    assert resumed["plan_seed"] == 20261103
    assert resumed["rung_seed"] == 20261104
    assert resumed["progress"]["rung_done"] is True
    assert resumed["progress"]["s_done"] == 80


def test_14_addendum2_item3_resume_after_s_lt_b_refuses_until_reset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Addendum 2 Item 3: Stop on S < B, resume without reset refuses with STAGE5_RESUME_AFTER_S_LT_B, resume with reset runs."""
    n = 40
    base_seed = 20261103
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(n, base_seed=base_seed)]
    client, _ = _make_fake_rewrite_client(worlds)

    fixture_rec = json.loads(MOCK_RECORD_FIXTURE_PATH.read_text(encoding="utf-8"))
    sampled_set = {
        (str(pair[0]), int(pair[1])) for pair in fixture_rec["rung"]["sampled_renderings"]
    }
    unsampled_key = next(
        (w.scenario_id, r_idx)
        for w in worlds
        for r_idx in (0, 1)
        if (w.scenario_id, r_idx) not in sampled_set
    )
    stop_sid, stop_r_idx = unsampled_key

    orig_swap = stage5_mod.swap_all_move_see
    sabotage = True

    def _bad_swap(prose: str, world: Any, plan: Any, **kwargs: Any) -> Any:
        res = orig_swap(prose, world, plan, **kwargs)
        if sabotage and world.scenario_id == stop_sid and kwargs.get("focal_agent") == world.agents[stop_r_idx]:
            return stage5_mod.SwapResult(
                prose=res.prose + " Quietly.",
                events=res.events,
                b_faults=res.b_faults,
                s_faults=res.s_faults,
                b_passed=res.b_passed,
                s_passed=False,
                seam_count=res.seam_count,
            )
        return res

    out_dir = tmp_path / "run_14"
    stage5_mod.swap_all_move_see = _bad_swap  # type: ignore[assignment]
    try:
        with pytest.raises(Stage5SLtBStopError):
            run_stage5_pipeline(
                count=n,
                base_seed=base_seed,
                output_dir=out_dir,
                writer_kind="mock",
                stage5_plan_seed=20261103,
                stage5_rung_seed=20261104,
                rewrite_client=client,
                worlds=worlds,
            )
    finally:
        stage5_mod.swap_all_move_see = orig_swap  # type: ignore[assignment]

    sabotage = False
    # 1. CLI resume without --stage5-redo-b exits non-zero with STAGE5_RESUME_AFTER_S_LT_B naming the rendering
    rc_refused = cli_mod.main(
        [
            "--stage5",
            "--resume",
            "--count",
            str(n),
            "--seed",
            str(base_seed),
            "--out-dir",
            str(out_dir),
        ],
        rewrite_client=client,
    )
    err_text = capsys.readouterr().err
    assert rc_refused == 1
    assert "STAGE5_RESUME_AFTER_S_LT_B" in err_text
    assert stop_sid in err_text
    assert f"rendering_index={stop_r_idx}" in err_text

    # 2. CLI resume with explicit --stage5-redo-b <sid>:<r_idx> runs and succeeds
    rc_ok = cli_mod.main(
        [
            "--stage5",
            "--resume",
            "--stage5-redo-b",
            f"{stop_sid}:{stop_r_idx}",
            "--count",
            str(n),
            "--seed",
            str(base_seed),
            "--out-dir",
            str(out_dir),
        ],
        rewrite_client=client,
    )
    _ = capsys.readouterr()
    assert rc_ok == 0
    rec_done = json.loads((out_dir / STAGE5_RUN_RECORD_FILENAME).read_text(encoding="utf-8"))
    assert rec_done["stop_reason"] is None
    assert rec_done["summary"]["s_passed"] == 80


def test_15_addendum2_item4_redo_b_sampled_by_rung_refuses_unless_redo_rung(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Addendum 2 Item 4: `--stage5-redo-b` on a rendering sampled by the rung refuses unless `--stage5-redo-rung` is also given (which redraws 0 seeds)."""
    n = 40
    base_seed = 20261103
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(n, base_seed=base_seed)]
    client, _ = _make_fake_rewrite_client(worlds)

    out_dir = tmp_path / "run_15"
    rec = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=out_dir,
        writer_kind="mock",
        stage5_plan_seed=20261103,
        stage5_rung_seed=20261104,
        rewrite_client=client,
        worlds=worlds,
    )
    sampled_pair = rec["rung"]["sampled_renderings"][0]
    sampled_sid, sampled_r_idx = str(sampled_pair[0]), int(sampled_pair[1])

    # 1. Refused without --stage5-redo-rung
    with pytest.raises(Stage5RedoBSampledByRungError) as exc_info:
        run_stage5_pipeline(
            count=n,
            base_seed=base_seed,
            output_dir=out_dir,
            writer_kind="mock",
            resume=True,
            redo_b=[f"{sampled_sid}:{sampled_r_idx}"],
            redo_rung=False,
            rewrite_client=client,
            worlds=worlds,
        )
    assert "STAGE5_REDO_B_SAMPLED_BY_RUNG" in str(exc_info.value)
    assert f"{sampled_sid}:{sampled_r_idx}" in str(exc_info.value)

    # 2. Succeeds with --stage5-redo-rung, records redo_rung=True, and makes 0 secrets draws (reuses rung_seed)
    draws: list[int] = []
    monkeypatch.setattr(secrets, "randbelow", lambda u: (draws.append(u), 999)[1])

    rec_redo = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=out_dir,
        writer_kind="mock",
        resume=True,
        redo_b=[f"{sampled_sid}:{sampled_r_idx}"],
        redo_rung=True,
        rewrite_client=client,
        worlds=worlds,
    )
    assert draws == []
    assert rec_redo["redo_b"] is True
    assert rec_redo["redo_rung"] is True
    assert rec_redo["rung_seed"] == 20261104
    assert rec_redo["rung"]["sampled_renderings"] == rec["rung"]["sampled_renderings"]


def test_16_addendum2_item5_over_cap_excluded_from_rung_and_zero_reads_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Addendum 2 Item 5: Over-cap rendering never enters the rung pool, and on resume from B makes 0 writer calls and 0 sealed-store reads."""
    n = 40
    base_seed = 20261103
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(n, base_seed=base_seed)]
    client, _ = _make_fake_rewrite_client(worlds)

    # Force ("scen_0000", 0) to be over cap via prompt_token_counter (call 1 of initial run)
    tok_call_count = 0

    def _token_counter(rendered_prompt: str) -> int:
        nonlocal tok_call_count
        _ = rendered_prompt
        tok_call_count += 1
        if tok_call_count == 1:
            return 5000
        return 3500

    out_dir = tmp_path / "run_16"
    b_calls_made_for: list[tuple[str, int]] = []
    crash_after_ten = True

    def _renderer(w: Any, fa: str) -> str:
        r_idx_cur = 0 if fa == w.agents[0] else 1
        b_calls_made_for.append((w.scenario_id, r_idx_cur))
        if crash_after_ten and len(b_calls_made_for) == 10:
            raise RuntimeError("Simulated crash in B after 10 non-over-cap renderings")
        return _render_mock_stage5_baseline(w, fa)

    with pytest.raises(RuntimeError, match="Simulated crash in B"):
        run_stage5_pipeline(
            count=n,
            base_seed=base_seed,
            output_dir=out_dir,
            writer_kind="mock",
            stage5_plan_seed=20261103,
            stage5_rung_seed=20261104,
            prompt_token_counter=_token_counter,
            rewrite_client=client,
            worlds=worlds,
            mock_custom_renderer=_renderer,
        )
    assert ("scen_0000", 0) not in b_calls_made_for

    # Spy on _read_sealed_b_prose_for_rendering during resume
    sealed_reads_for: list[tuple[str, int]] = []
    orig_read = stage5_mod._read_sealed_b_prose_for_rendering

    def _spy_read(path: Path, sid: str, r_idx: int, **kw: Any) -> str | None:
        sealed_reads_for.append((sid, int(r_idx)))
        return orig_read(path, sid, r_idx, **kw)

    monkeypatch.setattr(stage5_mod, "_read_sealed_b_prose_for_rendering", _spy_read)
    crash_after_ten = False
    b_calls_made_for.clear()

    rec = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=out_dir,
        writer_kind="mock",
        resume=True,
        prompt_token_counter=_token_counter,
        rewrite_client=client,
        worlds=worlds,
        mock_custom_renderer=_renderer,
    )

    # ("scen_0000", 0) made 0 writer calls and 0 sealed-store reads on resume, and never entered the rung pool
    assert ("scen_0000", 0) not in b_calls_made_for
    assert ("scen_0000", 0) not in sealed_reads_for
    assert ["scen_0000", 0] not in rec["rung"]["sampled_renderings"]
    assert rec["summary"]["ctx_over_cap"] == 1
    assert rec["renderings"][0]["ctx_over_cap"] is True


def test_17_addendum2_item6_mid_rung_crash_call_45_and_resume(tmp_path: Path) -> None:
    """Addendum 2 Item 6: Fake rewrite client raises on call 45 of the rung; resume restarts the rung from scratch with the same rung_seed and same 60 events, keeping rung_aborted_calls=45 apart from rung_calls=120."""
    n = 40
    base_seed = 20261103
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(n, base_seed=base_seed)]
    base_client, base_calls = _make_fake_rewrite_client(worlds)

    call_idx = 0
    crash_on_45 = True

    def _flaky_client(payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal call_idx
        call_idx += 1
        if crash_on_45 and call_idx == 45:
            raise RuntimeError("Simulated mid-rung crash on call 45")
        return base_client(payload)

    out_dir = tmp_path / "run_17"
    with pytest.raises(RuntimeError, match="Simulated mid-rung crash on call 45"):
        run_stage5_pipeline(
            count=n,
            base_seed=base_seed,
            output_dir=out_dir,
            writer_kind="mock",
            stage5_plan_seed=20261103,
            stage5_rung_seed=20261104,
            rewrite_client=_flaky_client,
            worlds=worlds,
        )

    aborted_rec = json.loads((out_dir / STAGE5_RUN_RECORD_FILENAME).read_text(encoding="utf-8"))
    assert aborted_rec["rung_aborted_calls"] == 45
    assert aborted_rec["call_counts"]["rung_aborted_calls"] == 45
    assert aborted_rec["call_counts"]["rung_calls"] == 0
    assert aborted_rec["progress"]["rung_done"] is False
    first_44_Before_crash = [(c["scenario_id"], c["target_form"]) for c in base_calls[:44]]

    # Resume -> restarts rung from scratch with same rung_seed and same 60 events
    crash_on_45 = False
    base_calls.clear()
    resumed_rec = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=out_dir,
        writer_kind="mock",
        resume=True,
        rewrite_client=_flaky_client,
        worlds=worlds,
    )
    assert resumed_rec["rung_aborted_calls"] == 45
    assert resumed_rec["call_counts"]["rung_aborted_calls"] == 45
    assert resumed_rec["call_counts"]["rung_calls"] == 120
    assert resumed_rec["progress"]["rung_done"] is True
    resumed_first_44 = [(c["scenario_id"], c["target_form"]) for c in base_calls[:44]]
    assert resumed_first_44 == first_44_Before_crash


def test_18_addendum2_item7_canonical_rung_pool_and_mode_order_invariance(tmp_path: Path) -> None:
    """Addendum 2 Item 7: First drafts completed in a shuffled order give the same 60 event ids, and the two rung modes run in both orders give identical results per event."""
    n = 40
    base_seed = 20261103
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(n, base_seed=base_seed)]

    all_r_keys = [(w.scenario_id, r_idx) for w in worlds for r_idx in (0, 1)]
    shuffled_r_keys = list(all_r_keys)
    random.Random(99999).shuffle(shuffled_r_keys)
    assert shuffled_r_keys != all_r_keys

    client_a, calls_a = _make_fake_rewrite_client(worlds)
    rec_a = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=tmp_path / "run_18_canonical",
        writer_kind="mock",
        stage5_plan_seed=20261103,
        stage5_rung_seed=20261104,
        rewrite_client=client_a,
        worlds=worlds,
        rung_mode_order=("facts_only", "with_draft"),
    )

    client_b, calls_b = _make_fake_rewrite_client(worlds)
    rec_b = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=tmp_path / "run_18_shuffled_reversed",
        writer_kind="mock",
        stage5_plan_seed=20261103,
        stage5_rung_seed=20261104,
        rewrite_client=client_b,
        worlds=worlds,
        rendering_execution_order=shuffled_r_keys,
        rung_mode_order=("with_draft", "facts_only"),
    )

    assert rec_a["rung"] == rec_b["rung"]
    # Compare the 60 facts_only and 60 with_draft rung calls in both runs
    fo_a = [(c["scenario_id"], c["target_form"], c["content"]) for c in calls_a[:60]]
    wd_a = [(c["scenario_id"], c["target_form"], c["content"]) for c in calls_a[60:120]]
    wd_b = [(c["scenario_id"], c["target_form"], c["content"]) for c in calls_b[:60]]
    fo_b = [(c["scenario_id"], c["target_form"], c["content"]) for c in calls_b[60:120]]
    assert fo_a == fo_b
    assert wd_a == wd_b
    assert rec_a["renderings"] == rec_b["renderings"]


def test_19_addendum2_item8_rewrite_input_enum_and_canary_test(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Addendum 2 Item 8: `rewrite_input` is only `facts_only` | `with_draft` | None, and unique canary tokens on every mock writer and fake rewrite reply appear only in the sealed store, never in record, stdout, stderr, or temp files."""
    n = 40
    base_seed = 20261103
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(n, base_seed=base_seed)]

    canary_tokens: list[str] = []
    b_idx = 0

    def _canary_b_renderer(w: Any, fa: str) -> str:
        nonlocal b_idx
        b_idx += 1
        tok = _build_runtime_canary_marker("draft", b_idx)
        canary_tokens.append(tok)
        base_prose = _render_mock_stage5_baseline(w, fa)
        # Include a canary token even on a line with no names (checker will refuse it as unlogged, or we embed it in raw B)
        return base_prose + f"\n{tok}."

    # Also test a passing run where every rewrite reply carries a unique canary token (including lines with no names)
    rw_idx = 0

    def _canary_rw_client(payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal rw_idx
        _ = payload
        rw_idx += 1
        tok = _build_runtime_canary_marker("rewrite", rw_idx)
        canary_tokens.append(tok)
        return {
            "message": {"role": "assistant", "content": f"The {tok} remained silent."},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 32,
            "eval_count": 4,
        }

    out_dir = tmp_path / "run_19_canary"
    rec = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=out_dir,
        writer_kind="mock",
        stage5_plan_seed=20261103,
        stage5_rung_seed=20261104,
        rewrite_client=_canary_rw_client,
        worlds=worlds,
    )
    # Also run a 1-scenario run with _canary_b_renderer into a sub-dir to verify B canary tokens on nameless lines
    out_dir_b = tmp_path / "run_19_canary_b"
    run_stage5_pipeline(
        count=1,
        base_seed=base_seed,
        output_dir=out_dir_b,
        writer_kind="mock",
        stage5_plan_seed=20261103,
        stage5_rung_seed=20261104,
        s_code_say=True,
        worlds=worlds[:1],
        mock_custom_renderer=_canary_b_renderer,
    )

    captured = capsys.readouterr()
    assert len(canary_tokens) > 40

    # Assert rewrite_input is strictly the enum ("facts_only", "with_draft", None)
    for r in rec["renderings"]:
        for ev in r["events"]:
            assert ev["rewrite_input"] in ("facts_only", "with_draft", None)

    # Check all files under out_dir and out_dir_b: no temp files exist, and non-sealed files + stdout + stderr contain ZERO canary tokens
    for root_dir in (out_dir, out_dir_b):
        for p in root_dir.rglob("*"):
            if p.is_file():
                assert not p.name.endswith(".tmp"), f"Leftover temp file found: {p}"
                if "sealed_store" not in p.parts:
                    txt = p.read_text(encoding="utf-8")
                    for tok in canary_tokens:
                        assert tok not in txt, f"Canary token {tok!r} leaked into {p.name}"

    for tok in canary_tokens:
        assert tok not in captured.out
        assert tok not in captured.err

    # Every canary token IS present in the sealed store
    combined_sealed = (
        (out_dir / "sealed_store" / "responses.sealed.jsonl").read_text(encoding="utf-8")
        + "\n"
        + (out_dir_b / "sealed_store" / "responses.sealed.jsonl").read_text(encoding="utf-8")
    )
    for tok in canary_tokens:
        assert tok in combined_sealed, f"Canary token {tok!r} missing from sealed store"


def test_20_addendum2_item9_write_order_and_atomic_os_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Addendum 2 Item 9: Each sealed blob is written and flushed before the record that refers to it is atomically replaced via `os.replace`."""
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(2, base_seed=20261103)]
    events_log: list[tuple[str, int]] = []

    orig_flush = stage5_mod._flush_sealed_store
    orig_replace = os.replace

    def _spy_flush(sealed_path: Path) -> None:
        orig_flush(sealed_path)
        line_count = sum(
            1 for ln in sealed_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        )
        events_log.append(("sealed_flushed", line_count))

    def _spy_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        dst_path = Path(dst)
        orig_replace(src, dst)
        if dst_path.name == STAGE5_RUN_RECORD_FILENAME:
            rec_obj = json.loads(dst_path.read_text(encoding="utf-8"))
            b_done = int(rec_obj["progress"]["b_done"])
            s_done = int(rec_obj["progress"]["s_done"])
            events_log.append(("record_replaced", b_done + s_done))
            if b_done + s_done > 0:
                # Most recent sealed_flushed must precede this record_replaced and have >= b_done + s_done entries
                flushes = [cnt for kind, cnt in events_log if kind == "sealed_flushed"]
                assert len(flushes) > 0 and flushes[-1] >= b_done + s_done

    monkeypatch.setattr(stage5_mod, "_flush_sealed_store", _spy_flush)
    monkeypatch.setattr(os, "replace", _spy_replace)

    out_dir = tmp_path / "run_20"
    run_stage5_pipeline(
        count=2,
        base_seed=20261103,
        output_dir=out_dir,
        writer_kind="mock",
        stage5_plan_seed=20261103,
        stage5_rung_seed=20261104,
        s_code_say=True,
        worlds=worlds,
    )
    assert any(k == "sealed_flushed" for k, _ in events_log)
    assert any(k == "record_replaced" for k, _ in events_log)


def test_21_addendum2_item10_resume_refuses_on_env_drift(tmp_path: Path) -> None:
    """Addendum 2 Item 10: Resume refuses with `STAGE5_RESUME_ENV_CHANGED` on model, digest, ollama_version, num_predict, max_num_ctx, or module sha256 drift."""
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(2, base_seed=20261103)]
    out_dir = tmp_path / "run_21"
    run_stage5_pipeline(
        count=2,
        base_seed=20261103,
        output_dir=out_dir,
        writer_kind="mock",
        stage5_plan_seed=20261103,
        stage5_rung_seed=20261104,
        s_code_say=True,
        worlds=worlds,
    )
    rec_path = out_dir / STAGE5_RUN_RECORD_FILENAME
    orig_rec = json.loads(rec_path.read_text(encoding="utf-8"))

    # 1. Parameter drift: num_predict and max_num_ctx
    with pytest.raises(Stage5ResumeEnvChangedError) as exc_np:
        run_stage5_pipeline(
            count=2,
            base_seed=20261103,
            output_dir=out_dir,
            writer_kind="mock",
            resume=True,
            num_predict=2048,
            s_code_say=True,
            worlds=worlds,
        )
    assert "STAGE5_RESUME_ENV_CHANGED" in str(exc_np.value)
    assert "num_predict" in exc_np.value.drifted_fields

    with pytest.raises(Stage5ResumeEnvChangedError) as exc_ctx:
        run_stage5_pipeline(
            count=2,
            base_seed=20261103,
            output_dir=out_dir,
            writer_kind="mock",
            resume=True,
            max_num_ctx=8192,
            s_code_say=True,
            worlds=worlds,
        )
    assert "max_num_ctx" in exc_ctx.value.drifted_fields

    # 2. Record mutation drift: model, model_digest, ollama_version, and each of render.py / shape_templates.py / rewrite.py
    mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        ("engine.model", lambda d: d["engine"].__setitem__("model", "other:model")),
        ("engine.model_digest", lambda d: d["engine"].__setitem__("model_digest", hashlib.sha256(b"drift-digest").hexdigest())),
        ("engine.ollama_version", lambda d: d["engine"].__setitem__("ollama_version", "0.99.0")),
        ("module_sha256s.render.py", lambda d: d["module_sha256s"].__setitem__("render.py", hashlib.sha256(b"drift-render").hexdigest())),
        ("module_sha256s.shape_templates.py", lambda d: d["module_sha256s"].__setitem__("shape_templates.py", hashlib.sha256(b"drift-templates").hexdigest())),
        ("module_sha256s.rewrite.py", lambda d: d["module_sha256s"].__setitem__("rewrite.py", hashlib.sha256(b"drift-rewrite").hexdigest())),
    ]
    for expected_field, mutator in mutations:
        tampered = json.loads(json.dumps(orig_rec))
        mutator(tampered)
        rec_path.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with pytest.raises(Stage5ResumeEnvChangedError) as exc_mut:
            run_stage5_pipeline(
                count=2,
                base_seed=20261103,
                output_dir=out_dir,
                writer_kind="mock",
                resume=True,
                s_code_say=True,
                worlds=worlds,
            )
        assert expected_field in exc_mut.value.drifted_fields


def test_22_addendum2_item11_dry_mode_upper_bound_and_over_cap_count() -> None:
    """Addendum 2 Item 11: Dry-mode call counts are labelled upper bounds before the context cap, and dry mode prints `ctx_over_cap` without a writer."""
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(2, base_seed=20261103)]
    tok_calls = 0

    def _counter(text: str) -> int:
        nonlocal tok_calls
        _ = text
        tok_calls += 1
        return 5000 if tok_calls == 1 else 3200

    dry_res = run_stage5_dry(
        count=2,
        base_seed=20261103,
        stage5_plan_seed=20261103,
        max_retries=1,
        prompt_token_counter=_counter,
        worlds=worlds,
    )
    assert dry_res["expected_call_counts_is_upper_bound_before_ctx_cap"] is True
    assert dry_res["expected_call_counts_upper_bound_before_ctx_cap"] == {
        "b_calls": 8,
        "rung_calls": 120,
        "rewrite_calls": 12,
    }
    assert dry_res["ctx_over_cap"] == 1


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Addendum 2 Item 12 STOP & REPORT: generate_stage5_shape_plan (src/udlab_generator/shape_templates.py:969) "
        "slices 20-scenario blocks rather than 20-event blocks per variant stream, resets last_kind_forms=[] on "
        "every scenario and block boundary (allowing 3-in-a-row across boundaries), and samples extra remainder "
        "forms independently per block without rotating the short shape (allowing global form spread > 1). "
        "Per card U4 & Addendum 2 Item 12: do not modify shape_templates.py."
    ),
)
def test_23_addendum2_item12_multi_block_plan_check() -> None:
    """Addendum 2 Item 12: Multi-block plan test over >= 3 blocks per variant stream against `generate_stage5_shape_plan` (`shape_templates.py:969`)."""
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(60, base_seed=20261103)]
    plan = generate_stage5_shape_plan(worlds, seed=20261103, block_size=20)
    recs = plan["plan_records"]

    for v, forms in STAGE5_ASSIGNED_VARIANTS.items():
        seq = [str(r["form_id"]) for r in recs if r["variant"] == v]
        assert len(seq) >= 60, f"Variant {v} needs at least 3 blocks of 20 events, got {len(seq)}"
        # 1. No three-in-a-row across block edges (indices 18-20, 38-40) or anywhere in stream
        for edge_start in (18, 38):
            window = seq[edge_start : edge_start + 3]
            assert len(set(window)) > 1, f"Three-in-a-row at edge {edge_start} for {v}: {window}"
        for i in range(len(seq) - 2):
            assert not (seq[i] == seq[i + 1] == seq[i + 2]), f"Three-in-a-row at {i} for {v}"
        # 2. Spread <= 1 on every 20-event block, including the tail block
        for b_start in range(0, len(seq), 20):
            blk = seq[b_start : b_start + 20]
            c = Counter(blk)
            spread = max(c.get(f, 0) for f in forms) - min(c.get(f, 0) for f in forms)
            assert spread <= 1, f"Block spread {spread} > 1 at {b_start} for {v}"
        # 3. Short shape rotated across blocks so global counts stay within 1
        g_counts = Counter(seq)
        global_spread = max(g_counts.get(f, 0) for f in forms) - min(g_counts.get(f, 0) for f in forms)
        assert global_spread <= 1, f"Global spread {global_spread} > 1 for {v}: {dict(g_counts)}"


def test_24_addendum2_item13_rung_runs_on_post_swap_scene(tmp_path: Path) -> None:
    """Addendum 2 Item 13: The rung applies `swap_all_move_see` in memory to each sampled rendering's B before running its two modes, without writing the rung's swap to S."""
    n = 40
    base_seed = 20261103
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(n, base_seed=base_seed)]
    client, _ = _make_fake_rewrite_client(worlds)

    out_dir = tmp_path / "run_24"
    rec = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=out_dir,
        writer_kind="mock",
        stage5_plan_seed=20261103,
        stage5_rung_seed=20261104,
        rewrite_client=client,
        worlds=worlds,
    )

    plan_bundle = generate_stage5_shape_plan(worlds, seed=20261103, block_size=20)
    plan_by_sid = plan_bundle["plan_by_scenario_id"]
    worlds_by_sid = {w.scenario_id: w for w in worlds}

    sealed_path = out_dir / "sealed_store" / "responses.sealed.jsonl"
    rung_entries = [
        json.loads(ln)
        for ln in sealed_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and json.loads(ln).get("retry_form") == "stage5_rung"
    ]
    assert len(rung_entries) == 120

    verified_diff_from_raw_b = 0
    for entry in rung_entries:
        sid = str(entry["scenario_id"])
        r_idx = int(entry["rendering_index"])
        w = worlds_by_sid[sid]
        focal = w.agents[r_idx]
        raw_b = _render_mock_stage5_baseline(w, focal)
        scen_plan = stage5_mod._scenario_stage5_plan_dict(plan_by_sid[sid], w)
        expected_post_swap = swap_all_move_see(
            raw_b, w, scen_plan, focal_agent=focal, checker_profile="d"
        ).prose
        thinking_obj = json.loads(entry["thinking_text"])
        assert thinking_obj["context_prose"] == expected_post_swap
        if expected_post_swap != raw_b:
            assert thinking_obj["context_prose"] != raw_b
            verified_diff_from_raw_b += 1

    assert verified_diff_from_raw_b > 0
    assert rec["summary"]["s_passed"] == 80


def test_25_addendum2_item14_at_most_one_rung_event_per_rendering_and_overlap() -> None:
    """Addendum 2 Item 14: At most one rung event is drawn per rendering, `rung_scenario_overlap` is exact, and a pool short under the 1-per-rendering cap refuses with `STAGE5_RUNG_SAMPLE_SHORT`."""
    # 1. Pool where every rendering has multiple eligible events (2 per rendering across 60 renderings):
    # yields at most 1 event per rendering across all 60 drawn events.
    eligible_multi: dict[str, list[tuple[str, int, int, str]]] = {
        "SAY_2": [],
        "SAY_3": [],
        "SAY_4": [],
    }
    for idx in range(35):
        sid = f"scen_{idx:04d}"
        for r_idx in (0, 1):
            for t_idx, t_form in enumerate(("SAY_2", "SAY_3", "SAY_4"), start=1):
                # Put two eligible events for each target in the same rendering
                eligible_multi[t_form].append((sid, r_idx, t_idx * 2 - 1, t_form))
                eligible_multi[t_form].append((sid, r_idx, t_idx * 2, t_form))

    sampled, overlap = _sample_rung_events_at_most_one_per_rendering(
        eligible_multi, rung_seed=20261104, per_target=20
    )
    assert len(sampled) == 60
    sampled_renderings = [(sid, r_idx) for sid, r_idx, _, _ in sampled]
    # Strictly at most 1 event per rendering across all 60 events!
    assert len(set(sampled_renderings)) == 60

    by_sid: dict[str, set[int]] = {}
    for sid, r_idx in sampled_renderings:
        by_sid.setdefault(sid, set()).add(r_idx)
    expected_overlap = sum(1 for s in by_sid.values() if len(s) == 2)
    assert overlap == expected_overlap

    # 2. A pool that has >= 20 total events for SAY_2 (e.g. 30 events) but all in only 15 renderings (< 20 renderings)
    # must refuse with STAGE5_RUNG_SAMPLE_SHORT under the 1-per-rendering cap!
    short_renderings_pool: dict[str, list[tuple[str, int, int, str]]] = {
        "SAY_2": [
            (f"scen_{i:04d}", 0, ev_i, "SAY_2")
            for i in range(15)
            for ev_i in (1, 2)  # 30 events, but only 15 distinct renderings!
        ],
        "SAY_3": [(f"scen_{i:04d}", 0, 3, "SAY_3") for i in range(15, 40)],
        "SAY_4": [(f"scen_{i:04d}", 1, 5, "SAY_4") for i in range(15, 40)],
    }
    with pytest.raises(Stage5RungSampleShortError) as exc_short:
        _sample_rung_events_at_most_one_per_rendering(
            short_renderings_pool, rung_seed=20261104, per_target=20
        )
    assert "STAGE5_RUNG_SAMPLE_SHORT" in str(exc_short.value)
    assert exc_short.value.counts["SAY_2"] == 15
