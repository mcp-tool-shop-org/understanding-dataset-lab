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
    assert isinstance(rec["model_endpoint"], dict)
    assert set(rec["model_endpoint"].keys()) == {"b", "rung", "swap", "rewrite"}
    for ep_val in rec["model_endpoint"].values():
        assert isinstance(ep_val, str) and ep_val.startswith(("http://", "https://"))
        assert "/" not in ep_val.split("://", 1)[1]
    assert isinstance(rec["step_timings"], dict)
    assert set(rec["step_timings"].keys()) == {"b", "rung", "swap", "rewrite"}
    for st_val in rec["step_timings"].values():
        assert isinstance(st_val, dict)
        assert st_val["start"] is None or isinstance(st_val["start"], str)
        assert st_val["end"] is None or isinstance(st_val["end"], str)
        assert isinstance(st_val["seconds"], (int, float)) and not isinstance(st_val["seconds"], bool)
        assert isinstance(st_val["calls"], int) and not isinstance(st_val["calls"], bool)
        assert isinstance(st_val["call_count"], int) and not isinstance(st_val["call_count"], bool)
        assert st_val["calls"] == st_val["call_count"]
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
    import datetime

    n = 40
    base_seed = 20261103
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(n, base_seed=base_seed)]
    client, calls = _make_fake_rewrite_client(worlds)

    utc_tick = 0
    mono_tick = 0

    def _fixture_utc_clock() -> str:
        nonlocal utc_tick
        dt = datetime.datetime(2026, 11, 3, 0, 0, 0, tzinfo=datetime.timezone.utc) + datetime.timedelta(
            seconds=utc_tick
        )
        utc_tick += 1
        return dt.isoformat().replace("+00:00", "Z")

    def _fixture_mono_clock() -> float:
        nonlocal mono_tick
        val = float(mono_tick)
        mono_tick += 1
        return val

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
        utc_clock=_fixture_utc_clock,
        monotonic_clock=_fixture_mono_clock,
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


def test_23_addendum2_item12_multi_block_plan_check() -> None:
    """Addendum 2 Item 12: Multi-block plan test over >= 3 blocks per variant stream against `generate_stage5_shape_plan` (`shape_templates.py:969`)."""
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(60, base_seed=20261103)]
    plan = generate_stage5_shape_plan(worlds, seed=20261103, block_size=20)
    recs = plan["plan_records"]

    empty_streams = {
        v
        for v in STAGE5_ASSIGNED_VARIANTS
        if not any(r["variant"] == v for r in recs)
    }
    assert empty_streams == {"move_occluded"}

    for v, forms in STAGE5_ASSIGNED_VARIANTS.items():
        seq = [str(r["form_id"]) for r in recs if r["variant"] == v]
        if not seq:
            continue
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


def test_26_u4_fix1_f2_ollama_version_drift_refusal_on_real_writer_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """U4-fix1 Item 1 (F2): On the real `ollama` writer path, `--resume` compares the owned server's live `version` before the first writer call, refuses with `STAGE5_RESUME_ENV_CHANGED` (`engine.ollama_version`) with zero writer calls, never overwrites the recorded version on close, and on a fresh run records the live version at start."""
    from types import SimpleNamespace
    from udlab_generator.render import (
        DEFAULT_WRITER_MODEL_PIN,
        GenerationCallRecord,
        SealedResponseEntry,
        append_sealed_response,
        compute_sealed_response_digests,
    )

    worlds = [generate_world(cfg) for cfg in make_balanced_configs(2, base_seed=20261103)]
    live_server_version = "0.6.1"
    writer_calls: list[tuple[str, int]] = []
    crash_on_second_call = True

    class _FakeOllamaWriter:
        def __init__(self, **kwargs: Any) -> None:
            self._sealed_responses_path: Path | None = kwargs.get("sealed_responses_path")
            self.owned_server = SimpleNamespace(
                version=live_server_version,
                ollama_version=live_server_version,
                stop=lambda: None,
            )
            self.manifest_digest = DEFAULT_WRITER_MODEL_PIN.manifest_digest
            self.call_records: list[GenerationCallRecord] = []
            self.sealed_responses: list[SealedResponseEntry] = []
            self._transport = lambda *a, **kw: {}

        def _chat_render(
            self,
            *,
            world: Any,
            scenario_id: str,
            rendering_index: int,
            focal_agent: str,
            prompt_text: str,
            allow_refusal: bool = False,
        ) -> str | None:
            _ = (prompt_text, allow_refusal)
            writer_calls.append((scenario_id, rendering_index))
            if crash_on_second_call and len(writer_calls) >= 2:
                raise RuntimeError("simulated crash after first B rendering")
            prose = _render_mock_stage5_baseline(world, focal_agent)
            c_idx = len(self.call_records) + 1
            resp_sha, think_sha = compute_sealed_response_digests(prose, "")
            rec = GenerationCallRecord(
                call_id=f"{scenario_id}:r{rendering_index}:call_{c_idx:04d}",
                call_index=c_idx,
                scenario_id=scenario_id,
                rendering_index=rendering_index,
                focal_agent=focal_agent,
                operation="render_scenario",
                model=DEFAULT_WRITER_MODEL_PIN.ollama_tag,
                manifest_digest=DEFAULT_WRITER_MODEL_PIN.manifest_digest,
                model_layer_digest=DEFAULT_WRITER_MODEL_PIN.model_layer_digest,
                licence_id=DEFAULT_WRITER_MODEL_PIN.licence_id,
                prompt_sha256=resp_sha,
                prompt_tokens_estimate=128,
                prompt_tokens_exact=128,
                prompt_eval_count=128,
                eval_count=64,
                thinking_words=0,
                num_predict=DEFAULT_WRITER_NUM_PREDICT,
                num_ctx=2048,
                max_num_ctx=DEFAULT_WRITER_MAX_NUM_CTX,
                think=False,
                temperature=0.0,
                min_p=None,
                done_reason="stop",
                seconds=0.01,
                vram_peak_mib=0,
                vram_halt_mib=26000,
                grant_id="cpu-mock",
                checks_passed=True,
                refusal_reasons=(),
                check_id=None,
                attempt=0,
                seed=12345,
                response_sha256=resp_sha,
                thinking_sha256=think_sha,
            )
            self.call_records.append(rec)
            sealed_entry = SealedResponseEntry(
                call_id=rec.call_id,
                call_index=c_idx,
                scenario_id=scenario_id,
                rendering_index=rendering_index,
                focal_agent=focal_agent,
                attempt=0,
                seed=12345,
                response_text=prose,
                thinking_text="",
                response_sha256=resp_sha,
                thinking_sha256=think_sha,
            )
            self.sealed_responses.append(sealed_entry)
            if self._sealed_responses_path is not None:
                append_sealed_response(self._sealed_responses_path, sealed_entry)
            return prose

        def close(self) -> None:
            pass

    monkeypatch.setattr(stage5_mod, "OllamaWriter", _FakeOllamaWriter)

    out_dir = tmp_path / "run_26_f2"
    version_at_first_call: list[str | None] = []

    def _inspect_record_before_first_call(rec_path: Path) -> None:
        loaded = json.loads(rec_path.read_text(encoding="utf-8"))
        version_at_first_call.append(loaded["engine"]["ollama_version"])

    with pytest.raises(RuntimeError, match="simulated crash after first B rendering"):
        run_stage5_pipeline(
            count=2,
            base_seed=20261103,
            output_dir=out_dir,
            writer_kind="ollama",
            stage5_plan_seed=20261103,
            stage5_rung_seed=20261104,
            s_code_say=True,
            worlds=worlds,
            on_before_first_writer_call=_inspect_record_before_first_call,
        )

    rec_after_crash = json.loads((out_dir / STAGE5_RUN_RECORD_FILENAME).read_text(encoding="utf-8"))
    assert rec_after_crash["engine"]["ollama_version"] == "0.6.1"

    # Now resume with a server reporting a different version ("0.6.2"): must refuse with STAGE5_RESUME_ENV_CHANGED and 0 writer calls
    writer_calls.clear()
    crash_on_second_call = False
    live_server_version = "0.6.2"

    with pytest.raises(Stage5ResumeEnvChangedError) as exc_info:
        run_stage5_pipeline(
            count=2,
            base_seed=20261103,
            output_dir=out_dir,
            writer_kind="ollama",
            resume=True,
            s_code_say=True,
            worlds=worlds,
        )
    assert "STAGE5_RESUME_ENV_CHANGED" in str(exc_info.value)
    assert "engine.ollama_version" in exc_info.value.drifted_fields
    assert len(writer_calls) == 0

    # Recorded version on disk must never be overwritten on close during a refused resume
    rec_after_refusal = json.loads((out_dir / STAGE5_RUN_RECORD_FILENAME).read_text(encoding="utf-8"))
    assert rec_after_refusal["engine"]["ollama_version"] == "0.6.1"
    # On a fresh run, the live version is recorded at start once the server reports it, before the first writer call
    assert version_at_first_call == ["0.6.1"]


def test_27_u4_fix1_f1_error_handler_scoped_to_stage5_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """U4-fix1 Item 2 (F1): `except (Stage5PipelineError, MaxNumCtxRefusalError)` applies only to `--stage5`; non-stage5 runs with `max_num_ctx` over the cap behave identically to base `f6277c6d`."""
    over_cap = DEFAULT_WRITER_MAX_NUM_CTX + 1

    # 1. Direct call to build_and_probe_dataset propagates MaxNumCtxRefusalError
    with pytest.raises(MaxNumCtxRefusalError, match="MAX_NUM_CTX_CAP"):
        build_and_probe_dataset(
            count=1,
            base_seed=20261030,
            output_dir=tmp_path / "non_stage5_direct",
            writer_kind="mock",
            max_num_ctx=over_cap,
        )

    # 2. CLI main() on non-stage5 mode hits the existing `except RenderCheckError` handler (at base f6277c6d),
    # printing `format_non_sealed_refusal_reason(check_id="max_num_ctx_exceeded", ...)` and returning 1.
    rc = cli_mod.main(
        [
            "--count",
            "1",
            "--seed",
            "20261030",
            "--writer",
            "mock",
            "--max-num-ctx",
            str(over_cap),
            "--out-dir",
            str(tmp_path / "non_stage5_cli"),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert (
        captured.err.strip()
        == "max_num_ctx_exceeded: event_index=None event_kind=None failing_sentence_sha256= failing_sentence_offset=None failing_sentence_length=None"
    )

    # 3. Verify that Stage5PipelineError is NOT caught by main() when --stage5 is False
    def _raise_stage5_err(*args: Any, **kwargs: Any) -> Any:
        raise Stage5ResumeNoPlanSeedError("unexpected stage5 error outside stage5")

    monkeypatch.setattr(cli_mod, "build_and_probe_dataset", _raise_stage5_err)
    with pytest.raises(Stage5ResumeNoPlanSeedError, match="unexpected stage5 error outside stage5"):
        cli_mod.main(
            [
                "--count",
                "1",
                "--seed",
                "20261030",
                "--writer",
                "mock",
                "--out-dir",
                str(tmp_path / "non_stage5_unhandled"),
            ]
        )


def _make_fake_owned_server_stage5_harness(
    worlds: list[Any],
    *,
    owned_port: int = 11499,
    fail_on_phase3_rewrite_call: int | None = None,
) -> dict[str, Any]:
    """Build a fake `popen_factory` and `http_transport` for full Stage 5 runs (B, rung, rewrites) against an owned server."""
    from types import SimpleNamespace

    b_queue = [(w, r_idx, w.agents[r_idx]) for w in worlds for r_idx in (0, 1)]
    b_idx = 0
    rewrite_helper, rewrite_calls = _make_fake_rewrite_client(worlds)
    recorded_requests: list[tuple[str, str, str]] = []
    stop_events: list[int] = []
    phase3_rewrite_calls = 0

    def _fake_popen(cmd: list[str], **kwargs: Any) -> Any:
        _ = (cmd, kwargs)
        return SimpleNamespace(
            pid=43210,
            returncode=None,
            poll=lambda: None,
            terminate=lambda: None,
            wait=lambda timeout=None: 0,
            kill=lambda: None,
        )

    def _fake_pid_killer(pid: int) -> None:
        _ = pid
        stop_events.append(len(recorded_requests))

    def _fake_transport(
        method: str, url: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        nonlocal b_idx, phase3_rewrite_calls
        if url.endswith("/api/version"):
            recorded_requests.append((method, url, "version"))
            return {"version": "0.6.1"}
        if url.endswith("/api/tags"):
            recorded_requests.append((method, url, "tags"))
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
            recorded_requests.append((method, url, "show"))
            return {"parameters": "temperature 0.6"}
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                recorded_requests.append((method, url, "unload"))
                return {"done": True}
            assert payload is not None
            opts = payload.get("options") or {}
            # Rewrite/rung calls set num_predict=80, num_ctx=2048; B calls set num_predict=DEFAULT_WRITER_NUM_PREDICT
            if int(opts.get("num_predict", 0)) == 80 and int(opts.get("num_ctx", 0)) == 2048:
                expected_rung_calls = 2 * len(stage5_mod.STAGE5_RUNG_TARGETS) * stage5_mod.STAGE5_RUNG_PER_TARGET
                is_rung = len(rewrite_calls) < expected_rung_calls
                recorded_requests.append((method, url, "rung" if is_rung else "rewrite"))
                if not is_rung:
                    phase3_rewrite_calls += 1
                    if (
                        fail_on_phase3_rewrite_call is not None
                        and phase3_rewrite_calls == fail_on_phase3_rewrite_call
                    ):
                        raise RuntimeError(
                            f"Simulated mid-rewrite transport error on call {phase3_rewrite_calls}"
                        )
                return rewrite_helper(payload)
            recorded_requests.append((method, url, "b"))
            w, _, focal = b_queue[b_idx]
            b_idx += 1
            prose = _render_mock_stage5_baseline(w, focal)
            return {
                "message": {"role": "assistant", "content": prose},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3200,
                "eval_count": 600,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    return {
        "owned_port": owned_port,
        "owned_base_url": f"http://127.0.0.1:{owned_port}",
        "popen_factory": _fake_popen,
        "pid_killer": _fake_pid_killer,
        "http_transport": _fake_transport,
        "recorded_requests": recorded_requests,
        "stop_events": stop_events,
    }


def test_28_owned_server_single_url_closed_once_and_model_endpoint(tmp_path: Path) -> None:
    """Card 2026-11-03 Items 1, 2, 3: Full mock Stage 5 run with a fake owned server sends 0 requests outside the owned URL, closes the owned server once after the last call, and records matching `model_endpoint` across `b`, `rung`, `swap`, `rewrite`."""
    n = 40
    base_seed = 20261103
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(n, base_seed=base_seed)]
    harness = _make_fake_owned_server_stage5_harness(worlds, owned_port=11499)

    out_dir = tmp_path / "run_28_owned"
    rec = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=out_dir,
        writer_kind="ollama",
        stage5_plan_seed=20261103,
        stage5_rung_seed=20261104,
        server_port=11499,
        start_owned_server=True,
        popen_factory=harness["popen_factory"],
        pid_killer=harness["pid_killer"],
        http_transport=harness["http_transport"],
        prompt_token_counter=lambda _: 3200,
        worlds=worlds,
    )

    reqs = harness["recorded_requests"]
    owned_url = harness["owned_base_url"]
    outside_urls = [url for _, url, _ in reqs if not url.startswith(f"{owned_url}/")]
    assert outside_urls == [], f"Requests sent outside owned server URL {owned_url}: {outside_urls[:5]}"
    assert sum(1 for _, _, kind in reqs if kind == "b") == 80
    assert sum(1 for _, _, kind in reqs if kind == "rung") == 120
    assert sum(1 for _, _, kind in reqs if kind == "rewrite") > 0

    # Owned server closed exactly once, after the last model call
    assert harness["stop_events"] == [len(reqs)]

    # Item 3: model_endpoint has scheme, host, and port only for b, rung, swap, rewrite, all equal under owned server
    assert rec["model_endpoint"] == {
        "b": owned_url,
        "rung": owned_url,
        "swap": owned_url,
        "rewrite": owned_url,
    }
    assert len(set(rec["model_endpoint"].values())) == 1

    # Also verify external URL (no owned server): every call uses that same URL
    ext_harness = _make_fake_owned_server_stage5_harness(worlds, owned_port=11488)
    ext_url = ext_harness["owned_base_url"]
    rec_ext = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=tmp_path / "run_28_external",
        writer_kind="ollama",
        stage5_plan_seed=20261103,
        stage5_rung_seed=20261104,
        ollama_base_url=ext_url,
        start_owned_server=False,
        http_transport=ext_harness["http_transport"],
        prompt_token_counter=lambda _: 3200,
        worlds=worlds,
    )
    ext_outside = [url for _, url, _ in ext_harness["recorded_requests"] if not url.startswith(f"{ext_url}/")]
    assert ext_outside == []
    assert ext_harness["stop_events"] == []
    assert rec_ext["model_endpoint"] == {
        "b": ext_url,
        "rung": ext_url,
        "swap": ext_url,
        "rewrite": ext_url,
    }


def test_29_owned_server_closed_once_on_mid_rewrite_error(tmp_path: Path) -> None:
    """Card 2026-11-03 Item 2 (error path): An exception mid-rewrite still sends 0 requests outside the owned URL and closes the owned server exactly once."""
    n = 40
    base_seed = 20261103
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(n, base_seed=base_seed)]
    harness = _make_fake_owned_server_stage5_harness(
        worlds,
        owned_port=11499,
        fail_on_phase3_rewrite_call=3,
    )

    with pytest.raises(RuntimeError, match="Simulated mid-rewrite transport error on call 3"):
        run_stage5_pipeline(
            count=n,
            base_seed=base_seed,
            output_dir=tmp_path / "run_29_err",
            writer_kind="ollama",
            stage5_plan_seed=20261103,
            stage5_rung_seed=20261104,
            server_port=11499,
            start_owned_server=True,
            popen_factory=harness["popen_factory"],
            pid_killer=harness["pid_killer"],
            http_transport=harness["http_transport"],
            prompt_token_counter=lambda _: 3200,
            worlds=worlds,
        )

    reqs = harness["recorded_requests"]
    owned_url = harness["owned_base_url"]
    outside_urls = [url for _, url, _ in reqs if not url.startswith(f"{owned_url}/")]
    assert outside_urls == []
    assert sum(1 for _, _, kind in reqs if kind == "rewrite") == 3
    assert harness["stop_events"] == [len(reqs)]


def test_30_step_timings_in_record_and_mirrored_in_status_with_fake_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Card 2026-11-03 Item 4: `stage5_run_record.json` records `step_timings` for `b`, `rung`, `swap`, `rewrite` (`start`, `end`, `seconds`, call count), and `stage5_status.json` mirrors `step_timings` as the run progresses."""
    import datetime

    n = 40
    base_seed = 20261103
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(n, base_seed=base_seed)]
    client, _ = _make_fake_rewrite_client(worlds)

    utc_base = datetime.datetime(2026, 11, 3, 12, 0, 0, tzinfo=datetime.timezone.utc)
    utc_step = 0
    mono_values = iter([100.0, 112.5, 200.0, 208.25, 300.0, 300.5, 400.0, 415.75])

    def _fake_utc_clock() -> str:
        nonlocal utc_step
        dt = utc_base + datetime.timedelta(seconds=utc_step * 10)
        utc_step += 1
        return dt.isoformat().replace("+00:00", "Z")

    def _fake_mono_clock() -> float:
        return next(mono_values)

    status_snapshots: list[dict[str, Any]] = []
    orig_atomic_write = stage5_mod._atomic_write_json

    def _spy_atomic_write(target_path: Path, payload: Mapping[str, Any]) -> None:
        orig_atomic_write(target_path, payload)
        if target_path.name == STAGE5_STATUS_FILENAME:
            status_snapshots.append(json.loads(target_path.read_text(encoding="utf-8")))

    monkeypatch.setattr(stage5_mod, "_atomic_write_json", _spy_atomic_write)

    out_dir = tmp_path / "run_30_timers"
    rec = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=out_dir,
        writer_kind="mock",
        stage5_plan_seed=20261103,
        stage5_rung_seed=20261104,
        rewrite_client=client,
        worlds=worlds,
        utc_clock=_fake_utc_clock,
        monotonic_clock=_fake_mono_clock,
    )

    timings = rec["step_timings"]
    assert set(timings.keys()) == {"b", "rung", "swap", "rewrite"}
    assert timings["b"] == {
        "start": "2026-11-03T12:00:00Z",
        "end": "2026-11-03T12:00:10Z",
        "seconds": 12.5,
        "calls": 80,
        "call_count": 80,
    }
    assert timings["rung"] == {
        "start": "2026-11-03T12:00:20Z",
        "end": "2026-11-03T12:00:30Z",
        "seconds": 8.25,
        "calls": 120,
        "call_count": 120,
    }
    assert timings["swap"] == {
        "start": "2026-11-03T12:00:40Z",
        "end": "2026-11-03T12:00:50Z",
        "seconds": 0.5,
        "calls": 0,
        "call_count": 0,
    }
    expected_rw_calls = rec["call_counts"]["rewrite_calls"]
    assert timings["rewrite"] == {
        "start": "2026-11-03T12:01:00Z",
        "end": "2026-11-03T12:01:10Z",
        "seconds": 15.75,
        "calls": expected_rw_calls,
        "call_count": expected_rw_calls,
    }

    # Verify stage5_status.json mirrored step_timings as the run progressed
    assert any(
        s["step_timings"]["b"]["end"] is not None
        and s["step_timings"]["rung"]["end"] is None
        for s in status_snapshots
    )
    assert any(
        s["step_timings"]["rung"]["end"] is not None
        and s["step_timings"]["swap"]["end"] is None
        for s in status_snapshots
    )
    assert any(
        s["step_timings"]["swap"]["end"] is not None
        and s["step_timings"]["rewrite"]["end"] is None
        for s in status_snapshots
    )
    assert status_snapshots[-1]["step_timings"] == timings


def test_31_colab_launch_reconcile_default_empty_and_no_rnd_local_in_tracked_files() -> None:
    """Card 2026-11-03 Item 5: `scripts/colab/launch.sh:104` defaults `RECONCILE_SH` to empty, keeps the `-x` check, and no tracked file outside leak-guard tests contains the private workspace marker."""
    import subprocess

    launch_sh = REPO_ROOT / "scripts" / "colab" / "launch.sh"
    lines = launch_sh.read_text(encoding="utf-8").splitlines()
    assert lines[103].strip() == 'RECONCILE_SH="${COLAB_RECONCILE_SH:-}"'
    assert any('if [[ -x "$RECONCILE_SH" ]]; then' in ln for ln in lines)

    needle = "-".join(("rnd", "local"))
    tracked_out = subprocess.check_output(
        ["git", "ls-files"], cwd=REPO_ROOT, text=True, encoding="utf-8"
    )
    offending: list[str] = []
    for rel_path in tracked_out.splitlines():
        rel = rel_path.strip()
        if not rel or rel == "tests/test_generator.py":
            continue
        p = REPO_ROOT / rel
        if p.is_file():
            txt = p.read_text(encoding="utf-8", errors="replace")
            if needle in txt:
                offending.append(rel)
    assert offending == [], f"Tracked files outside leak-guard tests contain {needle!r}: {offending}"


def test_32_plan_statistics_per_stream_20_event_blocks_and_planted_imbalance() -> None:
    """Card 2026-11-03 Item 6: `_compute_plan_statistics` uses per-stream 20-event blocks matching `generate_stage5_shape_plan` (`<= 1` on `count=80`) and catches a planted block imbalance."""
    dry_res = run_stage5_dry(
        count=80,
        base_seed=20261030,
        stage5_plan_seed=20261103,
    )
    spread_by_variant = dry_res["block_balance_max_spread_by_variant"]
    assert isinstance(spread_by_variant, dict)
    assert all(v <= 1 for v in spread_by_variant.values()), f"Expected <= 1 per variant, got {spread_by_variant}"
    assert dry_res["block_balance_max_spread"] <= 1

    # Plant an imbalance in the first 20-event block of the `say` stream
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(20, base_seed=20261030)]
    plan_bundle = generate_stage5_shape_plan(worlds, seed=20261103, block_size=20)
    tampered_by_sid = {
        sid: [dict(r) for r in recs]
        for sid, recs in plan_bundle["plan_by_scenario_id"].items()
    }
    say_seen = 0
    for w in worlds:
        for r in tampered_by_sid[w.scenario_id]:
            if r["variant"] == "say" and say_seen < 20:
                r["form_id"] = "SAY_1"
                say_seen += 1
    tampered_bundle = dict(plan_bundle)
    tampered_bundle["plan_by_scenario_id"] = tampered_by_sid
    _, _, planted_spread = stage5_mod._compute_plan_statistics(
        worlds, tampered_bundle, block_size=20
    )
    assert planted_spread["say"] == 20


def test_33_card_2026_11_08_a_owned_server_every_phase_no_default_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Card 2026-11-08 Test (a): A fake transport records every URL called; with an owned server set (including via `popen_factory`/`server_port` and on `--resume` after B into rung and rewrite), no request goes to the default URL (`127.0.0.1:11434`) in any phase, and the owned server verifies and unloads on its own URL."""
    monkeypatch.setattr(stage5_mod, "STAGE5_RUNG_PER_TARGET", 1)
    n = 4
    base_seed = 20261103
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(n, base_seed=base_seed)]
    harness = _make_fake_owned_server_stage5_harness(worlds, owned_port=11499)

    out_dir = tmp_path / "run_33_owned"
    # 1. Owned server configured via popen_factory + server_port (without explicit start_owned_server=True)
    rec = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=out_dir,
        writer_kind="ollama",
        stage5_plan_seed=20261103,
        stage5_rung_seed=20261104,
        server_port=11499,
        popen_factory=harness["popen_factory"],
        pid_killer=harness["pid_killer"],
        http_transport=harness["http_transport"],
        prompt_token_counter=lambda _: 3200,
        worlds=worlds,
    )

    reqs = harness["recorded_requests"]
    owned_url = harness["owned_base_url"]
    default_urls = [url for _, url, _ in reqs if url.startswith("http://127.0.0.1:11434")]
    assert default_urls == [], f"Requests sent to default 11434 URL: {default_urls}"
    outside_urls = [url for _, url, _ in reqs if not url.startswith(f"{owned_url}/")]
    assert outside_urls == [], f"Requests sent outside owned URL {owned_url}: {outside_urls}"
    assert sum(1 for _, _, kind in reqs if kind == "b") == 8
    assert sum(1 for _, _, kind in reqs if kind == "rung") == 6
    assert sum(1 for _, _, kind in reqs if kind == "rewrite") == 18
    assert sum(1 for _, _, kind in reqs if kind == "unload") == 1
    assert harness["stop_events"] == [len(reqs)]
    assert rec["model_endpoint"] == {
        "b": owned_url,
        "rung": owned_url,
        "swap": owned_url,
        "rewrite": owned_url,
    }

    # 2. Resume after B is already done (only rung + swap + rewrite run): still verifies and unloads on owned URL, 0 requests to 11434
    resume_harness = _make_fake_owned_server_stage5_harness(worlds, owned_port=11498)
    resume_url = resume_harness["owned_base_url"]
    run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=out_dir,
        writer_kind="ollama",
        resume=True,
        redo_rung=True,
        server_port=11498,
        popen_factory=resume_harness["popen_factory"],
        pid_killer=resume_harness["pid_killer"],
        http_transport=resume_harness["http_transport"],
        prompt_token_counter=lambda _: 3200,
        worlds=worlds,
    )
    resume_reqs = resume_harness["recorded_requests"]
    assert [url for _, url, _ in resume_reqs if not url.startswith(f"{resume_url}/")] == []
    assert sum(1 for _, _, kind in resume_reqs if kind == "tags") == 1
    assert sum(1 for _, _, kind in resume_reqs if kind == "rung") == 6
    assert sum(1 for _, _, kind in resume_reqs if kind == "rewrite") == 18
    assert sum(1 for _, _, kind in resume_reqs if kind == "unload") == 1
    assert resume_harness["stop_events"] == [len(resume_reqs)]


def test_34_card_2026_11_08_b_partial_offload_guard_both_endpoints_and_11434_unload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Card 2026-11-08 Test (b): The partial-offload guard runs on both the owned endpoint and `127.0.0.1:11434`, halting when either reports `size_vram < size`, and at the end of a run any model left loaded on `127.0.0.1:11434` is unloaded (`keep_alive: 0`) and `/api/ps` empty is recorded."""
    monkeypatch.setattr(stage5_mod, "STAGE5_RUNG_PER_TARGET", 1)
    n = 4
    base_seed = 20261103
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(n, base_seed=base_seed)]

    def _make_dual_endpoint_harness(
        *,
        owned_vram: int = 23_000_000_000,
        default_models_initial: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        base_harness = _make_fake_owned_server_stage5_harness(worlds, owned_port=11499)
        inner_transport = base_harness["http_transport"]
        owned_base = base_harness["owned_base_url"]
        default_base = "http://127.0.0.1:11434"
        default_models: list[dict[str, Any]] = (
            [dict(m) for m in default_models_initial] if default_models_initial is not None else []
        )
        ps_checks: list[str] = []
        default_unloads: list[dict[str, Any]] = []

        def _transport(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
            if method == "GET" and url == f"{owned_base}/api/ps":
                ps_checks.append(url)
                return {
                    "models": [
                        {
                            "name": "qwen3.6:35b",
                            "model": "qwen3.6:35b",
                            "size": 23_000_000_000,
                            "size_vram": owned_vram,
                        }
                    ]
                }
            if method == "GET" and url == f"{default_base}/api/ps":
                ps_checks.append(url)
                return {"models": [dict(m) for m in default_models]}
            if (
                method == "POST"
                and url == f"{default_base}/api/chat"
                and payload is not None
                and payload.get("keep_alive") == 0
                and not payload.get("messages")
            ):
                default_unloads.append(dict(payload))
                unloaded_name = payload.get("model")
                default_models[:] = [
                    m
                    for m in default_models
                    if m.get("model") != unloaded_name and m.get("name") != unloaded_name
                ]
                return {"model": str(unloaded_name), "done": True}
            return inner_transport(method, url, payload)

        return {
            **base_harness,
            "http_transport": _transport,
            "ps_checks": ps_checks,
            "default_unloads": default_unloads,
            "default_models": default_models,
        }

    # 1. Partial offload on owned port (11499) halts with STAGE5_PARTIAL_OFFLOAD
    h_owned_partial = _make_dual_endpoint_harness(owned_vram=18_000_000_000)
    with pytest.raises(stage5_mod.Stage5PartialOffloadError) as exc_owned:
        run_stage5_pipeline(
            count=n,
            base_seed=base_seed,
            output_dir=tmp_path / "run_34_owned_partial",
            writer_kind="ollama",
            stage5_plan_seed=20261103,
            stage5_rung_seed=20261104,
            server_port=11499,
            start_owned_server=True,
            guard_both_endpoints=True,
            popen_factory=h_owned_partial["popen_factory"],
            pid_killer=h_owned_partial["pid_killer"],
            http_transport=h_owned_partial["http_transport"],
            prompt_token_counter=lambda _: 3200,
            worlds=worlds,
        )
    assert exc_owned.value.failure_class == "STAGE5_PARTIAL_OFFLOAD"
    assert exc_owned.value.endpoint == "http://127.0.0.1:11499"
    assert exc_owned.value.is_own_model is True
    assert "run's own model" in str(exc_owned.value)

    # 2a. Partial offload on default port (11434) for the run's own model halts with STAGE5_PARTIAL_OFFLOAD
    h_def_partial = _make_dual_endpoint_harness(
        owned_vram=23_000_000_000,
        default_models_initial=[
            {
                "name": "qwen3.6:35b",
                "model": "qwen3.6:35b",
                "size": 23_000_000_000,
                "size_vram": 16_000_000_000,
            }
        ],
    )
    with pytest.raises(stage5_mod.Stage5PartialOffloadError) as exc_def:
        run_stage5_pipeline(
            count=n,
            base_seed=base_seed,
            output_dir=tmp_path / "run_34_def_partial",
            writer_kind="ollama",
            stage5_plan_seed=20261103,
            stage5_rung_seed=20261104,
            server_port=11499,
            start_owned_server=True,
            guard_both_endpoints=True,
            popen_factory=h_def_partial["popen_factory"],
            pid_killer=h_def_partial["pid_killer"],
            http_transport=h_def_partial["http_transport"],
            prompt_token_counter=lambda _: 3200,
            worlds=worlds,
        )
    assert exc_def.value.failure_class == "STAGE5_PARTIAL_OFFLOAD"
    assert exc_def.value.endpoint == "http://127.0.0.1:11434"
    assert exc_def.value.is_own_model is True
    assert "run's own model" in str(exc_def.value)
    assert "http://127.0.0.1:11499/api/ps" in h_def_partial["ps_checks"]
    assert "http://127.0.0.1:11434/api/ps" in h_def_partial["ps_checks"]

    # 2b. Partial offload on default port (11434) for another seat's model halts and identifies it as another seat's model
    h_def_foreign_partial = _make_dual_endpoint_harness(
        owned_vram=23_000_000_000,
        default_models_initial=[
            {
                "name": "foreign-seat:70b",
                "model": "foreign-seat:70b",
                "size": 40_000_000_000,
                "size_vram": 20_000_000_000,
            }
        ],
    )
    with pytest.raises(stage5_mod.Stage5PartialOffloadError) as exc_def_foreign:
        run_stage5_pipeline(
            count=n,
            base_seed=base_seed,
            output_dir=tmp_path / "run_34_def_foreign_partial",
            writer_kind="ollama",
            stage5_plan_seed=20261103,
            stage5_rung_seed=20261104,
            server_port=11499,
            start_owned_server=True,
            guard_both_endpoints=True,
            popen_factory=h_def_foreign_partial["popen_factory"],
            pid_killer=h_def_foreign_partial["pid_killer"],
            http_transport=h_def_foreign_partial["http_transport"],
            prompt_token_counter=lambda _: 3200,
            worlds=worlds,
        )
    assert exc_def_foreign.value.failure_class == "STAGE5_PARTIAL_OFFLOAD"
    assert exc_def_foreign.value.endpoint == "http://127.0.0.1:11434"
    assert exc_def_foreign.value.model == "foreign-seat:70b"
    assert exc_def_foreign.value.is_own_model is False
    assert "another seat's model" in str(exc_def_foreign.value)

    # 3. Full offload on both endpoints, with the run's own model left loaded on 11434: unloaded (keep_alive: 0) at end and new record shape recorded
    h_cleanup = _make_dual_endpoint_harness(
        owned_vram=23_000_000_000,
        default_models_initial=[
            {
                "name": "qwen3.6:35b",
                "model": "qwen3.6:35b",
                "size": 23_000_000_000,
                "size_vram": 23_000_000_000,
            }
        ],
    )
    rec = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=tmp_path / "run_34_cleanup",
        writer_kind="ollama",
        stage5_plan_seed=20261103,
        stage5_rung_seed=20261104,
        server_port=11499,
        start_owned_server=True,
        guard_both_endpoints=True,
        popen_factory=h_cleanup["popen_factory"],
        pid_killer=h_cleanup["pid_killer"],
        http_transport=h_cleanup["http_transport"],
        prompt_token_counter=lambda _: 3200,
        worlds=worlds,
    )
    assert "http://127.0.0.1:11499/api/ps" in h_cleanup["ps_checks"]
    assert "http://127.0.0.1:11434/api/ps" in h_cleanup["ps_checks"]
    assert len(h_cleanup["default_unloads"]) == 1
    assert h_cleanup["default_unloads"][0]["keep_alive"] == 0
    assert h_cleanup["default_unloads"][0]["model"] == "qwen3.6:35b"
    assert h_cleanup["default_models"] == []
    assert rec["default_endpoint_end"] == {
        "endpoint": "http://127.0.0.1:11434",
        "unloaded_models": ["qwen3.6:35b"],
        "foreign_models": [],
        "ps_after": {"models": []},
        "own_models_absent": True,
        "foreign_models_present": [],
    }
    assert "ps_empty" not in rec["default_endpoint_end"]


def test_35_card_2026_11_08_c_step_timings_and_call_counts_match_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Card 2026-11-08 Test (c): The run record carries per-step seconds and call counts (`step_timings` and `call_counts` including `swap_calls`), matching `tests/fixtures/stage5_step_timings_fixture.json`, and preserves completed `b` and `rung` timings across `--resume`."""
    import datetime

    fixture_path = REPO_ROOT / "tests" / "fixtures" / "stage5_step_timings_fixture.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(stage5_mod, "STAGE5_RUNG_PER_TARGET", int(fixture["rung_per_target"]))
    n = int(fixture["count"])
    base_seed = int(fixture["base_seed"])
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(n, base_seed=base_seed)]
    client, _ = _make_fake_rewrite_client(worlds)

    utc_base = datetime.datetime(2026, 11, 8, 12, 0, 0, tzinfo=datetime.timezone.utc)
    utc_step = 0
    mono_values = iter([100.0, 112.5, 200.0, 208.25, 300.0, 300.5, 400.0, 415.75])

    def _fake_utc_clock() -> str:
        nonlocal utc_step
        dt = utc_base + datetime.timedelta(seconds=utc_step * 10)
        utc_step += 1
        return dt.isoformat().replace("+00:00", "Z")

    def _fake_mono_clock() -> float:
        return next(mono_values)

    out_dir = tmp_path / "run_35_fixture"
    rec = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=out_dir,
        writer_kind="mock",
        stage5_plan_seed=int(fixture["plan_seed"]),
        stage5_rung_seed=int(fixture["rung_seed"]),
        rewrite_client=client,
        worlds=worlds,
        utc_clock=_fake_utc_clock,
        monotonic_clock=_fake_mono_clock,
    )

    assert rec["step_timings"] == fixture["step_timings"]
    assert rec["call_counts"] == fixture["call_counts"]

    # Also verify that resuming when B and rung are already completed preserves their recorded step_timings
    resume_mono = iter([500.0, 500.0, 600.0, 600.0, 700.0, 700.25, 800.0, 801.0])
    rec_resumed = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=out_dir,
        writer_kind="mock",
        resume=True,
        rewrite_client=client,
        worlds=worlds,
        utc_clock=_fake_utc_clock,
        monotonic_clock=lambda: next(resume_mono),
    )
    assert rec_resumed["step_timings"] == fixture["step_timings"]
    assert rec_resumed["call_counts"] == fixture["call_counts"]


def test_36_card_2026_11_08_d_plan_statistics_per_20_events_fixture() -> None:
    """Card 2026-11-08 Test (d): Plan statistics follow the per-20-events definition (including the tail block and global spread) on `tests/fixtures/stage5_plan_statistics_fixture.json`."""
    fixture_path = REPO_ROOT / "tests" / "fixtures" / "stage5_plan_statistics_fixture.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    worlds = [
        generate_world(cfg)
        for cfg in make_balanced_configs(int(fixture["count"]), base_seed=int(fixture["base_seed"]))
    ]
    plan_bundle = generate_stage5_shape_plan(
        worlds,
        seed=int(fixture["plan_seed"]),
        block_size=int(fixture["block_size"]),
    )
    stats = stage5_mod._compute_plan_statistics(
        worlds,
        plan_bundle,
        block_size=int(fixture["block_size"]),
    )
    assert hasattr(stats, "to_dict")
    stats_dict = stats.to_dict()
    assert stats_dict["planned_form_counts_by_kind"] == fixture["planned_form_counts_by_kind"]
    assert stats_dict["planned_form_counts_by_variant"] == fixture["planned_form_counts_by_variant"]
    assert stats_dict["stream_lengths_by_variant"] == fixture["stream_lengths_by_variant"]
    assert stats_dict["stream_block_spreads_by_variant"] == fixture["stream_block_spreads_by_variant"]
    assert stats_dict["block_balance_max_spread_by_variant"] == fixture["block_balance_max_spread_by_variant"]
    assert stats_dict["block_balance_max_spread"] == fixture["block_balance_max_spread"]
    assert stats_dict["global_balance_max_spread_by_variant"] == fixture["global_balance_max_spread_by_variant"]
    assert stats_dict["global_balance_max_spread"] == fixture["global_balance_max_spread"]

    dry_res = run_stage5_dry(
        count=int(fixture["count"]),
        base_seed=int(fixture["base_seed"]),
        stage5_plan_seed=int(fixture["plan_seed"]),
        worlds=worlds,
    )
    for k in (
        "planned_form_counts_by_kind",
        "planned_form_counts_by_variant",
        "stream_lengths_by_variant",
        "stream_block_spreads_by_variant",
        "block_balance_max_spread_by_variant",
        "block_balance_max_spread",
        "global_balance_max_spread_by_variant",
        "global_balance_max_spread",
    ):
        assert dry_res[k] == fixture[k]


def test_37_card_2026_11_08_fix1_own_and_foreign_model_unloads_own_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Card 2026-11-08 PR #21 Fix 1 Test 1: When 11434 lists the run's own model and a foreign model, only the run's model gets `keep_alive: 0`; the foreign one gets no request at all, `foreign_models` names it, and `own_models_absent` is True."""
    monkeypatch.setattr(stage5_mod, "STAGE5_RUNG_PER_TARGET", 1)
    n = 4
    base_seed = 20261103
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(n, base_seed=base_seed)]

    base_harness = _make_fake_owned_server_stage5_harness(worlds, owned_port=11499)
    inner_transport = base_harness["http_transport"]
    owned_base = base_harness["owned_base_url"]
    default_base = "http://127.0.0.1:11434"
    default_models: list[dict[str, Any]] = [
        {
            "name": "qwen3.6:35b",
            "model": "qwen3.6:35b",
            "size": 23_000_000_000,
            "size_vram": 23_000_000_000,
        },
        {
            "name": "other-seat:14b",
            "model": "other-seat:14b",
            "size": 9_000_000_000,
            "size_vram": 9_000_000_000,
        },
    ]
    default_post_requests: list[dict[str, Any]] = []

    def _transport(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if method == "GET" and url == f"{owned_base}/api/ps":
            return {
                "models": [
                    {
                        "name": "qwen3.6:35b",
                        "model": "qwen3.6:35b",
                        "size": 23_000_000_000,
                        "size_vram": 23_000_000_000,
                    }
                ]
            }
        if method == "GET" and url == f"{default_base}/api/ps":
            return {"models": [dict(m) for m in default_models]}
        if method == "POST" and url.startswith(default_base):
            default_post_requests.append({"url": url, "payload": dict(payload or {})})
            if payload is not None and payload.get("keep_alive") == 0 and not payload.get("messages"):
                unloaded_name = payload.get("model")
                default_models[:] = [
                    m
                    for m in default_models
                    if m.get("model") != unloaded_name and m.get("name") != unloaded_name
                ]
                return {"model": str(unloaded_name), "done": True}
        return inner_transport(method, url, payload)

    rec = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=tmp_path / "run_37_own_and_foreign",
        writer_kind="ollama",
        stage5_plan_seed=20261103,
        stage5_rung_seed=20261104,
        server_port=11499,
        start_owned_server=True,
        guard_both_endpoints=True,
        popen_factory=base_harness["popen_factory"],
        pid_killer=base_harness["pid_killer"],
        http_transport=_transport,
        prompt_token_counter=lambda _: 3200,
        worlds=worlds,
    )

    assert len(default_post_requests) == 1
    assert default_post_requests[0]["url"] == "http://127.0.0.1:11434/api/chat"
    assert default_post_requests[0]["payload"]["model"] == "qwen3.6:35b"
    assert default_post_requests[0]["payload"]["keep_alive"] == 0
    assert all(
        req["payload"].get("model") != "other-seat:14b" for req in default_post_requests
    )
    assert [m["model"] for m in default_models] == ["other-seat:14b"]

    end_rec = rec["default_endpoint_end"]
    assert end_rec["endpoint"] == "http://127.0.0.1:11434"
    assert end_rec["unloaded_models"] == ["qwen3.6:35b"]
    assert end_rec["foreign_models"] == ["other-seat:14b"]
    assert end_rec["own_models_absent"] is True
    assert end_rec["foreign_models_present"] == ["other-seat:14b"]
    assert "ps_empty" not in end_rec


def test_38_card_2026_11_08_fix1_foreign_model_only_sends_no_unload_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Card 2026-11-08 PR #21 Fix 1 Test 2: When 11434 lists only a foreign model, no unload request is sent at all."""
    monkeypatch.setattr(stage5_mod, "STAGE5_RUNG_PER_TARGET", 1)
    n = 4
    base_seed = 20261103
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(n, base_seed=base_seed)]

    base_harness = _make_fake_owned_server_stage5_harness(worlds, owned_port=11499)
    inner_transport = base_harness["http_transport"]
    owned_base = base_harness["owned_base_url"]
    default_base = "http://127.0.0.1:11434"
    default_models: list[dict[str, Any]] = [
        {
            "name": "bookkeeper:8b",
            "model": "bookkeeper:8b",
            "size": 5_000_000_000,
            "size_vram": 5_000_000_000,
        }
    ]
    default_post_requests: list[dict[str, Any]] = []

    def _transport(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if method == "GET" and url == f"{owned_base}/api/ps":
            return {
                "models": [
                    {
                        "name": "qwen3.6:35b",
                        "model": "qwen3.6:35b",
                        "size": 23_000_000_000,
                        "size_vram": 23_000_000_000,
                    }
                ]
            }
        if method == "GET" and url == f"{default_base}/api/ps":
            return {"models": [dict(m) for m in default_models]}
        if method == "POST" and url.startswith(default_base):
            default_post_requests.append({"url": url, "payload": dict(payload or {})})
            return {"done": True}
        return inner_transport(method, url, payload)

    rec = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=tmp_path / "run_38_foreign_only",
        writer_kind="ollama",
        stage5_plan_seed=20261103,
        stage5_rung_seed=20261104,
        server_port=11499,
        start_owned_server=True,
        guard_both_endpoints=True,
        popen_factory=base_harness["popen_factory"],
        pid_killer=base_harness["pid_killer"],
        http_transport=_transport,
        prompt_token_counter=lambda _: 3200,
        worlds=worlds,
    )

    assert default_post_requests == []
    assert [m["model"] for m in default_models] == ["bookkeeper:8b"]

    end_rec = rec["default_endpoint_end"]
    assert end_rec["endpoint"] == "http://127.0.0.1:11434"
    assert end_rec["unloaded_models"] == []
    assert end_rec["foreign_models"] == ["bookkeeper:8b"]
    assert end_rec["own_models_absent"] is True
    assert end_rec["foreign_models_present"] == ["bookkeeper:8b"]
    assert "ps_empty" not in end_rec
