"""Tests for the Stage 5 hybrid pipeline (`--stage5`) and `stage5_run_record.json` (Card U4)."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
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
from udlab_generator.shape_templates import render_event_sentence
from udlab_generator.stage5_pipeline import (
    STAGE5_RUN_RECORD_FILENAME,
    STAGE5_RUNG_RULE,
    STAGE5_SCHEMA_VERSION,
    Stage5ResumeNoPlanSeedError,
    Stage5RungSampleShortError,
    Stage5SLtBStopError,
    _render_mock_stage5_baseline,
    decide_rung_input_mode,
    run_stage5_dry,
    run_stage5_pipeline,
)
import udlab_generator.stage5_pipeline as stage5_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
MOCK_RECORD_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "stage5" / "stage5_run_record_mock.json"
MOCK_SEED_BANK_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "stage5" / "stage5_seed_bank_mock.json"

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
    fail_on_rendering: tuple[str, int] | None = None,
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

    rung = rec["rung"]
    assert isinstance(rung, dict)
    for k in ("facts_only", "with_draft"):
        assert isinstance(rung[k], dict)
        assert isinstance(rung[k]["attempts"], int) and not isinstance(rung[k]["attempts"], bool)
        assert isinstance(rung[k]["kept"], int) and not isinstance(rung[k]["kept"], bool)
    assert rung["chosen"] in ("facts_only", "with_draft")
    assert isinstance(rung["rule"], str) and rung["rule"] == STAGE5_RUNG_RULE

    assert rec["stop_reason"] is None or rec["stop_reason"] == "S_LT_B"

    prog = rec["progress"]
    assert isinstance(prog, dict)
    assert isinstance(prog["renderings_planned"], int)
    assert isinstance(prog["b_done"], int)
    assert isinstance(prog["rung_done"], bool)
    assert isinstance(prog["s_done"], int)
    calls_made = prog["calls_made"]
    assert isinstance(calls_made, dict)
    for ck in ("b_calls", "rung_calls", "rewrite_calls"):
        assert isinstance(calls_made[ck], int) and not isinstance(calls_made[ck], bool)

    assert isinstance(rec["redo_b"], bool)

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
    for ck in ("b_calls", "rung_calls", "rewrite_calls"):
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
    """Test 1: Full mock run on a synthetic bank (n=20 >= 12) with a fake rewrite client validates schema, summary, and fixture parity."""
    n = 20
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
    assert len(rs) == n * 2 == 40
    assert rec["summary"] == {
        "renderings": len(rs),
        "b_passed": sum(1 for r in rs if r["b_passed"]),
        "s_passed": sum(1 for r in rs if r["s_passed"]),
        "s_pass_b_fail": sum(1 for r in rs if r["s_passed"] and not r["b_passed"]),
        "ctx_over_cap": sum(1 for r in rs if r["ctx_over_cap"]),
    }
    assert rec["summary"]["b_passed"] == 40
    assert rec["summary"]["s_passed"] == 40
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
    assert rec["call_counts"]["b_calls"] == 40
    assert rec["call_counts"]["rung_calls"] == 120
    assert len(calls) == 120 + rec["call_counts"]["rewrite_calls"]

    written_path = out_dir / STAGE5_RUN_RECORD_FILENAME
    assert written_path.is_file()
    assert MOCK_RECORD_FIXTURE_PATH.is_file()
    assert written_path.read_text(encoding="utf-8") == MOCK_RECORD_FIXTURE_PATH.read_text(encoding="utf-8")


def test_2_s_ge_b_stop_and_resume(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test 2: S >= B stop (`STAGE5_S_LT_B`) on a rendering where B passed and S failed, plus resume and `--stage5-redo-b`."""
    n = 20
    base_seed = 20261103
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(n, base_seed=base_seed)]
    good_client, _ = _make_fake_rewrite_client(worlds)

    # Inject a faulty swap on rendering index 2 (scen_0001, r0) so B passes and S fails
    orig_swap = stage5_mod.swap_all_move_see
    inject_bad_swap = True

    def _sabotaged_swap(prose: str, world: Any, plan: Any, **kwargs: Any) -> Any:
        res = orig_swap(prose, world, plan, **kwargs)
        if inject_bad_swap and world.scenario_id == "scen_0001" and kwargs.get("focal_agent") == world.agents[0]:
            # Inject a banned adverb into S prose after swap so final profile-d check on S fails while B passed
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
    b_calls_by_sid: Counter[str] = Counter()

    def _counting_b_renderer(w: Any, fa: str) -> str:
        b_calls_by_sid[w.scenario_id] += 1
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
        assert exc_info.value.scenario_id == "scen_0001"
        assert exc_info.value.rendering_index == 0

        # Also verify CLI returns non-zero (1) and prints STAGE5_S_LT_B to stderr
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
    assert stopped_rec["stop_detail"]["scenario_id"] == "scen_0001"
    assert stopped_rec["stop_detail"]["rendering_index"] == 0
    assert stopped_rec["stop_detail"]["b_fault_rule_ids"] == []
    assert "c_fidelity_unlogged_event" in stopped_rec["stop_detail"]["s_fault_rule_ids"]

    # B completed for all 40 renderings, S completed up to and including stopping rendering #3 (index 2 -> s_done == 3)
    assert stopped_rec["progress"]["renderings_planned"] == 40
    assert stopped_rec["progress"]["b_done"] == 40
    assert stopped_rec["progress"]["rung_done"] is True
    assert stopped_rec["progress"]["s_done"] == 3
    assert stopped_rec["progress"]["calls_made"]["b_calls"] == 40
    assert stopped_rec["progress"]["calls_made"]["rung_calls"] == 120
    # Renderings after index 2 were not processed for S
    for later_r in stopped_rec["renderings"][3:]:
        assert later_r["s_passed"] is False
        assert all(ev["s_form"] is None for ev in later_r["events"])

    # Now test RESUME after stop: zero B calls for renderings with B in record, zero rung calls, byte-identical B fields
    inject_bad_swap = False
    b_calls_by_sid.clear()
    resume_client, resume_rw_calls = _make_fake_rewrite_client(worlds)
    resumed_rec = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=out_dir,
        writer_kind="mock",
        resume=True,
        rewrite_client=resume_client,
        worlds=worlds,
        mock_custom_renderer=_counting_b_renderer,
    )
    assert sum(b_calls_by_sid.values()) == 0
    assert resumed_rec["stop_reason"] is None
    assert resumed_rec["redo_b"] is False
    assert resumed_rec["progress"]["s_done"] == 40
    assert resumed_rec["summary"]["s_passed"] == 40
    # Zero rung calls on resume (only rewrite calls for renderings 2..39)
    assert resumed_rec["call_counts"]["rung_calls"] == 120
    assert len(resume_rw_calls) < 90  # renderings 0 and 1 were skipped

    # Verify byte-identical B fields before and after resume
    for r_before, r_after in zip(stopped_rec["renderings"], resumed_rec["renderings"]):
        assert r_before["b_passed"] == r_after["b_passed"]
        assert r_before["b_fault_rule_ids"] == r_after["b_fault_rule_ids"]
        assert [e["b_form"] for e in r_before["events"]] == [e["b_form"] for e in r_after["events"]]

    # Now test --stage5-redo-b on resume: B calls happen for all scenarios and redo_b=True is recorded
    b_calls_by_sid.clear()
    redo_client, _ = _make_fake_rewrite_client(worlds)
    redo_rec = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=out_dir,
        writer_kind="mock",
        resume=True,
        redo_b=True,
        rewrite_client=redo_client,
        worlds=worlds,
        mock_custom_renderer=_counting_b_renderer,
    )
    assert redo_rec["redo_b"] is True
    assert sum(b_calls_by_sid.values()) == 40
    assert all(c == 2 for c in b_calls_by_sid.values())


def test_3_s_passes_where_b_failed(tmp_path: Path) -> None:
    """Test 3: A rendering where B failed and S passed is allowed, does not stop the run, and is counted in `s_pass_b_fail`."""
    n = 20
    base_seed = 20261103
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(n, base_seed=base_seed)]
    client, _ = _make_fake_rewrite_client(worlds)

    def _b_fails_first_move_renderer(w: Any, fa: str) -> str:
        clean_b = _render_mock_stage5_baseline(w, fa)
        if w.scenario_id == "scen_0000" and fa == w.agents[0]:
            # Inject a banned adverb into a move sentence in B so B fails `a_banned_adverbs`,
            # and `swap_all_move_see` replaces that move sentence so S passes!
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
    assert rec["summary"]["renderings"] == 40
    assert rec["summary"]["b_passed"] == 39
    assert rec["summary"]["s_passed"] == 40
    assert rec["summary"]["s_pass_b_fail"] == 1
    r0 = rec["renderings"][0]
    assert r0["b_passed"] is False
    assert "b_banned_adverbs" in r0["b_fault_rule_ids"]
    assert r0["s_passed"] is True
    assert r0["s_fault_rule_ids"] == []


def test_4_context_cap_and_4a_cap_decision_per_rendering(tmp_path: Path) -> None:
    """Test 4 & 4a: Reuses the writer's existing `c_ctx_over_cap` refusal path; never drops over-cap renderings; populates `cap`."""
    n = 21  # 21 scenarios = 42 renderings so even with 1 over-cap rendering, >= 20 eligible per SAY target remain
    base_seed = 20261103
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(n, base_seed=base_seed)]
    worlds_by_sid = {w.scenario_id: w for w in worlds}
    client, rw_calls = _make_fake_rewrite_client(worlds)

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
        # Force only the first rendering (scen_0000, rendering 0) over the 4,864 prompt token limit (5000 + 4096 + 256 = 9352 > 9216)
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

    # 1. Rendering count equals input count (never dropped)
    assert len(rec["renderings"]) == n * 2 == 42
    assert rec["summary"]["renderings"] == 42
    assert rec["summary"]["ctx_over_cap"] == 1

    # 2. Over-cap rendering took the existing refusal path (no model call, c_ctx_over_cap in both fault lists, no swap/rewrite)
    assert len(chat_calls_made) == 41
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

    # 3. Test 4a: Every rendering's cap block is filled with exact fields
    for r in rec["renderings"]:
        cap = r["cap"]
        assert cap["fits"] == (not r["ctx_over_cap"])
        assert cap["prompt_tokens"] == r["prompt_tokens"]
        assert cap["num_predict"] == DEFAULT_WRITER_NUM_PREDICT
        assert cap["margin"] == NUM_CTX_MARGIN_TOKENS == 256
        assert cap["max_num_ctx"] == DEFAULT_WRITER_MAX_NUM_CTX == 9216

    # 4. Raising --max-num-ctx above DEFAULT_WRITER_MAX_NUM_CTX is refused with MaxNumCtxRefusalError
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
    """Test 5: Plan seed is drawn once via `secrets.randbelow`, written before the first writer call, and never redrawn on resume or with flag."""
    n = 20
    base_seed = 20261103
    worlds = [generate_world(cfg) for cfg in make_balanced_configs(n, base_seed=base_seed)]
    client, _ = _make_fake_rewrite_client(worlds)

    randbelow_calls: list[int] = []
    real_randbelow = secrets.randbelow

    def _spy_randbelow(upper: int) -> int:
        randbelow_calls.append(upper)
        return 20261103 + len(randbelow_calls) - 1

    monkeypatch.setattr(secrets, "randbelow", _spy_randbelow)

    # 1. Fresh run with rung_seed given by flag -> secrets.randbelow called EXACTLY ONCE (for plan_seed)
    out_fresh = tmp_path / "run_5_fresh"
    verified_before_writer = False

    def _assert_record_written_before_b( world: Any, focal: str) -> str:
        nonlocal verified_before_writer
        if not verified_before_writer:
            rec_file = out_fresh / STAGE5_RUN_RECORD_FILENAME
            assert rec_file.is_file(), "stage5_run_record.json must exist before the first writer call"
            on_disk = json.loads(rec_file.read_text(encoding="utf-8"))
            assert on_disk["plan_seed"] == 20261103
            assert on_disk["plan_seed_source"] == "secrets"
            verified_before_writer = True
        return _render_mock_stage5_baseline(world, focal)

    rec_fresh = run_stage5_pipeline(
        count=n,
        base_seed=base_seed,
        output_dir=out_fresh,
        writer_kind="mock",
        stage5_rung_seed=20261104,
        rewrite_client=client,
        worlds=worlds,
        mock_custom_renderer=_assert_record_written_before_b,
    )
    assert verified_before_writer is True
    assert len(randbelow_calls) == 1
    assert rec_fresh["plan_seed"] == 20261103
    assert rec_fresh["plan_seed_source"] == "secrets"
    assert rec_fresh["rung_seed"] == 20261104
    assert rec_fresh["rung_seed_source"] == "flag"

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
    # 1. Exact rational boundary tests (+0.10 -> with_draft; +0.0999 -> facts_only)
    assert decide_rung_input_mode(60, 36, 60, 42) == "with_draft"  # 42/60 - 36/60 = 6/60 = +0.10 exact
    assert decide_rung_input_mode(10000, 5000, 10000, 6000) == "with_draft"  # +0.1000
    assert decide_rung_input_mode(10000, 5000, 10000, 5999) == "facts_only"  # +0.0999
    assert decide_rung_input_mode(60, 37, 60, 42) == "facts_only"  # 5/60 = +0.0833...

    # 2. Verify 20/20/20 sample from passing B only, and both inputs called on the exact same 60 events
    n = 20
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
    # Both inputs run on the exact same 60 events in the same order
    assert [(c["scenario_id"], c["target_form"]) for c in fo_calls] == [
        (c["scenario_id"], c["target_form"]) for c in wd_rung_calls
    ]
    # And the main rewrite pass uses the chosen input ("with_draft")
    main_calls = wd_calls[120:]
    assert len(main_calls) > 0
    assert all(c["input_mode"] == "with_draft" for c in main_calls)

    # 3. Short-sample refusal: n=12 has only 18 events per target shape (< 20) -> refuses with STAGE5_RUNG_SAMPLE_SHORT
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
    assert exc_short.value.counts == {"SAY_2": 18, "SAY_3": 18, "SAY_4": 18}


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
    n = 20
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

    # stdout is a single JSON summary line of counts
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
    assert '"tests/test_stage5_templates.py",  # U3-fix1 authorized test-file edit' in rewrite_test_text


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
    sum_def_a = build_and_probe_dataset(
        count=2,
        base_seed=20261030,
        private_seed_hex="ab" * 32,
        output_dir=out_def_a,
        writer_kind="mock",
    )
    sum_def_b = build_and_probe_dataset(
        count=2,
        base_seed=20261030,
        private_seed_hex="ab" * 32,
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
        private_seed_hex="cd" * 32,
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
        private_seed_hex="cd" * 32,
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
    assert isinstance(raw_seeds, list) and len(raw_seeds) == 20
    expected_sha256 = hashlib.sha256(MOCK_SEED_BANK_FIXTURE_PATH.read_bytes()).hexdigest()
    expected_size = len(raw_seeds)

    worlds = [world_from_scenario_seed(int(s), index=i) for i, s in enumerate(raw_seeds)]
    client, _ = _make_fake_rewrite_client(worlds)

    # 1. Run `run_paired_seed_bank` on the exact same seed-bank file (with a fake CPU transport)
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
    assert isinstance(paired_renderings, list) and len(paired_renderings) == 40

    # 2. Run `--stage5` with `--paired-seed-bank` (and mock writer) on the same seed-bank file
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
    assert len(rec["renderings"]) == len(paired_renderings) == 40

    # Read sealed B entries from `sealed_store/responses.sealed.jsonl` to verify focal_agent per rendering
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

    # 3. Run `--stage5 --dry` with `--stage5-seed-bank` and assert sha256, size, and renderings count
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
    assert dry_doc["seed_bank_size"] == expected_size == 20
    assert dry_doc["scenarios"] == 20
    assert dry_doc["renderings"] == 40

