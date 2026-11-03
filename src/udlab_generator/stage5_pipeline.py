"""Stage 5 hybrid pipeline (`--stage5`), rewrite-input rung, and `stage5_run_record.json` writer.

Implements Card U4 (Design Revision 2, §4–§8):
1. Phase 1 (B, first draft): Stage 2/3 Prompt C (`constraint_layout="table"`, `accepted_forms_line="off"`,
   `shape_plan="off"`), with retries as Stage 3 and Profile D (`checker_profile="d"`) evaluation.
   Reuses the writer's existing `c_ctx_over_cap` refusal path (`CTX_OVER_CAP_CHECK_ID`).
2. Phase 2 (Rewrite-input rung): 60 eligible `say` events (20 × `SAY_2`, 20 × `SAY_3`, 20 × `SAY_4`) from
   renderings whose B passed and where `b_form != target`, drawn with `rung_seed`. Runs both `facts_only`
   and `with_draft` (120 calls) and selects `with_draft` iff `kept_rate(with_draft) - kept_rate(facts_only) >= 0.10`
   as exact fractions.
3. Phase 3 (Swap, Rewrite, Final Check, and S >= B Stop): runs `swap_all_move_see` followed by `rewrite_all_say`
   (or `swap_all_say` when `s_code_say=True`), evaluates Profile D on S, and stops immediately with
   `STAGE5_S_LT_B` (`stop_reason="S_LT_B"`) if any rendering with `b_passed=True` has `s_passed=False`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import secrets
from typing import Any, Callable, Mapping, Sequence

from .world import World, generate_world, make_balanced_configs
from .render import (
    ATTEMPT_0_TEMPERATURE,
    CTX_OVER_CAP_CHECK_ID,
    DEFAULT_GOAL_MODE,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_RETRY_FORM,
    DEFAULT_VRAM_HALT_MIB,
    DEFAULT_WRITER_MAX_NUM_CTX,
    DEFAULT_WRITER_MODEL_PIN,
    DEFAULT_WRITER_NUM_PREDICT,
    DEFAULT_WRITER_THINK,
    DEFAULT_WRITER_THINK_NUM_PREDICT,
    GENERATIONS_FILENAME,
    MaxNumCtxRefusalError,
    MockWriter,
    NUM_CTX_MARGIN_TOKENS,
    OllamaHttpTransport,
    OllamaWriter,
    PromptTokenCounter,
    SEALED_RESPONSES_FILENAME,
    SHAPE_PLAN_BLOCK_SIZE,
    SealedResponseEntry,
    VALID_GOAL_MODES,
    VALID_RETRY_FORMS,
    _load_seed_bank_seeds,
    _sentence_matches_folded_leave_enter,
    _split_prose_sentences,
    append_sealed_response,
    collect_prose_faults,
    compute_sealed_response_digests,
    count_prompt_tokens_exact,
    derive_call_seed,
    estimate_prompt_tokens,
    event_shape_plan_variant,
    format_writer_prompt,
    kill_pid_tree,
    load_writer_prompt_template,
    match_rendering_event_forms,
    normalize_prose_line_endings,
    prompt_fits_num_ctx_cap,
    render_world_template,
    render_world_template_goal_tokens,
    splice_goal_tokens,
    world_from_scenario_seed,
)
from .rewrite import (
    SAY_SHAPE_DESCRIPTIONS,
    RewriteResult,
    rewrite_all_say,
)
from .shape_templates import (
    MOVE_FORMS,
    SAY_FORMS,
    SEE_OCCLUDED_FORMS,
    SEE_UNOCCLUDED_FORMS,
    SwapEventRecord,
    SwapResult,
    generate_stage5_shape_plan,
     render_event_sentence,
    swap_all_move_see,
    swap_all_say,
)

STAGE5_RUN_RECORD_FILENAME: str = "stage5_run_record.json"
STAGE5_STATUS_FILENAME: str = "stage5_status.json"
STAGE5_SCHEMA_VERSION: str = "stage5_run_record_v1"
STAGE5_RUNG_RULE: str = "with_draft iff kept_rate(with_draft) - kept_rate(facts_only) >= 0.10"
STAGE5_RUNG_TARGETS: tuple[str, str, str] = ("SAY_2", "SAY_3", "SAY_4")
STAGE5_RUNG_PER_TARGET: int = 20

STAGE5_ASSIGNED_KINDS: frozenset[str] = frozenset({"move", "see", "say"})
STAGE5_ASSIGNED_VARIANTS: dict[str, tuple[str, ...]] = {
    "move_normal": MOVE_FORMS,
    "move_occluded": MOVE_FORMS,
    "see_unoccluded": SEE_UNOCCLUDED_FORMS,
    "see_occluded": SEE_OCCLUDED_FORMS,
    "say": SAY_FORMS,
}


class Stage5PipelineError(RuntimeError):
    """Base error for Stage 5 pipeline refusals and stops (carries `failure_class` with no prose)."""

    def __init__(self, failure_class: str, message: str) -> None:
        super().__init__(message)
        self.failure_class = failure_class


class Stage5SLtBStopError(Stage5PipelineError):
    """Raised when a rendering has `b_passed=True` and `s_passed=False` (`STAGE5_S_LT_B`)."""

    def __init__(
        self,
        *,
        scenario_id: str,
        rendering_index: int,
        b_fault_rule_ids: Sequence[str],
        s_fault_rule_ids: Sequence[str],
    ) -> None:
        self.scenario_id = scenario_id
        self.rendering_index = rendering_index
        self.b_fault_rule_ids = list(b_fault_rule_ids)
        self.s_fault_rule_ids = list(s_fault_rule_ids)
        msg = (
            f"STAGE5_S_LT_B: scenario_id={scenario_id} rendering_index={rendering_index} "
            f"b_fault_rule_ids={self.b_fault_rule_ids} s_fault_rule_ids={self.s_fault_rule_ids}"
        )
        super().__init__("STAGE5_S_LT_B", msg)


class Stage5ResumeNoPlanSeedError(Stage5PipelineError):
    """Raised when `--resume` is requested on a record lacking an integer `plan_seed` (`STAGE5_RESUME_NO_PLAN_SEED`)."""

    def __init__(self, message: str = "STAGE5_RESUME_NO_PLAN_SEED: resume record has no plan_seed.") -> None:
        super().__init__("STAGE5_RESUME_NO_PLAN_SEED", message)


class Stage5RungSampleShortError(Stage5PipelineError):
    """Raised when fewer than 20 eligible `say` events exist for any rung target (`STAGE5_RUNG_SAMPLE_SHORT`)."""

    def __init__(self, counts: Mapping[str, int]) -> None:
        self.counts = {str(k): int(v) for k, v in counts.items()}
        msg = (
            f"STAGE5_RUNG_SAMPLE_SHORT: fewer than {STAGE5_RUNG_PER_TARGET} eligible say events "
            f"from passing B renderings: {json.dumps(self.counts, sort_keys=True)}"
        )
        super().__init__("STAGE5_RUNG_SAMPLE_SHORT", msg)


def decide_rung_input_mode(
    facts_only_attempts: int,
    facts_only_kept: int,
    with_draft_attempts: int,
    with_draft_kept: int,
) -> str:
    """Apply the pre-registered rung rule using exact rational fractions (no float rounding).

    Returns `'with_draft'` iff `kept_rate(with_draft) - kept_rate(facts_only) >= 1/10`, else `'facts_only'`.
    """
    if facts_only_attempts <= 0 or with_draft_attempts <= 0:
        raise ValueError("Rung attempts must be > 0.")
    rate_fo = Fraction(int(facts_only_kept), int(facts_only_attempts))
    rate_wd = Fraction(int(with_draft_kept), int(with_draft_attempts))
    if rate_wd - rate_fo >= Fraction(1, 10):
        return "with_draft"
    return "facts_only"


def compute_event_witnessed_flags(world: World, focal_agent: str) -> list[bool]:
    """Compute per-event `witnessed` booleans for `focal_agent` in `world` using the room presence replay in `keys.py:165-219`.

    One definition for all event kinds: whether `focal_agent` was present in `ev.room` at that step.
    """
    occupants: dict[str, str] = dict(world.initial_agent_rooms)
    witnessed_by_pos: list[bool] = []
    for ev in world.events:
        if ev.kind == "enter" and ev.actor in world.agents:
            occupants[ev.actor] = ev.room
        witnessed_by_pos.append(occupants.get(focal_agent) == ev.room)
        if ev.kind == "leave":
            occupants.pop(ev.actor, None)
    return witnessed_by_pos


def compute_event_position_tercile_index(ev_pos: int, n_events: int) -> int:
    """Return 0, 1, or 2 using the exact Stage 4 formula (`render.py:6844`, 0-based `ev_pos`)."""
    return min(2, (ev_pos * 3) // max(1, n_events))


def compute_event_folded_flags(
    prose: str | None,
    world: World,
    s_rows: Sequence[Mapping[str, Any]] | None,
    *,
    goal_mode: str = DEFAULT_GOAL_MODE,
    checker_profile: str = "d",
) -> list[bool | None]:
    """Compute per-event `folded` flags for S: `bool` for `enter`/`leave` only (`_sentence_matches_folded_leave_enter`, `render.py:4452`), `None` for other kinds."""
    n_events = len(world.events)
    if prose is None or s_rows is None or len(s_rows) != n_events:
        return [False if ev.kind in ("enter", "leave") else None for ev in world.events]

    checked_prose = prose
    if goal_mode == "splice":
        try:
            checked_prose = splice_goal_tokens(prose, world).expanded_prose
        except Exception:
            pass
    norm_prose = normalize_prose_line_endings(checked_prose)
    paragraphs = [p.strip() for p in norm_prose.split("\n\n") if p.strip()]
    event_sentences: list[str] = []
    if len(paragraphs) >= 2:
        for ev_para in paragraphs[1:]:
            event_sentences.extend(_split_prose_sentences(ev_para))

    folded_flags: list[bool | None] = []
    for ev_pos, ev in enumerate(world.events):
        if ev.kind not in ("enter", "leave"):
            folded_flags.append(None)
            continue
        row = s_rows[ev_pos]
        s_idx_obj = row.get("sentence_index")
        if s_idx_obj is None:
            folded_flags.append(False)
            continue
        s_idx = int(s_idx_obj)
        if s_idx < 0 or s_idx >= len(event_sentences):
            folded_flags.append(False)
            continue
        is_folded = False
        if (
            ev.kind == "enter"
            and ev_pos > 0
            and s_rows[ev_pos - 1].get("sentence_index") == s_idx
            and _sentence_matches_folded_leave_enter(
                event_sentences[s_idx],
                world.events[ev_pos - 1],
                ev,
                world.noise_threshold,
                world=world,
                checker_profile=checker_profile,
            )
        ):
            is_folded = True
        elif (
            ev.kind == "leave"
            and ev_pos + 1 < n_events
            and s_rows[ev_pos + 1].get("sentence_index") == s_idx
            and _sentence_matches_folded_leave_enter(
                event_sentences[s_idx],
                ev,
                world.events[ev_pos + 1],
                world.noise_threshold,
                world=world,
                checker_profile=checker_profile,
            )
        ):
            is_folded = True
        folded_flags.append(is_folded)
    return folded_flags


def _render_mock_stage5_baseline(
    world: World,
    focal_agent: str,
    *,
    goal_mode: str = DEFAULT_GOAL_MODE,
) -> str:
    """Render Stage 5 mock B prose using `SAY_1` for `say` events (matching Prompt C baseline so `SAY_2`–`SAY_4` differ from B)."""
    base_prose = (
        render_world_template_goal_tokens(world, focal_agent)
        if goal_mode == "splice"
        else render_world_template(world, focal_agent)
    )
    say1_plan = {
        ev_idx: "SAY_1"
        for ev_idx, ev in enumerate(world.events, start=1)
        if ev.kind == "say"
    }
    if not say1_plan:
        return base_prose
    return swap_all_say(
        base_prose,
        world,
        say1_plan,
        focal_agent=focal_agent,
        goal_mode=goal_mode,
        checker_profile="d",
    ).prose


def _build_default_mock_rewrite_client(worlds: Sequence[World]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build a deterministic CPU-only rewrite client for `--writer mock` when no custom `rewrite_client` is supplied."""
    utterance_to_event: dict[tuple[str, str, str], tuple[World, Any]] = {}
    for w in worlds:
        for ev in w.events:
            if ev.kind == "say" and ev.utterance:
                utterance_to_event[(ev.actor, ev.room, ev.utterance)] = (w, ev)

    desc_to_form = {desc: form_id for form_id, desc in SAY_SHAPE_DESCRIPTIONS.items()}

    def _mock_client(payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages") or []
        content = str(messages[-1].get("content", "")) if messages else ""
        target_form = "SAY_2"
        for desc, form_id in desc_to_form.items():
            if f"Target sentence shape: {desc}" in content:
                target_form = form_id
                break

        for (actor, room, utterance), (w, ev) in utterance_to_event.items():
            if (
                f"- speaker: {actor}" in content
                and f"- room: {room}" in content
                and f'- utterance: "{utterance}"' in content
            ):
                reply_sent = render_event_sentence(ev, w, target_form, focal_agent=w.agents[0])
                return {
                    "message": {"role": "assistant", "content": reply_sent},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 64,
                    "eval_count": len(reply_sent.split()),
                }
        return {
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 64,
            "eval_count": 0,
        }

    return _mock_client


def _compute_plan_statistics(
    worlds: Sequence[World],
    plan_bundle: Mapping[str, Any],
    *,
    block_size: int = SHAPE_PLAN_BLOCK_SIZE,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]], dict[str, int]]:
    """Compute per-kind form counts, per-variant form counts, and block-balance max spread across `worlds`."""
    plan_by_sid: Mapping[str, Sequence[Mapping[str, Any]]] = plan_bundle["plan_by_scenario_id"]  # type: ignore[assignment]
    counts_by_kind: dict[str, dict[str, int]] = {
        "move": {f: 0 for f in (*MOVE_FORMS, "MOVE_0")},
        "see": {f: 0 for f in (*SEE_UNOCCLUDED_FORMS, *SEE_OCCLUDED_FORMS)},
        "say": {f: 0 for f in SAY_FORMS},
    }
    counts_by_variant: dict[str, dict[str, int]] = {
        v: {f: 0 for f in forms} for v, forms in STAGE5_ASSIGNED_VARIANTS.items()
    }
    counts_by_variant["move_station_change"] = {"MOVE_0": 0}

    for w in worlds:
        for rec in plan_by_sid.get(w.scenario_id, ()):
            kind = str(rec["kind"])
            variant = str(rec["variant"])
            form_id = str(rec["form_id"])
            if kind in counts_by_kind and form_id in counts_by_kind[kind]:
                counts_by_kind[kind][form_id] += 1
            if variant in counts_by_variant and form_id in counts_by_variant[variant]:
                counts_by_variant[variant][form_id] += 1

    max_spread_by_variant: dict[str, int] = {v: 0 for v in STAGE5_ASSIGNED_VARIANTS}
    for b_start in range(0, len(worlds), block_size):
        block_worlds = worlds[b_start : b_start + block_size]
        for v, forms in STAGE5_ASSIGNED_VARIANTS.items():
            b_counts = Counter()
            for w in block_worlds:
                for rec in plan_by_sid.get(w.scenario_id, ()):
                    if str(rec["variant"]) == v:
                        b_counts[str(rec["form_id"])] += 1
            if sum(b_counts.values()) > 0:
                vals = [b_counts[f] for f in forms]
                spread = max(vals) - min(vals)
                if spread > max_spread_by_variant[v]:
                    max_spread_by_variant[v] = spread

    return counts_by_kind, counts_by_variant, max_spread_by_variant


def _resolve_stage5_worlds_and_seed_bank(
    *,
    count: int,
    base_seed: int,
    seed_bank: Path | Sequence[int] | None,
    worlds: Sequence[World] | None,
) -> tuple[list[World], str | None, int | None]:
    """Resolve Stage 5 scenario worlds from `seed_bank` (via `world_from_scenario_seed`), `worlds`, or `(count, base_seed)`."""
    if seed_bank is not None:
        seeds = _load_seed_bank_seeds(seed_bank)
        resolved = [world_from_scenario_seed(scen_seed, index=idx) for idx, scen_seed in enumerate(seeds)]
        bank_size = len(seeds)
        if isinstance(seed_bank, Path):
            bank_sha256 = hashlib.sha256(seed_bank.read_bytes()).hexdigest()
        else:
            bank_sha256 = hashlib.sha256(json.dumps(list(seeds)).encode("utf-8")).hexdigest()
        return resolved, bank_sha256, bank_size
    if worlds is not None:
        return list(worlds), None, None
    return [generate_world(cfg) for cfg in make_balanced_configs(count, base_seed=base_seed)], None, None


def run_stage5_dry(
    *,
    count: int = 80,
    base_seed: int = 20261030,
    seed_bank: Path | Sequence[int] | None = None,
    stage5_plan_seed: int | None = None,
    max_retries: int = 0,
    worlds: Sequence[World] | None = None,
) -> dict[str, object]:
    """Execute `--stage5 --dry`: build worlds and the Stage 5 plan, compute counts, construct no writer, write no files."""
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}.")
    if stage5_plan_seed is not None:
        plan_seed = int(stage5_plan_seed)
        plan_seed_source = "flag"
    else:
        plan_seed = secrets.randbelow(2**63)
        plan_seed_source = "secrets"

    resolved_worlds, seed_bank_sha256, seed_bank_size = _resolve_stage5_worlds_and_seed_bank(
        count=count,
        base_seed=base_seed,
        seed_bank=seed_bank,
        worlds=worlds,
    )
    plan_bundle = generate_stage5_shape_plan(
        resolved_worlds,
        seed=plan_seed,
        block_size=SHAPE_PLAN_BLOCK_SIZE,
    )
    counts_by_kind, counts_by_variant, max_spread_by_variant = _compute_plan_statistics(
        resolved_worlds,
        plan_bundle,
        block_size=SHAPE_PLAN_BLOCK_SIZE,
    )
    num_scenarios = len(resolved_worlds)
    num_renderings = num_scenarios * 2
    say_events_per_rendering_pass = sum(
        2 for w in resolved_worlds for ev in w.events if ev.kind == "say"
    )
    expected_call_counts = {
        "b_calls": num_renderings * (1 + int(max_retries)),
        "rung_calls": 120,
        "rewrite_calls": say_events_per_rendering_pass,
    }
    max_spread_overall = max(max_spread_by_variant.values()) if max_spread_by_variant else 0
    return {
        "dry": True,
        "plan_seed": plan_seed,
        "plan_seed_source": plan_seed_source,
        "seed_bank_sha256": seed_bank_sha256,
        "seed_bank_size": seed_bank_size,
        "scenarios": num_scenarios,
        "renderings": num_renderings,
        "planned_form_counts_by_kind": counts_by_kind,
        "planned_form_counts_by_variant": counts_by_variant,
        "block_balance_max_spread_by_variant": max_spread_by_variant,
        "block_balance_max_spread": max_spread_overall,
        "expected_call_counts": expected_call_counts,
    }


def _scenario_stage5_plan_dict(scen_plan_records: Sequence[Mapping[str, Any]], world: World) -> dict[int, str | None]:
    """Return `{event_index: planned_form_or_None}` where `enter`/`leave`/other non-assigned events have `None`."""
    raw_by_idx: dict[int, str] = {
        int(r["event_index"]): str(r["form_id"]) for r in scen_plan_records
    }
    out: dict[int, str | None] = {}
    for ev_idx, ev in enumerate(world.events, start=1):
        if ev.kind in STAGE5_ASSIGNED_KINDS:
            out[ev_idx] = raw_by_idx.get(ev_idx)
        else:
            out[ev_idx] = None
    return out


def _build_initial_events_for_rendering(
    world: World,
    focal_agent: str,
    scen_plan_by_idx: Mapping[int, str | None],
    b_rows: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, object]]:
    """Build per-event record dicts after Phase 1 (B) before S has run."""
    n_events = len(world.events)
    witnessed_flags = compute_event_witnessed_flags(world, focal_agent)
    events_out: list[dict[str, object]] = []
    for ev_pos, ev in enumerate(world.events):
        ev_idx = ev_pos + 1
        b_form: str | None = None
        if b_rows is not None and ev_pos < len(b_rows):
            raw_bf = b_rows[ev_pos].get("realized_form_id")
            if raw_bf is not None:
                b_form = str(raw_bf)
        events_out.append(
            {
                "event_index": ev_idx,
                "kind": ev.kind,
                "variant": event_shape_plan_variant(ev, world),
                "position_tercile_index": compute_event_position_tercile_index(ev_pos, n_events),
                "witnessed": bool(witnessed_flags[ev_pos]),
                "plan_form": scen_plan_by_idx.get(ev_idx),
                "b_form": b_form,
                "s_form": None,
                "sentence_index": None,
                "action": "not_spliced",
                "rollback_cause": None,
                "rewrite_input": None,
                "seam_next_opens_ref": None,
                "folded": False if ev.kind in ("enter", "leave") else None,
            }
        )
    return events_out


def _assemble_final_events_for_rendering(
    world: World,
    focal_agent: str,
    scen_plan_by_idx: Mapping[int, str | None],
    b_rows: Sequence[Mapping[str, Any]],
    s_prose: str,
    swap_res: SwapResult,
    rewrite_res: RewriteResult | None,
    say_swap_res: SwapResult | None,
    chosen_input: str | None,
    *,
    goal_mode: str = DEFAULT_GOAL_MODE,
    checker_profile: str = "d",
) -> list[dict[str, object]]:
    """Assemble final per-event records for a completed S rendering."""
    n_events = len(world.events)
    witnessed_flags = compute_event_witnessed_flags(world, focal_agent)
    s_rows = match_rendering_event_forms(
        s_prose,
        world,
        focal_agent=focal_agent,
        goal_mode=goal_mode,
        checker_profile=checker_profile,
    )
    assert isinstance(s_rows, list)
    folded_flags = compute_event_folded_flags(
        s_prose,
        world,
        s_rows,
        goal_mode=goal_mode,
        checker_profile=checker_profile,
    )

    swap_by_idx: dict[int, SwapEventRecord] = {r.event_index: r for r in swap_res.events}
    rewrite_by_idx = {r.event_index: r for r in rewrite_res.events} if rewrite_res is not None else {}
    say_swap_by_idx = {r.event_index: r for r in say_swap_res.events} if say_swap_res is not None else {}

    events_out: list[dict[str, object]] = []
    for ev_pos, ev in enumerate(world.events):
        ev_idx = ev_pos + 1
        b_row = b_rows[ev_pos]
        s_row = s_rows[ev_pos]
        b_form = str(b_row["realized_form_id"]) if b_row.get("realized_form_id") is not None else None
        s_form = str(s_row["realized_form_id"]) if s_row.get("realized_form_id") is not None else None
        s_idx = int(s_row["sentence_index"]) if s_row.get("sentence_index") is not None else None  # type: ignore[arg-type]
        plan_form = scen_plan_by_idx.get(ev_idx)

        action = "not_spliced"
        rollback_cause: str | None = None
        rewrite_input_val: str | None = None
        seam_next_opens_ref: bool | None = None

        if ev.kind in ("move", "see"):
            sw_rec = swap_by_idx[ev_idx]
            action = sw_rec.action
            rollback_cause = sw_rec.cause
            seam_next_opens_ref = sw_rec.seam_next_opens_ref
        elif ev.kind == "say":
            if rewrite_res is not None:
                rw_rec = rewrite_by_idx[ev_idx]
                action = rw_rec.action
                rollback_cause = rw_rec.cause
                if rw_rec.action in ("rewritten", "rewrite_rolled_back"):
                    rewrite_input_val = chosen_input
            elif say_swap_res is not None:
                ss_rec = say_swap_by_idx[ev_idx]
                action = ss_rec.action
                rollback_cause = ss_rec.cause
                seam_next_opens_ref = ss_rec.seam_next_opens_ref
        else:
            action = "not_spliced"
            rollback_cause = None

        events_out.append(
            {
                "event_index": ev_idx,
                "kind": ev.kind,
                "variant": event_shape_plan_variant(ev, world),
                "position_tercile_index": compute_event_position_tercile_index(ev_pos, n_events),
                "witnessed": bool(witnessed_flags[ev_pos]),
                "plan_form": plan_form,
                "b_form": b_form,
                "s_form": s_form,
                "sentence_index": s_idx,
                "action": action,
                "rollback_cause": rollback_cause,
                "rewrite_input": rewrite_input_val,
                "seam_next_opens_ref": seam_next_opens_ref,
                "folded": folded_flags[ev_pos],
            }
        )
    return events_out


def _load_sealed_b_prose_by_rendering(
    sealed_responses_path: Path,
    *,
    goal_mode: str = DEFAULT_GOAL_MODE,
    worlds_by_sid: Mapping[str, World] | None = None,
) -> dict[tuple[str, int], str]:
    """Read the latest sealed B response for each `(scenario_id, rendering_index)` from `responses.sealed.jsonl`."""
    out: dict[tuple[str, int], str] = {}
    if not sealed_responses_path.is_file():
        return out
    for raw_line in sealed_responses_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        obj = json.loads(raw_line)
        if not isinstance(obj, dict):
            continue
        retry_form = str(obj.get("retry_form") or "")
        if retry_form in ("stage5_s", "stage5_rung", "stage5_rewrite"):
            continue
        sid = str(obj.get("scenario_id") or "")
        r_idx = int(obj.get("rendering_index", 0))
        resp_text = str(obj.get("response_text") or "")
        if not sid or not resp_text:
            continue
        if goal_mode == "splice" and worlds_by_sid and sid in worlds_by_sid:
            try:
                resp_text = splice_goal_tokens(resp_text, worlds_by_sid[sid]).expanded_prose
            except Exception:
                pass
        out[(sid, r_idx)] = resp_text
    return out


def _seal_rewrite_entries(
    sealed_responses_path: Path,
    *,
    scenario_id: str,
    rendering_index: int,
    focal_agent: str,
    phase_tag: str,
    sealed_entries: Sequence[Mapping[str, Any]],
    call_counter_start: int,
    goal_mode: str,
) -> int:
    """Append sealed prompt/reply entries from `RewriteResult.sealed` to `responses.sealed.jsonl`."""
    idx = call_counter_start
    for entry in sealed_entries:
        idx += 1
        ev_idx = int(entry.get("event_index", 0))
        reply_txt = str(entry.get("reply") or "")
        msgs = entry.get("messages") or []
        prompt_json = json.dumps(msgs, sort_keys=True, separators=(",", ":"))
        resp_sha, think_sha = compute_sealed_response_digests(reply_txt, prompt_json)
        append_sealed_response(
            sealed_responses_path,
            SealedResponseEntry(
                call_id=f"{scenario_id}:r{rendering_index}:{phase_tag}:ev{ev_idx}:call_{idx:04d}",
                call_index=idx,
                scenario_id=scenario_id,
                rendering_index=rendering_index,
                focal_agent=focal_agent,
                attempt=0,
                seed=0,
                temperature=0.0,
                min_p=None,
                retry_form=phase_tag,
                response_text=reply_txt,
                thinking_text=prompt_json,
                response_sha256=resp_sha,
                thinking_sha256=think_sha,
                goal_mode=goal_mode,
                presence_penalty=None,
                constraint_layout="table",
            ),
        )
    return idx


def _write_record_and_status(
    record_path: Path,
    status_path: Path,
    record: dict[str, Any],
    *,
    status_state: str,
    failure_class: str | None = None,
) -> None:
    """Write `stage5_run_record.json` and `stage5_status.json` atomically/cleanly with no prose."""
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status_doc = {
        "schema_version": STAGE5_SCHEMA_VERSION,
        "status": status_state,
        "stop_reason": record.get("stop_reason"),
        "failure_class": failure_class,
        "progress": record.get("progress"),
        "summary": record.get("summary"),
    }
    status_path.write_text(json.dumps(status_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _recompute_summary(renderings_list: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Compute the top-level `summary` counts from `renderings_list`."""
    total = len(renderings_list)
    b_pass = sum(1 for r in renderings_list if r.get("b_passed") is True)
    s_pass = sum(1 for r in renderings_list if r.get("s_passed") is True)
    s_pass_b_fail = sum(
        1 for r in renderings_list if r.get("s_passed") is True and not r.get("b_passed")
    )
    over_cap = sum(1 for r in renderings_list if r.get("ctx_over_cap") is True)
    return {
        "renderings": total,
        "b_passed": b_pass,
        "s_passed": s_pass,
        "s_pass_b_fail": s_pass_b_fail,
        "ctx_over_cap": over_cap,
    }


def run_stage5_pipeline(
    *,
    count: int = 80,
    base_seed: int = 20261030,
    seed_bank: Path | Sequence[int] | None = None,
    output_dir: Path,
    writer_kind: str = "mock",
    stage5_plan_seed: int | None = None,
    stage5_rung_seed: int | None = None,
    redo_b: bool = False,
    s_code_say: bool = False,
    vram_halt_mib: int = DEFAULT_VRAM_HALT_MIB,
    num_predict: int | None = None,
    max_num_ctx: int = DEFAULT_WRITER_MAX_NUM_CTX,
    think: bool = DEFAULT_WRITER_THINK,
    max_retries: int = 0,
    retry_form: str = DEFAULT_RETRY_FORM,
    multi_turn_retries: bool = False,
    goal_mode: str = DEFAULT_GOAL_MODE,
    presence_penalty: float | None = None,
    grant_id: str | None = None,
    resume: bool = False,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    server_port: int | None = None,
    start_owned_server: bool | None = None,
    http_transport: OllamaHttpTransport | None = None,
    rewrite_client: Any = None,
    vram_probe: Callable[[], int] | None = None,
    halt_pid: int | None = None,
    pid_killer: Callable[[int], None] = kill_pid_tree,
    popen_factory: Callable[..., Any] | None = None,
    prompt_token_counter: PromptTokenCounter | None = None,
    worlds: Sequence[World] | None = None,
    mock_custom_renderer: Callable[[World, str], str] | None = None,
    on_before_first_writer_call: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    """Run the full Stage 5 pipeline (B -> Rung -> Swap & Rewrite -> Final Check -> S >= B Stop) and write `stage5_run_record.json`."""
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
    if max_num_ctx > DEFAULT_WRITER_MAX_NUM_CTX:
        raise MaxNumCtxRefusalError(
            f"REFUSED (MAX_NUM_CTX_CAP): max_num_ctx={max_num_ctx} exceeds the hard VRAM ceiling limit "
            f"{DEFAULT_WRITER_MAX_NUM_CTX}."
        )

    resolved_goal_mode = str(goal_mode)
    resolved_think = bool(think)
    resolved_num_predict = (
        int(num_predict)
        if num_predict is not None
        else (DEFAULT_WRITER_THINK_NUM_PREDICT if resolved_think else DEFAULT_WRITER_NUM_PREDICT)
    )
    resolved_presence_penalty = float(presence_penalty) if presence_penalty is not None else None
    effective_writer_pin = replace(
        DEFAULT_WRITER_MODEL_PIN,
        num_predict=resolved_num_predict,
        max_num_ctx=int(max_num_ctx),
        think=resolved_think,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    sealed_dir = output_dir / "sealed_store"
    sealed_dir.mkdir(parents=True, exist_ok=True)
    generations_path = output_dir / GENERATIONS_FILENAME
    sealed_responses_path = sealed_dir / SEALED_RESPONSES_FILENAME
    record_path = output_dir / STAGE5_RUN_RECORD_FILENAME
    status_path = output_dir / STAGE5_STATUS_FILENAME

    existing_record: dict[str, Any] | None = None
    if resume:
        if not record_path.is_file():
            raise Stage5ResumeNoPlanSeedError(
                "STAGE5_RESUME_NO_PLAN_SEED: stage5_run_record.json does not exist for --resume."
            )
        loaded_obj = json.loads(record_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_obj, dict):
            raise Stage5ResumeNoPlanSeedError(
                "STAGE5_RESUME_NO_PLAN_SEED: stage5_run_record.json is not a JSON object."
            )
        raw_ps = loaded_obj.get("plan_seed")
        if isinstance(raw_ps, bool) or not isinstance(raw_ps, int):
            raise Stage5ResumeNoPlanSeedError(
                "STAGE5_RESUME_NO_PLAN_SEED: resume record has no valid integer plan_seed."
            )
        plan_seed = int(raw_ps)
        plan_seed_source = str(loaded_obj.get("plan_seed_source") or "secrets")
        raw_rs = loaded_obj.get("rung_seed")
        if not isinstance(raw_rs, bool) and isinstance(raw_rs, int):
            rung_seed = int(raw_rs)
            rung_seed_source = str(loaded_obj.get("rung_seed_source") or "secrets")
        elif stage5_rung_seed is not None:
            rung_seed = int(stage5_rung_seed)
            rung_seed_source = "flag"
        else:
            rung_seed = secrets.randbelow(2**63)
            rung_seed_source = "secrets"
        existing_record = loaded_obj
    else:
        if stage5_plan_seed is not None:
            plan_seed = int(stage5_plan_seed)
            plan_seed_source = "flag"
        else:
            plan_seed = secrets.randbelow(2**63)
            plan_seed_source = "secrets"

        if stage5_rung_seed is not None:
            rung_seed = int(stage5_rung_seed)
            rung_seed_source = "flag"
        else:
            rung_seed = secrets.randbelow(2**63)
            rung_seed_source = "secrets"

        if generations_path.exists():
            generations_path.unlink()
        if sealed_responses_path.exists():
            sealed_responses_path.unlink()

    resolved_worlds, seed_bank_sha256, seed_bank_size = _resolve_stage5_worlds_and_seed_bank(
        count=count,
        base_seed=base_seed,
        seed_bank=seed_bank,
        worlds=worlds,
    )
    if existing_record is not None and seed_bank_sha256 is None:
        prev_sb_sha = existing_record.get("seed_bank_sha256")
        prev_sb_sz = existing_record.get("seed_bank_size")
        if isinstance(prev_sb_sha, str):
            seed_bank_sha256 = prev_sb_sha
        if not isinstance(prev_sb_sz, bool) and isinstance(prev_sb_sz, int):
            seed_bank_size = int(prev_sb_sz)
    worlds_by_sid: dict[str, World] = {w.scenario_id: w for w in resolved_worlds}
    plan_bundle = generate_stage5_shape_plan(
        resolved_worlds,
        seed=plan_seed,
        block_size=SHAPE_PLAN_BLOCK_SIZE,
    )
    plan_by_sid: Mapping[str, Sequence[Mapping[str, Any]]] = plan_bundle["plan_by_scenario_id"]  # type: ignore[assignment]

    total_renderings = len(resolved_worlds) * 2
    redo_b_flag = bool(redo_b)

    # Restore state from existing_record if resuming without --stage5-redo-b
    existing_renderings_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    b_prose_by_key: dict[tuple[str, int], str] = {}
    existing_rung: dict[str, Any] | None = None
    existing_rung_done: bool = False
    b_calls_made = 0
    rung_calls_made = 0
    rewrite_calls_made = 0

    if existing_record is not None:
        prev_calls = existing_record.get("call_counts") or {}
        if isinstance(prev_calls, dict):
            b_calls_made = int(prev_calls.get("b_calls", 0))
            rung_calls_made = int(prev_calls.get("rung_calls", 0))
            rewrite_calls_made = int(prev_calls.get("rewrite_calls", 0))
        if redo_b_flag:
            # Explicit --stage5-redo-b resets every downstream field (rung and S) of affected renderings.
            existing_rung = None
            existing_rung_done = False
            rung_calls_made = 0
            rewrite_calls_made = 0
        else:
            b_prose_by_key = _load_sealed_b_prose_by_rendering(
                sealed_responses_path,
                goal_mode=resolved_goal_mode,
                worlds_by_sid=worlds_by_sid,
            )
            prev_prog = existing_record.get("progress") or {}
            prev_b_done = int(prev_prog.get("b_done", len(existing_record.get("renderings") or ())))
            prev_s_done = int(prev_prog.get("s_done", 0))
            for idx_r, r_entry in enumerate(existing_record.get("renderings") or ()):
                if idx_r >= prev_b_done:
                    break
                if isinstance(r_entry, dict):
                    k = (str(r_entry["scenario_id"]), int(r_entry["rendering_index"]))
                    cp = dict(r_entry)
                    cp["_s_completed"] = bool(
                        idx_r < prev_s_done
                        and not (cp.get("b_passed") and not cp.get("s_passed"))
                    )
                    existing_renderings_by_key[k] = cp
            prev_rung = existing_record.get("rung")
            if (
                isinstance(prev_rung, dict)
                and prev_rung.get("chosen") in ("facts_only", "with_draft")
                and isinstance(prev_prog, dict)
                and bool(prev_prog.get("rung_done"))
            ):
                existing_rung = dict(prev_rung)
                existing_rung_done = True

    engine_block: dict[str, object] = {
        "writer_kind": writer_kind,
        "ollama_version": None,
        "model": effective_writer_pin.ollama_tag,
        "model_digest": effective_writer_pin.manifest_digest if writer_kind == "ollama" else None,
    }

    default_rung_block: dict[str, Any] = (
        existing_rung
        if existing_rung is not None
        else {
            "facts_only": {"attempts": 0, "kept": 0},
            "with_draft": {"attempts": 0, "kept": 0},
            "chosen": "facts_only",
            "rule": STAGE5_RUNG_RULE,
        }
    )

    renderings_records: list[dict[str, Any]] = []
    record: dict[str, Any] = {
        "schema_version": STAGE5_SCHEMA_VERSION,
        "plan_seed": plan_seed,
        "plan_seed_source": plan_seed_source,
        "rung_seed": rung_seed,
        "rung_seed_source": rung_seed_source,
        "seed_bank_sha256": seed_bank_sha256,
        "seed_bank_size": seed_bank_size,
        "rung": default_rung_block,
        "stop_reason": None,
        "progress": {
            "renderings_planned": total_renderings,
            "b_done": len(existing_renderings_by_key),
            "rung_done": existing_rung_done,
            "s_done": sum(
                1
                for r in existing_renderings_by_key.values()
                if r.get("_s_completed") is True
            ),
            "calls_made": {
                "b_calls": b_calls_made,
                "rung_calls": rung_calls_made,
                "rewrite_calls": rewrite_calls_made,
            },
        },
        "redo_b": redo_b_flag,
        "engine": engine_block,
        "max_num_ctx": int(max_num_ctx),
        "num_predict": resolved_num_predict,
        "call_counts": {
            "b_calls": b_calls_made,
            "rung_calls": rung_calls_made,
            "rewrite_calls": rewrite_calls_made,
        },
        "summary": {
            "renderings": 0,
            "b_passed": 0,
            "s_passed": 0,
            "s_pass_b_fail": 0,
            "ctx_over_cap": 0,
        },
        "renderings": renderings_records,
    }

    # Write `plan_seed` and `rung_seed` to `stage5_run_record.json` BEFORE any writer call.
    _write_record_and_status(record_path, status_path, record, status_state="running")
    if on_before_first_writer_call is not None:
        on_before_first_writer_call(record_path)

    # -----------------------------------------------------------------------
    # Phase 1: B (First Draft)
    # Stage 2/3 Prompt C: constraint_layout="table", accepted_forms_line="off", shape_plan="off", checker_profile="d"
    # -----------------------------------------------------------------------
    prompt_template_c = load_writer_prompt_template(
        goal_mode=resolved_goal_mode,
        constraint_layout="table",
        accepted_forms_line="off",
        shape_plan="off",
    )

    ollama_writer: OllamaWriter | None = None
    mock_writer: MockWriter | None = None
    active_rewrite_client: Any = rewrite_client

    need_any_b = len(existing_renderings_by_key) < total_renderings
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
            short_circuit_dead_scenarios=False,
            goal_mode=resolved_goal_mode,
            constraint_layout="table",
            accepted_forms_line="off",
            checker_profile="d",
            shape_plan="off",
            shape_plan_seed=None,
            presence_penalty=resolved_presence_penalty,
            start_owned_server=should_own_server,
            popen_factory=popen_factory,
            prompt_token_counter=prompt_token_counter,
        )
        if active_rewrite_client is None:
            active_rewrite_client = ollama_writer._transport
    else:
        renderer_fn = (
            mock_custom_renderer
            if mock_custom_renderer is not None
            else lambda w, fa: _render_mock_stage5_baseline(w, fa, goal_mode=resolved_goal_mode)
        )
        if need_any_b:
            mock_writer = MockWriter(
                custom_renderer=renderer_fn,
                goal_mode=resolved_goal_mode,
                constraint_layout="table",
                checker_profile="d",
                accepted_forms_line="off",
                shape_plan="off",
                shape_plan_seed=None,
            )
        if active_rewrite_client is None:
            active_rewrite_client = _build_default_mock_rewrite_client(resolved_worlds)

    sealed_call_counter = 0
    if sealed_responses_path.is_file():
        sealed_call_counter = sum(
            1 for ln in sealed_responses_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        )

    b_rows_by_key: dict[tuple[str, int], list[dict[str, object]]] = {}
    scen_plan_dict_by_sid: dict[str, dict[int, str | None]] = {
        w.scenario_id: _scenario_stage5_plan_dict(plan_by_sid.get(w.scenario_id, ()), w)
        for w in resolved_worlds
    }

    s_completed_keys: set[tuple[str, int]] = {
        k for k, v in existing_renderings_by_key.items() if v.get("_s_completed") is True
    }
    for v in existing_renderings_by_key.values():
        v.pop("_s_completed", None)

    try:
        for w in resolved_worlds:
            sid = w.scenario_id
            scen_plan_by_idx = scen_plan_dict_by_sid[sid]
            for r_idx in (0, 1):
                r_key = (sid, r_idx)
                focal_ag = w.agents[r_idx]
                if r_key in existing_renderings_by_key:
                    prev_r = existing_renderings_by_key[r_key]
                    renderings_records.append(prev_r)
                    if not prev_r.get("ctx_over_cap") and r_key in b_prose_by_key:
                        b_rows_restored = match_rendering_event_forms(
                            b_prose_by_key[r_key],
                            w,
                            focal_agent=focal_ag,
                            goal_mode=resolved_goal_mode,
                            checker_profile="d",
                        )
                        assert isinstance(b_rows_restored, list)
                        b_rows_by_key[r_key] = b_rows_restored
                    continue

                prompt_text = format_writer_prompt(
                    w,
                    focal_ag,
                    prompt_template_c,
                    goal_mode=resolved_goal_mode,
                    constraint_layout="table",
                    accepted_forms_line="off",
                    shape_plan="off",
                )
                messages = [{"role": "user", "content": prompt_text}]

                if ollama_writer is not None:
                    prev_call_len = len(ollama_writer.call_records)
                    prev_sealed_len = len(ollama_writer.sealed_responses)
                    prose_out = ollama_writer._chat_render(
                        world=w,
                        scenario_id=sid,
                        rendering_index=r_idx,
                        focal_agent=focal_ag,
                        prompt_text=prompt_text,
                        allow_refusal=True,
                    )
                    new_call_recs = ollama_writer.call_records[prev_call_len:]
                    last_rec = new_call_recs[-1]
                    decision_tokens = (
                        last_rec.prompt_tokens_exact
                        if last_rec.prompt_tokens_exact is not None
                        else last_rec.prompt_tokens_estimate
                    )
                    is_over_cap = last_rec.check_id == CTX_OVER_CAP_CHECK_ID
                    if not is_over_cap:
                        b_calls_made += len(new_call_recs)
                        sealed_call_counter += len(ollama_writer.sealed_responses) - prev_sealed_len

                    if is_over_cap:
                        cap_block = {
                            "prompt_tokens": int(decision_tokens),
                            "num_predict": resolved_num_predict,
                            "margin": NUM_CTX_MARGIN_TOKENS,
                            "max_num_ctx": int(max_num_ctx),
                            "fits": False,
                        }
                        ev_init = _build_initial_events_for_rendering(
                            w, focal_ag, scen_plan_by_idx, None
                        )
                        renderings_records.append(
                            {
                                "scenario_id": sid,
                                "scenario_seed": int(w.seed),
                                "rendering_index": r_idx,
                                "focal_agent_sha256": hashlib.sha256(focal_ag.encode("utf-8")).hexdigest(),
                                "ctx_over_cap": True,
                                "prompt_tokens": int(decision_tokens),
                                "cap": cap_block,
                                "b_passed": False,
                                "b_fault_rule_ids": [CTX_OVER_CAP_CHECK_ID],
                                "s_passed": False,
                                "s_fault_rule_ids": [CTX_OVER_CAP_CHECK_ID],
                                "seam_count": 0,
                                "events": ev_init,
                            }
                        )
                        continue

                    if prose_out is not None:
                        b_prose = prose_out
                    else:
                        # All attempts failed check_rendered_prose; recover the final attempt's sealed prose
                        new_sealed = ollama_writer.sealed_responses[prev_sealed_len:]
                        raw_last = new_sealed[-1].response_text if new_sealed else ""
                        if resolved_goal_mode == "splice" and raw_last:
                            try:
                                b_prose = splice_goal_tokens(raw_last, w).expanded_prose
                            except Exception:
                                b_prose = raw_last
                        else:
                            b_prose = raw_last
                    decision_tokens_out = int(decision_tokens)
                else:
                    assert mock_writer is not None
                    p_tokens: int | None = (
                        count_prompt_tokens_exact(
                            messages, think=resolved_think, counter=prompt_token_counter
                        )
                        if prompt_token_counter is not None
                        else None
                    )
                    fits_cap = (
                        prompt_fits_num_ctx_cap(
                            p_tokens,
                            resolved_num_predict,
                            max_num_ctx,
                            margin=NUM_CTX_MARGIN_TOKENS,
                        )
                        if p_tokens is not None
                        else True
                    )
                    if not fits_cap:
                        assert p_tokens is not None
                        cap_block = {
                            "prompt_tokens": int(p_tokens),
                            "num_predict": resolved_num_predict,
                            "margin": NUM_CTX_MARGIN_TOKENS,
                            "max_num_ctx": int(max_num_ctx),
                            "fits": False,
                        }
                        ev_init = _build_initial_events_for_rendering(
                            w, focal_ag, scen_plan_by_idx, None
                        )
                        renderings_records.append(
                            {
                                "scenario_id": sid,
                                "scenario_seed": int(w.seed),
                                "rendering_index": r_idx,
                                "focal_agent_sha256": hashlib.sha256(focal_ag.encode("utf-8")).hexdigest(),
                                "ctx_over_cap": True,
                                "prompt_tokens": int(p_tokens),
                                "cap": cap_block,
                                "b_passed": False,
                                "b_fault_rule_ids": [CTX_OVER_CAP_CHECK_ID],
                                "s_passed": False,
                                "s_fault_rule_ids": [CTX_OVER_CAP_CHECK_ID],
                                "seam_count": 0,
                                "events": ev_init,
                            }
                        )
                        continue

                    decision_tokens_out: int | None = int(p_tokens) if p_tokens is not None else None
                    raw_b = mock_writer.render_raw_response(w, focal_ag)
                    b_calls_made += 1
                    sealed_call_counter += 1
                    call_seed = derive_call_seed(int(w.seed), r_idx, 0)
                    resp_sha, think_sha = compute_sealed_response_digests(raw_b, "")
                    append_sealed_response(
                        sealed_responses_path,
                        SealedResponseEntry(
                            call_id=f"{sid}:r{r_idx}:b_call_{sealed_call_counter:04d}",
                            call_index=sealed_call_counter,
                            scenario_id=sid,
                            rendering_index=r_idx,
                            focal_agent=focal_ag,
                            attempt=0,
                            seed=call_seed,
                            temperature=ATTEMPT_0_TEMPERATURE,
                            min_p=None,
                            retry_form="",
                            response_text=raw_b,
                            thinking_text="",
                            response_sha256=resp_sha,
                            thinking_sha256=think_sha,
                            goal_mode=resolved_goal_mode,
                            presence_penalty=resolved_presence_penalty,
                            constraint_layout="table",
                        ),
                    )
                    if resolved_goal_mode == "splice":
                        try:
                            b_prose = splice_goal_tokens(raw_b, w).expanded_prose
                        except Exception:
                            b_prose = raw_b
                    else:
                        b_prose = raw_b

                b_prose_by_key[r_key] = b_prose
                b_faults = collect_prose_faults(
                    b_prose,
                    w,
                    focal_agent=focal_ag,
                    goal_mode=resolved_goal_mode,
                    checker_profile="d",
                )
                b_passed = len(b_faults) == 0
                b_fault_rule_ids = [f.rule_id for f in b_faults]
                b_rows = match_rendering_event_forms(
                    b_prose,
                    w,
                    focal_agent=focal_ag,
                    goal_mode=resolved_goal_mode,
                    checker_profile="d",
                )
                assert isinstance(b_rows, list)
                b_rows_by_key[r_key] = b_rows

                cap_block = {
                    "prompt_tokens": decision_tokens_out,
                    "num_predict": resolved_num_predict,
                    "margin": NUM_CTX_MARGIN_TOKENS,
                    "max_num_ctx": int(max_num_ctx),
                    "fits": True,
                }
                ev_init = _build_initial_events_for_rendering(
                    w, focal_ag, scen_plan_by_idx, b_rows
                )
                renderings_records.append(
                    {
                        "scenario_id": sid,
                        "scenario_seed": int(w.seed),
                        "rendering_index": r_idx,
                        "focal_agent_sha256": hashlib.sha256(focal_ag.encode("utf-8")).hexdigest(),
                        "ctx_over_cap": False,
                        "prompt_tokens": decision_tokens_out,
                        "cap": cap_block,
                        "b_passed": b_passed,
                        "b_fault_rule_ids": b_fault_rule_ids,
                        "s_passed": False,
                        "s_fault_rule_ids": [],
                        "seam_count": 0,
                        "events": ev_init,
                    }
                )
    finally:
        if ollama_writer is not None:
            if ollama_writer.owned_server is not None:
                engine_block["ollama_version"] = ollama_writer.owned_server.ollama_version
            ollama_writer.close()

    # Update progress after Phase 1 (B done for all renderings)
    record["call_counts"]["b_calls"] = b_calls_made
    record["progress"]["b_done"] = len(renderings_records)
    record["progress"]["calls_made"]["b_calls"] = b_calls_made
    record["summary"] = _recompute_summary(renderings_records)
    _write_record_and_status(record_path, status_path, record, status_state="running")

    # -----------------------------------------------------------------------
    # Phase 2: The Rewrite-Input Rung
    # -----------------------------------------------------------------------
    if existing_rung_done and existing_rung is not None:
        chosen_input = str(existing_rung["chosen"])
        record["rung"] = existing_rung
    elif s_code_say:
        chosen_input = "facts_only"
        record["rung"] = {
            "facts_only": {"attempts": 0, "kept": 0},
            "with_draft": {"attempts": 0, "kept": 0},
            "chosen": "facts_only",
            "rule": STAGE5_RUNG_RULE,
        }
        record["progress"]["rung_done"] = True
        _write_record_and_status(record_path, status_path, record, status_state="running")
    else:
        eligible_by_target: dict[str, list[tuple[str, int, int, str]]] = {
            t: [] for t in STAGE5_RUNG_TARGETS
        }
        for r_entry in renderings_records:
            if not r_entry.get("b_passed") or r_entry.get("ctx_over_cap"):
                continue
            sid = str(r_entry["scenario_id"])
            r_idx = int(r_entry["rendering_index"])
            r_key = (sid, r_idx)
            w = worlds_by_sid[sid]
            b_rows = b_rows_by_key[r_key]
            scen_plan_by_idx = scen_plan_dict_by_sid[sid]
            for ev_pos, ev in enumerate(w.events):
                if ev.kind != "say":
                    continue
                ev_idx = ev_pos + 1
                target_form = scen_plan_by_idx.get(ev_idx)
                if target_form not in STAGE5_RUNG_TARGETS:
                    continue
                b_row = b_rows[ev_pos]
                if b_row.get("sentence_index") is None:
                    continue
                b_form = (
                    str(b_row["realized_form_id"])
                    if b_row.get("realized_form_id") is not None
                    else None
                )
                if b_form != target_form:
                    eligible_by_target[target_form].append((sid, r_idx, ev_idx, target_form))

        counts_by_target = {t: len(eligible_by_target[t]) for t in STAGE5_RUNG_TARGETS}
        if any(counts_by_target[t] < STAGE5_RUNG_PER_TARGET for t in STAGE5_RUNG_TARGETS):
            _write_record_and_status(
                record_path,
                status_path,
                record,
                status_state="stopped",
                failure_class="STAGE5_RUNG_SAMPLE_SHORT",
            )
            raise Stage5RungSampleShortError(counts_by_target)

        rung_rng = random.Random(rung_seed)
        sampled_rung_events: list[tuple[str, int, int, str]] = []
        for target_form in STAGE5_RUNG_TARGETS:
            sampled_rung_events.extend(
                rung_rng.sample(eligible_by_target[target_form], STAGE5_RUNG_PER_TARGET)
            )

        rung_stats: dict[str, dict[str, int]] = {
            "facts_only": {"attempts": 0, "kept": 0},
            "with_draft": {"attempts": 0, "kept": 0},
        }
        for mode in ("facts_only", "with_draft"):
            for sid, r_idx, ev_idx, target_form in sampled_rung_events:
                w = worlds_by_sid[sid]
                focal_ag = w.agents[r_idx]
                b_prose = b_prose_by_key[(sid, r_idx)]
                single_event_plan = {ev_idx: target_form}
                rung_res = rewrite_all_say(
                    b_prose,
                    w,
                    single_event_plan,
                    active_rewrite_client,
                    input_mode=mode,
                    focal_agent=focal_ag,
                    goal_mode=resolved_goal_mode,
                    checker_profile="d",
                    base_url=ollama_base_url,
                    model=effective_writer_pin.ollama_tag,
                )
                rung_calls_made += len(rung_res.sealed)
                sealed_call_counter = _seal_rewrite_entries(
                    sealed_responses_path,
                    scenario_id=sid,
                    rendering_index=r_idx,
                    focal_agent=focal_ag,
                    phase_tag="stage5_rung",
                    sealed_entries=rung_res.sealed,
                    call_counter_start=sealed_call_counter,
                    goal_mode=resolved_goal_mode,
                )
                ev_rec = rung_res.events[ev_idx - 1]
                rung_stats[mode]["attempts"] += 1
                if ev_rec.action == "rewritten":
                    rung_stats[mode]["kept"] += 1

        chosen_input = decide_rung_input_mode(
            rung_stats["facts_only"]["attempts"],
            rung_stats["facts_only"]["kept"],
            rung_stats["with_draft"]["attempts"],
            rung_stats["with_draft"]["kept"],
        )
        record["rung"] = {
            "facts_only": rung_stats["facts_only"],
            "with_draft": rung_stats["with_draft"],
            "chosen": chosen_input,
            "rule": STAGE5_RUNG_RULE,
        }
        record["call_counts"]["rung_calls"] = rung_calls_made
        record["progress"]["rung_done"] = True
        record["progress"]["calls_made"]["rung_calls"] = rung_calls_made
        _write_record_and_status(record_path, status_path, record, status_state="running")

    # -----------------------------------------------------------------------
    # Phase 3: Swap, Rewrite, Final Check, and S >= B Stop
    # -----------------------------------------------------------------------
    s_done_count = 0
    for r_entry in renderings_records:
        sid = str(r_entry["scenario_id"])
        r_idx = int(r_entry["rendering_index"])
        r_key = (sid, r_idx)
        w = worlds_by_sid[sid]
        focal_ag = w.agents[r_idx]
        scen_plan_by_idx = scen_plan_dict_by_sid[sid]

        if r_entry.get("ctx_over_cap"):
            s_done_count += 1
            record["progress"]["s_done"] = s_done_count
            continue

        if resume and not redo_b_flag and r_key in s_completed_keys:
            s_done_count += 1
            record["progress"]["s_done"] = s_done_count
            continue

        b_prose = b_prose_by_key[r_key]
        b_rows = b_rows_by_key[r_key]

        # 3. The swap pass (U2's swap_all_move_see) on B
        swap_res = swap_all_move_see(
            b_prose,
            w,
            scen_plan_by_idx,
            focal_agent=focal_ag,
            goal_mode=resolved_goal_mode,
            checker_profile="d",
        )

        # 4. The rewrite pass (U3's rewrite_all_say, or swap_all_say for S-code fallback)
        rewrite_res: RewriteResult | None = None
        say_swap_res: SwapResult | None = None
        if s_code_say:
            say_swap_res = swap_all_say(
                swap_res.prose,
                w,
                scen_plan_by_idx,
                focal_agent=focal_ag,
                goal_mode=resolved_goal_mode,
                checker_profile="d",
            )
            s_prose = say_swap_res.prose
            seam_count = swap_res.seam_count + say_swap_res.seam_count
        else:
            rewrite_res = rewrite_all_say(
                swap_res.prose,
                w,
                scen_plan_by_idx,
                active_rewrite_client,
                input_mode=chosen_input,
                focal_agent=focal_ag,
                goal_mode=resolved_goal_mode,
                checker_profile="d",
                base_url=ollama_base_url,
                model=effective_writer_pin.ollama_tag,
            )
            s_prose = rewrite_res.prose
            seam_count = swap_res.seam_count
            rewrite_calls_made += len(rewrite_res.sealed)
            sealed_call_counter = _seal_rewrite_entries(
                sealed_responses_path,
                scenario_id=sid,
                rendering_index=r_idx,
                focal_agent=focal_ag,
                phase_tag="stage5_rewrite",
                sealed_entries=rewrite_res.sealed,
                call_counter_start=sealed_call_counter,
                goal_mode=resolved_goal_mode,
            )

        # Seal S prose per rendering beside B in the sealed store
        sealed_call_counter += 1
        s_sha, s_think_sha = compute_sealed_response_digests(s_prose, "")
        append_sealed_response(
            sealed_responses_path,
            SealedResponseEntry(
                call_id=f"{sid}:r{r_idx}:s_final_{sealed_call_counter:04d}",
                call_index=sealed_call_counter,
                scenario_id=sid,
                rendering_index=r_idx,
                focal_agent=focal_ag,
                attempt=0,
                seed=derive_call_seed(int(w.seed), r_idx, 0),
                temperature=0.0,
                min_p=None,
                retry_form="stage5_s",
                response_text=s_prose,
                thinking_text="",
                response_sha256=s_sha,
                thinking_sha256=s_think_sha,
                goal_mode=resolved_goal_mode,
                presence_penalty=resolved_presence_penalty,
                constraint_layout="table",
            ),
        )

        # 5. The final check: profile d on S
        s_faults = collect_prose_faults(
            s_prose,
            w,
            focal_agent=focal_ag,
            goal_mode=resolved_goal_mode,
            checker_profile="d",
        )
        s_passed = len(s_faults) == 0
        s_fault_rule_ids = [f.rule_id for f in s_faults]

        final_events = _assemble_final_events_for_rendering(
            w,
            focal_ag,
            scen_plan_by_idx,
            b_rows,
            s_prose,
            swap_res,
            rewrite_res,
            say_swap_res,
            None if s_code_say else chosen_input,
            goal_mode=resolved_goal_mode,
            checker_profile="d",
        )

        r_entry["s_passed"] = s_passed
        r_entry["s_fault_rule_ids"] = s_fault_rule_ids
        r_entry["seam_count"] = seam_count
        r_entry["events"] = final_events

        s_done_count += 1
        record["call_counts"]["rewrite_calls"] = rewrite_calls_made
        record["progress"]["s_done"] = s_done_count
        record["progress"]["calls_made"]["rewrite_calls"] = rewrite_calls_made
        record["summary"] = _recompute_summary(renderings_records)

        # 6. The S >= B stop
        if r_entry["b_passed"] and not s_passed:
            record["stop_reason"] = "S_LT_B"
            record["stop_detail"] = {
                "failure_class": "STAGE5_S_LT_B",
                "scenario_id": sid,
                "rendering_index": r_idx,
                "b_fault_rule_ids": list(r_entry["b_fault_rule_ids"]),
                "s_fault_rule_ids": s_fault_rule_ids,
            }
            _write_record_and_status(
                record_path,
                status_path,
                record,
                status_state="stopped",
                failure_class="STAGE5_S_LT_B",
            )
            raise Stage5SLtBStopError(
                scenario_id=sid,
                rendering_index=r_idx,
                b_fault_rule_ids=r_entry["b_fault_rule_ids"],
                s_fault_rule_ids=s_fault_rule_ids,
            )

    record["stop_reason"] = None
    record.pop("stop_detail", None)
    record["summary"] = _recompute_summary(renderings_records)
    _write_record_and_status(record_path, status_path, record, status_state="completed")
    return record
