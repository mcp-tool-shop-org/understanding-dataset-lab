"""Stage 5 hybrid pipeline (`--stage5`), rewrite-input rung, and `stage5_run_record.json` writer.

Implements Card U4 (Design Revision 2, §4–§8) + Addenda 1 & 2:
1. Phase 1 (B, first draft): Stage 2/3 Prompt C (`constraint_layout="table"`, `accepted_forms_line="off"`,
   `shape_plan="off"`), with retries as Stage 3 and Profile D (`checker_profile="d"`) evaluation.
   Reuses the writer's existing `c_ctx_over_cap` refusal path (`CTX_OVER_CAP_CHECK_ID`).
2. Phase 2 (Rewrite-input rung): 60 eligible `say` events (20 × `SAY_2`, 20 × `SAY_3`, 20 × `SAY_4`), at most 1
   event per rendering, from canonically sorted renderings where `b_passed and not ctx_over_cap` and `b_form != target`,
   drawn with `rung_seed`. Applies `swap_all_move_see` in memory to each sampled rendering's B before running both
   `facts_only` and `with_draft` (120 calls), and selects `with_draft` iff `kept_with_draft - kept_facts_only >= 6`
   of 60 (exact integer/rational arithmetic, never float division).
3. Phase 3 (Swap, Rewrite, Final Check, and S >= B Stop): runs `swap_all_move_see` followed by `rewrite_all_say`
   (or `swap_all_say` when `s_code_say=True`), evaluates Profile D on S, and stops immediately with
   `STAGE5_S_LT_B` (`stop_reason="S_LT_B"`) if any rendering with `b_passed=True` has `s_passed=False`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import datetime
from fractions import Fraction
import hashlib
import inspect
from itertools import combinations
import json
import os
from pathlib import Path
import random
import secrets
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

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
    PINNED_WRITER_MODEL_TAG,
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
    _invoke_rewrite_client,
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
STAGE5_RUNG_RULE: str = (
    "with_draft iff kept_with_draft - kept_facts_only >= 6 of 60 "
    "(exact equivalent of kept_rate(with_draft) - kept_rate(facts_only) >= 0.10 at n = 60)"
)
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
STAGE5_UNLOAD_SKIPPED_NO_11434_CALL_REASON: str = "no model call to 11434 in this run"
MODEL_GENERATION_API_PATHS: frozenset[str] = frozenset(
    {"/api/chat", "/api/generate", "/api/embed", "/api/embeddings"}
)


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
    """Raised when `--resume` is requested on a record lacking an integer `plan_seed` or `rung_seed` (`STAGE5_RESUME_NO_PLAN_SEED`)."""

    def __init__(self, message: str = "STAGE5_RESUME_NO_PLAN_SEED: resume record has no plan_seed.") -> None:
        super().__init__("STAGE5_RESUME_NO_PLAN_SEED", message)


class Stage5ResumeAfterSLtBError(Stage5PipelineError):
    """Raised when `--resume` is requested after an `S_LT_B` stop without resetting the stopped rendering (`STAGE5_RESUME_AFTER_S_LT_B`)."""

    def __init__(self, *, scenario_id: str, rendering_index: int) -> None:
        self.scenario_id = scenario_id
        self.rendering_index = int(rendering_index)
        msg = (
            f"STAGE5_RESUME_AFTER_S_LT_B: cannot resume after S_LT_B stop on "
            f"scenario_id={scenario_id} rendering_index={self.rendering_index} without explicit --stage5-redo-b."
        )
        super().__init__("STAGE5_RESUME_AFTER_S_LT_B", msg)


class Stage5RedoBSampledByRungError(Stage5PipelineError):
    """Raised when `--stage5-redo-b` targets a rendering sampled by the rung without `--stage5-redo-rung` (`STAGE5_REDO_B_SAMPLED_BY_RUNG`)."""

    def __init__(self, sampled_renderings: Sequence[tuple[str, int]]) -> None:
        self.sampled_renderings = [(str(s), int(r)) for s, r in sampled_renderings]
        formatted = [f"{s}:{r}" for s, r in self.sampled_renderings]
        msg = (
            f"STAGE5_REDO_B_SAMPLED_BY_RUNG: --stage5-redo-b affects rendering(s) sampled by the rung "
            f"{formatted}; pass --stage5-redo-rung to reset the rung."
        )
        super().__init__("STAGE5_REDO_B_SAMPLED_BY_RUNG", msg)


class Stage5ResumeEnvChangedError(Stage5PipelineError):
    """Raised when `--resume` detects environment or module SHA-256 drift (`STAGE5_RESUME_ENV_CHANGED`)."""

    def __init__(self, drifted_fields: Sequence[str]) -> None:
        self.drifted_fields = [str(f) for f in drifted_fields]
        msg = (
            f"STAGE5_RESUME_ENV_CHANGED: resume refused due to environment drift in "
            f"{json.dumps(self.drifted_fields)}"
        )
        super().__init__("STAGE5_RESUME_ENV_CHANGED", msg)


class Stage5RungSampleShortError(Stage5PipelineError):
    """Raised when fewer than 20 eligible `say` events (at most 1 per rendering) exist for any rung target (`STAGE5_RUNG_SAMPLE_SHORT`)."""

    def __init__(self, counts: Mapping[str, int]) -> None:
        self.counts = {str(k): int(v) for k, v in counts.items()}
        msg = (
            f"STAGE5_RUNG_SAMPLE_SHORT: fewer than {STAGE5_RUNG_PER_TARGET} eligible say events "
            f"(at most 1 per rendering) from passing B renderings: {json.dumps(self.counts, sort_keys=True)}"
        )
        super().__init__("STAGE5_RUNG_SAMPLE_SHORT", msg)


class Stage5PartialOffloadError(Stage5PipelineError):
    """Raised when `/api/ps` on the owned endpoint or `127.0.0.1:11434` reports partial GPU offload (`size_vram < size`), stating whether the offloaded model is the run's own model or another seat's model."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        size: int,
        size_vram: int,
        is_own_model: bool = True,
    ) -> None:
        self.endpoint = str(endpoint)
        self.model = str(model)
        self.size = int(size)
        self.size_vram = int(size_vram)
        self.is_own_model = bool(is_own_model)
        self.model_owner = "run's own model" if self.is_own_model else "another seat's model"
        msg = (
            f"STAGE5_PARTIAL_OFFLOAD: partial offload detected on {self.endpoint} "
            f"for model={self.model} ({self.model_owner}; size_vram={self.size_vram} < size={self.size})"
        )
        super().__init__("STAGE5_PARTIAL_OFFLOAD", msg)



def compute_stage5_module_sha256s() -> dict[str, str]:
    """Compute SHA-256 digests of the checker, template, and guard modules (`render.py`, `shape_templates.py`, `rewrite.py`)."""
    pkg_dir = Path(__file__).resolve().parent
    return {
        name: hashlib.sha256((pkg_dir / name).read_bytes()).hexdigest()
        for name in ("render.py", "shape_templates.py", "rewrite.py")
    }


def decide_rung_input_mode(
    facts_only_attempts: int,
    facts_only_kept: int,
    with_draft_attempts: int,
    with_draft_kept: int,
) -> str:
    """Apply the pre-registered rung rule in exact integer / rational arithmetic (never float division).

    At `n = 60` (`facts_only_attempts == 60 and with_draft_attempts == 60`), chooses `'with_draft'`
    iff `kept_with_draft - kept_facts_only >= 6`. In general, compares exact `Fraction` rates against `Fraction(1, 10)`.
    """
    if facts_only_attempts <= 0 or with_draft_attempts <= 0:
        raise ValueError("Rung attempts must be > 0.")
    if int(facts_only_attempts) == 60 and int(with_draft_attempts) == 60:
        if int(with_draft_kept) - int(facts_only_kept) >= 6:
            return "with_draft"
        return "facts_only"
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


class PlanStatisticsResult(tuple):  # type: ignore[type-arg]
    """3-tuple `(counts_by_kind, counts_by_variant, max_spread_by_variant)` carrying per-20-events block spreads (including tail) and global spreads."""

    counts_by_kind: dict[str, dict[str, int]]
    counts_by_variant: dict[str, dict[str, int]]
    block_balance_max_spread_by_variant: dict[str, int]
    block_balance_max_spread: int
    global_balance_max_spread_by_variant: dict[str, int]
    global_balance_max_spread: int
    stream_block_spreads_by_variant: dict[str, list[int]]
    stream_lengths_by_variant: dict[str, int]

    def __new__(
        cls,
        counts_by_kind: dict[str, dict[str, int]],
        counts_by_variant: dict[str, dict[str, int]],
        max_spread_by_variant: dict[str, int],
        *,
        global_spread_by_variant: dict[str, int],
        stream_block_spreads_by_variant: dict[str, list[int]],
        stream_lengths_by_variant: dict[str, int],
    ) -> PlanStatisticsResult:
        obj = super().__new__(cls, (counts_by_kind, counts_by_variant, max_spread_by_variant))
        obj.counts_by_kind = counts_by_kind
        obj.counts_by_variant = counts_by_variant
        obj.block_balance_max_spread_by_variant = max_spread_by_variant
        obj.block_balance_max_spread = max(max_spread_by_variant.values()) if max_spread_by_variant else 0
        obj.global_balance_max_spread_by_variant = global_spread_by_variant
        obj.global_balance_max_spread = max(global_spread_by_variant.values()) if global_spread_by_variant else 0
        obj.stream_block_spreads_by_variant = stream_block_spreads_by_variant
        obj.stream_lengths_by_variant = stream_lengths_by_variant
        return obj

    def to_dict(self) -> dict[str, object]:
        return {
            "planned_form_counts_by_kind": self.counts_by_kind,
            "planned_form_counts_by_variant": self.counts_by_variant,
            "stream_lengths_by_variant": self.stream_lengths_by_variant,
            "stream_block_spreads_by_variant": self.stream_block_spreads_by_variant,
            "block_balance_max_spread_by_variant": self.block_balance_max_spread_by_variant,
            "block_balance_max_spread": self.block_balance_max_spread,
            "global_balance_max_spread_by_variant": self.global_balance_max_spread_by_variant,
            "global_balance_max_spread": self.global_balance_max_spread,
        }


def _compute_plan_statistics(
    worlds: Sequence[World],
    plan_bundle: Mapping[str, Any],
    *,
    block_size: int = SHAPE_PLAN_BLOCK_SIZE,
) -> PlanStatisticsResult:
    """Compute per-kind form counts, per-variant form counts, per-stream 20-event block spreads (including tail), and global spread across `worlds`."""
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
    global_spread_by_variant: dict[str, int] = {v: 0 for v in STAGE5_ASSIGNED_VARIANTS}
    stream_block_spreads_by_variant: dict[str, list[int]] = {v: [] for v in STAGE5_ASSIGNED_VARIANTS}
    stream_lengths_by_variant: dict[str, int] = {v: 0 for v in STAGE5_ASSIGNED_VARIANTS}
    for v, forms in STAGE5_ASSIGNED_VARIANTS.items():
        stream_forms: list[str] = [
            str(rec["form_id"])
            for w in worlds
            for rec in plan_by_sid.get(w.scenario_id, ())
            if str(rec["variant"]) == v
        ]
        stream_lengths_by_variant[v] = len(stream_forms)
        for b_start in range(0, len(stream_forms), block_size):
            block_forms = stream_forms[b_start : b_start + block_size]
            b_counts = Counter(block_forms)
            vals = [b_counts[f] for f in forms]
            spread = max(vals) - min(vals)
            stream_block_spreads_by_variant[v].append(spread)
            if spread > max_spread_by_variant[v]:
                max_spread_by_variant[v] = spread
        if stream_forms:
            g_vals = [counts_by_variant[v][f] for f in forms]
            global_spread_by_variant[v] = max(g_vals) - min(g_vals)

    return PlanStatisticsResult(
        counts_by_kind,
        counts_by_variant,
        max_spread_by_variant,
        global_spread_by_variant=global_spread_by_variant,
        stream_block_spreads_by_variant=stream_block_spreads_by_variant,
        stream_lengths_by_variant=stream_lengths_by_variant,
    )


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
    num_predict: int | None = None,
    max_num_ctx: int = DEFAULT_WRITER_MAX_NUM_CTX,
    think: bool = DEFAULT_WRITER_THINK,
    goal_mode: str = DEFAULT_GOAL_MODE,
    prompt_token_counter: PromptTokenCounter | None = None,
    worlds: Sequence[World] | None = None,
) -> dict[str, object]:
    """Execute `--stage5 --dry`: build worlds and the Stage 5 plan, compute upper-bound call counts and over-cap count without constructing a writer."""
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
    plan_stats = _compute_plan_statistics(
        resolved_worlds,
        plan_bundle,
        block_size=SHAPE_PLAN_BLOCK_SIZE,
    )
    counts_by_kind, counts_by_variant, max_spread_by_variant = plan_stats
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

    # Evaluate context cap without constructing any writer (`count_prompt_tokens_exact` at render.py:8063
    # when `prompt_token_counter` is provided, or `estimate_prompt_tokens` at render.py:7890 otherwise,
    # checked via `prompt_fits_num_ctx_cap` at render.py:7920).
    resolved_goal_mode = str(goal_mode)
    resolved_think = bool(think)
    resolved_num_predict = (
        int(num_predict)
        if num_predict is not None
        else (DEFAULT_WRITER_THINK_NUM_PREDICT if resolved_think else DEFAULT_WRITER_NUM_PREDICT)
    )
    prompt_template_c = load_writer_prompt_template(
        goal_mode=resolved_goal_mode,
        constraint_layout="table",
        accepted_forms_line="off",
        shape_plan="off",
    )
    ctx_over_cap_count = 0
    for w in resolved_worlds:
        for r_idx in (0, 1):
            focal_ag = w.agents[r_idx]
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
            p_tokens = (
                count_prompt_tokens_exact(
                    messages, think=resolved_think, counter=prompt_token_counter
                )
                if prompt_token_counter is not None
                else estimate_prompt_tokens(messages)
            )
            if not prompt_fits_num_ctx_cap(
                p_tokens,
                resolved_num_predict,
                int(max_num_ctx),
                margin=NUM_CTX_MARGIN_TOKENS,
            ):
                ctx_over_cap_count += 1

    max_spread_overall = max(max_spread_by_variant.values()) if max_spread_by_variant else 0
    return {
        "dry": True,
        "plan_seed": plan_seed,
        "plan_seed_source": plan_seed_source,
        "seed_bank_sha256": seed_bank_sha256,
        "seed_bank_size": seed_bank_size,
        "scenarios": num_scenarios,
        "renderings": num_renderings,
        "ctx_over_cap": ctx_over_cap_count,
        "planned_form_counts_by_kind": counts_by_kind,
        "planned_form_counts_by_variant": counts_by_variant,
        "stream_lengths_by_variant": plan_stats.stream_lengths_by_variant,
        "stream_block_spreads_by_variant": plan_stats.stream_block_spreads_by_variant,
        "block_balance_max_spread_by_variant": max_spread_by_variant,
        "block_balance_max_spread": max_spread_overall,
        "global_balance_max_spread_by_variant": plan_stats.global_balance_max_spread_by_variant,
        "global_balance_max_spread": plan_stats.global_balance_max_spread,
        "expected_call_counts_is_upper_bound_before_ctx_cap": True,
        "expected_call_counts_upper_bound_before_ctx_cap": expected_call_counts,
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

        if rewrite_input_val not in (None, "facts_only", "with_draft"):
            raise ValueError(f"Invalid rewrite_input enum value: {rewrite_input_val!r}")

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


def _read_sealed_b_prose_for_rendering(
    sealed_responses_path: Path,
    scenario_id: str,
    rendering_index: int,
    *,
    goal_mode: str = DEFAULT_GOAL_MODE,
    world: World | None = None,
) -> str | None:
    """Read the latest sealed B response for a single non-over-cap `(scenario_id, rendering_index)` from `responses.sealed.jsonl`."""
    if not sealed_responses_path.is_file():
        return None
    found_text: str | None = None
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
        if sid != scenario_id or r_idx != int(rendering_index):
            continue
        resp_text = str(obj.get("response_text") or "")
        if not resp_text:
            continue
        found_text = resp_text
    if found_text is not None and goal_mode == "splice" and world is not None:
        try:
            found_text = splice_goal_tokens(found_text, world).expanded_prose
        except Exception:
            pass
    return found_text


def _fsync_directory_if_supported(dir_path: Path) -> None:
    """Fsync `dir_path` where the platform allows (POSIX directories supporting `os.open` + `os.fsync`)."""
    if not dir_path.is_dir():
        return
    flags = getattr(os, "O_RDONLY", 0) | getattr(os, "O_DIRECTORY", 0)
    try:
        dir_fd = os.open(str(dir_path), flags)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        try:
            os.close(dir_fd)
        except OSError:
            pass


def _flush_sealed_store(sealed_responses_path: Path) -> None:
    """Flush and `os.fsync` `responses.sealed.jsonl` (and its directory where supported) to disk before updating the run record."""
    if sealed_responses_path.is_file():
        with sealed_responses_path.open("r+b") as fh:
            fh.flush()
            os.fsync(fh.fileno())
        _fsync_directory_if_supported(sealed_responses_path.parent)


def _append_and_flush_sealed_response(
    sealed_responses_path: Path,
    entry: SealedResponseEntry,
) -> None:
    """Append a sealed response entry and immediately flush `responses.sealed.jsonl`."""
    append_sealed_response(sealed_responses_path, entry)
    _flush_sealed_store(sealed_responses_path)


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
    context_prose: str | None = None,
) -> int:
    """Append and flush sealed prompt/reply entries from `RewriteResult.sealed` to `responses.sealed.jsonl`."""
    idx = call_counter_start
    for entry in sealed_entries:
        idx += 1
        ev_idx = int(entry.get("event_index", 0))
        reply_txt = str(entry.get("reply") or "")
        msgs = entry.get("messages") or []
        if context_prose is not None:
            thinking_payload: Any = {"context_prose": context_prose, "messages": msgs}
        else:
            thinking_payload = msgs
        prompt_json = json.dumps(thinking_payload, sort_keys=True, separators=(",", ":"))
        resp_sha, think_sha = compute_sealed_response_digests(reply_txt, prompt_json)
        _append_and_flush_sealed_response(
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


def _atomic_write_json(target_path: Path, payload: Mapping[str, Any]) -> None:
    """Write `payload` to a sibling `.tmp` file, flush + `os.fsync`, and atomically replace `target_path` via `os.replace`."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f".{target_path.name}.tmp")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tmp_path.open("wb") as fh:
        fh.write(encoded)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, target_path)
    _fsync_directory_if_supported(target_path.parent)


def _format_model_endpoint(raw_url: str) -> str:
    """Format an endpoint URL as `<scheme>://<host>:<port>` (scheme, host, and port only)."""
    cleaned = str(raw_url or DEFAULT_OLLAMA_BASE_URL).strip().rstrip("/")
    parsed = urlsplit(cleaned)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return cleaned


def _is_model_generation_call(
    method: str,
    url: str,
    payload: Mapping[str, Any] | None = None,
) -> bool:
    """Return True iff `(method, url, payload)` is a model generation HTTP call (excluding `/api/ps`, metadata reads, and `keep_alive: 0` unloads)."""
    if str(method).upper() != "POST":
        return False
    path = urlsplit(str(url).strip()).path.rstrip("/")
    if path not in MODEL_GENERATION_API_PATHS:
        return False
    if isinstance(payload, Mapping):
        ka = payload.get("keep_alive")
        is_zero_keep_alive = not isinstance(ka, bool) and ka in (0, "0", "0s")
        has_generation_input = bool(
            payload.get("messages") or payload.get("prompt") or payload.get("input")
        )
        if is_zero_keep_alive and not has_generation_input:
            return False
    return True


class _CountingTransportClientProxy:
    """Non-callable proxy preserving `base_url` while counting HTTP model calls through `_transport`."""

    def __init__(self, inner_client: Any, wrapped_transport: OllamaHttpTransport) -> None:
        self._inner_client = inner_client
        self._transport = wrapped_transport

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner_client, name)


def _wrap_rewrite_client_for_endpoint_counts(
    client: Any,
    wrap_transport: Callable[[OllamaHttpTransport], OllamaHttpTransport],
) -> Any:
    """Wrap an injected `rewrite_client` so HTTP-transport calls count in `model_calls_by_endpoint` without counting 1-arg mock callables."""
    if callable(client):
        try:
            sig = inspect.signature(client)
        except (ValueError, TypeError):
            return wrap_transport(client)
        params = [
            p
            for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
        if not has_varargs and len(params) == 1:
            return client
        return wrap_transport(client)
    if hasattr(client, "_transport") and callable(client._transport):
        return _CountingTransportClientProxy(client, wrap_transport(client._transport))
    return client


def _default_utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format (`...Z`)."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _default_monotonic() -> float:
    """Return the current monotonic clock value in seconds."""
    return time.monotonic()


def _make_empty_step_timing() -> dict[str, Any]:
    """Create an initial unstarted step-timing entry."""
    return {
        "start": None,
        "end": None,
        "seconds": 0.0,
        "calls": 0,
        "call_count": 0,
    }


def _make_empty_step_timings() -> dict[str, dict[str, Any]]:
    """Create the initial `step_timings` mapping for `b`, `rung`, `swap`, and `rewrite`."""
    return {step: _make_empty_step_timing() for step in ("b", "rung", "swap", "rewrite")}


def _normalize_own_model_tags(
    model_tag: str | Sequence[str] | set[str] | frozenset[str] | None = None,
    *,
    own_model_tags: Sequence[str] | set[str] | frozenset[str] | None = None,
) -> tuple[str, ...]:
    """Normalize the run's own model tag(s) (writer, critic, and any other model called by the run) into a deduplicated tuple."""
    tags: list[str] = []
    for source in (model_tag, own_model_tags):
        if source is None:
            continue
        if isinstance(source, str):
            cleaned = source.strip()
            if cleaned and cleaned not in tags:
                tags.append(cleaned)
        elif isinstance(source, (Sequence, set, frozenset)):
            for item in source:
                if isinstance(item, str):
                    cleaned = item.strip()
                    if cleaned and cleaned not in tags:
                        tags.append(cleaned)
    if not tags:
        tags.append(PINNED_WRITER_MODEL_TAG)
    return tuple(tags)


def _is_own_model_tag(model_name: str, own_tags: Sequence[str]) -> bool:
    """Return True iff `model_name` matches one of the run's own model tags (`own_tags`)."""
    cleaned = model_name.strip()
    if not cleaned:
        return False
    for tag in own_tags:
        t = tag.strip()
        if not t:
            continue
        if cleaned == t:
            return True
        if ":" not in t and cleaned == f"{t}:latest":
            return True
        if t.endswith(":latest") and cleaned == t[:-7]:
            return True
    return False


def check_endpoints_partial_offload(
    http_transport: OllamaHttpTransport,
    *,
    owned_base_url: str,
    default_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    model_tag: str | Sequence[str] | set[str] | frozenset[str] | None = None,
    own_model_tags: Sequence[str] | set[str] | frozenset[str] | None = None,
) -> list[dict[str, object]]:
    """Check `GET /api/ps` on both `owned_base_url` and `default_base_url` (`127.0.0.1:11434`).

    Raises `Stage5PartialOffloadError` if any loaded model on either endpoint reports
    `size <= 0` or `size_vram < size`, stating whether the offloaded model is the run's
    own model or another seat's model.
    """
    resolved_own_tags = _normalize_own_model_tags(model_tag, own_model_tags=own_model_tags)
    owned_ep = _format_model_endpoint(owned_base_url)
    default_ep = _format_model_endpoint(default_base_url)
    endpoints: list[str] = [owned_ep]
    if default_ep not in endpoints:
        endpoints.append(default_ep)

    results: list[dict[str, object]] = []
    for ep in endpoints:
        resp = http_transport("GET", f"{ep}/api/ps", None)
        raw_models = resp.get("models") if isinstance(resp, Mapping) else []
        models_list = raw_models if isinstance(raw_models, list) else []
        checked_models: list[dict[str, object]] = []
        for entry in models_list:
            if not isinstance(entry, Mapping):
                continue
            model_name = str(entry.get("model") or entry.get("name") or "unknown")
            is_own = _is_own_model_tag(model_name, resolved_own_tags)
            size = int(entry.get("size", 0) or 0)
            size_vram = int(entry.get("size_vram", 0) or 0)
            if size <= 0 or size_vram < size:
                raise Stage5PartialOffloadError(
                    endpoint=ep,
                    model=model_name,
                    size=size,
                    size_vram=size_vram,
                    is_own_model=is_own,
                )
            checked_models.append(
                {
                    "model": model_name,
                    "is_own_model": is_own,
                    "size": size,
                    "size_vram": size_vram,
                    "full_offload": True,
                }
            )
        results.append(
            {
                "endpoint": ep,
                "models": checked_models,
                "full_offload": True,
            }
        )
    return results


def unload_and_verify_default_endpoint_empty(
    http_transport: OllamaHttpTransport,
    *,
    default_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    model_tag: str | Sequence[str] | set[str] | frozenset[str] | None = None,
    own_model_tags: Sequence[str] | set[str] | frozenset[str] | None = None,
    model_calls_by_endpoint: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Unload (`keep_alive: 0`) own-tag models on `default_base_url` (`127.0.0.1:11434`) only when `model_calls_by_endpoint` shows at least one call to `default_base_url`.

    When `model_calls_by_endpoint` shows no calls to `default_base_url`, no `keep_alive: 0` request
    is sent; `/api/ps` is still read and own-tag models are recorded under `own_tag_models_not_ours`,
    foreign models under `foreign_models` and `foreign_models_present`, `own_models_absent` is True
    (since this run loaded none on `default_base_url`), and `unload_skipped_reason` is set.
    """
    resolved_own_tags = _normalize_own_model_tags(model_tag, own_model_tags=own_model_tags)
    default_ep = _format_model_endpoint(default_base_url)
    ps_url = f"{default_ep}/api/ps"
    chat_url = f"{default_ep}/api/chat"
    calls_to_default = sum(
        int(cnt)
        for ep_key, cnt in (model_calls_by_endpoint or {}).items()
        if not isinstance(cnt, bool) and isinstance(cnt, int) and _format_model_endpoint(str(ep_key)) == default_ep
    )
    before_resp = http_transport("GET", ps_url, None)
    raw_models = before_resp.get("models") if isinstance(before_resp, Mapping) else []
    models_list = raw_models if isinstance(raw_models, list) else []

    if calls_to_default < 1:
        own_tag_models_not_ours: list[str] = []
        foreign_models_skipped: list[str] = []
        for entry in models_list:
            if not isinstance(entry, Mapping):
                continue
            model_name = str(entry.get("model") or entry.get("name") or "").strip()
            if not model_name:
                continue
            if _is_own_model_tag(model_name, resolved_own_tags):
                if model_name not in own_tag_models_not_ours:
                    own_tag_models_not_ours.append(model_name)
            else:
                if model_name not in foreign_models_skipped:
                    foreign_models_skipped.append(model_name)
        return {
            "endpoint": default_ep,
            "unloaded_models": [],
            "own_tag_models_not_ours": own_tag_models_not_ours,
            "foreign_models": foreign_models_skipped,
            "ps_after": dict(before_resp) if isinstance(before_resp, Mapping) else {"models": []},
            "own_models_absent": True,
            "foreign_models_present": list(foreign_models_skipped),
            "unload_skipped_reason": STAGE5_UNLOAD_SKIPPED_NO_11434_CALL_REASON,
        }

    unloaded_models: list[str] = []
    foreign_models: list[str] = []
    for entry in models_list:
        if not isinstance(entry, Mapping):
            continue
        model_name = str(entry.get("model") or entry.get("name") or "").strip()
        if not model_name:
            continue
        if _is_own_model_tag(model_name, resolved_own_tags):
            http_transport(
                "POST",
                chat_url,
                {
                    "model": model_name,
                    "messages": [],
                    "keep_alive": 0,
                    "stream": False,
                },
            )
            unloaded_models.append(model_name)
        else:
            if model_name not in foreign_models:
                foreign_models.append(model_name)

    after_resp = http_transport("GET", ps_url, None)
    after_raw_models = after_resp.get("models") if isinstance(after_resp, Mapping) else []
    after_models = after_raw_models if isinstance(after_raw_models, list) else []
    own_models_after: list[str] = []
    foreign_models_present: list[str] = []
    for entry in after_models:
        if not isinstance(entry, Mapping):
            continue
        model_name = str(entry.get("model") or entry.get("name") or "").strip()
        if not model_name:
            continue
        if _is_own_model_tag(model_name, resolved_own_tags):
            own_models_after.append(model_name)
        else:
            if model_name not in foreign_models_present:
                foreign_models_present.append(model_name)
            if model_name not in foreign_models:
                foreign_models.append(model_name)

    return {
        "endpoint": default_ep,
        "unloaded_models": unloaded_models,
        "foreign_models": foreign_models,
        "ps_after": dict(after_resp) if isinstance(after_resp, Mapping) else {"models": []},
        "own_models_absent": len(own_models_after) == 0,
        "foreign_models_present": foreign_models_present,
    }



class _RewriteClientAbort(BaseException):
    """Internal `BaseException` wrapper so transport exceptions escape `call_rewrite`'s `except Exception:`."""

    def __init__(self, original: BaseException) -> None:
        super().__init__(str(original))
        self.original = original


def _write_record_and_status(
    record_path: Path,
    status_path: Path,
    record: dict[str, Any],
    *,
    status_state: str,
    failure_class: str | None = None,
) -> None:
    """Write `stage5_run_record.json` and `stage5_status.json` atomically (`os.replace`) with no prose."""
    _atomic_write_json(record_path, record)
    status_doc = {
        "schema_version": STAGE5_SCHEMA_VERSION,
        "status": status_state,
        "stop_reason": record.get("stop_reason"),
        "failure_class": failure_class,
        "model_endpoint": record.get("model_endpoint"),
        "step_timings": record.get("step_timings"),
        "progress": record.get("progress"),
        "summary": record.get("summary"),
    }
    _atomic_write_json(status_path, status_doc)


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


def _hall_condition_holds(
    renderings_by_target: Mapping[str, set[tuple[str, int]]],
    quotas: Mapping[str, int],
    used_renderings: set[tuple[str, int]],
) -> bool:
    """Return True iff Hall's marriage condition holds for `quotas` across `STAGE5_RUNG_TARGETS` outside `used_renderings`."""
    active_targets = [t for t in STAGE5_RUNG_TARGETS if quotas.get(t, 0) > 0]
    for r_len in range(1, len(active_targets) + 1):
        for subset in combinations(active_targets, r_len):
            needed = sum(quotas[t] for t in subset)
            union_avail: set[tuple[str, int]] = set()
            for t in subset:
                union_avail |= (renderings_by_target[t] - used_renderings)
            if len(union_avail) < needed:
                return False
    return True


def _sample_rung_events_at_most_one_per_rendering(
    eligible_by_target: Mapping[str, Sequence[tuple[str, int, int, str]]],
    rung_seed: int,
    *,
    per_target: int = STAGE5_RUNG_PER_TARGET,
) -> tuple[list[tuple[str, int, int, str]], int]:
    """Sample `per_target` events per target (`SAY_2`, `SAY_3`, `SAY_4`) with at most 1 event per rendering `(sid, r_idx)`.

    Pools are sorted canonically by `(scenario_id, rendering_index, event_index)` before sampling.
    Raises `Stage5RungSampleShortError` if any target cannot reach `per_target` under the 1-per-rendering cap.
    Returns `(sampled_events, rung_scenario_overlap)`.
    """
    sorted_by_target: dict[str, list[tuple[str, int, int, str]]] = {
        t: sorted(
            eligible_by_target.get(t, ()),
            key=lambda item: (str(item[0]), int(item[1]), int(item[2]), str(item[3])),
        )
        for t in STAGE5_RUNG_TARGETS
    }
    events_by_target_and_rendering: dict[str, dict[tuple[str, int], list[tuple[str, int, int, str]]]] = {
        t: {} for t in STAGE5_RUNG_TARGETS
    }
    renderings_by_target: dict[str, set[tuple[str, int]]] = {t: set() for t in STAGE5_RUNG_TARGETS}
    for t in STAGE5_RUNG_TARGETS:
        for ev_tuple in sorted_by_target[t]:
            r_key = (str(ev_tuple[0]), int(ev_tuple[1]))
            events_by_target_and_rendering[t].setdefault(r_key, []).append(ev_tuple)
            renderings_by_target[t].add(r_key)

    quotas: dict[str, int] = {t: int(per_target) for t in STAGE5_RUNG_TARGETS}
    used_renderings: set[tuple[str, int]] = set()
    if not _hall_condition_holds(renderings_by_target, quotas, used_renderings):
        # Compute per-target distinct rendering counts (and if union across targets is short, reflect capped availability)
        counts_by_target: dict[str, int] = {
            t: len(renderings_by_target[t]) for t in STAGE5_RUNG_TARGETS
        }
        if all(counts_by_target[t] >= per_target for t in STAGE5_RUNG_TARGETS):
            # Individual targets have >= per_target renderings, but their union is too small for disjoint 1-per-rendering draws
            greedy_used: set[tuple[str, int]] = set()
            for t in STAGE5_RUNG_TARGETS:
                avail_t = sorted(renderings_by_target[t] - greedy_used)
                take_t = avail_t[:per_target]
                counts_by_target[t] = len(take_t)
                greedy_used.update(take_t)
        raise Stage5RungSampleShortError(counts_by_target)

    rung_rng = random.Random(rung_seed)
    sampled_events: list[tuple[str, int, int, str]] = []
    for t in STAGE5_RUNG_TARGETS:
        for _ in range(per_target):
            next_quotas = dict(quotas)
            next_quotas[t] -= 1
            valid_events: list[tuple[str, int, int, str]] = []
            for r_key in sorted(renderings_by_target[t] - used_renderings):
                cand_used = used_renderings | {r_key}
                if _hall_condition_holds(renderings_by_target, next_quotas, cand_used):
                    valid_events.extend(events_by_target_and_rendering[t][r_key])
            chosen_ev = rung_rng.choice(valid_events)
            sampled_events.append(chosen_ev)
            used_renderings.add((str(chosen_ev[0]), int(chosen_ev[1])))
            quotas[t] -= 1

    renderings_by_sid: dict[str, set[int]] = {}
    for sid, r_idx, _, _ in sampled_events:
        renderings_by_sid.setdefault(str(sid), set()).add(int(r_idx))
    rung_scenario_overlap = sum(1 for r_set in renderings_by_sid.values() if len(r_set) == 2)
    return sampled_events, rung_scenario_overlap


def _parse_redo_b_spec(
    redo_b: bool | Sequence[str | tuple[str, int]] | None,
    resolved_worlds: Sequence[World],
) -> tuple[bool, set[tuple[str, int]]]:
    """Parse `redo_b` into `(redo_b_all, redo_b_targets)`."""
    if redo_b is True:
        all_keys = {(w.scenario_id, r_idx) for w in resolved_worlds for r_idx in (0, 1)}
        return True, all_keys
    if not redo_b:
        return False, set()
    targets: set[tuple[str, int]] = set()
    for item in redo_b:
        if isinstance(item, tuple) and len(item) == 2:
            targets.add((str(item[0]), int(item[1])))
        elif isinstance(item, str):
            raw = item.strip()
            if ":" in raw:
                sid_part, r_part = raw.split(":", 1)
                targets.add((sid_part.strip(), int(r_part.strip())))
            elif raw:
                targets.add((raw, 0))
                targets.add((raw, 1))
    return False, targets


def _check_resume_environment_drift(
    existing_record: Mapping[str, Any],
    *,
    writer_kind: str,
    model_tag: str,
    model_digest: str | None,
    ollama_version: str | None,
    num_predict: int,
    max_num_ctx: int,
    current_module_sha256s: Mapping[str, str],
    check_ollama_version: bool,
) -> None:
    """Refuse `--resume` with `Stage5ResumeEnvChangedError` (`STAGE5_RESUME_ENV_CHANGED`) if environment or module shas drifted."""
    drifted: list[str] = []
    prev_engine = existing_record.get("engine")
    if isinstance(prev_engine, Mapping):
        if prev_engine.get("writer_kind") != writer_kind:
            drifted.append("engine.writer_kind")
        if prev_engine.get("model") != model_tag:
            drifted.append("engine.model")
        if prev_engine.get("model_digest") != model_digest:
            drifted.append("engine.model_digest")
        if check_ollama_version and prev_engine.get("ollama_version") != ollama_version:
            drifted.append("engine.ollama_version")
    else:
        drifted.append("engine")

    if existing_record.get("num_predict") != int(num_predict):
        drifted.append("num_predict")
    if existing_record.get("max_num_ctx") != int(max_num_ctx):
        drifted.append("max_num_ctx")

    prev_mod_shas = existing_record.get("module_sha256s")
    if isinstance(prev_mod_shas, Mapping):
        for mod_name in ("render.py", "shape_templates.py", "rewrite.py"):
            if prev_mod_shas.get(mod_name) != current_module_sha256s.get(mod_name):
                drifted.append(f"module_sha256s.{mod_name}")
    else:
        drifted.append("module_sha256s")

    if drifted:
        raise Stage5ResumeEnvChangedError(drifted)


def run_stage5_pipeline(
    *,
    count: int = 80,
    base_seed: int = 20261030,
    seed_bank: Path | Sequence[int] | None = None,
    output_dir: Path,
    writer_kind: str = "mock",
    stage5_plan_seed: int | None = None,
    stage5_rung_seed: int | None = None,
    redo_b: bool | Sequence[str | tuple[str, int]] = False,
    redo_rung: bool = False,
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
    rung_mode_order: tuple[str, str] = ("facts_only", "with_draft"),
    rendering_execution_order: Sequence[tuple[str, int]] | None = None,
    utc_clock: Callable[[], datetime.datetime | str] | None = None,
    monotonic_clock: Callable[[], float] | None = None,
    guard_both_endpoints: bool = False,
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
    if sorted(rung_mode_order) != ["facts_only", "with_draft"]:
        raise ValueError(
            f"rung_mode_order must be a permutation of ('facts_only', 'with_draft'), got {rung_mode_order!r}."
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
    current_module_sha256s = compute_stage5_module_sha256s()

    output_dir.mkdir(parents=True, exist_ok=True)
    sealed_dir = output_dir / "sealed_store"
    sealed_dir.mkdir(parents=True, exist_ok=True)
    generations_path = sealed_dir / GENERATIONS_FILENAME
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
        raw_rs = loaded_obj.get("rung_seed")
        if isinstance(raw_rs, bool) or not isinstance(raw_rs, int):
            raise Stage5ResumeNoPlanSeedError(
                "STAGE5_RESUME_NO_PLAN_SEED: resume record has no valid integer rung_seed."
            )
        plan_seed = int(raw_ps)
        plan_seed_source = str(loaded_obj.get("plan_seed_source") or "secrets")
        rung_seed = int(raw_rs)
        rung_seed_source = str(loaded_obj.get("rung_seed_source") or "secrets")
        existing_record = loaded_obj

        # Check environment & module SHA-256 drift before any writer / server startup
        _check_resume_environment_drift(
            existing_record,
            writer_kind=writer_kind,
            model_tag=effective_writer_pin.ollama_tag,
            model_digest=effective_writer_pin.manifest_digest if writer_kind == "ollama" else None,
            ollama_version=None,
            num_predict=resolved_num_predict,
            max_num_ctx=int(max_num_ctx),
            current_module_sha256s=current_module_sha256s,
            check_ollama_version=(writer_kind == "mock"),
        )
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
    redo_b_all, redo_b_targets = _parse_redo_b_spec(redo_b, resolved_worlds)
    redo_b_flag = bool(redo_b_all or bool(redo_b_targets))
    redo_rung_flag = bool(redo_rung)

    existing_renderings_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    b_prose_by_key: dict[tuple[str, int], str] = {}
    existing_rung: dict[str, Any] | None = None
    existing_rung_done: bool = False
    b_calls_made = 0
    rung_calls_made = 0
    rung_aborted_calls = 0
    rewrite_calls_made = 0
    model_calls_by_endpoint: dict[str, int] = {}

    if existing_record is not None:
        prev_calls = existing_record.get("call_counts") or {}
        if isinstance(prev_calls, dict):
            b_calls_made = int(prev_calls.get("b_calls", 0))
            rung_calls_made = int(prev_calls.get("rung_calls", 0))
            rung_aborted_calls = int(
                existing_record.get(
                    "rung_aborted_calls",
                    prev_calls.get("rung_aborted_calls", 0),
                )
            )
            rewrite_calls_made = int(prev_calls.get("rewrite_calls", 0))

        prev_prog = existing_record.get("progress") or {}
        prev_b_done = int(prev_prog.get("b_done", len(existing_record.get("renderings") or ())))
        prev_s_done = int(prev_prog.get("s_done", 0))
        prev_renderings_list = list(existing_record.get("renderings") or ())

        # Addendum 2 Item 3: Resume after S_LT_B stop refuses unless the stopped rendering is explicitly reset via --stage5-redo-b
        stopped_s_lt_b_keys: list[tuple[str, int]] = []
        if existing_record.get("stop_reason") == "S_LT_B":
            sd = existing_record.get("stop_detail")
            if isinstance(sd, Mapping) and sd.get("scenario_id") is not None and sd.get("rendering_index") is not None:
                stopped_s_lt_b_keys.append((str(sd["scenario_id"]), int(sd["rendering_index"])))
        for idx_r, r_entry in enumerate(prev_renderings_list):
            if idx_r >= prev_s_done:
                break
            if isinstance(r_entry, Mapping) and r_entry.get("b_passed") is True and r_entry.get("s_passed") is False:
                k_stop = (str(r_entry["scenario_id"]), int(r_entry["rendering_index"]))
                if k_stop not in stopped_s_lt_b_keys:
                    stopped_s_lt_b_keys.append(k_stop)

        for stop_sid, stop_r_idx in stopped_s_lt_b_keys:
            if not redo_b_all and (stop_sid, stop_r_idx) not in redo_b_targets:
                raise Stage5ResumeAfterSLtBError(
                    scenario_id=stop_sid,
                    rendering_index=stop_r_idx,
                )

        # Addendum 2 Item 4: --stage5-redo-b on a rendering sampled by the rung refuses unless --stage5-redo-rung is set
        prev_rung = existing_record.get("rung")
        sampled_r_set: set[tuple[str, int]] = set()
        if isinstance(prev_rung, Mapping):
            raw_sampled_rs = prev_rung.get("sampled_renderings")
            if isinstance(raw_sampled_rs, Sequence):
                for item in raw_sampled_rs:
                    if isinstance(item, Sequence) and len(item) == 2:
                        sampled_r_set.add((str(item[0]), int(item[1])))
        if redo_b_flag and sampled_r_set:
            affected_sampled = sorted(sampled_r_set & redo_b_targets)
            if affected_sampled and not redo_rung_flag:
                raise Stage5RedoBSampledByRungError(affected_sampled)

        # Restore non-reset renderings up to prev_b_done
        for idx_r, r_entry in enumerate(prev_renderings_list):
            if idx_r >= prev_b_done:
                break
            if isinstance(r_entry, dict):
                k = (str(r_entry["scenario_id"]), int(r_entry["rendering_index"]))
                if k in redo_b_targets:
                    continue
                cp = dict(r_entry)
                cp["_s_completed"] = bool(
                    not redo_rung_flag
                    and idx_r < prev_s_done
                    and not (cp.get("b_passed") and not cp.get("s_passed"))
                )
                existing_renderings_by_key[k] = cp

        if redo_rung_flag:
            existing_rung = None
            existing_rung_done = False
            rung_calls_made = 0
            rewrite_calls_made = 0
        elif (
            isinstance(prev_rung, dict)
            and prev_rung.get("chosen") in ("facts_only", "with_draft")
            and isinstance(prev_prog, dict)
            and bool(prev_prog.get("rung_done"))
        ):
            existing_rung = dict(prev_rung)
            existing_rung_done = True
        else:
            existing_rung = None
            existing_rung_done = False
            rung_calls_made = 0

    prev_engine_ver = (
        existing_record.get("engine", {}).get("ollama_version")
        if isinstance(existing_record, Mapping) and isinstance(existing_record.get("engine"), Mapping)
        else None
    )
    engine_block: dict[str, object] = {
        "writer_kind": writer_kind,
        "ollama_version": prev_engine_ver if writer_kind == "ollama" else None,
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
            "rung_scenario_overlap": 0,
            "sampled_renderings": [],
        }
    )

    def _read_utc() -> str:
        if utc_clock is not None:
            raw_utc = utc_clock()
            if isinstance(raw_utc, datetime.datetime):
                return raw_utc.astimezone(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
            return str(raw_utc)
        return _default_utc_now()

    def _read_mono() -> float:
        if monotonic_clock is not None:
            return float(monotonic_clock())
        return _default_monotonic()

    active_ollama_base_url = _format_model_endpoint(
        f"http://127.0.0.1:{int(server_port)}"
        if (server_port is not None and ollama_base_url == DEFAULT_OLLAMA_BASE_URL)
        else ollama_base_url
    )
    model_endpoint_block: dict[str, str] = {
        "b": active_ollama_base_url,
        "rung": active_ollama_base_url,
        "swap": active_ollama_base_url,
        "rewrite": active_ollama_base_url,
    }
    step_timings_block: dict[str, dict[str, Any]] = _make_empty_step_timings()
    prev_b_timing_complete = False
    prev_rung_timing_complete = False
    prev_swap_timing_complete = False
    prev_rewrite_timing_complete = False
    if existing_record is not None and isinstance(existing_record.get("step_timings"), Mapping):
        prev_st = existing_record["step_timings"]
        for step_k in ("b", "rung", "swap", "rewrite"):
            raw_entry = prev_st.get(step_k)
            if isinstance(raw_entry, Mapping):
                raw_calls = int(raw_entry.get("calls", raw_entry.get("call_count", 0)) or 0)
                step_timings_block[step_k] = {
                    "start": raw_entry.get("start"),
                    "end": raw_entry.get("end"),
                    "seconds": float(raw_entry.get("seconds", 0.0) or 0.0),
                    "calls": raw_calls,
                    "call_count": int(raw_entry.get("call_count", raw_calls) or 0),
                }
        if (
            not redo_b_flag
            and len(existing_renderings_by_key) == total_renderings
            and step_timings_block["b"].get("start") is not None
            and step_timings_block["b"].get("end") is not None
        ):
            prev_b_timing_complete = True
        if (
            existing_rung_done
            and step_timings_block["rung"].get("start") is not None
            and step_timings_block["rung"].get("end") is not None
        ):
            prev_rung_timing_complete = True
        if (
            not redo_b_flag
            and not redo_rung_flag
            and step_timings_block["swap"].get("start") is not None
            and step_timings_block["swap"].get("end") is not None
        ):
            prev_swap_timing_complete = True
        if (
            not redo_b_flag
            and not redo_rung_flag
            and step_timings_block["rewrite"].get("start") is not None
            and step_timings_block["rewrite"].get("end") is not None
        ):
            prev_rewrite_timing_complete = True

    renderings_records: list[dict[str, Any]] = []
    record: dict[str, Any] = {
        "schema_version": STAGE5_SCHEMA_VERSION,
        "plan_seed": plan_seed,
        "plan_seed_source": plan_seed_source,
        "rung_seed": rung_seed,
        "rung_seed_source": rung_seed_source,
        "seed_bank_sha256": seed_bank_sha256,
        "seed_bank_size": seed_bank_size,
        "module_sha256s": current_module_sha256s,
        "model_calls_by_endpoint": model_calls_by_endpoint,
        "model_endpoint": model_endpoint_block,
        "step_timings": step_timings_block,
        "rung": default_rung_block,
        "rung_scenario_overlap": int(default_rung_block.get("rung_scenario_overlap", 0)),
        "rung_aborted_calls": rung_aborted_calls,
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
                "rung_aborted_calls": rung_aborted_calls,
                "rewrite_calls": rewrite_calls_made,
            },
        },
        "redo_b": redo_b_flag,
        "redo_rung": redo_rung_flag,
        "engine": engine_block,
        "max_num_ctx": int(max_num_ctx),
        "num_predict": resolved_num_predict,
        "call_counts": {
            "b_calls": b_calls_made,
            "rung_calls": rung_calls_made,
            "rung_aborted_calls": rung_aborted_calls,
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

    def _wrap_counting_transport(inner: OllamaHttpTransport) -> OllamaHttpTransport:
        def _counted_transport(
            method: str,
            url: str,
            payload: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            if _is_model_generation_call(method, url, payload):
                ep = _format_model_endpoint(url)
                model_calls_by_endpoint[ep] = model_calls_by_endpoint.get(ep, 0) + 1
            return inner(method, url, payload)

        return _counted_transport

    ollama_writer: OllamaWriter | None = None
    mock_writer: MockWriter | None = None
    active_rewrite_client: Any = (
        _wrap_rewrite_client_for_endpoint_counts(rewrite_client, _wrap_counting_transport)
        if rewrite_client is not None
        else None
    )

    need_any_b = len(existing_renderings_by_key) < total_renderings
    if writer_kind == "ollama":
        should_own_server = (
            start_owned_server
            if start_owned_server is not None
            else (http_transport is None or popen_factory is not None)
        )
        effective_base_url = (
            f"http://127.0.0.1:{int(server_port)}"
            if (server_port is not None and ollama_base_url == DEFAULT_OLLAMA_BASE_URL)
            else ollama_base_url
        )
        ollama_writer = OllamaWriter(
            model_pin=effective_writer_pin,
            base_url=effective_base_url,
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
        ollama_writer._transport = _wrap_counting_transport(ollama_writer._transport)

    run_own_model_tags: list[str] = []

    def _record_own_model_tag(tag_value: object) -> None:
        if isinstance(tag_value, str):
            cleaned_tag = tag_value.strip()
            if cleaned_tag and cleaned_tag not in run_own_model_tags:
                run_own_model_tags.append(cleaned_tag)

    _record_own_model_tag(effective_writer_pin.ollama_tag)
    if ollama_writer is not None:
        _record_own_model_tag(getattr(ollama_writer, "ollama_tag", None))
    for _k in ("model", "writer_model", "critic_model", "rewrite_model"):
        _record_own_model_tag(engine_block.get(_k))
        if rewrite_client is not None:
            _record_own_model_tag(getattr(rewrite_client, _k, None))

    def _check_partial_offload_or_stop(phase_name: str) -> None:
        if not guard_both_endpoints or ollama_writer is None:
            return
        raw_transport = getattr(ollama_writer._transport, "_raw", ollama_writer._transport)
        try:
            check_endpoints_partial_offload(
                raw_transport,
                owned_base_url=active_ollama_base_url,
                default_base_url=DEFAULT_OLLAMA_BASE_URL,
                model_tag=effective_writer_pin.ollama_tag,
                own_model_tags=tuple(run_own_model_tags),
            )
        except Stage5PartialOffloadError as po_exc:
            record["stop_reason"] = "PARTIAL_OFFLOAD"
            record["stop_detail"] = {
                "failure_class": "STAGE5_PARTIAL_OFFLOAD",
                "phase": phase_name,
                "endpoint": po_exc.endpoint,
                "model": po_exc.model,
                "is_own_model": po_exc.is_own_model,
                "model_owner": po_exc.model_owner,
                "size": po_exc.size,
                "size_vram": po_exc.size_vram,
            }
            _write_record_and_status(
                record_path,
                status_path,
                record,
                status_state="stopped",
                failure_class="STAGE5_PARTIAL_OFFLOAD",
            )
            raise

    try:
        if ollama_writer is not None:
            owned_base_url = (
                getattr(ollama_writer.owned_server, "base_url", None)
                if ollama_writer.owned_server is not None
                else None
            )
            if owned_base_url:
                active_ollama_base_url = _format_model_endpoint(str(owned_base_url))
            elif getattr(ollama_writer, "base_url", None):
                active_ollama_base_url = _format_model_endpoint(str(ollama_writer.base_url))
            for step_k in ("b", "rung", "swap", "rewrite"):
                model_endpoint_block[step_k] = active_ollama_base_url

            live_ollama_version: str | None = None
            live_model_digest: str | None = getattr(
                ollama_writer, "manifest_digest", effective_writer_pin.manifest_digest
            )
            if ollama_writer.owned_server is not None:
                live_ollama_version = getattr(ollama_writer.owned_server, "version", None)
                if live_ollama_version is None:
                    live_ollama_version = getattr(ollama_writer.owned_server, "ollama_version", None)
                owned_digest = getattr(ollama_writer.owned_server, "model_digest", None) or getattr(
                    ollama_writer.owned_server, "manifest_digest", None
                )
                if owned_digest is not None:
                    live_model_digest = str(owned_digest)
            if existing_record is not None:
                # U4-fix1 Item 1 (F2): Right after the owned server starts and reports its version, and BEFORE
                # the first writer call, compare live version (and model digest) with the record's.
                _check_resume_environment_drift(
                    existing_record,
                    writer_kind=writer_kind,
                    model_tag=effective_writer_pin.ollama_tag,
                    model_digest=live_model_digest,
                    ollama_version=live_ollama_version,
                    num_predict=resolved_num_predict,
                    max_num_ctx=int(max_num_ctx),
                    current_module_sha256s=current_module_sha256s,
                    check_ollama_version=True,
                )
            else:
                # On a fresh run, record the live version at start once the server reports it, not only at close.
                engine_block["ollama_version"] = live_ollama_version
                engine_block["model_digest"] = live_model_digest
            if active_rewrite_client is None:
                active_rewrite_client = ollama_writer._transport

        # Write `plan_seed`, `rung_seed`, and `engine` (including live `ollama_version` on fresh run) BEFORE any writer call.
        _write_record_and_status(record_path, status_path, record, status_state="running")
        if on_before_first_writer_call is not None:
            on_before_first_writer_call(record_path)

        if writer_kind != "ollama":
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

        ordered_rendering_keys: list[tuple[str, int]] = (
            list(rendering_execution_order)
            if rendering_execution_order is not None
            else [(w.scenario_id, r_idx) for w in resolved_worlds for r_idx in (0, 1)]
        )

        model_endpoint_block["b"] = active_ollama_base_url
        if need_any_b:
            _check_partial_offload_or_stop("b")
        if not need_any_b and prev_b_timing_complete:
            b_mono_start = None
        else:
            step_timings_block["b"]["start"] = _read_utc()
            b_mono_start = _read_mono()
        try:
            for sid, r_idx in ordered_rendering_keys:
                w = worlds_by_sid[sid]
                scen_plan_by_idx = scen_plan_dict_by_sid[sid]
                r_key = (sid, r_idx)
                focal_ag = w.agents[r_idx]
                if r_key in existing_renderings_by_key:
                    prev_r = existing_renderings_by_key[r_key]
                    renderings_records.append(prev_r)
                    # Addendum 2 Item 5: Over-cap renderings make 0 writer calls and 0 sealed-store reads on resume
                    if not prev_r.get("ctx_over_cap"):
                        restored_prose = _read_sealed_b_prose_for_rendering(
                            sealed_responses_path,
                            sid,
                            r_idx,
                            goal_mode=resolved_goal_mode,
                            world=w,
                        )
                        if restored_prose is not None:
                            b_prose_by_key[r_key] = restored_prose
                            b_rows_restored = match_rendering_event_forms(
                                restored_prose,
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
                    _flush_sealed_store(sealed_responses_path)
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
                    # Write and flush the sealed B response BEFORE updating the record
                    _append_and_flush_sealed_response(
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
            if b_mono_start is not None:
                step_timings_block["b"]["end"] = _read_utc()
                step_timings_block["b"]["seconds"] = round(max(0.0, _read_mono() - b_mono_start), 6)
            step_timings_block["b"]["calls"] = b_calls_made
            step_timings_block["b"]["call_count"] = b_calls_made
            if rendering_execution_order is not None:
                canonical_index = {
                    (w.scenario_id, r_idx): idx
                    for idx, (w, r_idx) in enumerate(
                        (w, r_idx) for w in resolved_worlds for r_idx in (0, 1)
                    )
                }
                renderings_records.sort(
                    key=lambda r: canonical_index.get((str(r["scenario_id"]), int(r["rendering_index"])), 0)
                )
            record["call_counts"]["b_calls"] = b_calls_made
            record["progress"]["b_done"] = len(renderings_records)
            record["progress"]["calls_made"]["b_calls"] = b_calls_made
            record["summary"] = _recompute_summary(renderings_records)
            _write_record_and_status(record_path, status_path, record, status_state="running")

        # -----------------------------------------------------------------------
        # Phase 2: The Rewrite-Input Rung
        # -----------------------------------------------------------------------
        model_endpoint_block["rung"] = active_ollama_base_url
        if existing_rung_done and existing_rung is not None:
            chosen_input = str(existing_rung["chosen"])
            record["rung"] = existing_rung
            record["rung_scenario_overlap"] = int(existing_rung.get("rung_scenario_overlap", 0))
            if not prev_rung_timing_complete:
                step_timings_block["rung"]["start"] = _read_utc()
                rung_mono_start = _read_mono()
                step_timings_block["rung"]["end"] = _read_utc()
                step_timings_block["rung"]["seconds"] = round(max(0.0, _read_mono() - rung_mono_start), 6)
            step_timings_block["rung"]["calls"] = rung_calls_made
            step_timings_block["rung"]["call_count"] = rung_calls_made
            _write_record_and_status(record_path, status_path, record, status_state="running")
        elif s_code_say:
            step_timings_block["rung"]["start"] = _read_utc()
            rung_mono_start = _read_mono()
            chosen_input = "facts_only"
            record["rung"] = {
                "facts_only": {"attempts": 0, "kept": 0},
                "with_draft": {"attempts": 0, "kept": 0},
                "chosen": "facts_only",
                "rule": STAGE5_RUNG_RULE,
                "rung_scenario_overlap": 0,
                "sampled_renderings": [],
            }
            record["rung_scenario_overlap"] = 0
            record["progress"]["rung_done"] = True
            step_timings_block["rung"]["end"] = _read_utc()
            step_timings_block["rung"]["seconds"] = round(max(0.0, _read_mono() - rung_mono_start), 6)
            step_timings_block["rung"]["calls"] = 0
            step_timings_block["rung"]["call_count"] = 0
            _write_record_and_status(record_path, status_path, record, status_state="running")
        else:
            _check_partial_offload_or_stop("rung")
            if ollama_writer is not None and not ollama_writer._verified_local_digest:
                ollama_writer._ensure_verified()
            step_timings_block["rung"]["start"] = _read_utc()
            rung_mono_start = _read_mono()
            eligible_by_target: dict[str, list[tuple[str, int, int, str]]] = {
                t: [] for t in STAGE5_RUNG_TARGETS
            }
            # Pool filter is `b_passed and not ctx_over_cap`, sorted canonically by (scenario_id, rendering_index, event_index)
            for r_entry in sorted(
                renderings_records,
                key=lambda r: (str(r["scenario_id"]), int(r["rendering_index"])),
            ):
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

            try:
                sampled_rung_events, rung_scenario_overlap = _sample_rung_events_at_most_one_per_rendering(
                    eligible_by_target,
                    rung_seed,
                    per_target=STAGE5_RUNG_PER_TARGET,
                )
            except Stage5RungSampleShortError:
                step_timings_block["rung"]["end"] = _read_utc()
                step_timings_block["rung"]["seconds"] = round(max(0.0, _read_mono() - rung_mono_start), 6)
                step_timings_block["rung"]["calls"] = 0
                step_timings_block["rung"]["call_count"] = 0
                _write_record_and_status(
                    record_path,
                    status_path,
                    record,
                    status_state="stopped",
                    failure_class="STAGE5_RUNG_SAMPLE_SHORT",
                )
                raise

            sampled_renderings_list = [
                [str(sid), int(r_idx)] for sid, r_idx, _, _ in sampled_rung_events
            ]

            # Addendum 2 Item 13: Apply U2's deterministic swap pass (`swap_all_move_see`, with the run's plan)
            # in memory to each sampled rendering's B before running the rung's two modes.
            rung_post_swap_prose_by_key: dict[tuple[str, int], str] = {}
            for sid, r_idx, _, _ in sampled_rung_events:
                r_key = (sid, r_idx)
                if r_key not in rung_post_swap_prose_by_key:
                    w = worlds_by_sid[sid]
                    focal_ag = w.agents[r_idx]
                    scen_plan_by_idx = scen_plan_dict_by_sid[sid]
                    in_memory_swap = swap_all_move_see(
                        b_prose_by_key[r_key],
                        w,
                        scen_plan_by_idx,
                        focal_agent=focal_ag,
                        goal_mode=resolved_goal_mode,
                        checker_profile="d",
                    )
                    rung_post_swap_prose_by_key[r_key] = in_memory_swap.prose

            # Addendum 2 Item 6: Track current rung calls and propagate any client exception to abort mid-rung
            current_rung_attempt_calls = 0

            def _rung_client_proxy(payload: dict[str, Any]) -> Any:
                nonlocal current_rung_attempt_calls
                current_rung_attempt_calls += 1
                if isinstance(payload, Mapping):
                    _record_own_model_tag(payload.get("model"))
                try:
                    return _invoke_rewrite_client(
                        active_rewrite_client,
                        f"{active_ollama_base_url}/api/chat",
                        payload,
                    )
                except BaseException as exc:
                    raise _RewriteClientAbort(exc) from exc

            rung_stats: dict[str, dict[str, int]] = {
                "facts_only": {"attempts": 0, "kept": 0},
                "with_draft": {"attempts": 0, "kept": 0},
            }
            try:
                for mode in rung_mode_order:
                    for sid, r_idx, ev_idx, target_form in sampled_rung_events:
                        w = worlds_by_sid[sid]
                        focal_ag = w.agents[r_idx]
                        post_swap_scene_prose = rung_post_swap_prose_by_key[(sid, r_idx)]
                        single_event_plan = {ev_idx: target_form}
                        try:
                            rung_res = rewrite_all_say(
                                post_swap_scene_prose,
                                w,
                                single_event_plan,
                                _rung_client_proxy,
                                input_mode=mode,
                                focal_agent=focal_ag,
                                goal_mode=resolved_goal_mode,
                                checker_profile="d",
                                base_url=active_ollama_base_url,
                                model=effective_writer_pin.ollama_tag,
                            )
                        except _RewriteClientAbort as abort_exc:
                            raise abort_exc.original from None
                        sealed_call_counter = _seal_rewrite_entries(
                            sealed_responses_path,
                            scenario_id=sid,
                            rendering_index=r_idx,
                            focal_agent=focal_ag,
                            phase_tag="stage5_rung",
                            sealed_entries=rung_res.sealed,
                            call_counter_start=sealed_call_counter,
                            goal_mode=resolved_goal_mode,
                            context_prose=post_swap_scene_prose,
                        )
                        ev_rec = rung_res.events[ev_idx - 1]
                        rung_stats[mode]["attempts"] += 1
                        if ev_rec.action == "rewritten":
                            rung_stats[mode]["kept"] += 1
            except BaseException:
                rung_aborted_calls += current_rung_attempt_calls
                step_timings_block["rung"]["end"] = _read_utc()
                step_timings_block["rung"]["seconds"] = round(max(0.0, _read_mono() - rung_mono_start), 6)
                step_timings_block["rung"]["calls"] = current_rung_attempt_calls
                step_timings_block["rung"]["call_count"] = current_rung_attempt_calls
                record["rung_aborted_calls"] = rung_aborted_calls
                record["call_counts"]["rung_aborted_calls"] = rung_aborted_calls
                record["call_counts"]["rung_calls"] = 0
                record["progress"]["rung_done"] = False
                record["progress"]["calls_made"]["rung_calls"] = 0
                record["progress"]["calls_made"]["rung_aborted_calls"] = rung_aborted_calls
                _write_record_and_status(
                    record_path,
                    status_path,
                    record,
                    status_state="stopped",
                    failure_class="STAGE5_RUNG_ABORTED",
                )
                raise

            rung_calls_made = current_rung_attempt_calls
            step_timings_block["rung"]["end"] = _read_utc()
            step_timings_block["rung"]["seconds"] = round(max(0.0, _read_mono() - rung_mono_start), 6)
            step_timings_block["rung"]["calls"] = rung_calls_made
            step_timings_block["rung"]["call_count"] = rung_calls_made
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
                "rung_scenario_overlap": rung_scenario_overlap,
                "sampled_renderings": sampled_renderings_list,
            }
            record["rung_scenario_overlap"] = rung_scenario_overlap
            record["call_counts"]["rung_calls"] = rung_calls_made
            record["progress"]["rung_done"] = True
            record["progress"]["calls_made"]["rung_calls"] = rung_calls_made
            _write_record_and_status(record_path, status_path, record, status_state="running")

        # -----------------------------------------------------------------------
        # Phase 3: Swap, Rewrite, Final Check, and S >= B Stop
        # -----------------------------------------------------------------------
        any_s_needed = any(
            not r_entry.get("ctx_over_cap")
            and not (resume and (str(r_entry["scenario_id"]), int(r_entry["rendering_index"])) in s_completed_keys)
            for r_entry in renderings_records
        )
        model_endpoint_block["swap"] = active_ollama_base_url
        if not any_s_needed and prev_swap_timing_complete:
            swap_mono_start = None
        else:
            step_timings_block["swap"]["start"] = _read_utc()
            swap_mono_start = _read_mono()
        swap_res_by_key: dict[tuple[str, int], SwapResult] = {}
        try:
            for r_entry in renderings_records:
                if r_entry.get("ctx_over_cap"):
                    continue
                sid = str(r_entry["scenario_id"])
                r_idx = int(r_entry["rendering_index"])
                r_key = (sid, r_idx)
                if resume and r_key in s_completed_keys:
                    continue
                w = worlds_by_sid[sid]
                focal_ag = w.agents[r_idx]
                scen_plan_by_idx = scen_plan_dict_by_sid[sid]
                swap_res_by_key[r_key] = swap_all_move_see(
                    b_prose_by_key[r_key],
                    w,
                    scen_plan_by_idx,
                    focal_agent=focal_ag,
                    goal_mode=resolved_goal_mode,
                    checker_profile="d",
                )
        finally:
            if swap_mono_start is not None:
                step_timings_block["swap"]["end"] = _read_utc()
                step_timings_block["swap"]["seconds"] = round(max(0.0, _read_mono() - swap_mono_start), 6)
            step_timings_block["swap"]["calls"] = 0
            step_timings_block["swap"]["call_count"] = 0
            _write_record_and_status(record_path, status_path, record, status_state="running")

        def _rewrite_client_proxy(payload: dict[str, Any]) -> Any:
            if isinstance(payload, Mapping):
                _record_own_model_tag(payload.get("model"))
            try:
                return _invoke_rewrite_client(
                    active_rewrite_client,
                    f"{active_ollama_base_url}/api/chat",
                    payload,
                )
            except BaseException as exc:
                raise _RewriteClientAbort(exc) from exc

        model_endpoint_block["rewrite"] = active_ollama_base_url
        if any_s_needed and not s_code_say:
            _check_partial_offload_or_stop("rewrite")
            if ollama_writer is not None and not ollama_writer._verified_local_digest:
                ollama_writer._ensure_verified()
        if not any_s_needed and prev_rewrite_timing_complete:
            rewrite_mono_start = None
        else:
            step_timings_block["rewrite"]["start"] = _read_utc()
            step_timings_block["rewrite"]["end"] = None
            rewrite_mono_start = _read_mono()
        s_done_count = 0
        try:
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

                if resume and r_key in s_completed_keys:
                    s_done_count += 1
                    record["progress"]["s_done"] = s_done_count
                    continue

                b_rows = b_rows_by_key[r_key]
                swap_res = swap_res_by_key[r_key]

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
                    try:
                        rewrite_res = rewrite_all_say(
                            swap_res.prose,
                            w,
                            scen_plan_by_idx,
                            _rewrite_client_proxy,
                            input_mode=chosen_input,
                            focal_agent=focal_ag,
                            goal_mode=resolved_goal_mode,
                            checker_profile="d",
                            base_url=active_ollama_base_url,
                            model=effective_writer_pin.ollama_tag,
                        )
                    except _RewriteClientAbort as abort_exc:
                        raise abort_exc.original from None
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

                # Seal and flush S prose per rendering beside B in the sealed store BEFORE updating the record
                sealed_call_counter += 1
                s_sha, s_think_sha = compute_sealed_response_digests(s_prose, "")
                _append_and_flush_sealed_response(
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
                    step_timings_block["rewrite"]["end"] = _read_utc()
                    step_timings_block["rewrite"]["seconds"] = round(
                        max(0.0, _read_mono() - rewrite_mono_start), 6
                    )
                    step_timings_block["rewrite"]["calls"] = rewrite_calls_made
                    step_timings_block["rewrite"]["call_count"] = rewrite_calls_made
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
        finally:
            if rewrite_mono_start is not None and step_timings_block["rewrite"]["end"] is None:
                step_timings_block["rewrite"]["end"] = _read_utc()
                step_timings_block["rewrite"]["seconds"] = round(
                    max(0.0, _read_mono() - rewrite_mono_start), 6
                )
            step_timings_block["rewrite"]["calls"] = rewrite_calls_made
            step_timings_block["rewrite"]["call_count"] = rewrite_calls_made

        record["stop_reason"] = None
        record.pop("stop_detail", None)
        record["summary"] = _recompute_summary(renderings_records)
        _write_record_and_status(record_path, status_path, record, status_state="completed")
        return record
    finally:
        if ollama_writer is not None:
            raw_transport = getattr(ollama_writer._transport, "_raw", ollama_writer._transport)
            ollama_writer.close()
            if guard_both_endpoints:
                default_end_status = unload_and_verify_default_endpoint_empty(
                    raw_transport,
                    default_base_url=DEFAULT_OLLAMA_BASE_URL,
                    model_tag=effective_writer_pin.ollama_tag,
                    own_model_tags=tuple(run_own_model_tags),
                    model_calls_by_endpoint=model_calls_by_endpoint,
                )
                record["default_endpoint_end"] = default_end_status
                if record_path.is_file():
                    _atomic_write_json(record_path, record)
