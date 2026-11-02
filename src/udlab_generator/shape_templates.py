"""Stage 5 deterministic code templates, scene-checked swap passes, and Option 3 control-arm renderer.

Implements Card U2 (`DESIGN-stage5-REV2-2026-11-02.md` §1, §2, §5, §7, §9):
- `render_event_sentence(ev, world, form_id, *, focal_agent) -> str`:
  Renders a single logged `Event` into its accepted Stage 4/5 sentence shape:
  - `move`: `MOVE_0` (station transitions only), `MOVE_1`, `MOVE_2` (fronted path), `MOVE_3` (passive), `MOVE_4` (P1 phrase);
  - `see`: `SEE_1`, `SEE_2` (unoccluded); `SEE_3`, `SEE_4`, `SEE_5` (occluded);
  - `say`: `SAY_1`, `SAY_2`, `SAY_3`, `SAY_4` (S-code fallback and control arm; `SAY_5` is out of the pool).
  Reuses `_render_event_sentence` imported from `udlab_generator.render` (`src/udlab_generator/render.py:7398-7472`,
  called by `render_world_template` at `src/udlab_generator/render.py:7566-7569`) for `MOVE_0`, `MOVE_1`, `SEE_1`,
  `SEE_4`, and `SAY_3` without copying.
- `swap_all_move_see(prose, world, plan, *, focal_agent, goal_mode=DEFAULT_GOAL_MODE, checker_profile="d") -> SwapResult`:
  Replaces every `move` and `see` sentence at its matched `sentence_index` in event order, re-checking the whole scene
  after each splice under profile `d` and rolling back any splice that grows the fault multiset, adds a new `rule_id`,
  alters a neighbour row's `(realized_form_id, sentence_index)`, changes the event-sentence count, or fails to realize
  the planned form on the target row.
- `swap_all_say(prose, world, plan, *, focal_agent, goal_mode=DEFAULT_GOAL_MODE, checker_profile="d") -> SwapResult`:
  Applies the same one-at-a-time scene-checked swap rule to `say` events using the `SAY_1`–`SAY_4` templates.
- `render_control_scene(world, focal_agent, plan) -> str`:
  Option 3 control arm: renders `render_world_template(world, focal_agent)`, then runs `swap_all_move_see` with `plan`,
  then runs `swap_all_say` with `plan`.

Seam-Flag Rule (Disclosed Count, Never a Gate):
For each kept swap (`action == "swapped"` at `sentence_index`), inspect the next event sentence in the scene
(`event_sentences[sentence_index + 1]`, if `sentence_index + 1 < len(event_sentences)`). Flag `seam_next_opens_ref = True`
(alias `seam_flag = True`) if and only if that next event sentence opens with either:
1. A pronoun from the fixed listed set `SEAM_OPENING_PRONOUNS`:
   `("he", "she", "they", "him", "her", "them", "his", "hers", "their", "theirs",
     "it", "its", "this", "that", "these", "those", "himself", "herself", "themselves")`
   matched case-insensitively on the first word token of the sentence; or
2. A participial phrase under the fixed rule: the first word token of the sentence (`[a-zA-Z]+`, length >= 4) ends
   case-insensitively in `-ing` or `-ed` and is followed by a comma `,` within the first 6 words of the sentence
   (`1 <= len(words_before_first_comma) <= 6`, with no sentence-ending punctuation before that comma).
If there is no next event sentence or neither opening pattern matches, `seam_next_opens_ref` is `False`. For events
that were not swapped (`action != "swapped"`), `seam_next_opens_ref` is `None`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import random
import re
from typing import Any

from udlab_generator.render import (
    DEFAULT_GOAL_MODE,
    ProseFaultRecord,
    RenderCheckError,
    SHAPE_PLAN_BLOCK_SIZE,
    VALID_CHECKER_PROFILES,
    VALID_GOAL_MODES,
    _GOAL_TOKEN_WITH_OPTIONAL_PUNCT_RE,
    _render_event_sentence,
    _split_prose_sentences,
    check_prose_render_fidelity,
    check_rendered_prose,
    collect_prose_faults,
    event_shape_plan_variant,
    match_rendering_event_forms,
    normalize_prose_line_endings,
    render_world_template,
    splice_goal_tokens,
    wildcard_form_id_for_event,
)
from udlab_generator.world import (
    CONTAINER_BAY_STATION,
    FAR_SIDE_ALCOVE_STATION,
    Event,
    World,
    is_station_transition_event,
)

STATION_MOVE_FORMS: tuple[str, ...] = ("MOVE_0",)
MOVE_FORMS: tuple[str, ...] = ("MOVE_1", "MOVE_2", "MOVE_3", "MOVE_4")
SEE_UNOCCLUDED_FORMS: tuple[str, ...] = ("SEE_1", "SEE_2")
SEE_OCCLUDED_FORMS: tuple[str, ...] = ("SEE_3", "SEE_4", "SEE_5")
SAY_FORMS: tuple[str, ...] = ("SAY_1", "SAY_2", "SAY_3", "SAY_4")

ALL_TEMPLATE_FORMS: tuple[str, ...] = (
    *STATION_MOVE_FORMS,
    *MOVE_FORMS,
    *SEE_UNOCCLUDED_FORMS,
    *SEE_OCCLUDED_FORMS,
    *SAY_FORMS,
)

STAGE5_VARIANT_FORMS: dict[str, tuple[str, ...]] = {
    "move_station_change": STATION_MOVE_FORMS,
    "move_normal": MOVE_FORMS,
    "move_occluded": MOVE_FORMS,
    "see_unoccluded": SEE_UNOCCLUDED_FORMS,
    "see_occluded": SEE_OCCLUDED_FORMS,
    "say": SAY_FORMS,
}

SEAM_OPENING_PRONOUNS: tuple[str, ...] = (
    "he",
    "she",
    "they",
    "him",
    "her",
    "them",
    "his",
    "hers",
    "their",
    "theirs",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "himself",
    "herself",
    "themselves",
)
_SEAM_OPENING_PRONOUN_SET: frozenset[str] = frozenset(SEAM_OPENING_PRONOUNS)
_FIRST_WORD_RE = re.compile(r"^\s*[\"“'‘(]*([a-zA-Z]+)\b")
_WORD_TOKENS_RE = re.compile(r"[a-zA-Z0-9']+")

VALID_SWAP_ACTIONS: frozenset[str] = frozenset(
    {"swapped", "swap_rolled_back", "kept", "not_spliced"}
)
VALID_ROLLBACK_CAUSES: frozenset[str] = frozenset(
    {"fault_grew", "neighbour_changed", "count_changed", "wrong_form"}
)

MOVE_4_P1_PHRASE: str = "working quickly"
SEE_2_P1_PHRASE: str = "standing quietly"


def applicable_forms_for_event(ev: Event, world: World | None = None) -> tuple[str, ...]:
    """Return the tuple of accepted Stage 5 template `form_id`s applicable to `ev`."""
    if ev.kind == "move":
        if is_station_transition_event(ev, world):
            return STATION_MOVE_FORMS
        return MOVE_FORMS
    if ev.kind == "see":
        if ev.occluded and ev.occluder:
            return SEE_OCCLUDED_FORMS
        return SEE_UNOCCLUDED_FORMS
    if ev.kind == "say":
        return SAY_FORMS
    return ()


def _see_location_clause(ev: Event) -> str:
    """Build the unoccluded `see` location clause matching `_render_event_sentence` (`render.py:7443-7450`)."""
    if (
        ev.to_location
        and ev.to_location not in (ev.room, CONTAINER_BAY_STATION, FAR_SIDE_ALCOVE_STATION)
    ):
        return f" at the {ev.to_location}"
    if ev.to_location in (CONTAINER_BAY_STATION, FAR_SIDE_ALCOVE_STATION):
        return f" in the {ev.to_location}"
    return ""


def _move_occlusion_suffix(ev: Event) -> str:
    """Build the occlusion clause for `MOVE_2`, `MOVE_3`, and `MOVE_4` when `ev.occluded and ev.occluder`."""
    if ev.occluded and ev.occluder:
        return f", the {ev.occluder} blocking sight"
    return ""


def _utterance_with_terminal_punct(utterance: str | None) -> str:
    """Return `utterance` guaranteed to end with `.`, `!`, or `?` for sentence-final quotes (`SAY_1`)."""
    raw = (utterance or "").strip()
    if not raw:
        return "."
    if raw[-1] in (".", "!", "?"):
        return raw
    return f"{raw}."


def _utterance_body_without_terminal_period(utterance: str | None) -> str:
    """Return `utterance` with at most one trailing `.`, `!`, or `?` stripped for pre-comma quotes (`SAY_2`, `SAY_4`)."""
    raw = (utterance or "").strip()
    if raw and raw[-1] in (".", "!", "?", ","):
        return raw[:-1].rstrip()
    return raw


def render_event_sentence(
    ev: Event,
    world: World,
    form_id: str,
    *,
    focal_agent: str,
) -> str:
    """Render `ev` into the single-sentence template for `form_id` focalized on `focal_agent`.

    Reuses `_render_event_sentence` imported from `udlab_generator.render` (`src/udlab_generator/render.py:7398-7472`,
    called by `render_world_template` at `src/udlab_generator/render.py:7566-7569`) for the five shapes it already
    writes (`MOVE_0`, `MOVE_1`, `SEE_1`, `SEE_4`, `SAY_3`).

    Matcher acceptance citations in `src/udlab_generator/render.py` for the nine new templates:
    - `MOVE_2`: `_matches_m2_fronted_path` (`src/udlab_generator/render.py:4218-4255`) and `matched_event_form_id`
      (`src/udlab_generator/render.py:4824-4842`).
    - `MOVE_3`: `_sentence_matches_passive_move_by_agent` (`src/udlab_generator/render.py:4258-4331`) and
      `matched_event_form_id` (`src/udlab_generator/render.py:4608-4613`, `src/udlab_generator/render.py:4788-4789`).
    - `MOVE_4`: `_actor_to_verb_matches_p1_phrase` (`src/udlab_generator/render.py:3404-3493`),
      `_find_actor_subject_verb_matches` (`src/udlab_generator/render.py:3590-3594`), and `matched_event_form_id`
      (`src/udlab_generator/render.py:4839-4843`).
    - `SEE_2`: `_actor_to_verb_matches_p1_phrase` (`src/udlab_generator/render.py:3404-3493`),
      `_find_actor_subject_verb_matches` (`src/udlab_generator/render.py:3590-3594`), and `matched_event_form_id`
      (`src/udlab_generator/render.py:4913-4940`).
    - `SEE_3`: `_sentence_states_occluder_blocked_actor_sight` (`src/udlab_generator/render.py:4158-4199`) and
      `matched_event_form_id` (`src/udlab_generator/render.py:4602-4607`, `src/udlab_generator/render.py:4870-4871`).
    - `SEE_5`: `_sentence_matches_v2_occluded_see_hid` (`src/udlab_generator/render.py:4334-4398`) and
      `matched_event_form_id` (`src/udlab_generator/render.py:4614-4619`, `src/udlab_generator/render.py:4866-4867`).
    - `SAY_1`: `matched_event_form_id` (`src/udlab_generator/render.py:4958-5009`, specifically `pre_quote_ok` at
      `src/udlab_generator/render.py:4982-4988` and return `"SAY_1"` at `src/udlab_generator/render.py:5009`).
    - `SAY_2`: `matched_event_form_id` (`src/udlab_generator/render.py:4958-5003`, specifically `post_quote_ok` at
      `src/udlab_generator/render.py:4989-4996` and return `"SAY_2"` at `src/udlab_generator/render.py:5002-5003`).
    - `SAY_4`: `matched_event_form_id` (`src/udlab_generator/render.py:4958-5008`, specifically `pre_quote_ok` at
      `src/udlab_generator/render.py:4982-4988` and post-quote room check at `src/udlab_generator/render.py:5006-5008`).
    """
    if focal_agent not in world.agents:
        raise ValueError(f"focal_agent '{focal_agent}' must be one of {world.agents}")
    if form_id == "SAY_5":
        raise ValueError("SAY_5 is out of the Stage 5 pool and cannot be rendered by render_event_sentence.")
    valid_forms = applicable_forms_for_event(ev, world)
    if not valid_forms:
        raise ValueError(f"Event kind {ev.kind!r} does not have Stage 5 shape templates.")
    if form_id not in valid_forms:
        raise ValueError(
            f"form_id {form_id!r} is not applicable to event (kind={ev.kind!r}, "
            f"variant={event_shape_plan_variant(ev, world)!r}); valid forms are {valid_forms}."
        )

    # Reuse `render_world_template`'s per-event renderer (`_render_event_sentence`, render.py:7398-7472)
    # for the shapes it already writes: MOVE_0, MOVE_1, SEE_1, SEE_4, SAY_3.
    if form_id in ("MOVE_0", "MOVE_1", "SEE_1", "SEE_4", "SAY_3"):
        return _render_event_sentence(ev, focal_agent, world.noise_threshold)

    if ev.kind == "move":
        occ_suffix = _move_occlusion_suffix(ev)
        if form_id == "MOVE_2":
            # Accepted by `_matches_m2_fronted_path` (src/udlab_generator/render.py:4218-4255)
            # and `matched_event_form_id` (src/udlab_generator/render.py:4824-4842).
            return (
                f"From the {ev.from_location} to the {ev.to_location}, "
                f"{ev.actor} moved the {ev.target_object} in the {ev.room}{occ_suffix}."
            )
        if form_id == "MOVE_3":
            # Accepted by `_sentence_matches_passive_move_by_agent` (src/udlab_generator/render.py:4258-4331)
            # and `matched_event_form_id` (src/udlab_generator/render.py:4608-4613, 4788-4789).
            return (
                f"In the {ev.room}, the {ev.target_object} was moved "
                f"from the {ev.from_location} to the {ev.to_location} by {ev.actor}{occ_suffix}."
            )
        if form_id == "MOVE_4":
            # Accepted by `_actor_to_verb_matches_p1_phrase` (src/udlab_generator/render.py:3404-3493)
            # and `matched_event_form_id` (src/udlab_generator/render.py:4839-4843).
            return (
                f"In the {ev.room}, {ev.actor}, {MOVE_4_P1_PHRASE}, moved the {ev.target_object} "
                f"from the {ev.from_location} to the {ev.to_location}{occ_suffix}."
            )

    if ev.kind == "see":
        if form_id == "SEE_2":
            # Accepted by `_actor_to_verb_matches_p1_phrase` (src/udlab_generator/render.py:3404-3493)
            # and `matched_event_form_id` (src/udlab_generator/render.py:4913-4940).
            loc_clause = _see_location_clause(ev)
            return (
                f"In the {ev.room}, {ev.actor}, {SEE_2_P1_PHRASE}, observed the {ev.target_object}"
                f"{loc_clause} in open view."
            )
        if form_id == "SEE_3":
            # Accepted by `_sentence_states_occluder_blocked_actor_sight` (src/udlab_generator/render.py:4158-4199)
            # and `matched_event_form_id` (src/udlab_generator/render.py:4602-4607, 4870-4871).
            return (
                f"In the {ev.room}, the {ev.occluder} blocked {ev.actor}'s sight "
                f"of the {ev.target_object} from the {FAR_SIDE_ALCOVE_STATION}."
            )
        if form_id == "SEE_5":
            # Accepted by `_sentence_matches_v2_occluded_see_hid` (src/udlab_generator/render.py:4334-4398)
            # and `matched_event_form_id` (src/udlab_generator/render.py:4614-4619, 4866-4867).
            return (
                f"In the {ev.room}, the {ev.occluder} hid the {ev.target_object} "
                f"from {ev.actor} at the {FAR_SIDE_ALCOVE_STATION}."
            )

    if ev.kind == "say":
        if form_id == "SAY_1":
            # Accepted by `matched_event_form_id` (src/udlab_generator/render.py:4958-5009,
            # `pre_quote_ok` at :4982-4988 and return "SAY_1" at :5009).
            utt_final = _utterance_with_terminal_punct(ev.utterance)
            return f'{ev.actor} said at {ev.noise_level} dB in the {ev.room}, "{utt_final}"'
        if form_id == "SAY_2":
            # Accepted by `matched_event_form_id` (src/udlab_generator/render.py:4958-5003,
            # `post_quote_ok` at :4989-4996 and return "SAY_2" at :5002-5003).
            utt_body = _utterance_body_without_terminal_period(ev.utterance)
            return f'"{utt_body}," {ev.actor} said at {ev.noise_level} dB in the {ev.room}.'
        if form_id == "SAY_4":
            # Accepted by `matched_event_form_id` (src/udlab_generator/render.py:4958-5008,
            # `pre_quote_ok` at :4982-4988 and post-quote room check at :5006-5008).
            utt_body = _utterance_body_without_terminal_period(ev.utterance)
            return f'{ev.actor} said at {ev.noise_level} dB, "{utt_body}," in the {ev.room}.'

    raise ValueError(f"Unsupported form_id {form_id!r} for event kind {ev.kind!r}.")


def sentence_opens_with_seam_reference(sentence: str) -> bool:
    """Return True if `sentence` opens with a pronoun in `SEAM_OPENING_PRONOUNS` or a participial phrase.

    Participial-phrase rule: the first word token (`[a-zA-Z]+`, length >= 4) ends with `-ing` or `-ed`
    (case-insensitive) and is followed by a comma `,` within the first 6 words of the sentence.
    """
    stripped = sentence.strip()
    if not stripped:
        return False
    m_first = _FIRST_WORD_RE.match(stripped)
    if m_first is None:
        return False
    first_word = m_first.group(1).lower()
    if first_word in _SEAM_OPENING_PRONOUN_SET:
        return True
    if len(first_word) >= 4 and (first_word.endswith("ing") or first_word.endswith("ed")):
        comma_pos = stripped.find(",")
        if comma_pos > 0:
            before_comma = stripped[:comma_pos]
            if not any(p in before_comma for p in (".", "!", "?", ";", ":")):
                words_before_comma = _WORD_TOKENS_RE.findall(before_comma)
                if 1 <= len(words_before_comma) <= 6:
                    return True
    return False


@dataclass(frozen=True)
class SwapEventRecord:
    """Per-event record produced by `swap_all_move_see` or `swap_all_say`."""

    event_index: int
    kind: str
    variant: str
    plan_form: str | None
    b_form: str | None
    final_form: str | None
    sentence_index: int | None
    action: str
    cause: str | None = None
    rule_ids: tuple[str, ...] = ()
    plan_matched_writer: bool = False
    seam_next_opens_ref: bool | None = None

    def __post_init__(self) -> None:
        if self.action not in VALID_SWAP_ACTIONS:
            raise ValueError(
                f"SwapEventRecord.action must be one of {sorted(VALID_SWAP_ACTIONS)}, got {self.action!r}."
            )
        if self.cause is not None and self.cause not in VALID_ROLLBACK_CAUSES:
            raise ValueError(
                f"SwapEventRecord.cause must be one of {sorted(VALID_ROLLBACK_CAUSES)} or None, got {self.cause!r}."
            )

    @property
    def plan(self) -> str | None:
        return self.plan_form

    @property
    def s_form(self) -> str | None:
        return self.final_form

    @property
    def rollback_cause(self) -> str | None:
        return self.cause

    @property
    def seam_flag(self) -> bool | None:
        return self.seam_next_opens_ref

    def to_dict(self) -> dict[str, object]:
        return {
            "event_index": self.event_index,
            "kind": self.kind,
            "variant": self.variant,
            "plan": self.plan_form,
            "plan_form": self.plan_form,
            "b_form": self.b_form,
            "final_form": self.final_form,
            "s_form": self.final_form,
            "sentence_index": self.sentence_index,
            "action": self.action,
            "cause": self.cause,
            "rollback_cause": self.cause,
            "rule_ids": list(self.rule_ids),
            "plan_matched_writer": self.plan_matched_writer,
            "seam_flag": self.seam_next_opens_ref,
            "seam_next_opens_ref": self.seam_next_opens_ref,
        }

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def get(self, key: str, default: object = None) -> object:
        return self.to_dict().get(key, default)


@dataclass(frozen=True)
class SwapResult:
    """Whole-scene swap outcome returned by `swap_all_move_see` and `swap_all_say`."""

    prose: str
    events: tuple[SwapEventRecord, ...]
    b_faults: tuple[ProseFaultRecord, ...]
    s_faults: tuple[ProseFaultRecord, ...]
    b_passed: bool
    s_passed: bool
    seam_count: int

    @property
    def event_records(self) -> tuple[SwapEventRecord, ...]:
        return self.events

    @property
    def final_faults(self) -> tuple[ProseFaultRecord, ...]:
        return self.s_faults

    @property
    def b_fault_rule_ids(self) -> tuple[str, ...]:
        return tuple(f.rule_id for f in self.b_faults)

    @property
    def s_fault_rule_ids(self) -> tuple[str, ...]:
        return tuple(f.rule_id for f in self.s_faults)

    def to_dict(self) -> dict[str, object]:
        return {
            "prose": self.prose,
            "events": [e.to_dict() for e in self.events],
            "b_faults": [f.to_dict(include_rule_id=True) for f in self.b_faults],
            "s_faults": [f.to_dict(include_rule_id=True) for f in self.s_faults],
            "b_fault_rule_ids": list(self.b_fault_rule_ids),
            "s_fault_rule_ids": list(self.s_fault_rule_ids),
            "b_passed": self.b_passed,
            "s_passed": self.s_passed,
            "seam_count": self.seam_count,
        }

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def get(self, key: str, default: object = None) -> object:
        return self.to_dict().get(key, default)


def _extract_event_sentences(prose: str, world: World, goal_mode: str) -> list[str]:
    """Extract the event-paragraph sentence list using the exact splitter path as `match_rendering_event_forms`."""
    checked_prose = prose
    if goal_mode == "splice":
        try:
            checked_prose = splice_goal_tokens(prose, world).expanded_prose
        except RenderCheckError:
            checked_prose = _GOAL_TOKEN_WITH_OPTIONAL_PUNCT_RE.sub("", prose)
    norm_prose = normalize_prose_line_endings(checked_prose)
    paragraphs = [p.strip() for p in norm_prose.split("\n\n") if p.strip()]
    event_sentences: list[str] = []
    if len(paragraphs) >= 2:
        for ev_para in paragraphs[1:]:
            event_sentences.extend(_split_prose_sentences(ev_para))
    return event_sentences


def splice_event_sentence_at_index(
    prose: str,
    sentence_index: int,
    new_sentence: str,
) -> str:
    """Replace the event sentence at 0-based `sentence_index` in `prose`, preserving setup and paragraph structure."""
    norm_prose = normalize_prose_line_endings(prose)
    paragraphs = [p.strip() for p in norm_prose.split("\n\n") if p.strip()]
    if len(paragraphs) < 2:
        raise ValueError("Cannot splice event sentence in prose with fewer than 2 paragraphs.")
    setup_para = paragraphs[0]
    new_event_paras: list[str] = []
    running_idx = 0
    replaced = False
    for ev_para in paragraphs[1:]:
        para_sents = _split_prose_sentences(ev_para)
        n_para = len(para_sents)
        if not replaced and running_idx <= sentence_index < running_idx + n_para:
            local_idx = sentence_index - running_idx
            updated_sents = list(para_sents)
            updated_sents[local_idx] = new_sentence
            new_event_paras.append(" ".join(updated_sents))
            replaced = True
        else:
            new_event_paras.append(ev_para)
        running_idx += n_para
    if not replaced:
        raise IndexError(
            f"sentence_index {sentence_index} out of range for event sentences (count={running_idx})."
        )
    return "\n\n".join([setup_para, *new_event_paras])


def _resolve_plan_by_event_index(world: World, plan: Any) -> dict[int, str]:
    """Normalize `plan` into a 1-based `{event_index: form_id}` dictionary for `world`."""
    if plan is None:
        return {}
    if isinstance(plan, Mapping):
        if "plan_by_scenario_id" in plan and isinstance(plan["plan_by_scenario_id"], Mapping):
            scen_plan = plan["plan_by_scenario_id"].get(world.scenario_id)
            if scen_plan is not None:
                return _resolve_plan_by_event_index(world, scen_plan)
        if "plan_records" in plan and isinstance(plan["plan_records"], Sequence):
            recs = [
                r
                for r in plan["plan_records"]
                if isinstance(r, Mapping) and str(r.get("scenario_id")) == world.scenario_id
            ]
            return _resolve_plan_by_event_index(world, recs)
        if world.scenario_id in plan and isinstance(plan[world.scenario_id], (Mapping, Sequence)):
            return _resolve_plan_by_event_index(world, plan[world.scenario_id])
        out: dict[int, str] = {}
        for k, v in plan.items():
            if v is None:
                continue
            idx = int(k)
            if isinstance(v, Mapping) and "form_id" in v:
                out[idx] = str(v["form_id"])
            else:
                out[idx] = str(v)
        return out
    if isinstance(plan, Sequence) and not isinstance(plan, (str, bytes)):
        out_seq: dict[int, str] = {}
        if all(isinstance(item, Mapping) for item in plan):
            for pos_idx, item in enumerate(plan, start=1):
                assert isinstance(item, Mapping)
                if "scenario_id" in item and str(item["scenario_id"]) != world.scenario_id:
                    continue
                ev_idx = int(item.get("event_index", pos_idx))
                if item.get("form_id") is not None:
                    out_seq[ev_idx] = str(item["form_id"])
            return out_seq
        if len(plan) != len(world.events):
            raise ValueError(
                f"Sequence plan length ({len(plan)}) must match world.events length ({len(world.events)})."
            )
        for ev_idx, item in enumerate(plan, start=1):
            if item is None:
                continue
            if isinstance(item, Mapping) and "form_id" in item:
                out_seq[ev_idx] = str(item["form_id"])
            else:
                out_seq[ev_idx] = str(item)
        return out_seq
    raise TypeError(f"Unsupported plan type: {type(plan)!r}")


def _fault_multiset(faults: Sequence[ProseFaultRecord]) -> Counter[tuple[str, int | None]]:
    """Build the `(rule_id, event_index)` multiset for `faults`."""
    return Counter((f.rule_id, f.event_index) for f in faults)


def _fault_list_grew(
    before_faults: Sequence[ProseFaultRecord],
    after_faults: Sequence[ProseFaultRecord],
) -> tuple[bool, tuple[str, ...]]:
    """Return `(grew, added_rule_ids)` comparing `after_faults` against `before_faults` as a multiset of `(rule_id, event_index)`."""
    if not before_faults and after_faults:
        return True, tuple(dict.fromkeys(f.rule_id for f in after_faults))
    before_ms = _fault_multiset(before_faults)
    after_ms = _fault_multiset(after_faults)
    before_rules = {f.rule_id for f in before_faults}
    after_rules = {f.rule_id for f in after_faults}
    added_ms = after_ms - before_ms
    new_rules = after_rules - before_rules
    if added_ms or new_rules or len(after_faults) > len(before_faults):
        added_ids: list[str] = []
        for f in after_faults:
            key = (f.rule_id, f.event_index)
            if added_ms.get(key, 0) > 0 or f.rule_id in new_rules:
                if f.rule_id not in added_ids:
                    added_ids.append(f.rule_id)
        if not added_ids:
            added_ids = list(dict.fromkeys(f.rule_id for f in after_faults))
        return True, tuple(added_ids)
    return False, ()


def _run_swap_pass(
    prose: str,
    world: World,
    plan: Any,
    *,
    target_kinds: frozenset[str],
    focal_agent: str,
    goal_mode: str = DEFAULT_GOAL_MODE,
    checker_profile: str = "d",
    template_fn: Callable[..., str] | None = None,
) -> SwapResult:
    """Shared implementation for `swap_all_move_see` and `swap_all_say`."""
    if focal_agent not in world.agents:
        raise ValueError(f"focal_agent '{focal_agent}' must be one of {world.agents}")
    if goal_mode not in VALID_GOAL_MODES:
        raise ValueError(f"goal_mode must be one of {sorted(VALID_GOAL_MODES)}, got {goal_mode!r}")
    if checker_profile not in VALID_CHECKER_PROFILES:
        raise ValueError(
            f"checker_profile must be one of {sorted(VALID_CHECKER_PROFILES)}, got {checker_profile!r}"
        )

    render_fn = template_fn if template_fn is not None else render_event_sentence
    plan_by_idx = _resolve_plan_by_event_index(world, plan)

    b_faults = collect_prose_faults(
        prose,
        world,
        focal_agent=focal_agent,
        goal_mode=goal_mode,
        checker_profile=checker_profile,
    )
    b_rows, b_sent_count = match_rendering_event_forms(
        prose,
        world,
        focal_agent=focal_agent,
        goal_mode=goal_mode,
        checker_profile=checker_profile,
        return_sentence_count=True,
    )

    current_prose = prose
    current_faults = b_faults
    current_rows = b_rows
    current_sent_count = b_sent_count

    per_event_action: dict[int, str] = {}
    per_event_cause: dict[int, str | None] = {}
    per_event_rule_ids: dict[int, tuple[str, ...]] = {}
    per_event_plan_form: dict[int, str | None] = {}
    per_event_b_form: dict[int, str | None] = {}
    per_event_matched_writer: dict[int, bool] = {}

    for ev_pos, ev in enumerate(world.events):
        ev_idx = ev_pos + 1
        b_row = b_rows[ev_pos]
        b_form = str(b_row["realized_form_id"]) if b_row.get("realized_form_id") is not None else None
        per_event_b_form[ev_idx] = b_form

        if ev.kind not in target_kinds:
            raw_plan = plan_by_idx.get(ev_idx)
            per_event_plan_form[ev_idx] = raw_plan
            per_event_matched_writer[ev_idx] = bool(
                b_form is not None and raw_plan is not None and b_form == raw_plan
            )
            per_event_action[ev_idx] = "kept"
            per_event_cause[ev_idx] = None
            per_event_rule_ids[ev_idx] = ()
            continue

        is_station = ev.kind == "move" and is_station_transition_event(ev, world)
        if is_station:
            planned_form: str | None = plan_by_idx.get(ev_idx, "MOVE_0")
            if planned_form == "MOVE_0" or planned_form is None:
                planned_form = "MOVE_0"
        else:
            planned_form = plan_by_idx.get(ev_idx)

        per_event_plan_form[ev_idx] = planned_form
        matched_writer = bool(
            b_form is not None and planned_form is not None and b_form == planned_form
        )
        per_event_matched_writer[ev_idx] = matched_writer

        cur_row = current_rows[ev_pos]
        s_idx_obj = cur_row.get("sentence_index")
        if s_idx_obj is None:
            per_event_action[ev_idx] = "not_spliced"
            per_event_cause[ev_idx] = None
            row_rule = str(cur_row["rule_id"]) if cur_row.get("rule_id") is not None else None
            per_event_rule_ids[ev_idx] = (row_rule,) if row_rule else ()
            continue

        if planned_form is None:
            per_event_action[ev_idx] = "kept"
            per_event_cause[ev_idx] = None
            per_event_rule_ids[ev_idx] = ()
            continue

        s_idx = int(s_idx_obj)  # type: ignore[arg-type]
        template_sentence = render_fn(ev, world, planned_form, focal_agent=focal_agent)
        candidate_prose = splice_event_sentence_at_index(current_prose, s_idx, template_sentence)

        if candidate_prose == current_prose:
            cand_faults = current_faults
            cand_rows = current_rows
            cand_sent_count = current_sent_count
        else:
            cand_faults = collect_prose_faults(
                candidate_prose,
                world,
                focal_agent=focal_agent,
                goal_mode=goal_mode,
                checker_profile=checker_profile,
            )
            cand_rows, cand_sent_count = match_rendering_event_forms(
                candidate_prose,
                world,
                focal_agent=focal_agent,
                goal_mode=goal_mode,
                checker_profile=checker_profile,
                return_sentence_count=True,
            )

        # Evaluate the four §5 splice conditions:
        # (iii) sentence count unchanged
        count_changed = cand_sent_count != current_sent_count
        # (i) scene fault list does not grow (multiset of (rule_id, event_index) and rule_id set)
        fault_grew, added_fault_rules = _fault_list_grew(current_faults, cand_faults)
        # (ii) every other row's realized_form_id and sentence_index are unchanged
        neighbour_changed = False
        changed_neighbour_rules: list[str] = []
        for other_pos in range(len(world.events)):
            if other_pos == ev_pos:
                continue
            prev_r = current_rows[other_pos]
            next_r = cand_rows[other_pos]
            if (
                prev_r.get("realized_form_id") != next_r.get("realized_form_id")
                or prev_r.get("sentence_index") != next_r.get("sentence_index")
            ):
                neighbour_changed = True
                nr_rule = next_r.get("rule_id")
                if isinstance(nr_rule, str) and nr_rule and nr_rule not in changed_neighbour_rules:
                    changed_neighbour_rules.append(nr_rule)
        # (iv) target row reports the planned form
        cand_target_row = cand_rows[ev_pos]
        wrong_form = cand_target_row.get("realized_form_id") != planned_form

        if not (count_changed or fault_grew or neighbour_changed or wrong_form):
            current_prose = candidate_prose
            current_faults = cand_faults
            current_rows = cand_rows
            current_sent_count = cand_sent_count
            per_event_action[ev_idx] = "swapped"
            per_event_cause[ev_idx] = None
            per_event_rule_ids[ev_idx] = ()
        else:
            # Roll back: restore `current_prose` untouched and record `swap_rolled_back` with cause and rule_ids.
            if count_changed:
                cause = "count_changed"
            elif fault_grew:
                cause = "fault_grew"
            elif neighbour_changed:
                cause = "neighbour_changed"
            else:
                cause = "wrong_form"

            involved_rules: list[str] = []
            for r_id in added_fault_rules:
                if r_id not in involved_rules:
                    involved_rules.append(r_id)
            target_rule = cand_target_row.get("rule_id")
            if isinstance(target_rule, str) and target_rule and target_rule not in involved_rules:
                involved_rules.append(target_rule)
            for r_id in changed_neighbour_rules:
                if r_id not in involved_rules:
                    involved_rules.append(r_id)
            if not involved_rules and cand_faults:
                for f in cand_faults:
                    if f.rule_id not in involved_rules:
                        involved_rules.append(f.rule_id)

            per_event_action[ev_idx] = "swap_rolled_back"
            per_event_cause[ev_idx] = cause
            per_event_rule_ids[ev_idx] = tuple(involved_rules)

    final_event_sentences = _extract_event_sentences(current_prose, world, goal_mode)
    records: list[SwapEventRecord] = []
    seam_count = 0

    for ev_pos, ev in enumerate(world.events):
        ev_idx = ev_pos + 1
        fin_row = current_rows[ev_pos]
        fin_form = (
            str(fin_row["realized_form_id"])
            if fin_row.get("realized_form_id") is not None
            else None
        )
        fin_s_idx = (
            int(fin_row["sentence_index"])  # type: ignore[arg-type]
            if fin_row.get("sentence_index") is not None
            else None
        )
        act = per_event_action[ev_idx]
        seam_flag: bool | None = None
        if act == "swapped":
            if fin_s_idx is not None and fin_s_idx + 1 < len(final_event_sentences):
                seam_flag = sentence_opens_with_seam_reference(final_event_sentences[fin_s_idx + 1])
            else:
                seam_flag = False
            if seam_flag:
                seam_count += 1

        records.append(
            SwapEventRecord(
                event_index=ev_idx,
                kind=ev.kind,
                variant=event_shape_plan_variant(ev, world),
                plan_form=per_event_plan_form[ev_idx],
                b_form=per_event_b_form[ev_idx],
                final_form=fin_form,
                sentence_index=fin_s_idx,
                action=act,
                cause=per_event_cause[ev_idx],
                rule_ids=per_event_rule_ids[ev_idx],
                plan_matched_writer=per_event_matched_writer[ev_idx],
                seam_next_opens_ref=seam_flag,
            )
        )

    return SwapResult(
        prose=current_prose,
        events=tuple(records),
        b_faults=b_faults,
        s_faults=current_faults,
        b_passed=len(b_faults) == 0,
        s_passed=len(current_faults) == 0,
        seam_count=seam_count,
    )


def swap_all_move_see(
    prose: str,
    world: World,
    plan: Any,
    *,
    focal_agent: str,
    goal_mode: str = DEFAULT_GOAL_MODE,
    checker_profile: str = "d",
    template_fn: Callable[..., str] | None = None,
) -> SwapResult:
    """Replace every `move` and `see` sentence at `sentence_index` with its planned template under the §5 scene re-check."""
    return _run_swap_pass(
        prose,
        world,
        plan,
        target_kinds=frozenset({"move", "see"}),
        focal_agent=focal_agent,
        goal_mode=goal_mode,
        checker_profile=checker_profile,
        template_fn=template_fn,
    )


def swap_all_say(
    prose: str,
    world: World,
    plan: Any,
    *,
    focal_agent: str,
    goal_mode: str = DEFAULT_GOAL_MODE,
    checker_profile: str = "d",
    template_fn: Callable[..., str] | None = None,
) -> SwapResult:
    """Replace every `say` sentence at `sentence_index` with its planned `SAY_1`–`SAY_4` template under the §5 scene re-check."""
    return _run_swap_pass(
        prose,
        world,
        plan,
        target_kinds=frozenset({"say"}),
        focal_agent=focal_agent,
        goal_mode=goal_mode,
        checker_profile=checker_profile,
        template_fn=template_fn,
    )


def render_control_scene(
    world: World,
    focal_agent: str,
    plan: Any,
) -> str:
    """Render the Option 3 control-arm scene: `render_world_template`, then `swap_all_move_see`, then `swap_all_say`."""
    base_prose = render_world_template(world, focal_agent)
    move_see_res = swap_all_move_see(
        base_prose,
        world,
        plan,
        focal_agent=focal_agent,
        goal_mode=DEFAULT_GOAL_MODE,
        checker_profile="d",
    )
    say_res = swap_all_say(
        move_see_res.prose,
        world,
        plan,
        focal_agent=focal_agent,
        goal_mode=DEFAULT_GOAL_MODE,
        checker_profile="d",
    )
    if not say_res.s_passed:
        check_rendered_prose(
            say_res.prose,
            world,
            focal_agent=focal_agent,
            goal_mode=DEFAULT_GOAL_MODE,
            checker_profile="d",
        )
    return say_res.prose


def generate_stage5_shape_plan(
    worlds: Sequence[World],
    *,
    seed: int,
    block_size: int = SHAPE_PLAN_BLOCK_SIZE,
) -> dict[str, object]:
    """Generate a deterministic Stage 5 shape plan across `worlds` (`MOVE_1..4`, `SEE_1..2`, `SEE_3..5`, `SAY_1..4`).

    Uses the Stage 4 block-of-20 quota balance (spread <= 1) and no-3-in-a-row rejection rule, with `SAY_5` excluded
    from the `say` pool (`SAY_1`–`SAY_4` only) and `MOVE_0` assigned to `move_station_change`.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {seed!r}")
    if block_size <= 0:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    assigned_variants: dict[str, tuple[str, ...]] = {
        "move_normal": MOVE_FORMS,
        "move_occluded": MOVE_FORMS,
        "see_unoccluded": SEE_UNOCCLUDED_FORMS,
        "see_occluded": SEE_OCCLUDED_FORMS,
        "say": SAY_FORMS,
    }
    stream_defs: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("move_normal", ("move_normal",)),
        ("move_occluded", ("move_occluded",)),
        ("see_unoccluded", ("see_unoccluded",)),
        ("see_occluded", ("see_occluded",)),
        ("say", ("say",)),
    )

    plan_by_scenario_id: dict[str, list[dict[str, object]]] = {}
    all_records: list[dict[str, object]] = []

    for block_idx, b_start in enumerate(range(0, len(worlds), block_size)):
        block_worlds = worlds[b_start : b_start + block_size]
        has_both_move_variants = (
            any(event_shape_plan_variant(ev, w) == "move_normal" for w in block_worlds for ev in w.events)
            and any(event_shape_plan_variant(ev, w) == "move_occluded" for w in block_worlds for ev in w.events)
        )
        effective_streams: tuple[tuple[str, tuple[str, ...]], ...] = (
            (
                ("move", ("move_normal", "move_occluded")),
                ("see_unoccluded", ("see_unoccluded",)),
                ("see_occluded", ("see_occluded",)),
                ("say", ("say",)),
            )
            if has_both_move_variants
            else stream_defs
        )

        drawn_by_scene_variant: dict[tuple[int, str], list[str]] = {}

        for stream_name, stream_variants in effective_streams:
            stream_seed_bytes = hashlib.blake2b(
                f"udlab.stage5.shape_plan.v1:{seed}:block_{block_idx}:{stream_name}".encode("utf-8"),
                digest_size=8,
            ).digest()
            stream_rng = random.Random(int.from_bytes(stream_seed_bytes, "big"))
            v_counts: Counter[str] = Counter()
            for w in block_worlds:
                for ev in w.events:
                    v = event_shape_plan_variant(ev, w)
                    if v in stream_variants:
                        v_counts[v] += 1
            if sum(v_counts.values()) == 0:
                continue

            succeeded = False
            for _attempt in range(10_000):
                quotas: dict[str, dict[str, int]] = {}
                for v in stream_variants:
                    forms = assigned_variants[v]
                    m = v_counts[v]
                    k = len(forms)
                    base_q = m // k
                    rem = m % k
                    extra = set(stream_rng.sample(forms, rem)) if rem else set()
                    quotas[v] = {f: base_q + (1 if f in extra else 0) for f in forms}

                ok = True
                candidate_draws: dict[tuple[int, str], list[str]] = {}
                for s_off, w in enumerate(block_worlds):
                    last_kind_forms: list[str] = []
                    for ev in w.events:
                        v = event_shape_plan_variant(ev, w)
                        if v == "move_station_change" and "move_normal" in stream_variants:
                            last_kind_forms.append("MOVE_0")
                            continue
                        if v not in stream_variants:
                            continue
                        forms = assigned_variants[v]
                        forbidden = (
                            last_kind_forms[-1]
                            if (len(last_kind_forms) >= 2 and last_kind_forms[-1] == last_kind_forms[-2])
                            else None
                        )
                        avail = [f for f in forms if quotas[v][f] > 0 and f != forbidden]
                        if not avail:
                            ok = False
                            break
                        while True:
                            cand = stream_rng.choice(forms)
                            if cand in avail:
                                break
                        quotas[v][cand] -= 1
                        last_kind_forms.append(cand)
                        candidate_draws.setdefault((s_off, v), []).append(cand)
                    if not ok:
                        break
                if ok:
                    drawn_by_scene_variant.update(candidate_draws)
                    succeeded = True
                    break
            if not succeeded:
                raise RuntimeError(
                    f"STOPPED: Stage 5 shape plan block-of-{block_size} balance infeasible "
                    f"(block={block_idx}, stream={stream_name})."
                )

        for s_off, w in enumerate(block_worlds):
            variant_cursors: Counter[str] = Counter()
            scen_records: list[dict[str, object]] = []
            for ev_idx, ev in enumerate(w.events, start=1):
                v = event_shape_plan_variant(ev, w)
                if v in assigned_variants:
                    cur = variant_cursors[v]
                    variant_cursors[v] += 1
                    form_id = drawn_by_scene_variant[(s_off, v)][cur]
                else:
                    form_id = wildcard_form_id_for_event(ev, w)
                rec: dict[str, object] = {
                    "scenario_id": w.scenario_id,
                    "event_index": ev_idx,
                    "kind": ev.kind,
                    "variant": v,
                    "form_id": form_id,
                }
                scen_records.append(rec)
                all_records.append(rec)
            plan_by_scenario_id[w.scenario_id] = scen_records

    return {
        "seed": seed,
        "block_size": block_size,
        "plan_records": all_records,
        "plan_by_scenario_id": plan_by_scenario_id,
    }


def draw_scene_shape_plan(world: World, *, seed: int) -> dict[int, str]:
    """Draw a deterministic uniform per-event Stage 5 shape plan `{event_index: form_id}` for a single `world`."""
    payload = f"udlab.stage5.scene_plan.v1:{seed}:{world.scenario_id}:{world.seed}".encode("utf-8")
    rng = random.Random(int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big"))
    out: dict[int, str] = {}
    for ev_idx, ev in enumerate(world.events, start=1):
        forms = applicable_forms_for_event(ev, world)
        if forms:
            out[ev_idx] = rng.choice(forms)
        else:
            out[ev_idx] = wildcard_form_id_for_event(ev, world)
    return out


def run_template_self_test(
    worlds: Sequence[World],
    *,
    plan_seed: int = 20261102,
) -> dict[str, object]:
    """Run the Card U2 Gate 0 template self-test across both renderings of every scenario in `worlds`.

    Checks:
    1. Single-event splices: for each scene from `render_world_template`, for each `move`, `see`, and `say` event,
       and for each shape applicable to it, splice that one event's sentence with `render_event_sentence` and verify:
       (a) the event's row reports that exact `form_id`;
       (b) every other row's `realized_form_id` and `sentence_index` are unchanged;
       (c) `event_sentence_count` is unchanged;
       (d) `check_prose_render_fidelity(..., checker_profile='d')` raises nothing.
    2. All-at-once swap: swap all moves and sightings via `swap_all_move_see` (uniform per-event plan from `plan_seed`)
       plus all says via `swap_all_say` (`SAY_1`–`SAY_4`). Verify every row reports its planned form and the scene passes.
    3. Control arm (`render_control_scene`): verify 100% pass under profile `d`.
    """
    events_by_kind_per_scenario: Counter[str] = Counter()
    events_by_variant_per_scenario: Counter[str] = Counter()
    events_by_kind_across_renderings: Counter[str] = Counter()
    events_by_variant_across_renderings: Counter[str] = Counter()

    attempted_by_shape: Counter[str] = Counter({f: 0 for f in ALL_TEMPLATE_FORMS})
    passed_by_shape: Counter[str] = Counter({f: 0 for f in ALL_TEMPLATE_FORMS})
    failures: list[dict[str, object]] = []

    all_at_once_attempted = 0
    all_at_once_passed = 0
    control_arm_attempted = 0
    control_arm_passed = 0

    for w in worlds:
        for ev in w.events:
            v = event_shape_plan_variant(ev, w)
            events_by_kind_per_scenario[ev.kind] += 1
            events_by_variant_per_scenario[v] += 1

        scene_plan = draw_scene_shape_plan(w, seed=plan_seed)

        for r_idx, focal_agent in ((0, w.agents[0]), (1, w.agents[1])):
            for ev in w.events:
                v = event_shape_plan_variant(ev, w)
                events_by_kind_across_renderings[ev.kind] += 1
                events_by_variant_across_renderings[v] += 1

            base_prose = render_world_template(w, focal_agent)
            base_rows, base_sent_count = match_rendering_event_forms(
                base_prose,
                w,
                focal_agent=focal_agent,
                checker_profile="d",
                return_sentence_count=True,
            )

            # 1. Single-event splices for every applicable shape
            for ev_pos, ev in enumerate(w.events):
                forms = applicable_forms_for_event(ev, w)
                if not forms:
                    continue
                base_row = base_rows[ev_pos]
                s_idx = int(base_row["sentence_index"])  # type: ignore[arg-type]

                for form_id in forms:
                    attempted_by_shape[form_id] += 1
                    cand_sent = render_event_sentence(ev, w, form_id, focal_agent=focal_agent)
                    spliced_prose = splice_event_sentence_at_index(base_prose, s_idx, cand_sent)

                    if spliced_prose == base_prose:
                        rows_after = base_rows
                        sent_count_after = base_sent_count
                        fidelity_rule_id = None
                        cond_d = True
                    else:
                        rows_after, sent_count_after = match_rendering_event_forms(
                            spliced_prose,
                            w,
                            focal_agent=focal_agent,
                            checker_profile="d",
                            return_sentence_count=True,
                        )
                        fidelity_rule_id = None
                        try:
                            check_prose_render_fidelity(
                                spliced_prose,
                                w,
                                focal_agent=focal_agent,
                                checker_profile="d",
                            )
                            cond_d = True
                        except RenderCheckError as exc:
                            cond_d = False
                            fidelity_rule_id = exc.rule_id or exc.check_id

                    target_row = rows_after[ev_pos]
                    cond_a = target_row.get("realized_form_id") == form_id
                    cond_b = all(
                        rows_after[i].get("realized_form_id") == base_rows[i].get("realized_form_id")
                        and rows_after[i].get("sentence_index") == base_rows[i].get("sentence_index")
                        for i in range(len(w.events))
                        if i != ev_pos
                    )
                    cond_c = sent_count_after == base_sent_count

                    if cond_a and cond_b and cond_c and cond_d:
                        passed_by_shape[form_id] += 1
                    else:
                        failures.append(
                            {
                                "scenario_id": w.scenario_id,
                                "rendering_index": r_idx,
                                "event_index": ev_pos + 1,
                                "kind": ev.kind,
                                "variant": event_shape_plan_variant(ev, w),
                                "form_id": form_id,
                                "realized_form_id": target_row.get("realized_form_id"),
                                "rule_id": fidelity_rule_id or target_row.get("rule_id"),
                                "cond_a_form": cond_a,
                                "cond_b_neighbours": cond_b,
                                "cond_c_count": cond_c,
                                "cond_d_fidelity": cond_d,
                            }
                        )

            # 2 & 3. All-moves-and-sightings swapped at once (`swap_all_move_see`), plus all says swapped to
            # SAY_1-4 (`swap_all_say`), via `render_control_scene` (which runs `render_world_template`,
            # then `swap_all_move_see`, then `swap_all_say`).
            all_at_once_attempted += 1
            control_arm_attempted += 1
            try:
                ctrl_prose = render_control_scene(w, focal_agent, scene_plan)
                check_prose_render_fidelity(
                    ctrl_prose,
                    w,
                    focal_agent=focal_agent,
                    checker_profile="d",
                )
                ctrl_passed = True
                ctrl_rule_ids: list[str] = []
            except RenderCheckError as exc:
                ctrl_prose = ""
                ctrl_passed = False
                ctrl_rule_ids = [exc.rule_id or exc.check_id]

            if ctrl_passed:
                control_arm_passed += 1
                final_rows = match_rendering_event_forms(
                    ctrl_prose,
                    w,
                    focal_agent=focal_agent,
                    checker_profile="d",
                )
                all_planned_matched = all(
                    final_rows[i].get("realized_form_id") == scene_plan[i + 1]
                    for i in range(len(w.events))
                )
                if all_planned_matched:
                    all_at_once_passed += 1
                else:
                    failures.append(
                        {
                            "scenario_id": w.scenario_id,
                            "rendering_index": r_idx,
                            "phase": "all_at_once",
                            "s_passed": True,
                            "all_planned_matched": False,
                            "rule_ids": [],
                        }
                    )
            else:
                failures.append(
                    {
                        "scenario_id": w.scenario_id,
                        "rendering_index": r_idx,
                        "phase": "control_arm",
                        "rule_ids": ctrl_rule_ids,
                    }
                )

    per_shape_report = {
        form_id: {
            "attempted": int(attempted_by_shape[form_id]),
            "passed": int(passed_by_shape[form_id]),
        }
        for form_id in ALL_TEMPLATE_FORMS
    }

    return {
        "scenarios": len(worlds),
        "renderings": len(worlds) * 2,
        "events_by_kind_per_scenario": dict(events_by_kind_per_scenario),
        "events_by_variant_per_scenario": dict(events_by_variant_per_scenario),
        "events_by_kind_across_renderings": dict(events_by_kind_across_renderings),
        "events_by_variant_across_renderings": dict(events_by_variant_across_renderings),
        "per_shape": per_shape_report,
        "total_single_splice_attempted": int(sum(attempted_by_shape.values())),
        "total_single_splice_passed": int(sum(passed_by_shape.values())),
        "all_at_once_attempted": all_at_once_attempted,
        "all_at_once_passed": all_at_once_passed,
        "control_arm_attempted": control_arm_attempted,
        "control_arm_passed": control_arm_passed,
        "failures": failures,
        "all_passed": (
            len(failures) == 0
            and sum(attempted_by_shape.values()) == sum(passed_by_shape.values())
            and all_at_once_attempted == all_at_once_passed
            and control_arm_attempted == control_arm_passed
        ),
    }
