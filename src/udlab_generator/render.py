"""Deterministic third-person perspective renderer, writer prompt, prose checks, and OllamaWriter."""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from typing import Any, NoReturn
from urllib.parse import urlparse
import urllib.error
import urllib.request

from udlab_generator.keys import ScenarioKeys, compute_scenario_keys
from udlab_generator.world import (
    AGENT_NAME_POOLS,
    BELIEF_PERSISTENCE_RULE,
    CONTAINER_BAY_STATION,
    DOMAINS,
    FAR_SIDE_ALCOVE_STATION,
    OCCLUSION_RULE,
    Event,
    World,
    is_station_transition_event,
)

BANNED_ADVERBS: tuple[str, ...] = (
    "nervously",
    "secretly",
    "suspiciously",
    "anxiously",
    "furtively",
    "stealthily",
    "guiltily",
    "gleefully",
    "angrily",
    "frustratedly",
    "eagerly",
    "reluctantly",
    "hesitantly",
    "deceitfully",
    "slyly",
    "cautiously",
    "fearfully",
    "bitterly",
    "jealously",
    "knowingly",
    "unknowingly",
    "unwittingly",
    "inadvertently",
    "deliberately",
    "intentionally",
    "purposely",
)

_BANNED_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in BANNED_ADVERBS) + r")\b",
    re.IGNORECASE,
)
_BRACKET_MARKER_RE = re.compile(r"\[[^\]]*\]")
_FACT_ID_RE = re.compile(r"\bF[0-9][a-z]?\b")
_STEP_NUMBER_RE = re.compile(r"\bstep\s+\d+\b", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])["\']?\s+')

ALL_KNOWN_PARTY_NAMES: frozenset[str] = frozenset(
    name for pool in AGENT_NAME_POOLS for name in pool
)

WRITER_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "writer_qwen3_6_35b.txt"

# Pinned writer model specification (DESIGN-udlab-r1-2026-10-29.md §0 & §6.8; rdlab models_and_build.py):
DEFAULT_OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
LOOPBACK_HOST: str = "127.0.0.1"
PINNED_WRITER_MODEL_TAG: str = "qwen3.6:35b"
PINNED_WRITER_MANIFEST_DIGEST: str = (
    "a7eb95c53bcf96b4bdd008d0fab4a5dac88047d9c1a7a9ab88ed453423fbd87c"
)
PINNED_WRITER_MODEL_LAYER_DIGEST: str = (
    "d372de8e934898a59e6ccfabc3368474711384d8f1fd4d22d87a3f0a45400cdc"
)
PINNED_WRITER_LICENCE_LAYER_DIGEST: str = (
    "5f3a3c817e78f5b8a4ad2d2c458a3e4b2cce470d6c12642c4eeb12cb8a9bf51d"
)
PINNED_WRITER_LICENCE_ID: str = "qwen3.6-35b-a3b-licence"
DEFAULT_WRITER_NUM_PREDICT: int = 4096
DEFAULT_WRITER_THINK_NUM_PREDICT: int = 5120
DEFAULT_WRITER_MAX_NUM_CTX: int = 9216
DEFAULT_WRITER_THINK: bool = False
DEFAULT_VRAM_HALT_MIB: int = 26_000
NUM_CTX_MARGIN_TOKENS: int = 256
NUM_CTX_STEP_TOKENS: int = 1024
PROMPT_TOKEN_ESTIMATE_RULE: str = "ceil(utf8_bytes / 3)"
CHECKPOINT_FILENAME: str = "checkpoint.json"
SEALED_CHECKPOINT_FILENAME: str = "checkpoint.sealed.json"
SEALED_RESPONSES_FILENAME: str = "responses.sealed.jsonl"
CHECKPOINT_GENESIS_HASH: str = "0" * 64
GENERATIONS_FILENAME: str = "generations.jsonl"

OllamaHttpTransport = Callable[[str, str, dict[str, Any] | None], dict[str, Any]]


@dataclass(frozen=True)
class PinnedWriterModelSpec:
    """Pinned Ollama writer model specification (`qwen3.6:35b`)."""

    ollama_tag: str = PINNED_WRITER_MODEL_TAG
    manifest_digest: str = PINNED_WRITER_MANIFEST_DIGEST
    model_layer_digest: str = PINNED_WRITER_MODEL_LAYER_DIGEST
    licence_layer_digest: str = PINNED_WRITER_LICENCE_LAYER_DIGEST
    licence_id: str = PINNED_WRITER_LICENCE_ID
    max_num_ctx: int = DEFAULT_WRITER_MAX_NUM_CTX
    num_predict: int = DEFAULT_WRITER_NUM_PREDICT
    think: bool = DEFAULT_WRITER_THINK

    def to_dict(self) -> dict[str, object]:
        return {
            "ollama_tag": self.ollama_tag,
            "manifest_digest": self.manifest_digest,
            "model_layer_digest": self.model_layer_digest,
            "licence_layer_digest": self.licence_layer_digest,
            "licence_id": self.licence_id,
            "max_num_ctx": self.max_num_ctx,
            "num_predict": self.num_predict,
            "think": self.think,
        }


DEFAULT_WRITER_MODEL_PIN = PinnedWriterModelSpec()


@dataclass(frozen=True)
class ProseRefusalRecord:
    """Structured refusal record emitted whenever a prose check or writer call refuses."""

    scenario_id: str
    rendering_index: int
    focal_agent: str
    check_id: str
    reason: str
    failing_sentence: str | None = None
    event_index: int | None = None
    event_kind: str | None = None
    failing_sentence_sha256: str = ""
    failing_sentence_offset: int | None = None
    failing_sentence_length: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "rendering_index": self.rendering_index,
            "focal_agent": self.focal_agent,
            "check_id": self.check_id,
            "reason": self.reason,
            "failing_sentence": self.failing_sentence,
            "event_index": self.event_index,
            "event_kind": self.event_kind,
            "failing_sentence_sha256": self.failing_sentence_sha256,
            "failing_sentence_offset": self.failing_sentence_offset,
            "failing_sentence_length": self.failing_sentence_length,
        }


class RenderCheckError(ValueError):
    """Raised when rendered scenario prose fails any check (a)-(d) or writer constraint."""

    def __init__(
        self,
        message: str,
        *,
        check_id: str = "render_check",
        failing_sentence: str | None = None,
        event_index: int | None = None,
        event_kind: str | None = None,
    ) -> None:
        super().__init__(message)
        self.check_id = check_id
        self.reason = message
        self.failing_sentence = failing_sentence
        self.event_index = event_index
        self.event_kind = event_kind


class ForbiddenModelCallError(RuntimeError):
    """Raised if any code attempts a real `/api/chat` or live model call in CPU-only mock mode."""


@dataclass(frozen=True)
class RenderedScenario:
    """One third-person rendering of a complete `World` focalized on `focal_agent`."""

    scenario_id: str
    rendering_index: int
    focal_agent: str
    prose: str
    writer_prompt: str
    keys: ScenarioKeys
    world: World | None = None


@dataclass(frozen=True)
class ProseRequirementSpec:
    """One structural requirement shared by the writer prompt and `check_rendered_prose`."""

    check_id: str
    rule_text: str


HARD_CONSTRAINTS_PLACEHOLDER: str = "{hard_constraints_block}"

SIGHTING_VERBS: tuple[str, ...] = (
    "observed",
    "looked",
    "saw",
    "see",
    "spotted",
    "noticed",
    "inspected",
    "checked",
    "viewed",
    "watched",
    "glanced",
    "examined",
)

BLOCKING_PHRASES: tuple[str, ...] = (
    "blocked",
    "could not see",
    "hidden behind",
    "obscured",
    "blocking sight",
)

NEGATION_PHRASES: tuple[str, ...] = (
    "not",
    "n't",
    "never",
    "no longer",
    "no one",
    "nobody",
    "nothing",
    "under no circumstances",
    "by no means",
    "at no time",
    "failed to",
    "refused to",
    "forgot to",
    "declined to",
    "neglected to",
    "avoided",
    "omitted to",
    "refrained from",
    "without",
    "neither",
    "nor",
    "unable to",
    "could not",
    "cannot",
    "did not",
    "didn't",
)

_NEGATION_PATTERN = re.compile(
    r"(?:"
    + "|".join(
        re.escape(p) if p.startswith("'") or p.startswith("n'") else rf"\b{re.escape(p)}\b"
        for p in NEGATION_PHRASES
    )
    + r")",
    re.IGNORECASE,
)
_ADJACENT_ROOMS_COULD_NOT_HEAR_RE = re.compile(r"\badjacent\s+rooms\s+could\s+not\s+hear\b", re.IGNORECASE)
_COULD_NOT_SEE_RE = re.compile(r"\bcould\s+not\s+see\b", re.IGNORECASE)
_FRONTED_TIME_PLACE_NEG_RE = re.compile(
    r"\b(?:not\s+long\s+(?:after(?:wards?)?|before)\s*,"
    r"|not\s+far\s+(?:from|away|off)(?:\s+[a-z0-9_-]+){0,6}\s*,"
    r"|without\s+(?:delay|pause|waiting|hesitation)(?:\s*,|\b))",
    re.IGNORECASE,
)
_POST_ARG_BENIGN_NEG_RE = re.compile(
    r"(?:\bwithout\s+(?:delay|pause|waiting|hesitation|incident|difficulty)\b"
    r"|\bnot\s+(?:far\s+(?:from|away|off)|long\s+(?:after(?:wards?)?|before))\b[^,;:.]*)",
    re.IGNORECASE,
)
_POST_ARG_GERUND_CLAUSE_RE = re.compile(r"\bwithout\s+([a-z]+ing)\b[^,;:.]*", re.IGNORECASE)
_POST_ARG_REL_CLAUSE_RE = re.compile(r",\s*(?:which|where|who)\b[^.;:]*", re.IGNORECASE)
_ARTICLE_OPT_RE = r"(?:(?:the|a|an)\s+)?"

SETUP_CHECKLIST_ORDER: str = (
    "rooms -> containers per room -> fixtures per room -> occluders -> doorways -> "
    "starting rooms -> goals -> stake -> belief rule -> hearing threshold"
)

_ACTION_VERBS_BY_KIND: dict[str, tuple[str, ...]] = {
    "enter": ("walked", "entered", "stepped", "came", "arrived", "went", "moved", "returned", "proceeded", "crossed"),
    "leave": ("exited", "left", "departed", "walked", "headed", "went", "stepped", "proceeded", "crossed"),
    "move": (
        "moved",
        "stepped",
        "placed",
        "put",
        "shifted",
        "transferred",
        "adjusted",
        "turned",
        "pulled",
        "set",
        "relocated",
        "carried",
        "took",
    ),
    "see": SIGHTING_VERBS,
    "say": ("said", "stated", "spoke", "announced", "uttered", "claimed", "reported", "told"),
    "message": ("sent", "messaged", "texted", "transmitted", "notified", "wrote"),
}
_FOLDED_LEAVE_ENTER_VERBS: tuple[str, ...] = tuple(
    dict.fromkeys(_ACTION_VERBS_BY_KIND["enter"] + _ACTION_VERBS_BY_KIND["leave"])
)

_GERUNDS_BY_KIND: dict[str, frozenset[str]] = {
    "enter": frozenset(
        {"walking", "entering", "stepping", "coming", "arriving", "going", "moving", "returning", "proceeding", "crossing"}
    ),
    "leave": frozenset(
        {"exiting", "leaving", "departing", "walking", "heading", "going", "stepping", "proceeding", "crossing", "moving"}
    ),
    "move": frozenset(
        {
            "moving",
            "stepping",
            "placing",
            "putting",
            "shifting",
            "transferring",
            "adjusting",
            "turning",
            "pulling",
            "setting",
            "relocating",
            "carrying",
            "taking",
        }
    ),
    "see": frozenset(
        {
            "observing",
            "looking",
            "seeing",
            "spotting",
            "noticing",
            "inspecting",
            "checking",
            "viewing",
            "watching",
            "glancing",
            "examining",
        }
    ),
    "say": frozenset(
        {"saying", "stating", "speaking", "announcing", "uttering", "claiming", "reporting", "telling"}
    ),
    "message": frozenset(
        {"sending", "messaging", "texting", "transmitting", "notifying", "writing"}
    ),
}
_BASE_VERBS_BY_KIND: dict[str, frozenset[str]] = {
    "enter": frozenset({"walk", "enter", "step", "come", "arrive", "go", "move", "return", "proceed", "cross"}),
    "leave": frozenset({"exit", "leave", "depart", "walk", "head", "go", "step", "proceed", "cross", "move"}),
    "move": frozenset(
        {"move", "step", "place", "put", "shift", "transfer", "adjust", "turn", "pull", "set", "relocate", "carry", "take"}
    ),
    "see": frozenset(
        {"observe", "look", "see", "spot", "notice", "inspect", "check", "view", "watch", "glance", "examine"}
    ),
    "say": frozenset({"say", "state", "speak", "announce", "utter", "claim", "report", "tell"}),
    "message": frozenset({"send", "message", "text", "transmit", "notify", "write"}),
}


def format_action_verbs_by_kind_summary() -> str:
    """Format `_ACTION_VERBS_BY_KIND` into a deterministic prompt string listing allowed verbs per event `kind`."""
    return "; ".join(f"{kind}: {', '.join(verbs)}" for kind, verbs in _ACTION_VERBS_BY_KIND.items())


def format_blocking_phrases_summary() -> str:
    """Format `BLOCKING_PHRASES` into a deterministic prompt string."""
    return ", ".join(f"'{p}'" for p in BLOCKING_PHRASES)


def format_negation_phrases_summary() -> str:
    """Format `NEGATION_PHRASES` into a deterministic prompt string."""
    return ", ".join(NEGATION_PHRASES)


WRITER_PROSE_REQUIREMENTS: tuple[ProseRequirementSpec, ...] = (
    ProseRequirementSpec(
        "c_fidelity_paragraph_split",
        f'Output two paragraphs separated by "\\n\\n": Paragraph 1 setup ({SETUP_CHECKLIST_ORDER}); Paragraph 2 events.',
    ),
    ProseRequirementSpec(
        "c_fidelity_focal_agent",
        "Name focal_agent in setup, framing transitions around focal_agent without omitting events.",
    ),
    ProseRequirementSpec(
        "c_fidelity_room_setup",
        "Name every room from rooms in setup.",
    ),
    ProseRequirementSpec(
        "c_fidelity_containers",
        "Name both containers of each room in setup.",
    ),
    ProseRequirementSpec(
        "c_fidelity_fixtures",
        "Name each room's fixture in setup.",
    ),
    ProseRequirementSpec(
        "c_fidelity_occluder",
        "Name each room's occluder in setup, between 'far-side alcove' and 'container bay' blocking sight from 'far-side alcove' while 'container bay' sees containers in open view.",
    ),
    ProseRequirementSpec(
        "c_fidelity_stations",
        "Name 'container bay' and 'far-side alcove' in setup.",
    ),
    ProseRequirementSpec(
        "c_fidelity_doorways",
        "For each doorway [room_a, room_b], name both rooms in one setup sentence (natural phrasing).",
    ),
    ProseRequirementSpec(
        "c_fidelity_initial_room",
        "For each agent, name agent and initial_agent_rooms[agent] in one setup sentence (grouped pass).",
    ),
    ProseRequirementSpec(
        "c_fidelity_goal",
        "Copy each agent's goal_text (goals[agent].goal_text) verbatim into setup.",
    ),
    ProseRequirementSpec(
        "c_fidelity_absent_party",
        "Name every absent party in setup.",
    ),
    ProseRequirementSpec(
        "c_fidelity_stake_edge",
        "Copy each stake description (stake_edges[].description) verbatim into setup.",
    ),
    ProseRequirementSpec(
        "d_rule_sentence",
        f'Include belief-persistence rule once in setup: "{BELIEF_PERSISTENCE_RULE}"',
    ),
    ProseRequirementSpec(
        "d_hearing_threshold",
        "State doorway hearing threshold in dB and 'doorway' in setup (>= threshold dB carries one doorway; quieter stays).",
    ),
    ProseRequirementSpec(
        "c_fidelity_event_order",
        "Write one sentence per event in order (event sentences state what happened, without negation: "
        f"{format_negation_phrases_summary()}); "
        "actor (no pronouns) precedes verb ("
        f"{format_action_verbs_by_kind_summary()}"
        ") with no intervening party; name room ('in/from/inside/at/of [the] <room>', no other room on intra-room); "
        "every enter sentence must name both from_location and to_location (never write a bare 'entered <to>' without 'from <from>'), and from_location precedes to_location on enter/leave/move (or 'entered/arrived in <to> ... from <from>' on enter), where from-locations on enter/move use 'from'/'out of' and to-locations on enter/leave/move use 'to'/'into'/'toward'/'at'/'in'/'onto'; "
        "on see, name target_object, and if unoccluded at a container place it at to_location ('container bay' optional); "
        "on unoccluded move/see, no blocking phrases ("
        f"{format_blocking_phrases_summary()}"
        "; 'in open view' optional); on occluded, state occluder + 'blocked'/'blocking sight' (+ 'far-side alcove' on occluded see, where 'could not see' is allowed); "
        "on say, '<noise_level> dB' + verbatim quote; on message, 'to/for/messaged/texted/notified <recipient>' + verbatim quote.",
    ),
    ProseRequirementSpec(
        "c_fidelity_missing_event",
        "Include every logged event without omitting or merging.",
    ),
    ProseRequirementSpec(
        "c_fidelity_unlogged_event",
        "Include no extra unlogged event sentences.",
    ),
    ProseRequirementSpec(
        "c_fidelity_contradictory_occlusion",
        "Never combine 'in open view' and 'blocked' in one sentence.",
    ),
    ProseRequirementSpec(
        "c_fidelity_unlogged_party",
        "Never name a party outside agents and absent_parties.",
    ),
    ProseRequirementSpec(
        "c_fidelity_unlogged_entity",
        "Never name an entity from another domain.",
    ),
    ProseRequirementSpec(
        "c_fidelity_unlogged_motive",
        "Never state an unlogged motive when stated_reason is absent.",
    ),
    ProseRequirementSpec(
        "c_fidelity_unlogged_placer",
        "Never attribute placement/movement of the unsigned note when unlogged.",
    ),
    ProseRequirementSpec(
        "a_no_markers_bracket",
        "Do not emit bracket markers ('[...]').",
    ),
    ProseRequirementSpec(
        "a_no_markers_fact_id",
        "Do not emit fact IDs (F0-F9).",
    ),
    ProseRequirementSpec(
        "a_no_markers_step_number",
        "Do not emit step numbers ('step N').",
    ),
    ProseRequirementSpec(
        "b_banned_adverbs",
        "Never use banned adverbs: " + ", ".join(BANNED_ADVERBS) + ".",
    ),
)

WRITER_PROSE_REQUIREMENTS_BY_ID: dict[str, ProseRequirementSpec] = {
    req.check_id: req for req in WRITER_PROSE_REQUIREMENTS
}

_PROMPT_REQ_LINE_RE = re.compile(r"^-\s+\(([a-z0-9_]+)\)\s+(.+)$")


def render_writer_prompt_constraints(
    requirements: Sequence[ProseRequirementSpec] = WRITER_PROSE_REQUIREMENTS,
) -> str:
    """Render the `HARD CONSTRAINTS (MANDATORY):` bullet lines from `WRITER_PROSE_REQUIREMENTS`."""
    return "\n".join(f"- ({req.check_id}) {req.rule_text}" for req in requirements)


def parse_writer_prompt_requirements(prompt_text: str) -> tuple[ProseRequirementSpec, ...]:
    """Parse the mandatory requirement list from `prompt_text` (expanding `{hard_constraints_block}` if unformatted) and reject any untracked requirement lines."""
    expanded_text = (
        prompt_text.replace(HARD_CONSTRAINTS_PLACEHOLDER, render_writer_prompt_constraints())
        if HARD_CONSTRAINTS_PLACEHOLDER in prompt_text
        else prompt_text
    )
    start_marker = "HARD CONSTRAINTS (MANDATORY):"
    end_marker = "INPUT SPECIFICATION:"
    start_idx = expanded_text.find(start_marker)
    end_idx = expanded_text.find(end_marker)
    if start_idx < 0 or end_idx < 0 or end_idx <= start_idx:
        raise ValueError("Writer prompt is missing HARD CONSTRAINTS (MANDATORY) or INPUT SPECIFICATION section.")
    section = expanded_text[start_idx + len(start_marker) : end_idx]
    parsed: list[ProseRequirementSpec] = []
    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = _PROMPT_REQ_LINE_RE.match(line)
        if m is None:
            raise ValueError(
                f"Unrecognized or untracked requirement line in writer prompt HARD CONSTRAINTS: {line!r}"
            )
        parsed.append(ProseRequirementSpec(check_id=m.group(1), rule_text=m.group(2)))
    if not parsed:
        raise ValueError("No structural requirements parsed from writer prompt.")
    return tuple(parsed)


def get_checker_enforced_check_ids() -> tuple[str, ...]:
    """Extract every `check_id` enforced by `check_prose_no_markers`, `check_prose_banned_adverbs`, `check_prose_rule_and_hearing_threshold`, and `check_prose_render_fidelity`."""
    checker_fns = (
        check_prose_no_markers,
        check_prose_banned_adverbs,
        check_prose_rule_and_hearing_threshold,
        check_prose_render_fidelity,
    )
    found: list[str] = []
    for fn in checker_fns:
        tree = ast.parse(inspect.getsource(fn))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else ""
                )
                if func_name == "_raise_prose_check_error" and node.args:
                    first_arg = node.args[0]
                    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                        if first_arg.value not in found:
                            found.append(first_arg.value)
                elif func_name == "RenderCheckError":
                    for kw in node.keywords:
                        if kw.arg == "check_id" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                            if kw.value.value not in found:
                                found.append(kw.value.value)
    return tuple(found)


def verify_prompt_and_checker_requirements_aligned(
    prompt_text: str | None = None,
) -> tuple[ProseRequirementSpec, ...]:
    """Verify that the writer prompt's parsed requirements, `WRITER_PROSE_REQUIREMENTS`, and the checker's enforced `check_id`s are identical."""
    if prompt_text is None:
        disk_raw = WRITER_PROMPT_PATH.read_text(encoding="utf-8")
        if HARD_CONSTRAINTS_PLACEHOLDER not in disk_raw:
            raise ValueError(
                f"On-disk writer prompt '{WRITER_PROMPT_PATH}' must contain {HARD_CONSTRAINTS_PLACEHOLDER!r} placeholder."
            )
        for line in disk_raw.splitlines():
            if _PROMPT_REQ_LINE_RE.match(line.strip()):
                raise ValueError(
                    f"On-disk writer prompt '{WRITER_PROMPT_PATH}' must not contain hand-copied requirement lines: {line!r}"
                )
        raw_text = disk_raw.replace(HARD_CONSTRAINTS_PLACEHOLDER, render_writer_prompt_constraints())
    else:
        raw_text = (
            prompt_text.replace(HARD_CONSTRAINTS_PLACEHOLDER, render_writer_prompt_constraints())
            if HARD_CONSTRAINTS_PLACEHOLDER in prompt_text
            else prompt_text
        )
    parsed_prompt_reqs = parse_writer_prompt_requirements(raw_text)
    if parsed_prompt_reqs != WRITER_PROSE_REQUIREMENTS:
        prompt_ids = [r.check_id for r in parsed_prompt_reqs]
        shared_ids = [r.check_id for r in WRITER_PROSE_REQUIREMENTS]
        raise ValueError(
            f"Writer prompt requirements diverge from WRITER_PROSE_REQUIREMENTS: prompt_ids={prompt_ids}, shared_ids={shared_ids}"
        )
    enforced_ids = set(get_checker_enforced_check_ids())
    shared_id_set = {r.check_id for r in WRITER_PROSE_REQUIREMENTS}
    if enforced_ids != shared_id_set:
        raise ValueError(
            f"Checker enforced check_ids diverge from WRITER_PROSE_REQUIREMENTS: only_in_checker={sorted(enforced_ids - shared_id_set)}, only_in_shared={sorted(shared_id_set - enforced_ids)}"
        )
    if set(_REPAIR_TARGET_BY_CHECK_ID.keys()) != shared_id_set:
        raise ValueError(
            f"_REPAIR_TARGET_BY_CHECK_ID keys diverge from WRITER_PROSE_REQUIREMENTS: {sorted(set(_REPAIR_TARGET_BY_CHECK_ID.keys()) ^ shared_id_set)}"
        )
    return parsed_prompt_reqs


def _raise_prose_check_error(
    check_id: str,
    message: str,
    *,
    failing_sentence: str | None = None,
    event_index: int | None = None,
    event_kind: str | None = None,
) -> NoReturn:
    """Raise `RenderCheckError` after verifying `check_id` is registered in `WRITER_PROSE_REQUIREMENTS`."""
    if check_id not in WRITER_PROSE_REQUIREMENTS_BY_ID:
        raise RuntimeError(f"Unregistered prose check_id {check_id!r} not in WRITER_PROSE_REQUIREMENTS.")
    raise RenderCheckError(
        message,
        check_id=check_id,
        failing_sentence=failing_sentence,
        event_index=event_index,
        event_kind=event_kind,
    )


_REPAIR_HINT_BY_EVENT_KIND: dict[str, str] = {
    "enter": "names the room the actor came from and the room entered, with the actor as subject",
    "leave": "names the room the actor left and the destination room, with the actor as subject",
    "move": "names the actor as subject, the target object, the source location, the destination location, and the room",
    "see": "names the actor as subject before a sighting verb, the target object, the room, and any required container or occlusion clause",
    "say": "names the actor as subject before a speech verb, the noise level in dB, and the verbatim quote",
    "message": "names the actor as subject before a message verb, the recipient, and the verbatim quote",
}

_REPAIR_TARGET_BY_CHECK_ID: dict[str, str] = {
    "c_fidelity_paragraph_split": "the two-paragraph setup and events layout",
    "c_fidelity_focal_agent": "the setup focal_agent framing",
    "c_fidelity_room_setup": "the setup room list",
    "c_fidelity_containers": "the setup containers per room",
    "c_fidelity_fixtures": "the setup fixtures per room",
    "c_fidelity_occluder": "the setup occluder description",
    "c_fidelity_stations": "the setup station names",
    "c_fidelity_doorways": "the setup doorway sentences",
    "c_fidelity_initial_room": "the setup starting-room sentence",
    "c_fidelity_goal": "the setup agent goal text",
    "c_fidelity_absent_party": "the setup absent-party sentence",
    "c_fidelity_stake_edge": "the setup stake description",
    "d_rule_sentence": "the setup belief-persistence rule sentence",
    "d_hearing_threshold": "the setup doorway hearing threshold sentence",
    "c_fidelity_event_order": "the out-of-order event sentence",
    "c_fidelity_missing_event": "the missing event sentence",
    "c_fidelity_unlogged_event": "the extra unlogged event sentence",
    "c_fidelity_contradictory_occlusion": "the contradictory occlusion sentence",
    "c_fidelity_unlogged_party": "the sentence naming an unlogged party",
    "c_fidelity_unlogged_entity": "the sentence naming an unlogged entity",
    "c_fidelity_unlogged_motive": "the sentence stating an unlogged motive",
    "c_fidelity_unlogged_placer": "the sentence attributing unlogged note placement",
    "a_no_markers_bracket": "the sentence containing a bracket marker",
    "a_no_markers_fact_id": "the sentence containing a fact ID",
    "a_no_markers_step_number": "the sentence containing a step number",
    "b_banned_adverbs": "the sentence containing a banned adverb",
}


def format_repair_retry_line(exc: RenderCheckError, *, retry_form: str = "multi_turn") -> str:
    """Format a single repair-retry line asking for one targeted fix, without key content or model text.

    When `retry_form == 'single_turn'`, omits 'keep your earlier text' / 'leaving every other sentence unchanged'
    because attempt-0 text is not included in the single-turn fallback request.
    """
    is_single_turn = retry_form == "single_turn"
    if exc.event_index is not None and exc.event_kind is not None:
        hint = _REPAIR_HINT_BY_EVENT_KIND.get(
            exc.event_kind,
            "states the logged event with its actor, verb, locations, and room",
        )
        if exc.check_id == "c_fidelity_event_order":
            if is_single_turn:
                return (
                    f"Ensure event {exc.event_index} ({exc.event_kind}, check {exc.check_id}) "
                    f"appears in its logged chronological position and {hint}."
                )
            return (
                f"Keep your earlier text exactly as it was; change only event {exc.event_index} "
                f"({exc.event_kind}, check {exc.check_id}) so that it appears in its logged chronological position "
                f"and {hint}, leaving every other sentence unchanged."
            )
        if is_single_turn:
            return (
                f"Ensure event {exc.event_index} ({exc.event_kind}, check {exc.check_id}) {hint}."
            )
        return (
            f"Keep your earlier text exactly as it was; change only event {exc.event_index} "
            f"({exc.event_kind}, check {exc.check_id}) so that it {hint}, leaving every other sentence unchanged."
        )
    target_item = _REPAIR_TARGET_BY_CHECK_ID.get(exc.check_id, "the failing item")
    rule_spec = WRITER_PROSE_REQUIREMENTS_BY_ID.get(exc.check_id)
    rule_hint = rule_spec.rule_text if rule_spec is not None else "satisfy this mandatory constraint."
    if is_single_turn:
        return f"Ensure {target_item} (check {exc.check_id}) satisfies: {rule_hint}"
    return (
        f"Keep your earlier text exactly as it was; change only {target_item} "
        f"(check {exc.check_id}) so that it satisfies: {rule_hint} Leave every other sentence unchanged."
    )


def _count_unchanged_sentences(attempt0_text: str, retry_text: str) -> tuple[int, int]:
    """Return `(unchanged_sentence_count, attempt0_sentence_count)` comparing `attempt0_text` and `retry_text`."""
    sents_0 = [
        s
        for para in attempt0_text.split("\n\n")
        if para.strip()
        for s in _split_prose_sentences(para.strip())
    ]
    sents_1 = [
        s
        for para in retry_text.split("\n\n")
        if para.strip()
        for s in _split_prose_sentences(para.strip())
    ]
    unchanged = sum((Counter(sents_0) & Counter(sents_1)).values())
    return unchanged, len(sents_0)


def load_writer_prompt_template(path: Path | None = None) -> str:
    """Load the `qwen3.6:35b` writer prompt text file, expand `{hard_constraints_block}` from `WRITER_PROSE_REQUIREMENTS`, and verify alignment."""
    target = path or WRITER_PROMPT_PATH
    raw_disk_text = target.read_text(encoding="utf-8")
    if target == WRITER_PROMPT_PATH:
        if HARD_CONSTRAINTS_PLACEHOLDER not in raw_disk_text:
            raise ValueError(
                f"Writer prompt template '{target}' must keep only non-constraint text plus {HARD_CONSTRAINTS_PLACEHOLDER!r}."
            )
        for line in raw_disk_text.splitlines():
            if _PROMPT_REQ_LINE_RE.match(line.strip()):
                raise ValueError(
                    f"Writer prompt template '{target}' must not contain hand-maintained requirement lines: {line!r}"
                )
    text = (
        raw_disk_text.replace(HARD_CONSTRAINTS_PLACEHOLDER, render_writer_prompt_constraints())
        if HARD_CONSTRAINTS_PLACEHOLDER in raw_disk_text
        else raw_disk_text
    )
    missing = [w for w in BANNED_ADVERBS if w not in text.lower()]
    if missing:
        raise ValueError(f"Writer prompt template is missing banned adverbs: {missing}")
    rule_count = text.count(BELIEF_PERSISTENCE_RULE)
    if rule_count != 1:
        raise ValueError(
            f"Writer prompt template must contain BELIEF_PERSISTENCE_RULE byte-identically exactly once; found {rule_count}."
        )
    if target == WRITER_PROMPT_PATH:
        verify_prompt_and_checker_requirements_aligned(text)
    return text


def format_writer_prompt(world: World, focal_agent: str, template: str | None = None) -> str:
    """Format the `qwen3.6:35b` writer prompt for `world` and `focal_agent`.

    Strips internal `fact_id` and `causal_preconditions` fields so no `F0`..`F9` markers leak
    into the prompt, and verifies that `BELIEF_PERSISTENCE_RULE` appears byte-identically once.
    """
    raw_template = template if template is not None else load_writer_prompt_template()
    world_setup = {
        "scenario_id": world.scenario_id,
        "domain": world.domain_name,
        "rooms": list(world.rooms),
        "doorways": [list(edge) for edge in world.doorways],
        "containers_by_room": {r: list(c) for r, c in world.containers_by_room.items()},
        "occluders_by_room": dict(world.occluders_by_room),
        "fixtures_by_room": dict(world.fixtures_by_room),
        "noise_threshold": world.noise_threshold,
        "agents": list(world.agents),
        "absent_parties": list(world.absent_parties),
        "initial_agent_rooms": dict(world.initial_agent_rooms),
        "stake_edges": [
            {"agent": s.agent, "variable": s.variable, "description": s.description}
            for s in world.stake_edges
        ],
        "goals": {
            ag: {
                "goal_text": g.goal_text,
            }
            for ag, g in world.goals.items()
        },
    }
    event_log: list[dict[str, object]] = []
    for idx, ev in enumerate(world.events, start=1):
        ev_entry: dict[str, object] = {
            "order_index": idx,
            "kind": ev.kind,
            "actor": ev.actor,
            "room": ev.room,
        }
        if ev.target_object is not None:
            ev_entry["target_object"] = ev.target_object
        if ev.from_location is not None:
            ev_entry["from_location"] = ev.from_location
        if ev.to_location is not None:
            ev_entry["to_location"] = ev.to_location
        if ev.utterance is not None:
            ev_entry["utterance"] = ev.utterance
        if ev.recipient is not None:
            ev_entry["recipient"] = ev.recipient
        if ev.occluded:
            ev_entry["occluded"] = True
        if ev.occluder is not None:
            ev_entry["occluder"] = ev.occluder
        if ev.noise_level:
            ev_entry["noise_level"] = ev.noise_level
        if ev.is_bluff:
            ev_entry["is_bluff"] = True
        if ev.stated_reason is not None:
            ev_entry["stated_reason"] = ev.stated_reason
        event_log.append(ev_entry)

    expanded_template = (
        raw_template.replace(HARD_CONSTRAINTS_PLACEHOLDER, render_writer_prompt_constraints())
        if HARD_CONSTRAINTS_PLACEHOLDER in raw_template
        else raw_template
    )
    formatted = (
        expanded_template.replace("{focal_agent}", focal_agent)
        .replace("{noise_threshold}", str(world.noise_threshold))
        .replace("{world_setup_json}", json.dumps(world_setup, separators=(",", ":"), sort_keys=True))
        .replace("{event_log_json}", json.dumps(event_log, separators=(",", ":"), sort_keys=True))
    )
    rule_count = formatted.count(BELIEF_PERSISTENCE_RULE)
    if rule_count != 1:
        raise RenderCheckError(
            f"Formatted writer prompt must contain BELIEF_PERSISTENCE_RULE byte-identically once; found {rule_count}.",
            check_id="prompt_belief_rule_count",
        )
    return formatted


_WORD_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
_SILENT_H_PREFIXES: tuple[str, ...] = ("hour", "honest", "honor", "honour", "heir", "herbarium")
_CONSONANT_SOUND_VOWEL_PREFIXES: tuple[str, ...] = (
    "uni",
    "use",
    "usu",
    "ute",
    "uti",
    "uto",
    "eul",
    "eup",
    "eur",
    "ew",
    "one",
    "once",
)
_VOWEL_SOUND_ACRONYMS: frozenset[str] = frozenset({"f", "h", "l", "m", "n", "r", "s", "x"})


def indefinite_article(phrase: str) -> str:
    """Return `'a'` or `'an'` based on the phonetic sound of the first word in `phrase`."""
    match = _WORD_TOKEN_RE.search(phrase.strip())
    if not match:
        return "a"
    raw_word = match.group(0)
    word = raw_word.lower()

    if any(word.startswith(pref) for pref in _SILENT_H_PREFIXES):
        return "an"
    if any(word.startswith(pref) for pref in _CONSONANT_SOUND_VOWEL_PREFIXES):
        return "a"
    if len(word) == 1 and word in _VOWEL_SOUND_ACRONYMS:
        return "an"
    if len(raw_word) >= 2 and raw_word.isupper() and word[0] in _VOWEL_SOUND_ACRONYMS:
        return "an"
    if word[0] in ("a", "e", "i", "o", "u"):
        return "an"
    return "a"


_SPEECH_AND_MESSAGE_VERBS_RE = re.compile(
    r"\b(?:"
    + "|".join(
        re.escape(v)
        for v in (_ACTION_VERBS_BY_KIND["say"] + _ACTION_VERBS_BY_KIND["message"])
    )
    + r")\b",
    re.IGNORECASE,
)


def _clause_until_next_unquoted_sentence_end(text: str) -> str:
    """Return the prefix of `text` up to the first sentence-ending `.!?` outside double quotes."""
    in_q = False
    out: list[str] = []
    for i, ch in enumerate(text):
        out.append(ch)
        if ch in ('"', "“", "”"):
            in_q = not in_q
        elif not in_q and ch in (".", "!", "?"):
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if not nxt or nxt.isspace():
                break
    return "".join(out)


def _split_prose_sentences(prose: str) -> list[str]:
    """Split `prose` into sentences without splitting on periods inside quoted speech when attribution follows."""
    sentences: list[str] = []
    current: list[str] = []
    pre_quote_outside: list[str] = []
    in_quote = False
    idx = 0
    n = len(prose)
    while idx < n:
        ch = prose[idx]
        if ch in ('"', "“", "”"):
            current.append(ch)
            prev_ch = prose[idx - 1] if idx > 0 else ""
            in_quote = not in_quote
            if not in_quote and prev_ch in (".", "!", "?"):
                after = prose[idx + 1 :]
                if not after or after[0].isspace():
                    rest = after.lstrip()
                    if not rest:
                        sent = "".join(current).strip()
                        if sent:
                            sentences.append(sent)
                        current = []
                        pre_quote_outside = []
                        break
                    pre_outside_str = "".join(pre_quote_outside)
                    next_clause = _clause_until_next_unquoted_sentence_end(rest)
                    attribution_follows = (
                        rest[0].islower()
                        or rest[0] in (",", ";", "—", "-")
                        or (
                            _SPEECH_AND_MESSAGE_VERBS_RE.search(pre_outside_str) is None
                            and _SPEECH_AND_MESSAGE_VERBS_RE.search(next_clause) is not None
                        )
                    )
                    if not attribution_follows:
                        sent = "".join(current).strip()
                        if sent:
                            sentences.append(sent)
                        current = []
                        pre_quote_outside = []
                        while idx + 1 < n and prose[idx + 1].isspace():
                            idx += 1
            idx += 1
            continue

        current.append(ch)
        if not in_quote:
            pre_quote_outside.append(ch)
            if ch in (".", "!", "?"):
                nxt = prose[idx + 1] if idx + 1 < n else ""
                if nxt in ('"', "'", "”", "’"):
                    if nxt in ("'", "’"):
                        after_q = prose[idx + 2 :]
                        if not after_q or after_q[0].isspace():
                            current.append(nxt)
                            idx += 1
                            sent = "".join(current).strip()
                            if sent:
                                sentences.append(sent)
                            current = []
                            pre_quote_outside = []
                            while idx + 1 < n and prose[idx + 1].isspace():
                                idx += 1
                elif not nxt or nxt.isspace():
                    sent = "".join(current).strip()
                    if sent:
                        sentences.append(sent)
                    current = []
                    pre_quote_outside = []
                    while idx + 1 < n and prose[idx + 1].isspace():
                        idx += 1
        idx += 1

    tail = "".join(current).strip()
    if tail:
        sentences.append(tail)
    return sentences


def _is_setup_sentence(sentence: str) -> bool:
    s_low = sentence.lower()
    return any(
        marker in s_low
        for marker in (
            "contains the ",
            "doorways link the rooms",
            "each agent assumes an object remains",
            "at the start,",
            "absent off-site",
            "stated goals:",
            "depends on the location",
            "depends on the baseline",
        )
    )


def _actor_precedes_verb_as_subject(
    s_low: str,
    actor: str,
    action_verbs: Sequence[str],
    all_parties: Sequence[str],
    *,
    ev: Event | None = None,
) -> bool:
    """Return True if `actor` appears as the subject immediately preceding an unnegated action verb in `s_low`."""
    if not action_verbs:
        return bool(re.search(rf"\b{re.escape(actor.lower())}\b", s_low))
    verb_alt = "|".join(re.escape(v) for v in action_verbs)
    actor_esc = re.escape(actor.lower())
    for m_act in re.finditer(rf"\b{actor_esc}\b", s_low):
        # Ignore possessive occurrences such as "From Ana's station,"
        after_actor = s_low[m_act.end() :]
        if after_actor.startswith("'s") or after_actor.startswith("’s"):
            continue
        for m_verb in re.finditer(rf"\b(?:{verb_alt})\b", after_actor):
            between = after_actor[: m_verb.start()]
            # Allow bare "see" as a sighting verb ONLY after "could not" on an occluded see
            if m_verb.group(0).lower() == "see":
                if re.search(r"\bcould\s+not\s+$", between) is None:
                    continue
                if ev is not None and not (ev.kind == "see" and ev.occluded and ev.occluder):
                    continue
            # No other party name (or second occurrence of actor) may intervene as the closer subject before the verb
            intervening_party = any(
                re.search(rf"\b{re.escape(p.lower())}\b", between)
                for p in all_parties
            )
            if intervening_party:
                continue
            if ev is not None and _verb_is_scoped_by_negation(s_low, m_act.start(), m_act.end() + m_verb.start(), m_act.end() + m_verb.end(), ev):
                continue
            return True
    return False


def _post_verb_core_args_end(frame_low: str, verb_end: int, ev: Event) -> int:
    """Return the index (`>= verb_end`) in `frame_low` where `ev`'s post-verb object and location arguments end."""
    after_verb = frame_low[verb_end:]
    core_args: list[str] = []
    if ev.kind in ("enter", "leave"):
        if ev.from_location:
            core_args.append(ev.from_location)
        if ev.to_location:
            core_args.append(ev.to_location)
    elif ev.kind == "move":
        for piece in (ev.target_object, ev.from_location, ev.to_location, ev.occluder if ev.occluded else None):
            if piece:
                core_args.append(piece)
    elif ev.kind == "see":
        if ev.target_object:
            core_args.append(ev.target_object)
        if ev.occluded and ev.occluder:
            core_args.extend((ev.occluder, FAR_SIDE_ALCOVE_STATION))
        elif ev.to_location:
            core_args.append(ev.to_location)
    elif ev.kind == "say":
        core_args.append(f"{ev.noise_level} db")
    elif ev.kind == "message" and ev.recipient:
        core_args.append(ev.recipient)

    max_rel_end = 0
    for arg in core_args:
        m = re.search(rf"\b{re.escape(arg.lower())}\b", after_verb)
        if m is not None and m.end() > max_rel_end:
            max_rel_end = m.end()
    return verb_end + max_rel_end


def _adjunct_negates_event_verb_object_or_location(clause_low: str, ev: Event) -> bool:
    """Return True if a post-argument `without <gerund>` or relative clause negates `ev`'s own verb, target object, or location."""
    kind_gerunds = _GERUNDS_BY_KIND.get(ev.kind, frozenset())
    m_gerund = _POST_ARG_GERUND_CLAUSE_RE.match(clause_low.strip())
    if m_gerund is not None:
        gerund_word = m_gerund.group(1).lower()
        if gerund_word in kind_gerunds:
            return True
    else:
        for g_word in kind_gerunds:
            if re.search(rf"\b{re.escape(g_word)}\b", clause_low):
                return True
        kind_verbs = set(_ACTION_VERBS_BY_KIND.get(ev.kind, ())) | set(_BASE_VERBS_BY_KIND.get(ev.kind, frozenset()))
        if ev.kind in ("enter", "leave"):
            kind_verbs |= (
                set(_ACTION_VERBS_BY_KIND["enter"])
                | set(_ACTION_VERBS_BY_KIND["leave"])
                | set(_BASE_VERBS_BY_KIND["enter"])
                | set(_BASE_VERBS_BY_KIND["leave"])
            )
        for v_word in kind_verbs:
            if re.search(rf"\b{re.escape(v_word)}\b", clause_low):
                return True

    for piece in (ev.target_object, ev.from_location, ev.to_location, ev.room):
        if piece and re.search(rf"\b{re.escape(piece.lower())}\b", clause_low):
            return True
    return False


def _strip_benign_post_arg_adjuncts(tail_low: str, ev: Event) -> str:
    """Strip post-argument `without <gerund>` and relative clauses from `tail_low` only when they do not negate `ev`'s verb, object, or location."""
    cleaned = _POST_ARG_BENIGN_NEG_RE.sub("", tail_low)

    def _replace_gerund(m: re.Match[str]) -> str:
        clause = m.group(0)
        if _adjunct_negates_event_verb_object_or_location(clause, ev):
            return clause
        return ""

    def _replace_rel(m: re.Match[str]) -> str:
        clause = m.group(0)
        if _NEGATION_PATTERN.search(clause) is not None and _adjunct_negates_event_verb_object_or_location(clause, ev):
            return clause
        return ""

    cleaned = _POST_ARG_GERUND_CLAUSE_RE.sub(_replace_gerund, cleaned)
    cleaned = _POST_ARG_REL_CLAUSE_RE.sub(_replace_rel, cleaned)
    return cleaned


def _verb_is_scoped_by_negation(
    frame_low: str,
    actor_start: int,
    verb_start: int,
    verb_end: int,
    ev: Event,
) -> bool:
    """Return True if the event verb at `[verb_start, verb_end)` is scoped by a negation from `NEGATION_PHRASES`."""
    # 1. Before the actor: refuse sentence-level negation unless the fronted part is a plain time or place phrase.
    before_actor = frame_low[:actor_start]
    cleaned_before = _ADJACENT_ROOMS_COULD_NOT_HEAR_RE.sub("", before_actor)
    cleaned_before = _FRONTED_TIME_PLACE_NEG_RE.sub("", cleaned_before)
    if _NEGATION_PATTERN.search(cleaned_before) is not None:
        return True

    # 2. From the actor through the event verb `[actor_start, verb_end)`: any negation directly scopes the verb
    # (except "could not see" on an occluded see).
    actor_to_verb = frame_low[actor_start:verb_start] + frame_low[verb_start:verb_end]
    cleaned_actor_to_verb = _ADJACENT_ROOMS_COULD_NOT_HEAR_RE.sub("", actor_to_verb)
    if ev.kind == "see" and ev.occluded and ev.occluder:
        cleaned_actor_to_verb = _COULD_NOT_SEE_RE.sub("", cleaned_actor_to_verb)
    if _NEGATION_PATTERN.search(cleaned_actor_to_verb) is not None:
        return True

    # 3. After `verb_end`: negation before the event's object and location (`[verb_end, core_end)`) scopes the VP
    # and is refused; after `core_end`, non-scoping clauses/adjuncts are allowed unless they negate the event's verb, object, or location.
    core_end = _post_verb_core_args_end(frame_low, verb_end, ev)
    core_span = frame_low[verb_end:core_end]
    cleaned_core = _ADJACENT_ROOMS_COULD_NOT_HEAR_RE.sub("", core_span)
    if ev.kind == "see" and ev.occluded and ev.occluder:
        cleaned_core = _COULD_NOT_SEE_RE.sub("", cleaned_core)
    if _NEGATION_PATTERN.search(cleaned_core) is not None:
        return True

    post_arg_tail = frame_low[core_end:]
    cleaned_tail = _ADJACENT_ROOMS_COULD_NOT_HEAR_RE.sub("", post_arg_tail)
    if ev.kind == "see" and ev.occluded and ev.occluder:
        cleaned_tail = _COULD_NOT_SEE_RE.sub("", cleaned_tail)
    cleaned_tail = _strip_benign_post_arg_adjuncts(cleaned_tail, ev)
    if _NEGATION_PATTERN.search(cleaned_tail) is not None:
        return True
    return False


def _ordered_from_to_positions(
    s_low: str,
    from_loc: str,
    to_loc: str,
    *,
    require_from_prep: bool = True,
) -> bool:
    """Return True if `from_loc` appears in the source phrase strictly before `to_loc` in the destination phrase."""
    from_esc = re.escape(from_loc.lower())
    to_esc = re.escape(to_loc.lower())
    if require_from_prep:
        from_matches = list(
            re.finditer(rf"\b(?:from|out of|exited|left|departed)\s+{_ARTICLE_OPT_RE}{from_esc}\b", s_low)
        )
    else:
        from_matches = list(re.finditer(rf"\b{from_esc}\b", s_low))
    if not from_matches:
        return False

    to_matches = list(
        re.finditer(
            rf"\b(?:to|into|toward|towards|for|onto|in|at|behind|around|entered|arrived\s+in)\s+{_ARTICLE_OPT_RE}{to_esc}\b",
            s_low,
        )
    )
    if not to_matches:
        return False

    if re.search(rf"\b(?:entered|arrived\s+in)\s+{_ARTICLE_OPT_RE}{to_esc}\b.*\b(?:from|out of)\s+{_ARTICLE_OPT_RE}{from_esc}\b", s_low):
        return True

    for fm in from_matches:
        for tm in to_matches:
            if fm.end() <= tm.start():
                return True
    return False


def _has_forbidden_event_negation(
    frame_low: str,
    ev: Event,
    all_parties: Sequence[str] = (),
    *,
    action_verbs: Sequence[str] | None = None,
) -> bool:
    """Return True if the event's verb in `frame_low` is scoped by any negation from `NEGATION_PHRASES` outside the allowed occluded-see blocking clause."""
    parties = all_parties if all_parties else (ev.actor, *([ev.recipient] if ev.recipient else []))
    verbs = action_verbs if action_verbs is not None else _ACTION_VERBS_BY_KIND.get(ev.kind, ())
    if not _actor_precedes_verb_as_subject(frame_low, ev.actor, verbs, parties):
        cleaned = _ADJACENT_ROOMS_COULD_NOT_HEAR_RE.sub("", frame_low)
        if ev.kind == "see" and ev.occluded and ev.occluder:
            cleaned = _COULD_NOT_SEE_RE.sub("", cleaned)
        return _NEGATION_PATTERN.search(cleaned) is not None
    return not _actor_precedes_verb_as_subject(frame_low, ev.actor, verbs, parties, ev=ev)


def _sentence_has_folded_from_and_arrival(
    s_low: str,
    from_loc: str,
    to_loc: str,
) -> bool:
    """Return True if `s_low` explicitly names both `from <from_loc>` and arrival at `<to_loc>` (`to/into/onto/in/at/entered/arrived in <to_loc>`)."""
    from_esc = re.escape(from_loc.lower())
    to_esc = re.escape(to_loc.lower())
    if not re.search(rf"\bfrom\s+{_ARTICLE_OPT_RE}{from_esc}\b", s_low):
        return False
    if not re.search(
        rf"\b(?:to|into|onto|in|at|entered|arrived\s+in)\s+{_ARTICLE_OPT_RE}{to_esc}\b",
        s_low,
    ):
        return False
    return _ordered_from_to_positions(
        s_low,
        from_loc,
        to_loc,
        require_from_prep=True,
    )


def _preceding_sentence_is_matching_leave(
    prev_sentence: str | None,
    prev_event: Event | None,
    ev: Event,
    noise_threshold: int,
    world: World | None = None,
) -> bool:
    """Return True if `prev_event` and `prev_sentence` form the matching `leave` by the same actor naming `ev.from_location`."""
    if prev_sentence is None or prev_event is None:
        return False
    if (
        prev_event.kind != "leave"
        or prev_event.actor != ev.actor
        or not ev.from_location
        or not prev_event.from_location
        or prev_event.from_location.lower() != ev.from_location.lower()
        or prev_event.room.lower() != ev.from_location.lower()
        or not ev.to_location
        or not prev_event.to_location
        or prev_event.to_location.lower() != ev.to_location.lower()
    ):
        return False
    if not _sentence_matches_event(prev_sentence, prev_event, noise_threshold, world=world):
        return False
    return bool(re.search(rf"\b{re.escape(ev.from_location.lower())}\b", prev_sentence.lower()))


def _sentence_matches_folded_leave_enter(
    prev_sentence: str | None,
    prev_event: Event | None,
    ev: Event,
    noise_threshold: int,
    world: World | None = None,
) -> bool:
    """Return True if `prev_sentence` is a folded movement sentence satisfying both consecutive `prev_event` (leave) and `ev` (enter)."""
    if (
        prev_sentence is None
        or prev_event is None
        or ev.kind != "enter"
        or prev_event.kind != "leave"
        or prev_event.actor != ev.actor
        or not ev.from_location
        or not prev_event.from_location
        or prev_event.from_location.lower() != ev.from_location.lower()
        or prev_event.room.lower() != ev.from_location.lower()
        or not ev.to_location
        or not prev_event.to_location
        or prev_event.to_location.lower() != ev.to_location.lower()
    ):
        return False
    if not _sentence_matches_event(prev_sentence, prev_event, noise_threshold, world=world):
        return False
    if not _sentence_matches_event(prev_sentence, ev, noise_threshold, world=world):
        return False
    return _sentence_has_folded_from_and_arrival(
        prev_sentence.lower(),
        ev.from_location,
        ev.to_location,
    )


def _sentence_matches_event(
    sentence: str,
    ev: Event,
    noise_threshold: int,
    world: World | None = None,
    *,
    prev_sentence: str | None = None,
    prev_event: Event | None = None,
    next_sentence: str | None = None,
    next_event: Event | None = None,
) -> bool:
    """Return True if `sentence` states `ev` with role-bound actor, action, object, ordered from/to, and room."""
    s_low = sentence.lower()
    s_unquoted_low = re.sub(r'"[^"]*"|“[^”]*”', '""', s_low)
    frame_low = s_unquoted_low if ev.kind in ("say", "message") else s_low
    all_parties: Sequence[str] = (
        tuple(world.agents) + tuple(world.absent_parties)
        if world is not None
        else (ev.actor, *([ev.recipient] if ev.recipient else []))
    )
    is_folded_movement = (
        ev.kind in ("enter", "leave")
        and bool(ev.from_location and ev.to_location)
        and _sentence_has_folded_from_and_arrival(
            s_low,
            ev.from_location or "",
            ev.to_location or "",
        )
    )
    if (
        is_folded_movement
        and ev.kind == "leave"
        and next_sentence is not None
        and next_event is not None
        and next_event.kind == "enter"
        and next_event.actor == ev.actor
        and next_event.from_location
        and next_event.to_location
        and _actor_precedes_verb_as_subject(
            next_sentence.lower(),
            next_event.actor,
            _FOLDED_LEAVE_ENTER_VERBS,
            all_parties,
            ev=next_event,
        )
        and _sentence_has_folded_from_and_arrival(
            next_sentence.lower(),
            next_event.from_location,
            next_event.to_location,
        )
    ):
        is_folded_movement = False
    action_verbs: Sequence[str] = (
        _FOLDED_LEAVE_ENTER_VERBS
        if is_folded_movement
        else _ACTION_VERBS_BY_KIND.get(ev.kind, ())
    )
    if _has_forbidden_event_negation(frame_low, ev, all_parties, action_verbs=action_verbs):
        return False
    if not _actor_precedes_verb_as_subject(frame_low, ev.actor, action_verbs, all_parties, ev=ev):
        return False

    if ev.target_object:
        target_esc = re.escape(ev.target_object.lower())
        if re.search(rf"\bno\s+{_ARTICLE_OPT_RE}{target_esc}\b", frame_low):
            return False

    if ev.room.lower() not in frame_low:
        if ev.kind not in ("say", "message"):
            return False
    else:
        # Reject if another room from world.rooms is asserted as the room of an intra-room action
        if world is not None and ev.kind not in ("enter", "leave"):
            other_rooms = [r for r in world.rooms if r.lower() != ev.room.lower()]
            for o_room in other_rooms:
                if o_room.lower() in frame_low:
                    return False
            room_esc = re.escape(ev.room.lower())
            # Strip any "not far from [the] <room>" before checking that the event's room is stated as the location
            frame_for_room = re.sub(
                rf"\bnot\s+far\s+from\s+{_ARTICLE_OPT_RE}{room_esc}\b",
                "",
                frame_low,
            )
            if not re.search(rf"\b(?:in|from|inside|at|of)\s+{_ARTICLE_OPT_RE}{room_esc}\b", frame_for_room):
                return False

    if ev.kind in ("say", "message") and ev.room.lower() not in frame_low and world is not None:
        other_rooms = [r for r in world.rooms if r.lower() != ev.room.lower()]
        for o_room in other_rooms:
            if o_room.lower() in frame_low:
                return False

    if ev.kind == "enter":
        if not (ev.from_location and ev.to_location):
            return False
        if world is not None:
            allowed_rooms = {ev.room.lower(), ev.from_location.lower(), ev.to_location.lower()}
            for o_room in world.rooms:
                if o_room.lower() not in allowed_rooms and o_room.lower() in s_low:
                    return False
        if _ordered_from_to_positions(
            s_low,
            ev.from_location,
            ev.to_location,
            require_from_prep=True,
        ):
            # Prevent two consecutive folded movement sentences from covering one logged leave+enter pair:
            if (
                prev_sentence is not None
                and prev_event is not None
                and _sentence_matches_folded_leave_enter(
                    prev_sentence, prev_event, ev, noise_threshold, world=world
                )
            ):
                return False
            return True
        to_esc = re.escape(ev.to_location.lower())
        has_to_dest = bool(
            re.search(
                rf"\b(?:to|into|toward|towards|for|onto|in|at|entered|arrived\s+in)\s+{_ARTICLE_OPT_RE}{to_esc}\b",
                s_low,
            )
        )
        if has_to_dest and _preceding_sentence_is_matching_leave(
            prev_sentence, prev_event, ev, noise_threshold, world=world
        ):
            return True
        return False

    if ev.kind == "leave":
        if not (ev.from_location and ev.to_location):
            return False
        if world is not None:
            allowed_rooms = {ev.room.lower(), ev.from_location.lower(), ev.to_location.lower()}
            for o_room in world.rooms:
                if o_room.lower() not in allowed_rooms and o_room.lower() in s_low:
                    return False
        return _ordered_from_to_positions(
            s_low,
            ev.from_location,
            ev.to_location,
            require_from_prep=False,
        )

    if ev.kind == "move":
        if is_station_transition_event(ev):
            if not (
                ev.target_object
                and ev.target_object.lower() in s_low
                and ev.from_location
                and ev.to_location
            ):
                return False
            return _ordered_from_to_positions(
                s_low,
                ev.from_location,
                ev.to_location,
                require_from_prep=True,
            )
        if not (
            ev.target_object
            and ev.target_object.lower() in s_low
            and ev.from_location
            and ev.to_location
        ):
            return False
        if not _ordered_from_to_positions(
            s_low,
            ev.from_location,
            ev.to_location,
            require_from_prep=True,
        ):
            return False
        if ev.occluded and ev.occluder:
            if ev.occluder.lower() not in s_low or (
                "blocked" not in s_low and "blocking sight" not in s_low
            ):
                return False
        else:
            if any(bp in s_low for bp in BLOCKING_PHRASES):
                return False
        return True

    if ev.kind == "see":
        if not _actor_precedes_verb_as_subject(frame_low, ev.actor, SIGHTING_VERBS, all_parties, ev=ev):
            return False
        if not (ev.target_object and ev.target_object.lower() in s_low):
            return False
        if ev.occluded and ev.occluder:
            return (
                ev.occluder.lower() in s_low
                and ("blocked" in s_low or "blocking sight" in s_low)
                and FAR_SIDE_ALCOVE_STATION.lower() in s_low
            )
        if any(bp in s_low for bp in BLOCKING_PHRASES):
            return False
        all_containers = (
            {c.lower() for pair in world.containers_by_room.values() for c in pair}
            if world is not None
            else {c.lower() for dom in DOMAINS for pair in dom.containers for c in pair}
        )
        if ev.to_location and ev.to_location.lower() in all_containers:
            if ev.to_location.lower() not in s_low:
                return False
            for other_c in all_containers:
                if other_c != ev.to_location.lower() and other_c in s_low:
                    return False
        return True

    if ev.kind == "say":
        if f"{ev.noise_level} db" not in frame_low:
            return False
        if ev.utterance:
            utt_core = ev.utterance.rstrip(".!?").lower()
            if utt_core not in s_low:
                return False
        return True

    if ev.kind == "message":
        if not ev.recipient:
            return False
        recip_esc = re.escape(ev.recipient.lower())
        if not re.search(rf"\b(?:to|for|messaged|texted|notified)\s+{recip_esc}\b", frame_low):
            return False
        if ev.utterance:
            utt_core = ev.utterance.rstrip(".!?").lower()
            if utt_core not in s_low:
                return False
        return True

    return False


def _find_sentence_with_substring(prose: str, substring: str) -> str:
    """Return the first sentence in `prose` containing `substring` (case-insensitive), or `'no candidate sentence'`."""
    sub_low = substring.lower()
    for para in prose.split("\n\n"):
        for sent in _split_prose_sentences(para.strip()):
            if sub_low in sent.lower():
                return sent
    return "no candidate sentence"


def check_prose_no_markers(prose: str) -> None:
    """Prose check (a): refuse if any bracket marker `[...]`, fact ID `F0`..`F9`, or `step N` survives."""
    bracket_match = _BRACKET_MARKER_RE.search(prose)
    if bracket_match is not None:
        cand = _find_sentence_with_substring(prose, bracket_match.group(0))
        _raise_prose_check_error(
            "a_no_markers_bracket",
            f"Render check (a) failed: surviving bracket marker {bracket_match.group(0)!r} in prose (candidate sentence: {cand!r}).",
            failing_sentence=cand,
        )
    fact_match = _FACT_ID_RE.search(prose)
    if fact_match is not None:
        cand = _find_sentence_with_substring(prose, fact_match.group(0))
        _raise_prose_check_error(
            "a_no_markers_fact_id",
            f"Render check (a) failed: surviving fact ID {fact_match.group(0)!r} in prose (candidate sentence: {cand!r}).",
            failing_sentence=cand,
        )
    step_match = _STEP_NUMBER_RE.search(prose)
    if step_match is not None:
        cand = _find_sentence_with_substring(prose, step_match.group(0))
        _raise_prose_check_error(
            "a_no_markers_step_number",
            f"Render check (a) failed: surviving step number {step_match.group(0)!r} in prose (candidate sentence: {cand!r}).",
            failing_sentence=cand,
        )


def check_prose_banned_adverbs(prose: str) -> tuple[str, ...]:
    """Prose check (b): refuse if any banned mental/emotional adverb from `BANNED_ADVERBS` appears."""
    hits = tuple(dict.fromkeys(m.group(1).lower() for m in _BANNED_PATTERN.finditer(prose)))
    if hits:
        cand = _find_sentence_with_substring(prose, hits[0])
        _raise_prose_check_error(
            "b_banned_adverbs",
            f"Render check (b) failed: prose contains banned unlogged mental/emotional adverb(s): {hits} (candidate sentence: {cand!r}).",
            failing_sentence=cand,
        )
    return ()


def check_prose_rule_and_hearing_threshold(prose: str, world: World) -> None:
    """Prose check (d): refuse unless the belief-persistence rule sentence and hearing threshold appear in the setup prose."""
    paragraphs = [p.strip() for p in prose.split("\n\n") if p.strip()]
    setup_text = paragraphs[0] if paragraphs else prose
    rule_count = setup_text.count(BELIEF_PERSISTENCE_RULE)
    if rule_count != 1:
        cand = _find_sentence_with_substring(setup_text, "assumes")
        _raise_prose_check_error(
            "d_rule_sentence",
            f"Render check (d) failed: setup prose must contain BELIEF_PERSISTENCE_RULE byte-identically once (found {rule_count}; candidate sentence: {cand!r}).",
            failing_sentence=cand,
        )
    threshold_str = f"{world.noise_threshold} dB"
    if threshold_str.lower() not in setup_text.lower() or "doorway" not in setup_text.lower():
        cand = _find_sentence_with_substring(setup_text, "doorway")
        _raise_prose_check_error(
            "d_hearing_threshold",
            f"Render check (d) failed: setup prose is missing the doorway hearing threshold '{threshold_str}' (candidate sentence: {cand!r}).",
            failing_sentence=cand,
        )


def check_prose_render_fidelity(
    prose: str,
    world: World,
    focal_agent: str | None = None,
) -> None:
    """Prose check (c): validate complete event-log and setup fidelity using `world` (never keys).

    Verifies:
    1. Setup states every room, container pair, occluder (between `far-side alcove` and `container bay`),
       doorway edge (both room names in one setup sentence), initial agent room (each agent and starting
       room in one setup sentence), absent party, verbatim goal, and verbatim stake edge.
    2. Every event in `world.events` is stated in the prose with its actor, action, object, place,
       and room in exact relative chronological order.
    3. Nothing the event log lacks is asserted (no unlogged party names, rooms, containers, objects,
       unlogged motives on unknowable `why` actions, unlogged note placers on unknowable `who` actions,
       or extra unlogged event sentences).
    """
    prose_low = prose.lower()
    paragraphs = [p.strip() for p in prose.split("\n\n") if p.strip()]
    setup_text = paragraphs[0] if paragraphs else prose
    setup_low = setup_text.lower()
    setup_sents = _split_prose_sentences(setup_text)
    setup_sents_low = [s.lower() for s in setup_sents]

    if focal_agent is not None and focal_agent.lower() not in setup_low:
        cand = setup_sents[0] if setup_sents else "no candidate sentence"
        _raise_prose_check_error(
            "c_fidelity_focal_agent",
            f"Render check (c) failed: focal_agent '{focal_agent}' is missing from setup prose (candidate sentence: {cand!r}).",
            failing_sentence=cand,
        )

    # 1. Verify per-room containers and occluders in setup
    for room in world.rooms:
        if room.lower() not in setup_low:
            cand = setup_sents[0] if setup_sents else "no candidate sentence"
            _raise_prose_check_error(
                "c_fidelity_room_setup",
                f"Render check (c) failed: room '{room}' is missing from setup prose (candidate sentence: {cand!r}).",
                failing_sentence=cand,
            )
        c0, c1 = world.containers_by_room[room]
        if c0.lower() not in setup_low or c1.lower() not in setup_low:
            cand = _find_sentence_with_substring(setup_text, room)
            _raise_prose_check_error(
                "c_fidelity_containers",
                f"Render check (c) failed: containers ({c0!r}, {c1!r}) for room '{room}' missing from setup prose (candidate sentence: {cand!r}).",
                failing_sentence=cand,
            )
        fixture = world.fixtures_by_room.get(room)
        if fixture and fixture.lower() not in setup_low:
            cand = _find_sentence_with_substring(setup_text, room)
            _raise_prose_check_error(
                "c_fidelity_fixtures",
                f"Render check (c) failed: fixture '{fixture}' for room '{room}' missing from setup prose (candidate sentence: {cand!r}).",
                failing_sentence=cand,
            )
        occ = world.occluders_by_room[room]
        if occ.lower() not in setup_low:
            cand = _find_sentence_with_substring(setup_text, room)
            _raise_prose_check_error(
                "c_fidelity_occluder",
                f"Render check (c) failed: occluder '{occ}' for room '{room}' missing from setup prose (candidate sentence: {cand!r}).",
                failing_sentence=cand,
            )

    if CONTAINER_BAY_STATION.lower() not in setup_low or FAR_SIDE_ALCOVE_STATION.lower() not in setup_low:
        cand = setup_sents[0] if setup_sents else "no candidate sentence"
        _raise_prose_check_error(
            "c_fidelity_stations",
            f"Render check (c) failed: station names ('container bay', 'far-side alcove') missing from setup prose (candidate sentence: {cand!r}).",
            failing_sentence=cand,
        )

    # Verify doorways: both room names of each edge must appear in one setup sentence
    for r_a, r_b in world.doorways:
        if not any(r_a.lower() in s_low and r_b.lower() in s_low for s_low in setup_sents_low):
            cand = _find_sentence_with_substring(setup_text, "doorway")
            _raise_prose_check_error(
                "c_fidelity_doorways",
                f"Render check (c) failed: doorway ({r_a!r}, {r_b!r}) missing from setup prose (candidate sentence: {cand!r}).",
                failing_sentence=cand,
            )

    # Verify initial agent rooms (agent name and starting room in the same setup sentence), absent parties, goals, and stakes
    for ag in world.agents:
        init_r = world.initial_agent_rooms[ag]
        if not any(
            re.search(rf"\b{re.escape(ag.lower())}\b", s_low) and init_r.lower() in s_low
            for s_low in setup_sents_low
        ):
            cand = _find_sentence_with_substring(setup_text, ag)
            _raise_prose_check_error(
                "c_fidelity_initial_room",
                f"Render check (c) failed: initial room '{init_r}' for agent '{ag}' missing from setup prose (candidate sentence: {cand!r}).",
                failing_sentence=cand,
            )
        goal_txt = world.goals[ag].goal_text.lower()
        if goal_txt not in setup_low:
            cand = _find_sentence_with_substring(setup_text, ag)
            _raise_prose_check_error(
                "c_fidelity_goal",
                f"Render check (c) failed: goal '{world.goals[ag].goal_text}' for agent '{ag}' missing from setup prose (candidate sentence: {cand!r}).",
                failing_sentence=cand,
            )

    for abs_p in world.absent_parties:
        if not re.search(rf"\b{re.escape(abs_p.lower())}\b", setup_low):
            cand = _find_sentence_with_substring(setup_text, "off-site")
            _raise_prose_check_error(
                "c_fidelity_absent_party",
                f"Render check (c) failed: absent party '{abs_p}' missing from setup prose (candidate sentence: {cand!r}).",
                failing_sentence=cand,
            )

    for stake in world.stake_edges:
        if stake.description.lower() not in setup_low:
            cand = _find_sentence_with_substring(setup_text, stake.agent)
            _raise_prose_check_error(
                "c_fidelity_stake_edge",
                f"Render check (c) failed: stake edge '{stake.description}' missing from setup prose (candidate sentence: {cand!r}).",
                failing_sentence=cand,
            )

    # 2. Verify no unlogged party names or foreign domain entities appear anywhere in the prose
    allowed_parties = set(world.agents) | set(world.absent_parties)
    for party_name in ALL_KNOWN_PARTY_NAMES - allowed_parties:
        if re.search(rf"\b{re.escape(party_name)}\b", prose):
            cand = _find_sentence_with_substring(prose, party_name)
            _raise_prose_check_error(
                "c_fidelity_unlogged_party",
                f"Render check (c) failed: unlogged party name '{party_name}' asserted in prose (candidate sentence: {cand!r}).",
                failing_sentence=cand,
            )

    current_domain = next((d for d in DOMAINS if d.name == world.domain_name), None)
    for dom in DOMAINS:
        if dom.name == world.domain_name:
            continue
        foreign_phrases = (
            *dom.rooms,
            *dom.objects,
            *(c for pair in dom.containers for c in pair),
            *dom.occluders,
            *dom.fixtures,
            dom.note_item,
            dom.side_mechanism,
            *dom.side_reasons,
        )
        for phrase in foreign_phrases:
            if phrase.lower() in prose_low:
                cand = _find_sentence_with_substring(prose, phrase)
                _raise_prose_check_error(
                    "c_fidelity_unlogged_entity",
                    f"Render check (c) failed: unlogged entity/phrase '{phrase}' from domain '{dom.name}' asserted in prose (candidate sentence: {cand!r}).",
                    failing_sentence=cand,
                )

    # Check that unlogged side_reasons or unlogged note placement are never asserted when unknowable
    if current_domain is not None:
        logged_reasons = {ev.stated_reason.lower() for ev in world.events if ev.stated_reason}
        for reason_str in current_domain.side_reasons:
            if reason_str.lower() not in logged_reasons and reason_str.lower() in prose_low:
                cand = _find_sentence_with_substring(prose, reason_str)
                _raise_prose_check_error(
                    "c_fidelity_unlogged_motive",
                    f"Render check (c) failed: unlogged motive '{reason_str}' asserted in prose (candidate sentence: {cand!r}).",
                    failing_sentence=cand,
                )
        note_moves = [
            ev
            for ev in world.events
            if ev.kind == "move" and ev.target_object == current_domain.note_item
        ]
        if not note_moves:
            for ag in allowed_parties:
                if (
                    f"{ag.lower()} placed the {current_domain.note_item.lower()}" in prose_low
                    or f"{ag.lower()} moved the {current_domain.note_item.lower()}" in prose_low
                ):
                    cand = _find_sentence_with_substring(prose, current_domain.note_item)
                    _raise_prose_check_error(
                        "c_fidelity_unlogged_placer",
                        f"Render check (c) failed: unlogged placement of '{current_domain.note_item}' by '{ag}' asserted in prose (candidate sentence: {cand!r}).",
                        failing_sentence=cand,
                    )

    # 3. Verify every event in `world.events` is stated in exact relative order and no extra event sentences exist
    # Paragraph 1 (`paragraphs[0]`) is structurally the setup paragraph; `paragraphs[1:]` are event paragraphs.
    if len(paragraphs) < 2:
        cand = setup_sents[-1] if setup_sents else "no candidate sentence"
        _raise_prose_check_error(
            "c_fidelity_paragraph_split",
            f"Render check (c) failed: prose must separate the setup paragraph from the event paragraph(s) with a blank line ('\\n\\n') (candidate sentence: {cand!r}).",
            failing_sentence=cand,
        )
    event_sentences: list[str] = []
    for ev_para in paragraphs[1:]:
        event_sentences.extend(_split_prose_sentences(ev_para))

    for sent in event_sentences:
        s_low = sent.lower()
        if "in open view" in s_low and "blocked" in s_low:
            _raise_prose_check_error(
                "c_fidelity_contradictory_occlusion",
                f"Render check (c) failed: contradictory visibility ('in open view' and 'blocked') in sentence: {sent!r}",
                failing_sentence=sent,
            )

    matched_sentence_indices: list[int] = []
    search_start = 0
    for ev_idx, ev in enumerate(world.events):
        prev_ev = world.events[ev_idx - 1] if ev_idx > 0 else None
        next_ev = world.events[ev_idx + 1] if ev_idx + 1 < len(world.events) else None
        found_idx: int | None = None
        if (
            ev.kind == "enter"
            and prev_ev is not None
            and matched_sentence_indices
            and search_start == matched_sentence_indices[-1] + 1
        ):
            s_prev = matched_sentence_indices[-1]
            if search_start < len(event_sentences) and _sentence_matches_event(
                event_sentences[search_start],
                ev,
                world.noise_threshold,
                world=world,
                prev_sentence=event_sentences[s_prev],
                prev_event=prev_ev,
            ):
                found_idx = search_start
            elif _sentence_matches_folded_leave_enter(
                event_sentences[s_prev],
                prev_ev,
                ev,
                world.noise_threshold,
                world=world,
            ):
                found_idx = s_prev
        if found_idx is None:
            for s_idx in range(search_start, len(event_sentences)):
                prev_sent = (
                    event_sentences[s_idx - 1]
                    if (
                        ev_idx > 0
                        and s_idx > 0
                        and matched_sentence_indices
                        and matched_sentence_indices[-1] == s_idx - 1
                    )
                    else None
                )
                next_sent = (
                    event_sentences[s_idx + 1]
                    if s_idx + 1 < len(event_sentences)
                    else None
                )
                if _sentence_matches_event(
                    event_sentences[s_idx],
                    ev,
                    world.noise_threshold,
                    world=world,
                    prev_sentence=prev_sent,
                    prev_event=prev_ev,
                    next_sentence=next_sent,
                    next_event=next_ev,
                ) and not (
                    ev.kind == "leave"
                    and ev_idx + 1 < len(world.events)
                    and world.events[ev_idx + 1].kind == "enter"
                    and world.events[ev_idx + 1].actor == ev.actor
                    and (
                        (
                            s_idx > search_start
                            and s_idx - 1 not in matched_sentence_indices
                            and ev.actor.lower() in event_sentences[s_idx - 1].lower()
                            and _sentence_matches_folded_leave_enter(
                                event_sentences[s_idx],
                                ev,
                                world.events[ev_idx + 1],
                                world.noise_threshold,
                                world=world,
                            )
                        )
                        or (
                            any(
                                i not in matched_sentence_indices
                                and _sentence_matches_event(
                                    event_sentences[i],
                                    ev,
                                    world.noise_threshold,
                                    world=world,
                                )
                                and not _sentence_matches_folded_leave_enter(
                                    event_sentences[i],
                                    ev,
                                    world.events[ev_idx + 1],
                                    world.noise_threshold,
                                    world=world,
                                )
                                for i in range(0, search_start)
                            )
                            and _sentence_matches_folded_leave_enter(
                                event_sentences[s_idx],
                                ev,
                                world.events[ev_idx + 1],
                                world.noise_threshold,
                                world=world,
                            )
                        )
                    )
                ):
                    found_idx = s_idx
                    break
        if found_idx is None:
            # Check if the event appears out of order earlier in event_sentences
            earlier_idx = next(
                (
                    i
                    for i in range(0, search_start)
                    if i not in matched_sentence_indices
                    and _sentence_matches_event(event_sentences[i], ev, world.noise_threshold, world=world)
                ),
                None,
            )
            if earlier_idx is not None:
                cand_sent = event_sentences[earlier_idx]
                _raise_prose_check_error(
                    "c_fidelity_event_order",
                    f"Render check (c) failed: event #{ev_idx + 1} ({ev.kind} by {ev.actor} in {ev.room}) is out of relative chronological order (candidate sentence: {cand_sent!r}).",
                    failing_sentence=cand_sent,
                    event_index=ev_idx + 1,
                    event_kind=ev.kind,
                )
            cand_sent = (
                event_sentences[search_start]
                if search_start < len(event_sentences)
                else "no candidate sentence"
            )
            _raise_prose_check_error(
                "c_fidelity_missing_event",
                f"Render check (c) failed: missing or altered event #{ev_idx + 1} (step={ev.step}, kind={ev.kind}, actor={ev.actor}, room={ev.room}, target={ev.target_object!r}, from={ev.from_location!r}, to={ev.to_location!r}; candidate sentence: {cand_sent!r}).",
                failing_sentence=cand_sent,
                event_index=ev_idx + 1,
                event_kind=ev.kind,
            )
        matched_sentence_indices.append(found_idx)
        search_start = found_idx + 1

    unmatched = [
        event_sentences[i]
        for i in range(len(event_sentences))
        if i not in set(matched_sentence_indices)
    ]
    if unmatched:
        cand_sent = unmatched[0]
        _raise_prose_check_error(
            "c_fidelity_unlogged_event",
            f"Render check (c) failed: prose asserts {len(unmatched)} unlogged event sentence(s) not in world.events: {unmatched[:2]} (candidate sentence: {cand_sent!r}).",
            failing_sentence=cand_sent,
        )


def check_rendered_prose(
    prose: str,
    world: World | None = None,
    focal_agent: str | None = None,
) -> tuple[str, ...]:
    """Run all prose checks (a)-(d) on `prose`, raising `RenderCheckError` on any violation.

    Checks:
    - (a) No markers survive: no `[...]` bracket markers, no fact IDs (`F0`–`F9`), no `"step N"`.
    - (b) Banned mental/emotional adverbs (`BANNED_ADVERBS`).
    - (c) Render fidelity against `world` (when `world` is provided): all events, actors, actions,
          objects, places, rooms, and relative order, with nothing unlogged asserted.
    - (d) Belief-persistence rule sentence and doorway hearing threshold in setup prose (when `world` is provided).
    """
    check_prose_banned_adverbs(prose)
    check_prose_no_markers(prose)
    if world is not None:
        check_prose_rule_and_hearing_threshold(prose, world)
        check_prose_render_fidelity(prose, world, focal_agent=focal_agent)
    return ()


def _render_event_sentence(event: Event, focal_agent: str, noise_threshold: int) -> str:
    """Render a single `Event` in third-person prose focalized on `focal_agent` without step numbers or markers."""
    prefix = (
        f"From {focal_agent}'s station"
        if event.actor == focal_agent
        else "Next"
    )

    if event.kind == "enter":
        return f"{prefix}, {event.actor} walked through the doorway from the {event.from_location} into the {event.room}."

    if event.kind == "leave":
        return f"{prefix}, {event.actor} exited the {event.room} heading toward the {event.to_location}."

    if event.kind == "move":
        if is_station_transition_event(event):
            direction_prep = (
                "behind" if event.to_location == FAR_SIDE_ALCOVE_STATION else "around"
            )
            return (
                f"{prefix}, in the {event.room}, {event.actor} stepped from the {event.from_location} "
                f"{direction_prep} the {event.target_object} into the {event.to_location}."
            )
        occ_clause = (
            f" behind the {event.occluder}, which blocked the line of sight for other occupants in the {event.room}"
            if event.occluded and event.occluder
            else f" in open view at the {CONTAINER_BAY_STATION}"
            if event.from_location not in ("latch position A", "dispatch folder")
            else " in open view"
        )
        return (
            f"{prefix}, in the {event.room}, {event.actor} moved the {event.target_object} "
            f"from the {event.from_location} to the {event.to_location}{occ_clause}."
        )

    if event.kind == "see":
        if event.occluded and event.occluder:
            return (
                f"{prefix}, in the {event.room}, from the {FAR_SIDE_ALCOVE_STATION} behind the {event.occluder}, "
                f"{event.actor} looked toward the {event.target_object}, "
                f"but the {event.occluder} blocked the line of sight."
            )
        loc_clause = (
            f" at the {event.to_location}"
            if event.to_location
            and event.to_location not in (event.room, CONTAINER_BAY_STATION, FAR_SIDE_ALCOVE_STATION)
            else f" in the {event.to_location}"
            if event.to_location in (CONTAINER_BAY_STATION, FAR_SIDE_ALCOVE_STATION)
            else ""
        )
        return (
            f"{prefix}, in the {event.room}, {event.actor} observed the {event.target_object}"
            f"{loc_clause} in open view."
        )

    if event.kind == "say":
        noise_desc = (
            f"at {event.noise_level} dB (at or above the {noise_threshold} dB doorway threshold, audible in the room and doorway-adjacent rooms)"
            if event.noise_level >= noise_threshold
            else f"at {event.noise_level} dB (below the {noise_threshold} dB doorway threshold, so quieter speech stayed in the {event.room} and adjacent rooms could not hear)"
        )
        return (
            f"{prefix}, in the {event.room} {noise_desc}, {event.actor} said: \"{event.utterance}\""
        )

    if event.kind == "message":
        return (
            f"{prefix}, from the {event.room}, {event.actor} sent a direct text message to {event.recipient}: "
            f"\"{event.utterance}\""
        )

    raise ValueError(f"Unknown event kind: {event.kind}")


def render_world_template(world: World, focal_agent: str) -> str:
    """Deterministically render `world` in third-person prose focalized on `focal_agent`."""
    if focal_agent not in world.agents:
        raise ValueError(f"focal_agent '{focal_agent}' must be one of {world.agents}")

    ordered_agents = (focal_agent,) + tuple(a for a in world.agents if a != focal_agent)

    doorway_strs = [f"the {a} connects to the {b}" for a, b in world.doorways]
    room_setup_strs: list[str] = []
    for r in world.rooms:
        c_pair = world.containers_by_room[r]
        fix = world.fixtures_by_room[r]
        occ = world.occluders_by_room[r]
        occ_art = indefinite_article(occ)
        room_setup_strs.append(
            f"the {r} contains the {c_pair[0]} and the {c_pair[1]} in the {CONTAINER_BAY_STATION} "
            f"and the {fix} in the {FAR_SIDE_ALCOVE_STATION} "
            f"(with {occ_art} {occ} standing between the {FAR_SIDE_ALCOVE_STATION} and the {CONTAINER_BAY_STATION} "
            f"that blocks an agent in the {FAR_SIDE_ALCOVE_STATION} from seeing events at or the contents of "
            f"the containers behind it, while an agent in the {CONTAINER_BAY_STATION} sees them in open view)"
        )

    agent_pos_strs = [
        f"{ag} starts in the {world.initial_agent_rooms[ag]}" for ag in ordered_agents
    ]
    goal_strs = [world.goals[ag].goal_text for ag in ordered_agents]
    stake_strs = [s.description for s in world.stake_edges]

    setup_para = (
        f"Following {focal_agent} through the {world.domain_name.replace('_', ' ')}: "
        f"{'; '.join(room_setup_strs)}. "
        f"Doorways link the rooms as follows: {', '.join(doorway_strs)}, and the doorway hearing threshold is {world.noise_threshold} dB "
        f"(a say at or above {world.noise_threshold} dB carries one doorway; quieter speech stays in its room). "
        f"{BELIEF_PERSISTENCE_RULE} "
        f"At the start, {', '.join(agent_pos_strs)}. "
        f"Meanwhile, {', '.join(world.absent_parties)} "
        f"{'is' if len(world.absent_parties) == 1 else 'are'} absent off-site. "
        f"Stated goals: {'; '.join(goal_strs)}. "
        f"Stakes: {' '.join(stake_strs)}"
    )

    event_sentences = [
        _render_event_sentence(ev, focal_agent, world.noise_threshold)
        for ev in world.events
    ]
    events_para = " ".join(event_sentences)

    prose = f"{setup_para}\n\n{events_para}"
    check_rendered_prose(prose, world, focal_agent=focal_agent)
    return prose


def _reconstruct_world_from_writer_prompt(world: World, prompt_str: str) -> World:
    """Reconstruct `World` from the JSON event log embedded in `prompt_str` to verify perspective-flip invariance."""
    marker = "Complete Event Log:\n"
    pos = prompt_str.find(marker)
    if pos < 0:
        return world
    raw_after = prompt_str[pos + len(marker) :]
    end_pos = raw_after.find("\n\nOUTPUT FORMAT:")
    json_block = raw_after[:end_pos] if end_pos >= 0 else raw_after
    parsed_list = json.loads(json_block)
    if len(parsed_list) != len(world.events):
        raise RenderCheckError(
            "Writer prompt event log is incomplete.",
            check_id="prompt_event_log_incomplete",
        )
    by_order: dict[int, dict[str, Any]] = {}
    for idx, item in enumerate(parsed_list, start=1):
        order_key = int(item.get("order_index", item.get("step", idx)))
        by_order[order_key] = item
    expected_indices = set(range(1, len(world.events) + 1))
    if set(by_order.keys()) != expected_indices:
        raise RenderCheckError(
            "Writer prompt event log is incomplete.",
            check_id="prompt_event_log_incomplete",
        )
    reconstructed_events = tuple(
        replace(
            ev,
            kind=str(by_order[idx]["kind"]),
            actor=str(by_order[idx]["actor"]),
            room=str(by_order[idx]["room"]),
            target_object=by_order[idx].get("target_object"),
            from_location=by_order[idx].get("from_location"),
            to_location=by_order[idx].get("to_location"),
            utterance=by_order[idx].get("utterance"),
            recipient=by_order[idx].get("recipient"),
            occluded=bool(by_order[idx].get("occluded", False)),
            occluder=by_order[idx].get("occluder"),
            noise_level=int(by_order[idx].get("noise_level", 0)),
            is_bluff=bool(by_order[idx].get("is_bluff", False)),
            stated_reason=by_order[idx].get("stated_reason"),
        )
        for idx, ev in enumerate(world.events, start=1)
    )
    return replace(
        world,
        events=reconstructed_events,
        unknowable_spec=replace(
            world.unknowable_spec,
            trace_a=reconstructed_events,
            trace_b=reconstructed_events,
        ),
    )


class MockWriter:
    """CPU-only mock writer for `qwen3.6:35b` that refuses any real `/api/chat` call."""

    def __init__(
        self,
        custom_renderer: Callable[[World, str], str] | None = None,
    ) -> None:
        self._custom_renderer = custom_renderer
        self._prompt_template = load_writer_prompt_template()
        self._refusals: list[ProseRefusalRecord] = []

    @property
    def refusals(self) -> tuple[ProseRefusalRecord, ...]:
        return tuple(self._refusals)

    def call_api_chat(self, *args: object, **kwargs: object) -> None:
        """Hard guard: real `/api/chat` calls are forbidden in the CPU mock writer."""
        raise ForbiddenModelCallError(
            "Real /api/chat calls and GPU model loads are forbidden in MockWriter (CPU mock writer only)."
        )

    def render_scenario_pair(self, world: World) -> tuple[RenderedScenario, RenderedScenario]:
        """Render `world` twice, focalized on two distinct agents, over the complete log."""
        focal_0 = world.agents[0]
        focal_1 = world.agents[1]

        prompt_0 = format_writer_prompt(world, focal_0, self._prompt_template)
        prompt_1 = format_writer_prompt(world, focal_1, self._prompt_template)

        render_fn = self._custom_renderer or render_world_template
        rendered_texts: list[str] = []
        for r_idx, focal_ag in ((0, focal_0), (1, focal_1)):
            try:
                prose_text = render_fn(world, focal_ag)
                check_rendered_prose(prose_text, world, focal_agent=focal_ag)
                rendered_texts.append(prose_text)
            except RenderCheckError as exc:
                self._refusals.append(
                    ProseRefusalRecord(
                        scenario_id=world.scenario_id,
                        rendering_index=r_idx,
                        focal_agent=focal_ag,
                        check_id=exc.check_id,
                        reason=str(exc),
                        failing_sentence=exc.failing_sentence,
                        event_index=exc.event_index,
                        event_kind=exc.event_kind,
                    )
                )
                raise

        prose_0, prose_1 = rendered_texts[0], rendered_texts[1]

        keys_0 = compute_scenario_keys(_reconstruct_world_from_writer_prompt(world, prompt_0))
        keys_1 = compute_scenario_keys(_reconstruct_world_from_writer_prompt(world, prompt_1))
        if keys_0 != keys_1:
            raise AssertionError("Perspective-flip invariance violated: keys differ between renderings.")

        return (
            RenderedScenario(
                scenario_id=world.scenario_id,
                rendering_index=0,
                focal_agent=focal_0,
                prose=prose_0,
                writer_prompt=prompt_0,
                keys=keys_0,
                world=world,
            ),
            RenderedScenario(
                scenario_id=world.scenario_id,
                rendering_index=1,
                focal_agent=focal_1,
                prose=prose_1,
                writer_prompt=prompt_1,
                keys=keys_1,
                world=world,
            ),
        )


# ---------------------------------------------------------------------------
# Loopback Ollama server, VRAM guard, generations.jsonl, checkpoint, and OllamaWriter
# ---------------------------------------------------------------------------


def estimate_prompt_tokens(messages: Sequence[Mapping[str, str]] | str) -> int:
    """Conservatively estimate prompt tokens using `ceil(utf8_bytes / 3)`."""
    if isinstance(messages, str):
        utf8_bytes = len(messages.encode("utf-8"))
    else:
        utf8_bytes = sum(
            len(str(m.get("role", "")).encode("utf-8"))
            + len(str(m.get("content", "")).encode("utf-8"))
            for m in messages
        )
    return max(1, (utf8_bytes + 2) // 3)


def round_up_num_ctx(needed_tokens: int, step: int = NUM_CTX_STEP_TOKENS) -> int:
    """Round `needed_tokens` up to the next multiple of `step` (default 1024)."""
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    return max(step, ((max(1, needed_tokens) + step - 1) // step) * step)


def compute_num_ctx(
    prompt_tokens_estimate: int,
    num_predict: int,
    margin: int = NUM_CTX_MARGIN_TOKENS,
    step: int = NUM_CTX_STEP_TOKENS,
) -> int:
    """Compute `options.num_ctx = round_up(prompt_tokens_estimate + num_predict + margin, 1024)`."""
    return round_up_num_ctx(prompt_tokens_estimate + num_predict + margin, step=step)


def require_loopback_ollama_url(url: str) -> str:
    """Enforce loopback-only Ollama endpoint: only plain `http://127.0.0.1:<port>` is permitted."""
    raw = url.strip() if isinstance(url, str) else ""
    parsed = urlparse(raw)
    if (
        parsed.scheme != "http"
        or parsed.hostname != LOOPBACK_HOST
        or parsed.port is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(
            f"REFUSED (NON_LOOPBACK_URL): Ollama endpoint must be plain http://{LOOPBACK_HOST}:<port> on the loopback, got {url!r}."
        )
    return raw.rstrip("/")


def default_ollama_http_transport(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Default loopback-only HTTP transport for Ollama `/api/tags`, `/api/version`, and `/api/chat`."""
    require_loopback_ollama_url(url)
    if url.endswith("/api/pull"):
        raise ValueError(f"REFUSED (PULL_FORBIDDEN): Auto-pull is forbidden ({method} {url}).")
    data = (
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if body is not None
        else None
    )
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:  # noqa: S310 - loopback verified above
            raw_bytes = resp.read()
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"Loopback Ollama request failed ({method} {url}): {exc}") from exc

    try:
        parsed = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Loopback Ollama returned invalid JSON from {method} {url}: {exc}") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError(f"Loopback Ollama returned non-object JSON from {method} {url}.")
    return parsed


def verify_local_ollama_model_digest(
    *,
    base_url: str,
    ollama_tag: str,
    expected_digest: str,
    http_transport: OllamaHttpTransport = default_ollama_http_transport,
) -> str:
    """Verify via `GET /api/tags` that `ollama_tag` is present locally and matches `expected_digest`. Never pulls."""
    clean_base = require_loopback_ollama_url(base_url)
    tags_url = f"{clean_base}/api/tags"
    resp = http_transport("GET", tags_url, None)
    models = resp.get("models")
    if not isinstance(models, list):
        models = []

    candidates = {ollama_tag, f"{ollama_tag}:latest"}
    entry = next(
        (
            m
            for m in models
            if isinstance(m, dict)
            and (m.get("name") in candidates or m.get("model") in candidates)
        ),
        None,
    )
    if entry is None:
        raise ValueError(
            f"REFUSED (MODEL_MISSING): Pinned model '{ollama_tag}' is not present in local Ollama at {tags_url}. Auto-pull is forbidden."
        )

    got_digest = str(entry.get("digest") or "").strip().lower().removeprefix("sha256:")
    norm_expected = expected_digest.strip().lower().removeprefix("sha256:")
    if got_digest != norm_expected:
        raise ValueError(
            f"REFUSED (DIGEST_MISMATCH): Local Ollama model '{ollama_tag}' has digest '{got_digest}', "
            f"which differs from pinned manifest_digest '{norm_expected}'."
        )
    return norm_expected


def query_gpu_vram_peak_mib() -> int:
    """Query current GPU memory usage in MiB via `nvidia-smi` (returns 0 if unavailable)."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return 0
        vals = [
            int(line.strip())
            for line in proc.stdout.splitlines()
            if line.strip().isdigit()
        ]
        return max(vals, default=0)
    except Exception:  # pragma: no cover
        return 0


def kill_pid_tree(pid: int) -> None:
    """Terminate a process and all of its descendants by PID tree (`taskkill /PID <pid> /T /F` on Windows)."""
    if pid <= 4:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        return

    # POSIX: process-group kill for owned child server session/group, plus recursive child cleanup
    sig_kill = getattr(signal, "SIGKILL", signal.SIGTERM)
    killpg_fn = getattr(os, "killpg", None)
    getpgid_fn = getattr(os, "getpgid", None)
    getpgrp_fn = getattr(os, "getpgrp", None)
    own_pgrp = int(getpgrp_fn()) if callable(getpgrp_fn) else -1
    if callable(killpg_fn):
        pgid = pid
        if callable(getpgid_fn):
            try:
                pgid = int(getpgid_fn(pid))
            except OSError:  # pragma: no cover
                pgid = pid
        if pgid > 4 and pgid != own_pgrp:
            try:
                killpg_fn(pgid, sig_kill)
            except OSError:  # pragma: no cover
                pass
        if pid != pgid and pid != own_pgrp:  # pragma: no cover
            try:
                killpg_fn(pid, sig_kill)
            except OSError:
                pass

    child_pids: set[int] = set()  # pragma: no cover
    proc_root = Path("/proc")  # pragma: no cover
    if proc_root.is_dir():  # pragma: no cover
        for stat_file in proc_root.glob("[0-9]*/stat"):  # pragma: no cover
            try:  # pragma: no cover
                raw_stat = stat_file.read_text(encoding="utf-8")  # pragma: no cover
                after_comm = raw_stat.rsplit(")", 1)[1].split()  # pragma: no cover
                if len(after_comm) >= 2 and int(after_comm[1]) == pid:  # pragma: no cover
                    child_pids.add(int(stat_file.parent.name))  # pragma: no cover
            except Exception:  # pragma: no cover
                pass  # pragma: no cover
    try:  # pragma: no cover
        children = subprocess.run(
            ["pgrep", "-P", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        for line in children.stdout.splitlines():  # pragma: no cover
            if line.strip().isdigit():  # pragma: no cover
                child_pids.add(int(line.strip()))  # pragma: no cover
    except Exception:  # pragma: no cover
        pass  # pragma: no cover
    for cpid in sorted(child_pids):  # pragma: no cover
        if cpid != pid:  # pragma: no cover
            kill_pid_tree(cpid)  # pragma: no cover
    try:
        os.kill(pid, sig_kill)
    except OSError:
        pass


def _pick_free_localhost_port() -> int:
    """Bind an ephemeral TCP socket on 127.0.0.1 and return its free port number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class OwnedOllamaServer:
    """Child `ollama serve` process started and owned on loopback `127.0.0.1:<port>`."""

    process: Any
    pid: int
    port: int
    host: str
    base_url: str
    models_dir: str | None
    started_at: str
    stopped_at: str | None = None
    version: str | None = None
    unloaded: bool = False
    pid_killer: Callable[[int], None] = kill_pid_tree

    def stop(self) -> None:
        """Stop the child `ollama serve` process and its PID tree, recording its UTC stop timestamp."""
        if self.stopped_at is not None:
            return
        try:
            if self.process is not None and self.process.poll() is None:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=3)
                except Exception:  # pragma: no cover
                    pass
                if self.process.poll() is None:  # pragma: no cover
                    try:
                        self.process.kill()
                        self.process.wait(timeout=3)
                    except Exception:
                        pass
            if self.pid > 4:
                try:
                    self.pid_killer(self.pid)
                except Exception:  # pragma: no cover
                    pass
        finally:
            self.stopped_at = datetime.datetime.now(datetime.UTC).isoformat()

    def to_dict(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "port": self.port,
            "host": self.host,
            "base_url": self.base_url,
            "models_dir": self.models_dir,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "version": self.version,
            "owned": True,
            "unloaded": self.unloaded,
        }


def wait_for_ollama_version(
    *,
    base_url: str,
    process: Any,
    http_transport: OllamaHttpTransport = default_ollama_http_transport,
    timeout_s: float = 30.0,
    poll_interval_s: float = 0.05,
) -> str:
    """Wait for `GET <base_url>/api/version` on the owned child server."""
    clean_base = require_loopback_ollama_url(base_url)
    version_url = f"{clean_base}/api/version"
    deadline = time.monotonic() + max(0.0, timeout_s)
    last_exc: Exception | None = None
    while True:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"Owned Ollama server process (PID {getattr(process, 'pid', None)}) exited with code "
                f"{process.returncode} before {version_url} became ready."
            )
        try:
            resp = http_transport("GET", version_url, None)
            if isinstance(resp, Mapping):
                return str(resp.get("version") or "unknown")
        except Exception as exc:
            last_exc = exc
        if time.monotonic() >= deadline:
            break
        time.sleep(poll_interval_s)
    raise RuntimeError(
        f"Timed out waiting for owned Ollama server (PID {getattr(process, 'pid', None)}) at {version_url}: {last_exc}"
    )


def start_owned_ollama_server(
    *,
    http_transport: OllamaHttpTransport = default_ollama_http_transport,
    popen_factory: Callable[..., Any] | None = None,
    port: int | None = None,
    startup_timeout_s: float = 30.0,
    poll_interval_s: float = 0.05,
    ollama_models_dir: Path | str | None = None,
    pid_killer: Callable[[int], None] = kill_pid_tree,
) -> OwnedOllamaServer:
    """Launch `ollama serve` as an owned child process on `127.0.0.1:<port>` and wait for `/api/version`."""
    resolved_port = int(port) if port is not None else _pick_free_localhost_port()
    host = f"127.0.0.1:{resolved_port}"
    base_url = f"http://{host}"
    require_loopback_ollama_url(base_url)

    env = dict(os.environ)
    env["OLLAMA_HOST"] = host
    if ollama_models_dir is not None:
        env["OLLAMA_MODELS"] = str(ollama_models_dir)
    elif os.environ.get("OLLAMA_MODELS", "").strip():
        env["OLLAMA_MODELS"] = os.environ["OLLAMA_MODELS"].strip()
    models_dir = env.get("OLLAMA_MODELS")

    popen_kwargs: dict[str, Any] = {
        "env": env,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        create_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        if create_group:
            popen_kwargs["creationflags"] = create_group
    else:  # pragma: no cover
        popen_kwargs["start_new_session"] = True

    started_at = datetime.datetime.now(datetime.UTC).isoformat()
    launcher = popen_factory if popen_factory is not None else subprocess.Popen
    proc = launcher(["ollama", "serve"], **popen_kwargs)
    raw_pid = getattr(proc, "pid", None)
    if not isinstance(raw_pid, int) or isinstance(raw_pid, bool) or raw_pid <= 4:
        try:
            proc.terminate()
        except Exception:  # pragma: no cover
            pass
        raise RuntimeError(f"Owned Ollama child process returned invalid PID {raw_pid!r}.")

    server = OwnedOllamaServer(
        process=proc,
        pid=int(raw_pid),
        port=resolved_port,
        host=host,
        base_url=base_url,
        models_dir=models_dir,
        started_at=started_at,
        pid_killer=pid_killer,
    )
    try:
        server.version = wait_for_ollama_version(
            base_url=base_url,
            process=proc,
            http_transport=http_transport,
            timeout_s=startup_timeout_s,
            poll_interval_s=poll_interval_s,
        )
    except Exception:
        server.stop()
        raise
    return server


@dataclass(frozen=True)
class SealedResponseEntry:
    """Single writer response entry saved in `sealed_store/responses.sealed.jsonl`, keyed by `call_id`."""

    call_id: str
    call_index: int
    scenario_id: str
    rendering_index: int
    focal_agent: str
    attempt: int
    seed: int
    response_text: str
    thinking_text: str
    response_sha256: str
    thinking_sha256: str
    checks_passed: bool = True
    check_id: str | None = None
    event_index: int | None = None
    event_kind: str | None = None
    failing_sentence: str | None = None
    failing_sentence_sha256: str = ""
    failing_sentence_offset: int | None = None
    failing_sentence_length: int | None = None
    refusal_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "call_id": self.call_id,
            "call_index": self.call_index,
            "scenario_id": self.scenario_id,
            "rendering_index": self.rendering_index,
            "focal_agent": self.focal_agent,
            "attempt": self.attempt,
            "seed": self.seed,
            "response_text": self.response_text,
            "thinking_text": self.thinking_text,
            "response_sha256": self.response_sha256,
            "thinking_sha256": self.thinking_sha256,
            "checks_passed": self.checks_passed,
            "check_id": self.check_id,
            "event_index": self.event_index,
            "event_kind": self.event_kind,
            "failing_sentence": self.failing_sentence,
            "failing_sentence_sha256": self.failing_sentence_sha256,
            "failing_sentence_offset": self.failing_sentence_offset,
            "failing_sentence_length": self.failing_sentence_length,
            "refusal_reasons": list(self.refusal_reasons),
        }


def compute_sealed_response_digests(
    response_text: str,
    thinking_text: str = "",
) -> tuple[str, str]:
    """Return `(response_sha256, thinking_sha256)` for a writer response and optional thinking text."""
    resp_sha = _sha256_bytes(response_text.encode("utf-8"))
    think_sha = _sha256_bytes(thinking_text.encode("utf-8")) if thinking_text else ""
    return resp_sha, think_sha


def locate_failing_sentence_in_response(
    response_text: str,
    failing_sentence: str | None,
) -> tuple[str, int | None, int | None]:
    """Locate `failing_sentence` in `response_text` and return `(sha256_hex, offset, length)`."""
    if not failing_sentence or failing_sentence == "no candidate sentence":
        return "", None, None
    idx = response_text.find(failing_sentence)
    if idx < 0:
        return "", None, None
    length = len(failing_sentence)
    sha256_hex = _sha256_bytes(failing_sentence.encode("utf-8"))
    return sha256_hex, idx, length


def resolve_sealed_failing_sentence(
    response_text: str,
    *,
    failing_sentence_sha256: str | None,
    failing_sentence_offset: int | None,
    failing_sentence_length: int | None,
) -> tuple[str | None, bool]:
    """Slice `response_text[offset : offset + length]` and verify its SHA-256 matches `failing_sentence_sha256`."""
    expected_sha = str(failing_sentence_sha256 or "").strip().lower()
    if (
        not expected_sha
        and failing_sentence_offset is None
        and failing_sentence_length is None
    ):
        return None, True
    if (
        not expected_sha
        or not isinstance(failing_sentence_offset, int)
        or isinstance(failing_sentence_offset, bool)
        or not isinstance(failing_sentence_length, int)
        or isinstance(failing_sentence_length, bool)
        or failing_sentence_offset < 0
        or failing_sentence_length < 0
        or failing_sentence_offset + failing_sentence_length > len(response_text)
    ):
        return None, False
    sliced = response_text[failing_sentence_offset : failing_sentence_offset + failing_sentence_length]
    actual_sha = _sha256_bytes(sliced.encode("utf-8"))
    if actual_sha != expected_sha:
        return sliced, False
    return sliced, True


def format_non_sealed_refusal_reason(
    *,
    check_id: str,
    event_index: int | None = None,
    event_kind: str | None = None,
    failing_sentence_sha256: str = "",
    failing_sentence_offset: int | None = None,
    failing_sentence_length: int | None = None,
) -> str:
    """Format a non-sealed refusal summary string containing only check_id, event number/kind, and sealed sentence sha/offset/length."""
    parts = [
        f"event_index={event_index}",
        f"event_kind={event_kind}",
        f"failing_sentence_sha256={failing_sentence_sha256}",
        f"failing_sentence_offset={failing_sentence_offset}",
        f"failing_sentence_length={failing_sentence_length}",
    ]
    return f"{check_id}: " + " ".join(parts)


def parse_generation_refusal_metadata(gen_row: Mapping[str, Any]) -> dict[str, Any]:
    """Parse refusal metadata from a `generations.jsonl` row, supporting both `sealed_offset_v1` and `legacy_inline_v0`."""
    checks_passed = bool(gen_row.get("checks_passed", True))
    if checks_passed:
        return {
            "format": "sealed_offset_v1",
            "check_id": None,
            "event_index": None,
            "event_kind": None,
            "failing_sentence_sha256": "",
            "failing_sentence_offset": None,
            "failing_sentence_length": None,
        }
    has_new_fields = (
        "failing_sentence_sha256" in gen_row
        or "failing_sentence_offset" in gen_row
        or "failing_sentence_length" in gen_row
        or "check_id" in gen_row
    )
    raw_reasons = gen_row.get("refusal_reasons") or ()
    first_reason = str(raw_reasons[0]) if isinstance(raw_reasons, (list, tuple)) and raw_reasons else ""
    has_inline_candidate = "candidate sentence:" in first_reason
    if has_new_fields and not has_inline_candidate:
        raw_cid = gen_row.get("check_id")
        cid = (
            str(raw_cid).strip()
            if raw_cid is not None and str(raw_cid).strip()
            else (first_reason.split(":", 1)[0].strip() if ":" in first_reason else first_reason.strip() or None)
        )
        raw_ev_idx = gen_row.get("event_index")
        ev_idx = (
            int(raw_ev_idx)
            if isinstance(raw_ev_idx, int) and not isinstance(raw_ev_idx, bool)
            else None
        )
        raw_ev_kind = gen_row.get("event_kind")
        ev_kind = str(raw_ev_kind) if raw_ev_kind is not None and str(raw_ev_kind).strip() else None
        raw_sha = str(gen_row.get("failing_sentence_sha256") or "")
        raw_off = gen_row.get("failing_sentence_offset")
        off = (
            int(raw_off)
            if isinstance(raw_off, int) and not isinstance(raw_off, bool)
            else None
        )
        raw_len = gen_row.get("failing_sentence_length")
        length = (
            int(raw_len)
            if isinstance(raw_len, int) and not isinstance(raw_len, bool)
            else None
        )
        return {
            "format": "sealed_offset_v1",
            "check_id": cid,
            "event_index": ev_idx,
            "event_kind": ev_kind,
            "failing_sentence_sha256": raw_sha,
            "failing_sentence_offset": off,
            "failing_sentence_length": length,
        }

    cid_legacy = (
        first_reason.split(":", 1)[0].strip()
        if ":" in first_reason
        else (first_reason.strip() or None)
    )
    m_ev = re.search(r"event\s+#(\d+)", first_reason)
    ev_idx_legacy = int(m_ev.group(1)) if m_ev else None
    m_kind = re.search(r"\bkind=([a-z_]+)\b", first_reason)
    ev_kind_legacy = m_kind.group(1) if m_kind else None
    return {
        "format": "legacy_inline_v0",
        "check_id": cid_legacy,
        "event_index": ev_idx_legacy,
        "event_kind": ev_kind_legacy,
        "failing_sentence_sha256": "",
        "failing_sentence_offset": None,
        "failing_sentence_length": None,
    }


def _sanitize_non_sealed_logprobs(raw_logprobs: Any) -> tuple[dict[str, object], ...]:
    """Strip any model token strings from `logprobs` before storing in non-sealed `GenerationCallRecord`."""
    if not isinstance(raw_logprobs, (list, tuple)):
        return ()
    sanitized: list[dict[str, object]] = []
    for item in raw_logprobs[:5]:
        if not isinstance(item, dict):
            continue
        clean_item: dict[str, object] = {}
        for k, v in item.items():
            if k in ("token", "bytes", "text", "piece") and isinstance(v, str):
                clean_item["token_sha256"] = _sha256_bytes(v.encode("utf-8"))
            elif isinstance(v, (int, float, bool)) or v is None:
                clean_item[str(k)] = v
        sanitized.append(clean_item)
    return tuple(sanitized)


def append_sealed_response(
    sealed_responses_path: Path,
    entry: SealedResponseEntry,
) -> None:
    """Append a `SealedResponseEntry` row to `sealed_store/responses.sealed.jsonl`."""
    sealed_responses_path.parent.mkdir(parents=True, exist_ok=True)
    with sealed_responses_path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(_canonical_json_line(entry.to_dict()) + "\n")


@dataclass(frozen=True)
class GenerationCallRecord:
    """Single line record in `generations.jsonl` per design §6.8."""

    call_id: str
    call_index: int
    scenario_id: str
    rendering_index: int
    focal_agent: str
    operation: str
    model: str
    manifest_digest: str
    model_layer_digest: str
    licence_id: str
    prompt_sha256: str
    prompt_tokens_estimate: int
    prompt_eval_count: int
    eval_count: int
    thinking_words: int
    num_predict: int
    num_ctx: int
    max_num_ctx: int
    think: bool
    temperature: float
    done_reason: str
    seconds: float
    vram_peak_mib: int
    vram_halt_mib: int
    grant_id: str
    top5_logprobs: tuple[dict[str, object], ...] = ()
    checks_passed: bool = True
    refusal_reasons: tuple[str, ...] = ()
    check_id: str | None = None
    event_index: int | None = None
    event_kind: str | None = None
    failing_sentence_sha256: str = ""
    failing_sentence_offset: int | None = None
    failing_sentence_length: int | None = None
    attempt: int = 0
    seed: int = 0
    response_sha256: str = ""
    thinking_sha256: str = ""
    repair_line: str = ""
    repair_line_sha256: str = ""
    retry_form: str = ""
    retry_fallback_reason: str = ""
    retry_unchanged_sentences: int | None = None
    retry_attempt0_sentences: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "call_id": self.call_id,
            "call_index": self.call_index,
            "scenario_id": self.scenario_id,
            "rendering_index": self.rendering_index,
            "focal_agent": self.focal_agent,
            "attempt": self.attempt,
            "seed": self.seed,
            "operation": self.operation,
            "model": self.model,
            "manifest_digest": self.manifest_digest,
            "model_layer_digest": self.model_layer_digest,
            "licence_id": self.licence_id,
            "pins": {
                "model": self.model,
                "manifest_digest": self.manifest_digest,
                "model_layer_digest": self.model_layer_digest,
                "licence_id": self.licence_id,
            },
            "prompt_sha256": self.prompt_sha256,
            "response_sha256": self.response_sha256,
            "thinking_sha256": self.thinking_sha256,
            "repair_line": self.repair_line,
            "repair_line_sha256": self.repair_line_sha256,
            "retry_form": self.retry_form,
            "retry_fallback_reason": self.retry_fallback_reason,
            "retry_unchanged_sentences": self.retry_unchanged_sentences,
            "retry_attempt0_sentences": self.retry_attempt0_sentences,
            "tokens": {
                "prompt_tokens_estimate": self.prompt_tokens_estimate,
                "prompt_eval_count": self.prompt_eval_count,
                "eval_count": self.eval_count,
                "thinking_words": self.thinking_words,
            },
            "prompt_tokens_estimate": self.prompt_tokens_estimate,
            "prompt_eval_count": self.prompt_eval_count,
            "eval_count": self.eval_count,
            "thinking_words": self.thinking_words,
            "num_predict": self.num_predict,
            "num_ctx": self.num_ctx,
            "max_num_ctx": self.max_num_ctx,
            "think": self.think,
            "temperature": self.temperature,
            "done_reason": self.done_reason,
            "seconds": round(self.seconds, 6),
            "vram_peak_mib": self.vram_peak_mib,
            "vram_halt_mib": self.vram_halt_mib,
            "grant_id": self.grant_id,
            "top5_logprobs": list(self.top5_logprobs),
            "checks_passed": self.checks_passed,
            "check_id": self.check_id,
            "event_index": self.event_index,
            "event_kind": self.event_kind,
            "failing_sentence_sha256": self.failing_sentence_sha256,
            "failing_sentence_offset": self.failing_sentence_offset,
            "failing_sentence_length": self.failing_sentence_length,
            "refusal_reasons": list(self.refusal_reasons),
        }


@dataclass(frozen=True)
class OllamaUnloadRecord:
    """Record of an Ollama `keep_alive: 0` model unload request."""

    model: str
    manifest_digest: str
    keep_alive: int
    wall_time_s: float

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "manifest_digest": self.manifest_digest,
            "keep_alive": self.keep_alive,
            "wall_time_s": round(self.wall_time_s, 6),
        }


class VramHaltError(RuntimeError):
    """Raised when GPU VRAM usage exceeds `vram_halt_mib` and the owned server PID tree is killed."""


class LengthStopRefusalError(RenderCheckError):
    """Raised when an Ollama `/api/chat` call terminates with `done_reason == 'length'`."""

    def __init__(self, message: str) -> None:
        super().__init__(message, check_id="length_stop")


class MaxNumCtxRefusalError(RenderCheckError):
    """Raised when a writer call requires `num_ctx > max_num_ctx`."""

    def __init__(self, message: str) -> None:
        super().__init__(message, check_id="max_num_ctx_exceeded")


class GpuGuardMonitor:
    """Monitors per-call timing, token counts, and VRAM peak, halting by PID-tree kill above `vram_halt_mib`."""

    def __init__(
        self,
        *,
        grant_id: str = "cpu-mock",
        vram_halt_mib: int = DEFAULT_VRAM_HALT_MIB,
        vram_probe: Callable[[], int] = query_gpu_vram_peak_mib,
        halt_pid: int | None = None,
        pid_killer: Callable[[int], None] = kill_pid_tree,
    ) -> None:
        if vram_halt_mib <= 0:
            raise ValueError(f"vram_halt_mib must be positive, got {vram_halt_mib}.")
        self.grant_id = grant_id
        self.vram_halt_mib = vram_halt_mib
        self._vram_probe = vram_probe
        self._halt_pid = halt_pid
        self._pid_killer = pid_killer
        self._peak_vram_mib: int = 0
        self._halted: bool = False

    @property
    def peak_vram_mib(self) -> int:
        return self._peak_vram_mib

    @property
    def halted(self) -> bool:
        return self._halted

    @property
    def halt_pid(self) -> int | None:
        return self._halt_pid

    def set_halt_pid(self, pid: int | None) -> None:
        self._halt_pid = pid

    def _trigger_halt(self, current_mib: int) -> None:
        if not self._halted:
            self._halted = True
            if self._halt_pid is not None:
                self._pid_killer(self._halt_pid)
        raise VramHaltError(
            f"REFUSED (VRAM_HALT_EXCEEDED): VRAM usage {current_mib} MiB exceeded halt ceiling "
            f"{self.vram_halt_mib} MiB (killed PID tree {self._halt_pid})."
        )

    def check_vram(self) -> int:
        if self._halted:
            self._trigger_halt(self._peak_vram_mib)
        current_mib = int(self._vram_probe())
        if current_mib > self._peak_vram_mib:
            self._peak_vram_mib = current_mib
        if current_mib > self.vram_halt_mib:
            self._trigger_halt(current_mib)
        return current_mib

    def wrap_transport(self, inner: OllamaHttpTransport) -> OllamaHttpTransport:
        """Wrap an `OllamaHttpTransport` so every `/api/chat` call checks VRAM before and after."""

        def _guarded(
            method: str,
            url: str,
            payload: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            is_unload = (
                method.upper() == "POST"
                and url.endswith("/api/chat")
                and payload is not None
                and payload.get("keep_alive") == 0
                and not payload.get("messages")
            )
            if self._halted and not is_unload:
                self._trigger_halt(self._peak_vram_mib)
            require_loopback_ollama_url(url)
            if url.endswith("/api/pull"):
                raise ValueError(
                    f"REFUSED (PULL_FORBIDDEN): Auto-pull is forbidden ({method} {url}); refuse when the pinned writer digest is missing."
                )
            if method.upper() == "POST" and url.endswith("/api/chat") and not is_unload:
                self.check_vram()
            resp = inner(method, url, payload)
            if method.upper() == "POST" and url.endswith("/api/chat") and payload is not None and not is_unload:
                if payload.get("keep_alive") != 0 and payload.get("messages"):
                    self.check_vram()
            return resp

        return _guarded


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically write bytes to `path` via temp file + flush + `os.fsync` + `os.replace`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp_path, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)


def _canonical_json_line(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _canonical_json_doc(obj: Any) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class WriterCheckpointManager:
    """Per-scenario atomic sealed checkpoint with SHA-256 hash chain and `--resume` verification."""

    def __init__(
        self,
        *,
        output_dir: Path,
        count: int,
        base_seed: int,
        private_seed_hex: str,
        writer_kind: str,
        writer_pin: PinnedWriterModelSpec = DEFAULT_WRITER_MODEL_PIN,
    ) -> None:
        self.output_dir = output_dir
        self.public_path = output_dir / CHECKPOINT_FILENAME
        self.sealed_path = output_dir / "sealed_store" / SEALED_CHECKPOINT_FILENAME
        self.count = count
        self.base_seed = base_seed
        self.private_seed_hex = private_seed_hex
        self.writer_kind = writer_kind
        self.writer_pin = writer_pin

        self._pub_entries: list[dict[str, Any]] = []
        self._sealed_entries: list[dict[str, Any]] = []
        self._chain_head: str = CHECKPOINT_GENESIS_HASH

    @property
    def completed_count(self) -> int:
        return len(self._sealed_entries)

    @property
    def chain_head_sha256(self) -> str:
        return self._chain_head

    def _public_header(self) -> dict[str, object]:
        return {
            "schema_version": "udlab.checkpoint.v1",
            "count": self.count,
            "base_seed": self.base_seed,
            "writer_kind": self.writer_kind,
            "writer_model": self.writer_pin.ollama_tag,
            "writer_manifest_digest": self.writer_pin.manifest_digest,
        }

    def _sealed_header(self) -> dict[str, object]:
        return {
            **self._public_header(),
            "private_seed_hex": self.private_seed_hex,
            "writer_pin": self.writer_pin.to_dict(),
        }

    def _write_both_atomically(self) -> None:
        sealed_doc = {
            "header": self._sealed_header(),
            "scenarios_completed": len(self._sealed_entries),
            "chain_head_sha256": self._chain_head,
            "scenarios": self._sealed_entries,
        }
        pub_doc = {
            "header": self._public_header(),
            "scenarios_completed": len(self._pub_entries),
            "chain_head_sha256": self._chain_head,
            "scenarios": self._pub_entries,
        }
        _atomic_write_bytes(self.sealed_path, _canonical_json_doc(sealed_doc))
        _atomic_write_bytes(self.public_path, _canonical_json_doc(pub_doc))

    def initialize_fresh(self) -> None:
        """Write an initial empty checkpoint atomically at the start of a fresh build."""
        self._pub_entries = []
        self._sealed_entries = []
        self._chain_head = CHECKPOINT_GENESIS_HASH
        self._write_both_atomically()

    def append_scenario(
        self,
        *,
        scenario_index: int,
        pair: tuple[RenderedScenario, RenderedScenario] | None,
        call_records: Sequence[GenerationCallRecord],
        world: World | None = None,
    ) -> None:
        """Append a finished scenario (kept pair or refused scenario) and its generation call records to the checkpoint atomically."""
        if pair is not None:
            r0, r1 = pair
            w_obj = world or r0.world or r1.world
            if w_obj is not None:
                check_rendered_prose(r0.prose, w_obj, focal_agent=r0.focal_agent)
                check_rendered_prose(r1.prose, w_obj, focal_agent=r1.focal_agent)
            scen_id = r0.scenario_id
            focal_agents = [r0.focal_agent, r1.focal_agent]
            prose_sha256 = [
                _sha256_bytes(r0.prose.encode("utf-8")),
                _sha256_bytes(r1.prose.encode("utf-8")),
            ]
            rendering_0_dict: dict[str, Any] | None = {
                "rendering_index": r0.rendering_index,
                "focal_agent": r0.focal_agent,
                "prose": r0.prose,
                "writer_prompt": r0.writer_prompt,
            }
            rendering_1_dict: dict[str, Any] | None = {
                "rendering_index": r1.rendering_index,
                "focal_agent": r1.focal_agent,
                "prose": r1.prose,
                "writer_prompt": r1.writer_prompt,
            }
            kept = True
        else:
            if world is None:
                raise ValueError("world is required when appending a refused scenario to checkpoint.")
            scen_id = world.scenario_id
            focal_agents = [world.agents[0], world.agents[1]]
            prose_sha256 = []
            rendering_0_dict = None
            rendering_1_dict = None
            kept = False

        prev_hash = self._chain_head
        sealed_payload: dict[str, Any] = {
            "scenario_index": scenario_index,
            "scenario_id": scen_id,
            "kept": kept,
            "rendering_0": rendering_0_dict,
            "rendering_1": rendering_1_dict,
            "call_records": [rec.to_dict() for rec in call_records],
            "prev_hash": prev_hash,
        }
        sealed_entry_sha256 = _sha256_bytes(_canonical_json_line(sealed_payload).encode("utf-8"))

        pub_payload: dict[str, Any] = {
            "scenario_index": scenario_index,
            "kept": kept,
            "focal_agents": focal_agents,
            "prose_sha256": prose_sha256,
            "call_records": [
                {
                    "call_id": rec.call_id,
                    "call_index": rec.call_index,
                    "rendering_index": rec.rendering_index,
                    "attempt": rec.attempt,
                    "seed": rec.seed,
                    "checks_passed": rec.checks_passed,
                    "eval_count": rec.eval_count,
                    "prompt_eval_count": rec.prompt_eval_count,
                    "seconds": round(rec.seconds, 6),
                    "vram_peak_mib": rec.vram_peak_mib,
                }
                for rec in call_records
            ],
            "prev_hash": prev_hash,
        }
        chain_hash = _sha256_bytes(
            f"{prev_hash}:{_canonical_json_line(pub_payload)}:{sealed_entry_sha256}".encode("utf-8")
        )

        self._sealed_entries.append(
            {
                **sealed_payload,
                "sealed_entry_sha256": sealed_entry_sha256,
                "chain_hash": chain_hash,
            }
        )
        self._pub_entries.append(
            {
                **pub_payload,
                "chain_hash": chain_hash,
            }
        )
        self._chain_head = chain_hash
        self._write_both_atomically()

    def load_and_verify(
        self,
        *,
        worlds: Sequence[World],
    ) -> tuple[
        list[tuple[RenderedScenario, RenderedScenario]],
        list[GenerationCallRecord],
        str,
    ]:
        """Load public and sealed checkpoints, verify hash chain and prose checks, and return restored state."""
        if not self.public_path.is_file() or not self.sealed_path.is_file():
            raise ValueError(
                f"Cannot --resume: checkpoint files not found in '{self.output_dir}'."
            )
        try:
            pub_doc = json.loads(self.public_path.read_text(encoding="utf-8"))
            sealed_doc = json.loads(self.sealed_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Cannot --resume: checkpoint JSON is corrupt ({exc}).") from exc

        pub_header = pub_doc.get("header", {})
        sealed_header = sealed_doc.get("header", {})
        if (
            pub_header.get("count") != self.count
            or sealed_header.get("count") != self.count
            or pub_header.get("base_seed") != self.base_seed
            or sealed_header.get("base_seed") != self.base_seed
            or pub_header.get("writer_kind") != self.writer_kind
            or sealed_header.get("writer_kind") != self.writer_kind
            or pub_header.get("writer_manifest_digest") != self.writer_pin.manifest_digest
            or sealed_header.get("writer_manifest_digest") != self.writer_pin.manifest_digest
        ):
            raise ValueError(
                "Cannot --resume: checkpoint header (count, seed, writer_kind, or writer_manifest_digest) mismatches requested run."
            )

        restored_private_seed = str(sealed_header.get("private_seed_hex") or "")
        if len(restored_private_seed) < 32:
            raise ValueError("Cannot --resume: invalid private_seed_hex in sealed checkpoint.")
        self.private_seed_hex = restored_private_seed

        pub_scenarios = pub_doc.get("scenarios")
        sealed_scenarios = sealed_doc.get("scenarios")
        if (
            not isinstance(pub_scenarios, list)
            or not isinstance(sealed_scenarios, list)
            or len(pub_scenarios) != len(sealed_scenarios)
            or pub_doc.get("scenarios_completed") != len(pub_scenarios)
            or sealed_doc.get("scenarios_completed") != len(sealed_scenarios)
            or len(sealed_scenarios) > len(worlds)
        ):
            raise ValueError("Cannot --resume: checkpoint scenario list is truncated or inconsistent.")

        restored_pairs: list[tuple[RenderedScenario, RenderedScenario]] = []
        restored_calls: list[GenerationCallRecord] = []
        running_prev = CHECKPOINT_GENESIS_HASH

        for idx, (pub_rec, sealed_rec) in enumerate(zip(pub_scenarios, sealed_scenarios, strict=True)):
            if (
                pub_rec.get("scenario_index") != idx
                or sealed_rec.get("scenario_index") != idx
                or pub_rec.get("prev_hash") != running_prev
                or sealed_rec.get("prev_hash") != running_prev
            ):
                raise ValueError(f"Cannot --resume: checkpoint hash chain broken at scenario index {idx}.")

            sealed_payload_check = {
                k: v for k, v in sealed_rec.items() if k not in ("sealed_entry_sha256", "chain_hash")
            }
            expected_sealed_sha = _sha256_bytes(_canonical_json_line(sealed_payload_check).encode("utf-8"))
            if sealed_rec.get("sealed_entry_sha256") != expected_sealed_sha:
                raise ValueError(f"Cannot --resume: sealed checkpoint entry digest mismatch at scenario {idx}.")

            pub_payload_check = {k: v for k, v in pub_rec.items() if k != "chain_hash"}
            expected_chain_hash = _sha256_bytes(
                f"{running_prev}:{_canonical_json_line(pub_payload_check)}:{expected_sealed_sha}".encode("utf-8")
            )
            if (
                pub_rec.get("chain_hash") != expected_chain_hash
                or sealed_rec.get("chain_hash") != expected_chain_hash
            ):
                raise ValueError(f"Cannot --resume: checkpoint chain_hash mismatch at scenario {idx}.")
            running_prev = expected_chain_hash

            world = worlds[idx]
            if sealed_rec.get("scenario_id") != world.scenario_id:
                raise ValueError(
                    f"Cannot --resume: checkpoint scenario_id {sealed_rec.get('scenario_id')!r} != {world.scenario_id!r}."
                )

            r0_dict = sealed_rec.get("rendering_0")
            r1_dict = sealed_rec.get("rendering_1")
            is_kept = bool(sealed_rec.get("kept", True)) and r0_dict is not None and r1_dict is not None
            if is_kept:
                assert isinstance(r0_dict, dict) and isinstance(r1_dict, dict)
                prose_0 = str(r0_dict["prose"])
                prose_1 = str(r1_dict["prose"])
                focal_0 = str(r0_dict["focal_agent"])
                focal_1 = str(r1_dict["focal_agent"])
                prompt_0 = str(r0_dict["writer_prompt"])
                prompt_1 = str(r1_dict["writer_prompt"])

                check_rendered_prose(prose_0, world, focal_agent=focal_0)
                check_rendered_prose(prose_1, world, focal_agent=focal_1)
                keys = compute_scenario_keys(world)

                restored_pairs.append(
                    (
                        RenderedScenario(
                            scenario_id=world.scenario_id,
                            rendering_index=0,
                            focal_agent=focal_0,
                            prose=prose_0,
                            writer_prompt=prompt_0,
                            keys=keys,
                            world=world,
                        ),
                        RenderedScenario(
                            scenario_id=world.scenario_id,
                            rendering_index=1,
                            focal_agent=focal_1,
                            prose=prose_1,
                            writer_prompt=prompt_1,
                            keys=keys,
                            world=world,
                        ),
                    )
                )

            for c_dict in sealed_rec.get("call_records", ()):
                restored_calls.append(
                    GenerationCallRecord(
                        call_id=str(c_dict["call_id"]),
                        call_index=int(c_dict["call_index"]),
                        scenario_id=str(c_dict["scenario_id"]),
                        rendering_index=int(c_dict["rendering_index"]),
                        focal_agent=str(c_dict["focal_agent"]),
                        operation=str(c_dict["operation"]),
                        model=str(c_dict["model"]),
                        manifest_digest=str(c_dict["manifest_digest"]),
                        model_layer_digest=str(c_dict["model_layer_digest"]),
                        licence_id=str(c_dict["licence_id"]),
                        prompt_sha256=str(c_dict["prompt_sha256"]),
                        prompt_tokens_estimate=int(c_dict["prompt_tokens_estimate"]),
                        prompt_eval_count=int(c_dict["prompt_eval_count"]),
                        eval_count=int(c_dict["eval_count"]),
                        thinking_words=int(c_dict["thinking_words"]),
                        num_predict=int(c_dict["num_predict"]),
                        num_ctx=int(c_dict["num_ctx"]),
                        max_num_ctx=int(c_dict["max_num_ctx"]),
                        think=bool(c_dict["think"]),
                        temperature=float(c_dict["temperature"]),
                        done_reason=str(c_dict["done_reason"]),
                        seconds=float(c_dict["seconds"]),
                        vram_peak_mib=int(c_dict["vram_peak_mib"]),
                        vram_halt_mib=int(c_dict["vram_halt_mib"]),
                        grant_id=str(c_dict["grant_id"]),
                        top5_logprobs=tuple(c_dict.get("top5_logprobs") or ()),
                        checks_passed=bool(c_dict.get("checks_passed", True)),
                        refusal_reasons=tuple(str(x) for x in (c_dict.get("refusal_reasons") or ())),
                        check_id=(
                            str(c_dict["check_id"])
                            if c_dict.get("check_id") is not None
                            else None
                        ),
                        event_index=(
                            int(c_dict["event_index"])
                            if c_dict.get("event_index") is not None
                            else None
                        ),
                        event_kind=(
                            str(c_dict["event_kind"])
                            if c_dict.get("event_kind") is not None
                            else None
                        ),
                        failing_sentence_sha256=str(c_dict.get("failing_sentence_sha256") or ""),
                        failing_sentence_offset=(
                            int(c_dict["failing_sentence_offset"])
                            if c_dict.get("failing_sentence_offset") is not None
                            else None
                        ),
                        failing_sentence_length=(
                            int(c_dict["failing_sentence_length"])
                            if c_dict.get("failing_sentence_length") is not None
                            else None
                        ),
                        attempt=int(c_dict.get("attempt", 0)),
                        seed=int(c_dict.get("seed", 0)),
                        response_sha256=str(c_dict.get("response_sha256") or ""),
                        thinking_sha256=str(c_dict.get("thinking_sha256") or ""),
                        repair_line=str(c_dict.get("repair_line") or ""),
                        repair_line_sha256=str(c_dict.get("repair_line_sha256") or ""),
                        retry_form=str(c_dict.get("retry_form") or ""),
                        retry_fallback_reason=str(c_dict.get("retry_fallback_reason") or ""),
                        retry_unchanged_sentences=(
                            int(c_dict["retry_unchanged_sentences"])
                            if c_dict.get("retry_unchanged_sentences") is not None
                            else None
                        ),
                        retry_attempt0_sentences=(
                            int(c_dict["retry_attempt0_sentences"])
                            if c_dict.get("retry_attempt0_sentences") is not None
                            else None
                        ),
                    )
                )

        if (
            pub_doc.get("chain_head_sha256") != running_prev
            or sealed_doc.get("chain_head_sha256") != running_prev
        ):
            raise ValueError("Cannot --resume: checkpoint chain_head_sha256 does not match end of chain.")

        self._pub_entries = [dict(r) for r in pub_scenarios]
        self._sealed_entries = [dict(r) for r in sealed_scenarios]
        self._chain_head = running_prev
        return restored_pairs, restored_calls, restored_private_seed


class OllamaWriter:
    """Real loopback Ollama writer adapter (`qwen3.6:35b`) with the same interface as `MockWriter`.

    Enforces:
    - Pinned model `qwen3.6:35b` verified against `PINNED_WRITER_MANIFEST_DIGEST` via `GET /api/tags`
      before the first `/api/chat` call (never pulls).
    - Loopback-only endpoint (`http://127.0.0.1:<port>`), optionally owning its own child
      `ollama serve` process and unloading (`keep_alive: 0`) on exit.
    - VRAM ceiling (`vram_halt_mib`, default `26000` MiB) killing the owned PID tree if exceeded.
    - `length_stop` refusal (`done_reason == 'length'`) and `max_num_ctx` refusal.
    - All prose checks (a)-(d) on every rendered scenario, recording any refusal with a reason.
    - Per-call telemetry written to `generations.jsonl` per design §6.8.
    """

    def __init__(
        self,
        *,
        model_pin: PinnedWriterModelSpec = DEFAULT_WRITER_MODEL_PIN,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        num_predict: int | None = None,
        max_num_ctx: int = DEFAULT_WRITER_MAX_NUM_CTX,
        think: bool | None = None,
        vram_halt_mib: int = DEFAULT_VRAM_HALT_MIB,
        grant_id: str | None = None,
        http_transport: OllamaHttpTransport | None = None,
        vram_probe: Callable[[], int] | None = None,
        halt_pid: int | None = None,
        pid_killer: Callable[[int], None] = kill_pid_tree,
        generations_path: Path | None = None,
        sealed_responses_path: Path | None = None,
        max_retries: int = 0,
        start_owned_server: bool | None = None,
        popen_factory: Callable[..., Any] | None = None,
        server_port: int | None = None,
        ollama_models_dir: Path | str | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        resolved_think = bool(model_pin.think if think is None else think)
        resolved_num_predict = (
            int(num_predict)
            if num_predict is not None
            else (DEFAULT_WRITER_THINK_NUM_PREDICT if resolved_think else DEFAULT_WRITER_NUM_PREDICT)
        )
        if resolved_num_predict < 1:
            raise ValueError(f"num_predict must be >= 1, got {resolved_num_predict}")
        if max_num_ctx < NUM_CTX_STEP_TOKENS:
            raise ValueError(f"max_num_ctx must be >= {NUM_CTX_STEP_TOKENS}, got {max_num_ctx}")
        if max_num_ctx > DEFAULT_WRITER_MAX_NUM_CTX:
            raise MaxNumCtxRefusalError(
                f"REFUSED (MAX_NUM_CTX_CAP): max_num_ctx={max_num_ctx} exceeds the hard VRAM ceiling limit "
                f"{DEFAULT_WRITER_MAX_NUM_CTX}."
            )
        self._model_pin = replace(
            model_pin,
            num_predict=resolved_num_predict,
            max_num_ctx=int(max_num_ctx),
            think=resolved_think,
        )
        self._num_predict = resolved_num_predict
        self._max_num_ctx = int(max_num_ctx)
        self._think = resolved_think
        self._prompt_template = load_writer_prompt_template()

        is_real_transport = http_transport is None or http_transport is default_ollama_http_transport
        resolved_grant = (grant_id or os.environ.get("UDLAB_GPU_GRANT_ID") or "").strip()
        if is_real_transport and not resolved_grant:
            raise PermissionError(
                "REFUSED (GPU_GRANT_REQUIRED): Refused to run OllamaWriter with live HTTP transport "
                "without an explicit GPU grant id (--grant-id or UDLAB_GPU_GRANT_ID)."
            )
        self._grant_id = resolved_grant or "test-transport-no-gpu"

        raw_transport = http_transport if http_transport is not None else default_ollama_http_transport
        should_own_server = (
            bool(start_owned_server)
            if start_owned_server is not None
            else (http_transport is None)
        )
        self._owned_server: OwnedOllamaServer | None = None
        if should_own_server:
            self._owned_server = start_owned_ollama_server(
                http_transport=raw_transport,
                popen_factory=popen_factory,
                port=server_port,
                ollama_models_dir=ollama_models_dir,
                pid_killer=pid_killer,
            )
            active_url = self._owned_server.base_url
            resolved_halt_pid = self._owned_server.pid if halt_pid is None else halt_pid
        else:
            active_url = require_loopback_ollama_url(base_url)
            resolved_halt_pid = halt_pid

        try:
            self._base_url = require_loopback_ollama_url(active_url)
            resolved_vram_probe = (
                vram_probe
                if vram_probe is not None
                else (query_gpu_vram_peak_mib if is_real_transport else (lambda: 0))
            )
            self._gpu_guard = GpuGuardMonitor(
                grant_id=self._grant_id,
                vram_halt_mib=vram_halt_mib,
                vram_probe=resolved_vram_probe,
                halt_pid=resolved_halt_pid,
                pid_killer=pid_killer,
            )
            self._transport: OllamaHttpTransport = self._gpu_guard.wrap_transport(raw_transport)
            self._generations_path = generations_path
            self._sealed_responses_path = sealed_responses_path
            self._max_retries = int(max_retries)
            self._verified_local_digest = False
            self._call_records: list[GenerationCallRecord] = []
            self._sealed_responses: list[SealedResponseEntry] = []
            self._unload_records: list[OllamaUnloadRecord] = []
            self._refusals: list[ProseRefusalRecord] = []
            self._pending_sealed_failing_sentence: str | None = None
            self._pending_sealed_refusal_reasons: tuple[str, ...] = ()
        except Exception:
            if self._owned_server is not None:
                self._owned_server.stop()
            raise

    @property
    def model_pin(self) -> PinnedWriterModelSpec:
        return self._model_pin

    @property
    def ollama_tag(self) -> str:
        return self._model_pin.ollama_tag

    @property
    def manifest_digest(self) -> str:
        return self._model_pin.manifest_digest

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def num_predict(self) -> int:
        return self._num_predict

    @property
    def max_num_ctx(self) -> int:
        return self._max_num_ctx

    @property
    def think(self) -> bool:
        return self._think

    @property
    def vram_halt_mib(self) -> int:
        return self._gpu_guard.vram_halt_mib

    @property
    def peak_vram_mib(self) -> int:
        return self._gpu_guard.peak_vram_mib

    @property
    def grant_id(self) -> str:
        return self._grant_id

    @property
    def owned_server(self) -> OwnedOllamaServer | None:
        return self._owned_server

    @property
    def call_records(self) -> tuple[GenerationCallRecord, ...]:
        return tuple(self._call_records)

    @property
    def sealed_responses(self) -> tuple[SealedResponseEntry, ...]:
        return tuple(self._sealed_responses)

    @property
    def max_retries(self) -> int:
        return self._max_retries

    @property
    def unload_records(self) -> tuple[OllamaUnloadRecord, ...]:
        return tuple(self._unload_records)

    @property
    def refusals(self) -> tuple[ProseRefusalRecord, ...]:
        return tuple(self._refusals)

    def restore_call_records(self, records: Sequence[GenerationCallRecord]) -> None:
        """Restore checkpointed `GenerationCallRecord` entries on `--resume` and rewrite `generations.jsonl`."""
        self._call_records = list(records)
        if self._generations_path is not None:
            self._generations_path.parent.mkdir(parents=True, exist_ok=True)
            lines = "".join(_canonical_json_line(r.to_dict()) + "\n" for r in self._call_records)
            _atomic_write_bytes(self._generations_path, lines.encode("utf-8"))

    def _append_generation_record(
        self,
        rec: GenerationCallRecord,
        *,
        response_text: str | None = None,
        thinking_text: str = "",
    ) -> None:
        self._call_records.append(rec)
        if self._generations_path is not None:
            self._generations_path.parent.mkdir(parents=True, exist_ok=True)
            with self._generations_path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(_canonical_json_line(rec.to_dict()) + "\n")
        if response_text is not None:
            sealed_entry = SealedResponseEntry(
                call_id=rec.call_id,
                call_index=rec.call_index,
                scenario_id=rec.scenario_id,
                rendering_index=rec.rendering_index,
                focal_agent=rec.focal_agent,
                attempt=rec.attempt,
                seed=rec.seed,
                response_text=response_text,
                thinking_text=thinking_text,
                response_sha256=rec.response_sha256,
                thinking_sha256=rec.thinking_sha256,
                checks_passed=rec.checks_passed,
                check_id=rec.check_id,
                event_index=rec.event_index,
                event_kind=rec.event_kind,
                failing_sentence=self._pending_sealed_failing_sentence,
                failing_sentence_sha256=rec.failing_sentence_sha256,
                failing_sentence_offset=rec.failing_sentence_offset,
                failing_sentence_length=rec.failing_sentence_length,
                refusal_reasons=self._pending_sealed_refusal_reasons,
            )
            self._pending_sealed_failing_sentence = None
            self._pending_sealed_refusal_reasons = ()
            self._sealed_responses.append(sealed_entry)
            if self._sealed_responses_path is not None:
                append_sealed_response(self._sealed_responses_path, sealed_entry)

    def _ensure_verified(self) -> None:
        if not self._verified_local_digest:
            verify_local_ollama_model_digest(
                base_url=self._base_url,
                ollama_tag=self.ollama_tag,
                expected_digest=self.manifest_digest,
                http_transport=self._transport,
            )
            self._verified_local_digest = True

    def _chat_render(
        self,
        *,
        world: World,
        scenario_id: str,
        rendering_index: int,
        focal_agent: str,
        prompt_text: str,
        allow_refusal: bool = False,
    ) -> str | None:
        self._ensure_verified()
        messages = [
            {
                "role": "user",
                "content": prompt_text,
            }
        ]
        prompt_sha = _sha256_bytes(_canonical_json_line(messages).encode("utf-8"))
        prompt_tokens_estimate = estimate_prompt_tokens(messages)
        if self._think and prompt_tokens_estimate + self._num_predict <= self._max_num_ctx:
            num_ctx = min(
                self._max_num_ctx,
                compute_num_ctx(
                    prompt_tokens_estimate,
                    self._num_predict,
                    margin=NUM_CTX_MARGIN_TOKENS,
                    step=NUM_CTX_STEP_TOKENS,
                ),
            )
        else:
            num_ctx = compute_num_ctx(
                prompt_tokens_estimate,
                self._num_predict,
                margin=NUM_CTX_MARGIN_TOKENS,
                step=NUM_CTX_STEP_TOKENS,
            )

        base_seed = int(world.seed)

        if num_ctx > self._max_num_ctx or (prompt_tokens_estimate + self._num_predict > self._max_num_ctx):
            call_index = len(self._call_records) + 1
            call_id = f"{scenario_id}:r{rendering_index}:call_{call_index:04d}"
            reason_msg = (
                f"Writer call '{call_id}' requires num_ctx={num_ctx} "
                f"(prompt_tokens_estimate={prompt_tokens_estimate} + num_predict={self._num_predict} + "
                f"margin={NUM_CTX_MARGIN_TOKENS}), which exceeds max_num_ctx={self._max_num_ctx}."
            )
            rec = GenerationCallRecord(
                call_id=call_id,
                call_index=call_index,
                scenario_id=scenario_id,
                rendering_index=rendering_index,
                focal_agent=focal_agent,
                operation="render_scenario",
                model=self.ollama_tag,
                manifest_digest=self.manifest_digest,
                model_layer_digest=self._model_pin.model_layer_digest,
                licence_id=self._model_pin.licence_id,
                prompt_sha256=prompt_sha,
                prompt_tokens_estimate=prompt_tokens_estimate,
                prompt_eval_count=0,
                eval_count=0,
                thinking_words=0,
                num_predict=self._num_predict,
                num_ctx=num_ctx,
                max_num_ctx=self._max_num_ctx,
                think=self._think,
                temperature=0.0,
                done_reason="max_num_ctx_exceeded",
                seconds=0.0,
                vram_peak_mib=self._gpu_guard.peak_vram_mib,
                vram_halt_mib=self._gpu_guard.vram_halt_mib,
                grant_id=self._grant_id,
                checks_passed=False,
                refusal_reasons=(format_non_sealed_refusal_reason(check_id="max_num_ctx_exceeded"),),
                check_id="max_num_ctx_exceeded",
                attempt=0,
                seed=base_seed,
            )
            self._append_generation_record(rec)
            self._refusals.append(
                ProseRefusalRecord(
                    scenario_id=scenario_id,
                    rendering_index=rendering_index,
                    focal_agent=focal_agent,
                    check_id="max_num_ctx_exceeded",
                    reason=reason_msg,
                )
            )
            raise MaxNumCtxRefusalError(reason_msg)

        chat_url = f"{self._base_url}/api/chat"
        last_exc: RenderCheckError | None = None
        attempt_0_content: str | None = None
        attempt_0_response_sha256: str = ""
        attempt_0_eval_pair: tuple[int, int] = (0, 0)

        for attempt in range(self._max_retries + 1):
            call_index = len(self._call_records) + 1
            call_id = f"{scenario_id}:r{rendering_index}:call_{call_index:04d}"
            call_seed = base_seed
            if attempt > 0 and last_exc is not None:
                mt_repair_line = format_repair_retry_line(last_exc, retry_form="multi_turn")
                mt_repair_tokens = estimate_prompt_tokens(mt_repair_line)
                p0_tokens = (
                    attempt_0_eval_pair[0]
                    if attempt_0_eval_pair[0] > 0
                    else prompt_tokens_estimate
                )
                e0_tokens = (
                    attempt_0_eval_pair[1]
                    if attempt_0_eval_pair[0] > 0
                    else estimate_prompt_tokens(attempt_0_content or "")
                )
                multiturn_need = (
                    p0_tokens + e0_tokens + mt_repair_tokens + self._num_predict + NUM_CTX_MARGIN_TOKENS
                )
                if multiturn_need <= self._max_num_ctx:
                    if (
                        attempt_0_content is None
                        or not attempt_0_response_sha256
                        or _sha256_bytes(attempt_0_content.encode("utf-8")) != attempt_0_response_sha256
                    ):
                        raise AssertionError(
                            "Multi-turn retry assistant turn is not byte-identical to the sealed attempt-0 response."
                        )
                    if self._sealed_responses:
                        sealed_a0 = next(
                            (
                                sr
                                for sr in reversed(self._sealed_responses)
                                if sr.scenario_id == scenario_id
                                and sr.rendering_index == rendering_index
                                and sr.attempt == 0
                            ),
                            None,
                        )
                        if (
                            sealed_a0 is None
                            or sealed_a0.response_sha256 != attempt_0_response_sha256
                            or _sha256_bytes(attempt_0_content.encode("utf-8")) != sealed_a0.response_sha256
                        ):
                            raise AssertionError(
                                "Multi-turn retry assistant turn sha256 does not match sealed_store attempt-0 response_sha256."
                            )
                    retry_form = "multi_turn"
                    retry_fallback_reason = ""
                    repair_line = mt_repair_line
                    repair_line_sha = _sha256_bytes(repair_line.encode("utf-8"))
                    attempt_messages = [
                        {
                            "role": "user",
                            "content": prompt_text,
                        },
                        {
                            "role": "assistant",
                            "content": attempt_0_content,
                        },
                        {
                            "role": "user",
                            "content": repair_line,
                        },
                    ]
                    attempt_num_ctx = compute_num_ctx(
                        p0_tokens + e0_tokens + mt_repair_tokens,
                        self._num_predict,
                        margin=NUM_CTX_MARGIN_TOKENS,
                        step=NUM_CTX_STEP_TOKENS,
                    )
                else:
                    retry_form = "single_turn"
                    repair_line = format_repair_retry_line(last_exc, retry_form="single_turn")
                    repair_line_sha = _sha256_bytes(repair_line.encode("utf-8"))
                    st_repair_tokens = estimate_prompt_tokens(repair_line)
                    retry_fallback_reason = (
                        f"multiturn_need_exceeds_max_num_ctx: need={multiturn_need} "
                        f"(prompt_eval_count={p0_tokens} + eval_count={e0_tokens} + "
                        f"repair_tokens={mt_repair_tokens} + num_predict={self._num_predict} + "
                        f"margin={NUM_CTX_MARGIN_TOKENS}) > max_num_ctx={self._max_num_ctx}"
                    )
                    single_turn_prompt = f"{prompt_text.rstrip('\n')}\n{repair_line}"
                    attempt_messages = [
                        {
                            "role": "user",
                            "content": single_turn_prompt,
                        }
                    ]
                    single_turn_tokens = p0_tokens + st_repair_tokens
                    if single_turn_tokens + self._num_predict + NUM_CTX_MARGIN_TOKENS <= self._max_num_ctx:
                        attempt_num_ctx = compute_num_ctx(
                            single_turn_tokens,
                            self._num_predict,
                            margin=NUM_CTX_MARGIN_TOKENS,
                            step=NUM_CTX_STEP_TOKENS,
                        )
                    else:
                        attempt_num_ctx = self._max_num_ctx
            else:
                repair_line = ""
                repair_line_sha = ""
                retry_form = ""
                retry_fallback_reason = ""
                attempt_messages = [
                    {
                        "role": "user",
                        "content": prompt_text,
                    }
                ]
                attempt_num_ctx = num_ctx
            attempt_prompt_sha = _sha256_bytes(_canonical_json_line(attempt_messages).encode("utf-8"))
            attempt_prompt_est = estimate_prompt_tokens(attempt_messages)
            payload: dict[str, Any] = {
                "model": self.ollama_tag,
                "messages": attempt_messages,
                "think": self._think,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "seed": call_seed,
                    "num_predict": self._num_predict,
                    "num_ctx": attempt_num_ctx,
                },
            }
            t0 = time.perf_counter()
            resp = self._transport("POST", chat_url, payload)
            wall_s = max(0.0, time.perf_counter() - t0)

            msg = resp.get("message")
            if not isinstance(msg, dict):
                msg = {}
            content = str(msg.get("content") or "").strip()
            thinking_text = str(msg.get("thinking") or "")
            done_reason = str(resp.get("done_reason") or "stop").strip()
            resp_sha, think_sha = compute_sealed_response_digests(content, thinking_text)

            raw_eval = resp.get("eval_count")
            eval_count = (
                raw_eval
                if isinstance(raw_eval, int) and not isinstance(raw_eval, bool) and raw_eval >= 0
                else len(content.split())
            )
            raw_prompt_eval = resp.get("prompt_eval_count")
            prompt_eval_count = (
                raw_prompt_eval
                if isinstance(raw_prompt_eval, int) and not isinstance(raw_prompt_eval, bool) and raw_prompt_eval >= 0
                else 0
            )
            thinking_words = len(thinking_text.split()) if thinking_text.strip() else 0
            raw_logprobs = resp.get("logprobs")
            top5_logprobs = _sanitize_non_sealed_logprobs(raw_logprobs)
            if attempt == 0:
                attempt_0_content = content
                attempt_0_response_sha256 = resp_sha
                attempt_0_eval_pair = (prompt_eval_count, eval_count)
                retry_unchanged: int | None = None
                retry_attempt0_total: int | None = None
            else:
                retry_unchanged, retry_attempt0_total = _count_unchanged_sentences(
                    attempt_0_content or "", content
                )

            if done_reason == "length":
                reason_msg = (
                    f"Writer call '{call_id}' stopped with done_reason='length' "
                    f"(num_predict={self._num_predict}, eval_count={eval_count})."
                )
                rec = GenerationCallRecord(
                    call_id=call_id,
                    call_index=call_index,
                    scenario_id=scenario_id,
                    rendering_index=rendering_index,
                    focal_agent=focal_agent,
                    operation="render_scenario",
                    model=self.ollama_tag,
                    manifest_digest=self.manifest_digest,
                    model_layer_digest=self._model_pin.model_layer_digest,
                    licence_id=self._model_pin.licence_id,
                    prompt_sha256=attempt_prompt_sha,
                    prompt_tokens_estimate=attempt_prompt_est,
                    prompt_eval_count=prompt_eval_count,
                    eval_count=eval_count,
                    thinking_words=thinking_words,
                    num_predict=self._num_predict,
                    num_ctx=attempt_num_ctx,
                    max_num_ctx=self._max_num_ctx,
                    think=self._think,
                    temperature=0.0,
                    done_reason=done_reason,
                    seconds=wall_s,
                    vram_peak_mib=self._gpu_guard.peak_vram_mib,
                    vram_halt_mib=self._gpu_guard.vram_halt_mib,
                    grant_id=self._grant_id,
                    top5_logprobs=top5_logprobs,
                    checks_passed=False,
                    refusal_reasons=(format_non_sealed_refusal_reason(check_id="length_stop"),),
                    check_id="length_stop",
                    attempt=attempt,
                    seed=call_seed,
                    response_sha256=resp_sha,
                    thinking_sha256=think_sha,
                    repair_line=repair_line,
                    repair_line_sha256=repair_line_sha,
                    retry_form=retry_form,
                    retry_fallback_reason=retry_fallback_reason,
                    retry_unchanged_sentences=retry_unchanged,
                    retry_attempt0_sentences=retry_attempt0_total,
                )
                self._pending_sealed_failing_sentence = None
                self._pending_sealed_refusal_reasons = (f"length_stop: {reason_msg}",)
                self._append_generation_record(
                    rec,
                    response_text=content,
                    thinking_text=thinking_text,
                )
                self._refusals.append(
                    ProseRefusalRecord(
                        scenario_id=scenario_id,
                        rendering_index=rendering_index,
                        focal_agent=focal_agent,
                        check_id="length_stop",
                        reason=reason_msg,
                    )
                )
                raise LengthStopRefusalError(reason_msg)

            try:
                check_rendered_prose(content, world, focal_agent=focal_agent)
            except RenderCheckError as exc:
                last_exc = exc
                fail_sha, fail_off, fail_len = locate_failing_sentence_in_response(
                    content, exc.failing_sentence
                )
                non_sealed_reason = format_non_sealed_refusal_reason(
                    check_id=exc.check_id,
                    event_index=exc.event_index,
                    event_kind=exc.event_kind,
                    failing_sentence_sha256=fail_sha,
                    failing_sentence_offset=fail_off,
                    failing_sentence_length=fail_len,
                )
                full_sealed_reason = f"{exc.check_id}: {exc}"
                rec = GenerationCallRecord(
                    call_id=call_id,
                    call_index=call_index,
                    scenario_id=scenario_id,
                    rendering_index=rendering_index,
                    focal_agent=focal_agent,
                    operation="render_scenario",
                    model=self.ollama_tag,
                    manifest_digest=self.manifest_digest,
                    model_layer_digest=self._model_pin.model_layer_digest,
                    licence_id=self._model_pin.licence_id,
                    prompt_sha256=attempt_prompt_sha,
                    prompt_tokens_estimate=attempt_prompt_est,
                    prompt_eval_count=prompt_eval_count,
                    eval_count=eval_count,
                    thinking_words=thinking_words,
                    num_predict=self._num_predict,
                    num_ctx=attempt_num_ctx,
                    max_num_ctx=self._max_num_ctx,
                    think=self._think,
                    temperature=0.0,
                    done_reason=done_reason,
                    seconds=wall_s,
                    vram_peak_mib=self._gpu_guard.peak_vram_mib,
                    vram_halt_mib=self._gpu_guard.vram_halt_mib,
                    grant_id=self._grant_id,
                    top5_logprobs=top5_logprobs,
                    checks_passed=False,
                    refusal_reasons=(non_sealed_reason,),
                    check_id=exc.check_id,
                    event_index=exc.event_index,
                    event_kind=exc.event_kind,
                    failing_sentence_sha256=fail_sha,
                    failing_sentence_offset=fail_off,
                    failing_sentence_length=fail_len,
                    attempt=attempt,
                    seed=call_seed,
                    response_sha256=resp_sha,
                    thinking_sha256=think_sha,
                    repair_line=repair_line,
                    repair_line_sha256=repair_line_sha,
                    retry_form=retry_form,
                    retry_fallback_reason=retry_fallback_reason,
                    retry_unchanged_sentences=retry_unchanged,
                    retry_attempt0_sentences=retry_attempt0_total,
                )
                self._pending_sealed_failing_sentence = exc.failing_sentence
                self._pending_sealed_refusal_reasons = (full_sealed_reason,)
                self._append_generation_record(
                    rec,
                    response_text=content,
                    thinking_text=thinking_text,
                )
                self._refusals.append(
                    ProseRefusalRecord(
                        scenario_id=scenario_id,
                        rendering_index=rendering_index,
                        focal_agent=focal_agent,
                        check_id=exc.check_id,
                        reason=str(exc),
                        failing_sentence=exc.failing_sentence,
                        event_index=exc.event_index,
                        event_kind=exc.event_kind,
                        failing_sentence_sha256=fail_sha,
                        failing_sentence_offset=fail_off,
                        failing_sentence_length=fail_len,
                    )
                )
                if attempt < self._max_retries:
                    continue
                if allow_refusal:
                    return None
                raise

            rec = GenerationCallRecord(
                call_id=call_id,
                call_index=call_index,
                scenario_id=scenario_id,
                rendering_index=rendering_index,
                focal_agent=focal_agent,
                operation="render_scenario",
                model=self.ollama_tag,
                manifest_digest=self.manifest_digest,
                model_layer_digest=self._model_pin.model_layer_digest,
                licence_id=self._model_pin.licence_id,
                prompt_sha256=attempt_prompt_sha,
                prompt_tokens_estimate=attempt_prompt_est,
                prompt_eval_count=prompt_eval_count,
                eval_count=eval_count,
                thinking_words=thinking_words,
                num_predict=self._num_predict,
                num_ctx=attempt_num_ctx,
                max_num_ctx=self._max_num_ctx,
                think=self._think,
                temperature=0.0,
                done_reason=done_reason,
                seconds=wall_s,
                vram_peak_mib=self._gpu_guard.peak_vram_mib,
                vram_halt_mib=self._gpu_guard.vram_halt_mib,
                grant_id=self._grant_id,
                top5_logprobs=top5_logprobs,
                checks_passed=True,
                refusal_reasons=(),
                attempt=attempt,
                seed=call_seed,
                response_sha256=resp_sha,
                thinking_sha256=think_sha,
                repair_line=repair_line,
                repair_line_sha256=repair_line_sha,
                retry_form=retry_form,
                retry_fallback_reason=retry_fallback_reason,
                retry_unchanged_sentences=retry_unchanged,
                retry_attempt0_sentences=retry_attempt0_total,
            )
            self._append_generation_record(rec, response_text=content, thinking_text=thinking_text)
            return content

        if allow_refusal:
            return None
        assert last_exc is not None
        raise last_exc

    def render_scenario_pair(
        self,
        world: World,
        *,
        allow_refusal: bool = False,
    ) -> tuple[RenderedScenario, RenderedScenario] | None:
        """Render `world` twice via `qwen3.6:35b`, run all prose checks (a)-(d), and verify key invariance."""
        focal_0 = world.agents[0]
        focal_1 = world.agents[1]

        prompt_0 = format_writer_prompt(world, focal_0, self._prompt_template)
        prompt_1 = format_writer_prompt(world, focal_1, self._prompt_template)

        prose_0 = self._chat_render(
            world=world,
            scenario_id=world.scenario_id,
            rendering_index=0,
            focal_agent=focal_0,
            prompt_text=prompt_0,
            allow_refusal=allow_refusal,
        )
        prose_1 = self._chat_render(
            world=world,
            scenario_id=world.scenario_id,
            rendering_index=1,
            focal_agent=focal_1,
            prompt_text=prompt_1,
            allow_refusal=allow_refusal,
        )
        if prose_0 is None or prose_1 is None:
            return None

        keys_0 = compute_scenario_keys(_reconstruct_world_from_writer_prompt(world, prompt_0))
        keys_1 = compute_scenario_keys(_reconstruct_world_from_writer_prompt(world, prompt_1))
        if keys_0 != keys_1:
            raise AssertionError("Perspective-flip invariance violated: keys differ between renderings.")

        return (
            RenderedScenario(
                scenario_id=world.scenario_id,
                rendering_index=0,
                focal_agent=focal_0,
                prose=prose_0,
                writer_prompt=prompt_0,
                keys=keys_0,
                world=world,
            ),
            RenderedScenario(
                scenario_id=world.scenario_id,
                rendering_index=1,
                focal_agent=focal_1,
                prose=prose_1,
                writer_prompt=prompt_1,
                keys=keys_1,
                world=world,
            ),
        )

    def unload_model(self) -> OllamaUnloadRecord:
        """Unload the writer model from VRAM using `keep_alive: 0`."""
        chat_url = f"{self._base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": self.ollama_tag,
            "messages": [],
            "keep_alive": 0,
            "stream": False,
        }
        t0 = time.perf_counter()
        self._transport("POST", chat_url, payload)
        wall_s = max(0.0, time.perf_counter() - t0)
        rec = OllamaUnloadRecord(
            model=self.ollama_tag,
            manifest_digest=self.manifest_digest,
            keep_alive=0,
            wall_time_s=wall_s,
        )
        self._unload_records.append(rec)
        if self._owned_server is not None:
            self._owned_server.unloaded = True
        return rec

    def close(self) -> None:
        """Unload model (if verified) and stop the owned child `ollama serve` process."""
        try:
            if self._verified_local_digest and not self._unload_records:
                try:
                    self.unload_model()
                except Exception:
                    pass
        finally:
            if self._owned_server is not None:
                self._owned_server.stop()
