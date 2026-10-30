"""Deterministic third-person perspective renderer, writer prompt, prose checks, and OllamaWriter."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from typing import Any
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
DEFAULT_WRITER_MAX_NUM_CTX: int = 9216
DEFAULT_VRAM_HALT_MIB: int = 26_000
NUM_CTX_MARGIN_TOKENS: int = 256
NUM_CTX_STEP_TOKENS: int = 1024
PROMPT_TOKEN_ESTIMATE_RULE: str = "ceil(utf8_bytes / 3)"
CHECKPOINT_FILENAME: str = "checkpoint.json"
SEALED_CHECKPOINT_FILENAME: str = "checkpoint.sealed.json"
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
    think: bool = True

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

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "rendering_index": self.rendering_index,
            "focal_agent": self.focal_agent,
            "check_id": self.check_id,
            "reason": self.reason,
        }


class RenderCheckError(ValueError):
    """Raised when rendered scenario prose fails any check (a)-(d) or writer constraint."""

    def __init__(self, message: str, *, check_id: str = "render_check") -> None:
        super().__init__(message)
        self.check_id = check_id
        self.reason = message


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


def load_writer_prompt_template(path: Path | None = None) -> str:
    """Load the `qwen3.6:35b` writer prompt text file and verify banned-adverb and belief-rule coverage."""
    target = path or WRITER_PROMPT_PATH
    text = target.read_text(encoding="utf-8")
    missing = [w for w in BANNED_ADVERBS if w not in text.lower()]
    if missing:
        raise ValueError(f"Writer prompt template is missing banned adverbs: {missing}")
    rule_count = text.count(BELIEF_PERSISTENCE_RULE)
    if rule_count != 1:
        raise ValueError(
            f"Writer prompt template must contain BELIEF_PERSISTENCE_RULE byte-identically exactly once; found {rule_count}."
        )
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

    formatted = (
        raw_template.replace("{focal_agent}", focal_agent)
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


def _split_prose_sentences(prose: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(prose) if s.strip()]


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


_ACTION_VERBS_BY_KIND: dict[str, tuple[str, ...]] = {
    "enter": ("walked", "entered", "stepped", "came", "arrived", "went", "moved", "returned"),
    "leave": ("exited", "left", "departed", "walked", "headed", "went", "stepped"),
    "move": ("moved", "stepped", "placed", "put", "shifted", "transferred", "adjusted", "turned", "pulled", "set"),
    "see": ("observed", "looked", "saw", "spotted", "noticed", "inspected", "checked", "viewed", "watched"),
    "say": ("said", "stated", "spoke", "announced", "uttered", "claimed", "reported", "told"),
    "message": ("sent", "messaged", "texted", "transmitted", "notified", "wrote"),
}


def _sentence_matches_event(sentence: str, ev: Event, noise_threshold: int) -> bool:
    """Return True if `sentence` states `ev` with its actor, action, object, place, and room."""
    if _is_setup_sentence(sentence):
        return False
    s_low = sentence.lower()
    if not re.search(rf"\b{re.escape(ev.actor.lower())}\b", s_low):
        return False
    if ev.room.lower() not in s_low:
        return False
    action_verbs = _ACTION_VERBS_BY_KIND.get(ev.kind, ())
    if action_verbs and not any(re.search(rf"\b{re.escape(v)}\b", s_low) for v in action_verbs):
        return False

    if ev.kind == "enter":
        return bool(ev.from_location and ev.from_location.lower() in s_low)

    if ev.kind == "leave":
        return bool(ev.to_location and ev.to_location.lower() in s_low)

    if ev.kind == "move":
        if is_station_transition_event(ev):
            return bool(
                ev.target_object
                and ev.target_object.lower() in s_low
                and ev.from_location
                and ev.from_location.lower() in s_low
                and ev.to_location
                and ev.to_location.lower() in s_low
            )
        if not (
            ev.target_object
            and ev.target_object.lower() in s_low
            and ev.from_location
            and ev.from_location.lower() in s_low
            and ev.to_location
            and ev.to_location.lower() in s_low
        ):
            return False
        if ev.occluded and ev.occluder:
            if ev.occluder.lower() not in s_low or "blocked" not in s_low:
                return False
        else:
            if "open view" not in s_low:
                return False
        return True

    if ev.kind == "see":
        if not (ev.target_object and ev.target_object.lower() in s_low):
            return False
        if ev.occluded and ev.occluder:
            return (
                ev.occluder.lower() in s_low
                and "blocked" in s_low
                and FAR_SIDE_ALCOVE_STATION.lower() in s_low
            )
        if "open view" not in s_low or "blocked" in s_low:
            return False
        if ev.to_location and ev.to_location.lower() not in s_low:
            return False
        return True

    if ev.kind == "say":
        if f"{ev.noise_level} db" not in s_low:
            return False
        if ev.utterance and ev.utterance.lower() not in s_low:
            return False
        return True

    if ev.kind == "message":
        if not (ev.recipient and re.search(rf"\b{re.escape(ev.recipient.lower())}\b", s_low)):
            return False
        if ev.utterance and ev.utterance.lower() not in s_low:
            return False
        return True

    return False


def check_prose_no_markers(prose: str) -> None:
    """Prose check (a): refuse if any bracket marker `[...]`, fact ID `F0`..`F9`, or `step N` survives."""
    bracket_match = _BRACKET_MARKER_RE.search(prose)
    if bracket_match is not None:
        raise RenderCheckError(
            f"Render check (a) failed: surviving bracket marker {bracket_match.group(0)!r} in prose.",
            check_id="a_no_markers_bracket",
        )
    fact_match = _FACT_ID_RE.search(prose)
    if fact_match is not None:
        raise RenderCheckError(
            f"Render check (a) failed: surviving fact ID {fact_match.group(0)!r} in prose.",
            check_id="a_no_markers_fact_id",
        )
    step_match = _STEP_NUMBER_RE.search(prose)
    if step_match is not None:
        raise RenderCheckError(
            f"Render check (a) failed: surviving step number {step_match.group(0)!r} in prose.",
            check_id="a_no_markers_step_number",
        )


def check_prose_banned_adverbs(prose: str) -> tuple[str, ...]:
    """Prose check (b): refuse if any banned mental/emotional adverb from `BANNED_ADVERBS` appears."""
    hits = tuple(dict.fromkeys(m.group(1).lower() for m in _BANNED_PATTERN.finditer(prose)))
    if hits:
        raise RenderCheckError(
            f"Render check (b) failed: prose contains banned unlogged mental/emotional adverb(s): {hits}",
            check_id="b_banned_adverbs",
        )
    return ()


def check_prose_rule_and_hearing_threshold(prose: str, world: World) -> None:
    """Prose check (d): refuse unless the belief-persistence rule sentence and hearing threshold appear in the setup prose."""
    paragraphs = [p.strip() for p in prose.split("\n\n") if p.strip()]
    setup_text = paragraphs[0] if paragraphs else prose
    rule_count = setup_text.count(BELIEF_PERSISTENCE_RULE)
    if rule_count != 1:
        raise RenderCheckError(
            f"Render check (d) failed: setup prose must contain BELIEF_PERSISTENCE_RULE byte-identically once (found {rule_count}).",
            check_id="d_rule_sentence",
        )
    threshold_str = f"{world.noise_threshold} dB"
    if threshold_str.lower() not in setup_text.lower() or "doorway" not in setup_text.lower():
        raise RenderCheckError(
            f"Render check (d) failed: setup prose is missing the doorway hearing threshold '{threshold_str}'.",
            check_id="d_hearing_threshold",
        )


def check_prose_render_fidelity(
    prose: str,
    world: World,
    focal_agent: str | None = None,
) -> None:
    """Prose check (c): validate complete event-log and setup fidelity using `world` (never keys).

    Verifies:
    1. Setup states every room, container pair, occluder (between `far-side alcove` and `container bay`),
       doorway edge, initial agent room, absent party, goal, and stake edge.
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

    if focal_agent is not None and focal_agent.lower() not in setup_low:
        raise RenderCheckError(
            f"Render check (c) failed: focal_agent '{focal_agent}' is missing from setup prose.",
            check_id="c_fidelity_focal_agent",
        )

    # 1. Verify per-room containers and occluders in setup
    for room in world.rooms:
        if room.lower() not in setup_low:
            raise RenderCheckError(
                f"Render check (c) failed: room '{room}' is missing from setup prose.",
                check_id="c_fidelity_room_setup",
            )
        c0, c1 = world.containers_by_room[room]
        if c0.lower() not in setup_low or c1.lower() not in setup_low:
            raise RenderCheckError(
                f"Render check (c) failed: containers ({c0!r}, {c1!r}) for room '{room}' missing from setup prose.",
                check_id="c_fidelity_containers",
            )
        occ = world.occluders_by_room[room]
        if occ.lower() not in setup_low:
            raise RenderCheckError(
                f"Render check (c) failed: occluder '{occ}' for room '{room}' missing from setup prose.",
                check_id="c_fidelity_occluder",
            )

    if CONTAINER_BAY_STATION.lower() not in setup_low or FAR_SIDE_ALCOVE_STATION.lower() not in setup_low:
        raise RenderCheckError(
            "Render check (c) failed: station names ('container bay', 'far-side alcove') missing from setup prose.",
            check_id="c_fidelity_stations",
        )

    # Verify doorways
    for r_a, r_b in world.doorways:
        adj_a_b = f"{r_a.lower()} connects to the {r_b.lower()}"
        adj_b_a = f"{r_b.lower()} connects to the {r_a.lower()}"
        if adj_a_b not in setup_low and adj_b_a not in setup_low:
            raise RenderCheckError(
                f"Render check (c) failed: doorway ({r_a!r}, {r_b!r}) missing from setup prose.",
                check_id="c_fidelity_doorways",
            )

    # Verify initial agent rooms, absent parties, goals, and stakes
    for ag in world.agents:
        init_r = world.initial_agent_rooms[ag]
        if f"{ag.lower()} starts in the {init_r.lower()}" not in setup_low:
            raise RenderCheckError(
                f"Render check (c) failed: initial room '{init_r}' for agent '{ag}' missing from setup prose.",
                check_id="c_fidelity_initial_room",
            )
        goal_txt = world.goals[ag].goal_text.lower()
        if goal_txt not in setup_low:
            raise RenderCheckError(
                f"Render check (c) failed: goal '{world.goals[ag].goal_text}' for agent '{ag}' missing from setup prose.",
                check_id="c_fidelity_goal",
            )

    for abs_p in world.absent_parties:
        if not re.search(rf"\b{re.escape(abs_p.lower())}\b", setup_low):
            raise RenderCheckError(
                f"Render check (c) failed: absent party '{abs_p}' missing from setup prose.",
                check_id="c_fidelity_absent_party",
            )

    for stake in world.stake_edges:
        if stake.description.lower() not in setup_low:
            raise RenderCheckError(
                f"Render check (c) failed: stake edge '{stake.description}' missing from setup prose.",
                check_id="c_fidelity_stake_edge",
            )

    # 2. Verify no unlogged party names or foreign domain entities appear anywhere in the prose
    allowed_parties = set(world.agents) | set(world.absent_parties)
    for party_name in ALL_KNOWN_PARTY_NAMES - allowed_parties:
        if re.search(rf"\b{re.escape(party_name)}\b", prose):
            raise RenderCheckError(
                f"Render check (c) failed: unlogged party name '{party_name}' asserted in prose.",
                check_id="c_fidelity_unlogged_party",
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
                raise RenderCheckError(
                    f"Render check (c) failed: unlogged entity/phrase '{phrase}' from domain '{dom.name}' asserted in prose.",
                    check_id="c_fidelity_unlogged_entity",
                )

    # Check that unlogged side_reasons or unlogged note placement are never asserted when unknowable
    if current_domain is not None:
        logged_reasons = {ev.stated_reason.lower() for ev in world.events if ev.stated_reason}
        for reason_str in current_domain.side_reasons:
            if reason_str.lower() not in logged_reasons and reason_str.lower() in prose_low:
                raise RenderCheckError(
                    f"Render check (c) failed: unlogged motive '{reason_str}' asserted in prose.",
                    check_id="c_fidelity_unlogged_motive",
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
                    raise RenderCheckError(
                        f"Render check (c) failed: unlogged placement of '{current_domain.note_item}' by '{ag}' asserted in prose.",
                        check_id="c_fidelity_unlogged_placer",
                    )

    # 3. Verify every event in `world.events` is stated in exact relative order and no extra event sentences exist
    all_sentences = _split_prose_sentences(prose)
    event_sentences = [s for s in all_sentences if not _is_setup_sentence(s)]

    for sent in event_sentences:
        s_low = sent.lower()
        if "in open view" in s_low and "blocked" in s_low:
            raise RenderCheckError(
                f"Render check (c) failed: contradictory visibility ('in open view' and 'blocked') in sentence: {sent!r}",
                check_id="c_fidelity_contradictory_occlusion",
            )

    matched_sentence_indices: list[int] = []
    search_start = 0
    for ev_idx, ev in enumerate(world.events):
        found_idx: int | None = None
        for s_idx in range(search_start, len(event_sentences)):
            if _sentence_matches_event(event_sentences[s_idx], ev, world.noise_threshold):
                found_idx = s_idx
                break
        if found_idx is None:
            # Check if the event appears out of order earlier in event_sentences
            earlier_idx = next(
                (
                    i
                    for i in range(0, search_start)
                    if i not in matched_sentence_indices
                    and _sentence_matches_event(event_sentences[i], ev, world.noise_threshold)
                ),
                None,
            )
            if earlier_idx is not None:
                raise RenderCheckError(
                    f"Render check (c) failed: event #{ev_idx + 1} ({ev.kind} by {ev.actor} in {ev.room}) is out of relative chronological order.",
                    check_id="c_fidelity_event_order",
                )
            raise RenderCheckError(
                f"Render check (c) failed: missing or altered event #{ev_idx + 1} (step={ev.step}, kind={ev.kind}, actor={ev.actor}, room={ev.room}, target={ev.target_object!r}, from={ev.from_location!r}, to={ev.to_location!r}).",
                check_id="c_fidelity_missing_event",
            )
        matched_sentence_indices.append(found_idx)
        search_start = found_idx + 1

    if len(event_sentences) != len(world.events):
        unmatched = [
            event_sentences[i]
            for i in range(len(event_sentences))
            if i not in set(matched_sentence_indices)
        ]
        raise RenderCheckError(
            f"Render check (c) failed: prose asserts {len(unmatched)} unlogged event sentence(s) not in world.events: {unmatched[:2]}",
            check_id="c_fidelity_unlogged_event",
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
        occ = world.occluders_by_room[r]
        occ_art = indefinite_article(occ)
        room_setup_strs.append(
            f"the {r} contains the {c_pair[0]} and the {c_pair[1]} in the {CONTAINER_BAY_STATION} "
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

    sig_kill = getattr(signal, "SIGKILL", signal.SIGTERM)
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

    def stop(self) -> None:
        """Stop the child `ollama serve` process and record its UTC stop timestamp."""
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

    def to_dict(self) -> dict[str, object]:
        return {
            "call_id": self.call_id,
            "call_index": self.call_index,
            "scenario_id": self.scenario_id,
            "rendering_index": self.rendering_index,
            "focal_agent": self.focal_agent,
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
            if self._halted:
                self._trigger_halt(self._peak_vram_mib)
            require_loopback_ollama_url(url)
            if url.endswith("/api/pull"):
                raise ValueError(
                    f"REFUSED (PULL_FORBIDDEN): Auto-pull is forbidden ({method} {url}); refuse when the pinned writer digest is missing."
                )
            if method.upper() == "POST" and url.endswith("/api/chat"):
                self.check_vram()
            resp = inner(method, url, payload)
            if method.upper() == "POST" and url.endswith("/api/chat") and payload is not None:
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
        pair: tuple[RenderedScenario, RenderedScenario],
        call_records: Sequence[GenerationCallRecord],
    ) -> None:
        """Append a finished scenario pair and its generation call records to the checkpoint atomically."""
        r0, r1 = pair
        world = r0.world or r1.world
        if world is not None:
            check_rendered_prose(r0.prose, world, focal_agent=r0.focal_agent)
            check_rendered_prose(r1.prose, world, focal_agent=r1.focal_agent)

        prev_hash = self._chain_head
        sealed_payload: dict[str, Any] = {
            "scenario_index": scenario_index,
            "scenario_id": r0.scenario_id,
            "rendering_0": {
                "rendering_index": r0.rendering_index,
                "focal_agent": r0.focal_agent,
                "prose": r0.prose,
                "writer_prompt": r0.writer_prompt,
            },
            "rendering_1": {
                "rendering_index": r1.rendering_index,
                "focal_agent": r1.focal_agent,
                "prose": r1.prose,
                "writer_prompt": r1.writer_prompt,
            },
            "call_records": [rec.to_dict() for rec in call_records],
            "prev_hash": prev_hash,
        }
        sealed_entry_sha256 = _sha256_bytes(_canonical_json_line(sealed_payload).encode("utf-8"))

        pub_payload: dict[str, Any] = {
            "scenario_index": scenario_index,
            "focal_agents": [r0.focal_agent, r1.focal_agent],
            "prose_sha256": [
                _sha256_bytes(r0.prose.encode("utf-8")),
                _sha256_bytes(r1.prose.encode("utf-8")),
            ],
            "call_records": [
                {
                    "call_id": rec.call_id,
                    "call_index": rec.call_index,
                    "rendering_index": rec.rendering_index,
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

            r0_dict = sealed_rec["rendering_0"]
            r1_dict = sealed_rec["rendering_1"]
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
        num_predict: int = DEFAULT_WRITER_NUM_PREDICT,
        max_num_ctx: int = DEFAULT_WRITER_MAX_NUM_CTX,
        vram_halt_mib: int = DEFAULT_VRAM_HALT_MIB,
        grant_id: str | None = None,
        http_transport: OllamaHttpTransport | None = None,
        vram_probe: Callable[[], int] | None = None,
        halt_pid: int | None = None,
        pid_killer: Callable[[int], None] = kill_pid_tree,
        generations_path: Path | None = None,
        start_owned_server: bool = False,
        popen_factory: Callable[..., Any] | None = None,
        server_port: int | None = None,
        ollama_models_dir: Path | str | None = None,
    ) -> None:
        if num_predict < 1:
            raise ValueError(f"num_predict must be >= 1, got {num_predict}")
        if max_num_ctx < NUM_CTX_STEP_TOKENS:
            raise ValueError(f"max_num_ctx must be >= {NUM_CTX_STEP_TOKENS}, got {max_num_ctx}")
        self._model_pin = model_pin
        self._num_predict = int(num_predict)
        self._max_num_ctx = int(max_num_ctx)
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
        self._owned_server: OwnedOllamaServer | None = None
        if start_owned_server:
            self._owned_server = start_owned_ollama_server(
                http_transport=raw_transport,
                popen_factory=popen_factory,
                port=server_port,
                ollama_models_dir=ollama_models_dir,
            )
            active_url = self._owned_server.base_url
            resolved_halt_pid = self._owned_server.pid if halt_pid is None else halt_pid
        else:
            active_url = require_loopback_ollama_url(base_url)
            resolved_halt_pid = halt_pid

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
        self._verified_local_digest = False
        self._call_records: list[GenerationCallRecord] = []
        self._unload_records: list[OllamaUnloadRecord] = []
        self._refusals: list[ProseRefusalRecord] = []

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

    def _append_generation_record(self, rec: GenerationCallRecord) -> None:
        self._call_records.append(rec)
        if self._generations_path is not None:
            self._generations_path.parent.mkdir(parents=True, exist_ok=True)
            with self._generations_path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(_canonical_json_line(rec.to_dict()) + "\n")

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
        scenario_id: str,
        rendering_index: int,
        focal_agent: str,
        prompt_text: str,
    ) -> str:
        self._ensure_verified()
        messages = [
            {
                "role": "user",
                "content": prompt_text,
            }
        ]
        prompt_sha = _sha256_bytes(_canonical_json_line(messages).encode("utf-8"))
        prompt_tokens_estimate = estimate_prompt_tokens(messages)
        num_ctx = compute_num_ctx(
            prompt_tokens_estimate,
            self._num_predict,
            margin=NUM_CTX_MARGIN_TOKENS,
            step=NUM_CTX_STEP_TOKENS,
        )
        call_index = len(self._call_records) + 1
        call_id = f"{scenario_id}:r{rendering_index}:call_{call_index:04d}"

        if num_ctx > self._max_num_ctx:
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
                think=self._model_pin.think,
                temperature=0.0,
                done_reason="max_num_ctx_exceeded",
                seconds=0.0,
                vram_peak_mib=self._gpu_guard.peak_vram_mib,
                vram_halt_mib=self._gpu_guard.vram_halt_mib,
                grant_id=self._grant_id,
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
        payload: dict[str, Any] = {
            "model": self.ollama_tag,
            "messages": messages,
            "think": self._model_pin.think,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": self._num_predict,
                "num_ctx": num_ctx,
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
        top5_logprobs: tuple[dict[str, object], ...] = (
            tuple(x for x in raw_logprobs[:5] if isinstance(x, dict))
            if isinstance(raw_logprobs, list)
            else ()
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
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_count,
            thinking_words=thinking_words,
            num_predict=self._num_predict,
            num_ctx=num_ctx,
            max_num_ctx=self._max_num_ctx,
            think=self._model_pin.think,
            temperature=0.0,
            done_reason=done_reason,
            seconds=wall_s,
            vram_peak_mib=self._gpu_guard.peak_vram_mib,
            vram_halt_mib=self._gpu_guard.vram_halt_mib,
            grant_id=self._grant_id,
            top5_logprobs=top5_logprobs,
        )
        self._append_generation_record(rec)

        if done_reason == "length":
            reason_msg = (
                f"Writer call '{call_id}' stopped with done_reason='length' "
                f"(num_predict={self._num_predict}, eval_count={eval_count})."
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

        return content

    def render_scenario_pair(self, world: World) -> tuple[RenderedScenario, RenderedScenario]:
        """Render `world` twice via `qwen3.6:35b`, run all prose checks (a)-(d), and verify key invariance."""
        focal_0 = world.agents[0]
        focal_1 = world.agents[1]

        prompt_0 = format_writer_prompt(world, focal_0, self._prompt_template)
        prompt_1 = format_writer_prompt(world, focal_1, self._prompt_template)

        prose_0 = self._chat_render(
            scenario_id=world.scenario_id,
            rendering_index=0,
            focal_agent=focal_0,
            prompt_text=prompt_0,
        )
        try:
            check_rendered_prose(prose_0, world, focal_agent=focal_0)
        except RenderCheckError as exc:
            self._refusals.append(
                ProseRefusalRecord(
                    scenario_id=world.scenario_id,
                    rendering_index=0,
                    focal_agent=focal_0,
                    check_id=exc.check_id,
                    reason=str(exc),
                )
            )
            raise

        prose_1 = self._chat_render(
            scenario_id=world.scenario_id,
            rendering_index=1,
            focal_agent=focal_1,
            prompt_text=prompt_1,
        )
        try:
            check_rendered_prose(prose_1, world, focal_agent=focal_1)
        except RenderCheckError as exc:
            self._refusals.append(
                ProseRefusalRecord(
                    scenario_id=world.scenario_id,
                    rendering_index=1,
                    focal_agent=focal_1,
                    check_id=exc.check_id,
                    reason=str(exc),
                )
            )
            raise

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
