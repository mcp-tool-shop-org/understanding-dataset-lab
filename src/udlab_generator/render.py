"""Deterministic third-person perspective renderer, writer prompt, and banned-adverb check."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
from typing import Callable

from udlab_generator.keys import ScenarioKeys, compute_scenario_keys
from udlab_generator.world import (
    CONTAINER_BAY_STATION,
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

WRITER_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "writer_qwen3_6_35b.txt"


class RenderCheckError(ValueError):
    """Raised when rendered scenario prose contains a banned mental or emotional adverb."""


class ForbiddenModelCallError(RuntimeError):
    """Raised if any code attempts a real `/api/chat` or live model call in CPU-only mode."""


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
    """Load the `qwen3.6:35b` writer prompt text file and verify banned-adverb coverage."""
    target = path or WRITER_PROMPT_PATH
    text = target.read_text(encoding="utf-8")
    missing = [w for w in BANNED_ADVERBS if w not in text.lower()]
    if missing:
        raise ValueError(f"Writer prompt template is missing banned adverbs: {missing}")
    return text


def format_writer_prompt(world: World, focal_agent: str, template: str | None = None) -> str:
    """Format the `qwen3.6:35b` writer prompt for `world` and `focal_agent`."""
    raw_template = template if template is not None else load_writer_prompt_template()
    world_setup = {
        "scenario_id": world.scenario_id,
        "domain": world.domain_name,
        "rooms": list(world.rooms),
        "doorways": [list(edge) for edge in world.doorways],
        "containers_by_room": {r: list(c) for r, c in world.containers_by_room.items()},
        "occluders_by_room": dict(world.occluders_by_room),
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
                "causal_preconditions": list(g.causal_preconditions),
            }
            for ag, g in world.goals.items()
        },
    }
    event_log = [
        {
            "step": ev.step,
            "kind": ev.kind,
            "actor": ev.actor,
            "room": ev.room,
            "target_object": ev.target_object,
            "from_location": ev.from_location,
            "to_location": ev.to_location,
            "utterance": ev.utterance,
            "recipient": ev.recipient,
            "occluded": ev.occluded,
            "occluder": ev.occluder,
            "noise_level": ev.noise_level,
            "fact_id": ev.fact_id,
            "is_bluff": ev.is_bluff,
            "stated_reason": ev.stated_reason,
        }
        for ev in world.events
    ]
    return (
        raw_template.replace("{focal_agent}", focal_agent)
        .replace("{world_setup_json}", json.dumps(world_setup, indent=2, sort_keys=True))
        .replace("{event_log_json}", json.dumps(event_log, indent=2, sort_keys=True))
    )


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


def check_rendered_prose(prose: str) -> tuple[str, ...]:
    """Validate that `prose` contains no banned mental/emotional adverbs.

    Raises `RenderCheckError` if any banned word from `BANNED_ADVERBS` is found.
    Returns an empty tuple on pass.
    """
    hits = tuple(dict.fromkeys(m.group(1).lower() for m in _BANNED_PATTERN.finditer(prose)))
    if hits:
        raise RenderCheckError(
            f"Render check failed: prose contains banned unlogged mental/emotional adverb(s): {hits}"
        )
    return ()


def _render_event_sentence(event: Event, focal_agent: str, noise_threshold: int) -> str:
    """Render a single `Event` in third-person prose focalized on `focal_agent`."""
    prefix = (
        f"From {focal_agent}'s station at step {event.step}"
        if event.actor == focal_agent
        else f"At step {event.step}"
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
        fact_tag = f" [establishing {event.fact_id}]" if event.fact_id and event.fact_id.startswith("F") and event.fact_id in ("F1", "F2", "F3", "F4") else ""
        return (
            f"{prefix}, in the {event.room}, {event.actor} moved the {event.target_object} "
            f"from the {event.from_location} to the {event.to_location}{occ_clause}{fact_tag}."
        )

    if event.kind == "see":
        if event.occluded and event.occluder:
            return (
                f"{prefix}, in the {event.room}, from the {FAR_SIDE_ALCOVE_STATION} behind the {event.occluder}, "
                f"{event.actor} looked toward the {event.target_object}, "
                f"but the {event.occluder} blocked the line of sight."
            )
        fact_tag = f" [establishing {event.fact_id}]" if event.fact_id and event.fact_id in ("F1", "F2", "F3", "F4") else ""
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
            f"{loc_clause} in open view{fact_tag}."
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
    goal_strs = [
        f"{ world.goals[ag].goal_text } (preconditions: {', '.join(world.goals[ag].causal_preconditions)})"
        for ag in ordered_agents
    ]
    stake_strs = [s.description for s in world.stake_edges]

    setup_para = (
        f"Following {focal_agent} through the {world.domain_name.replace('_', ' ')}: "
        f"{'; '.join(room_setup_strs)}. "
        f"Doorways link the rooms as follows: {', '.join(doorway_strs)}, and the doorway hearing threshold is {world.noise_threshold} dB "
        f"(a say at or above {world.noise_threshold} dB carries one doorway; quieter speech stays in its room). "
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
    check_rendered_prose(prose)
    return prose


class MockWriter:
    """CPU-only mock writer for `qwen3.6:35b` that refuses any real `/api/chat` call."""

    def __init__(
        self,
        custom_renderer: Callable[[World, str], str] | None = None,
    ) -> None:
        self._custom_renderer = custom_renderer
        self._prompt_template = load_writer_prompt_template()

    def call_api_chat(self, *args: object, **kwargs: object) -> None:
        """Hard guard: real `/api/chat` calls are forbidden in the CPU generator build."""
        raise ForbiddenModelCallError(
            "Real /api/chat calls and GPU model loads are forbidden in Card qh (CPU mock writer only)."
        )

    def render_scenario_pair(self, world: World) -> tuple[RenderedScenario, RenderedScenario]:
        """Render `world` twice, focalized on two distinct agents, over the complete log."""
        focal_0 = world.agents[0]
        focal_1 = world.agents[1]

        render_fn = self._custom_renderer or render_world_template
        prose_0 = render_fn(world, focal_0)
        prose_1 = render_fn(world, focal_1)

        # Both renderings must pass the banned-adverb render check and cover the complete event log.
        check_rendered_prose(prose_0)
        check_rendered_prose(prose_1)
        for idx_label, prose_text in (("rendering 0", prose_0), ("rendering 1", prose_1)):
            for ev in world.events:
                if f"step {ev.step}" not in prose_text:
                    raise RenderCheckError(
                        f"Render check failed: incomplete event log in {idx_label} (missing step {ev.step})."
                    )

        prompt_0 = format_writer_prompt(world, focal_0, self._prompt_template)
        prompt_1 = format_writer_prompt(world, focal_1, self._prompt_template)

        # Verify perspective-flip invariance by reconstructing the event trace from each writer prompt
        # and computing `ScenarioKeys` independently on both focalized representations.
        def _world_from_prompt(prompt_str: str) -> World:
            marker = "Complete Event Log:\n"
            pos = prompt_str.find(marker)
            if pos < 0:
                return world
            raw_after = prompt_str[pos + len(marker) :]
            end_pos = raw_after.find("\n\nOUTPUT FORMAT:")
            json_block = raw_after[:end_pos] if end_pos >= 0 else raw_after
            parsed_list = json.loads(json_block)
            step_by_num = {int(item["step"]): item for item in parsed_list}
            if set(step_by_num.keys()) != {ev.step for ev in world.events}:
                raise RenderCheckError("Writer prompt event log is incomplete.")
            reconstructed_events = tuple(
                replace(
                    ev,
                    kind=step_by_num[ev.step]["kind"],
                    actor=step_by_num[ev.step]["actor"],
                    room=step_by_num[ev.step]["room"],
                    target_object=step_by_num[ev.step]["target_object"],
                    from_location=step_by_num[ev.step]["from_location"],
                    to_location=step_by_num[ev.step]["to_location"],
                    occluded=bool(step_by_num[ev.step]["occluded"]),
                    occluder=step_by_num[ev.step]["occluder"],
                    noise_level=int(step_by_num[ev.step]["noise_level"]),
                    is_bluff=bool(step_by_num[ev.step]["is_bluff"]),
                    stated_reason=step_by_num[ev.step]["stated_reason"],
                )
                for ev in world.events
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

        keys_0 = compute_scenario_keys(_world_from_prompt(prompt_0))
        keys_1 = compute_scenario_keys(_world_from_prompt(prompt_1))
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
