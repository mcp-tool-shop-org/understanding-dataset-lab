"""Stage 5 spoken-line rewrite pass, prompt builders, pre-splice guards, and keep-rate bookkeeping.

Implements Card U3 (`DESIGN-stage5-REV2-2026-11-02.md` §3, §4, §5, §7, §8):
1. `build_rewrite_messages(ev, world, target_form, *, draft_sentence=None) -> list[dict[str, str]]`:
   Builds the single-user-message `/api/chat` payload for `facts_only` (`draft_sentence=None`) or
   `with_draft` (`draft_sentence` included as `"The current sentence: ..."`), using:
   - the event facts from code (speaker, exact utterance, loudness in dB, room);
   - the target shape's plain-words description (`SAY_1`, `SAY_2`, `SAY_3`, `SAY_4`; `SAY_5` is excluded);
   - one code-built example of that shape rendered by `render_event_sentence` on a fixed, invented,
     non-scene event (`EXAMPLE_EVENT` in `EXAMPLE_WORLD`).
2. `call_rewrite(client, messages) -> RewriteCallResult`:
   Calls Ollama `/api/chat` via the injected `client` transport with `temperature=0`, `think=False`,
   `num_predict=80`, `num_ctx=2048`, and no assistant prefill. Records `model`, `done_reason`,
   `prompt_eval_count`, `eval_count`, and `duration_s`. Converts `done_reason == "length"`, any
   transport exception, or an empty reply into `call_error`.
3. Pre-splice guards (`check_rewrite_guards`), evaluated in strict order before any splice:
   (1) `not_one_sentence`: `_split_prose_sentences(reply.strip())` must return a single sentence;
   (2) `extra_names`: no agent, absent party, room, container, object, occluder, or fixture of `world`
       outside the quoted utterance except `ev.actor` and `ev.room` (case-insensitive word boundaries);
   (3) `verb`: the speech verb (`_ACTION_VERBS_BY_KIND["say"]`) must equal the draft's speech verb or
       be `"said"`;
   (4) `tag_echo`: no shape key or tag token (`SAY_n`, `MOVE_n`, `SEE_n`, or any form id), no legend
       wording, and no bracketed role label (wrapping `_has_tag_echo` from `udlab_generator.render`);
   (5) `utterance_changed`: quoted text must match `ev.utterance` under `_quote_matches_utterance_exactly`.
4. Splice and scene re-check via U2's public wrapper `check_splice_keep` in `shape_templates.py`:
   rolls back byte-identically with `fault_grew`, `neighbour_changed`, `count_changed`, or `wrong_form`.
5. `rewrite_all_say(prose, world, plan, client, *, input_mode, focal_agent, goal_mode, checker_profile="d") -> RewriteResult`:
   Runs at most one attempt per eligible `say` event (where `plan_form != before_form`), with no retry.
   Prompts and replies go only into `RewriteResult.sealed`; `RewriteResult.record` and log lines never
   carry prose, prompts, replies, speaker names, room names, or utterances.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import inspect
import json
import logging
import re
import time
from typing import Any, Literal

from udlab_generator.render import (
     DEFAULT_GOAL_MODE,
    DEFAULT_OLLAMA_BASE_URL,
    PINNED_WRITER_MODEL_TAG,
    ProseFaultRecord,
    SHAPE_PLAN_VARIANT_FORMS,
    SHAPE_PLAN_WILDCARD_BY_KIND,
    STAGE4_LEGEND_V1_LINES,
    TAG_ECHO_TOKENS,
    VALID_CHECKER_PROFILES,
    VALID_GOAL_MODES,
    _ACTION_VERBS_BY_KIND,
    _BRACKET_MARKER_RE,
    _canonical_json_line,
    _extract_quoted_spans,
    _find_actor_subject_verb_matches,
    _has_tag_echo,
    _normalize_utterance_or_quote,
    _quote_matches_utterance_exactly,
    _sha256_bytes,
    _split_prose_sentences,
    _strip_quoted_spans,
    _tag_echo_token_pattern,
    collect_prose_faults,
    event_shape_plan_variant,
    match_rendering_event_forms,
    normalize_prose_line_endings,
)
from udlab_generator.shape_templates import (
    ALL_TEMPLATE_FORMS,
    SAY_FORMS,
    SpliceKeepCheckResult,
    _extract_event_sentences,
    _resolve_plan_by_event_index,
    check_splice_keep,
    render_event_sentence,
    splice_event_sentence_at_index,
)
from udlab_generator.world import (
    AgentGoal,
    Event,
    FactSpec,
    ScenarioConfig,
    SettledTwinSpec,
    StakeEdge,
    UnknowableSpec,
    World,
)

logger = logging.getLogger(__name__)

RewriteInputMode = Literal["facts_only", "with_draft"]
VALID_REWRITE_INPUT_MODES: frozenset[str] = frozenset({"facts_only", "with_draft"})

REWRITE_TEMPERATURE: float = 0.0
REWRITE_THINK: bool = False
REWRITE_NUM_PREDICT: int = 80
REWRITE_NUM_CTX: int = 2048

SAY_REWRITE_TARGET_FORMS: tuple[str, ...] = ("SAY_2", "SAY_3", "SAY_4")

# Plain-words descriptions of SAY_1..SAY_4 from DESIGN-stage5-REV2-2026-11-02.md §3 (lines 109-112):
SAY_SHAPE_DESCRIPTIONS: dict[str, str] = {
    "SAY_1": "standard: actor verb, loudness and room, then the quote",
    "SAY_2": "quote first, attribution after",
    "SAY_3": "loudness fronted before the actor",
    "SAY_4": "the room placed after the quote",
}

# Fixed, invented, non-scene constants for the prompt example (disjoint from all DOMAINS and AGENT_NAME_POOLS):
EXAMPLE_SPEAKER: str = "Soren"
EXAMPLE_FOCAL_AGENT: str = "Talia"
EXAMPLE_ABSENT_PARTY: str = "Vesper"
EXAMPLE_AGENT_NAMES: tuple[str, ...] = (EXAMPLE_SPEAKER, EXAMPLE_FOCAL_AGENT)
EXAMPLE_ROOM: str = "lantern vestibule"
EXAMPLE_OTHER_ROOMS: tuple[str, ...] = ("amber gallery", "basalt rotunda", "cinder atrium")
EXAMPLE_UTTERANCE: str = "The cobalt pendulum rests on the quartz shelf."
EXAMPLE_NOISE_LEVEL: int = 65
EXAMPLE_NOISE_THRESHOLD: int = 55

EXAMPLE_EVENT: Event = Event(
    step=1,
    kind="say",
    actor=EXAMPLE_SPEAKER,
    room=EXAMPLE_ROOM,
    utterance=EXAMPLE_UTTERANCE,
    noise_level=EXAMPLE_NOISE_LEVEL,
)


def _build_invented_example_world() -> World:
    """Construct the fixed, invented, non-scene `World` used solely to render the prompt example via `render_event_sentence`."""
    rooms = (EXAMPLE_ROOM, *EXAMPLE_OTHER_ROOMS)
    doorways = (
        (rooms[0], rooms[1]),
        (rooms[1], rooms[2]),
        (rooms[2], rooms[3]),
    )
    containers_by_room = {
        rooms[0]: ("quartz shelf", "pewter niche"),
        rooms[1]: ("basalt tray", "cedar coffer"),
        rooms[2]: ("amber bin", "slate locker"),
        rooms[3]: ("cobalt case", "linen trough"),
    }
    occluders_by_room = {
        rooms[0]: "alabaster screen",
        rooms[1]: "tapestried partition",
        rooms[2]: "stone pillar",
        rooms[3]: "woven divider",
    }
    fixtures_by_room = {
        rooms[0]: "sundial plaque",
        rooms[1]: "mercury column",
        rooms[2]: "pendulum bracket",
        rooms[3]: "draft indicator",
    }
    goals = {
        EXAMPLE_SPEAKER: AgentGoal(
            agent=EXAMPLE_SPEAKER,
            goal_text=f"{EXAMPLE_SPEAKER} plans to calibrate the sundial plaque",
            causal_preconditions=("F1",),
        ),
        EXAMPLE_FOCAL_AGENT: AgentGoal(
            agent=EXAMPLE_FOCAL_AGENT,
            goal_text=f"{EXAMPLE_FOCAL_AGENT} plans to inspect the mercury column",
            causal_preconditions=("F1",),
        ),
    }
    facts = {
        "F1": FactSpec(
            fact_id="F1",
            variable="loc_cobalt_pendulum",
            value="quartz shelf",
            description="The cobalt pendulum is on the quartz shelf.",
        )
    }
    stake_edges = (
        StakeEdge(
            agent=EXAMPLE_ABSENT_PARTY,
            variable="F1",
            description=(
                f"{EXAMPLE_ABSENT_PARTY}'s sign-off for {EXAMPLE_SPEAKER} depends on the location of the cobalt pendulum."
            ),
        ),
    )
    unknowable = UnknowableSpec(
        subtype="who",
        question_text="Who adjusted the sundial plaque?",
        alternative_a=EXAMPLE_SPEAKER,
        alternative_b=EXAMPLE_FOCAL_AGENT,
        target_agent=EXAMPLE_SPEAKER,
        side_action_desc="adjusted the sundial plaque",
        trace_a=(EXAMPLE_EVENT,),
        trace_b=(EXAMPLE_EVENT,),
    )
    settled = SettledTwinSpec(
        subtype="who",
        question_text="Who spoke in the lantern vestibule?",
        settled_answer=EXAMPLE_SPEAKER,
        target_agent=EXAMPLE_SPEAKER,
        side_action_desc="spoke in the lantern vestibule",
    )
    cfg = ScenarioConfig(
        scenario_id="invented_rewrite_example",
        seed=0,
        p_can_see=True,
        k_knows=True,
        b1_true_belief=True,
        b2_true_belief=True,
        x_notified=False,
        u_is_who=True,
        num_agents=2,
        num_absent=1,
        topology_kind="chain",
    )
    return World(
        scenario_id="invented_rewrite_example",
        seed=0,
        domain_name="invented_domain",
        rooms=rooms,
        doorways=doorways,
        objects=("cobalt pendulum", "brass astrolabe", "silver plumb"),
        containers_by_room=containers_by_room,
        occluders_by_room=occluders_by_room,
        fixtures_by_room=fixtures_by_room,
        noise_threshold=EXAMPLE_NOISE_THRESHOLD,
        agents=EXAMPLE_AGENT_NAMES,
        absent_parties=(EXAMPLE_ABSENT_PARTY,),
        initial_agent_rooms={EXAMPLE_SPEAKER: EXAMPLE_ROOM, EXAMPLE_FOCAL_AGENT: EXAMPLE_ROOM},
        stake_edges=stake_edges,
        goals=goals,
        facts=facts,
        events=(EXAMPLE_EVENT,),
        unknowable_spec=unknowable,
        settled_twin_spec=settled,
        primary_object="cobalt pendulum",
        primary_variable="F1",
        secondary_object="brass astrolabe",
        secondary_variable="F2",
        aux_object="silver plumb",
        aux_variable="F3",
        bluff_event_step=1,
        bluffer_agent=EXAMPLE_SPEAKER,
        bluff_listener=EXAMPLE_FOCAL_AGENT,
        bluff_claimed_location="quartz shelf",
        p_agent=EXAMPLE_SPEAKER,
        k_agent=EXAMPLE_FOCAL_AGENT,
        b1_agent=EXAMPLE_SPEAKER,
        b2_observer=EXAMPLE_FOCAL_AGENT,
        b2_actor=EXAMPLE_SPEAKER,
        d_helper=EXAMPLE_SPEAKER,
        d_target=EXAMPLE_FOCAL_AGENT,
        primary_absent=EXAMPLE_ABSENT_PARTY,
        config=cfg,
    )


EXAMPLE_WORLD: World = _build_invented_example_world()

VALID_REWRITE_ACTIONS: frozenset[str] = frozenset(
    {"rewritten", "rewrite_rolled_back", "kept", "not_spliced"}
)
VALID_REWRITE_CAUSES: frozenset[str] = frozenset(
    {
        "call_error",
        "not_one_sentence",
        "extra_names",
        "verb",
        "tag_echo",
        "utterance_changed",
        "fault_grew",
        "neighbour_changed",
        "count_changed",
        "wrong_form",
    }
)

_SAY_VERBS: tuple[str, ...] = _ACTION_VERBS_BY_KIND["say"]
_SAY_VERB_ALT: str = "|".join(re.escape(v) for v in _SAY_VERBS)
_SAY_VERB_WORD_RE = re.compile(rf"\b(?:{_SAY_VERB_ALT})\b", re.IGNORECASE)

# Extended form-id and legend-wording tokens for the rewrite `tag_echo` guard, wrapping `_has_tag_echo`:
_ALL_FORM_IDS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *ALL_TEMPLATE_FORMS,
            *SHAPE_PLAN_WILDCARD_BY_KIND.values(),
            *(f for forms in SHAPE_PLAN_VARIANT_FORMS.values() for f in forms),
            *(f"{prefix}_{n}" for prefix in ("SAY", "MOVE", "SEE", "ENTER", "LEAVE", "MESSAGE") for n in range(10)),
        )
    )
)
_ANGLE_PLACEHOLDER_RE = re.compile(
    r"<\s*(?:actor|verb|room|utterance|noise_level|target_object|from_location|to_location|occluder|short\s+phrase)\s*>",
    re.IGNORECASE,
)
_LEGEND_PHRASES: tuple[str, ...] = (
    "Shape key for the tag on each event row",
    "The tag names the sentence shape for that event",
    "Never write a tag or this key in the story",
    "any accepted move shape",
    "any accepted see shape",
    "any accepted say shape",
    "the shape in the table",
    *SAY_SHAPE_DESCRIPTIONS.values(),
)
_EXTENDED_TAG_ECHO_RE = re.compile(
    r"(?:"
    + "|".join(
        _tag_echo_token_pattern(tok)
        for tok in dict.fromkeys((*TAG_ECHO_TOKENS, *_ALL_FORM_IDS, *_LEGEND_PHRASES))
    )
    + r")",
    re.IGNORECASE,
)


def render_rewrite_example_sentence(target_form: str) -> str:
    """Render the fixed, invented, non-scene example sentence for `target_form` via U2's `render_event_sentence`."""
    if target_form == "SAY_5":
        raise ValueError("SAY_5 is out of the Stage 5 say pool.")
    if target_form not in SAY_FORMS:
        raise ValueError(f"target_form must be one of {SAY_FORMS}, got {target_form!r}.")
    return render_event_sentence(
        EXAMPLE_EVENT,
        EXAMPLE_WORLD,
        target_form,
        focal_agent=EXAMPLE_FOCAL_AGENT,
    )


def build_rewrite_messages(
    ev: Event,
    world: World,
    target_form: str,
    *,
    draft_sentence: str | None = None,
) -> list[dict[str, str]]:
    """Build the single-user-message `/api/chat` payload for a spoken-line rewrite call.

    - `facts-only`: `draft_sentence=None`. The draft sentence never appears in the message.
    - `with-draft`: `draft_sentence` is added as `"The current sentence: <draft_sentence>"`.
    - Returns `[{"role": "user", "content": ...}]` (one user message, no system message).
    """
    if ev.kind != "say":
        raise ValueError(f"build_rewrite_messages requires a 'say' event, got kind={ev.kind!r}.")
    if target_form == "SAY_5":
        raise ValueError("SAY_5 is out of the Stage 5 pool and cannot be targeted by build_rewrite_messages.")
    if target_form not in SAY_FORMS:
        raise ValueError(f"target_form must be one of {SAY_FORMS}, got {target_form!r}.")
    if world is None:
        raise ValueError("world must not be None.")

    shape_desc = SAY_SHAPE_DESCRIPTIONS[target_form]
    example_sentence = render_rewrite_example_sentence(target_form)
    utterance_text = ev.utterance or ""

    if draft_sentence is None:
        lines = [
            "Write one spoken-line sentence from the facts below in the target sentence shape.",
            "Facts:",
            f"- speaker: {ev.actor}",
            f'- utterance: "{utterance_text}"',
            f"- loudness: {ev.noise_level} dB",
            f"- room: {ev.room}",
            f"Target sentence shape: {shape_desc}",
            "Example of this shape on a different event:",
            example_sentence,
            'Keep the exact quoted utterance, loudness in dB, room, and speaker, name no other person, place, or object, use the verb "said", and answer with that one sentence only.',
        ]
    else:
        clean_draft = draft_sentence.strip()
        lines = [
            "Rewrite the current sentence into the target sentence shape using the facts below.",
            f"The current sentence: {clean_draft}",
            "Facts:",
            f"- speaker: {ev.actor}",
            f'- utterance: "{utterance_text}"',
            f"- loudness: {ev.noise_level} dB",
            f"- room: {ev.room}",
            f"Target sentence shape: {shape_desc}",
            "Example of this shape on a different event:",
            example_sentence,
            'Keep the exact quoted utterance, loudness in dB, room, and speaker, name no other person, place, or object, keep the current sentence\'s speech verb or use "said", and answer with that one sentence only.',
        ]

    return [{"role": "user", "content": "\n".join(lines)}]


@dataclass(frozen=True)
class RewriteCallResult:
    """Result of a single `call_rewrite` invocation.

    Note: `to_dict()` / `to_record_dict()` intentionally omit `reply` so serialized records never carry model text.
    """

    ok: bool
    cause: str | None
    error_kind: str | None
    model: str
    done_reason: str
    prompt_eval_count: int
    eval_count: int
    duration_s: float
    reply: str
    reply_sha256: str
    prompt_sha256: str
    temperature: float = REWRITE_TEMPERATURE
    think: bool = REWRITE_THINK
    num_predict: int = REWRITE_NUM_PREDICT
    num_ctx: int = REWRITE_NUM_CTX

    @property
    def duration(self) -> float:
        return self.duration_s

    @property
    def seconds(self) -> float:
        return self.duration_s

    def to_record_dict(self) -> dict[str, object]:
        """Serialize only non-sealed call metadata (counts, hashes, timing, done_reason, options)."""
        return {
            "ok": self.ok,
            "cause": self.cause,
            "error_kind": self.error_kind,
            "model": self.model,
            "done_reason": self.done_reason,
            "prompt_eval_count": self.prompt_eval_count,
            "eval_count": self.eval_count,
            "duration_s": round(self.duration_s, 6),
            "duration": round(self.duration_s, 6),
            "seconds": round(self.duration_s, 6),
            "reply_sha256": self.reply_sha256,
            "prompt_sha256": self.prompt_sha256,
            "temperature": self.temperature,
            "think": self.think,
            "num_predict": self.num_predict,
            "num_ctx": self.num_ctx,
        }

    def to_dict(self) -> dict[str, object]:
        return self.to_record_dict()


def _invoke_rewrite_client(
    client: Any,
    chat_url: str,
    payload: dict[str, Any],
) -> Mapping[str, Any]:
    """Invoke the injected transport `client` (`OllamaHttpTransport` or adapter) with `payload`."""
    if callable(client):
        try:
            sig = inspect.signature(client)
            params = [
                p
                for p in sig.parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
            if not has_varargs and len(params) == 1:
                resp = client(payload)
            else:
                resp = client("POST", chat_url, payload)
        except (ValueError, TypeError):
            resp = client("POST", chat_url, payload)
    elif hasattr(client, "_transport") and callable(client._transport):
        base = str(getattr(client, "base_url", DEFAULT_OLLAMA_BASE_URL)).rstrip("/")
        resp = client._transport("POST", f"{base}/api/chat", payload)
    elif hasattr(client, "chat") and callable(client.chat):
        resp = client.chat(payload)
    elif hasattr(client, "call_api_chat") and callable(client.call_api_chat):
        resp = client.call_api_chat("POST", chat_url, payload)
    else:
        raise TypeError(f"Unsupported rewrite client type: {type(client)!r}")

    if not isinstance(resp, Mapping):
        raise RuntimeError(f"Rewrite client returned non-mapping response: {type(resp)!r}")
    return resp


def call_rewrite(
    client: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    model: str = PINNED_WRITER_MODEL_TAG,
) -> RewriteCallResult:
    """Execute one spoken-line rewrite call via the injected `client`.

    Request options:
    - `temperature`: `0` (T=0)
    - `think`: `False`
    - `num_predict`: `80`
    - `num_ctx`: `2048`
    - no prefill (no trailing assistant message)

    A `length` done_reason, a transport error, or an empty reply becomes `cause="call_error"`.
    """
    if not messages:
        raise ValueError("messages must not be empty.")
    normalized_messages: list[dict[str, str]] = []
    for idx, m in enumerate(messages):
        role = str(m.get("role", ""))
        content = str(m.get("content", ""))
        if idx == len(messages) - 1 and role == "assistant":
            raise ValueError("Prefill (trailing assistant message) is forbidden in call_rewrite.")
        if role != "user":
            raise ValueError(f"call_rewrite allows only a single user message (no system or prefill), got role={role!r}.")
        normalized_messages.append({"role": role, "content": content})
    if len(normalized_messages) != 1:
        raise ValueError(f"call_rewrite expects exactly 1 user message, got {len(normalized_messages)}.")

    prompt_sha256 = _sha256_bytes(_canonical_json_line(normalized_messages).encode("utf-8"))
    payload: dict[str, Any] = {
        "model": model,
        "messages": normalized_messages,
        "think": REWRITE_THINK,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": REWRITE_NUM_PREDICT,
            "num_ctx": REWRITE_NUM_CTX,
        },
    }
    chat_url = f"{base_url.rstrip('/')}/api/chat"

    t0 = time.perf_counter()
    try:
        resp = _invoke_rewrite_client(client, chat_url, payload)
    except Exception:
        wall_s = max(0.0, time.perf_counter() - t0)
        return RewriteCallResult(
            ok=False,
            cause="call_error",
            error_kind="transport_error",
            model=model,
            done_reason="transport_error",
            prompt_eval_count=0,
            eval_count=0,
            duration_s=wall_s,
            reply="",
            reply_sha256="",
            prompt_sha256=prompt_sha256,
        )
    wall_s = max(0.0, time.perf_counter() - t0)

    resp_model = str(resp.get("model") or model)
    done_reason = str(resp.get("done_reason") or "stop").strip()

    raw_prompt_eval = resp.get("prompt_eval_count")
    prompt_eval_count = (
        int(raw_prompt_eval)
        if isinstance(raw_prompt_eval, int) and not isinstance(raw_prompt_eval, bool) and raw_prompt_eval >= 0
        else 0
    )

    msg = resp.get("message")
    if isinstance(msg, Mapping):
        raw_content = msg.get("content")
    else:
        raw_content = resp.get("response", resp.get("content"))
    reply = normalize_prose_line_endings(str(raw_content or "")).strip()
    reply_sha256 = _sha256_bytes(reply.encode("utf-8")) if reply else ""

    raw_eval = resp.get("eval_count")
    eval_count = (
        int(raw_eval)
        if isinstance(raw_eval, int) and not isinstance(raw_eval, bool) and raw_eval >= 0
        else (len(reply.split()) if reply else 0)
    )

    raw_duration = resp.get("duration_s", resp.get("seconds"))
    if isinstance(raw_duration, (int, float)) and not isinstance(raw_duration, bool) and raw_duration >= 0:
        duration_s = float(raw_duration)
    elif (
        isinstance(resp.get("total_duration"), int)
        and not isinstance(resp.get("total_duration"), bool)
        and int(resp["total_duration"]) > 0
    ):
        duration_s = float(resp["total_duration"]) / 1_000_000_000.0
    else:
        duration_s = wall_s

    if done_reason == "length":
        return RewriteCallResult(
            ok=False,
            cause="call_error",
            error_kind="length",
            model=resp_model,
            done_reason=done_reason,
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_count,
            duration_s=duration_s,
            reply=reply,
            reply_sha256=reply_sha256,
            prompt_sha256=prompt_sha256,
        )

    if not reply:
        return RewriteCallResult(
            ok=False,
            cause="call_error",
            error_kind="empty_reply",
            model=resp_model,
            done_reason=done_reason,
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_count,
            duration_s=duration_s,
            reply="",
            reply_sha256="",
            prompt_sha256=prompt_sha256,
        )

    return RewriteCallResult(
        ok=True,
        cause=None,
        error_kind=None,
        model=resp_model,
        done_reason=done_reason,
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
        duration_s=duration_s,
        reply=reply,
        reply_sha256=reply_sha256,
        prompt_sha256=prompt_sha256,
    )


def _world_forbidden_extra_names(world: World, ev: Event) -> tuple[str, ...]:
    """Return all world names (agents, absent parties, rooms, containers, objects, occluders, fixtures)
    except the speaker (`ev.actor`) and the event's room (`ev.room`)."""
    allowed = {ev.actor.strip().lower(), ev.room.strip().lower()}
    candidates: list[str] = []
    candidates.extend(world.agents)
    candidates.extend(world.absent_parties)
    candidates.extend(world.rooms)
    for pair in world.containers_by_room.values():
        candidates.extend(pair)
    candidates.extend(world.objects)
    candidates.extend(world.occluders_by_room.values())
    candidates.extend(world.fixtures_by_room.values())

    forbidden: list[str] = []
    seen_low: set[str] = set()
    for name in candidates:
        clean = name.strip()
        low = clean.lower()
        if not low or low in allowed or low in seen_low:
            continue
        seen_low.add(low)
        forbidden.append(low)
    return tuple(forbidden)


def _extract_draft_say_verb(draft_sentence: str | None, ev: Event, world: World) -> str | None:
    """Extract the speech verb from `draft_sentence` (outside quoted spans), or return `None`."""
    if not draft_sentence:
        return None
    unquoted_low = _strip_quoted_spans(draft_sentence.lower())
    frame_no_thresh = re.sub(r"\([^)]*threshold[^)]*\)", "", unquoted_low)
    all_parties = tuple(world.agents) + tuple(world.absent_parties)
    subj_matches = _find_actor_subject_verb_matches(
        frame_no_thresh,
        ev.actor,
        _SAY_VERBS,
        all_parties,
        ev=ev,
        world=world,
        checker_profile="d",
    )
    if subj_matches:
        return subj_matches[0][1].group(0).lower()
    m = _SAY_VERB_WORD_RE.search(frame_no_thresh)
    return m.group(0).lower() if m is not None else None


def has_rewrite_tag_echo(reply: str) -> bool:
    """Return True if `reply` contains any shape key, tag token (`SAY_n`, `MOVE_n`, `SEE_n`, or form id),
    legend wording, or bracketed role label.

    Wraps `_has_tag_echo` (`src/udlab_generator/render.py:116`) without modifying `render.py`.
    """
    if _has_tag_echo(reply):
        return True
    if _EXTENDED_TAG_ECHO_RE.search(reply) is not None:
        return True
    unquoted = _strip_quoted_spans(reply)
    if _BRACKET_MARKER_RE.search(unquoted) is not None or _ANGLE_PLACEHOLDER_RE.search(unquoted) is not None:
        return True
    if _BRACKET_MARKER_RE.search(reply) is not None or _ANGLE_PLACEHOLDER_RE.search(reply) is not None:
        return True
    for line in STAGE4_LEGEND_V1_LINES:
        clean_line = line.strip()
        if len(clean_line) >= 16 and clean_line.lower() in reply.lower():
            return True
    return False


def check_rewrite_guards(
    reply: str,
    ev: Event,
    world: World,
    *,
    draft_sentence: str | None = None,
) -> str | None:
    """Apply the five pre-splice rewrite guards in strict order.

    Returns `None` if all guards pass, or the failure cause:
    1. `"not_one_sentence"`
    2. `"extra_names"`
    3. `"verb"`
    4. `"tag_echo"`
    5. `"utterance_changed"`
    """
    cleaned = normalize_prose_line_endings(reply).strip()

    # 1. not_one_sentence: after stripping whitespace, _split_prose_sentences(reply) must return exactly one sentence.
    if not cleaned or "\n\n" in cleaned:
        return "not_one_sentence"
    split_sents = _split_prose_sentences(cleaned)
    if len(split_sents) != 1:
        return "not_one_sentence"

    # 2. extra_names: no agent, absent party, room, container, object, occluder or fixture of the world
    #    except the speaker and the event's room (outside the quoted utterance), case-insensitive on word boundaries.
    unquoted_low = _strip_quoted_spans(cleaned.lower())
    for forbidden_name in _world_forbidden_extra_names(world, ev):
        if re.search(rf"\b{re.escape(forbidden_name)}\b", unquoted_low, flags=re.IGNORECASE) is not None:
            return "extra_names"

    # 3. verb: the speech verb must equal the draft's verb, or be `said`.
    #    The verb list is `_ACTION_VERBS_BY_KIND["say"]` (render.py:815). A reply with none of those verbs is `verb`.
    frame_no_thresh = re.sub(r"\([^)]*threshold[^)]*\)", "", unquoted_low)
    reply_verbs = [m.group(0).lower() for m in _SAY_VERB_WORD_RE.finditer(frame_no_thresh)]
    if not reply_verbs:
        return "verb"
    draft_verb = _extract_draft_say_verb(draft_sentence, ev, world)
    allowed_verbs = {"said"}
    if draft_verb is not None:
        allowed_verbs.add(draft_verb)
    if any(v not in allowed_verbs for v in reply_verbs):
        return "verb"

    # 4. tag_echo: no shape key or tag token, no legend wording, and no bracketed role label (before utterance check).
    if has_rewrite_tag_echo(cleaned):
        return "tag_echo"

    # 5. utterance_changed: quoted text must equal the event's utterance (normalized the way the say matcher does).
    quoted_spans = _extract_quoted_spans(cleaned.lower())
    if not quoted_spans or not _quote_matches_utterance_exactly(quoted_spans, ev.utterance or ""):
        return "utterance_changed"

    return None


@dataclass(frozen=True)
class RewriteEventRecord:
    """Per-event record produced by `rewrite_all_say` (carries no prose, prompt, reply, speaker, room, or utterance)."""

    event_index: int
    kind: str
    variant: str
    plan_form: str | None
    before_form: str | None
    after_form: str | None
    sentence_index: int | None
    action: str
    cause: str | None = None
    rule_ids: tuple[str, ...] = ()
    call: RewriteCallResult | None = None

    def __post_init__(self) -> None:
        if self.action not in VALID_REWRITE_ACTIONS:
            raise ValueError(
                f"RewriteEventRecord.action must be one of {sorted(VALID_REWRITE_ACTIONS)}, got {self.action!r}."
            )
        if self.cause is not None and self.cause not in VALID_REWRITE_CAUSES:
            raise ValueError(
                f"RewriteEventRecord.cause must be one of {sorted(VALID_REWRITE_CAUSES)} or None, got {self.cause!r}."
            )

    @property
    def plan(self) -> str | None:
        return self.plan_form

    @property
    def b_form(self) -> str | None:
        return self.before_form

    @property
    def final_form(self) -> str | None:
        return self.after_form

    @property
    def s_form(self) -> str | None:
        return self.after_form

    @property
    def rollback_cause(self) -> str | None:
        return self.cause

    @property
    def call_record(self) -> dict[str, object] | None:
        return self.call.to_record_dict() if self.call is not None else None

    def to_dict(self) -> dict[str, object]:
        call_dict = self.call.to_record_dict() if self.call is not None else None
        return {
            "event_index": self.event_index,
            "kind": self.kind,
            "variant": self.variant,
            "plan": self.plan_form,
            "plan_form": self.plan_form,
            "before_form": self.before_form,
            "b_form": self.before_form,
            "after_form": self.after_form,
            "final_form": self.after_form,
            "s_form": self.after_form,
            "sentence_index": self.sentence_index,
            "action": self.action,
            "cause": self.cause,
            "rollback_cause": self.cause,
            "rule_ids": list(self.rule_ids),
            "call": call_dict,
            "call_record": call_dict,
        }

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def get(self, key: str, default: object = None) -> object:
        return self.to_dict().get(key, default)


def compute_say_rewrite_bookkeeping(
    events: Sequence[RewriteEventRecord | Mapping[str, Any]],
) -> dict[str, object]:
    """Compute keep-rate and rollback-cause bookkeeping across `RewriteEventRecord` items.

    - Attempts and kept per target shape for `SAY_2`, `SAY_3`, and `SAY_4` only (with rate `kept / attempts`);
    - `SAY_1` reported as `(draft_matched + rewrite_kept) / assigned`, with no `rate` key;
    - Rollbacks counted by cause.
    """
    attempts_by_shape: Counter[str] = Counter({f: 0 for f in SAY_REWRITE_TARGET_FORMS})
    kept_by_shape: Counter[str] = Counter({f: 0 for f in SAY_REWRITE_TARGET_FORMS})
    rollbacks_by_cause: Counter[str] = Counter()

    say1_assigned = 0
    say1_draft_matched = 0
    say1_rewrite_attempted = 0
    say1_rewrite_kept = 0

    for item in events:
        if isinstance(item, RewriteEventRecord):
            kind = item.kind
            plan_form = item.plan_form
            before_form = item.before_form
            after_form = item.after_form
            action = item.action
            cause = item.cause
        else:
            kind = str(item.get("kind") or "")
            plan_form = item.get("plan_form", item.get("plan"))
            before_form = item.get("before_form", item.get("b_form"))
            after_form = item.get("after_form", item.get("s_form", item.get("final_form")))
            action = str(item.get("action") or "")
            cause = item.get("cause", item.get("rollback_cause"))

        if kind != "say":
            continue

        if action == "rewrite_rolled_back" and cause is not None:
            rollbacks_by_cause[str(cause)] += 1

        if plan_form in SAY_REWRITE_TARGET_FORMS:
            if action in ("rewritten", "rewrite_rolled_back"):
                attempts_by_shape[plan_form] += 1
                if action == "rewritten" and after_form == plan_form:
                    kept_by_shape[plan_form] += 1
        elif plan_form == "SAY_1":
            say1_assigned += 1
            if before_form == "SAY_1" and action == "kept":
                say1_draft_matched += 1
            elif action in ("rewritten", "rewrite_rolled_back"):
                say1_rewrite_attempted += 1
                if action == "rewritten" and after_form == "SAY_1":
                    say1_rewrite_kept += 1

    per_shape: dict[str, dict[str, object]] = {}
    for form_id in SAY_REWRITE_TARGET_FORMS:
        att = int(attempts_by_shape[form_id])
        kp = int(kept_by_shape[form_id])
        per_shape[form_id] = {
            "attempts": att,
            "kept": kp,
            "rate": round(kp / att, 4) if att > 0 else None,
        }

    pooled_att = int(sum(attempts_by_shape.values()))
    pooled_kept = int(sum(kept_by_shape.values()))
    pooled = {
        "attempts": pooled_att,
        "kept": pooled_kept,
        "rate": round(pooled_kept / pooled_att, 4) if pooled_att > 0 else None,
    }

    say1_realized = say1_draft_matched + say1_rewrite_kept
    say1_report: dict[str, object] = {
        "assigned": say1_assigned,
        "draft_matched": say1_draft_matched,
        "rewrite_attempted": say1_rewrite_attempted,
        "rewrite_kept": say1_rewrite_kept,
        "realized": say1_realized,
        "expression": f"({say1_draft_matched} + {say1_rewrite_kept}) / {say1_assigned}",
    }

    causes_dict = {k: int(v) for k, v in sorted(rollbacks_by_cause.items())}
    return {
        "per_shape": per_shape,
        "pooled": pooled,
        "say_1": say1_report,
        "say1_realization": say1_report,
        "rollbacks_by_cause": causes_dict,
        "rollback_causes": causes_dict,
    }


@dataclass(frozen=True)
class RewriteResult:
    """Outcome of `rewrite_all_say`.

    - `prose`: final scene prose (with kept rewrites spliced and failed rewrites restored byte-identically).
    - `sealed`: list of `{"event_index": int, "messages": list[dict[str, str]], "reply": str}` for each call.
    - `record`: non-sealed telemetry record containing only counts, token counts, causes, form ids, indices,
      and `reply_sha256` (never prompts, replies, prose, speaker names, room names, or utterances).
    """

    prose: str
    events: tuple[RewriteEventRecord, ...]
    sealed: list[dict[str, object]]
    record: dict[str, object]
    bookkeeping: dict[str, object]
    b_faults: tuple[ProseFaultRecord, ...]
    s_faults: tuple[ProseFaultRecord, ...]
    b_passed: bool
    s_passed: bool

    @property
    def event_records(self) -> tuple[RewriteEventRecord, ...]:
        return self.events

    @property
    def b_fault_rule_ids(self) -> tuple[str, ...]:
        return tuple(f.rule_id for f in self.b_faults)

    @property
    def s_fault_rule_ids(self) -> tuple[str, ...]:
        return tuple(f.rule_id for f in self.s_faults)

    def to_dict(self) -> dict[str, object]:
        return dict(self.record)

    def __getitem__(self, key: str) -> object:
        return self.record[key]

    def get(self, key: str, default: object = None) -> object:
        return self.record.get(key, default)


def rewrite_all_say(
    prose: str,
    world: World,
    plan: Any,
    client: Any,
    *,
    input_mode: RewriteInputMode = "facts_only",
    focal_agent: str,
    goal_mode: str = DEFAULT_GOAL_MODE,
    checker_profile: str = "d",
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    model: str = PINNED_WRITER_MODEL_TAG,
) -> RewriteResult:
    """Run the Stage 5 spoken-line rewrite pass across all eligible `say` events in `world`.

    - For each `say` event in event order whose planned form differs from its current realized form,
      performs exactly one rewrite call (no retry), runs the 5 pre-splice guards in order, and if all
      pass, splices at `sentence_index` and runs U2's `check_splice_keep`.
    - Prompts and replies are stored only in `RewriteResult.sealed`.
    - `RewriteResult.record` and log lines carry only counts, token counts, causes, form ids, indices,
      and `reply_sha256`.
    """
    if focal_agent not in world.agents:
        raise ValueError(f"focal_agent '{focal_agent}' must be one of {world.agents}")
    if input_mode not in VALID_REWRITE_INPUT_MODES:
        raise ValueError(
            f"input_mode must be one of {sorted(VALID_REWRITE_INPUT_MODES)}, got {input_mode!r}"
        )
    if goal_mode not in VALID_GOAL_MODES:
        raise ValueError(f"goal_mode must be one of {sorted(VALID_GOAL_MODES)}, got {goal_mode!r}")
    if checker_profile not in VALID_CHECKER_PROFILES:
        raise ValueError(
            f"checker_profile must be one of {sorted(VALID_CHECKER_PROFILES)}, got {checker_profile!r}"
        )

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
    per_event_before_form: dict[int, str | None] = {}
    per_event_call: dict[int, RewriteCallResult | None] = {}
    sealed_entries: list[dict[str, object]] = []

    for ev_pos, ev in enumerate(world.events):
        ev_idx = ev_pos + 1
        cur_row = current_rows[ev_pos]
        cur_form = (
            str(cur_row["realized_form_id"])
            if cur_row.get("realized_form_id") is not None
            else None
        )
        per_event_before_form[ev_idx] = cur_form
        planned_form = plan_by_idx.get(ev_idx)
        per_event_plan_form[ev_idx] = planned_form

        if ev.kind != "say":
            per_event_action[ev_idx] = "kept"
            per_event_cause[ev_idx] = None
            per_event_rule_ids[ev_idx] = ()
            per_event_call[ev_idx] = None
            continue

        if planned_form == "SAY_5":
            raise ValueError("SAY_5 is out of the Stage 5 say pool.")
        if planned_form is not None and planned_form not in SAY_FORMS:
            raise ValueError(f"Unsupported say plan form {planned_form!r} at event_index={ev_idx}.")

        s_idx_obj = cur_row.get("sentence_index")
        if s_idx_obj is None:
            per_event_action[ev_idx] = "not_spliced"
            per_event_cause[ev_idx] = None
            row_rule = str(cur_row["rule_id"]) if cur_row.get("rule_id") is not None else None
            per_event_rule_ids[ev_idx] = (row_rule,) if row_rule else ()
            per_event_call[ev_idx] = None
            continue

        if planned_form is None or cur_form == planned_form:
            per_event_action[ev_idx] = "kept"
            per_event_cause[ev_idx] = None
            per_event_rule_ids[ev_idx] = ()
            per_event_call[ev_idx] = None
            continue

        s_idx = int(s_idx_obj)  # type: ignore[arg-type]
        event_sents = _extract_event_sentences(current_prose, world, goal_mode)
        draft_sentence = event_sents[s_idx]

        messages = build_rewrite_messages(
            ev,
            world,
            planned_form,
            draft_sentence=draft_sentence if input_mode == "with_draft" else None,
        )

        # Exactly one attempt per eligible event; no retry.
        call_res = call_rewrite(
            client,
            messages,
            base_url=base_url,
            model=model,
        )
        per_event_call[ev_idx] = call_res
        sealed_entries.append(
            {
                "event_index": ev_idx,
                "messages": messages,
                "reply": call_res.reply,
            }
        )

        if not call_res.ok:
            per_event_action[ev_idx] = "rewrite_rolled_back"
            per_event_cause[ev_idx] = "call_error"
            per_event_rule_ids[ev_idx] = ()
            logger.info(
                "rewrite_say event_index=%d sentence_index=%d before_form=%s target_form=%s "
                "after_form=%s action=rewrite_rolled_back cause=call_error done_reason=%s "
                "prompt_eval_count=%d eval_count=%d reply_sha256=%s",
                ev_idx,
                s_idx,
                cur_form,
                planned_form,
                cur_form,
                call_res.done_reason,
                call_res.prompt_eval_count,
                call_res.eval_count,
                call_res.reply_sha256,
            )
            continue

        guard_cause = check_rewrite_guards(
            call_res.reply,
            ev,
            world,
            draft_sentence=draft_sentence,
        )
        if guard_cause is not None:
            per_event_action[ev_idx] = "rewrite_rolled_back"
            per_event_cause[ev_idx] = guard_cause
            per_event_rule_ids[ev_idx] = ()
            logger.info(
                "rewrite_say event_index=%d sentence_index=%d before_form=%s target_form=%s "
                "after_form=%s action=rewrite_rolled_back cause=%s done_reason=%s "
                "prompt_eval_count=%d eval_count=%d reply_sha256=%s",
                ev_idx,
                s_idx,
                cur_form,
                planned_form,
                cur_form,
                guard_cause,
                call_res.done_reason,
                call_res.prompt_eval_count,
                call_res.eval_count,
                call_res.reply_sha256,
            )
            continue

        cleaned_reply = normalize_prose_line_endings(call_res.reply).strip()
        candidate_prose = splice_event_sentence_at_index(current_prose, s_idx, cleaned_reply)
        keep_res: SpliceKeepCheckResult = check_splice_keep(
            current_prose,
            candidate_prose,
            world,
            ev_pos=ev_pos,
            target_form=planned_form,
            current_faults=current_faults,
            current_rows=current_rows,
            current_sent_count=current_sent_count,
            focal_agent=focal_agent,
            goal_mode=goal_mode,
            checker_profile=checker_profile,
        )

        if keep_res.kept:
            current_prose = candidate_prose
            current_faults = keep_res.cand_faults
            current_rows = keep_res.cand_rows
            current_sent_count = keep_res.cand_sent_count
            per_event_action[ev_idx] = "rewritten"
            per_event_cause[ev_idx] = None
            per_event_rule_ids[ev_idx] = ()
            after_form_log = planned_form
        else:
            # Failure restores `current_prose` byte-identically.
            per_event_action[ev_idx] = "rewrite_rolled_back"
            per_event_cause[ev_idx] = keep_res.cause
            per_event_rule_ids[ev_idx] = keep_res.rule_ids
            after_form_log = cur_form

        logger.info(
            "rewrite_say event_index=%d sentence_index=%d before_form=%s target_form=%s "
            "after_form=%s action=%s cause=%s done_reason=%s "
            "prompt_eval_count=%d eval_count=%d reply_sha256=%s",
            ev_idx,
            s_idx,
            cur_form,
            planned_form,
            after_form_log,
            per_event_action[ev_idx],
            per_event_cause[ev_idx],
            call_res.done_reason,
            call_res.prompt_eval_count,
            call_res.eval_count,
            call_res.reply_sha256,
        )

    records: list[RewriteEventRecord] = []
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
        records.append(
            RewriteEventRecord(
                event_index=ev_idx,
                kind=ev.kind,
                variant=event_shape_plan_variant(ev, world),
                plan_form=per_event_plan_form[ev_idx],
                before_form=per_event_before_form[ev_idx],
                after_form=fin_form,
                sentence_index=fin_s_idx,
                action=per_event_action[ev_idx],
                cause=per_event_cause[ev_idx],
                rule_ids=per_event_rule_ids[ev_idx],
                call=per_event_call[ev_idx],
            )
        )

    bookkeeping = compute_say_rewrite_bookkeeping(records)
    b_passed = len(b_faults) == 0
    s_passed = len(current_faults) == 0
    record_dict: dict[str, object] = {
        "scenario_id": world.scenario_id,
        "input_mode": input_mode,
        "checker_profile": checker_profile,
        "goal_mode": goal_mode,
        "b_passed": b_passed,
        "s_passed": s_passed,
        "b_fault_rule_ids": [f.rule_id for f in b_faults],
        "s_fault_rule_ids": [f.rule_id for f in current_faults],
        "calls_made": len(sealed_entries),
        "events": [r.to_dict() for r in records],
        "bookkeeping": bookkeeping,
    }

    return RewriteResult(
        prose=current_prose,
        events=tuple(records),
        sealed=sealed_entries,
        record=record_dict,
        bookkeeping=bookkeeping,
        b_faults=b_faults,
        s_faults=current_faults,
        b_passed=b_passed,
        s_passed=s_passed,
    )
