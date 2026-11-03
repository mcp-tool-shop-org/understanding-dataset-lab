"""Stage 5 Card U3 tests: spoken-line rewrite pass, prompt builders, pre-splice guards, sealed payloads, and keep-rate bookkeeping."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import inspect
import json
import logging
from pathlib import Path
import subprocess
from typing import Any

import pytest

from udlab_generator.render import (
    PINNED_WRITER_MODEL_TAG,
    collect_prose_faults,
    match_rendering_event_forms,
    render_world_template,
    world_from_scenario_seed,
)
from udlab_generator.rewrite import (
    EXAMPLE_ABSENT_PARTY,
    EXAMPLE_AGENT_NAMES,
    EXAMPLE_EVENT,
    EXAMPLE_FOCAL_AGENT,
    EXAMPLE_OTHER_ROOMS,
    EXAMPLE_ROOM,
    EXAMPLE_SPEAKER,
    EXAMPLE_UTTERANCE,
    EXAMPLE_WORLD,
    REWRITE_NUM_CTX,
    REWRITE_NUM_PREDICT,
    REWRITE_TEMPERATURE,
    REWRITE_THINK,
    SAY_REWRITE_TARGET_FORMS,
    SAY_SHAPE_DESCRIPTIONS,
    RewriteCallResult,
    RewriteEventRecord,
    build_rewrite_messages,
    call_rewrite,
    check_rewrite_guards,
    compute_say_rewrite_bookkeeping,
    has_rewrite_tag_echo,
    render_rewrite_example_sentence,
    rewrite_all_say,
)
from udlab_generator.shape_templates import (
    SAY_FORMS,
    SpliceKeepCheckResult,
    check_splice_keep,
    render_event_sentence,
    splice_event_sentence_at_index,
    swap_all_move_see,
    swap_all_say,
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
    generate_world,
    make_balanced_configs,
)

STAGE5_SELF_TEST_BASE_SEED = 2026110202
FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "stage5"
    / "stage5_rewrite_fixtures.json"
)


def _load_fixture_doc() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class FakeRewriteClient:
    """Deterministic fake Ollama `/api/chat` transport (zero network, zero model load)."""

    def __init__(
        self,
        responses: Sequence[dict[str, Any] | Exception | str],
    ) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, method: str, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "payload": json.loads(json.dumps(payload)),
            }
        )
        if not self._responses:
            raise AssertionError("FakeRewriteClient received more calls than configured responses.")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            return {
                "model": payload.get("model", PINNED_WRITER_MODEL_TAG),
                "done_reason": "stop",
                "prompt_eval_count": 140,
                "eval_count": 22,
                "total_duration": 125_000_000,
                "message": {"role": "assistant", "content": item},
            }
        return dict(item)


def _make_invented_fixture_world(events: tuple[Event, ...] | None = None) -> World:
    """Build a synthetic world using the invented fixture names from `stage5_rewrite_fixtures.json`."""
    fx = _load_fixture_doc()["fixture_scene"]
    speaker = str(fx["speaker"])
    focal = str(fx["focal_agent"])
    other_agent = str(fx["other_agent"])
    absent = str(fx["absent_party"])
    room0 = str(fx["room"])
    other_rooms = tuple(str(r) for r in fx["other_rooms"])
    rooms = (room0, *other_rooms)
    doorways = (
        (rooms[0], rooms[1]),
        (rooms[1], rooms[2]),
        (rooms[2], rooms[3]),
    )
    default_say = Event(
        step=1,
        kind="say",
        actor=speaker,
        room=room0,
        utterance=str(fx["utterance"]),
        noise_level=int(fx["noise_level"]),
    )
    resolved_events = events if events is not None else (default_say,)

    containers_by_room = {
        rooms[0]: ("cedar cabinet", "iron locker"),
        rooms[1]: ("brass tray", "oak coffer"),
        rooms[2]: ("slate drawer", "pewter chest"),
        rooms[3]: ("granite bin", "linen rack"),
    }
    occluders_by_room = {
        rooms[0]: "velvet curtain",
        rooms[1]: "carved screen",
        rooms[2]: "stone column",
        rooms[3]: "glass partition",
    }
    fixtures_by_room = {
        rooms[0]: "wall chronometer",
        rooms[1]: "pressure gauge",
        rooms[2]: "index plaque",
        rooms[3]: "signal bell",
    }
    goals = {
        speaker: AgentGoal(
            agent=speaker,
            goal_text=f"{speaker} plans to verify the wall chronometer",
            causal_preconditions=("F1",),
        ),
        focal: AgentGoal(
            agent=focal,
            goal_text=f"{focal} plans to inspect the pressure gauge",
            causal_preconditions=("F1",),
        ),
        other_agent: AgentGoal(
            agent=other_agent,
            goal_text=f"{other_agent} plans to check the index plaque",
            causal_preconditions=("F1",),
        ),
    }
    facts = {
        "F1": FactSpec(
            fact_id="F1",
            variable="location:bronze ledger",
            value="cedar cabinet",
            description="The bronze ledger is inside the cedar cabinet.",
        )
    }
    stake_edges = (
        StakeEdge(
            agent=absent,
            variable="location:bronze ledger",
            description=(
                f"{absent} is off-site and depends on the location of the bronze ledger for the evening dispatch."
            ),
        ),
    )
    unknowable = UnknowableSpec(
        subtype="who",
        question_text="Who checked the wall chronometer?",
        alternative_a=speaker,
        alternative_b=focal,
        target_agent=speaker,
        side_action_desc="checked the wall chronometer",
        trace_a=resolved_events,
        trace_b=resolved_events,
    )
    settled = SettledTwinSpec(
        subtype="who",
        question_text=f"Who spoke in the {room0}?",
        settled_answer=speaker,
        target_agent=speaker,
        side_action_desc=f"spoke in the {room0}",
    )
    cfg = ScenarioConfig(
        scenario_id=str(fx["scenario_id"]),
        seed=42,
        p_can_see=True,
        k_knows=True,
        b1_true_belief=True,
        b2_true_belief=True,
        x_notified=False,
        u_is_who=True,
        num_agents=3,
        num_absent=1,
        topology_kind="chain",
    )
    return World(
        scenario_id=str(fx["scenario_id"]),
        seed=42,
        domain_name="invented_fixture_domain",
        rooms=rooms,
        doorways=doorways,
        objects=("bronze ledger", "silver compass", "ivory seal"),
        containers_by_room=containers_by_room,
        occluders_by_room=occluders_by_room,
        fixtures_by_room=fixtures_by_room,
        noise_threshold=int(fx["noise_threshold"]),
        agents=(speaker, focal, other_agent),
        absent_parties=(absent,),
        initial_agent_rooms={speaker: room0, focal: room0, other_agent: room0},
        stake_edges=stake_edges,
        goals=goals,
        facts=facts,
        events=resolved_events,
        unknowable_spec=unknowable,
        settled_twin_spec=settled,
        primary_object="bronze ledger",
        primary_variable="location:bronze ledger",
        secondary_object="silver compass",
        secondary_variable="location:silver compass",
        aux_object="ivory seal",
        aux_variable="location:ivory seal",
        bluff_event_step=1,
        bluffer_agent=speaker,
        bluff_listener=focal,
        bluff_claimed_location="cedar cabinet",
        p_agent=speaker,
        k_agent=focal,
        b1_agent=speaker,
        b2_observer=focal,
        b2_actor=speaker,
        d_helper=speaker,
        d_target=focal,
        primary_absent=absent,
        config=cfg,
    )


def test_case_1_two_sentence_reply_not_one_sentence_byte_identical() -> None:
    """Case 1: A two-sentence reply gives `not_one_sentence`, and the scene is byte-identical."""
    doc = _load_fixture_doc()
    w = _make_invented_fixture_world()
    focal = w.agents[1]
    prose_b = render_world_template(w, focal)
    two_sent_reply = str(doc["guard_cases"]["case_1_two_sentences"])

    assert check_rewrite_guards(two_sent_reply, w.events[0], w) == "not_one_sentence"

    client = FakeRewriteClient([two_sent_reply])
    res = rewrite_all_say(
        prose_b,
        w,
        plan={1: "SAY_2"},
        client=client,
        input_mode="facts_only",
        focal_agent=focal,
    )
    assert res.prose.encode("utf-8") == prose_b.encode("utf-8")
    assert len(client.calls) == 1
    ev_rec = res.events[0]
    assert ev_rec.action == "rewrite_rolled_back"
    assert ev_rec.cause == "not_one_sentence"
    assert ev_rec.before_form == "SAY_3"
    assert ev_rec.after_form == "SAY_3"


def test_case_2_quote_first_inner_punctuation_split_not_one_sentence() -> None:
    """Case 2: A quote-first reply whose inner punctuation makes the splitter cut it gives `not_one_sentence`."""
    doc = _load_fixture_doc()
    w = _make_invented_fixture_world()
    focal = w.agents[1]
    prose_b = render_world_template(w, focal)
    split_quote_reply = str(doc["guard_cases"]["case_2_quote_first_inner_split"])
    period_split_reply = str(doc["guard_cases"]["case_2_quote_first_period_split"])

    assert check_rewrite_guards(split_quote_reply, w.events[0], w) == "not_one_sentence"
    assert check_rewrite_guards(period_split_reply, w.events[0], w) == "not_one_sentence"

    client = FakeRewriteClient([split_quote_reply])
    res = rewrite_all_say(
        prose_b,
        w,
        plan={1: "SAY_2"},
        client=client,
        input_mode="facts_only",
        focal_agent=focal,
    )
    assert res.prose.encode("utf-8") == prose_b.encode("utf-8")
    assert res.events[0].action == "rewrite_rolled_back"
    assert res.events[0].cause == "not_one_sentence"
    assert res.events[0].after_form == "SAY_3"


def test_case_3_extra_names_outside_vs_inside_quote() -> None:
    """Case 3: A reply naming another agent (or world entity) outside the quote gives `extra_names`. The same name inside the quote is allowed."""
    doc = _load_fixture_doc()
    gc = doc["guard_cases"]
    w = _make_invented_fixture_world()
    focal = w.agents[1]
    prose_b = render_world_template(w, focal)

    # Another agent ("Lira"), another room ("copper gallery"), or an object ("silver compass") outside the quote -> extra_names
    for bad_reply in (
        str(gc["case_3_extra_agent_outside_quote"]),
        str(gc["case_3_extra_room_outside_quote"]),
        str(gc["case_3_extra_object_outside_quote"]),
        # Case-insensitive check ("lira" in lowercase outside the quote)
        '"The bronze ledger is inside the cedar cabinet," Mira said near lira at 62 dB in the archive vestibule.',
    ):
        assert check_rewrite_guards(bad_reply, w.events[0], w) == "extra_names"

    client_bad = FakeRewriteClient([str(gc["case_3_extra_agent_outside_quote"])])
    res_bad = rewrite_all_say(
        prose_b,
        w,
        plan={1: "SAY_2"},
        client=client_bad,
        input_mode="facts_only",
        focal_agent=focal,
    )
    assert res_bad.prose == prose_b
    assert res_bad.events[0].action == "rewrite_rolled_back"
    assert res_bad.events[0].cause == "extra_names"

    # Now place the other agent's name ("Lira") INSIDE the quoted utterance: allowed and kept!
    ev_quote_with_name = Event(
        step=1,
        kind="say",
        actor="Mira",
        room="archive vestibule",
        utterance="Lira left the bronze ledger inside the cedar cabinet.",
        noise_level=62,
    )
    w_inside = _make_invented_fixture_world((ev_quote_with_name,))
    prose_inside_b = render_world_template(w_inside, focal)
    reply_name_inside_quote = (
        '"Lira left the bronze ledger inside the cedar cabinet," Mira said at 62 dB in the archive vestibule.'
    )
    assert check_rewrite_guards(reply_name_inside_quote, ev_quote_with_name, w_inside) is None

    client_ok = FakeRewriteClient([reply_name_inside_quote])
    res_ok = rewrite_all_say(
        prose_inside_b,
        w_inside,
        plan={1: "SAY_2"},
        client=client_ok,
        input_mode="facts_only",
        focal_agent=focal,
    )
    assert res_ok.events[0].action == "rewritten"
    assert res_ok.events[0].cause is None
    assert res_ok.events[0].after_form == "SAY_2"


def test_case_4_verb_guard_claimed_told_rejected_said_kept() -> None:
    """Case 4: A reply using `claimed` or `told` when the draft said `said` gives `verb`. Using `said` is kept."""
    doc = _load_fixture_doc()
    fx = doc["fixture_scene"]
    gc = doc["guard_cases"]
    w = _make_invented_fixture_world()
    focal = w.agents[1]
    prose_b = render_world_template(w, focal)
    draft_said = str(fx["draft_say_3"])

    # Draft used `said`: `claimed` and `told` are rejected with `verb`
    assert (
        check_rewrite_guards(
            str(gc["case_4_wrong_verb_claimed"]),
            w.events[0],
            w,
            draft_sentence=draft_said,
        )
        == "verb"
    )
    assert (
        check_rewrite_guards(
            str(gc["case_4_wrong_verb_told"]),
            w.events[0],
            w,
            draft_sentence=draft_said,
        )
        == "verb"
    )
    # Reply with none of `_ACTION_VERBS_BY_KIND["say"]` (e.g., "whispered") is also rejected with `verb`
    assert (
        check_rewrite_guards(
            '"The bronze ledger is inside the cedar cabinet," Mira whispered at 62 dB in the archive vestibule.',
            w.events[0],
            w,
            draft_sentence=draft_said,
        )
        == "verb"
    )

    client_claimed = FakeRewriteClient([str(gc["case_4_wrong_verb_claimed"])])
    res_claimed = rewrite_all_say(
        prose_b,
        w,
        plan={1: "SAY_2"},
        client=client_claimed,
        input_mode="with_draft",
        focal_agent=focal,
    )
    assert res_claimed.prose == prose_b
    assert res_claimed.events[0].action == "rewrite_rolled_back"
    assert res_claimed.events[0].cause == "verb"

    # Using `said` is kept
    valid_say_2 = str(fx["valid_say_2"])
    client_said = FakeRewriteClient([valid_say_2])
    res_said = rewrite_all_say(
        prose_b,
        w,
        plan={1: "SAY_2"},
        client=client_said,
        input_mode="with_draft",
        focal_agent=focal,
    )
    assert res_said.events[0].action == "rewritten"
    assert res_said.events[0].cause is None
    assert res_said.events[0].after_form == "SAY_2"

    # And if the draft itself used `claimed`, then `claimed` (matching the draft's verb) or `said` passes the guard
    assert (
        check_rewrite_guards(
            str(gc["case_4_reply_claimed"]),
            w.events[0],
            w,
            draft_sentence=str(gc["case_4_draft_claimed"]),
        )
        is None
    )


def test_case_5_changed_utterance_utterance_changed() -> None:
    """Case 5: A changed utterance gives `utterance_changed`."""
    doc = _load_fixture_doc()
    w = _make_invented_fixture_world()
    focal = w.agents[1]
    prose_b = render_world_template(w, focal)
    changed_utt_reply = str(doc["guard_cases"]["case_5_changed_utterance"])

    assert check_rewrite_guards(changed_utt_reply, w.events[0], w) == "utterance_changed"

    client = FakeRewriteClient([changed_utt_reply])
    res = rewrite_all_say(
        prose_b,
        w,
        plan={1: "SAY_2"},
        client=client,
        input_mode="facts_only",
        focal_agent=focal,
    )
    assert res.prose.encode("utf-8") == prose_b.encode("utf-8")
    assert res.events[0].action == "rewrite_rolled_back"
    assert res.events[0].cause == "utterance_changed"
    assert res.events[0].after_form == "SAY_3"


def test_case_6_passes_guards_adds_fault_fault_grew() -> None:
    """Case 6: A reply that passes the 5 pre-splice guards but adds a new scene fault gives `fault_grew`."""
    doc = _load_fixture_doc()
    w = _make_invented_fixture_world()
    focal = w.agents[1]
    prose_b = render_world_template(w, focal)
    fault_adding_reply = str(doc["guard_cases"]["case_6_adds_fault_pronoun_subject"])

    # Verify it passes all 5 pre-splice guards
    assert check_rewrite_guards(fault_adding_reply, w.events[0], w) is None

    # Splicing it adds a scene fault (`defect_4_actor_subject` / `c_fidelity_missing_event`) -> rolled back with `fault_grew`
    client = FakeRewriteClient([fault_adding_reply])
    res = rewrite_all_say(
        prose_b,
        w,
        plan={1: "SAY_1"},
        client=client,
        input_mode="facts_only",
        focal_agent=focal,
    )
    assert res.b_passed is True
    assert res.s_passed is True
    assert res.prose.encode("utf-8") == prose_b.encode("utf-8")
    ev_rec = res.events[0]
    assert ev_rec.action == "rewrite_rolled_back"
    assert ev_rec.cause == "fault_grew"
    assert len(ev_rec.rule_ids) > 0
    assert ev_rec.after_form == "SAY_3"


def test_case_7_moves_neighbour_index_neighbour_changed() -> None:
    """Case 7: A reply that passes the guards and adds no fault, but moves a neighbour's index, gives `neighbour_changed`."""
    doc = _load_fixture_doc()
    fx = doc["fixture_scene"]
    gc = doc["guard_cases"]

    ev_1 = Event(
        step=1,
        kind="say",
        actor=str(fx["speaker"]),
        room=str(fx["room"]),
        utterance=str(fx["utterance"]),
        noise_level=int(fx["noise_level"]),
        from_location="intake",
        to_location=None,
    )
    ev_2 = Event(
        step=2,
        kind="say",
        actor=str(fx["speaker"]),
        room=str(fx["room"]),
        utterance=str(fx["utterance"]),
        noise_level=int(fx["noise_level"]),
        from_location=None,
        to_location=None,
    )
    w = _make_invented_fixture_world((ev_1, ev_2))
    focal = w.agents[1]
    setup_para = render_world_template(w, focal).split("\n\n", 1)[0]
    s_0 = str(gc["case_7_s0_only_matches_ev2"])
    s_1 = str(gc["case_7_s1_matches_both_ev1_and_ev2"])
    prose_b = f"{setup_para}\n\n{s_0} {s_1}"

    # Before rewrite: Event 1 has sentence_index=1 (SAY_3) and Event 2 has sentence_index=0 (SAY_3).
    rows_before = match_rendering_event_forms(prose_b, w, focal_agent=focal, checker_profile="d")
    assert [r["sentence_index"] for r in rows_before] == [1, 0]

    # Event 2 (at sentence_index=0) is rewritten to SAY_2 (`valid_say_2`), which passes all 5 guards
    # and adds no new fault, but causes Event 1 to match index 0 and Event 2 to match index 1.
    valid_say_2 = str(fx["valid_say_2"])
    assert check_rewrite_guards(valid_say_2, ev_2, w, draft_sentence=s_0) is None
    cand_prose = splice_event_sentence_at_index(prose_b, 0, valid_say_2)
    rows_after = match_rendering_event_forms(cand_prose, w, focal_agent=focal, checker_profile="d")
    assert [r["sentence_index"] for r in rows_after] == [0, 1]

    client = FakeRewriteClient([valid_say_2])
    res = rewrite_all_say(
        prose_b,
        w,
        plan={1: "SAY_3", 2: "SAY_2"},
        client=client,
        input_mode="facts_only",
        focal_agent=focal,
    )
    assert len(client.calls) == 1
    assert res.prose.encode("utf-8") == prose_b.encode("utf-8")
    assert res.events[0].action == "kept"
    assert res.events[0].sentence_index == 1
    assert res.events[1].action == "rewrite_rolled_back"
    assert res.events[1].cause == "neighbour_changed"
    assert res.events[1].sentence_index == 0


def test_case_8_realizes_different_say_form_wrong_form() -> None:
    """Case 8: A reply that passes the guards and adds no fault, but realizes a different SAY form (`SAY_1` when `SAY_2` was planned), gives `wrong_form`."""
    doc = _load_fixture_doc()
    w = _make_invented_fixture_world()
    focal = w.agents[1]
    prose_b = render_world_template(w, focal)
    say_1_reply = str(doc["guard_cases"]["case_8_wrong_form_say_1_when_say_2_planned"])

    assert check_rewrite_guards(say_1_reply, w.events[0], w) is None

    client = FakeRewriteClient([say_1_reply])
    res = rewrite_all_say(
        prose_b,
        w,
        plan={1: "SAY_2"},
        client=client,
        input_mode="facts_only",
        focal_agent=focal,
    )
    assert res.prose.encode("utf-8") == prose_b.encode("utf-8")
    ev_rec = res.events[0]
    assert ev_rec.action == "rewrite_rolled_back"
    assert ev_rec.cause == "wrong_form"
    assert ev_rec.before_form == "SAY_3"
    assert ev_rec.after_form == "SAY_3"


def test_case_9_length_transport_error_empty_reply_call_error() -> None:
    """Case 9: A `length` done_reason, a transport error, or an empty reply gives `call_error`, with no splice."""
    doc = _load_fixture_doc()
    fx = doc["fixture_scene"]
    w = _make_invented_fixture_world()
    focal = w.agents[1]
    prose_b = render_world_template(w, focal)

    # 1. done_reason == "length"
    length_resp = {
        "model": PINNED_WRITER_MODEL_TAG,
        "done_reason": "length",
        "prompt_eval_count": 120,
        "eval_count": 80,
        "total_duration": 250_000_000,
        "message": {"role": "assistant", "content": str(fx["valid_say_2"])},
    }
    # 2. transport error (RuntimeError)
    transport_exc = RuntimeError("simulated connection refused")
    # 3. empty reply
    empty_resp = {
        "model": PINNED_WRITER_MODEL_TAG,
        "done_reason": "stop",
        "prompt_eval_count": 120,
        "eval_count": 0,
        "total_duration": 50_000_000,
        "message": {"role": "assistant", "content": "   \n  "},
    }

    for subcase_resp, expected_err_kind in (
        (length_resp, "length"),
        (transport_exc, "transport_error"),
        (empty_resp, "empty_reply"),
    ):
        client = FakeRewriteClient([subcase_resp])
        res = rewrite_all_say(
            prose_b,
            w,
            plan={1: "SAY_2"},
            client=client,
            input_mode="facts_only",
            focal_agent=focal,
        )
        assert res.prose.encode("utf-8") == prose_b.encode("utf-8")
        assert len(client.calls) == 1
        sent_payload = client.calls[0]["payload"]
        assert sent_payload["think"] is REWRITE_THINK is False
        assert sent_payload["options"]["temperature"] == REWRITE_TEMPERATURE == 0
        assert sent_payload["options"]["num_predict"] == REWRITE_NUM_PREDICT == 80
        assert sent_payload["options"]["num_ctx"] == REWRITE_NUM_CTX == 2048
        assert len(sent_payload["messages"]) == 1
        assert sent_payload["messages"][0]["role"] == "user"

        ev_rec = res.events[0]
        assert ev_rec.action == "rewrite_rolled_back"
        assert ev_rec.cause == "call_error"
        assert ev_rec.call is not None
        assert ev_rec.call.ok is False
        assert ev_rec.call.cause == "call_error"
        assert ev_rec.call.error_kind == expected_err_kind


def test_case_9a_shape_key_tag_token_bracket_label_tag_echo_before_utterance() -> None:
    """Case 9a: A reply containing a shape key (`SAY_2`), a tag token (`MOVE_3`), legend wording, or a bracketed role label gives `tag_echo`, evaluated before `utterance_changed`."""
    doc = _load_fixture_doc()
    gc = doc["guard_cases"]
    w = _make_invented_fixture_world()
    focal = w.agents[1]
    prose_b = render_world_template(w, focal)

    # Notice each fixture in 9a ALSO alters the utterance or lacks exact quote match,
    # proving `tag_echo` (Guard 4) fires BEFORE `utterance_changed` (Guard 5).
    for tagged_reply in (
        str(gc["case_9a_shape_key"]),
        str(gc["case_9a_tag_token"]),
        str(gc["case_9a_bracket_label"]),
        'Shape key for the tag on each event row: "The bronze ledger is inside the iron locker," Mira said at 62 dB in the archive vestibule.',
    ):
        assert has_rewrite_tag_echo(tagged_reply) is True
        assert check_rewrite_guards(tagged_reply, w.events[0], w) == "tag_echo"

    client = FakeRewriteClient([str(gc["case_9a_shape_key"])])
    res = rewrite_all_say(
        prose_b,
        w,
        plan={1: "SAY_2"},
        client=client,
        input_mode="facts_only",
        focal_agent=focal,
    )
    assert res.prose == prose_b
    assert res.events[0].action == "rewrite_rolled_back"
    assert res.events[0].cause == "tag_echo"


def test_case_9b_sealed_prompt_grep_absent_from_record_and_logs_present_in_sealed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Case 9b: The fixture's utterance, speaker, and room are absent from the serialized record and all captured log output, and present only in `sealed`."""
    doc = _load_fixture_doc()
    fx = doc["fixture_scene"]
    w = _make_invented_fixture_world()
    focal = w.agents[1]
    prose_b = render_world_template(w, focal)
    valid_say_2 = str(fx["valid_say_2"])

    client = FakeRewriteClient([valid_say_2])
    with caplog.at_level(logging.DEBUG, logger="udlab_generator"):
        res = rewrite_all_say(
            prose_b,
            w,
            plan={1: "SAY_2"},
            client=client,
            input_mode="with_draft",
            focal_agent=focal,
        )

    assert res.events[0].action == "rewritten"
    assert res.events[0].after_form == "SAY_2"

    serialized_record = json.dumps(res.record, sort_keys=True)
    serialized_event_dicts = json.dumps([e.to_dict() for e in res.events], sort_keys=True)
    captured_logs = "\n".join(caplog.messages) + "\n" + caplog.text

    speaker = str(fx["speaker"])
    room = str(fx["room"])
    utterance = str(fx["utterance"])

    for secret_needle in (speaker, room, utterance, valid_say_2):
        assert secret_needle.lower() not in serialized_record.lower(), (
            f"Leaked {secret_needle!r} into RewriteResult.record"
        )
        assert secret_needle.lower() not in serialized_event_dicts.lower(), (
            f"Leaked {secret_needle!r} into RewriteEventRecord.to_dict()"
        )
        assert secret_needle.lower() not in captured_logs.lower(), (
            f"Leaked {secret_needle!r} into log output"
        )

    # Verify they ARE present in `RewriteResult.sealed`
    assert len(res.sealed) == 1
    sealed_entry = res.sealed[0]
    assert sealed_entry["event_index"] == 1
    assert sealed_entry["reply"] == valid_say_2
    sealed_json = json.dumps(res.sealed)
    assert speaker in sealed_json
    assert room in sealed_json
    assert utterance in sealed_json


def test_case_9c_example_provenance_disjoint_from_270_scenarios() -> None:
    """Case 9c: The example's names, room, and utterance match no scenario in the 270 generated self-test scenarios (`make_balanced_configs(270, base_seed=2026110202)`)."""
    configs = make_balanced_configs(270, base_seed=STAGE5_SELF_TEST_BASE_SEED)
    worlds = [generate_world(cfg) for cfg in configs]
    assert len(worlds) == 270

    example_names_low = {
        n.strip().lower()
        for n in (*EXAMPLE_AGENT_NAMES, EXAMPLE_ABSENT_PARTY, EXAMPLE_SPEAKER, EXAMPLE_FOCAL_AGENT)
    }
    example_rooms_low = {r.strip().lower() for r in (EXAMPLE_ROOM, *EXAMPLE_OTHER_ROOMS)}
    example_utterance_low = EXAMPLE_UTTERANCE.strip().lower()

    for w in worlds:
        scenario_names_low = {n.strip().lower() for n in (*w.agents, *w.absent_parties)}
        scenario_rooms_low = {r.strip().lower() for r in w.rooms}
        scenario_utterances_low = {
            (ev.utterance or "").strip().lower()
            for ev in w.events
            if ev.kind == "say" and ev.utterance
        }

        assert example_names_low.isdisjoint(scenario_names_low), (
            f"Overlap between example names and scenario {w.scenario_id}: {example_names_low & scenario_names_low}"
        )
        assert example_rooms_low.isdisjoint(scenario_rooms_low), (
            f"Overlap between example rooms and scenario {w.scenario_id}: {example_rooms_low & scenario_rooms_low}"
        )
        assert example_utterance_low not in scenario_utterances_low, (
            f"Example utterance matched scenario {w.scenario_id}"
        )

    # Also verify `render_rewrite_example_sentence` uses U2's `render_event_sentence` on `EXAMPLE_EVENT` for SAY_1..SAY_4
    for form_id in SAY_FORMS:
        rendered_ex = render_rewrite_example_sentence(form_id)
        assert rendered_ex == render_event_sentence(
            EXAMPLE_EVENT,
            EXAMPLE_WORLD,
            form_id,
            focal_agent=EXAMPLE_FOCAL_AGENT,
        )
    with pytest.raises(ValueError, match="SAY_5"):
        render_rewrite_example_sentence("SAY_5")


def test_case_10_facts_only_vs_with_draft_messages_substring() -> None:
    """Case 10: `facts-only` messages never contain the draft sentence; `with-draft` messages do (as 'The current sentence')."""
    doc = _load_fixture_doc()
    fx = doc["fixture_scene"]
    w = _make_invented_fixture_world()
    ev = w.events[0]
    draft_sent = str(fx["draft_say_3"])

    for target_form in SAY_FORMS:
        msgs_facts = build_rewrite_messages(ev, w, target_form, draft_sentence=None)
        assert len(msgs_facts) == 1
        assert msgs_facts[0]["role"] == "user"
        content_facts = msgs_facts[0]["content"]
        assert draft_sent not in content_facts
        assert "The current sentence" not in content_facts
        assert str(fx["speaker"]) in content_facts
        assert str(fx["utterance"]) in content_facts
        assert f"{fx['noise_level']} dB" in content_facts
        assert str(fx["room"]) in content_facts
        assert SAY_SHAPE_DESCRIPTIONS[target_form] in content_facts
        assert render_rewrite_example_sentence(target_form) in content_facts

        msgs_draft = build_rewrite_messages(ev, w, target_form, draft_sentence=draft_sent)
        assert len(msgs_draft) == 1
        assert msgs_draft[0]["role"] == "user"
        content_draft = msgs_draft[0]["content"]
        assert draft_sent in content_draft
        assert f"The current sentence: {draft_sent}" in content_draft
        assert SAY_SHAPE_DESCRIPTIONS[target_form] in content_draft
        assert render_rewrite_example_sentence(target_form) in content_draft


def test_case_11_no_retry_one_call_per_eligible_event() -> None:
    """Case 11: No retry: exactly one fake call per eligible event, counted even when calls or guards fail."""
    doc = _load_fixture_doc()
    fx = doc["fixture_scene"]
    gc = doc["guard_cases"]

    ev_1 = Event(
        step=1,
        kind="say",
        actor=str(fx["speaker"]),
        room=str(fx["room"]),
        utterance=str(fx["utterance"]),
        noise_level=int(fx["noise_level"]),
    )
    ev_2 = Event(
        step=2,
        kind="say",
        actor=str(fx["speaker"]),
        room=str(fx["room"]),
        utterance="The silver compass is on the brass tray.",
        noise_level=64,
    )
    ev_3 = Event(
        step=3,
        kind="say",
        actor=str(fx["speaker"]),
        room=str(fx["room"]),
        utterance="The ivory seal is inside the iron locker.",
        noise_level=60,
    )
    w = _make_invented_fixture_world((ev_1, ev_2, ev_3))
    focal = w.agents[1]
    prose_b = render_world_template(w, focal)

    # Event 1 fails with `not_one_sentence`, Event 2 fails with `call_error`, Event 3 succeeds (`SAY_4`).
    # If there were any retry, FakeRewriteClient would either consume the wrong response or raise AssertionError.
    valid_ev3_say4 = (
        'Mira said at 60 dB, "The ivory seal is inside the iron locker," in the archive vestibule.'
    )
    client = FakeRewriteClient(
        [
            str(gc["case_1_two_sentences"]),
            RuntimeError("timeout on event 2"),
            valid_ev3_say4,
        ]
    )
    res = rewrite_all_say(
        prose_b,
        w,
        plan={1: "SAY_2", 2: "SAY_1", 3: "SAY_4"},
        client=client,
        input_mode="facts_only",
        focal_agent=focal,
    )
    assert len(client.calls) == 3
    assert res.record["calls_made"] == 3
    assert len(res.sealed) == 3
    assert res.events[0].action == "rewrite_rolled_back"
    assert res.events[0].cause == "not_one_sentence"
    assert res.events[1].action == "rewrite_rolled_back"
    assert res.events[1].cause == "call_error"
    assert res.events[2].action == "rewritten"
    assert res.events[2].after_form == "SAY_4"


def test_case_12_plan_equals_current_form_not_called() -> None:
    """Case 12: Events whose planned form equals the current realized form are not called."""
    doc = _load_fixture_doc()
    fx = doc["fixture_scene"]

    ev_1 = Event(
        step=1,
        kind="say",
        actor=str(fx["speaker"]),
        room=str(fx["room"]),
        utterance=str(fx["utterance"]),
        noise_level=int(fx["noise_level"]),
    )
    ev_2 = Event(
        step=2,
        kind="say",
        actor=str(fx["speaker"]),
        room=str(fx["room"]),
        utterance="The silver compass is on the brass tray.",
        noise_level=64,
    )
    w = _make_invented_fixture_world((ev_1, ev_2))
    focal = w.agents[1]
    prose_b = render_world_template(w, focal)

    # In `render_world_template`, both say events start in `SAY_3`.
    # Plan assigns `SAY_3` to Event 1 (matches current form -> NOT called) and `SAY_2` to Event 2 (called once).
    valid_ev2_say2 = '"The silver compass is on the brass tray," Mira said at 64 dB in the archive vestibule.'
    client = FakeRewriteClient([valid_ev2_say2])
    res = rewrite_all_say(
        prose_b,
        w,
        plan={1: "SAY_3", 2: "SAY_2"},
        client=client,
        input_mode="facts_only",
        focal_agent=focal,
    )
    assert len(client.calls) == 1
    assert len(res.sealed) == 1
    assert res.sealed[0]["event_index"] == 2
    assert res.events[0].action == "kept"
    assert res.events[0].before_form == "SAY_3"
    assert res.events[0].after_form == "SAY_3"
    assert res.events[0].call is None
    assert res.events[1].action == "rewritten"
    assert res.events[1].before_form == "SAY_3"
    assert res.events[1].after_form == "SAY_2"
    assert res.events[1].call is not None


def test_case_13_determinism_byte_identical_outputs() -> None:
    """Case 13: Determinism: the same fake replies give byte-identical outputs (`prose`, `record`, and `sealed`)."""
    doc = _load_fixture_doc()
    fx = doc["fixture_scene"]
    w = _make_invented_fixture_world()
    focal = w.agents[1]
    prose_b = render_world_template(w, focal)
    reply_say_4 = str(fx["valid_say_4"])

    client_1 = FakeRewriteClient([reply_say_4])
    client_2 = FakeRewriteClient([reply_say_4])

    res_1 = rewrite_all_say(
        prose_b,
        w,
        plan={1: "SAY_4"},
        client=client_1,
        input_mode="with_draft",
        focal_agent=focal,
    )
    res_2 = rewrite_all_say(
        prose_b,
        w,
        plan={1: "SAY_4"},
        client=client_2,
        input_mode="with_draft",
        focal_agent=focal,
    )

    assert res_1.prose.encode("utf-8") == res_2.prose.encode("utf-8")
    assert json.dumps(res_1.record, sort_keys=True).encode("utf-8") == json.dumps(
        res_2.record, sort_keys=True
    ).encode("utf-8")
    assert json.dumps(res_1.sealed, sort_keys=True).encode("utf-8") == json.dumps(
        res_2.sealed, sort_keys=True
    ).encode("utf-8")


def test_case_14_keep_rate_bookkeeping_say2_to_4_and_say1_no_rate() -> None:
    """Case 14: Keep-rate bookkeeping: attempts and kept per target shape for SAY_2..SAY_4 only; SAY_1 reported as (draft matched + rewrite kept) / assigned with no rate; rollbacks counted by cause."""
    events = [
        # SAY_1 already matched by writer draft
        RewriteEventRecord(
            event_index=1,
            kind="say",
            variant="say",
            plan_form="SAY_1",
            before_form="SAY_1",
            after_form="SAY_1",
            sentence_index=0,
            action="kept",
        ),
        # SAY_1 rewritten from SAY_3 and kept
        RewriteEventRecord(
            event_index=2,
            kind="say",
            variant="say",
            plan_form="SAY_1",
            before_form="SAY_3",
            after_form="SAY_1",
            sentence_index=1,
            action="rewritten",
        ),
        # SAY_1 rewrite rolled back (utterance_changed)
        RewriteEventRecord(
            event_index=3,
            kind="say",
            variant="say",
            plan_form="SAY_1",
            before_form="SAY_3",
            after_form="SAY_3",
            sentence_index=2,
            action="rewrite_rolled_back",
            cause="utterance_changed",
        ),
        # SAY_2: 2 attempts, 1 kept, 1 rolled back (not_one_sentence)
        RewriteEventRecord(
            event_index=4,
            kind="say",
            variant="say",
            plan_form="SAY_2",
            before_form="SAY_1",
            after_form="SAY_2",
            sentence_index=3,
            action="rewritten",
        ),
        RewriteEventRecord(
            event_index=5,
            kind="say",
            variant="say",
            plan_form="SAY_2",
            before_form="SAY_1",
            after_form="SAY_1",
            sentence_index=4,
            action="rewrite_rolled_back",
            cause="not_one_sentence",
        ),
        # SAY_3: 1 attempt, 1 kept
        RewriteEventRecord(
            event_index=6,
            kind="say",
            variant="say",
            plan_form="SAY_3",
            before_form="SAY_1",
            after_form="SAY_3",
            sentence_index=5,
            action="rewritten",
        ),
        # SAY_4: 2 attempts, 1 kept, 1 rolled back (not_one_sentence)
        RewriteEventRecord(
            event_index=7,
            kind="say",
            variant="say",
            plan_form="SAY_4",
            before_form="SAY_1",
            after_form="SAY_4",
            sentence_index=6,
            action="rewritten",
        ),
        RewriteEventRecord(
            event_index=8,
            kind="say",
            variant="say",
            plan_form="SAY_4",
            before_form="SAY_1",
            after_form="SAY_1",
            sentence_index=7,
            action="rewrite_rolled_back",
            cause="not_one_sentence",
        ),
    ]

    bk = compute_say_rewrite_bookkeeping(events)
    per_shape = bk["per_shape"]
    assert isinstance(per_shape, dict)
    # Only SAY_2, SAY_3, SAY_4 are in per_shape; SAY_1 is NOT in per_shape
    assert tuple(per_shape.keys()) == SAY_REWRITE_TARGET_FORMS == ("SAY_2", "SAY_3", "SAY_4")
    assert "SAY_1" not in per_shape
    assert per_shape["SAY_2"] == {"attempts": 2, "kept": 1, "rate": 0.5}
    assert per_shape["SAY_3"] == {"attempts": 1, "kept": 1, "rate": 1.0}
    assert per_shape["SAY_4"] == {"attempts": 2, "kept": 1, "rate": 0.5}

    pooled = bk["pooled"]
    assert pooled == {"attempts": 5, "kept": 3, "rate": 0.6}

    say1 = bk["say_1"]
    assert isinstance(say1, dict)
    assert "rate" not in say1
    assert say1["assigned"] == 3
    assert say1["draft_matched"] == 1
    assert say1["rewrite_attempted"] == 2
    assert say1["rewrite_kept"] == 1
    assert say1["realized"] == 2
    assert say1["expression"] == "(1 + 1) / 3"

    assert bk["rollbacks_by_cause"] == {
        "not_one_sentence": 2,
        "utterance_changed": 1,
    }


def test_case_15_u2_public_wrapper_parity_and_count_changed() -> None:
    """Case 15: U2's public wrapper `check_splice_keep` in `shape_templates.py` is exposed, matches `swap_all_move_see`/`swap_all_say` behaviour byte-identically, and covers `count_changed`."""
    w = world_from_scenario_seed(2026110500, index=0)
    focal = w.agents[0]
    prose_b = render_world_template(w, focal)

    b_faults = collect_prose_faults(prose_b, w, focal_agent=focal, checker_profile="d")
    b_rows, b_sent_count = match_rendering_event_forms(
        prose_b,
        w,
        focal_agent=focal,
        checker_profile="d",
        return_sentence_count=True,
    )

    say_pos = next(i for i, ev in enumerate(w.events) if ev.kind == "say")
    say_ev = w.events[say_pos]
    s_idx = int(b_rows[say_pos]["sentence_index"])  # type: ignore[arg-type]

    # 1. Valid SAY_2 candidate -> kept=True
    cand_say2 = render_event_sentence(say_ev, w, "SAY_2", focal_agent=focal)
    cand_prose_ok = splice_event_sentence_at_index(prose_b, s_idx, cand_say2)
    keep_ok: SpliceKeepCheckResult = check_splice_keep(
        prose_b,
        cand_prose_ok,
        w,
        ev_pos=say_pos,
        target_form="SAY_2",
        current_faults=b_faults,
        current_rows=b_rows,
        current_sent_count=b_sent_count,
        focal_agent=focal,
        checker_profile="d",
    )
    assert keep_ok.kept is True
    assert keep_ok.cause is None

    # Compare with `swap_all_say` on the same event
    swap_res = swap_all_say(prose_b, w, plan={say_pos + 1: "SAY_2"}, focal_agent=focal, checker_profile="d")
    assert swap_res.prose.encode("utf-8") == cand_prose_ok.encode("utf-8")

    # 2. Two-sentence candidate -> kept=False, cause="count_changed"
    cand_prose_two = splice_event_sentence_at_index(prose_b, s_idx, f"{cand_say2} {w.agents[0]} waited.")
    keep_count_changed = check_splice_keep(
        prose_b,
        cand_prose_two,
        w,
        ev_pos=say_pos,
        target_form="SAY_2",
        current_faults=b_faults,
        current_rows=b_rows,
        current_sent_count=b_sent_count,
        focal_agent=focal,
        checker_profile="d",
    )
    assert keep_count_changed.kept is False
    assert keep_count_changed.cause == "count_changed"

    # 3. Verify U2's test file, render.py, and cli.py are unchanged at U3 commit a7ca5831 vs U2 base 842a2e6b
    repo_root = Path(__file__).resolve().parent.parent
    u2_sha = "842a2e6b6ef15c46ca92fb85d2412ec82404d187"
    u3_sha = "a7ca5831a96d2b034a9f995ac58ea0a85e73206f"
    for sha in (u2_sha, u3_sha):
        check_proc = subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if check_proc.returncode != 0:
            pytest.skip(f"Commit {sha} missing from clone (shallow checkout)")
    proc = subprocess.run(
        ["git", "diff", "--name-only", u2_sha, u3_sha],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    changed = {line.strip() for line in proc.stdout.splitlines() if line.strip()}
    allowed_u3 = {
        "src/udlab_generator/shape_templates.py",
        "src/udlab_generator/rewrite.py",
        "tests/test_stage5_rewrite.py",
        "tests/fixtures/stage5/stage5_rewrite_fixtures.json",
    }
    assert changed <= allowed_u3, f"Unexpected modified files vs U2 base: {sorted(changed - allowed_u3)}"
    assert "tests/test_stage5_templates.py" not in changed
    assert "src/udlab_generator/render.py" not in changed
    assert "src/udlab_generator/cli.py" not in changed


class _RecordingCallableClient:
    """Fake callable rewrite client that records signature-inspection and call order, arity, and payloads."""

    def __init__(
        self,
        *,
        arity: int = 1,
        signature_exc: Exception | None = None,
        first_call_exc: Exception | None = None,
        reply: str = "",
    ) -> None:
        self._arity = arity
        self._signature_exc = signature_exc
        self._first_call_exc = first_call_exc
        self._reply = reply
        self.events: list[str] = []
        self.call_arities: list[int] = []
        self.payloads: list[dict[str, Any]] = []
        self.calls: list[tuple[Any, ...]] = []
        self.call_count: int = 0

    @property
    def __signature__(self) -> inspect.Signature:
        self.events.append("signature_inspected")
        if self._signature_exc is not None:
            raise self._signature_exc
        if self._arity == 1:
            return inspect.Signature(
                [inspect.Parameter("payload", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
            )
        return inspect.Signature(
            [
                inspect.Parameter("method", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter("url", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter("payload", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            ]
        )

    def __call__(self, *args: Any) -> dict[str, Any]:
        self.events.append("called")
        self.call_count += 1
        self.call_arities.append(len(args))
        self.calls.append(args)
        payload = args[0] if len(args) == 1 else args[2]
        assert isinstance(payload, dict)
        self.payloads.append(json.loads(json.dumps(payload)))
        if self.call_count == 1 and self._first_call_exc is not None:
            raise self._first_call_exc
        return {
            "model": payload.get("model", PINNED_WRITER_MODEL_TAG),
            "done_reason": "stop",
            "prompt_eval_count": 140,
            "eval_count": 22,
            "total_duration": 125_000_000,
            "message": {"role": "assistant", "content": self._reply},
        }


def _assert_payload_and_sealed_identical_across_call_forms(
    prose_b: str,
    w: World,
    focal: str,
    valid_reply: str,
    *,
    input_mode: str = "with_draft",
) -> None:
    """Run the same event once through a 1-arg client and once through a (method, url, payload) client and assert parity."""
    client_1arg = _RecordingCallableClient(arity=1, reply=valid_reply)
    client_3arg = _RecordingCallableClient(arity=3, reply=valid_reply)

    res_1arg = rewrite_all_say(
        prose_b,
        w,
        plan={1: "SAY_2"},
        client=client_1arg,
        focal_agent=focal,
        input_mode=input_mode,
    )
    res_3arg = rewrite_all_say(
        prose_b,
        w,
        plan={1: "SAY_2"},
        client=client_3arg,
        focal_agent=focal,
        input_mode=input_mode,
    )

    assert client_1arg.events == ["signature_inspected", "called"]
    assert client_3arg.events == ["signature_inspected", "called"]
    assert client_1arg.call_arities == [1]
    assert client_3arg.call_arities == [3]
    assert len(client_1arg.payloads) == 1
    assert len(client_3arg.payloads) == 1
    assert client_1arg.payloads[0] == client_3arg.payloads[0]
    assert len(res_1arg.sealed) == 1
    assert len(res_3arg.sealed) == 1
    assert res_1arg.sealed[0]["messages"] == res_3arg.sealed[0]["messages"]


def test_u3fix_client_typeerror_is_one_call_and_call_error() -> None:
    """U3-fix1: 1-arg callable client raising TypeError on first call is called once, rolls back with call_error, and leaves prose unchanged."""
    w = _make_invented_fixture_world()
    focal = w.agents[1]
    prose_b = render_world_template(w, focal)
    valid_say2 = render_event_sentence(w.events[0], w, "SAY_2", focal_agent=focal)

    client = _RecordingCallableClient(
        arity=1,
        first_call_exc=TypeError("simulated client TypeError on first call"),
        reply=valid_say2,
    )
    res = rewrite_all_say(
        prose_b,
        w,
        plan={1: "SAY_2"},
        client=client,
        focal_agent=focal,
        input_mode="with_draft",
    )

    # No call precedes signature inspection / calling-form choice, and exactly 1 call is made.
    assert client.events[0] == "signature_inspected"
    assert client.events.index("signature_inspected") < client.events.index("called")
    assert client.call_count == 1
    assert client.events == ["signature_inspected", "called"]
    assert client.call_arities == [1]

    ev_rec = next(r for r in res.events if r.event_index == 1)
    assert ev_rec.action == "rewrite_rolled_back"
    assert ev_rec.cause == "call_error"
    assert res.prose.encode("utf-8") == prose_b.encode("utf-8")

    _assert_payload_and_sealed_identical_across_call_forms(
        prose_b,
        w,
        focal,
        valid_say2,
        input_mode="with_draft",
    )


def test_u3fix_client_valueerror_is_one_call_and_call_error() -> None:
    """U3-fix1: 1-arg callable client raising ValueError on first call is called once, rolls back with call_error, and leaves prose unchanged."""
    w = _make_invented_fixture_world()
    focal = w.agents[1]
    prose_b = render_world_template(w, focal)
    valid_say2 = render_event_sentence(w.events[0], w, "SAY_2", focal_agent=focal)

    client = _RecordingCallableClient(
        arity=1,
        first_call_exc=ValueError("simulated client ValueError on first call"),
        reply=valid_say2,
    )
    res = rewrite_all_say(
        prose_b,
        w,
        plan={1: "SAY_2"},
        client=client,
        focal_agent=focal,
        input_mode="facts_only",
    )

    # No call precedes signature inspection / calling-form choice, and exactly 1 call is made.
    assert client.events[0] == "signature_inspected"
    assert client.events.index("signature_inspected") < client.events.index("called")
    assert client.call_count == 1
    assert client.events == ["signature_inspected", "called"]
    assert client.call_arities == [1]

    ev_rec = next(r for r in res.events if r.event_index == 1)
    assert ev_rec.action == "rewrite_rolled_back"
    assert ev_rec.cause == "call_error"
    assert res.prose.encode("utf-8") == prose_b.encode("utf-8")

    _assert_payload_and_sealed_identical_across_call_forms(
        prose_b,
        w,
        focal,
        valid_say2,
        input_mode="facts_only",
    )


def test_u3fix_uninspectable_client_signature_is_one_fallback_call() -> None:
    """U3-fix1: Client whose signature cannot be inspected gets 1 call in fallback (method, url, payload) form after inspection."""
    w = _make_invented_fixture_world()
    focal = w.agents[1]
    prose_b = render_world_template(w, focal)
    valid_say2 = render_event_sentence(w.events[0], w, "SAY_2", focal_agent=focal)

    for sig_exc in (ValueError("no signature found for builtin"), TypeError("uninspectable callable")):
        uninspectable_ok = _RecordingCallableClient(
            arity=3,
            signature_exc=sig_exc,
            reply=valid_say2,
        )
        res_ok = rewrite_all_say(
            prose_b,
            w,
            plan={1: "SAY_2"},
            client=uninspectable_ok,
            focal_agent=focal,
            input_mode="with_draft",
        )
        assert uninspectable_ok.events[0] == "signature_inspected"
        assert uninspectable_ok.events.index("signature_inspected") < uninspectable_ok.events.index("called")
        assert uninspectable_ok.call_count == 1
        assert uninspectable_ok.events == ["signature_inspected", "called"]
        assert uninspectable_ok.call_arities == [3]
        method_arg, url_arg, payload_arg = uninspectable_ok.calls[0]
        assert method_arg == "POST"
        assert str(url_arg).endswith("/api/chat")
        assert isinstance(payload_arg, dict)
        ev_rec_ok = next(r for r in res_ok.events if r.event_index == 1)
        assert ev_rec_ok.action == "rewritten"

        # Compare payload and sealed prompt against a 1-arg client on the same event.
        client_1arg = _RecordingCallableClient(arity=1, reply=valid_say2)
        res_1arg = rewrite_all_say(
            prose_b,
            w,
            plan={1: "SAY_2"},
            client=client_1arg,
            focal_agent=focal,
            input_mode="with_draft",
        )
        assert client_1arg.events == ["signature_inspected", "called"]
        assert client_1arg.payloads[0] == uninspectable_ok.payloads[0]
        assert res_1arg.sealed[0]["messages"] == res_ok.sealed[0]["messages"]

    # Also verify a 3-arg fallback client that raises ValueError on its first call gets 1 call (not 2 via except fallback).
    fallback_err_client = _RecordingCallableClient(
        arity=3,
        first_call_exc=ValueError("simulated fallback client ValueError on first call"),
        reply=valid_say2,
    )
    res_err = rewrite_all_say(
        prose_b,
        w,
        plan={1: "SAY_2"},
        client=fallback_err_client,
        focal_agent=focal,
        input_mode="with_draft",
    )
    assert fallback_err_client.events[0] == "signature_inspected"
    assert fallback_err_client.call_count == 1
    assert fallback_err_client.events == ["signature_inspected", "called"]
    ev_rec_err = next(r for r in res_err.events if r.event_index == 1)
    assert ev_rec_err.action == "rewrite_rolled_back"
    assert ev_rec_err.cause == "call_error"
    assert res_err.prose.encode("utf-8") == prose_b.encode("utf-8")


