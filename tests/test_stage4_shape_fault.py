"""Stage 4 shape-fault (`form_not_accepted`) and `tag_echo` fixture and unit tests."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from udlab_generator import (
    EVENT_FIDELITY_SUBRULE_IDS,
    TAG_ECHO_TOKENS,
    VALID_FAULT_RULE_IDS,
    AgentGoal,
    Event,
    FactSpec,
    RenderCheckError,
    StakeEdge,
    World,
    check_rendered_prose,
    collect_prose_faults,
    render_world_template,
    rescore_prose_records,
)
from udlab_generator.render import (
    _event_content_present,
    _render_event_sentence,
    _sentence_matches_event,
    _sentence_rejection_rule_id,
)

STAGE3_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "stage3" / "stage3_gate0_fixtures.json"
)
STAGE4_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "stage4" / "stage4_shape_fault_fixtures.json"
)


def _load_stage3_world() -> World:
    from udlab_generator import world_from_scenario_seed

    doc = json.loads(STAGE3_FIXTURE_PATH.read_text(encoding="utf-8"))
    spec = doc["world"]
    base = world_from_scenario_seed(0, index=0)
    rooms = tuple(str(r) for r in spec["rooms"])
    agents = tuple(str(a) for a in spec["agents"])
    absent = tuple(str(a) for a in spec["absent_parties"])
    objects = tuple(str(o) for o in spec["objects"])
    doorways = tuple((str(a), str(b)) for a, b in spec["doorways"])
    containers_by_room = {
        str(k): (str(v[0]), str(v[1]))
        for k, v in spec["containers_by_room"].items()
    }
    occluders_by_room = {
        str(k): str(v) for k, v in spec["occluders_by_room"].items()
    }
    default_fixtures = ("wall bench", "corner desk", "floor rack")
    fixtures_by_room = {
        str(k): (str(v) if v is not None else default_fixtures[i % len(default_fixtures)])
        for i, (k, v) in enumerate(spec["fixtures_by_room"].items())
    }
    return replace(
        base,
        scenario_id=str(spec["scenario_id"]),
        seed=int(spec["seed"]),
        domain_name=str(spec["domain_name"]),
        rooms=rooms,
        doorways=doorways,
        agents=agents,
        absent_parties=absent,
        objects=objects,
        containers_by_room=containers_by_room,
        occluders_by_room=occluders_by_room,
        fixtures_by_room=fixtures_by_room,
        initial_agent_rooms={ag: rooms[0] for ag in agents},
        stake_edges=(
            StakeEdge(
                agent=absent[0],
                variable="F1",
                description=(
                    f"{absent[0]}'s audit sign-off for {agents[0]} depends on the location of the {objects[0]}."
                ),
            ),
        ),
        goals={
            ag: AgentGoal(
                agent=ag,
                goal_text=f"{ag} plans to inspect the {rooms[0]}.",
                causal_preconditions=("F1",),
            )
            for ag in agents
        },
        noise_threshold=int(spec["noise_threshold"]),
    )


def _load_stage4_fixture_doc() -> dict[str, Any]:
    return json.loads(STAGE4_FIXTURE_PATH.read_text(encoding="utf-8"))


def _build_stage4_events(doc: dict[str, Any]) -> dict[str, Event]:
    events_spec: dict[str, dict[str, Any]] = doc["events"]
    out: dict[str, Event] = {}
    for idx, (key, raw) in enumerate(events_spec.items(), start=1):
        out[key] = Event(
            step=idx,
            kind=str(raw["kind"]),
            actor=str(raw["actor"]),
            room=str(raw["room"]),
            target_object=raw.get("target_object"),
            from_location=raw.get("from_location"),
            to_location=raw.get("to_location"),
            utterance=raw.get("utterance"),
            noise_level=int(raw["noise_level"]) if raw.get("noise_level") is not None else None,
            occluded=bool(raw.get("occluded", False)),
            occluder=raw.get("occluder"),
        )
    return out


_STAGE4_DOC = _load_stage4_fixture_doc()
_STAGE4_EVENTS = _build_stage4_events(_STAGE4_DOC)
_STAGE4_CASES: list[dict[str, Any]] = list(_STAGE4_DOC["fixtures"])


def test_stage4_fixture_file_sha256_and_counts() -> None:
    """Verify the Stage 4 fixture file SHA-256 and exact category counts (6/9/15/5 = 35)."""
    digest = hashlib.sha256(STAGE4_FIXTURE_PATH.read_bytes()).hexdigest()
    assert digest.startswith("552096b3")
    assert len(_STAGE4_CASES) == 35
    by_expected: dict[str, int] = {}
    for row in _STAGE4_CASES:
        exp = str(row["expect"])
        by_expected[exp] = by_expected.get(exp, 0) + 1
    assert by_expected == {
        "accepted": 6,
        "form_not_accepted": 9,
        "meaning": 15,
        "tag_echo": 5,
    }


def test_stage4_rule_ids_and_tag_echo_tokens_registered() -> None:
    """Verify `form_not_accepted`, `tag_echo`, and `TAG_ECHO_TOKENS` registration."""
    assert "form_not_accepted" in EVENT_FIDELITY_SUBRULE_IDS
    assert "form_not_accepted" in VALID_FAULT_RULE_IDS
    assert "tag_echo" in EVENT_FIDELITY_SUBRULE_IDS
    assert "tag_echo" in VALID_FAULT_RULE_IDS
    expected_tokens = (
        *(f"SAY_{n}" for n in range(10)),
        *(f"MOVE_{n}" for n in range(10)),
        *(f"SEE_{n}" for n in range(10)),
        "[Actor]",
        "[Room]",
        "[Object]",
        "[Occluder]",
    )
    assert TAG_ECHO_TOKENS == expected_tokens


@pytest.mark.parametrize("case", _STAGE4_CASES, ids=[ str(c["id"]) for c in _STAGE4_CASES ])
def test_stage4_gate0_fixture_row_profile_d(case: dict[str, Any]) -> None:
    """Gate 0: Every row in `stage4_shape_fault_fixtures.json` meets its expectation under profile `d`."""
    base_world = _load_stage3_world()
    ev = _STAGE4_EVENTS[str(case["event"])]
    sentence = str(case["sentence"])
    expected = str(case["expect"])

    world_1ev = replace(base_world, events=(ev,))
    focal = world_1ev.agents[0]
    template_prose = render_world_template(world_1ev, focal)
    setup_para, _ = template_prose.split("\n\n", 1)
    candidate_prose = f"{setup_para}\n\n{sentence}"

    matched = _sentence_matches_event(
        sentence,
        ev,
        world_1ev.noise_threshold,
        world=world_1ev,
        checker_profile="d",
    )
    content_ok = _event_content_present(sentence, ev, world_1ev, "d")
    faults = collect_prose_faults(
        candidate_prose,
        world_1ev,
        focal_agent=focal,
        checker_profile="d",
    )
    rescored = rescore_prose_records(
        [{"world": world_1ev, "focal_agent": focal, "prose": candidate_prose}],
        checker_version="stage4-v1",
        checker_profile="d",
    )[0]

    if expected == "accepted":
        assert matched is True
        assert content_ok is True
        assert faults == ()
        assert rescored["passed"] is True
        assert rescored["primary_rule_id"] is None
        check_rendered_prose(candidate_prose, world_1ev, focal_agent=focal, checker_profile="d")
    elif expected == "form_not_accepted":
        assert matched is False
        assert content_ok is True
        rule_id = _sentence_rejection_rule_id(
            sentence,
            ev,
            world_1ev.noise_threshold,
            world=world_1ev,
            checker_profile="d",
        )
        assert rule_id == "form_not_accepted"
        assert len(faults) == 1
        assert faults[0].check_id == "c_fidelity_missing_event"
        assert faults[0].rule_id == "form_not_accepted"
        assert rescored["passed"] is False
        assert rescored["primary_rule_id"] == "form_not_accepted"
        assert rescored["rule_id_counts"] == {"form_not_accepted": 1}
    elif expected == "meaning":
        assert matched is False
        assert content_ok is False
        rule_id = _sentence_rejection_rule_id(
            sentence,
            ev,
            world_1ev.noise_threshold,
            world=world_1ev,
            checker_profile="d",
        )
        assert rule_id in VALID_FAULT_RULE_IDS
        assert rule_id not in ("form_not_accepted", "tag_echo")
        assert len(faults) >= 1
        assert all(f.rule_id not in ("form_not_accepted", "tag_echo") for f in faults)
        assert rescored["passed"] is False
        assert rescored["primary_rule_id"] not in ("form_not_accepted", "tag_echo")
    elif expected == "tag_echo":
        assert matched is False
        assert content_ok is False
        rule_id = _sentence_rejection_rule_id(
            sentence,
            ev,
            world_1ev.noise_threshold,
            world=world_1ev,
            checker_profile="d",
        )
        assert rule_id == "tag_echo"
        assert len(faults) >= 1
        assert all(f.rule_id == "tag_echo" for f in faults)
        assert rescored["passed"] is False
        assert rescored["primary_rule_id"] == "tag_echo"
    else:  # pragma: no cover
        pytest.fail(f"Unexpected fixture category: {expected!r}")


@pytest.mark.parametrize("profile", ["c", "d"])
@pytest.mark.parametrize(
    "tag_token",
    ["SAY_0", "SAY_9", "MOVE_0", "MOVE_9", "SEE_0", "SEE_9", "[Actor]", "[Room]", "[Object]", "[Occluder]", "say_4", "move_7", "see_1", "[actor]", "[occluder]"],
)
def test_tag_echo_rejected_in_both_profiles(profile: str, tag_token: str) -> None:
    """Every tag token in `TAG_ECHO_TOKENS` (case-insensitive, word boundaries) is rejected as `tag_echo` in C and D."""
    world = _load_stage3_world()
    ev = _STAGE4_EVENTS["move"]
    sent = f"In the amber room, {tag_token} Mira moved the copper token from the pine crate to the tin locker."
    assert not _sentence_matches_event(sent, ev, world.noise_threshold, world=world, checker_profile=profile)
    assert not _event_content_present(sent, ev, world, profile)
    assert (
        _sentence_rejection_rule_id(sent, ev, world.noise_threshold, world=world, checker_profile=profile)
        == "tag_echo"
    )


def test_sequential_checker_alignment_with_form_not_accepted() -> None:
    """Sequential checker labels `form_not_accepted` only for the sentence that `_event_content_present` ties to the event."""
    base_world = _load_stage3_world()
    ev1 = replace(_STAGE4_EVENTS["say"], step=1)
    ev2 = replace(_STAGE4_EVENTS["move"], step=2)
    ev3 = replace(_STAGE4_EVENTS["see_clear"], step=3)
    world = replace(base_world, events=(ev1, ev2, ev3))
    focal = world.agents[0]
    template_prose = render_world_template(world, focal)
    setup_para, _ = template_prose.split("\n\n", 1)

    s1_valid = _render_event_sentence(ev1, focal, world.noise_threshold)
    s2_fna = "It was Mira who moved the copper token from the pine crate to the tin locker in the amber room."
    s3_valid = _render_event_sentence(ev3, focal, world.noise_threshold)

    # Case 1: ev1 omitted (absent), ev2 present in unaccepted form, ev3 valid
    prose_ev1_absent = f"{setup_para}\n\n{s2_fna} {s3_valid}"
    faults_absent = collect_prose_faults(prose_ev1_absent, world, focal_agent=focal, checker_profile="d")
    assert len(faults_absent) == 2
    assert faults_absent[0].event_index == 1
    assert faults_absent[0].failure_kind == "absent"
    assert faults_absent[0].rule_id == "c_fidelity_missing_event"
    assert faults_absent[1].event_index == 2
    assert faults_absent[1].failure_kind == "altered"
    assert faults_absent[1].rule_id == "form_not_accepted"

    # Case 2: aligned ev2 sentence is labeled form_not_accepted, while a misaligned sentence for ev2 is NOT labeled form_not_accepted
    prose_aligned_fna = f"{setup_para}\n\n{s1_valid} {s2_fna} {s3_valid}"
    faults_aligned = collect_prose_faults(prose_aligned_fna, world, focal_agent=focal, checker_profile="d")
    assert len(faults_aligned) == 1
    assert faults_aligned[0].event_index == 2
    assert faults_aligned[0].rule_id == "form_not_accepted"

    wrong_sent_for_ev2 = "The hallway clock chimed once."
    prose_misaligned = f"{setup_para}\n\n{s1_valid} {wrong_sent_for_ev2} {s3_valid}"
    faults_misaligned = collect_prose_faults(prose_misaligned, world, focal_agent=focal, checker_profile="d")
    assert len(faults_misaligned) == 1
    assert faults_misaligned[0].event_index == 2
    assert faults_misaligned[0].rule_id != "form_not_accepted"


def test_event_content_present_edge_cases_and_other_kinds() -> None:
    """Cover `_event_content_present` branches for invalid profile, empty sentence, pronoun, message, enter, leave, occluded move, and station move."""
    world = _load_stage3_world()
    ev_move = _STAGE4_EVENTS["move"]
    ev_say = _STAGE4_EVENTS["say"]
    ev_see = _STAGE4_EVENTS["see_clear"]
    ev_see_occ = _STAGE4_EVENTS["see_occ"]

    with pytest.raises(ValueError, match="checker_profile must be one of"):
        _event_content_present("In the amber room, Mira moved the copper token from the pine crate to the tin locker.", ev_move, world, "z")

    assert not _event_content_present("", ev_move, world, "d")
    assert not _event_content_present("no candidate sentence", ev_move, world, "d")

    # Pronoun rejection
    assert not _event_content_present(
        "In the amber room, Mira moved the copper token from the pine crate to her tin locker.",
        ev_move,
        world,
        "d",
    )

    # Extra container / fixture / occluder / object rejections
    assert not _event_content_present(
        "In the amber room, Mira moved the copper token from the pine crate to the tin locker near the slate box.",
        ev_move,
        world,
        "d",
    )
    assert not _event_content_present(
        "In the amber room, Mira moved the copper token from the pine crate to the tin locker near the wall sundial.",
        ev_move,
        world,
        "d",
    )
    assert not _event_content_present(
        "In the amber room, Mira moved the copper token from the pine crate to the tin locker near the felt curtain.",
        ev_move,
        world,
        "d",
    )

    # Negated target or location (`no copper token`, `no pine crate`)
    assert not _event_content_present(
        "In the amber room, Mira moved no copper token from the pine crate to the tin locker.",
        ev_move,
        world,
        "d",
    )
    assert not _event_content_present(
        "In the amber room, Mira moved the copper token from no pine crate to the tin locker.",
        ev_move,
        world,
        "d",
    )

    # Say clause boundary and threshold
    assert not _event_content_present(
        'In the amber room, Mira said, "The copper token is stored in the tin locker," while Nolan was at 55 dB.',
        ev_say,
        world,
        "d",
    )
    assert not _event_content_present(
        'In the amber room, Mira whispered at 55 dB, "The copper token is stored in the tin locker."',
        ev_say,
        world,
        "d",
    )

    # Unoccluded see with blocking phrase or hid/hidden
    assert not _event_content_present(
        "In the amber room, Mira saw the copper token in the pine crate behind the hidden panel.",
        ev_see,
        world,
        "d",
    )
    assert not _event_content_present(
        "In the amber room, Mira looked for the copper token in the pine crate.",
        ev_see,
        world,
        "d",
    )
    assert not _event_content_present(
        "In the amber room, Mira saw the copper token.",
        ev_see,
        world,
        "d",
    )

    # Occluded see missing far-side alcove or asserting active sighting
    assert not _event_content_present(
        "In the amber room, the pine crate was hidden from Mira by the felt curtain.",
        ev_see_occ,
        world,
        "d",
    )
    assert not _event_content_present(
        "In the amber room at the far-side alcove, Mira saw the pine crate behind the felt curtain which blocked sight.",
        ev_see_occ,
        world,
        "d",
    )

    # Message event
    ev_msg = Event(
        step=5,
        kind="message",
        actor="Mira",
        room="amber room",
        recipient="Quin",
        utterance="The copper token is stored in the tin locker.",
    )
    assert _event_content_present(
        'Mira sent a message to Quin: "The copper token is stored in the tin locker."',
        ev_msg,
        world,
        "d",
    )
    assert _event_content_present(
        "Mira sent a message to Quin that the copper token is stored in the tin locker.",
        ev_msg,
        world,
        "d",
    )
    assert not _event_content_present(
        'Mira sent a message to Pia: "The copper token is stored in the tin locker."',
        ev_msg,
        world,
        "d",
    )

    # Enter and leave events
    ev_leave = Event(
        step=6,
        kind="leave",
        actor="Mira",
        room="amber room",
        from_location="amber room",
        to_location="birch room",
    )
    ev_enter = Event(
        step=7,
        kind="enter",
        actor="Mira",
        room="birch room",
        from_location="amber room",
        to_location="birch room",
    )
    assert _event_content_present(
        "Mira exited the amber room toward the birch room.",
        ev_leave,
        world,
        "d",
    )
    assert _event_content_present(
        "Mira entered the birch room from the amber room.",
        ev_enter,
        world,
        "d",
    )
    assert not _event_content_present(
        "Mira exited the amber room toward the cedar room.",
        ev_leave,
        world,
        "d",
    )
