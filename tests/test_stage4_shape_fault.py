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
    RenderCheckError,
    StakeEdge,
    World,
    check_prose_no_markers,
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
EXPECTED_VERDICTS_PATH = (
    Path(__file__).resolve().parent / "stage4_verdict_invariance_expected.json"
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
    # The fixture world's "pine crate" comes from the "customs_wharf" domain.
    return replace(
        base,
        scenario_id=str(spec["scenario_id"]),
        seed=int(spec["seed"]),
        domain_name="customs_wharf",
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

_STATION_MOVE_EVENTS: dict[str, Event] = {
    "station_to_alcove": Event(
        step=1,
        kind="move",
        actor="Mira",
        room="amber room",
        target_object="felt curtain",
        from_location="container bay",
        to_location="far-side alcove",
        occluded=False,
    ),
    "station_to_bay": Event(
        step=1,
        kind="move",
        actor="Mira",
        room="amber room",
        target_object="felt curtain",
        from_location="far-side alcove",
        to_location="container bay",
        occluded=False,
    ),
}

_STATION_MOVE_FIXTURES: list[dict[str, str]] = [
    {
        "id": "acc_station_move_to_alcove",
        "event": "station_to_alcove",
        "expect": "accepted",
        "sentence": "In the amber room, Mira stepped behind the felt curtain to the far-side alcove.",
    },
    {
        "id": "acc_station_move_return_to_bay",
        "event": "station_to_bay",
        "expect": "accepted",
        "sentence": "In the amber room, Mira stepped back around the felt curtain to the container bay.",
    },
    {
        "id": "fna_station_move_cleft_to_alcove",
        "event": "station_to_alcove",
        "expect": "form_not_accepted",
        "sentence": "It was Mira who stepped behind the felt curtain to the far-side alcove in the amber room.",
    },
    {
        "id": "fna_station_move_cleft_return_to_bay",
        "event": "station_to_bay",
        "expect": "form_not_accepted",
        "sentence": "It was Mira who stepped around the felt curtain to the container bay in the amber room.",
    },
    {
        "id": "mean_station_move_wrong_occluder_cleft",
        "event": "station_to_alcove",
        "expect": "meaning",
        "sentence": "It was Mira who stepped behind the timber screen to the far-side alcove in the amber room.",
    },
    {
        "id": "mean_station_move_wrong_occluder_std",
        "event": "station_to_alcove",
        "expect": "meaning",
        "sentence": "In the amber room, Mira stepped behind the timber screen to the far-side alcove.",
    },
]


def _build_verdict_invariance_battery() -> list[dict[str, Any]]:
    """Construct the 32 multi-event scenes in the fixture world for verdict-invariance testing."""
    ev_map: dict[str, Event] = {
        **_STAGE4_EVENTS,
        **_STATION_MOVE_EVENTS,
        "leave_ab": Event(
            step=1,
            kind="leave",
            actor="Mira",
            room="amber room",
            from_location="amber room",
            to_location="birch room",
        ),
        "enter_ab": Event(
            step=1,
            kind="enter",
            actor="Mira",
            room="birch room",
            from_location="amber room",
            to_location="birch room",
        ),
    }
    fix_by_id = {str(r["id"]): str(r["sentence"]) for r in _STAGE4_CASES}
    w = _load_stage3_world()
    focal = w.agents[0]
    nt = w.noise_threshold

    s_see_clear_std = _render_event_sentence(ev_map["see_clear"], focal, nt)
    s_see_occ_std = _render_event_sentence(ev_map["see_occ"], focal, nt)
    s_stat_alc_std = _render_event_sentence(ev_map["station_to_alcove"], focal, nt)
    s_stat_bay_std = _render_event_sentence(ev_map["station_to_bay"], focal, nt)
    s_leave_std = _render_event_sentence(ev_map["leave_ab"], focal, nt)
    s_enter_std = _render_event_sentence(ev_map["enter_ab"], focal, nt)
    s_stat_alc_fna = "It was Mira who stepped behind the felt curtain to the far-side alcove in the amber room."
    s_stat_alc_wrong_occ = "In the amber room, Mira stepped behind the timber screen to the far-side alcove."

    specs: list[tuple[str, str, list[str], list[str]]] = [
        ("v01_valid_say_move_see", "valid", ["say", "move", "see_clear"], [fix_by_id["acc_say_std"], fix_by_id["acc_move_std"], s_see_clear_std]),
        ("v02_valid_station_occ_return", "valid", ["station_to_alcove", "see_occ", "station_to_bay"], [s_stat_alc_std, s_see_occ_std, s_stat_bay_std]),
        ("v03_valid_leave_enter_say", "valid", ["leave_ab", "enter_ab", "say"], [s_leave_std, s_enter_std, fix_by_id["acc_say_std"]]),
        ("v04_valid_profile_d_widenings", "valid_d_only", ["say", "move", "see_occ", "see_clear"], [fix_by_id["acc_say_s2"], fix_by_id["acc_move_m3"], fix_by_id["acc_see_v2"], fix_by_id["acc_see_p1"]]),
        ("v05_valid_four_events_canonical", "valid", ["say", "see_clear", "move", "station_to_alcove"], [fix_by_id["acc_say_std"], s_see_clear_std, fix_by_id["acc_move_std"], s_stat_alc_std]),
        ("v06_valid_folded_leave_enter_move", "valid", ["leave_ab", "enter_ab", "move"], ["Mira walked from the amber room into the birch room.", fix_by_id["acc_move_std"]]),
        ("fna07_first_say_split", "form_bad", ["say", "move", "see_clear"], [fix_by_id["fna_say_split"], fix_by_id["acc_move_std"], s_see_clear_std]),
        ("fna08_first_say_inverted", "form_bad", ["say", "move"], [fix_by_id["fna_say_inverted"], fix_by_id["acc_move_std"]]),
        ("fna09_second_say_indirect", "form_bad", ["move", "say", "see_clear"], [fix_by_id["acc_move_std"], fix_by_id["fna_say_indirect"], s_see_clear_std]),
        ("fna10_second_move_cleft", "form_bad", ["say", "move", "see_clear"], [fix_by_id["acc_say_std"], fix_by_id["fna_move_cleft"], s_see_clear_std]),
        ("fna11_third_move_to_first", "form_bad", ["say", "see_clear", "move"], [fix_by_id["acc_say_std"], s_see_clear_std, fix_by_id["fna_move_to_first"]]),
        ("fna12_first_move_obj_fronted", "form_bad", ["move", "see_clear"], [fix_by_id["fna_move_object_fronted"], s_see_clear_std]),
        ("fna13_second_see_passive", "form_bad", ["move", "see_clear", "say"], [fix_by_id["acc_move_std"], fix_by_id["fna_see_passive"], fix_by_id["acc_say_std"]]),
        ("fna14_third_see_cleft", "form_bad", ["say", "move", "see_clear"], [fix_by_id["acc_say_std"], fix_by_id["acc_move_std"], fix_by_id["fna_see_cleft"]]),
        ("fna15_second_see_occ_passive", "form_bad", ["station_to_alcove", "see_occ", "station_to_bay"], [s_stat_alc_std, fix_by_id["fna_see_occ_passive"], s_stat_bay_std]),
        ("fna16_station_move_cleft", "form_bad", ["station_to_alcove", "see_occ"], [s_stat_alc_fna, s_see_occ_std]),
        ("mean17_wrong_actor_and_valid", "meaning_bad", ["say", "move", "see_clear"], [fix_by_id["mean_say_inverted_wrong_actor"], fix_by_id["acc_move_std"], s_see_clear_std]),
        ("mean18_wrong_room_and_fna", "meaning_bad", ["say", "move"], [fix_by_id["mean_say_split_wrong_room"], fix_by_id["fna_move_cleft"]]),
        ("mean19_reversed_move_and_valid", "meaning_bad", ["say", "move", "see_clear"], [fix_by_id["acc_say_std"], fix_by_id["mean_move_to_first_reversed"], s_see_clear_std]),
        ("mean20_negated_move_and_hedged_see", "meaning_bad", ["move", "see_clear"], [fix_by_id["mean_move_cleft_negated"], fix_by_id["mean_see_cleft_hedged"]]),
        ("mean21_wrong_occluder_and_clear_occ", "meaning_bad", ["station_to_alcove", "see_occ"], [s_stat_alc_wrong_occ, fix_by_id["mean_see_occ_passive_clear"]]),
        ("omit22_review_scene_a", "omitted_and_form_bad", ["say", "see_clear", "move"], [fix_by_id["acc_say_std"], fix_by_id["fna_move_cleft"]]),
        ("omit23_omit_ev1_then_fna_ev2_valid_ev3", "omitted_and_form_bad", ["say", "move", "see_clear"], [fix_by_id["fna_move_cleft"], s_see_clear_std]),
        ("omit24_omit_ev1_then_valid_ev2_valid_ev3", "omitted", ["say", "move", "see_clear"], [fix_by_id["acc_move_std"], s_see_clear_std]),
        ("omit25_omit_ev3_after_valid_and_fna", "omitted_and_form_bad", ["say", "move", "see_clear"], [fix_by_id["acc_say_std"], fix_by_id["fna_move_to_first"]]),
        ("omit26_omit_middle_meaning_bad_last", "omitted_and_meaning_bad", ["say", "see_clear", "move"], [fix_by_id["acc_say_std"], fix_by_id["mean_move_cleft_wrong_object"]]),
        ("reord27_review_scene_b", "reordered_form_bad", ["move", "see_clear"], [fix_by_id["fna_see_cleft"], fix_by_id["fna_move_cleft"]]),
        ("reord28_swap_valid_say_move", "reordered", ["say", "move", "see_clear"], [fix_by_id["acc_move_std"], fix_by_id["acc_say_std"], s_see_clear_std]),
        ("reord29_swap_valid_and_fna", "reordered_and_form_bad", ["move", "see_clear", "say"], [s_see_clear_std, fix_by_id["fna_move_cleft"], fix_by_id["acc_say_std"]]),
        ("bund30_bundled_move_and_see_one_sent", "bundled", ["move", "see_clear"], ["In the amber room, Mira moved the copper token from the pine crate to the tin locker, and then Mira saw the copper token in the pine crate."]),
        ("bund31_bundled_with_trailing_valid", "bundled", ["say", "move", "see_clear"], [fix_by_id["acc_say_std"], "In the amber room, Mira moved the copper token from the pine crate to the tin locker; Mira saw the copper token in the pine crate.", s_see_clear_std]),
        ("mix32_fna_meaning_extra_unlogged", "mixed", ["say", "move", "see_clear"], [fix_by_id["fna_say_inverted"], fix_by_id["mean_move_cleft_bystander"], s_see_clear_std, "In the amber room, Mira saw the glass prism in the tin locker."]),
    ]

    battery: list[dict[str, Any]] = []
    for scene_id, category, ev_keys, sents in specs:
        evs = tuple(replace(ev_map[k], step=i) for i, k in enumerate(ev_keys, start=1))
        w_scene = replace(w, events=evs)
        template_prose = render_world_template(w_scene, focal)
        setup_para, _ = template_prose.split("\n\n", 1)
        prose = f"{setup_para}\n\n" + " ".join(sents)
        battery.append(
            {
                "id": scene_id,
                "category": category,
                "world": w_scene,
                "focal": focal,
                "prose": prose,
            }
        )
    return battery


_BATTERY_SCENES = _build_verdict_invariance_battery()
_EXPECTED_VERDICTS_DOC: dict[str, Any] = json.loads(
    EXPECTED_VERDICTS_PATH.read_text(encoding="utf-8")
)
_EXPECTED_BY_ID: dict[str, dict[str, Any]] = {
    str(row["id"]): row for row in _EXPECTED_VERDICTS_DOC["scenes"]
}


def test_stage4_fixture_file_sha256_and_counts() -> None:
    """Verify the Stage 4 fixture file SHA-256 and exact category counts (6/9/15/5 = 35)."""
    digest = hashlib.sha256(STAGE4_FIXTURE_PATH.read_bytes()).hexdigest()
    assert digest == "552096b373544a7f970c3a377e469e42ae3d3125ca910e060c98a2beba75c88c"
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


@pytest.mark.parametrize("case", _STAGE4_CASES, ids=[str(c["id"]) for c in _STAGE4_CASES])
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


@pytest.mark.parametrize(
    "case",
    _STATION_MOVE_FIXTURES,
    ids=[str(c["id"]) for c in _STATION_MOVE_FIXTURES],
)
def test_station_move_content_recognition_and_fault_labels(case: dict[str, str]) -> None:
    """Card 1b item 1: station moves use content recognition in `_event_content_present` rather than `_sentence_matches_station_move`."""
    base_world = _load_stage3_world()
    ev = _STATION_MOVE_EVENTS[case["event"]]
    sentence = case["sentence"]
    expected = case["expect"]

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

    if expected == "accepted":
        assert matched is True
        assert content_ok is True
        assert faults == ()
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
        assert rule_id != "form_not_accepted"
        assert len(faults) >= 1
        assert all(f.rule_id != "form_not_accepted" for f in faults)
    else:  # pragma: no cover
        pytest.fail(f"Unexpected station-move expectation: {expected!r}")


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


@pytest.mark.parametrize(
    "scene",
    _BATTERY_SCENES,
    ids=[str(s["id"]) for s in _BATTERY_SCENES],
)
def test_stage4_verdict_invariance_battery_matches_base_856df86(scene: dict[str, Any]) -> None:
    """Card 1b item 2: scene pass/fail at head equals pass/fail recorded at base `856df86` across >=30 multi-event scenes."""
    assert len(_BATTERY_SCENES) >= 30
    assert _EXPECTED_VERDICTS_DOC["base_commit"] == "856df86ec04cf66e789c6346db6c2448a44289ba"
    scene_id = str(scene["id"])
    expected_row = _EXPECTED_BY_ID[scene_id]
    world_scene: World = scene["world"]
    focal: str = scene["focal"]
    prose: str = scene["prose"]

    for prof, key in (("c", "profile_c"), ("d", "profile_d")):
        exp_passed = bool(expected_row[key]["passed"])
        try:
            check_rendered_prose(prose, world_scene, focal_agent=focal, checker_profile=prof)
            head_passed = True
        except RenderCheckError:
            head_passed = False
        faults = collect_prose_faults(prose, world_scene, focal_agent=focal, checker_profile=prof)
        assert (len(faults) == 0) == head_passed
        assert head_passed == exp_passed, (
            f"Verdict mismatch on scene {scene_id!r} under profile {prof!r}: "
            f"head={head_passed} vs base={exp_passed}"
        )


def test_review_scenes_a_and_b_fault_attribution() -> None:
    """Card 1b item 2: explicit fault-list verification for the review's two scenes (a) and (b) at base vs head."""
    scene_by_id = {str(s["id"]): s for s in _BATTERY_SCENES}

    # Scene (a): [say, see_clear, move], where see_clear is omitted and move is a cleft ("It was Mira who moved ...")
    scene_a = scene_by_id["omit22_review_scene_a"]
    base_faults_a = [
        (f["event_index"], f["check_id"], f["rule_id"], f["failure_kind"])
        for f in _EXPECTED_BY_ID["omit22_review_scene_a"]["profile_d"]["faults"]
    ]
    assert base_faults_a == [
        (2, "c_fidelity_missing_event", "wrong_verb", "altered"),
        (3, "c_fidelity_missing_event", "c_fidelity_missing_event", "absent"),
    ]
    head_faults_a = [
        (f.event_index, f.check_id, f.rule_id, f.failure_kind)
        for f in collect_prose_faults(
            scene_a["prose"],
            scene_a["world"],
            focal_agent=scene_a["focal"],
            checker_profile="d",
        )
    ]
    assert head_faults_a == [
        (2, "c_fidelity_missing_event", "c_fidelity_missing_event", "absent"),
        (3, "c_fidelity_missing_event", "form_not_accepted", "altered"),
    ]

    # Scene (b): [move, see_clear], with two form-bad sentences, the second a cleft move
    scene_b = scene_by_id["reord27_review_scene_b"]
    base_faults_b = [
        (f["event_index"], f["check_id"], f["rule_id"], f["failure_kind"])
        for f in _EXPECTED_BY_ID["reord27_review_scene_b"]["profile_d"]["faults"]
    ]
    assert base_faults_b == [
        (1, "c_fidelity_missing_event", "wrong_verb", "altered"),
        (2, "c_fidelity_missing_event", "wrong_verb", "altered"),
    ]
    head_faults_b = [
        (f.event_index, f.check_id, f.rule_id, f.failure_kind)
        for f in collect_prose_faults(
            scene_b["prose"],
            scene_b["world"],
            focal_agent=scene_b["focal"],
            checker_profile="d",
        )
    ]
    assert head_faults_b == [
        (1, "c_fidelity_missing_event", "c_fidelity_missing_event", "absent"),
        (2, "c_fidelity_missing_event", "form_not_accepted", "altered"),
        (None, "c_fidelity_unlogged_event", "c_fidelity_unlogged_event", "altered"),
    ]


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


def test_label_only_extras_bracket_tag_echo_and_wider_hedge() -> None:
    """Card 1b item 4: test the two label-only extras (`check_prose_no_markers` bracket `tag_echo` and wider `negation_or_hedge`)."""
    world = _load_stage3_world()
    ev_see = _STAGE4_EVENTS["see_clear"]

    # 1. Bracket markers in check_prose_no_markers: role-label brackets get rule_id="tag_echo", other brackets keep "a_no_markers_bracket"
    for role_bracket in ("[Actor]", "[Room]", "[Object]", "[Occluder]", "[actor]"):
        with pytest.raises(RenderCheckError) as exc_role:
            check_prose_no_markers(f"In the amber room, {role_bracket} Mira moved the copper token.")
        assert exc_role.value.check_id == "a_no_markers_bracket"
        assert exc_role.value.rule_id == "tag_echo"

    with pytest.raises(RenderCheckError) as exc_other:
        check_prose_no_markers("In the amber room, [draft] Mira moved the copper token.")
    assert exc_other.value.check_id == "a_no_markers_bracket"
    assert exc_other.value.rule_id == "a_no_markers_bracket"

    # 2. Wider negation_or_hedge vocabulary in _sentence_rejection_rule_id (_P1_HEDGE_AND_MODAL_WORDS and _P1_HEDGE_PHRASES)
    for hedged_sent in (
        "It was Mira who might have seen the copper token in the pine crate in the amber room.",
        "In the amber room, Mira, perhaps in haste, saw the copper token in the pine crate.",
        "In the amber room, Mira, it seemed, saw the copper token in the pine crate.",
    ):
        assert not _sentence_matches_event(hedged_sent, ev_see, world.noise_threshold, world=world, checker_profile="d")
        assert not _event_content_present(hedged_sent, ev_see, world, "d")
        assert (
            _sentence_rejection_rule_id(hedged_sent, ev_see, world.noise_threshold, world=world, checker_profile="d")
            == "negation_or_hedge"
        )


def test_event_content_present_edge_cases_and_other_kinds() -> None:
    """Cover `_event_content_present` branches for invalid profile, empty sentence, pronoun, message, enter, leave, occluded move, and station move."""
    world = _load_stage3_world()
    ev_move = _STAGE4_EVENTS["move"]
    ev_say = _STAGE4_EVENTS["say"]
    ev_see = _STAGE4_EVENTS["see_clear"]
    ev_see_occ = _STAGE4_EVENTS["see_occ"]
    ev_stat = _STATION_MOVE_EVENTS["station_to_alcove"]

    with pytest.raises(ValueError, match="checker_profile must be one of"):
        _event_content_present("In the amber room, Mira moved the copper token from the pine crate to the tin locker.", ev_move, world, "z")

    assert not _event_content_present("", ev_move, world, "d")
    assert not _event_content_present("no candidate sentence", ev_move, world, "d")

    # Station move edge cases (direct manipulation of occluder, missing verb, reversed stations)
    assert not _event_content_present(
        "In the amber room, Mira moved the felt curtain to the far-side alcove.",
        ev_stat,
        world,
        "d",
    )
    assert not _event_content_present(
        "In the amber room, Mira stood behind the felt curtain at the far-side alcove.",
        ev_stat,
        world,
        "d",
    )
    assert not _event_content_present(
        "In the amber room, Mira stepped behind the felt curtain from the far-side alcove to the container bay.",
        ev_stat,
        world,
        "d",
    )

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
