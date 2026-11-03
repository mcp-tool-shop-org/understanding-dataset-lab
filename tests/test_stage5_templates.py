"""Stage 5 Card U2 tests: code templates, scene-checked swap pass, S-code say swap, and Option 3 control arm."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from udlab_generator.render import (
    _render_event_sentence,
    check_prose_render_fidelity,
    collect_prose_faults,
    event_shape_plan_variant,
    match_rendering_event_forms,
    matched_event_form_id,
    render_world_template,
    world_from_scenario_seed,
)
from udlab_generator.shape_templates import (
    ALL_TEMPLATE_FORMS,
    MOVE_FORMS,
    SAY_FORMS,
    SEE_OCCLUDED_FORMS,
    SEE_UNOCCLUDED_FORMS,
    STATION_MOVE_FORMS,
    applicable_forms_for_event,
    draw_scene_shape_plan,
    generate_stage5_shape_plan,
    render_control_scene,
    render_event_sentence,
    run_template_self_test,
    sentence_opens_with_seam_reference,
    splice_event_sentence_at_index,
    swap_all_move_see,
    swap_all_say,
)
from udlab_generator.world import (
    CONTAINER_BAY_STATION,
    FAR_SIDE_ALCOVE_STATION,
    AgentGoal,
    Event,
    StakeEdge,
    World,
    generate_world,
    make_balanced_configs,
)

BASE_COMMIT_SHA = "50441f46159d073eae0ea3d99c36903514b4bddf"
STAGE5_SELF_TEST_BASE_SEED = 2026110202
STAGE5_SELF_TEST_PLAN_SEED = 2026110299
FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "stage5"
    / "stage5_template_swap_fixtures.json"
)


def _load_fixture_doc() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _make_base_archival_world() -> World:
    """Create a deterministic synthetic archival_registry world (scen_0000) for unit/rollback tests."""
    return world_from_scenario_seed(2026110500, index=0)


def _make_custom_archival_world(events: tuple[Event, ...]) -> World:
    """Build a synthetic archival_registry world with agents ('Ana', 'Ben', 'Cara') and custom `events`."""
    base_world = _make_base_archival_world()
    agents = ("Ana", "Ben", "Cara")
    absent_parties = ("Dan",)
    rooms = ("vault room", "reading room", "bindery room", "sorting room")
    doorways = (
        ("vault room", "reading room"),
        ("reading room", "bindery room"),
        ("bindery room", "sorting room"),
    )
    goals = {
        "Ana": AgentGoal(
            agent="Ana",
            goal_text="Ana plans to complete the quarterly audit packet",
            causal_preconditions=("F1", "F2"),
        ),
        "Ben": AgentGoal(
            agent="Ben",
            goal_text="Ben plans to verify the municipal deed transfer",
            causal_preconditions=("F1", "F3"),
        ),
        "Cara": AgentGoal(
            agent="Cara",
            goal_text="Cara plans to catalog the incoming charter folio",
            causal_preconditions=("F1", "F2", "F4"),
        ),
    }
    stake_edges = (
        StakeEdge(
            agent="Dan",
            variable="location:brass ledger",
            description="Dan must close tonight's audit, so his sign-off depends on whether the brass ledger is still in the vault room.",
        ),
    )
    return replace(
        base_world,
        agents=agents,
        absent_parties=absent_parties,
        rooms=rooms,
        doorways=doorways,
        initial_agent_rooms={"Ana": "vault room", "Ben": "vault room", "Cara": "vault room"},
        goals=goals,
        stake_edges=stake_edges,
        events=events,
        unknowable_spec=replace(base_world.unknowable_spec, trace_a=events, trace_b=events),
    )


def test_gate0_template_self_test_270_generated_scenarios() -> None:
    """Card U2 Item 2 & 6 (Gate 0): 270 generated scenarios x 2 renderings = 540 scenes, 100% pass across all shapes, all-at-once swap, and control arm."""
    configs = make_balanced_configs(270, base_seed=STAGE5_SELF_TEST_BASE_SEED)
    worlds = [generate_world(cfg) for cfg in configs]
    assert len(worlds) == 270

    report = run_template_self_test(worlds, plan_seed=STAGE5_SELF_TEST_PLAN_SEED)
    assert report["scenarios"] == 270
    assert report["renderings"] == 540
    assert report["failures"] == [], f"Template self-test failures: {report['failures'][:5]}"
    assert report["all_passed"] is True

    # Every event kind and generated variant appears across the 270 scenarios
    by_kind = report["events_by_kind_per_scenario"]
    by_variant = report["events_by_variant_per_scenario"]
    assert isinstance(by_kind, dict) and isinstance(by_variant, dict)
    for req_kind in ("enter", "leave", "move", "see", "say", "message"):
        assert by_kind.get(req_kind, 0) > 0, f"Missing event kind {req_kind}"
    for req_variant in ("move_station_change", "move_normal", "see_unoccluded", "see_occluded", "say"):
        assert by_variant.get(req_variant, 0) > 0, f"Missing event variant {req_variant}"

    # 100% pass on every applicable template shape
    per_shape = report["per_shape"]
    assert isinstance(per_shape, dict)
    for form_id in ALL_TEMPLATE_FORMS:
        entry = per_shape[form_id]
        assert entry["attempted"] > 0, f"Shape {form_id} had 0 attempts"
        assert entry["passed"] == entry["attempted"], f"Shape {form_id} failed: {entry}"

    # All-at-once move+see+say swap and Option 3 control arm both pass 540/540 (100%)
    assert report["all_at_once_attempted"] == 540
    assert report["all_at_once_passed"] == 540
    assert report["control_arm_attempted"] == 540
    assert report["control_arm_passed"] == 540


def test_template_self_test_occluded_move_variants_and_reused_forms() -> None:
    """Verify `move_occluded` (`ev.occluded=True`, `ev.occluder` set) across `MOVE_1`..`MOVE_4`, reused forms, and `SAY_5` refusal."""
    base_world = _make_base_archival_world()
    focal = base_world.agents[0]

    # Locate a normal container move in base_world and construct its occluded counterpart
    move_pos = next(
        i
        for i, ev in enumerate(base_world.events)
        if event_shape_plan_variant(ev, base_world) == "move_normal"
    )
    orig_move = base_world.events[move_pos]
    occ_name = base_world.occluders_by_room[orig_move.room]
    occluded_move = replace(orig_move, occluded=True, occluder=occ_name)
    occ_events = list(base_world.events)
    occ_events[move_pos] = occluded_move
    occ_world = replace(
        base_world,
        events=tuple(occ_events),
        unknowable_spec=replace(
            base_world.unknowable_spec,
            trace_a=tuple(occ_events),
            trace_b=tuple(occ_events),
        ),
    )
    assert event_shape_plan_variant(occluded_move, occ_world) == "move_occluded"
    assert applicable_forms_for_event(occluded_move, occ_world) == MOVE_FORMS

    base_prose = render_world_template(occ_world, focal)
    base_rows, base_count = match_rendering_event_forms(
        base_prose,
        occ_world,
        focal_agent=focal,
        checker_profile="d",
        return_sentence_count=True,
    )
    s_idx = int(base_rows[move_pos]["sentence_index"])  # type: ignore[arg-type]

    for form_id in MOVE_FORMS:
        sent = render_event_sentence(occluded_move, occ_world, form_id, focal_agent=focal)
        assert matched_event_form_id(
            sent,
            occluded_move,
            occ_world.noise_threshold,
            world=occ_world,
            checker_profile="d",
        ) == form_id
        spliced = splice_event_sentence_at_index(base_prose, s_idx, sent)
        rows_after, count_after = match_rendering_event_forms(
            spliced,
            occ_world,
            focal_agent=focal,
            checker_profile="d",
            return_sentence_count=True,
        )
        assert count_after == base_count
        assert rows_after[move_pos]["realized_form_id"] == form_id
        check_prose_render_fidelity(spliced, occ_world, focal_agent=focal, checker_profile="d")

    # Verify reused shapes call `_render_event_sentence` identically
    for ev in occ_world.events:
        v = event_shape_plan_variant(ev, occ_world)
        reused_map = {
            "move_station_change": "MOVE_0",
            "move_normal": "MOVE_1",
            "move_occluded": "MOVE_1",
            "see_unoccluded": "SEE_1",
            "see_occluded": "SEE_4",
            "say": "SAY_3",
        }
        if v in reused_map:
            rf = reused_map[v]
            assert render_event_sentence(ev, occ_world, rf, focal_agent=focal) == _render_event_sentence(
                ev, focal, occ_world.noise_threshold
            )

    # Verify SAY_5 is refused as out of the Stage 5 pool
    say_ev = next(ev for ev in base_world.events if ev.kind == "say")
    with pytest.raises(ValueError, match="SAY_5 is out of the Stage 5 pool"):
        render_event_sentence(say_ev, base_world, "SAY_5", focal_agent=focal)


def test_rollback_1_splice_makes_passing_scene_fail_fault_grew() -> None:
    """Case 1: A splice that makes a passing scene fail (fronted path MOVE_2 next to a folded leave/enter) is rolled back with `fault_grew`, and the scene is byte-identical to before."""
    fixtures = _load_fixture_doc()["rollback_case_1_fault_grew"]
    focal = "Ana"

    ev_leave = Event(
        step=1,
        kind="leave",
        actor="Ana",
        room="slate tray",
        from_location="slate tray",
        to_location="reading room",
    )
    ev_enter = Event(
        step=2,
        kind="enter",
        actor="Ana",
        room="reading room",
        from_location="slate tray",
        to_location="reading room",
    )
    ev_move = Event(
        step=3,
        kind="move",
        actor="Ana",
        room="reading room",
        target_object="brass ledger",
        from_location="slate tray",
        to_location="cedar shelf",
        occluded=False,
    )
    w = _make_custom_archival_world((ev_leave, ev_enter, ev_move))
    setup_para = render_world_template(_make_custom_archival_world((ev_move,)), focal).split("\n\n", 1)[0]
    s_0 = str(fixtures["s_0_folded_leave_enter"])
    s_1 = str(fixtures["s_1_passive_move_3"])
    passing_prose = f"{setup_para}\n\n{s_0} {s_1}"

    # Before swap: scene passes profile d (0 faults)
    b_faults = collect_prose_faults(passing_prose, w, focal_agent=focal, checker_profile="d")
    assert b_faults == ()

    res = swap_all_move_see(
        passing_prose,
        w,
        plan={3: str(fixtures["planned_form_for_ev3"])},
        focal_agent=focal,
        checker_profile="d",
    )
    assert res.b_passed is True
    assert res.s_passed is True
    assert res.prose == passing_prose  # byte-identical to before
    ev3_rec = res.events[2]
    assert ev3_rec.event_index == 3
    assert ev3_rec.action == "swap_rolled_back"
    assert ev3_rec.cause == fixtures["expected_cause"]
    assert fixtures["expected_rule_id"] in ev3_rec.rule_ids
    assert ev3_rec.b_form == "MOVE_3"
    assert ev3_rec.final_form == "MOVE_3"


def test_rollback_2_neighbour_sentence_index_changes_same_form() -> None:
    """Case 2: A neighbour row's `sentence_index` changes while its form id stays the same: rolled back with `neighbour_changed`."""
    fixtures = _load_fixture_doc()["rollback_case_2_neighbour_index_shifts_same_form"]
    focal = "Ana"

    ev_leave = Event(
        step=1,
        kind="leave",
        actor="Ana",
        room="slate tray",
        from_location="slate tray",
        to_location="reading room",
    )
    ev_enter = Event(
        step=2,
        kind="enter",
        actor="Ana",
        room="reading room",
        from_location="slate tray",
        to_location="reading room",
    )
    ev_move = Event(
        step=3,
        kind="move",
        actor="Ana",
        room="reading room",
        target_object="brass ledger",
        from_location="slate tray",
        to_location="cedar shelf",
        occluded=False,
    )
    w = _make_custom_archival_world((ev_leave, ev_enter, ev_move))
    setup_para = render_world_template(_make_custom_archival_world((ev_move,)), focal).split("\n\n", 1)[0]
    s_0 = str(fixtures["s_0_folded_leave_enter"])
    s_1 = str(fixtures["s_1_passive_move_3"])
    s_2 = str(fixtures["s_2_standard_folded_leave_enter"])
    s_3 = str(fixtures["s_3_fronted_move_2"])
    prose_b = f"{setup_para}\n\n{s_0} {s_1} {s_2} {s_3}"

    rows_before = match_rendering_event_forms(prose_b, w, focal_agent=focal, checker_profile="d")
    assert [r["sentence_index"] for r in rows_before] == fixtures["before_indices"]
    assert [r["realized_form_id"] for r in rows_before[:2]] == fixtures["unchanged_neighbour_forms"]

    # Verify what happens if s_1 is spliced with MOVE_2 directly: neighbours keep form IDs LEAVE_0, ENTER_0
    # but their sentence_index shifts from 0 to 2.
    cand_sent = render_event_sentence(ev_move, w, "MOVE_2", focal_agent=focal)
    cand_prose = splice_event_sentence_at_index(prose_b, 1, cand_sent)
    rows_cand = match_rendering_event_forms(cand_prose, w, focal_agent=focal, checker_profile="d")
    assert [r["realized_form_id"] for r in rows_cand[:2]] == fixtures["unchanged_neighbour_forms"]
    assert [r["sentence_index"] for r in rows_cand] == fixtures["after_attempted_splice_indices"]

    # Now run swap_all_move_see: must roll back with neighbour_changed
    res = swap_all_move_see(
        prose_b,
        w,
        plan={3: "MOVE_2"},
        focal_agent=focal,
        checker_profile="d",
    )
    assert res.prose == prose_b
    ev3_rec = res.events[2]
    assert ev3_rec.action == "swap_rolled_back"
    assert ev3_rec.cause == "neighbour_changed"
    assert ev3_rec.sentence_index == 1
    assert ev3_rec.final_form == "MOVE_3"


def test_rollback_3_sentence_count_changes_two_sentences() -> None:
    """Case 3: The sentence count changes (a template emitting two sentences, forced with a fixture template): rolled back with `count_changed`."""
    fixtures = _load_fixture_doc()["rollback_case_3_two_sentence_template"]
    w = _make_base_archival_world()
    focal = w.agents[0]
    prose_b = render_world_template(w, focal)

    move_rec_idx = next(
        i + 1
        for i, ev in enumerate(w.events)
        if event_shape_plan_variant(ev, w) == "move_normal"
    )
    forced_two_sent = str(fixtures["forced_two_sentence_template"])

    def _fixture_template_fn(ev: Event, world: World, form_id: str, *, focal_agent: str) -> str:
        if ev.step == w.events[move_rec_idx - 1].step:
            return forced_two_sent
        return render_event_sentence(ev, world, form_id, focal_agent=focal_agent)

    res = swap_all_move_see(
        prose_b,
        w,
        plan={move_rec_idx: "MOVE_2"},
        focal_agent=focal,
        checker_profile="d",
        template_fn=_fixture_template_fn,
    )
    assert res.prose == prose_b
    target_rec = res.events[move_rec_idx - 1]
    assert target_rec.action == "swap_rolled_back"
    assert target_rec.cause == "count_changed"


def test_rollback_4_target_reports_wrong_form() -> None:
    """Case 4: The target reports the wrong form: rolled back with `wrong_form`."""
    fixtures = _load_fixture_doc()["rollback_case_4_wrong_form_template"]
    w = _make_base_archival_world()
    focal = w.agents[0]
    prose_b = render_world_template(w, focal)

    move_rec_idx = next(
        i + 1
        for i, ev in enumerate(w.events)
        if event_shape_plan_variant(ev, w) == "move_normal"
    )
    target_ev = w.events[move_rec_idx - 1]

    def _wrong_form_template_fn(ev: Event, world: World, form_id: str, *, focal_agent: str) -> str:
        if ev.step == target_ev.step and form_id == fixtures["planned_form"]:
            # Emit MOVE_1 sentence when MOVE_3 was planned
            return render_event_sentence(
                ev,
                world,
                str(fixtures["realized_form"]),
                focal_agent=focal_agent,
            )
        return render_event_sentence(ev, world, form_id, focal_agent=focal_agent)

    res = swap_all_move_see(
        prose_b,
        w,
        plan={move_rec_idx: str(fixtures["planned_form"])},
        focal_agent=focal,
        checker_profile="d",
        template_fn=_wrong_form_template_fn,
    )
    assert res.prose == prose_b
    target_rec = res.events[move_rec_idx - 1]
    assert target_rec.action == "swap_rolled_back"
    assert target_rec.cause == "wrong_form"


def test_rollback_5_b_failed_scene_removes_fault_passes_and_new_rule_id_rolls_back() -> None:
    """Case 5: A B-failed scene: a swap that removes a fault is kept and reported as S passing where B failed; a swap that adds a different rule id is rolled back."""
    fixtures = _load_fixture_doc()["rollback_case_5_b_failed_scene"]
    focal = "Ana"

    ev_move = Event(
        step=1,
        kind="move",
        actor="Ana",
        room="vault room",
        target_object="brass ledger",
        from_location="oak drawer",
        to_location="steel cabinet",
        occluded=False,
    )
    ev_see = Event(
        step=2,
        kind="see",
        actor="Ana",
        room="vault room",
        target_object="wall barometer",
        from_location=FAR_SIDE_ALCOVE_STATION,
        to_location=FAR_SIDE_ALCOVE_STATION,
        occluded=False,
    )
    w = _make_custom_archival_world((ev_move, ev_see))
    setup_para = render_world_template(w, focal).split("\n\n", 1)[0]
    bad_move_sent = str(fixtures["bad_move_sentence_pronoun"])
    good_see_sent = render_event_sentence(ev_see, w, "SEE_1", focal_agent=focal)
    b_failed_prose = f"{setup_para}\n\n{bad_move_sent} {good_see_sent}"

    # Part A: B fails with pronoun_lead on event 1, but event 1 still gets sentence_index=0 via content alignment;
    # swapping event 1 to MOVE_3 removes the fault so S passes where B failed.
    res_fixed = swap_all_move_see(
        b_failed_prose,
        w,
        plan={1: "MOVE_3", 2: "SEE_2"},
        focal_agent=focal,
        checker_profile="d",
    )
    assert res_fixed.b_passed is False
    assert fixtures["bad_move_rule_id"] in res_fixed.b_fault_rule_ids
    assert res_fixed.s_passed is True
    assert res_fixed.s_faults == ()
    assert res_fixed.events[0].action == "swapped"
    assert res_fixed.events[0].final_form == "MOVE_3"
    assert res_fixed.events[1].action == "swapped"
    assert res_fixed.events[1].final_form == "SEE_2"

    # Part B: A swap on a B-failed scene that replaces the fault with a DIFFERENT rule_id ("third_room") is rolled back.
    third_room_sent = str(fixtures["new_fault_template_third_room"])

    def _third_room_template_fn(ev: Event, world: World, form_id: str, *, focal_agent: str) -> str:
        if ev.step == 1:
            return third_room_sent
        return render_event_sentence(ev, world, form_id, focal_agent=focal_agent)

    res_new_rule = swap_all_move_see(
        b_failed_prose,
        w,
        plan={1: "MOVE_1"},
        focal_agent=focal,
        checker_profile="d",
        template_fn=_third_room_template_fn,
    )
    assert res_new_rule.b_passed is False
    assert res_new_rule.prose == b_failed_prose
    assert res_new_rule.events[0].action == "swap_rolled_back"
    assert res_new_rule.events[0].cause == "fault_grew"
    assert fixtures["new_fault_rule_id"] in res_new_rule.events[0].rule_ids


def test_edge_6_absent_event_no_sentence_index_not_spliced() -> None:
    """Case 6: An absent event (no `sentence_index`): `not_spliced`, and the scene is unchanged."""
    focal = "Ana"

    ev_move = Event(
        step=1,
        kind="move",
        actor="Ana",
        room="vault room",
        target_object="brass ledger",
        from_location="oak drawer",
        to_location="steel cabinet",
        occluded=False,
    )
    ev_see = Event(
        step=2,
        kind="see",
        actor="Ana",
        room="vault room",
        target_object="wall barometer",
        from_location=FAR_SIDE_ALCOVE_STATION,
        to_location=FAR_SIDE_ALCOVE_STATION,
        occluded=False,
    )
    w = _make_custom_archival_world((ev_move, ev_see))
    setup_para = render_world_template(w, focal).split("\n\n", 1)[0]
    only_see_sent = render_event_sentence(ev_see, w, "SEE_1", focal_agent=focal)
    # Event 1 (ev_move) is completely absent from the prose; only Event 2 is present.
    absent_prose = f"{setup_para}\n\n{only_see_sent}"

    rows = match_rendering_event_forms(absent_prose, w, focal_agent=focal, checker_profile="d")
    assert rows[0]["sentence_index"] is None
    assert rows[1]["sentence_index"] == 0

    res = swap_all_move_see(
        absent_prose,
        w,
        plan={1: "MOVE_3"},
        focal_agent=focal,
        checker_profile="d",
    )
    assert res.prose == absent_prose
    assert res.events[0].action == "not_spliced"
    assert res.events[0].sentence_index is None
    assert res.events[0].cause is None


def test_edge_7_station_change_uses_move_0_only() -> None:
    """Case 7: The station change uses `MOVE_0` only."""
    w = _make_base_archival_world()
    focal = w.agents[0]
    station_pos, station_ev = next(
        (i, ev)
        for i, ev in enumerate(w.events)
        if event_shape_plan_variant(ev, w) == "move_station_change"
    )
    assert applicable_forms_for_event(station_ev, w) == STATION_MOVE_FORMS == ("MOVE_0",)

    sent_m0 = render_event_sentence(station_ev, w, "MOVE_0", focal_agent=focal)
    assert matched_event_form_id(
        sent_m0,
        station_ev,
        w.noise_threshold,
        world=w,
        checker_profile="d",
    ) == "MOVE_0"

    for invalid_form in ("MOVE_1", "MOVE_2", "MOVE_3", "MOVE_4", "SEE_1", "SAY_1"):
        with pytest.raises(ValueError, match="not applicable to event"):
            render_event_sentence(station_ev, w, invalid_form, focal_agent=focal)

    prose_b = render_world_template(w, focal)
    res = swap_all_move_see(
        prose_b,
        w,
        plan={station_pos + 1: "MOVE_0"},
        focal_agent=focal,
        checker_profile="d",
    )
    assert res.events[station_pos].plan_form == "MOVE_0"
    assert res.events[station_pos].final_form == "MOVE_0"
    assert res.events[station_pos].action == "swapped"


def test_edge_8_seam_flag_pronoun_participial_and_plain_openings() -> None:
    """Case 8: Seam-flag fixtures: a pronoun opening, a participial opening, and a plain opening."""
    seam_fixtures = _load_fixture_doc()["seam_flag_fixtures"]

    for sent in seam_fixtures["pronoun_openings"]:
        assert sentence_opens_with_seam_reference(sent) is True, f"Expected True for pronoun opening: {sent!r}"
    for sent in seam_fixtures["participial_openings"]:
        assert sentence_opens_with_seam_reference(sent) is True, f"Expected True for participial opening: {sent!r}"
    for sent in seam_fixtures["plain_openings"]:
        assert sentence_opens_with_seam_reference(sent) is False, f"Expected False for plain opening: {sent!r}"

    # Verify in a full scene swap: event 1 is swapped, and we vary event 2's sentence opening across
    # pronoun, participial, and plain openings.
    focal = "Ana"
    ev_move = Event(
        step=1,
        kind="move",
        actor="Ana",
        room="vault room",
        target_object="brass ledger",
        from_location="oak drawer",
        to_location="steel cabinet",
        occluded=False,
    )
    ev_leave = Event(
        step=2,
        kind="leave",
        actor="Ana",
        room="vault room",
        from_location="vault room",
        to_location="reading room",
    )
    ev_enter = Event(
        step=3,
        kind="enter",
        actor="Ana",
        room="reading room",
        from_location="vault room",
        to_location="reading room",
    )
    w = _make_custom_archival_world((ev_move, ev_leave, ev_enter))
    setup_para = render_world_template(w, focal).split("\n\n", 1)[0]
    s_move = render_event_sentence(ev_move, w, "MOVE_1", focal_agent=focal)

    # 1. Next sentence opens with a participial phrase ("Walking from the vault room, Ana left into the reading room.")
    s_part_leave = "Walking from the vault room, Ana left into the reading room."
    res_part = swap_all_move_see(
        f"{setup_para}\n\n{s_move} {s_part_leave}",
        w,
        plan={1: "MOVE_3"},
        focal_agent=focal,
        checker_profile="d",
    )
    assert res_part.s_passed is True
    assert res_part.events[0].action == "swapped"
    assert res_part.events[0].seam_next_opens_ref is True
    assert res_part.seam_count == 1

    # 2. Next sentence opens with a pronoun ("She walked from the vault room into the reading room.")
    s_pron_leave = "She walked from the vault room into the reading room."
    res_pron = swap_all_move_see(
        f"{setup_para}\n\n{s_move} {s_pron_leave}",
        w,
        plan={1: "MOVE_3"},
        focal_agent=focal,
        checker_profile="d",
    )
    assert res_pron.events[0].action == "swapped"
    assert res_pron.events[0].seam_next_opens_ref is True
    assert res_pron.seam_count == 1

    # 3. Next sentence has a plain opening ("From the vault room, Ana walked into the reading room and left.")
    s_plain_leave = "From the vault room, Ana walked into the reading room and left."
    res_plain = swap_all_move_see(
        f"{setup_para}\n\n{s_move} {s_plain_leave}",
        w,
        plan={1: "MOVE_3"},
        focal_agent=focal,
        checker_profile="d",
    )
    assert res_plain.s_passed is True
    assert res_plain.events[0].action == "swapped"
    assert res_plain.events[0].seam_next_opens_ref is False
    assert res_plain.seam_count == 0


def test_edge_9_determinism_byte_identical_outputs() -> None:
    """Case 9: Determinism: the same inputs give byte-identical outputs."""
    configs = make_balanced_configs(20, base_seed=STAGE5_SELF_TEST_BASE_SEED)
    worlds = [generate_world(cfg) for cfg in configs]

    plan_a = generate_stage5_shape_plan(worlds, seed=20261102)
    plan_b = generate_stage5_shape_plan(worlds, seed=20261102)
    assert json.dumps(plan_a, sort_keys=True) == json.dumps(plan_b, sort_keys=True)
    # Verify SAY_5 never appears in Stage 5 plans
    for rec in plan_a["plan_records"]:  # type: ignore[union-attr]
        assert rec["form_id"] != "SAY_5"

    for w in worlds[:5]:
        focal = w.agents[0]
        scene_plan_1 = draw_scene_shape_plan(w, seed=20261102)
        scene_plan_2 = draw_scene_shape_plan(w, seed=20261102)
        assert scene_plan_1 == scene_plan_2

        ctrl_1 = render_control_scene(w, focal, scene_plan_1)
        ctrl_2 = render_control_scene(w, focal, scene_plan_2)
        assert ctrl_1.encode("utf-8") == ctrl_2.encode("utf-8")

        base_prose = render_world_template(w, focal)
        ms_1 = swap_all_move_see(base_prose, w, scene_plan_1, focal_agent=focal)
        ms_2 = swap_all_move_see(base_prose, w, scene_plan_2, focal_agent=focal)
        assert ms_1.prose.encode("utf-8") == ms_2.prose.encode("utf-8")
        assert json.dumps(ms_1.to_dict(), sort_keys=True) == json.dumps(ms_2.to_dict(), sort_keys=True)

        say_1 = swap_all_say(ms_1.prose, w, scene_plan_1, focal_agent=focal)
        say_2 = swap_all_say(ms_2.prose, w, scene_plan_2, focal_agent=focal)
        assert say_1.prose.encode("utf-8") == say_2.prose.encode("utf-8")
        assert json.dumps(say_1.to_dict(), sort_keys=True) == json.dumps(say_2.to_dict(), sort_keys=True)


def test_edge_10_no_existing_module_changes() -> None:
    """Case 10: `git diff --name-only <base>` shows only the allowed files and zero edits to `render.py` or `cli.py`."""
    repo_root = Path(__file__).resolve().parent.parent
    u1_sha = "50441f46159d073eae0ea3d99c36903514b4bddf"
    u2_sha = "842a2e6b6ef15c46ca92fb85d2412ec82404d187"
    for sha in (u1_sha, u2_sha):
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
        [
            "git",
            "diff",
            "--name-only",
            "50441f46159d073eae0ea3d99c36903514b4bddf",
            "842a2e6b6ef15c46ca92fb85d2412ec82404d187",
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    changed = {line.strip() for line in proc.stdout.splitlines() if line.strip()}
    allowed = {
        "src/udlab_generator/shape_templates.py",
        "tests/test_stage5_templates.py",
        "tests/test_stage5_sentence_index.py",
        "tests/fixtures/stage5/stage5_template_swap_fixtures.json",
    }
    assert changed <= allowed, f"Unexpected modified files vs {BASE_COMMIT_SHA}: {sorted(changed - allowed)}"
    assert "src/udlab_generator/render.py" not in changed
    assert "src/udlab_generator/cli.py" not in changed


def test_rollback_11_two_events_trade_sentence_indices_neighbour_changed() -> None:
    """Case 11: A swap that would reorder two events' sentences (event j's template is matched before event j-1's sentence, so the two events trade sentence indices) is rolled back with `neighbour_changed`."""
    fixtures = _load_fixture_doc()["rollback_case_11_two_events_trade_indices"]
    focal = "Ana"

    # Event 1 (j-1): unoccluded see by Ana of brass ledger in vault room with from_location="intake"
    # (so P1 phrase "near the intake" in s_0 is rejected for Event 1 by _actor_to_verb_matches_p1_phrase, render.py:3453,
    #  while s_1 matches Event 1 at sentence_index=1).
    ev_1 = Event(
        step=1,
        kind="see",
        actor="Ana",
        room="vault room",
        target_object="brass ledger",
        from_location="intake",
        to_location=None,
        occluded=False,
    )
    # Event 2 (j): unoccluded see by Ana of brass ledger in vault room with from_location=None
    # (so s_0 matches Event 2 via earlier_idx=0 at sentence_index=0).
    ev_2 = Event(
        step=2,
        kind="see",
        actor="Ana",
        room="vault room",
        target_object="brass ledger",
        from_location=None,
        to_location=None,
        occluded=False,
    )
    w = _make_custom_archival_world((ev_1, ev_2))
    setup_para = render_world_template(w, focal).split("\n\n", 1)[0]
    s_0 = str(fixtures["s_0_only_matches_ev2"])
    s_1 = str(fixtures["s_1_matches_both_ev1_and_ev2"])
    prose_b = f"{setup_para}\n\n{s_0} {s_1}"

    # Before the attempted splice on Event 2 (at sentence_index=0):
    # Event 1 has sentence_index=1 (SEE_1) and Event 2 has sentence_index=0 (SEE_2).
    rows_before = match_rendering_event_forms(prose_b, w, focal_agent=focal, checker_profile="d")
    indices_before = [r["sentence_index"] for r in rows_before]
    assert indices_before == fixtures["before_indices"] == [1, 0]

    # When Event 2's sentence at index 0 is spliced with Event 2's SEE_1 template,
    # that template has no "intake" phrase, so Event 1 (j-1) matches index 0 and Event 2 (j) matches index 1:
    # the two events trade sentence indices from [1, 0] to [0, 1]!
    cand_sent_ev2 = render_event_sentence(
        ev_2,
        w,
        str(fixtures["planned_form_for_ev2"]),
        focal_agent=focal,
    )
    cand_prose = splice_event_sentence_at_index(prose_b, 0, cand_sent_ev2)
    rows_after_splice = match_rendering_event_forms(cand_prose, w, focal_agent=focal, checker_profile="d")
    indices_after = [r["sentence_index"] for r in rows_after_splice]
    assert indices_after == fixtures["after_attempted_splice_indices"] == [0, 1]

    # Running swap_all_move_see with plan={2: "SEE_1"} attempts the splice on Event 2 and rolls it back
    # with cause == "neighbour_changed" because Event 1's sentence_index changed from 1 to 0.
    res = swap_all_move_see(
        prose_b,
        w,
        plan={2: str(fixtures["planned_form_for_ev2"])},
        focal_agent=focal,
        checker_profile="d",
    )
    assert res.prose == prose_b
    ev2_rec = res.events[1]
    assert ev2_rec.event_index == 2
    assert ev2_rec.action == "swap_rolled_back"
    assert ev2_rec.cause == fixtures["expected_cause"] == "neighbour_changed"
    assert [e.sentence_index for e in res.events] == [1, 0]


# ---------------------------------------------------------------------------
# Card U2-fix: Multi-block Stage 5 shape-plan tests (Do 2.1 - 2.6, 2.8)
# ---------------------------------------------------------------------------

_PLAN_STREAM_VARIANTS: dict[str, tuple[str, ...]] = {
    "move_normal": MOVE_FORMS,
    "move_occluded": MOVE_FORMS,
    "see_unoccluded": SEE_UNOCCLUDED_FORMS,
    "see_occluded": SEE_OCCLUDED_FORMS,
    "say": SAY_FORMS,
}


def _make_multiblock_plan_worlds() -> list[World]:
    """Build a 62-scenario bank where all 5 variant streams have >= 3 full blocks of 20 plus a non-zero tail."""
    raw_worlds = [generate_world(cfg) for cfg in make_balanced_configs(62, base_seed=20261103)]
    out_worlds: list[World] = []
    occ_moves_added = 0
    for w in raw_worlds:
        new_evs: list[Event] = []
        changed = False
        for ev in w.events:
            if event_shape_plan_variant(ev, w) == "move_normal" and occ_moves_added < 65:
                occ_name = w.occluders_by_room[ev.room]
                new_evs.append(replace(ev, occluded=True, occluder=occ_name))
                occ_moves_added += 1
                changed = True
            else:
                new_evs.append(ev)
        if changed:
            evs_t = tuple(new_evs)
            w = replace(
                w,
                events=evs_t,
                unknowable_spec=replace(w.unknowable_spec, trace_a=evs_t, trace_b=evs_t),
            )
        out_worlds.append(w)
    return out_worlds


def _assert_stream_plan_rules(
    seq: list[str],
    forms: tuple[str, ...],
    *,
    block_size: int = 20,
    variant_name: str = "",
) -> None:
    """Assert per-block spread <= 1 (full and tail), global spread <= 1, and no three-in-a-row anywhere."""
    from collections import Counter

    for b_start in range(0, len(seq), block_size):
        blk = seq[b_start : b_start + block_size]
        counts = Counter(blk)
        spread = max(counts.get(f, 0) for f in forms) - min(counts.get(f, 0) for f in forms)
        assert spread <= 1, (
            f"Stream {variant_name!r} block at {b_start} (len={len(blk)}) has spread {spread} > 1: {dict(counts)}"
        )

    global_counts = Counter(seq)
    global_spread = max(global_counts.get(f, 0) for f in forms) - min(
        global_counts.get(f, 0) for f in forms
    )
    assert global_spread <= 1, (
        f"Stream {variant_name!r} global spread {global_spread} > 1: {dict(global_counts)}"
    )

    for edge_start in (18, 38):
        if len(seq) >= edge_start + 4:
            edge_slice = seq[edge_start : edge_start + 4]
            assert not (edge_slice[0] == edge_slice[1] == edge_slice[2]) and not (
                edge_slice[1] == edge_slice[2] == edge_slice[3]
            ), f"Stream {variant_name!r} three-in-a-row across block edge {edge_start}..{edge_start + 3}: {edge_slice}"

    for idx in range(len(seq) - 2):
        assert not (seq[idx] == seq[idx + 1] == seq[idx + 2]), (
            f"Stream {variant_name!r} three-in-a-row at index {idx}: {seq[idx : idx + 3]}"
        )


def test_plan_1_several_blocks_balance_and_no_triples() -> None:
    """Card U2-fix Do 2.1: A bank where every stream has at least 3 full blocks plus a tail satisfies block spread <= 1, global spread <= 1, and no three in a row anywhere (including indices 18-21 and 38-41)."""
    worlds = _make_multiblock_plan_worlds()
    plan = generate_stage5_shape_plan(worlds, seed=20261103, block_size=20)
    recs: list[dict[str, Any]] = plan["plan_records"]  # type: ignore[assignment]

    for v, forms in _PLAN_STREAM_VARIANTS.items():
        seq = [str(r["form_id"]) for r in recs if r["variant"] == v]
        assert len(seq) > 60 and len(seq) % 20 != 0, (
            f"Stream {v!r} must have >= 3 full blocks of 20 plus a non-zero tail, got len={len(seq)}"
        )
        _assert_stream_plan_rules(seq, forms, block_size=20, variant_name=v)


def test_plan_2_rotation_across_three_blocks_k3() -> None:
    """Card U2-fix Do 2.2: On a stream of exactly 3 blocks with k = 3 (`see_occluded`), rotation cycles the extra/short uses across the 3 blocks so global counts are exact."""
    from collections import Counter

    base_w = _make_base_archival_world()
    occ_name = base_w.occluders_by_room["vault room"]
    occ_see_ev = Event(
        step=1,
        kind="see",
        actor="Ana",
        room="vault room",
        target_object="brass ledger",
        from_location=FAR_SIDE_ALCOVE_STATION,
        to_location=FAR_SIDE_ALCOVE_STATION,
        occluded=True,
        occluder=occ_name,
    )

    # 1. 3 blocks of block_size=10 (30 events, k=3, 10 = 3*3 + 1): each of SEE_3..5 gets the extra use (4 vs 3) in literally 1 block
    worlds_30 = [
        replace(
            _make_custom_archival_world((replace(occ_see_ev, step=1), replace(occ_see_ev, step=2))),
            scenario_id=f"scen_rot10_{i:04d}",
        )
        for i in range(15)
    ]
    plan_30 = generate_stage5_shape_plan(worlds_30, seed=20261103, block_size=10)
    seq_30 = [
        str(r["form_id"])
        for r in plan_30["plan_records"]  # type: ignore[union-attr]
        if r["variant"] == "see_occluded"
    ]
    assert len(seq_30) == 30
    extra_block_counts_10: Counter[str] = Counter()
    for b_idx in range(3):
        blk = seq_30[b_idx * 10 : (b_idx + 1) * 10]
        c = Counter(blk)
        assert sorted(c[f] for f in SEE_OCCLUDED_FORMS) == [3, 3, 4]
        for f in SEE_OCCLUDED_FORMS:
            if c[f] == 4:
                extra_block_counts_10[f] += 1
    assert {f: extra_block_counts_10[f] for f in SEE_OCCLUDED_FORMS} == {
        "SEE_3": 1,
        "SEE_4": 1,
        "SEE_5": 1,
    }
    assert Counter(seq_30) == {"SEE_3": 10, "SEE_4": 10, "SEE_5": 10}

    # 2. 3 blocks of block_size=20 (60 events, k=3, 20 = 6*3 + 2): each form gets the short count (6) in 1 block and extra (7) in 2 blocks, totaling 20 each
    worlds_60 = [
        replace(
            _make_custom_archival_world((replace(occ_see_ev, step=1), replace(occ_see_ev, step=2))),
            scenario_id=f"scen_rot20_{i:04d}",
        )
        for i in range(30)
    ]
    plan_60 = generate_stage5_shape_plan(worlds_60, seed=20261103, block_size=20)
    seq_60 = [
        str(r["form_id"])
        for r in plan_60["plan_records"]  # type: ignore[union-attr]
        if r["variant"] == "see_occluded"
    ]
    assert len(seq_60) == 60
    short_block_counts_20: Counter[str] = Counter()
    for b_idx in range(3):
        blk = seq_60[b_idx * 20 : (b_idx + 1) * 20]
        c = Counter(blk)
        assert sorted(c[f] for f in SEE_OCCLUDED_FORMS) == [6, 7, 7]
        for f in SEE_OCCLUDED_FORMS:
            if c[f] == 6:
                short_block_counts_20[f] += 1
    assert {f: short_block_counts_20[f] for f in SEE_OCCLUDED_FORMS} == {
        "SEE_3": 1,
        "SEE_4": 1,
        "SEE_5": 1,
    }
    assert Counter(seq_60) == {"SEE_3": 20, "SEE_4": 20, "SEE_5": 20}


def test_plan_3_tail_and_short_streams_7_and_47() -> None:
    """Card U2-fix Do 2.3: A short stream of 7 events and a multi-block + tail stream of 47 events both meet all plan rules across every variant stream."""
    for target_len in (7, 47):
        worlds_n: list[World] = []
        base_w = _make_base_archival_world()
        occ_name = base_w.occluders_by_room["vault room"]
        for i in range(target_len):
            evs = (
                Event(
                    step=1,
                    kind="move",
                    actor="Ana",
                    room="vault room",
                    target_object="brass ledger",
                    from_location="oak drawer",
                    to_location="steel cabinet",
                    occluded=False,
                ),
                Event(
                    step=2,
                    kind="move",
                    actor="Ana",
                    room="vault room",
                    target_object="brass ledger",
                    from_location="steel cabinet",
                    to_location="oak drawer",
                    occluded=True,
                    occluder=occ_name,
                ),
                Event(
                    step=3,
                    kind="see",
                    actor="Ana",
                    room="vault room",
                    target_object="brass ledger",
                    from_location=CONTAINER_BAY_STATION,
                    to_location="oak drawer",
                    occluded=False,
                ),
                Event(
                    step=4,
                    kind="see",
                    actor="Ana",
                    room="vault room",
                    target_object="brass ledger",
                    from_location=FAR_SIDE_ALCOVE_STATION,
                    to_location=FAR_SIDE_ALCOVE_STATION,
                    occluded=True,
                    occluder=occ_name,
                ),
                Event(
                    step=5,
                    kind="say",
                    actor="Ana",
                    room="vault room",
                    utterance="The brass ledger is in the oak drawer.",
                    noise_level=60,
                    occluded=False,
                ),
            )
            worlds_n.append(
                replace(_make_custom_archival_world(evs), scenario_id=f"scen_len{target_len}_{i:04d}")
            )

        for s in (20261103, 20261104, 20261105):
            plan = generate_stage5_shape_plan(worlds_n, seed=s, block_size=20)
            recs: list[dict[str, Any]] = plan["plan_records"]  # type: ignore[assignment]
            for v, forms in _PLAN_STREAM_VARIANTS.items():
                seq = [str(r["form_id"]) for r in recs if r["variant"] == v]
                assert len(seq) == target_len
                _assert_stream_plan_rules(seq, forms, block_size=20, variant_name=f"{v}_len{target_len}_seed{s}")


def test_plan_4_seed_sweep_200_seeds() -> None:
    """Card U2-fix Do 2.4: The three plan properties hold across 200 seeds (1000..1199) on the multi-block bank in under 60 s."""
    import time

    worlds = _make_multiblock_plan_worlds()
    t0 = time.perf_counter()
    for s in range(1000, 1200):
        plan = generate_stage5_shape_plan(worlds, seed=s, block_size=20)
        recs: list[dict[str, Any]] = plan["plan_records"]  # type: ignore[assignment]
        for v, forms in _PLAN_STREAM_VARIANTS.items():
            seq = [str(r["form_id"]) for r in recs if r["variant"] == v]
            _assert_stream_plan_rules(seq, forms, block_size=20, variant_name=f"{v}_seed{s}")
    elapsed = time.perf_counter() - t0
    assert elapsed < 60.0, f"200-seed sweep took {elapsed:.2f}s >= 60.0s"


def test_plan_5_determinism_byte_identical_plan() -> None:
    """Card U2-fix Do 2.5: The same seed gives a byte-identical plan on the multi-block bank."""
    worlds = _make_multiblock_plan_worlds()
    plan_1 = generate_stage5_shape_plan(worlds, seed=20261103, block_size=20)
    plan_2 = generate_stage5_shape_plan(worlds, seed=20261103, block_size=20)
    bytes_1 = json.dumps(plan_1, sort_keys=True).encode("utf-8")
    bytes_2 = json.dumps(plan_2, sort_keys=True).encode("utf-8")
    assert bytes_1 == bytes_2


def test_plan_6_named_failure_stage5_plan_generation_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Card U2-fix Do 2.6: A forced impossible case raises `STAGE5_PLAN_GENERATION_FAILED` naming the stream and block index."""
    import udlab_generator.shape_templates as st_mod

    # 1. Natural mathematical impossibility: 21-event stream (block 0 of 20 ending in (f*, f*) + 1-event tail block 1 with quota {f*: 1})
    see_ev = Event(
        step=1,
        kind="see",
        actor="Ana",
        room="vault room",
        target_object="brass ledger",
        from_location=CONTAINER_BAY_STATION,
        to_location="oak drawer",
        occluded=False,
    )
    worlds_21 = [
        replace(_make_custom_archival_world((see_ev,)), scenario_id=f"scen_imp_{i:04d}")
        for i in range(21)
    ]
    with pytest.raises(RuntimeError, match="STAGE5_PLAN_GENERATION_FAILED") as nat_exc:
        generate_stage5_shape_plan(worlds_21, seed=4, block_size=20)
    nat_msg = str(nat_exc.value)
    assert "STAGE5_PLAN_GENERATION_FAILED" in nat_msg
    assert "stream=see_unoccluded" in nat_msg
    assert "block=1" in nat_msg

    # 2. Forced unsatisfiable block via sampler feasibility hook
    worlds = [_make_base_archival_world()]
    monkeypatch.setattr(st_mod, "_can_complete_stage5_plan_block", lambda *a, **kw: False, raising=False)
    with pytest.raises(RuntimeError, match="STAGE5_PLAN_GENERATION_FAILED") as exc_info:
        generate_stage5_shape_plan(worlds, seed=20261103, block_size=20)
    msg = str(exc_info.value)
    assert "STAGE5_PLAN_GENERATION_FAILED" in msg
    assert "stream=move_normal" in msg
    assert "block=0" in msg


def test_plan_8_registered_switch_rate_band_200_seeds() -> None:
    """Card U2-fix Do 2.8: Over the 200-seed sweep (1000..1199), the two-form stream (`see_unoccluded`, k=2) switch rate stays inside the registered 0.60-0.71 band alongside the three plan rules."""
    from udlab_generator.render import SHAPE_PLAN_DESIGN_CHOICES

    band = SHAPE_PLAN_DESIGN_CHOICES["two_form_switch_rate_band"]
    low = float(band["low"])  # type: ignore[arg-type]
    high = float(band["high"])  # type: ignore[arg-type]
    assert (low, high) == (0.60, 0.71)

    worlds = _make_multiblock_plan_worlds()
    rates_by_stream: dict[str, list[float]] = {v: [] for v in _PLAN_STREAM_VARIANTS}
    for s in range(1000, 1200):
        plan = generate_stage5_shape_plan(worlds, seed=s, block_size=20)
        recs: list[dict[str, Any]] = plan["plan_records"]  # type: ignore[assignment]
        for v, forms in _PLAN_STREAM_VARIANTS.items():
            seq = [str(r["form_id"]) for r in recs if r["variant"] == v]
            _assert_stream_plan_rules(seq, forms, block_size=20, variant_name=f"{v}_seed{s}")
            switches = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
            sw_rate = switches / (len(seq) - 1)
            rates_by_stream[v].append(sw_rate)
            if len(forms) == 2:
                assert low <= sw_rate <= high, (
                    f"Two-form stream {v!r} seed={s} switch rate {sw_rate:.6f} outside [{low}, {high}]"
                )

