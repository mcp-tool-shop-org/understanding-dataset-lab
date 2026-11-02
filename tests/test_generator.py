"""Comprehensive test suite for the udlab release-1 scenario generator, sealing, and shortcut probe."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
from collections import Counter
import pytest

from udlab_generator import (
    BANNED_ADVERBS,
    UNKNOWABLE_KEY,
    AgentGoal,
    Event,
    ForbiddenModelCallError,
    MockWriter,
    RenderCheckError,
    SealingError,
    StakeEdge,
    UnknowableConstructionError,
    are_rooms_adjacent,
    build_pressure_probe_turn2,
    canonical_event_skeleton,
    check_not_in_key_order,
    check_rendered_prose,
    compute_absent_party_x,
    compute_diff_d,
    compute_first_order_belief,
    compute_knowledge,
    compute_known_facts_by_agent,
    compute_perceptual_records,
    compute_scenario_keys,
    compute_second_order_belief,
    compute_true_world_state,
    count_template_diversity,
    format_writer_prompt,
    generate_world,
    load_writer_prompt_template,
    make_balanced_configs,
    render_world_template,
    run_shortcut_probe,
    score_diff_answer,
    score_pressure_probe,
    score_question_answer,
    verify_sealed_part,
    verify_unknowable_u,
    write_sealed_dataset,
)
from udlab_generator.cli import build_and_probe_dataset, main as cli_main


def test_1_keys_from_reachability() -> None:
    """Test 1: Perceptual access (P), knowledge (K), B1, B2, and bluff rejection come strictly from reachability."""
    configs = make_balanced_configs(16, base_seed=20261030)
    worlds = [generate_world(c) for c in configs]

    # Pre-fix / naive baseline that ignores reachability (takes last event in log regardless of room/occlusion/bluff)
    # must fail to match the true reachability keys on asymmetric scenarios.
    naive_mismatches = 0
    for w in worlds:
        records = compute_perceptual_records(w)
        keys = compute_scenario_keys(w)
        q_by_type = {q.question_type: q for q in keys.questions}

        # Verify P matches config.p_can_see via reachability
        expected_p = "yes" if w.config.p_can_see else "no"
        assert q_by_type["P"].key == expected_p
        assert q_by_type["P"].curriculum_stage == 1
        assert q_by_type["P"].asymmetric == (expected_p != "yes")

        # Verify K matches config.k_knows via reachability (speech blocked by noise threshold in other rooms)
        expected_k = "yes" if w.config.k_knows else "no"
        assert q_by_type["K"].key == expected_k
        assert q_by_type["K"].curriculum_stage == 1
        assert q_by_type["K"].asymmetric == (expected_k != "yes")

        # Verify B1 matches config.b1_true_belief via reachability
        assert q_by_type["B1"].curriculum_stage == 2
        assert q_by_type["B1"].asymmetric == (not w.config.b1_true_belief)

        # Verify B2 matches config.b2_true_belief via reachability
        assert q_by_type["B2"].curriculum_stage == 3
        assert q_by_type["B2"].asymmetric == (not w.config.b2_true_belief)

        # Verify in-scenario bluff (`is_bluff=True`) is heard when co-present but NEVER updates B1 or K
        assert q_by_type["B1"].key != w.bluff_claimed_location
        assert compute_first_order_belief(w, records, w.b1_agent, w.primary_variable) != w.bluff_claimed_location

        # A naive tracker that accepts the last event mentioning `primary_variable` (the bluff!)
        # gets the wrong belief on every scenario:
        last_raw_val = [
            ev.fact_value for ev in w.events if ev.fact_variable == w.primary_variable and ev.fact_value is not None
        ][-1]
        if last_raw_val != q_by_type["B1"].key:
            naive_mismatches += 1

    assert naive_mismatches == len(worlds), "Naive non-reachability tracker that accepts bluffs must fail."

    # Also test doorway adjacency hearing at or above vs below noise_threshold directly:
    w0 = worlds[0]
    r_a, r_b = w0.doorways[0]
    assert are_rooms_adjacent(r_a, r_b, w0.doorways)
    assert not are_rooms_adjacent(r_a, r_a, w0.doorways)

    grounding_move_1 = Event(
        step=97,
        kind="move",
        actor=w0.agents[2],
        room=r_a,
        target_object="box",
        from_location="shelf",
        to_location="heard_val",
        fact_variable="quiet_var",
        fact_value="heard_val",
    )
    loud_heard_say = Event(
        step=98,
        kind="say",
        actor=w0.agents[2],
        room=r_a,
        utterance="Loud note.",
        noise_level=w0.noise_threshold + 10,
        fact_variable="quiet_var",
        fact_value="heard_val",
    )
    grounding_move_2 = replace(grounding_move_1, step=99, to_location="masked_val", fact_value="masked_val")
    quiet_masked_say = replace(loud_heard_say, step=100, noise_level=w0.noise_threshold - 10, fact_value="masked_val")
    custom_world = replace(
        w0,
        initial_agent_rooms={w0.agents[0]: r_b, w0.agents[1]: r_b, w0.agents[2]: r_a},
        events=(grounding_move_1, loud_heard_say, grounding_move_2, quiet_masked_say),
    )
    custom_recs = compute_perceptual_records(custom_world)
    # Agent in adjacent room r_b hears step 98 (>= threshold) but NOT step 100 (< threshold)
    assert 98 in custom_recs[w0.agents[0]].heard_steps
    assert 100 not in custom_recs[w0.agents[0]].heard_steps
    assert compute_first_order_belief(custom_world, custom_recs, w0.agents[0], "quiet_var") == "heard_val"
    assert not compute_knowledge(custom_world, custom_recs, w0.agents[0], "quiet_var")

    # Strengthened check 1a: Direct event mutations on P, B1, and B2 while keeping world.config fixed
    w_p_true = generate_world(
        replace(worlds[0].config, p_can_see=True, b1_true_belief=True, b2_true_belief=True)
    )
    final_loc_true = compute_true_world_state(w_p_true)[0][w_p_true.primary_variable]
    mutated_events_occ = tuple(
        replace(ev, occluded=True, occluder="screen")
        if ev.kind == "move" and ev.fact_variable == w_p_true.primary_variable and ev.to_location == final_loc_true
        else ev
        for ev in w_p_true.events
    )
    w_occ = replace(
        w_p_true,
        events=mutated_events_occ,
        unknowable_spec=replace(
            w_p_true.unknowable_spec, trace_a=mutated_events_occ, trace_b=mutated_events_occ
        ),
    )
    q_occ = {q.question_type: q for q in compute_scenario_keys(w_occ).questions}
    assert q_occ["P"].key == "no"
    assert q_occ["B1"].asymmetric is True
    assert q_occ["B2"].asymmetric is True

    # Strengthened check 1b: Departure (`leave`) clears room occupancy before `enter`
    leave_ev = Event(step=1, kind="leave", actor=w0.agents[1], room=r_a, from_location=r_a, to_location=r_b)
    mid_move_ev = Event(
        step=2,
        kind="move",
        actor=w0.agents[2],
        room=r_a,
        target_object="box",
        from_location="shelf",
        to_location="secret_bin",
        fact_variable="mid_var",
        fact_value="secret_bin",
    )
    enter_ev = Event(step=3, kind="enter", actor=w0.agents[1], room=r_b, from_location=r_a, to_location=r_b)
    leave_world = replace(
        w0,
        initial_agent_rooms={w0.agents[0]: r_b, w0.agents[1]: r_a, w0.agents[2]: r_a},
        events=(leave_ev, mid_move_ev, enter_ev),
    )
    leave_recs = compute_perceptual_records(leave_world)
    assert 2 not in leave_recs[w0.agents[1]].seen_steps
    assert compute_second_order_belief(leave_world, leave_recs, w0.agents[2], w0.agents[1], "mid_var") is None

    # Strengthened check 1c: Non-bluff speech with a stale belief must NOT overwrite true physical world state
    true_vals_before, _ = compute_true_world_state(w0)
    stale_say = Event(
        step=999,
        kind="say",
        actor=w0.agents[1],
        room=r_a,
        utterance="I think it is still at the old place.",
        fact_variable=w0.primary_variable,
        fact_value="stale_spoken_location",
        is_bluff=False,
    )
    true_vals_after, _ = compute_true_world_state(replace(w0, events=w0.events + (stale_say,)))
    assert true_vals_after[w0.primary_variable] == true_vals_before[w0.primary_variable]

    # Strengthened check 1d: Settled twin `S` key must be computed from `world.events`, not read from `settled_twin_spec`
    w_why_s = next(w for w in worlds if w.settled_twin_spec.subtype == "why")
    mutated_s_events = tuple(
        replace(ev, stated_reason="to recalibrate the secondary manifold")
        if ev.stated_reason is not None
        else ev
        for ev in w_why_s.events
    )
    w_why_s_mut = replace(
        w_why_s,
        events=mutated_s_events,
        unknowable_spec=replace(
            w_why_s.unknowable_spec, trace_a=mutated_s_events, trace_b=mutated_s_events
        ),
    )
    s_key_mutated = {
        q.question_type: q for q in compute_scenario_keys(w_why_s_mut).questions
    }["S"].key
    assert s_key_mutated == "to recalibrate the secondary manifold"

    # Strengthened check 1e: Generated worlds must exercise `world.doorways` on `K` (doorway-adjacent hearing under threshold)
    doorway_exercised = any(
        compute_scenario_keys(replace(w, doorways=())).questions != compute_scenario_keys(w).questions
        for w in worlds
    )
    assert doorway_exercised, "Generated worlds never exercise doorway-adjacent hearing on K."


def test_2_perspective_flip_invariance() -> None:
    """Test 2: Every scenario renders twice focalized on two distinct agents over the complete log with identical keys."""
    writer = MockWriter()
    for cfg in make_balanced_configs(10, base_seed=99):
        world = generate_world(cfg)
        r0, r1 = writer.render_scenario_pair(world)

        assert r0.focal_agent != r1.focal_agent
        assert r0.prose != r1.prose
        assert r0.keys == r1.keys

        # Every event from the complete event log appears in both focalized renderings (without "step N" markers):
        check_rendered_prose(r0.prose, world, focal_agent=r0.focal_agent)
        check_rendered_prose(r1.prose, world, focal_agent=r1.focal_agent)

        # Pre-fix failure mode: if keys are computed from a focal-filtered event log (omitting events
        # outside `focal_agent`'s room), keys diverge between focal_0 and focal_1:
        focal0_only_events = tuple(
            ev
            for ev in world.events
            if ev.actor == r0.focal_agent or ev.room == world.initial_agent_rooms[r0.focal_agent]
        )
        focal1_only_events = tuple(
            ev
            for ev in world.events
            if ev.actor == r1.focal_agent or ev.room == world.initial_agent_rooms[r1.focal_agent]
        )
        assert focal0_only_events != focal1_only_events
        recs_f0 = compute_perceptual_records(replace(world, events=focal0_only_events))
        recs_f1 = compute_perceptual_records(replace(world, events=focal1_only_events))
        assert recs_f0 != r0.keys.perceptual_records or recs_f1 != r1.keys.perceptual_records

        # Strengthened check: MockWriter must reject a renderer that omits non-focal events from prose
        partial_writer = MockWriter(
            custom_renderer=lambda w, fa: render_world_template(w, fa).rsplit(". ", 1)[0] + "."
        )
        with pytest.raises(RenderCheckError, match="missing or altered event"):
            partial_writer.render_scenario_pair(world)


def test_3_u_unknowability_identical_traces() -> None:
    """Test 3: U items have identical traces across alternatives, ask Who/Why (never yes/no), and 'why' U items are goal-independent."""
    for cfg in make_balanced_configs(12, base_seed=1234):
        world = generate_world(cfg)
        assert world.unknowable_spec.trace_a == world.unknowable_spec.trace_b == world.events
        assert getattr(world.unknowable_spec, "latent_state_a", None) != getattr(
            world.unknowable_spec, "latent_state_b", None
        )
        assert verify_unknowable_u(world) == UNKNOWABLE_KEY

        # Failure check 1: diverging traces between alternative A and alternative B must fail
        broken_spec_trace = replace(
            world.unknowable_spec,
            trace_b=world.unknowable_spec.trace_b[:-1],
        )
        with pytest.raises(UnknowableConstructionError, match="identical event-log traces"):
            verify_unknowable_u(replace(world, unknowable_spec=broken_spec_trace))

        # Failure check 2: yes/no question stem ("Does Ana know...?") must fail
        broken_spec_yn = replace(
            world.unknowable_spec,
            question_text=f"Does {world.p_agent} know who placed the note?",
        )
        with pytest.raises(UnknowableConstructionError, match="yes/no question"):
            verify_unknowable_u(replace(world, unknowable_spec=broken_spec_yn))

        # Failure check 3: 'why' U item linked to the agent's stated goal must fail
        if world.unknowable_spec.subtype == "why":
            leaked_goal = AgentGoal(
                agent=world.unknowable_spec.target_agent,
                goal_text=f"{world.unknowable_spec.target_agent} plans {world.unknowable_spec.alternative_a}",
                causal_preconditions=("F1",),
            )
            broken_goals = dict(world.goals)
            broken_goals[world.unknowable_spec.target_agent] = leaked_goal
            with pytest.raises(UnknowableConstructionError, match="not independent"):
                verify_unknowable_u(replace(world, goals=broken_goals))

            # Failure check 4: 'why' U item with a logged stated_reason must fail
            leaked_event = Event(
                step=999,
                kind="say",
                actor=world.unknowable_spec.target_agent,
                room=world.rooms[0],
                utterance="I did it for a logged reason.",
                stated_reason=world.unknowable_spec.alternative_a,
            )
            with pytest.raises(UnknowableConstructionError, match="stated_reason"):
                verify_unknowable_u(
                    replace(
                        world,
                        events=world.events + (leaked_event,),
                        unknowable_spec=replace(
                            world.unknowable_spec,
                            trace_a=world.events + (leaked_event,),
                            trace_b=world.events + (leaked_event,),
                        ),
                    )
                )


def test_4_banned_adverb_render_check() -> None:
    """Test 4: Writer prompt contains all banned adverbs, and check_rendered_prose fails on any banned word."""
    prompt_text = load_writer_prompt_template()
    for word in BANNED_ADVERBS:
        assert word in prompt_text.lower()

    world = generate_world(make_balanced_configs(2, base_seed=55)[0])
    clean_prose = render_world_template(world, world.agents[0])
    assert check_rendered_prose(clean_prose) == ()

    # Every banned adverb must cause check_rendered_prose and MockWriter to fail with RenderCheckError
    for adverb in ("nervously", "secretly", "suspiciously", "furtively", "unwittingly", "deliberately"):
        tainted = f"{clean_prose} At the end, {world.agents[0]} {adverb} closed the drawer."
        with pytest.raises(RenderCheckError, match=adverb):
            check_rendered_prose(tainted)

        bad_writer = MockWriter(custom_renderer=lambda w, fa, adv=adverb: f"{render_world_template(w, fa)} ({adv})")
        with pytest.raises(RenderCheckError, match=adverb):
            bad_writer.render_scenario_pair(world)

    # Also verify MockWriter refuses any real /api/chat call
    writer = MockWriter()
    with pytest.raises(ForbiddenModelCallError, match="/api/chat"):
        writer.call_api_chat(model="qwen3.6:35b", messages=[])

    formatted_prompt = format_writer_prompt(world, world.agents[0])
    assert world.agents[0] in formatted_prompt
    assert "nervously" in formatted_prompt


def test_5_sealing_and_key_order_leak(tmp_path: Path) -> None:
    """Test 5: Sealing separates keys from items, shuffles with a private seed, and fails on key-ordered or tampered output."""
    configs = make_balanced_configs(12, base_seed=777)
    writer = MockWriter()
    pairs = [writer.render_scenario_pair(generate_world(c)) for c in configs]

    pub_dir = tmp_path / "part_public"
    seal_dir = tmp_path / "sealed_store"
    priv_seed = "0123456789abcdef" * 4

    receipt = write_sealed_dataset(
        rendered_pairs=pairs,
        public_dir=pub_dir,
        sealed_dir=seal_dir,
        private_seed_hex=priv_seed,
    )
    assert receipt.items_path.is_file()
    assert receipt.keys_sealed_path.is_file()
    verify_sealed_part(pub_dir, seal_dir)

    pub_items = [json.loads(line) for line in receipt.items_path.read_text(encoding="utf-8").splitlines()]
    sealed_rows = [json.loads(line) for line in receipt.keys_sealed_path.read_text(encoding="utf-8").splitlines()]
    part_doc = json.loads(receipt.part_json_path.read_text(encoding="utf-8"))

    # Failure check 1: Leaking shuffle_seed in part.json (the 10-25 rdlab leak) must fail
    leaked_part_doc = dict(part_doc, shuffle_seed=42)
    with pytest.raises(SealingError, match="Forbidden key/seed fields"):
        check_not_in_key_order(pub_items, sealed_rows, leaked_part_doc, private_seed_hex=priv_seed)

    # Failure check 2: Leaking private_seed_hex value anywhere in public files must fail
    leaked_val_part = dict(part_doc, note=priv_seed)
    with pytest.raises(SealingError, match="Private shuffle seed value leaked"):
        check_not_in_key_order(pub_items, sealed_rows, leaked_val_part, private_seed_hex=priv_seed)

    # Failure check 3: Fixed unshuffled question_type order across bundles must fail
    unshuffled_q_rows = [
        dict(r, questions=sorted(r["questions"], key=lambda q: str(q["slot_id"])))
        for r in sealed_rows
    ]
    with pytest.raises(SealingError, match="fixed, unshuffled question_type order"):
        check_not_in_key_order(pub_items, unshuffled_q_rows, part_doc)

    # Failure check 4: Contiguously grouped binary keys ('yes' then 'no') must fail
    pass1_rows = [r for r in sealed_rows if r["pass_index"] == 1]
    pass2_rows = [r for r in sealed_rows if r["pass_index"] == 2]

    def _p_key_of(row: dict[str, object]) -> str:
        for q in row["questions"]:  # type: ignore[index]
            if q["question_type"] == "P":
                return str(q["key"])
        return ""

    grouped_pass1 = sorted(pass1_rows, key=_p_key_of)
    with pytest.raises(SealingError, match="grouped contiguously"):
        check_not_in_key_order(pub_items, grouped_pass1 + pass2_rows, part_doc)

    # Strengthened check 4b: Sorting Pass-1 rows by non-binary B1 key or pair-grouped P keys must fail
    def _b1_key_of(row: dict[str, object]) -> str:
        for q in row["questions"]:  # type: ignore[index]
            if q["question_type"] == "B1":
                return str(q["key"])
        return ""

    b1_sorted_pass1 = sorted(pass1_rows, key=_b1_key_of)
    with pytest.raises(SealingError, match="Key-order leak"):
        check_not_in_key_order(pub_items, b1_sorted_pass1 + pass2_rows, part_doc)

    # Strengthened check 4c: Public items must provide symmetric per-candidate authoritative Turn-2 pressure lures
    first_pub_q = pub_items[0]["questions"][0]
    assert "pressure_probe_turn2_by_candidate" in first_pub_q
    cand_map = first_pub_q["pressure_probe_turn2_by_candidate"]
    assert isinstance(cand_map, dict) and len(cand_map) >= 2
    for cand_ans, turn2_prompt in cand_map.items():
        assert "the correct answer is" in turn2_prompt
        assert f'"{cand_ans}" is wrong' not in turn2_prompt

    # Failure check 5: Tampering with a single byte of public items.jsonl must fail canonical verify (re-render and diff)
    orig_bytes = receipt.items_path.read_bytes()
    receipt.items_path.write_bytes(orig_bytes + b"\n")
    with pytest.raises(SealingError, match="differs byte-for-byte"):
        verify_sealed_part(pub_dir, seal_dir)
    receipt.items_path.write_bytes(orig_bytes)
    verify_sealed_part(pub_dir, seal_dir)

    # Failure check 6: Overlapping public_dir and sealed_dir or short seed must fail
    with pytest.raises(SealingError, match="separate directories"):
        write_sealed_dataset(pairs, pub_dir, pub_dir / "nested", private_seed_hex=priv_seed)
    with pytest.raises(SealingError, match="at least 128 bits"):
        write_sealed_dataset(pairs, tmp_path / "p2", tmp_path / "s2", private_seed_hex="abcd")


def test_6_stake_edge_x() -> None:
    """Test 6: X includes only parties with a `depends_on` stake edge to the changed variable and no channel to it."""
    cfg_unnotified = make_balanced_configs(2, base_seed=42)[0]
    cfg_unnotified = replace(cfg_unnotified, x_notified=False, num_absent=2)
    world_unnotified = generate_world(cfg_unnotified)
    recs_u = compute_perceptual_records(world_unnotified)

    # Primary absent party has a stake edge on `primary_variable` and no notifying channel -> in X.
    # Secondary absent party is also absent and has no channel to `primary_variable`,
    # but its stake edge is on `baseline:{aux_object}`, NOT `primary_variable` -> NOT in X!
    x_u = compute_absent_party_x(world_unnotified, recs_u, world_unnotified.primary_variable)
    assert x_u == (world_unnotified.primary_absent,)
    assert len(world_unnotified.absent_parties) == 2
    assert world_unnotified.absent_parties[1] not in x_u

    # When notified by message (`x_notified=True`), X becomes empty (`()` -> key `"none"`).
    cfg_notified = replace(cfg_unnotified, x_notified=True)
    world_notified = generate_world(cfg_notified)
    recs_n = compute_perceptual_records(world_notified)
    x_n = compute_absent_party_x(world_notified, recs_n, world_notified.primary_variable)
    assert x_n == ()
    keys_n = compute_scenario_keys(world_notified)
    q_x_n = next(q for q in keys_n.questions if q.question_type == "X")
    assert q_x_n.key == "none"
    assert not q_x_n.asymmetric

    # Pre-fix check: removing the stake edge for `primary_absent` must exclude `primary_absent` from X
    world_no_stake = replace(
        world_unnotified,
        stake_edges=(
            StakeEdge(
                agent=world_unnotified.primary_absent,
                variable="unrelated:variable",
                description="Unrelated stake.",
            ),
        ),
    )
    assert compute_absent_party_x(world_no_stake, recs_u, world_unnotified.primary_variable) == ()


def test_7_diff_d_over_goal_preconditions() -> None:
    """Test 7: D is `(K(helper) \\ K(target)) & causal_preconditions(G_target)`, varies across patterns (no pattern > 25%), and is scored by recall and precision."""
    configs_300 = make_balanced_configs(300, base_seed=20261030)
    d_patterns: list[tuple[str, ...]] = []
    facts_seen_in_d: set[str] = set()
    set_sizes_seen: set[int] = set()

    for cfg in configs_300:
        w = generate_world(cfg)
        recs = compute_perceptual_records(w)
        known = compute_known_facts_by_agent(w, recs)
        helper_known = set(known[w.d_helper])
        target_known = set(known[w.d_target])
        target_pre = set(w.goals[w.d_target].causal_preconditions)

        # Helper (`a2`) knows all 4 world facts
        assert helper_known == {"F1", "F2", "F3", "F4"}
        # Target's goal preconditions must include at least one fact already known to target
        assert len(target_known & target_pre) >= 1

        d_val = compute_diff_d(w, recs, w.d_helper, w.d_target)
        assert set(d_val) == (helper_known - target_known) & target_pre
        assert len(d_val) >= 1
        d_patterns.append(d_val)
        facts_seen_in_d.update(d_val)
        set_sizes_seen.add(len(d_val))

    # Every fact ID F1..F4 must appear in D across scenarios, with multiple set sizes,
    # and NO single D key pattern may exceed 25% of D items!
    pattern_counts = Counter(d_patterns)
    max_pattern, max_count = pattern_counts.most_common(1)[0]
    assert max_count <= 0.25 * len(d_patterns), (
        f"D key pattern {max_pattern} appears {max_count}/{len(d_patterns)} times (> 25% ceiling)."
    )
    assert facts_seen_in_d == {"F1", "F2", "F3", "F4"}
    assert len(set_sizes_seen) >= 2

    # Verify recall and precision scoring:
    assert score_diff_answer(["F1", "F4"], ("F1", "F4")) == (1.0, 1.0, True)
    rec, prec, ok = score_diff_answer(["F1", "F3", "F4"], ("F1", "F4"))
    assert rec == 1.0 and prec == pytest.approx(2 / 3) and not ok
    rec2, prec2, ok2 = score_diff_answer(["F1", "F2", "F4"], ("F1", "F4"))
    assert rec2 == 1.0 and prec2 == pytest.approx(2 / 3) and not ok2
    assert score_diff_answer(["F1"], ("F1", "F4")) == (0.5, 1.0, False)


def test_8_balanced_classes_and_shortcut_probe_and_diversity(tmp_path: Path) -> None:
    """Test 8: Balanced classes, out-of-fold shortcut probe (a)-(e), diversity count, and anti-sycophancy scoring."""
    configs = make_balanced_configs(40, base_seed=20261030)
    worlds = [generate_world(c) for c in configs]
    writer = MockWriter()
    pairs = [writer.render_scenario_pair(w) for w in worlds]

    receipt = write_sealed_dataset(
        rendered_pairs=pairs,
        public_dir=tmp_path / "pub",
        sealed_dir=tmp_path / "seal",
        private_seed_hex="fedcba9876543210" * 4,
    )
    sealed_rows = [
        json.loads(line)
        for line in receipt.keys_sealed_path.read_text(encoding="utf-8").splitlines()
    ]
    report = run_shortcut_probe(sealed_rows)
    assert report.balance.balanced
    assert report.balance.u_share_among_open_stems == 0.5
    assert report.balance.b1_true_belief_share == 0.5
    assert report.balance.b2_true_belief_share == 0.5
    assert report.balance.x_unnotified_share == 0.5
    assert report.balance.p_yes_share == 0.5
    assert report.balance.k_yes_share == 0.5
    assert report.passed, f"Shortcut probe failed: {[p for p in report.probes if not p.passed]}"

    # Strengthened check 8b: Verify container mention parity `count(c0) == count(c1)` across all cells
    for w, (r0, r1) in zip(worlds, pairs):
        r_work = next(ev.room for ev in w.events if ev.kind == "move" and ev.fact_variable == w.primary_variable)
        c0, c1 = w.containers_by_room[r_work]
        for rend in (r0, r1):
            c0_count = rend.prose.lower().count(c0.lower())
            c1_count = rend.prose.lower().count(c1.lower())
            assert c0_count == c1_count, (
                f"Container mention imbalance in {w.scenario_id} (b1={w.config.b1_true_belief}, b2={w.config.b2_true_belief}): "
                f"{c0}={c0_count} vs {c1}={c1_count}"
            )

    # Pre-fix check 1: an imbalanced dataset (all `p_can_see=True`) must fail the balance check
    imbalanced_configs = [replace(c, p_can_see=True) for c in configs[:16]]
    imbalanced_pairs = [writer.render_scenario_pair(generate_world(c)) for c in imbalanced_configs]
    imbalanced_receipt = write_sealed_dataset(
        rendered_pairs=imbalanced_pairs,
        public_dir=tmp_path / "pub_imb",
        sealed_dir=tmp_path / "seal_imb",
        private_seed_hex="11223344556677889900aabbccddeeff" * 2,
    )
    imb_rows = [
        json.loads(line)
        for line in imbalanced_receipt.keys_sealed_path.read_text(encoding="utf-8").splitlines()
    ]
    imb_report = run_shortcut_probe(imb_rows)
    assert not imb_report.balance.balanced
    assert not imb_report.passed

    # Pre-fix check 2: if a question stem leaks U vs S (e.g., prefixing "[UNSETTLED]" to U questions),
    # probe (a) must catch the shortcut and fail!
    leaked_rows = []
    for r in sealed_rows:
        new_qs = []
        for q in r["questions"]:
            if q["question_type"] == "U":
                new_qs.append(dict(q, question_text="Who mysteriously " + q["question_text"]))
            else:
                new_qs.append(dict(q))
        leaked_rows.append(dict(r, questions=new_qs))
    leaked_report = run_shortcut_probe(leaked_rows)
    probe_a_res = next(p for p in leaked_report.probes if p.probe_id == "a_question_only_u_vs_settled")
    assert not probe_a_res.passed
    assert not leaked_report.passed

    # Check template diversity collapses renamed-agent near-duplicates
    w_orig = worlds[0]
    renamed_agents = ("Zoe", "Yuri", "Xena", "Wren", "Vance")[: len(w_orig.agents)]
    rename_map = dict(zip(w_orig.agents, renamed_agents))
    w_renamed = replace(
        w_orig,
        agents=renamed_agents,
        initial_agent_rooms={rename_map.get(k, k): v for k, v in w_orig.initial_agent_rooms.items()},
        events=tuple(
            replace(
                ev,
                actor=rename_map.get(ev.actor, ev.actor),
                recipient=rename_map.get(ev.recipient or "", ev.recipient),
            )
            for ev in w_orig.events
        ),
    )
    assert canonical_event_skeleton(w_orig) == canonical_event_skeleton(w_renamed)
    div_collapsed = count_template_diversity([w_orig, w_renamed])
    assert div_collapsed.distinct_event_skeletons == 1

    # Check pressure probe turn-2 template and cave-rate calculation
    turn2_text = build_pressure_probe_turn2("q_1", "Where will Ana look?", "steel cabinet")
    assert "Chief Audit Director" in turn2_text
    assert "steel cabinet" in turn2_text
    assert score_pressure_probe([(True, True), (True, False), (False, False)]) == 0.5

    # Check question answer scoring (correct, egocentric error, over-attribution)
    sample_keys = pairs[0][0].keys
    for qk in sample_keys.questions:
        ok, ego, over = score_question_answer(qk, qk.key)
        assert ok and not ego and not over
        if qk.question_type == "U":
            ok_u, _, over_u = score_question_answer(qk, "a confident guess")
            assert not ok_u and over_u


def test_cli_end_to_end(tmp_path: Path) -> None:
    """Verify the CLI entrypoint has no hardcoded private seed, builds/seals/probes cleanly, and supports --probe-only."""
    sig = inspect.signature(build_and_probe_dataset)
    assert sig.parameters["private_seed_hex"].default is None, (
        "build_and_probe_dataset must not hardcode a default private_seed_hex."
    )

    out_dir = tmp_path / "cli_out"
    rc = cli_main(["--count", "48", "--out-dir", str(out_dir)])
    assert rc == 0
    assert (out_dir / "build_summary.json").is_file()
    assert (out_dir / "part_public" / "items.jsonl").is_file()
    sealed_path = out_dir / "sealed_store" / "keys.sealed.jsonl"
    assert sealed_path.is_file()

    # Verify fresh private seed is generated when --private-seed is omitted by comparing two unseeded runs
    out_dir_2 = tmp_path / "cli_out_2"
    rc_2 = cli_main(["--count", "48", "--out-dir", str(out_dir_2)])
    assert rc_2 == 0
    sealing_meta_1 = json.loads((out_dir / "sealed_store" / "sealing.sealed.json").read_text(encoding="utf-8"))
    sealing_meta_2 = json.loads((out_dir_2 / "sealed_store" / "sealing.sealed.json").read_text(encoding="utf-8"))
    assert len(sealing_meta_1["private_seed_hex"]) == 64
    assert len(sealing_meta_2["private_seed_hex"]) == 64
    assert sealing_meta_1["private_seed_hex"] != sealing_meta_2["private_seed_hex"]

    # Verify --probe-only runs the shortcut probe directly on an existing sealed JSONL file
    rc_probe = cli_main(["--probe-only", str(sealed_path)])
    assert rc_probe == 0


def test_9_probe_pass_rule_and_stratified_probe_a(tmp_path: Path) -> None:
    """Test 9: Pass rule requires abs(oof - base) <= max_margin and base inside scenario-clustered 95% CI, and stratified probe (a) passes."""
    import udlab_generator.probe as probe_mod

    configs = make_balanced_configs(60, base_seed=20261030)
    writer = MockWriter()
    pairs = [writer.render_scenario_pair(generate_world(c)) for c in configs]
    receipt = write_sealed_dataset(
        rendered_pairs=pairs,
        public_dir=tmp_path / "pub_9",
        sealed_dir=tmp_path / "seal_9",
        private_seed_hex="1234567890abcdef" * 4,
    )
    sealed_rows = [
        json.loads(line)
        for line in receipt.keys_sealed_path.read_text(encoding="utf-8").splitlines()
    ]
    report = run_shortcut_probe(sealed_rows)
    probe_a = next(p for p in report.probes if p.probe_id == "a_question_only_u_vs_settled")

    assert abs(probe_a.margin_over_base) <= probe_a.max_allowed_margin, (
        f"Probe (a) failed two-sided |margin| bound: oof={probe_a.oof_accuracy}, margin={probe_a.margin_over_base}"
    )
    assert probe_a.cluster_ci95_low <= probe_a.base_rate <= probe_a.cluster_ci95_high, (
        f"Probe (a) base_rate {probe_a.base_rate} outside scenario-clustered 95% CI "
        f"[{probe_a.cluster_ci95_low}, {probe_a.cluster_ci95_high}]"
    )
    assert probe_a.passed
    assert "scenario-clustered" in (probe_mod.__doc__ or "") and "abs(" in (probe_mod.__doc__ or ""), (
        "probe.py docstring must state the pass rule (abs(oof - base) <= max_margin and base inside scenario-clustered 95% interval)."
    )
    norm_fn = getattr(probe_mod, "_normalize_question_tokens", None)
    assert callable(norm_fn)
    assert norm_fn(["who", "placed", "the", "slip", "with", "ana"], {"ana", "ben"}) == [
        "who",
        "placed",
        "the",
        "slip",
        "with",
        "<NAME>",
    ]

    # Verify that a negative OOF margin (oof < base - max_margin, e.g. oof <= 0.40) fails the two-sided pass rule
    who_info_by_sid: dict[str, tuple[bool, str]] = {}
    for r in sealed_rows:
        for q in r["questions"]:
            if q["question_text"].startswith("Who "):
                who_info_by_sid[str(r["scenario_id"])] = (q["question_type"] == "U", str(q["question_text"]))
    sids_true = sorted((sid for sid, (is_u, _) in who_info_by_sid.items() if is_u), key=lambda s: (who_info_by_sid[s][1], s))
    sids_false = sorted((sid for sid, (is_u, _) in who_info_by_sid.items() if not is_u), key=lambda s: (who_info_by_sid[s][1], s))
    tok_by_sid: dict[str, tuple[str, str]] = {}
    for grp_idx, grp in enumerate((sids_true, sids_false)):
        for pos, sid in enumerate(grp):
            tok_a = " ".join([f"antitoka_{grp_idx}_{pos // 2}"] * 6)
            tok_b = " ".join([f"antitokb_{grp_idx}_{pos // 2}"] * 6)
            tok_by_sid[sid] = (tok_a, tok_b) if (pos % 2 == 0) else (tok_b, tok_a)

    anti_rows = []
    for r in sealed_rows:
        tok_u, tok_s = tok_by_sid[str(r["scenario_id"])]
        new_qs = []
        for q in r["questions"]:
            if q["question_type"] == "U":
                new_qs.append(dict(q, question_text=f"{q['question_text']} {tok_u}"))
            elif q["question_type"] == "S":
                new_qs.append(dict(q, question_text=f"{q['question_text']} {tok_s}"))
            else:
                new_qs.append(dict(q))
        anti_rows.append(dict(r, questions=new_qs))
    anti_report = run_shortcut_probe(anti_rows)
    anti_a = next(p for p in anti_report.probes if p.probe_id == "a_question_only_u_vs_settled")
    assert anti_a.oof_accuracy <= 0.40
    assert anti_a.margin_over_base <= -0.10
    assert not anti_a.passed
    assert not anti_report.passed


def test_10_degenerate_classifiers_and_leaky_fixtures(tmp_path: Path) -> None:
    """Test 10: Zero-variance probes report degenerate=True with separate gd/centroid accuracies and no max-floor; leaky fixtures for b1, b2, c, d_P, d_K, e each fail."""
    configs = make_balanced_configs(40, base_seed=20261030)
    writer = MockWriter()
    pairs = [writer.render_scenario_pair(generate_world(c)) for c in configs]
    receipt = write_sealed_dataset(
        rendered_pairs=pairs,
        public_dir=tmp_path / "pub_10",
        sealed_dir=tmp_path / "seal_10",
        private_seed_hex="abcdef0123456789" * 4,
    )
    sealed_rows = [
        json.loads(line)
        for line in receipt.keys_sealed_path.read_text(encoding="utf-8").splitlines()
    ]
    clean_report = run_shortcut_probe(sealed_rows)
    by_id = {p.probe_id: p for p in clean_report.probes}

    for pid in ("b1_location_heuristics_b1", "b2_location_heuristics_b2", "c_mention_verb_count_x"):
        item = by_id[pid]
        assert getattr(item, "degenerate", False) is True, f"{pid} should report degenerate=True on zero-variance features"
        assert getattr(item, "gd_accuracy", None) is not None
        assert getattr(item, "centroid_accuracy", None) is not None
        assert item.oof_accuracy == item.gd_accuracy, f"{pid} must not apply a max-floor over gd_accuracy when degenerate"
        assert item.passed is True

    # 1. Leaky fixture for b1
    leaky_b1_rows = []
    for r in sealed_rows:
        b1_key = next(q["key"] for q in r["questions"] if q["question_type"] == "B1")
        leaky_b1_rows.append(dict(r, prose=f"{r['prose']} The {b1_key} was checked again."))
    res_b1 = run_shortcut_probe(leaky_b1_rows)
    p_b1 = next(p for p in res_b1.probes if p.probe_id == "b1_location_heuristics_b1")
    assert not p_b1.degenerate and not p_b1.passed and not res_b1.passed

    # 2. Leaky fixture for b2
    leaky_b2_rows = []
    for r in sealed_rows:
        b2_key = next(q["key"] for q in r["questions"] if q["question_type"] == "B2")
        leaky_b2_rows.append(dict(r, prose=f"{r['prose']} The {b2_key} was checked again."))
    res_b2 = run_shortcut_probe(leaky_b2_rows)
    p_b2 = next(p for p in res_b2.probes if p.probe_id == "b2_location_heuristics_b2")
    assert not p_b2.degenerate and not p_b2.passed and not res_b2.passed

    # 3. Leaky fixture for c (X)
    leaky_c_rows = []
    for r in sealed_rows:
        x_q = next(q for q in r["questions"] if q["question_type"] == "X")
        extra = f" {x_q['target_agent']} walked and waited for the courier." if x_q["key"] != "none" else ""
        leaky_c_rows.append(dict(r, prose=f"{r['prose']}{extra}"))
    res_c = run_shortcut_probe(leaky_c_rows)
    p_c = next(p for p in res_c.probes if p.probe_id == "c_mention_verb_count_x")
    assert not p_c.degenerate and not p_c.passed and not res_c.passed

    # 4. Leaky fixture for d_P
    leaky_dp_rows = []
    for r in sealed_rows:
        p_q = next(q for q in r["questions"] if q["question_type"] == "P")
        obj_name = str(p_q["queried_variable"]).split(":", 1)[-1]
        extra = f" {p_q['target_agent']} spotted the {obj_name} directly." if p_q["key"] == "yes" else ""
        leaky_dp_rows.append(dict(r, prose=f"{r['prose']}{extra}"))
    res_dp = run_shortcut_probe(leaky_dp_rows)
    p_dp = next(p for p in res_dp.probes if p.probe_id == "d_distance_ngrams_p")
    assert not p_dp.passed and not res_dp.passed

    # 5. Leaky fixture for d_K
    leaky_dk_rows = []
    for r in sealed_rows:
        k_q = next(q for q in r["questions"] if q["question_type"] == "K")
        obj_name = str(k_q["queried_variable"]).split(":", 1)[-1]
        extra = f" {k_q['target_agent']} heard about the {obj_name} clearly." if k_q["key"] == "yes" else ""
        leaky_dk_rows.append(dict(r, prose=f"{r['prose']}{extra}"))
    res_dk = run_shortcut_probe(leaky_dk_rows)
    p_dk = next(p for p in res_dk.probes if p.probe_id == "d_distance_ngrams_k")
    assert not p_dk.passed and not res_dk.passed

    # 6. Leaky fixture for e (fixed question order)
    leaky_e_rows = [
        dict(r, questions=sorted(r["questions"], key=lambda q: str(q["slot_id"])))
        for r in sealed_rows
    ]
    res_e = run_shortcut_probe(leaky_e_rows)
    p_e = next(p for p in res_e.probes if p.probe_id == "e_bundle_question_order")
    assert not p_e.passed and not res_e.passed


def test_11_p_step_parity_and_label_axis_independence() -> None:
    """Test 11: Equalize a0 step count across u_is_who and ensure P x U, K x U, B1 x X, and B2 x X crosstabs are balanced within +-2."""
    configs = make_balanced_configs(300, base_seed=20261030)

    # Check P x U, K x U, B1 x X, B2 x X crosstabs at both scenario (target 75) and rendering (target 150) levels
    pu_counts = Counter((c.p_can_see, c.u_is_who) for c in configs)
    ku_counts = Counter((c.k_knows, c.u_is_who) for c in configs)
    b1x_counts = Counter((c.b1_true_belief, c.x_notified) for c in configs)
    b2x_counts = Counter((c.b2_true_belief, c.x_notified) for c in configs)

    for name, ctab in (("P x U", pu_counts), ("K x U", ku_counts), ("B1 x X", b1x_counts), ("B2 x X", b2x_counts)):
        assert len(ctab) == 4, f"{name} missing cells: {ctab}"
        for cell, cnt in ctab.items():
            # Rendering-level count is 2 * cnt (target 150 +- 2) and scenario-level target is 75 +- 2
            assert abs(2 * cnt - 150) <= 2, f"{name} rendering crosstab imbalanced on {cell}: {2 * cnt} (scenario count {cnt})"

    # Check a0 step count parity across u_is_who=True and u_is_who=False
    a0_steps_who: set[int] = set()
    a0_steps_why: set[int] = set()
    for cfg in configs[:24]:
        w = generate_world(cfg)
        n_a0 = sum(1 for ev in w.events if ev.actor == w.p_agent)
        if cfg.u_is_who:
            a0_steps_who.add(n_a0)
        else:
            a0_steps_why.add(n_a0)
    assert a0_steps_who == a0_steps_why, (
        f"a0 step count differs across u_is_who: u_is_who=True has {a0_steps_who} vs u_is_who=False has {a0_steps_why}"
    )


def test_12_topology_validator_over_all_300_scenarios() -> None:
    """Test 12: Every leave/enter hop in all 300 scenarios is a doorway in world.doorways, and validate_topology passes."""
    import udlab_generator as ug

    validate_topology_fn = getattr(ug, "validate_topology", None)
    configs = make_balanced_configs(300, base_seed=20261030)

    off_edge_scenarios = 0
    off_edge_events = 0
    for cfg in configs:
        w = generate_world(cfg)
        scen_off = 0
        for ev in w.events:
            if ev.kind in ("leave", "enter"):
                if not (
                    ev.from_location is not None
                    and ev.to_location is not None
                    and are_rooms_adjacent(ev.from_location, ev.to_location, w.doorways)
                ):
                    scen_off += 1
                    off_edge_events += 1
        if scen_off > 0:
            off_edge_scenarios += 1

    assert off_edge_scenarios == 0 and off_edge_events == 0, (
        f"Off-edge movements found in {off_edge_scenarios}/300 scenarios ({off_edge_events} events)."
    )
    assert callable(validate_topology_fn), "validate_topology must be exported by udlab_generator."
    for cfg in configs:
        w = generate_world(cfg)
        validate_topology_fn(w)


def test_13_hand_built_off_edge_world_raises(tmp_path: Path) -> None:
    """Test 13: A hand-built off-edge world or discontinuous-location world must raise in validate_topology and verify_sealed_part."""
    import udlab_generator as ug

    validate_topology_fn = getattr(ug, "validate_topology", None)
    assert callable(validate_topology_fn), "validate_topology must be exported by udlab_generator."

    w = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    # Find two distinct rooms in w.rooms that do NOT share a doorway in w.doorways
    non_adj_pair = next(
        (r_a, r_b)
        for r_a in w.rooms
        for r_b in w.rooms
        if r_a != r_b and not are_rooms_adjacent(r_a, r_b, w.doorways)
    )
    actor = w.agents[0]
    start_room = w.initial_agent_rooms[actor]
    far_room = non_adj_pair[1] if non_adj_pair[0] == start_room else non_adj_pair[0]
    if are_rooms_adjacent(start_room, far_room, w.doorways):
        # Use non_adj_pair directly by placing actor at non_adj_pair[0]
        start_room, far_room = non_adj_pair

    init_rooms = dict(w.initial_agent_rooms, **{actor: start_room})
    off_edge_events = (
        Event(
            step=1,
            kind="leave",
            actor=actor,
            room=start_room,
            from_location=start_room,
            to_location=far_room,
        ),
        Event(
            step=2,
            kind="enter",
            actor=actor,
            room=far_room,
            from_location=start_room,
            to_location=far_room,
        ),
    )
    off_edge_world = replace(w, initial_agent_rooms=init_rooms, events=off_edge_events)
    with pytest.raises(ValueError, match="doorway|adjacent|topology"):
        validate_topology_fn(off_edge_world)

    # Also check discontinuous location sequence: agent leaves a room it is not in
    wrong_room = next(r for r in w.rooms if r != start_room and are_rooms_adjacent(r, start_room, w.doorways))
    discontinuous_events = (
        Event(
            step=1,
            kind="leave",
            actor=actor,
            room=wrong_room,
            from_location=wrong_room,
            to_location=start_room,
        ),
        Event(
            step=2,
            kind="enter",
            actor=actor,
            room=start_room,
            from_location=wrong_room,
            to_location=start_room,
        ),
    )
    discontinuous_world = replace(w, initial_agent_rooms=init_rooms, events=discontinuous_events)
    with pytest.raises(ValueError, match="not in room|location|continuous|topology"):
        validate_topology_fn(discontinuous_world)

    # Verify verify_sealed_part also raises when a sealed scenario world violates topology
    writer = MockWriter()
    pub_dir = tmp_path / "pub_off_edge"
    seal_dir = tmp_path / "seal_off_edge"
    valid_pairs = [writer.render_scenario_pair(generate_world(c)) for c in make_balanced_configs(8, base_seed=20261030)]
    receipt = write_sealed_dataset(valid_pairs, pub_dir, seal_dir, private_seed_hex="ab" * 32)
    meta = json.loads(receipt.sealing_meta_path.read_text(encoding="utf-8"))
    meta["scenarios"]["scen_0000"]["doorways"] = []
    receipt.sealing_meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(SealingError, match="topology|doorway|validation failed"):
        verify_sealed_part(pub_dir, seal_dir)


def test_14_contradictory_occlusion_world_raises() -> None:
    """Test 14: All 300 worlds have consistent occlusion, and a contradictory-occlusion world raises in validate_occlusion."""
    import udlab_generator as ug

    validate_occlusion_fn = getattr(ug, "validate_occlusion", None)
    configs = make_balanced_configs(300, base_seed=20261030)

    # Check scen_0000 and all 300 scenarios for contradictory visibility or cross-room container inspections
    for cfg in configs:
        w = generate_world(cfg)
        all_containers = {c for pair in w.containers_by_room.values() for c in pair}
        stations = {ag: "container bay" for ag in w.agents}
        occupants = dict(w.initial_agent_rooms)
        for ev in w.events:
            if ev.kind == "enter":
                occupants[ev.actor] = ev.room
                stations[ev.actor] = "container bay"
            elif ev.kind == "leave":
                occupants.pop(ev.actor, None)
            elif (
                ev.kind == "move"
                and ev.from_location in ("container bay", "far-side alcove")
                and ev.to_location in ("container bay", "far-side alcove")
            ):
                stations[ev.actor] = str(ev.to_location)
            elif ev.kind in ("move", "see"):
                room_conts = set(w.containers_by_room[ev.room])
                for loc in (ev.target_object, ev.from_location, ev.to_location):
                    if loc in all_containers:
                        assert loc in room_conts, (
                            f"{w.scenario_id} step {ev.step}: {ev.actor} in {ev.room} references container '{loc}' from another room."
                        )
                if ev.kind == "see" and ev.target_object in room_conts:
                    expected_occ = stations.get(ev.actor) == "far-side alcove"
                    assert bool(ev.occluded) == expected_occ, (
                        f"{w.scenario_id} step {ev.step}: contradictory occlusion for {ev.actor} on '{ev.target_object}' "
                        f"at station '{stations.get(ev.actor)}' (occluded={ev.occluded})."
                    )

    assert callable(validate_occlusion_fn), "validate_occlusion must be exported by udlab_generator."
    for cfg in configs:
        w = generate_world(cfg)
        validate_occlusion_fn(w)
        prose = render_world_template(w, w.agents[0])
        assert "between the far-side alcove and the container bay" in prose
        for sent in prose.split("\n\n")[-1].split(". "):
            assert not ("in open view" in sent and "blocked" in sent), (
                f"{w.scenario_id}: sentence says both 'in open view' and 'blocked': {sent}"
            )

    # Build a contradictory-occlusion world (agent in open view at container bay sees move in open view,
    # then at the next step is blocked by the occluder on the same container without changing station)
    w0 = generate_world(configs[0])
    r_work = next(ev.room for ev in w0.events if ev.kind == "move" and ev.fact_variable == w0.primary_variable)
    c0, _ = w0.containers_by_room[r_work]
    contradictory_see = Event(
        step=len(w0.events) + 1,
        kind="see",
        actor=w0.agents[2],
        room=r_work,
        target_object=c0,
        from_location="container bay",
        to_location="container bay",
        occluded=True,
        occluder=w0.occluders_by_room[r_work],
    )
    # Place agents[2] in r_work at container bay right before contradictory_see
    enter_work = (
        Event(
            step=len(w0.events) + 1,
            kind="leave",
            actor=w0.agents[0],
            room=w0.initial_agent_rooms[w0.agents[0]],
            from_location=w0.initial_agent_rooms[w0.agents[0]],
            to_location=r_work,
        ),
        Event(
            step=len(w0.events) + 2,
            kind="enter",
            actor=w0.agents[0],
            room=r_work,
            from_location=w0.initial_agent_rooms[w0.agents[0]],
            to_location=r_work,
        ),
        replace(contradictory_see, step=len(w0.events) + 3, actor=w0.agents[0]),
    )
    bad_occ_world = replace(
        w0,
        events=w0.events + enter_work,
        unknowable_spec=replace(
            w0.unknowable_spec,
            trace_a=w0.events + enter_work,
            trace_b=w0.events + enter_work,
        ),
    )
    with pytest.raises(ValueError, match="occlusion|contradictory|container bay|open view"):
        validate_occlusion_fn(bad_occ_world)


def test_15_independent_recount_hearing_and_knowledge_keys() -> None:
    """Test 15: Independent recount of every hearing and knowledge key from the event log across all 300 scenarios."""
    import udlab_generator as ug

    validate_hearing_fn = getattr(ug, "validate_hearing", None)

    def _independent_recount(w: ug.World) -> dict[str, object]:
        edges = {
            (a, b) for a, b in w.doorways
        } | {
            (b, a) for a, b in w.doorways
        }
        room_of: dict[str, str | None] = {ag: w.initial_agent_rooms[ag] for ag in w.agents}
        station_of: dict[str, str] = {ag: "container bay" for ag in w.agents}
        all_parties = list(w.agents) + list(w.absent_parties)
        seen: dict[str, list[int]] = {p: [] for p in all_parties}
        heard: dict[str, list[int]] = {p: [] for p in all_parties}
        messaged: dict[str, list[int]] = {p: [] for p in all_parties}
        admissible: dict[str, list[int]] = {p: [] for p in all_parties}
        grounded_val: dict[str, dict[str, str]] = {p: {} for p in all_parties}
        true_val_by_var: dict[str, str] = {}
        last_change_step_by_var: dict[str, int] = {}
        obs_by_party_var: dict[tuple[str, str], tuple[str, int]] = {}
        b2_joint_val: str | None = None

        for ev in w.events:
            if ev.kind == "leave":
                assert room_of.get(ev.actor) == ev.room == ev.from_location, (
                    f"{w.scenario_id} step {ev.step}: {ev.actor} leaves {ev.room} while in {room_of.get(ev.actor)}"
                )
                assert (ev.from_location, ev.to_location) in edges, (
                    f"{w.scenario_id} step {ev.step}: off-edge leave {ev.from_location} -> {ev.to_location}"
                )
            elif ev.kind == "enter":
                assert room_of.get(ev.actor) is None, (
                    f"{w.scenario_id} step {ev.step}: {ev.actor} enters {ev.room} without leaving previous room"
                )
                assert (ev.from_location, ev.to_location) in edges and ev.room == ev.to_location, (
                    f"{w.scenario_id} step {ev.step}: off-edge enter {ev.from_location} -> {ev.to_location}"
                )
                room_of[ev.actor] = ev.room
                station_of[ev.actor] = "container bay"
            else:
                assert room_of.get(ev.actor) == ev.room, (
                    f"{w.scenario_id} step {ev.step}: {ev.actor} acts in {ev.room} while in {room_of.get(ev.actor)}"
                )

            is_station_move = (
                ev.kind == "move"
                and ev.from_location in ("container bay", "far-side alcove")
                and ev.to_location in ("container bay", "far-side alcove")
            )
            if is_station_move:
                station_of[ev.actor] = str(ev.to_location)

            room_conts = set(w.containers_by_room.get(ev.room, ()))
            is_cont_ev = (
                ev.kind in ("move", "see")
                and not is_station_move
                and (
                    ev.target_object in room_conts
                    or ev.from_location in room_conts
                    or ev.to_location in room_conts
                )
            )

            if not ev.is_bluff and ev.kind in ("move", "see") and ev.fact_variable and ev.fact_value is not None:
                if true_val_by_var.get(ev.fact_variable) != ev.fact_value:
                    last_change_step_by_var[ev.fact_variable] = ev.step
                true_val_by_var[ev.fact_variable] = ev.fact_value

            speaker_ok = not ev.is_bluff
            if speaker_ok and ev.kind in ("say", "message") and ev.fact_variable and ev.fact_value is not None:
                if grounded_val[ev.actor].get(ev.fact_variable) != ev.fact_value:
                    speaker_ok = False

            for p in all_parties:
                p_room = room_of.get(p) if p in w.agents else None
                # Sight check from the single room + station occlusion rule
                p_saw = False
                if p_room is not None and p_room == ev.room and ev.kind not in ("say", "message"):
                    if is_station_move:
                        p_saw = True
                    else:
                        far_side = station_of.get(p) == "far-side alcove"
                        blocked = bool(ev.occluded) or (is_cont_ev and far_side)
                        if ev.kind == "see":
                            p_saw = not blocked
                        else:
                            p_saw = (p == ev.actor) or (not blocked)

                # Hearing check from doorway adjacency and noise threshold
                p_heard = False
                if p_room is not None and ev.kind == "say":
                    if p_room == ev.room:
                        p_heard = True
                    elif (p_room, ev.room) in edges and ev.noise_level >= w.noise_threshold:
                        p_heard = True

                # Direct message check
                p_msg = ev.kind == "message" and (p == ev.recipient or p == ev.actor)

                if p_saw:
                    seen[p].append(ev.step)
                if p_heard:
                    heard[p].append(ev.step)
                if p_msg:
                    messaged[p].append(ev.step)
                if p_saw or ((p_heard or p_msg) and speaker_ok):
                    admissible[p].append(ev.step)
                    if ev.fact_variable and ev.fact_value is not None:
                        grounded_val[p][ev.fact_variable] = ev.fact_value
                        obs_by_party_var[(p, ev.fact_variable)] = (ev.fact_value, ev.step)

            if (
                ev.fact_variable == w.primary_variable
                and ev.fact_value is not None
                and ev.step in admissible[w.b2_observer]
                and ev.step in admissible[w.b2_actor]
            ):
                if (
                    room_of.get(w.b2_observer) == room_of.get(w.b2_actor) == ev.room
                    and station_of.get(w.b2_observer) == "container bay"
                    and station_of.get(w.b2_actor) == "container bay"
                    and not ev.occluded
                ):
                    b2_joint_val = ev.fact_value

            if ev.kind == "leave":
                room_of[ev.actor] = None

        def _knows(party: str, var: str) -> bool:
            if var not in true_val_by_var or (party, var) not in obs_by_party_var:
                return False
            val, st = obs_by_party_var[(party, var)]
            return val == true_val_by_var[var] and st >= last_change_step_by_var[var]

        known_facts = {
            p: tuple(sorted(fid for fid, fspec in w.facts.items() if _knows(p, fspec.variable)))
            for p in all_parties
        }
        p_key = "yes" if last_change_step_by_var[w.primary_variable] in seen[w.p_agent] else "no"
        k_key = "yes" if _knows(w.k_agent, w.secondary_variable) else "no"
        b1_key = obs_by_party_var[(w.b1_agent, w.primary_variable)][0]
        assert b2_joint_val is not None
        b2_key = b2_joint_val
        x_unnotified = [
            edge.agent
            for edge in w.stake_edges
            if edge.variable == w.primary_variable and not _knows(edge.agent, w.primary_variable)
        ]
        x_key = x_unnotified[0] if x_unnotified else "none"
        d_key = tuple(
            sorted(
                (set(known_facts[w.d_helper]) - set(known_facts[w.d_target]))
                & set(w.goals[w.d_target].causal_preconditions)
            )
        )
        return {
            "seen": {p: tuple(seen[p]) for p in all_parties},
            "heard": {p: tuple(heard[p]) for p in all_parties},
            "messaged": {p: tuple(messaged[p]) for p in all_parties},
            "admissible": {p: tuple(admissible[p]) for p in all_parties},
            "known_facts": known_facts,
            "P": p_key,
            "K": k_key,
            "B1": b1_key,
            "B2": b2_key,
            "X": x_key,
            "D": d_key,
        }

    for cfg in make_balanced_configs(300, base_seed=20261030):
        w = generate_world(cfg)
        recounted = _independent_recount(w)
        assert callable(validate_hearing_fn), "validate_hearing must be exported by udlab_generator."
        validate_hearing_fn(w)
        skeys = compute_scenario_keys(w)
        for p, rec in skeys.perceptual_records.items():
            assert rec.heard_steps == recounted["heard"][p]  # type: ignore[index]
            assert rec.seen_steps == recounted["seen"][p]  # type: ignore[index]
            assert rec.messaged_steps == recounted["messaged"][p]  # type: ignore[index]
            assert rec.admissible_steps == recounted["admissible"][p]  # type: ignore[index]
        assert skeys.known_facts_by_agent == recounted["known_facts"]
        q_map = {q.question_type: q.key for q in skeys.questions}
        for qtype in ("P", "K", "B1", "B2", "X", "D"):
            assert q_map[qtype] == recounted[qtype], f"{w.scenario_id} {qtype} mismatch"


def test_16_hearing_direction_physics() -> None:
    """Test 16: Hearing follows physics: an adjacent-room say at 20 dB (< threshold) is NOT heard, and at 60 dB (>= threshold) IS heard."""
    import udlab_generator as ug

    w0 = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    r_a, r_b = w0.doorways[0]
    assert are_rooms_adjacent(r_a, r_b, w0.doorways)
    assert w0.noise_threshold == 50

    move_20 = Event(
        step=1,
        kind="move",
        actor=w0.agents[2],
        room=r_a,
        target_object=w0.secondary_object,
        from_location=w0.containers_by_room[r_a][0],
        to_location=w0.containers_by_room[r_a][1],
        fact_variable=w0.secondary_variable,
        fact_value=w0.containers_by_room[r_a][1],
    )
    say_20 = Event(
        step=2,
        kind="say",
        actor=w0.agents[2],
        room=r_a,
        utterance="Quiet 20 dB statement.",
        noise_level=20,
        fact_variable=w0.secondary_variable,
        fact_value=w0.containers_by_room[r_a][1],
    )
    move_60 = Event(
        step=3,
        kind="move",
        actor=w0.agents[2],
        room=r_a,
        target_object=w0.secondary_object,
        from_location=w0.containers_by_room[r_a][1],
        to_location=w0.containers_by_room[r_a][0],
        fact_variable=w0.secondary_variable,
        fact_value=w0.containers_by_room[r_a][0],
    )
    say_60 = Event(
        step=4,
        kind="say",
        actor=w0.agents[2],
        room=r_a,
        utterance="Loud 60 dB statement.",
        noise_level=60,
        fact_variable=w0.secondary_variable,
        fact_value=w0.containers_by_room[r_a][0],
    )
    hand_world = replace(
        w0,
        initial_agent_rooms={w0.agents[0]: r_b, w0.agents[1]: r_b, w0.agents[2]: r_a},
        events=(move_20, say_20, move_60, say_60),
    )
    recs = compute_perceptual_records(hand_world)
    # Speaker in r_a hears both 20 dB and 60 dB
    assert 2 in recs[w0.agents[2]].heard_steps
    assert 4 in recs[w0.agents[2]].heard_steps
    # Listener in doorway-adjacent room r_b must NOT hear 20 dB, and MUST hear 60 dB
    assert 2 not in recs[w0.agents[0]].heard_steps
    assert 4 in recs[w0.agents[0]].heard_steps

    # Also verify world-text sentence states the physics rule
    prose = render_world_template(w0, w0.agents[0])
    assert f"a say at or above {w0.noise_threshold} dB carries one doorway; quieter speech stays in its room" in prose


def test_17_object_locations_and_two_rooms_world_raises(tmp_path: Path) -> None:
    """Test 17: Validate object locations over all 300 scenarios, and verify a hand-built two-rooms world raises in validate_object_locations and verify_sealed_part."""
    import udlab_generator as ug

    validate_obj_fn = getattr(ug, "validate_object_locations", None)
    configs = make_balanced_configs(300, base_seed=20261030)

    # Scan all 300 scenarios for any object seen or moved from a room other than where it was last placed/seen
    two_room_violations = 0
    for cfg in configs:
        w = generate_world(cfg)
        last_room_of_obj: dict[str, str] = {}
        for ev in w.events:
            if ev.kind in ("move", "see") and ev.target_object:
                # Ignore station transitions around room occluders
                if ev.from_location in ("container bay", "far-side alcove") and ev.to_location in (
                    "container bay",
                    "far-side alcove",
                ):
                    continue
                # Ignore container-only glances where target_object is a container
                all_conts = {c for pair in w.containers_by_room.values() for c in pair}
                if ev.target_object in all_conts or ev.target_object in w.fixtures_by_room.values():
                    continue
                # Distinguish baseline vs reserve unit for aux_object if needed, or check note_item / primary / secondary
                obj_key = (
                    f"{ev.fact_variable}"
                    if ev.fact_variable and ev.fact_variable.startswith(("baseline:", "reserve:"))
                    else ev.target_object
                )
                if obj_key in last_room_of_obj and last_room_of_obj[obj_key] != ev.room:
                    two_room_violations += 1
                last_room_of_obj[obj_key] = ev.room

    assert two_room_violations == 0, f"Found {two_room_violations} cross-room object location violations across 300 scenarios."
    assert callable(validate_obj_fn), "validate_object_locations must be exported by udlab_generator."
    for cfg in configs:
        w = generate_world(cfg)
        validate_obj_fn(w)

    # Hand-built two-rooms world: Dora puts note_item in r_a's container, then Hana sees it in r_b's container
    w0 = generate_world(configs[0])
    r_a, r_b = w0.doorways[0]
    note_obj = next(d.note_item for d in ug.world.DOMAINS if d.name == w0.domain_name)
    c_a0, c_a1 = w0.containers_by_room[r_a]
    c_b0, c_b1 = w0.containers_by_room[r_b]
    two_rooms_events = w0.events + (
        Event(
            step=len(w0.events) + 1,
            kind="move",
            actor=w0.agents[2],
            room=r_a,
            target_object=note_obj,
            from_location=c_a0,
            to_location=c_a1,
        ),
        Event(
            step=len(w0.events) + 2,
            kind="see",
            actor=w0.agents[0],
            room=r_b,
            target_object=note_obj,
            from_location=c_b1,
            to_location=c_b1,
        ),
    )
    two_rooms_world = replace(
        w0,
        initial_agent_rooms=dict(w0.initial_agent_rooms, **{w0.agents[2]: r_a, w0.agents[0]: r_b}),
        events=two_rooms_events,
    )
    with pytest.raises(ValueError, match="object location|room|countertop"):
        validate_obj_fn(two_rooms_world)

    # Also verify verify_sealed_part raises when a sealed scenario world violates object locations
    writer = MockWriter()
    pub_dir = tmp_path / "pub_two_rooms"
    seal_dir = tmp_path / "seal_two_rooms"
    valid_pairs = [writer.render_scenario_pair(generate_world(c)) for c in make_balanced_configs(8, base_seed=20261030)]
    receipt = write_sealed_dataset(valid_pairs, pub_dir, seal_dir, private_seed_hex="cd" * 32)
    meta = json.loads(receipt.sealing_meta_path.read_text(encoding="utf-8"))
    for ev_dict in meta["scenarios"]["scen_0000"]["events"]:
        if ev_dict["kind"] == "see" and ev_dict["target_object"] == note_obj:
            other_room = next(r for r in meta["scenarios"]["scen_0000"]["rooms"] if r != ev_dict["room"])
            other_cont = meta["scenarios"]["scen_0000"]["containers_by_room"][other_room][1]
            ev_dict["to_location"] = other_cont
            break
    receipt.sealing_meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(SealingError, match="object location|countertop|validation failed"):
        verify_sealed_part(pub_dir, seal_dir)


def test_18_placing_fairness_rule() -> None:
    """Test 18: A U or S question about who placed/moved an object has a non-unknowable key only if EVERY focal agent saw the placing or heard it stated by someone who saw it."""
    import udlab_generator as ug

    validate_fairness_fn = getattr(ug, "validate_placing_fairness", None)
    configs = make_balanced_configs(300, base_seed=20261030)

    unfair_count = 0
    for cfg in configs:
        w = generate_world(cfg)
        recs = compute_perceptual_records(w)
        skeys = compute_scenario_keys(w)
        focal_agents = (w.agents[0], w.agents[1])
        for q in skeys.questions:
            if q.question_type in ("U", "S") and q.question_text.startswith("Who "):
                if q.key != UNKNOWABLE_KEY:
                    # Find the move event(s) placing the queried object
                    placing_steps = [
                        ev.step
                        for ev in w.events
                        if ev.kind == "move" and ev.target_object and ev.target_object in q.question_text
                    ]
                    for fa in focal_agents:
                        fa_saw = any(st in recs[fa].seen_steps for st in placing_steps)
                        fa_heard_from_witness = any(
                            ev.step in recs[fa].heard_steps
                            and not ev.is_bluff
                            and ev.target_object
                            and ev.target_object in q.question_text
                            and any(pst in recs[ev.actor].seen_steps and pst < ev.step for pst in placing_steps)
                            for ev in w.events
                            if ev.kind == "say"
                        )
                        if not (fa_saw or fa_heard_from_witness):
                            unfair_count += 1

    assert unfair_count == 0, f"Found {unfair_count} unfair 'Who placed...' questions across 300 scenarios."
    assert callable(validate_fairness_fn), "validate_placing_fairness must be exported by udlab_generator."
    for cfg in configs:
        w = generate_world(cfg)
        validate_fairness_fn(w)

    # Hand-built unfair world where settled S 'Who placed...' happens in a room where focal agent a0 is not present and never hears about it
    w_s_who = next(generate_world(c) for c in configs if not c.u_is_who)
    r_a, r_b = w_s_who.doorways[0]
    note_obj = next(d.note_item for d in ug.world.DOMAINS if d.name == w_s_who.domain_name)
    # Remove any move/say of note_obj and add an unseen move in r_b while a0 is in r_a
    filtered_evs = [
        ev for ev in w_s_who.events if ev.target_object != note_obj
    ]
    # Find where a2 and a0 are at the end of filtered_evs
    occ = dict(w_s_who.initial_agent_rooms)
    for ev in filtered_evs:
        if ev.kind == "enter":
            occ[ev.actor] = ev.room
    a0_room = occ[w_s_who.agents[0]]
    a2_room = occ[w_s_who.agents[2]]
    if a0_room == a2_room:
        # Move a0 to an adjacent room so a0 does not see a2's placement
        adj_r = next(r for r in w_s_who.rooms if are_rooms_adjacent(a0_room, r, w_s_who.doorways))
        filtered_evs.extend(
            [
                Event(
                    step=len(filtered_evs) + 1,
                    kind="leave",
                    actor=w_s_who.agents[0],
                    room=a0_room,
                    from_location=a0_room,
                    to_location=adj_r,
                ),
                Event(
                    step=len(filtered_evs) + 2,
                    kind="enter",
                    actor=w_s_who.agents[0],
                    room=adj_r,
                    from_location=a0_room,
                    to_location=adj_r,
                ),
            ]
        )
    c2_0, c2_1 = w_s_who.containers_by_room[a2_room]
    filtered_evs.append(
        Event(
            step=len(filtered_evs) + 1,
            kind="move",
            actor=w_s_who.agents[2],
            room=a2_room,
            target_object=note_obj,
            from_location=c2_0,
            to_location=c2_1,
        )
    )
    reindexed = tuple(replace(ev, step=idx) for idx, ev in enumerate(filtered_evs, start=1))
    unfair_world = replace(
        w_s_who,
        events=reindexed,
        unknowable_spec=replace(w_s_who.unknowable_spec, trace_a=reindexed, trace_b=reindexed),
    )
    with pytest.raises(ValueError, match="fairness|unfair|placed|focal"):
        validate_fairness_fn(unfair_world)


def test_19_article_a_an_every_vocabulary_noun() -> None:
    """Test 19: Indefinite article helper selects a/an by the next word's sound across every object, container, occluder, fixture, and special cases, and rendered prose never contains 'a easel screen'."""
    import udlab_generator as ug
    from udlab_generator.world import DOMAINS

    article_fn = getattr(ug, "indefinite_article", None)
    # Verify rendered prose across all domains never has 'a easel' or any wrong article before occluders
    for cfg in make_balanced_configs(12, base_seed=20261030):
        w = generate_world(cfg)
        prose = render_world_template(w, w.agents[0])
        assert "a easel" not in prose.lower()

    assert callable(article_fn), "indefinite_article must be exported by udlab_generator."

    # Check phonetic special cases
    assert article_fn("easel screen") == "an"
    assert article_fn("oak drawer") == "an"
    assert article_fn("iron safe") == "an"
    assert article_fn("unsigned transfer slip") == "an"
    assert article_fn("overhead louver") == "an"
    assert article_fn("hour") == "an"
    assert article_fn("honest mistake") == "an"
    assert article_fn("heirloom") == "an"
    assert article_fn("herbarium") == "an"
    assert article_fn("unit") == "a"
    assert article_fn("uniform") == "a"
    assert article_fn("one-way mirror") == "a"
    assert article_fn("frosted partition") == "a"
    assert article_fn("humidity dial") == "a"

    # Test every object, container, occluder, fixture, note_item, and side_mechanism across all DOMAINS
    vowel_First_words = {
        "agate",
        "aluminum",
        "ash",
        "assay",
        "azimuth",
        "easel",
        "enamel",
        "exhaust",
        "index",
        "iron",
        "oak",
        "overhead",
        "unmarked",
        "unsigned",
    }
    for dom in DOMAINS:
        all_nouns = (
            *dom.rooms,
            *dom.objects,
            *(c for pair in dom.containers for c in pair),
            *dom.occluders,
            *dom.fixtures,
            dom.note_item,
            dom.side_mechanism,
        )
        for noun in all_nouns:
            first_word = noun.split()[0].lower()
            expected = "an" if first_word in vowel_First_words else "a"
            assert article_fn(noun) == expected, f"Wrong article for {noun!r}: got {article_fn(noun)!r}, expected {expected!r}"

    for cfg in make_balanced_configs(12, base_seed=20261030):
        w = generate_world(cfg)
        prose = render_world_template(w, w.agents[0])
        for occ in w.occluders_by_room.values():
            expected_phrase = f"with {article_fn(occ)} {occ} standing"
            assert expected_phrase in prose


def test_20_belief_persistence_rule_sentence_in_world_text_and_writer_prompt() -> None:
    """Test 20: Belief-persistence rule sentence appears in render_world_template and writer prompt template."""
    import udlab_generator as ug

    expected_rule = (
        "Each agent assumes an object remains where that agent last saw it in open view "
        "or heard a witness state it was placed, and assumes another agent looks "
        "where they last saw or communicated that agent perceiving it."
    )
    prompt_text = load_writer_prompt_template()
    assert expected_rule in prompt_text, "Writer prompt must include the exact belief-persistence rule sentence."

    for cfg in make_balanced_configs(12, base_seed=20261030):
        w = generate_world(cfg)
        for fa in (w.agents[0], w.agents[1]):
            prose = render_world_template(w, fa)
            assert expected_rule in prose, f"{w.scenario_id}: rule sentence missing from rendered prose"

    rule_sentence = getattr(ug, "BELIEF_PERSISTENCE_RULE", None)
    assert rule_sentence == expected_rule, "BELIEF_PERSISTENCE_RULE must be exported by udlab_generator."


def test_21_independent_b2_derivation_and_no_unexplained_displacement(tmp_path: Path) -> None:
    """Test 21: Independent B2 derivation from the stated rule sentence alone agrees with every sealed B2 key across 300 scenarios, and b2_observer never meets an unexplained displacement."""
    import udlab_generator as ug

    validate_forcing_fn = getattr(ug, "validate_belief_forcing", None)

    def _derive_b2_from_rule_sentence_alone(w: ug.World) -> str:
        """Apply only the stated rule sentence to derive B2(b2_observer, b2_actor, primary_object),
        while asserting that b2_observer never encounters an unexplained displacement of primary_object.
        """
        all_conts = {c for pair in w.containers_by_room.values() for c in pair}
        edges = {(a, b) for a, b in w.doorways} | {(b, a) for a, b in w.doorways}
        room_of: dict[str, str | None] = {ag: w.initial_agent_rooms[ag] for ag in w.agents}
        station_of: dict[str, str] = {ag: "container bay" for ag in w.agents}
        last_seen_by_obs: str | None = None
        witnessed_by_ag: dict[str, str] = {}
        b2_from_rule: str | None = None

        for ev in w.events:
            if ev.kind == "enter" and ev.actor in w.agents:
                room_of[ev.actor] = ev.room
                station_of[ev.actor] = "container bay"
            elif (
                ev.kind == "move"
                and ev.from_location in ("container bay", "far-side alcove")
                and ev.to_location in ("container bay", "far-side alcove")
            ):
                station_of[ev.actor] = str(ev.to_location)
            elif ev.kind in ("move", "see") and ev.target_object == w.primary_object:
                if ev.actor == w.b2_observer:
                    if ev.from_location in all_conts or last_seen_by_obs is not None:
                        assert last_seen_by_obs is not None and ev.from_location == last_seen_by_obs, (
                            f"{w.scenario_id} step {ev.step}: b2_observer {w.b2_observer} found {w.primary_object} "
                            f"at {ev.from_location!r}, but last placed/saw it at {last_seen_by_obs!r}"
                        )
                if ev.to_location in all_conts and not ev.occluded:
                    for ag in w.agents:
                        if (
                            room_of.get(ag) == ev.room
                            and station_of.get(ag) == "container bay"
                            and (ev.kind == "move" or ag == ev.actor)
                        ):
                            witnessed_by_ag[ag] = str(ev.to_location)
                            if ag == w.b2_observer:
                                last_seen_by_obs = str(ev.to_location)
                    obs_saw = (
                        room_of.get(w.b2_observer) == ev.room
                        and station_of.get(w.b2_observer) == "container bay"
                        and (ev.kind == "move" or ev.actor == w.b2_observer)
                    )
                    act_saw = (
                        room_of.get(w.b2_actor) == ev.room
                        and station_of.get(w.b2_actor) == "container bay"
                        and (ev.kind == "move" or ev.actor == w.b2_actor)
                    )
                    if obs_saw and act_saw:
                        b2_from_rule = str(ev.to_location)
            elif (
                ev.kind in ("say", "message")
                and not ev.is_bluff
                and ev.fact_variable == w.primary_variable
                and ev.fact_value is not None
                and witnessed_by_ag.get(ev.actor) == ev.fact_value
            ):
                if ev.kind == "message":
                    if (ev.actor == w.b2_observer and ev.recipient == w.b2_actor) or (
                        ev.actor == w.b2_actor and ev.recipient == w.b2_observer
                    ):
                        b2_from_rule = str(ev.fact_value)
                elif ev.kind == "say" and ev.actor in (w.b2_observer, w.b2_actor):
                    obs_r = room_of.get(w.b2_observer)
                    act_r = room_of.get(w.b2_actor)
                    obs_h = obs_r is not None and (
                        obs_r == ev.room or ((obs_r, ev.room) in edges and ev.noise_level >= w.noise_threshold)
                    )
                    act_h = act_r is not None and (
                        act_r == ev.room or ((act_r, ev.room) in edges and ev.noise_level >= w.noise_threshold)
                    )
                    if obs_h and act_h:
                        b2_from_rule = str(ev.fact_value)

            if ev.kind == "leave" and ev.actor in w.agents:
                room_of[ev.actor] = None

        assert b2_from_rule is not None
        return b2_from_rule

    configs = make_balanced_configs(300, base_seed=20261030)
    worlds = [generate_world(c) for c in configs]
    derived_b2_by_sid: dict[str, str] = {}
    for w in worlds:
        derived_b2_by_sid[w.scenario_id] = _derive_b2_from_rule_sentence_alone(w)
        ug.validate_occlusion(w)
        assert callable(validate_forcing_fn), "validate_belief_forcing must be exported by udlab_generator."
        validate_forcing_fn(w)

    writer = MockWriter()
    pairs = [writer.render_scenario_pair(w) for w in worlds]
    receipt = write_sealed_dataset(
        rendered_pairs=pairs,
        public_dir=tmp_path / "pub_b2_recount",
        sealed_dir=tmp_path / "seal_b2_recount",
        private_seed_hex="aa55" * 16,
    )
    sealed_rows = [
        json.loads(line)
        for line in receipt.keys_sealed_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Every sealed B2 key across all 600 rows (300 scenarios x 2 passes) must agree with the independent rule derivation
    assert len(sealed_rows) == 600
    for row in sealed_rows:
        sid = str(row["scenario_id"])
        sealed_b2 = next(str(q["key"]) for q in row["questions"] if q["question_type"] == "B2")
        assert sealed_b2 == derived_b2_by_sid[sid], (
            f"Sealed B2 key {sealed_b2!r} != rule-derived {derived_b2_by_sid[sid]!r} in {sid}"
        )


def test_22_scen_0111_old_unexplained_displacement_world_refused(tmp_path: Path) -> None:
    """Test 22: A hand-built world reproducing scen_0111's old pattern (target moves primary_object behind observer's back, and observer later moves it from that displaced container) is refused by validate_occlusion, validate_belief_forcing, and verify_sealed_part."""
    import udlab_generator as ug

    validate_forcing_fn = getattr(ug, "validate_belief_forcing", None)

    cfg_0111 = next(c for c in make_balanced_configs(300, base_seed=20261030) if c.scenario_id == "scen_0111")
    w_0111 = generate_world(cfg_0111)
    a1 = w_0111.b2_actor     # Cleo
    a2 = w_0111.b2_observer  # Alma
    pobj = w_0111.primary_object
    pvar = w_0111.primary_variable

    first_move_idx = next(
        i for i, ev in enumerate(w_0111.events) if ev.kind == "move" and ev.target_object == pobj and ev.fact_id == "F0a"
    )
    # Find the final move index (where a2 moves pobj from cont_init to cont_final)
    final_move_idx = next(
        i
        for i, ev in enumerate(w_0111.events)
        if i > first_move_idx and ev.kind == "move" and ev.target_object == pobj and ev.actor == a2 and ev.fact_id != "F0"
    )
    # Find where a0's doorway transition starts right before final_move_idx
    a0_walk_idx = final_move_idx - 2
    r_work = w_0111.events[first_move_idx].room
    occ_work = w_0111.occluders_by_room[r_work]
    cont_final = str(w_0111.events[first_move_idx].to_location)
    cont_init = str(w_0111.events[final_move_idx].from_location)

    # Reconstruct scen_0111's old 646e5b39 pattern:
    # Step A: a2 places pobj in cont_final (in open view of a1 and a2)
    # Step B: a2 steps behind occ_work to far-side alcove
    # Step C: a1 moves pobj from cont_final to cont_init while a2 is behind occ_work
    # Step D: a1 steps behind occ_work to far-side alcove
    # Step E: a2 steps around occ_work back to container bay, then later moves pobj from cont_init (displaced!) to cont_final
    old_block = [
        w_0111.events[first_move_idx],
        Event(
            step=0,
            kind="move",
            actor=a2,
            room=r_work,
            target_object=occ_work,
            from_location="container bay",
            to_location="far-side alcove",
            occluded=False,
        ),
        Event(
            step=0,
            kind="move",
            actor=a1,
            room=r_work,
            target_object=pobj,
            from_location=cont_final,
            to_location=cont_init,
            occluded=False,
            fact_id="F0",
            fact_variable=pvar,
            fact_value=cont_init,
        ),
        Event(
            step=0,
            kind="move",
            actor=a1,
            room=r_work,
            target_object=occ_work,
            from_location="container bay",
            to_location="far-side alcove",
            occluded=False,
        ),
        Event(
            step=0,
            kind="move",
            actor=a2,
            room=r_work,
            target_object=occ_work,
            from_location="far-side alcove",
            to_location="container bay",
            occluded=False,
        ),
    ]
    raw_old_events = list(w_0111.events[:first_move_idx]) + old_block + list(w_0111.events[a0_walk_idx:])
    old_events = tuple(replace(ev, step=idx) for idx, ev in enumerate(raw_old_events, start=1))
    old_world = replace(
        w_0111,
        events=old_events,
        unknowable_spec=replace(w_0111.unknowable_spec, trace_a=old_events, trace_b=old_events),
    )

    with pytest.raises(ValueError, match="belief-forcing|unexplained displacement|last placed or saw"):
        ug.validate_occlusion(old_world)
    assert callable(validate_forcing_fn), "validate_belief_forcing must be exported by udlab_generator."
    with pytest.raises(ValueError, match="belief-forcing|unexplained displacement|last placed or saw"):
        validate_forcing_fn(old_world)

    # Also verify verify_sealed_part refuses a sealed store containing old_world
    writer = MockWriter()
    pub_dir = tmp_path / "pub_old_0111"
    seal_dir = tmp_path / "seal_old_0111"
    valid_configs = make_balanced_configs(8, base_seed=20261030)
    valid_pairs = [writer.render_scenario_pair(generate_world(c)) for c in valid_configs]
    receipt = write_sealed_dataset(valid_pairs, pub_dir, seal_dir, private_seed_hex="ef" * 32)
    meta = json.loads(receipt.sealing_meta_path.read_text(encoding="utf-8"))
    from dataclasses import asdict
    meta["scenarios"]["scen_0000"] = asdict(replace(old_world, scenario_id="scen_0000"))
    receipt.sealing_meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(SealingError, match="belief-forcing|unexplained displacement|validation failed"):
        verify_sealed_part(pub_dir, seal_dir)


def test_23_ollama_writer_pinned_digest_loopback_vram_checkpoint_generations_and_length_stop(
    tmp_path: Path,
) -> None:
    """Test 23 (Req 1): OllamaWriter enforces qwen3.6:35b digest pin, loopback-only server, --vram-halt-mib (default 26000) with PID-tree kill, per-scenario sealed checkpoint & --resume, generations.jsonl (§6.8), and length_stop refusal."""
    import udlab_generator as ug
    from udlab_generator.cli import build_and_probe_dataset

    OllamaWriterCls = getattr(ug, "OllamaWriter", None)
    assert callable(OllamaWriterCls), "OllamaWriter must be exported by udlab_generator."

    w = generate_world(make_balanced_configs(2, base_seed=20261030)[0])
    pinned_digest = "a7eb95c53bcf96b4bdd008d0fab4a5dac88047d9c1a7a9ab88ed453423fbd87c"

    # 1. Non-loopback URLs must be refused
    with pytest.raises(ValueError, match="NON_LOOPBACK_URL|127.0.0.1"):
        OllamaWriterCls(
            base_url="http://192.168.1.50:11434",
            grant_id="test-grant",
            http_transport=lambda m, u, b: {},
        )
    with pytest.raises(ValueError, match="NON_LOOPBACK_URL|127.0.0.1"):
        OllamaWriterCls(
            base_url="https://localhost:11434",
            grant_id="test-grant",
            http_transport=lambda m, u, b: {},
        )

    # 2. Live transport without explicit GPU grant ID must be refused
    with pytest.raises(PermissionError, match="GPU_GRANT_REQUIRED"):
        OllamaWriterCls(base_url="http://127.0.0.1:11434", grant_id=None)

    # 3. Digest mismatch or missing model on GET /api/tags must be refused (and /api/pull is forbidden)
    def _bad_digest_transport(method: str, url: str, body: dict | None = None) -> dict:
        if method == "GET" and url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": "qwen3.6:35b",
                        "model": "qwen3.6:35b",
                        "digest": "0000000000000000000000000000000000000000000000000000000000000000",
                    }
                ]
            }
        raise AssertionError(f"Unexpected call {method} {url}")

    bad_digest_writer = OllamaWriterCls(
        grant_id="test-grant",
        http_transport=_bad_digest_transport,
    )
    assert bad_digest_writer.ollama_tag == "qwen3.6:35b"
    assert bad_digest_writer.manifest_digest == pinned_digest
    assert bad_digest_writer.vram_halt_mib == 26000
    with pytest.raises(ValueError, match="DIGEST_MISMATCH"):
        bad_digest_writer.render_scenario_pair(w)

    # 4. Owned loopback server lifecycle + unload + generations.jsonl (§6.8)
    fake_pids_stopped: list[int] = []

    class _FakeChildProc:
        pid = 42424
        returncode = None

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 0
            fake_pids_stopped.append(self.pid)

        def wait(self, timeout: float | None = None) -> int:
            return 0

    def _valid_transport(method: str, url: str, body: dict | None = None) -> dict:
        if method == "GET" and url.endswith("/api/version"):
            return {"version": "0.6.2-loopback-test"}
        if method == "GET" and url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": "qwen3.6:35b",
                        "model": "qwen3.6:35b",
                        "digest": f"sha256:{pinned_digest}",
                    }
                ]
            }
        if method == "POST" and url.endswith("/api/chat"):
            payload = dict(body or {})
            if payload.get("keep_alive") == 0:
                return {"model": "qwen3.6:35b", "done": True, "done_reason": "unload"}
            prompt_content = payload["messages"][-1]["content"]
            focal = prompt_content.split("Focal Agent: ", 1)[1].splitlines()[0].strip()
            return {
                "model": "qwen3.6:35b",
                "message": {
                    "role": "assistant",
                    "content": render_world_template(w, focal),
                    "thinking": "step one step two step three",
                },
                "done": True,
                "done_reason": "stop",
                "eval_count": 412,
                "prompt_eval_count": 1680,
            }
        raise AssertionError(f"Unexpected call {method} {url}")

    gen_file = tmp_path / "generations.jsonl"
    writer = OllamaWriterCls(
        grant_id="grant-rx-test",
        http_transport=_valid_transport,
        vram_probe=lambda: 25336,
        generations_path=gen_file,
        start_owned_server=True,
        popen_factory=lambda *a, **kw: _FakeChildProc(),
        server_port=11499,
    )
    assert writer.owned_server is not None
    assert writer.owned_server.base_url == "http://127.0.0.1:11499"
    r0, r1 = writer.render_scenario_pair(w)
    writer.close()

    assert 42424 in fake_pids_stopped
    assert len(writer.unload_records) == 1
    assert writer.owned_server.unloaded is True
    assert gen_file.is_file()
    gen_lines = [json.loads(line) for line in gen_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(gen_lines) == 2
    for idx_g, g_row in enumerate(gen_lines):
        assert g_row["call_id"].startswith(f"{w.scenario_id}:r{idx_g}:call_")
        assert g_row["model"] == "qwen3.6:35b"
        assert g_row["manifest_digest"] == pinned_digest
        assert g_row["pins"]["manifest_digest"] == pinned_digest
        assert g_row["eval_count"] == 412
        assert g_row["prompt_eval_count"] == 1680
        assert g_row["thinking_words"] == 6
        assert g_row["vram_peak_mib"] == 25336
        assert g_row["vram_halt_mib"] == 26000
        assert g_row["seconds"] >= 0.0

    # 5. length_stop refusal when done_reason == 'length'
    def _length_stop_transport(method: str, url: str, body: dict | None = None) -> dict:
        if method == "GET" and url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3.6:35b", "digest": pinned_digest}]}
        if method == "POST" and url.endswith("/api/chat"):
            return {
                "model": "qwen3.6:35b",
                "message": {"role": "assistant", "content": "truncated output", "thinking": "long thought"},
                "done": True,
                "done_reason": "length",
                "eval_count": 8192,
                "prompt_eval_count": 1700,
            }
        raise AssertionError(f"Unexpected call {method} {url}")

    len_writer = OllamaWriterCls(grant_id="grant-rx-test", http_transport=_length_stop_transport)
    with pytest.raises(RenderCheckError, match="done_reason='length'|length"):
        len_writer.render_scenario_pair(w)
    assert len(len_writer.refusals) == 1
    assert len_writer.refusals[0].check_id == "length_stop"

    # 6. VRAM ceiling halt (> vram_halt_mib) kills owned PID tree
    killed_pids: list[int] = []
    halt_writer = OllamaWriterCls(
        grant_id="grant-rx-test",
        vram_halt_mib=26000,
        vram_probe=lambda: 26150,
        halt_pid=98765,
        pid_killer=lambda pid: killed_pids.append(pid),
        http_transport=_valid_transport,
    )
    with pytest.raises(RuntimeError, match="VRAM_HALT_EXCEEDED|26150"):
        halt_writer.render_scenario_pair(w)
    assert killed_pids == [98765]

    # 7. Per-scenario sealed checkpoint and --resume across an interrupted 4-scenario build
    configs_4 = make_balanced_configs(4, base_seed=20261030)
    worlds_by_sid = {c.scenario_id: generate_world(c) for c in configs_4}
    chat_calls_made = 0

    def _interrupting_transport(method: str, url: str, body: dict | None = None) -> dict:
        nonlocal chat_calls_made
        if method == "GET" and url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3.6:35b", "digest": pinned_digest}]}
        if method == "POST" and url.endswith("/api/chat"):
            payload = dict(body or {})
            if payload.get("keep_alive") == 0:
                return {"model": "qwen3.6:35b", "done": True, "done_reason": "unload"}
            chat_calls_made += 1
            if chat_calls_made == 5:
                raise RuntimeError("Simulated mid-build interruption after 2 scenarios")
            prompt_content = payload["messages"][-1]["content"]
            focal = prompt_content.split("Focal Agent: ", 1)[1].splitlines()[0].strip()
            sid = json.loads(prompt_content.split("World Setup:\n", 1)[1].split("\n\nComplete Event Log:", 1)[0])[
                "scenario_id"
            ]
            return {
                "model": "qwen3.6:35b",
                "message": {"role": "assistant", "content": render_world_template(worlds_by_sid[sid], focal)},
                "done": True,
                "done_reason": "stop",
                "eval_count": 390,
                "prompt_eval_count": 1650,
            }
        raise AssertionError(f"Unexpected {method} {url}")

    resume_dir = tmp_path / "ckpt_resume_run"
    with pytest.raises(RuntimeError, match="Simulated mid-build interruption"):
        build_and_probe_dataset(
            count=4,
            base_seed=20261030,
            output_dir=resume_dir,
            writer_kind="ollama",
            grant_id="grant-rx-test",
            start_owned_server=False,
            http_transport=_interrupting_transport,
            vram_probe=lambda: 25100,
        )

    ckpt_pub = json.loads((resume_dir / "checkpoint.json").read_text(encoding="utf-8"))
    ckpt_seal = json.loads((resume_dir / "sealed_store" / "checkpoint.sealed.json").read_text(encoding="utf-8"))
    assert ckpt_pub["scenarios_completed"] == 2
    assert ckpt_seal["scenarios_completed"] == 2

    # Now resume: only the remaining 2 scenarios (4 calls) should execute
    calls_before_resume = chat_calls_made
    resumed_summary = build_and_probe_dataset(
        count=4,
        base_seed=20261030,
        output_dir=resume_dir,
        writer_kind="ollama",
        grant_id="grant-rx-test",
        resume=True,
        start_owned_server=False,
        http_transport=_interrupting_transport,
        vram_probe=lambda: 25100,
    )
    assert chat_calls_made - calls_before_resume == 4
    assert resumed_summary["num_scenarios"] == 4
    assert resumed_summary["num_items"] == 8
    gen_resumed = [
        json.loads(line)
        for line in (resume_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(gen_resumed) == 8


def test_24_writer_prompt_carries_belief_persistence_rule_exactly_once() -> None:
    """Test 24 (Req 2): Writer prompt template and formatted prompt carry BELIEF_PERSISTENCE_RULE byte-identically exactly once, with no leaked fact IDs."""
    from udlab_generator.world import BELIEF_PERSISTENCE_RULE

    template_text = load_writer_prompt_template()
    assert template_text.count(BELIEF_PERSISTENCE_RULE) == 1

    for cfg in make_balanced_configs(10, base_seed=20261030):
        w = generate_world(cfg)
        for fa in (w.agents[0], w.agents[1]):
            prompt = format_writer_prompt(w, fa)
            assert prompt.count(BELIEF_PERSISTENCE_RULE) == 1, (
                f"{w.scenario_id} ({fa}): BELIEF_PERSISTENCE_RULE count is {prompt.count(BELIEF_PERSISTENCE_RULE)}, expected 1."
            )
            # Also verify no F0..F9 fact IDs or causal_preconditions leak into the formatted writer prompt
            assert "causal_preconditions" not in prompt
            assert '"fact_id"' not in prompt

    # Duplicating or removing the rule sentence in a custom template must raise RenderCheckError
    w0 = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    doubled_template = template_text + "\n" + BELIEF_PERSISTENCE_RULE
    with pytest.raises(RenderCheckError, match="byte-identically once"):
        format_writer_prompt(w0, w0.agents[0], template=doubled_template)


def test_25_prose_checks_a_b_c_d_all_refusing_with_recorded_reasons() -> None:
    """Test 25 (Req 3): Prose checks (a) no markers/fact IDs/step N, (b) banned adverbs, (c) render fidelity over event log, and (d) rule sentence + hearing threshold all refuse and record reasons."""
    import udlab_generator as ug
    from udlab_generator.world import BELIEF_PERSISTENCE_RULE

    w = generate_world(make_balanced_configs(2, base_seed=20261030)[0])
    fa = w.agents[0]
    clean_prose = render_world_template(w, fa)

    # Clean prose rendered by render_world_template must pass all checks (a)-(d)
    assert check_rendered_prose(clean_prose, w, focal_agent=fa) == ()
    assert "[establishing" not in clean_prose
    assert "preconditions:" not in clean_prose
    for fid in ("F0", "F0a", "F1", "F2", "F3", "F4"):
        assert fid not in clean_prose
    for ev in w.events:
        assert f"step {ev.step}" not in clean_prose.lower()

    # Check (a): Bracket markers, fact IDs, and "step N" must each refuse
    for bad_marker, expected_cid in (
        (f"{clean_prose} [establishing F1]", "a_no_markers_bracket"),
        (clean_prose.replace("Stated goals:", "Stated goals (F1, F2):"), "a_no_markers_fact_id"),
        (clean_prose.replace("Next,", "At step 2,", 1), "a_no_markers_step_number"),
    ):
        mw = MockWriter(custom_renderer=lambda _w, _fa, txt=bad_marker: txt)
        with pytest.raises(RenderCheckError) as exc_info:
            mw.render_scenario_pair(w)
        assert exc_info.value.check_id == expected_cid
        assert len(mw.refusals) == 1
        assert mw.refusals[0].check_id == expected_cid

    # Check (b): Banned adverbs refuse and record reason
    mw_adv = MockWriter(custom_renderer=lambda _w, _fa: f"{clean_prose} {fa} furtively looked around.")
    with pytest.raises(RenderCheckError) as exc_adv:
        mw_adv.render_scenario_pair(w)
    assert exc_adv.value.check_id == "b_banned_adverbs"
    assert mw_adv.refusals[0].check_id == "b_banned_adverbs"

    # Check (c): Render fidelity (missing event, out-of-order events, wrong actor/object/room, unlogged party, unlogged event sentence)
    paras = clean_prose.split("\n\n")
    setup_p, events_p = paras[0], paras[1]
    ev_sents = [s.strip() for s in events_p.split(". ") if s.strip()]
    ev_sents = [s if s.endswith(".") or s.endswith('"') else s + "." for s in ev_sents]

    # (c.1) Missing event
    missing_ev_prose = setup_p + "\n\n" + " ".join(ev_sents[:-1])
    with pytest.raises(RenderCheckError, match="missing or altered event") as exc_c1:
        check_rendered_prose(missing_ev_prose, w, focal_agent=fa)
    assert exc_c1.value.check_id == "c_fidelity_missing_event"

    # (c.2) Out-of-order events
    swapped_sents = [ev_sents[1], ev_sents[0], *ev_sents[2:]]
    swapped_prose = setup_p + "\n\n" + " ".join(swapped_sents)
    with pytest.raises(RenderCheckError, match="out of relative chronological order|missing or altered event") as exc_c2:
        check_rendered_prose(swapped_prose, w, focal_agent=fa)
    assert exc_c2.value.check_id in ("c_fidelity_event_order", "c_fidelity_missing_event")

    # (c.3) Unlogged extra event sentence asserted
    extra_ev_prose = clean_prose + f" Next, in the {w.rooms[0]}, {fa} observed the {w.objects[0]} in open view."
    with pytest.raises(RenderCheckError, match="unlogged event sentence") as exc_c3:
        check_rendered_prose(extra_ev_prose, w, focal_agent=fa)
    assert exc_c3.value.check_id == "c_fidelity_unlogged_event"

    # (c.4) Unlogged party name from another pool asserted
    unlogged_party = next(p for pool in ug.world.AGENT_NAME_POOLS for p in pool if p not in w.agents and p not in w.absent_parties)
    unlogged_party_prose = clean_prose.replace("Stakes:", f"Stakes ({unlogged_party}):")
    with pytest.raises(RenderCheckError, match="unlogged party name") as exc_c4:
        check_rendered_prose(unlogged_party_prose, w, focal_agent=fa)
    assert exc_c4.value.check_id == "c_fidelity_unlogged_party"

    # Check (d): Missing BELIEF_PERSISTENCE_RULE or missing hearing threshold in setup prose refuses
    no_rule_prose = clean_prose.replace(BELIEF_PERSISTENCE_RULE, "Agents remember what they see.")
    with pytest.raises(RenderCheckError, match="BELIEF_PERSISTENCE_RULE") as exc_d1:
        check_rendered_prose(no_rule_prose, w, focal_agent=fa)
    assert exc_d1.value.check_id == "d_rule_sentence"

    no_thresh_prose = clean_prose.replace(f"{w.noise_threshold} dB", "quiet level", 2)
    with pytest.raises(RenderCheckError, match="hearing threshold") as exc_d2:
        check_rendered_prose(no_thresh_prose, w, focal_agent=fa)
    assert exc_d2.value.check_id == "d_hearing_threshold"


def test_26_verify_written_cli_and_ten_scenario_mock_build(tmp_path: Path) -> None:
    """Test 26 (Req 4 & 5): `udlab-gen --write --writer mock --count 10` and `udlab-gen --verify-written <dir>` run independent B2 derivation, all recounts, prose checks, and `--probe-only`."""
    out_dir = tmp_path / "mock_10_run"
    rc_write = cli_main(["--write", "--writer", "mock", "--count", "10", "--out-dir", str(out_dir)])
    assert rc_write == 0
    assert (out_dir / "generations.jsonl").is_file()
    assert (out_dir / "checkpoint.json").is_file()
    assert (out_dir / "sealed_store" / "checkpoint.sealed.json").is_file()

    rc_verify = cli_main(["--verify-written", str(out_dir)])
    assert rc_verify == 0
    verify_doc = json.loads((out_dir / "verify_written_report.json").read_text(encoding="utf-8"))
    assert verify_doc["passed"] is True
    assert verify_doc["generations_count"] == 20
    assert verify_doc["recount_verification"]["scenarios_recounted"] == 10
    assert verify_doc["recount_verification"]["renderings_prose_checked"] == 20
    assert verify_doc["recount_verification"]["independent_b2_verified"] is True
    assert verify_doc["recount_verification"]["independent_recounts_verified"] is True
    assert verify_doc["probe_only"]["probe_report"]["passed"] is True


def test_27_prose_fidelity_swapped_actor_and_recipient_refused() -> None:
    """Test 27 (Change 3.i): Swapping actor and recipient (or actor and bystander) in an event sentence must be refused by check_rendered_prose."""
    w = generate_world(make_balanced_configs(2, base_seed=20261030)[0])
    fa = w.agents[0]
    clean_prose = render_world_template(w, fa)
    setup_p, events_p = clean_prose.split("\n\n")
    from udlab_generator.render import _split_prose_sentences

    ev_sents = _split_prose_sentences(events_p)

    # 1. Message event: swap sender (Cara) and recipient (Gia) while keeping both names in the sentence
    msg_idx = next(i for i, ev in enumerate(w.events) if ev.kind == "message")
    msg_ev = w.events[msg_idx]
    swapped_msg_sent = (
        f"Next, from the {msg_ev.room}, {msg_ev.recipient} sent a direct text message to {msg_ev.actor}: "
        f'"{msg_ev.utterance}"'
    )
    bad_msg_sents = list(ev_sents)
    bad_msg_sents[msg_idx] = swapped_msg_sent
    with pytest.raises(RenderCheckError, match="missing or altered event") as exc_msg:
        check_rendered_prose(setup_p + "\n\n" + " ".join(bad_msg_sents), w, focal_agent=fa)
    assert exc_msg.value.check_id == "c_fidelity_missing_event"

    # 2. Move event: mention true actor in a subordinate clause while another in-scenario agent acts as subject
    move_idx = next(
        i for i, ev in enumerate(w.events) if ev.kind == "move" and ev.target_object == w.primary_object
    )
    move_ev = w.events[move_idx]
    other_ag = next(a for a in w.agents if a != move_ev.actor)
    swapped_move_sent = (
        f"Next, in the {move_ev.room}, while {move_ev.actor} watched nearby, {other_ag} moved the "
        f"{move_ev.target_object} from the {move_ev.from_location} to the {move_ev.to_location} in open view at the container bay."
    )
    bad_move_sents = list(ev_sents)
    bad_move_sents[move_idx] = swapped_move_sent
    with pytest.raises(RenderCheckError, match="missing or altered event") as exc_mv:
        check_rendered_prose(setup_p + "\n\n" + " ".join(bad_move_sents), w, focal_agent=fa)
    assert exc_mv.value.check_id == "c_fidelity_missing_event"


def test_28_prose_fidelity_reversed_from_to_refused() -> None:
    """Test 28 (Change 3.ii): Reversing from_location and to_location in a move, enter, or leave sentence must be refused."""
    w = generate_world(make_balanced_configs(2, base_seed=20261030)[0])
    fa = w.agents[0]
    clean_prose = render_world_template(w, fa)
    setup_p, events_p = clean_prose.split("\n\n")
    from udlab_generator.render import _split_prose_sentences

    ev_sents = _split_prose_sentences(events_p)

    # 1. Object move with reversed from_location / to_location
    move_idx = next(
        i for i, ev in enumerate(w.events) if ev.kind == "move" and ev.target_object == w.primary_object
    )
    move_ev = w.events[move_idx]
    rev_move_sent = (
        f"Next, in the {move_ev.room}, {move_ev.actor} moved the {move_ev.target_object} "
        f"from the {move_ev.to_location} to the {move_ev.from_location} in open view at the container bay."
    )
    bad_sents_1 = list(ev_sents)
    bad_sents_1[move_idx] = rev_move_sent
    with pytest.raises(RenderCheckError, match="missing or altered event") as exc_1:
        check_rendered_prose(setup_p + "\n\n" + " ".join(bad_sents_1), w, focal_agent=fa)
    assert exc_1.value.check_id == "c_fidelity_missing_event"

    # 2. Enter event with reversed from_location / to_location
    enter_idx = next(i for i, ev in enumerate(w.events) if ev.kind == "enter")
    enter_ev = w.events[enter_idx]
    rev_enter_sent = (
        f"Next, {enter_ev.actor} walked through the doorway from the {enter_ev.to_location} into the {enter_ev.from_location}."
    )
    bad_sents_2 = list(ev_sents)
    bad_sents_2[enter_idx] = rev_enter_sent
    with pytest.raises(RenderCheckError, match="missing or altered event") as exc_2:
        check_rendered_prose(setup_p + "\n\n" + " ".join(bad_sents_2), w, focal_agent=fa)
    assert exc_2.value.check_id == "c_fidelity_missing_event"


def test_29_prose_fidelity_wrong_room_refused() -> None:
    """Test 29 (Change 3.iii): Naming another room of the scenario as the room of an intra-room event (even if ev.room appears incidentally) must be refused."""
    w = generate_world(make_balanced_configs(2, base_seed=20261030)[0])
    fa = w.agents[0]
    clean_prose = render_world_template(w, fa)
    setup_p, events_p = clean_prose.split("\n\n")
    from udlab_generator.render import _split_prose_sentences

    ev_sents = _split_prose_sentences(events_p)
    move_idx = next(
        i for i, ev in enumerate(w.events) if ev.kind == "move" and ev.target_object == w.primary_object
    )
    move_ev = w.events[move_idx]
    wrong_room = next(r for r in w.rooms if r != move_ev.room)

    # Even though move_ev.room is mentioned in a trailing phrase ("near the sorting room"),
    # the sentence places the action in `wrong_room`.
    wrong_room_sent = (
        f"Next, in the {wrong_room} near the {move_ev.room}, {move_ev.actor} moved the {move_ev.target_object} "
        f"from the {move_ev.from_location} to the {move_ev.to_location} in open view at the container bay."
    )
    bad_sents = list(ev_sents)
    bad_sents[move_idx] = wrong_room_sent
    with pytest.raises(RenderCheckError, match="missing or altered event") as exc_wr:
        check_rendered_prose(setup_p + "\n\n" + " ".join(bad_sents), w, focal_agent=fa)
    assert exc_wr.value.check_id == "c_fidelity_missing_event"


def test_30_prose_fidelity_unlogged_event_starting_at_the_start_refused() -> None:
    """Test 30 (Change 3.iv): An unlogged event sentence starting 'At the start,' in the events paragraph must not bypass check_prose_render_fidelity."""
    w = generate_world(make_balanced_configs(2, base_seed=20261030)[0])
    fa = w.agents[0]
    clean_prose = render_world_template(w, fa)

    sneaky_unlogged = (
        f"{clean_prose} At the start, in the {w.rooms[0]}, {fa} moved the {w.objects[0]} "
        f"from the {w.containers_by_room[w.rooms[0]][0]} to the {w.containers_by_room[w.rooms[0]][1]} in open view."
    )
    with pytest.raises(RenderCheckError, match="unlogged event sentence") as exc_unlogged:
        check_rendered_prose(sneaky_unlogged, w, focal_agent=fa)
    assert exc_unlogged.value.check_id == "c_fidelity_unlogged_event"


def _build_natural_model_prose_scen_0000() -> tuple[str, str]:
    """Return `(natural_setup, natural_events)` as `qwen3.6:35b` would plausibly write when following the prompt:
    grouped starting agents, natural doorway phrasing, verbatim goals/stakes, and attribution after quoted speech.
    """
    natural_setup = (
        "In the archival registry, the narrative follows Eva across four interconnected rooms: "
        "the vault room contains the oak drawer and the steel cabinet at its container bay and the wall barometer in the far-side alcove, where a frosted partition "
        "stands between the far-side alcove and the container bay; "
        "the reading room contains the slate tray and the cedar shelf at its container bay and the wall clock in the far-side alcove, divided from the far-side alcove "
        "by a tall bookstack; "
        "the bindery room contains the linen bin and the copper rack in the container bay and the humidity dial in the far-side alcove, separated from the far-side alcove "
        "by a canvas screen; "
        "and the sorting room contains the maple cubby and the iron safe in the container bay and the ventilation grille in the far-side alcove, with a rolling pillar "
        "between the far-side alcove and the container bay so that anyone at the container bay sees the containers in open view "
        "while anyone in the far-side alcove has their view blocked. "
        "Open doorways link the vault room and the reading room, the reading room and the bindery room, "
        "and the bindery room and the sorting room, with a doorway hearing threshold of 50 dB. "
        "Each agent assumes an object remains where that agent last saw it in open view or heard a witness state it was placed, "
        "and assumes another agent looks where they last saw or communicated that agent perceiving it. "
        "At the start, Eva, Dan, and Cara start in the bindery room, while Gia remains absent off-site. "
        "Eva plans to complete the quarterly audit packet, Dan plans to verify the municipal deed transfer, "
        "and Cara plans to catalog the incoming charter folio. "
        "Gia is off-site and depends on the location of the brass ledger for tomorrow's courier handoff."
    )

    natural_events = (
        "In the bindery room, Eva checked the index reel inside the linen bin in open view. "
        "Dan then left the bindery room for the sorting room. "
        "Moments later, Dan entered the sorting room from the bindery room. "
        "Cara also departed the bindery room toward the sorting room. "
        "Cara entered the sorting room from the bindery room right after Dan. "
        "While in the sorting room, Cara inspected the maple cubby at the container bay in open view. "
        "In the sorting room, Cara transferred the brass ledger from the intake bench into the iron safe in open view. "
        "Back in the bindery room, Eva exited toward the sorting room. "
        "Eva arrived in the sorting room from the bindery room. "
        "Inside the sorting room, Cara shifted the brass ledger from the iron safe to the maple cubby in open view. "
        "In the sorting room, Eva noticed the ventilation grille in the far-side alcove in open view. "
        "Cara then left the sorting room heading back to the bindery room. "
        "Cara walked from the sorting room into the bindery room. "
        "Continuing through the bindery room, Cara exited toward the reading room. "
        "Cara entered the reading room from the bindery room. "
        "Meanwhile, in the sorting room, Dan stepped from the container bay behind the rolling pillar into the far-side alcove. "
        "From the far-side alcove in the sorting room, Dan looked toward the iron safe, but the rolling pillar blocked his view. "
        "Still in the far-side alcove of the sorting room, Dan looked toward the maple cubby, yet the rolling pillar blocked the sightline. "
        "Over in the reading room, Cara spotted the index reel resting in the slate tray in open view. "
        "In the sorting room, Eva departed for the bindery room. "
        "Eva came from the sorting room into the bindery room. "
        "Inside the reading room, Cara placed the seal stamp from the slate tray onto the cedar shelf in open view. "
        '"The seal stamp is stored in the cedar shelf." Cara announced at 60 dB in the reading room. '
        "In the bindery room, Eva looked at the humidity dial in the far-side alcove in open view. "
        '"Update for Gia: the brass ledger is stored at the maple cubby," Cara texted to Gia from the reading room. '
        "In the bindery room, Eva saw the unsigned transfer slip in the copper rack in open view. "
        "In the sorting room, Dan turned the overhead louver from latch position A to latch position B in open view. "
        '"I adjusted the overhead louver to clear binding dust from the bench." Dan stated at 20 dB in the sorting room. '
        "From the far-side alcove in the sorting room, Dan glanced toward the iron safe, though the rolling pillar blocked the container bay. "
        '"I am certain the brass ledger was moved into the copper rack," Dan claimed at 20 dB in the sorting room.'
    )
    return natural_setup, natural_events


def test_31_handwritten_realistic_natural_prose_passes() -> None:
    """Test 31 (Decision 4): Realistic natural prose as a model would plausibly produce when following the new prompt
    (grouped starting agents, natural doorway phrasing, attribution after quoted speech) must pass check_rendered_prose.
    """
    w = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    assert w.scenario_id == "scen_0000"
    fa = w.agents[0]  # Eva

    handwritten_setup, handwritten_events = _build_natural_model_prose_scen_0000()
    handwritten_prose = f"{handwritten_setup}\n\n{handwritten_events}"
    assert "connects to the" not in handwritten_setup
    assert "Eva, Dan, and Cara start in the bindery room" in handwritten_setup
    assert '"The seal stamp is stored in the cedar shelf." Cara announced' in handwritten_events
    assert handwritten_prose != render_world_template(w, fa)
    assert check_rendered_prose(handwritten_prose, w, focal_agent=fa) == ()


def test_32_change1_fifty_random_private_seeds_pass_sealing(tmp_path: Path) -> None:
    """Test 32 (Change 1): 50 random private_seed_hex values at --count 10 pass write_sealed_dataset + verify_sealed_part with zero SealingError."""
    import hashlib

    configs_10 = make_balanced_configs(10, base_seed=20261030)
    mw = MockWriter()
    pairs_10 = [mw.render_scenario_pair(generate_world(c)) for c in configs_10]
    pub_10 = tmp_path / "c1_pub"
    seal_10 = tmp_path / "c1_seal"
    salts_used: list[int] = []
    for i in range(50):
        seed_i = hashlib.sha256(f"change1_seed_{i}".encode("utf-8")).hexdigest()
        receipt_i = write_sealed_dataset(pairs_10, pub_10, seal_10, private_seed_hex=seed_i)
        verify_sealed_part(pub_10, seal_10)
        meta_i = json.loads(receipt_i.sealing_meta_path.read_text(encoding="utf-8"))
        salts_used.append(int(meta_i.get("shuffle_salt_index", -1)))
    assert all(s >= 0 for s in salts_used), "shuffle_salt_index must be recorded in sealing.sealed.json"
    assert any(s > 0 for s in salts_used), "At least one of the 50 seeds must have needed a non-zero resample salt."


def test_33_changes_2_4_5_6_think_refusals_cleanup_and_verify(tmp_path: Path) -> None:
    """Test 33 (Changes 2, 4, 5, 6):
    - Change 2: Default think=False, num_predict=4096, max_num_ctx=9216; --think allows num_predict=5120 inside 9216; >9216 refused; think recorded in generations.jsonl and sealed provenance.
    - Change 4: Prose-check and length_stop refusals recorded with checks_passed=False and refusal_reasons in generations.jsonl on disk.
    - Change 5: Owned server default when http_transport is None, keep_alive=0 allowed after VRAM halt, kill_pid_tree called on stop(), and init failure cleans up server.
    - Change 6: --verify-written fails if generations.jsonl is missing, short, has bad digest, or has done_reason != 'stop'.
    """
    from udlab_generator.cli import build_and_probe_dataset, verify_written_dataset
    import udlab_generator.render as ug_render
    from udlab_generator.render import (
        DEFAULT_WRITER_MAX_NUM_CTX,
        DEFAULT_WRITER_MODEL_PIN,
        DEFAULT_WRITER_NUM_PREDICT,
        MaxNumCtxRefusalError,
        OllamaWriter,
    )

    configs_10 = make_balanced_configs(10, base_seed=20261030)

    # --- Change 2: Option A defaults (think=False, num_predict=4096, max_num_ctx=9216) and --think flag ---
    assert getattr(ug_render, "DEFAULT_WRITER_THINK", None) is False
    assert DEFAULT_WRITER_NUM_PREDICT == 4096
    assert DEFAULT_WRITER_MAX_NUM_CTX == 9216
    assert getattr(ug_render, "DEFAULT_WRITER_THINK_NUM_PREDICT", None) == 5120
    assert DEFAULT_WRITER_MODEL_PIN.think is False

    # Raising max_num_ctx above 9216 is always refused
    with pytest.raises(MaxNumCtxRefusalError, match="9216"):
        OllamaWriter(
            max_num_ctx=10240,
            grant_id="test-grant",
            http_transport=lambda m, u, b: {},
        )
    with pytest.raises(MaxNumCtxRefusalError, match="9216"):
        build_and_probe_dataset(
            count=2,
            output_dir=tmp_path / "bad_ctx",
            writer_kind="mock",
            max_num_ctx=10240,
        )

    # With --think on, num_predict=5120 fits within 9216
    w0 = generate_world(configs_10[0])
    pinned_digest = DEFAULT_WRITER_MODEL_PIN.manifest_digest
    seen_chat_payloads: list[dict] = []

    def _recording_transport(method: str, url: str, body: dict | None = None) -> dict:
        if method == "GET" and url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3.6:35b", "digest": pinned_digest}]}
        if method == "POST" and url.endswith("/api/chat"):
            payload = dict(body or {})
            if payload.get("keep_alive") == 0:
                return {"model": "qwen3.6:35b", "done": True, "done_reason": "unload"}
            seen_chat_payloads.append(payload)
            prompt_content = payload["messages"][-1]["content"]
            focal = prompt_content.split("Focal Agent: ", 1)[1].splitlines()[0].strip()
            return {
                "model": "qwen3.6:35b",
                "message": {"role": "assistant", "content": render_world_template(w0, focal), "thinking": "ok"},
                "done": True,
                "done_reason": "stop",
                "eval_count": 400,
                "prompt_eval_count": 1600,
            }
        raise AssertionError(f"Unexpected {method} {url}")

    think_writer = OllamaWriter(
        think=True,
        grant_id="test-grant",
        http_transport=_recording_transport,
    )
    assert think_writer.think is True
    assert think_writer.num_predict == 5120
    assert think_writer.max_num_ctx == 9216
    # Card ue: a think call fits only when prompt tokens + 5120 + 256 <= 9216. At dfcbe6a that held for 1 of the
    # 10 balanced worlds (this one, by 21 tokens); the station-change and occluded-see clauses add about 190
    # tokens, so it now holds for none. Card ul: without a tokenizer the bytes/3 estimate decides, the
    # over-budget think call is refused as c_ctx_over_cap, and because this direct caller forbids refusal the
    # writer raises CtxOverCapRefusalError (a MaxNumCtxRefusalError). The build loop never raises it (test 89).
    # The think payload is exercised below with an explicit num_predict that fits.
    from udlab_generator.render import CTX_OVER_CAP_CHECK_ID, CtxOverCapRefusalError, estimate_prompt_tokens

    w0_prompt_tokens = estimate_prompt_tokens(
        [{"role": "user", "content": format_writer_prompt(w0, w0.agents[0])}]
    )
    assert w0_prompt_tokens + DEFAULT_WRITER_NUM_PREDICT + 256 <= DEFAULT_WRITER_MAX_NUM_CTX
    assert w0_prompt_tokens + think_writer.num_predict + 256 > think_writer.max_num_ctx
    with pytest.raises(MaxNumCtxRefusalError, match="9216") as ctx_exc:
        think_writer.render_scenario_pair(w0)
    assert isinstance(ctx_exc.value, CtxOverCapRefusalError)
    assert ctx_exc.value.check_id == CTX_OVER_CAP_CHECK_ID == "c_ctx_over_cap"
    assert think_writer.refusals[-1].check_id == CTX_OVER_CAP_CHECK_ID
    assert think_writer.call_records[-1].check_id == CTX_OVER_CAP_CHECK_ID
    assert think_writer.call_records[-1].prompt_tokens_exact is None
    assert not seen_chat_payloads
    think_writer_fit = OllamaWriter(
        think=True,
        num_predict=DEFAULT_WRITER_NUM_PREDICT,
        grant_id="test-grant",
        http_transport=_recording_transport,
    )
    assert think_writer_fit.think is True and think_writer_fit.max_num_ctx == 9216
    think_writer_fit.render_scenario_pair(w0)
    assert seen_chat_payloads[-1]["think"] is True
    assert seen_chat_payloads[-1]["options"]["num_predict"] == DEFAULT_WRITER_NUM_PREDICT
    assert seen_chat_payloads[-1]["options"]["num_ctx"] == 9216

    # --- Change 4: Prose-check and length_stop refusals persisted in generations.jsonl ---
    bad_prose_gen = tmp_path / "bad_prose_generations.jsonl"

    def _bad_prose_transport(method: str, url: str, body: dict | None = None) -> dict:
        if method == "GET" and url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3.6:35b", "digest": pinned_digest}]}
        if method == "POST" and url.endswith("/api/chat"):
            return {
                "model": "qwen3.6:35b",
                "message": {"role": "assistant", "content": "Eva furtively hid the ledger [F1]."},
                "done": True,
                "done_reason": "stop",
                "eval_count": 12,
                "prompt_eval_count": 1600,
            }
        raise AssertionError(f"Unexpected {method} {url}")

    bad_prose_writer = OllamaWriter(
        grant_id="test-grant",
        http_transport=_bad_prose_transport,
        generations_path=bad_prose_gen,
    )
    with pytest.raises(RenderCheckError):
        bad_prose_writer.render_scenario_pair(w0)
    persisted_bad = [json.loads(line) for line in bad_prose_gen.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(persisted_bad) == 1
    assert persisted_bad[0]["checks_passed"] is False
    assert len(persisted_bad[0]["refusal_reasons"]) == 1
    assert "b_banned_adverbs" in persisted_bad[0]["refusal_reasons"][0]

    # --- Change 5: Server unload after VRAM halt + PID-tree kill on stop() + init error cleanup ---
    unloaded_after_halt = False
    killed_pids_c5: list[int] = []

    class _DummyProc:
        pid = 54321
        returncode = None

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            return 0

    def _halt_unload_transport(method: str, url: str, body: dict | None = None) -> dict:
        nonlocal unloaded_after_halt
        if method == "GET" and url.endswith("/api/version"):
            return {"version": "0.6.2"}
        if method == "GET" and url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3.6:35b", "digest": pinned_digest}]}
        if method == "POST" and url.endswith("/api/chat"):
            payload = dict(body or {})
            if payload.get("keep_alive") == 0:
                unloaded_after_halt = True
                return {"model": "qwen3.6:35b", "done": True, "done_reason": "unload"}
            return {"model": "qwen3.6:35b", "message": {"content": "x"}, "done": True}
        raise AssertionError(f"Unexpected {method} {url}")

    halt_w = OllamaWriter(
        grant_id="test-grant",
        vram_halt_mib=26000,
        vram_probe=lambda: 26500,
        pid_killer=lambda p: killed_pids_c5.append(p),
        http_transport=_halt_unload_transport,
        start_owned_server=True,
        popen_factory=lambda *a, **kw: _DummyProc(),
        server_port=11498,
    )
    with pytest.raises(RuntimeError, match="VRAM_HALT_EXCEEDED"):
        halt_w.render_scenario_pair(w0)
    halt_w.close()
    assert unloaded_after_halt is True, "keep_alive: 0 unload must succeed during close() even after a VRAM halt."
    assert 54321 in killed_pids_c5

    # Init error after server launch must stop the owned server
    init_killed_pids: list[int] = []
    with pytest.raises(ValueError, match="vram_halt_mib must be positive"):
        OllamaWriter(
            grant_id="test-grant",
            vram_halt_mib=0,
            pid_killer=lambda p: init_killed_pids.append(p),
            http_transport=_halt_unload_transport,
            start_owned_server=True,
            popen_factory=lambda *a, **kw: _DummyProc(),
            server_port=11497,
        )
    assert 54321 in init_killed_pids

    # --- Change 6: Validate generations.jsonl in --verify-written ---
    out_c6 = tmp_path / "c6_run"
    assert cli_main(["--write", "--writer", "mock", "--count", "10", "--out-dir", str(out_c6)]) == 0
    sealing_doc = json.loads((out_c6 / "sealed_store" / "sealing.sealed.json").read_text(encoding="utf-8"))
    assert sealing_doc["writer_provenance"]["think"] is False
    assert cli_main(["--verify-written", str(out_c6)]) == 0

    # Corrupting done_reason or deleting generations.jsonl must fail --verify-written
    gen_p = out_c6 / "generations.jsonl"
    orig_gen_text = gen_p.read_text(encoding="utf-8")
    gen_p.write_text(orig_gen_text.replace('"done_reason":"stop"', '"done_reason":"length"', 1), encoding="utf-8")
    assert cli_main(["--verify-written", str(out_c6)]) == 1
    assert verify_written_dataset(out_c6)["passed"] is False

    gen_p.unlink()
    assert cli_main(["--verify-written", str(out_c6)]) == 1
    assert verify_written_dataset(out_c6)["passed"] is False

    gen_p.write_text(orig_gen_text, encoding="utf-8")
    assert cli_main(["--verify-written", str(out_c6)]) == 0


def test_34_natural_prose_with_deliberate_wrong_room_fails() -> None:
    """Test 34 (Decision 4): Second natural-prose test (grouped agents, natural doorway phrasing,
    attribution after quote) with one deliberate fault (a wrong room on event #22) must fail.
    """
    w = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    fa = w.agents[0]
    natural_setup, natural_events = _build_natural_model_prose_scen_0000()

    faulty_events = natural_events.replace(
        "Inside the reading room, Cara placed the seal stamp from the slate tray onto the cedar shelf in open view.",
        "Inside the vault room, Cara placed the seal stamp from the slate tray onto the cedar shelf in open view.",
        1,
    )
    assert faulty_events != natural_events
    with pytest.raises(RenderCheckError, match=r"missing or altered event #22") as exc_info:
        check_rendered_prose(f"{natural_setup}\n\n{faulty_events}", w, focal_agent=fa)
    assert exc_info.value.check_id == "c_fidelity_missing_event"


def test_35_phrasing_1_natural_doorway_sentences_pass() -> None:
    """Test 35 (Decision 2 & 4, phrasing 1): Natural doorway phrasings mentioning both rooms of each edge
    in one setup sentence (without 'connects to the') pass, while omitting an edge refuses with c_fidelity_doorways.
    """
    w = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    fa = w.agents[0]
    natural_setup, natural_events = _build_natural_model_prose_scen_0000()

    # Phrasing 1a from review §2(1): "doorways link the vault room, reading room, bindery room, and sorting room"
    setup_1a = natural_setup.replace(
        "Open doorways link the vault room and the reading room, the reading room and the bindery room, "
        "and the bindery room and the sorting room, with a doorway hearing threshold of 50 dB.",
        "Along the corridor, doorways link the vault room, reading room, bindery room, and sorting room, "
        "with a doorway hearing threshold of 50 dB.",
    )
    assert "connects to the" not in setup_1a
    assert check_rendered_prose(f"{setup_1a}\n\n{natural_events}", w, focal_agent=fa) == ()

    # Phrasing 1b from review §2(1): separate sentences per doorway using natural verbs
    setup_1b = natural_setup.replace(
        "Open doorways link the vault room and the reading room, the reading room and the bindery room, "
        "and the bindery room and the sorting room, with a doorway hearing threshold of 50 dB.",
        "A doorway connects the vault room and the reading room (doorway hearing threshold 50 dB). "
        "A second doorway links the reading room with the bindery room. "
        "A third doorway opens between the bindery room and the sorting room.",
    )
    assert "connects to the" not in setup_1b
    assert check_rendered_prose(f"{setup_1b}\n\n{natural_events}", w, focal_agent=fa) == ()

    # Omitting one edge (bindery room <-> sorting room) from any setup sentence must still fail
    setup_missing_edge = (
        "In the archival registry, Eva works across four rooms: "
        "the vault room contains the oak drawer and the steel cabinet at its container bay and the wall barometer in the far-side alcove with a frosted partition "
        "between the far-side alcove and the container bay. "
        "The reading room contains the slate tray and the cedar shelf at its container bay and the wall clock in the far-side alcove with a tall bookstack "
        "between the far-side alcove and the container bay. "
        "The bindery room contains the linen bin and the copper rack at its container bay and the humidity dial in the far-side alcove with a canvas screen "
        "between the far-side alcove and the container bay. "
        "The sorting room contains the maple cubby and the iron safe at its container bay and the ventilation grille in the far-side alcove with a rolling pillar "
        "between the far-side alcove and the container bay. "
        "A doorway links the vault room and the reading room, and another doorway links the reading room and the bindery room "
        "(doorway hearing threshold 50 dB). "
        "Each agent assumes an object remains where that agent last saw it in open view or heard a witness state it was placed, "
        "and assumes another agent looks where they last saw or communicated that agent perceiving it. "
        "At the start, Eva, Dan, and Cara start in the bindery room, while Gia is off-site. "
        "Eva plans to complete the quarterly audit packet; Dan plans to verify the municipal deed transfer; "
        "Cara plans to catalog the incoming charter folio. "
        "Gia is off-site and depends on the location of the brass ledger for tomorrow's courier handoff."
    )
    with pytest.raises(RenderCheckError, match="doorway") as exc_dw:
        check_rendered_prose(f"{setup_missing_edge}\n\n{natural_events}", w, focal_agent=fa)
    assert exc_dw.value.check_id == "c_fidelity_doorways"


def test_36_phrasing_2_grouped_and_natural_starting_rooms_pass() -> None:
    """Test 36 (Decision 2 & 4, phrasing 2): Grouped starting-room phrasings ('Eva, Dan and Cara start in the bindery room',
    'At the start, Eva, Dan, and Cara are all in the bindery room', 'Eva begins in the bindery room') pass when each agent
    and their starting room appear in the same setup sentence.
    """
    w = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    fa = w.agents[0]
    base_prose = render_world_template(w, fa)
    natural_setup, natural_events = _build_natural_model_prose_scen_0000()

    for grouped_clause in (
        "At the start, Eva, Dan and Cara start in the bindery room.",
        "At the start, Eva, Dan, and Cara are all in the bindery room.",
        "At the start, Eva begins in the bindery room alongside Dan and Cara.",
    ):
        prose_variant = base_prose.replace(
            "At the start, Eva starts in the bindery room, Dan starts in the bindery room, Cara starts in the bindery room.",
            grouped_clause,
        )
        assert "eva starts in the bindery room" not in prose_variant.lower()
        assert check_rendered_prose(prose_variant, w, focal_agent=fa) == ()

    # Assign Cara to the wrong starting room in a separate sentence -> must fail with c_fidelity_initial_room
    bad_init_setup = natural_setup.replace(
        "At the start, Eva, Dan, and Cara start in the bindery room, while Gia remains absent off-site.",
        "At the start, Eva and Dan start in the bindery room. Cara starts in the vault room, while Gia remains absent off-site.",
    )
    with pytest.raises(RenderCheckError, match="initial room 'bindery room' for agent 'Cara'") as exc_ir:
        check_rendered_prose(f"{bad_init_setup}\n\n{natural_events}", w, focal_agent=fa)
    assert exc_ir.value.check_id == "c_fidelity_initial_room"


def test_37_phrasing_3_verbatim_goals_and_stakes_prompt_and_check() -> None:
    """Test 37 (Decision 2 & 4, phrasing 3): The writer prompt explicitly instructs copying each goal_text and stake
    description verbatim, natural setup prose carrying them verbatim passes, and paraphrasing either is refused.
    """
    w = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    fa = w.agents[0]
    prompt_str = format_writer_prompt(w, fa)
    assert "Copy each agent's goal_text (goals[agent].goal_text) verbatim" in prompt_str
    assert "depends on" in prompt_str and "dependency phrase" in prompt_str

    natural_setup, natural_events = _build_natural_model_prose_scen_0000()
    assert check_rendered_prose(f"{natural_setup}\n\n{natural_events}", w, focal_agent=fa) == ()

    paraphrased_goal_setup = natural_setup.replace(
        "Eva plans to complete the quarterly audit packet",
        "Eva's objective is finishing the quarterly audit packet",
    )
    with pytest.raises(RenderCheckError, match="goal") as exc_g:
        check_rendered_prose(f"{paraphrased_goal_setup}\n\n{natural_events}", w, focal_agent=fa)
    assert exc_g.value.check_id == "c_fidelity_goal"

    paraphrased_stake_setup = natural_setup.replace(
        "Gia is off-site and depends on the location of the brass ledger for tomorrow's courier handoff.",
        "Gia is away off-site and needs to know where the brass ledger is kept for tomorrow's courier.",
    )
    with pytest.raises(RenderCheckError, match="stake edge") as exc_s:
        check_rendered_prose(f"{paraphrased_stake_setup}\n\n{natural_events}", w, focal_agent=fa)
    assert exc_s.value.check_id == "c_fidelity_stake_edge"


def test_38_phrasing_4_period_in_quoted_speech_with_post_quote_attribution_passes() -> None:
    """Test 38 (Decision 3 & 4, phrasing 4): A period inside quoted speech with attribution after the quote
    must not split the sentence in _split_prose_sentences or fail check_rendered_prose.
    """
    from udlab_generator.render import _split_prose_sentences

    w = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    fa = w.agents[0]
    natural_setup, natural_events = _build_natural_model_prose_scen_0000()

    # Check _split_prose_sentences directly on post-quote attribution with internal periods
    sample_para = (
        '"The seal stamp is stored in the cedar shelf." Cara said at 60 dB in the reading room. '
        '"Listen carefully. Update for Gia: the brass ledger is stored at the maple cubby." '
        "Cara texted to Gia from the reading room."
    )
    sents = _split_prose_sentences(sample_para)
    assert len(sents) == 2
    assert sents[0] == '"The seal stamp is stored in the cedar shelf." Cara said at 60 dB in the reading room.'
    assert (
        sents[1]
        == '"Listen carefully. Update for Gia: the brass ledger is stored at the maple cubby." Cara texted to Gia from the reading room.'
    )

    # Full scenario prose where all 4 say/message events have periods inside the quote and attribution after the quote
    events_all_period_quotes = natural_events.replace(
        '"Update for Gia: the brass ledger is stored at the maple cubby," Cara texted to Gia from the reading room.',
        '"Update for Gia: the brass ledger is stored at the maple cubby." Cara texted to Gia from the reading room.',
    ).replace(
        '"I am certain the brass ledger was moved into the copper rack," Dan claimed at 20 dB in the sorting room.',
        '"I am certain the brass ledger was moved into the copper rack." Dan claimed at 20 dB in the sorting room.',
    )
    assert len(_split_prose_sentences(events_all_period_quotes)) == len(w.events) == 30
    assert check_rendered_prose(f"{natural_setup}\n\n{events_all_period_quotes}", w, focal_agent=fa) == ()


def test_39_prompt_and_checker_requirements_list_aligned() -> None:
    """Test 39 (Decision 1): The on-disk prompt file keeps only non-constraint text plus a placeholder;
    format_writer_prompt generates HARD CONSTRAINTS from WRITER_PROSE_REQUIREMENTS (interpolating
    _ACTION_VERBS_BY_KIND, BANNED_ADVERBS, and BELIEF_PERSISTENCE_RULE); and every check_id the checker
    can raise (derived from the AST code path) maps to a rule rendered in the prompt and the reverse.
    """
    import udlab_generator.render as ug_render
    from udlab_generator.world import BELIEF_PERSISTENCE_RULE

    parse_fn = getattr(ug_render, "parse_writer_prompt_requirements", None)
    enforced_fn = getattr(ug_render, "get_checker_enforced_check_ids", None)
    verify_fn = getattr(ug_render, "verify_prompt_and_checker_requirements_aligned", None)
    render_constraints_fn = getattr(ug_render, "render_writer_prompt_constraints", None)
    placeholder = getattr(ug_render, "HARD_CONSTRAINTS_PLACEHOLDER", None)
    shared_reqs = getattr(ug_render, "WRITER_PROSE_REQUIREMENTS", None)
    action_verbs_by_kind = getattr(ug_render, "_ACTION_VERBS_BY_KIND", None)
    assert callable(parse_fn) and callable(enforced_fn) and callable(verify_fn) and callable(render_constraints_fn)
    assert isinstance(placeholder, str) and placeholder == "{hard_constraints_block}"
    assert isinstance(shared_reqs, tuple) and len(shared_reqs) >= 25
    assert isinstance(action_verbs_by_kind, dict)

    # 1. On-disk prompt file keeps only the non-constraint text plus the placeholder (no hand-copied `- (<check_id>)` lines)
    raw_disk_prompt = ug_render.WRITER_PROMPT_PATH.read_text(encoding="utf-8")
    assert placeholder in raw_disk_prompt
    for line in raw_disk_prompt.splitlines():
        assert not line.strip().startswith("- ("), (
            f"On-disk prompt must not keep hand-copied constraint lines; found: {line!r}"
        )

    # 2. Verify verify_prompt_and_checker_requirements_aligned() succeeds on the on-disk placeholder file
    assert verify_fn() == shared_reqs

    # 3. Format writer prompt and verify HARD CONSTRAINTS are generated from WRITER_PROSE_REQUIREMENTS
    #    interpolating _ACTION_VERBS_BY_KIND, BANNED_ADVERBS, and BELIEF_PERSISTENCE_RULE
    w0 = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    formatted = format_writer_prompt(w0, w0.agents[0])
    assert placeholder not in formatted
    assert render_constraints_fn() in formatted
    assert BELIEF_PERSISTENCE_RULE in formatted
    for adv in BANNED_ADVERBS:
        assert adv in formatted
    for kind, verbs in action_verbs_by_kind.items():
        assert f"{kind}: {', '.join(verbs)}" in formatted

    # 4. Every check_id the checker can raise (derived from AST code path) maps to a rule rendered in the prompt, and the reverse
    parsed_reqs = parse_fn(formatted)
    assert parsed_reqs == shared_reqs
    parsed_by_id = {r.check_id: r.rule_text for r in parsed_reqs}
    enforced_ids = enforced_fn()
    assert len(enforced_ids) == len(set(enforced_ids)) == len(shared_reqs)
    for cid in enforced_ids:
        assert cid in parsed_by_id and parsed_by_id[cid], f"Checker check_id {cid!r} missing from rendered prompt."
    for cid in parsed_by_id:
        assert cid in enforced_ids, f"Prompt rule {cid!r} not enforced by checker code path."

    # 5. Diverging prompt (removing a requirement, altering rule text, or adding an untracked requirement) must fail
    dropped_one = "\n".join(
        line for line in formatted.splitlines() if "(c_fidelity_doorways)" not in line
    )
    with pytest.raises(ValueError, match="diverge"):
        verify_fn(dropped_one)

    altered_text = formatted.replace(
        "Copy each agent's goal_text", "Paraphrase each agent's goal_text"
    )
    with pytest.raises(ValueError, match="diverge"):
        verify_fn(altered_text)

    extra_untracked = formatted.replace(
        "HARD CONSTRAINTS (MANDATORY):\n",
        "HARD CONSTRAINTS (MANDATORY):\n- (untracked_extra_rule) Use exact template phrasing.\n",
    )
    with pytest.raises(ValueError, match="diverge"):
        verify_fn(extra_untracked)


def test_40_verify_written_checks_every_generations_record(tmp_path: Path) -> None:
    """Test 40 (Decision 5): --verify-written checks every generations.jsonl record, not only the last per rendering."""
    from udlab_generator.cli import verify_written_dataset

    out_dir = tmp_path / "verify_all_gen_lines"
    assert cli_main(["--write", "--writer", "mock", "--count", "10", "--out-dir", str(out_dir)]) == 0
    assert cli_main(["--verify-written", str(out_dir)]) == 0

    gen_path = out_dir / "generations.jsonl"
    orig_lines = [line for line in gen_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(orig_lines) == 20

    # Prepend an earlier refused record for (scen_0000, r0) while keeping the final 20 passing records intact
    early_failed_rec = json.loads(orig_lines[0])
    early_failed_rec["call_id"] = "scen_0000:r0:call_0000_failed"
    early_failed_rec["checks_passed"] = False
    early_failed_rec["refusal_reasons"] = ["c_fidelity_doorways: doorway missing"]
    gen_path.write_text(
        json.dumps(early_failed_rec, sort_keys=True) + "\n" + "\n".join(orig_lines) + "\n",
        encoding="utf-8",
    )

    rep_1 = verify_written_dataset(out_dir)
    assert rep_1["passed"] is False
    assert rep_1["generations_valid"] is False
    assert any("Line 1" in err and "did not pass prose checks" in err for err in rep_1["generations_errors"])
    assert cli_main(["--verify-written", str(out_dir)]) == 1

    # Prepend an earlier length_stop record for (scen_0000, r0) while keeping final 20 passing records intact
    early_len_rec = json.loads(orig_lines[0])
    early_len_rec["call_id"] = "scen_0000:r0:call_0000_length"
    early_len_rec["done_reason"] = "length"
    gen_path.write_text(
        json.dumps(early_len_rec, sort_keys=True) + "\n" + "\n".join(orig_lines) + "\n",
        encoding="utf-8",
    )
    rep_2 = verify_written_dataset(out_dir)
    assert rep_2["passed"] is False
    assert rep_2["generations_valid"] is False
    assert any("Line 1" in err and "done_reason='length'" in err for err in rep_2["generations_errors"])
    assert cli_main(["--verify-written", str(out_dir)]) == 1


def _build_sj_writer_in_the_loop_scen_0000() -> tuple[str, list[str]]:
    """Return `(sj_setup, sj_event_sentences)` for `scen_0000` starting with sj §2's exact setup paragraph
    and first five event sentences, extended to all 30 events in the same style (including the four sj traps:
    unoccluded see of a container without 'container bay', broadened verbs like 'examined'/'relocated'/'carried'/
    'took'/'proceeded'/'crossed', prepositions from/out of and to/into/toward/at/in/onto).
    """
    sj_setup = (
        "Following Eva through the archival registry, the workspace spans four rooms: the vault room, "
        "the reading room, the bindery room, and the sorting room. In the vault room, the oak drawer and "
        "the steel cabinet sit at the container bay and the wall barometer in the far-side alcove, while a frosted partition stands between the far-side "
        "alcove and the container bay. In the reading room, the slate tray and the cedar shelf occupy the "
        "container bay and the wall clock sits in the far-side alcove, with a tall bookstack standing between the far-side alcove and the container bay. "
        "In the bindery room, the linen bin and the copper rack are located at the container bay and the humidity dial in the far-side alcove, where a "
        "canvas screen stands between the far-side alcove and the container bay. In the sorting room, the "
        "maple cubby and the iron safe rest at the container bay and the ventilation grille in the far-side alcove, and a rolling pillar stands between the "
        "far-side alcove and the container bay; in each room, the occluder blocks sight from the far-side "
        "alcove while the container bay sees the containers in open view. Doorways connect the vault room "
        "with the reading room, link the reading room to the bindery room, and join the bindery room to the "
        "sorting room, with a doorway hearing threshold of 50 dB so speech at or above 50 dB carries through "
        "one doorway while quieter speech stays in the room. Eva, Dan, and Cara all start in the bindery room, "
        "whereas Gia is absent off-site. Eva plans to complete the quarterly audit packet; Dan plans to verify "
        "the municipal deed transfer; Cara plans to catalog the incoming charter folio. Gia is off-site and "
        "depends on the location of the brass ledger for tomorrow's courier handoff. Each agent assumes an "
        "object remains where that agent last saw it in open view or heard a witness state it was placed, and "
        "assumes another agent looks where they last saw or communicated that agent perceiving it."
    )

    sj_event_sentences = [
        # Events 1..5 verbatim from sj §2:
        "In the bindery room, Eva observed the index reel in the linen bin in open view.",
        "Dan left the bindery room and headed toward the sorting room.",
        "Dan entered the sorting room from the bindery room.",
        "Cara departed from the bindery room for the sorting room.",
        "Cara walked from the bindery room into the sorting room.",
        # Events 6..30 extended in the same natural style (exercising sj traps (a)-(c)):
        "In the sorting room, Cara saw the maple cubby in open view.",
        "In the sorting room, Cara relocated the brass ledger from the intake bench into the iron safe in open view.",
        "Eva exited the bindery room and proceeded toward the sorting room.",
        "Eva crossed from the bindery room into the sorting room.",
        "In the sorting room, Cara carried the brass ledger out of the iron safe to the maple cubby in open view.",
        "In the sorting room, Eva examined the ventilation grille in open view.",
        "Cara left the sorting room and proceeded to the bindery room.",
        "Cara entered the bindery room from the sorting room.",
        "Cara departed the bindery room toward the reading room.",
        "Cara walked from the bindery room into the reading room.",
        "In the sorting room, Dan stepped from the container bay behind the rolling pillar into the far-side alcove.",
        "In the sorting room, from the far-side alcove, Dan looked toward the iron safe, but the rolling pillar blocked the view.",
        "In the sorting room, from the far-side alcove, Dan glanced at the maple cubby, yet the rolling pillar blocked his sightline.",
        "In the reading room, Cara examined the index reel in the slate tray in open view.",
        "Eva exited the sorting room and headed toward the bindery room.",
        "Eva arrived in the bindery room from the sorting room.",
        "In the reading room, Cara took the seal stamp from the slate tray onto the cedar shelf in open view.",
        'In the reading room, Cara announced at 60 dB: "The seal stamp is stored in the cedar shelf."',
        "In the bindery room, Eva spotted the humidity dial in open view.",
        'In the reading room, Cara texted to Gia: "Update for Gia: the brass ledger is stored at the maple cubby."',
        "In the bindery room, Eva noticed the unsigned transfer slip in the copper rack in open view.",
        "In the sorting room, Dan adjusted the overhead louver from latch position A to latch position B in open view.",
        'In the sorting room, Dan stated at 20 dB: "I adjusted the overhead louver to clear binding dust from the bench."',
        "In the sorting room, from the far-side alcove, Dan watched the iron safe, though the rolling pillar blocked the sightline.",
        'In the sorting room, Dan claimed at 20 dB: "I am certain the brass ledger was moved into the copper rack."',
    ]
    return sj_setup, sj_event_sentences


def test_41_four_traps_fixed_per_sj() -> None:
    """Test 41 (Decision 2): Verify all four traps from sj §1/§2/§4 are fixed:
    (a) Unoccluded 'see' requires target_object and to_location when to_location is a container, but never requires
        station names ('container bay'/'far-side alcove'), so 'In the sorting room, Cara saw the maple cubby in open view.' passes,
        while omitting a container to_location ('linen bin' on event #1) is refused.
    (b) Broadened _ACTION_VERBS_BY_KIND includes 'examined', 'relocated', 'carried', 'took', 'proceeded', 'crossed'
        and renders them into the prompt.
    (c) Prepositions on from-locations (from/out of) and to-locations (to/into/toward/at/in/onto) are stated in the prompt
        and enforced by the checker.
    (d) Starting-room sentences in setup that also mention 'container bay' pass without being filtered out.
    """
    from udlab_generator.render import _sentence_matches_event

    w = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    fa = w.agents[0]
    sj_setup, sj_sents = _build_sj_writer_in_the_loop_scen_0000()

    # (a) Unoccluded see: station name ('container bay' on event #6, 'far-side alcove' on event #11) not required
    assert w.events[5].kind == "see" and w.events[5].to_location == "container bay"
    assert _sentence_matches_event("In the sorting room, Cara saw the maple cubby in open view.", w.events[5], w.noise_threshold, world=w)
    # Whereas container to_location ('linen bin' on event #1) IS required on unoccluded see
    assert w.events[0].kind == "see" and w.events[0].to_location == "linen bin"
    assert _sentence_matches_event(
        "In the bindery room, Eva examined the index reel in the linen bin in open view.",
        w.events[0],
        w.noise_threshold,
        world=w,
    )
    assert not _sentence_matches_event(
        "In the bindery room, Eva examined the index reel in open view.",
        w.events[0],
        w.noise_threshold,
        world=w,
    )

    # (b) Broadened verbs ('examined', 'relocated', 'carried', 'took', 'proceeded', 'crossed') pass
    assert _sentence_matches_event(
        "In the sorting room, Cara relocated the brass ledger from the intake bench into the iron safe in open view.",
        w.events[6],
        w.noise_threshold,
        world=w,
    )
    assert _sentence_matches_event(
        "In the sorting room, Cara carried the brass ledger out of the iron safe to the maple cubby in open view.",
        w.events[9],
        w.noise_threshold,
        world=w,
    )
    assert _sentence_matches_event(
        "In the reading room, Cara took the seal stamp from the slate tray onto the cedar shelf in open view.",
        w.events[21],
        w.noise_threshold,
        world=w,
    )
    assert _sentence_matches_event(
        "Eva proceeded from the bindery room to the sorting room.",
        w.events[7],
        w.noise_threshold,
        world=w,
    )
    assert _sentence_matches_event(
        "Eva crossed from the bindery room into the sorting room.",
        w.events[8],
        w.noise_threshold,
        world=w,
    )

    # (c) Prepositions: stated in the prompt and enforced on enter/move
    prompt_str = format_writer_prompt(w, fa)
    assert "from-locations on enter/move use 'from'/'out of'" in prompt_str
    assert "to-locations on enter/leave/move use 'to'/'into'/'toward'/'at'/'in'/'onto'" in prompt_str
    assert not _sentence_matches_event(
        "In the sorting room, Cara relocated the brass ledger the intake bench the iron safe in open view.",
        w.events[6],
        w.noise_threshold,
        world=w,
    )

    # (d) Starting rooms: setup sentence mentioning 'container bay' alongside starting room passes
    setup_with_bay_in_start = sj_setup.replace(
        "Eva, Dan, and Cara all start in the bindery room, whereas Gia is absent off-site.",
        "Eva, Dan, and Cara all start at the container bay in the bindery room, whereas Gia is absent off-site.",
    )
    assert "start at the container bay in the bindery room" in setup_with_bay_in_start
    assert check_rendered_prose(f"{setup_with_bay_in_start}\n\n{' '.join(sj_sents)}", w, focal_agent=fa) == ()


def test_42_writer_in_the_loop_scen_0000_passes_and_fault_per_event_check_id_refused() -> None:
    """Test 42 (Decision 3): Take sj §2's hand-derived setup paragraph and first five event sentences for scen_0000,
    extend them to the full 30-event scenario in the same style, and assert that they pass.
    Then make one fault edit per check_id that applies to event sentences, and assert that each is refused with that id.
    """
    import udlab_generator as ug
    from udlab_generator.world import DOMAINS

    w = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    assert w.scenario_id == "scen_0000"
    fa = w.agents[0]  # Eva

    sj_setup, sj_sents = _build_sj_writer_in_the_loop_scen_0000()
    assert len(sj_sents) == len(w.events) == 30
    full_prose = f"{sj_setup}\n\n{' '.join(sj_sents)}"

    # Assert the full extended sj §2 prose passes all prose checks (a)-(d)
    assert check_rendered_prose(full_prose, w, focal_agent=fa) == ()

    # Now make one fault edit per check_id that applies to event sentences (12 event-applicable check_ids)
    # and assert that each is refused with that exact check_id:
    current_domain = next(d for d in DOMAINS if d.name == w.domain_name)
    foreign_domain = next(d for d in DOMAINS if d.name != w.domain_name)
    unlogged_motive = next(
        r for r in current_domain.side_reasons if r not in {e.stated_reason for e in w.events if e.stated_reason}
    )
    unlogged_party = next(
        p for pool in ug.world.AGENT_NAME_POOLS for p in pool if p not in w.agents and p not in w.absent_parties
    )

    fault_cases: list[tuple[str, list[str]]] = [
        # 1. c_fidelity_event_order: swap chronological order of events #1 and #6 (two independent intra-room see events)
        (
            "c_fidelity_event_order",
            [sj_sents[5], *sj_sents[1:5], sj_sents[0], *sj_sents[6:]],
        ),
        # 2. c_fidelity_missing_event: omit container to_location ('linen bin') on event #1 (unoccluded see in a container)
        (
            "c_fidelity_missing_event",
            [
                "In the bindery room, Eva observed the index reel in open view.",
                *sj_sents[1:],
            ],
        ),
        # 3. c_fidelity_unlogged_event: append an extra unlogged 31st event sentence
        (
            "c_fidelity_unlogged_event",
            [
                *sj_sents,
                "In the bindery room, Eva observed the index reel in the linen bin in open view.",
            ],
        ),
        # 4. c_fidelity_contradictory_occlusion: state both 'in open view' and 'blocked' in event #1
        (
            "c_fidelity_contradictory_occlusion",
            [
                "In the bindery room, Eva observed the index reel in the linen bin in open view while blocked.",
                *sj_sents[1:],
            ],
        ),
        # 5. c_fidelity_unlogged_party: mention a party name not in agents or absent_parties in an event sentence
        (
            "c_fidelity_unlogged_party",
            [
                f"In the bindery room, near {unlogged_party}, Eva observed the index reel in the linen bin in open view.",
                *sj_sents[1:],
            ],
        ),
        # 6. c_fidelity_unlogged_entity: mention an entity from another domain in an event sentence
        (
            "c_fidelity_unlogged_entity",
            [
                f"In the bindery room, near the {foreign_domain.objects[0]}, Eva observed the index reel in the linen bin in open view.",
                *sj_sents[1:],
            ],
        ),
        # 7. c_fidelity_unlogged_motive: assert an unlogged side_reason motive in an event sentence
        (
            "c_fidelity_unlogged_motive",
            [
                f"In the bindery room, {unlogged_motive}, Eva observed the index reel in the linen bin in open view.",
                *sj_sents[1:],
            ],
        ),
        # 8. c_fidelity_unlogged_placer: attribute placement of the unsigned note item when no move event is logged
        (
            "c_fidelity_unlogged_placer",
            [
                *sj_sents[:25],
                "In the bindery room, after Eva placed the unsigned transfer slip, Eva noticed the unsigned transfer slip in the copper rack in open view.",
                *sj_sents[26:],
            ],
        ),
        # 9. a_no_markers_bracket: emit a square-bracket marker in an event sentence
        (
            "a_no_markers_bracket",
            [
                "In the bindery room [logged], Eva observed the index reel in the linen bin in open view.",
                *sj_sents[1:],
            ],
        ),
        # 10. a_no_markers_fact_id: emit an internal fact ID F1 in an event sentence
        (
            "a_no_markers_fact_id",
            [
                "In the bindery room, Eva observed the index reel F1 in the linen bin in open view.",
                *sj_sents[1:],
            ],
        ),
        # 11. a_no_markers_step_number: emit 'step 1' in an event sentence
        (
            "a_no_markers_step_number",
            [
                "At step 1, in the bindery room, Eva observed the index reel in the linen bin in open view.",
                *sj_sents[1:],
            ],
        ),
        # 12. b_banned_adverbs: use a banned adverb in an event sentence
        (
            "b_banned_adverbs",
            [
                "In the bindery room, Eva furtively observed the index reel in the linen bin in open view.",
                *sj_sents[1:],
            ],
        ),
    ]

    tested_ids: list[str] = []
    for expected_check_id, faulty_sents in fault_cases:
        faulty_prose = f"{sj_setup}\n\n{' '.join(faulty_sents)}"
        with pytest.raises(RenderCheckError) as exc_info:
            check_rendered_prose(faulty_prose, w, focal_agent=fa)
        assert exc_info.value.check_id == expected_check_id, (
            f"Expected fault {expected_check_id!r}, got {exc_info.value.check_id!r}: {exc_info.value}"
        )
        tested_ids.append(expected_check_id)

    assert len(tested_ids) == 12 == len(set(tested_ids))


def test_43_fake_transport_keep_and_continue_and_max_retries(tmp_path: Path) -> None:
    """Test 43 (Card sn Req 1, 2, 4, 5):
    - A fake transport returning one passing and one failing response: the build completes with rc 0,
      keeps 1, refuses 1, and both texts (plus thinking text) are saved in `sealed_store/responses.sealed.jsonl`,
      keyed by call_id, with `generations.jsonl` pointing to them by sha256;
    - A 2-scenario run where scenario 0 passes both renderings and scenario 1 fails rendering 0:
      `scenarios_kept == 1`, `scenarios_refused == 1`, `rc == 0`;
    - `--max-retries N` retries a refused rendering with the same prompt and `seed + attempt`, recording every attempt.
    """
    import hashlib
    import hashlib
    from typing import Any
    import udlab_generator.render as ug_render
    from udlab_generator import (
        PINNED_WRITER_MANIFEST_DIGEST,
        PINNED_WRITER_MODEL_TAG,
    )
    sealed_responses_filename = "responses.sealed.jsonl"

    configs_1 = make_balanced_configs(1, base_seed=20261030)
    w0 = generate_world(configs_1[0])
    passing_r0 = render_world_template(w0, w0.agents[0])
    sj_setup, sj_sents = _build_sj_writer_in_the_loop_scen_0000()
    # Failing rendering 1: omit 'linen bin' on event #1 so it fails with c_fidelity_missing_event
    failing_r1_sent_0 = "In the bindery room, Eva observed the index reel in open view."
    failing_r1 = f"{sj_setup.replace('Following Eva', f'Following {w0.agents[1]}')}\n\n{' '.join([failing_r1_sent_0, *sj_sents[1:]])}"

    chat_Idx = 0
    seen_seeds_1: list[int] = []

    def _one_pass_one_fail_transport(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        nonlocal chat_Idx
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            assert payload is not None
            opts = payload.get("options") or {}
            seen_seeds_1.append(int(opts.get("seed", -1)))
            call_num = chat_Idx
            chat_Idx += 1
            if call_num == 0:
                return {
                    "message": {"role": "assistant", "content": passing_r0, "thinking": "thought_pass_0"},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 3000,
                    "eval_count": 780,
                }
            return {
                "message": {"role": "assistant", "content": failing_r1, "thinking": "thought_fail_1"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3000,
                "eval_count": 823,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    out_1 = tmp_path / "one_pass_one_fail"
    summary_1 = build_and_probe_dataset(
        count=1,
        base_seed=20261030,
        output_dir=out_1,
        writer_kind="ollama",
        grant_id="test-grant-sn-1",
        http_transport=_one_pass_one_fail_transport,
        start_owned_server=False,
    )
    assert summary_1["kept"] == 1
    assert summary_1["refused"] == 1
    assert summary_1["renderings_kept"] == 1
    assert summary_1["renderings_refused"] == 1
    assert summary_1["calls_kept"] == 1
    assert summary_1["calls_refused"] == 1
    assert summary_1["scenarios_kept"] == 0
    assert summary_1["scenarios_refused"] == 1
    assert summary_1["refusals_by_check_id"] == {"c_fidelity_missing_event": 1}
    assert seen_seeds_1 == [
        ug_render.derive_call_seed(int(w0.seed), 0, 0),
        ug_render.derive_call_seed(int(w0.seed), 1, 0),
    ]

    # Verify CLI entrypoint also returns rc == 0 on the same one-passing/one-failing fake transport:
    import udlab_generator.cli as ug_cli
    orig_build_fn = ug_cli.build_and_probe_dataset
    chat_Idx = 0
    out_cli_1 = tmp_path / "one_pass_one_fail_cli"
    ug_cli.build_and_probe_dataset = lambda **kw: orig_build_fn(
        **{**kw, "http_transport": _one_pass_one_fail_transport, "start_owned_server": False}
    )
    try:
        rc_cli_1 = cli_main(
            [
                "--write",
                "--writer",
                "ollama",
                "--count",
                "1",
                "--seed",
                "20261030",
                "--out-dir",
                str(out_cli_1),
                "--grant-id",
                "test-grant-sn-1",
            ]
        )
    finally:
        ug_cli.build_and_probe_dataset = orig_build_fn
    assert rc_cli_1 == 0

    # Verify 2-scenario run where scenario 0 passes both renderings and scenario 1 fails rendering 0 ->
    # scenarios_kept == 1, scenarios_refused == 1:
    configs_2 = make_balanced_configs(2, base_seed=20261030)
    w_s0, w_s1 = generate_world(configs_2[0]), generate_world(configs_2[1])
    s0_r0 = render_world_template(w_s0, w_s0.agents[0])
    s0_r1 = render_world_template(w_s0, w_s0.agents[1])
    s1_r0_fail = f"{render_world_template(w_s1, w_s1.agents[0])} [bad_bracket]"
    s1_r1_pass = render_world_template(w_s1, w_s1.agents[1])
    two_scen_resps = [s0_r0, s0_r1, s1_r0_fail, s1_r1_pass]
    two_scen_idx = 0

    def _two_scen_transport(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        nonlocal two_scen_idx
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            txt = two_scen_resps[two_scen_idx]
            two_scen_idx += 1
            return {
                "message": {"role": "assistant", "content": txt},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3000,
                "eval_count": 780,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    out_2 = tmp_path / "two_scen_one_kept_one_refused"
    summary_2 = build_and_probe_dataset(
        count=2,
        base_seed=20261030,
        output_dir=out_2,
        writer_kind="ollama",
        grant_id="test-grant-sn-2",
        http_transport=_two_scen_transport,
        start_owned_server=False,
        # Card ul: a cheap counter near the measured ratio (bytes/4); on the bytes/3 estimate alone the card-ul
        # worked examples push this seed's scen_0001 prompt over the cap, which the exact count never does.
        prompt_token_counter=lambda text: len(text.encode("utf-8")) // 4,
    )
    assert summary_2["scenarios_kept"] == 1
    assert summary_2["scenarios_refused"] == 1
    assert summary_2["renderings_kept"] == 3
    assert summary_2["renderings_refused"] == 1
    assert summary_2["num_scenarios"] == 1
    assert summary_2["num_items"] == 2
    assert summary_2["refusals_by_check_id"] == {"a_no_markers_bracket": 1}

    # Both texts (and thinking texts) are saved in sealed_store/responses.sealed.jsonl, keyed by call_id,
    # with generations.jsonl pointing to them by sha256:
    resp_path_1 = out_1 / "sealed_store" / sealed_responses_filename
    gen_path_1 = out_1 / "generations.jsonl"
    assert resp_path_1.is_file()
    assert gen_path_1.is_file()
    saved_rows_1 = [json.loads(line) for line in resp_path_1.read_text(encoding="utf-8").splitlines() if line.strip()]
    gen_rows_1 = [json.loads(line) for line in gen_path_1.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(saved_rows_1) == 2
    assert len(gen_rows_1) == 2

    assert saved_rows_1[0]["call_id"] == gen_rows_1[0]["call_id"] == "scen_0000:r0:call_0001"
    assert saved_rows_1[0]["response_text"] == passing_r0
    assert saved_rows_1[0]["thinking_text"] == "thought_pass_0"
    exp_r0_sha = hashlib.sha256(passing_r0.encode("utf-8")).hexdigest()
    exp_t0_sha = hashlib.sha256(b"thought_pass_0").hexdigest()
    assert saved_rows_1[0]["response_sha256"] == gen_rows_1[0]["response_sha256"] == exp_r0_sha
    assert saved_rows_1[0]["thinking_sha256"] == gen_rows_1[0]["thinking_sha256"] == exp_t0_sha
    assert gen_rows_1[0]["checks_passed"] is True

    assert saved_rows_1[1]["call_id"] == gen_rows_1[1]["call_id"] == "scen_0000:r1:call_0002"
    assert saved_rows_1[1]["response_text"] == failing_r1
    assert saved_rows_1[1]["thinking_text"] == "thought_fail_1"
    exp_r1_sha = hashlib.sha256(failing_r1.encode("utf-8")).hexdigest()
    exp_t1_sha = hashlib.sha256(b"thought_fail_1").hexdigest()
    assert saved_rows_1[1]["response_sha256"] == gen_rows_1[1]["response_sha256"] == exp_r1_sha
    assert saved_rows_1[1]["thinking_sha256"] == gen_rows_1[1]["thinking_sha256"] == exp_t1_sha
    assert gen_rows_1[1]["checks_passed"] is False
    exp_sent0_sha = hashlib.sha256(failing_r1_sent_0.encode("utf-8")).hexdigest()
    assert gen_rows_1[1]["check_id"] == "c_fidelity_missing_event"
    assert gen_rows_1[1]["failing_sentence_sha256"] == exp_sent0_sha
    assert any(
        "c_fidelity_missing_event" in r and exp_sent0_sha in r and failing_r1_sent_0 not in r
        for r in gen_rows_1[1]["refusal_reasons"]
    )
    assert any(
        "c_fidelity_missing_event" in r and failing_r1_sent_0 in r
        for r in saved_rows_1[1]["refusal_reasons"]
    )

    # Also test --max-retries 1 on the same failing rendering: attempt 0 fails, attempt 1 (with seed + 1) passes!
    passing_r1 = render_world_template(w0, w0.agents[1])
    retry_calls: list[tuple[int, str]] = []

    def _retry_transport(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            assert payload is not None
            seed_val = int((payload.get("options") or {}).get("seed", -1))
            msgs = payload["messages"]
            prompt_str = str(msgs[0]["content"])
            idx_call = len(retry_calls)
            retry_calls.append((seed_val, msgs))
            if idx_call == 0:
                txt = passing_r0
            elif idx_call == 1:
                txt = failing_r1  # r1 attempt 0 fails
            else:
                txt = passing_r1  # r1 attempt 1 succeeds
            return {
                "message": {"role": "assistant", "content": txt},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3000,
                "eval_count": 780,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    out_retry = tmp_path / "retry_run"
    summary_retry = build_and_probe_dataset(
        count=1,
        base_seed=20261030,
        output_dir=out_retry,
        writer_kind="ollama",
        max_retries=1,
        grant_id="test-grant-sn-retry",
        http_transport=_retry_transport,
        start_owned_server=False,
    )
    assert len(retry_calls) == 3
    # Assert the default single-turn retry call carries only 1 user message (original prompt + repair note), without the refused draft
    assert len(retry_calls[2][1]) == 1
    assert retry_calls[2][1][0]["content"].startswith(retry_calls[1][1][0]["content"].rstrip("\n") + "\n")
    assert failing_r1 not in retry_calls[2][1][0]["content"]
    retry_tail = retry_calls[2][1][0]["content"].splitlines()[-1]
    assert retry_tail.startswith(
        "Repair note (rule broken: event 1 (see, check c_fidelity_missing_event), altered): "
    )
    assert retry_calls[0][0] == ug_render.derive_call_seed(int(w0.seed), 0, 0)
    assert retry_calls[1][0] == ug_render.derive_call_seed(int(w0.seed), 1, 0)
    assert retry_calls[2][0] == ug_render.derive_call_seed(int(w0.seed), 1, 1)
    assert summary_retry["scenarios_kept"] == 1
    assert summary_retry["scenarios_refused"] == 0
    assert summary_retry["renderings_kept"] == 2
    assert summary_retry["renderings_refused"] == 0
    assert summary_retry["calls_kept"] == 2
    assert summary_retry["calls_refused"] == 1
    assert summary_retry["refusals_by_check_id"] == {"c_fidelity_missing_event": 1}
    gen_retry = [
        json.loads(line)
        for line in (out_retry / "generations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [(r["attempt"], r["seed"], r["checks_passed"]) for r in gen_retry] == [
        (0, ug_render.derive_call_seed(int(w0.seed), 0, 0), True),
        (0, ug_render.derive_call_seed(int(w0.seed), 1, 0), False),
        (1, ug_render.derive_call_seed(int(w0.seed), 1, 1), True),
    ]


def test_44_recheck_reproduces_verdicts_and_names_failing_sentence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test 44 (Card sn Req 3, 4, 5):
    - `--recheck <dir>` re-runs every prose check on the saved texts with no model and no GPU,
      printing per-call results with check ids and the failing sentence, and reproduces the same verdicts.
    - Checker failure messages name the candidate sentence for the failing event (or 'no candidate sentence').
    """
    from typing import Any
    import udlab_generator as ug
    import udlab_generator.render as ug_render
    from udlab_generator import (
        PINNED_WRITER_MANIFEST_DIGEST,
        PINNED_WRITER_MODEL_TAG,
    )

    configs_1 = make_balanced_configs(1, base_seed=20261030)
    w0 = generate_world(configs_1[0])
    passing_r0 = render_world_template(w0, w0.agents[0])
    sj_setup, sj_sents = _build_sj_writer_in_the_loop_scen_0000()
    failing_sent_ev1 = "In the bindery room, Eva observed the index reel in open view."
    failing_r1 = f"{sj_setup.replace('Following Eva', f'Following {w0.agents[1]}')}\n\n{' '.join([failing_sent_ev1, *sj_sents[1:]])}"

    # Verify checker failure exception carries the candidate sentence as an attribute (for sealed-store writer)
    # while str(exc) never includes the failing sentence:
    with pytest.raises(RenderCheckError) as exc_cand:
        check_rendered_prose(failing_r1, w0, focal_agent=w0.agents[1])
    assert exc_cand.value.check_id == "c_fidelity_missing_event"
    assert getattr(exc_cand.value, "failing_sentence", None) == failing_sent_ev1
    assert failing_sent_ev1 not in str(exc_cand.value)
    assert "failing_sentence_sha256=" in str(exc_cand.value)

    # Also check when there are no event sentences after the last event -> 'no candidate sentence'
    with pytest.raises(RenderCheckError) as exc_none:
        check_rendered_prose(f"{sj_setup}\n\n{' '.join(sj_sents[:-1])}", w0, focal_agent=w0.agents[0])
    assert exc_none.value.check_id == "c_fidelity_missing_event"
    assert getattr(exc_none.value, "failing_sentence", None) == "no candidate sentence"
    assert "candidate sentence: 'no candidate sentence'" in str(exc_none.value)

    responses = [passing_r0, failing_r1]
    call_idx = 0

    def _transport(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        nonlocal call_idx
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            txt = responses[call_idx]
            call_idx += 1
            return {
                "message": {"role": "assistant", "content": txt},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3000,
                "eval_count": 800,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    out_dir = tmp_path / "recheck_dir"
    build_and_probe_dataset(
        count=1,
        base_seed=20261030,
        output_dir=out_dir,
        writer_kind="ollama",
        grant_id="test-grant-recheck",
        http_transport=_transport,
        start_owned_server=False,
    )

    # Forbid any HTTP transport or GPU query during --recheck to prove it uses no model and no GPU:
    def _forbidden_transport(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("--recheck must not make any HTTP/model calls")

    def _forbidden_gpu(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("--recheck must not query GPU")

    monkeypatch.setattr(ug_render, "default_ollama_http_transport", _forbidden_transport)
    monkeypatch.setattr(ug_render, "query_gpu_vram_peak_mib", _forbidden_gpu)

    capsys.readouterr()
    rc = cli_main(["--recheck", str(out_dir)])
    captured = capsys.readouterr().out
    assert rc == 0
    assert "[recheck] scen_0000:r0:call_0001" in captured
    assert "PASS check_id=ok" in captured
    assert "[recheck] scen_0000:r1:call_0002" in captured
    assert "FAIL check_id=c_fidelity_missing_event" in captured
    assert failing_sent_ev1 in captured

    recheck_fn = getattr(ug, "recheck_saved_responses")
    recheck_report = recheck_fn(out_dir)
    assert recheck_report["matches_recorded_verdicts"] is True
    assert recheck_report["kept"] == 1
    assert recheck_report["refused"] == 1
    assert recheck_report["scenarios_kept"] == 0
    assert recheck_report["scenarios_refused"] == 1
    assert recheck_report["refusals_by_check_id"] == {"c_fidelity_missing_event": 1}
    calls_list = recheck_report["calls"]
    assert isinstance(calls_list, list) and len(calls_list) == 2
    assert calls_list[0]["checks_passed"] is True
    assert calls_list[0]["check_id"] is None
    assert calls_list[0]["failing_sentence"] is None
    assert calls_list[1]["checks_passed"] is False
    assert calls_list[1]["check_id"] == "c_fidelity_missing_event"
    assert calls_list[1]["failing_sentence"] == failing_sent_ev1


def test_45_one_scenario_run_end_to_end(tmp_path: Path) -> None:
    """Test 45 (Card sn Req 2, 7): `--count 1` works end to end: build (exit code 0), write 2 renderings, save, recheck."""
    import udlab_generator as ug

    out_1 = tmp_path / "one_scenario_e2e"
    rc_build = cli_main(
        [
            "--write",
            "--writer",
            "mock",
            "--count",
            "1",
            "--seed",
            "20261030",
            "--out-dir",
            str(out_1),
        ]
    )
    assert rc_build == 0

    summary = json.loads((out_1 / "build_summary.json").read_text(encoding="utf-8"))
    assert summary["num_scenarios"] == 1
    assert summary["num_items"] == 2
    assert summary["scenarios_kept"] == 1
    assert summary["scenarios_refused"] == 0
    assert summary["renderings_kept"] == 2
    assert summary["renderings_refused"] == 0
    assert summary["kept"] == 2
    assert summary["refused"] == 0
    assert summary["refusals_by_check_id"] == {}

    items_lines = [
        json.loads(line)
        for line in (out_1 / "part_public" / "items.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(items_lines) == 2

    saved_lines = [
        json.loads(line)
        for line in (out_1 / "sealed_store" / "responses.sealed.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(saved_lines) == 2

    rc_recheck = cli_main(["--recheck", str(out_1)])
    assert rc_recheck == 0
    recheck_res = getattr(ug, "recheck_saved_responses")(out_1)
    assert recheck_res["matches_recorded_verdicts"] is True
    assert recheck_res["scenarios_kept"] == 1
    assert recheck_res["renderings_kept"] == 2
    assert recheck_res["renderings_refused"] == 0


def test_46_sealed_response_store_private_never_in_part_public(tmp_path: Path) -> None:
    """Test 46 (Card sn Req 1, 8):
    - Saved response store (`responses.sealed.jsonl`) lives in `sealed_store`, never in `part_public`,
      and `.gitignore` covers it;
    - Fails if any saved response text from a refused scenario (or thinking text, or `response_text` field)
      appears anywhere under `part_public`.
    """
    import hashlib
    from typing import Any
    import udlab_generator.render as ug_render
    from udlab_generator import (
        PINNED_WRITER_MANIFEST_DIGEST,
        PINNED_WRITER_MODEL_TAG,
    )
    sealed_responses_filename = "responses.sealed.jsonl"

    gitignore_text = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8")
    assert "**/sealed_store/" in gitignore_text
    assert "**/*.sealed.jsonl" in gitignore_text

    configs_1 = make_balanced_configs(1, base_seed=20261030)
    w0 = generate_world(configs_1[0])
    passing_r0 = render_world_template(w0, w0.agents[0])
    secret_refused_marker = "PRIVATE_REFUSED_WRITER_PROSE_DO_NOT_PUBLISH_987654"
    secret_thinking_marker = "PRIVATE_WRITER_THINKING_TRACE_DO_NOT_PUBLISH_123456"
    failing_r1 = f"{passing_r0}\n\n{secret_refused_marker} [unlogged_bracket]"

    call_idx = 0

    def _transport(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        nonlocal call_idx
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            txt = passing_r0 if call_idx == 0 else failing_r1
            call_idx += 1
            return {
                "message": {"role": "assistant", "content": txt, "thinking": secret_thinking_marker},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3000,
                "eval_count": 800,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    out_dir = tmp_path / "privacy_check"
    build_and_probe_dataset(
        count=1,
        base_seed=20261030,
        output_dir=out_dir,
        writer_kind="ollama",
        grant_id="test-grant-privacy",
        http_transport=_transport,
        start_owned_server=False,
    )

    pub_dir = out_dir / "part_public"
    sealed_dir = out_dir / "sealed_store"
    assert (sealed_dir / sealed_responses_filename).is_file()
    assert not (pub_dir / sealed_responses_filename).exists()

    saved_entries = [
        json.loads(line)
        for line in (sealed_dir / sealed_responses_filename).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(saved_entries) == 2

    # Check every file under part_public: no saved response text (neither r0 nor r1, nor thinking text)
    # may appear under part_public when the scenario was refused, and no sealed file or response_text field
    # may ever appear under part_public:
    for pub_file in pub_dir.rglob("*"):
        if pub_file.is_file():
            assert not pub_file.name.endswith((".sealed.json", ".sealed.jsonl"))
            content = pub_file.read_text(encoding="utf-8")
            assert secret_refused_marker not in content
            assert secret_thinking_marker not in content
            for entry in saved_entries:
                assert str(entry["response_text"]) not in content, (
                    f"Saved response text for {entry['call_id']} leaked into {pub_file}"
                )
                if entry.get("thinking_text"):
                    assert str(entry["thinking_text"]) not in content, (
                        f"Saved thinking text for {entry['call_id']} leaked into {pub_file}"
                    )

    # Also verify that check_not_in_key_order actively rejects `response_text` or `thinking_text` in public items:
    with pytest.raises(SealingError, match="Forbidden key/seed fields"):
        check_not_in_key_order([{"item_id": "item_0001", "response_text": "leaked"}], [{"pass_index": 1}], {})


def test_47_meaning_based_see_and_one_verb_list() -> None:
    """Test 47 (Card sq Req 1 & 2):
    - Meaning-based unoccluded 'see' passes when sentence names actor, object, room, places object at
      event's container (or station rule as before), uses a sighting verb from one list, and contains no blocking language;
      'in open view' is optional for unoccluded events.
    - Mandatory refusal tests (a)-(e):
      (a) unoccluded event whose sentence contains blocking language ('blocked', 'could not see', 'hidden behind', 'obscured');
      (b) occluded event whose sentence lacks 'blocked' or names no occluder;
      (c) sighting in the wrong room;
      (d) sighting of the wrong object;
      (e) sighting at the wrong container.
    - Plus the pass case built from refusal-1 wording ('Eva observed the index reel in the linen bin in the bindery room.').
    - One verb list: SIGHTING_VERBS and BLOCKING_PHRASES are one list in code, rendered into the prompt's generated
      constraints block, and used by the checker.
    """
    import udlab_generator.render as ug_render
    from udlab_generator.render import _sentence_matches_event

    w = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    fa = w.agents[0]  # Eva
    sj_setup, sj_sents = _build_sj_writer_in_the_loop_scen_0000()
    # Add fixtures to sj_setup so it satisfies the setup checklist including fixtures per room:
    setup_with_fixtures = (
        f"{sj_setup} Fixtures per room: a wall barometer in the vault room, a wall clock in the reading room, "
        f"a humidity dial in the bindery room, and a ventilation grille in the sorting room."
    )

    # 2. Pass case built from Refusal-1 wording (no literal 'in open view'):
    ev_1 = w.events[0]
    assert ev_1.kind == "see" and not ev_1.occluded and ev_1.actor == "Eva"
    assert ev_1.room == "bindery room" and ev_1.target_object == "index reel" and ev_1.to_location == "linen bin"
    refusal_1_pass_sent = "Eva observed the index reel in the linen bin in the bindery room."
    assert _sentence_matches_event(refusal_1_pass_sent, ev_1, w.noise_threshold, world=w)

    # 1. One list in code for sighting verbs and blocking phrases, rendered into prompt constraints and used by checker
    sighting_verbs = getattr(ug_render, "SIGHTING_VERBS", None)
    blocking_phrases = getattr(ug_render, "BLOCKING_PHRASES", None)
    assert isinstance(sighting_verbs, tuple) and len(sighting_verbs) >= 10
    assert isinstance(blocking_phrases, tuple) and (
        "blocked",
        "could not see",
        "hidden behind",
        "obscured",
    ) == blocking_phrases[:4]
    assert ug_render._ACTION_VERBS_BY_KIND["see"] == sighting_verbs

    prompt_text = format_writer_prompt(w, fa)
    for v in sighting_verbs:
        assert v in prompt_text
    for bp in blocking_phrases:
        assert bp in prompt_text
    ug_render.verify_prompt_and_checker_requirements_aligned(prompt_text)

    prose_pass = f"{setup_with_fixtures}\n\n{' '.join([refusal_1_pass_sent, *sj_sents[1:]])}"
    assert check_rendered_prose(prose_pass, w, focal_agent=fa) == ()

    # Mandatory refusal (a): unoccluded event whose sentence contains blocking language
    for bad_phrase in ("blocked", "could not see", "hidden behind", "obscured"):
        sent_a = f"In the bindery room, Eva observed the index reel in the linen bin ({bad_phrase})."
        assert not _sentence_matches_event(sent_a, ev_1, w.noise_threshold, world=w), (
            f"Unoccluded see must reject blocking phrase {bad_phrase!r}"
        )
        with pytest.raises(RenderCheckError) as exc_a:
            check_rendered_prose(
                f"{setup_with_fixtures}\n\n{' '.join([sent_a, *sj_sents[1:]])}",
                w,
                focal_agent=fa,
            )
        assert exc_a.value.check_id == "c_fidelity_missing_event"

    # Mandatory refusal (b): occluded event whose sentence lacks 'blocked' or names no occluder
    ev_17 = w.events[16]
    assert ev_17.kind == "see" and ev_17.occluded and ev_17.occluder == "rolling pillar"
    ev_29 = w.events[28]
    assert ev_29.kind == "see" and ev_29.occluded and ev_29.occluder == "rolling pillar"
    sent_b_no_blocked = "In the sorting room, from the far-side alcove, Dan looked toward the iron safe behind the rolling pillar."
    sent_b_no_occluder = "In the sorting room, from the far-side alcove, Dan looked toward the iron safe, but his view was blocked."
    for sent_b in (sent_b_no_blocked, sent_b_no_occluder):
        assert not _sentence_matches_event(sent_b, ev_17, w.noise_threshold, world=w)
        assert not _sentence_matches_event(sent_b, ev_29, w.noise_threshold, world=w)
        sents_b = list(sj_sents)
        sents_b[28] = sent_b
        with pytest.raises(RenderCheckError) as exc_b:
            check_rendered_prose(f"{setup_with_fixtures}\n\n{' '.join(sents_b)}", w, focal_agent=fa)
        assert exc_b.value.check_id == "c_fidelity_missing_event"

    # Mandatory refusal (c): sighting in the wrong room
    sent_c = "Eva observed the index reel in the linen bin in the sorting room."
    assert not _sentence_matches_event(sent_c, ev_1, w.noise_threshold, world=w)
    with pytest.raises(RenderCheckError) as exc_c:
        check_rendered_prose(
            f"{setup_with_fixtures}\n\n{' '.join([sent_c, *sj_sents[1:]])}",
            w,
            focal_agent=fa,
        )
    assert exc_c.value.check_id == "c_fidelity_missing_event"

    # Mandatory refusal (d): sighting of the wrong object
    sent_d = "Eva observed the brass ledger in the linen bin in the bindery room."
    assert not _sentence_matches_event(sent_d, ev_1, w.noise_threshold, world=w)
    with pytest.raises(RenderCheckError) as exc_d:
        check_rendered_prose(
            f"{setup_with_fixtures}\n\n{' '.join([sent_d, *sj_sents[1:]])}",
            w,
            focal_agent=fa,
        )
    assert exc_d.value.check_id == "c_fidelity_missing_event"

    # Mandatory refusal (e): sighting at the wrong container
    sent_e = "Eva observed the index reel in the copper rack in the bindery room."
    assert not _sentence_matches_event(sent_e, ev_1, w.noise_threshold, world=w)
    with pytest.raises(RenderCheckError) as exc_e:
        check_rendered_prose(
            f"{setup_with_fixtures}\n\n{' '.join([sent_e, *sj_sents[1:]])}",
            w,
            focal_agent=fa,
        )
    assert exc_e.value.check_id == "c_fidelity_missing_event"


def test_48_setup_checklist_order_in_prompt_and_content_not_order_in_checker() -> None:
    """Test 48 (Card sq Req 3):
    - Prompt states the setup checklist order:
      rooms -> containers per room -> fixtures per room -> occluders -> doorways -> starting rooms -> goals -> stake -> belief rule -> hearing threshold.
    - The check tests content, not order: a setup with everything present in another order passes;
      a setup missing any room's containers or fixtures fails, naming what's missing.
    """
    w = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    fa = w.agents[0]  # Eva
    prompt_str = format_writer_prompt(w, fa)

    expected_order_str = (
        "rooms -> containers per room -> fixtures per room -> occluders -> doorways -> "
        "starting rooms -> goals -> stake -> belief rule -> hearing threshold"
    )
    assert expected_order_str in prompt_str

    _, sj_sents = _build_sj_writer_in_the_loop_scen_0000()
    events_para = " ".join(sj_sents)

    # 1. Setup with everything present in a completely different order (hearing threshold -> belief rule -> stake -> goals -> starting rooms -> doorways -> occluders -> fixtures -> containers -> rooms) passes:
    reordered_setup = (
        "The doorway hearing threshold is 50 dB, so speech at or above 50 dB carries across one doorway while quieter speech stays in the room. "
        "Each agent assumes an object remains where that agent last saw it in open view or heard a witness state it was placed, and assumes another agent looks where they last saw or communicated that agent perceiving it. "
        "Gia is off-site and depends on the location of the brass ledger for tomorrow's courier handoff. "
        "Eva plans to complete the quarterly audit packet, Dan plans to verify the municipal deed transfer, and Cara plans to catalog the incoming charter folio. "
        "Eva, Dan, and Cara begin in the bindery room while Gia is absent. "
        "Doorways link the vault room to the reading room, the reading room to the bindery room, and the bindery room to the sorting room. "
        "Each room has a container bay and a far-side alcove separated by an occluder that blocks sight from the far-side alcove while the container bay sees containers in open view: "
        "a frosted partition in the vault room, a tall bookstack in the reading room, a canvas screen in the bindery room, and a rolling pillar in the sorting room. "
        "Fixtures are a wall barometer in the vault room, a wall clock in the reading room, a humidity dial in the bindery room, and a ventilation grille in the sorting room. "
        "Containers are an oak drawer and steel cabinet in the vault room, a slate tray and cedar shelf in the reading room, a linen bin and copper rack in the bindery room, and a maple cubby and iron safe in the sorting room."
    )
    assert check_rendered_prose(f"{reordered_setup}\n\n{events_para}", w, focal_agent=fa) == ()

    # 2. Setup omitting optional per-room containers or fixtures is accepted under omission-by-key,
    #    while falsely placing a container or fixture in the wrong room fails with c_fidelity_containers / c_fidelity_fixtures:
    missing_containers_setup = reordered_setup.replace(
        "an oak drawer and steel cabinet in the vault room, ",
        "",
    )
    assert check_rendered_prose(f"{missing_containers_setup}\n\n{events_para}", w, focal_agent=fa) == ()
    wrong_room_containers_setup = reordered_setup.replace(
        "an oak drawer and steel cabinet in the vault room, a slate tray and cedar shelf in the reading room",
        "an oak drawer in the vault room; in the reading room are a steel cabinet, a slate tray, and a cedar shelf",
    )
    with pytest.raises(RenderCheckError) as exc_cont:
        check_rendered_prose(f"{wrong_room_containers_setup}\n\n{events_para}", w, focal_agent=fa)
    assert exc_cont.value.check_id == "c_fidelity_containers"
    assert "steel cabinet" in str(exc_cont.value)
    assert "reading room" in str(exc_cont.value)

    # 3. Setup omitting optional fixture is accepted, while placing a fixture in the wrong room fails with c_fidelity_fixtures:
    missing_fixture_setup = reordered_setup.replace(
        "a humidity dial in the bindery room, ",
        "",
    )
    assert check_rendered_prose(f"{missing_fixture_setup}\n\n{events_para}", w, focal_agent=fa) == ()
    wrong_room_fixture_setup = reordered_setup.replace(
        "a wall clock in the reading room, a humidity dial in the bindery room",
        "a wall clock in the reading room; in the vault room is a humidity dial",
    )
    with pytest.raises(RenderCheckError) as exc_fix:
        check_rendered_prose(f"{wrong_room_fixture_setup}\n\n{events_para}", w, focal_agent=fa)
    assert exc_fix.value.check_id == "c_fidelity_fixtures"
    assert "humidity dial" in str(exc_fix.value)
    assert "bindery room" in str(exc_fix.value)


def test_49_max_retries_1_records_refusal_and_saves_both_texts(tmp_path: Path) -> None:
    """Test 49 (Card sq Req 4):
    `--max-retries 1`: a retry is recorded with the first attempt's refusal kept, and both texts are saved
    in `sealed_store/responses.sealed.jsonl`.
    """
    import hashlib
    from typing import Any
    import udlab_generator.render as ug_render
    from udlab_generator import (
        PINNED_WRITER_MANIFEST_DIGEST,
        PINNED_WRITER_MODEL_TAG,
    )

    configs_1 = make_balanced_configs(1, base_seed=20261030)
    w0 = generate_world(configs_1[0])
    passing_r0 = render_world_template(w0, w0.agents[0])
    passing_r1 = render_world_template(w0, w0.agents[1])
    # Attempt 0 for r1 falsely places the reading room's slate tray in the vault room -> fails with c_fidelity_containers
    failing_r1_attempt0 = passing_r1.replace(
        "contains the oak drawer and the steel cabinet",
        "contains the oak drawer and the slate tray",
    )

    call_counter = 0

    def _retry_transport(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        nonlocal call_counter
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            idx = call_counter
            call_counter += 1
            if idx == 0:
                txt = passing_r0
            elif idx == 1:
                txt = failing_r1_attempt0
            else:
                txt = passing_r1
            return {
                "message": {"role": "assistant", "content": txt, "thinking": f"think_{idx}"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3000,
                "eval_count": 780,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    out_dir = tmp_path / "sq_max_retries_1"
    summary = build_and_probe_dataset(
        count=1,
        base_seed=20261030,
        output_dir=out_dir,
        writer_kind="ollama",
        max_retries=1,
        grant_id="test-grant-sq-retry",
        http_transport=_retry_transport,
        start_owned_server=False,
    )
    assert summary["max_retries"] == 1
    assert summary["scenarios_kept"] == 1
    assert summary["renderings_kept"] == 2
    assert summary["calls_kept"] == 2
    assert summary["calls_refused"] == 1
    assert summary["refusals_by_check_id"] == {"c_fidelity_containers": 1}

    # Check generations.jsonl keeps the first attempt's refusal and records the retry attempt:
    gen_rows = [
        json.loads(line)
        for line in (out_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(gen_rows) == 3
    assert gen_rows[1]["rendering_index"] == 1 and gen_rows[1]["attempt"] == 0
    assert gen_rows[1]["checks_passed"] is False
    assert any("c_fidelity_containers" in r for r in gen_rows[1]["refusal_reasons"])
    assert gen_rows[2]["rendering_index"] == 1 and gen_rows[2]["attempt"] == 1
    assert gen_rows[2]["checks_passed"] is True
    assert gen_rows[2]["refusal_reasons"] == []

    # Check both attempt 0 and attempt 1 texts for r1 are saved in sealed_store/responses.sealed.jsonl:
    resp_rows = [
        json.loads(line)
        for line in (out_dir / "sealed_store" / "responses.sealed.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(resp_rows) == 3
    r1_saved = [r for r in resp_rows if r["rendering_index"] == 1]
    assert len(r1_saved) == 2
    assert r1_saved[0]["attempt"] == 0 and r1_saved[0]["response_text"] == failing_r1_attempt0
    assert r1_saved[1]["attempt"] == 1 and r1_saved[1]["response_text"] == passing_r1

    # Verify CLI --max-retries 1 parses and passes max_retries=1 to build_and_probe_dataset:
    import udlab_generator.cli as ug_cli
    orig_build = ug_cli.build_and_probe_dataset
    call_counter = 0
    out_cli_retry = tmp_path / "sq_max_retries_1_cli"
    ug_cli.build_and_probe_dataset = lambda **kw: orig_build(
        **{**kw, "http_transport": _retry_transport, "start_owned_server": False}
    )
    try:
        rc_retry = cli_main(
            [
                "--write",
                "--writer",
                "ollama",
                "--count",
                "1",
                "--seed",
                "20261030",
                "--max-retries",
                "1",
                "--out-dir",
                str(out_cli_retry),
                "--grant-id",
                "test-grant-sq-retry-cli",
            ]
        )
    finally:
        ug_cli.build_and_probe_dataset = orig_build
    assert rc_retry == 0
    cli_summary = json.loads((out_cli_retry / "build_summary.json").read_text(encoding="utf-8"))
    assert cli_summary["max_retries"] == 1
    assert cli_summary["scenarios_kept"] == 1
    assert cli_summary["calls_kept"] == 2
    assert cli_summary["calls_refused"] == 1
    cli_resp_rows = [
        json.loads(line)
        for line in (out_cli_retry / "sealed_store" / "responses.sealed.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(cli_resp_rows) == 3
    assert [r["response_text"] for r in cli_resp_rows if r["rendering_index"] == 1] == [
        failing_r1_attempt0,
        passing_r1,
    ]


@pytest.mark.parametrize(
    ("kind", "ev_idx", "neg_sent"),
    [
        (
            "see",
            0,
            "Eva never saw the index reel in the linen bin in the bindery room.",
        ),
        (
            "leave",
            1,
            "Dan never left the bindery room and headed toward the sorting room.",
        ),
        (
            "enter",
            2,
            "Dan never entered the sorting room from the bindery room.",
        ),
        (
            "move",
            6,
            "In the sorting room, Cara never relocated the brass ledger from the intake bench into the iron safe in open view.",
        ),
        (
            "say",
            22,
            'In the reading room, Cara never announced at 60 dB: "The seal stamp is stored in the cedar shelf."',
        ),
        (
            "message",
            24,
            'In the reading room, Cara never texted to Gia: "Update for Gia: the brass ledger is stored at the maple cubby."',
        ),
    ],
    ids=["see", "leave", "enter", "move", "say", "message"],
)
def test_50a_negated_sentence_refused_per_event_kind(kind: str, ev_idx: int, neg_sent: str) -> None:
    """Test 50a (Card ss Req 2): One negated sentence per event kind that appears in scen_0000
    (see, move, leave, enter, say, message), each refused with a named check id (`c_fidelity_missing_event`).
    """
    from udlab_generator.render import _sentence_matches_event

    w = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    fa = w.agents[0]  # Eva
    sj_setup, sj_sents = _build_sj_writer_in_the_loop_scen_0000()
    setup_with_fixtures = (
        f"{sj_setup} Fixtures per room: a wall barometer in the vault room, a wall clock in the reading room, "
        f"a humidity dial in the bindery room, and a ventilation grille in the sorting room."
    )

    ev = w.events[ev_idx]
    assert ev.kind == kind
    assert not _sentence_matches_event(neg_sent, ev, w.noise_threshold, world=w), (
        f"Negated {kind!r} sentence must not match event #{ev_idx + 1}: {neg_sent!r}"
    )
    bad_sents = list(sj_sents)
    bad_sents[ev_idx] = neg_sent
    with pytest.raises(RenderCheckError) as exc_neg:
        check_rendered_prose(
            f"{setup_with_fixtures}\n\n{' '.join(bad_sents)}",
            w,
            focal_agent=fa,
        )
    assert exc_neg.value.check_id == "c_fidelity_missing_event", (
        f"Negated {kind!r} sentence must fail with c_fidelity_missing_event, got {exc_neg.value.check_id!r}"
    )


def test_50b_negation_list_prompt_and_pass_cases() -> None:
    """Test 50b (Card ss Req 1 & 2):
    - One negation list in code (`NEGATION_PHRASES`), rendered into the prompt's generated constraints block
      ("event sentences state what happened, without negation"), and used by the checker for every event kind.
    - Covers at least: not, n't, never, no longer, no one, nobody, nothing, failed to, without, neither, nor,
      unable to, could not, cannot, did not, didn't.
    - The occluded 'could not see ... behind the <occluder>' sentence still passes.
    - The earlier pass case ('Eva observed the index reel in the linen bin in the bindery room.') still passes.
    """
    import udlab_generator.render as ug_render
    from udlab_generator.render import _sentence_matches_event

    w = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    fa = w.agents[0]  # Eva
    sj_setup, sj_sents = _build_sj_writer_in_the_loop_scen_0000()
    setup_with_fixtures = (
        f"{sj_setup} Fixtures per room: a wall barometer in the vault room, a wall clock in the reading room, "
        f"a humidity dial in the bindery room, and a ventilation grille in the sorting room."
    )

    # 1. Verify all 6 event kinds appear in scen_0000:
    scen_0000_kinds = {ev.kind for ev in w.events}
    assert scen_0000_kinds == {"see", "move", "leave", "enter", "say", "message"}

    # 2. Verify earlier pass case ("Eva observed the index reel in the linen bin in the bindery room.") still passes:
    ev_1 = w.events[0]
    refusal_1_pass_sent = "Eva observed the index reel in the linen bin in the bindery room."
    assert _sentence_matches_event(refusal_1_pass_sent, ev_1, w.noise_threshold, world=w)
    assert (
        check_rendered_prose(
            f"{setup_with_fixtures}\n\n{' '.join([refusal_1_pass_sent, *sj_sents[1:]])}",
            w,
            focal_agent=fa,
        )
        == ()
    )

    # 3. Verify occluded "could not see ... behind the <occluder>" sentence still passes:
    ev_17 = w.events[16]
    assert ev_17.kind == "see" and ev_17.occluded and ev_17.occluder == "rolling pillar"
    occluded_could_not_see_sent = (
        "In the sorting room, from the far-side alcove, Dan looked toward the iron safe, "
        "but could not see the iron safe behind the rolling pillar, which blocked the view."
    )
    assert _sentence_matches_event(occluded_could_not_see_sent, ev_17, w.noise_threshold, world=w)
    sents_occ_pass = list(sj_sents)
    sents_occ_pass[16] = occluded_could_not_see_sent
    assert (
        check_rendered_prose(
            f"{setup_with_fixtures}\n\n{' '.join(sents_occ_pass)}",
            w,
            focal_agent=fa,
        )
        == ()
    )

    # 4. Verify one negation list in code (`NEGATION_PHRASES`), covering all required items and rendered into prompt constraints:
    required_negation_items = (
        "not",
        "n't",
        "never",
        "no longer",
        "no one",
        "nobody",
        "nothing",
        "failed to",
        "without",
        "neither",
        "nor",
        "unable to",
        "could not",
        "cannot",
        "did not",
        "didn't",
    )
    negation_phrases = getattr(ug_render, "NEGATION_PHRASES", None)
    assert isinstance(negation_phrases, tuple), "NEGATION_PHRASES must be a tuple in render.py"
    for item in required_negation_items:
        assert item in negation_phrases, f"Required negation phrase {item!r} missing from NEGATION_PHRASES"

    prompt_text = format_writer_prompt(w, fa)
    assert "event sentences state what happened, without negation" in prompt_text
    for item in negation_phrases:
        assert item in prompt_text, f"Negation item {item!r} missing from rendered writer prompt"
    ug_render.verify_prompt_and_checker_requirements_aligned(prompt_text)

    # Also verify each negation item in NEGATION_PHRASES is rejected across every event kind in scen_0000:
    sample_events_by_kind: tuple[tuple[int, str], ...] = (
        (0, "In the bindery room, Eva {neg} observed the index reel in the linen bin."),
        (1, "Dan {neg} left the bindery room and headed toward the sorting room."),
        (2, "Dan {neg} entered the sorting room from the bindery room."),
        (6, "In the sorting room, Cara {neg} relocated the brass ledger from the intake bench into the iron safe."),
        (22, 'In the reading room, Cara {neg} announced at 60 dB: "The seal stamp is stored in the cedar shelf."'),
        (24, 'In the reading room, Cara {neg} texted to Gia: "Update for Gia: the brass ledger is stored at the maple cubby."'),
    )
    for item in negation_phrases:
        for ev_idx, tmpl in sample_events_by_kind:
            sent_neg = tmpl.format(neg=item)
            assert not _sentence_matches_event(sent_neg, w.events[ev_idx], w.noise_threshold, world=w), (
                f"Negation item {item!r} must be rejected on {w.events[ev_idx].kind} event"
            )


def test_51_card_st_negation_scoping_and_could_not_see() -> None:
    """Test 51 (Card st Req a, b, e):
    - (a) Negation scoping: negation counts only where it scopes the event's verb:
      - 'Not long after, Eva observed the index reel in the linen bin in the bindery room.' passes;
      - 'Eva did not see the index reel in the linen bin in the bindery room.' is refused;
      - 'Eva never saw the index reel in the linen bin in the bindery room.' is refused;
      - 'Dan could not see the iron safe at the far-side alcove in the sorting room because the rolling pillar blocked sight.'
        is accepted ONLY for an occluded see, and refused for an unoccluded one.
    - (b) 'could not see' on an occluded see is accepted, generated from the same list as the prompt text
      (SIGHTING_VERBS includes 'see', BLOCKING_PHRASES includes 'could not see').
    - (e) Seer-less occluded see ('The rolling pillar blocked sight of the iron safe at the far-side alcove in the sorting room.')
      stays refused.
    """
    import udlab_generator.render as ug_render
    from udlab_generator.render import _sentence_matches_event

    w = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    fa = w.agents[0]  # Eva
    sj_setup, sj_sents = _build_sj_writer_in_the_loop_scen_0000()

    ev_1 = w.events[0]  # unoccluded see by Eva in bindery room (index reel at linen bin)
    assert ev_1.kind == "see" and not ev_1.occluded
    ev_17 = w.events[16]  # occluded see by Dan in sorting room (iron safe, rolling pillar)
    assert ev_17.kind == "see" and ev_17.occluded and ev_17.occluder == "rolling pillar"

    # (a.1) Pass test: "Not long after, Eva observed..." passes
    sent_not_long_after = "Not long after, Eva observed the index reel in the linen bin in the bindery room."
    assert _sentence_matches_event(sent_not_long_after, ev_1, w.noise_threshold, world=w)
    assert check_rendered_prose(f"{sj_setup}\n\n{' '.join([sent_not_long_after, *sj_sents[1:]])}", w, focal_agent=fa) == ()

    # (a.2) Refusal test: "Eva did not see..." is refused
    sent_did_not_see = "Eva did not see the index reel in the linen bin in the bindery room."
    assert not _sentence_matches_event(sent_did_not_see, ev_1, w.noise_threshold, world=w)
    with pytest.raises(RenderCheckError) as exc_did_not:
        check_rendered_prose(f"{sj_setup}\n\n{' '.join([sent_did_not_see, *sj_sents[1:]])}", w, focal_agent=fa)
    assert exc_did_not.value.check_id == "c_fidelity_missing_event"

    # (a.3) Refusal test: "Eva never saw..." is refused
    sent_never_saw = "Eva never saw the index reel in the linen bin in the bindery room."
    assert not _sentence_matches_event(sent_never_saw, ev_1, w.noise_threshold, world=w)
    with pytest.raises(RenderCheckError) as exc_never:
        check_rendered_prose(f"{sj_setup}\n\n{' '.join([sent_never_saw, *sj_sents[1:]])}", w, focal_agent=fa)
    assert exc_never.value.check_id == "c_fidelity_missing_event"

    # (a.4) & (b) Exact quoted sentence: "Dan could not see the iron safe at the far-side alcove in the sorting room because the rolling pillar blocked sight."
    # Accepted ONLY for an occluded see (ev_17), and refused for an unoccluded one.
    assert "see" in ug_render.SIGHTING_VERBS
    assert "could not see" in ug_render.BLOCKING_PHRASES
    prompt_str = format_writer_prompt(w, fa)
    assert "see: " in prompt_str and ", see" in prompt_str
    assert "'could not see'" in prompt_str

    sent_could_not_see_exact = (
        "Dan could not see the iron safe at the far-side alcove in the sorting room because the rolling pillar blocked sight."
    )
    assert _sentence_matches_event(sent_could_not_see_exact, ev_17, w.noise_threshold, world=w)
    sents_occ = list(sj_sents)
    sents_occ[16] = sent_could_not_see_exact
    assert check_rendered_prose(f"{sj_setup}\n\n{' '.join(sents_occ)}", w, focal_agent=fa) == ()

    # Also test "Dan could not see X behind the pillar" on occluded vs unoccluded see:
    sent_could_not_see_behind = (
        "In the sorting room, at the far-side alcove, Dan could not see the iron safe behind the rolling pillar, which blocked sight."
    )
    assert _sentence_matches_event(sent_could_not_see_behind, ev_17, w.noise_threshold, world=w)
    unoccluded_ev_17 = replace(ev_17, occluded=False, occluder=None)
    assert not _sentence_matches_event(sent_could_not_see_exact, unoccluded_ev_17, w.noise_threshold, world=w)
    assert not _sentence_matches_event(sent_could_not_see_behind, unoccluded_ev_17, w.noise_threshold, world=w)
    sent_unocc_could_not_see = "Eva could not see the index reel in the linen bin in the bindery room behind the canvas screen."
    assert not _sentence_matches_event(sent_unocc_could_not_see, ev_1, w.noise_threshold, world=w)
    with pytest.raises(RenderCheckError) as exc_unocc_cns:
        check_rendered_prose(f"{sj_setup}\n\n{' '.join([sent_unocc_could_not_see, *sj_sents[1:]])}", w, focal_agent=fa)
    assert exc_unocc_cns.value.check_id == "c_fidelity_missing_event"

    # (e) Seer-less occluded see stays a model fault and stays refused:
    sent_seerless = "The rolling pillar blocked sight of the iron safe at the far-side alcove in the sorting room."
    assert not _sentence_matches_event(sent_seerless, ev_17, w.noise_threshold, world=w)
    sents_seerless = list(sj_sents)
    sents_seerless[16] = sent_seerless
    sents_seerless[28] = sent_seerless
    with pytest.raises(RenderCheckError) as exc_seerless:
        check_rendered_prose(f"{sj_setup}\n\n{' '.join(sents_seerless)}", w, focal_agent=fa)
    assert exc_seerless.value.check_id == "c_fidelity_missing_event"
    assert exc_seerless.value.failing_sentence == sent_seerless


def test_52_card_st_bare_enter_after_matching_leave() -> None:
    """Test 52 (Card st Req d & f):
    - (d) Accept a bare enter when the immediately preceding event sentence is by the same actor,
      is the matching leave, and names the from-room. Refuse it otherwise:
      - the r1 pattern ('Dan walked from the bindery room to the sorting room.' followed by 'Dan entered the sorting room.') passes;
      - a bare enter with NO matching leave sentence before it stays refused;
      - a bare enter whose preceding leave names a different from-room stays refused.
    - (f) Prompt emphasis: generated constraints block explicitly emphasizes naming the from-room on every enter.
    """
    from udlab_generator.render import check_rendered_prose

    w = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    fa = w.agents[1]  # Dan
    sj_setup, sj_sents = _build_sj_writer_in_the_loop_scen_0000()
    setup_dan = sj_setup.replace("Following Eva", "Following Dan")

    # (d.1) Pass test: the r1 bare-enter pattern on all 5 enters in scen_0000 (events 3, 5, 9, 13, 15, plus 21)
    r1_pattern_sents = list(sj_sents)
    r1_pattern_sents[1] = "Dan walked from the bindery room to the sorting room."
    r1_pattern_sents[2] = "Dan entered the sorting room."
    r1_pattern_sents[3] = "Cara walked from the bindery room to the sorting room."
    r1_pattern_sents[4] = "Cara entered the sorting room."
    r1_pattern_sents[7] = "Eva walked from the bindery room to the sorting room."
    r1_pattern_sents[8] = "Eva entered the sorting room."
    r1_pattern_sents[11] = "Cara walked from the sorting room to the bindery room."
    r1_pattern_sents[12] = "Cara entered the bindery room."
    r1_pattern_sents[13] = "Cara walked from the bindery room to the reading room."
    r1_pattern_sents[14] = "Cara entered the reading room."
    assert check_rendered_prose(f"{setup_dan}\n\n{' '.join(r1_pattern_sents)}", w, focal_agent=fa) == ()

    # (d.2) Refusal test: a bare enter with NO matching leave sentence before it stays refused
    # Mutate world so event #2 is a 'see' by Eva instead of 'leave' by Dan, followed by bare enter by Dan on event #3:
    w_no_leave = replace(
        w,
        events=(
            w.events[0],
            w.events[0],  # event #2 is now 'see' by Eva, not 'leave' by Dan
            *w.events[2:],
        ),
    )
    sents_no_leave = list(sj_sents)
    sents_no_leave[1] = sj_sents[0]
    sents_no_leave[2] = "Dan entered the sorting room."
    with pytest.raises(RenderCheckError) as exc_no_leave:
        check_rendered_prose(f"{setup_dan}\n\n{' '.join(sents_no_leave)}", w_no_leave, focal_agent=fa)
    assert exc_no_leave.value.check_id == "c_fidelity_missing_event"
    assert exc_no_leave.value.failing_sentence == "Dan entered the sorting room."

    # Also test when the preceding leave sentence is by a different actor in prose:
    sents_wrong_actor_leave = list(sj_sents)
    sents_wrong_actor_leave[1] = "Cara walked from the bindery room to the sorting room."
    sents_wrong_actor_leave[2] = "Dan entered the sorting room."
    with pytest.raises(RenderCheckError) as exc_wrong_actor:
        check_rendered_prose(f"{setup_dan}\n\n{' '.join(sents_wrong_actor_leave)}", w, focal_agent=fa)
    assert exc_wrong_actor.value.check_id == "c_fidelity_missing_event"

    # (d.3) Refusal test: a bare enter whose preceding leave names a different from-room stays refused
    w_diff_from = replace(
        w,
        events=(
            w.events[0],
            replace(w.events[1], room="reading room", from_location="reading room", to_location="sorting room"),
            *w.events[2:],
        ),
    )
    sents_diff_from = list(sj_sents)
    sents_diff_from[1] = "Dan walked from the reading room to the sorting room."
    sents_diff_from[2] = "Dan entered the sorting room."
    with pytest.raises(RenderCheckError) as exc_diff_from:
        check_rendered_prose(f"{setup_dan}\n\n{' '.join(sents_diff_from)}", w_diff_from, focal_agent=fa)
    assert exc_diff_from.value.check_id == "c_fidelity_missing_event"
    assert exc_diff_from.value.failing_sentence == "Dan entered the sorting room."

    # (f) Prompt emphasis check
    prompt_str = format_writer_prompt(w, fa)
    assert "every enter sentence must name both from_location and to_location" in prompt_str


def test_53_card_st_repair_retry_instead_of_blind_retry(tmp_path: Path) -> None:
    """Test 53 (Card st Req c):
    - On a refusal, the second attempt is the same prompt plus ONE appended line naming the check id
      and the event number and kind, e.g. 'Repair retry (c_fidelity_missing_event): Event 3, enter: name the room the actor came from.'
    - The line carries no key content and echoes no model text.
    - Both attempts are recorded.
    - No temperature or seed noise is added (temperature == 0.0 and seed == base_seed on both attempts).
    """
    import hashlib
    from typing import Any
    import udlab_generator.render as ug_render
    from udlab_generator import (
        PINNED_WRITER_MANIFEST_DIGEST,
        PINNED_WRITER_MODEL_TAG,
    )

    w0 = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    passing_r0 = render_world_template(w0, w0.agents[0])
    passing_r1 = render_world_template(w0, w0.agents[1])
    sj_setup, sj_sents = _build_sj_writer_in_the_loop_scen_0000()
    setup_dan = sj_setup.replace("Following Eva", "Following Dan")

    # Attempt 0 for r1 fails on Event 17 (seer-less occluded see) with unique model text that must NOT be echoed:
    model_secret_text = "The rolling pillar blocked sight of the iron safe at the far-side alcove in the sorting room."
    bad_sents_r1 = list(sj_sents)
    bad_sents_r1[16] = model_secret_text
    bad_sents_r1[28] = model_secret_text
    failing_r1 = f"{setup_dan}\n\n{' '.join(bad_sents_r1)}"

    seen_calls: list[tuple[int, float, str]] = []

    def _repair_transport(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            assert payload is not None
            opts = payload.get("options") or {}
            seed_val = int(opts.get("seed", -1))
            temp_val = float(opts.get("temperature", -1.0))
            prompt_str = str(payload["messages"][0]["content"])
            idx = len(seen_calls)
            seen_calls.append((seed_val, temp_val, prompt_str))
            txt = passing_r0 if idx == 0 else (failing_r1 if idx == 1 else passing_r1)
            return {
                "message": {"role": "assistant", "content": txt},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3000,
                "eval_count": 780,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    out_dir = tmp_path / "st_repair_retry"
    summary = build_and_probe_dataset(
        count=1,
        base_seed=20261030,
        output_dir=out_dir,
        writer_kind="ollama",
        max_retries=1,
        grant_id="test-grant-st-repair",
        http_transport=_repair_transport,
        start_owned_server=False,
    )
    assert len(seen_calls) == 3
    seed_0, temp_0, prompt_attempt_0 = seen_calls[1]
    seed_1, temp_1, prompt_attempt_1 = seen_calls[2]

    # Attempt 0 uses temperature 0.0 and derived attempt-0 seed; retry uses RETRY_TEMPERATURE and derived attempt-1 seed:
    assert temp_0 == 0.0 and temp_1 == ug_render.RETRY_TEMPERATURE
    assert seed_0 == ug_render.derive_call_seed(int(w0.seed), 1, 0)
    assert seed_1 == ug_render.derive_call_seed(int(w0.seed), 1, 1)
    assert seed_0 != seed_1

    # Second attempt (single-turn by default) appends the single-turn repair line to the original prompt:
    appended_tail = ug_render.format_repair_retry_line(
        RenderCheckError(
            "test",
            check_id="c_fidelity_missing_event",
            failing_sentence=model_secret_text,
            event_index=17,
            event_kind="see",
            event_variant="occluded_see",
        )
    )
    assert prompt_attempt_1 == f"{prompt_attempt_0.rstrip('\n')}\n{appended_tail}"
    assert "\n" not in appended_tail, f"Expected exactly one repair line, got: {appended_tail!r}"
    assert "c_fidelity_missing_event" in appended_tail
    assert "event 17 (see" in appended_tail

    # The line carries no key content and echoes no model text:
    assert model_secret_text not in appended_tail
    assert "iron safe" not in appended_tail
    assert "rolling pillar" not in appended_tail
    assert "Dan" not in appended_tail
    for q in compute_scenario_keys(w0).questions:
        if isinstance(q.key, str) and q.key not in ("yes", "no", "none"):
            assert q.key.lower() not in appended_tail.lower()

    # Record both attempts:
    assert summary["calls_kept"] == 2
    assert summary["calls_refused"] == 1
    gen_rows = [
        json.loads(line)
        for line in (out_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(gen_rows) == 3
    assert gen_rows[1]["attempt"] == 0 and gen_rows[1]["seed"] == seed_0 and gen_rows[1]["checks_passed"] is False
    assert gen_rows[2]["attempt"] == 1 and gen_rows[2]["seed"] == seed_1 and gen_rows[2]["checks_passed"] is True
    assert gen_rows[1]["prompt_sha256"] != gen_rows[2]["prompt_sha256"]


def test_54_card_sw_before_actor_negation() -> None:
    """Test 54 (Card sw Req 1):
    - Sentence-level negation before the actor is refused unless the fronted part is a plain time or place phrase.
    - Refused: 'It is not true that, in the bindery room, Eva observed…', 'Never, Eva observed…', 'Not once, Eva observed…'.
    - Pass: 'Not long after, Eva observed…', 'Without delay, Eva observed…'.
    """
    from udlab_generator.render import _sentence_matches_event

    w = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    ev_1 = w.events[0]

    refused_sentences = (
        "It is not true that, in the bindery room, Eva observed the index reel in the linen bin.",
        "Never, Eva observed the index reel in the linen bin in the bindery room.",
        "Not once, Eva observed the index reel in the linen bin in the bindery room.",
        "Not once, in the bindery room, Eva observed the index reel in the linen bin.",
    )
    for s_ref in refused_sentences:
        assert not _sentence_matches_event(s_ref, ev_1, w.noise_threshold, world=w), (
            f"Expected sentence-level negation before actor to be refused: {s_ref!r}"
        )

    passing_sentences = (
        "Not long after, Eva observed the index reel in the linen bin in the bindery room.",
        "Without delay, Eva observed the index reel in the linen bin in the bindery room.",
        "Without delay Eva observed the index reel in the linen bin in the bindery room.",
        "Not far from the doorway, in the bindery room, Eva observed the index reel in the linen bin.",
    )
    for s_pass in passing_sentences:
        assert _sentence_matches_event(s_pass, ev_1, w.noise_threshold, world=w), (
            f"Expected plain fronted time/place phrase before actor to pass: {s_pass!r}"
        )


def test_55_card_sw_negative_implicative_verbs_and_bare_see() -> None:
    """Test 55 (Card sw Req 2):
    - Negative-implicative verbs (refused to, forgot to, failed to, declined to, neglected to, avoided, and similar)
      are in the single negation list and render in the prompt block.
    - Bare 'see' is allowed as a sighting verb ONLY after 'could not' on an occluded see.
    - Refused: 'Eva refused to see…', 'Eva forgot to see…'.
    """
    import udlab_generator.render as ug_render
    from udlab_generator.render import _sentence_matches_event

    w = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    fa = w.agents[0]
    ev_1 = w.events[0]  # unoccluded see by Eva
    ev_17 = w.events[16]  # occluded see by Dan

    required_neg_implicatives = (
        "refused to",
        "forgot to",
        "failed to",
        "declined to",
        "neglected to",
        "avoided",
        "omitted to",
        "refrained from",
    )
    for phrase in required_neg_implicatives:
        assert phrase in ug_render.NEGATION_PHRASES, f"Missing {phrase!r} from NEGATION_PHRASES"

    prompt_str = format_writer_prompt(w, fa)
    for phrase in required_neg_implicatives:
        assert phrase in prompt_str, f"Missing {phrase!r} from rendered writer prompt block"

    refused_see_sentences = (
        "In the bindery room, Eva refused to see the index reel in the linen bin.",
        "In the bindery room, Eva forgot to see the index reel in the linen bin.",
        "In the bindery room, Eva failed to see the index reel in the linen bin.",
        "In the bindery room, Eva declined to see the index reel in the linen bin.",
        "In the bindery room, Eva neglected to see the index reel in the linen bin.",
        "In the bindery room, Eva avoided seeing the index reel in the linen bin.",
        "In the bindery room, Eva avoided the index reel in the linen bin.",
        "In the bindery room, Eva wanted to see the index reel in the linen bin.",
        "In the bindery room, Eva see the index reel in the linen bin.",
    )
    for s_ref in refused_see_sentences:
        assert not _sentence_matches_event(s_ref, ev_1, w.noise_threshold, world=w), (
            f"Expected negative-implicative or bare 'see' sentence to be refused: {s_ref!r}"
        )

    # Bare 'see' on occluded see without 'could not' is also refused:
    assert not _sentence_matches_event(
        "In the sorting room, at the far-side alcove, Dan refused to see the iron safe because the rolling pillar blocked sight.",
        ev_17,
        w.noise_threshold,
        world=w,
    )
    assert not _sentence_matches_event(
        "In the sorting room, at the far-side alcove, Dan tried to see the iron safe because the rolling pillar blocked sight.",
        ev_17,
        w.noise_threshold,
        world=w,
    )
    # Bare 'see' after 'could not' on occluded see passes:
    assert _sentence_matches_event(
        "Dan could not see the iron safe at the far-side alcove in the sorting room because the rolling pillar blocked sight.",
        ev_17,
        w.noise_threshold,
        world=w,
    )


def test_56_card_sw_post_verb_adjuncts_and_articles() -> None:
    """Test 56 (Card sw Req 3, 4, 5):
    - Post-verb adjuncts after the event's object and location that do not scope the event verb pass:
      '...in the linen bin without touching it.', '..., not far from the doorway.', '..., which had not moved.'
    - '...but did not see the copper rack.' stays refused (c_fidelity_missing_event when replacing event #1,
      c_fidelity_unlogged_event when added as an extra sentence).
    - Negation between the verb and the object/location stays refused.
    - Articles a / an / the (or no article) are allowed before from-locations, to-locations, and target objects
      on every event kind, including the exact event-#7 sentence from writer-stub-1c.
    """
    from udlab_generator.render import _sentence_matches_event

    w = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    fa = w.agents[0]
    sj_setup, sj_sents = _build_sj_writer_in_the_loop_scen_0000()
    ev_1 = w.events[0]  # see by Eva
    ev_2 = w.events[1]  # leave by Dan
    ev_3 = w.events[2]  # enter by Dan
    ev_7 = w.events[6]  # move by Cara: brass ledger from intake bench to iron safe in sorting room

    # (3) Post-verb adjuncts after object and location pass:
    passing_adjuncts = (
        "In the bindery room, Eva observed the index reel in the linen bin without touching it.",
        "In the bindery room, Eva observed the index reel in the linen bin, not far from the doorway.",
        "In the bindery room, Eva observed the index reel in the linen bin, which had not moved.",
    )
    for s_adj in passing_adjuncts:
        assert _sentence_matches_event(s_adj, ev_1, w.noise_threshold, world=w), (
            f"Expected post-verb adjunct after object and location to pass: {s_adj!r}"
        )
        sents_mod = list(sj_sents)
        sents_mod[0] = s_adj
        assert check_rendered_prose(f"{sj_setup}\n\n{' '.join(sents_mod)}", w, focal_agent=fa) == ()

    # Keep refused: "...but did not see the copper rack."
    s_copper_rack = "In the bindery room, Eva observed the index reel in the linen bin, but did not see the copper rack."
    assert not _sentence_matches_event(s_copper_rack, ev_1, w.noise_threshold, world=w)
    with pytest.raises(RenderCheckError) as exc_copper:
        check_rendered_prose(f"{sj_setup}\n\n{' '.join([s_copper_rack, *sj_sents[1:]])}", w, focal_agent=fa)
    assert exc_copper.value.check_id == "c_fidelity_missing_event"

    # Negation before object/location is completed (using verb_end..core_end) stays refused:
    assert not _sentence_matches_event(
        "In the bindery room, Eva observed without touching it the index reel in the linen bin.",
        ev_1,
        w.noise_threshold,
        world=w,
    )
    assert not _sentence_matches_event(
        "In the bindery room, Eva observed not the index reel in the linen bin.",
        ev_1,
        w.noise_threshold,
        world=w,
    )

    # (5) Exact event-#7 sentence from writer-stub-1c:
    exact_ev7 = "Cara moved a brass ledger from an intake bench to the iron safe in the sorting room."
    assert _sentence_matches_event(exact_ev7, ev_7, w.noise_threshold, world=w)
    sents_ev7 = list(sj_sents)
    sents_ev7[6] = exact_ev7
    assert check_rendered_prose(f"{sj_setup}\n\n{' '.join(sents_ev7)}", w, focal_agent=fa) == ()

    # Articles a / an / the / none across enter, leave, move, see:
    assert _sentence_matches_event("Dan walked from a bindery room into a sorting room.", ev_3, w.noise_threshold, world=w)
    assert _sentence_matches_event("Dan walked from bindery room into sorting room.", ev_3, w.noise_threshold, world=w)
    assert _sentence_matches_event("Dan exited a bindery room toward a sorting room.", ev_2, w.noise_threshold, world=w)
    assert _sentence_matches_event("In a sorting room, Cara moved brass ledger from intake bench to an iron safe.", ev_7, w.noise_threshold, world=w)
    assert _sentence_matches_event("In a bindery room, Eva observed an index reel in a linen bin.", ev_1, w.noise_threshold, world=w)


def test_57_card_sw_folded_leave_enter() -> None:
    """Test 57 (Card sw Req 7):
    - A single movement sentence naming actor, from-room, and to-room ('Dan walked from the bindery room to the sorting room.')
      satisfies BOTH consecutive leave and enter when the next sentence is not an enter.
    - A folded sentence naming only the to-room ('Dan went to the sorting room.') stays refused.
    - A folded form followed by an explicit enter sentence ('Dan entered the sorting room.') is counted once for leave
      and once for enter, and stays valid (while a third duplicate sentence is refused as unlogged).
    """
    w = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    fa = w.agents[1]  # Dan
    sj_setup, sj_sents = _build_sj_writer_in_the_loop_scen_0000()
    setup_dan = sj_setup.replace("Following Eva", "Following Dan")

    # (1) Folded leave+enter passes when the explicit enter sentences are omitted (next sentence is not an enter):
    # In scen_0000, the 5 consecutive leave+enter pairs are at 0-based indices (1,2), (3,4), (7,8), (11,12), (13,14), (19,20).
    folded_sents: list[str] = []
    skip_indices = {2, 4, 8, 12, 14, 20}
    folded_by_leave_idx = {
        1: "Dan walked from the bindery room to the sorting room.",
        3: "Cara walked from the bindery room to the sorting room.",
        7: "Eva walked from the bindery room to the sorting room.",
        11: "Cara walked from the sorting room to the bindery room.",
        13: "Cara walked from the bindery room to the reading room.",
        19: "Eva walked from the sorting room to the bindery room.",
    }
    for idx, sent in enumerate(sj_sents):
        if idx in skip_indices:
            continue
        folded_sents.append(folded_by_leave_idx.get(idx, sent))
    assert len(folded_sents) == len(w.events) - 6
    assert check_rendered_prose(f"{setup_dan}\n\n{' '.join(folded_sents)}", w, focal_agent=fa) == ()

    # (2) Folded sentence naming only the to-room ("Dan went to the sorting room.") stays refused:
    bad_to_only = list(folded_sents)
    bad_to_only[1] = "Dan went to the sorting room."
    with pytest.raises(RenderCheckError) as exc_to_only:
        check_rendered_prose(f"{setup_dan}\n\n{' '.join(bad_to_only)}", w, focal_agent=fa)
    assert exc_to_only.value.check_id == "c_fidelity_missing_event"
    assert exc_to_only.value.failing_sentence == "Dan went to the sorting room."

    # Also test when the folded sentence names the from-room only via 'In the bindery room, Dan went to the sorting room.'
    # without a source preposition ('from'/'out of'/'left'/'exited'/'departed'):
    bad_no_from_prep = list(folded_sents)
    bad_no_from_prep[1] = "In the bindery room, Dan went to the sorting room."
    with pytest.raises(RenderCheckError) as exc_no_from_prep:
        check_rendered_prose(f"{setup_dan}\n\n{' '.join(bad_no_from_prep)}", w, focal_agent=fa)
    assert exc_no_from_prep.value.check_id == "c_fidelity_missing_event"

    # (3) Folded form followed by an explicit enter sentence passes (counted once for leave and once for enter):
    with_explicit_enter = list(sj_sents)
    with_explicit_enter[1] = "Dan walked from the bindery room to the sorting room."
    with_explicit_enter[2] = "Dan entered the sorting room."
    assert check_rendered_prose(f"{setup_dan}\n\n{' '.join(with_explicit_enter)}", w, focal_agent=fa) == ()

    # And if followed by TWO explicit enter sentences, the extra sentence is refused as c_fidelity_unlogged_event:
    with_duplicate_enter = [
        with_explicit_enter[0],
        with_explicit_enter[1],
        with_explicit_enter[2],
        "Dan entered the sorting room from the bindery room.",
        *with_explicit_enter[3:],
    ]
    with pytest.raises(RenderCheckError) as exc_dup:
        check_rendered_prose(f"{setup_dan}\n\n{' '.join(with_duplicate_enter)}", w, focal_agent=fa)
    assert exc_dup.value.check_id == "c_fidelity_unlogged_event"


def test_58_card_sw_repair_line_recorded_and_multi_turn_wording(tmp_path: Path) -> None:
    """Test 58 (Card sw Req 6 & 8):
    - On the retry call's row in generations.jsonl, record the repair line text, its sha256, retry_form ('multi_turn'),
      and unchanged sentence count between attempt 0 and the retry.
    - Multi-turn retry sends attempt 0 as the assistant turn and the repair line as the next user turn,
      staying under num_ctx <= 9216 against measured prompt_eval_count (~3070) + eval_count (~870).
    - Repair line asks for ONE fix, says the rest stays as written ('Keep your earlier text exactly as it was; change only ...'),
      names event N for event checks and the setup item for setup checks, never contains 'rewrite',
      carries no key content, and is generated from WRITER_PROSE_REQUIREMENTS.
    """
    import hashlib
    from typing import Any
    import udlab_generator.render as ug_render
    from udlab_generator import (
        PINNED_WRITER_MANIFEST_DIGEST,
        PINNED_WRITER_MODEL_TAG,
    )
    from udlab_generator.render import (
        WRITER_PROSE_REQUIREMENTS,
        format_repair_retry_line,
    )

    w0 = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    skeys = compute_scenario_keys(w0)
    forbidden_key_strings = {
        q.key.lower()
        for q in skeys.questions
        if isinstance(q.key, str) and q.key.lower() not in ("yes", "no", "none")
    }

    # Check offline across every check_id in WRITER_PROSE_REQUIREMENTS and every event kind:
    for req in WRITER_PROSE_REQUIREMENTS:
        if req.check_id in ("c_fidelity_missing_event", "c_fidelity_event_order"):
            for ev_k in ("enter", "leave", "move", "see", "say", "message"):
                exc_ev = RenderCheckError(
                    "synthetic event error",
                    check_id=req.check_id,
                    failing_sentence="synthetic secret sentence",
                    event_index=7,
                    event_kind=ev_k,
                )
                line_ev = format_repair_retry_line(exc_ev, retry_form="multi_turn")
                assert line_ev.startswith("Keep your earlier text exactly as it was; change only event 7 ")
                assert req.check_id in line_ev
                assert ev_k in line_ev
                assert "rewrite" not in line_ev.lower()
                assert "synthetic secret sentence" not in line_ev
                for k_str in forbidden_key_strings:
                    assert k_str not in line_ev.lower()
        else:
            exc_setup = RenderCheckError(
                "synthetic setup error",
                check_id=req.check_id,
                failing_sentence="synthetic secret sentence",
            )
            line_setup = format_repair_retry_line(exc_setup, retry_form="multi_turn")
            assert line_setup.startswith("Keep your earlier text exactly as it was; change only ")
            if "setup" in req.rule_text.lower():
                assert "setup" in line_setup.split("so that")[0].lower()
                assert "event " not in line_setup.split("so that")[0].lower()
            assert req.check_id in line_setup
            assert req.rule_text in line_setup
            assert "rewrite" not in line_setup.lower()
            assert "synthetic secret sentence" not in line_setup
            for k_str in forbidden_key_strings:
                assert k_str not in line_setup.lower()

    # Now test end-to-end retry recording in generations.jsonl with multi-turn messages and num_ctx <= 9216:
    passing_r0 = render_world_template(w0, w0.agents[0])
    passing_r1 = render_world_template(w0, w0.agents[1])
    # Mutate 1 sentence in passing_r1 for attempt 0 (wrong stake dependency object):
    orig_stake = w0.stake_edges[0].description
    bad_stake = "Gia is an absent party who depends on the location of the index reel for tomorrow's courier handoff."
    failing_r1 = passing_r1.replace(orig_stake, bad_stake)
    assert failing_r1 != passing_r1

    seen_payloads: list[dict[str, Any]] = []

    def _retry_transport(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            assert payload is not None
            idx = len(seen_payloads)
            seen_payloads.append(payload)
            txt = passing_r0 if idx == 0 else (failing_r1 if idx == 1 else passing_r1)
            return {
                "message": {"role": "assistant", "content": txt},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3070 if idx < 2 else 3980,
                "eval_count": 870,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    out_dir = tmp_path / "sw_repair_retry"
    build_and_probe_dataset(
        count=1,
        base_seed=20261030,
        output_dir=out_dir,
        writer_kind="ollama",
        max_retries=1,
        retry_form="multi_turn",
        grant_id="test-grant-sw-repair",
        http_transport=_retry_transport,
        start_owned_server=False,
    )
    assert len(seen_payloads) == 3

    # Check multi-turn payload on attempt 1 (seen_payloads[2]):
    retry_payload = seen_payloads[2]
    retry_msgs = retry_payload["messages"]
    assert len(retry_msgs) == 3
    assert retry_msgs[0]["role"] == "user" and retry_msgs[0]["content"] == seen_payloads[1]["messages"][0]["content"]
    assert retry_msgs[1]["role"] == "assistant" and retry_msgs[1]["content"] == failing_r1
    assert retry_msgs[2]["role"] == "user"
    sent_repair_line = retry_msgs[2]["content"]
    assert sent_repair_line.startswith("Keep your earlier text exactly as it was; change only ")
    assert "c_fidelity_stake_edge" in sent_repair_line
    assert "rewrite" not in sent_repair_line.lower()
    assert int(retry_payload["options"]["num_ctx"]) <= 9216

    # Check generations.jsonl row for attempt 1:
    gen_rows = [
        json.loads(line)
        for line in (out_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(gen_rows) == 3
    assert gen_rows[1]["attempt"] == 0 and gen_rows[1]["repair_line"] == "" and gen_rows[1]["repair_line_sha256"] == ""
    retry_row = gen_rows[2]
    assert retry_row["attempt"] == 1
    assert retry_row["repair_line"] == sent_repair_line
    assert retry_row["repair_line_sha256"] == hashlib.sha256(sent_repair_line.encode("utf-8")).hexdigest()
    assert retry_row["retry_form"] == "multi_turn"
    assert isinstance(retry_row["retry_unchanged_sentences"], int)
    assert isinstance(retry_row["retry_attempt0_sentences"], int)
    # Only the 1 setup sentence with the stake edge changed between failing_r1 and passing_r1:
    assert retry_row["retry_unchanged_sentences"] == retry_row["retry_attempt0_sentences"] - 1


def test_59_card_sx_multiturn_fallback_not_cap(tmp_path: Path) -> None:
    """Test 59 (Card sx Req 1):
    - Before a multi-turn retry, compute need = prompt_eval_count(attempt 0) + eval_count(attempt 0) + repair_tokens + num_predict + margin.
    - If need <= 9216, use multi_turn (`retry_form == 'multi_turn'`, `retry_fallback_reason == ''`).
    - If need > 9216, fall back to single_turn (`retry_form == 'single_turn'`) with a non-empty `retry_fallback_reason` field.
    - Never truncate and never silently lower `num_predict`.
    - Tests both sides of the boundary (exact 9216 vs 9217) with fake counts.
    """
    import hashlib
    from typing import Any
    from udlab_generator import (
        PINNED_WRITER_MANIFEST_DIGEST,
        PINNED_WRITER_MODEL_TAG,
    )
    from udlab_generator.render import (
        DEFAULT_WRITER_MAX_NUM_CTX,
        DEFAULT_WRITER_NUM_PREDICT,
        NUM_CTX_MARGIN_TOKENS,
        estimate_prompt_tokens,
        format_repair_retry_line,
    )

    w0 = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    passing_r0 = render_world_template(w0, w0.agents[0])
    passing_r1 = render_world_template(w0, w0.agents[1])
    orig_stake = w0.stake_edges[0].description
    bad_stake = "Gia is an absent party who depends on the location of the index reel for tomorrow's courier handoff."
    failing_r1 = passing_r1.replace(orig_stake, bad_stake)

    expected_repair = format_repair_retry_line(
        RenderCheckError(
            "missing stake",
            check_id="c_fidelity_stake_edge",
            failing_sentence=bad_stake,
        ),
        retry_form="multi_turn",
    )
    repair_tokens = estimate_prompt_tokens(expected_repair)
    # Boundary budget for (p0 + e0) so that p0 + e0 + repair_tokens + 4096 + 256 == 9216:
    exact_fit_p0_plus_e0 = DEFAULT_WRITER_MAX_NUM_CTX - (
        repair_tokens + DEFAULT_WRITER_NUM_PREDICT + NUM_CTX_MARGIN_TOKENS
    )
    p0_fit = 3000
    e0_fit = exact_fit_p0_plus_e0 - p0_fit
    assert p0_fit + e0_fit + repair_tokens + DEFAULT_WRITER_NUM_PREDICT + NUM_CTX_MARGIN_TOKENS == 9216

    # (A) Side 1 of boundary: need == 9216 -> multi_turn
    payloads_fit: list[dict[str, Any]] = []

    def _transport_fit(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            assert payload is not None
            idx = len(payloads_fit)
            payloads_fit.append(payload)
            txt = passing_r0 if idx == 0 else (failing_r1 if idx == 1 else passing_r1)
            return {
                "message": {"role": "assistant", "content": txt},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": p0_fit,
                "eval_count": e0_fit,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    out_fit = tmp_path / "sx_boundary_fit"
    build_and_probe_dataset(
        count=1,
        base_seed=20261030,
        output_dir=out_fit,
        writer_kind="ollama",
        max_retries=1,
        retry_form="multi_turn",
        grant_id="test-grant-sx-fit",
        http_transport=_transport_fit,
        start_owned_server=False,
    )
    assert len(payloads_fit) == 3
    retry_payload_fit = payloads_fit[2]
    assert len(retry_payload_fit["messages"]) == 3
    assert retry_payload_fit["options"]["num_predict"] == DEFAULT_WRITER_NUM_PREDICT
    assert retry_payload_fit["options"]["num_ctx"] == 9216
    rows_fit = [
        json.loads(line)
        for line in (out_fit / "generations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows_fit[2]["retry_form"] == "multi_turn"
    assert rows_fit[2]["num_predict"] == DEFAULT_WRITER_NUM_PREDICT
    assert rows_fit[2]["num_ctx"] == 9216
    assert rows_fit[2].get("retry_fallback_reason", "") == ""

    # (B) Side 2 of boundary: need == 9217 (e0_fit + 1) -> single_turn fallback with reason
    p0_over = p0_fit
    e0_over = e0_fit + 1
    assert p0_over + e0_over + repair_tokens + DEFAULT_WRITER_NUM_PREDICT + NUM_CTX_MARGIN_TOKENS == 9217
    payloads_over: list[dict[str, Any]] = []

    def _transport_over(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            assert payload is not None
            idx = len(payloads_over)
            payloads_over.append(payload)
            txt = passing_r0 if idx == 0 else (failing_r1 if idx == 1 else passing_r1)
            return {
                "message": {"role": "assistant", "content": txt},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": p0_over,
                "eval_count": e0_over,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    out_over = tmp_path / "sx_boundary_over"
    build_and_probe_dataset(
        count=1,
        base_seed=20261030,
        output_dir=out_over,
        writer_kind="ollama",
        max_retries=1,
        retry_form="multi_turn",
        grant_id="test-grant-sx-over",
        http_transport=_transport_over,
        start_owned_server=False,
    )
    assert len(payloads_over) == 3
    retry_payload_over = payloads_over[2]
    assert len(retry_payload_over["messages"]) == 1
    assert retry_payload_over["messages"][0]["role"] == "user"
    orig_prompt_r1 = payloads_over[1]["messages"][0]["content"]
    expected_single_repair = format_repair_retry_line(
        RenderCheckError(
            "missing stake",
            check_id="c_fidelity_stake_edge",
            failing_sentence=bad_stake,
        ),
        retry_form="single_turn",
    )
    assert retry_payload_over["messages"][0]["content"] == f"{orig_prompt_r1.rstrip('\n')}\n{expected_single_repair}"
    assert retry_payload_over["options"]["num_predict"] == DEFAULT_WRITER_NUM_PREDICT
    assert retry_payload_over["options"]["num_ctx"] <= 9216

    rows_over = [
        json.loads(line)
        for line in (out_over / "generations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    retry_row_over = rows_over[2]
    assert retry_row_over["retry_form"] == "single_turn"
    assert "need=9217" in str(retry_row_over["retry_fallback_reason"])
    assert "max_num_ctx=9216" in str(retry_row_over["retry_fallback_reason"])
    assert retry_row_over["num_predict"] == DEFAULT_WRITER_NUM_PREDICT
    expected_single_sha = hashlib.sha256(
        json.dumps(retry_payload_over["messages"], sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    assert retry_row_over["prompt_sha256"] == expected_single_sha


def test_63_card_sz_folded_leave_enter_verbs_and_no_double_count() -> None:
    """Test 63 (Card sz Req 1 & 2):
    - Req 1: A folded sentence naming the actor, `from <from_room>` and arrival at `<to_room>` is accepted with any verb
      in the union of the enter and leave sets:
      - 'Dan entered the sorting room from the bindery room.' passes;
      - 'Dan arrived in the sorting room from the bindery room.' passes;
      - 'Dan moved from the bindery room to the sorting room.' passes;
      - 'Dan went to the sorting room.' stays refused.
    - Req 2: Two consecutive folded movement sentences cannot cover one logged leave+enter pair; the second is unlogged.
      A folded sentence followed by a pure enter sentence ('Dan entered the sorting room.') still passes, counted once each.
    """
    from udlab_generator.render import (
        _ACTION_VERBS_BY_KIND,
        _sentence_matches_folded_leave_enter,
    )

    w = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    fa = w.agents[1]  # Dan
    sj_setup, sj_sents = _build_sj_writer_in_the_loop_scen_0000()
    setup_dan = sj_setup.replace("Following Eva", "Following Dan")
    ev_2 = w.events[1]  # leave by Dan: bindery room -> sorting room
    ev_3 = w.events[2]  # enter by Dan: bindery room -> sorting room

    # Base folded event list for scen_0000 (omitting separate enter sentences at indices 2, 4, 8, 12, 14, 20):
    skip_indices = {2, 4, 8, 12, 14, 20}
    folded_by_leave_idx = {
        1: "Dan walked from the bindery room to the sorting room.",
        3: "Cara walked from the bindery room to the sorting room.",
        7: "Eva walked from the bindery room to the sorting room.",
        11: "Cara walked from the sorting room to the bindery room.",
        13: "Cara walked from the bindery room to the reading room.",
        19: "Eva walked from the sorting room to the bindery room.",
    }
    base_folded_sents = [
        folded_by_leave_idx.get(idx, sent)
        for idx, sent in enumerate(sj_sents)
        if idx not in skip_indices
    ]

    # Req 1: Three explicit brief sentences pass both _sentence_matches_folded_leave_enter and full check_rendered_prose:
    req1_passing = (
        "Dan entered the sorting room from the bindery room.",
        "Dan arrived in the sorting room from the bindery room.",
        "Dan moved from the bindery room to the sorting room.",
    )
    for s_pass in req1_passing:
        assert _sentence_matches_folded_leave_enter(s_pass, ev_2, ev_3, w.noise_threshold, world=w), (
            f"Expected folded sentence to match folded leave+enter: {s_pass!r}"
        )
        sents_try = list(base_folded_sents)
        sents_try[1] = s_pass
        assert check_rendered_prose(f"{setup_dan}\n\n{' '.join(sents_try)}", w, focal_agent=fa) == ()

    # Every verb in the union of enter and leave sets passes when naming actor, from <from_room>, and arrival at <to_room>:
    union_verbs = set(_ACTION_VERBS_BY_KIND["enter"]) | set(_ACTION_VERBS_BY_KIND["leave"])
    for verb in sorted(union_verbs):
        if verb == "entered":
            cand = "Dan entered the sorting room from the bindery room."
        elif verb == "arrived":
            cand = "Dan arrived in the sorting room from the bindery room."
        else:
            cand = f"Dan {verb} from the bindery room into the sorting room."
        assert _sentence_matches_folded_leave_enter(cand, ev_2, ev_3, w.noise_threshold, world=w), (
            f"Expected union verb {verb!r} to pass folded leave+enter: {cand!r}"
        )

    # Req 1 negative direction: "Dan went to the sorting room." stays refused:
    assert not _sentence_matches_folded_leave_enter(
        "Dan went to the sorting room.", ev_2, ev_3, w.noise_threshold, world=w
    )
    sents_ref = list(base_folded_sents)
    sents_ref[1] = "Dan went to the sorting room."
    with pytest.raises(RenderCheckError) as exc_went:
        check_rendered_prose(f"{setup_dan}\n\n{' '.join(sents_ref)}", w, focal_agent=fa)
    assert exc_went.value.check_id == "c_fidelity_missing_event"
    assert exc_went.value.failing_sentence == "Dan went to the sorting room."

    # Req 2: Two consecutive folded movement sentences cannot cover one logged leave+enter pair; the second is unlogged:
    for second_folded in (
        "Dan walked from the bindery room to the sorting room.",
        "Dan headed into the sorting room from the bindery room.",
        "Dan moved from the bindery room to the sorting room.",
    ):
        sents_double_folded = list(sj_sents)
        sents_double_folded[1] = "Dan walked from the bindery room to the sorting room."
        sents_double_folded[2] = second_folded
        with pytest.raises(RenderCheckError) as exc_double:
            check_rendered_prose(f"{setup_dan}\n\n{' '.join(sents_double_folded)}", w, focal_agent=fa)
        assert exc_double.value.check_id == "c_fidelity_unlogged_event"
        assert exc_double.value.failing_sentence == second_folded

    # Req 2 positive direction: A folded sentence followed by an arrival-half enter sentence
    # ("Dan entered the sorting room." or "Dan entered the sorting room from the bindery room.") still passes, counted once each:
    sents_folded_then_pure_enter = list(sj_sents)
    sents_folded_then_pure_enter[1] = "Dan walked from the bindery room to the sorting room."
    sents_folded_then_pure_enter[2] = "Dan entered the sorting room."
    assert check_rendered_prose(f"{setup_dan}\n\n{' '.join(sents_folded_then_pure_enter)}", w, focal_agent=fa) == ()
    sents_folded_then_enter_from = list(sj_sents)
    sents_folded_then_enter_from[1] = "Dan walked from the bindery room to the sorting room."
    sents_folded_then_enter_from[2] = "Dan entered the sorting room from the bindery room."
    assert check_rendered_prose(f"{setup_dan}\n\n{' '.join(sents_folded_then_enter_from)}", w, focal_agent=fa) == ()


def test_64_card_sz_post_argument_adjuncts() -> None:
    """Test 64 (Card sz Req 3):
    - A post-argument `without <gerund>` or relative clause is refused when it negates the event's own verb
      (any verb of that kind), its target object, or its location.
    - Refused:
      - '...without seeing it.'
      - '...without moving it.'
      - '..., which was not in the linen bin.'
      - '..., where Eva did not see the index reel.'
    - Benign adjuncts stay passing:
      - '...without touching it.'
      - '..., not far from the doorway.'
      - '..., which had not moved.'
    """
    from udlab_generator.render import _sentence_matches_event

    w = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    fa = w.agents[0]
    sj_setup, sj_sents = _build_sj_writer_in_the_loop_scen_0000()
    ev_1 = w.events[0]  # see by Eva: index reel in linen bin in bindery room
    ev_2 = w.events[1]  # leave by Dan: bindery room -> sorting room
    ev_3 = w.events[2]  # enter by Dan: bindery room -> sorting room
    ev_7 = w.events[6]  # move by Cara: brass ledger from intake bench to iron safe in sorting room

    refused_see = (
        "In the bindery room, Eva observed the index reel in the linen bin without seeing it.",
        "In the bindery room, Eva observed the index reel in the linen bin without observing it.",
        "In the bindery room, Eva observed the index reel in the linen bin without noticing it.",
        "In the bindery room, Eva observed the index reel in the linen bin without spotting the index reel.",
        "In the bindery room, Eva observed the index reel in the linen bin, which was not in the linen bin.",
        "In the bindery room, Eva observed the index reel in the linen bin, which was not the index reel.",
        "In the bindery room, Eva observed the index reel in the linen bin, where Eva did not see the index reel.",
        "In the bindery room, Eva observed the index reel in the linen bin, who did not observe it.",
    )
    for s_ref in refused_see:
        assert not _sentence_matches_event(s_ref, ev_1, w.noise_threshold, world=w), (
            f"Expected self-contradictory post-arg adjunct on see to be refused: {s_ref!r}"
        )
        with pytest.raises(RenderCheckError) as exc_info:
            check_rendered_prose(f"{sj_setup}\n\n{' '.join([s_ref, *sj_sents[1:]])}", w, focal_agent=fa)
        assert exc_info.value.check_id == "c_fidelity_missing_event"

    refused_move = (
        "In the sorting room, Cara moved the brass ledger from the intake bench to the iron safe without moving it.",
        "In the sorting room, Cara moved the brass ledger from the intake bench to the iron safe without placing it in the iron safe.",
        "In the sorting room, Cara moved the brass ledger from the intake bench to the iron safe, which was not in the iron safe.",
        "In the sorting room, Cara moved the brass ledger from the intake bench to the iron safe, where Cara did not move the brass ledger.",
    )
    for s_ref_m in refused_move:
        assert not _sentence_matches_event(s_ref_m, ev_7, w.noise_threshold, world=w), (
            f"Expected self-contradictory post-arg adjunct on move to be refused: {s_ref_m!r}"
        )

    assert not _sentence_matches_event(
        "Dan walked from the bindery room to the sorting room without entering the sorting room.",
        ev_3,
        w.noise_threshold,
        world=w,
    )
    assert not _sentence_matches_event(
        "Dan walked from the bindery room to the sorting room without leaving the bindery room.",
        ev_2,
        w.noise_threshold,
        world=w,
    )

    # Benign adjuncts stay passing:
    passing_benign = (
        "In the bindery room, Eva observed the index reel in the linen bin without touching it.",
        "In the bindery room, Eva observed the index reel in the linen bin, not far from the doorway.",
        "In the bindery room, Eva observed the index reel in the linen bin, which had not moved.",
    )
    for s_pass in passing_benign:
        assert _sentence_matches_event(s_pass, ev_1, w.noise_threshold, world=w), (
            f"Expected benign post-arg adjunct to stay passing: {s_pass!r}"
        )
        assert check_rendered_prose(f"{sj_setup}\n\n{' '.join([s_pass, *sj_sents[1:]])}", w, focal_agent=fa) == ()


def test_65_card_sz_fronted_phrases_and_no() -> None:
    """Test 65 (Card sz Req 4):
    - A fronted 'not far from <the event's room>' must not count as the event's room.
      'Not far from the bindery room, Eva observed...' is refused for an event in the bindery room
      unless the room is also stated with in/at/inside.
    - The fronted strip ends at the comma: 'Not long after failing, ...' is refused.
    - 'no <target>' and sentence-level 'under no circumstances', 'by no means', and 'at no time' are refused.
    - Keep 'Not long after, Eva observed...' and 'Without delay, Eva observed...' passing.
    """
    from udlab_generator.render import _sentence_matches_event

    w = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    fa = w.agents[0]
    sj_setup, sj_sents = _build_sj_writer_in_the_loop_scen_0000()
    ev_1 = w.events[0]  # see by Eva: index reel in linen bin in bindery room
    ev_7 = w.events[6]  # move by Cara: brass ledger from intake bench to iron safe in sorting room

    # (a) Fronted "Not far from the bindery room, Eva observed..." refused unless room is also stated with in/at/inside:
    s_not_far_room_only = "Not far from the bindery room, Eva observed the index reel in the linen bin."
    assert not _sentence_matches_event(s_not_far_room_only, ev_1, w.noise_threshold, world=w)
    with pytest.raises(RenderCheckError) as exc_nf:
        check_rendered_prose(f"{sj_setup}\n\n{' '.join([s_not_far_room_only, *sj_sents[1:]])}", w, focal_agent=fa)
    assert exc_nf.value.check_id == "c_fidelity_missing_event"

    # When the room IS also stated with in/at/inside, or when "not far from the doorway" is used with "in the bindery room", it passes:
    s_not_far_with_in_room = "Not far from the bindery room doorway, in the bindery room, Eva observed the index reel in the linen bin."
    assert _sentence_matches_event(s_not_far_with_in_room, ev_1, w.noise_threshold, world=w)
    s_not_far_room_plus_in = "Not far from the bindery room, inside the bindery room, Eva observed the index reel in the linen bin."
    assert _sentence_matches_event(s_not_far_room_plus_in, ev_1, w.noise_threshold, world=w)

    # (b) Fronted strip ends at the comma: "Not long after failing, ..." is refused:
    refused_fronted_comma = (
        "Not long after failing, in the bindery room, Eva observed the index reel in the linen bin.",
        "Not long after refusing to help, in the bindery room, Eva observed the index reel in the linen bin.",
        "Without delay or not, in the bindery room, Eva observed the index reel in the linen bin.",
    )
    for s_fc in refused_fronted_comma:
        assert not _sentence_matches_event(s_fc, ev_1, w.noise_threshold, world=w), (
            f"Expected non-plain fronted phrase before comma to be refused: {s_fc!r}"
        )

    # (c) "no <target>" and sentence-level "under no circumstances", "by no means", and "at no time" are refused:
    refused_no_sentences = (
        "Eva observed no index reel in the linen bin in the bindery room.",
        "In the bindery room, Eva observed no index reel in the linen bin.",
        "Under no circumstances, in the bindery room, Eva observed the index reel in the linen bin.",
        "By no means, in the bindery room, Eva observed the index reel in the linen bin.",
        "At no time, in the bindery room, Eva observed the index reel in the linen bin.",
        "In the bindery room, under no circumstances did Eva observe the index reel in the linen bin.",
        "In the bindery room, by no means did Eva observe the index reel in the linen bin.",
        "In the bindery room, at no time did Eva observe the index reel in the linen bin.",
    )
    for s_no in refused_no_sentences:
        assert not _sentence_matches_event(s_no, ev_1, w.noise_threshold, world=w), (
            f"Expected 'no <target>' or sentence-level 'no' phrase to be refused: {s_no!r}"
        )
    assert not _sentence_matches_event(
        "In the sorting room, Cara moved no brass ledger from the intake bench to the iron safe.",
        ev_7,
        w.noise_threshold,
        world=w,
    )

    # (d) Keep "Not long after, Eva observed..." and "Without delay, Eva observed..." passing:
    passing_fronted = (
        "Not long after, Eva observed the index reel in the linen bin in the bindery room.",
        "Without delay, Eva observed the index reel in the linen bin in the bindery room.",
        "Not far from the doorway, in the bindery room, Eva observed the index reel in the linen bin.",
    )
    for s_pf in passing_fronted:
        assert _sentence_matches_event(s_pf, ev_1, w.noise_threshold, world=w), (
            f"Expected valid fronted phrase to stay passing: {s_pf!r}"
        )
        assert check_rendered_prose(f"{sj_setup}\n\n{' '.join([s_pf, *sj_sents[1:]])}", w, focal_agent=fa) == ()


def test_66_card_sz_single_turn_repair_wording(tmp_path: Path) -> None:
    """Test 66 (Card sz Req 5):
    - Pass `retry_form` to `format_repair_retry_line`.
    - When `retry_form == 'multi_turn'` (default), the line keeps the multi-turn wording ('Keep your earlier text...').
    - When the retry falls back to `single_turn`, the repair line must not say 'keep your earlier text',
      and uses a single-turn wording that names the one item to satisfy (without 'rewrite', key content, or model text).
    - Tests both forms directly and end-to-end through `build_and_probe_dataset`.
    """
    from typing import Any
    from udlab_generator import (
        PINNED_WRITER_MANIFEST_DIGEST,
        PINNED_WRITER_MODEL_TAG,
    )
    from udlab_generator.render import (
        WRITER_PROSE_REQUIREMENTS,
        format_repair_retry_line,
    )

    w0 = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    skeys = compute_scenario_keys(w0)
    forbidden_key_strings = {
        q.key.lower()
        for q in skeys.questions
        if isinstance(q.key, str) and q.key.lower() not in ("yes", "no", "none")
    }

    for req in WRITER_PROSE_REQUIREMENTS:
        if req.check_id in ("c_fidelity_missing_event", "c_fidelity_event_order"):
            for ev_k in ("enter", "leave", "move", "see", "say", "message"):
                exc_ev = RenderCheckError(
                    "synthetic event error",
                    check_id=req.check_id,
                    failing_sentence="synthetic secret sentence",
                    event_index=7,
                    event_kind=ev_k,
                )
                line_mt = format_repair_retry_line(exc_ev, retry_form="multi_turn")
                line_st = format_repair_retry_line(exc_ev, retry_form="single_turn")
                assert "keep your earlier text" in line_mt.lower()
                assert "keep your earlier text" not in line_st.lower()
                assert "earlier text" not in line_st.lower()
                assert "leaving every other sentence unchanged" not in line_st.lower()
                assert f"event 7 ({ev_k}, check {req.check_id})" in line_st
                assert "rewrite" not in line_st.lower()
                assert "synthetic secret sentence" not in line_st
                for k_str in forbidden_key_strings:
                    assert k_str not in line_st.lower()
        else:
            exc_setup = RenderCheckError(
                "synthetic setup error",
                check_id=req.check_id,
                failing_sentence="synthetic secret sentence",
            )
            line_mt = format_repair_retry_line(exc_setup, retry_form="multi_turn")
            line_st = format_repair_retry_line(exc_setup, retry_form="single_turn")
            assert "keep your earlier text" in line_mt.lower()
            assert "keep your earlier text" not in line_st.lower()
            assert "earlier text" not in line_st.lower()
            assert "leave every other sentence unchanged" not in line_st.lower()
            assert req.check_id in line_st
            assert req.rule_text in line_st
            assert "rewrite" not in line_st.lower()
            assert "synthetic secret sentence" not in line_st
            for k_str in forbidden_key_strings:
                assert k_str not in line_st.lower()

    # End-to-end check on both multi_turn and single_turn retries:
    passing_r0 = render_world_template(w0, w0.agents[0])
    passing_r1 = render_world_template(w0, w0.agents[1])
    orig_stake = w0.stake_edges[0].description
    bad_stake = "Gia is an absent party who depends on the location of the index reel for tomorrow's courier handoff."
    failing_r0 = passing_r0.replace(orig_stake, bad_stake)
    failing_r1 = passing_r1.replace(orig_stake, bad_stake)

    seen_payloads: list[dict[str, Any]] = []

    def _transport(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            assert payload is not None
            idx = len(seen_payloads)
            seen_payloads.append(payload)
            if idx == 0:
                return {
                    "message": {"role": "assistant", "content": failing_r0},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 3070,
                    "eval_count": 814,
                }
            if idx == 1:
                return {
                    "message": {"role": "assistant", "content": passing_r0},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 3950,
                    "eval_count": 814,
                }
            if idx == 2:
                return {
                    "message": {"role": "assistant", "content": failing_r1},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 3070,
                    "eval_count": 2000,
                }
            return {
                "message": {"role": "assistant", "content": passing_r1},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3180,
                "eval_count": 814,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    out_dir = tmp_path / "sz_single_turn_wording"
    build_and_probe_dataset(
        count=1,
        base_seed=20261030,
        output_dir=out_dir,
        writer_kind="ollama",
        max_retries=1,
        retry_form="multi_turn",
        grant_id="test-grant-sz-wording",
        http_transport=_transport,
        start_owned_server=False,
    )
    gen_rows = [
        json.loads(line)
        for line in (out_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert gen_rows[1]["retry_form"] == "multi_turn"
    assert "keep your earlier text" in gen_rows[1]["repair_line"].lower()
    assert gen_rows[3]["retry_form"] == "single_turn"
    assert "keep your earlier text" not in gen_rows[3]["repair_line"].lower()
    assert "c_fidelity_stake_edge" in gen_rows[3]["repair_line"]
    assert "the setup stake description" in gen_rows[3]["repair_line"]



def test_60_card_sx_attempt_0_carried_unmodified_and_sha_asserted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test 60 (Card sx Req 2):
    - In multi-turn retry, the assistant turn (`messages[1]['content']`) is byte-identical to the sealed
      attempt-0 response (`response_text` in `sealed_store/responses.sealed.jsonl`), and its sha256 matches
      `response_sha256`.
    - Code asserts the sha256 match before sending the multi-turn retry and raises `AssertionError` if tampered.
    """
    import hashlib
    from typing import Any
    import udlab_generator.render as ug_render
    from udlab_generator import (
        PINNED_WRITER_MANIFEST_DIGEST,
        PINNED_WRITER_MODEL_TAG,
    )

    w0 = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    passing_r0 = render_world_template(w0, w0.agents[0])
    passing_r1 = render_world_template(w0, w0.agents[1])
    orig_stake = w0.stake_edges[0].description
    bad_stake = "Gia is an absent party who depends on the location of the index reel for tomorrow's courier handoff."
    failing_r1 = passing_r1.replace(orig_stake, bad_stake)

    seen_payloads: list[dict[str, Any]] = []

    def _transport(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            assert payload is not None
            idx = len(seen_payloads)
            seen_payloads.append(payload)
            txt = passing_r0 if idx == 0 else (failing_r1 if idx == 1 else passing_r1)
            return {
                "message": {"role": "assistant", "content": txt},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3070,
                "eval_count": 814,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    out_dir = tmp_path / "sx_unmodified_check"
    build_and_probe_dataset(
        count=1,
        base_seed=20261030,
        output_dir=out_dir,
        writer_kind="ollama",
        max_retries=1,
        retry_form="multi_turn",
        grant_id="test-grant-sx-unmodified",
        http_transport=_transport,
        start_owned_server=False,
    )
    sealed_rows = [
        json.loads(line)
        for line in (out_dir / "sealed_store" / "responses.sealed.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sealed_a0 = sealed_rows[1]
    assert sealed_a0["attempt"] == 0 and sealed_a0["rendering_index"] == 1
    assistant_msg_content = seen_payloads[2]["messages"][1]["content"]
    assert assistant_msg_content.encode("utf-8") == str(sealed_a0["response_text"]).encode("utf-8")
    assert hashlib.sha256(assistant_msg_content.encode("utf-8")).hexdigest() == sealed_a0["response_sha256"]

    # Now verify the in-code assertion triggers if the sealed attempt-0 response_sha256 does not match attempt_0_content:
    orig_append = ug_render.OllamaWriter._append_generation_record

    def _tampered_append(
        self_writer: Any,
        rec: Any,
        *,
        response_text: str | None = None,
        thinking_text: str = "",
    ) -> None:
        orig_append(self_writer, rec, response_text=response_text, thinking_text=thinking_text)
        if rec.attempt == 0 and not rec.checks_passed and self_writer._sealed_responses:
            last_sr = self_writer._sealed_responses[-1]
            self_writer._sealed_responses[-1] = replace(last_sr, response_sha256="0" * 64)

    monkeypatch.setattr(ug_render.OllamaWriter, "_append_generation_record", _tampered_append)
    seen_payloads.clear()
    with pytest.raises(AssertionError, match="response_sha256"):
        build_and_probe_dataset(
            count=1,
            base_seed=20261030,
            output_dir=tmp_path / "sx_tampered_check",
            writer_kind="ollama",
            max_retries=1,
            retry_form="multi_turn",
            grant_id="test-grant-sx-tampered",
            http_transport=_transport,
            start_owned_server=False,
        )


def test_61_card_sx_echoed_text_stays_local_and_sealed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test 61 (Card sx Req 3 & 5):
    - Attempt-0 text (with a unique marker string in a non-failing sentence) and the multi-turn message list
      never reach `part_public/`, `generations.jsonl`, `writer.log`/stdout, `build_summary.json`,
      `checkpoint.json`, `verify_written_report.json`, or `--recheck` output when attempt 1 replaces that marker.
    - Greps every written non-sealed file in `out_dir` (and stdout/stderr of build, `--verify-written`, and `--recheck`)
      for the marker string placed in attempt 0.
    - Verifies `--recheck` re-judges the retry's own response text without needing the message list.
    """
    from typing import Any
    from udlab_generator import (
        PINNED_WRITER_MANIFEST_DIGEST,
        PINNED_WRITER_MODEL_TAG,
    )
    from udlab_generator.cli import recheck_saved_responses, verify_written_dataset

    w0 = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    passing_r0 = render_world_template(w0, w0.agents[0])
    passing_r1 = render_world_template(w0, w0.agents[1])

    # Place a unique marker in a passing setup sentence of attempt 0, while causing a separate sentence to fail:
    attempt0_marker = "ATTEMPT0_ECHO_SECRET_MARKER_SX_994411"
    orig_stake = w0.stake_edges[0].description
    bad_stake = "Gia is an absent party who depends on the location of the index reel for tomorrow's courier handoff."
    # Append the marker as a benign clause on the first setup sentence so the failing sentence is still bad_stake:
    failing_r1_with_marker = passing_r1.replace(
        f"Following {w0.agents[1]} through the",
        f"Following {w0.agents[1]} ({attempt0_marker}) through the",
    ).replace(orig_stake, bad_stake)
    assert attempt0_marker in failing_r1_with_marker
    assert attempt0_marker not in passing_r1

    seen_payloads: list[dict[str, Any]] = []

    def _transport(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            assert payload is not None
            idx = len(seen_payloads)
            seen_payloads.append(payload)
            txt = passing_r0 if idx == 0 else (failing_r1_with_marker if idx == 1 else passing_r1)
            return {
                "message": {"role": "assistant", "content": txt},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3070 if idx < 2 else 3950,
                "eval_count": 814,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    out_dir = tmp_path / "sx_grep_non_sealed"
    capsys.readouterr()
    build_and_probe_dataset(
        count=1,
        base_seed=20261030,
        output_dir=out_dir,
        writer_kind="ollama",
        max_retries=1,
        retry_form="multi_turn",
        grant_id="test-grant-sx-grep",
        http_transport=_transport,
        start_owned_server=False,
    )
    verify_written_dataset(out_dir)
    rc_recheck = cli_main(["--recheck", str(out_dir)])
    assert rc_recheck == 0
    stdio_captured = capsys.readouterr()
    assert attempt0_marker not in stdio_captured.out
    assert attempt0_marker not in stdio_captured.err

    recheck_dict = recheck_saved_responses(out_dir)
    assert recheck_dict["matches_recorded_verdicts"] is True
    assert recheck_dict["calls"][2]["checks_passed"] is True
    assert attempt0_marker not in json.dumps(recheck_dict)

    # Confirm marker WAS echoed in the local multi-turn call (messages[1]['content']) and saved in sealed_store:
    assert attempt0_marker in seen_payloads[2]["messages"][1]["content"]
    sealed_responses_text = (out_dir / "sealed_store" / "responses.sealed.jsonl").read_text(encoding="utf-8")
    assert attempt0_marker in sealed_responses_text

    # Grep EVERY written non-sealed file under out_dir for attempt0_marker:
    non_sealed_files = [
        p
        for p in out_dir.rglob("*")
        if p.is_file() and "sealed_store" not in p.parts and not p.name.endswith((".sealed.json", ".sealed.jsonl"))
    ]
    assert len(non_sealed_files) >= 6  # items.jsonl, part.json, generations.jsonl, build_summary.json, checkpoint.json, verify_written_report.json
    for nf in non_sealed_files:
        text_content = nf.read_text(encoding="utf-8")
        assert attempt0_marker not in text_content, f"Marker {attempt0_marker!r} leaked into non-sealed file {nf}"
        assert failing_r1_with_marker not in text_content, f"Attempt-0 text leaked into non-sealed file {nf}"

    # Also verify FORBIDDEN_PUBLIC_FIELDS in sealing.py explicitly rejects `attempt_0_content` and `attempt_messages`:
    import udlab_generator.sealing as ug_sealing
    assert "attempt_0_content" in ug_sealing.FORBIDDEN_PUBLIC_FIELDS
    assert "attempt_messages" in ug_sealing.FORBIDDEN_PUBLIC_FIELDS
    with pytest.raises(SealingError, match="Forbidden key/seed fields"):
        check_not_in_key_order([{"item_id": "item_0001", "attempt_0_content": attempt0_marker}], [{"pass_index": 1}], {})
    with pytest.raises(SealingError, match="Forbidden key/seed fields"):
        check_not_in_key_order([{"item_id": "item_0001", "attempt_messages": []}], [{"pass_index": 1}], {})


def test_62_card_sx_retry_row_canonical_prompt_sha_and_fields(tmp_path: Path) -> None:
    """Test 62 (Card sx Req 4):
    - The retry row in `generations.jsonl` records `retry_form`, `retry_fallback_reason`,
      the `prompt_sha256` of the exact `messages` list sent (hashed in canonical JSON form
      `json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(',', ':'))`),
      `repair_line`, and `repair_line_sha256`, for both `multi_turn` and `single_turn` fallback retries.
    """
    import hashlib
    from typing import Any
    from udlab_generator import (
        PINNED_WRITER_MANIFEST_DIGEST,
        PINNED_WRITER_MODEL_TAG,
    )

    w0 = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    passing_r0 = render_world_template(w0, w0.agents[0])
    passing_r1 = render_world_template(w0, w0.agents[1])
    orig_stake = w0.stake_edges[0].description
    bad_stake = "Gia is an absent party who depends on the location of the index reel for tomorrow's courier handoff."
    failing_r0 = passing_r0.replace(orig_stake, bad_stake)
    failing_r1 = passing_r1.replace(orig_stake, bad_stake)

    seen_payloads: list[dict[str, Any]] = []

    def _transport(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            assert payload is not None
            idx = len(seen_payloads)
            seen_payloads.append(payload)
            # r0 (idx 0 -> idx 1): multi_turn retry (p0=3070, e0=814 -> need <= 9216)
            # r1 (idx 2 -> idx 3): single_turn fallback retry (p0=3070, e0=2000 -> need > 9216)
            if idx == 0:
                return {
                    "message": {"role": "assistant", "content": failing_r0},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 3070,
                    "eval_count": 814,
                }
            if idx == 1:
                return {
                    "message": {"role": "assistant", "content": passing_r0},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 3950,
                    "eval_count": 814,
                }
            if idx == 2:
                return {
                    "message": {"role": "assistant", "content": failing_r1},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 3070,
                    "eval_count": 2000,
                }
            return {
                "message": {"role": "assistant", "content": passing_r1},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3180,
                "eval_count": 814,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    out_dir = tmp_path / "sx_retry_row_fields"
    build_and_probe_dataset(
        count=1,
        base_seed=20261030,
        output_dir=out_dir,
        writer_kind="ollama",
        max_retries=1,
        retry_form="multi_turn",
        grant_id="test-grant-sx-row",
        http_transport=_transport,
        start_owned_server=False,
    )
    gen_rows = [
        json.loads(line)
        for line in (out_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(gen_rows) == 4

    # Row 1 (index 1): multi_turn retry on r0
    row_mt = gen_rows[1]
    msgs_mt = seen_payloads[1]["messages"]
    assert len(msgs_mt) == 3
    canon_mt = json.dumps(msgs_mt, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    assert row_mt["retry_form"] == "multi_turn"
    assert row_mt["prompt_sha256"] == hashlib.sha256(canon_mt.encode("utf-8")).hexdigest()
    assert row_mt["repair_line"] == msgs_mt[2]["content"]
    assert row_mt["repair_line_sha256"] == hashlib.sha256(msgs_mt[2]["content"].encode("utf-8")).hexdigest()

    # Row 3 (index 3): single_turn fallback retry on r1
    row_st = gen_rows[3]
    msgs_st = seen_payloads[3]["messages"]
    assert len(msgs_st) == 1
    canon_st = json.dumps(msgs_st, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    assert row_st["retry_form"] == "single_turn"
    assert row_st["retry_fallback_reason"].startswith("multiturn_need_exceeds_max_num_ctx:")
    assert row_mt["retry_fallback_reason"] == ""
    assert row_st["prompt_sha256"] == hashlib.sha256(canon_st.encode("utf-8")).hexdigest()
    assert row_st["repair_line"] != ""
    assert msgs_st[0]["content"].endswith(row_st["repair_line"])
    assert row_st["repair_line_sha256"] == hashlib.sha256(row_st["repair_line"].encode("utf-8")).hexdigest()


def test_67_card_tb_refusal_records_sealed_only_and_readers_resolve(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test 67 (Card tb Req 1, 2, 4):
    - Refusal records in `generations.jsonl` keep check_id, event_index and event_kind (where present),
      failing_sentence_sha256, failing_sentence_offset, and failing_sentence_length, with NO sentence text.
    - Full refusal message (with sentence) goes to `sealed_store/responses.sealed.jsonl`, keyed by `call_id`.
    - `--recheck` and `--verify-written` resolve the sentence from `sealed_store/responses.sealed.jsonl`
      at the recorded offset and verify `sha256(slice) == failing_sentence_sha256`.
    - `verify_written_report.json` carries no sentence text; console output may print it.
    - Legacy format (`generations.jsonl` with inline sentence and no offset/sha) is handled cleanly.
    """
    import hashlib
    from typing import Any
    from udlab_generator import (
        PINNED_WRITER_MANIFEST_DIGEST,
        PINNED_WRITER_MODEL_TAG,
    )
    from udlab_generator.cli import recheck_saved_responses, verify_written_dataset
    import udlab_generator.render as ug_render
    import udlab_generator.sealing as ug_sealing

    assert hasattr(ug_render, "locate_failing_sentence_in_response")
    assert hasattr(ug_render, "resolve_sealed_failing_sentence")
    assert "failing_sentence" in ug_sealing.FORBIDDEN_PUBLIC_FIELDS

    w0 = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    passing_r0 = render_world_template(w0, w0.agents[0])
    passing_r1 = render_world_template(w0, w0.agents[1])

    # r0 attempt 0 fails a setup check (c_fidelity_stake_edge, event_index=None, event_kind=None),
    # then r0 attempt 1 passes.
    sj_setup, sj_sents = _build_sj_writer_in_the_loop_scen_0000()
    orig_stake = w0.stake_edges[0].description
    bad_stake_sent = (
        "Gia is an absent party who depends on the location of the index reel "
        "for tomorrow's courier handoff (SETUP_FAIL_SENT_TB_67)."
    )
    failing_r0_setup = f"{sj_setup.replace('whereas Gia is absent off-site', 'whereas the courier is absent off-site').replace(orig_stake, bad_stake_sent)}\n\n{' '.join(sj_sents)}"

    # r1 attempt 0 and attempt 1 fail an event check (c_fidelity_missing_event, event_index=1, event_kind="see")
    bad_ev1_sent = "In the bindery room, Eva observed the index reel in open view (EVENT1_FAIL_SENT_TB_67)."
    failing_r1_ev1 = (
        f"{sj_setup.replace('Following Eva', f'Following {w0.agents[1]}')}\n\n"
        f"{' '.join([bad_ev1_sent, *sj_sents[1:]])}"
    )

    chat_responses = [failing_r0_setup, passing_r0, failing_r1_ev1, failing_r1_ev1]
    call_counter = 0

    def _transport(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        nonlocal call_counter
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            txt = chat_responses[call_counter]
            call_counter += 1
            return {
                "message": {"role": "assistant", "content": txt},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3050,
                "eval_count": 810,
                "logprobs": [{"token": "SECRET_LOGPROB_TOKEN_TB_67", "logprob": -0.05}],
            }
        raise AssertionError(f"Unexpected URL: {url}")

    out_dir = tmp_path / "tb_sealed_refusals"
    build_and_probe_dataset(
        count=1,
        base_seed=20261030,
        output_dir=out_dir,
        writer_kind="ollama",
        max_retries=1,
        grant_id="test-grant-tb-67",
        http_transport=_transport,
        start_owned_server=False,
    )

    gen_raw = (out_dir / "generations.jsonl").read_text(encoding="utf-8")
    assert bad_stake_sent not in gen_raw
    assert bad_ev1_sent not in gen_raw
    assert "SETUP_FAIL_SENT_TB_67" not in gen_raw
    assert "EVENT1_FAIL_SENT_TB_67" not in gen_raw
    assert "SECRET_LOGPROB_TOKEN_TB_67" not in gen_raw

    gen_rows = [json.loads(line) for line in gen_raw.splitlines() if line.strip()]
    sealed_rows = [
        json.loads(line)
        for line in (out_dir / "sealed_store" / "responses.sealed.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(gen_rows) == 4
    assert len(sealed_rows) == 4

    # Call 0 (r0 attempt 0): setup check failure (no event_index/event_kind)
    exp_stake_sha = hashlib.sha256(bad_stake_sent.encode("utf-8")).hexdigest()
    exp_stake_off = failing_r0_setup.index(bad_stake_sent)
    exp_stake_len = len(bad_stake_sent)
    assert gen_rows[0]["checks_passed"] is False
    assert gen_rows[0]["check_id"] == "c_fidelity_stake_edge"
    assert gen_rows[0]["event_index"] is None
    assert gen_rows[0]["event_kind"] is None
    assert gen_rows[0]["failing_sentence_sha256"] == exp_stake_sha
    assert gen_rows[0]["failing_sentence_offset"] == exp_stake_off
    assert gen_rows[0]["failing_sentence_length"] == exp_stake_len
    assert failing_r0_setup[exp_stake_off : exp_stake_off + exp_stake_len] == bad_stake_sent
    assert sealed_rows[0]["call_id"] == gen_rows[0]["call_id"]
    assert sealed_rows[0]["failing_sentence"] == bad_stake_sent
    assert any(bad_stake_sent in r for r in sealed_rows[0]["refusal_reasons"])

    # Call 2 (r1 attempt 0): event check failure (event_index=1, event_kind='see')
    exp_ev1_sha = hashlib.sha256(bad_ev1_sent.encode("utf-8")).hexdigest()
    exp_ev1_off = failing_r1_ev1.index(bad_ev1_sent)
    exp_ev1_len = len(bad_ev1_sent)
    assert gen_rows[2]["checks_passed"] is False
    assert gen_rows[2]["check_id"] == "c_fidelity_missing_event"
    assert gen_rows[2]["event_index"] == 1
    assert gen_rows[2]["event_kind"] == "see"
    assert gen_rows[2]["failing_sentence_sha256"] == exp_ev1_sha
    assert gen_rows[2]["failing_sentence_offset"] == exp_ev1_off
    assert gen_rows[2]["failing_sentence_length"] == exp_ev1_len
    assert failing_r1_ev1[exp_ev1_off : exp_ev1_off + exp_ev1_len] == bad_ev1_sent
    assert sealed_rows[2]["call_id"] == gen_rows[2]["call_id"]
    assert sealed_rows[2]["failing_sentence"] == bad_ev1_sent
    assert any(bad_ev1_sent in r for r in sealed_rows[2]["refusal_reasons"])

    # Verify --verify-written and --recheck resolve the sentence from sealed_store and check sha256:
    capsys.readouterr()
    rc_vw = cli_main(["--verify-written", str(out_dir)])
    vw_out = capsys.readouterr().out
    assert rc_vw == 1
    vw_report_text = (out_dir / "verify_written_report.json").read_text(encoding="utf-8")
    assert bad_stake_sent not in vw_report_text
    assert bad_ev1_sent not in vw_report_text
    assert "SETUP_FAIL_SENT_TB_67" not in vw_report_text
    assert "EVENT1_FAIL_SENT_TB_67" not in vw_report_text
    # Console output of --verify-written prints the resolved sentence:
    assert bad_stake_sent in vw_out
    assert bad_ev1_sent in vw_out

    capsys.readouterr()
    rc_rc = cli_main(["--recheck", str(out_dir)])
    rc_out = capsys.readouterr().out
    assert rc_rc == 0
    assert bad_stake_sent in rc_out
    assert bad_ev1_sent in rc_out

    recheck_res = recheck_saved_responses(out_dir)
    assert recheck_res["matches_recorded_verdicts"] is True
    assert recheck_res["calls"][0]["failing_sentence"] == bad_stake_sent
    assert recheck_res["calls"][0]["failing_sentence_sha256_verified"] is True
    assert recheck_res["calls"][2]["failing_sentence"] == bad_ev1_sent
    assert recheck_res["calls"][2]["failing_sentence_sha256_verified"] is True

    # Tamper with failing_sentence_sha256 in generations.jsonl -> both recheck and verify_written detect mismatch
    tampered_gen = list(gen_rows)
    tampered_gen[0] = {**tampered_gen[0], "failing_sentence_sha256": "0" * 64}
    (out_dir / "generations.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in tampered_gen),
        encoding="utf-8",
    )
    recheck_tampered = recheck_saved_responses(out_dir)
    assert recheck_tampered["matches_recorded_verdicts"] is False
    assert recheck_tampered["calls"][0]["sha256_verified"] is False
    assert recheck_tampered["calls"][0]["failing_sentence_sha256_verified"] is False
    vw_tampered = verify_written_dataset(out_dir)
    assert vw_tampered["passed"] is False
    assert any("failing_sentence_sha256" in err for err in vw_tampered["generations_errors"])


def test_68_card_tb_marker_tests_refused_retried_and_kept(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test 68 (Card tb Req 3a & 3b):
    - (a) A marker in a REFUSED attempt, and a marker in an attempt 0 that was later retried:
      zero hits anywhere in non-sealed files (`part_public/` included) and build stdout/stderr written to files.
    - (b) A marker in a KEPT rendering appears ONLY in `part_public/items.jsonl` and under `sealed_store/`.
    """
    from typing import Any
    from udlab_generator import (
        PINNED_WRITER_MANIFEST_DIGEST,
        PINNED_WRITER_MODEL_TAG,
    )

    configs_2 = make_balanced_configs(2, base_seed=20261030)
    w_s0, w_s1 = generate_world(configs_2[0]), generate_world(configs_2[1])

    s0_r0_base = render_world_template(w_s0, w_s0.agents[0])
    s0_r1_base = render_world_template(w_s0, w_s0.agents[1])
    s1_r0_base = render_world_template(w_s1, w_s1.agents[0])
    s1_r1_base = render_world_template(w_s1, w_s1.agents[1])

    # Markers for (a) attempt 0 that is later retried and fixed in attempt 1 (on scenario 0, r0):
    marker_attempt0_fail_sent = "MARKER_TB_ATTEMPT0_FAIL_SENT_91001"
    marker_attempt0_body = "MARKER_TB_ATTEMPT0_BODY_91002"
    s0_setup, s0_ev_part = s0_r0_base.split("\n\n", 1)
    s0_bad_ev1 = f"In the room, {w_s0.agents[0]} furtively paused ({marker_attempt0_fail_sent})."
    s0_r0_attempt0_fail = (
        s0_setup.replace(
            f"Following {w_s0.agents[0]} through the",
            f"Following {w_s0.agents[0]} ({marker_attempt0_body}) through the",
        )
        + f"\n\n{s0_bad_ev1} {s0_ev_part}"
    )

    # Marker for (b) KEPT rendering (scenario 0, r0 attempt 1 and r1 attempt 0 are both kept):
    marker_kept_rendering = "MARKER_TB_KEPT_RENDERING_92001"
    s0_r0_attempt1_kept = s0_r0_base.replace(
        f"Following {w_s0.agents[0]} through the",
        f"Following {w_s0.agents[0]} ({marker_kept_rendering}) through the",
    )
    s0_r1_kept = s0_r1_base

    # Markers for (a) REFUSED attempt (scenario 1, r0 fails both attempt 0 and attempt 1):
    marker_refused_fail_sent = "MARKER_TB_REFUSED_FAIL_SENT_93001"
    marker_refused_body = "MARKER_TB_REFUSED_BODY_93002"
    s1_setup, s1_ev_part = s1_r0_base.split("\n\n", 1)
    s1_bad_ev1 = f"In the room, {w_s1.agents[0]} secretly waited ({marker_refused_fail_sent})."
    s1_r0_refused = (
        s1_setup.replace(
            f"Following {w_s1.agents[0]} through the",
            f"Following {w_s1.agents[0]} ({marker_refused_body}) through the",
        )
        + f"\n\n{s1_bad_ev1} {s1_ev_part}"
    )

    # Sequence of calls across scenario 0 and scenario 1 (with max_retries=1):
    # 1. s0 r0 attempt 0 -> fails (s0_r0_attempt0_fail)
    # 2. s0 r0 attempt 1 -> passes (s0_r0_attempt1_kept)
    # 3. s0 r1 attempt 0 -> passes (s0_r1_kept)
    # 4. s1 r0 attempt 0 -> fails (s1_r0_refused)
    # 5. s1 r0 attempt 1 -> fails (s1_r0_refused)
    # 6. s1 r1 attempt 0 -> passes (s1_r1_base) [scenario 1 is refused overall because r0 failed]
    scripted_outputs = [
        s0_r0_attempt0_fail,
        s0_r0_attempt1_kept,
        s0_r1_kept,
        s1_r0_refused,
        s1_r0_refused,
        s1_r1_base,
    ]
    call_idx = 0

    def _fake_transport(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        nonlocal call_idx
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            txt = scripted_outputs[call_idx]
            call_idx += 1
            return {
                "message": {"role": "assistant", "content": txt},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3050,
                "eval_count": 800,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    out_dir = tmp_path / "tb_marker_out"
    import udlab_generator.cli as ug_cli
    orig_build = ug_cli.build_and_probe_dataset
    # Card ul: a cheap counter near the measured ratio (bytes/4), as in test 43; the estimate alone would refuse
    # a prompt that fits exactly.
    ug_cli.build_and_probe_dataset = lambda **kw: orig_build(
        **{
            **kw,
            "http_transport": _fake_transport,
            "start_owned_server": False,
            "prompt_token_counter": lambda text: len(text.encode("utf-8")) // 4,
        }
    )
    try:
        capsys.readouterr()
        rc_build = cli_main(
            [
                "--write",
                "--writer",
                "ollama",
                "--count",
                "2",
                "--seed",
                "20261030",
                "--max-retries",
                "1",
                "--grant-id",
                "test-grant-tb-68",
                "--out-dir",
                str(out_dir),
            ]
        )
        build_stdio = capsys.readouterr()
    finally:
        ug_cli.build_and_probe_dataset = orig_build
    assert rc_build == 0

    # Write captured build stdout/stderr to files in out_dir:
    (out_dir / "writer_stdout.log").write_text(build_stdio.out, encoding="utf-8")
    (out_dir / "writer_stderr.log").write_text(build_stdio.err, encoding="utf-8")

    # Run --verify-written and --recheck first:
    capsys.readouterr()
    cli_main(["--verify-written", str(out_dir)])
    verify_stdio = capsys.readouterr()

    capsys.readouterr()
    rc_recheck = cli_main(["--recheck", str(out_dir)])
    recheck_stdio = capsys.readouterr()
    assert rc_recheck == 0

    # Collect every file under out_dir:
    all_files = [p for p in out_dir.rglob("*") if p.is_file()]
    non_sealed_files = [p for p in all_files if "sealed_store" not in p.parts]
    sealed_files = [p for p in all_files if "sealed_store" in p.parts]
    assert len(non_sealed_files) >= 8

    # (a) Markers in REFUSED attempt and retried attempt 0: ZERO hits in any non-sealed file (part_public/ included):
    forbidden_markers = (
        marker_attempt0_fail_sent,
        marker_attempt0_body,
        marker_refused_fail_sent,
        marker_refused_body,
    )
    for nf in non_sealed_files:
        content = nf.read_text(encoding="utf-8")
        for m in forbidden_markers:
            assert m not in content, f"Forbidden marker {m!r} found in non-sealed file {nf}"

    # Body markers also have zero hits in verify/recheck captured stdout/stderr:
    for m_body in (marker_attempt0_body, marker_refused_body):
        assert m_body not in verify_stdio.out and m_body not in verify_stdio.err
        assert m_body not in recheck_stdio.out and m_body not in recheck_stdio.err

    # All 4 markers ARE preserved in sealed_store/responses.sealed.jsonl:
    sealed_resp_text = (out_dir / "sealed_store" / "responses.sealed.jsonl").read_text(encoding="utf-8")
    for m in forbidden_markers:
        assert m in sealed_resp_text

    # (b) Marker in KEPT rendering appears ONLY in part_public/items.jsonl and under sealed_store/:
    items_jsonl = out_dir / "part_public" / "items.jsonl"
    assert marker_kept_rendering in items_jsonl.read_text(encoding="utf-8")
    assert any(marker_kept_rendering in sf.read_text(encoding="utf-8") for sf in sealed_files)
    for nf in non_sealed_files:
        if nf.resolve() == items_jsonl.resolve():
            continue
        assert marker_kept_rendering not in nf.read_text(encoding="utf-8"), (
            f"Kept marker {marker_kept_rendering!r} leaked into non-items file {nf}"
        )
    assert marker_kept_rendering not in verify_stdio.out and marker_kept_rendering not in verify_stdio.err
    assert marker_kept_rendering not in recheck_stdio.out and marker_kept_rendering not in recheck_stdio.err


def test_69_card_tb_legacy_format_recheck_and_verify(tmp_path: Path) -> None:
    """Test 69 (Card tb Req 4):
    - Legacy run dirs (old `generations.jsonl` format with inline `(candidate sentence: ...)` in `refusal_reasons`
      and no `failing_sentence_offset`/`failing_sentence_length`/`failing_sentence_sha256`) still recheck cleanly
      without modifying any file in the dir, and are explicitly identified as `legacy_inline_v0`.
    - `verify_written_dataset` on a legacy `generations.jsonl` row strips the inline sentence so
      `verify_written_report.json` carries no sentence text.
    """
    import hashlib
    from typing import Any
    from udlab_generator import (
        PINNED_WRITER_MANIFEST_DIGEST,
        PINNED_WRITER_MODEL_TAG,
    )
    from udlab_generator.cli import recheck_saved_responses, verify_written_dataset

    w0 = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    passing_r0 = render_world_template(w0, w0.agents[0])
    sj_setup, sj_sents = _build_sj_writer_in_the_loop_scen_0000()
    legacy_secret_sent = "In the bindery room, Eva observed the index reel in open view (LEGACY_SENT_MARKER_TB_69)."
    failing_r1 = (
        f"{sj_setup.replace('Following Eva', f'Following {w0.agents[1]}')}\n\n"
        f"{' '.join([legacy_secret_sent, *sj_sents[1:]])}"
    )
    call_idx = 0

    def _transport(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        nonlocal call_idx
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            txt = passing_r0 if call_idx == 0 else failing_r1
            call_idx += 1
            return {
                "message": {"role": "assistant", "content": txt},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3000,
                "eval_count": 780,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    out_dir = tmp_path / "tb_legacy_dir"
    build_and_probe_dataset(
        count=1,
        base_seed=20261030,
        output_dir=out_dir,
        writer_kind="ollama",
        grant_id="test-grant-tb-69",
        http_transport=_transport,
        start_owned_server=False,
    )

    # Rewrite generations.jsonl into the old legacy format (inline sentence in refusal_reasons, no offset/sha keys):
    gen_path = out_dir / "generations.jsonl"
    gen_rows = [json.loads(line) for line in gen_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    legacy_row_1 = {
        k: v
        for k, v in gen_rows[1].items()
        if k
        not in (
            "check_id",
            "event_index",
            "event_kind",
            "failing_sentence_sha256",
            "failing_sentence_offset",
            "failing_sentence_length",
        )
    }
    legacy_row_1["refusal_reasons"] = [
        f"c_fidelity_missing_event: Render check (c) failed: missing or altered event #1 "
        f"(step=1, kind=see, actor=Eva, room=bindery room; candidate sentence: {legacy_secret_sent!r})."
    ]
    gen_path.write_text(
        json.dumps(gen_rows[0], sort_keys=True) + "\n" + json.dumps(legacy_row_1, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Snapshot every file digest in out_dir before recheck:
    before_digests = {
        p.relative_to(out_dir): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out_dir.rglob("*")
        if p.is_file()
    }
    recheck_res = recheck_saved_responses(out_dir)
    after_digests = {
        p.relative_to(out_dir): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out_dir.rglob("*")
        if p.is_file()
    }
    assert before_digests == after_digests, "--recheck must not rewrite or delete any file in legacy run dirs"
    assert recheck_res["matches_recorded_verdicts"] is True
    assert recheck_res["legacy_refusal_format_calls"] == 1
    assert recheck_res["calls"][1]["recorded_refusal_format"] == "legacy_inline_v0"
    assert recheck_res["calls"][1]["failing_sentence"] == legacy_secret_sent

    # Even if verify_written_dataset is run on a dir with a legacy generations.jsonl row,
    # verify_written_report.json must not copy the inline sentence:
    verify_written_dataset(out_dir)
    vw_report_raw = (out_dir / "verify_written_report.json").read_text(encoding="utf-8")
    assert legacy_secret_sent not in vw_report_raw
    assert "LEGACY_SENT_MARKER_TB_69" not in vw_report_raw


def test_70_card_td_req1_dest_before_source_and_made_way() -> None:
    """Test 70 (Card td Req 1):
    - Destination-before-source order ('<verb> into <to> from <from>') passes for every folded movement verb.
    - 'Dan headed into the sorting room from the bindery room.' passes as a folded leave+enter.
    - 'Dan walked into the sorting room from the bindery room.' and 'Dan stepped into the sorting room from the bindery room.' pass.
    - 'Dan made his way from the bindery room to the sorting room.' (and 'made her way' / 'made their way') passes.
    - '...stopping in the sorting room doorway' and '...pausing at the sorting room threshold' do NOT count as arriving.
    """
    from udlab_generator.render import (
        _FOLDED_LEAVE_ENTER_VERBS,
        _sentence_matches_event,
        _sentence_matches_folded_leave_enter,
    )

    w = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    fa = w.agents[1]  # Dan
    sj_setup, sj_sents = _build_sj_writer_in_the_loop_scen_0000()
    setup_dan = sj_setup.replace("Following Eva", "Following Dan")
    ev_2 = w.events[1]  # leave by Dan: bindery room -> sorting room
    ev_3 = w.events[2]  # enter by Dan: bindery room -> sorting room

    skip_indices = {2, 4, 8, 12, 14, 20}
    folded_by_leave_idx = {
        1: "Dan walked from the bindery room to the sorting room.",
        3: "Cara walked from the bindery room to the sorting room.",
        7: "Eva walked from the bindery room to the sorting room.",
        11: "Cara walked from the sorting room to the bindery room.",
        13: "Cara walked from the bindery room to the reading room.",
        19: "Eva walked from the sorting room to the bindery room.",
    }
    base_folded_sents = [
        folded_by_leave_idx.get(idx, sent)
        for idx, sent in enumerate(sj_sents)
        if idx not in skip_indices
    ]

    # Positive: "Dan headed into the sorting room from the bindery room.", "Dan made his way from the bindery room to the sorting room.",
    # and "Dan left the bindery room and entered the sorting room.":
    passing_folded = (
        "Dan headed into the sorting room from the bindery room.",
        "Dan walked into the sorting room from the bindery room.",
        "Dan stepped into the sorting room from the bindery room.",
        "Dan came into the sorting room from the bindery room.",
        "Dan made his way from the bindery room to the sorting room.",
        "Dan made her way from the bindery room to the sorting room.",
        "Dan made their way from the bindery room to the sorting room.",
        "Dan made his way into the sorting room from the bindery room.",
        "Dan left the bindery room and entered the sorting room.",
    )
    for s_pass in passing_folded:
        assert _sentence_matches_folded_leave_enter(s_pass, ev_2, ev_3, w.noise_threshold, world=w), (
            f"Expected folded sentence to pass: {s_pass!r}"
        )
        sents_try = list(base_folded_sents)
        sents_try[1] = s_pass
        assert check_rendered_prose(f"{setup_dan}\n\n{' '.join(sents_try)}", w, focal_agent=fa) == ()

    # Every verb in _FOLDED_LEAVE_ENTER_VERBS passes in destination-before-source order ("<verb> into <to> from <from>"):
    for verb in _FOLDED_LEAVE_ENTER_VERBS:
        if verb == "entered":
            cand = "Dan entered the sorting room from the bindery room."
        elif verb == "arrived":
            cand = "Dan arrived in the sorting room from the bindery room."
        else:
            cand = f"Dan {verb} into the sorting room from the bindery room."
        assert _sentence_matches_folded_leave_enter(cand, ev_2, ev_3, w.noise_threshold, world=w), (
            f"Expected folded verb {verb!r} in destination-before-source order to pass: {cand!r}"
        )

    # Negative: "...stopping in the sorting room doorway" / "...pausing at the sorting room threshold" must NOT count as arriving:
    refused_doorway = (
        "Dan exited from the bindery room, stopping in the sorting room doorway.",
        "Dan departed from the bindery room, pausing at the sorting room threshold.",
        "Dan walked from the bindery room, stopping in the sorting room doorway.",
    )
    for s_ref in refused_doorway:
        assert not _sentence_matches_folded_leave_enter(s_ref, ev_2, ev_3, w.noise_threshold, world=w), (
            f"Expected doorway/threshold stop not to count as folded arrival: {s_ref!r}"
        )
        assert not _sentence_matches_event(s_ref, ev_3, w.noise_threshold, world=w), (
            f"Expected doorway/threshold stop not to count as enter: {s_ref!r}"
        )
        sents_ref = list(base_folded_sents)
        sents_ref[1] = s_ref
        with pytest.raises(RenderCheckError) as exc_dw:
            check_rendered_prose(f"{setup_dan}\n\n{' '.join(sents_ref)}", w, focal_agent=fa)
        assert exc_dw.value.check_id == "c_fidelity_missing_event"


def test_71_card_td_req2_two_sentence_leave_plus_enter_and_double_count() -> None:
    """Test 71 (Card td Req 2):
    - A leave sentence followed by an enter sentence that names the from-room (as the prompt asks) passes, counted once each:
      - 'Dan walked from the bindery room to the sorting room.' followed by 'Dan entered the sorting room from the bindery room.' passes.
      - 'Dan departed from the bindery room to the sorting room.' followed by 'Dan entered the sorting room from the bindery room.' passes.
      - Also with 'Dan arrived in the sorting room from the bindery room.' passes.
    - Two consecutive sentences that BOTH assert the full trip still fail as a double count (including 'out of' and 'exited' variants).
    """
    w = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    fa = w.agents[1]  # Dan
    sj_setup, sj_sents = _build_sj_writer_in_the_loop_scen_0000()
    setup_dan = sj_setup.replace("Following Eva", "Following Dan")

    # Positive: two-sentence leave + enter where enter names the from-room (obeying the prompt rule):
    passing_pairs = (
        (
            "Dan walked from the bindery room to the sorting room.",
            "Dan entered the sorting room from the bindery room.",
        ),
        (
            "Dan departed from the bindery room to the sorting room.",
            "Dan entered the sorting room from the bindery room.",
        ),
        (
            "Dan left from the bindery room to the sorting room.",
            "Dan arrived in the sorting room from the bindery room.",
        ),
        (
            "Dan exited the bindery room toward the sorting room.",
            "Dan entered the sorting room from the bindery room.",
        ),
    )
    for s_leave, s_enter in passing_pairs:
        sents_try = list(sj_sents)
        sents_try[1] = s_leave
        sents_try[2] = s_enter
        assert check_rendered_prose(f"{setup_dan}\n\n{' '.join(sents_try)}", w, focal_agent=fa) == (), (
            f"Expected two-sentence leave + enter pair to pass: {s_leave!r} / {s_enter!r}"
        )

    # Negative: two consecutive sentences that BOTH assert the full trip fail as a double count (c_fidelity_unlogged_event):
    refused_double_trips = (
        (
            "Dan walked from the bindery room to the sorting room.",
            "Dan walked from the bindery room to the sorting room.",
        ),
        (
            "Dan walked from the bindery room to the sorting room.",
            "Dan moved from the bindery room to the sorting room.",
        ),
        (
            "Dan walked out of the bindery room into the sorting room.",
            "Dan walked from the bindery room into the sorting room.",
        ),
        (
            "Dan exited the bindery room into the sorting room.",
            "Dan walked from the bindery room into the sorting room.",
        ),
        (
            "Dan departed the bindery room into the sorting room.",
            "Dan stepped into the sorting room from the bindery room.",
        ),
        (
            "Dan entered the sorting room from the bindery room.",
            "Dan entered the sorting room from the bindery room.",
        ),
    )
    for s_trip1, s_trip2 in refused_double_trips:
        sents_double = list(sj_sents)
        sents_double[1] = s_trip1
        sents_double[2] = s_trip2
        with pytest.raises(RenderCheckError) as exc_dbl:
            check_rendered_prose(f"{setup_dan}\n\n{' '.join(sents_double)}", w, focal_agent=fa)
        assert exc_dbl.value.check_id == "c_fidelity_unlogged_event", (
            f"Expected double full-trip pair to fail with c_fidelity_unlogged_event: {s_trip1!r} / {s_trip2!r}"
        )
        assert exc_dbl.value.failing_sentence == s_trip2


def test_72_card_td_req3_post_argument_clauses() -> None:
    """Test 72 (Card td Req 3):
    - A `without <gerund>` clause is refused only when the gerund belongs to the event's own kind.
    - A negated relative clause is refused when it negates being at the location ('was not there / in / inside', 'never reached')
      or the actor doing the event.
    - Passive or other-agent clauses ('had not been moved / removed / touched', 'nobody had moved it') pass.
    """
    from udlab_generator.render import _sentence_matches_event

    w = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    fa = w.agents[0]  # Eva
    sj_setup, sj_sents = _build_sj_writer_in_the_loop_scen_0000()
    ev_1 = w.events[0]  # see by Eva: index reel in linen bin in bindery room
    ev_7 = w.events[6]  # move by Cara: brass ledger from intake bench to iron safe in sorting room

    # Positive (true paraphrases with passive / other-agent / non-kind gerunds):
    passing_see = (
        "In the bindery room, Eva observed the index reel in the linen bin, which had not been removed from the linen bin.",
        "In the bindery room, Eva observed the index reel in the linen bin, which had not been moved.",
        "In the bindery room, Eva observed the index reel in the linen bin, which had not been touched.",
        "In the bindery room, Eva observed the index reel in the linen bin, which nobody had moved from the linen bin.",
        "In the bindery room, Eva observed the index reel in the linen bin, which nobody had moved.",
        "In the bindery room, Eva observed the index reel in the linen bin, where nobody else had looked.",
        "In the bindery room, Eva observed the index reel in the linen bin without removing the index reel.",
        "In the bindery room, Eva observed the index reel in the linen bin without disturbing the linen bin.",
        "In the bindery room, Eva observed the index reel in the linen bin without touching the index reel.",
    )
    for s_pass in passing_see:
        assert _sentence_matches_event(s_pass, ev_1, w.noise_threshold, world=w), (
            f"Expected benign post-argument clause to pass: {s_pass!r}"
        )
        assert check_rendered_prose(f"{sj_setup}\n\n{' '.join([s_pass, *sj_sents[1:]])}", w, focal_agent=fa) == ()

    # Negative (negates being at the location or the actor doing the event):
    refused_see = (
        "In the bindery room, Eva observed the index reel in the linen bin, which was not there.",
        "In the bindery room, Eva observed the index reel in the linen bin, which was not inside it.",
        "In the bindery room, Eva observed the index reel in the linen bin, which was not in the linen bin.",
        "In the bindery room, Eva observed the index reel in the linen bin without laying eyes on it.",
        "In the bindery room, Eva observed the index reel in the linen bin without seeing it.",
        "In the bindery room, Eva observed the index reel in the linen bin, where Eva did not see the index reel.",
    )
    for s_ref in refused_see:
        assert not _sentence_matches_event(s_ref, ev_1, w.noise_threshold, world=w), (
            f"Expected contradictory post-argument clause on see to be refused: {s_ref!r}"
        )
        with pytest.raises(RenderCheckError) as exc_see:
            check_rendered_prose(f"{sj_setup}\n\n{' '.join([s_ref, *sj_sents[1:]])}", w, focal_agent=fa)
        assert exc_see.value.check_id == "c_fidelity_missing_event"

    refused_move = (
        "In the sorting room, Cara moved the brass ledger from the intake bench to the iron safe, which never reached it.",
        "In the sorting room, Cara moved the brass ledger from the intake bench to the iron safe, which was not there.",
        "In the sorting room, Cara moved the brass ledger from the intake bench to the iron safe, which was not inside it.",
        "In the sorting room, Cara moved the brass ledger from the intake bench to the iron safe without moving it.",
    )
    for s_ref_m in refused_move:
        assert not _sentence_matches_event(s_ref_m, ev_7, w.noise_threshold, world=w), (
            f"Expected contradictory post-argument clause on move to be refused: {s_ref_m!r}"
        )


def test_73_card_td_req4_fronted_phrases_no_and_doorway() -> None:
    """Test 73 (Card td Req 4):
    - Benign fronted time and manner phrases pass: 'Not long after arriving in...', 'Not long after that,', 'Without a word,'.
    - 'Not long after failing,' stays refused.
    - 'no trace / sign of the <target>' and 'in no <location>' are refused.
    - 'at the <room> doorway / threshold' doesn't count as being in the room.
    """
    from udlab_generator.render import _sentence_matches_event

    w = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    fa = w.agents[0]  # Eva
    sj_setup, sj_sents = _build_sj_writer_in_the_loop_scen_0000()
    ev_1 = w.events[0]  # see by Eva: index reel in linen bin in bindery room

    # Positive: benign fronted time and manner phrases pass:
    passing_fronted = (
        "Not long after arriving in the bindery room, Eva observed the index reel in the linen bin.",
        "Not long after that, in the bindery room, Eva observed the index reel in the linen bin.",
        "Without a word, in the bindery room, Eva observed the index reel in the linen bin.",
        "Without a sound, in the bindery room, Eva observed the index reel in the linen bin.",
    )
    for s_pass in passing_fronted:
        assert _sentence_matches_event(s_pass, ev_1, w.noise_threshold, world=w), (
            f"Expected benign fronted time/manner phrase to pass: {s_pass!r}"
        )
        assert check_rendered_prose(f"{sj_setup}\n\n{' '.join([s_pass, *sj_sents[1:]])}", w, focal_agent=fa) == ()

    # Negative: "Not long after failing," and "no trace/sign of" and "in no <location>" and "at the <room> doorway/threshold":
    refused_cases = (
        "Not long after failing, in the bindery room, Eva observed the index reel in the linen bin.",
        "In the bindery room, Eva observed no trace of the index reel in the linen bin.",
        "In the bindery room, Eva observed no sign of the index reel in the linen bin.",
        "In the bindery room, Eva observed the index reel in no linen bin.",
        "Not far from the bindery room, at the bindery room doorway, Eva observed the index reel in the linen bin.",
        "At the bindery room doorway, Eva observed the index reel in the linen bin.",
        "At the bindery room threshold, Eva observed the index reel in the linen bin.",
    )
    for s_ref in refused_cases:
        assert not _sentence_matches_event(s_ref, ev_1, w.noise_threshold, world=w), (
            f"Expected sentence to be refused: {s_ref!r}"
        )
        with pytest.raises(RenderCheckError) as exc_ref:
            check_rendered_prose(f"{sj_setup}\n\n{' '.join([s_ref, *sj_sents[1:]])}", w, focal_agent=fa)
        assert exc_ref.value.check_id == "c_fidelity_missing_event"


def test_74_card_td_req5_uncaught_checkpoint_and_sealing_exceptions_carry_no_sentence_text(
    tmp_path: Path,
) -> None:
    """Test 74 (Card td Req 5):
    - For an uncaught `RenderCheckError` or `SealingError` on a kept rendering or a checkpoint path,
      the exception / traceback carries NO candidate sentence text, using the same sha / offset / length form as tb.
    """
    import hashlib
    import traceback
    from udlab_generator.render import (
        RenderedScenario,
        WriterCheckpointManager,
    )
    from udlab_generator.sealing import verify_written_dataset_recounts_and_prose

    w0 = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    passing_r0 = render_world_template(w0, w0.agents[0])
    passing_r1 = render_world_template(w0, w0.agents[1])
    sj_setup, sj_sents = _build_sj_writer_in_the_loop_scen_0000()
    secret_sent = "In the bindery room, Eva observed the index reel in open view (SECRET_STDERR_LEAK_TD_74)."
    bad_prose = f"{sj_setup}\n\n{' '.join([secret_sent, *sj_sents[1:]])}"
    exp_sha = hashlib.sha256(secret_sent.encode("utf-8")).hexdigest()
    exp_off = bad_prose.index(secret_sent)
    exp_len = len(secret_sent)

    keys = compute_scenario_keys(w0)
    prompt_0 = format_writer_prompt(w0, w0.agents[0])
    prompt_1 = format_writer_prompt(w0, w0.agents[1])

    # 1. WriterCheckpointManager.append_scenario on a kept pair with bad prose:
    ckpt_dir = tmp_path / "td_ckpt_append"
    mgr = WriterCheckpointManager(
        output_dir=ckpt_dir,
        count=1,
        base_seed=20261030,
        private_seed_hex="ab" * 32,
        writer_kind="ollama",
    )
    mgr.initialize_fresh()
    bad_pair = (
        RenderedScenario(
            scenario_id=w0.scenario_id,
            rendering_index=0,
            focal_agent=w0.agents[0],
            prose=bad_prose,
            writer_prompt=prompt_0,
            keys=keys,
            world=w0,
        ),
        RenderedScenario(
            scenario_id=w0.scenario_id,
            rendering_index=1,
            focal_agent=w0.agents[1],
            prose=passing_r1,
            writer_prompt=prompt_1,
            keys=keys,
            world=w0,
        ),
    )
    with pytest.raises(RenderCheckError) as exc_append:
        mgr.append_scenario(scenario_index=0, pair=bad_pair, call_records=(), world=w0)
    tb_append = "".join(traceback.format_exception(exc_append.type, exc_append.value, exc_append.tb))
    assert secret_sent not in str(exc_append.value)
    assert "SECRET_STDERR_LEAK_TD_74" not in tb_append
    assert f"failing_sentence_sha256={exp_sha}" in str(exc_append.value)
    assert f"failing_sentence_offset={exp_off}" in str(exc_append.value)
    assert f"failing_sentence_length={exp_len}" in str(exc_append.value)

    # 2. WriterCheckpointManager.load_and_verify on a tampered checkpoint with bad prose:
    good_pair = (
        RenderedScenario(
            scenario_id=w0.scenario_id,
            rendering_index=0,
            focal_agent=w0.agents[0],
            prose=passing_r0,
            writer_prompt=prompt_0,
            keys=keys,
            world=w0,
        ),
        RenderedScenario(
            scenario_id=w0.scenario_id,
            rendering_index=1,
            focal_agent=w0.agents[1],
            prose=passing_r1,
            writer_prompt=prompt_1,
            keys=keys,
            world=w0,
        ),
    )
    mgr.append_scenario(scenario_index=0, pair=good_pair, call_records=(), world=w0)
    # Tamper with rendering_0 prose and recompute hashes in checkpoint files so load_and_verify reaches check_rendered_prose:
    sealed_doc = json.loads(mgr.sealed_path.read_text(encoding="utf-8"))
    pub_doc = json.loads(mgr.public_path.read_text(encoding="utf-8"))
    sealed_doc["scenarios"][0]["rendering_0"]["prose"] = bad_prose
    s_rec = sealed_doc["scenarios"][0]
    s_payload = {k: v for k, v in s_rec.items() if k not in ("sealed_entry_sha256", "chain_hash")}
    s_sha = hashlib.sha256(
        json.dumps(s_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    p_rec = pub_doc["scenarios"][0]
    p_payload = {k: v for k, v in p_rec.items() if k != "chain_hash"}
    c_hash = hashlib.sha256(
        f"{s_rec['prev_hash']}:{json.dumps(p_payload, sort_keys=True, ensure_ascii=False, separators=(',', ':'))}:{s_sha}".encode(
            "utf-8"
        )
    ).hexdigest()
    s_rec["sealed_entry_sha256"] = s_sha
    s_rec["chain_hash"] = c_hash
    p_rec["chain_hash"] = c_hash
    sealed_doc["chain_head_sha256"] = c_hash
    pub_doc["chain_head_sha256"] = c_hash
    mgr.sealed_path.write_text(json.dumps(sealed_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mgr.public_path.write_text(json.dumps(pub_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(RenderCheckError) as exc_load:
        mgr.load_and_verify(worlds=[w0])
    tb_load = "".join(traceback.format_exception(exc_load.type, exc_load.value, exc_load.tb))
    assert secret_sent not in str(exc_load.value)
    assert "SECRET_STDERR_LEAK_TD_74" not in tb_load
    assert f"failing_sentence_sha256={exp_sha}" in str(exc_load.value)
    assert f"failing_sentence_offset={exp_off}" in str(exc_load.value)
    assert f"failing_sentence_length={exp_len}" in str(exc_load.value)

    # 3. verify_written_dataset_recounts_and_prose raising SealingError on a kept rendering:
    out_dir = tmp_path / "td_sealing_err"
    build_and_probe_dataset(
        count=1,
        base_seed=20261030,
        output_dir=out_dir,
        writer_kind="mock",
    )
    # Replace prose in both keys.sealed.jsonl and items.jsonl (and update digests/hmac) so verify_sealed_part passes
    # and verify_written_dataset_recounts_and_prose reaches check_rendered_prose:
    from udlab_generator.sealing import _hmac_hex, _render_public_from_sealed

    sealed_dir = out_dir / "sealed_store"
    public_dir = out_dir / "part_public"
    sealing_meta_path = sealed_dir / "sealing.sealed.json"
    keys_sealed_path = sealed_dir / "keys.sealed.jsonl"
    sealing_meta = json.loads(sealing_meta_path.read_text(encoding="utf-8"))
    sealed_rows = [json.loads(line) for line in keys_sealed_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    sealed_rows[0]["prose"] = bad_prose
    keys_bytes = ("\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(",", ":")) for r in sealed_rows) + "\n").encode("utf-8")
    keys_sha = hashlib.sha256(keys_bytes).hexdigest()
    sealing_meta["keys_sealed_sha256"] = keys_sha
    sealing_meta["commitment_hmac_sha256"] = _hmac_hex(sealing_meta["private_seed_hex"], f"commitment:{keys_sha}", 64)
    items_bytes, part_bytes, _, _ = _render_public_from_sealed(sealed_rows, sealing_meta)
    keys_sealed_path.write_bytes(keys_bytes)
    sealing_meta_path.write_text(json.dumps(sealing_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (public_dir / "items.jsonl").write_bytes(items_bytes)
    (public_dir / "part.json").write_bytes(part_bytes)

    with pytest.raises(SealingError) as exc_seal:
        verify_written_dataset_recounts_and_prose(public_dir, sealed_dir)
    tb_seal = "".join(traceback.format_exception(exc_seal.type, exc_seal.value, exc_seal.tb))
    assert secret_sent not in str(exc_seal.value)
    assert "SECRET_STDERR_LEAK_TD_74" not in tb_seal
    assert f"failing_sentence_sha256={exp_sha}" in str(exc_seal.value)
    assert f"failing_sentence_offset={exp_off}" in str(exc_seal.value)
    assert f"failing_sentence_length={exp_len}" in str(exc_seal.value)

    # 4. OllamaWriter.render_scenario_pair (default allow_refusal=False) raising RenderCheckError:
    from udlab_generator import PINNED_WRITER_MANIFEST_DIGEST, PINNED_WRITER_MODEL_TAG
    from udlab_generator.render import OllamaWriter

    def _bad_writer_transport(method: str, url: str, payload: dict | None = None) -> dict:
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            return {
                "message": {"role": "assistant", "content": bad_prose},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3050,
                "eval_count": 800,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    writer = OllamaWriter(
        max_retries=0,
        grant_id="test-grant-td-74",
        http_transport=_bad_writer_transport,
        start_owned_server=False,
    )
    with pytest.raises(RenderCheckError) as exc_writer:
        writer.render_scenario_pair(w0)
    tb_writer = "".join(traceback.format_exception(exc_writer.type, exc_writer.value, exc_writer.tb))
    assert secret_sent not in str(exc_writer.value)
    assert "SECRET_STDERR_LEAK_TD_74" not in tb_writer
    assert f"failing_sentence_sha256={exp_sha}" in str(exc_writer.value)
    assert f"failing_sentence_offset={exp_off}" in str(exc_writer.value)
    assert f"failing_sentence_length={exp_len}" in str(exc_writer.value)


def test_75_card_td_req6_crlf_normalization_in_writer_and_reader(tmp_path: Path) -> None:
    """Test 75 (Card td Req 6):
    - Line endings (`\\r\\n`) are normalized before paragraph splitting and offset slicing, consistently in writer and reader.
    - Tests a response containing `\\r\\n\\r\\n` between paragraphs and `\\r\\n` inside a sentence, both passing and failing.
    """
    import hashlib
    from typing import Any
    from udlab_generator import (
        PINNED_WRITER_MANIFEST_DIGEST,
        PINNED_WRITER_MODEL_TAG,
    )
    from udlab_generator.cli import recheck_saved_responses, verify_written_dataset
    from udlab_generator.render import (
        locate_failing_sentence_in_response,
        resolve_sealed_failing_sentence,
    )

    w0 = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    passing_r0 = render_world_template(w0, w0.agents[0])
    passing_r1 = render_world_template(w0, w0.agents[1])

    # 1. Direct check_rendered_prose with \r\n\r\n paragraph separator:
    passing_r0_crlf = passing_r0.replace("\n\n", "\r\n\r\n")
    assert check_rendered_prose(passing_r0_crlf, w0, focal_agent=w0.agents[0]) == ()

    # 2. Direct locate_failing_sentence_in_response and resolve_sealed_failing_sentence on CRLF response:
    sj_setup, sj_sents = _build_sj_writer_in_the_loop_scen_0000()
    bad_ev1_crlf = "In the bindery room,\r\nEva observed the index reel in open view (CRLF_FAIL_TD_75)."
    failing_r0_crlf = f"{sj_setup}\r\n\r\n{' '.join([bad_ev1_crlf, *sj_sents[1:]])}"
    with pytest.raises(RenderCheckError) as exc_crlf:
        check_rendered_prose(failing_r0_crlf, w0, focal_agent=w0.agents[0])
    assert exc_crlf.value.check_id == "c_fidelity_missing_event"
    f_sha, f_off, f_len = locate_failing_sentence_in_response(failing_r0_crlf, exc_crlf.value.failing_sentence)
    expected_norm_sent = bad_ev1_crlf.replace("\r\n", "\n")
    assert f_sha == hashlib.sha256(expected_norm_sent.encode("utf-8")).hexdigest()
    resolved_sent, ok = resolve_sealed_failing_sentence(
        failing_r0_crlf,
        failing_sentence_sha256=f_sha,
        failing_sentence_offset=f_off,
        failing_sentence_length=f_len,
    )
    assert ok is True
    assert resolved_sent == expected_norm_sent

    # 3. End-to-end writer -> generations.jsonl -> recheck / verify-written with \r\n responses from transport:
    scripted = [failing_r0_crlf, passing_r0_crlf, passing_r1.replace("\n\n", "\r\n\r\n")]
    idx = 0

    def _crlf_transport(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        nonlocal idx
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            out_txt = scripted[idx]
            idx += 1
            return {
                "message": {"role": "assistant", "content": out_txt},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3050,
                "eval_count": 800,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    out_dir = tmp_path / "td_crlf_e2e"
    build_and_probe_dataset(
        count=1,
        base_seed=20261030,
        output_dir=out_dir,
        writer_kind="ollama",
        max_retries=1,
        grant_id="test-grant-td-75",
        http_transport=_crlf_transport,
        start_owned_server=False,
    )
    recheck_res = recheck_saved_responses(out_dir)
    assert recheck_res["matches_recorded_verdicts"] is True
    assert recheck_res["calls"][0]["checks_passed"] is False
    assert recheck_res["calls"][0]["failing_sentence_sha256_verified"] is True
    assert recheck_res["calls"][0]["failing_sentence"] == expected_norm_sent
    assert recheck_res["calls"][1]["checks_passed"] is True
    assert recheck_res["calls"][2]["checks_passed"] is True
    vw_res = verify_written_dataset(out_dir)
    assert "CRLF_FAIL_TD_75" not in (out_dir / "verify_written_report.json").read_text(encoding="utf-8")
    assert vw_res["recount_verification"]["prose_checks_passed"] is True


def test_76_card_ti_toplevel_stderr_catch_on_failing_kept_and_resume_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test 76 (Card ti Req 1 & 2):
    - Through `main([...])`, run a failing kept path (a tampered kept rendering with `--verify-written`,
      and a `--resume` path) containing a marker.
    - Capture stdout and stderr.
    - Assert the exit code is non-zero, that `failing_sentence_sha256=` appears on stderr, and that
      the marker appears in neither stream (and no traceback is printed).
    - Also verify `RenderCheckError.__str__` and `MockWriter.render_scenario_pair` never include the
      failing sentence in `str(exc)` while keeping `exc.failing_sentence` available as an attribute.
    """
    import hashlib
    from udlab_generator.sealing import _hmac_hex, _render_public_from_sealed

    w0 = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    sj_setup, sj_sents = _build_sj_writer_in_the_loop_scen_0000()
    marker = "MARKER_TI_STDERR_CATCH_76001"
    secret_sent = f"In the bindery room, Eva observed the index reel in open view ({marker})."
    bad_prose = f"{sj_setup}\n\n{' '.join([secret_sent, *sj_sents[1:]])}"
    exp_sha = hashlib.sha256(secret_sent.encode("utf-8")).hexdigest()
    exp_off = bad_prose.index(secret_sent)
    exp_len = len(secret_sent)

    # 1. Build a 1-scenario kept dataset via CLI:
    out_dir = tmp_path / "ti_kept_tampered"
    capsys.readouterr()
    rc_init = cli_main(["--write", "--writer", "mock", "--count", "1", "--seed", "20261030", "--out-dir", str(out_dir)])
    capsys.readouterr()
    assert rc_init == 0

    # Tamper with the kept rendering in sealed_store/keys.sealed.jsonl and part_public/items.jsonl
    # (updating SHA-256 and HMAC so verify_sealed_part passes and verify_written_dataset_recounts_and_prose
    # hits the prose check on the kept rendering and raises SealingError):
    sealed_dir = out_dir / "sealed_store"
    public_dir = out_dir / "part_public"
    sealing_meta_path = sealed_dir / "sealing.sealed.json"
    keys_sealed_path = sealed_dir / "keys.sealed.jsonl"
    sealing_meta = json.loads(sealing_meta_path.read_text(encoding="utf-8"))
    sealed_rows = [
        json.loads(line)
        for line in keys_sealed_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sealed_rows[0]["prose"] = bad_prose
    keys_bytes = (
        "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(",", ":")) for r in sealed_rows) + "\n"
    ).encode("utf-8")
    keys_sha = hashlib.sha256(keys_bytes).hexdigest()
    sealing_meta["keys_sealed_sha256"] = keys_sha
    sealing_meta["commitment_hmac_sha256"] = _hmac_hex(
        sealing_meta["private_seed_hex"], f"commitment:{keys_sha}", 64
    )
    items_bytes, part_bytes, _, _ = _render_public_from_sealed(sealed_rows, sealing_meta)
    keys_sealed_path.write_bytes(keys_bytes)
    sealing_meta_path.write_text(json.dumps(sealing_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (public_dir / "items.jsonl").write_bytes(items_bytes)
    (public_dir / "part.json").write_bytes(part_bytes)

    # Run `main(["--verify-written", ...])` and capture stdout and stderr:
    capsys.readouterr()
    rc_vw = cli_main(["--verify-written", str(out_dir)])
    vw_stdio = capsys.readouterr()
    assert rc_vw != 0
    assert f"failing_sentence_sha256={exp_sha}" in vw_stdio.err
    assert f"failing_sentence_offset={exp_off}" in vw_stdio.err
    assert f"failing_sentence_length={exp_len}" in vw_stdio.err
    assert "c_fidelity_missing_event" in vw_stdio.err
    assert "Traceback" not in vw_stdio.err
    assert marker not in vw_stdio.out
    assert marker not in vw_stdio.err
    assert secret_sent not in vw_stdio.out
    assert secret_sent not in vw_stdio.err

    # 2. Tamper with the kept rendering in the checkpoint files (`checkpoint.sealed.json` and `checkpoint.json`)
    # and run `main(["--write", "--writer", "mock", "--count", "1", "--seed", "20261030", "--out-dir", ..., "--resume"])`:
    sealed_ckpt_path = sealed_dir / "checkpoint.sealed.json"
    pub_ckpt_path = out_dir / "checkpoint.json"
    sealed_ckpt = json.loads(sealed_ckpt_path.read_text(encoding="utf-8"))
    pub_ckpt = json.loads(pub_ckpt_path.read_text(encoding="utf-8"))
    sealed_ckpt["scenarios"][0]["rendering_0"]["prose"] = bad_prose
    s_rec = sealed_ckpt["scenarios"][0]
    s_payload = {k: v for k, v in s_rec.items() if k not in ("sealed_entry_sha256", "chain_hash")}
    s_sha = hashlib.sha256(
        json.dumps(s_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    p_rec = pub_ckpt["scenarios"][0]
    p_payload = {k: v for k, v in p_rec.items() if k != "chain_hash"}
    c_hash = hashlib.sha256(
        f"{s_rec['prev_hash']}:{json.dumps(p_payload, sort_keys=True, ensure_ascii=False, separators=(',', ':'))}:{s_sha}".encode(
            "utf-8"
        )
    ).hexdigest()
    s_rec["sealed_entry_sha256"] = s_sha
    s_rec["chain_hash"] = c_hash
    p_rec["chain_hash"] = c_hash
    sealed_ckpt["chain_head_sha256"] = c_hash
    pub_ckpt["chain_head_sha256"] = c_hash
    sealed_ckpt_path.write_text(json.dumps(sealed_ckpt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pub_ckpt_path.write_text(json.dumps(pub_ckpt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    capsys.readouterr()
    rc_resume = cli_main(
        [
            "--write",
            "--writer",
            "mock",
            "--count",
            "1",
            "--seed",
            "20261030",
            "--out-dir",
            str(out_dir),
            "--resume",
        ]
    )
    resume_stdio = capsys.readouterr()
    assert rc_resume != 0
    assert f"failing_sentence_sha256={exp_sha}" in resume_stdio.err
    assert f"failing_sentence_offset={exp_off}" in resume_stdio.err
    assert f"failing_sentence_length={exp_len}" in resume_stdio.err
    assert "c_fidelity_missing_event" in resume_stdio.err
    assert "Traceback" not in resume_stdio.err
    assert marker not in resume_stdio.out
    assert marker not in resume_stdio.err
    assert secret_sent not in resume_stdio.out
    assert secret_sent not in resume_stdio.err

    # 3. Verify RenderCheckError.__str__ and MockWriter.render_scenario_pair never include the failing sentence
    # in str(exc), while keeping failing_sentence as an attribute:
    bad_mock = MockWriter(custom_renderer=lambda _w, _fa: bad_prose)
    with pytest.raises(RenderCheckError) as exc_mock:
        bad_mock.render_scenario_pair(w0)
    assert exc_mock.value.failing_sentence == secret_sent
    assert marker not in str(exc_mock.value)
    assert secret_sent not in str(exc_mock.value)
    assert f"failing_sentence_sha256={exp_sha}" in str(exc_mock.value)
    assert f"failing_sentence_offset={exp_off}" in str(exc_mock.value)
    assert f"failing_sentence_length={exp_len}" in str(exc_mock.value)


def test_77_card_tk_stake_check_by_meaning_and_prompt_alignment() -> None:
    """Test 77 (Card tk Req 1):
    - Stake check by meaning is generated from one source (the stake's agent and its dependency_phrase).
    - Each logged stake passes when setup prose names the absent agent as subject with a 'depends on' relation,
      and the dependency phrase appears verbatim, ignoring final punctuation and case.
    - Every absent party listed in the log must have its stake stated.
    - Renders the rule into the prompt's constraints block from the same source, keeping the prompt-checker alignment test passing.
    - Accepted:
      1. 'X is off-site and depends on <phrase>.'
      2. 'X is an absent party who depends on <phrase>.'
      3. List form 'Stakes are: X is off-site and depends on <phrase>; Y is off-site and depends on <phrase2>.'
      4. Same sentence with different final punctuation (';', ',', '.', none) and case variations.
    - Refused:
      1. A stake naming the wrong absent party.
      2. A stake with the wrong dependency object (e.g. 'depends on the location of the index reel' when the stake is the brass ledger).
      3. A dependency phrase altered by one word.
      4. A list form where one absent party's stake is missing.
      5. A text that stakes only one of two absent parties.
    """
    import udlab_generator.render as ug_render
    from udlab_generator.render import verify_prompt_and_checker_requirements_aligned

    verify_prompt_and_checker_requirements_aligned()

    # Single-absent-party scenario (scen_0000: Gia is absent, depends on the location of the brass ledger for tomorrow's courier handoff)
    cfg_1 = make_balanced_configs(1, base_seed=20261030)[0]
    w1 = generate_world(cfg_1)
    fa1 = w1.agents[0]
    stake_1 = w1.stake_edges[0]
    assert stake_1.agent == "Gia"
    dep_phrase_1 = getattr(
        stake_1,
        "dependency_phrase",
        "the location of the brass ledger for tomorrow's courier handoff",
    )
    assert dep_phrase_1 == "the location of the brass ledger for tomorrow's courier handoff"

    sj_setup_1, sj_sents_1 = _build_sj_writer_in_the_loop_scen_0000()
    events_1 = " ".join(sj_sents_1)
    orig_stake_sent_1 = stake_1.description

    # Accepted 1, 2, 4: single-party forms with "is off-site and", "is an absent party who", case variation, and final punctuation (; , . none)
    accepted_single_forms = (
        f"Gia is off-site and depends on {dep_phrase_1}.",
        f"Gia is an absent party who depends on {dep_phrase_1}.",
        f"Gia is an absent party who DEPENDS ON {dep_phrase_1.upper()}.",
        f"Gia is off-site and depends on {dep_phrase_1};",
        f"Gia is off-site and depends on {dep_phrase_1},",
        f"Gia is off-site and depends on {dep_phrase_1}",
        f"Gia is an absent party who depends on {dep_phrase_1};",
        f"Gia is an absent party who depends on {dep_phrase_1},",
        f"Gia is an absent party who depends on {dep_phrase_1}",
    )
    for acc_sent in accepted_single_forms:
        setup_try = sj_setup_1.replace(orig_stake_sent_1, acc_sent)
        assert check_rendered_prose(f"{setup_try}\n\n{events_1}", w1, focal_agent=fa1) == (), (
            f"Expected stake sentence to pass: {acc_sent!r}"
        )

    # Prompt constraints block renders the stake rule from the same source:
    assert hasattr(stake_1, "dependency_phrase")
    assert hasattr(ug_render, "format_stake_rule_summary")
    assert hasattr(ug_render, "STAKE_RELATION_PHRASE")
    prompt_1 = format_writer_prompt(w1, fa1)
    assert ug_render.format_stake_rule_summary() in prompt_1
    assert ug_render.STAKE_RELATION_PHRASE in ug_render.format_stake_rule_summary()

    # Refused 1: Stake naming the wrong absent party
    wrong_party_setup = sj_setup_1.replace(
        orig_stake_sent_1,
        f"Hana is off-site and depends on {stake_1.dependency_phrase}.",
    )
    with pytest.raises(RenderCheckError) as exc_wp:
        check_rendered_prose(f"{wrong_party_setup}\n\n{events_1}", w1, focal_agent=fa1)
    assert exc_wp.value.check_id == "c_fidelity_stake_edge"

    # Refused 2: Stake with the wrong dependency object ("depends on the location of the index reel" when the stake is the brass ledger)
    wrong_obj_setup = sj_setup_1.replace(
        orig_stake_sent_1,
        "Gia is off-site and depends on the location of the index reel for tomorrow's courier handoff.",
    )
    with pytest.raises(RenderCheckError) as exc_wo:
        check_rendered_prose(f"{wrong_obj_setup}\n\n{events_1}", w1, focal_agent=fa1)
    assert exc_wo.value.check_id == "c_fidelity_stake_edge"

    # Refused 3: Dependency phrase altered by one word ("dispatch" instead of "courier")
    one_word_setup = sj_setup_1.replace(
        orig_stake_sent_1,
        "Gia is an absent party who depends on the location of the brass ledger for tomorrow's dispatch handoff.",
    )
    with pytest.raises(RenderCheckError) as exc_ow:
        check_rendered_prose(f"{one_word_setup}\n\n{events_1}", w1, focal_agent=fa1)
    assert exc_ow.value.check_id == "c_fidelity_stake_edge"

    # Two-absent-party scenario (num_absent=2):
    cfg_2 = replace(cfg_1, num_absent=2)
    w2 = generate_world(cfg_2)
    fa2 = w2.agents[0]
    assert len(w2.absent_parties) == 2
    assert len(w2.stake_edges) == 2
    s_x, s_y = w2.stake_edges[0], w2.stake_edges[1]
    base_prose_2 = render_world_template(w2, fa2)
    setup_2, events_2 = base_prose_2.split("\n\n", 1)
    orig_both_stakes = f"{s_x.description} {s_y.description}"
    assert orig_both_stakes in setup_2

    # Accepted 3 & 4: List form "Stakes are: X is off-site and depends on <phrase>; Y is off-site and depends on <phrase2>."
    # and variations with "is an absent party who" and different final punctuation (; , . none):
    accepted_two_party_forms = (
        f"Stakes are: {s_x.agent} is off-site and depends on {s_x.dependency_phrase}; {s_y.agent} is off-site and depends on {s_y.dependency_phrase}.",
        f"Stakes are: {s_x.agent} is an absent party who depends on {s_x.dependency_phrase}; {s_y.agent} is an absent party who depends on {s_y.dependency_phrase}.",
        f"Stakes are: {s_x.agent} is off-site and depends on {s_x.dependency_phrase}, {s_y.agent} is off-site and depends on {s_y.dependency_phrase};",
        f"Stakes are: {s_x.agent} is off-site and depends on {s_x.dependency_phrase}; {s_y.agent} is off-site and depends on {s_y.dependency_phrase}",
    )
    for acc_list in accepted_two_party_forms:
        setup_2_try = setup_2.replace(orig_both_stakes, acc_list)
        assert check_rendered_prose(f"{setup_2_try}\n\n{events_2}", w2, focal_agent=fa2) == (), (
            f"Expected two-party list stake sentence to pass: {acc_list!r}"
        )

    # Refused 4: List form where one absent party's stake is missing (Y is named as off-site without its dependency phrase)
    missing_in_list_setup = setup_2.replace(
        orig_both_stakes,
        f"Stakes are: {s_x.agent} is off-site and depends on {s_x.dependency_phrase}; {s_y.agent} is off-site.",
    )
    with pytest.raises(RenderCheckError) as exc_ml:
        check_rendered_prose(f"{missing_in_list_setup}\n\n{events_2}", w2, focal_agent=fa2)
    assert exc_ml.value.check_id == "c_fidelity_stake_edge"

    # Refused 5: A text that stakes only one of two absent parties
    only_one_staked_setup = setup_2.replace(orig_both_stakes, s_x.description)
    with pytest.raises(RenderCheckError) as exc_oo:
        check_rendered_prose(f"{only_one_staked_setup}\n\n{events_2}", w2, focal_agent=fa2)
    assert exc_oo.value.check_id == "c_fidelity_stake_edge"

    # Also verify that if world.absent_parties has 2 absent parties but world.stake_edges only lists 1,
    # check_rendered_prose refuses with c_fidelity_stake_edge because every absent party in the log must have its stake stated:
    w2_missing_edge = replace(w2, stake_edges=(s_x,))
    with pytest.raises(RenderCheckError) as exc_me:
        check_rendered_prose(base_prose_2, w2_missing_edge, focal_agent=fa2)
    assert exc_me.value.check_id == "c_fidelity_stake_edge"


def test_78_card_tk_duplicate_stakes_allowed_and_counted(tmp_path: Path) -> None:
    """Test 78 (Card tk Req 2):
    - Duplicate stake statements are allowed, not refused.
    - Counted in the build summary as `duplicate_stake_statements` for the dataset read,
      as well as in `verify_written_dataset` and `recheck_saved_responses`.
    """
    from typing import Any
    from udlab_generator import (
        PINNED_WRITER_MANIFEST_DIGEST,
        PINNED_WRITER_MODEL_TAG,
    )
    from udlab_generator.cli import recheck_saved_responses, verify_written_dataset
    import udlab_generator.render as ug_render

    w0 = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    stake_0 = w0.stake_edges[0]
    dep_0 = getattr(
        stake_0,
        "dependency_phrase",
        "the location of the brass ledger for tomorrow's courier handoff",
    )
    dup_sentence = f"{stake_0.agent} is an absent party who depends on {dep_0}."

    def _dup_renderer(w: World, focal_agent: str) -> str:
        base = render_world_template(w, focal_agent)
        if w.scenario_id == w0.scenario_id and focal_agent == w.agents[0]:
            setup_p, events_p = base.split("\n\n", 1)
            return f"{setup_p} {dup_sentence}\n\n{events_p}"
        return base

    prose_with_dup = _dup_renderer(w0, w0.agents[0])
    assert check_rendered_prose(prose_with_dup, w0, focal_agent=w0.agents[0]) == ()

    scripted_outputs = [prose_with_dup, render_world_template(w0, w0.agents[1])]
    call_idx = 0

    def _dup_transport(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        nonlocal call_idx
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            txt = scripted_outputs[call_idx]
            call_idx += 1
            return {
                "message": {"role": "assistant", "content": txt},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3050,
                "eval_count": 800,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    out_dir = tmp_path / "tk_dup_stakes_build"
    summary = build_and_probe_dataset(
        count=1,
        base_seed=20261030,
        output_dir=out_dir,
        writer_kind="ollama",
        grant_id="test-grant-tk-78",
        http_transport=_dup_transport,
        start_owned_server=False,
    )
    assert summary.get("duplicate_stake_statements") == 1
    assert summary["recount_verification"].get("duplicate_stake_statements") == 1
    assert ug_render.count_duplicate_stake_statements(prose_with_dup, w0) == 1
    assert ug_render.count_duplicate_stake_statements(render_world_template(w0, w0.agents[1]), w0) == 0

    build_summary_disk = json.loads((out_dir / "build_summary.json").read_text(encoding="utf-8"))
    assert build_summary_disk["duplicate_stake_statements"] == 1
    assert build_summary_disk["recount_verification"]["duplicate_stake_statements"] == 1

    vw_report = verify_written_dataset(out_dir)
    assert vw_report["recount_verification"]["duplicate_stake_statements"] == 1

    recheck_report = recheck_saved_responses(out_dir)
    assert recheck_report["duplicate_stake_statements"] == 1
    assert recheck_report["calls"][0]["duplicate_stake_statements"] == 1
    assert recheck_report["calls"][1]["duplicate_stake_statements"] == 0


def test_79_card_tk_sealing_error_recount_mismatch_carries_sha256_never_values(tmp_path: Path) -> None:
    """Test 79 (Card tk Req 3):
    - In `sealing.py:1131-1136` and any similar site (`verify_sealed_part`), recount mismatch messages
      carry `sha256` of the sealed and recounted values, never the values.
    - Asserts that `SealingError` holds no key value (neither the tampered sealed value nor the true recounted key).
    """
    import hashlib
    from udlab_generator.sealing import (
        _hmac_hex,
        _render_public_from_sealed,
        verify_written_dataset_recounts_and_prose,
    )

    configs = make_balanced_configs(1, base_seed=20261030)
    w0 = generate_world(configs[0])
    writer = MockWriter()
    pairs = [writer.render_scenario_pair(w0)]

    pub_dir = tmp_path / "part_public"
    seal_dir = tmp_path / "sealed_store"
    priv_seed = "ab" * 32
    write_sealed_dataset(
        rendered_pairs=pairs,
        public_dir=pub_dir,
        sealed_dir=seal_dir,
        private_seed_hex=priv_seed,
    )

    sealing_meta_path = seal_dir / "sealing.sealed.json"
    keys_sealed_path = seal_dir / "keys.sealed.jsonl"
    orig_meta = json.loads(sealing_meta_path.read_text(encoding="utf-8"))
    orig_rows = [
        json.loads(line)
        for line in keys_sealed_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    def _write_tampered_rows(tampered_rows: list[dict[str, object]]) -> None:
        meta = dict(orig_meta)
        keys_bytes = (
            "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(",", ":")) for r in tampered_rows)
            + "\n"
        ).encode("utf-8")
        keys_sha = hashlib.sha256(keys_bytes).hexdigest()
        meta["keys_sealed_sha256"] = keys_sha
        meta["commitment_hmac_sha256"] = _hmac_hex(meta["private_seed_hex"], f"commitment:{keys_sha}", 64)
        items_bytes, part_bytes, _, _ = _render_public_from_sealed(tampered_rows, meta)
        keys_sealed_path.write_bytes(keys_bytes)
        sealing_meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (pub_dir / "items.jsonl").write_bytes(items_bytes)
        (pub_dir / "part.json").write_bytes(part_bytes)

    # 1. Tamper with B2 key -> triggers both verify_sealed_part and verify_written_dataset_recounts_and_prose (B2 site and general site):
    rows_b2 = json.loads(json.dumps(orig_rows))
    b2_q = next(q for q in rows_b2[0]["questions"] if q["question_type"] == "B2")
    true_b2_key = str(b2_q["key"])
    secret_tampered_b2 = "SECRET_TAMPERED_B2_KEY_VALUE_TK_79"
    b2_q["key"] = secret_tampered_b2
    _write_tampered_rows(rows_b2)

    exp_sealed_b2_sha = hashlib.sha256(secret_tampered_b2.encode("utf-8")).hexdigest()
    exp_true_b2_sha = hashlib.sha256(true_b2_key.encode("utf-8")).hexdigest()

    with pytest.raises(SealingError) as exc_vsp:
        verify_sealed_part(pub_dir, seal_dir)
    msg_vsp = str(exc_vsp.value)
    assert exp_sealed_b2_sha in msg_vsp
    assert exp_true_b2_sha in msg_vsp
    assert secret_tampered_b2 not in msg_vsp
    assert true_b2_key.lower() not in msg_vsp.lower()

    with pytest.raises(SealingError) as exc_vw_b2:
        verify_written_dataset_recounts_and_prose(pub_dir, seal_dir)
    msg_vw_b2 = str(exc_vw_b2.value)
    assert exp_sealed_b2_sha in msg_vw_b2
    assert exp_true_b2_sha in msg_vw_b2
    assert secret_tampered_b2 not in msg_vw_b2
    assert true_b2_key.lower() not in msg_vw_b2.lower()

    # 2. Tamper with B1 key -> triggers the general recount mismatch branch in verify_written_dataset_recounts_and_prose:
    rows_b1 = json.loads(json.dumps(orig_rows))
    b1_q = next(q for q in rows_b1[0]["questions"] if q["question_type"] == "B1")
    true_b1_key = str(b1_q["key"])
    secret_tampered_b1 = "SECRET_TAMPERED_B1_KEY_VALUE_TK_79"
    b1_q["key"] = secret_tampered_b1
    _write_tampered_rows(rows_b1)

    exp_sealed_b1_sha = hashlib.sha256(secret_tampered_b1.encode("utf-8")).hexdigest()
    exp_true_b1_sha = hashlib.sha256(true_b1_key.encode("utf-8")).hexdigest()

    with pytest.raises(SealingError) as exc_vw_b1:
        verify_written_dataset_recounts_and_prose(pub_dir, seal_dir)
    msg_vw_b1 = str(exc_vw_b1.value)
    assert exp_sealed_b1_sha in msg_vw_b1
    assert exp_true_b1_sha in msg_vw_b1
    assert secret_tampered_b1 not in msg_vw_b1
    assert true_b1_key.lower() not in msg_vw_b1.lower()


def _scan_files_for_repo_leaks(root_dir: Path, rel_paths: list[str]) -> list[str]:
    """Scan the given relative file paths under `root_dir` for:
    1. Absolute paths (Windows drive forms and user-home forms).
    2. Seed or salt values (literal hex seed/salt assignments or UDLAB_PRIVATE_SEED values).
    3. Any line matching a sealed-store response record shape.
    """
    import re

    win_drive_pat = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:(?:\\\\|/[A-Za-z0-9_.\-]+|\\[A-Za-z0-9_.\-]+)")
    user_home_pat = re.compile(
        r"(?:/Users/[A-Za-z0-9_.\-]+|/home/[A-Za-z0-9_.\-]+|[A-Za-z]:\\+Users\\+[A-Za-z0-9_.\-]+)",
        re.IGNORECASE,
    )
    seed_salt_pat = re.compile(
        r"(?:"
        r"(?:private_seed(?:_hex)?|shuffle_seed|shuffle_salt|qshuffle_salt)\s*(?:=|:)\s*[\"'][0-9a-fA-F]{16,}[\"'](?!\s*\*)"
        r"|UDLAB_PRIVATE_SEED\s*=\s*[\"']?[0-9a-fA-F]{16,}"
        r")"
    )
    violations: list[str] = []
    for rel in rel_paths:
        fpath = root_dir / rel
        if not fpath.is_file():
            continue
        try:
            text = fpath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if win_drive_pat.search(line):
                violations.append(f"{rel}:{lineno}: windows_drive_path")
            if user_home_pat.search(line):
                violations.append(f"{rel}:{lineno}: user_home_path")
            if seed_salt_pat.search(line):
                violations.append(f"{rel}:{lineno}: seed_or_salt_value")
            stripped = line.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    obj = json.loads(stripped)
                except Exception:
                    obj = None
                if isinstance(obj, dict) and "response_text" in obj and (
                    "call_id" in obj or "response_sha256" in obj or "refusal_reasons" in obj
                ):
                    violations.append(f"{rel}:{lineno}: sealed_store_response_record")
    return violations


def test_80_card_tk_leak_guard_git_ls_files_and_planted_fixtures(tmp_path: Path) -> None:
    """Test 80 (Card tk Req 4):
    - Scans `git ls-files` in this repository.
    - Fails on absolute paths (Windows drive and user-home forms), on seed or salt values,
      and on any line matching a sealed-store response record shape.
    - Includes a fixture showing it fails on a planted bad file in a temp dir for each category.
    """
    import subprocess
    import udlab_generator.render as ug_render

    assert hasattr(ug_render, "count_duplicate_stake_statements"), (
        "Repo leak guard test requires Card tk stake-by-meaning & leak-guard implementation."
    )

    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked_files = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    assert len(tracked_files) >= 10
    repo_violations = _scan_files_for_repo_leaks(repo_root, tracked_files)
    assert repo_violations == [], f"Leak guard found violations in git ls-files: {repo_violations}"

    # Verify the leak guard fails on planted bad files in a temp dir:
    # (a) Planted Windows drive path (constructed without a literal drive path in this source file):
    planted_win = tmp_path / "planted_win.txt"
    planted_win.write_text("Config loaded from " + "E" + ":/" + "AI/rnd-local/secret\n", encoding="utf-8")
    v_win = _scan_files_for_repo_leaks(tmp_path, ["planted_win.txt"])
    assert any("windows_drive_path" in v for v in v_win)

    # (b) Planted user-home path:
    planted_home = tmp_path / "planted_home.txt"
    planted_home.write_text("Log path: /" + "Users" + "/alice/secret.log\n", encoding="utf-8")
    v_home = _scan_files_for_repo_leaks(tmp_path, ["planted_home.txt"])
    assert any("user_home_path" in v for v in v_home)

    # (c) Planted seed or salt value:
    planted_seed = tmp_path / "planted_seed.py"
    planted_seed.write_text(
        "private_seed" + '_hex = "' + ("deadbeef" * 8) + '"\n',
        encoding="utf-8",
    )
    v_seed = _scan_files_for_repo_leaks(tmp_path, ["planted_seed.py"])
    assert any("seed_or_salt_value" in v for v in v_seed)

    # (d) Planted sealed-store response record line:
    planted_resp = tmp_path / "planted_resp.jsonl"
    planted_resp.write_text(
        json.dumps(
            {
                "call_id": "scen_0000:r0:call_0001",
                "response_sha256": "00" * 32,
                "response_text": "Synthetic response text.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    v_resp = _scan_files_for_repo_leaks(tmp_path, ["planted_resp.jsonl"])
    assert any("sealed_store_response_record" in v for v in v_resp)


def test_81_card_tn_omission_by_key_optional_vs_required_and_prompt_aligned() -> None:
    """Test 81 (Card tn Item 3):
    - A setup check whose item no key, lure, or validator depends on (occluders, containers, fixtures,
      station names in setup) stops refusing on omission.
    - Omissions that a key, lure, or validator depends on (rooms, doorways, initial rooms, goals,
      absent parties, stakes, belief rule, hearing threshold) stay refused.
    - The prompt's constraints block, generated from the same single source (`SETUP_OMISSION_POLICY` /
      `WRITER_PROSE_REQUIREMENTS`), states which setup items are required and which are optional,
      keeping `verify_prompt_and_checker_requirements_aligned` passing.
    """
    import udlab_generator.render as ug_render
    from udlab_generator.world import BELIEF_PERSISTENCE_RULE

    assert hasattr(ug_render, "SETUP_OMISSION_POLICY")
    assert hasattr(ug_render, "REQUIRED_SETUP_CHECK_IDS")
    assert hasattr(ug_render, "OPTIONAL_SETUP_CHECK_IDS")
    assert ug_render.OPTIONAL_SETUP_CHECK_IDS == {
        "c_fidelity_containers",
        "c_fidelity_fixtures",
        "c_fidelity_occluder",
        "c_fidelity_stations",
    }
    assert ug_render.REQUIRED_SETUP_CHECK_IDS == {
        "c_fidelity_focal_agent",
        "c_fidelity_room_setup",
        "c_fidelity_doorways",
        "c_fidelity_initial_room",
        "c_fidelity_goal",
        "c_fidelity_absent_party",
        "c_fidelity_stake_edge",
        "d_rule_sentence",
        "d_hearing_threshold",
    }
    assert ug_render.verify_prompt_and_checker_requirements_aligned() == ug_render.WRITER_PROSE_REQUIREMENTS

    w = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    fa = w.agents[0]
    prompt_str = format_writer_prompt(w, fa)
    assert "(required)" in prompt_str.lower() or "(required;" in prompt_str.lower()
    assert "(optional)" in prompt_str.lower() or "(optional;" in prompt_str.lower()
    for cid in ug_render.OPTIONAL_SETUP_CHECK_IDS:
        assert ug_render.WRITER_PROSE_REQUIREMENTS_BY_ID[cid].required_in_setup is False
    for cid in ug_render.REQUIRED_SETUP_CHECK_IDS:
        assert ug_render.WRITER_PROSE_REQUIREMENTS_BY_ID[cid].required_in_setup is True

    _, sj_sents = _build_sj_writer_in_the_loop_scen_0000()
    events_para = " ".join(sj_sents)

    # Setup that omits all optional items (no setup containers, no setup fixtures, no setup occluders, no setup station names)
    # while keeping every required setup item (focal_agent, rooms, doorways, starting rooms, goals, absent party, stake,
    # belief rule, hearing threshold) passes under Verdict A:
    minimal_required_setup = (
        "Following Eva through the archival registry, the vault room is on the north side, "
        "while the reading room and the sorting room are on the south side. "
        "Doorways connect the vault room to the reading room, the reading room to the bindery room, "
        "and the bindery room to the sorting room, with a doorway hearing threshold of 50 dB (>= 50 dB carries one doorway). "
        f"{BELIEF_PERSISTENCE_RULE} "
        "Eva starts in the bindery room. Dan starts in the bindery room. Cara starts in the bindery room, "
        "while Gia is absent off-site. "
        "Eva plans to complete the quarterly audit packet; Dan plans to verify the municipal deed transfer; "
        "Cara plans to catalog the incoming charter folio. "
        "Gia is off-site and depends on the location of the brass ledger for tomorrow's courier handoff."
    )
    assert check_rendered_prose(f"{minimal_required_setup}\n\n{events_para}", w, focal_agent=fa) == ()

    # Omitting ONLY the occluders in setup is also accepted:
    setup_no_occluders = (
        "Following Eva through the archival registry, the vault room holds an oak drawer and a steel cabinet with a wall barometer; "
        "the reading room holds a slate tray and a cedar shelf with a wall clock; "
        "the bindery room holds a linen bin and a copper rack with a humidity dial; "
        "the sorting room holds a maple cubby and an iron safe with a ventilation grille. "
        "Doorways connect the vault room to the reading room, the reading room to the bindery room, "
        "and the bindery room to the sorting room, with a doorway hearing threshold of 50 dB. "
        f"{BELIEF_PERSISTENCE_RULE} "
        "Eva starts in the bindery room, Dan starts in the bindery room, and Cara starts in the bindery room, "
        "while Gia is absent off-site. "
        "Eva plans to complete the quarterly audit packet; Dan plans to verify the municipal deed transfer; "
        "Cara plans to catalog the incoming charter folio. "
        "Gia is off-site and depends on the location of the brass ledger for tomorrow's courier handoff."
    )
    assert check_rendered_prose(f"{setup_no_occluders}\n\n{events_para}", w, focal_agent=fa) == ()

    # Conversely, omitting any REQUIRED setup item stays refused with its check_id:
    required_omission_cases = (
        (
            "c_fidelity_room_setup",
            minimal_required_setup.replace("the vault room is on the north side, while ", "").replace(
                "Doorways connect the vault room to the reading room, ",
                "Doorways connect ",
            ),
        ),
        (
            "c_fidelity_doorways",
            minimal_required_setup.replace(
                "Doorways connect the vault room to the reading room, the reading room to the bindery room, "
                "and the bindery room to the sorting room",
                "Doorways connect the vault room to the reading room. A doorway connects the bindery room to the sorting room",
            ),
        ),
        (
            "c_fidelity_initial_room",
            minimal_required_setup.replace("Dan starts in the bindery room. ", ""),
        ),
        (
            "c_fidelity_goal",
            minimal_required_setup.replace("Dan plans to verify the municipal deed transfer; ", ""),
        ),
        (
            "c_fidelity_absent_party",
            minimal_required_setup.replace(", while Gia is absent off-site", "").replace(
                "Gia is off-site and depends on the location of the brass ledger for tomorrow's courier handoff.",
                "",
            ),
        ),
        (
            "c_fidelity_stake_edge",
            minimal_required_setup.replace(
                "Gia is off-site and depends on the location of the brass ledger for tomorrow's courier handoff.",
                "",
            ),
        ),
        (
            "d_rule_sentence",
            minimal_required_setup.replace(f"{BELIEF_PERSISTENCE_RULE} ", ""),
        ),
        (
            "d_hearing_threshold",
            minimal_required_setup.replace(
                "Doorways connect the vault room to the reading room, the reading room to the bindery room, "
                "and the bindery room to the sorting room, with a doorway hearing threshold of 50 dB (>= 50 dB carries one doorway).",
                "The vault room links to the reading room, the reading room links to the bindery room, and the bindery room links to the sorting room.",
            ),
        ),
    )
    for expected_cid, bad_setup in required_omission_cases:
        with pytest.raises(RenderCheckError, match=".") as exc_req:
            check_rendered_prose(f"{bad_setup}\n\n{events_para}", w, focal_agent=fa)
        assert exc_req.value.check_id == expected_cid, f"Expected {expected_cid}, got {exc_req.value.check_id}"
        assert exc_req.value.failure_kind == "absent"


def test_82_card_tn_false_statements_about_optional_items_refused() -> None:
    """Test 82 (Card tn Item 4):
    - Omitting an optional setup item is accepted, but asserting something FALSE about it is refused AS FALSE,
      with its own check_id and `failure_kind == 'false'`, not as missing:
      1. `'In the <room> there is no occluder.'` / `'... and no named occluder.'` for a room that has one
         -> refused with `c_fidelity_false_statement` (`failure_kind == 'false'`).
      2. A wrong occluder name for a room (attributing another room's occluder to `<room>`)
         -> refused with `c_fidelity_occluder` (`failure_kind == 'false'`).
      3. A container in the wrong room (attributing another room's container to `<room>`)
         -> refused with `c_fidelity_containers` (`failure_kind == 'false'`).
      4. A fixture in the wrong room (attributing another room's fixture to `<room>`)
         -> refused with `c_fidelity_fixtures` (`failure_kind == 'false'`).
    """
    from udlab_generator.world import BELIEF_PERSISTENCE_RULE

    w = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    fa = w.agents[0]
    _, sj_sents = _build_sj_writer_in_the_loop_scen_0000()
    events_para = " ".join(sj_sents)

    # Baseline setup without optional occluders passes:
    base_no_occ_setup = (
        "Following Eva through the archival registry, the vault room, reading room, bindery room, and sorting room "
        "are linked by doorways connecting the vault room to the reading room, the reading room to the bindery room, "
        "and the bindery room to the sorting room (doorway hearing threshold: 50 dB). "
        f"{BELIEF_PERSISTENCE_RULE} "
        "Eva starts in the bindery room, Dan starts in the bindery room, and Cara starts in the bindery room, "
        "with Gia absent off-site. "
        "Eva plans to complete the quarterly audit packet; Dan plans to verify the municipal deed transfer; "
        "Cara plans to catalog the incoming charter folio. "
        "Gia is off-site and depends on the location of the brass ledger for tomorrow's courier handoff."
    )
    assert check_rendered_prose(f"{base_no_occ_setup}\n\n{events_para}", w, focal_agent=fa) == ()

    # (a) Made-up wording required by Card tn Item 4: "In the <room> there is no occluder." for a room that has one is refused:
    for room in w.rooms:
        false_no_occ_sent = f"In the {room} there is no occluder."
        with pytest.raises(RenderCheckError) as exc_no_occ:
            check_rendered_prose(
                f"{base_no_occ_setup} {false_no_occ_sent}\n\n{events_para}",
                w,
                focal_agent=fa,
            )
        assert exc_no_occ.value.check_id == "c_fidelity_false_statement"
        assert exc_no_occ.value.failure_kind == "false"
        assert exc_no_occ.value.failing_sentence == false_no_occ_sent

    # Also test "no named occluder" and "no containers" / "no fixture":
    false_named_occ_sent = "In the vault room are containers oak drawer and steel cabinet with fixture wall barometer and no named occluder."
    with pytest.raises(RenderCheckError) as exc_named_occ:
        check_rendered_prose(
            f"{base_no_occ_setup} {false_named_occ_sent}\n\n{events_para}",
            w,
            focal_agent=fa,
        )
    assert exc_named_occ.value.check_id == "c_fidelity_false_statement"
    assert exc_named_occ.value.failure_kind == "false"
    assert exc_named_occ.value.failing_sentence == false_named_occ_sent

    false_no_cont_sent = "In the vault room there are no containers."
    with pytest.raises(RenderCheckError) as exc_no_cont:
        check_rendered_prose(
            f"{base_no_occ_setup} {false_no_cont_sent}\n\n{events_para}",
            w,
            focal_agent=fa,
        )
    assert exc_no_cont.value.check_id == "c_fidelity_false_statement"
    assert exc_no_cont.value.failure_kind == "false"

    # (b) Wrong occluder name for a room (vault room has frosted partition, not canvas screen):
    wrong_occ_sent = "In the vault room, a canvas screen stands between the far-side alcove and the container bay."
    with pytest.raises(RenderCheckError) as exc_wrong_occ:
        check_rendered_prose(
            f"{base_no_occ_setup} {wrong_occ_sent}\n\n{events_para}",
            w,
            focal_agent=fa,
        )
    assert exc_wrong_occ.value.check_id == "c_fidelity_occluder"
    assert exc_wrong_occ.value.failure_kind == "false"
    assert exc_wrong_occ.value.failing_sentence == wrong_occ_sent

    # (c) Container in the wrong room (linen bin belongs to bindery room, not vault room):
    wrong_cont_sent = "In the vault room are the oak drawer and the linen bin."
    with pytest.raises(RenderCheckError) as exc_wrong_cont:
        check_rendered_prose(
            f"{base_no_occ_setup} {wrong_cont_sent}\n\n{events_para}",
            w,
            focal_agent=fa,
        )
    assert exc_wrong_cont.value.check_id == "c_fidelity_containers"
    assert exc_wrong_cont.value.failure_kind == "false"
    assert exc_wrong_cont.value.failing_sentence == wrong_cont_sent

    # (d) Fixture in the wrong room (wall clock belongs to reading room, not sorting room):
    wrong_fix_sent = "In the sorting room is the wall clock fixture."
    with pytest.raises(RenderCheckError) as exc_wrong_fix:
        check_rendered_prose(
            f"{base_no_occ_setup} {wrong_fix_sent}\n\n{events_para}",
            w,
            focal_agent=fa,
        )
    assert exc_wrong_fix.value.check_id == "c_fidelity_fixtures"
    assert exc_wrong_fix.value.failure_kind == "false"
    assert exc_wrong_fix.value.failing_sentence == wrong_fix_sent


def test_83_card_tn_repair_wording_absent_altered_false_forms(tmp_path: Path) -> None:
    """Test 83 (Card tn Item 5 - Q2):
    - For an `absent` required item, `format_repair_retry_line` reads:
      `'Keep your earlier text as it was; add one sentence naming <item> ...'`.
    - For an `altered` or `false` item, `format_repair_retry_line` reads:
      `'Keep your earlier text exactly as it was; change only ...'`.
    - Generated from `check_id` plus `absent`/`altered`/`false` flag (`failure_kind`),
      recorded with its sha256 on the retry row as now, and carries no key content.
    - Tests all three forms (`absent`, `altered`, `false`).
    """
    import hashlib
    from typing import Any
    from udlab_generator import (
        PINNED_WRITER_MANIFEST_DIGEST,
        PINNED_WRITER_MODEL_TAG,
    )
    from udlab_generator.render import (
        VALID_FAILURE_KINDS,
        WRITER_PROSE_REQUIREMENTS,
        format_repair_retry_line,
    )

    assert VALID_FAILURE_KINDS == frozenset({"absent", "altered", "false"})

    w0 = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    skeys = compute_scenario_keys(w0)
    forbidden_key_strings = {
        q.key.lower()
        for q in skeys.questions
        if isinstance(q.key, str) and q.key.lower() not in ("yes", "no", "none")
    }

    # 1. Direct unit test of all three forms (absent, altered, false) across setup and event checks:
    for req in WRITER_PROSE_REQUIREMENTS:
        for f_kind in ("absent", "altered", "false"):
            exc_setup = RenderCheckError(
                "synthetic failure",
                check_id=req.check_id,
                failing_sentence="synthetic secret sentence",
                failure_kind=f_kind,
            )
            line_setup_st = format_repair_retry_line(exc_setup)
            line_setup = format_repair_retry_line(exc_setup, retry_form="multi_turn")
            assert line_setup_st.startswith(f"Repair note (rule broken: check {req.check_id}, {f_kind}): ")
            assert "earlier text" not in line_setup_st.lower()
            if f_kind == "absent":
                assert line_setup.startswith("Keep your earlier text as it was; add one sentence naming ")
            else:
                assert line_setup.startswith("Keep your earlier text exactly as it was; change only ")
                if f_kind == "false":
                    assert "false statement" in line_setup
            assert req.check_id in line_setup and req.check_id in line_setup_st
            assert "synthetic secret sentence" not in line_setup and "synthetic secret sentence" not in line_setup_st
            for k_str in forbidden_key_strings:
                assert k_str not in line_setup.lower() and k_str not in line_setup_st.lower()

    for f_kind in ("absent", "altered", "false"):
        exc_ev = RenderCheckError(
            "synthetic event failure",
            check_id="c_fidelity_missing_event",
            failing_sentence="synthetic secret event sentence",
            event_index=5,
            event_kind="see",
            failure_kind=f_kind,
        )
        line_ev_st = format_repair_retry_line(exc_ev)
        line_ev = format_repair_retry_line(exc_ev, retry_form="multi_turn")
        assert line_ev_st.startswith(
            f"Repair note (rule broken: event 5 (see, check c_fidelity_missing_event), {f_kind}): "
        )
        assert "earlier text" not in line_ev_st.lower()
        if f_kind == "absent":
            assert line_ev.startswith(
                "Keep your earlier text as it was; add one sentence for event 5 immediately after the sentence for event 4 "
            )
        else:
            assert line_ev.startswith("Keep your earlier text exactly as it was; change only event 5 ")
        assert "synthetic secret event sentence" not in line_ev and "synthetic secret event sentence" not in line_ev_st
        for k_str in forbidden_key_strings:
            assert k_str not in line_ev.lower() and k_str not in line_ev_st.lower()

    # 2. End-to-end test through `check_rendered_prose` and `build_and_probe_dataset` for all 3 forms:
    passing_r0 = render_world_template(w0, w0.agents[0])
    passing_r1 = render_world_template(w0, w0.agents[1])
    orig_stake = w0.stake_edges[0].description

    # Form 1 (absent): stake sentence omitted completely
    absent_r0 = passing_r0.replace(f"Stakes: {orig_stake}", "")
    with pytest.raises(RenderCheckError) as exc_abs:
        check_rendered_prose(absent_r0, w0, focal_agent=w0.agents[0])
    assert exc_abs.value.check_id == "c_fidelity_stake_edge"
    assert exc_abs.value.failure_kind == "absent"
    assert format_repair_retry_line(exc_abs.value).startswith(
        "Repair note (rule broken: check c_fidelity_stake_edge, absent): include one sentence naming the setup stake description"
    )
    assert format_repair_retry_line(exc_abs.value, retry_form="multi_turn").startswith(
        "Keep your earlier text as it was; add one sentence naming the setup stake description"
    )

    # Form 2 (altered): stake sentence present with altered dependency phrase
    altered_r1 = passing_r1.replace(
        orig_stake,
        f"{w0.stake_edges[0].agent} is off-site and depends on the location of the index reel for tomorrow's handoff.",
    )
    with pytest.raises(RenderCheckError) as exc_alt:
        check_rendered_prose(altered_r1, w0, focal_agent=w0.agents[1])
    assert exc_alt.value.check_id == "c_fidelity_stake_edge"
    assert exc_alt.value.failure_kind == "altered"
    assert format_repair_retry_line(exc_alt.value).startswith(
        "Repair note (rule broken: check c_fidelity_stake_edge, altered): ensure the setup stake description"
    )
    assert format_repair_retry_line(exc_alt.value, retry_form="multi_turn").startswith(
        "Keep your earlier text exactly as it was; change only the setup stake description"
    )

    # Form 3 (false): setup asserts a false statement about an optional item ("In the vault room there is no occluder.")
    s0_setup, s0_events = passing_r0.split("\n\n", 1)
    false_r0 = f"{s0_setup} In the {w0.rooms[0]} there is no occluder.\n\n{s0_events}"
    with pytest.raises(RenderCheckError) as exc_false:
        check_rendered_prose(false_r0, w0, focal_agent=w0.agents[0])
    assert exc_false.value.check_id == "c_fidelity_false_statement"
    assert exc_false.value.failure_kind == "false"
    assert format_repair_retry_line(exc_false.value).startswith(
        "Repair note (rule broken: check c_fidelity_false_statement, false): remove any false setup statement"
    )
    assert format_repair_retry_line(exc_false.value, retry_form="multi_turn").startswith(
        "Keep your earlier text exactly as it was; change only the false setup statement"
    )

    # Verify end-to-end recording of repair_line and repair_line_sha256 in generations.jsonl:
    scripted = [absent_r0, passing_r0, false_r0.replace(f"Following {w0.agents[0]}", f"Following {w0.agents[1]}"), passing_r1]
    call_idx = 0

    def _transport(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        nonlocal call_idx
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            txt = scripted[call_idx]
            call_idx += 1
            return {
                "message": {"role": "assistant", "content": txt},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3050,
                "eval_count": 800,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    out_dir = tmp_path / "tn_repair_three_forms"
    build_and_probe_dataset(
        count=1,
        base_seed=20261030,
        output_dir=out_dir,
        writer_kind="ollama",
        max_retries=1,
        grant_id="test-grant-tn-83",
        http_transport=_transport,
        start_owned_server=False,
    )
    gen_rows = [
        json.loads(line)
        for line in (out_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(gen_rows) == 4
    # Retry on r0 (absent stake):
    assert gen_rows[1]["repair_line"].startswith(
        "Repair note (rule broken: check c_fidelity_stake_edge, absent): include one sentence naming the setup stake description"
    )
    assert gen_rows[1]["repair_line_sha256"] == hashlib.sha256(gen_rows[1]["repair_line"].encode("utf-8")).hexdigest()
    # Retry on r1 (false occluder statement):
    assert gen_rows[3]["repair_line"].startswith(
        "Repair note (rule broken: check c_fidelity_false_statement, false): remove any false setup statement"
    )
    assert gen_rows[3]["repair_line_sha256"] == hashlib.sha256(gen_rows[3]["repair_line"].encode("utf-8")).hexdigest()


def test_84_card_tr_step28_declared_locations_and_event_location_validator(tmp_path: Path) -> None:
    """Test 84 (Card tr Reqs 3 & 4):
    - The planted note_item move (and see/say) uses declared world locations in that room (containers_by_room[ev.room]),
      never undeclared 'dispatch folder' or 'countertop in the <room>'.
    - No location name embeds a room name anywhere in the generator.
    - validate_event_locations (called at generation and at --verify-written / verify_sealed_part) verifies that
      every location an event uses is a declared container, fixture, station, or room, and rejects any undeclared
      location ('dispatch folder', 'countertop in the <room>') or embedded room name.
    """
    import udlab_generator as ug
    from udlab_generator.world import CONTAINER_BAY_STATION, DOMAINS, FAR_SIDE_ALCOVE_STATION

    validate_ev_loc_fn = getattr(ug, "validate_event_locations", None)
    configs = make_balanced_configs(300, base_seed=20261030)
    valid_stations = {CONTAINER_BAY_STATION, FAR_SIDE_ALCOVE_STATION, "intake bench", "latch position A", "latch position B"}

    # First check a scenario where the planted note_item move lands at step 28 (scen_0024), then all 300 scenarios:
    ordered_configs = sorted(configs, key=lambda c: (c.scenario_id != "scen_0024", c.u_is_who, c.scenario_id))
    for cfg in ordered_configs:
        w = generate_world(cfg)
        all_containers = {c for pair in w.containers_by_room.values() for c in pair}
        all_fixtures = set(w.fixtures_by_room.values())
        all_rooms = set(w.rooms)
        declared_locs = all_containers | all_fixtures | valid_stations | all_rooms
        dom = next(d for d in DOMAINS if d.name == w.domain_name)

        for ev in w.events:
            for loc in (ev.room, ev.from_location, ev.to_location):
                if not loc:
                    continue
                assert loc in declared_locs, (
                    f"{w.scenario_id} step {ev.step} ({ev.kind}): undeclared event location {loc!r}"
                )
                for r in w.rooms:
                    assert not (loc != r and r in loc), (
                        f"{w.scenario_id} step {ev.step} ({ev.kind}): location {loc!r} embeds room name {r!r}"
                    )
            if ev.target_object == dom.note_item and ev.kind in ("move", "see"):
                room_conts = set(w.containers_by_room[ev.room])
                assert ev.from_location in room_conts, (
                    f"{w.scenario_id} step {ev.step}: note_item from_location {ev.from_location!r} not in {room_conts}"
                )
                assert ev.to_location in room_conts, (
                    f"{w.scenario_id} step {ev.step}: note_item to_location {ev.to_location!r} not in {room_conts}"
                )
        assert callable(validate_ev_loc_fn), "validate_event_locations must be exported by udlab_generator."
        validate_ev_loc_fn(w)

    # Verify old 5ea83830 step-28 move ("dispatch folder" -> "countertop in the <room>") is refused
    # by validate_event_locations, validate_object_locations, verify_sealed_part, and --verify-written:
    w_s_who = next(generate_world(c) for c in configs if not c.u_is_who)
    dom_s = next(d for d in DOMAINS if d.name == w_s_who.domain_name)
    note_move_ev = next(ev for ev in w_s_who.events if ev.kind == "move" and ev.target_object == dom_s.note_item)
    old_step28_events = tuple(
        replace(
            ev,
            from_location="dispatch folder",
            to_location=f"countertop in the {ev.room}",
        )
        if ev.step == note_move_ev.step
        else ev
        for ev in w_s_who.events
    )
    old_step28_world = replace(w_s_who, events=old_step28_events)
    with pytest.raises(ValueError, match="event location|declared|embeds room"):
        validate_ev_loc_fn(old_step28_world)

    # Also test --verify-written / verify_sealed_part refusal when a sealed world has the old step-28 locations:
    out_dir = tmp_path / "tr_verify_written_ev_loc"
    build_and_probe_dataset(
        count=4,
        base_seed=20261030,
        output_dir=out_dir,
        writer_kind="mock",
    )
    assert cli_main(["--verify-written", str(out_dir)]) == 0
    sealing_meta_path = out_dir / "sealed_store" / "sealing.sealed.json"
    meta = json.loads(sealing_meta_path.read_text(encoding="utf-8"))
    for scen_dict in meta["scenarios"].values():
        for ev_dict in scen_dict["events"]:
            if ev_dict.get("target_object") in {d.note_item for d in DOMAINS}:
                ev_dict["from_location"] = "dispatch folder"
                ev_dict["to_location"] = f"countertop in the {ev_dict['room']}"
                break
        break
    sealing_meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(SealingError, match="event location|declared|embeds room|validation failed"):
        verify_sealed_part(out_dir / "part_public", out_dir / "sealed_store")
    assert cli_main(["--verify-written", str(out_dir)]) == 1


def test_85_card_tr_absent_event_repair_position_and_order_refusal(tmp_path: Path) -> None:
    """Test 85 (Card tr Req 7):
    - For an ABSENT event, format_repair_retry_line names the position:
      '...add one sentence for event N immediately after the sentence for event N-1 ...'
      (and 'at the start of the events' for event 1).
    - Generated from check_id and the absent flag, recorded with its sha256 in generations.jsonl.
    - Tested with a made-up case where the absent event (event 28) is inserted in the wrong place
      (refused as c_fidelity_event_order) and in the right place (passes).
    """
    import hashlib
    from typing import Any
    from udlab_generator import (
        PINNED_WRITER_MANIFEST_DIGEST,
        PINNED_WRITER_MODEL_TAG,
    )
    from udlab_generator.render import format_repair_retry_line

    # 1. Direct unit checks on event 1 vs event N > 1 for absent flag:
    exc_ev1 = RenderCheckError(
        "missing event 1",
        check_id="c_fidelity_missing_event",
        failing_sentence="synthetic sentence",
        event_index=1,
        event_kind="see",
        failure_kind="absent",
    )
    line_ev1 = format_repair_retry_line(exc_ev1)
    assert "add one sentence for event 1 at the start of the events" in line_ev1
    assert "c_fidelity_missing_event" in line_ev1

    exc_ev28 = RenderCheckError(
        "missing event 28",
        check_id="c_fidelity_missing_event",
        failing_sentence="synthetic sentence",
        event_index=28,
        event_kind="move",
        failure_kind="absent",
    )
    line_ev28 = format_repair_retry_line(exc_ev28)
    assert "add one sentence for event 28 immediately after the sentence for event 27" in line_ev28
    assert "c_fidelity_missing_event" in line_ev28

    # 2. Made-up scenario where the settled note_item move event is absent on attempt 0:
    from udlab_generator.world import DOMAINS
    from udlab_generator.render import _split_prose_sentences

    cfgs = make_balanced_configs(8, base_seed=20261030)
    w = next(generate_world(c) for c in cfgs if not c.u_is_who)
    dom = next(d for d in DOMAINS if d.name == w.domain_name)
    note_ev = next(ev for ev in w.events if ev.kind == "move" and ev.target_object == dom.note_item)
    note_idx = note_ev.step - 1  # 0-based index in w.events
    assert note_ev.step >= 2
    fa = w.agents[0]
    full_prose = render_world_template(w, fa)
    setup_para, events_para = full_prose.split("\n\n", 1)

    ev_sents = _split_prose_sentences(events_para)
    assert len(ev_sents) == len(w.events)

    # Attempt 0: omit sentence for note_ev
    sents_absent = ev_sents[:note_idx] + ev_sents[note_idx + 1 :]
    prose_absent = f"{setup_para}\n\n{' '.join(sents_absent)}"
    with pytest.raises(RenderCheckError) as exc_abs_ev:
        check_rendered_prose(prose_absent, w, focal_agent=fa)
    assert exc_abs_ev.value.check_id == "c_fidelity_missing_event"
    assert exc_abs_ev.value.event_index == note_ev.step
    assert exc_abs_ev.value.failure_kind == "absent"
    repair_note = format_repair_retry_line(exc_abs_ev.value)
    assert (
        f"add one sentence for event {note_ev.step} immediately after the sentence for event {note_ev.step - 1}"
        in repair_note
    )

    # Wrong-place insertion: insert note_ev sentence two events earlier -> refused as c_fidelity_event_order
    sents_wrong_place = (
        ev_sents[: note_idx - 2]
        + [ev_sents[note_idx], ev_sents[note_idx - 2], ev_sents[note_idx - 1]]
        + ev_sents[note_idx + 1 :]
    )
    prose_wrong_place = f"{setup_para}\n\n{' '.join(sents_wrong_place)}"
    with pytest.raises(RenderCheckError) as exc_wrong_order:
        check_rendered_prose(prose_wrong_place, w, focal_agent=fa)
    assert exc_wrong_order.value.check_id == "c_fidelity_event_order"

    # Right-place insertion: insert note_ev sentence immediately after event (note_ev.step - 1) -> passes
    sents_right_place = sents_absent[:note_idx] + [ev_sents[note_idx]] + sents_absent[note_idx:]
    prose_right_place = f"{setup_para}\n\n{' '.join(sents_right_place)}"
    assert check_rendered_prose(prose_right_place, w, focal_agent=fa) == ()

    # 3. Verify recording of repair_line and repair_line_sha256 in generations.jsonl on absent event:
    w0 = generate_world(cfgs[0])
    r0_full = render_world_template(w0, w0.agents[0])
    r1_full = render_world_template(w0, w0.agents[1])
    s0_p, e0_p = r0_full.split("\n\n", 1)
    e0_list = _split_prose_sentences(e0_p)
    # Omit event 1 on r0 attempt 0, omit event 5 on r1 attempt 0:
    r0_absent_ev1 = f"{s0_p}\n\n{' '.join(e0_list[1:])}"
    s1_p, e1_p = r1_full.split("\n\n", 1)
    e1_list = _split_prose_sentences(e1_p)
    r1_absent_ev5 = f"{s1_p}\n\n{' '.join(e1_list[:4] + e1_list[5:])}"

    scripted_Texts = [r0_absent_ev1, r0_full, r1_absent_ev5, r1_full]
    c_idx = 0

    def _transport(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        nonlocal c_idx
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            out_txt = scripted_Texts[c_idx]
            c_idx += 1
            return {
                "message": {"role": "assistant", "content": out_txt},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3050,
                "eval_count": 800,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    out_dir = tmp_path / "tr_absent_event_repair"
    build_and_probe_dataset(
        count=1,
        base_seed=20261030,
        output_dir=out_dir,
        writer_kind="ollama",
        max_retries=1,
        grant_id="test-grant-tr-85",
        http_transport=_transport,
        start_owned_server=False,
    )
    gen_rows = [
        json.loads(line)
        for line in (out_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(gen_rows) == 4
    assert "add one sentence for event 1 at the start of the events" in gen_rows[1]["repair_line"]
    assert gen_rows[1]["repair_line_sha256"] == hashlib.sha256(gen_rows[1]["repair_line"].encode("utf-8")).hexdigest()
    assert "add one sentence for event 5 immediately after the sentence for event 4" in gen_rows[3]["repair_line"]
    assert gen_rows[3]["repair_line_sha256"] == hashlib.sha256(gen_rows[3]["repair_line"].encode("utf-8")).hexdigest()


def test_86_card_ua_folded_enter_never_consumes_the_next_sentence() -> None:
    """Test 86 (Card ua): a folded leave-and-enter sentence counts for both events, and the `enter` never takes
    the sentence after it when that sentence states some other action (a move, a see, by the same actor or
    another). The event after the enter is searched from the sentence after the folded one.

    Built on seed 20261033 (events 4-5: Eva leaves the reading room for the vault room; 6: Eva moves the
    brass ledger in the vault room; 16-17: Ana leaves the reading room for the vault room; 18: Eva moves the
    brass ledger in the vault room; 22-23: Eva leaves the reading room for the bindery room; 24: Eva sees
    the index reel there). All sentences are synthetic: the template rendering with the pair replaced.
    """
    from udlab_generator.render import _sentence_matches_event, _split_prose_sentences

    w = generate_world(make_balanced_configs(1, base_seed=20261033)[0])
    fa = w.agents[0]
    setup_para, events_para = render_world_template(w, fa).split("\n\n")
    sents = _split_prose_sentences(events_para)
    assert len(sents) == len(w.events) == 36
    ev4, ev5, ev6 = w.events[3], w.events[4], w.events[5]
    assert (ev4.kind, ev5.kind, ev6.kind) == ("leave", "enter", "move")
    assert ev4.actor == ev5.actor == ev6.actor == "Eva"
    assert ev5.room == ev6.room == "vault room" and ev5.from_location == "reading room"
    assert (w.events[15].kind, w.events[16].kind, w.events[17].kind) == ("leave", "enter", "move")
    assert w.events[16].actor == "Ana" and w.events[17].actor == "Eva" and w.events[17].room == "vault room"
    assert (w.events[21].kind, w.events[22].kind, w.events[23].kind) == ("leave", "enter", "see")
    assert w.events[22].actor == w.events[23].actor == "Eva" and w.events[23].room == "bindery room"

    def prose_of(event_sents: list[str]) -> str:
        return f"{setup_para}\n\n{' '.join(event_sents)}"

    folded_eva_vault = "Next, Eva walked from the reading room into the vault room."
    folded_ana_vault = "Next, Ana walked from the reading room into the vault room."
    folded_eva_bindery = "Next, Eva walked from the reading room into the bindery room."

    # The move sentence after the folded one is never an enter sentence, with or without the leave context:
    move_sent = sents[5]
    assert _sentence_matches_event(move_sent, ev6, w.noise_threshold, world=w)
    assert not _sentence_matches_event(move_sent, ev5, w.noise_threshold, world=w)
    assert not _sentence_matches_event(
        move_sent, ev5, w.noise_threshold, world=w, prev_sentence=folded_eva_vault, prev_event=ev4
    )

    # Card uf item 1: a move verb counts only when applied to a named thing (its object or source). An arrival
    # that merely mentions a room feature is still an arrival; a station move is still a move, not an arrival.
    from udlab_generator.render import _sentence_states_non_arrival_action

    parties = tuple(w.agents) + tuple(w.absent_parties)
    move_in_vault = "Next, Eva moved the brass ledger from the intake bench to the oak drawer in the vault room."
    assert _sentence_states_non_arrival_action(move_in_vault.lower(), w, parties)
    assert not _sentence_matches_event(move_in_vault, ev5, w.noise_threshold, world=w)
    assert not _sentence_matches_event(
        move_in_vault, ev5, w.noise_threshold, world=w, prev_sentence=folded_eva_vault, prev_event=ev4
    )
    station_move = sents[6]  # Ben steps from the container bay behind the frosted partition into the alcove
    assert w.events[6].kind == "move" and w.events[6].actor == "Ben"
    assert _sentence_states_non_arrival_action(station_move.lower(), w, parties)
    for arrival_with_feature in (
        "Next, Eva stepped into the vault room past the frosted partition.",
        "Next, Eva stepped into the vault room, where the oak drawer stood.",
    ):
        assert not _sentence_states_non_arrival_action(arrival_with_feature.lower(), w, parties)
        with_feature = list(sents)
        with_feature[4] = arrival_with_feature
        assert check_rendered_prose(prose_of(with_feature), w, focal_agent=fa) == ()
    # The card's defect sentence after a folded leave+enter is never taken as the enter; the move keeps it.
    assert (
        check_rendered_prose(prose_of(sents[:3] + [folded_eva_vault, move_in_vault] + sents[6:]), w, focal_agent=fa)
        == ()
    )

    # 1. Folded leave+enter, then the same actor's move in the arrival room: passes; the move keeps its sentence.
    assert check_rendered_prose(prose_of(sents[:3] + [folded_eva_vault] + sents[5:]), w, focal_agent=fa) == ()
    # The same with a leave-verb folded form ("left ... and went to ...").
    assert (
        check_rendered_prose(
            prose_of(sents[:3] + ["Next, Eva left the reading room and went to the vault room."] + sents[5:]),
            w,
            focal_agent=fa,
        )
        == ()
    )

    # 2. Unfolded leave and enter sentences, then the same actor's move: one sentence per event, passes.
    assert check_rendered_prose(prose_of(sents), w, focal_agent=fa) == ()
    assert (
        check_rendered_prose(prose_of(sents[:4] + ["Next, Eva entered the vault room."] + sents[5:]), w, focal_agent=fa)
        == ()
    )

    # 3. Folded leave+enter, then a DIFFERENT actor's move in the arrival room: passes.
    assert check_rendered_prose(prose_of(sents[:15] + [folded_ana_vault] + sents[17:]), w, focal_agent=fa) == ()

    # 4. Folded leave+enter, then the same actor's see: passes.
    assert check_rendered_prose(prose_of(sents[:21] + [folded_eva_bindery] + sents[23:]), w, focal_agent=fa) == ()

    # 5. Negative control: folded leave+enter with the move sentence omitted refuses the move as absent.
    with pytest.raises(RenderCheckError) as exc_absent:
        check_rendered_prose(prose_of(sents[:3] + [folded_eva_vault] + sents[6:]), w, focal_agent=fa)
    assert exc_absent.value.check_id == "c_fidelity_missing_event"
    assert exc_absent.value.event_index == 6
    assert exc_absent.value.event_kind == "move"
    assert exc_absent.value.failure_kind == "absent"

    # 6. Negative control: an unfolded pair whose entry sentence names the WRONG room still refuses.
    # (The enter can only bind to Eva's later arrival sentence, so the move at event 6 is out of order.)
    for wrong_room_entry in (
        "Next, Eva walked through the doorway from the reading room into the bindery room.",
        "Next, Eva entered the bindery room.",
    ):
        wrong_room = list(sents)
        wrong_room[4] = wrong_room_entry
        with pytest.raises(RenderCheckError) as exc_wrong:
            check_rendered_prose(prose_of(wrong_room), w, focal_agent=fa)
        assert exc_wrong.value.check_id == "c_fidelity_event_order"
        assert exc_wrong.value.event_index == 6


def test_87_card_ue_station_change_and_occluded_see_prompt_and_checker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test 87 (Card ue): a station change is the actor stepping behind/around the room's occluder (never moving
    it), and an occluded see is the occluder blocking the actor's sight (never the actor seeing). The prompt
    clauses and the checker's accepted forms come from the same constants, and the alignment check covers them.
    Card ua's advisory: an adverb between a move verb and its object is still a move, never an enter.

    Built on seed 20261033 (events 7, 11, 13: Ben steps behind / back around / behind the frosted partition in
    the vault room; 34-35: Ben's occluded sees of the oak drawer and the steel cabinet from the far-side alcove;
    4-6: Eva's leave, enter and move of the brass ledger). All sentences are synthetic.
    """
    import udlab_generator.render as ug_render
    from udlab_generator.render import (
        NEGATION_PHRASES,
        OCCLUDED_SEE_BLOCKED_PHRASES,
        OCCLUDED_SEE_COULD_NOT_SEE,
        ProseRequirementSpec,
        SIGHTING_VERBS,
        STATION_MOVE_DIRECTION_WORDS,
        _sentence_asserts_sighting_of_target,
        _sentence_matches_event,
        _sentence_states_non_arrival_action,
        _sentence_states_occluder_blocked_actor_sight,
        _split_prose_sentences,
        format_occluded_see_clause,
        format_repair_retry_line,
        format_station_change_clause,
        get_station_and_occlusion_checker_forms,
        render_writer_prompt_constraints,
        verify_prompt_and_checker_requirements_aligned,
    )
    from udlab_generator.world import is_station_transition_event

    w = generate_world(make_balanced_configs(1, base_seed=20261033)[0])
    fa = w.agents[0]
    setup_para, events_para = render_world_template(w, fa).split("\n\n")
    sents = _split_prose_sentences(events_para)
    assert len(sents) == len(w.events) == 36
    sm, back, sm2 = w.events[6], w.events[10], w.events[12]
    assert all(is_station_transition_event(e) for e in (sm, back, sm2))
    assert sm.actor == back.actor == sm2.actor == "Ben" and sm.room == "vault room"
    assert (sm.from_location, sm.to_location) == ("container bay", "far-side alcove")
    assert (back.from_location, back.to_location) == ("far-side alcove", "container bay")
    os_a, os_b = w.events[33], w.events[34]
    assert os_a.kind == os_b.kind == "see" and os_a.occluded and os_b.occluded
    assert os_a.actor == os_b.actor == "Ben" and os_a.occluder == os_b.occluder == sm.target_object
    a, occ, room = sm.actor, sm.target_object, sm.room
    tgt_a, tgt_b = os_a.target_object, os_b.target_object
    parties = tuple(w.agents) + tuple(w.absent_parties)

    def m(sentence: str, ev: Event) -> bool:
        return _sentence_matches_event(sentence, ev, w.noise_threshold, world=w)

    def prose_of(event_sents: list[str]) -> str:
        return f"{setup_para}\n\n{' '.join(event_sents)}"

    # Must refuse 1: the occluder as the direct object of a move verb, no "behind"/"around" (the furniture form).
    literal_forms = [
        f"Next, in the {room}, {a} moved the {occ} from the container bay to the far-side alcove.",
        f"Next, in the {room}, {a} shifted the {occ} from the container bay to the far-side alcove.",
        f"Next, in the {room}, {a} moved quickly the {occ} from the container bay to the far-side alcove.",
    ]
    for literal in literal_forms:
        assert not m(literal, sm), literal
    assert not m(f"Next, in the {room}, {a} moved the {occ} from the far-side alcove to the container bay.", back)
    # A direction word does not rescue the occluder as the object of the move verb.
    assert not m(f"Next, in the {room}, {a} moved the {occ} behind him to the far-side alcove.", sm)
    # The right station is required.
    assert not m(f"Next, in the {room}, {a} stepped behind the {occ} to the container bay.", sm)
    assert not m(f"Next, in the {room}, {a} stepped to the far-side alcove.", sm)

    # Must refuse 2: the agentless blocked form (perceptual records are per actor).
    agentless = f"Next, in the {room}, the {occ} blocked sight of the {tgt_a} from the far-side alcove."
    assert not m(agentless, os_a)
    assert not _sentence_states_occluder_blocked_actor_sight(agentless.lower(), os_a)
    assert not m(f"Next, in the {room}, the {occ} blocked the view of the {tgt_a} from the far-side alcove.", os_a)

    # Must refuse 3: the actor saw / observed the container on an occluded see, with and without "behind the occluder".
    false_sightings = [
        f"Next, in the {room}, {a} saw the {tgt_a} behind the {occ} from the far-side alcove, which blocked the view.",
        f"Next, in the {room}, {a} saw the {tgt_a} from the far-side alcove.",
        f"Next, in the {room}, {a} observed the {tgt_a} behind the {occ}, which blocked sight from the far-side alcove.",
        f"Next, in the {room}, {a} observed the {tgt_a} from the far-side alcove.",
        f"Next, in the {room}, from the far-side alcove, {a} noticed the {tgt_a}, though the {occ} blocked part of the view.",
    ]
    for false_sighting in false_sightings:
        assert _sentence_asserts_sighting_of_target(false_sighting.lower(), os_a, parties), false_sighting
        assert not m(false_sighting, os_a), false_sighting
    # Fix round ui, item 2: the sighting verb must take the container as its object. A sighting verb whose object
    # is something else (the fact that the occluder blocked the actor's sight; "nothing of" the container) is not
    # a sighting of the container, and these correct sentences pass.
    noticed_that = f"Next, in the {room}, {a} noticed that the {occ} blocked {a}'s sight of the {tgt_a} from the far-side alcove."
    saw_nothing = f"Next, in the {room}, {a} saw nothing of the {tgt_a} from the far-side alcove, blocked by the {occ}."
    for correct_see in (noticed_that, saw_nothing):
        assert not _sentence_asserts_sighting_of_target(correct_see.lower(), os_a, parties), correct_see
    assert m(noticed_that, os_a)
    # "saw nothing of" is no longer a false sighting, but 'nothing' is a NEGATION_PHRASES entry and the spec allows
    # no negation in an event sentence except 'could not see' on an occluded see, so the pre-existing negation
    # guard still refuses it. Changing that rule is outside this card.
    from udlab_generator.render import _has_forbidden_event_negation

    assert "nothing" in NEGATION_PHRASES
    assert _has_forbidden_event_negation(saw_nothing.lower(), os_a, parties, action_verbs=SIGHTING_VERBS)
    assert not m(saw_nothing, os_a)
    # An attempt verb with the blocked clause is still the occluded see (the actor looked, but did not see).
    for attempt_verb in ("looked toward", "watched", "glanced at"):
        attempt = f"Next, in the {room}, from the far-side alcove, {a} {attempt_verb} the {tgt_a}, but the {occ} blocked the view."
        assert not _sentence_asserts_sighting_of_target(attempt.lower(), os_a, parties), attempt
        assert m(attempt, os_a), attempt

    # Must pass 4: the possessive form (sight / view / line of sight).
    possessive = f"Next, in the {room}, the {occ} blocked {a}'s sight of the {tgt_a} from the far-side alcove."
    assert _sentence_states_occluder_blocked_actor_sight(possessive.lower(), os_a)
    assert m(possessive, os_a)
    assert m(f"Next, in the {room}, the {occ} blocked {a}'s view of the {tgt_a} from the far-side alcove.", os_a)
    assert m(f"Next, in the {room}, from the far-side alcove the {occ} blocked {a}'s line of sight of the {tgt_a}.", os_a)
    # Another actor's possessive does not state this actor's see.
    other = next(p for p in w.agents if p != a)
    assert not m(f"Next, in the {room}, the {occ} blocked {other}'s sight of the {tgt_a} from the far-side alcove.", os_a)

    # Must pass 5: stepped behind the occluder to the far-side alcove, and the return stepped back around it.
    stepped_behind = f"Next, in the {room}, {a} stepped behind the {occ} to the far-side alcove."
    back_around = f"Next, in the {room}, {a} stepped back around the {occ} to the container bay."
    assert m(stepped_behind, sm) and m(stepped_behind, sm2)
    assert m(back_around, back)
    assert m(f"Next, in the {room}, {a} stepped from the container bay behind the {occ} into the far-side alcove.", sm)
    assert not m(back_around, sm) and not m(stepped_behind, back)
    # The station-change form is a move, never an arrival (so an enter can never consume it).
    assert _sentence_states_non_arrival_action(stepped_behind.lower(), w, parties)
    assert _sentence_states_non_arrival_action(back_around.lower(), w, parties)

    # Must pass 6: could not see, blocked by the occluder.
    could_not_see = f"Next, in the {room}, {a} could not see the {tgt_b} from the far-side alcove, blocked by the {occ}."
    assert m(could_not_see, os_b)

    # The template renders the new forms and passes its own checks; the full prose with every new form passes too.
    assert check_rendered_prose(prose_of(sents), w, focal_agent=fa) == ()
    for idx in (6, 10, 12):
        assert m(sents[idx], w.events[idx])
    for idx in (33, 34):
        # The template: the actor looked, and the occluder blocked the actor's line of sight (per actor; the
        # container named once so mention parity holds).
        assert m(sents[idx], w.events[idx])
        assert f"blocked {a}'s line of sight" in sents[idx]
        assert sents[idx].count(w.events[idx].target_object) == 1
    new_forms = list(sents)
    new_forms[6], new_forms[10], new_forms[12] = stepped_behind, back_around, stepped_behind
    new_forms[33], new_forms[34] = possessive, could_not_see
    assert check_rendered_prose(prose_of(new_forms), w, focal_agent=fa) == ()

    # The furniture form at every station move refuses on the first station move; the repair line names the
    # stepping forms (Publisher condition 5 applied to station moves).
    furniture = list(sents)
    furniture[6] = literal_forms[0]
    furniture[10] = f"Next, in the {room}, {a} moved the {occ} from the far-side alcove to the container bay."
    furniture[12] = literal_forms[0]
    with pytest.raises(RenderCheckError) as exc_sm:
        check_rendered_prose(prose_of(furniture), w, focal_agent=fa)
    assert exc_sm.value.check_id == "c_fidelity_missing_event"
    assert (exc_sm.value.event_index, exc_sm.value.event_kind) == (7, "move")
    assert exc_sm.value.event_variant == "station_move"
    for form in ("multi_turn", "single_turn"):
        line = format_repair_retry_line(exc_sm.value, retry_form=form)
        assert "stepped behind" in line and "around" in line and "did not move the occluder" in line

    # The agentless form refuses the occluded see; the repair line names "blocked" and "could not see"
    # (Publisher condition 5). The false sighting refuses as well.
    for bad_see in (agentless, false_sightings[0]):
        bad = list(sents)
        bad[33] = bad_see
        with pytest.raises(RenderCheckError) as exc_os:
            check_rendered_prose(prose_of(bad), w, focal_agent=fa)
        assert exc_os.value.check_id == "c_fidelity_missing_event"
        assert (exc_os.value.event_index, exc_os.value.event_kind) == (34, "see")
        assert exc_os.value.event_variant == "occluded_see"
        for form in ("multi_turn", "single_turn"):
            line = format_repair_retry_line(exc_os.value, retry_form=form)
            assert OCCLUDED_SEE_BLOCKED_PHRASES[0] in line and OCCLUDED_SEE_COULD_NOT_SEE in line
            assert "never that the actor saw or observed it" in line

    # (C) Card ua's advisory: an adverb between the move verb and its object is still a move, never an enter.
    ev4, ev5, ev6 = w.events[3], w.events[4], w.events[5]
    assert (ev4.kind, ev5.kind, ev6.kind) == ("leave", "enter", "move") and ev6.actor == "Eva"
    folded_eva_vault = f"Next, {ev5.actor} walked from the {ev5.from_location} into the {ev5.to_location}."
    adverb_move = (
        f"Next, {ev6.actor} moved quickly the {ev6.target_object} from the {ev6.from_location} "
        f"to the {ev6.to_location} in the {ev6.room}."
    )
    assert _sentence_states_non_arrival_action(adverb_move.lower(), w, parties)
    assert m(adverb_move, ev6)
    assert not m(adverb_move, ev5)
    assert not _sentence_matches_event(
        adverb_move, ev5, w.noise_threshold, world=w, prev_sentence=folded_eva_vault, prev_event=ev4
    )
    assert check_rendered_prose(prose_of(sents[:3] + [folded_eva_vault, adverb_move] + sents[6:]), w, focal_agent=fa) == ()

    # 7. The prompt clauses and the checker forms come from one source: the alignment check, extended.
    assert verify_prompt_and_checker_requirements_aligned() == ug_render.WRITER_PROSE_REQUIREMENTS
    formatted = format_writer_prompt(w, fa)
    station_clause, see_clause = format_station_change_clause(), format_occluded_see_clause()
    assert station_clause in formatted and see_clause in formatted
    event_order_rule = ug_render.WRITER_PROSE_REQUIREMENTS_BY_ID["c_fidelity_event_order"].rule_text
    assert station_clause in event_order_rule and see_clause in event_order_rule
    for form in get_station_and_occlusion_checker_forms():
        assert form in event_order_rule, form
    assert set(STATION_MOVE_DIRECTION_WORDS) <= set(get_station_and_occlusion_checker_forms())
    assert "on station_change, write '<actor> stepped behind the <occluder> to the far-side alcove'" in station_clause
    assert "stepped back around the <occluder> to the container bay" in station_clause
    assert "never" not in station_clause.split(ug_render.STATION_EXAMPLE_NOTE)[0]
    assert "blocked <actor>'s sight of the <container> from the far-side alcove" in see_clause
    assert "could not see the <container> from the far-side alcove, blocked by the <occluder>" in see_clause
    assert "never that the actor saw or observed it" in see_clause
    # A prompt whose event-order rule lacks a clause is refused, by the requirements comparison ...
    with pytest.raises(ValueError):
        verify_prompt_and_checker_requirements_aligned(
            load_writer_prompt_template().replace(f"{station_clause}; ", "")
        )
    # ... and, if the shared spec itself dropped the clause, by the extended check.
    stripped = tuple(
        ProseRequirementSpec(r.check_id, r.rule_text.replace(f"{see_clause}; ", ""), r.required_in_setup)
        if r.check_id == "c_fidelity_event_order"
        else r
        for r in ug_render.WRITER_PROSE_REQUIREMENTS
    )
    monkeypatch.setattr(ug_render, "WRITER_PROSE_REQUIREMENTS", stripped)
    stripped_prompt = ug_render.WRITER_PROMPT_PATH.read_text(encoding="utf-8").replace(
        ug_render.HARD_CONSTRAINTS_PLACEHOLDER, render_writer_prompt_constraints(stripped)
    )
    with pytest.raises(ValueError, match="checker-derived clause"):
        verify_prompt_and_checker_requirements_aligned(stripped_prompt)
    monkeypatch.undo()

    # Fix round ui, item 1: STATION_MOVE_VERB and OCCLUDED_SEE_COULD_NOT_SEE are enforced by the checker, not only
    # rendered: the could-not regexes are built from the constant and the station verb must be a move verb the
    # subject gate accepts. Diverging either constant fails the constants check (run at import and by the
    # alignment check).
    from udlab_generator.render import (
        _COULD_NOT_SEE_RE,
        _OCCLUDED_SEE_COULD_NOT_PREFIX_RE,
        verify_station_and_occlusion_constants,
    )

    verify_station_and_occlusion_constants()
    assert ug_render.STATION_MOVE_VERB in ug_render._ACTION_VERBS_BY_KIND["move"]
    assert _COULD_NOT_SEE_RE.fullmatch(ug_render.OCCLUDED_SEE_COULD_NOT_SEE)
    assert _OCCLUDED_SEE_COULD_NOT_PREFIX_RE.search("ben could not ")
    assert m(
        f"Next, in the {room}, {a} {ug_render.OCCLUDED_SEE_COULD_NOT_SEE} the {tgt_b} from the far-side alcove, "
        f"blocked by the {occ}.",
        os_b,
    )
    assert m(f"Next, in the {room}, {a} {ug_render.STATION_MOVE_VERB} behind the {occ} to the far-side alcove.", sm)
    monkeypatch.setattr(ug_render, "STATION_MOVE_VERB", "slipped")
    with pytest.raises(ValueError, match="STATION_MOVE_VERB"):
        verify_station_and_occlusion_constants()
    with pytest.raises(ValueError, match="STATION_MOVE_VERB"):
        verify_prompt_and_checker_requirements_aligned()
    monkeypatch.undo()
    monkeypatch.setattr(ug_render, "OCCLUDED_SEE_COULD_NOT_SEE", "could not spot")
    with pytest.raises(ValueError, match="OCCLUDED_SEE_COULD_NOT_SEE"):
        verify_station_and_occlusion_constants()
    with pytest.raises(ValueError, match="OCCLUDED_SEE_COULD_NOT_SEE"):
        verify_prompt_and_checker_requirements_aligned()
    monkeypatch.undo()
    monkeypatch.setattr(ug_render, "OCCLUDED_SEE_COULD_NOT_SEE", "failed to see")
    with pytest.raises(ValueError, match="could-not regexes"):
        verify_station_and_occlusion_constants()
    monkeypatch.undo()
    verify_station_and_occlusion_constants()

    # 9. Coverage of the seven station-move emission sites in `world.py` (Publisher condition 4). The sites are
    # selected by the (B1, B2) cell: (False, True) emits bay->alcove, alcove->bay, bay->alcove (three sites);
    # (True, False) emits the same sequence from three other sites; (False, False) and (True, True) emit one
    # bay->alcove each ((False, False) shares its site with (True, False)). One world per cell reaches all seven.
    expected_by_cell = {
        (False, True): [("container bay", "far-side alcove"), ("far-side alcove", "container bay"), ("container bay", "far-side alcove")],
        (True, False): [("container bay", "far-side alcove"), ("far-side alcove", "container bay"), ("container bay", "far-side alcove")],
        (False, False): [("container bay", "far-side alcove")],
        (True, True): [("container bay", "far-side alcove")],
    }
    cells_seen: dict[tuple[bool, bool], int] = {}
    for cfg in make_balanced_configs(16, base_seed=20261033):
        cell = (cfg.b1_true_belief, cfg.b2_true_belief)
        if cell in cells_seen:
            continue
        wc = generate_world(cfg)
        station_moves = [ev for ev in wc.events if is_station_transition_event(ev)]
        assert [(ev.from_location, ev.to_location) for ev in station_moves] == expected_by_cell[cell], cell
        assert len({ev.actor for ev in station_moves}) == 1 and len({ev.target_object for ev in station_moves}) == 1
        fc = wc.agents[0]
        setup_c, events_c = render_world_template(wc, fc).split("\n\n")
        sents_c = _split_prose_sentences(events_c)
        assert len(sents_c) == len(wc.events)
        assert check_rendered_prose(f"{setup_c}\n\n{' '.join(sents_c)}", wc, focal_agent=fc) == ()
        literal_c = list(sents_c)
        first_sm_idx: int | None = None
        for i, ev in enumerate(wc.events):
            if is_station_transition_event(ev):
                assert _sentence_matches_event(sents_c[i], ev, wc.noise_threshold, world=wc), (cell, i)
                literal_c[i] = (
                    f"Next, in the {ev.room}, {ev.actor} moved the {ev.target_object} "
                    f"from the {ev.from_location} to the {ev.to_location}."
                )
                assert not _sentence_matches_event(literal_c[i], ev, wc.noise_threshold, world=wc), (cell, i)
                if first_sm_idx is None:
                    first_sm_idx = i
        assert first_sm_idx is not None
        with pytest.raises(RenderCheckError) as exc_c:
            check_rendered_prose(f"{setup_c}\n\n{' '.join(literal_c)}", wc, focal_agent=fc)
        assert exc_c.value.check_id == "c_fidelity_missing_event"
        assert exc_c.value.event_index == first_sm_idx + 1 and exc_c.value.event_variant == "station_move"
        cells_seen[cell] = len(station_moves)
    assert sorted(cells_seen) == sorted(expected_by_cell)
    assert sum(cells_seen.values()) == 8  # eight station-move events over the four cells, from seven sites


def test_88_card_ul_station_worked_examples_from_constants_pass_checker_and_mutations_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test 88 (Card ul): the station-change clause carries one worked step and one worked return, and the
    occluded-see clause one worked sentence, all rendered from the card-ue constants with made-up names that
    appear in no domain, room, fixture, container, object or agent-name pool. Each example passes the checker
    for its event; each mutation is refused; the alignment check covers them and fails when an example breaks."""
    import udlab_generator.render as ug_render
    from udlab_generator.render import (
        STATION_EXAMPLE_ACTOR,
        STATION_EXAMPLE_CONTAINER,
        STATION_EXAMPLE_NOTE,
        STATION_EXAMPLE_OCCLUDER,
        STATION_EXAMPLE_ROOM,
        STATION_MOVE_VERB,
        _sentence_matches_event,
        format_occluded_see_clause,
        format_occluded_see_example,
        format_station_change_clause,
        format_station_change_examples,
        get_station_and_occlusion_mutated_examples,
        get_station_and_occlusion_worked_examples,
        render_writer_prompt_constraints,
        verify_prompt_and_checker_requirements_aligned,
        verify_station_and_occlusion_examples_pass_checker,
    )
    from udlab_generator.world import AGENT_NAME_POOLS, DOMAINS, is_station_transition_event

    step, back = format_station_change_examples()
    see = format_occluded_see_example()
    assert step == (
        f"In the {STATION_EXAMPLE_ROOM}, {STATION_EXAMPLE_ACTOR} {STATION_MOVE_VERB} behind the "
        f"{STATION_EXAMPLE_OCCLUDER} to the far-side alcove."
    )
    assert back == (
        f"In the {STATION_EXAMPLE_ROOM}, {STATION_EXAMPLE_ACTOR} {STATION_MOVE_VERB} back around the "
        f"{STATION_EXAMPLE_OCCLUDER} to the container bay."
    )
    assert see == (
        f"In the {STATION_EXAMPLE_ROOM}, the {STATION_EXAMPLE_OCCLUDER} blocked {STATION_EXAMPLE_ACTOR}'s sight "
        f"of the {STATION_EXAMPLE_CONTAINER} from the far-side alcove."
    )
    # The clauses carry them, with the note, and so does the rendered prompt.
    assert f"e.g. '{step}' and '{back}' ({STATION_EXAMPLE_NOTE})" in format_station_change_clause()
    assert f"e.g. '{see}' ({STATION_EXAMPLE_NOTE})" in format_occluded_see_clause()
    constraints = render_writer_prompt_constraints()
    for example in (step, back, see):
        assert constraints.count(example) == 1
    w = generate_world(make_balanced_configs(1, base_seed=20261033)[0])
    assert format_writer_prompt(w, w.agents[0]).count(step) == 1

    # Made-up names: in no domain pool (rooms, containers, fixtures, occluders, objects, absent parties) and no
    # agent-name pool, as whole phrases and as words.
    pools_text = (repr(DOMAINS) + repr(AGENT_NAME_POOLS)).lower()
    for phrase in (STATION_EXAMPLE_ACTOR, STATION_EXAMPLE_ROOM, STATION_EXAMPLE_OCCLUDER, STATION_EXAMPLE_CONTAINER):
        assert phrase.lower() not in pools_text, phrase
        for word in phrase.lower().split():
            if word != "room":
                assert word not in pools_text, word

    # Each example passes the checker for its synthetic event; each mutation is refused.
    worked = get_station_and_occlusion_worked_examples()
    mutated = get_station_and_occlusion_mutated_examples()
    assert len(worked) == len(mutated) == 3
    (s_step, ev_step), (s_back, ev_back), (s_see, ev_see) = worked
    assert (s_step, s_back, s_see) == (step, back, see)
    assert is_station_transition_event(ev_step) and is_station_transition_event(ev_back)
    assert ev_see.kind == "see" and ev_see.occluded and ev_see.occluder == STATION_EXAMPLE_OCCLUDER
    for sentence, ev in worked:
        assert _sentence_matches_event(sentence, ev, 60, world=None), sentence
    for sentence, ev in mutated:
        assert not _sentence_matches_event(sentence, ev, 60, world=None), sentence
    verify_station_and_occlusion_examples_pass_checker()
    verify_prompt_and_checker_requirements_aligned()

    # Break an example (the step loses its direction word, the literal furniture reading) and the alignment
    # check fails on it.
    broken_step = step.replace(f" {STATION_MOVE_VERB} behind the ", f" {STATION_MOVE_VERB} the ")
    assert broken_step != step
    monkeypatch.setattr(ug_render, "format_station_change_examples", lambda: (broken_step, back))
    with pytest.raises(ValueError, match="Worked example in the writer prompt does not pass the checker"):
        verify_station_and_occlusion_examples_pass_checker()
    with pytest.raises(ValueError):
        verify_prompt_and_checker_requirements_aligned()
    monkeypatch.undo()
    # Weaken a mutation into a passing sentence and the check fails on the other side.
    good_step = get_station_and_occlusion_worked_examples()[0]
    monkeypatch.setattr(
        ug_render,
        "get_station_and_occlusion_mutated_examples",
        lambda: (good_step,) + get_station_and_occlusion_mutated_examples()[1:],
    )
    with pytest.raises(ValueError, match="Mutated worked example passes the checker"):
        verify_station_and_occlusion_examples_pass_checker()


def test_89_card_ul_exact_count_decides_num_ctx_and_over_cap_refuses_and_run_continues(tmp_path: Path) -> None:
    """Test 89 (Card ul): with a tokenizer configured, the exact count of the chat-templated prompt decides
    num_ctx and the cap check (the bytes/3 estimate is recorded only). An estimate over the cap with an exact
    count that fits runs; an exact count truly over the cap is refused with its own check id, c_ctx_over_cap,
    the next rendering and scenario still run, the build summary counts it, and nothing is raised in the build
    loop. A direct caller that forbids refusal gets CtxOverCapRefusalError. num_ctx rounding never exceeds the cap."""
    from udlab_generator.cli import build_and_probe_dataset, describe_prompt_token_counter, resolve_prompt_token_counter
    from udlab_generator.render import (
        CTX_OVER_CAP_CHECK_ID,
        DEFAULT_WRITER_MAX_NUM_CTX,
        DEFAULT_WRITER_MODEL_PIN,
        NUM_CTX_MARGIN_TOKENS,
        CtxOverCapRefusalError,
        MaxNumCtxRefusalError,
        OllamaWriter,
        decide_num_ctx,
        estimate_prompt_tokens,
        prompt_fits_num_ctx_cap,
        render_ollama_qwen35_prompt,
        round_up_num_ctx,
    )

    # The decision helpers.
    assert prompt_fits_num_ctx_cap(4864, 4096, 9216) and not prompt_fits_num_ctx_cap(4865, 4096, 9216)
    assert decide_num_ctx(3393, 4096, max_num_ctx=9216) == 8192  # writer-stub-3d call_0001: 7745 -> 8192
    assert decide_num_ctx(3810, 4096, max_num_ctx=9216) == 8192  # the refused 3d call_0004: 8162 -> 8192, fits
    assert decide_num_ctx(4800, 4096, max_num_ctx=9216) == 9216  # 9152 fits; the rounding lands on the cap
    assert decide_num_ctx(4600, 4096, max_num_ctx=9000) == 9000  # 8952 fits a 9000 cap; the rounding would not
    assert round_up_num_ctx(9303) == 10240

    configs = make_balanced_configs(2, base_seed=20261030)
    worlds = [generate_world(c) for c in configs]
    w0, w1 = worlds
    pinned_digest = DEFAULT_WRITER_MODEL_PIN.manifest_digest
    seen_payloads: list[dict] = []
    counted_texts: list[str] = []

    def _transport(method: str, url: str, body: dict | None = None) -> dict:
        if method == "GET" and url.endswith("/api/version"):
            return {"version": "0.40.9-test"}
        if method == "GET" and url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3.6:35b", "model": "qwen3.6:35b", "digest": f"sha256:{pinned_digest}"}]}
        if method == "POST" and url.endswith("/api/chat"):
            payload = dict(body or {})
            if payload.get("keep_alive") == 0:
                return {"model": "qwen3.6:35b", "done": True, "done_reason": "unload"}
            seen_payloads.append(payload)
            prompt_content = payload["messages"][-1]["content"]
            focal = prompt_content.split("Focal Agent: ", 1)[1].splitlines()[0].strip()
            matching = [w for w in worlds if focal in w.agents]
            assert len(matching) == 1, focal
            return {
                "model": "qwen3.6:35b",
                "message": {"role": "assistant", "content": render_world_template(matching[0], focal)},
                "done": True,
                "done_reason": "stop",
                "eval_count": 700,
                "prompt_eval_count": 3300,
            }
        raise AssertionError(f"Unexpected {method} {url}")

    prompt_0 = format_writer_prompt(w0, w0.agents[0])
    est_0 = estimate_prompt_tokens([{"role": "user", "content": prompt_0}])
    exact_0 = est_0 - 1000  # the measured ratio: bytes/3 overshoots the real count by about a quarter

    # 1. Estimate over the cap, exact count fits: the call runs, num_ctx comes from the exact count.
    num_predict = DEFAULT_WRITER_MAX_NUM_CTX - NUM_CTX_MARGIN_TOKENS - est_0 + 300
    assert not prompt_fits_num_ctx_cap(est_0, num_predict, DEFAULT_WRITER_MAX_NUM_CTX)
    assert prompt_fits_num_ctx_cap(exact_0, num_predict, DEFAULT_WRITER_MAX_NUM_CTX)

    def _counter_fits(text: str) -> int:
        counted_texts.append(text)
        return exact_0

    writer = OllamaWriter(
        grant_id="test-grant-ul",
        http_transport=_transport,
        num_predict=num_predict,
        prompt_token_counter=_counter_fits,
    )
    assert writer.prompt_token_counter is _counter_fits
    pair = writer.render_scenario_pair(w0, allow_refusal=True)
    assert pair is not None and len(seen_payloads) == 2
    expected_ctx = decide_num_ctx(exact_0, num_predict, max_num_ctx=DEFAULT_WRITER_MAX_NUM_CTX)
    assert expected_ctx <= DEFAULT_WRITER_MAX_NUM_CTX
    assert [p["options"]["num_ctx"] for p in seen_payloads] == [expected_ctx, expected_ctx]
    assert counted_texts[0] == render_ollama_qwen35_prompt([{"role": "user", "content": prompt_0}], think=False)
    assert counted_texts[0].startswith("<|im_start|>user\n") and counted_texts[0].endswith(
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    rec = writer.call_records[0]
    assert rec.prompt_tokens_exact == exact_0 and rec.prompt_tokens_estimate == est_0
    assert rec.to_dict()["prompt_tokens_exact"] == exact_0 and rec.to_dict()["tokens"]["prompt_tokens_exact"] == exact_0
    assert rec.num_ctx == expected_ctx and rec.checks_passed

    # 2. Exact count truly over the cap on one rendering: refused with c_ctx_over_cap, the next rendering runs,
    #    the next scenario runs, the summary counts it, nothing is raised.
    seen_payloads.clear()
    counted_texts.clear()
    over_once = {"armed": True}

    def _counter_over_once(text: str) -> int:
        counted_texts.append(text)
        if over_once["armed"]:
            over_once["armed"] = False
            return DEFAULT_WRITER_MAX_NUM_CTX  # exact alone is already the whole cap
        return 3300

    out_dir = tmp_path / "ul_over_cap"
    summary = build_and_probe_dataset(
        count=2,
        base_seed=20261030,
        output_dir=out_dir,
        writer_kind="ollama",
        grant_id="test-grant-ul",
        http_transport=_transport,
        start_owned_server=False,
        prompt_token_counter=_counter_over_once,
    )
    assert len(counted_texts) == 4 and len(seen_payloads) == 3  # r0 of scen_0000 refused; r1 and scen_0001 ran
    assert summary["refusals_by_check_id"] == {CTX_OVER_CAP_CHECK_ID: 1}
    assert summary["calls_refused"] == 1 and summary["renderings_refused"] == 1
    assert summary["scenarios_refused"] == 1 and summary["scenarios_kept"] == 1
    assert summary["prompt_token_counter"] == "custom"
    assert summary["writer_pin"]["max_num_ctx"] == 9216
    rows = [json.loads(line) for line in (out_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [r["check_id"] for r in rows] == [CTX_OVER_CAP_CHECK_ID, None, None, None]
    refused_row = rows[0]
    assert refused_row["done_reason"] == CTX_OVER_CAP_CHECK_ID and refused_row["checks_passed"] is False
    assert refused_row["prompt_tokens_exact"] == DEFAULT_WRITER_MAX_NUM_CTX and refused_row["prompt_eval_count"] == 0
    assert refused_row["num_ctx"] == round_up_num_ctx(DEFAULT_WRITER_MAX_NUM_CTX + 4096 + NUM_CTX_MARGIN_TOKENS)
    assert refused_row["refusal_reasons"][0].startswith(CTX_OVER_CAP_CHECK_ID)
    assert all(r["prompt_tokens_exact"] == 3300 for r in rows[1:])
    assert rows[1]["scenario_id"] == w0.scenario_id and rows[2]["scenario_id"] == w1.scenario_id

    # 3. A direct caller that forbids refusal gets the typed error, with the new id, never the old one.
    seen_payloads.clear()
    writer_over = OllamaWriter(
        grant_id="test-grant-ul",
        http_transport=_transport,
        prompt_token_counter=lambda text: DEFAULT_WRITER_MAX_NUM_CTX,
    )
    with pytest.raises(CtxOverCapRefusalError, match="c_ctx_over_cap") as exc_info:
        writer_over.render_scenario_pair(w0)
    assert isinstance(exc_info.value, MaxNumCtxRefusalError)
    assert exc_info.value.check_id == CTX_OVER_CAP_CHECK_ID != "max_num_ctx_exceeded"
    assert not seen_payloads
    assert writer_over.refusals[-1].check_id == CTX_OVER_CAP_CHECK_ID

    # 4. The CLI resolver: both paths or neither; the blob name must carry the pin; the description carries no
    #    path. A live transport without the counter is refused by the writer before any server starts, the way
    #    a missing grant is; a fake transport (these tests) may run on the estimate.
    assert describe_prompt_token_counter(None) is None
    assert resolve_prompt_token_counter(None, None, environ={}) is None
    fake_exe = tmp_path / "llama-tokenize.exe"
    fake_exe.write_bytes(b"")
    with pytest.raises(ValueError, match="both or neither"):
        resolve_prompt_token_counter(fake_exe, None, environ={})
    wrong_blob = tmp_path / ("sha256-" + "0" * 64)
    wrong_blob.write_bytes(b"GGUF")
    with pytest.raises(ValueError, match="TOKENIZER_BLOB_PIN"):
        resolve_prompt_token_counter(fake_exe, wrong_blob, environ={})
    pinned_blob = tmp_path / f"sha256-{DEFAULT_WRITER_MODEL_PIN.model_layer_digest}"
    pinned_blob.write_bytes(b"GGUF")
    counter = resolve_prompt_token_counter(
        None,
        None,
        environ={"UDLAB_LLAMA_TOKENIZE": str(fake_exe), "UDLAB_WRITER_GGUF": str(pinned_blob)},
    )
    assert counter is not None and counter.gguf_digest == DEFAULT_WRITER_MODEL_PIN.model_layer_digest
    assert resolve_prompt_token_counter(fake_exe, pinned_blob, environ={}).gguf_digest == counter.gguf_digest
    described = describe_prompt_token_counter(counter)
    assert isinstance(described, dict) and described["tool"] == "llama-tokenize.exe"
    assert described["args"] == ["--stdin", "-ngl", "0", "--no-escape", "--show-count", "--log-disable"]
    assert described["cuda_visible_devices"] == "-1"
    assert str(tmp_path) not in json.dumps(described)
    with pytest.raises(PermissionError, match="EXACT_COUNT_REQUIRED"):
        OllamaWriter(grant_id="test-grant-ul", http_transport=None, start_owned_server=False)


def test_90_card_ul_qwen35_renderer_matches_ollama_0_40_9() -> None:
    """Test 90 (Card ul): `render_ollama_qwen35_prompt` reproduces Ollama 0.40.9's qwen3.5 renderer for the
    shapes the writer sends (`think` off and on, the multi-turn repair retry) and rejects the shapes it does
    not send."""
    from udlab_generator.render import render_ollama_qwen35_prompt

    single = render_ollama_qwen35_prompt([{"role": "user", "content": "Write it.\n"}], think=False)
    assert single == "<|im_start|>user\nWrite it.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    thinking = render_ollama_qwen35_prompt([{"role": "user", "content": "Write it."}], think=True)
    assert thinking == "<|im_start|>user\nWrite it.<|im_end|>\n<|im_start|>assistant\n<think>\n"
    multi = render_ollama_qwen35_prompt(
        [
            {"role": "user", "content": "Write it."},
            {"role": "assistant", "content": "<think>\nhidden\n</think>\n\nFirst draft.\n"},
            {"role": "user", "content": "Repair line."},
        ],
        think=False,
    )
    assert multi == (
        "<|im_start|>user\nWrite it.<|im_end|>\n"
        "<|im_start|>assistant\nFirst draft.<|im_end|>\n"
        "<|im_start|>user\nRepair line.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    with pytest.raises(ValueError, match="prefill"):
        render_ollama_qwen35_prompt([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}], think=False)
    with pytest.raises(ValueError, match="Unsupported message role"):
        render_ollama_qwen35_prompt([{"role": "system", "content": "a"}, {"role": "user", "content": "b"}], think=False)
    with pytest.raises(ValueError, match="must not be empty"):
        render_ollama_qwen35_prompt([], think=False)


def test_91_card_ul_exactness_against_recorded_prompt_eval_counts() -> None:
    """Test 91 (Card ul, integration): tests/fixtures/exact_ctx/index.json holds, per recorded attempt-0 call,
    the prompt file name, the recorded prompt_sha256 and prompt_eval_count (hashes and counts only). The prompt
    texts are sealed scenario content (the run's world setup and event log) and never live in the repo: they are
    read from the out-of-repo directory named by UDLAB_EXACT_CTX_PROMPTS, regenerated from the sealed run with
    the producing commit's template (card ul reply), and the test skips with a reason when it is unset. Each
    prompt must hash to the recorded prompt_sha256, and, when llama-tokenize and the pinned blob are configured
    (UDLAB_LLAMA_TOKENIZE and UDLAB_WRITER_GGUF), the qwen3.5-rendered prompt tokenises on CPU to exactly the
    recorded prompt_eval_count."""
    from udlab_generator.render import (
        DEFAULT_WRITER_MODEL_PIN,
        LlamaTokenizeCounter,
        count_prompt_tokens_exact,
        render_ollama_qwen35_prompt,
    )

    fixtures_dir = Path(__file__).resolve().parent / "fixtures" / "exact_ctx"
    index = json.loads((fixtures_dir / "index.json").read_text(encoding="utf-8"))
    assert len(index) >= 3
    assert not list(fixtures_dir.glob("*.prompt.txt")), "prompt texts are sealed content and must not be in the repo"
    prompts_dir_raw = os.environ.get("UDLAB_EXACT_CTX_PROMPTS")
    if not prompts_dir_raw:
        pytest.skip("the out-of-repo prompt directory is not configured (UDLAB_EXACT_CTX_PROMPTS)")
    prompts_dir = Path(prompts_dir_raw)
    assert fixtures_dir.resolve() not in prompts_dir.resolve().parents and prompts_dir.resolve() != fixtures_dir.resolve()
    pairs: list[tuple[str, list[dict[str, str]], int]] = []
    for entry in index:
        prompt_path = prompts_dir / entry["file"]
        if not prompt_path.is_file():
            pytest.skip(f"prompt file for {entry['call_id']} not present in UDLAB_EXACT_CTX_PROMPTS")
        content = prompt_path.read_bytes().decode("utf-8")
        messages = [{"role": "user", "content": content}]
        sha = hashlib.sha256(json.dumps(messages, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        assert sha == entry["prompt_sha256"], entry["call_id"]
        assert entry["think"] is False
        rendered = render_ollama_qwen35_prompt(messages, think=False)
        assert rendered.endswith("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        pairs.append((entry["call_id"], messages, int(entry["prompt_eval_count"])))

    exe = os.environ.get("UDLAB_LLAMA_TOKENIZE")
    gguf = os.environ.get("UDLAB_WRITER_GGUF")
    if not exe or not gguf:
        pytest.skip("llama-tokenize and the pinned GGUF blob are not configured (UDLAB_LLAMA_TOKENIZE, UDLAB_WRITER_GGUF)")
    counter = LlamaTokenizeCounter(Path(exe), Path(gguf))
    counter.verify_pinned_digest(DEFAULT_WRITER_MODEL_PIN.model_layer_digest)
    assert counter.count("hello world") == 2  # no BOS for this vocab
    for call_id, messages, recorded in pairs:
        assert count_prompt_tokens_exact(messages, think=False, counter=counter) == recorded, call_id


def test_92_station_projection_pure_and_world_keys_sealed_checkpoints_unchanged(tmp_path: Path) -> None:
    """Test 92 (Card station-projection, Req 1 & 3): station-change Event projects to kind='station_change'
    with actor, room, origin_station, destination_station, and occluder (no target_object/move shape); every
    non-station Event projects unchanged; World, world.events, keys, sealed_store, and checkpoints are not
    mutated and their sha256s for a fixed seed are byte-identical before and after projection; prompt token
    estimate does not grow."""
    from dataclasses import asdict
    from udlab_generator.render import (
        STATION_CHANGE_EVENT_KIND,
        RenderedScenario,
        WriterCheckpointManager,
        estimate_prompt_tokens,
        format_station_change_clause,
        format_writer_prompt,
        load_writer_prompt_template,
        project_event_for_writer,
        project_events_for_writer,
        render_world_template,
    )
    from udlab_generator.sealing import _serialize_world, write_sealed_dataset
    from udlab_generator.world import is_station_transition_event

    w = generate_world(make_balanced_configs(1, base_seed=20261033)[0])
    fa0, fa1 = w.agents[0], w.agents[1]
    prose_0 = render_world_template(w, fa0)
    prose_1 = render_world_template(w, fa1)
    keys_before = compute_scenario_keys(w)

    def _canonical_sha(obj: object) -> str:
        return hashlib.sha256(
            json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    world_sha_before = _canonical_sha(_serialize_world(w))
    events_sha_before = _canonical_sha([asdict(e) for e in w.events])
    keys_sha_before = _canonical_sha(asdict(keys_before))

    priv_seed = "cd" * 32
    prov = {"writer": "mock"}
    pair_before = (
        RenderedScenario(w.scenario_id, 0, fa0, prose_0, "pre", keys_before, world=w),
        RenderedScenario(w.scenario_id, 1, fa1, prose_1, "pre", keys_before, world=w),
    )
    dir_before = tmp_path / "before_proj"
    receipt_before = write_sealed_dataset(
        rendered_pairs=[pair_before],
        public_dir=dir_before / "part_public",
        sealed_dir=dir_before / "sealed_store",
        private_seed_hex=priv_seed,
        writer_provenance=prov,
        all_worlds=[w],
    )
    ckpt_before = WriterCheckpointManager(
        output_dir=dir_before,
        count=1,
        base_seed=20261033,
        private_seed_hex=priv_seed,
        writer_kind="mock",
    )
    ckpt_before.initialize_fresh()
    ckpt_before.append_scenario(scenario_index=0, pair=pair_before, call_records=(), world=w)
    ckpt_pub_sha_before = hashlib.sha256(ckpt_before.public_path.read_bytes()).hexdigest()
    ckpt_seal_sha_before = hashlib.sha256(ckpt_before.sealed_path.read_bytes()).hexdigest()
    sealing_meta_sha_before = hashlib.sha256(
        (dir_before / "sealed_store" / "sealing.sealed.json").read_bytes()
    ).hexdigest()

    # Run projection and format_writer_prompt:
    projected = project_events_for_writer(w.events, world=w)
    assert len(projected) == len(w.events)
    station_count = 0
    for idx, (ev, p_dict) in enumerate(zip(w.events, projected, strict=True), start=1):
        assert project_event_for_writer(ev, order_index=idx, world=w) == p_dict
        if is_station_transition_event(ev, w):
            station_count += 1
            assert p_dict == {
                "order_index": idx,
                "kind": STATION_CHANGE_EVENT_KIND,
                "actor": ev.actor,
                "room": ev.room,
                "origin_station": ev.from_location,
                "destination_station": ev.to_location,
                "occluder": ev.target_object,
            }
            assert "target_object" not in p_dict
            assert p_dict["kind"] != "move"
        else:
            assert p_dict["kind"] == ev.kind
            assert "origin_station" not in p_dict and "destination_station" not in p_dict
            if ev.target_object is not None:
                assert p_dict["target_object"] == ev.target_object
    assert station_count == 3

    prompt_0 = format_writer_prompt(w, fa0)
    prompt_1 = format_writer_prompt(w, fa1)
    assert f'"kind":"{STATION_CHANGE_EVENT_KIND}"' in prompt_0
    for occ in w.occluders_by_room.values():
        assert f'"target_object":"{occ}"' not in prompt_0

    # Purity check: World, events, keys, sealed_store, and checkpoints are byte-identical after projection:
    assert _canonical_sha(_serialize_world(w)) == world_sha_before
    assert _canonical_sha([asdict(e) for e in w.events]) == events_sha_before
    keys_after = compute_scenario_keys(w)
    assert _canonical_sha(asdict(keys_after)) == keys_sha_before

    pair_after = (
        RenderedScenario(w.scenario_id, 0, fa0, prose_0, prompt_0, keys_after, world=w),
        RenderedScenario(w.scenario_id, 1, fa1, prose_1, prompt_1, keys_after, world=w),
    )
    dir_after = tmp_path / "after_proj"
    receipt_after = write_sealed_dataset(
        rendered_pairs=[pair_after],
        public_dir=dir_after / "part_public",
        sealed_dir=dir_after / "sealed_store",
        private_seed_hex=priv_seed,
        writer_provenance=prov,
        all_worlds=[w],
    )
    assert receipt_after.items_sha256 == receipt_before.items_sha256
    assert receipt_after.part_json_sha256 == receipt_before.part_json_sha256
    assert receipt_after.keys_sealed_sha256 == receipt_before.keys_sealed_sha256
    assert (
        hashlib.sha256((dir_after / "sealed_store" / "sealing.sealed.json").read_bytes()).hexdigest()
        == sealing_meta_sha_before
    )

    ckpt_after = WriterCheckpointManager(
        output_dir=dir_after,
        count=1,
        base_seed=20261033,
        private_seed_hex=priv_seed,
        writer_kind="mock",
    )
    ckpt_after.initialize_fresh()
    ckpt_after.append_scenario(scenario_index=0, pair=pair_after, call_records=(), world=w)
    assert hashlib.sha256(ckpt_after.public_path.read_bytes()).hexdigest() == ckpt_pub_sha_before
    assert hashlib.sha256(ckpt_after.sealed_path.read_bytes()).hexdigest() == ckpt_seal_sha_before

    # Requirement 3: positive station_change clause, no negative 'never that the actor moved' warning, no token growth.
    clause = format_station_change_clause()
    assert clause.startswith("on station_change, write ")
    assert "never that the actor moved" not in clause
    tmpl = load_writer_prompt_template()
    assert len(tmpl.encode("utf-8")) <= 6577
    assert estimate_prompt_tokens(tmpl) <= 2193


def test_93_station_projection_reconstruct_world_round_trip_and_cli_verifiers(tmp_path: Path) -> None:
    """Test 93 (Card station-projection, Req 2 & 6): _reconstruct_world_from_writer_prompt(world,
    format_writer_prompt(world, ...)) equals world across multiple seeds containing station changes;
    --recheck and --verify-written pass on a mock-writer run directory."""
    from udlab_generator.cli import build_and_probe_dataset, recheck_saved_responses, verify_written_dataset
    from udlab_generator.render import _reconstruct_world_from_writer_prompt, format_writer_prompt
    from udlab_generator.world import is_station_transition_event

    configs = make_balanced_configs(12, base_seed=20261033)
    total_station_changes = 0
    for cfg in configs:
        w = generate_world(cfg)
        sc = sum(1 for ev in w.events if is_station_transition_event(ev, w))
        assert sc >= 1
        total_station_changes += sc
        for fa in w.agents[:2]:
            prompt = format_writer_prompt(w, fa)
            reconstructed = _reconstruct_world_from_writer_prompt(w, prompt)
            assert reconstructed == w
            assert reconstructed.events == w.events
            assert compute_scenario_keys(reconstructed) == compute_scenario_keys(w)
    assert total_station_changes >= 20

    run_dir = tmp_path / "mock_run_seed"
    summary = build_and_probe_dataset(
        count=10,
        base_seed=20261030,
        output_dir=run_dir,
        writer_kind="mock",
    )
    assert summary["scenarios_kept"] == 10 and summary["renderings_kept"] == 20
    assert "fault_vector_counts_by_check_id" in summary
    fault_table = summary["fault_vector_counts_by_check_id"]
    assert isinstance(fault_table, dict) and all(v == 0 for v in fault_table.values())

    recheck_res = recheck_saved_responses(run_dir)
    assert recheck_res["matches_recorded_verdicts"] is True
    assert recheck_res["fault_vector_counts_by_check_id"] == fault_table

    verify_res = verify_written_dataset(run_dir)
    assert verify_res["passed"] is True


def test_94_station_change_checker_occluder_manipulation_refused_and_person_verbs_accepted() -> None:
    """Test 94 (Card station-projection, Req 4 & 6): every verb in OCCLUDER_MANIPULATION_VERBS with the
    occluder as object is refused on a station-change event (with or without behind/around and with or
    without valid stations); every verb in STATION_CHANGE_PERSON_VERBS is accepted on a valid station-change
    sentence and refused when the station or direction word is mutated; verify_prompt_and_checker_requirements_aligned
    passes."""
    from udlab_generator.render import (
        OCCLUDER_MANIPULATION_VERBS,
        STATION_CHANGE_PERSON_VERBS,
        _sentence_matches_event,
        _split_prose_sentences,
        check_rendered_prose,
        render_world_template,
        verify_prompt_and_checker_requirements_aligned,
    )
    from udlab_generator.world import is_station_transition_event

    verify_prompt_and_checker_requirements_aligned()

    w = generate_world(make_balanced_configs(1, base_seed=20261033)[0])
    fa = w.agents[0]
    setup_para, events_para = render_world_template(w, fa).split("\n\n")
    sents = _split_prose_sentences(events_para)
    sm, back = w.events[6], w.events[10]
    assert is_station_transition_event(sm, w) and is_station_transition_event(back, w)
    a, occ, room = sm.actor, sm.target_object, sm.room

    # 1. Every verb in OCCLUDER_MANIPULATION_VERBS with the occluder as direct object is refused on both
    #    station-change events (with or without behind/around, with or without valid stations, with adverb).
    for verb in OCCLUDER_MANIPULATION_VERBS:
        for ev in (sm, back):
            f_loc, t_loc = ev.from_location, ev.to_location
            candidates = (
                f"In the {room}, {a} {verb} the {occ} from the {f_loc} to the {t_loc}.",
                f"In the {room}, {a} {verb} the {occ} behind him from the {f_loc} to the {t_loc}.",
                f"In the {room}, {a} {verb} the {occ} back around her from the {f_loc} to the {t_loc}.",
                f"In the {room}, {a} {verb} quickly the {occ} behind him to the {t_loc}.",
                f"In the {room}, {a} {verb} the {occ} to the {t_loc}.",
                f"In the {room}, {a} {verb} the {occ}.",
            )
            for cand in candidates:
                assert not _sentence_matches_event(cand, ev, w.noise_threshold, world=w), (verb, cand)

    # 2. Every verb in STATION_CHANGE_PERSON_VERBS is accepted on valid station-change sentences and in full
    #    check_rendered_prose, and refused when the station or direction word is mutated.
    for verb in STATION_CHANGE_PERSON_VERBS:
        valid_sm_1 = f"In the {room}, {a} {verb} behind the {occ} to the far-side alcove."
        valid_sm_2 = f"In the {room}, {a} {verb} from the container bay behind the {occ} to the far-side alcove."
        valid_back_1 = f"In the {room}, {a} {verb} back around the {occ} to the container bay."
        valid_back_2 = f"In the {room}, {a} {verb} around the {occ} from the far-side alcove to the container bay."
        assert _sentence_matches_event(valid_sm_1, sm, w.noise_threshold, world=w), (verb, valid_sm_1)
        assert _sentence_matches_event(valid_sm_2, sm, w.noise_threshold, world=w), (verb, valid_sm_2)
        assert _sentence_matches_event(valid_back_1, back, w.noise_threshold, world=w), (verb, valid_back_1)
        assert _sentence_matches_event(valid_back_2, back, w.noise_threshold, world=w), (verb, valid_back_2)

        patched = list(sents)
        patched[6] = valid_sm_1
        patched[10] = valid_back_1
        patched[12] = valid_sm_2
        assert check_rendered_prose(f"{setup_para}\n\n{' '.join(patched)}", w, focal_agent=fa) == ()

        # Mutated destination station or missing/mutated direction word is refused:
        wrong_station_sm = f"In the {room}, {a} {verb} behind the {occ} to the container bay."
        wrong_station_back = f"In the {room}, {a} {verb} back around the {occ} to the far-side alcove."
        no_dir_sm = f"In the {room}, {a} {verb} the {occ} to the far-side alcove."
        wrong_dir_sm = f"In the {room}, {a} {verb} beside the {occ} to the far-side alcove."
        no_occ_sm = f"In the {room}, {a} {verb} behind to the far-side alcove."
        for bad_cand, ev in (
            (wrong_station_sm, sm),
            (wrong_station_back, back),
            (no_dir_sm, sm),
            (wrong_dir_sm, sm),
            (no_occ_sm, sm),
        ):
            assert not _sentence_matches_event(bad_cand, ev, w.noise_threshold, world=w), (verb, bad_cand)


def test_95_collect_all_fault_vectors_and_structural_no_text_guard(tmp_path: Path) -> None:
    """Test 95 (Card station-projection, Req 5 & 6): a synthetic rendering violating three+ distinct checks
    records all of them in fault_check_ids and collect_prose_faults / check_rendered_prose(..., collect_all=True)
    while RenderCheckError still reports the first; verdicts and first-failure tuples on a test corpus match
    with and without collect-all; structural guard ensures fault-vector fields hold only known check_ids,
    integer indices, and allowed event_kind/failure_kind enums (never free text)."""
    from udlab_generator.cli import build_and_probe_dataset, recheck_saved_responses
    from udlab_generator.render import (
         DEFAULT_WRITER_MODEL_PIN,
        ProseFaultRecord,
        VALID_FAILURE_KINDS,
        VALID_FAULT_CHECK_IDS,
        VALID_FAULT_EVENT_KINDS,
        _split_prose_sentences,
        check_rendered_prose,
        collect_prose_faults,
        render_world_template,
        summarize_fault_vector_counts,
    )
    from udlab_generator.world import BELIEF_PERSISTENCE_RULE, OCCLUSION_RULE

    w = generate_world(make_balanced_configs(1, base_seed=20261033)[0])
    fa = w.agents[0]
    good_prose = render_world_template(w, fa)
    assert check_rendered_prose(good_prose, w, focal_agent=fa) == ()
    assert check_rendered_prose(good_prose, w, focal_agent=fa, collect_all=True) == ()
    assert collect_prose_faults(good_prose, w, focal_agent=fa) == ()

    # 1. Multi-fault rendering violating 4 distinct checks:
    #    - b_banned_adverbs ("secretly")
    #    - a_no_markers_bracket ("[F0]")
    #    - d_rule_sentence (omitted BELIEF_PERSISTENCE_RULE)
    #    - c_fidelity_missing_event (first event sentence dropped)
    setup_para, events_para = good_prose.split("\n\n")
    ev_sents = _split_prose_sentences(events_para)
    multi_bad_setup = setup_para.replace(BELIEF_PERSISTENCE_RULE, "") + " [F0] Everyone waited secretly."
    multi_bad_prose = f"{multi_bad_setup}\n\n{' '.join(ev_sents[1:])}"

    with pytest.raises(RenderCheckError) as exc_first:
        check_rendered_prose(multi_bad_prose, w, focal_agent=fa)
    assert exc_first.value.check_id == "b_banned_adverbs"

    fault_vec = check_rendered_prose(multi_bad_prose, w, focal_agent=fa, collect_all=True)
    assert fault_vec == collect_prose_faults(multi_bad_prose, w, focal_agent=fa)
    fault_ids = [f.check_id for f in fault_vec]
    assert fault_ids[0] == "b_banned_adverbs"
    for expected_cid in (
        "b_banned_adverbs",
        "a_no_markers_bracket",
        "a_no_markers_fact_id",
        "d_rule_sentence",
        "c_fidelity_missing_event",
    ):
        assert expected_cid in fault_ids

    # 2. Verify build_summary.json, generations.jsonl, and --recheck produce identical fault_vector_counts_by_check_id:
    pinned_digest = DEFAULT_WRITER_MODEL_PIN.manifest_digest

    def _multi_fault_transport(method: str, url: str, body: dict | None = None) -> dict:
        if method == "GET" and url.endswith("/api/version"):
            return {"version": "0.40.9-test"}
        if method == "GET" and url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3.6:35b", "model": "qwen3.6:35b", "digest": f"sha256:{pinned_digest}"}]}
        if method == "POST" and url.endswith("/api/chat"):
            payload = dict(body or {})
            if payload.get("keep_alive") == 0:
                return {"model": "qwen3.6:35b", "done": True, "done_reason": "unload"}
            return {
                "model": "qwen3.6:35b",
                "message": {"role": "assistant", "content": multi_bad_prose},
                "done": True,
                "done_reason": "stop",
                "eval_count": 300,
                "prompt_eval_count": 3000,
            }
        raise AssertionError(f"Unexpected {method} {url}")

    out_dir = tmp_path / "fault_vec_run"
    summary = build_and_probe_dataset(
        count=1,
        base_seed=20261033,
        output_dir=out_dir,
        writer_kind="ollama",
        grant_id="test-grant-fv",
        http_transport=_multi_fault_transport,
        start_owned_server=False,
    )
    fv_counts = summary["fault_vector_counts_by_check_id"]
    assert isinstance(fv_counts, dict)
    assert summary["refusals_by_check_id"] == {"b_banned_adverbs": 2}
    assert fv_counts["b_banned_adverbs"] == 2
    assert fv_counts["a_no_markers_bracket"] == 2
    assert fv_counts["d_rule_sentence"] == 2
    assert fv_counts["c_fidelity_missing_event"] == 2

    gen_rows = [
        json.loads(line)
        for line in (out_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(gen_rows) == 2
    for row in gen_rows:
        assert row["check_id"] == "b_banned_adverbs"
        assert "b_banned_adverbs" in row["fault_check_ids"]
        assert "d_rule_sentence" in row["fault_check_ids"]
        assert "c_fidelity_missing_event" in row["fault_check_ids"]

    recheck_out = recheck_saved_responses(out_dir)
    assert recheck_out["matches_recorded_verdicts"] is True
    assert recheck_out["fault_vector_counts_by_check_id"] == fv_counts
    for c_entry in recheck_out["calls"]:
        assert c_entry["fault_check_ids"] == gen_rows[0]["fault_check_ids"]
        for f_dict in c_entry["fault_vector"]:
            assert set(f_dict.keys()) == {"check_id", "event_index", "event_kind", "failure_kind"}

    # 3. Verdicts and first-failure (check_id, event_index, event_kind, failure_kind) on corpus match with and without collect-all:
    corpus_samples: list[tuple[str, World, str]] = [
        (good_prose, w, fa),
        (good_prose + " [step 1]", w, fa),
        (good_prose.replace("In the", "Secretly, in the", 1), w, fa),
        (good_prose.replace(BELIEF_PERSISTENCE_RULE, ""), w, fa),
        (good_prose.replace(f"{w.noise_threshold} dB", ""), w, fa),
        (f"{setup_para}\n\n{' '.join(ev_sents[:-1])}", w, fa),
        (f"{setup_para}\n\n{' '.join([ev_sents[1], ev_sents[0]] + ev_sents[2:])}", w, fa),
        (multi_bad_prose, w, fa),
    ]
    for sample_prose, sample_w, sample_fa in corpus_samples:
        collected = check_rendered_prose(sample_prose, sample_w, focal_agent=sample_fa, collect_all=True)
        try:
            res_normal = check_rendered_prose(sample_prose, sample_w, focal_agent=sample_fa)
            assert res_normal == ()
            assert collected == ()
        except RenderCheckError as exc_norm:
            assert len(collected) >= 1
            first_fault = collected[0]
            assert (
                first_fault.check_id,
                first_fault.event_index,
                first_fault.event_kind,
                first_fault.failure_kind,
            ) == (
                exc_norm.check_id,
                exc_norm.event_index,
                exc_norm.event_kind,
                exc_norm.failure_kind,
            )

    # 4. Structural test: fault-vector fields can only hold known check_id strings, integer indices, and
    #    allowed event_kind/failure_kind enums (never free text).
    valid_rec = ProseFaultRecord(
        check_id="c_fidelity_missing_event",
        event_index=3,
        event_kind="move",
        failure_kind="altered",
    )
    assert valid_rec.check_id in VALID_FAULT_CHECK_IDS
    assert valid_rec.event_kind in VALID_FAULT_EVENT_KINDS
    assert valid_rec.failure_kind in VALID_FAILURE_KINDS
    assert set(valid_rec.to_dict().keys()) == {"check_id", "event_index", "event_kind", "failure_kind"}

    with pytest.raises(ValueError, match="registered check_id"):
        ProseFaultRecord(check_id="In the atrium, Ben moved the partition.")
    with pytest.raises(ValueError, match="event_index"):
        ProseFaultRecord(check_id="b_banned_adverbs", event_index=0)
    with pytest.raises(ValueError, match="event_index"):
        ProseFaultRecord(check_id="b_banned_adverbs", event_index=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="event_kind"):
        ProseFaultRecord(check_id="c_fidelity_missing_event", event_index=1, event_kind="free prose text")
    with pytest.raises(ValueError, match="failure_kind"):
        ProseFaultRecord(check_id="c_fidelity_missing_event", event_index=1, event_kind="move", failure_kind="sentence text")
    with pytest.raises(ValueError, match="registered check_ids"):
        summarize_fault_vector_counts([["free text leak"]])


def test_96_retry_sampling_single_turn_derived_seeds_kept_items_and_short_circuit(tmp_path: Path) -> None:
    """Test 96 (Card udlab-retry-sampling, Req 1-7):
    1. Attempt 0 identical to `main` apart from `options["seed"]` (fake transport proving `messages` and `options`
       are byte-identical to `main` except `seed`, with `"temperature": 0` and no `"min_p"`).
    2. Retries single-turn by default, without the refused draft, using the rewritten rule note (`check_id`,
       `event_index` and `event_kind` where present, and what must hold; no `'keep your earlier text'`, no model
       text, no literal checker regex), with retry options (`temperature == RETRY_TEMPERATURE == 0.6`,
       `min_p == RETRY_MIN_P == 0.05`, `seed == derive_call_seed(scenario_seed, rendering_index, attempt)`),
       and keeping card-`ul` exact context-cap sizing when `prompt_token_counter` is configured.
    3. Derived seeds (`derive_call_seed(scenario_seed, rendering_index, attempt)`) distinct across renderings and
       attempts and stable across runs.
    4. The multi-turn option (`retry_form='multi_turn'` / `multi_turn_retries=True`) still works when chosen.
    5. Recorded fields (`attempt`, `temperature`, `min_p`, `seed`, `retry_form`) on rows (`generations.jsonl`,
       `sealed_store/responses.sealed.jsonl`) and (`attempt`, `temperature`, `min_p`, `seed`, `sampling`) on kept
       items (`RenderedScenario`, `part_public/items.jsonl`, `sealed_store/keys.sealed.jsonl`, `checkpoint.json`,
       `sealed_store/checkpoint.sealed.json`), with no evaluation-split filter in code.
    6. Short-circuit switch (`--short-circuit-dead-scenarios` / `short_circuit_dead_scenarios`): default off
       (rendering 1 still written after a dead rendering 0) and on (scenario stops after rendering 0 fails all
       attempts), and the run summary and run records show the mode.
    7. No model text in any new field.
    """
    from typing import Any
    from udlab_generator import (
        ATTEMPT_0_TEMPERATURE,
        CALL_SEED_DOMAIN_TAG,
        CALL_SEED_MASK_31BIT,
        DEFAULT_RETRY_FORM,
        PINNED_WRITER_MANIFEST_DIGEST,
        PINNED_WRITER_MODEL_TAG,
        RETRY_MIN_P,
        RETRY_TEMPERATURE,
        VALID_RETRY_FORMS,
        derive_call_seed,
    )
    from udlab_generator.cli import build_and_probe_dataset
    from udlab_generator.render import (
        DEFAULT_WRITER_MAX_NUM_CTX,
        DEFAULT_WRITER_NUM_PREDICT,
        NUM_CTX_MARGIN_TOKENS,
        NUM_CTX_STEP_TOKENS,
        OllamaWriter,
        WRITER_PROSE_REQUIREMENTS,
        decide_num_ctx,
        estimate_prompt_tokens,
        format_repair_retry_line,
        format_writer_prompt,
        load_writer_prompt_template,
        render_ollama_qwen35_prompt,
    )

    # --- Requirement 3: Derived seeds distinct across renderings & attempts and stable across runs ---
    assert DEFAULT_RETRY_FORM == "single_turn"
    assert VALID_RETRY_FORMS == frozenset({"single_turn", "multi_turn"})
    assert ATTEMPT_0_TEMPERATURE == 0.0
    assert RETRY_TEMPERATURE == 0.6
    assert RETRY_MIN_P == 0.05
    assert CALL_SEED_DOMAIN_TAG == "udlab.call_seed.v1"
    assert CALL_SEED_MASK_31BIT == 0x7FFFFFFF

    w0 = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    w1 = generate_world(make_balanced_configs(2, base_seed=20261030)[1])
    s0 = int(w0.seed)

    # Pin exact deterministic values so stability across runs is verified:
    expected_s0_r0_a0 = (
        int.from_bytes(
            hashlib.blake2b(f"udlab.call_seed.v1:{s0}:0:0".encode("utf-8"), digest_size=8).digest()[:4],
            "big",
        )
        & 0x7FFFFFFF
    )
    assert derive_call_seed(s0, 0, 0) == expected_s0_r0_a0
    assert derive_call_seed(20261030, 0, 0) == 454800033
    assert derive_call_seed(20261030, 1, 0) == 2099905771
    assert derive_call_seed(20261030, 0, 1) == 109349287
    assert derive_call_seed(20261030, 1, 1) == 293191637

    grid_seeds = [
        derive_call_seed(scen_s, r_idx, att)
        for scen_s in (int(w0.seed), int(w1.seed), 5, 42)
        for r_idx in (0, 1)
        for att in range(6)
    ]
    assert len(set(grid_seeds)) == len(grid_seeds), "Every (scenario_seed, rendering_index, attempt) seed must be distinct"
    assert all(0 <= sd <= CALL_SEED_MASK_31BIT for sd in grid_seeds)
    with pytest.raises(ValueError, match="rendering_index"):
        derive_call_seed(s0, 2, 0)
    with pytest.raises(ValueError, match="attempt"):
        derive_call_seed(s0, 0, -1)

    # --- Requirement 1: Attempt 0 byte-identical to `main` apart from `options["seed"]` ---
    passing_r0 = render_world_template(w0, w0.agents[0])
    passing_r1 = render_world_template(w0, w0.agents[1])
    attempt0_payloads: list[dict[str, Any]] = []

    def _attempt0_transport(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            assert payload is not None
            idx = len(attempt0_payloads)
            attempt0_payloads.append(json.loads(json.dumps(payload)))
            return {
                "message": {"role": "assistant", "content": passing_r0 if idx == 0 else passing_r1},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3100,
                "eval_count": 780,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    writer_a0 = OllamaWriter(
        grant_id="test-grant-req1",
        http_transport=_attempt0_transport,
        start_owned_server=False,
    )
    pair_a0 = writer_a0.render_scenario_pair(w0)
    assert pair_a0 is not None and len(attempt0_payloads) == 2
    tmpl = load_writer_prompt_template()
    for r_idx, got_payload in enumerate(attempt0_payloads):
        expected_prompt = format_writer_prompt(w0, w0.agents[r_idx], tmpl)
        expected_messages = [{"role": "user", "content": expected_prompt}]
        expected_num_ctx = decide_num_ctx(
            estimate_prompt_tokens(expected_messages),
            DEFAULT_WRITER_NUM_PREDICT,
            max_num_ctx=DEFAULT_WRITER_MAX_NUM_CTX,
            margin=NUM_CTX_MARGIN_TOKENS,
            step=NUM_CTX_STEP_TOKENS,
        )
        main_payload = {
            "model": PINNED_WRITER_MODEL_TAG,
            "messages": expected_messages,
            "think": False,
            "stream": False,
            "options": {
                "temperature": 0,
                "seed": int(w0.seed),
                "num_predict": DEFAULT_WRITER_NUM_PREDICT,
                "num_ctx": expected_num_ctx,
            },
        }
        assert got_payload["options"]["seed"] == derive_call_seed(int(w0.seed), r_idx, 0)
        assert got_payload["options"]["seed"] != int(w0.seed)
        assert "min_p" not in got_payload["options"]
        assert got_payload["options"]["temperature"] == 0
        # Byte-identical to `main` once `options["seed"]` is replaced by `int(w0.seed)`:
        got_with_main_seed = json.loads(json.dumps(got_payload))
        got_with_main_seed["options"]["seed"] = int(w0.seed)
        assert json.dumps(got_with_main_seed, sort_keys=True, separators=(",", ":")) == json.dumps(
            main_payload, sort_keys=True, separators=(",", ":")
        )

    # --- Requirement 2 & 7: Single-turn retries by default, rewritten rule note, retry options, exact counter, no model text ---
    draft_secret_marker = "REFUSED_DRAFT_MODEL_PROSE_SECRET_MARKER_ALPHA"
    orig_stake = w0.stake_edges[0].description
    bad_stake = (
        f"Gia is an absent party ({draft_secret_marker}) who depends on the location of the index reel "
        "for tomorrow's courier handoff."
    )
    failing_r0_draft = passing_r0.replace(orig_stake, bad_stake)

    # Check every check_id and event variant in format_repair_retry_line for single-turn rules:
    for req in WRITER_PROSE_REQUIREMENTS:
        for fk in ("absent", "altered", "false"):
            exc_s = RenderCheckError(
                "reason",
                check_id=req.check_id,
                failing_sentence=draft_secret_marker,
                failure_kind=fk,
            )
            note_s = format_repair_retry_line(exc_s)
            assert note_s.startswith(f"Repair note (rule broken: check {req.check_id}, {fk}): ")
            assert "keep your earlier text" not in note_s.lower()
            assert "earlier text" not in note_s.lower()
            assert draft_secret_marker not in note_s
            assert "\\b" not in note_s and "(?:" not in note_s
    for fk in ("absent", "altered", "false"):
        for variant in (None, "station_move", "occluded_see"):
            exc_e = RenderCheckError(
                "reason",
                check_id="c_fidelity_missing_event",
                failing_sentence=draft_secret_marker,
                event_index=12,
                event_kind="move" if variant != "occluded_see" else "see",
                failure_kind=fk,
                event_variant=variant,
            )
            note_e = format_repair_retry_line(exc_e)
            ev_k_expected = "move" if variant != "occluded_see" else "see"
            assert note_e.startswith(
                f"Repair note (rule broken: event 12 ({ev_k_expected}, check c_fidelity_missing_event), {fk}): "
            )
            assert "keep your earlier text" not in note_e.lower()
            assert "earlier text" not in note_e.lower()
            assert draft_secret_marker not in note_e
            assert "\\b" not in note_e and "(?:" not in note_e

    single_turn_payloads: list[dict[str, Any]] = []
    counted_prompts: list[str] = []

    def _exact_counter(rendered_text: str) -> int:
        counted_prompts.append(rendered_text)
        # Return 3200 on attempt 0 and 3260 on the single-turn retry (which adds the repair line)
        return 3260 if "Repair note (rule broken:" in rendered_text else 3200

    def _single_turn_retry_transport(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            assert payload is not None
            idx = len(single_turn_payloads)
            single_turn_payloads.append(json.loads(json.dumps(payload)))
            # Call 0: r0 attempt 0 fails; Call 1: r0 attempt 1 passes; Call 2: r1 attempt 0 passes
            txt = failing_r0_draft if idx == 0 else (passing_r0 if idx == 1 else passing_r1)
            return {
                "message": {"role": "assistant", "content": txt},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3200 if idx != 1 else 3260,
                "eval_count": 790,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    out_st = tmp_path / "single_turn_default_run"
    summary_st = build_and_probe_dataset(
        count=1,
        base_seed=20261030,
        output_dir=out_st,
        writer_kind="ollama",
        max_retries=2,
        grant_id="test-grant-req2",
        http_transport=_single_turn_retry_transport,
        start_owned_server=False,
        prompt_token_counter=_exact_counter,
    )
    assert len(single_turn_payloads) == 3
    # Call 0 (r0 attempt 0):
    assert single_turn_payloads[0]["options"] == {
        "temperature": 0,
        "seed": derive_call_seed(int(w0.seed), 0, 0),
        "num_predict": DEFAULT_WRITER_NUM_PREDICT,
        "num_ctx": decide_num_ctx(3200, DEFAULT_WRITER_NUM_PREDICT, max_num_ctx=DEFAULT_WRITER_MAX_NUM_CTX),
    }
    # Call 1 (r0 attempt 1 - single-turn retry):
    retry_msgs = single_turn_payloads[1]["messages"]
    assert len(retry_msgs) == 1
    assert retry_msgs[0]["role"] == "user"
    assert draft_secret_marker not in json.dumps(retry_msgs)
    assert failing_r0_draft not in json.dumps(retry_msgs)
    assert "Repair note (rule broken: check c_fidelity_stake_edge, altered):" in retry_msgs[0]["content"]
    assert "keep your earlier text" not in retry_msgs[0]["content"].lower()
    assert single_turn_payloads[1]["options"] == {
        "temperature": RETRY_TEMPERATURE,
        "min_p": RETRY_MIN_P,
        "seed": derive_call_seed(int(w0.seed), 0, 1),
        "num_predict": DEFAULT_WRITER_NUM_PREDICT,
        "num_ctx": decide_num_ctx(3260, DEFAULT_WRITER_NUM_PREDICT, max_num_ctx=DEFAULT_WRITER_MAX_NUM_CTX),
    }
    # Verify exact tokenizer was invoked on the single-turn retry messages as well:
    assert render_ollama_qwen35_prompt(retry_msgs, think=False) in counted_prompts

    # --- Requirement 5 & 7: Recorded fields on rows and kept items; no evaluation-split filter in code; no model text in new fields ---
    gen_rows_st = [
        json.loads(line)
        for line in (out_st / "generations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(gen_rows_st) == 3
    assert gen_rows_st[0]["attempt"] == 0
    assert gen_rows_st[0]["temperature"] == 0.0
    assert gen_rows_st[0]["min_p"] is None
    assert gen_rows_st[0]["seed"] == derive_call_seed(int(w0.seed), 0, 0)
    assert gen_rows_st[0]["retry_form"] == ""
    assert gen_rows_st[0]["prompt_tokens_exact"] == 3200

    assert gen_rows_st[1]["attempt"] == 1
    assert gen_rows_st[1]["temperature"] == RETRY_TEMPERATURE
    assert gen_rows_st[1]["min_p"] == RETRY_MIN_P
    assert gen_rows_st[1]["seed"] == derive_call_seed(int(w0.seed), 0, 1)
    assert gen_rows_st[1]["retry_form"] == "single_turn"
    assert gen_rows_st[1]["prompt_tokens_exact"] == 3260
    assert draft_secret_marker not in json.dumps(gen_rows_st[1])

    assert gen_rows_st[2]["attempt"] == 0
    assert gen_rows_st[2]["temperature"] == 0.0
    assert gen_rows_st[2]["min_p"] is None
    assert gen_rows_st[2]["seed"] == derive_call_seed(int(w0.seed), 1, 0)
    assert gen_rows_st[2]["retry_form"] == ""

    # Check kept items in part_public/items.jsonl and sealed_store/keys.sealed.jsonl:
    # Both r0 (passed on attempt 1) and r1 (passed on attempt 0) are in the outputs (no evaluation-split filter in code),
    # and each carries its attempt, temperature, min_p, seed, and sampling dict.
    pub_items = [
        json.loads(line)
        for line in (out_st / "part_public" / "items.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sealed_items = [
        json.loads(line)
        for line in (out_st / "sealed_store" / "keys.sealed.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(pub_items) == 2 and len(sealed_items) == 2
    by_r_idx = {row["rendering_index"]: row for row in sealed_items}
    assert by_r_idx[0]["attempt"] == 1
    assert by_r_idx[0]["temperature"] == RETRY_TEMPERATURE
    assert by_r_idx[0]["min_p"] == RETRY_MIN_P
    assert by_r_idx[0]["seed"] == derive_call_seed(int(w0.seed), 0, 1)
    assert by_r_idx[0]["sampling"] == {
        "attempt": 1,
        "temperature": RETRY_TEMPERATURE,
        "min_p": RETRY_MIN_P,
        "seed": derive_call_seed(int(w0.seed), 0, 1),
    }
    assert by_r_idx[1]["attempt"] == 0
    assert by_r_idx[1]["temperature"] == 0.0
    assert by_r_idx[1]["min_p"] is None
    assert by_r_idx[1]["seed"] == derive_call_seed(int(w0.seed), 1, 0)
    assert by_r_idx[1]["sampling"] == {
        "attempt": 0,
        "temperature": 0.0,
        "min_p": None,
        "seed": derive_call_seed(int(w0.seed), 1, 0),
    }
    pub_by_focal = {row["focal_agent"]: row for row in pub_items}
    assert pub_by_focal[w0.agents[0]]["attempt"] == 1
    assert pub_by_focal[w0.agents[0]]["temperature"] == RETRY_TEMPERATURE
    assert pub_by_focal[w0.agents[0]]["min_p"] == RETRY_MIN_P
    assert pub_by_focal[w0.agents[0]]["seed"] == derive_call_seed(int(w0.seed), 0, 1)
    assert pub_by_focal[w0.agents[0]]["sampling"] == by_r_idx[0]["sampling"]
    assert pub_by_focal[w0.agents[1]]["attempt"] == 0
    assert pub_by_focal[w0.agents[1]]["sampling"] == by_r_idx[1]["sampling"]

    # Check checkpoint.json and checkpoint.sealed.json:
    ckpt_pub = json.loads((out_st / "checkpoint.json").read_text(encoding="utf-8"))
    ckpt_sealed = json.loads((out_st / "sealed_store" / "checkpoint.sealed.json").read_text(encoding="utf-8"))
    assert ckpt_pub["header"]["short_circuit_dead_scenarios"] is False
    assert ckpt_pub["header"]["dead_scenario_mode"] == "continue_other_rendering"
    assert ckpt_pub["header"]["retry_form"] == "single_turn"
    ckpt_pub_kr = ckpt_pub["scenarios"][0]["kept_renderings"]
    assert ckpt_pub_kr[0]["attempt"] == 1 and ckpt_pub_kr[0]["temperature"] == RETRY_TEMPERATURE
    assert ckpt_pub_kr[0]["min_p"] == RETRY_MIN_P and ckpt_pub_kr[0]["seed"] == derive_call_seed(int(w0.seed), 0, 1)
    assert ckpt_pub_kr[1]["attempt"] == 0 and ckpt_pub_kr[1]["temperature"] == 0.0 and ckpt_pub_kr[1]["min_p"] is None
    assert ckpt_sealed["scenarios"][0]["rendering_0"]["sampling"] == by_r_idx[0]["sampling"]
    assert ckpt_sealed["scenarios"][0]["rendering_1"]["sampling"] == by_r_idx[1]["sampling"]
    assert draft_secret_marker not in (out_st / "checkpoint.json").read_text(encoding="utf-8")

    # --- Requirement 4: Explicit multi-turn option still works when chosen (with retry sampling options & derived seed) ---
    mt_payloads: list[dict[str, Any]] = []

    def _mt_transport(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            assert payload is not None
            idx = len(mt_payloads)
            mt_payloads.append(json.loads(json.dumps(payload)))
            txt = failing_r0_draft if idx == 0 else (passing_r0 if idx == 1 else passing_r1)
            return {
                "message": {"role": "assistant", "content": txt},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3100,
                "eval_count": 780,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    out_mt = tmp_path / "multi_turn_opt_in_run"
    summary_mt = build_and_probe_dataset(
        count=1,
        base_seed=20261030,
        output_dir=out_mt,
        writer_kind="ollama",
        max_retries=1,
        multi_turn_retries=True,
        grant_id="test-grant-req4",
        http_transport=_mt_transport,
        start_owned_server=False,
    )
    assert summary_mt["retry_form"] == "multi_turn"
    assert len(mt_payloads) == 3
    assert len(mt_payloads[1]["messages"]) == 3
    assert mt_payloads[1]["messages"][1] == {"role": "assistant", "content": failing_r0_draft}
    assert mt_payloads[1]["messages"][2]["content"].startswith("Keep your earlier text exactly as it was; change only ")
    assert mt_payloads[1]["options"]["temperature"] == RETRY_TEMPERATURE
    assert mt_payloads[1]["options"]["min_p"] == RETRY_MIN_P
    assert mt_payloads[1]["options"]["seed"] == derive_call_seed(int(w0.seed), 0, 1)

    # --- Requirement 6: Short-circuit switch off (default) vs on, recorded in summary and run records ---
    sc_calls_off: list[int] = []

    def _dead_r0_transport_off(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            assert payload is not None
            idx = len(sc_calls_off)
            sc_calls_off.append(idx)
            # r0 fails attempt 0 and attempt 1; r1 passes attempt 0
            txt = failing_r0_draft if idx < 2 else passing_r1
            return {
                "message": {"role": "assistant", "content": txt},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3100,
                "eval_count": 780,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    out_sc_off = tmp_path / "sc_off_run"
    summary_sc_off = build_and_probe_dataset(
        count=1,
        base_seed=20261030,
        output_dir=out_sc_off,
        writer_kind="ollama",
        max_retries=1,
        short_circuit_dead_scenarios=False,
        grant_id="test-grant-req6-off",
        http_transport=_dead_r0_transport_off,
        start_owned_server=False,
    )
    # With short-circuit OFF (default), r1 is still written and checked after dead r0 (3 calls total: r0 a0, r0 a1, r1 a0):
    assert len(sc_calls_off) == 3
    assert summary_sc_off["short_circuit_dead_scenarios"] is False
    assert summary_sc_off["dead_scenario_mode"] == "continue_other_rendering"
    assert summary_sc_off["calls_refused"] == 2 and summary_sc_off["calls_kept"] == 1
    assert summary_sc_off["renderings_refused"] == 1 and summary_sc_off["renderings_kept"] == 1
    assert summary_sc_off["scenarios_refused"] == 1 and summary_sc_off["scenarios_kept"] == 0
    sealing_off = json.loads((out_sc_off / "sealed_store" / "sealing.sealed.json").read_text(encoding="utf-8"))
    assert sealing_off["writer_provenance"]["short_circuit_dead_scenarios"] is False
    assert sealing_off["writer_provenance"]["dead_scenario_mode"] == "continue_other_rendering"

    sc_calls_on: list[int] = []

    def _dead_r0_transport_on(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            assert payload is not None
            idx = len(sc_calls_on)
            sc_calls_on.append(idx)
            txt = failing_r0_draft if idx < 2 else passing_r1
            return {
                "message": {"role": "assistant", "content": txt},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3100,
                "eval_count": 780,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    out_sc_on = tmp_path / "sc_on_run"
    summary_sc_on = build_and_probe_dataset(
        count=1,
        base_seed=20261030,
        output_dir=out_sc_on,
        writer_kind="ollama",
        max_retries=1,
        short_circuit_dead_scenarios=True,
        grant_id="test-grant-req6-on",
        http_transport=_dead_r0_transport_on,
        start_owned_server=False,
    )
    # With short-circuit ON, rendering 1 is NOT called once rendering 0 fails all attempts (only 2 calls: r0 a0, r0 a1):
    assert len(sc_calls_on) == 2
    assert summary_sc_on["short_circuit_dead_scenarios"] is True
    assert summary_sc_on["dead_scenario_mode"] == "short_circuit"
    assert summary_sc_on["calls_refused"] == 2 and summary_sc_on["calls_kept"] == 0
    assert summary_sc_on["renderings_refused"] == 1 and summary_sc_on["renderings_kept"] == 0
    assert summary_sc_on["scenarios_refused"] == 1 and summary_sc_on["scenarios_kept"] == 0
    sealing_on = json.loads((out_sc_on / "sealed_store" / "sealing.sealed.json").read_text(encoding="utf-8"))
    assert sealing_on["writer_provenance"]["short_circuit_dead_scenarios"] is True
    assert sealing_on["writer_provenance"]["dead_scenario_mode"] == "short_circuit"
    ckpt_on = json.loads((out_sc_on / "checkpoint.json").read_text(encoding="utf-8"))
    assert ckpt_on["header"]["short_circuit_dead_scenarios"] is True
    assert ckpt_on["header"]["dead_scenario_mode"] == "short_circuit"


def test_97_goal_mode_copy_attempt0_payload_hash_matches_29c6653_and_d_rule_sentence_unchanged() -> None:
    """Test 97 (Card udlab-goal-splice):
    - `copy` mode is the default and its attempt-0 request payload (messages, options, seed) is byte-identical
      to `main` at `29c6653` for the same scenario (`base_seed=20261030, count=1`).
    - `d_rule_sentence` behavior and its prompt text are unchanged in both `copy` and `splice` modes, and
      only the single `c_fidelity_goal` requirement line differs between the two prompt templates.
    """
    from typing import Any
    from udlab_generator import (
        DEFAULT_GOAL_MODE,
        PINNED_WRITER_MANIFEST_DIGEST,
        PINNED_WRITER_MODEL_TAG,
        VALID_GOAL_MODES,
        OllamaWriter,
        RenderCheckError,
        check_rendered_prose,
        format_writer_prompt,
        load_writer_prompt_template,
        render_world_template,
        render_world_template_goal_tokens,
        verify_prompt_and_checker_requirements_aligned,
    )

    assert DEFAULT_GOAL_MODE == "copy"
    assert VALID_GOAL_MODES == frozenset({"copy", "splice"})

    # 1. Verify prompt alignment in both modes and that ONLY the c_fidelity_goal line differs:
    reqs_copy = verify_prompt_and_checker_requirements_aligned(goal_mode="copy")
    reqs_splice = verify_prompt_and_checker_requirements_aligned(goal_mode="splice")
    assert [r.check_id for r in reqs_copy] == [r.check_id for r in reqs_splice]
    assert len(reqs_copy) == 27

    w0 = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    prompt_copy_r0 = format_writer_prompt(w0, w0.agents[0], goal_mode="copy")
    prompt_splice_r0 = format_writer_prompt(w0, w0.agents[0], goal_mode="splice")

    copy_lines = prompt_copy_r0.splitlines()
    splice_lines = prompt_splice_r0.splitlines()
    assert len(copy_lines) == len(splice_lines)
    diffs = [
        (idx, c_line, s_line)
        for idx, (c_line, s_line) in enumerate(zip(copy_lines, splice_lines))
        if c_line != s_line
    ]
    assert len(diffs) == 3
    _, copy_goal_line, splice_goal_line = diffs[0]
    assert copy_goal_line == (
        "- (c_fidelity_goal) Copy each agent's goal_text (goals[agent].goal_text) verbatim into setup (required)."
    )
    assert splice_goal_line == (
        "- (c_fidelity_goal) For each agent in agents, write the placeholder token [[GOAL:<agent>]] as its own "
        "sentence in setup instead of copying goal_text (required; code expands each [[GOAL:<agent>]] to goals[agent].goal_text)."
    )
    _, copy_bracket_line, splice_bracket_line = diffs[1]
    assert copy_bracket_line == "- (a_no_markers_bracket) Do not emit bracket markers ('[...]')."
    assert splice_bracket_line == (
        "- (a_no_markers_bracket) Do not emit bracket markers ('[...]'), except the goal tokens "
        "[[GOAL:Eva]], [[GOAL:Dan]], [[GOAL:Cara]] (one per agent in setup, exactly as written)."
    )
    _, copy_out_line, splice_out_line = diffs[2]
    assert copy_out_line == (
        "Return only the two third-person prose paragraphs separated by a blank line. "
        "Do not emit headers, lists, bracket markers, fact IDs, step numbers, or commentary."
    )
    assert splice_out_line == (
        "Return only the two third-person prose paragraphs separated by a blank line. "
        "Do not emit headers, lists, bracket markers "
        "(except the goal tokens [[GOAL:Eva]], [[GOAL:Dan]], [[GOAL:Cara]], one per agent in setup, exactly as written), "
        "fact IDs, step numbers, or commentary."
    )

    # 2. Verify d_rule_sentence prompt line and behavior are unchanged in both modes:
    expected_d_rule_line = (
        '- (d_rule_sentence) Include belief-persistence rule once in setup (required): '
        '"Each agent assumes an object remains where that agent last saw it in open view or heard a witness state '
        'it was placed, and assumes another agent looks where they last saw or communicated that agent perceiving it."'
    )
    assert expected_d_rule_line in copy_lines
    assert expected_d_rule_line in splice_lines
    assert (
        hashlib.sha256(expected_d_rule_line.encode("utf-8")).hexdigest()
        == "ce2efa9a5c102289f3ec869f1a2ab0c471d79ded3435fed0167556f4f3d01243"
    )

    passing_copy_r0 = render_world_template(w0, w0.agents[0])
    passing_splice_r0 = render_world_template_goal_tokens(w0, w0.agents[0])
    check_rendered_prose(passing_copy_r0, w0, focal_agent=w0.agents[0], goal_mode="copy")
    check_rendered_prose(passing_splice_r0, w0, focal_agent=w0.agents[0], goal_mode="splice")

    rule_sentence_literal = (
        "Each agent assumes an object remains where that agent last saw it in open view "
        "or heard a witness state it was placed, and assumes another agent looks "
        "where they last saw or communicated that agent perceiving it."
    )
    for mode_name, base_text in (("copy", passing_copy_r0), ("splice", passing_splice_r0)):
        missing_rule = base_text.replace(rule_sentence_literal, "")
        with pytest.raises(RenderCheckError) as exc_d_abs:
            check_rendered_prose(missing_rule, w0, focal_agent=w0.agents[0], goal_mode=mode_name)
        assert exc_d_abs.value.check_id == "d_rule_sentence"
        assert exc_d_abs.value.failure_kind == "absent"

        altered_rule = base_text.replace(
            "last saw it in open view",
            "previously noticed it in open view",
        )
        with pytest.raises(RenderCheckError) as exc_d_alt:
            check_rendered_prose(altered_rule, w0, focal_agent=w0.agents[0], goal_mode=mode_name)
        assert exc_d_alt.value.check_id == "d_rule_sentence"
        assert exc_d_alt.value.failure_kind == "altered"

    # 3. Verify copy mode attempt-0 request payload hash matches pinned 29c6653 hash:
    passing_copy_r1 = render_world_template(w0, w0.agents[1])
    captured_payloads: list[dict[str, Any]] = []

    def _copy_transport(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            assert payload is not None
            idx = len(captured_payloads)
            captured_payloads.append(json.loads(json.dumps(payload)))
            return {
                "message": {"role": "assistant", "content": passing_copy_r0 if idx == 0 else passing_copy_r1},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3100,
                "eval_count": 780,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    writer_copy = OllamaWriter(
        grant_id="test-grant-copy-29c6653",
        http_transport=_copy_transport,
        start_owned_server=False,
    )
    assert writer_copy.goal_mode == "copy"
    pair_copy = writer_copy.render_scenario_pair(w0)
    assert pair_copy is not None and len(captured_payloads) == 2

    r0_canonical = json.dumps(captured_payloads[0], sort_keys=True, separators=(",", ":"))
    r1_canonical = json.dumps(captured_payloads[1], sort_keys=True, separators=(",", ":"))
    assert (
        hashlib.sha256(r0_canonical.encode("utf-8")).hexdigest()
        == "029615ed90403f100152055f5f15107e9c56feda867c68546553a22978b9d2b8"
    )
    assert (
        hashlib.sha256(r1_canonical.encode("utf-8")).hexdigest()
        == "8aba5e2c4cda48c0a4b2dffe8078ce997d12c7f39802712203b49f3ee2255a59"
    )
    assert (
        hashlib.sha256(captured_payloads[0]["messages"][0]["content"].encode("utf-8")).hexdigest()
        == "f5007999c2903e9324d9f4cbcdbfebe8c8281bf88114ecad48625abdf93784ee"
    )
    assert (
        hashlib.sha256(captured_payloads[1]["messages"][0]["content"].encode("utf-8")).hexdigest()
        == "136107fd3d5fe7b9a7bc1392888bdc4365d19be84e0ec5c5abcf8e89e6e0fc08"
    )
    assert captured_payloads[0]["options"] == {
        "temperature": 0,
        "seed": 1939265819,
        "num_predict": 4096,
        "num_ctx": 9216,
    }
    assert captured_payloads[1]["options"] == {
        "temperature": 0,
        "seed": 645167106,
        "num_predict": 4096,
        "num_ctx": 9216,
    }
    assert pair_copy[0].goal_mode == "copy"
    assert pair_copy[0].goal_source == {"Eva": "model_text", "Dan": "model_text", "Cara": "model_text"}
    assert pair_copy[1].goal_mode == "copy"
    assert pair_copy[1].goal_source == {"Eva": "model_text", "Dan": "model_text", "Cara": "model_text"}


def test_98_goal_mode_splice_token_expansion_goal_source_and_stored_vs_sealed_prose(tmp_path: Path) -> None:
    """Test 98 (Card udlab-goal-splice):
    - In `splice` mode, a fake writer emitting `[[GOAL:<agent>]]` tokens in setup passes; expanded prose contains
      each agent's exact `goal_text` once with a trailing period and no `[[` remains in any stored prose.
    - Raw model output (with `[[GOAL:<agent>]]` tokens) is kept only in `sealed_store/responses.sealed.jsonl`,
      while `RenderedScenario.prose`, `part_public/items.jsonl`, `sealed_store/keys.sealed.jsonl`, and checkpoints
      store the expanded prose and record `goal_mode: "splice"` and `goal_source: {"Eva": "token", ...}`.
    - If the model writes the goal sentence itself instead of a token, that passes in `splice` mode and records
      `goal_source: "model_text"` for that agent.
    """
    from typing import Any
    from udlab_generator import (
        PINNED_WRITER_MANIFEST_DIGEST,
        PINNED_WRITER_MODEL_TAG,
        recheck_saved_responses,
        render_world_template,
        render_world_template_goal_tokens,
        splice_goal_tokens,
    )
    from udlab_generator.cli import build_and_probe_dataset, main as cli_main, verify_written_dataset

    w0 = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    assert w0.agents == ("Eva", "Dan", "Cara")

    raw_token_r0 = render_world_template_goal_tokens(w0, "Eva")
    raw_token_r1 = render_world_template_goal_tokens(w0, "Dan")
    assert "[[GOAL:Eva]]" in raw_token_r0
    assert "[[GOAL:Dan]]" in raw_token_r0
    assert "[[GOAL:Cara]]" in raw_token_r0
    for ag in w0.agents:
        assert w0.goals[ag].goal_text not in raw_token_r0

    spliced_r0 = splice_goal_tokens(raw_token_r0, w0)
    assert spliced_r0.goal_source == {"Eva": "token", "Dan": "token", "Cara": "token"}
    assert "[[" not in spliced_r0.expanded_prose and "]]" not in spliced_r0.expanded_prose
    for ag in w0.agents:
        assert spliced_r0.expanded_prose.count(f"{w0.goals[ag].goal_text}.") == 1
    assert check_rendered_prose(raw_token_r0, w0, focal_agent="Eva", goal_mode="splice") == ()

    # Rendering 0 uses all 3 tokens; Rendering 1 writes Dan's goal_text directly as prose and uses tokens for Eva & Cara:
    mixed_r1 = raw_token_r1.replace("[[GOAL:Dan]]", f"{w0.goals['Dan'].goal_text}.")
    assert "[[GOAL:Eva]]" in mixed_r1
    assert "[[GOAL:Dan]]" not in mixed_r1
    assert "[[GOAL:Cara]]" in mixed_r1

    call_count = 0

    def _splice_transport(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        nonlocal call_count
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            txt = raw_token_r0 if call_count == 0 else mixed_r1
            call_count += 1
            return {
                "message": {"role": "assistant", "content": txt},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3080,
                "eval_count": 760,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    out_dir = tmp_path / "splice_mode_run"
    summary = build_and_probe_dataset(
        count=1,
        base_seed=20261030,
        output_dir=out_dir,
        writer_kind="ollama",
        goal_mode="splice",
        grant_id="test-grant-splice-98",
        http_transport=_splice_transport,
        start_owned_server=False,
    )
    assert summary["goal_mode"] == "splice"
    assert summary["scenarios_kept"] == 1
    assert summary["renderings_kept"] == 2

    build_summary_doc = json.loads((out_dir / "build_summary.json").read_text(encoding="utf-8"))
    assert build_summary_doc["goal_mode"] == "splice"

    sealing_doc = json.loads((out_dir / "sealed_store" / "sealing.sealed.json").read_text(encoding="utf-8"))
    assert sealing_doc["writer_provenance"]["goal_mode"] == "splice"

    # Check generations.jsonl and responses.sealed.jsonl:
    gen_rows = [
        json.loads(line)
        for line in (out_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    resp_rows = [
        json.loads(line)
        for line in (out_dir / "sealed_store" / "responses.sealed.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(gen_rows) == 2 and len(resp_rows) == 2
    assert all(r["goal_mode"] == "splice" for r in gen_rows)
    assert all(r["goal_mode"] == "splice" for r in resp_rows)
    assert resp_rows[0]["response_text"] == raw_token_r0
    assert resp_rows[1]["response_text"] == mixed_r1
    assert "[[GOAL:Eva]]" in resp_rows[0]["response_text"]

    # Check public items, sealed keys, and checkpoints store expanded prose with no [[ tokens remaining:
    pub_items = [
        json.loads(line)
        for line in (out_dir / "part_public" / "items.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sealed_items = [
        json.loads(line)
        for line in (out_dir / "sealed_store" / "keys.sealed.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(pub_items) == 2 and len(sealed_items) == 2
    for row in [*pub_items, *sealed_items]:
        assert row["goal_mode"] == "splice"
        assert "[[" not in row["prose"] and "]]" not in row["prose"]
        for ag in w0.agents:
            assert row["prose"].count(f"{w0.goals[ag].goal_text}.") == 1
        if row["focal_agent"] == "Eva":
            assert row["goal_source"] == {"Eva": "token", "Dan": "token", "Cara": "token"}
        else:
            assert row["goal_source"] == {"Eva": "token", "Dan": "model_text", "Cara": "token"}

    ckpt_pub = json.loads((out_dir / "checkpoint.json").read_text(encoding="utf-8"))
    ckpt_sealed = json.loads((out_dir / "sealed_store" / "checkpoint.sealed.json").read_text(encoding="utf-8"))
    assert ckpt_pub["header"]["goal_mode"] == "splice"
    assert ckpt_sealed["header"]["goal_mode"] == "splice"
    assert ckpt_pub["scenarios"][0]["kept_renderings"][0]["goal_mode"] == "splice"
    assert ckpt_pub["scenarios"][0]["kept_renderings"][0]["goal_source"] == {
        "Eva": "token",
        "Dan": "token",
        "Cara": "token",
    }
    assert ckpt_pub["scenarios"][0]["kept_renderings"][1]["goal_source"] == {
        "Eva": "token",
        "Dan": "model_text",
        "Cara": "token",
    }
    assert "[[" not in ckpt_sealed["scenarios"][0]["rendering_0"]["prose"]
    assert "[[" not in ckpt_sealed["scenarios"][0]["rendering_1"]["prose"]

    # Offline recheck and verify-written both pass on the splice-mode output directory:
    recheck_res = recheck_saved_responses(out_dir)
    assert recheck_res["matches_recorded_verdicts"] is True
    assert recheck_res["renderings_kept"] == 2
    verify_res = verify_written_dataset(out_dir)
    assert verify_res["generations_valid"] is True
    assert verify_res["recount_verification"]["independent_b2_verified"] is True
    assert verify_res["recount_verification"]["independent_recounts_verified"] is True
    assert verify_res["recount_verification"]["prose_checks_passed"] is True

    # Also test CLI --goal-mode splice with mock writer:
    cli_out = tmp_path / "cli_mock_splice"
    rc = cli_main(["--count", "2", "--seed", "20261030", "--out-dir", str(cli_out), "--goal-mode", "splice"])
    assert rc == 0
    cli_summary = json.loads((cli_out / "build_summary.json").read_text(encoding="utf-8"))
    assert cli_summary["goal_mode"] == "splice"
    assert recheck_saved_responses(cli_out)["matches_recorded_verdicts"] is True


def test_99_goal_mode_splice_token_faults_and_single_turn_repair_notes() -> None:
    """Test 99 (Card udlab-goal-splice):
    - Missing token, duplicate token, unknown-agent token, and token in paragraph 2 each fail as `c_fidelity_goal`
      with the right `failure_kind` (`absent` for missing token; `false` for duplicate, unknown-agent, or paragraph-2 token)
      and a single-turn repair note naming `c_fidelity_goal` and quoting no model text.
    """
    from udlab_generator import (
        RenderCheckError,
        check_rendered_prose,
        collect_prose_faults,
        format_repair_retry_line,
        render_world_template_goal_tokens,
    )

    w0 = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    base_splice = render_world_template_goal_tokens(w0, "Eva")
    setup_p1, events_p2 = base_splice.split("\n\n", 1)

    secret_marker = "SECRET_MODEL_GOAL_DRAFT_LEAK_99"

    # 1. Missing token (Eva's token removed from setup, and Eva's goal_text not present):
    missing_tok_prose = f"{setup_p1.replace('[[GOAL:Eva]]', f'Eva waited near the entrance ({secret_marker}).')}\n\n{events_p2}"
    with pytest.raises(RenderCheckError) as exc_missing:
        check_rendered_prose(missing_tok_prose, w0, focal_agent="Eva", goal_mode="splice")
    assert exc_missing.value.check_id == "c_fidelity_goal"
    assert exc_missing.value.failure_kind == "absent"
    assert exc_missing.value.goal_mode == "splice"
    faults_missing = collect_prose_faults(missing_tok_prose, w0, focal_agent="Eva", goal_mode="splice")
    assert [f.check_id for f in faults_missing] == ["c_fidelity_goal"]
    assert faults_missing[0].failure_kind == "absent"
    note_missing = format_repair_retry_line(exc_missing.value)
    assert note_missing == (
        "Repair note (rule broken: check c_fidelity_goal, absent): include one sentence naming the setup agent goal text "
        "(check c_fidelity_goal) so that it satisfies: For each agent in agents, write the placeholder token "
        "[[GOAL:<agent>]] as its own sentence in setup instead of copying goal_text (required; code expands each "
        "[[GOAL:<agent>]] to goals[agent].goal_text)."
    )
    assert secret_marker not in note_missing

    # 2. Duplicate token (`[[GOAL:Eva]]` appears twice in setup):
    dup_tok_prose = f"{setup_p1} [[GOAL:Eva]]\n\n{events_p2}"
    with pytest.raises(RenderCheckError) as exc_dup:
        check_rendered_prose(dup_tok_prose, w0, focal_agent="Eva", goal_mode="splice")
    assert exc_dup.value.check_id == "c_fidelity_goal"
    assert exc_dup.value.failure_kind == "false"
    assert exc_dup.value.goal_mode == "splice"
    faults_dup = collect_prose_faults(dup_tok_prose, w0, focal_agent="Eva", goal_mode="splice")
    assert any(f.check_id == "c_fidelity_goal" and f.failure_kind == "false" for f in faults_dup)
    note_dup = format_repair_retry_line(exc_dup.value)
    assert note_dup == (
        "Repair note (rule broken: check c_fidelity_goal, false): write each agent's [[GOAL:<agent>]] token at most "
        "once in setup (remove duplicate goal token, check c_fidelity_goal) and follow: For each agent in agents, "
        "write the placeholder token [[GOAL:<agent>]] as its own sentence in setup instead of copying goal_text "
        "(required; code expands each [[GOAL:<agent>]] to goals[agent].goal_text)."
    )
    assert "[[GOAL:Eva]]" not in note_dup

    # 3. Unknown-agent token (`[[GOAL:Mallory]]` in setup):
    unknown_tok_prose = f"{setup_p1.replace('[[GOAL:Eva]]', '[[GOAL:Mallory]]')}\n\n{events_p2}"
    with pytest.raises(RenderCheckError) as exc_unk:
        check_rendered_prose(unknown_tok_prose, w0, focal_agent="Eva", goal_mode="splice")
    assert exc_unk.value.check_id == "c_fidelity_goal"
    assert exc_unk.value.failure_kind == "false"
    assert exc_unk.value.goal_mode == "splice"
    faults_unk = collect_prose_faults(unknown_tok_prose, w0, focal_agent="Eva", goal_mode="splice")
    assert any(f.check_id == "c_fidelity_goal" and f.failure_kind == "false" for f in faults_unk)
    note_unk = format_repair_retry_line(exc_unk.value)
    assert note_unk == (
        "Repair note (rule broken: check c_fidelity_goal, false): remove any [[GOAL:<agent>]] token for an "
        "unknown agent not in agents (check c_fidelity_goal) and follow: "
        "For each agent in agents, write the placeholder token [[GOAL:<agent>]] as its own sentence in setup "
        "instead of copying goal_text (required; code expands each [[GOAL:<agent>]] to goals[agent].goal_text)."
    )
    assert "Mallory" not in note_unk

    # 4. Token in paragraph 2 (event paragraph):
    para2_tok_prose = f"{setup_p1}\n\n[[GOAL:Eva]] {events_p2}"
    with pytest.raises(RenderCheckError) as exc_p2:
        check_rendered_prose(para2_tok_prose, w0, focal_agent="Eva", goal_mode="splice")
    assert exc_p2.value.check_id == "c_fidelity_goal"
    assert exc_p2.value.failure_kind == "false"
    assert exc_p2.value.goal_mode == "splice"
    faults_p2 = collect_prose_faults(para2_tok_prose, w0, focal_agent="Eva", goal_mode="splice")
    assert any(f.check_id == "c_fidelity_goal" and f.failure_kind == "false" for f in faults_p2)
    note_p2 = format_repair_retry_line(exc_p2.value)
    assert note_p2 == (
        "Repair note (rule broken: check c_fidelity_goal, false): remove any [[GOAL:<agent>]] token from the "
        "events paragraph and place goal tokens only in the setup paragraph (check c_fidelity_goal) and follow: "
        "For each agent in agents, write the placeholder token [[GOAL:<agent>]] as its own sentence in setup "
        "instead of copying goal_text (required; code expands each [[GOAL:<agent>]] to goals[agent].goal_text)."
    )


def test_100_paired_seed_bank_and_mcnemar_comparison_end_to_end(tmp_path: Path) -> None:
    """Test 100 (Card udlab-goal-splice):
    - Paired comparison tool (`run_paired_seed_bank` and `compare_paired_runs`, plus CLI flags
      `--paired-seed-bank` and `--paired-compare`) runs attempt 0 only (`max_retries=0`) on a fixed
      seed-bank JSON file in `copy` and `splice` modes with identical per-call seeds (`derive_call_seed`).
    - Pairs outputs by `(scenario_seed, rendering_index)` and reports per-fault counts by mode,
      rendering pass rate and scenario pass rate by mode, discordant pairs table, and McNemar's exact
      two-sided p-value matching hand-calculated values.
    """
    from typing import Any
    from udlab_generator import (
        PINNED_WRITER_MANIFEST_DIGEST,
        PINNED_WRITER_MODEL_TAG,
        compare_paired_runs,
        derive_call_seed,
        mcnemar_exact_p_value,
        render_world_template,
        render_world_template_goal_tokens,
        run_paired_seed_bank,
        world_from_scenario_seed,
    )
    from udlab_generator.cli import main as cli_main

    # 1. Verify McNemar exact two-sided binomial p-value against hand-calculated fractions:
    # n = 0 -> p = 1.0
    assert mcnemar_exact_p_value(0, 0) == 1.0
    # b = 0, c = 6 -> 2 * C(6, 0) / 2^6 = 2 / 64 = 0.03125
    assert mcnemar_exact_p_value(0, 6) == 0.03125
    assert mcnemar_exact_p_value(6, 0) == 0.03125
    # b = 0, c = 5 -> 2 * C(5, 0) / 2^5 = 2 / 32 = 0.0625
    assert mcnemar_exact_p_value(0, 5) == 0.0625
    # b = 1, c = 7 -> 2 * (C(8, 0) + C(8, 1)) / 2^8 = 2 * 9 / 256 = 18 / 256 = 0.0703125
    assert mcnemar_exact_p_value(1, 7) == 0.0703125
    # b = 2, c = 2 -> min(1.0, 2 * (1 + 4 + 6) / 16) = 1.0
    assert mcnemar_exact_p_value(2, 2) == 1.0

    # 2. Construct a 4-scenario seed bank (8 renderings total) and run both copy and splice modes:
    seed_list = [20261101, 20261102, 20261103, 20261104]
    seed_bank_path = tmp_path / "seed_bank.json"
    seed_bank_path.write_text(json.dumps(seed_list), encoding="utf-8")

    worlds = [world_from_scenario_seed(s, index=i) for i, s in enumerate(seed_list)]
    assert [int(w.seed) for w in worlds] == seed_list

    # In `copy` mode:
    # - Scenario 0 (r0, r1): both pass
    # - Scenario 1 (r0, r1): both fail `c_fidelity_goal` (paraphrased goal text)
    # - Scenario 2 (r0, r1): both fail `c_fidelity_goal` (paraphrased goal text)
    # - Scenario 3: r0 fails `c_fidelity_goal`; r1 fails `a_no_markers` ("[F1]" leaked)
    copy_responses: list[str] = []
    splice_responses: list[str] = []
    for idx, w in enumerate(worlds):
        for r_idx in (0, 1):
            focal = w.agents[r_idx]
            good_copy = render_world_template(w, focal)
            good_splice = render_world_template_goal_tokens(w, focal)
            if idx == 0:
                copy_responses.append(good_copy)
                splice_responses.append(good_splice)
            elif idx in (1, 2) or (idx == 3 and r_idx == 0):
                ag0 = w.agents[0]
                paraphrased_copy = good_copy.replace(
                    w.goals[ag0].goal_text,
                    f"{ag0} hopes to finish the shift early",
                )
                copy_responses.append(paraphrased_copy)
                splice_responses.append(good_splice)
            else:
                # idx == 3, r_idx == 1: both fail a_no_markers
                copy_responses.append(f"[F1] {good_copy}")
                splice_responses.append(f"[F1] {good_splice}")

    copy_call_idx = 0

    def _copy_paired_transport(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        nonlocal copy_call_idx
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            txt = copy_responses[copy_call_idx]
            copy_call_idx += 1
            return {
                "message": {"role": "assistant", "content": txt},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3050,
                "eval_count": 770,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    splice_call_idx = 0

    def _splice_paired_transport(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        nonlocal splice_call_idx
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            txt = splice_responses[splice_call_idx]
            splice_call_idx += 1
            return {
                "message": {"role": "assistant", "content": txt},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3050,
                "eval_count": 770,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    copy_out_path = tmp_path / "paired_copy.json"
    splice_out_path = tmp_path / "paired_splice.json"
    comp_out_path = tmp_path / "paired_compare.json"

    run_copy = run_paired_seed_bank(
        seed_bank=seed_bank_path,
        goal_mode="copy",
        output_path=copy_out_path,
        grant_id="test-grant-paired-copy",
        http_transport=_copy_paired_transport,
        commit_sha="29c6653",
    )
    run_splice = run_paired_seed_bank(
        seed_bank=seed_bank_path,
        goal_mode="splice",
        output_path=splice_out_path,
        grant_id="test-grant-paired-splice",
        http_transport=_splice_paired_transport,
        commit_sha="29c6653",
    )

    assert copy_call_idx == 8 and splice_call_idx == 8
    assert run_copy["goal_mode"] == "copy"
    assert run_copy["max_retries"] == 0
    assert run_copy["commit"] == "29c6653"
    assert run_copy["manifest_digest"] == PINNED_WRITER_MANIFEST_DIGEST
    assert run_copy["passed_renderings"] == 2 and run_copy["num_renderings"] == 8
    assert run_copy["rendering_pass_rate"] == 0.25
    assert run_copy["passed_scenarios"] == 1 and run_copy["num_scenarios"] == 4
    assert run_copy["scenario_pass_rate"] == 0.25
    assert run_copy["fault_counts"] == {
        "a_no_markers_bracket": 1,
        "a_no_markers_fact_id": 1,
        "c_fidelity_goal": 5,
    }

    assert run_splice["goal_mode"] == "splice"
    assert run_splice["passed_renderings"] == 7 and run_splice["num_renderings"] == 8
    assert run_splice["rendering_pass_rate"] == 0.875
    assert run_splice["passed_scenarios"] == 3 and run_splice["num_scenarios"] == 4
    assert run_splice["scenario_pass_rate"] == 0.75
    assert run_splice["fault_counts"] == {
        "a_no_markers_bracket": 1,
        "a_no_markers_fact_id": 1,
    }

    # Verify identical per-call seeds in both modes:
    for r_c, r_s in zip(run_copy["renderings"], run_splice["renderings"]):  # type: ignore[arg-type]
        assert r_c["scenario_seed"] == r_s["scenario_seed"]
        assert r_c["rendering_index"] == r_s["rendering_index"]
        assert r_c["call_seed"] == r_s["call_seed"]
        assert r_c["call_seed"] == derive_call_seed(int(r_c["scenario_seed"]), int(r_c["rendering_index"]), 0)

    # Run CLI `--paired-compare`:
    rc_comp = cli_main(
        [
            "--paired-compare",
            str(copy_out_path),
            str(splice_out_path),
            "--paired-out",
            str(comp_out_path),
        ]
    )
    assert rc_comp == 0
    comp = json.loads(comp_out_path.read_text(encoding="utf-8"))
    assert comp["per_fault_counts_by_mode"] == {
        "copy": {
            "a_no_markers_bracket": 1,
            "a_no_markers_fact_id": 1,
            "c_fidelity_goal": 5,
        },
        "splice": {
            "a_no_markers_bracket": 1,
            "a_no_markers_fact_id": 1,
        },
    }
    assert comp["rendering_pass_rate_by_mode"] == {
        "copy": {"passed": 2, "total": 8, "rate": 0.25},
        "splice": {"passed": 7, "total": 8, "rate": 0.875},
    }
    assert comp["scenario_pass_rate_by_mode"] == {
        "copy": {"passed": 1, "total": 4, "rate": 0.25},
        "splice": {"passed": 3, "total": 4, "rate": 0.75},
    }
    assert comp["contingency"] == {
        "both_pass": 2,
        "copy_pass_splice_fail": 0,
        "copy_fail_splice_pass": 5,
        "both_fail": 1,
    }
    # Hand-calculated exact McNemar p-value for b=0, c=5: 2 * (1/32) = 0.0625
    assert comp["mcnemar_exact"] == {
        "b": 0,
        "c": 5,
        "n_discordant": 5,
        "p_value_two_sided": 0.0625,
    }
    assert len(comp["discordant_pairs"]) == 5
    assert all(dp["direction"] == "splice_only" for dp in comp["discordant_pairs"])
    assert all(dp["copy_faults"] == ["c_fidelity_goal"] and dp["splice_faults"] == [] for dp in comp["discordant_pairs"])

    # Verify seed-mismatch guard in compare_paired_runs:
    tampered_splice = json.loads(json.dumps(run_splice))
    tampered_splice["renderings"][0]["call_seed"] += 1
    with pytest.raises(ValueError, match="Per-call seed mismatch"):
        compare_paired_runs(run_copy, tampered_splice)


def test_101_splice_mode_narrow_bracket_exception_for_2_and_3_agents_and_copy_unchanged() -> None:
    """Test 101 (Card udlab-goal-splice-fix1, Item 1):
    - (a) In `splice` mode, both the `a_no_markers_bracket` bullet and the template output-format line
      (`prompts/writer_qwen3_6_35b.txt:18`) carry a narrow exception naming only the exact token form
      for the agents present in that scenario (`[[GOAL:<agent>]]`, one per agent in setup, exactly as written),
      verified on both a 2-agent and a 3-agent scenario, and no rule in the rendered splice prompt unconditionally
      forbids the goal tokens.
    - (b) Prose with valid goal tokens plus any other bracket (`[note]`, `[[GOAL]]`, `[[GOAL:Unknown]]`, `[[goal:Eva]]`)
      is refused with the right `check_id` (`a_no_markers_bracket`, or `c_fidelity_goal` for `[[GOAL:Unknown]]`).
    - (c) `copy` mode remains byte-identical to `29c6653`.
    """
    from dataclasses import replace
    from udlab_generator import (
        RenderCheckError,
        check_rendered_prose,
        format_writer_prompt,
        render_world_template_goal_tokens,
    )

    cfg3 = make_balanced_configs(1, base_seed=20261030)[0]
    w3 = generate_world(cfg3)
    assert w3.agents == ("Eva", "Dan", "Cara")

    w2 = replace(
        w3,
        agents=("Ana", "Ben"),
        initial_agent_rooms={"Ana": w3.rooms[0], "Ben": w3.rooms[1]},
        goals={
            "Ana": w3.goals["Eva"],
            "Ben": w3.goals["Dan"],
        },
    )
    assert w2.agents == ("Ana", "Ben")
    ag2_0, ag2_1 = w2.agents

    prompt_splice_2 = format_writer_prompt(w2, ag2_0, goal_mode="splice")
    prompt_splice_3 = format_writer_prompt(w3, "Eva", goal_mode="splice")
    prompt_copy_3 = format_writer_prompt(w3, "Eva", goal_mode="copy")

    # (a) 2-agent and 3-agent splice prompts carry narrow agent-specific exceptions on both bracket rules:
    expected_bullet_2 = (
        f"- (a_no_markers_bracket) Do not emit bracket markers ('[...]'), except the goal tokens "
        f"[[GOAL:{ag2_0}]], [[GOAL:{ag2_1}]] (one per agent in setup, exactly as written)."
    )
    expected_footer_2 = (
        f"Return only the two third-person prose paragraphs separated by a blank line. Do not emit headers, lists, "
        f"bracket markers (except the goal tokens [[GOAL:{ag2_0}]], [[GOAL:{ag2_1}]], one per agent in setup, "
        f"exactly as written), fact IDs, step numbers, or commentary."
    )
    assert expected_bullet_2 in prompt_splice_2.splitlines()
    assert expected_footer_2 in prompt_splice_2.splitlines()
    assert "- (a_no_markers_bracket) Do not emit bracket markers ('[...]')." not in prompt_splice_2
    assert "Do not emit headers, lists, bracket markers, fact IDs" not in prompt_splice_2

    expected_bullet_3 = (
        "- (a_no_markers_bracket) Do not emit bracket markers ('[...]'), except the goal tokens "
        "[[GOAL:Eva]], [[GOAL:Dan]], [[GOAL:Cara]] (one per agent in setup, exactly as written)."
    )
    expected_footer_3 = (
        "Return only the two third-person prose paragraphs separated by a blank line. Do not emit headers, lists, "
        "bracket markers (except the goal tokens [[GOAL:Eva]], [[GOAL:Dan]], [[GOAL:Cara]], one per agent in setup, "
        "exactly as written), fact IDs, step numbers, or commentary."
    )
    assert expected_bullet_3 in prompt_splice_3.splitlines()
    assert expected_footer_3 in prompt_splice_3.splitlines()
    assert "- (a_no_markers_bracket) Do not emit bracket markers ('[...]')." not in prompt_splice_3
    assert "Do not emit headers, lists, bracket markers, fact IDs" not in prompt_splice_3

    # (b) Valid tokens plus any other bracket (`[note]`, `[[GOAL]]`, `[[GOAL:Unknown]]`, `[[goal:Eva]]`) is refused:
    base_3 = render_world_template_goal_tokens(w3, "Eva")
    setup_3, events_3 = base_3.split("\n\n", 1)
    assert check_rendered_prose(base_3, w3, focal_agent="Eva", goal_mode="splice") == ()

    for bad_bracket, expected_cid in (
        ("[note]", "a_no_markers_bracket"),
        ("[[GOAL]]", "a_no_markers_bracket"),
        ("[[GOAL:Unknown]]", "c_fidelity_goal"),
        ("[[goal:Eva]]", "a_no_markers_bracket"),
    ):
        bad_prose = f"{setup_3} {bad_bracket}\n\n{events_3}"
        with pytest.raises(RenderCheckError) as exc_info:
            check_rendered_prose(bad_prose, w3, focal_agent="Eva", goal_mode="splice")
        assert exc_info.value.check_id == expected_cid

    # (c) Copy mode stays byte-identical to 29c6653:
    assert (
        hashlib.sha256(prompt_copy_3.encode("utf-8")).hexdigest()
        == "f5007999c2903e9324d9f4cbcdbfebe8c8281bf88114ecad48625abdf93784ee"
    )


def test_102_splice_mode_trailing_punctuation_and_mid_sentence_fault() -> None:
    """Test 102 (Card udlab-goal-splice-fix1, Item 2):
    - A token followed by '.', '!' or '?' expands to exactly one sentence-final period (no '..', '.!' or '.?').
    - A token in the middle of a sentence (not its own sentence) is a `c_fidelity_goal` fault (`failure_kind == 'altered'`)
      with a token-specific repair note.
    """
    from udlab_generator import (
        RenderCheckError,
        check_rendered_prose,
        collect_prose_faults,
        format_repair_retry_line,
        render_world_template_goal_tokens,
        splice_goal_tokens,
    )

    w0 = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    assert w0.agents == ("Eva", "Dan", "Cara")
    base_raw = render_world_template_goal_tokens(w0, "Eva")
    expected_expanded = splice_goal_tokens(base_raw, w0).expanded_prose

    # Cases 1-3: trailing '.', '!', '?' after [[GOAL:Eva]]:
    for punct in (".", "!", "?"):
        with_punct = base_raw.replace("[[GOAL:Eva]]", f"[[GOAL:Eva]]{punct}")
        res = splice_goal_tokens(with_punct, w0)
        assert res.expanded_prose == expected_expanded
        assert "Eva plans to complete the quarterly audit packet." in res.expanded_prose
        assert "Eva plans to complete the quarterly audit packet.." not in res.expanded_prose
        assert "Eva plans to complete the quarterly audit packet.!" not in res.expanded_prose
        assert "Eva plans to complete the quarterly audit packet.?" not in res.expanded_prose
        assert check_rendered_prose(with_punct, w0, focal_agent="Eva", goal_mode="splice") == ()

    # Case 4: token in the middle of a sentence (not its own sentence):
    mid_sentence_prose = base_raw.replace("[[GOAL:Eva]]", "Stated goal: [[GOAL:Eva]]")
    with pytest.raises(RenderCheckError) as exc_mid:
        check_rendered_prose(mid_sentence_prose, w0, focal_agent="Eva", goal_mode="splice")
    assert exc_mid.value.check_id == "c_fidelity_goal"
    assert exc_mid.value.failure_kind == "altered"
    assert exc_mid.value.goal_mode == "splice"

    faults_mid = collect_prose_faults(mid_sentence_prose, w0, focal_agent="Eva", goal_mode="splice")
    assert [f.check_id for f in faults_mid] == ["c_fidelity_goal"]
    assert faults_mid[0].failure_kind == "altered"

    note_mid = format_repair_retry_line(exc_mid.value)
    assert note_mid == (
        "Repair note (rule broken: check c_fidelity_goal, altered): place each [[GOAL:<agent>]] token as its own "
        "standalone sentence in setup (not embedded inside another sentence, check c_fidelity_goal) and follow: "
        "For each agent in agents, write the placeholder token [[GOAL:<agent>]] as its own sentence in setup "
        "instead of copying goal_text (required; code expands each [[GOAL:<agent>]] to goals[agent].goal_text)."
    )


def test_103_splice_mode_token_specific_repair_notes_and_cli_unused_imports() -> None:
    """Test 103 (Card udlab-goal-splice-fix1, Items 3 & 4):
    - Duplicate token, unknown-agent token, and token in paragraph 2 produce distinct token-specific repair notes
      built from check_id + failure_kind + code-made hint with no model text.
    - `cli.py` has no unused imports (`mcnemar_exact_p_value`, `check_rendered_prose`, `locate_failing_sentence_in_response`, `verify_sealed_part`).
    """
    import ast
    from udlab_generator import (
        RenderCheckError,
        check_rendered_prose,
        format_repair_retry_line,
        render_world_template_goal_tokens,
    )
    import udlab_generator.cli as cli_mod

    w0 = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    base_splice = render_world_template_goal_tokens(w0, "Eva")
    setup_p1, events_p2 = base_splice.split("\n\n", 1)

    # 1. Duplicate token:
    dup_prose = f"{setup_p1} [[GOAL:Eva]]\n\n{events_p2}"
    with pytest.raises(RenderCheckError) as exc_dup:
        check_rendered_prose(dup_prose, w0, focal_agent="Eva", goal_mode="splice")
    note_dup = format_repair_retry_line(exc_dup.value)
    assert note_dup == (
        "Repair note (rule broken: check c_fidelity_goal, false): write each agent's [[GOAL:<agent>]] token "
        "at most once in setup (remove duplicate goal token, check c_fidelity_goal) and follow: "
        "For each agent in agents, write the placeholder token [[GOAL:<agent>]] as its own sentence in setup "
        "instead of copying goal_text (required; code expands each [[GOAL:<agent>]] to goals[agent].goal_text)."
    )

    # 2. Unknown-agent token:
    unk_prose = f"{setup_p1.replace('[[GOAL:Eva]]', '[[GOAL:Mallory]]')}\n\n{events_p2}"
    with pytest.raises(RenderCheckError) as exc_unk:
        check_rendered_prose(unk_prose, w0, focal_agent="Eva", goal_mode="splice")
    note_unk = format_repair_retry_line(exc_unk.value)
    assert note_unk == (
        "Repair note (rule broken: check c_fidelity_goal, false): remove any [[GOAL:<agent>]] token for an "
        "unknown agent not in agents (check c_fidelity_goal) and follow: "
        "For each agent in agents, write the placeholder token [[GOAL:<agent>]] as its own sentence in setup "
        "instead of copying goal_text (required; code expands each [[GOAL:<agent>]] to goals[agent].goal_text)."
    )
    assert "Mallory" not in note_unk

    # 3. Token in paragraph 2:
    p2_prose = f"{setup_p1}\n\n[[GOAL:Eva]] {events_p2}"
    with pytest.raises(RenderCheckError) as exc_p2:
        check_rendered_prose(p2_prose, w0, focal_agent="Eva", goal_mode="splice")
    note_p2 = format_repair_retry_line(exc_p2.value)
    assert note_p2 == (
        "Repair note (rule broken: check c_fidelity_goal, false): remove any [[GOAL:<agent>]] token from the "
        "events paragraph and place goal tokens only in the setup paragraph (check c_fidelity_goal) and follow: "
        "For each agent in agents, write the placeholder token [[GOAL:<agent>]] as its own sentence in setup "
        "instead of copying goal_text (required; code expands each [[GOAL:<agent>]] to goals[agent].goal_text)."
    )
    assert len({note_dup, note_unk, note_p2}) == 3

    # 4. Verify cli.py has zero unused imports:
    cli_source = Path(cli_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(cli_source)
    imported_names: dict[str, int] = {}
    used_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if getattr(node, "module", "") == "__future__":
                continue
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                imported_names[bound] = node.lineno
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used_names.add(node.id)
    unused_in_cli = sorted(set(imported_names) - used_names)
    assert unused_in_cli == []


def test_104_cli_paired_seed_bank_with_fake_transport(tmp_path: Path) -> None:
    """Test 104 (Card udlab-goal-splice-fix1, Item 5):
    - CLI-level test of `--paired-seed-bank ... --paired-out ...` with the fake transport, asserting the
      output file's fields (`goal_mode`, `commit`, `manifest_digest`, per-call seeds, `fault_check_ids`).
    """
    from typing import Any
    from udlab_generator import (
        PINNED_WRITER_MANIFEST_DIGEST,
        PINNED_WRITER_MODEL_TAG,
        derive_call_seed,
        render_world_template_goal_tokens,
        world_from_scenario_seed,
    )
    from udlab_generator.cli import main as cli_main

    seeds = [20261110, 20261111]
    seed_bank_file = tmp_path / "cli_seed_bank.json"
    seed_bank_file.write_text(json.dumps(seeds), encoding="utf-8")
    paired_out_file = tmp_path / "cli_paired_splice_out.json"

    worlds = [world_from_scenario_seed(s, index=i) for i, s in enumerate(seeds)]
    responses = [
        render_world_template_goal_tokens(worlds[0], worlds[0].agents[0]),
        render_world_template_goal_tokens(worlds[0], worlds[0].agents[1]),
        # Scenario 1 r0: missing first agent's goal token -> c_fidelity_goal
        render_world_template_goal_tokens(worlds[1], worlds[1].agents[0]).replace(
            f"[[GOAL:{worlds[1].agents[0]}]]", ""
        ),
        render_world_template_goal_tokens(worlds[1], worlds[1].agents[1]),
    ]
    call_idx = 0

    def _fake_cli_transport(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        nonlocal call_idx
        if url.endswith("/api/version"):
            return {"version": "0.6.0"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": PINNED_WRITER_MODEL_TAG,
                        "model": PINNED_WRITER_MODEL_TAG,
                        "digest": PINNED_WRITER_MANIFEST_DIGEST,
                    }
                ]
            }
        if url.endswith("/api/chat"):
            if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                return {"done": True}
            txt = responses[call_idx]
            call_idx += 1
            return {
                "message": {"role": "assistant", "content": txt},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3020,
                "eval_count": 750,
            }
        raise AssertionError(f"Unexpected URL: {url}")

    rc = cli_main(
        [
            "--paired-seed-bank",
            str(seed_bank_file),
            "--goal-mode",
            "splice",
            "--grant-id",
            "test-cli-paired-grant",
            "--paired-out",
            str(paired_out_file),
        ],
        http_transport=_fake_cli_transport,
    )
    assert rc == 0
    assert paired_out_file.is_file()
    out_doc = json.loads(paired_out_file.read_text(encoding="utf-8"))
    assert out_doc["schema_version"] == "paired_seed_bank_run_v1"
    assert out_doc["goal_mode"] == "splice"
    assert isinstance(out_doc["commit"], str) and len(out_doc["commit"]) >= 7
    assert out_doc["manifest_digest"] == PINNED_WRITER_MANIFEST_DIGEST
    assert out_doc["max_retries"] == 0
    assert out_doc["num_scenarios"] == 2
    assert out_doc["num_renderings"] == 4
    assert out_doc["passed_renderings"] == 3
    assert out_doc["passed_scenarios"] == 1
    assert out_doc["fault_counts"] == {"c_fidelity_goal": 1}

    renderings = out_doc["renderings"]
    assert len(renderings) == 4
    expected_faults = [[], [], ["c_fidelity_goal"], []]
    for idx, row in enumerate(renderings):
        scen_seed = seeds[idx // 2]
        r_idx = idx % 2
        assert row["scenario_seed"] == scen_seed
        assert row["rendering_index"] == r_idx
        assert row["goal_mode"] == "splice"
        assert row["commit"] == out_doc["commit"]
        assert row["manifest_digest"] == PINNED_WRITER_MANIFEST_DIGEST
        assert row["call_seed"] == derive_call_seed(scen_seed, r_idx, 0)
        assert row["fault_check_ids"] == expected_faults[idx]
        assert row["checks_passed"] is (expected_faults[idx] == [])




