"""Comprehensive test suite for the udlab release-1 scenario generator, sealing, and shortcut probe."""

from __future__ import annotations

from dataclasses import replace
import inspect
import json
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

    # Also test doorway adjacency hearing below vs above noise_threshold directly:
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
    quiet_say = Event(
        step=98,
        kind="say",
        actor=w0.agents[2],
        room=r_a,
        utterance="Quiet note.",
        noise_level=w0.noise_threshold - 10,
        fact_variable="quiet_var",
        fact_value="heard_val",
    )
    grounding_move_2 = replace(grounding_move_1, step=99, to_location="masked_val", fact_value="masked_val")
    loud_masked_say = replace(quiet_say, step=100, noise_level=w0.noise_threshold + 10, fact_value="masked_val")
    custom_world = replace(
        w0,
        initial_agent_rooms={w0.agents[0]: r_b, w0.agents[1]: r_b, w0.agents[2]: r_a},
        events=(grounding_move_1, quiet_say, grounding_move_2, loud_masked_say),
    )
    custom_recs = compute_perceptual_records(custom_world)
    # Agent in adjacent room r_b hears step 98 (under threshold) but NOT step 100 (over threshold)
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

        # Every event step from the complete event log appears in both focalized renderings:
        for ev in world.events:
            assert f"step {ev.step}" in r0.prose
            assert f"step {ev.step}" in r1.prose

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
        partial_writer = MockWriter(custom_renderer=lambda w, fa: f"Following {fa} at step 1 only.")
        with pytest.raises(RenderCheckError, match="incomplete event log|missing step"):
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
    priv_seed = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

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
        private_seed_hex="fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
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
        private_seed_hex="11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff",
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
        private_seed_hex="1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
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
        private_seed_hex="abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
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
