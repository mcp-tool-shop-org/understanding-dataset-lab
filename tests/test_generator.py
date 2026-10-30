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

    # Hand-built two-rooms world: Dora puts note_item on countertop in room A, then Hana sees it on countertop in room B
    w0 = generate_world(configs[0])
    r_a, r_b = w0.doorways[0]
    note_obj = next(d.note_item for d in ug.world.DOMAINS if d.name == w0.domain_name)
    two_rooms_events = w0.events + (
        Event(
            step=len(w0.events) + 1,
            kind="move",
            actor=w0.agents[2],
            room=r_a,
            target_object=note_obj,
            from_location="dispatch folder",
            to_location=f"countertop in the {r_a}",
        ),
        Event(
            step=len(w0.events) + 2,
            kind="see",
            actor=w0.agents[0],
            room=r_b,
            target_object=note_obj,
            from_location=f"countertop in the {r_b}",
            to_location=f"countertop in the {r_b}",
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
        if ev_dict["kind"] == "see" and ev_dict["target_object"] and "countertop" in str(ev_dict.get("to_location", "")):
            other_room = next(r for r in meta["scenarios"]["scen_0000"]["rooms"] if r != ev_dict["room"])
            ev_dict["to_location"] = f"countertop in the {other_room}"
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
    filtered_evs.append(
        Event(
            step=len(filtered_evs) + 1,
            kind="move",
            actor=w_s_who.agents[2],
            room=a2_room,
            target_object=note_obj,
            from_location="dispatch folder",
            to_location=f"countertop in the {a2_room}",
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
        "the vault room contains the oak drawer and the steel cabinet at its container bay, where a frosted partition "
        "stands between the far-side alcove and the container bay; "
        "the reading room contains the slate tray and the cedar shelf at its container bay, divided from the far-side alcove "
        "by a tall bookstack; "
        "the bindery room contains the linen bin and the copper rack in the container bay, separated from the far-side alcove "
        "by a canvas screen; "
        "and the sorting room contains the maple cubby and the iron safe in the container bay, with a rolling pillar "
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
        "In the bindery room, Eva saw the unsigned transfer slip on the countertop in the bindery room in open view. "
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
    think_writer.render_scenario_pair(w0)
    assert seen_chat_payloads[-1]["think"] is True
    assert seen_chat_payloads[-1]["options"]["num_predict"] == 5120
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
        "the vault room contains the oak drawer and the steel cabinet at its container bay with a frosted partition "
        "between the far-side alcove and the container bay. "
        "The reading room contains the slate tray and the cedar shelf at its container bay with a tall bookstack "
        "between the far-side alcove and the container bay. "
        "The bindery room contains the linen bin and the copper rack at its container bay with a canvas screen "
        "between the far-side alcove and the container bay. "
        "The sorting room contains the maple cubby and the iron safe at its container bay with a rolling pillar "
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
    assert "Copy each stake description (stake_edges[].description) verbatim" in prompt_str

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
    """Test 39 (Decision 1): Parse the writer prompt and the checker's requirement list and fail if they diverge."""
    import udlab_generator.render as ug_render

    parse_fn = getattr(ug_render, "parse_writer_prompt_requirements", None)
    enforced_fn = getattr(ug_render, "get_checker_enforced_check_ids", None)
    verify_fn = getattr(ug_render, "verify_prompt_and_checker_requirements_aligned", None)
    shared_reqs = getattr(ug_render, "WRITER_PROSE_REQUIREMENTS", None)
    assert callable(parse_fn) and callable(enforced_fn) and callable(verify_fn)
    assert isinstance(shared_reqs, tuple) and len(shared_reqs) >= 25

    raw_prompt = ug_render.WRITER_PROMPT_PATH.read_text(encoding="utf-8")
    parsed_reqs = parse_fn(raw_prompt)
    assert parsed_reqs == shared_reqs

    enforced_ids = enforced_fn()
    assert set(enforced_ids) == {r.check_id for r in shared_reqs}
    assert verify_fn(raw_prompt) == shared_reqs

    # Also verify formatted prompt parses identically
    w0 = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    formatted = format_writer_prompt(w0, w0.agents[0])
    assert parse_fn(formatted) == shared_reqs

    # Diverging prompt (removing a requirement, altering rule text, or adding an untracked requirement) must fail
    dropped_one = "\n".join(
        line for line in raw_prompt.splitlines() if "(c_fidelity_doorways)" not in line
    )
    with pytest.raises(ValueError, match="diverge"):
        verify_fn(dropped_one)

    altered_text = raw_prompt.replace(
        "Copy each agent's goal_text", "Paraphrase each agent's goal_text"
    )
    with pytest.raises(ValueError, match="diverge"):
        verify_fn(altered_text)

    extra_untracked = raw_prompt.replace(
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





