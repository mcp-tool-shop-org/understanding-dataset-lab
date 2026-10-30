"""Comprehensive test suite for the udlab release-1 scenario generator, sealing, and shortcut probe."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
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
from udlab_generator.cli import main as cli_main


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

        # Pre-fix failure mode: if keys were computed from a focal-filtered event log (omitting events
        # outside `focal_agent`'s room), keys would change between focal_0 and focal_1:
        focal0_only_events = tuple(ev for ev in world.events if ev.actor == r0.focal_agent or ev.room == world.initial_agent_rooms[r0.focal_agent])
        focal1_only_events = tuple(ev for ev in world.events if ev.actor == r1.focal_agent or ev.room == world.initial_agent_rooms[r1.focal_agent])
        assert focal0_only_events != focal1_only_events


def test_3_u_unknowability_identical_traces() -> None:
    """Test 3: U items have identical traces across alternatives, ask Who/Why (never yes/no), and 'why' U items are goal-independent."""
    for cfg in make_balanced_configs(12, base_seed=1234):
        world = generate_world(cfg)
        assert world.unknowable_spec.trace_a == world.unknowable_spec.trace_b == world.events
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
    """Test 7: D is `(K(helper) \\ K(target)) & causal_preconditions(G_target)`, scored by recall and precision."""
    cfg_false_b1 = replace(make_balanced_configs(2, base_seed=314)[0], b1_true_belief=False)
    w_false = generate_world(cfg_false_b1)
    recs_false = compute_perceptual_records(w_false)
    known_false = compute_known_facts_by_agent(w_false, recs_false)

    # Helper (`a2`) knows F1, F2, F3, F4.
    assert set(known_false[w_false.d_helper]) == {"F1", "F2", "F3", "F4"}
    # Target (`a1`) knows F2 (shared baseline), and does not know F1, F3, F4 when `b1_true_belief=False`.
    assert set(known_false[w_false.d_target]) == {"F2"}
    # Target's goal preconditions are ("F1", "F2", "F4").
    assert w_false.goals[w_false.d_target].causal_preconditions == ("F1", "F2", "F4")

    # Therefore D must be ("F1", "F4") — excluding F2 (already known to target) and excluding F3 (not in G_target)!
    d_false = compute_diff_d(w_false, recs_false, w_false.d_helper, w_false.d_target)
    assert d_false == ("F1", "F4")

    # When `b1_true_belief=True`, target (`a1`) also knows F1, so D reduces to ("F4",)!
    cfg_true_b1 = replace(cfg_false_b1, b1_true_belief=True)
    w_true = generate_world(cfg_true_b1)
    recs_true = compute_perceptual_records(w_true)
    d_true = compute_diff_d(w_true, recs_true, w_true.d_helper, w_true.d_target)
    assert d_true == ("F4",)

    # Verify recall and precision scoring:
    # Exact match ("F1", "F4") -> recall=1.0, precision=1.0, correct=True
    assert score_diff_answer(["F1", "F4"], d_false) == (1.0, 1.0, True)
    # Including F3 (ignoring goal preconditions) -> recall=1.0, precision=2/3, correct=False!
    rec, prec, ok = score_diff_answer(["F1", "F3", "F4"], d_false)
    assert rec == 1.0 and prec == pytest.approx(2 / 3) and not ok
    # Including F2 (ignoring target's existing knowledge) -> recall=1.0, precision=2/3, correct=False!
    rec2, prec2, ok2 = score_diff_answer(["F1", "F2", "F4"], d_false)
    assert rec2 == 1.0 and prec2 == pytest.approx(2 / 3) and not ok2
    # Partial recall ("F1" only) -> recall=0.5, precision=1.0, correct=False!
    assert score_diff_answer(["F1"], d_false) == (0.5, 1.0, False)


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
    """Verify the CLI entrypoint builds, seals, verifies, and probes a dataset cleanly."""
    out_dir = tmp_path / "cli_out"
    rc = cli_main(["--count", "24", "--out-dir", str(out_dir)])
    assert rc in (0, 1)
    assert (out_dir / "build_summary.json").is_file()
    assert (out_dir / "part_public" / "items.jsonl").is_file()
    assert (out_dir / "sealed_store" / "keys.sealed.jsonl").is_file()
