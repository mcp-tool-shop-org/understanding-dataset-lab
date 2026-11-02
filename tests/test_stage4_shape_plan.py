"""Stage 4 shape-plan tests: legend v1, per-row tags, token parity, plan balance, realized form matching, and telemetry."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import ast
import hashlib
import inspect
import json
from pathlib import Path
import re
from typing import Any

import pytest

from udlab_generator.cli import (
    main,
    resolve_prompt_token_counter,
)
from udlab_generator.diversity import count_template_diversity
from udlab_generator.render import (
    ACCEPTED_FORMS_LINE_TEXT,
    BELIEF_PERSISTENCE_RULE,
    DEFAULT_WRITER_MODEL_PIN,
    SHAPE_PLAN_ALL_ASSIGNED_FORMS,
    SHAPE_PLAN_BLOCK_SIZE,
    SHAPE_PLAN_DESIGN_CHOICES,
    SHAPE_PLAN_JSON_CONTEXT_TEMPLATES,
    SHAPE_PLAN_TAG_PAIRS,
    SHAPE_PLAN_VARIANT_FORMS,
    STAGE4_LEGEND_V1_PATH,
    STAGE4_LEGEND_V1_RAW_TEXT,
    STAGE4_LEGEND_V1_SHA256,
    STAGE4_LEGEND_V1_TEXT,
    TAG_ECHO_TOKENS,
    _TAG_ECHO_RE,
    _has_tag_echo,
    LlamaTokenizeCounter,
    MockWriter,
    RenderCheckError,
    check_prose_no_markers,
    check_rendered_prose,
    compute_hard_constraints_sha256,
    compute_shape_plan_run_telemetry,
    event_shape_plan_variant,
    format_writer_prompt,
    generate_shape_plan,
    match_rendering_event_forms,
    matched_event_form_id,
    project_event_for_writer,
    project_events_for_writer,
    render_world_template,
    render_writer_prompt_constraints,
    verify_prompt_and_checker_requirements_aligned,
    verify_shape_plan_json_context_token_parity,
    verify_shape_plan_prompt_token_parity,
    verify_shape_plan_tag_pairs_token_parity,
    wildcard_form_id_for_event,
    world_from_scenario_seed,
)
from udlab_generator.world import (
    AgentGoal,
    Event,
    StakeEdge,
    World,
    generate_world,
    make_balanced_configs,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "stage4" / "stage4_shape_plan_fixtures.json"
EXPECTED_FIXTURE_SHA256 = "3c427e7d3b28e9d09b306d0cf6a8f67f26dede7265ce7841e55bf46882f896af"
EXPECTED_LEGEND_SHA256 = "fb073147abacfd536a2ba7931cb4e178903a7fb33c3b0427b3d983985895564d"

PINNED_OFF_HARD_CONSTRAINTS_SHA256 = {
    ("copy", "line", "off"): "6a2b9c6f51c12f840fa80f60fe94f59ad20da41b66084731f75460b297820304",
    ("copy", "line", "on"): "68ba5db9756969c94be8ad05958050142508e3b25dbc14ff04065929a62d46e5",
    ("copy", "table", "off"): "67491e4b67e879ea4f5e8a5691e6e2615f6295d79cd0576c49e4f15462c44724",
    ("copy", "table", "on"): "b57f06d9d23eb1bdfb647bcb211836f3b485df5b865ce90a9da371ad4cd64143",
    ("splice", "line", "off"): "b570f664290705b9155f913203ecdb54bd8053910df0e330b4a2175c3f4bd944",
    ("splice", "line", "on"): "a626ccf388ea83f3912284b6b8506210df22dbfb7e0156103d53b4dea18607de",
    ("splice", "table", "off"): "284c8e95e0a1cff3ab2d864d7aa50d0b5c5c980d644259be1e0967c8bfaec671",
    ("splice", "table", "on"): "89c0bdf313066badf4f22ba2e09e40ca4c2a729a39acb46cf36410cc92641da8",
}

PINNED_OFF_FORMAT_WRITER_PROMPT_SHA256 = {
    ("copy", "line", "off"): "7a34d382917fff173eef92c3de8dae611cf7f769e2ebc8a1e45dbc79d3ed3282",
    ("copy", "line", "on"): "800d6cebbabd84752a345883759f181979751956104a73e3416a962687a9c760",
    ("copy", "table", "off"): "2e8896939a84ab348346373e69642bf6fe58d91f2aaecf3d53cd1b931f03edd2",
    ("copy", "table", "on"): "ba1a059da7b194eaad950e0affd6102821be3d717ba6ad5a384bcea8920d0000",
    ("splice", "line", "off"): "3e82b900ec6d3a7b62f51c79e70e314e0073f46984f5dc571233543b2bb09e33",
    ("splice", "line", "on"): "7f321b0969a20b8babfa07a8f14684e183bdc47695e571be65e90a99c599eead",
    ("splice", "table", "off"): "e42ecc85e776f0e6705e28f27642d132612d7237846f07a30acf935a254df235",
    ("splice", "table", "on"): "24e0e75c3e952429941aaea6d5f6b1e0b7a441ef2b6f95c5d3956b3a0aa89b48",
}


def _resolve_gguf_counter_or_skip() -> LlamaTokenizeCounter:
    """Resolve the CPU llama-tokenize counter against the pinned writer GGUF via env vars, or skip if unset."""
    counter = resolve_prompt_token_counter(None, None)
    if counter is not None:
        return counter
    pytest.skip(
        "llama-tokenize and the pinned GGUF blob are not configured (UDLAB_LLAMA_TOKENIZE, UDLAB_WRITER_GGUF)"
    )


def _build_fixture_world_and_events(fixture_data: dict[str, Any]) -> tuple[World, dict[str, Event]]:
    """Construct a World and Event map from `stage4_shape_plan_fixtures.json`."""
    base_world = world_from_scenario_seed(20261030, index=0)
    w_raw = fixture_data["world"]
    rooms = tuple(str(r) for r in w_raw["rooms"])
    doorways = tuple((str(a), str(b)) for a, b in w_raw["doorways"])
    agents = tuple(str(a) for a in w_raw["agents"])
    absent_parties = tuple(str(a) for a in w_raw["absent_parties"])
    objects = tuple(str(o) for o in w_raw["objects"])
    containers_by_room = {
        str(r): (str(pair[0]), str(pair[1]))
        for r, pair in w_raw["containers_by_room"].items()
    }
    occluders_by_room = {str(r): str(occ) for r, occ in w_raw["occluders_by_room"].items()}
    default_fixtures = ("wall bench", "corner desk", "floor rack")
    fixtures_by_room = {
        str(r): (str(fix) if fix is not None else default_fixtures[idx % len(default_fixtures)])
        for idx, (r, fix) in enumerate(w_raw["fixtures_by_room"].items())
    }
    noise_threshold = int(w_raw["noise_threshold"])

    events_map: dict[str, Event] = {}
    for ev_key, ev_raw in fixture_data["events"].items():
        events_map[ev_key] = Event(
            step=int(ev_raw["step"]),
            kind=str(ev_raw["kind"]),  # type: ignore[arg-type]
            actor=str(ev_raw["actor"]),
            room=str(ev_raw["room"]),
            target_object=ev_raw["target_object"],
            from_location=ev_raw["from_location"],
            to_location=ev_raw["to_location"],
            utterance=ev_raw["utterance"],
            noise_level=int(ev_raw["noise_level"] or 0),
            occluded=bool(ev_raw["occluded"]),
            occluder=ev_raw["occluder"],
            recipient=ev_raw["recipient"],
            stated_reason=ev_raw["stated_reason"],
        )

    goals = {
        ag: AgentGoal(
            agent=ag,
            goal_text=f"{ag} plans to inspect the {rooms[0]}.",
            causal_preconditions=("F1",),
        )
        for ag in agents
    }
    stake_edges = (
        StakeEdge(
            agent=absent_parties[0],
            variable="F1",
            description=(
                f"{absent_parties[0]}'s audit sign-off for {agents[0]} depends on the location of the {objects[0]}."
            ),
        ),
    )

    # The fixture world's "pine crate" comes from the "customs_wharf" domain.
    fixture_world = replace(
        base_world,
        scenario_id=str(w_raw["scenario_id"]),
        seed=int(w_raw["seed"]),
        domain_name="customs_wharf",
        rooms=rooms,
        doorways=doorways,
        agents=agents,
        absent_parties=absent_parties,
        objects=objects,
        containers_by_room=containers_by_room,
        occluders_by_room=occluders_by_room,
        fixtures_by_room=fixtures_by_room,
        noise_threshold=noise_threshold,
        initial_agent_rooms={ag: rooms[0] for ag in agents},
        goals=goals,
        stake_edges=stake_edges,
    )
    return fixture_world, events_map


def test_stage4_copied_files_sha256() -> None:
    """Verify byte-for-byte SHA-256 of copied legend and fixture files."""
    assert STAGE4_LEGEND_V1_PATH.is_file()
    actual_legend_sha = hashlib.sha256(STAGE4_LEGEND_V1_PATH.read_bytes()).hexdigest()
    assert actual_legend_sha == EXPECTED_LEGEND_SHA256
    assert STAGE4_LEGEND_V1_SHA256 == EXPECTED_LEGEND_SHA256

    assert FIXTURE_PATH.is_file()
    actual_fixture_sha = hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()
    assert actual_fixture_sha == EXPECTED_FIXTURE_SHA256

    fixture_data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert fixture_data["legend_sha256"] == EXPECTED_LEGEND_SHA256
    assert fixture_data["legend_text"] == STAGE4_LEGEND_V1_RAW_TEXT
    assert (
        fixture_data["plan_properties"]["two_form_switch_rate_band"]
        == SHAPE_PLAN_DESIGN_CHOICES["two_form_switch_rate_band"]
    )


def test_tag_echo_tokens_and_regex_agree() -> None:
    """Card 2b item 1: TAG_ECHO_TOKENS and _TAG_ECHO_RE agree 1-to-1 and the legend phrase has a word boundary at its end."""
    expected_tokens = (
        *(f"SAY_{n}" for n in range(10)),
        *(f"MOVE_{n}" for n in range(10)),
        *(f"SEE_{n}" for n in range(10)),
        "ENTER_0",
        "LEAVE_0",
        "MESSAGE_0",
        "[Actor]",
        "[Room]",
        "[Object]",
        "[Occluder]",
        "Shape key for the tag on each event row",
    )
    assert TAG_ECHO_TOKENS == expected_tokens

    # Every token in the tuple must match the regex
    for token in TAG_ECHO_TOKENS:
        assert _TAG_ECHO_RE.search(token) is not None, f"Token {token!r} in TAG_ECHO_TOKENS did not match _TAG_ECHO_RE"
        assert _has_tag_echo(f"In the amber room, {token} happened.")

    # Every alternative in the regex must be in the tuple
    pattern = _TAG_ECHO_RE.pattern
    assert pattern.startswith("(?:") and pattern.endswith(")")
    raw_alternatives = pattern[3:-1].split("|")
    decoded_alternatives: list[str] = []
    for alt in raw_alternatives:
        if alt.startswith(r"(?<!\w)") and alt.endswith(r"(?!\w)"):
            inner = alt[len(r"(?<!\w)") : -len(r"(?!\w)")]
        elif alt.startswith(r"\b") and alt.endswith(r"\b"):
            inner = alt[len(r"\b") : -len(r"\b")]
        else:
            pytest.fail(f"Alternative {alt!r} lacks expected word or bracket boundary wrapper")
        unescaped = re.sub(r"\\(.)", r"\1", inner)
        assert unescaped in TAG_ECHO_TOKENS, f"Regex alternative {unescaped!r} not in TAG_ECHO_TOKENS"
        decoded_alternatives.append(unescaped)
    assert tuple(decoded_alternatives) == TAG_ECHO_TOKENS

    # Word boundary at the end of the legend phrase and no unlisted ENTER_1/LEAVE_1/MESSAGE_1
    assert _has_tag_echo("Shape key for the tag on each event row.")
    assert not _has_tag_echo("Shape key for the tag on each event rows.")
    assert not _has_tag_echo("Shape key for the tag on each event rowboat.")
    for non_token in ("ENTER_1", "LEAVE_1", "MESSAGE_1"):
        assert non_token not in TAG_ECHO_TOKENS
        assert not _has_tag_echo(f"Mira walked ({non_token}) into the room.")


def test_shape_plan_off_byte_identical_across_all_8_combinations() -> None:
    """Gate 0 & Build 1: --shape-plan off is byte-identical to pre-change prompts across all 8 combinations."""
    w = world_from_scenario_seed(20261030, index=0)
    focal = w.agents[0]

    for (goal_mode, layout, afl), expected_hc_sha in PINNED_OFF_HARD_CONSTRAINTS_SHA256.items():
        got_hc_sha = compute_hard_constraints_sha256(
            goal_mode=goal_mode,
            constraint_layout=layout,
            accepted_forms_line=afl,
            shape_plan="off",
        )
        assert got_hc_sha == expected_hc_sha, f"Mismatch in hard_constraints_sha256 for {(goal_mode, layout, afl)}"

        got_default_hc_sha = compute_hard_constraints_sha256(
            goal_mode=goal_mode,
            constraint_layout=layout,
            accepted_forms_line=afl,
        )
        assert got_default_hc_sha == expected_hc_sha

        prompt_off = format_writer_prompt(
            w,
            focal,
            goal_mode=goal_mode,
            constraint_layout=layout,
            accepted_forms_line=afl,
            shape_plan="off",
        )
        prompt_default = format_writer_prompt(
            w,
            focal,
            goal_mode=goal_mode,
            constraint_layout=layout,
            accepted_forms_line=afl,
        )
        assert prompt_off == prompt_default
        got_prompt_sha = hashlib.sha256(prompt_off.encode("utf-8")).hexdigest()
        assert got_prompt_sha == PINNED_OFF_FORMAT_WRITER_PROMPT_SHA256[(goal_mode, layout, afl)]
        assert "Shape key for the tag on each event row" not in prompt_off
        assert "shape_tag" not in prompt_off


def test_legend_placement_and_tag_echo_refusal() -> None:
    """Gate 0 & Build 2: Legend placement directly after constraint table and tag_echo refusal."""
    for layout in ("table", "line"):
        for afl in ("off", "on"):
            for mode in ("assigned", "wildcard"):
                constraints = render_writer_prompt_constraints(
                    goal_mode="splice",
                    constraint_layout=layout,
                    accepted_forms_line=afl,
                    shape_plan=mode,
                )
                assert STAGE4_LEGEND_V1_TEXT in constraints
                legend_pos = constraints.index(STAGE4_LEGEND_V1_TEXT)
                missing_ev_pos = constraints.index("- (c_fidelity_missing_event)")
                assert legend_pos < missing_ev_pos
                if afl == "on":
                    afl_pos = constraints.index(ACCEPTED_FORMS_LINE_TEXT)
                    assert legend_pos < afl_pos < missing_ev_pos
                assert verify_prompt_and_checker_requirements_aligned(
                    goal_mode="splice",
                    constraint_layout=layout,
                    accepted_forms_line=afl,
                    shape_plan=mode,
                )

    # Verify matched_event_form_id and check_rendered_prose refuse every legend form token and header phrase with rule_id="tag_echo"
    fixture_data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    fixture_world, events_map = _build_fixture_world_and_events(fixture_data)
    ev_move = events_map["move"]
    single_ev_world = replace(fixture_world, events=(ev_move,))
    setup_para = render_world_template(single_ev_world, single_ev_world.agents[0]).split("\n\n", 1)[0]
    valid_move_sentence = str(fixture_data["legend_rows"][0]["sentence"])

    for token in (
        *SHAPE_PLAN_ALL_ASSIGNED_FORMS,
        "MOVE_0",
        "SEE_0",
        "SAY_0",
        "ENTER_0",
        "LEAVE_0",
        "MESSAGE_0",
        "Shape key for the tag on each event row",
    ):
        echo_sentence = f"{valid_move_sentence[:-1]} ({token})."
        assert (
            matched_event_form_id(
                echo_sentence,
                ev_move,
                single_ev_world.noise_threshold,
                world=single_ev_world,
                checker_profile="d",
            )
            is None
        )
        with pytest.raises(RenderCheckError) as exc_info:
            check_rendered_prose(
                f"{setup_para}\n\n{echo_sentence}",
                single_ev_world,
                focal_agent=single_ev_world.agents[0],
                goal_mode="copy",
                checker_profile="d",
            )
        assert exc_info.value.check_id == "c_fidelity_missing_event"
        assert exc_info.value.rule_id == "tag_echo"


def test_per_row_shape_tags_and_token_parity() -> None:
    """Gate 0 & Build 3 & 4: Per-row shape_tag placement, 14 tag pairs equal at 5 tokens each, and full prompt token parity."""
    fixture_data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    fixture_pairs = [(r["assigned"], r["wildcard"]) for r in fixture_data["tag_pairs"]]
    assert tuple(fixture_pairs) == SHAPE_PLAN_TAG_PAIRS
    assert len(fixture_pairs) == 14

    worlds = [world_from_scenario_seed(20261030 + i, index=i) for i in range(20)]
    plan_bundle = generate_shape_plan(worlds, seed=20261030, block_size=20)
    plan_by_sid = plan_bundle["plan_by_scenario_id"]

    for w in worlds[:3]:
        scen_plan = plan_by_sid[w.scenario_id]
        assigned_tags = [str(r["form_id"]) for r in scen_plan]
        wildcard_tags = [wildcard_form_id_for_event(ev, w) for ev in w.events]

        rows_off = project_events_for_writer(w.events, world=w)
        rows_assigned = project_events_for_writer(w.events, world=w, shape_tags=assigned_tags)
        rows_wildcard = project_events_for_writer(w.events, world=w, shape_tags=wildcard_tags)

        for ev, r_off, r_asg, r_wld, p_rec in zip(
            w.events, rows_off, rows_assigned, rows_wildcard, scen_plan, strict=True
        ):
            assert "shape_tag" not in r_off
            assert r_asg["shape_tag"] == p_rec["form_id"]
            assert r_wld["shape_tag"] == wildcard_form_id_for_event(ev, w)
            if p_rec["variant"] not in SHAPE_PLAN_VARIANT_FORMS:
                assert r_asg["shape_tag"] == r_wld["shape_tag"]

    # In-process token counter coverage (always runs in CI) plus mismatch refusal checks
    inproc_counter = lambda text: len(text.split())
    inproc_tag_rows = verify_shape_plan_tag_pairs_token_parity(inproc_counter)
    assert len(inproc_tag_rows) == 14
    assert all(r["equal"] is True for r in inproc_tag_rows)

    inproc_json_rows = verify_shape_plan_json_context_token_parity(inproc_counter)
    assert len(inproc_json_rows) == 14 * len(SHAPE_PLAN_JSON_CONTEXT_TEMPLATES)
    assert all(r["equal"] is True for r in inproc_json_rows)

    inproc_prompt_parity = verify_shape_plan_prompt_token_parity(
        worlds[0],
        worlds[0].agents[0],
        inproc_counter,
        goal_mode="splice",
        constraint_layout="table",
        accepted_forms_line="off",
        scenario_shape_plan=plan_by_sid[worlds[0].scenario_id],
        shape_plan_seed=20261030,
        think=False,
    )
    assert inproc_prompt_parity["equal"] is True

    call_seq = iter(range(1, 100))
    skewed_counter = lambda _text: next(call_seq)
    with pytest.raises(ValueError, match="STOPPED: tag pair"):
        verify_shape_plan_tag_pairs_token_parity(skewed_counter)
    with pytest.raises(ValueError, match="STOPPED: JSON context"):
        verify_shape_plan_json_context_token_parity(skewed_counter)
    with pytest.raises(ValueError, match="STOPPED: prompt token parity mismatch"):
        verify_shape_plan_prompt_token_parity(
            worlds[0],
            worlds[0].agents[0],
            skewed_counter,
            scenario_shape_plan=plan_by_sid[worlds[0].scenario_id],
            shape_plan_seed=20261030,
        )


def test_json_context_token_parity_on_tag_pairs() -> None:
    """Card 2b item 2: CPU llama-tokenize token parity (tolerance 0) for all 14 (assigned, wildcard) pairs in real JSON event-log contexts across >=3 key-order prefixes and all emitted following-key suffixes."""
    assert len(SHAPE_PLAN_JSON_CONTEXT_TEMPLATES) >= 5
    # Verify the templates cover at least 3 distinct key-order prefixes and the following-key suffixes
    prefixes = {tmpl.split('"shape_tag":"<TAG>"', 1)[0] for tmpl in SHAPE_PLAN_JSON_CONTEXT_TEMPLATES.values()}
    suffixes = {tmpl.split('"shape_tag":"<TAG>"', 1)[1] for tmpl in SHAPE_PLAN_JSON_CONTEXT_TEMPLATES.values()}
    assert len(prefixes) >= 3
    assert ',"target_object":"copper token","to_location":"tin locker"}' in suffixes
    assert ',"target_object":"copper token"}' in suffixes
    assert ',"utterance":"The copper token is stored in the tin locker."}' in suffixes
    assert ',"stated_reason":"audit check","utterance":"The copper token is stored in the tin locker."}' in suffixes
    assert "}" in suffixes

    counter = _resolve_gguf_counter_or_skip()
    json_parity_rows = verify_shape_plan_json_context_token_parity(counter)
    assert len(json_parity_rows) == 14 * len(SHAPE_PLAN_JSON_CONTEXT_TEMPLATES)
    for row in json_parity_rows:
        assert row["equal"] is True
        assert row["assigned_tokens"] == row["wildcard_tokens"]

    # Also verify real project_event_for_writer output on fixture events under sort_keys=True and sort_keys=False
    fixture_data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    fixture_world, events_map = _build_fixture_world_and_events(fixture_data)
    for ev_key, variant_key, wildcard_tag in (
        ("move", "move_normal", "MOVE_0"),
        ("move_occ", "move_occluded", "MOVE_0"),
        ("see_clear", "see_unoccluded", "SEE_0"),
        ("see_occ", "see_occluded", "SEE_0"),
        ("say", "say", "SAY_0"),
    ):
        ev = events_map[ev_key]
        for sort_keys in (True, False):
            w_proj = project_event_for_writer(ev, order_index=1, world=fixture_world, shape_tag=wildcard_tag)
            w_json = json.dumps(w_proj, separators=(",", ":"), sort_keys=sort_keys)
            w_tok = counter(w_json)
            for assigned_tag in SHAPE_PLAN_VARIANT_FORMS[variant_key]:
                a_proj = project_event_for_writer(ev, order_index=1, world=fixture_world, shape_tag=assigned_tag)
                a_json = json.dumps(a_proj, separators=(",", ":"), sort_keys=sort_keys)
                a_tok = counter(a_json)
                assert a_tok == w_tok, f"JSON parity mismatch for {ev_key} sort_keys={sort_keys}: {assigned_tag}={a_tok} vs {wildcard_tag}={w_tok}"


def test_gguf_token_parity_on_tag_pairs_and_prompts() -> None:
    """Gate 0 & Build 4: CPU llama-tokenize token parity on all 14 tag pairs (5 tokens each) and full prompts against the pinned GGUF."""
    worlds = [world_from_scenario_seed(20261030 + i, index=i) for i in range(5)]
    plan_bundle = generate_shape_plan(worlds, seed=20261030, block_size=5)
    plan_by_sid = plan_bundle["plan_by_scenario_id"]

    counter = _resolve_gguf_counter_or_skip()
    tag_parity_rows = verify_shape_plan_tag_pairs_token_parity(counter)
    assert len(tag_parity_rows) == 14
    for row in tag_parity_rows:
        assert row["equal"] is True
        assert row["assigned_tokens"] == 5
        assert row["wildcard_tokens"] == 5

    for w in worlds:
        scen_plan = plan_by_sid[w.scenario_id]
        for focal in w.agents[:2]:
            parity = verify_shape_plan_prompt_token_parity(
                w,
                focal,
                counter,
                goal_mode="splice",
                constraint_layout="table",
                accepted_forms_line="off",
                scenario_shape_plan=scen_plan,
                shape_plan_seed=20261030,
                think=False,
            )
            assert parity["equal"] is True
            assert parity["assigned_raw_tokens"] == parity["wildcard_raw_tokens"]
            assert parity["assigned_chat_tokens"] == parity["wildcard_chat_tokens"]


def test_seeded_shape_plan_properties_on_200_scenario_synthetic_bank() -> None:
    """Gate 0 & Build 5 & Card 2b items 3, 4, 7: 200-scenario bank of distinct worlds, plan properties, 0.60-0.71 switch band, step-relabel invariance, and sequence-dependent run-length draw with flat tercile marginals."""
    fixture_data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    plan_props = fixture_data["plan_properties"]
    expected_fields = tuple(plan_props["plan_record_fields"])

    # Card 2b item 3: 200 distinct worlds built with make_balanced_configs and generate_world
    bank_configs = make_balanced_configs(200, base_seed=20261103)
    bank_worlds = [generate_world(cfg) for cfg in bank_configs]
    div_report = count_template_diversity(bank_worlds)
    distinct_variant_sigs = len({tuple(event_shape_plan_variant(ev, w) for ev in w.events) for w in bank_worlds})
    distinct_event_tuples = len({tuple(w.events) for w in bank_worlds})
    assert div_report.distinct_event_skeletons == 186
    assert div_report.distinct_domain_skeletons == 195
    assert distinct_variant_sigs == 102
    assert distinct_event_tuples == 200

    plan_a = generate_shape_plan(bank_worlds, seed=20261103, block_size=SHAPE_PLAN_BLOCK_SIZE)
    plan_b = generate_shape_plan(bank_worlds, seed=20261103, block_size=SHAPE_PLAN_BLOCK_SIZE)
    assert plan_a == plan_b

    # Design choice labels present (Card 2b item 7: band 0.60-0.71 with basis)
    assert plan_a["block_balance_max_spread"]["value"] == 1
    assert plan_a["block_balance_max_spread"]["design_choice"] is True
    assert plan_a["two_form_switch_rate_band"]["low"] == 0.6
    assert plan_a["two_form_switch_rate_band"]["high"] == 0.71
    assert plan_a["two_form_switch_rate_band"]["expected"] == 0.6667
    assert plan_a["two_form_switch_rate_band"]["design_choice"] is True
    assert (
        plan_a["two_form_switch_rate_band"]["basis"]
        == "widened 2026-11-02 before freeze; 60 seeds on the 80-scenario dev bank gave mean 0.647, min 0.6227"
    )

    # Plan record fields
    for rec in plan_a["plan_records"]:
        assert tuple(rec.keys()) == expected_fields

    # Check every 20-scenario block for spread <= 1 per variant and no 3-in-a-row same-kind same-form
    plan_by_sid = plan_a["plan_by_scenario_id"]
    for b_start in range(0, 200, 20):
        block_worlds = bank_worlds[b_start : b_start + 20]
        counts_by_variant: dict[str, Counter[str]] = {v: Counter() for v in SHAPE_PLAN_VARIANT_FORMS}
        for w in block_worlds:
            scen_recs = plan_by_sid[w.scenario_id]
            last_by_kind: dict[str, list[str]] = {}
            for rec in scen_recs:
                v = str(rec["variant"])
                k = str(rec["kind"])
                fid = str(rec["form_id"])
                if v in SHAPE_PLAN_VARIANT_FORMS:
                    counts_by_variant[v][fid] += 1
                    hist = last_by_kind.setdefault(k, [])
                    hist.append(fid)
                    if len(hist) >= 3:
                        assert not (hist[-1] == hist[-2] == hist[-3]), (
                            f"Three-in-a-row violation in {w.scenario_id} for kind {k}: {hist[-3:]}"
                        )
                elif v == "move_station_change":
                    last_by_kind.setdefault("move", []).append("MOVE_0")
        for v, forms in SHAPE_PLAN_VARIANT_FORMS.items():
            if sum(counts_by_variant[v].values()) > 0:
                form_counts = [counts_by_variant[v][f] for f in forms]
                spread = max(form_counts) - min(form_counts)
                assert spread <= 1, f"Block {b_start // 20} variant {v} spread {spread} > 1: {counts_by_variant[v]}"

    # Two-form switch rate within [0.60, 0.71]
    two_form_switch = float(plan_a["two_form_switch_rate"])
    assert 0.60 <= two_form_switch <= 0.71, f"two_form_switch_rate={two_form_switch} outside [0.60, 0.71]"

    # One plan per scenario shared across both renderings r0 and r1
    w0 = bank_worlds[0]
    scen_plan_0 = plan_by_sid[w0.scenario_id]
    p_r0 = format_writer_prompt(
        w0,
        w0.agents[0],
        goal_mode="splice",
        constraint_layout="table",
        accepted_forms_line="off",
        shape_plan="assigned",
        scenario_shape_plan=scen_plan_0,
    )
    p_r1 = format_writer_prompt(
        w0,
        w0.agents[1],
        goal_mode="splice",
        constraint_layout="table",
        accepted_forms_line="off",
        shape_plan="assigned",
        scenario_shape_plan=scen_plan_0,
    )
    e_r0 = json.loads(p_r0.split("Complete Event Log:\n", 1)[1].split("\n\nOUTPUT FORMAT:", 1)[0])
    e_r1 = json.loads(p_r1.split("Complete Event Log:\n", 1)[1].split("\n\nOUTPUT FORMAT:", 1)[0])
    assert [item["shape_tag"] for item in e_r0] == [item["shape_tag"] for item in e_r1]

    # Card 2b item 4: Position & step tests.
    # Note: generate_shape_plan never reads ev.step or event_index when drawing forms,
    # while the draw is sequence-dependent through the 3-in-a-row run-length rejection rule
    # and per-variant stream order (not order-independent).
    for fn in (generate_shape_plan, event_shape_plan_variant):
        assert not any(
            isinstance(node, ast.Attribute) and node.attr == "step"
            for node in ast.walk(ast.parse(inspect.getsource(fn)))
        )

    # Relabelling ev.step leaves the plan identical across the entire 200-scenario bank
    relabeled_worlds = [
        replace(w, events=tuple(replace(ev, step=1000 + idx * 13) for idx, ev in enumerate(w.events)))
        for w in bank_worlds
    ]
    plan_relabeled = generate_shape_plan(relabeled_worlds, seed=20261103, block_size=SHAPE_PLAN_BLOCK_SIZE)
    assert plan_relabeled == plan_a

    # Reordering assigned events in their slots across the 200-scenario bank preserves flat form marginals by tercile
    reordered_assigned_worlds: list[World] = []
    for w in bank_worlds:
        assigned_evs = [ev for ev in w.events if event_shape_plan_variant(ev, w) in SHAPE_PLAN_VARIANT_FORMS]
        reversed_assigned = iter(reversed(assigned_evs))
        new_events = tuple(
            next(reversed_assigned) if event_shape_plan_variant(ev, w) in SHAPE_PLAN_VARIANT_FORMS else ev
            for ev in w.events
        )
        reordered_assigned_worlds.append(replace(w, events=new_events))

    plan_reordered = generate_shape_plan(reordered_assigned_worlds, seed=20261103, block_size=SHAPE_PLAN_BLOCK_SIZE)
    for ws, p_bundle in ((bank_worlds, plan_a), (reordered_assigned_worlds, plan_reordered)):
        by_sid = p_bundle["plan_by_scenario_id"]
        tercile_counts: dict[str, dict[str, Counter[str]]] = {
            v: {t: Counter() for t in ("T1", "T2", "T3")}
            for v in SHAPE_PLAN_VARIANT_FORMS
        }
        for w in ws:
            n_ev = len(w.events)
            for rec in by_sid[w.scenario_id]:
                v = str(rec["variant"])
                if v in SHAPE_PLAN_VARIANT_FORMS:
                    ev_idx = int(rec["event_index"]) - 1
                    t_label = f"T{min(2, (ev_idx * 3) // max(1, n_ev)) + 1}"
                    tercile_counts[v][t_label][str(rec["form_id"])] += 1
        for v, forms in SHAPE_PLAN_VARIANT_FORMS.items():
            expected_share = 1.0 / len(forms)
            for t_label in ("T1", "T2", "T3"):
                tot = sum(tercile_counts[v][t_label].values())
                if tot >= 100:
                    for f in forms:
                        share = tercile_counts[v][t_label][f] / tot
                        assert abs(share - expected_share) <= 0.08, (
                            f"Non-flat marginal for {v} {t_label} {f}: {share:.3f} vs {expected_share:.3f}"
                        )

    # Moving non-assigned events (enter, leave, message) and renumbering steps leaves per-variant draws unchanged
    permuted_events = list(w0.events)
    non_assigned = [ev for ev in permuted_events if event_shape_plan_variant(ev, w0) not in SHAPE_PLAN_VARIANT_FORMS]
    assigned_only = [ev for ev in permuted_events if event_shape_plan_variant(ev, w0) in SHAPE_PLAN_VARIANT_FORMS]
    reordered_events = tuple(
        replace(ev, step=idx + 10)
        for idx, ev in enumerate(non_assigned + assigned_only)
    )
    w0_permuted = replace(w0, events=reordered_events)
    plan_orig_1 = generate_shape_plan([w0], seed=20261103, block_size=1)
    plan_perm_1 = generate_shape_plan([w0_permuted], seed=20261103, block_size=1)
    for v in SHAPE_PLAN_VARIANT_FORMS:
        forms_orig = [r["form_id"] for r in plan_orig_1["plan_records"] if r["variant"] == v]
        forms_perm = [r["form_id"] for r in plan_perm_1["plan_records"] if r["variant"] == v]
        assert forms_orig == forms_perm


def test_shape_plan_block_size_rule_full_and_partial_blocks() -> None:
    """Card 2b item 5: Blocks of 20 scenarios in bank order; 80 and 120 give full blocks, 335 gives 16 full + 1 of 15, and 230 works (11 full + 1 of 10), each block balanced with spread <= 1."""
    expected_block_sizes_by_count = {
        80: [20] * 4,
        120: [20] * 6,
        335: [20] * 16 + [15],
        230: [20] * 11 + [10],
    }
    for count, expected_block_sizes in expected_block_sizes_by_count.items():
        worlds = [generate_world(cfg) for cfg in make_balanced_configs(count, base_seed=20261103)]
        plan = generate_shape_plan(worlds, seed=20261103)
        assert plan["block_size"] == SHAPE_PLAN_BLOCK_SIZE == 20
        assert plan["block_sizes"] == expected_block_sizes
        plan_by_sid = plan["plan_by_scenario_id"]

        for b_idx, b_start in enumerate(range(0, len(worlds), SHAPE_PLAN_BLOCK_SIZE)):
            block_worlds = worlds[b_start : b_start + SHAPE_PLAN_BLOCK_SIZE]
            assert len(block_worlds) == expected_block_sizes[b_idx]
            counts_by_variant: dict[str, Counter[str]] = {v: Counter() for v in SHAPE_PLAN_VARIANT_FORMS}
            for w in block_worlds:
                last_by_kind: dict[str, list[str]] = {}
                for rec in plan_by_sid[w.scenario_id]:
                    v = str(rec["variant"])
                    k = str(rec["kind"])
                    fid = str(rec["form_id"])
                    if v in SHAPE_PLAN_VARIANT_FORMS:
                        counts_by_variant[v][fid] += 1
                        hist = last_by_kind.setdefault(k, [])
                        hist.append(fid)
                        if len(hist) >= 3:
                            assert not (hist[-1] == hist[-2] == hist[-3])
                    elif v == "move_station_change":
                        last_by_kind.setdefault("move", []).append("MOVE_0")
            for v, forms in SHAPE_PLAN_VARIANT_FORMS.items():
                if sum(counts_by_variant[v].values()) > 0:
                    form_counts = [counts_by_variant[v][f] for f in forms]
                    assert max(form_counts) - min(form_counts) <= 1, (
                        f"count={count} block={b_idx} (len={len(block_worlds)}) variant={v} spread > 1: {counts_by_variant[v]}"
                    )

    # Also verify compute_shape_plan_run_telemetry uses block_size=20 on a non-multiple bank when plan_bundle is omitted
    worlds_25 = [generate_world(cfg) for cfg in make_balanced_configs(25, base_seed=20261103)]
    tel_25 = compute_shape_plan_run_telemetry(worlds_25, shape_plan="assigned", shape_plan_seed=20261103)
    assert tel_25["block_size"] == 20
    assert tel_25["block_sizes"] == [20, 5]


def test_realized_form_exposure_on_all_18_fixture_legend_rows() -> None:
    """Gate 0 & Build 5 & 6: All 18 legend_rows in stage4_shape_plan_fixtures.json are accepted under Profile D, have 0 echoes, and match expected form_id."""
    fixture_data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert fixture_data["variant_forms"] == {
        k: list(v) for k, v in SHAPE_PLAN_VARIANT_FORMS.items()
    }
    fixture_world, events_map = _build_fixture_world_and_events(fixture_data)
    legend_rows = fixture_data["legend_rows"]
    assert len(legend_rows) == 18

    for row in legend_rows:
        expected_form_id = str(row["form_id"])
        ev_key = str(row["event"])
        sentence = str(row["sentence"])
        assert row["expect"] == "accepted_profile_d"

        # 0 echoes in fixture sentence
        check_prose_no_markers(sentence)

        ev = events_map[ev_key]
        single_ev_world = replace(fixture_world, events=(ev,))

        got_form = matched_event_form_id(
            sentence,
            ev,
            single_ev_world.noise_threshold,
            world=single_ev_world,
            checker_profile="d",
        )
        assert got_form == expected_form_id, (
            f"Fixture row {expected_form_id} ({ev_key}): got {got_form!r} for {sentence!r}"
        )

        # Also test full rendering event-form matching via match_rendering_event_forms
        setup_para = render_world_template(single_ev_world, single_ev_world.agents[0]).split("\n\n", 1)[0]
        full_prose = f"{setup_para}\n\n{sentence}"
        check_rendered_prose(
            full_prose,
            single_ev_world,
            focal_agent=single_ev_world.agents[0],
            goal_mode="copy",
            checker_profile="d",
        )
        ev_outcomes = match_rendering_event_forms(
            full_prose,
            single_ev_world,
            focal_agent=single_ev_world.agents[0],
            goal_mode="copy",
            checker_profile="d",
        )
        assert len(ev_outcomes) == 1
        assert ev_outcomes[0]["realized_form_id"] == expected_form_id
        assert ev_outcomes[0]["outcome"] == expected_form_id


def test_shape_plan_telemetry_and_confusion_matrix(tmp_path: Path) -> None:
    """Gate 0 & Build 6: Telemetry and confusion matrix per kind, position tercile, and assigned form."""
    fixture_data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    fixture_world, events_map = _build_fixture_world_and_events(fixture_data)
    by_event_and_form = {
        (str(r["event"]), str(r["form_id"])): str(r["sentence"])
        for r in fixture_data["legend_rows"]
    }

    # Build a 3-event world (move, see_clear, say) using the fixture events
    three_events = (
        replace(events_map["move"], step=1),
        replace(events_map["see_clear"], step=2),
        replace(events_map["say"], step=3),
    )
    w = replace(fixture_world, events=three_events)
    setup_r0 = render_world_template(w, w.agents[0]).split("\n\n", 1)[0]
    setup_r1 = render_world_template(w, w.agents[1]).split("\n\n", 1)[0]

    plan_bundle = generate_shape_plan([w], seed=20261103, block_size=1)
    scen_plan = plan_bundle["plan_by_scenario_id"][w.scenario_id]
    assigned_move = str(scen_plan[0]["form_id"])
    assigned_see = str(scen_plan[1]["form_id"])
    assigned_say = str(scen_plan[2]["form_id"])

    # 1. Compliant prose: realizes exact assigned forms on both renderings
    compliant_events_para = " ".join(
        [
            by_event_and_form[("move", assigned_move)],
            by_event_and_form[("see_clear", assigned_see)],
            by_event_and_form[("say", assigned_say)],
        ]
    )
    prose_Map_compliant = {
        (w.scenario_id, 0): f"{setup_r0}\n\n{compliant_events_para}",
        (w.scenario_id, 1): f"{setup_r1}\n\n{compliant_events_para}",
    }
    tel_diag = compute_shape_plan_run_telemetry(
        [w],
        shape_plan="assigned",
        shape_plan_seed=20261103,
        plan_bundle=plan_bundle,
        attempt0_prose_by_rendering=prose_Map_compliant,
        final_prose_by_rendering=prose_Map_compliant,
        goal_mode="copy",
        checker_profile="d",
    )
    assert tel_diag["baseline_only"] is False
    assert tel_diag["compliance_by_kind"]["overall"]["compliance_rate"] == 1.0
    assert tel_diag["confusion_matrix_by_kind"]["move"][assigned_move][assigned_move] == 2
    assert tel_diag["confusion_matrix_by_kind"]["see"][assigned_see][assigned_see] == 2
    assert tel_diag["confusion_matrix_by_kind"]["say"][assigned_say][assigned_say] == 2
    assert tel_diag["outcomes_by_assigned_form"][assigned_move]["first_attempt"]["compliant"] == 2

    # Card 2b item 6: Off-mode telemetry carries "baseline_only": True
    tel_off = compute_shape_plan_run_telemetry(
        [w],
        shape_plan="off",
        shape_plan_seed=20261103,
        plan_bundle=plan_bundle,
        attempt0_prose_by_rendering=prose_Map_compliant,
        final_prose_by_rendering=prose_Map_compliant,
        goal_mode="copy",
        checker_profile="d",
    )
    assert tel_off["baseline_only"] is True
    assert tel_off["shape_plan"] == "off"

    # 2. Non-compliant / faulted prose:
    # - move uses a different valid form (e.g. MOVE_1 if assigned != MOVE_1 else MOVE_2) -> off-diagonal realized form
    # - see uses an unaccepted form (actor not subject) -> form_not_accepted / other_fault
    # - say echoes a tag token -> tag_echo
    alt_move = "MOVE_2" if assigned_move == "MOVE_1" else "MOVE_1"
    mixed_events_para = " ".join(
        [
            by_event_and_form[("move", alt_move)],
            "In the amber room, the copper token was seen in the pine crate by Mira.",
            f"Mira said ({assigned_say}) at 55 dB in the amber room, \"The copper token is stored in the tin locker.\"",
        ]
    )
    other_fault_events_para = " ".join(
        [
            "Mira moved the glass prism from the pine crate to the tin locker in the amber room.",
            by_event_and_form[("see_clear", assigned_see)],
            by_event_and_form[("say", assigned_say)],
        ]
    )
    prose_map_mixed = {
        (w.scenario_id, 0): f"{setup_r0}\n\n{mixed_events_para}",
        (w.scenario_id, 1): f"{setup_r1}\n\n{other_fault_events_para}",
    }
    tel_mixed = compute_shape_plan_run_telemetry(
        [w],
        shape_plan="assigned",
        shape_plan_seed=20261103,
        plan_bundle=plan_bundle,
        attempt0_prose_by_rendering=prose_map_mixed,
        final_prose_by_rendering=prose_map_mixed,
        goal_mode="copy",
        checker_profile="d",
    )
    assert tel_mixed["confusion_matrix_by_kind"]["move"][assigned_move][alt_move] == 1
    assert tel_mixed["confusion_matrix_by_kind"]["move"][assigned_move]["other_fault"] == 1
    assert tel_mixed["confusion_matrix_by_kind"]["see"][assigned_see]["form_not_accepted"] == 1
    assert tel_mixed["confusion_matrix_by_kind"]["say"][assigned_say]["tag_echo"] == 1

    # 3. End-to-end CLI build_and_probe_dataset with --shape-plan assigned and --shape-plan off
    out_dir = tmp_path / "assigned_build"
    rc = main(
        [
            "--count",
            "6",
            "--seed",
            "20261103",
            "--out-dir",
            str(out_dir),
            "--writer",
            "mock",
            "--checker-profile",
            "d",
            "--constraint-layout",
            "table",
            "--shape-plan",
            "assigned",
            "--shape-plan-seed",
            "20261103",
        ]
    )
    assert rc == 0
    summary = json.loads((out_dir / "build_summary.json").read_text(encoding="utf-8"))
    assert summary["shape_plan"] == "assigned"
    assert summary["shape_plan_seed"] == 20261103
    assert "shape_plan_telemetry" in summary
    assert summary["shape_plan_telemetry"]["baseline_only"] is False
    assert summary["shape_plan_telemetry"]["block_size"] == 20
    assert summary["shape_plan_telemetry"]["block_sizes"] == [6]
    assert summary["two_form_switch_rate_band"] == SHAPE_PLAN_DESIGN_CHOICES["two_form_switch_rate_band"]
    assert "plan_records" in summary
    assert "lag1_persistence_by_kind" in summary
    assert "compliance_by_kind" in summary
    assert "compliance_by_position_tercile" in summary
    assert "confusion_matrix_by_kind" in summary
    assert "outcomes_by_assigned_form" in summary
    assert summary["recount_verification"]["prose_checks_passed"] is True

    out_dir_off = tmp_path / "off_build"
    rc_off = main(
        [
            "--count",
            "4",
            "--seed",
            "20261103",
            "--out-dir",
            str(out_dir_off),
            "--writer",
            "mock",
            "--checker-profile",
            "d",
            "--constraint-layout",
            "table",
            "--shape-plan",
            "off",
        ]
    )
    assert rc_off == 0
    summary_off = json.loads((out_dir_off / "build_summary.json").read_text(encoding="utf-8"))
    assert summary_off["shape_plan"] == "off"
    assert summary_off["shape_plan_telemetry"]["baseline_only"] is True


def test_80_scenario_prompts_byte_identical_to_40c7dc5e() -> None:
    """Card 2b proof: across the 80 dev-bank-shaped synthetic scenarios, rendered prompts under off, assigned, and wildcard are byte-identical to 40c7dc5e."""
    expected_80_bank_prompt_sha256_at_40c7dc5e = {
        "off": "08085c820abc297d2cd6f6be53888b1f466ba44188f9d68c524a2d7672e0673c",
        "assigned": "efdfc43b88b937ce293994bdc0ff327adc873562cfaa2730205764806d4ae246",
        "wildcard": "7c117487729fa576f37beddcc38d0fd38946446000e3ee418e31d271aebac4b4",
    }
    worlds_80 = [generate_world(cfg) for cfg in make_balanced_configs(80, base_seed=20261103)]
    plan_80 = generate_shape_plan(worlds_80, seed=20261103, block_size=SHAPE_PLAN_BLOCK_SIZE)
    by_sid_80 = plan_80["plan_by_scenario_id"]

    for mode, expected_digest in expected_80_bank_prompt_sha256_at_40c7dc5e.items():
        h = hashlib.sha256()
        for w in worlds_80:
            sp = by_sid_80[w.scenario_id]
            for focal in w.agents[:2]:
                p = format_writer_prompt(
                    w,
                    focal,
                    goal_mode="splice",
                    constraint_layout="table",
                    accepted_forms_line="off",
                    shape_plan=mode,
                    scenario_shape_plan=sp,
                    shape_plan_seed=20261103,
                )
                h.update(p.encode("utf-8"))
                h.update(b"\0")
        assert h.hexdigest() == expected_digest, f"80-scenario bank digest mismatch for mode={mode}"

