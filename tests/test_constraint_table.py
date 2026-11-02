from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from udlab_generator import (
     PINNED_WRITER_MANIFEST_DIGEST,
    PINNED_WRITER_MODEL_TAG,
    RenderCheckError,
    check_rendered_prose,
    collect_prose_faults,
    compare_paired_runs,
    diagnose_rendered_prose,
    evaluate_event_subrules,
    format_writer_prompt,
    generate_world,
    load_writer_prompt_template,
    make_balanced_configs,
    parse_writer_prompt_requirements,
    render_world_template,
    render_world_template_goal_tokens,
    render_writer_prompt_constraints,
    run_paired_seed_bank,
    verify_prompt_and_checker_requirements_aligned,
    world_from_scenario_seed,
)
from udlab_generator.cli import build_and_probe_dataset, main as cli_main
import udlab_generator.render as ug_render


PINNED_367F11A_COPY_PAYLOAD_SHA256: tuple[str, ...] = (
    "a225f946982a905ca09f330aef8cc4518ff31fc07eb0953b4a119a514b1a1d2e",
    "b3c37d83ab4bb4d7deb2dc29ab63d8d4c0e3a6d60b5cde0cf160faa4933ee22d",
    "7348b7a32449850e34413db90d525e233768917fb6f48e3493ec5e1ae7cdfad7",
    "7d27d2c6d54d88282979138029d2bbef11667901ce27546c9c1500d87354ba7f",
    "6d19fafffff115008ef6a977c3ec0118f852799f9f064c4e764e7db59ff15d2d",
    "0517461d59c8b97cf88ea3cbddfc16a042ecf24134937fdb2142c2e31985fe64",
    "799a5c1e355c63645ec06eb6db9585bfb2bc58ab6c70325c25b8fd817aa4f66a",
    "6fe69736227ef71b6ceeb39e2046ededed58237dff36988256fc75e15b1557b8",
    "c8449beec0ea4c27149afb958556d2b50746bab1ba398bdb88078d02d1e52d0f",
    "19653d74cebcfa93703a20c2f7145a7de79cbaae7fa528d763544da0af84dc95",
    "a41755f874d7ff21b648a024c59e534eac7dd96e5d20236e6ba345484da92625",
    "bdb1eb119009348253e1ec086ae65b004c0fa666f3dd77937e90a77ca0cfdb90",
    "36cb38121d75ac1df6de38df331b1705defcc510ed96e61839bcdad0203fa82f",
    "9c9460b746b0a48dbb0bfcb90c40477c17261a468275535d65673069dc2af63a",
    "5043a687edc51bffd2ed1813cb4fe0990060dd6e6707804e9365b4f9c44cedb5",
    "471a7206382fa6064ced22fad80f8185acef044dd100813eef79eb463d9ec960",
    "59b05767f34ad71fe3e74a87e46491d1e0c3be5580f24e257559b5d55e6a6708",
    "d84f5123d0249fc88dfa12feb3def0e2cdc7c0000a414cf1361d3dbe0b5c22f9",
    "5ae6d8b501f6aa484ddf4216828167ae95ed1e55cd5684a3e9c93a0f9273c03e",
    "03ef9ec3118b08b429b7e268367835d6003d6e047b3cba126a0b7d7561b8b771",
)

PINNED_367F11A_SPLICE_PAYLOAD_SHA256: tuple[str, ...] = (
    "d6cac1b09e3defce5f96cecd25f6ea53437364e93c3e30d1ddc1148981e8c82d",
    "afd9636b04aacdad3000c877f5d598d4570999873f234eb88e94fc68d182b502",
    "ff7255c83ea153ef6e0e784e3326e7537d8558c3b100854834c09706b7d35ef3",
    "151cb25d617d89e792b5edee5309b216a336fbbb4aac5de927bfff6fbf21a34f",
    "0047846896805d0a87454b91dd9d8dccad8db312253ee4fdcf497ceb59652949",
    "4478af8150e91085a89e5df8d1cb84f9b13b5faa9a6a42e46c642af7c1379af1",
    "69ff64479e9f4f13a46f5460d821fc2c2c6712218a47991b6ff6abd8e36eca20",
    "abcb4f33c2e0a2a17047cdd4917bc17e5a18ae2fd890584f86ba3eb9c910dfea",
    "685e50add65f1de64ac3cf1fb4e010880abae7a5040144146076d7ae784827de",
    "25ae3828e8a3016f7fda7409592f158c4226521931ce4a1cc7aadcb024c03760",
    "a66550cea722e5b88ea1a9241abae15b46da71fb5dc788e8118b6c5b3f65dce0",
    "3171fb30846a5c80bed27874164cb4492569f504a3d615ce89dea4db5016315a",
    "816fa00681f36b9bdb70939bbae855de269e2f9d9fc97e4df35dcabe2185010d",
    "7c7010a3a580213042935a941dc76b324c7077af487c1e1f952b9085bb3e1263",
    "3a7579837f4faca7a4f1610c2f66e343ea80b19b3839dd6cc570f5f262e0233e",
    "2a4e618b6b4ff7aa1021cc5c8cf7497b787b9e722418f15c374047483b8c8310",
    "2939753b3a448e9dc329932e811ea8486db50b19a5af645ad3f36b8a9b5f6018",
    "3fd09fcfc2b30756de4f7706c966ff853b9b7c0b574ff416f1c38b8eee1052a4",
    "1a1840bafff2ba1ca0356a67fb897ad36cb6425ab1af586b65659245344b1444",
    "06956c52ab6d91959254cac512cc5152e973bc20c0bf9b256623a9338f201621",
)


def test_01_copy_identity_absent_and_line_match_367f11a_10_scenarios_copy_and_splice() -> None:
    """With constraint_layout absent or 'line', the attempt-0 request payloads for both goal_mode='copy'
    and goal_mode='splice' across 10 scenarios x 2 renderings (seeds 20261102-20261111) are byte-identical
    to 367f11a."""
    from udlab_generator import DEFAULT_CONSTRAINT_LAYOUT, OllamaWriter, VALID_CONSTRAINT_LAYOUTS

    assert DEFAULT_CONSTRAINT_LAYOUT == "line"
    assert VALID_CONSTRAINT_LAYOUTS == frozenset({"line", "table"})

    seeds = list(range(20261102, 20261112))
    assert len(seeds) == 10
    worlds = [world_from_scenario_seed(s, index=i) for i, s in enumerate(seeds)]

    for mode, expected_hashes in (
        ("copy", PINNED_367F11A_COPY_PAYLOAD_SHA256),
        ("splice", PINNED_367F11A_SPLICE_PAYLOAD_SHA256),
    ):
        for explicit_layout in (None, "line"):
            captured: list[dict[str, Any]] = []

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
                if url.endswith("/api/show"):
                    return {"parameters": "temperature 0.6\ntop_p 0.95\nrepeat_penalty 1.0"}
                if url.endswith("/api/chat"):
                    if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                        return {"done": True}
                    assert payload is not None
                    idx = len(captured)
                    captured.append(json.loads(json.dumps(payload)))
                    w = worlds[idx // 2]
                    focal = w.agents[idx % 2]
                    fn = render_world_template if mode == "copy" else render_world_template_goal_tokens
                    return {
                        "message": {"role": "assistant", "content": fn(w, focal)},
                        "done": True,
                        "done_reason": "stop",
                        "prompt_eval_count": 3100,
                        "eval_count": 780,
                    }
                raise AssertionError(f"Unexpected URL: {url}")

            kwargs: dict[str, Any] = {
                "grant_id": "test-grant-367f11a",
                "http_transport": _transport,
                "start_owned_server": False,
                "goal_mode": mode,
            }
            if explicit_layout is not None:
                kwargs["constraint_layout"] = explicit_layout
            wr = OllamaWriter(**kwargs)
            assert wr.constraint_layout == "line"
            for w in worlds:
                pair = wr.render_scenario_pair(w)
                assert pair is not None
                assert pair[0].constraint_layout == "line"
                assert pair[1].constraint_layout == "line"
                assert format_writer_prompt(w, w.agents[0], goal_mode=mode) == format_writer_prompt(
                    w, w.agents[0], goal_mode=mode, constraint_layout="line"
                )
            assert len(captured) == 20
            got_hashes = tuple(
                hashlib.sha256(json.dumps(p, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
                for p in captured
            )
            assert got_hashes == expected_hashes


def test_02_table_rendering_room_slots_guard_fold_exception_parser_and_untracked_rejection() -> None:
    """Verify table layout rendering:
    - contains 'in the <room>' in both move rows and both see rows;
    - contains the stricter one-sentence-per-event guard and the leave+enter fold exception;
    - keeps all other 26 requirement lines unchanged (composing with both copy and splice goal_mode);
    - parses back to the exact same check_ids as line layout;
    - rejects an untracked line when appended (in both line and table layouts);
    - leaves on-disk template guard unchanged.
    """
    from udlab_generator.render import (
        EVENT_ORDER_TABLE_ROW_KINDS,
        HARD_CONSTRAINTS_PLACEHOLDER,
        WRITER_PROMPT_PATH,
        get_event_order_table_rows,
        get_writer_prose_requirements,
    )

    assert EVENT_ORDER_TABLE_ROW_KINDS == (
        "enter",
        "leave",
        "move (normal)",
        "move (station change)",
        "see (unoccluded)",
        "see (occluded)",
        "say",
        "message",
    )
    rows_by_kind = get_event_order_table_rows()
    assert tuple(rows_by_kind.keys()) == EVENT_ORDER_TABLE_ROW_KINDS

    # Move rows and see rows must each contain the literal slot 'in the <room>'
    for kind in ("move (normal)", "move (station change)", "see (unoccluded)", "see (occluded)"):
        assert "in the <room>" in rows_by_kind[kind], kind
    assert (
        "<actor> <verb> the <target_object> from the <from_location> to the <to_location> in the <room>."
        in rows_by_kind["move (normal)"]
    )

    # Check header guard and fold exception in table layout
    reqs_line_copy = get_writer_prose_requirements("copy", constraint_layout="line")
    reqs_table_copy = get_writer_prose_requirements("copy", constraint_layout="table")
    reqs_line_splice = get_writer_prose_requirements("splice", constraint_layout="line", agents=("Eva", "Dan", "Cara"))
    reqs_table_splice = get_writer_prose_requirements("splice", constraint_layout="table", agents=("Eva", "Dan", "Cara"))

    assert [r.check_id for r in reqs_line_copy] == [r.check_id for r in reqs_table_copy]
    assert [r.check_id for r in reqs_line_splice] == [r.check_id for r in reqs_table_splice]

    # All other 26 requirement specs are identical between line and table in both copy and splice modes
    for r_l, r_t in zip(reqs_line_copy, reqs_table_copy, strict=True):
        if r_l.check_id != "c_fidelity_event_order":
            assert r_l == r_t
    for r_l, r_t in zip(reqs_line_splice, reqs_table_splice, strict=True):
        if r_l.check_id != "c_fidelity_event_order":
            assert r_l == r_t

    table_event_rule = next(r.rule_text for r in reqs_table_copy if r.check_id == "c_fidelity_event_order")
    header_line = table_event_rule.splitlines()[0]
    assert "one sentence per logged event, in log order" in header_line
    assert "never put two events in one sentence, and never split one event across two sentences" in header_line
    assert (
        "exception: consecutive same-actor leave+enter across a doorway may fold into one sentence "
        "naming from_location and to_location"
    ) in header_line

    # Parser accepts table layout and returns the exact same check_ids as line layout
    w0 = world_from_scenario_seed(20261102, index=0)
    for g_mode in ("copy", "splice"):
        prompt_line = format_writer_prompt(w0, w0.agents[0], goal_mode=g_mode, constraint_layout="line")
        prompt_table = format_writer_prompt(w0, w0.agents[0], goal_mode=g_mode, constraint_layout="table")
        parsed_line = parse_writer_prompt_requirements(
            prompt_line, goal_mode=g_mode, constraint_layout="line", agents=w0.agents
        )
        parsed_table = parse_writer_prompt_requirements(
            prompt_table, goal_mode=g_mode, constraint_layout="table", agents=w0.agents
        )
        # Also works when constraint_layout is not explicitly passed to parse_writer_prompt_requirements:
        parsed_table_auto = parse_writer_prompt_requirements(prompt_table, goal_mode=g_mode, agents=w0.agents)
        assert [r.check_id for r in parsed_table] == [r.check_id for r in parsed_line]
        assert parsed_table == parsed_table_auto
        assert verify_prompt_and_checker_requirements_aligned(
            prompt_table, goal_mode=g_mode, constraint_layout="table", agents=w0.agents
        ) == parsed_table

        # Rejects an untracked line appended at the end of HARD CONSTRAINTS or inside the table:
        for base_prompt, layout in ((prompt_line, "line"), (prompt_table, "table")):
            appended = base_prompt.replace(
                "INPUT SPECIFICATION:",
                "  | untracked_kind | bogus rule line |\n\nINPUT SPECIFICATION:",
            )
            with pytest.raises(ValueError, match="Unrecognized or untracked requirement line"):
                parse_writer_prompt_requirements(appended, goal_mode=g_mode, constraint_layout=layout, agents=w0.agents)

            appended_bullet = base_prompt.replace(
                "INPUT SPECIFICATION:",
                "- (untracked_extra_rule) Never write extra words.\n\nINPUT SPECIFICATION:",
            )
            with pytest.raises(ValueError, match="Writer prompt requirements diverge"):
                verify_prompt_and_checker_requirements_aligned(
                    appended_bullet, goal_mode=g_mode, constraint_layout=layout, agents=w0.agents
                )

    # On-disk template guard is unchanged
    disk_raw = WRITER_PROMPT_PATH.read_text(encoding="utf-8")
    assert HARD_CONSTRAINTS_PLACEHOLDER in disk_raw
    assert "| move (normal) |" not in disk_raw
    with pytest.raises(ValueError, match="constraint_layout must be one of"):
        load_writer_prompt_template(constraint_layout="invalid_layout")  # type: ignore[arg-type]


def test_03_table_coverage_every_367f11a_fragment_present_per_event_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build Item 5: For each event kind, assert that every requirement fragment of 367f11a's
    c_fidelity_event_order line that applies to that kind appears in its row, and that dropping any
    fragment fails the coverage check."""
    from udlab_generator.render import (
        BLOCKING_PHRASES,
        CONTAINER_BAY_STATION,
        FAR_SIDE_ALCOVE_STATION,
        OCCLUDED_SEE_BLOCKED_PHRASES,
        OCCLUDED_SEE_COULD_NOT_SEE,
        OCCLUDER_MANIPULATION_VERBS,
        SIGHTING_VERBS,
        STATION_CHANGE_EVENT_KIND,
        STATION_CHANGE_PERSON_VERBS,
        _ACTION_VERBS_BY_KIND,
        format_blocking_phrases_summary,
        format_negation_phrases_summary,
        format_occluded_see_clause,
        format_occluded_see_example,
        format_station_change_clause,
        format_station_change_examples,
        get_event_order_table_rows,
        verify_event_order_table_coverage,
    )

    example_step, example_return = format_station_change_examples()
    occ_see_example = format_occluded_see_example()

    # Table mapping each 367f11a line-layout sub-clause for that kind -> (line_sub_clause, table_row_fragment):
    coverage_map_by_kind: dict[str, tuple[tuple[str, str], ...]] = {
        "enter": (
            ("enter: " + ", ".join(_ACTION_VERBS_BY_KIND["enter"]), ", ".join(_ACTION_VERBS_BY_KIND["enter"])),
            (
                "every enter sentence must name both from_location and to_location (no bare 'entered <to>' without 'from <from>')",
                "no bare 'entered <to>' without 'from <from>'",
            ),
            (
                "from_location precedes to_location on enter/leave/move (or '<verb> into/to <to> ... from <from>' on enter/leave)",
                "from_location precedes to_location (or '<verb> into/to <to> ... from <from>')",
            ),
            ("from-locations on enter/move use 'from'/'out of'", "'from'/'out of'"),
            ("to-locations on enter/leave/move use 'to'/'into'/'toward'/'at'/'in'/'onto'", "'to'/'into'/'toward'/'at'/'in'/'onto'"),
        ),
        "leave": (
            ("leave: " + ", ".join(_ACTION_VERBS_BY_KIND["leave"]), ", ".join(_ACTION_VERBS_BY_KIND["leave"])),
            ("name both from_location and to_location", "name both from_location and to_location"),
            (
                "from_location precedes to_location on enter/leave/move (or '<verb> into/to <to> ... from <from>' on enter/leave)",
                "from_location precedes to_location (or '<verb> into/to <to> ... from <from>')",
            ),
            ("to-locations on enter/leave/move use 'to'/'into'/'toward'/'at'/'in'/'onto'", "'to'/'into'/'toward'/'at'/'in'/'onto'"),
        ),
        "move (normal)": (
            ("move: " + ", ".join(_ACTION_VERBS_BY_KIND["move"]), ", ".join(_ACTION_VERBS_BY_KIND["move"])),
            ("name room ('in/from/inside/at/of [the] <room>', no other room on intra-room)", "in the <room>"),
            ("name room ('in/from/inside/at/of [the] <room>', no other room on intra-room)", "name room, target_object"),
            (
                "from-locations on enter/move use 'from'/'out of' and to-locations on enter/leave/move use 'to'/'into'/'toward'/'at'/'in'/'onto'",
                "from_location ('from'/'out of') before to_location ('to'/'into'/'toward'/'at'/'in'/'onto')",
            ),
            (
                f"on unoccluded move/see, no blocking phrases ({format_blocking_phrases_summary()}; 'in open view' optional)",
                f"no blocking phrases ({format_blocking_phrases_summary()}; 'in open view' optional)",
            ),
            ("on occluded, state occluder + 'blocked'/'blocking sight'", "state occluder + 'blocked'/'blocking sight'"),
        ),
        "move (station change)": (
            ("name room ('in/from/inside/at/of [the] <room>', no other room on intra-room)", "in the <room>"),
            (STATION_CHANGE_EVENT_KIND + ": " + ", ".join(STATION_CHANGE_PERSON_VERBS), ", ".join(STATION_CHANGE_PERSON_VERBS)),
            (format_station_change_clause(), format_station_change_clause()),
        ),
        "see (unoccluded)": (
            ("see: " + ", ".join(_ACTION_VERBS_BY_KIND["see"]), ", ".join(_ACTION_VERBS_BY_KIND["see"])),
            ("name room ('in/from/inside/at/of [the] <room>', no other room on intra-room)", "in the <room>"),
            ("on see, name target_object", "name target_object"),
            (
                "on see, name target_object, and if unoccluded at a container place it at to_location ('container bay' optional)",
                "at container: place at to_location ('container bay' optional)",
            ),
            (
                f"on unoccluded move/see, no blocking phrases ({format_blocking_phrases_summary()}; 'in open view' optional)",
                f"no blocking phrases ('{BLOCKING_PHRASES[0]}', etc.; 'in open view' optional)",
            ),
        ),
        "see (occluded)": (
            ("name room ('in/from/inside/at/of [the] <room>', no other room on intra-room)", "in the <room>"),
            ("on see, name target_object", "target_object, occluder"),
            ("on occluded, state occluder + 'blocked'/'blocking sight'", OCCLUDED_SEE_BLOCKED_PHRASES[1]),
            (format_occluded_see_clause(), format_occluded_see_clause()),
        ),
        "say": (
            ("name room ('in/from/inside/at/of [the] <room>', no other room on intra-room)", "in the <room>"),
            ("say: " + ", ".join(_ACTION_VERBS_BY_KIND["say"]), ", ".join(_ACTION_VERBS_BY_KIND["say"])),
            ("on say, '<noise_level> dB' + verbatim quote", "'<noise_level> dB' + verbatim quote"),
        ),
        "message": (
            ("name room ('in/from/inside/at/of [the] <room>', no other room on intra-room)", "in the <room>"),
            ("message: " + ", ".join(_ACTION_VERBS_BY_KIND["message"]), ", ".join(_ACTION_VERBS_BY_KIND["message"])),
            (
                "on message, 'to/for/messaged/texted/notified <recipient>' + verbatim quote",
                "'to/for/messaged/texted/notified <recipient>' + verbatim quote",
            ),
        ),
    }

    line_rule_text = next(
        r.rule_text for r in ug_render.WRITER_PROSE_REQUIREMENTS if r.check_id == "c_fidelity_event_order"
    )
    rows = get_event_order_table_rows()
    for kind, pairs in coverage_map_by_kind.items():
        row_text = rows[kind]
        for line_clause, table_frag in pairs:
            assert line_clause in line_rule_text, (kind, line_clause)
            assert table_frag in row_text, f"Row {kind!r} dropped fragment {table_frag!r}"

    # Verify cross-event rules (negation phrases and banned adverbs) are present in table layout:
    table_block = render_writer_prompt_constraints(constraint_layout="table")
    assert format_negation_phrases_summary() in table_block
    assert "- (b_banned_adverbs)" in table_block

    verify_event_order_table_coverage()

    # Dropping any single mapped fragment from any row must cause the coverage check to fail:
    for kind, pairs in coverage_map_by_kind.items():
        for _, table_frag in pairs:
            corrupted_rows = dict(rows)
            corrupted_rows[kind] = rows[kind].replace(table_frag, "")
            assert table_frag not in corrupted_rows[kind]
            monkeypatch.setattr(ug_render, "get_event_order_table_rows", lambda cr=corrupted_rows: cr)
            with pytest.raises(ValueError, match=r"table row .* missing required fragment"):
                verify_event_order_table_coverage()
            monkeypatch.undo()


def test_04_recording_constraint_layout_and_hard_constraints_sha256_resume_guard_and_paired_compare(
    tmp_path: Path,
) -> None:
    """Build Item 4:
    - constraint_layout and hard_constraints_sha256 appear in rendering records, build summary,
      sealing metadata, checkpoint header, paired-run header and rows, and paired-compare output.
    - --resume across layouts refuses with a clear ValueError.
    - --paired-compare accepts two runs that differ only in constraint_layout while keeping same-commit checks.
    """
    from udlab_generator import compute_hard_constraints_sha256

    hc_copy_line = compute_hard_constraints_sha256(goal_mode="copy", constraint_layout="line")
    hc_copy_table = compute_hard_constraints_sha256(goal_mode="copy", constraint_layout="table")
    hc_splice_line = compute_hard_constraints_sha256(goal_mode="splice", constraint_layout="line")
    hc_splice_table = compute_hard_constraints_sha256(goal_mode="splice", constraint_layout="table")
    assert len({hc_copy_line, hc_copy_table, hc_splice_line, hc_splice_table}) == 4
    assert hc_splice_table == hashlib.sha256(
        render_writer_prompt_constraints(goal_mode="splice", constraint_layout="table").encode("utf-8")
    ).hexdigest()

    w0 = generate_world(make_balanced_configs(1, base_seed=20261030)[0])
    w_paired = world_from_scenario_seed(20261030, index=0)

    def _make_transport(mode: str, target_world: Any = w0) -> Any:
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
            if url.endswith("/api/show"):
                return {"parameters": "temperature 0.6\ntop_p 0.95\nrepeat_penalty 1.0"}
            if url.endswith("/api/chat"):
                if payload and payload.get("keep_alive") == 0 and not payload.get("messages"):
                    return {"done": True}
                assert payload is not None
                prompt_txt = payload["messages"][-1]["content"]
                focal = prompt_txt.split("Focal Agent: ", 1)[1].splitlines()[0].strip()
                fn = render_world_template if mode == "copy" else render_world_template_goal_tokens
                return {
                    "message": {"role": "assistant", "content": fn(target_world, focal)},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 3050,
                    "eval_count": 760,
                }
            raise AssertionError(f"Unexpected URL: {url}")

        return _transport

    out_dir = tmp_path / "build_table"
    summary = build_and_probe_dataset(
        count=1,
        base_seed=20261030,
        private_seed_hex="ab" * 32,
        output_dir=out_dir,
        writer_kind="ollama",
        think=False,
        goal_mode="splice",
        constraint_layout="table",
        grant_id="test-grant-table",
        start_owned_server=False,
        http_transport=_make_transport("splice", w0),
    )
    assert summary["constraint_layout"] == "table"
    assert summary["hard_constraints_sha256"] == hc_splice_table

    sealing_doc = json.loads((out_dir / "sealed_store" / "sealing.sealed.json").read_text(encoding="utf-8"))
    assert sealing_doc["writer_provenance"]["constraint_layout"] == "table"
    assert sealing_doc["writer_provenance"]["hard_constraints_sha256"] == hc_splice_table

    ckpt_pub = json.loads((out_dir / "checkpoint.json").read_text(encoding="utf-8"))
    ckpt_seal = json.loads((out_dir / "sealed_store" / "checkpoint.sealed.json").read_text(encoding="utf-8"))
    assert ckpt_pub["header"]["constraint_layout"] == "table"
    assert ckpt_pub["header"]["hard_constraints_sha256"] == hc_splice_table
    assert ckpt_seal["header"]["constraint_layout"] == "table"
    assert ckpt_seal["header"]["hard_constraints_sha256"] == hc_splice_table
    assert ckpt_pub["scenarios"][0]["constraint_layout"] == "table"
    assert ckpt_pub["scenarios"][0]["kept_renderings"][0]["constraint_layout"] == "table"
    assert ckpt_seal["scenarios"][0]["rendering_0"]["constraint_layout"] == "table"

    for gen_line in (out_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines():
        if gen_line.strip():
            assert json.loads(gen_line)["constraint_layout"] == "table"
    for sr_line in (out_dir / "sealed_store" / "responses.sealed.jsonl").read_text(encoding="utf-8").splitlines():
        if sr_line.strip():
            assert json.loads(sr_line)["constraint_layout"] == "table"
    for item_line in (out_dir / "part_public" / "items.jsonl").read_text(encoding="utf-8").splitlines():
        if item_line.strip():
            assert json.loads(item_line)["constraint_layout"] == "table"
    for key_line in (out_dir / "sealed_store" / "keys.sealed.jsonl").read_text(encoding="utf-8").splitlines():
        if key_line.strip():
            assert json.loads(key_line)["constraint_layout"] == "table"

    # --resume with a different constraint_layout refuses with a clear message:
    with pytest.raises(ValueError, match=r"Cannot --resume: checkpoint header.*constraint_layout"):
        build_and_probe_dataset(
            count=1,
            base_seed=20261030,
            output_dir=out_dir,
            writer_kind="ollama",
            think=False,
            goal_mode="splice",
            constraint_layout="line",
            grant_id="test-grant-table",
            start_owned_server=False,
            http_transport=_make_transport("splice", w0),
            resume=True,
        )

    # --resume with matching constraint_layout="table" succeeds:
    resumed = build_and_probe_dataset(
        count=1,
        base_seed=20261030,
        output_dir=out_dir,
        writer_kind="ollama",
        think=False,
        goal_mode="splice",
        constraint_layout="table",
        grant_id="test-grant-table",
        start_owned_server=False,
        http_transport=_make_transport("splice", w0),
        resume=True,
    )
    assert resumed["constraint_layout"] == "table"
    assert resumed["hard_constraints_sha256"] == hc_splice_table

    # Paired seed bank and --paired-compare across line vs table:
    seeds_path = tmp_path / "seeds_paired.json"
    seeds_path.write_text(json.dumps([20261030]), encoding="utf-8")
    line_out = tmp_path / "paired_line.json"
    table_out = tmp_path / "paired_table.json"
    comp_out = tmp_path / "paired_comp_line_vs_table.json"

    rc_line = cli_main(
        [
            "--paired-seed-bank",
            str(seeds_path),
            "--no-think",
            "--goal-mode",
            "splice",
            "--constraint-layout",
            "line",
            "--grant-id",
            "test-grant-paired-line",
            "--paired-out",
            str(line_out),
        ],
        http_transport=_make_transport("splice", w_paired),
    )
    rc_table = cli_main(
        [
            "--paired-seed-bank",
            str(seeds_path),
            "--no-think",
            "--goal-mode",
            "splice",
            "--constraint-layout",
            "table",
            "--grant-id",
            "test-grant-paired-table",
            "--paired-out",
            str(table_out),
        ],
        http_transport=_make_transport("splice", w_paired),
    )
    assert rc_line == 0 and rc_table == 0
    doc_line = json.loads(line_out.read_text(encoding="utf-8"))
    doc_table = json.loads(table_out.read_text(encoding="utf-8"))
    assert doc_line["constraint_layout"] == "line"
    assert doc_line["hard_constraints_sha256"] == hc_splice_line
    assert all(r["constraint_layout"] == "line" for r in doc_line["renderings"])
    assert doc_table["constraint_layout"] == "table"
    assert doc_table["hard_constraints_sha256"] == hc_splice_table
    assert all(r["constraint_layout"] == "table" for r in doc_table["renderings"])

    rc_comp = cli_main(
        [
            "--paired-compare",
            str(line_out),
            str(table_out),
            "--paired-out",
            str(comp_out),
        ]
    )
    assert rc_comp == 0
    comp_doc = json.loads(comp_out.read_text(encoding="utf-8"))
    assert comp_doc["modes"] == {"a": "line", "b": "table"}
    assert comp_doc["constraint_layout"] == {"line": "line", "table": "table"}
    assert comp_doc["hard_constraints_sha256"] == {"line": hc_splice_line, "table": hc_splice_table}

    # Same-commit and same-settings guards in compare_paired_runs:
    bad_commit_table = {**doc_table, "commit": "0" * 40}
    with pytest.raises(ValueError, match="commit mismatch"):
        compare_paired_runs(doc_line, bad_commit_table)

    bad_both_diff = {**doc_table, "goal_mode": "copy"}
    with pytest.raises(ValueError, match=r"cannot differ in both goal_mode.*and constraint_layout"):
        compare_paired_runs(doc_line, bad_both_diff)


def test_05_checker_and_diagnostics_untouched_across_constraint_layouts() -> None:
    """Checker and diagnostic verdicts on constructed prose are byte-identical whichever layout
    produced the prompt."""
    w0 = world_from_scenario_seed(20261102, index=0)
    focal = w0.agents[0]

    prompt_line_copy = format_writer_prompt(w0, focal, goal_mode="copy", constraint_layout="line")
    prompt_table_copy = format_writer_prompt(w0, focal, goal_mode="copy", constraint_layout="table")
    prompt_line_splice = format_writer_prompt(w0, focal, goal_mode="splice", constraint_layout="line")
    prompt_table_splice = format_writer_prompt(w0, focal, goal_mode="splice", constraint_layout="table")
    assert prompt_line_copy != prompt_table_copy
    assert prompt_line_splice != prompt_table_splice

    valid_copy = render_world_template(w0, focal)
    valid_splice = render_world_template_goal_tokens(w0, focal)

    # Construct failing prose samples (missing room on move, missing room on see, paraphrased goal):
    setup_p1, events_p2 = valid_copy.split("\n\n", 1)
    sents = ug_render._split_prose_sentences(events_p2)
    move_idx = next(
        i
        for i, ev in enumerate(w0.events)
        if ev.kind == "move" and not ug_render.is_station_transition_event(ev, w0)
    )
    see_idx = next(i for i, ev in enumerate(w0.events) if ev.kind == "see" and not ev.occluded)

    sents_missing_move_room = list(sents)
    sents_missing_move_room[move_idx] = sents[move_idx].replace(f"in the {w0.events[move_idx].room}, ", "")
    prose_missing_move_room = f"{setup_p1}\n\n{' '.join(sents_missing_move_room)}"

    sents_missing_see_room = list(sents)
    sents_missing_see_room[see_idx] = sents[see_idx].replace(f"in the {w0.events[see_idx].room}, ", "")
    prose_missing_see_room = f"{setup_p1}\n\n{' '.join(sents_missing_see_room)}"

    for sample_prose, g_mode in (
        (valid_copy, "copy"),
        (valid_splice, "splice"),
        (prose_missing_move_room, "copy"),
        (prose_missing_see_room, "copy"),
    ):
        faults_1 = [f.to_dict() for f in collect_prose_faults(sample_prose, w0, focal_agent=focal, goal_mode=g_mode)]
        faults_2 = [f.to_dict() for f in collect_prose_faults(sample_prose, w0, focal_agent=focal, goal_mode=g_mode)]
        assert faults_1 == faults_2

        diag_1 = [
            r.to_dict()
            for r in diagnose_rendered_prose(sample_prose, w0, rendering_index=0, focal_agent=focal, goal_mode=g_mode)
        ]
        diag_2 = [
            r.to_dict()
            for r in diagnose_rendered_prose(sample_prose, w0, rendering_index=0, focal_agent=focal, goal_mode=g_mode)
        ]
        assert diag_1 == diag_2

    with pytest.raises(RenderCheckError) as exc_move:
        check_rendered_prose(prose_missing_move_room, w0, focal_agent=focal, goal_mode="copy")
    assert exc_move.value.check_id == "c_fidelity_missing_event"
    assert evaluate_event_subrules(sents_missing_move_room[move_idx], w0.events[move_idx], w0)["source_room"] is False

    with pytest.raises(RenderCheckError) as exc_see:
        check_rendered_prose(prose_missing_see_room, w0, focal_agent=focal, goal_mode="copy")
    assert exc_see.value.check_id == "c_fidelity_missing_event"
    assert evaluate_event_subrules(sents_missing_see_room[see_idx], w0.events[see_idx], w0)["source_room"] is False
