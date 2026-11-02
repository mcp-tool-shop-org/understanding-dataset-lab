"""Stage 5 Card U1: `sentence_index` and `return_sentence_count` on `match_rendering_event_forms`."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from udlab_generator import (
    AgentGoal,
    Event,
    StakeEdge,
    World,
    compute_shape_plan_run_telemetry,
    generate_shape_plan,
    generate_world,
    make_balanced_configs,
    match_rendering_event_forms,
    matched_event_form_id,
    render_world_template,
    rescore_prose_records,
    world_from_scenario_seed,
)
from udlab_generator.render import (
    _event_content_present,
    _render_event_sentence,
    _sentence_matches_event,
    _sentence_matches_folded_leave_enter,
    _sentence_rejection_rule_id,
    _split_prose_sentences,
    event_shape_plan_variant,
    normalize_prose_line_endings,
)

STAGE3_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "stage3" / "stage3_gate0_fixtures.json"
)
STAGE4_FAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "stage4" / "stage4_shape_fault_fixtures.json"
)
STAGE4_PLAN_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "stage4" / "stage4_shape_plan_fixtures.json"
)
PARITY_FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "stage5"
    / "stage5_sentence_index_parity_e0e010c0.json"
)
EXPECTED_PARITY_FIXTURE_SHA256 = (
    "2c1e49d106a61f476087a1bf56f08e2ee13cbf5102ad5ca7ce62dfca0a93a215"
)


def _load_stage3_world() -> World:
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


_STAGE4_DOC: dict[str, Any] = json.loads(STAGE4_FAULT_FIXTURE_PATH.read_text(encoding="utf-8"))
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


def _build_fixture_world_and_events(
    fixture_data: dict[str, Any],
) -> tuple[World, dict[str, Event]]:
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
    occluders_by_room = {
        str(r): str(occ) for r, occ in w_raw["occluders_by_room"].items()
    }
    default_fixtures = ("wall bench", "corner desk", "floor rack")
    fixtures_by_room = {
        str(r): (str(fix) if fix is not None else default_fixtures[idx % len(default_fixtures)])
        for idx, (r, fix) in enumerate(w_raw["fixtures_by_room"].items())
    }
    world = replace(
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
        initial_agent_rooms={ag: rooms[0] for ag in agents},
        stake_edges=(
            StakeEdge(
                agent=absent_parties[0],
                variable="F1",
                description=(
                    f"{absent_parties[0]}'s audit sign-off for {agents[0]} depends on the location of the {objects[0]}."
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
        noise_threshold=int(w_raw["noise_threshold"]),
    )
    events_map: dict[str, Event] = {}
    for k, ev_raw in fixture_data["events"].items():
        events_map[str(k)] = Event(
            step=int(ev_raw["step"]),
            kind=str(ev_raw["kind"]),
            actor=str(ev_raw["actor"]),
            room=str(ev_raw["room"]),
            target_object=ev_raw["target_object"],
            from_location=ev_raw["from_location"],
            to_location=ev_raw["to_location"],
            utterance=ev_raw["utterance"],
            noise_level=ev_raw["noise_level"],
            occluded=bool(ev_raw["occluded"]),
            occluder=ev_raw["occluder"],
            recipient=ev_raw["recipient"],
            stated_reason=ev_raw["stated_reason"],
        )
    return world, events_map


def _build_verdict_invariance_battery() -> list[dict[str, Any]]:
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
        "leave_nolan_ab": Event(
            step=1,
            kind="leave",
            actor="Nolan",
            room="amber room",
            from_location="amber room",
            to_location="birch room",
        ),
        "enter_nolan_ab": Event(
            step=1,
            kind="enter",
            actor="Nolan",
            room="birch room",
            from_location="amber room",
            to_location="birch room",
        ),
        "move_nolan_prism": Event(
            step=1,
            kind="move",
            actor="Nolan",
            room="birch room",
            target_object="glass prism",
            from_location="slate box",
            to_location="reed basket",
            occluded=False,
        ),
        "see_nolan_prism": Event(
            step=1,
            kind="see",
            actor="Nolan",
            room="birch room",
            target_object="glass prism",
            to_location="reed basket",
            occluded=False,
        ),
        "msg_nolan_quin": Event(
            step=1,
            kind="message",
            actor="Nolan",
            room="birch room",
            recipient="Quin",
            utterance="The glass prism is in the reed basket.",
        ),
        "move_pia_key": Event(
            step=1,
            kind="move",
            actor="Pia",
            room="cobalt room",
            target_object="key to the archive",
            from_location="cedar trunk",
            to_location="iron bin",
            occluded=False,
        ),
        "stat_pia_cobalt_alcove": Event(
            step=1,
            kind="move",
            actor="Pia",
            room="cobalt room",
            target_object="canvas partition",
            from_location="container bay",
            to_location="far-side alcove",
            occluded=False,
        ),
        "see_pia_cobalt_occ": Event(
            step=1,
            kind="see",
            actor="Pia",
            room="cobalt room",
            target_object="cedar trunk",
            from_location="far-side alcove",
            to_location="cedar trunk",
            occluded=True,
            occluder="canvas partition",
        ),
        "say_pia_cobalt_quiet": Event(
            step=1,
            kind="say",
            actor="Pia",
            room="cobalt room",
            utterance="The cedar trunk is behind the canvas partition.",
            noise_level=35,
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
        (
            "v33_valid_nolan_birch_prism_message",
            "valid",
            ["leave_nolan_ab", "enter_nolan_ab", "move_nolan_prism", "see_nolan_prism", "msg_nolan_quin"],
            [
                "Nolan exited the amber room toward the birch room.",
                "Nolan entered the birch room from the amber room.",
                "Nolan transferred the glass prism from the slate box to the reed basket in the birch room.",
                "Nolan observed the glass prism in the reed basket in the birch room.",
                'In the birch room, Nolan sent a message to Quin: "The glass prism is in the reed basket."',
            ],
        ),
        (
            "v34_valid_pia_cobalt_key_station_occ",
            "valid",
            ["move_pia_key", "stat_pia_cobalt_alcove", "see_pia_cobalt_occ", "say_pia_cobalt_quiet"],
            [
                "In the cobalt room, Pia placed the key to the archive from the cedar trunk into the iron bin.",
                "In the cobalt room, Pia stepped behind the canvas partition to the far-side alcove.",
                "In the cobalt room, the canvas partition blocked Pia's sight of the cedar trunk from the far-side alcove.",
                '"The cedar trunk is behind the canvas partition," Pia stated at 35 dB in the cobalt room.',
            ],
        ),
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
                "events": list(ev_keys),
                "num_sentences": len(sents),
                "world": w_scene,
                "focal": focal,
                "prose": prose,
            }
        )
    return battery


def _build_tag_echo_battery_scene() -> dict[str, Any]:
    fix_by_id = {str(r["id"]): str(r["sentence"]) for r in _STAGE4_CASES}
    w = _load_stage3_world()
    focal = w.agents[0]
    nt = w.noise_threshold
    s_see_clear_std = _render_event_sentence(_STAGE4_EVENTS["see_clear"], focal, nt)
    ev_keys = ["say", "move", "see_clear"]
    sents = [fix_by_id["acc_say_std"], fix_by_id["echo_move_tag"], s_see_clear_std]
    evs = tuple(replace(_STAGE4_EVENTS[k], step=i) for i, k in enumerate(ev_keys, start=1))
    w_scene = replace(w, events=evs)
    template_prose = render_world_template(w_scene, focal)
    setup_para, _ = template_prose.split("\n\n", 1)
    prose = f"{setup_para}\n\n" + " ".join(sents)
    return {
        "id": "echo35_tag_echo_move_token",
        "category": "tag_echo_tightened",
        "events": list(ev_keys),
        "num_sentences": len(sents),
        "world": w_scene,
        "focal": focal,
        "prose": prose,
    }


_BATTERY_SCENES = _build_verdict_invariance_battery()
_TAG_ECHO_BATTERY_SCENE = _build_tag_echo_battery_scene()


def _extract_event_sentences(prose: str) -> list[str]:
    norm_prose = normalize_prose_line_endings(prose)
    paragraphs = [p.strip() for p in norm_prose.split("\n\n") if p.strip()]
    event_sentences: list[str] = []
    if len(paragraphs) >= 2:
        for ev_para in paragraphs[1:]:
            event_sentences.extend(_split_prose_sentences(ev_para))
    return event_sentences


def _strip_sentence_index(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{k: v for k, v in r.items() if k != "sentence_index"} for r in rows]


def test_sentence_index_correctness_on_template_scenes_and_all_kinds() -> None:
    """Every row's `sentence_index` on `render_world_template` scenes points at the sentence realizing that event, verified by re-matching that single sentence across all kinds (move, see, say, enter, leave, message, and station MOVE_0)."""
    fixed_seeds = [20261103, 1001, 1002]
    covered_kinds: set[str] = set()
    covered_variants: set[str] = set()
    covered_station_move0 = False

    for seed in fixed_seeds:
        for cfg in make_balanced_configs(4, base_seed=seed):
            w = generate_world(cfg)
            for r_idx in (0, 1):
                focal = w.agents[r_idx]
                prose = render_world_template(w, focal)
                event_sentences = _extract_event_sentences(prose)
                assert len(event_sentences) == len(w.events)

                for prof in ("c", "d"):
                    rows, sent_count = match_rendering_event_forms(
                        prose,
                        w,
                        focal_agent=focal,
                        goal_mode="copy",
                        checker_profile=prof,
                        return_sentence_count=True,
                    )
                    assert sent_count == len(event_sentences)
                    assert len(rows) == len(w.events)

                    for ev_idx, (ev, row) in enumerate(zip(w.events, rows, strict=True)):
                        s_idx = row["sentence_index"]
                        assert isinstance(s_idx, int)
                        assert s_idx == ev_idx
                        assert 0 <= s_idx < len(event_sentences)

                        single_sent = event_sentences[s_idx]
                        rematched_fid = matched_event_form_id(
                            single_sent,
                            ev,
                            w.noise_threshold,
                            world=w,
                            checker_profile=prof,
                        )
                        assert rematched_fid is not None
                        assert rematched_fid == row["realized_form_id"]
                        assert row["outcome"] == rematched_fid
                        assert row["rule_id"] is None

                        covered_kinds.add(ev.kind)
                        variant = event_shape_plan_variant(ev, w)
                        covered_variants.add(variant)
                        if variant == "move_station_change" and rematched_fid == "MOVE_0":
                            covered_station_move0 = True

    assert covered_kinds == {"move", "see", "say", "enter", "leave", "message"}
    assert "move_station_change" in covered_variants
    assert covered_station_move0 is True


def test_sentence_index_on_stage4_shape_fault_and_legend_fixtures() -> None:
    """Check `sentence_index` across Stage 4 shape-fault fixtures, station-move fixtures, and shape-plan legend rows by re-matching each single sentence."""
    base_world = _load_stage3_world()
    focal_base = base_world.agents[0]

    # 1. Stage 4 35 shape-fault fixtures
    for case in _STAGE4_CASES:
        ev = _STAGE4_EVENTS[str(case["event"])]
        sentence = str(case["sentence"])
        expected = str(case["expect"])
        w_1ev = replace(base_world, events=(ev,))
        setup_para = render_world_template(w_1ev, focal_base).split("\n\n", 1)[0]
        prose = f"{setup_para}\n\n{sentence}"
        event_sentences = _extract_event_sentences(prose)
        assert event_sentences == [sentence]

        rows, sent_count = match_rendering_event_forms(
            prose,
            w_1ev,
            focal_agent=focal_base,
            goal_mode="copy",
            checker_profile="d",
            return_sentence_count=True,
        )
        assert sent_count == 1
        assert len(rows) == 1
        row = rows[0]
        assert row["sentence_index"] == 0

        if expected == "accepted":
            rematched_fid = matched_event_form_id(
                event_sentences[0],
                ev,
                w_1ev.noise_threshold,
                world=w_1ev,
                checker_profile="d",
            )
            assert rematched_fid == row["realized_form_id"]
            assert rematched_fid is not None
            assert row["rule_id"] is None
        else:
            assert row["realized_form_id"] is None
            assert _sentence_rejection_rule_id(
                event_sentences[0],
                ev,
                w_1ev.noise_threshold,
                world=w_1ev,
                checker_profile="d",
            ) == row["rule_id"]

    # 2. Station move fixtures (MOVE_0 accepted and fault cases)
    for case in _STATION_MOVE_FIXTURES:
        ev = _STATION_MOVE_EVENTS[str(case["event"])]
        sentence = str(case["sentence"])
        expected = str(case["expect"])
        w_1ev = replace(base_world, events=(ev,))
        setup_para = render_world_template(w_1ev, focal_base).split("\n\n", 1)[0]
        prose = f"{setup_para}\n\n{sentence}"
        event_sentences = _extract_event_sentences(prose)
        assert event_sentences == [sentence]

        rows, sent_count = match_rendering_event_forms(
            prose,
            w_1ev,
            focal_agent=focal_base,
            goal_mode="copy",
            checker_profile="d",
            return_sentence_count=True,
        )
        assert sent_count == 1
        assert len(rows) == 1
        row = rows[0]
        assert row["sentence_index"] == 0
        if expected == "accepted":
            assert row["realized_form_id"] == "MOVE_0"
            assert (
                matched_event_form_id(
                    event_sentences[0],
                    ev,
                    w_1ev.noise_threshold,
                    world=w_1ev,
                    checker_profile="d",
                )
                == "MOVE_0"
            )
        else:
            assert row["realized_form_id"] is None

    # 3. Stage 4 shape-plan 18 legend rows
    plan_fixture_data = json.loads(STAGE4_PLAN_FIXTURE_PATH.read_text(encoding="utf-8"))
    plan_world, plan_events_map = _build_fixture_world_and_events(plan_fixture_data)
    focal_plan = plan_world.agents[0]
    for legend_row in plan_fixture_data["legend_rows"]:
        ev = plan_events_map[str(legend_row["event"])]
        sentence = str(legend_row["sentence"])
        expected_fid = str(legend_row["form_id"])
        w_1ev = replace(plan_world, events=(ev,))
        setup_para = render_world_template(w_1ev, focal_plan).split("\n\n", 1)[0]
        prose = f"{setup_para}\n\n{sentence}"
        rows, sent_count = match_rendering_event_forms(
            prose,
            w_1ev,
            focal_agent=focal_plan,
            goal_mode="copy",
            checker_profile="d",
            return_sentence_count=True,
        )
        assert sent_count == 1
        assert len(rows) == 1
        assert rows[0]["sentence_index"] == 0
        assert rows[0]["realized_form_id"] == expected_fid


def test_folded_leave_enter_shares_sentence_index() -> None:
    """When one sentence realizes a folded leave/enter pair, both rows carry the same `sentence_index`."""
    scene_by_id = {str(s["id"]): s for s in _BATTERY_SCENES}
    folded_scene = scene_by_id["v06_valid_folded_leave_enter_move"]
    w_scene = folded_scene["world"]
    focal = str(folded_scene["focal"])
    prose = str(folded_scene["prose"])
    event_sentences = _extract_event_sentences(prose)
    assert len(event_sentences) == 2

    rows, sent_count = match_rendering_event_forms(
        prose,
        w_scene,
        focal_agent=focal,
        goal_mode="copy",
        checker_profile="d",
        return_sentence_count=True,
    )
    assert sent_count == 2
    assert len(rows) == 3
    assert [r["kind"] for r in rows] == ["leave", "enter", "move"]
    assert [r["realized_form_id"] for r in rows] == ["LEAVE_0", "ENTER_0", "MOVE_1"]
    assert rows[0]["sentence_index"] == 0
    assert rows[1]["sentence_index"] == 0
    assert rows[0]["sentence_index"] == rows[1]["sentence_index"]
    assert rows[2]["sentence_index"] == 1
    assert _sentence_matches_folded_leave_enter(
        event_sentences[0],
        w_scene.events[0],
        w_scene.events[1],
        w_scene.noise_threshold,
        world=w_scene,
        checker_profile="d",
    )

    # Also check when folded leave/enter occurs at sentence index 1 after a preceding event at index 0
    say_ev = _STAGE4_EVENTS["say"]
    w_shifted = replace(
        w_scene,
        events=(
            replace(say_ev, step=1),
            replace(w_scene.events[0], step=2),
            replace(w_scene.events[1], step=3),
            replace(w_scene.events[2], step=4),
        ),
    )
    setup_para = render_world_template(w_shifted, focal).split("\n\n", 1)[0]
    say_sent = _render_event_sentence(w_shifted.events[0], focal, w_shifted.noise_threshold)
    prose_shifted = f"{setup_para}\n\n{say_sent} " + " ".join(event_sentences)
    rows_shifted, sent_count_shifted = match_rendering_event_forms(
        prose_shifted,
        w_shifted,
        focal_agent=focal,
        goal_mode="copy",
        checker_profile="d",
        return_sentence_count=True,
    )
    assert sent_count_shifted == 3
    assert [r["sentence_index"] for r in rows_shifted] == [0, 1, 1, 2]
    assert rows_shifted[1]["realized_form_id"] == "LEAVE_0"
    assert rows_shifted[2]["realized_form_id"] == "ENTER_0"
    assert rows_shifted[3]["realized_form_id"] == "MOVE_1"


def test_fault_cases_altered_absent_and_earlier_order() -> None:
    """Verify `sentence_index` on fault cases: `altered` carries the consumed sentence index, `absent` carries `None` (`null`), and earlier-sentence `c_fidelity_event_order` carries the earlier sentence index."""
    scene_by_id = {str(s["id"]): s for s in _BATTERY_SCENES}

    # 1. Omitted middle event + altered third event (`omit22_review_scene_a`: [say, see_clear, move], 2 sentences)
    scene_a = scene_by_id["omit22_review_scene_a"]
    rows_a, count_a = match_rendering_event_forms(
        scene_a["prose"],
        scene_a["world"],
        focal_agent=scene_a["focal"],
        checker_profile="d",
        return_sentence_count=True,
    )
    assert count_a == 2
    assert len(rows_a) == 3
    # Event 1 (say) matches sentence 0
    assert rows_a[0]["realized_form_id"] == "SAY_1"
    assert rows_a[0]["rule_id"] is None
    assert rows_a[0]["sentence_index"] == 0
    # Event 2 (see_clear) is absent -> None (null in JSON)
    assert rows_a[1]["realized_form_id"] is None
    assert rows_a[1]["rule_id"] == "c_fidelity_missing_event"
    assert rows_a[1]["sentence_index"] is None
    # Event 3 (move) consumes sentence 1 as altered (`form_not_accepted`) -> sentence_index 1
    assert rows_a[2]["realized_form_id"] is None
    assert rows_a[2]["rule_id"] == "form_not_accepted"
    assert rows_a[2]["sentence_index"] == 1
    sents_a = _extract_event_sentences(scene_a["prose"])
    assert _event_content_present(sents_a[1], scene_a["world"].events[2], scene_a["world"], "d")

    # 2. Trailing absent event when no candidate sentence remains (`omit25_omit_ev3_after_valid_and_fna`)
    scene_omit25 = scene_by_id["omit25_omit_ev3_after_valid_and_fna"]
    rows_omit25, count_omit25 = match_rendering_event_forms(
        scene_omit25["prose"],
        scene_omit25["world"],
        focal_agent=scene_omit25["focal"],
        checker_profile="d",
        return_sentence_count=True,
    )
    assert count_omit25 == 2
    assert [r["sentence_index"] for r in rows_omit25] == [0, 1, None]
    assert [r["rule_id"] for r in rows_omit25] == [
        None,
        "form_not_accepted",
        "c_fidelity_missing_event",
    ]

    # 3. Earlier-sentence `c_fidelity_event_order` (`reord28_swap_valid_say_move`: events [say, move, see_clear], sentences [move, say, see_clear])
    scene_reord28 = scene_by_id["reord28_swap_valid_say_move"]
    rows_reord28, count_reord28 = match_rendering_event_forms(
        scene_reord28["prose"],
        scene_reord28["world"],
        focal_agent=scene_reord28["focal"],
        checker_profile="d",
        return_sentence_count=True,
    )
    assert count_reord28 == 3
    # Event 1 (say) matches sentence 1 -> search_start becomes 2
    assert rows_reord28[0]["realized_form_id"] == "SAY_1"
    assert rows_reord28[0]["sentence_index"] == 1
    # Event 2 (move) is found at earlier sentence 0 -> c_fidelity_event_order with sentence_index 0
    assert rows_reord28[1]["realized_form_id"] is None
    assert rows_reord28[1]["outcome"] == "other_fault"
    assert rows_reord28[1]["rule_id"] == "c_fidelity_event_order"
    assert rows_reord28[1]["sentence_index"] == 0
    sents_reord28 = _extract_event_sentences(scene_reord28["prose"])
    assert _sentence_matches_event(
        sents_reord28[0],
        scene_reord28["world"].events[1],
        scene_reord28["world"].noise_threshold,
        world=scene_reord28["world"],
        checker_profile="d",
    )
    # Event 3 (see_clear) matches sentence 2 -> sentence_index 2
    assert rows_reord28[2]["realized_form_id"] == "SEE_1"
    assert rows_reord28[2]["sentence_index"] == 2

    # 4. Reordered + form-bad (`reord29_swap_valid_and_fna`: events [move, see_clear, say], sentences [see_clear, fna_move_cleft, say])
    scene_reord29 = scene_by_id["reord29_swap_valid_and_fna"]
    rows_reord29, count_reord29 = match_rendering_event_forms(
        scene_reord29["prose"],
        scene_reord29["world"],
        focal_agent=scene_reord29["focal"],
        checker_profile="d",
        return_sentence_count=True,
    )
    assert count_reord29 == 3
    # Event 1 (move) sees sentence 0 (see_clear, which matches event 2), so event 1 is marked absent (None)
    assert rows_reord29[0]["rule_id"] == "c_fidelity_missing_event"
    assert rows_reord29[0]["sentence_index"] is None
    # Event 2 (see_clear) matches sentence 0 -> search_start becomes 1
    assert rows_reord29[1]["realized_form_id"] == "SEE_1"
    assert rows_reord29[1]["sentence_index"] == 0
    # Event 3 (say) matches sentence 2 -> sentence_index 2 (sentence 1 is skipped as unlogged)
    assert rows_reord29[2]["realized_form_id"] == "SAY_1"
    assert rows_reord29[2]["sentence_index"] == 2


def test_parity_with_e0e010c0_fixture_and_existing_callers() -> None:
    """Parity requirement: every row with `sentence_index` removed is byte-identical (`json.dumps(..., sort_keys=True)`) to the fixture generated at e0e010c0, and `compute_shape_plan_run_telemetry` and `rescore_prose_records` produce identical results."""
    digest = hashlib.sha256(PARITY_FIXTURE_PATH.read_bytes()).hexdigest()
    assert digest == EXPECTED_PARITY_FIXTURE_SHA256
    fixture_doc = json.loads(PARITY_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert fixture_doc["generated_from_commit"] == "e0e010c036cfd4c72c08b79ad60641f1db73fb25"

    fixed_seeds: list[int] = [int(s) for s in fixture_doc["fixed_seeds"]]
    telemetry_worlds = []
    a0_prose_map: dict[tuple[str, int], str] = {}
    fin_prose_map: dict[tuple[str, int], str] = {}
    rescore_inputs: list[dict[str, Any]] = []

    # 1. Template scenes parity
    actual_template_scenes: list[dict[str, Any]] = []
    for seed in fixed_seeds:
        for cfg in make_balanced_configs(4, base_seed=seed):
            w = generate_world(cfg)
            if seed == 20261103:
                telemetry_worlds.append(w)
            for r_idx in (0, 1):
                focal = w.agents[r_idx]
                prose = render_world_template(w, focal)
                if seed == 20261103:
                    a0_prose_map[(w.scenario_id, r_idx)] = prose
                    fin_prose_map[(w.scenario_id, r_idx)] = prose
                    rescore_inputs.append(
                        {
                            "world": w,
                            "rendering_index": r_idx,
                            "focal_agent": focal,
                            "prose": prose,
                            "goal_mode": "copy",
                        }
                    )
                rows_c = match_rendering_event_forms(
                    prose, w, focal_agent=focal, goal_mode="copy", checker_profile="c"
                )
                rows_d = match_rendering_event_forms(
                    prose, w, focal_agent=focal, goal_mode="copy", checker_profile="d"
                )
                actual_template_scenes.append(
                    {
                        "seed": seed,
                        "scenario_id": w.scenario_id,
                        "rendering_index": r_idx,
                        "focal_agent": focal,
                        "rows_profile_c": _strip_sentence_index(rows_c),
                        "rows_profile_d": _strip_sentence_index(rows_d),
                    }
                )

    for act_scene, exp_scene in zip(
        actual_template_scenes, fixture_doc["template_scenes"], strict=True
    ):
        for key in ("rows_profile_c", "rows_profile_d"):
            for act_row, exp_row in zip(act_scene[key], exp_scene[key], strict=True):
                assert json.dumps(act_row, sort_keys=True) == json.dumps(
                    exp_row, sort_keys=True
                )
        assert json.dumps(act_scene, sort_keys=True) == json.dumps(exp_scene, sort_keys=True)

    # 2. Stage 4 shape-fault rows parity
    base_world = _load_stage3_world()
    focal_base = base_world.agents[0]
    for case, exp_entry in zip(_STAGE4_CASES, fixture_doc["stage4_fault_rows"], strict=True):
        ev = _STAGE4_EVENTS[str(case["event"])]
        sentence = str(case["sentence"])
        w_1ev = replace(base_world, events=(ev,))
        setup_para = render_world_template(w_1ev, focal_base).split("\n\n", 1)[0]
        prose = f"{setup_para}\n\n{sentence}"
        rescore_inputs.append(
            {
                "world": w_1ev,
                "rendering_index": 0,
                "focal_agent": focal_base,
                "prose": prose,
                "goal_mode": "copy",
            }
        )
        rows_c = match_rendering_event_forms(
            prose, w_1ev, focal_agent=focal_base, goal_mode="copy", checker_profile="c"
        )
        rows_d = match_rendering_event_forms(
            prose, w_1ev, focal_agent=focal_base, goal_mode="copy", checker_profile="d"
        )
        assert json.dumps(_strip_sentence_index(rows_c)[0], sort_keys=True) == json.dumps(
            exp_entry["rows_profile_c"][0], sort_keys=True
        )
        assert json.dumps(_strip_sentence_index(rows_d)[0], sort_keys=True) == json.dumps(
            exp_entry["rows_profile_d"][0], sort_keys=True
        )

    # 3. Station move rows parity
    for case, exp_entry in zip(
        _STATION_MOVE_FIXTURES, fixture_doc["station_move_rows"], strict=True
    ):
        ev = _STATION_MOVE_EVENTS[str(case["event"])]
        sentence = str(case["sentence"])
        w_1ev = replace(base_world, events=(ev,))
        setup_para = render_world_template(w_1ev, focal_base).split("\n\n", 1)[0]
        prose = f"{setup_para}\n\n{sentence}"
        rows_c = match_rendering_event_forms(
            prose, w_1ev, focal_agent=focal_base, goal_mode="copy", checker_profile="c"
        )
        rows_d = match_rendering_event_forms(
            prose, w_1ev, focal_agent=focal_base, goal_mode="copy", checker_profile="d"
        )
        assert json.dumps(_strip_sentence_index(rows_c)[0], sort_keys=True) == json.dumps(
            exp_entry["rows_profile_c"][0], sort_keys=True
        )
        assert json.dumps(_strip_sentence_index(rows_d)[0], sort_keys=True) == json.dumps(
            exp_entry["rows_profile_d"][0], sort_keys=True
        )

    # 4. Stage 4 shape-plan legend rows parity
    plan_fixture_data = json.loads(STAGE4_PLAN_FIXTURE_PATH.read_text(encoding="utf-8"))
    plan_world, plan_events_map = _build_fixture_world_and_events(plan_fixture_data)
    focal_plan = plan_world.agents[0]
    for legend_row, exp_entry in zip(
        plan_fixture_data["legend_rows"], fixture_doc["stage4_legend_rows"], strict=True
    ):
        ev = plan_events_map[str(legend_row["event"])]
        sentence = str(legend_row["sentence"])
        w_1ev = replace(plan_world, events=(ev,))
        setup_para = render_world_template(w_1ev, focal_plan).split("\n\n", 1)[0]
        prose = f"{setup_para}\n\n{sentence}"
        rows_c = match_rendering_event_forms(
            prose, w_1ev, focal_agent=focal_plan, goal_mode="copy", checker_profile="c"
        )
        rows_d = match_rendering_event_forms(
            prose, w_1ev, focal_agent=focal_plan, goal_mode="copy", checker_profile="d"
        )
        assert json.dumps(_strip_sentence_index(rows_c)[0], sort_keys=True) == json.dumps(
            exp_entry["rows_profile_c"][0], sort_keys=True
        )
        assert json.dumps(_strip_sentence_index(rows_d)[0], sort_keys=True) == json.dumps(
            exp_entry["rows_profile_d"][0], sort_keys=True
        )

    # 5. Multi-event battery scenes parity
    for scene, exp_entry in zip(
        [*_BATTERY_SCENES, _TAG_ECHO_BATTERY_SCENE], fixture_doc["battery_rows"], strict=True
    ):
        w_scene = scene["world"]
        focal_scene = str(scene["focal"])
        prose_scene = str(scene["prose"])
        rescore_inputs.append(
            {
                "world": w_scene,
                "rendering_index": 0,
                "focal_agent": focal_scene,
                "prose": prose_scene,
                "goal_mode": "copy",
            }
        )
        rows_c = match_rendering_event_forms(
            prose_scene, w_scene, focal_agent=focal_scene, goal_mode="copy", checker_profile="c"
        )
        rows_d = match_rendering_event_forms(
            prose_scene, w_scene, focal_agent=focal_scene, goal_mode="d" if False else "copy", checker_profile="d"
        )
        for act_r, exp_r in zip(
            _strip_sentence_index(rows_c), exp_entry["rows_profile_c"], strict=True
        ):
            assert json.dumps(act_r, sort_keys=True) == json.dumps(exp_r, sort_keys=True)
        for act_r, exp_r in zip(
            _strip_sentence_index(rows_d), exp_entry["rows_profile_d"], strict=True
        ):
            assert json.dumps(act_r, sort_keys=True) == json.dumps(exp_r, sort_keys=True)

    # 6. Existing callers parity: compute_shape_plan_run_telemetry & rescore_prose_records
    plan_bundle = generate_shape_plan(telemetry_worlds, seed=20261103, block_size=20)
    telemetry_out_d = compute_shape_plan_run_telemetry(
        telemetry_worlds,
        shape_plan="assigned",
        shape_plan_seed=20261103,
        plan_bundle=plan_bundle,
        attempt0_prose_by_rendering=a0_prose_map,
        final_prose_by_rendering=fin_prose_map,
        goal_mode="copy",
        checker_profile="d",
    )
    rescore_out_d = rescore_prose_records(
        rescore_inputs,
        checker_version="stage5-u1-parity",
        checker_profile="d",
    )
    rescore_out_c = rescore_prose_records(
        rescore_inputs,
        checker_version="stage5-u1-parity",
        checker_profile="c",
    )

    exp_caller = fixture_doc["caller_parity"]
    assert json.dumps(telemetry_out_d, sort_keys=True) == json.dumps(
        exp_caller["telemetry_profile_d"], sort_keys=True
    )
    assert json.dumps(rescore_out_d, sort_keys=True) == json.dumps(
        exp_caller["rescore_profile_d"], sort_keys=True
    )
    assert json.dumps(rescore_out_c, sort_keys=True) == json.dumps(
        exp_caller["rescore_profile_c"], sort_keys=True
    )


def test_rows_carry_sentence_index() -> None:
    """Head-only assertion moved from `test_parity_with_e0e010c0_fixture_and_existing_callers` so the parity test runs unchanged against e0e010c0 source."""
    fixture_doc = json.loads(PARITY_FIXTURE_PATH.read_text(encoding="utf-8"))
    fixed_seeds: list[int] = [int(s) for s in fixture_doc["fixed_seeds"]]
    for seed in fixed_seeds:
        for cfg in make_balanced_configs(4, base_seed=seed):
            w = generate_world(cfg)
            for r_idx in (0, 1):
                focal = w.agents[r_idx]
                prose = render_world_template(w, focal)
                rows_c = match_rendering_event_forms(
                    prose, w, focal_agent=focal, goal_mode="copy", checker_profile="c"
                )
                rows_d = match_rendering_event_forms(
                    prose, w, focal_agent=focal, goal_mode="copy", checker_profile="d"
                )
                assert all("sentence_index" in r for r in rows_c)
                assert all("sentence_index" in r for r in rows_d)


def test_return_sentence_count_and_default_return_type() -> None:
    """Verify `return_sentence_count=True` returns `(rows, event_sentence_count)` including on 0-event worlds and multi-paragraph/extra-sentence prose, while the default call returns `list[dict[str, object]]`."""
    w = generate_world(make_balanced_configs(1, base_seed=20261103)[0])
    focal = w.agents[0]
    prose = render_world_template(w, focal)
    expected_count = len(_extract_event_sentences(prose))
    assert expected_count == len(w.events)

    # Default call: return_sentence_count omitted or False
    rows_default = match_rendering_event_forms(prose, w, focal_agent=focal)
    assert isinstance(rows_default, list)
    rows_false = match_rendering_event_forms(
        prose, w, focal_agent=focal, return_sentence_count=False
    )
    assert isinstance(rows_false, list)
    assert rows_default == rows_false

    # With return_sentence_count=True
    ret_true = match_rendering_event_forms(
        prose, w, focal_agent=focal, return_sentence_count=True
    )
    assert isinstance(ret_true, tuple)
    assert len(ret_true) == 2
    rows_true, count_true = ret_true
    assert rows_true == rows_default
    assert isinstance(count_true, int)
    assert count_true == expected_count

    # Extra unlogged trailing sentence increases event_sentence_count without changing matched indices
    prose_extra = prose + " In the amber room, Mira waited."
    rows_extra, count_extra = match_rendering_event_forms(
        prose_extra, w, focal_agent=focal, return_sentence_count=True
    )
    assert count_extra == expected_count + 1
    assert rows_extra == rows_default

    # Single-paragraph prose (no event paragraph) -> event_sentence_count == 0
    setup_only = prose.split("\n\n", 1)[0]
    rows_no_ev_para, count_no_ev_para = match_rendering_event_forms(
        setup_only, w, focal_agent=focal, return_sentence_count=True
    )
    assert count_no_ev_para == 0
    assert all(r["sentence_index"] is None for r in rows_no_ev_para)

    # Empty world (0 events)
    w_empty = replace(w, events=())
    assert match_rendering_event_forms(prose, w_empty, focal_agent=focal) == []
    assert match_rendering_event_forms(
        prose, w_empty, focal_agent=focal, return_sentence_count=True
    ) == ([], expected_count)
    assert match_rendering_event_forms(
        setup_only, w_empty, focal_agent=focal, return_sentence_count=True
    ) == ([], 0)
