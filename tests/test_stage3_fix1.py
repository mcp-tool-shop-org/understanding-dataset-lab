"""Stage 3 fix1 tests: B1 (quote binding), B2 (P1 hedges/modals), N3 (V2 trailing sighting), N6 (quote normalisation), N7 (CLI help), N2 (M3 order)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from udlab_generator import (
    AgentGoal,
    Event,
    World,
    world_from_scenario_seed,
)
from udlab_generator.cli import main as cli_main
from udlab_generator.render import _sentence_matches_event


def _load_stage3_world() -> World:
    from dataclasses import replace

    fix_path = Path(__file__).resolve().parent / "fixtures" / "stage3" / "stage3_gate0_fixtures.json"
    doc = json.loads(fix_path.read_text(encoding="utf-8"))
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
    fixtures_by_room = {
        str(k): (str(v) if v is not None else "")
        for k, v in spec["fixtures_by_room"].items()
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
        goals={
            ag: AgentGoal(
                agent=ag,
                goal_text=f"{ag} plans to inspect the {rooms[0]}",
                causal_preconditions=("F1",),
            )
            for ag in agents
        },
        noise_threshold=int(spec["noise_threshold"]),
    )


def _ev_say() -> Event:
    return Event(
        step=1,
        kind="say",
        actor="Mira",
        room="amber room",
        utterance="The copper token is stored in the tin locker.",
        noise_level=55,
    )


def _ev_message() -> Event:
    return Event(
        step=2,
        kind="message",
        actor="Mira",
        room="amber room",
        recipient="Quin",
        utterance="Update for Quin: the copper token is stored in the tin locker.",
    )


def _ev_see_clear() -> Event:
    return Event(
        step=3,
        kind="see",
        actor="Mira",
        room="amber room",
        target_object="copper token",
        to_location="pine crate",
        occluded=False,
    )


def _ev_see_occ() -> Event:
    return Event(
        step=4,
        kind="see",
        actor="Mira",
        room="amber room",
        target_object="pine crate",
        occluded=True,
        occluder="felt curtain",
    )


def _ev_move() -> Event:
    return Event(
        step=5,
        kind="move",
        actor="Mira",
        room="amber room",
        target_object="copper token",
        from_location="pine crate",
        to_location="tin locker",
        occluded=False,
    )


@pytest.mark.parametrize("profile", ["c", "d"])
def test_b1_say_split_quote_with_second_quoted_span_rejected(profile: str) -> None:
    """B1: '"<utt>," Mira said at 55 dB in the amber room, "and the glass prism is lost."' must fail in both C and D."""
    world = _load_stage3_world()
    ev = _ev_say()
    # Valid control for S2 shape
    valid_s2 = '"The copper token is stored in the tin locker," Mira said at 55 dB in the amber room.'
    assert _sentence_matches_event(valid_s2, ev, world.noise_threshold, world=world, checker_profile=profile)

    # Invalid: extra quoted span after the attribution clause
    invalid_s2_extra = (
        '"The copper token is stored in the tin locker," Mira said at 55 dB in the amber room, '
        '"and the glass prism is lost."'
    )
    assert not _sentence_matches_event(
        invalid_s2_extra, ev, world.noise_threshold, world=world, checker_profile=profile
    )


@pytest.mark.parametrize("profile", ["c", "d"])
def test_b1_say_wrong_verb_adjacent_quote_followed_by_utterance_rejected(profile: str) -> None:
    """B1: 'Mira said at 55 dB in the amber room, "and the glass prism is lost," "<utt>."' must fail in both C and D."""
    world = _load_stage3_world()
    ev = _ev_say()
    # Valid control for S1 shape
    valid_s1 = 'Mira said at 55 dB in the amber room, "The copper token is stored in the tin locker."'
    assert _sentence_matches_event(valid_s1, ev, world.noise_threshold, world=world, checker_profile=profile)

    # Invalid: verb-adjacent quote is wrong, second quote has the utterance
    invalid_s1_two_quotes = (
        'Mira said at 55 dB in the amber room, "and the glass prism is lost," '
        '"The copper token is stored in the tin locker."'
    )
    assert not _sentence_matches_event(
        invalid_s1_two_quotes, ev, world.noise_threshold, world=world, checker_profile=profile
    )


@pytest.mark.parametrize("profile", ["c", "d"])
def test_b1_message_with_extra_quoted_span_rejected(profile: str) -> None:
    """B1: A message sentence with one extra quoted span must fail in both C and D."""
    world = _load_stage3_world()
    ev = _ev_message()
    # Valid controls for message shape (post-verb quote and pre-verb quote)
    valid_msg_post = (
        'In the amber room, Mira texted to Quin, '
        '"Update for Quin: the copper token is stored in the tin locker."'
    )
    valid_msg_pre = (
        '"Update for Quin: the copper token is stored in the tin locker," '
        'Mira texted to Quin from the amber room.'
    )
    assert _sentence_matches_event(valid_msg_post, ev, world.noise_threshold, world=world, checker_profile=profile)
    assert _sentence_matches_event(valid_msg_pre, ev, world.noise_threshold, world=world, checker_profile=profile)

    # Invalid: message with an extra quoted span
    invalid_msg_extra_after = (
        'In the amber room, Mira texted to Quin, '
        '"Update for Quin: the copper token is stored in the tin locker," "and the glass prism is lost."'
    )
    invalid_msg_extra_before = (
        'In the amber room, Mira texted to Quin, "and the glass prism is lost," '
        '"Update for Quin: the copper token is stored in the tin locker."'
    )
    invalid_msg_split = (
        '"Update for Quin: the copper token is stored in the tin locker," '
        'Mira texted to Quin from the amber room, "and the glass prism is lost."'
    )
    assert not _sentence_matches_event(
        invalid_msg_extra_after, ev, world.noise_threshold, world=world, checker_profile=profile
    )
    assert not _sentence_matches_event(
        invalid_msg_extra_before, ev, world.noise_threshold, world=world, checker_profile=profile
    )
    assert not _sentence_matches_event(
        invalid_msg_split, ev, world.noise_threshold, world=world, checker_profile=profile
    )


_B2_P1_HEDGE_AND_MODAL_PHRASES: tuple[tuple[str, str], ...] = (
    ("perhaps", "perhaps"),
    ("maybe", "maybe"),
    ("might", "might be tired"),
    ("may", "may be tired"),
    ("could", "could be tired"),
    ("would", "would be tired"),
    ("should", "should be ready"),
    ("likely", "likely tired"),
    ("probably", "probably tired"),
    ("possibly", "possibly tired"),
    ("apparently", "apparently calm"),
    ("seemingly", "seemingly calm"),
    ("supposedly", "supposedly calm"),
    ("it seems", "it seems"),
    ("it seemed", "it seemed"),
    ("I think", "I think"),
    ("I believe", "I believe"),
    ("I guess", "I guess"),
)


def test_b2_p1_valid_control_passes_in_d_and_fails_in_c() -> None:
    """B2 control: valid P1 phrase without hedges passes in Profile D and fails in Profile C."""
    world = _load_stage3_world()
    ev = _ev_see_clear()
    valid_p1 = "Mira, standing quietly, saw the copper token in the pine crate in the amber room."
    assert not _sentence_matches_event(valid_p1, ev, world.noise_threshold, world=world, checker_profile="c")
    assert _sentence_matches_event(valid_p1, ev, world.noise_threshold, world=world, checker_profile="d")


@pytest.mark.parametrize(
    ("item_label", "p1_phrase"),
    _B2_P1_HEDGE_AND_MODAL_PHRASES,
    ids=[x[0].replace(" ", "_") for x in _B2_P1_HEDGE_AND_MODAL_PHRASES],
)
def test_b2_p1_hedges_and_modals_rejected_in_profile_d(item_label: str, p1_phrase: str) -> None:
    """B2: Every hedge/modal in P1's dedicated set is rejected inside a P1 phrase under Profile D (and C)."""
    from udlab_generator.render import _NON_FACTIVE_OR_HEDGE_WORDS

    # Shared set must remain untouched (do not extend _NON_FACTIVE_OR_HEDGE_WORDS)
    assert "perhaps" not in _NON_FACTIVE_OR_HEDGE_WORDS
    assert "maybe" not in _NON_FACTIVE_OR_HEDGE_WORDS
    assert "might" not in _NON_FACTIVE_OR_HEDGE_WORDS

    world = _load_stage3_world()
    ev = _ev_see_clear()
    sent = f"Mira, {p1_phrase}, saw the copper token in the pine crate in the amber room."
    assert not _sentence_matches_event(
        sent, ev, world.noise_threshold, world=world, checker_profile="d"
    ), f"P1 hedge/modal {item_label!r} ({p1_phrase!r}) must be rejected under Profile D: {sent!r}"
    assert not _sentence_matches_event(
        sent, ev, world.noise_threshold, world=world, checker_profile="c"
    )


@pytest.mark.parametrize(
    "trailing_clause",
    [
        ", who saw the pine crate.",
        ", who noticed the pine crate.",
        ", who saw the copper token.",
        ", who noticed the copper token.",
    ],
    ids=["who_saw_target", "who_noticed_target", "who_saw_object", "who_noticed_object"],
)
def test_n3_v2_trailing_who_saw_or_noticed_target_rejected(trailing_clause: str) -> None:
    """N3: V2 with a trailing ', who saw|noticed <target>' contradicts the occlusion and must be rejected."""
    world = _load_stage3_world()
    ev = _ev_see_occ()
    # Valid V2 control
    valid_v2 = "In the amber room at the far-side alcove, the felt curtain hid the pine crate from Mira."
    assert _sentence_matches_event(valid_v2, ev, world.noise_threshold, world=world, checker_profile="d")
    assert not _sentence_matches_event(valid_v2, ev, world.noise_threshold, world=world, checker_profile="c")

    # Contradictory trailing relative clause on V2
    invalid_v2 = f"In the amber room at the far-side alcove, the felt curtain hid the pine crate from Mira{trailing_clause}"
    assert not _sentence_matches_event(invalid_v2, ev, world.noise_threshold, world=world, checker_profile="d")
    assert not _sentence_matches_event(invalid_v2, ev, world.noise_threshold, world=world, checker_profile="c")


@pytest.mark.parametrize(
    "case_name",
    ["trailing_semicolon", "trailing_colon", "doubled_inner_spaces"],
)
@pytest.mark.parametrize("profile", ["c", "d"])
def test_n6_quote_normalisation_rejects_trailing_semicolon_colon_and_doubled_inner_spaces(
    profile: str,
    case_name: str,
) -> None:
    """N6: Quote normalisation is lowercase + trim outer whitespace + strip at most one trailing `.`, `,`, `!` or `?`.
    It must NOT strip `;` or `:`, and must NOT collapse inner whitespace.
    """
    world = _load_stage3_world()
    ev = _ev_say()

    # Valid controls: trailing `.`, `,`, `!`, `?`, or bare
    for punct in (".", ",", "!", "?", ""):
        valid_sent = f'In the amber room, Mira said at 55 dB, "The copper token is stored in the tin locker{punct}"'
        assert _sentence_matches_event(valid_sent, ev, world.noise_threshold, world=world, checker_profile=profile)

    invalid_by_case = {
        "trailing_semicolon": 'In the amber room, Mira said at 55 dB, "The copper token is stored in the tin locker;"',
        "trailing_colon": 'In the amber room, Mira said at 55 dB, "The copper token is stored in the tin locker:"',
        "doubled_inner_spaces": 'In the amber room, Mira said at 55 dB, "The copper  token is stored in the tin locker."',
    }
    assert not _sentence_matches_event(
        invalid_by_case[case_name], ev, world.noise_threshold, world=world, checker_profile=profile
    )


def test_n7_cli_help_accepted_forms_line_says_four_lines(capsys: pytest.CaptureFixture[str]) -> None:
    """N7: The CLI `--help` for `--accepted-forms-line` states that the flag adds four lines, not two."""
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["--help"])
    assert exc_info.value.code == 0
    help_out = capsys.readouterr().out
    normalized_help = " ".join(help_out.split())
    assert "four accepted-forms lines" in normalized_help
    assert "two accepted-forms lines" not in normalized_help


def test_n2_m3_was_verb_by_actor_from_to_order_pinned() -> None:
    """N2: M3 accepts `was <verb> by <actor> from the <from> to the <to>` in Profile D (and rejects in Profile C)."""
    world = _load_stage3_world()
    ev = _ev_move()
    sent_by_first = (
        "In the amber room, the copper token was moved by Mira from the pine crate to the tin locker."
    )
    assert not _sentence_matches_event(sent_by_first, ev, world.noise_threshold, world=world, checker_profile="c")
    assert _sentence_matches_event(sent_by_first, ev, world.noise_threshold, world=world, checker_profile="d")
