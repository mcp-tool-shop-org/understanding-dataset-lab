"""World state and deterministic event-log generator for udlab release 1."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import itertools
import random
from typing import Literal

EventKind = Literal["enter", "leave", "move", "say", "see", "message"]

CONTAINER_BAY_STATION: str = "container bay"
FAR_SIDE_ALCOVE_STATION: str = "far-side alcove"

OCCLUSION_RULE: str = (
    "In each room, the room's occluder stands between the far-side alcove "
    "(where the room fixture and side mechanism sit) and the container bay "
    "(where the room's two storage containers stand). "
    "An agent at the container bay is in open view of both containers and any "
    "event at them (occluded=False); an agent at the far-side alcove is on the "
    "far side of the occluder, which blocks line of sight to events at, and the "
    "contents of, the room's containers (occluded=True)."
)

FACT_ID_PERMUTATIONS: tuple[tuple[str, str, str, str], ...] = tuple(
    itertools.permutations(("F1", "F2", "F3", "F4"))
)


@dataclass(frozen=True)
class StakeEdge:
    """Represents `depends_on(agent, variable)` for active or absent parties."""

    agent: str
    variable: str
    description: str


@dataclass(frozen=True)
class FactSpec:
    """Canonical world fact tracked by fact_id (`F1`, `F2`, ...)."""

    fact_id: str
    variable: str
    value: str
    description: str


@dataclass(frozen=True)
class AgentGoal:
    """An active agent's goal `G_a` and its causal precondition fact IDs."""

    agent: str
    goal_text: str
    causal_preconditions: tuple[str, ...]


@dataclass(frozen=True)
class Event:
    """A single chronological event in the world event log."""

    step: int
    kind: EventKind
    actor: str
    room: str
    target_object: str | None = None
    from_location: str | None = None
    to_location: str | None = None
    utterance: str | None = None
    recipient: str | None = None
    occluded: bool = False
    occluder: str | None = None
    noise_level: int = 20
    fact_id: str | None = None
    fact_variable: str | None = None
    fact_value: str | None = None
    is_bluff: bool = False
    stated_reason: str | None = None


@dataclass(frozen=True)
class UnknowableSpec:
    """Two counterfactual world alternatives with identical event-log traces (`U`)."""

    subtype: Literal["who", "why"]
    question_text: str
    alternative_a: str
    alternative_b: str
    target_agent: str
    side_action_desc: str
    trace_a: tuple[Event, ...]
    trace_b: tuple[Event, ...]
    latent_state_a: dict[str, str] | None = None
    latent_state_b: dict[str, str] | None = None


@dataclass(frozen=True)
class SettledTwinSpec:
    """The settled twin (`S`) matching the `Who`/`Why` question stem distribution."""

    subtype: Literal["who", "why"]
    question_text: str
    settled_answer: str
    target_agent: str
    side_action_desc: str


@dataclass(frozen=True)
class ScenarioConfig:
    """Balanced structural parameters driving world construction."""

    scenario_id: str
    seed: int
    p_can_see: bool
    k_knows: bool
    b1_true_belief: bool
    b2_true_belief: bool
    x_notified: bool
    u_is_who: bool
    num_agents: int
    num_absent: int
    topology_kind: str


@dataclass(frozen=True)
class World:
    """Complete generated scenario world state and event log."""

    scenario_id: str
    seed: int
    domain_name: str
    rooms: tuple[str, ...]
    doorways: tuple[tuple[str, str], ...]
    objects: tuple[str, ...]
    containers_by_room: dict[str, tuple[str, ...]]
    occluders_by_room: dict[str, str]
    fixtures_by_room: dict[str, str]
    noise_threshold: int
    agents: tuple[str, ...]
    absent_parties: tuple[str, ...]
    initial_agent_rooms: dict[str, str]
    stake_edges: tuple[StakeEdge, ...]
    goals: dict[str, AgentGoal]
    facts: dict[str, FactSpec]
    events: tuple[Event, ...]
    unknowable_spec: UnknowableSpec
    settled_twin_spec: SettledTwinSpec
    primary_object: str
    primary_variable: str
    secondary_object: str
    secondary_variable: str
    aux_object: str
    aux_variable: str
    bluff_event_step: int
    bluffer_agent: str
    bluff_listener: str
    bluff_claimed_location: str
    p_agent: str
    k_agent: str
    b1_agent: str
    b2_observer: str
    b2_actor: str
    d_helper: str
    d_target: str
    primary_absent: str
    config: ScenarioConfig


@dataclass(frozen=True)
class DomainSpec:
    """Vocabulary specification for a physical setting."""

    name: str
    rooms: tuple[str, str, str, str]
    objects: tuple[str, str, str]
    containers: tuple[tuple[str, str], tuple[str, str], tuple[str, str], tuple[str, str]]
    occluders: tuple[str, str, str, str]
    fixtures: tuple[str, str, str, str]
    note_item: str
    side_mechanism: str
    side_reasons: tuple[str, str]
    goal_templates: tuple[str, str, str]


DOMAINS: tuple[DomainSpec, ...] = (
    DomainSpec(
        name="archival_registry",
        rooms=("vault room", "reading room", "bindery room", "sorting room"),
        objects=("brass ledger", "seal stamp", "index reel"),
        containers=(
            ("oak drawer", "steel cabinet"),
            ("slate tray", "cedar shelf"),
            ("linen bin", "copper rack"),
            ("maple cubby", "iron safe"),
        ),
        occluders=("frosted partition", "tall bookstack", "canvas screen", "rolling pillar"),
        fixtures=("wall barometer", "wall clock", "humidity dial", "ventilation grille"),
        note_item="unsigned transfer slip",
        side_mechanism="overhead louver",
        side_reasons=("to clear binding dust from the bench", "to equalize draft pressure before sealing"),
        goal_templates=(
            "complete the quarterly audit packet",
            "verify the municipal deed transfer",
            "catalog the incoming charter folio",
        ),
    ),
    DomainSpec(
        name="conservatory_lab",
        rooms=("potting hall", "seed annex", "mist chamber", "drying loft"),
        objects=("grafting knife", "spore vial", "ph meter"),
        containers=(
            ("zinc trough", "wicker basket"),
            ("glass jar", "cork rack"),
            ("ceramic basin", "bamboo crate"),
            ("enamel box", "slatted bin"),
        ),
        occluders=("fern trellis", "burlap curtain", "potting locker", "reed divider"),
        fixtures=("soil thermometer", "water gauge", "light sensor", "drain valve"),
        note_item="unmarked watering tag",
        side_mechanism="roof vent crank",
        side_reasons=("to vent excess condensation from the glass", "to cool the upper bench before noon"),
        goal_templates=(
            "prepare the orchid propagation tray",
            "stabilize the fern cultivar sample",
            "log the highland seed shipment",
        ),
    ),
    DomainSpec(
        name="observatory_station",
        rooms=("dome gallery", "chart room", "clock vault", "lens workshop"),
        objects=("prism mount", "star ephemeris", "calibration weight"),
        containers=(
            ("velvet case", "brass cradle"),
            ("mahogany chest", "aluminum tray"),
            ("felt drawer", "pewter canister"),
            ("graphite box", "leather satchel"),
        ),
        occluders=("telescope pier", "blackout drape", "mirror shroud", "baffle board"),
        fixtures=("sidereal dial", "azimuth scale", "shutter lever", "level bubble"),
        note_item="unsigned zenith card",
        side_mechanism="counterweight clamp",
        side_reasons=("to balance the meridian guide rail", "to lock the slewing gear during maintenance"),
        goal_templates=(
            "align the meridian transit spectrograph",
            "record the lunar occultation series",
            "verify the chronometer drift log",
        ),
    ),
    DomainSpec(
        name="customs_wharf",
        rooms=("manifest hall", "weighing bay", "bonded cage", "sample locker"),
        objects=("tariff stamp", "assay scale", "cargo key"),
        containers=(
            ("canvas pouch", "tin lockbox"),
            ("pine crate", "galvanized tub"),
            ("rope hamper", "steel locker"),
            ("tarpaulin bin", "oak casket"),
        ),
        occluders=("bale stack", "timber bulkhead", "tarpaulin curtain", "hoist column"),
        fixtures=("tide gauge", "crane winch", "scale pointer", "hatch bolt"),
        note_item="unsigned clearance chit",
        side_mechanism="sluice gate wheel",
        side_reasons=("to drain wash water from the deck", "to clear silt from the intake grate"),
        goal_templates=(
            "clear the coastal freight manifest",
            "seal the bonded spice consignment",
            "audit the harbour tonnage register",
        ),
    ),
    DomainSpec(
        name="restoration_studio",
        rooms=("varnish room", "framing bay", "pigment vault", "kiln alcove"),
        objects=("gilding brush", "agate burnisher", "resin flask"),
        containers=(
            ("walnut caddy", "porcelain dish"),
            ("ash drawer", "tin canister"),
            ("stoneware crock", "brass tube"),
            ("beech tray", "slate coffer"),
        ),
        occluders=("easel screen", "drying rack", "linen backdrop", "plaster column"),
        fixtures=("hygrometer dial", "exhaust hood", "kiln pyrometer", "solvent trap"),
        note_item="unsigned condition card",
        side_mechanism="damper rod",
        side_reasons=("to draw turpentine fumes out of the bay", "to steady the airflow over the canvas"),
        goal_templates=(
            "stabilize the sixteenth-century panel",
            "apply the protective mastic coat",
            "mount the restored altarpiece frame",
        ),
    ),
    DomainSpec(
        name="railway_dispatch",
        rooms=("signal room", "telegraph bay", "timetable hall", "lamp room"),
        objects=("brass token", "signal lantern", "switch key"),
        containers=(
            ("iron rack", "leather pouch"),
            ("oak pigeonhole", "steel bin"),
            ("copper box", "enamel tray"),
            ("canvas bag", "wall holster"),
        ),
        occluders=("lever frame", "relay cabinet", "chalkboard stand", "partition wall"),
        fixtures=("block bell", "line ammeter", "pressure manifold", "repeater arm"),
        note_item="unsigned track memo",
        side_mechanism="bell muffler lever",
        side_reasons=("to test the spring tension on the bell", "to reset the local relay indicator"),
        goal_templates=(
            "authorize the northbound express clearance",
            "log the single-line token exchange",
            "update the junction switching sheet",
        ),
    ),
)

AGENT_NAME_POOLS: tuple[tuple[str, ...], ...] = (
    ("Ana", "Ben", "Cara", "Dan", "Eva", "Finn", "Gia"),
    ("Iris", "Jude", "Kira", "Leo", "Maya", "Noel", "Opal"),
    ("Nina", "Omar", "Pia", "Quinn", "Rory", "Sara", "Theo"),
    ("Vera", "Will", "Yara", "Zane", "Alma", "Bram", "Cleo"),
    ("Dora", "Eli", "Faye", "Gabe", "Hana", "Ivan", "June"),
    ("Lena", "Milo", "Nora", "Owen", "Petra", "Reid", "Tess"),
)

TOPOLOGIES: tuple[str, ...] = ("chain", "cycle", "star", "tee", "diamond")


def _build_doorways(rooms: tuple[str, ...], topology: str) -> tuple[tuple[str, str], ...]:
    """Construct an undirected doorway graph over 4 rooms."""
    r0, r1, r2, r3 = rooms
    if topology == "chain":
        edges = ((r0, r1), (r1, r2), (r2, r3))
    elif topology == "cycle":
        edges = ((r0, r1), (r1, r2), (r2, r3), (r3, r0))
    elif topology == "star":
        edges = ((r0, r1), (r0, r2), (r0, r3))
    elif topology == "tee":
        edges = ((r0, r1), (r1, r2), (r1, r3))
    else:  # diamond
        edges = ((r0, r1), (r0, r2), (r1, r3), (r2, r3))
    return edges


def are_rooms_adjacent(
    room_a: str,
    room_b: str,
    doorways: tuple[tuple[str, str], ...],
) -> bool:
    """Return True if two distinct rooms share a doorway."""
    if room_a == room_b:
        return False
    return (room_a, room_b) in doorways or (room_b, room_a) in doorways


def shortest_doorway_path(
    start_room: str,
    end_room: str,
    doorways: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    """Return the shortest doorway path `(start_room, ..., end_room)` over `doorways`."""
    if start_room == end_room:
        return (start_room,)
    queue: deque[tuple[str, ...]] = deque([(start_room,)])
    visited: set[str] = {start_room}
    neighbors_by_room: dict[str, list[str]] = {}
    for r_a, r_b in doorways:
        neighbors_by_room.setdefault(r_a, []).append(r_b)
        neighbors_by_room.setdefault(r_b, []).append(r_a)

    while queue:
        path = queue.popleft()
        cur = path[-1]
        for nxt in neighbors_by_room.get(cur, ()):
            if nxt == end_room:
                return (*path, nxt)
            if nxt not in visited:
                visited.add(nxt)
                queue.append((*path, nxt))
    raise ValueError(f"No doorway path between '{start_room}' and '{end_room}' in {doorways}.")


def _append_doorway_walk(
    events: list[Event],
    step: int,
    actor: str,
    start_room: str,
    end_room: str,
    doorways: tuple[tuple[str, str], ...],
) -> int:
    """Append one `leave`/`enter` pair per doorway hop along the shortest path from `start_room` to `end_room`."""
    path = shortest_doorway_path(start_room, end_room, doorways)
    for hop_idx in range(len(path) - 1):
        r_from = path[hop_idx]
        r_to = path[hop_idx + 1]
        events.append(
            Event(
                step=step,
                kind="leave",
                actor=actor,
                room=r_from,
                from_location=r_from,
                to_location=r_to,
            )
        )
        step += 1
        events.append(
            Event(
                step=step,
                kind="enter",
                actor=actor,
                room=r_to,
                from_location=r_from,
                to_location=r_to,
            )
        )
        step += 1
    return step


def is_station_transition_event(event: Event, world: World | None = None) -> bool:
    """Return True if `event` moves an agent between `container bay` and `far-side alcove` within a room."""
    if event.kind != "move":
        return False
    valid_stations = (CONTAINER_BAY_STATION, FAR_SIDE_ALCOVE_STATION)
    if event.from_location not in valid_stations or event.to_location not in valid_stations:
        return False
    if world is not None and event.room in world.occluders_by_room:
        return event.target_object == world.occluders_by_room[event.room]
    return True


def is_container_event(event: Event, world: World) -> bool:
    """Return True if `event` is a `move` or `see` involving a storage container in `world`."""
    if event.kind not in ("move", "see") or is_station_transition_event(event, world):
        return False
    all_containers = {c for pair in world.containers_by_room.values() for c in pair}
    return (
        event.target_object in all_containers
        or event.from_location in all_containers
        or event.to_location in all_containers
    )


def validate_topology(world: World) -> None:
    """Validate that every `leave`/`enter` hop is a doorway in `world.doorways` and every agent's location sequence is continuous."""
    valid_rooms = set(world.rooms)
    current_room: dict[str, str | None] = {}
    pending_hop: dict[str, tuple[str, str] | None] = {}

    for ag in world.agents:
        init_r = world.initial_agent_rooms.get(ag)
        if init_r is None or init_r not in valid_rooms:
            raise ValueError(
                f"Scenario {world.scenario_id} topology error: agent '{ag}' has invalid initial room '{init_r}'."
            )
        current_room[ag] = init_r
        pending_hop[ag] = None

    for ev in world.events:
        if ev.actor not in current_room:
            raise ValueError(
                f"Scenario {world.scenario_id} step {ev.step} topology error: unknown active agent '{ev.actor}'."
            )
        if ev.room not in valid_rooms:
            raise ValueError(
                f"Scenario {world.scenario_id} step {ev.step} topology error: invalid room '{ev.room}'."
            )

        if ev.kind == "leave":
            if pending_hop[ev.actor] is not None:
                raise ValueError(
                    f"Scenario {world.scenario_id} step {ev.step} topology error: "
                    f"{ev.actor} leaves '{ev.room}' while already in transit {pending_hop[ev.actor]} (not continuous)."
                )
            if current_room[ev.actor] != ev.room:
                raise ValueError(
                    f"Scenario {world.scenario_id} step {ev.step} topology error: "
                    f"{ev.actor} is not in room '{ev.room}' (current location is '{current_room[ev.actor]}', not continuous)."
                )
            if ev.from_location != ev.room or ev.to_location is None or ev.to_location not in valid_rooms:
                raise ValueError(
                    f"Scenario {world.scenario_id} step {ev.step} topology error: "
                    f"invalid leave endpoints {ev.from_location!r} -> {ev.to_location!r}."
                )
            if not are_rooms_adjacent(ev.from_location, ev.to_location, world.doorways):
                raise ValueError(
                    f"Scenario {world.scenario_id} step {ev.step} topology error: "
                    f"leave hop '{ev.from_location}' -> '{ev.to_location}' is not a doorway (not adjacent in doorways)."
                )
            current_room[ev.actor] = None
            pending_hop[ev.actor] = (ev.from_location, ev.to_location)

        elif ev.kind == "enter":
            hop = pending_hop[ev.actor]
            if hop is None:
                raise ValueError(
                    f"Scenario {world.scenario_id} step {ev.step} topology error: "
                    f"{ev.actor} enters '{ev.room}' without a preceding leave (location sequence not continuous)."
                )
            if (ev.from_location, ev.to_location) != hop or ev.room != ev.to_location:
                raise ValueError(
                    f"Scenario {world.scenario_id} step {ev.step} topology error: "
                    f"{ev.actor} enter endpoints ({ev.from_location!r} -> {ev.to_location!r} in {ev.room!r}) "
                    f"do not match pending hop {hop} (not continuous)."
                )
            if not are_rooms_adjacent(ev.from_location, ev.to_location, world.doorways):
                raise ValueError(
                    f"Scenario {world.scenario_id} step {ev.step} topology error: "
                    f"enter hop '{ev.from_location}' -> '{ev.to_location}' is not a doorway (not adjacent in doorways)."
                )
            current_room[ev.actor] = ev.room
            pending_hop[ev.actor] = None

        else:
            if pending_hop[ev.actor] is not None or current_room[ev.actor] != ev.room:
                raise ValueError(
                    f"Scenario {world.scenario_id} step {ev.step} topology error: "
                    f"{ev.actor} performs '{ev.kind}' in room '{ev.room}' while not in room "
                    f"(current location is '{current_room[ev.actor]}', not continuous)."
                )

    for ag, hop in pending_hop.items():
        if hop is not None:
            raise ValueError(
                f"Scenario {world.scenario_id} topology error: agent '{ag}' left in transit {hop} at end of log."
            )


def make_balanced_configs(count: int, base_seed: int = 20261030) -> list[ScenarioConfig]:
    """Build `count` scenario configurations with exact 50/50 balance and orthogonal label axes."""
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(base_seed)
    design_rng = random.Random(base_seed ^ 0x3C6EF372)

    # Construct 4-bit state vectors `s = (s0, s1, s2, s3)` arranged inblocks of 4 quads (16 scenarios).
    # Each quad consists of two complementary pairs `{z, z ^ 0b1111}` and `{z ^ 0b0110, z ^ 0b1001}`
    # for `z in {(0,0,0,0), (0,0,0,1), (0,1,0,0), (0,1,0,1)}`.
    # Mapping the 7 binary axes to distinct odd-weight linear combinations over F_2^4:
    #   U       = s0             (1, 0, 0, 0)
    #   B1      = s1             (0, 1, 0, 0)
    #   B2      = s2             (0, 0, 1, 0)
    #   loc_bit = s3             (0, 0, 0, 1)
    #   X       = s0 ^ s1 ^ s2   (1, 1, 1, 0)
    #   P       = s0 ^ s2 ^ s3   (1, 0, 1, 1)
    #   K       = s0 ^ s1 ^ s3   (1, 1, 0, 1)
    # guarantees:
    # 1. Every consecutive pair `(2*m, 2*m + 1)` has opposite values on all 7 axes (50/50 per domain).
    # 2. Every quad of 4 scenarios has exact 1/1/1/1 balance on P x U, K x U, B1 x X, B2 x X,
    #    B1 x loc_bit, and B2 x loc_bit.
    # 3. Every block of 16 scenarios is an exact 4/4/4/4 orthogonal array on all 21 axis pairs
    #    and an exact 2x2x2 factorial on (B1, B2, loc_bit).
    quad_base_states = [0b0000, 0b0001, 0b0100, 0b0101]
    states: list[int] = []
    while len(states) < count:
        quads = list(quad_base_states)
        design_rng.shuffle(quads)
        for z in quads:
            pairs = [
                [z, z ^ 0b1111],
                [z ^ 0b0110, z ^ 0b1001],
            ]
            design_rng.shuffle(pairs)
            first_p_target = design_rng.randint(0, 1)
            for p_idx, pair in enumerate(pairs):
                target_p = first_p_target ^ p_idx
                s_first = pair[0]
                p_first = ((s_first >> 3) & 1) ^ ((s_first >> 1) & 1) ^ (s_first & 1)
                if p_first != target_p:
                    pair.reverse()
                states.extend(pair)

    configs: list[ScenarioConfig] = []
    for idx in range(count):
        s = states[idx]
        s0 = (s >> 3) & 1
        s1 = (s >> 2) & 1
        s2 = (s >> 1) & 1
        s3 = s & 1

        u_val = bool(s0)
        b1_val = bool(s1)
        b2_val = bool(s2)
        loc_bit = s3
        x_val = bool(s0 ^ s1 ^ s2)
        p_val = bool(s0 ^ s2 ^ s3)
        k_val = bool(s0 ^ s1 ^ s3)

        raw_seed = rng.randint(100_000, 999_999_999) ^ (idx * 7919)
        scen_seed = (raw_seed & ~1) | loc_bit
        num_agents = 3 + (idx % 3)  # 3, 4, or 5 active agents
        num_absent = 1 + (idx % 2)  # 1 or 2 absent parties
        topology_kind = TOPOLOGIES[idx % len(TOPOLOGIES)]
        configs.append(
            ScenarioConfig(
                scenario_id=f"scen_{idx:04d}",
                seed=scen_seed,
                p_can_see=p_val,
                k_knows=k_val,
                b1_true_belief=b1_val,
                b2_true_belief=b2_val,
                x_notified=x_val,
                u_is_who=u_val,
                num_agents=num_agents,
                num_absent=num_absent,
                topology_kind=topology_kind,
            )
        )
    return configs


def _select_room_roles(
    perm_rooms: list[str],
    doorways: tuple[tuple[str, str], ...],
    need_hub_sec_adjacent: bool,
) -> tuple[str, str, str, str]:
    """Select `(r_hub, r_work, r_sec_a, r_sec_b)` from `perm_rooms` matching doorway adjacency.

    Guarantees:
    - `(r_hub, r_work)` is always a doorway in `doorways`.
    - `r_hub` has at least one additional doorway neighbor in `{r_sec_a, r_sec_b}` for `p_can_see=False`.
    - When `need_hub_sec_adjacent` is True, `(r_hub, r_sec_a)` is a doorway in `doorways`.
    - When `need_hub_sec_adjacent` is False, prefers `(r_hub, r_sec_a)` non-adjacent with `(r_work, r_sec_a)`
      and `(r_hub, r_sec_b)` adjacent (possible in `chain`, `cycle`, `diamond`), falling back to the
      degree-3 hub in `star` and `tee`.
    """
    if need_hub_sec_adjacent:
        for p in itertools.permutations(perm_rooms):
            if (
                are_rooms_adjacent(p[0], p[1], doorways)
                and are_rooms_adjacent(p[0], p[2], doorways)
                and are_rooms_adjacent(p[1], p[2], doorways)
            ):
                return p[0], p[1], p[2], p[3]
        for p in itertools.permutations(perm_rooms):
            if are_rooms_adjacent(p[0], p[1], doorways) and are_rooms_adjacent(p[0], p[2], doorways):
                return p[0], p[1], p[2], p[3]
    else:
        for p in itertools.permutations(perm_rooms):
            if (
                are_rooms_adjacent(p[0], p[1], doorways)
                and are_rooms_adjacent(p[0], p[3], doorways)
                and not are_rooms_adjacent(p[0], p[2], doorways)
                and are_rooms_adjacent(p[1], p[2], doorways)
            ):
                return p[0], p[1], p[2], p[3]
        for p in itertools.permutations(perm_rooms):
            if (
                are_rooms_adjacent(p[0], p[1], doorways)
                and are_rooms_adjacent(p[0], p[3], doorways)
                and not are_rooms_adjacent(p[0], p[2], doorways)
            ):
                return p[0], p[1], p[2], p[3]
        for p in itertools.permutations(perm_rooms):
            if are_rooms_adjacent(p[0], p[1], doorways) and are_rooms_adjacent(p[0], p[2], doorways):
                return p[0], p[1], p[2], p[3]
    return perm_rooms[0], perm_rooms[1], perm_rooms[2], perm_rooms[3]


def _build_event_trace(
    *,
    config: ScenarioConfig,
    domain: DomainSpec,
    rooms: tuple[str, ...],
    doorways: tuple[tuple[str, str], ...],
    containers_by_room: dict[str, tuple[str, ...]],
    occluders_by_room: dict[str, str],
    fixtures_by_room: dict[str, str],
    agents: tuple[str, ...],
    absent_parties: tuple[str, ...],
    primary_object: str,
    primary_variable: str,
    secondary_object: str,
    secondary_variable: str,
    aux_object: str,
    aux_variable: str,
    noise_threshold: int,
    fact_id_map: tuple[str, str, str, str],
    u_variant: Literal["A", "B"],
) -> tuple[
    tuple[Event, ...],
    dict[str, str],
    dict[str, FactSpec],
    int,
    str,
    str,
    str,
    str,
    str,
    dict[str, str],
    bool,
]:
    """Construct a symmetric, shortcut-free chronological event trace for `config`.

    `u_variant='A'` and `u_variant='B'` represent two distinct latent world states
    whose observable projection onto `Event` is identical.
    """
    rng = random.Random(config.seed)
    a0, a1, a2 = agents[0], agents[1], agents[2]
    extra_agents = agents[3:]
    reason_a, reason_b = domain.side_reasons

    # Construct the counterfactual latent world state for `u_variant` ('A' vs 'B'):
    if config.u_is_who:
        latent_note_placer = a1 if u_variant == "A" else a2
        latent_side_reason = reason_a
    else:
        latent_note_placer = a2
        latent_side_reason = reason_a if u_variant == "A" else reason_b

    latent_state: dict[str, str] = {
        "u_variant": u_variant,
        "note_item_placer": latent_note_placer,
        "side_mechanism_reason": latent_side_reason,
    }

    fid_primary, fid_shared, fid_secondary, fid_aux = fact_id_map

    # Permute the 4 domain rooms per scenario so no room name carries a prior for any key.
    # In `k_doorway_mode` (50% of scenarios), `a2` speaks at or above `noise_threshold` (60 dB) and `a0`
    # enters `r_hub`, so `K` is decided directly by whether `(r_hub, r_sec_a)` is a doorway in `doorways`.
    perm_rooms = list(rooms)
    rng.shuffle(perm_rooms)
    k_doorway_mode = bool((config.seed >> 2) & 1)
    need_adj = config.k_knows if k_doorway_mode else True
    r_hub, r_work, r_sec_a, r_sec_b = _select_room_roles(perm_rooms, doorways, need_adj)

    # Assign which container in r_work is `cont_init` vs `cont_final` using `config.seed & 1`,
    # which is stratified 50/50 inside every `(b1_true_belief, b2_true_belief)` cell.
    c0, c1 = containers_by_room[r_work]
    if (config.seed & 1) == 1:
        cont_init, cont_final = c1, c0
    else:
        cont_init, cont_final = c0, c1

    cont_sec_init, cont_sec_final = containers_by_room[r_sec_a]
    cont_hub_init, cont_hub_final = containers_by_room[r_hub]

    # All core agents begin in `r_hub` to witness the shared baseline fact together.
    initial_rooms: dict[str, str] = {
        a0: r_hub,
        a1: r_hub,
        a2: r_hub,
    }
    for idx, ea in enumerate(extra_agents):
        initial_rooms[ea] = perm_rooms[(idx + 1) % len(perm_rooms)]

    events: list[Event] = []
    step = 1

    # Step 1: Shared baseline fact (`fid_shared`) observed by a0, a1, a2 in r_hub.
    shared_var = f"baseline:{aux_object}"
    shared_val = cont_hub_init
    events.append(
        Event(
            step=step,
            kind="see",
            actor=a0,
            room=r_hub,
            target_object=aux_object,
            from_location=shared_val,
            to_location=shared_val,
            occluded=False,
            occluder=None,
            noise_level=20,
            fact_id=fid_shared,
            fact_variable=shared_var,
            fact_value=shared_val,
        )
    )
    step += 1

    # Move a1 and a2 from r_hub into r_work along doorway (r_hub, r_work) to initialize primary_object.
    for actor in (a1, a2):
        step = _append_doorway_walk(events, step, actor, r_hub, r_work, doorways)

    current_agent_rooms: dict[str, str] = dict(initial_rooms)
    current_agent_rooms[a1] = r_work
    current_agent_rooms[a2] = r_work

    # Optional extra-agent movements along doorway edges (creates diverse canonical event skeletons).
    for ea_idx, ea in enumerate(extra_agents):
        ea_cur = current_agent_rooms[ea]
        ea_neighbors = [r for r in perm_rooms if are_rooms_adjacent(ea_cur, r, doorways)]
        ea_next = ea_neighbors[(ea_idx + (config.seed % 2)) % len(ea_neighbors)]
        if (config.seed >> ea_idx) & 1:
            step = _append_doorway_walk(events, step, ea, ea_cur, ea_next, doorways)
            current_agent_rooms[ea] = ea_next
        if (config.seed + ea_idx) % 3 == 0:
            ea_room = current_agent_rooms[ea]
            events.append(
                Event(
                    step=step,
                    kind="see",
                    actor=ea,
                    room=ea_room,
                    target_object=fixtures_by_room[ea_room],
                    from_location=FAR_SIDE_ALCOVE_STATION,
                    to_location=FAR_SIDE_ALCOVE_STATION,
                    occluded=False,
                )
            )
            step += 1

    occ_work = occluders_by_room[r_work]
    a1_current_room = r_work

    # Execute the initial placement of `primary_object` in `r_work` using the single station-based
    # occlusion rule (`container bay` in open view vs `far-side alcove` behind `occ_work`) with exact
    # mention parity (`count(cont_init) == count(cont_final)`) across all 4 `(B1, B2)` cells:
    if not config.b1_true_belief and config.b2_true_belief:
        # Cell (B1=False, B2=True):
        # 1. Both a1 and a2 at `container bay` see a2 place `primary_object` in `cont_final` (`F0a`).
        # 2. a2 steps behind `occ_work` to `far-side alcove`.
        # 3. While a2 is occluded at `far-side alcove`, a1 moves `primary_object` from `cont_final` to `cont_init` (`F0`).
        # 4. a1 steps behind `occ_work` to `far-side alcove`, and a2 returns to `container bay` before the final move.
        events.append(
            Event(
                step=step,
                kind="move",
                actor=a2,
                room=r_work,
                target_object=primary_object,
                from_location="intake bench",
                to_location=cont_final,
                occluded=False,
                fact_id="F0a",
                fact_variable=primary_variable,
                fact_value=cont_final,
            )
        )
        step += 1
        events.append(
            Event(
                step=step,
                kind="move",
                actor=a2,
                room=r_work,
                target_object=occ_work,
                from_location=CONTAINER_BAY_STATION,
                to_location=FAR_SIDE_ALCOVE_STATION,
                occluded=False,
            )
        )
        step += 1
        events.append(
            Event(
                step=step,
                kind="move",
                actor=a1,
                room=r_work,
                target_object=primary_object,
                from_location=cont_final,
                to_location=cont_init,
                occluded=False,
                fact_id="F0",
                fact_variable=primary_variable,
                fact_value=cont_init,
            )
        )
        step += 1
        events.append(
            Event(
                step=step,
                kind="move",
                actor=a1,
                room=r_work,
                target_object=occ_work,
                from_location=CONTAINER_BAY_STATION,
                to_location=FAR_SIDE_ALCOVE_STATION,
                occluded=False,
            )
        )
        step += 1
        events.append(
            Event(
                step=step,
                kind="move",
                actor=a2,
                room=r_work,
                target_object=occ_work,
                from_location=FAR_SIDE_ALCOVE_STATION,
                to_location=CONTAINER_BAY_STATION,
                occluded=False,
            )
        )
        step += 1
    else:
        # First inspect `cont_final` at `container bay` in `r_work` before placing `primary_object` in `cont_init`
        # so both `cont_init` and `cont_final` have equal mention counts before and after the final move.
        events.append(
            Event(
                step=step,
                kind="see",
                actor=a2,
                room=r_work,
                target_object=cont_final,
                from_location=CONTAINER_BAY_STATION,
                to_location=CONTAINER_BAY_STATION,
                occluded=False,
            )
        )
        step += 1
        events.append(
            Event(
                step=step,
                kind="move",
                actor=a2,
                room=r_work,
                target_object=primary_object,
                from_location="intake bench",
                to_location=cont_init,
                occluded=False,
                fact_id="F0",
                fact_variable=primary_variable,
                fact_value=cont_init,
            )
        )
        step += 1
        if not (config.b1_true_belief and config.b2_true_belief):
            # In (B1=True, B2=False) and (B1=False, B2=False), a1 steps behind `occ_work` to `far-side alcove`
            # before the final move so `occ_work` blocks a1's view of the final move.
            events.append(
                Event(
                    step=step,
                    kind="move",
                    actor=a1,
                    room=r_work,
                    target_object=occ_work,
                    from_location=CONTAINER_BAY_STATION,
                    to_location=FAR_SIDE_ALCOVE_STATION,
                    occluded=False,
                )
            )
            step += 1

    # Symmetric 1-hop doorway transition for `a0` before the final move of `primary_object` (`p_can_see`):
    r_p_false = r_sec_b if are_rooms_adjacent(r_hub, r_sec_b, doorways) else r_sec_a
    r_p_dest = r_work if config.p_can_see else r_p_false
    step = _append_doorway_walk(events, step, a0, r_hub, r_p_dest, doorways)
    current_agent_rooms[a0] = r_p_dest

    # Final move of `primary_object` in `r_work` from `cont_init` to `cont_final` (`fid_primary`) at `container bay`.
    events.append(
        Event(
            step=step,
            kind="move",
            actor=a2,
            room=r_work,
            target_object=primary_object,
            from_location=cont_init,
            to_location=cont_final,
            occluded=False,
            occluder=None,
            noise_level=20,
            fact_id=fid_primary,
            fact_variable=primary_variable,
            fact_value=cont_final,
        )
    )
    step += 1

    # Immediate post-move fixture check by `a0` in `r_p_dest` (keeps sentence distance == 1).
    events.append(
        Event(
            step=step,
            kind="see",
            actor=a0,
            room=r_p_dest,
            target_object=fixtures_by_room[r_p_dest],
            from_location=FAR_SIDE_ALCOVE_STATION,
            to_location=FAR_SIDE_ALCOVE_STATION,
            occluded=False,
        )
    )
    step += 1

    # Transition `a2` from `r_work` to `r_sec_a` along the shortest doorway path for the secondary object phase.
    step = _append_doorway_walk(events, step, a2, r_work, r_sec_a, doorways)
    current_agent_rooms[a2] = r_sec_a

    # Post-move steps in `r_work` for `a1` (maintaining exact `(3, 3)` container mentions and ending at `far-side alcove`):
    if config.b1_true_belief and not config.b2_true_belief:
        # While `a2` is in `r_sec_a`, `a1` steps around `occ_work` to `container bay`, inspects `cont_init`
        # and `primary_object` at `cont_final` in open view (`occluded=False`), then steps back to `far-side alcove`.
        events.append(
            Event(
                step=step,
                kind="move",
                actor=a1,
                room=r_work,
                target_object=occ_work,
                from_location=FAR_SIDE_ALCOVE_STATION,
                to_location=CONTAINER_BAY_STATION,
                occluded=False,
            )
        )
        step += 1
        events.append(
            Event(
                step=step,
                kind="see",
                actor=a1,
                room=r_work,
                target_object=cont_init,
                from_location=CONTAINER_BAY_STATION,
                to_location=CONTAINER_BAY_STATION,
                occluded=False,
            )
        )
        step += 1
        events.append(
            Event(
                step=step,
                kind="see",
                actor=a1,
                room=r_work,
                target_object=primary_object,
                from_location=cont_final,
                to_location=cont_final,
                occluded=False,
                fact_id=fid_primary,
                fact_variable=primary_variable,
                fact_value=cont_final,
            )
        )
        step += 1
        events.append(
            Event(
                step=step,
                kind="move",
                actor=a1,
                room=r_work,
                target_object=occ_work,
                from_location=CONTAINER_BAY_STATION,
                to_location=FAR_SIDE_ALCOVE_STATION,
                occluded=False,
            )
        )
        step += 1
    elif not config.b1_true_belief and config.b2_true_belief:
        # `a1` is already at `far-side alcove` behind `occ_work`; looks toward `cont_init` (`occluded=True`).
        events.append(
            Event(
                step=step,
                kind="see",
                actor=a1,
                room=r_work,
                target_object=cont_init,
                from_location=FAR_SIDE_ALCOVE_STATION,
                to_location=FAR_SIDE_ALCOVE_STATION,
                occluded=True,
                occluder=occ_work,
            )
        )
        step += 1
    else:
        if config.b1_true_belief and config.b2_true_belief:
            # `a1` steps behind `occ_work` to `far-side alcove` before looking toward `cont_init` and `cont_final`.
            events.append(
                Event(
                    step=step,
                    kind="move",
                    actor=a1,
                    room=r_work,
                    target_object=occ_work,
                    from_location=CONTAINER_BAY_STATION,
                    to_location=FAR_SIDE_ALCOVE_STATION,
                    occluded=False,
                )
            )
            step += 1
        for c_eq in (cont_init, cont_final):
            events.append(
                Event(
                    step=step,
                    kind="see",
                    actor=a1,
                    room=r_work,
                    target_object=c_eq,
                    from_location=FAR_SIDE_ALCOVE_STATION,
                    to_location=FAR_SIDE_ALCOVE_STATION,
                    occluded=True,
                    occluder=occ_work,
                )
            )
            step += 1

    # Aux reserve fact (`fid_aux`): `a2` observes `aux_object` reserve in `r_sec_a` at `cont_sec_init` (known to `a2`, unknown to `a1`).
    events.append(
        Event(
            step=step,
            kind="see",
            actor=a2,
            room=r_sec_a,
            target_object=aux_object,
            from_location=cont_sec_init,
            to_location=cont_sec_init,
            occluded=False,
            noise_level=20,
            fact_id=fid_aux,
            fact_variable=aux_variable,
            fact_value=cont_sec_init,
        )
    )
    step += 1

    # Symmetric 1-hop doorway transition for `a0` before `a2` moves and speaks about `secondary_object` (`k_knows`):
    if not k_doorway_mode and config.k_knows and are_rooms_adjacent(r_p_dest, r_sec_a, doorways):
        r_k_dest = r_sec_a
        k_say_noise = noise_threshold - 10
    else:
        r_k_dest = r_hub
        if config.k_knows:
            k_say_noise = noise_threshold + 10
        else:
            k_say_noise = (
                noise_threshold + 10
                if not are_rooms_adjacent(r_hub, r_sec_a, doorways)
                else noise_threshold - 10
            )

    a1_hears_secondary = (
        a1_current_room == r_sec_a
        or (are_rooms_adjacent(a1_current_room, r_sec_a, doorways) and k_say_noise >= noise_threshold)
    )

    step = _append_doorway_walk(events, step, a0, r_p_dest, r_k_dest, doorways)
    current_agent_rooms[a0] = r_k_dest

    # Secondary fact (`fid_secondary`): `a2` moves `secondary_object` in `r_sec_a` from `cont_sec_init` to `cont_sec_final`.
    events.append(
        Event(
            step=step,
            kind="move",
            actor=a2,
            room=r_sec_a,
            target_object=secondary_object,
            from_location=cont_sec_init,
            to_location=cont_sec_final,
            occluded=False,
            noise_level=20,
            fact_id=fid_secondary,
            fact_variable=secondary_variable,
            fact_value=cont_sec_final,
        )
    )
    step += 1

    events.append(
        Event(
            step=step,
            kind="say",
            actor=a2,
            room=r_sec_a,
            target_object=secondary_object,
            to_location=cont_sec_final,
            utterance=f"The {secondary_object} is stored in the {cont_sec_final}.",
            occluded=False,
            noise_level=k_say_noise,
            fact_id=fid_secondary,
            fact_variable=secondary_variable,
            fact_value=cont_sec_final,
        )
    )
    step += 1

    # Immediate post-say fixture check by `a0` in `r_k_dest` (keeps sentence distance == 1).
    events.append(
        Event(
            step=step,
            kind="see",
            actor=a0,
            room=r_k_dest,
            target_object=fixtures_by_room[r_k_dest],
            from_location=FAR_SIDE_ALCOVE_STATION,
            to_location=FAR_SIDE_ALCOVE_STATION,
            occluded=False,
        )
    )
    step += 1

    # When `x_notified` is True and `cont_init == c0` (`cont_final == c1`), `a1` looks toward `cont_init` (`c0`)
    # from `far-side alcove` before `a2`'s message mentioning `cont_final` (`c1`) so `c1` is always last-mentioned.
    if config.x_notified and cont_init == c0:
        events.append(
            Event(
                step=step,
                kind="see",
                actor=a1,
                room=r_work,
                target_object=cont_init,
                from_location=FAR_SIDE_ALCOVE_STATION,
                to_location=FAR_SIDE_ALCOVE_STATION,
                occluded=True,
                occluder=occ_work,
            )
        )
        step += 1

    # Absent party message step (`x_notified`):
    primary_absent = absent_parties[0]
    if config.x_notified:
        events.append(
            Event(
                step=step,
                kind="message",
                actor=a2,
                room=r_sec_a,
                target_object=primary_object,
                to_location=cont_final,
                recipient=primary_absent,
                utterance=f"Update for {primary_absent}: the {primary_object} is stored at the {cont_final}.",
                fact_id=fid_primary,
                fact_variable=primary_variable,
                fact_value=cont_final,
            )
        )
    else:
        events.append(
            Event(
                step=step,
                kind="message",
                actor=a2,
                room=r_sec_a,
                target_object=aux_object,
                to_location=shared_val,
                recipient=primary_absent,
                utterance=f"Update for {primary_absent}: the {aux_object} is stored at the {shared_val}.",
                fact_id=fid_shared,
                fact_variable=shared_var,
                fact_value=shared_val,
            )
        )
    step += 1

    if len(absent_parties) > 1:
        secondary_absent = absent_parties[1]
        events.append(
            Event(
                step=step,
                kind="message",
                actor=a2,
                room=r_sec_a,
                target_object=aux_object,
                to_location=shared_val,
                recipient=secondary_absent,
                utterance=f"Update for {secondary_absent}: the {aux_object} is stored at the {shared_val}.",
                fact_id=fid_shared,
                fact_variable=shared_var,
                fact_value=shared_val,
            )
        )
        step += 1

    # Side events for `U` (unknowable) and `S` (settled twin) projected from `latent_state`:
    note_surface = f"countertop in the {r_k_dest}"
    if config.u_is_who:
        # `latent_state["note_item_placer"]` placed `note_item` on the countertop in `r_k_dest` prior to step 1 (unlogged),
        # so only `a0` seeing it already on the countertop in `r_k_dest` is projected to the observable event log.
        events.append(
            Event(
                step=step,
                kind="see",
                actor=a0,
                room=r_k_dest,
                target_object=domain.note_item,
                from_location=note_surface,
                to_location=note_surface,
                occluded=False,
            )
        )
        step += 1
        settled_why = latent_state["side_mechanism_reason"]
        events.append(
            Event(
                step=step,
                kind="move",
                actor=a1,
                room=a1_current_room,
                target_object=domain.side_mechanism,
                from_location="latch position A",
                to_location="latch position B",
                occluded=False,
                stated_reason=settled_why,
            )
        )
        step += 1
        events.append(
            Event(
                step=step,
                kind="say",
                actor=a1,
                room=a1_current_room,
                target_object=domain.side_mechanism,
                utterance=f"I adjusted the {domain.side_mechanism} {settled_why}.",
                stated_reason=settled_why,
                noise_level=20,
            )
        )
        step += 1
    else:
        settled_who = latent_state["note_item_placer"]
        if current_agent_rooms[settled_who] != r_k_dest:
            step = _append_doorway_walk(
                events, step, settled_who, current_agent_rooms[settled_who], r_k_dest, doorways
            )
            current_agent_rooms[settled_who] = r_k_dest
        events.append(
            Event(
                step=step,
                kind="move",
                actor=settled_who,
                room=r_k_dest,
                target_object=domain.note_item,
                from_location="dispatch folder",
                to_location=note_surface,
                occluded=False,
            )
        )
        step += 1
        # Equalize `a0`'s step count across `u_is_who` so `len(agent_sent_indices)` does not leak into P/K,
        # and confirm `a0` sees `note_item` on the countertop in the same room `r_k_dest`.
        events.append(
            Event(
                step=step,
                kind="see",
                actor=a0,
                room=r_k_dest,
                target_object=domain.note_item,
                from_location=note_surface,
                to_location=note_surface,
                occluded=False,
            )
        )
        step += 1
        # `settled_who` states the placement in `r_hub` at >= `noise_threshold` so both focal agents (`a0` in `r_k_dest`
        # and `a1` in `r_work`, adjacent to `r_hub`) hear who placed `note_item` from the witness who saw/performed it.
        if current_agent_rooms[settled_who] != r_hub:
            step = _append_doorway_walk(
                events, step, settled_who, current_agent_rooms[settled_who], r_hub, doorways
            )
            current_agent_rooms[settled_who] = r_hub
        events.append(
            Event(
                step=step,
                kind="say",
                actor=settled_who,
                room=r_hub,
                target_object=domain.note_item,
                to_location=note_surface,
                utterance=f"I placed the {domain.note_item} on the {note_surface}.",
                occluded=False,
                noise_level=noise_threshold + 10,
            )
        )
        step += 1
        # `a1` adjusts `side_mechanism` for `latent_state["side_mechanism_reason"]`, which is an unstated
        # internal motive (`stated_reason=None` in the observable projection).
        events.append(
            Event(
                step=step,
                kind="move",
                actor=a1,
                room=a1_current_room,
                target_object=domain.side_mechanism,
                from_location="latch position A",
                to_location="latch position B",
                occluded=False,
                stated_reason=None,
            )
        )
        step += 1

    # Post-message container mention equalizer (keeping total mentions of `c0` and `c1` equal at 5 == 5
    # and ensuring `c1` is always the last-mentioned container in `r_work`):
    if config.x_notified:
        if cont_init == c1:
            events.append(
                Event(
                    step=step,
                    kind="see",
                    actor=a1,
                    room=r_work,
                    target_object=cont_init,
                    from_location=FAR_SIDE_ALCOVE_STATION,
                    to_location=FAR_SIDE_ALCOVE_STATION,
                    occluded=True,
                    occluder=occ_work,
                )
            )
            step += 1
    else:
        for c_eq_msg in (c0, c1):
            events.append(
                Event(
                    step=step,
                    kind="see",
                    actor=a1,
                    room=r_work,
                    target_object=c_eq_msg,
                    from_location=FAR_SIDE_ALCOVE_STATION,
                    to_location=FAR_SIDE_ALCOVE_STATION,
                    occluded=True,
                    occluder=occ_work,
                )
            )
            step += 1

    # In-scenario bluff (`is_bluff=True`):
    bluffer = extra_agents[0] if extra_agents else a1
    bluff_room = current_agent_rooms.get(bluffer, a1_current_room) if extra_agents else a1_current_room
    bluff_claimed_location = cont_hub_final
    bluff_step = step
    events.append(
        Event(
            step=step,
            kind="say",
            actor=bluffer,
            room=bluff_room,
            target_object=primary_object,
            to_location=bluff_claimed_location,
            utterance=f"I am certain the {primary_object} was moved into the {bluff_claimed_location}.",
            occluded=False,
            noise_level=20,
            fact_id=None,
            fact_variable=primary_variable,
            fact_value=bluff_claimed_location,
            is_bluff=True,
        )
    )

    raw_facts: dict[str, FactSpec] = {
        fid_primary: FactSpec(
            fact_id=fid_primary,
            variable=primary_variable,
            value=cont_final,
            description=f"The {primary_object} is located in the {cont_final}.",
        ),
        fid_shared: FactSpec(
            fact_id=fid_shared,
            variable=shared_var,
            value=shared_val,
            description=f"The {aux_object} baseline reading is at the {shared_val}.",
        ),
        fid_secondary: FactSpec(
            fact_id=fid_secondary,
            variable=secondary_variable,
            value=cont_sec_final,
            description=f"The {secondary_object} is stored in the {cont_sec_final}.",
        ),
        fid_aux: FactSpec(
            fact_id=fid_aux,
            variable=aux_variable,
            value=cont_sec_init,
            description=f"The {aux_object} reserve unit is verified in the {cont_sec_init}.",
        ),
    }
    facts = {fid: raw_facts[fid] for fid in sorted(raw_facts.keys())}

    return (
        tuple(events),
        initial_rooms,
        facts,
        bluff_step,
        bluffer,
        a0,
        bluff_claimed_location,
        reason_a,
        reason_b,
        latent_state,
        a1_hears_secondary,
    )


def _parse_scenario_index(scenario_id: str, seed: int) -> int:
    if scenario_id.startswith("scen_") and scenario_id[5:].isdigit():
        return int(scenario_id[5:])
    return int(hashlib.sha256(f"{scenario_id}:{seed}".encode("utf-8")).hexdigest()[:8], 16)


def generate_world(config: ScenarioConfig) -> World:
    """Generate a complete, deterministic `World` from `ScenarioConfig`."""
    rng = random.Random(config.seed)
    scen_idx = _parse_scenario_index(config.scenario_id, config.seed)
    domain = DOMAINS[(scen_idx // 2) % len(DOMAINS)]

    name_pool = list(AGENT_NAME_POOLS[scen_idx % len(AGENT_NAME_POOLS)])
    rng.shuffle(name_pool)

    agents = tuple(name_pool[: config.num_agents])
    absent_parties = tuple(name_pool[config.num_agents : config.num_agents + config.num_absent])

    rooms = domain.rooms
    doorways = _build_doorways(rooms, config.topology_kind)

    containers_by_room = {rooms[i]: domain.containers[i] for i in range(len(rooms))}
    occluders_by_room = {rooms[i]: domain.occluders[i] for i in range(len(rooms))}
    fixtures_by_room = {rooms[i]: domain.fixtures[i] for i in range(len(rooms))}

    primary_object, secondary_object, aux_object = domain.objects
    primary_variable = f"location:{primary_object}"
    secondary_variable = f"location:{secondary_object}"
    aux_variable = f"reserve:{aux_object}"
    noise_threshold = 50

    # Permute which fact IDs (`F1`..`F4`) correspond to `(primary, shared, secondary, aux)`.
    fact_id_map = FACT_ID_PERMUTATIONS[(scen_idx * 7 + (config.seed >> 4)) % len(FACT_ID_PERMUTATIONS)]
    fid_primary, fid_shared, fid_secondary, fid_aux = fact_id_map

    (
        trace_a,
        initial_rooms,
        facts,
        bluff_step,
        bluffer,
        bluff_listener,
        bluff_claimed_location,
        reason_a,
        reason_b,
        latent_state_a,
        a1_hears_secondary,
    ) = _build_event_trace(
        config=config,
        domain=domain,
        rooms=rooms,
        doorways=doorways,
        containers_by_room=containers_by_room,
        occluders_by_room=occluders_by_room,
        fixtures_by_room=fixtures_by_room,
        agents=agents,
        absent_parties=absent_parties,
        primary_object=primary_object,
        primary_variable=primary_variable,
        secondary_object=secondary_object,
        secondary_variable=secondary_variable,
        aux_object=aux_object,
        aux_variable=aux_variable,
        noise_threshold=noise_threshold,
        fact_id_map=fact_id_map,
        u_variant="A",
    )

    trace_b, *_, latent_state_b, _ = _build_event_trace(
        config=config,
        domain=domain,
        rooms=rooms,
        doorways=doorways,
        containers_by_room=containers_by_room,
        occluders_by_room=occluders_by_room,
        fixtures_by_room=fixtures_by_room,
        agents=agents,
        absent_parties=absent_parties,
        primary_object=primary_object,
        primary_variable=primary_variable,
        secondary_object=secondary_object,
        secondary_variable=secondary_variable,
        aux_object=aux_object,
        aux_variable=aux_variable,
        noise_threshold=noise_threshold,
        fact_id_map=fact_id_map,
        u_variant="B",
    )

    a0, a1, a2 = agents[0], agents[1], agents[2]
    primary_absent = absent_parties[0]

    stake_list: list[StakeEdge] = [
        StakeEdge(
            agent=primary_absent,
            variable=primary_variable,
            description=(
                f"{primary_absent} is off-site and depends on the location of the {primary_object} "
                f"for tomorrow's courier handoff."
            ),
        )
    ]
    if len(absent_parties) > 1:
        sec_absent = absent_parties[1]
        stake_list.append(
            StakeEdge(
                agent=sec_absent,
                variable=f"baseline:{aux_object}",
                description=(
                    f"{sec_absent} is off-site and depends on the baseline reading of the {aux_object}."
                ),
            )
        )

    # Randomize `causal_preconditions(G_target)` across scenarios so `D` varies across all 14 non-empty
    # subsets of `{F1, F2, F3, F4}` of size 1, 2, and 3 while always including `fid_shared` (already known to `a1`).
    unknown_to_a1: list[str] = [fid_aux]
    if not config.b1_true_belief:
        unknown_to_a1.append(fid_primary)
    if not a1_hears_secondary:
        unknown_to_a1.append(fid_secondary)
    unknown_to_a1.sort()

    diff_rng = random.Random(config.seed ^ 0xA5A5F0F0 ^ (scen_idx * 131))
    diff_candidates = list(unknown_to_a1)
    diff_rng.shuffle(diff_candidates)
    target_diff_size = 1 + ((scen_idx + (config.seed >> 5)) % len(diff_candidates))
    chosen_diff = diff_candidates[:target_diff_size]
    a1_preconditions = tuple(sorted({fid_shared, *chosen_diff}))

    goals: dict[str, AgentGoal] = {
        a0: AgentGoal(
            agent=a0,
            goal_text=f"{a0} plans to {domain.goal_templates[0]}",
            causal_preconditions=tuple(sorted({fid_shared, fid_secondary})),
        ),
        a1: AgentGoal(
            agent=a1,
            goal_text=f"{a1} plans to {domain.goal_templates[1]}",
            causal_preconditions=a1_preconditions,
        ),
        a2: AgentGoal(
            agent=a2,
            goal_text=f"{a2} plans to {domain.goal_templates[2]}",
            causal_preconditions=tuple(sorted({fid_primary, fid_secondary, fid_aux})),
        ),
    }
    for idx, ea in enumerate(agents[3:]):
        goals[ea] = AgentGoal(
            agent=ea,
            goal_text=f"{ea} plans to assist with {domain.goal_templates[idx % len(domain.goal_templates)]}",
            causal_preconditions=(fid_shared,),
        )

    who_q_text = f"Who placed the {domain.note_item} on the countertop?"
    why_q_text = f"Why did {a1} adjust the {domain.side_mechanism}?"

    if config.u_is_who:
        unknowable_spec = UnknowableSpec(
            subtype="who",
            question_text=who_q_text,
            alternative_a=a1,
            alternative_b=a2,
            target_agent=a0,
            side_action_desc=f"placing the {domain.note_item} on the countertop",
            trace_a=trace_a,
            trace_b=trace_b,
            latent_state_a=latent_state_a,
            latent_state_b=latent_state_b,
        )
        settled_twin_spec = SettledTwinSpec(
            subtype="why",
            question_text=why_q_text,
            settled_answer=reason_a,
            target_agent=a1,
            side_action_desc=f"adjusting the {domain.side_mechanism}",
        )
    else:
        unknowable_spec = UnknowableSpec(
            subtype="why",
            question_text=why_q_text,
            alternative_a=reason_a,
            alternative_b=reason_b,
            target_agent=a1,
            side_action_desc=f"adjusting the {domain.side_mechanism}",
            trace_a=trace_a,
            trace_b=trace_b,
            latent_state_a=latent_state_a,
            latent_state_b=latent_state_b,
        )
        settled_twin_spec = SettledTwinSpec(
            subtype="who",
            question_text=who_q_text,
            settled_answer=a2,
            target_agent=a2,
            side_action_desc=f"placing the {domain.note_item} on the countertop",
        )

    world = World(
        scenario_id=config.scenario_id,
        seed=config.seed,
        domain_name=domain.name,
        rooms=rooms,
        doorways=doorways,
        objects=domain.objects,
        containers_by_room=containers_by_room,
        occluders_by_room=occluders_by_room,
        fixtures_by_room=fixtures_by_room,
        noise_threshold=noise_threshold,
        agents=agents,
        absent_parties=absent_parties,
        initial_agent_rooms=initial_rooms,
        stake_edges=tuple(stake_list),
        goals=goals,
        facts=facts,
        events=trace_a,
        unknowable_spec=unknowable_spec,
        settled_twin_spec=settled_twin_spec,
        primary_object=primary_object,
        primary_variable=primary_variable,
        secondary_object=secondary_object,
        secondary_variable=secondary_variable,
        aux_object=aux_object,
        aux_variable=aux_variable,
        bluff_event_step=bluff_step,
        bluffer_agent=bluffer,
        bluff_listener=bluff_listener,
        bluff_claimed_location=bluff_claimed_location,
        p_agent=a0,
        k_agent=a0,
        b1_agent=a1,
        b2_observer=a2,
        b2_actor=a1,
        d_helper=a2,
        d_target=a1,
        primary_absent=primary_absent,
        config=config,
    )
    from udlab_generator.keys import (
        validate_hearing,
        validate_object_locations,
        validate_occlusion,
        validate_placing_fairness,
    )

    validate_topology(world)
    validate_occlusion(world)
    validate_hearing(world)
    validate_object_locations(world)
    validate_placing_fairness(world)
    return world
