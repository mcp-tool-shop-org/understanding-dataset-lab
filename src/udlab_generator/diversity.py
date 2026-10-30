"""Template-diversity counter collapsing renamed-agent and renamed-entity near-duplicates."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json

from udlab_generator.world import World


@dataclass(frozen=True)
class DiversityReport:
    """Summary of distinct canonical event skeletons across generated scenarios."""

    total_scenarios: int
    distinct_event_skeletons: int
    distinct_domain_skeletons: int
    max_cluster_size: int
    mean_cluster_size: float
    skeleton_counts: dict[str, int]


def canonical_event_skeleton(world: World) -> tuple[object, ...]:
    """Compute a name-invariant canonical event skeleton for `world`.

    All active agents, absent parties, rooms, and objects/targets are alpha-renamed
    by order of first appearance in the event trace (`A0, A1, ...`, `X0, X1, ...`,
    `R0, R1, ...`, `O0, O1, ...`). Thus two scenarios with identical event structure
    and renamed agents, rooms, or objects collapse to the exact same skeleton tuple.
    """
    agent_map: dict[str, str] = {}
    absent_set = set(world.absent_parties)
    room_map: dict[str, str] = {}
    obj_map: dict[str, str] = {}

    def _canon_party(name: str | None) -> str | None:
        if name is None:
            return None
        if name not in agent_map:
            prefix = "X" if name in absent_set else "A"
            idx = sum(1 for k in agent_map.values() if k.startswith(prefix))
            agent_map[name] = f"{prefix}{idx}"
        return agent_map[name]

    def _canon_room(room: str | None) -> str | None:
        if room is None:
            return None
        if room not in room_map:
            room_map[room] = f"R{len(room_map)}"
        return room_map[room]

    def _canon_obj(obj: str | None) -> str | None:
        if obj is None:
            return None
        if obj not in obj_map:
            obj_map[obj] = f"O{len(obj_map)}"
        return obj_map[obj]

    canon_events: list[tuple[object, ...]] = []
    for ev in world.events:
        c_actor = _canon_party(ev.actor)
        c_room = _canon_room(ev.room)
        c_obj = _canon_obj(ev.target_object)
        c_recip = _canon_party(ev.recipient)
        c_from = _canon_room(ev.from_location) if ev.kind in ("enter", "leave") else None
        c_to = _canon_room(ev.to_location) if ev.kind in ("enter", "leave") else None
        canon_events.append(
            (
                ev.kind,
                c_actor,
                c_room,
                c_from,
                c_to,
                c_obj,
                bool(ev.occluded),
                bool(ev.noise_level >= world.noise_threshold),
                bool(ev.is_bluff),
                bool(ev.stated_reason is not None),
                c_recip,
            )
        )

    # Register any remaining rooms/agents so doorway graph can be canonicalized.
    for r in world.rooms:
        _canon_room(r)
    for a in world.agents:
        _canon_party(a)
    for x in world.absent_parties:
        _canon_party(x)

    canon_edges = tuple(
        sorted(
            tuple(sorted((_canon_room(a) or "", _canon_room(b) or "")))
            for a, b in world.doorways
        )
    )

    return (
        len(world.agents),
        len(world.absent_parties),
        canon_edges,
        tuple(canon_events),
    )


def skeleton_digest(world: World) -> str:
    """Return a short deterministic SHA-256 hex digest of `canonical_event_skeleton(world)`."""
    skel = canonical_event_skeleton(world)
    raw = json.dumps(skel, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def count_template_diversity(worlds: Sequence[World]) -> DiversityReport:
    """Count distinct canonical event skeletons across `worlds`, collapsing renamed near-duplicates."""
    if not worlds:
        return DiversityReport(
            total_scenarios=0,
            distinct_event_skeletons=0,
            distinct_domain_skeletons=0,
            max_cluster_size=0,
            mean_cluster_size=0.0,
            skeleton_counts={},
        )

    skel_counter: Counter[str] = Counter()
    domain_skel_set: set[tuple[str, str]] = set()
    for w in worlds:
        dig = skeleton_digest(w)
        skel_counter[dig] += 1
        domain_skel_set.add((w.domain_name, dig))

    n_total = len(worlds)
    n_distinct = len(skel_counter)
    max_cluster = max(skel_counter.values())
    mean_cluster = n_total / n_distinct

    return DiversityReport(
        total_scenarios=n_total,
        distinct_event_skeletons=n_distinct,
        distinct_domain_skeletons=len(domain_skel_set),
        max_cluster_size=max_cluster,
        mean_cluster_size=round(mean_cluster, 2),
        skeleton_counts=dict(sorted(skel_counter.items())),
    )
