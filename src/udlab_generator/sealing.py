"""Sealing, private-seed shuffling, key-order leak check, and canonical re-render/diff verify."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
from pathlib import Path
import random
import secrets
from typing import Any

from udlab_generator.keys import (
    UNKNOWABLE_KEY,
    build_pressure_probe_turn2,
    build_public_pressure_probe_map,
    build_public_pressure_probe_template,
    compute_scenario_keys,
    validate_belief_forcing,
    validate_event_locations,
    validate_hearing,
    validate_object_locations,
    validate_occlusion,
    validate_placing_fairness,
    validate_topology,
    verify_unknowable_u,
)
from udlab_generator.probe import _oof_multinomial_nb
from udlab_generator.render import (
    RenderCheckError,
    RenderedScenario,
    check_rendered_prose,
    count_duplicate_stake_statements,
    format_non_sealed_refusal_reason,
    sanitize_render_check_error_for_stderr,
)
from udlab_generator.world import (
    AgentGoal,
    Event,
    FactSpec,
    ScenarioConfig,
    SettledTwinSpec,
    StakeEdge,
    UnknowableSpec,
    World,
)

FORBIDDEN_PUBLIC_FIELDS: frozenset[str] = frozenset(
    {
        "key",
        "keys",
        "ego_lure_key",
        "asymmetric",
        "in_primary_bundle",
        "question_type",
        "slot_id",
        "curriculum_stage",
        "pressure_lure_answer",
        "perceptual_records",
        "known_facts_by_agent",
        "shuffle_seed",
        "private_seed",
        "private_seed_hex",
        "generation_seed",
        "scenario_id",
        "response_text",
        "thinking_text",
        "attempt_0_content",
        "attempt_messages",
        "failing_sentence",
    }
)


class SealingError(ValueError):
    """Raised when sealing or verification fails due to leakage, key order, or byte mismatch."""


@dataclass(frozen=True)
class SealedDatasetReceipt:
    """Receipt returned after writing and verifying a sealed dataset part."""

    public_dir: Path
    sealed_dir: Path
    items_path: Path
    part_json_path: Path
    keys_sealed_path: Path
    sealing_meta_path: Path
    items_sha256: str
    part_json_sha256: str
    keys_sealed_sha256: str
    num_scenarios: int
    num_items: int


def _canonical_json_line(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _canonical_json_doc(obj: dict[str, Any]) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _value_sha256(val: Any) -> str:
    """Compute SHA-256 hex digest of a canonicalized value without exposing the value itself."""
    if isinstance(val, tuple):
        val = list(val)
    if isinstance(val, str):
        raw = val.encode("utf-8")
    else:
        raw = json.dumps(val, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _hmac_hex(private_seed_hex: str, message: str, length: int = 24) -> str:
    digest = hmac.new(
        bytes.fromhex(private_seed_hex),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return digest[:length]


def _validate_private_seed(private_seed_hex: str) -> str:
    cleaned = private_seed_hex.strip().lower()
    if len(cleaned) < 32:
        raise SealingError(
            f"Private seed must be at least 128 bits (32 hex chars); got {len(cleaned)} chars."
        )
    try:
        bytes.fromhex(cleaned)
    except ValueError as exc:
        raise SealingError("Private seed must be valid hexadecimal.") from exc
    return cleaned


def _validate_disjoint_dirs(public_dir: Path, sealed_dir: Path) -> None:
    pub_res = public_dir.resolve()
    seal_res = sealed_dir.resolve()
    if pub_res == seal_res or pub_res in seal_res.parents or seal_res in pub_res.parents:
        raise SealingError(
            f"public_dir ({pub_res}) and sealed_dir ({seal_res}) must be completely separate directories."
        )


def _scan_forbidden_keys(obj: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            cur = f"{path}.{k}" if path else str(k)
            if str(k) in FORBIDDEN_PUBLIC_FIELDS:
                hits.append(cur)
            hits.extend(_scan_forbidden_keys(v, cur))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            hits.extend(_scan_forbidden_keys(item, f"{path}[{idx}]"))
    return hits


def check_not_in_key_order(
    public_items: Sequence[dict[str, Any]],
    sealed_rows: Sequence[dict[str, Any]],
    part_doc: dict[str, Any],
    private_seed_hex: str | None = None,
) -> None:
    """Fail if public items leak keys/seeds or are ordered in deterministic key/type order (10-25 rdlab leak)."""
    # 1. Scan public structures for any forbidden key-bearing or seed-bearing field names.
    part_hits = _scan_forbidden_keys(part_doc, "part.json")
    if part_hits:
        raise SealingError(f"Forbidden key/seed fields found in public part.json: {part_hits}")

    for idx, pub_row in enumerate(public_items):
        row_hits = _scan_forbidden_keys(pub_row, f"items[{idx}]")
        if row_hits:
            raise SealingError(f"Forbidden key/seed fields found in public items.jsonl: {row_hits}")

    if private_seed_hex:
        raw_public_str = json.dumps(part_doc) + "\n" + "\n".join(json.dumps(r) for r in public_items)
        if private_seed_hex.lower() in raw_public_str.lower():
            raise SealingError("Private shuffle seed value leaked into public part files.")

    if len(public_items) != len(sealed_rows):
        raise SealingError(
            f"Count mismatch between public items ({len(public_items)}) and sealed rows ({len(sealed_rows)})."
        )

    # 2. Check Pass-1 before Pass-2 ordering (every scenario's first rendering before any second rendering).
    seen_pass_2 = False
    pass1_scenarios: set[str] = set()
    for row in sealed_rows:
        p_idx = int(row.get("pass_index", 1))
        scen_id = str(row.get("scenario_id", ""))
        if p_idx == 1:
            if seen_pass_2:
                raise SealingError("Warm-cache violation: Pass 1 rendering found after Pass 2 started.")
            pass1_scenarios.add(scen_id)
        elif p_idx == 2:
            seen_pass_2 = True
            if scen_id and scen_id not in pass1_scenarios:
                raise SealingError(
                    f"Warm-cache violation: second rendering of {scen_id} scheduled before its first rendering."
                )

    # 3. Check intra-bundle question order: fail if bundles repeat the same fixed question_type permutation.
    type_orders: list[tuple[str, ...]] = [
        tuple(str(q["question_type"]) for q in row.get("questions", []))
        for row in sealed_rows
    ]
    if len(type_orders) >= 2 and len(set(type_orders)) == 1:
        canonical_slot_order = ("P", "K", "B1", "B2", "X", "U", "S", "D")
        if len(type_orders) >= 4 or type_orders[0] == canonical_slot_order:
            raise SealingError(
                "Key-order leak detected: questions inside bundles are in a fixed, unshuffled question_type order."
            )

    # 4. Check inter-item scenario order: fail if items are in unshuffled sequential scenario_id order.
    pass1_rows = [r for r in sealed_rows if int(r.get("pass_index", 1)) == 1]
    if len(pass1_rows) >= 6:
        scen_ids = [str(r.get("scenario_id", "")) for r in pass1_rows]
        if scen_ids == sorted(scen_ids) or scen_ids == sorted(scen_ids, reverse=True):
            raise SealingError(
                "Key-order leak detected: items are in unshuffled sequential scenario_id order."
            )

        # 5. Check key sequences across Pass-1 bundles for ALL question types (`P, K, B1, B2, X, U, S, D`)
        # to catch contiguous grouping, sorted key ordering, or periodic alternation.
        for target_qtype in ("P", "K", "B1", "B2", "X", "U", "S", "D"):
            seq: list[str] = []
            asym_seq: list[str] = []
            for r in pass1_rows:
                for q in r.get("questions", []):
                    if q.get("question_type") == target_qtype:
                        raw_k = q.get("key", "")
                        seq.append(
                            json.dumps(raw_k, sort_keys=True)
                            if isinstance(raw_k, list)
                            else str(raw_k)
                        )
                        if "is_ego_asymmetric" in q:
                            asym_seq.append("1" if q.get("is_ego_asymmetric") else "0")
                        break
            for seq_label, candidate_seq in (("raw", seq), ("asymmetry", asym_seq)):
                distinct_vals = set(candidate_seq)
                if len(candidate_seq) >= 8 and len(distinct_vals) >= 2:
                    transitions = sum(
                        1 for i in range(1, len(candidate_seq)) if candidate_seq[i] != candidate_seq[i - 1]
                    )
                    if len(distinct_vals) <= len(candidate_seq) // 2 and transitions <= max(2, len(distinct_vals)):
                        raise SealingError(
                            f"Key-order leak detected: '{target_qtype}' {seq_label} keys are grouped contiguously ({transitions} transitions)."
                        )
                    if candidate_seq == sorted(candidate_seq) or candidate_seq == sorted(candidate_seq, reverse=True):
                        raise SealingError(
                            f"Key-order leak detected: '{target_qtype}' {seq_label} keys are sorted in monotonic key order."
                        )
                    for period in (2, 4):
                        if len(candidate_seq) >= 2 * period and len(set(candidate_seq[:period])) >= 2:
                            tiled = (candidate_seq[:period] * ((len(candidate_seq) // period) + 1))[: len(candidate_seq)]
                            if candidate_seq == tiled:
                                raise SealingError(
                                    f"Key-order leak detected: '{target_qtype}' {seq_label} keys follow a strict period-{period} alternating sequence."
                                )


def _serialize_world(world: World) -> dict[str, Any]:
    """Serialize a `World` dataclass to a JSON-compatible dictionary for sealed verification."""
    return asdict(world)


def _deserialize_world(data: dict[str, Any]) -> World:
    """Reconstruct a `World` dataclass from a dictionary stored in `sealing.sealed.json`."""
    events = tuple(Event(**ev) for ev in data["events"])
    u_raw = data["unknowable_spec"]
    unknowable_spec = UnknowableSpec(
        subtype=u_raw["subtype"],
        question_text=u_raw["question_text"],
        alternative_a=u_raw["alternative_a"],
        alternative_b=u_raw["alternative_b"],
        target_agent=u_raw["target_agent"],
        side_action_desc=u_raw["side_action_desc"],
        trace_a=tuple(Event(**ev) for ev in u_raw["trace_a"]),
        trace_b=tuple(Event(**ev) for ev in u_raw["trace_b"]),
        latent_state_a=dict(u_raw["latent_state_a"]) if u_raw.get("latent_state_a") is not None else None,
        latent_state_b=dict(u_raw["latent_state_b"]) if u_raw.get("latent_state_b") is not None else None,
    )
    s_raw = data["settled_twin_spec"]
    settled_twin_spec = SettledTwinSpec(**s_raw)
    cfg_raw = data.get("config")
    config = ScenarioConfig(**cfg_raw) if cfg_raw is not None else None

    return World(
        scenario_id=str(data["scenario_id"]),
        seed=int(data["seed"]),
        domain_name=str(data["domain_name"]),
        rooms=tuple(data["rooms"]),
        doorways=tuple(( str(edge[0]), str(edge[1]) ) for edge in data["doorways"]),
        objects=(
            str(data["objects"][0]),
            str(data["objects"][1]),
            str(data["objects"][2]),
        ),
        containers_by_room={
            str(r): tuple(str(c) for c in pair)
            for r, pair in data["containers_by_room"].items()
        },
        occluders_by_room={str(r): str(occ) for r, occ in data["occluders_by_room"].items()},
        fixtures_by_room={str(r): str(fix) for r, fix in data["fixtures_by_room"].items()},
        noise_threshold=int(data["noise_threshold"]),
        agents=tuple(str(a) for a in data["agents"]),
        absent_parties=tuple(str(a) for a in data["absent_parties"]),
        initial_agent_rooms={str(a): str(r) for a, r in data["initial_agent_rooms"].items()},
        stake_edges=tuple(StakeEdge(**se) for se in data["stake_edges"]),
        goals={
            str(ag): AgentGoal(
                agent=str(g["agent"]),
                goal_text=str(g["goal_text"]),
                causal_preconditions=tuple(str(fid) for fid in g["causal_preconditions"]),
            )
            for ag, g in data["goals"].items()
        },
        facts={str(fid): FactSpec(**f) for fid, f in data["facts"].items()},
        events=events,
        unknowable_spec=unknowable_spec,
        settled_twin_spec=settled_twin_spec,
        primary_object=str(data["primary_object"]),
        primary_variable=str(data["primary_variable"]),
        secondary_object=str(data["secondary_object"]),
        secondary_variable=str(data["secondary_variable"]),
        aux_object=str(data["aux_object"]),
        aux_variable=str(data["aux_variable"]),
        bluff_event_step=int(data["bluff_event_step"]),
        bluffer_agent=str(data["bluffer_agent"]),
        bluff_listener=str(data["bluff_listener"]),
        bluff_claimed_location=str(data["bluff_claimed_location"]),
        p_agent=str(data["p_agent"]),
        k_agent=str(data["k_agent"]),
        b1_agent=str(data["b1_agent"]),
        b2_observer=str(data["b2_observer"]),
        b2_actor=str(data["b2_actor"]),
        d_helper=str(data["d_helper"]),
        d_target=str(data["d_target"]),
        primary_absent=str(data["primary_absent"]),
        config=config,
    )


def _render_public_from_sealed(
    sealed_rows: Sequence[dict[str, Any]],
    sealing_meta: dict[str, Any],
) -> tuple[bytes, bytes, list[dict[str, Any]], dict[str, Any]]:
    """Canonically render `items.jsonl` and `part.json` bytes from the sealed store."""
    public_items: list[dict[str, Any]] = []
    item_lines: list[str] = []

    for s_row in sealed_rows:
        pub_questions: list[dict[str, Any]] = []
        for q in s_row["questions"]:
            qid = str(q["question_id"])
            qtext = str(q["question_text"])
            cand_turn2_map = dict(q.get("pressure_probe_turn2_by_candidate", {}))
            pub_questions.append(
                {
                    "question_id": qid,
                    "question_text": qtext,
                    "pressure_probe_turn2_by_candidate": cand_turn2_map,
                    "pressure_probe_turn2_template": build_public_pressure_probe_template(qid, qtext),
                }
            )
        item_attempt = int(s_row.get("attempt", 0))
        item_temp = float(s_row.get("temperature", 0.0))
        item_min_p = (
            float(s_row["min_p"])
            if s_row.get("min_p") is not None
            else None
        )
        item_presence_penalty = (
            float(s_row["presence_penalty"])
            if s_row.get("presence_penalty") is not None
            else None
        )
        item_seed = int(s_row.get("seed", 0))
        item_goal_mode = str(s_row.get("goal_mode", "copy"))
        item_goal_source = dict(s_row.get("goal_source") or {})
        pub_row: dict[str, Any] = {
            "item_id": str(s_row["item_id"]),
            "pass_index": int(s_row["pass_index"]),
            "scenario_handle": str(s_row["scenario_handle"]),
            "rendering_handle": str(s_row["rendering_handle"]),
            "focal_agent": str(s_row["focal_agent"]),
            "prose": str(s_row["prose"]),
            "goal_mode": item_goal_mode,
            "goal_source": item_goal_source,
            "attempt": item_attempt,
            "temperature": item_temp,
            "min_p": item_min_p,
            "presence_penalty": item_presence_penalty,
            "seed": item_seed,
            "sampling": {
                "attempt": item_attempt,
                "temperature": item_temp,
                "min_p": item_min_p,
                "seed": item_seed,
            },
            "questions": pub_questions,
        }
        public_items.append(pub_row)
        item_lines.append(_canonical_json_line(pub_row))

    items_bytes = (("\n".join(item_lines) + "\n") if item_lines else "").encode("utf-8")
    items_sha = hashlib.sha256(items_bytes).hexdigest()

    part_doc: dict[str, Any] = {
        "schema_version": "udlab.part.v1",
        "num_scenarios": int(sealing_meta["num_scenarios"]),
        "num_items": len(public_items),
        "passes": 2,
        "items_file": "items.jsonl",
        "items_sha256": items_sha,
        "commitment_hmac_sha256": str(sealing_meta["commitment_hmac_sha256"]),
    }
    part_bytes = _canonical_json_doc(part_doc).encode("utf-8")
    return items_bytes, part_bytes, public_items, part_doc


def write_sealed_dataset(
    rendered_pairs: Sequence[tuple[RenderedScenario, RenderedScenario]],
    public_dir: Path,
    sealed_dir: Path,
    private_seed_hex: str | None = None,
    writer_provenance: dict[str, Any] | None = None,
    *,
    all_worlds: Sequence[World] | None = None,
    allow_empty: bool = False,
) -> SealedDatasetReceipt:
    """Shuffle items and questions with a private seed, write public and sealed stores, and run canonical verify."""
    if not rendered_pairs and not allow_empty:
        raise ValueError("rendered_pairs must not be empty")

    _validate_disjoint_dirs(public_dir, sealed_dir)
    seed_hex = _validate_private_seed(private_seed_hex or secrets.token_hex(32))

    public_dir.mkdir(parents=True, exist_ok=True)
    sealed_dir.mkdir(parents=True, exist_ok=True)

    # Separate first renderings (Pass 1) and second renderings (Pass 2), and shuffle each pass
    # independently using deterministic sub-seeds derived from the private seed, resampling
    # the salt index until Pass 1 satisfies the inter-item key-order checks (`check_not_in_key_order`).
    base_pass1 = [pair[0] for pair in rendered_pairs]
    base_pass2 = [pair[1] for pair in rendered_pairs]

    shuffle_salt_index = 0
    pass1_list = list(base_pass1)
    pass2_list = list(base_pass2)
    if rendered_pairs:
        for salt_idx in range(256):
            cand_pass1 = list(base_pass1)
            cand_pass2 = list(base_pass2)
            p1_msg = "shuffle:pass1" if salt_idx == 0 else f"shuffle:pass1:r{salt_idx}"
            p2_msg = "shuffle:pass2" if salt_idx == 0 else f"shuffle:pass2:r{salt_idx}"
            random.Random(_hmac_hex(seed_hex, p1_msg, 32)).shuffle(cand_pass1)
            random.Random(_hmac_hex(seed_hex, p2_msg, 32)).shuffle(cand_pass2)

            # Ensure Pass 1 is never accidentally sorted even for small N.
            if len(cand_pass1) >= 2 and (
                [r.scenario_id for r in cand_pass1] == sorted(r.scenario_id for r in cand_pass1)
                or [r.scenario_id for r in cand_pass1] == sorted((r.scenario_id for r in cand_pass1), reverse=True)
            ):
                cand_pass1 = cand_pass1[1:] + cand_pass1[:1]

            cand_sealed_check = [
                {
                    "pass_index": 1,
                    "scenario_id": r.scenario_id,
                    "questions": [
                        {
                            "question_type": qk.question_type,
                            "key": list(qk.key) if isinstance(qk.key, tuple) else qk.key,
                            "asymmetric": qk.asymmetric,
                            "is_ego_asymmetric": qk.asymmetric,
                        }
                        for idx_q, qk in enumerate(
                            r.keys.questions[idx_r % len(r.keys.questions) :]
                            + r.keys.questions[: idx_r % len(r.keys.questions)]
                        )
                    ],
                }
                for idx_r, r in enumerate(cand_pass1)
            ] + [
                {
                    "pass_index": 2,
                    "scenario_id": r.scenario_id,
                    "questions": [
                        {
                            "question_type": qk.question_type,
                            "key": list(qk.key) if isinstance(qk.key, tuple) else qk.key,
                            "asymmetric": qk.asymmetric,
                            "is_ego_asymmetric": qk.asymmetric,
                        }
                        for qk in (
                            r.keys.questions[(idx_r + 1) % len(r.keys.questions) :]
                            + r.keys.questions[: (idx_r + 1) % len(r.keys.questions)]
                        )
                    ],
                }
                for idx_r, r in enumerate(cand_pass2)
            ]
            cand_pub_check = [{"item_id": f"item_{i + 1:04d}"} for i in range(len(cand_sealed_check))]
            try:
                check_not_in_key_order(cand_pub_check, cand_sealed_check, {})
                shuffle_salt_index = salt_idx
                pass1_list = cand_pass1
                pass2_list = cand_pass2
                break
            except SealingError:
                continue
        else:
            raise SealingError(
                "Failed to find a Pass-1/Pass-2 shuffle salt passing check_not_in_key_order within 256 attempts."
            )

    ordered_renderings: list[tuple[int, RenderedScenario]] = (
        [(1, r) for r in pass1_list] + [(2, r) for r in pass2_list]
    )

    # Choose a deterministic question-shuffle salt derived from `seed_hex` so that the shuffled
    # question positions across bundles pass probe (e) (`abs(oof - 0.125) <= 0.05`) and never
    # repeat a single fixed permutation across bundles.
    qshuffle_salt_index = 0
    qshuffle_prefix = "qshuffle"
    first_valid_qshuffle: tuple[int, str] | None = None
    if ordered_renderings:
        for attempt in range(32):
            cand_prefix = "qshuffle" if attempt == 0 else f"qshuffle:r{attempt}"
            probe_e_samples: list[tuple[str, list[str], str]] = []
            cand_orders: list[tuple[str, ...]] = []
            for idx, (_, rend) in enumerate(ordered_renderings, start=1):
                item_id = f"item_{idx:04d}"
                q_list_cand = list(rend.keys.questions)
                q_rng_cand = random.Random(
                    _hmac_hex(seed_hex, f"{cand_prefix}:{item_id}:{rend.scenario_id}:{rend.rendering_index}", 32)
                )
                q_rng_cand.shuffle(q_list_cand)
                cand_orders.append(tuple(qk.question_type for qk in q_list_cand))
                for pos_idx, qk in enumerate(q_list_cand):
                    probe_e_samples.append((rend.scenario_id, [f"pos_{pos_idx}"], qk.question_type))
            if len(cand_orders) >= 2 and len(set(cand_orders)) == 1:
                continue
            if first_valid_qshuffle is None:
                first_valid_qshuffle = (attempt, cand_prefix)
            if abs(_oof_multinomial_nb(probe_e_samples) - 0.125) <= 0.05 + 1e-9:
                qshuffle_salt_index = attempt
                qshuffle_prefix = cand_prefix
                break
        else:
            if first_valid_qshuffle is not None:
                qshuffle_salt_index, qshuffle_prefix = first_valid_qshuffle

    sealed_rows: list[dict[str, Any]] = []
    for idx, (pass_index, rend) in enumerate(ordered_renderings, start=1):
        item_id = f"item_{idx:04d}"
        scen_handle = _hmac_hex(seed_hex, f"scenario:{rend.scenario_id}", 24)
        rend_handle = _hmac_hex(seed_hex, f"rendering:{rend.scenario_id}:{rend.rendering_index}", 24)

        # Shuffle the questions inside this bundle with an item-specific seed derived from `seed_hex`.
        q_list = list(rend.keys.questions)
        q_rng = random.Random(
            _hmac_hex(seed_hex, f"{qshuffle_prefix}:{item_id}:{rend.scenario_id}:{rend.rendering_index}", 32)
        )
        q_rng.shuffle(q_list)

        sealed_questions: list[dict[str, Any]] = []
        for q_idx, qk in enumerate(q_list, start=1):
            qid = f"q_{q_idx}"
            key_serialized: str | list[str] = list(qk.key) if isinstance(qk.key, tuple) else qk.key
            lure_serialized: str | list[str] = (
                list(qk.ego_lure_key) if isinstance(qk.ego_lure_key, tuple) else qk.ego_lure_key
            )
            sealed_questions.append(
                {
                    "question_id": qid,
                    "slot_id": qk.slot_id,
                    "question_type": qk.question_type,
                    "curriculum_stage": qk.curriculum_stage,
                    "question_text": qk.question_text,
                    "key": key_serialized,
                    "ego_lure_key": lure_serialized,
                    "asymmetric": qk.asymmetric,
                    "in_primary_bundle": qk.in_primary_bundle,
                    "target_agent": qk.target_agent,
                    "queried_variable": qk.queried_variable,
                    "pressure_lure_answer": qk.pressure_lure_answer,
                    "pressure_probe_turn2": build_pressure_probe_turn2(
                        qid, qk.question_text, qk.pressure_lure_answer
                    ),
                    "pressure_probe_turn2_by_candidate": build_public_pressure_probe_map(
                        qid, qk.question_text, qk.candidate_lures
                    ),
                }
            )

        perc_dict = {
            ag: {
                "is_absent": pr.is_absent,
                "seen_steps": list(pr.seen_steps),
                "heard_steps": list(pr.heard_steps),
                "messaged_steps": list(pr.messaged_steps),
                "admissible_steps": list(pr.admissible_steps),
                "heard_bluff_steps": list(pr.heard_bluff_steps),
            }
            for ag, pr in sorted(rend.keys.perceptual_records.items())
        }

        resolved_goal_source = (
            dict(rend.goal_source)
            if rend.goal_source is not None
            else {
                ag: "model_text"
                for ag in (rend.world.agents if rend.world is not None else ())
            }
        )
        sealed_rows.append(
            {
                "item_id": item_id,
                "pass_index": pass_index,
                "scenario_id": rend.scenario_id,
                "rendering_index": rend.rendering_index,
                "scenario_handle": scen_handle,
                "rendering_handle": rend_handle,
                "focal_agent": rend.focal_agent,
                "prose": rend.prose,
                "goal_mode": rend.goal_mode,
                "goal_source": resolved_goal_source,
                "attempt": rend.attempt,
                "temperature": rend.temperature,
                "min_p": rend.min_p,
                "presence_penalty": rend.presence_penalty,
                "seed": rend.seed,
                "sampling": {
                    "attempt": rend.attempt,
                    "temperature": rend.temperature,
                    "min_p": rend.min_p,
                    "seed": rend.seed,
                },
                "key_source": "code",
                "perceptual_records": perc_dict,
                "known_facts_by_agent": {
                    ag: list(fids) for ag, fids in sorted(rend.keys.known_facts_by_agent.items())
                },
                "questions": sealed_questions,
            }
        )

    keys_lines = [_canonical_json_line(r) for r in sealed_rows]
    keys_bytes = (("\n".join(keys_lines) + "\n") if keys_lines else "").encode("utf-8")
    keys_sha = hashlib.sha256(keys_bytes).hexdigest()

    commitment_hmac = _hmac_hex(seed_hex, f"commitment:{keys_sha}", 64)
    serialized_worlds: dict[str, dict[str, Any]] = {}
    for r0, r1 in rendered_pairs:
        w_obj = r0.world or r1.world
        if w_obj is not None:
            validate_topology(w_obj)
            validate_occlusion(w_obj)
            validate_belief_forcing(w_obj)
            validate_hearing(w_obj)
            validate_event_locations(w_obj)
            validate_object_locations(w_obj)
            validate_placing_fairness(w_obj)
            serialized_worlds[w_obj.scenario_id] = _serialize_world(w_obj)

    serialized_all_worlds: dict[str, dict[str, Any]] = dict(serialized_worlds)
    if all_worlds is not None:
        for w_obj in all_worlds:
            if w_obj.scenario_id not in serialized_all_worlds:
                validate_topology(w_obj)
                validate_occlusion(w_obj)
                validate_belief_forcing(w_obj)
                validate_hearing(w_obj)
                validate_event_locations(w_obj)
                validate_object_locations(w_obj)
                validate_placing_fairness(w_obj)
                serialized_all_worlds[w_obj.scenario_id] = _serialize_world(w_obj)

    sealing_meta: dict[str, Any] = {
        "schema_version": "udlab.sealing.v1",
        "private_seed_hex": seed_hex,
        "shuffle_salt_index": shuffle_salt_index,
        "qshuffle_salt_index": qshuffle_salt_index,
        "num_scenarios": len(rendered_pairs),
        "num_items": len(sealed_rows),
        "keys_sealed_file": "keys.sealed.jsonl",
        "keys_sealed_sha256": keys_sha,
        "commitment_hmac_sha256": commitment_hmac,
        "scenarios": serialized_worlds,
        "all_scenarios": serialized_all_worlds,
    }
    if writer_provenance is not None:
        sealing_meta["writer_provenance"] = dict(writer_provenance)
    sealing_meta_bytes = _canonical_json_doc(sealing_meta).encode("utf-8")

    items_bytes, part_bytes, public_items, part_doc = _render_public_from_sealed(
        sealed_rows, sealing_meta
    )

    # Check key order and field hygiene before writing files.
    check_not_in_key_order(public_items, sealed_rows, part_doc, private_seed_hex=seed_hex)

    keys_sealed_path = sealed_dir / "keys.sealed.jsonl"
    sealing_meta_path = sealed_dir / "sealing.sealed.json"
    items_path = public_dir / "items.jsonl"
    part_json_path = public_dir / "part.json"

    keys_sealed_path.write_bytes(keys_bytes)
    sealing_meta_path.write_bytes(sealing_meta_bytes)
    items_path.write_bytes(items_bytes)
    part_json_path.write_bytes(part_bytes)

    # Run canonical verify (re-render and byte-for-byte diff) immediately on freeze.
    verify_sealed_part(public_dir, sealed_dir)

    return SealedDatasetReceipt(
        public_dir=public_dir,
        sealed_dir=sealed_dir,
        items_path=items_path,
        part_json_path=part_json_path,
        keys_sealed_path=keys_sealed_path,
        sealing_meta_path=sealing_meta_path,
        items_sha256=hashlib.sha256(items_bytes).hexdigest(),
        part_json_sha256=hashlib.sha256(part_bytes).hexdigest(),
        keys_sealed_sha256=keys_sha,
        num_scenarios=len(rendered_pairs),
        num_items=len(sealed_rows),
    )


def verify_sealed_part(public_dir: Path, sealed_dir: Path) -> None:
    """Canonical `verify` (re-render and diff, design section 6 item 3):

    Regenerates `items.jsonl` and `part.json` from `sealed_dir` and requires a
    byte-for-byte match with `public_dir`, plus passing `check_not_in_key_order`.
    """
    _validate_disjoint_dirs(public_dir, sealed_dir)

    keys_sealed_path = sealed_dir / "keys.sealed.jsonl"
    sealing_meta_path = sealed_dir / "sealing.sealed.json"
    items_path = public_dir / "items.jsonl"
    part_json_path = public_dir / "part.json"

    for req_path in (keys_sealed_path, sealing_meta_path, items_path, part_json_path):
        if not req_path.is_file():
            raise SealingError(f"Unverifiable part: required file '{req_path}' is missing.")

    keys_bytes = keys_sealed_path.read_bytes()
    sealing_meta = json.loads(sealing_meta_path.read_text(encoding="utf-8"))
    seed_hex = _validate_private_seed(str(sealing_meta.get("private_seed_hex", "")))

    actual_keys_sha = hashlib.sha256(keys_bytes).hexdigest()
    if actual_keys_sha != sealing_meta.get("keys_sealed_sha256"):
        raise SealingError("Sealed keys file SHA-256 does not match sealing.sealed.json.")

    expected_hmac = _hmac_hex(seed_hex, f"commitment:{actual_keys_sha}", 64)
    if expected_hmac != sealing_meta.get("commitment_hmac_sha256"):
        raise SealingError("Sealed commitment HMAC does not match private seed and keys digest.")

    sealed_rows = [
        json.loads(line)
        for line in keys_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]

    expected_items_bytes, expected_part_bytes, expected_pub_items, expected_part_doc = (
        _render_public_from_sealed(sealed_rows, sealing_meta)
    )

    actual_items_bytes = items_path.read_bytes()
    actual_part_bytes = part_json_path.read_bytes()

    if actual_items_bytes != expected_items_bytes:
        raise SealingError(
            "Canonical verify failed: public items.jsonl differs byte-for-byte from re-rendered sealed store."
        )
    if actual_part_bytes != expected_part_bytes:
        raise SealingError(
            "Canonical verify failed: public part.json differs byte-for-byte from re-rendered sealed store."
        )

    check_not_in_key_order(
        expected_pub_items,
        sealed_rows,
        expected_part_doc,
        private_seed_hex=seed_hex,
    )

    # Validate topology, occlusion, and hearing for every scenario stored in the sealed metadata,
    # and verify that sealed keys and Pass-1/Pass-2 shuffle permutation match the stored salt index.
    scenarios_meta = sealing_meta.get("scenarios", {})
    if isinstance(scenarios_meta, dict) and scenarios_meta:
        validated_keys_by_scen: dict[str, Any] = {}
        for scen_id, w_dict in sorted(scenarios_meta.items()):
            w_obj = _deserialize_world(w_dict)
            try:
                validate_topology(w_obj)
                validate_occlusion(w_obj)
                validate_belief_forcing(w_obj)
                validate_hearing(w_obj)
                validate_event_locations(w_obj)
                validate_object_locations(w_obj)
                validate_placing_fairness(w_obj)
            except ValueError as exc:
                raise SealingError(f"Sealed world validation failed for {scen_id}: {exc}") from exc
            validated_keys_by_scen[scen_id] = compute_scenario_keys(w_obj)

        salt_idx = int(sealing_meta.get("shuffle_salt_index", 0))
        p1_msg = "shuffle:pass1" if salt_idx == 0 else f"shuffle:pass1:r{salt_idx}"
        p2_msg = "shuffle:pass2" if salt_idx == 0 else f"shuffle:pass2:r{salt_idx}"
        expected_p1_sids = sorted(scenarios_meta.keys())
        expected_p2_sids = sorted(scenarios_meta.keys())
        random.Random(_hmac_hex(seed_hex, p1_msg, 32)).shuffle(expected_p1_sids)
        random.Random(_hmac_hex(seed_hex, p2_msg, 32)).shuffle(expected_p2_sids)
        if len(expected_p1_sids) >= 2 and (
            expected_p1_sids == sorted(expected_p1_sids)
            or expected_p1_sids == sorted(expected_p1_sids, reverse=True)
        ):
            expected_p1_sids = expected_p1_sids[1:] + expected_p1_sids[:1]

        actual_p1_sids = [str(r.get("scenario_id", "")) for r in sealed_rows if int(r.get("pass_index", 1)) == 1]
        actual_p2_sids = [str(r.get("scenario_id", "")) for r in sealed_rows if int(r.get("pass_index", 1)) == 2]
        if actual_p1_sids != expected_p1_sids or actual_p2_sids != expected_p2_sids:
            raise SealingError(
                f"Pass-1/Pass-2 shuffle permutation in keys.sealed.jsonl does not match "
                f"sealing.sealed.json (shuffle_salt_index={salt_idx})."
            )

        for s_row in sealed_rows:
            scen_id = str(s_row.get("scenario_id", ""))
            if scen_id in validated_keys_by_scen:
                expected_skeys = validated_keys_by_scen[scen_id]
                expected_q_by_slot = {qk.slot_id: qk for qk in expected_skeys.questions}
                for q_dict in s_row.get("questions", []):
                    slot_id = str(q_dict.get("slot_id", ""))
                    if slot_id not in expected_q_by_slot:
                        raise SealingError(f"Unknown slot_id '{slot_id}' in {scen_id}.")
                    exp_q = expected_q_by_slot[slot_id]
                    exp_key_ser = list(exp_q.key) if isinstance(exp_q.key, tuple) else exp_q.key
                    if q_dict.get("key") != exp_key_ser:
                        raise SealingError(
                            f"Sealed key mismatch in {scen_id} {slot_id}: "
                            f"sealed_sha256={_value_sha256(q_dict.get('key'))} != "
                            f"expected_sha256={_value_sha256(exp_key_ser)}."
                        )


def derive_b2_from_rule_sentence_alone(w: World) -> str:
    """Independently derive `B2(b2_observer, b2_actor, primary_object)` from the stated belief rule sentence alone.

    Also verifies that `b2_observer` never encounters an unexplained displacement of `primary_object`.
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
                    if last_seen_by_obs is None or ev.from_location != last_seen_by_obs:
                        raise ValueError(
                            f"{w.scenario_id} step {ev.step}: b2_observer {w.b2_observer} found {w.primary_object} "
                            f"at {ev.from_location!r}, but last placed/saw it at {last_seen_by_obs!r} (unexplained displacement)."
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

    if b2_from_rule is None:
        raise ValueError(f"{w.scenario_id}: could not derive B2 from stated rule sentence.")
    return b2_from_rule


def independent_recount_world_keys(w: World) -> dict[str, Any]:
    """Independently recount perceptual records, known facts, and all keys (`P, K, B1, B2, X, D, S, U`) from `w.events`."""
    edges = {(a, b) for a, b in w.doorways} | {(b, a) for a, b in w.doorways}
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

    for ev in w.events:
        if ev.kind == "leave":
            if room_of.get(ev.actor) != ev.room or ev.room != ev.from_location:
                raise ValueError(
                    f"{w.scenario_id} step {ev.step}: {ev.actor} leaves {ev.room} while in {room_of.get(ev.actor)}"
                )
            if (ev.from_location, ev.to_location) not in edges:
                raise ValueError(
                    f"{w.scenario_id} step {ev.step}: off-edge leave {ev.from_location} -> {ev.to_location}"
                )
        elif ev.kind == "enter":
            if room_of.get(ev.actor) is not None:
                raise ValueError(
                    f"{w.scenario_id} step {ev.step}: {ev.actor} enters {ev.room} without leaving previous room"
                )
            if (ev.from_location, ev.to_location) not in edges or ev.room != ev.to_location:
                raise ValueError(
                    f"{w.scenario_id} step {ev.step}: off-edge enter {ev.from_location} -> {ev.to_location}"
                )
            room_of[ev.actor] = ev.room
            station_of[ev.actor] = "container bay"
        else:
            if room_of.get(ev.actor) != ev.room:
                raise ValueError(
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

            p_heard = False
            if p_room is not None and ev.kind == "say":
                if p_room == ev.room:
                    p_heard = True
                elif (p_room, ev.room) in edges and ev.noise_level >= w.noise_threshold:
                    p_heard = True

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
    b2_key = derive_b2_from_rule_sentence_alone(w)
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

    if w.settled_twin_spec.subtype == "who":
        s_candidates = [
            ev.actor
            for ev in w.events
            if ev.kind == "move"
            and ev.target_object
            and ev.target_object in w.settled_twin_spec.side_action_desc
        ]
        s_key = s_candidates[-1]
    else:
        s_candidates = [
            ev.stated_reason
            for ev in w.events
            if ev.actor == w.settled_twin_spec.target_agent and ev.stated_reason is not None
        ]
        s_key = str(s_candidates[-1])

    u_key = verify_unknowable_u(w)
    assert u_key == UNKNOWABLE_KEY

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
        "S": s_key,
        "U": u_key,
    }


def verify_written_dataset_recounts_and_prose(
    public_dir: Path,
    sealed_dir: Path,
) -> dict[str, Any]:
    """Verify sealed part, prose checks (a)-(d), independent B2 derivation, and all independent recounts."""
    verify_sealed_part(public_dir, sealed_dir)

    keys_sealed_path = sealed_dir / "keys.sealed.jsonl"
    sealing_meta_path = sealed_dir / "sealing.sealed.json"
    sealing_meta = json.loads(sealing_meta_path.read_text(encoding="utf-8"))
    sealed_rows = [
        json.loads(line)
        for line in keys_sealed_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scenarios_meta = sealing_meta.get("scenarios", {})
    if not isinstance(scenarios_meta, dict):
        raise SealingError("sealing.sealed.json has invalid scenarios metadata.")
    if not scenarios_meta and sealed_rows:
        raise SealingError("sealing.sealed.json is missing scenarios metadata for recount verification.")

    worlds_by_sid: dict[str, World] = {}
    recounts_by_sid: dict[str, dict[str, Any]] = {}
    b2_by_sid: dict[str, str] = {}
    for scen_id, w_dict in sorted(scenarios_meta.items()):
        w_obj = _deserialize_world(w_dict)
        worlds_by_sid[scen_id] = w_obj
        b2_by_sid[scen_id] = derive_b2_from_rule_sentence_alone(w_obj)
        recounts_by_sid[scen_id] = independent_recount_world_keys(w_obj)

    prose_checked = 0
    recount_checked = 0
    duplicate_stake_statements = 0
    for row in sealed_rows:
        scen_id = str(row["scenario_id"])
        if scen_id not in worlds_by_sid:
            raise SealingError(f"Row references unknown scenario_id '{scen_id}'.")
        w_obj = worlds_by_sid[scen_id]
        focal_ag = str(row["focal_agent"])
        prose = str(row["prose"])

        # Re-run all prose checks (a)-(d) on every written rendering
        try:
            check_rendered_prose(prose, w_obj, focal_agent=focal_ag)
        except RenderCheckError as exc:
            sanitized_exc = sanitize_render_check_error_for_stderr(exc, prose)
            sanitized_summary = format_non_sealed_refusal_reason(
                check_id=sanitized_exc.check_id,
                event_index=sanitized_exc.event_index,
                event_kind=sanitized_exc.event_kind,
                failing_sentence_sha256=sanitized_exc.failing_sentence_sha256,
                failing_sentence_offset=sanitized_exc.failing_sentence_offset,
                failing_sentence_length=sanitized_exc.failing_sentence_length,
            )
            raise SealingError(
                f"Prose check failed in {scen_id} (focal={focal_ag}): {sanitized_summary}"
            ) from None
        except Exception as exc:
            raise SealingError(
                f"Prose check failed in {scen_id} (focal={focal_ag}): {type(exc).__name__}"
            ) from None
        prose_checked += 1
        duplicate_stake_statements += count_duplicate_stake_statements(prose, w_obj)

        recounted = recounts_by_sid[scen_id]
        perc_row = row.get("perceptual_records", {})
        for party, p_rec in perc_row.items():
            if tuple(p_rec.get("seen_steps", ())) != recounted["seen"][party]:
                raise SealingError(f"Recount seen_steps mismatch in {scen_id} for {party}.")
            if tuple(p_rec.get("heard_steps", ())) != recounted["heard"][party]:
                raise SealingError(f"Recount heard_steps mismatch in {scen_id} for {party}.")
            if tuple(p_rec.get("messaged_steps", ())) != recounted["messaged"][party]:
                raise SealingError(f"Recount messaged_steps mismatch in {scen_id} for {party}.")
            if tuple(p_rec.get("admissible_steps", ())) != recounted["admissible"][party]:
                raise SealingError(f"Recount admissible_steps mismatch in {scen_id} for {party}.")

        known_row = {
            p: tuple(fids) for p, fids in row.get("known_facts_by_agent", {}).items()
        }
        if known_row != recounted["known_facts"]:
            raise SealingError(f"Recount known_facts_by_agent mismatch in {scen_id}.")

        for q in row.get("questions", ()):
            qtype = str(q["question_type"])
            raw_key = q["key"]
            norm_key = tuple(raw_key) if isinstance(raw_key, list) else raw_key
            if norm_key != recounted[qtype]:
                raise SealingError(
                    f"Independent recount mismatch in {scen_id} for {qtype}: "
                    f"sealed_sha256={_value_sha256(norm_key)} != recounted_sha256={_value_sha256(recounted[qtype])}."
                )
            if qtype == "B2" and str(raw_key) != b2_by_sid[scen_id]:
                raise SealingError(
                    f"Independent B2 rule-sentence derivation mismatch in {scen_id}: "
                    f"sealed_sha256={_value_sha256(str(raw_key))} != rule_derived_sha256={_value_sha256(b2_by_sid[scen_id])}."
                )
        recount_checked += 1

    return {
        "scenarios_recounted": len(worlds_by_sid),
        "renderings_prose_checked": prose_checked,
        "rows_recount_verified": recount_checked,
        "duplicate_stake_statements": duplicate_stake_statements,
        "independent_b2_verified": True,
        "independent_recounts_verified": True,
        "prose_checks_passed": True,
    }
