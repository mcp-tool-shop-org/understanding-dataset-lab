"""Sealing, private-seed shuffling, key-order leak check, and canonical re-render/diff verify."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path
import random
import secrets
from typing import Any

from udlab_generator.keys import build_public_pressure_probe_template, build_pressure_probe_turn2
from udlab_generator.render import RenderedScenario

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
    if len(type_orders) >= 4 and len(set(type_orders)) == 1:
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

        # 5. Check binary key sequences across Pass-1 bundles for contiguous grouping or periodic alternation.
        for target_qtype in ("P", "K"):
            seq: list[str] = []
            for r in pass1_rows:
                for q in r.get("questions", []):
                    if q.get("question_type") == target_qtype:
                        seq.append(str(q.get("key", "")))
                        break
            if len(seq) >= 8 and len(set(seq)) >= 2:
                transitions = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
                if transitions <= 2:
                    raise SealingError(
                        f"Key-order leak detected: '{target_qtype}' keys are grouped contiguously ({transitions} transitions)."
                    )
                half = len(seq) // 2
                if len(seq) == 2 * half and (
                    seq == ["yes", "no"] * half or seq == ["no", "yes"] * half
                ):
                    raise SealingError(
                        f"Key-order leak detected: '{target_qtype}' keys follow a strict periodic alternating sequence."
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
            pub_questions.append(
                {
                    "question_id": qid,
                    "question_text": qtext,
                    "pressure_probe_turn2_template": build_public_pressure_probe_template(qid, qtext),
                }
            )
        pub_row: dict[str, Any] = {
            "item_id": str(s_row["item_id"]),
            "pass_index": int(s_row["pass_index"]),
            "scenario_handle": str(s_row["scenario_handle"]),
            "rendering_handle": str(s_row["rendering_handle"]),
            "focal_agent": str(s_row["focal_agent"]),
            "prose": str(s_row["prose"]),
            "questions": pub_questions,
        }
        public_items.append(pub_row)
        item_lines.append(_canonical_json_line(pub_row))

    items_bytes = ("\n".join(item_lines) + "\n").encode("utf-8")
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
) -> SealedDatasetReceipt:
    """Shuffle items and questions with a private seed, write public and sealed stores, and run canonical verify."""
    if not rendered_pairs:
        raise ValueError("rendered_pairs must not be empty")

    _validate_disjoint_dirs(public_dir, sealed_dir)
    seed_hex = _validate_private_seed(private_seed_hex or secrets.token_hex(32))

    public_dir.mkdir(parents=True, exist_ok=True)
    sealed_dir.mkdir(parents=True, exist_ok=True)

    # Separate first renderings (Pass 1) and second renderings (Pass 2), and shuffle each pass
    # independently using deterministic sub-seeds derived from the private seed.
    pass1_list = [pair[0] for pair in rendered_pairs]
    pass2_list = [pair[1] for pair in rendered_pairs]

    rng_pass1 = random.Random(_hmac_hex(seed_hex, "shuffle:pass1", 32))
    rng_pass2 = random.Random(_hmac_hex(seed_hex, "shuffle:pass2", 32))
    rng_pass1.shuffle(pass1_list)
    rng_pass2.shuffle(pass2_list)

    # Ensure Pass 1 is never accidentally sorted even for small N.
    if len(pass1_list) >= 2 and [r.scenario_id for r in pass1_list] == sorted(r.scenario_id for r in pass1_list):
        pass1_list = pass1_list[1:] + pass1_list[:1]

    ordered_renderings: list[tuple[int, RenderedScenario]] = (
        [(1, r) for r in pass1_list] + [(2, r) for r in pass2_list]
    )

    sealed_rows: list[dict[str, Any]] = []
    for idx, (pass_index, rend) in enumerate(ordered_renderings, start=1):
        item_id = f"item_{idx:04d}"
        scen_handle = _hmac_hex(seed_hex, f"scenario:{rend.scenario_id}", 24)
        rend_handle = _hmac_hex(seed_hex, f"rendering:{rend.scenario_id}:{rend.rendering_index}", 24)

        # Shuffle the questions inside this bundle with an item-specific seed derived from `seed_hex`.
        q_list = list(rend.keys.questions)
        q_rng = random.Random(_hmac_hex(seed_hex, f"qshuffle:{item_id}:{rend.scenario_id}:{rend.rendering_index}", 32))
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
                "key_source": "code",
                "perceptual_records": perc_dict,
                "known_facts_by_agent": {
                    ag: list(fids) for ag, fids in sorted(rend.keys.known_facts_by_agent.items())
                },
                "questions": sealed_questions,
            }
        )

    keys_lines = [_canonical_json_line(r) for r in sealed_rows]
    keys_bytes = ("\n".join(keys_lines) + "\n").encode("utf-8")
    keys_sha = hashlib.sha256(keys_bytes).hexdigest()

    commitment_hmac = _hmac_hex(seed_hex, f"commitment:{keys_sha}", 64)
    sealing_meta: dict[str, Any] = {
        "schema_version": "udlab.sealing.v1",
        "private_seed_hex": seed_hex,
        "num_scenarios": len(rendered_pairs),
        "num_items": len(sealed_rows),
        "keys_sealed_file": "keys.sealed.jsonl",
        "keys_sealed_sha256": keys_sha,
        "commitment_hmac_sha256": commitment_hmac,
    }
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
