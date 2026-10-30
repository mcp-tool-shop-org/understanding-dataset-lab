"""CLI entrypoint for the udlab release-1 scenario generator, sealing, and shortcut probe."""

from __future__ import annotations

from collections import Counter
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from udlab_generator.diversity import count_template_diversity
from udlab_generator.probe import run_shortcut_probe
from udlab_generator.render import MockWriter
from udlab_generator.sealing import verify_sealed_part, write_sealed_dataset
from udlab_generator.world import generate_world, make_balanced_configs


def _summarize_d_key_patterns(sealed_rows: Sequence[dict[str, object]]) -> dict[str, object]:
    """Summarize the distribution of D key patterns and set sizes across scenarios."""
    pattern_counts: Counter[str] = Counter()
    size_counts: Counter[str] = Counter()
    total_d = 0
    for row in sealed_rows:
        if int(row.get("pass_index", 1)) != 1:  # type: ignore[arg-type]
            continue
        questions = row.get("questions", [])
        if not isinstance(questions, list):
            continue
        for q in questions:
            if isinstance(q, dict) and q.get("question_type") == "D":
                raw_key = q.get("key", [])
                if isinstance(raw_key, (list, tuple)):
                    fact_ids = tuple(sorted(str(part).strip() for part in raw_key if str(part).strip()))
                else:
                    fact_ids = tuple(sorted(part.strip() for part in str(raw_key).split(",") if part.strip()))
                pattern_label = ",".join(fact_ids) if fact_ids else "NONE"
                pattern_counts[pattern_label] += 1
                size_counts[str(len(fact_ids))] += 1
                total_d += 1
    max_pattern_count = max(pattern_counts.values(), default=0)
    max_pattern_share = round(max_pattern_count / total_d, 6) if total_d > 0 else 0.0
    return {
        "total_scenarios": total_d,
        "distinct_patterns": len(pattern_counts),
        "max_pattern_count": max_pattern_count,
        "max_pattern_share": max_pattern_share,
        "counts_by_pattern": dict(sorted(pattern_counts.items())),
        "counts_by_set_size": dict(sorted(size_counts.items())),
    }


def run_probe_only(sealed_jsonl_path: Path) -> dict[str, object]:
    """Run the shortcut probe directly on any sealed JSONL file (including writer-rendered prose)."""
    sealed_rows = [
        json.loads(line)
        for line in sealed_jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    probe_report = run_shortcut_probe(sealed_rows)
    return {
        "sealed_jsonl_path": str(sealed_jsonl_path),
        "num_rows": len(sealed_rows),
        "d_key_pattern_distribution": _summarize_d_key_patterns(sealed_rows),
        "probe_report": asdict(probe_report),
    }


def build_and_probe_dataset(
    *,
    count: int = 300,
    base_seed: int = 20261030,
    private_seed_hex: str | None = None,
    output_dir: Path,
) -> dict[str, object]:
    """Generate `count` scenarios, seal keys, verify part, and run shortcut probes and diversity count."""
    configs = make_balanced_configs(count, base_seed=base_seed)
    worlds = [generate_world(cfg) for cfg in configs]

    writer = MockWriter()
    rendered_pairs = [writer.render_scenario_pair(w) for w in worlds]

    public_dir = output_dir / "part_public"
    sealed_dir = output_dir / "sealed_store"

    receipt = write_sealed_dataset(
        rendered_pairs=rendered_pairs,
        public_dir=public_dir,
        sealed_dir=sealed_dir,
        private_seed_hex=private_seed_hex,
    )
    verify_sealed_part(public_dir, sealed_dir)

    sealed_rows = [
        json.loads(line)
        for line in receipt.keys_sealed_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    probe_report = run_shortcut_probe(sealed_rows)
    diversity_report = count_template_diversity(worlds)
    d_distribution = _summarize_d_key_patterns(sealed_rows)

    summary = {
        "num_scenarios": receipt.num_scenarios,
        "num_items": receipt.num_items,
        "items_sha256": receipt.items_sha256,
        "part_json_sha256": receipt.part_json_sha256,
        "keys_sealed_sha256": receipt.keys_sealed_sha256,
        "d_key_pattern_distribution": d_distribution,
        "probe_report": asdict(probe_report),
        "diversity_report": {
            "total_scenarios": diversity_report.total_scenarios,
            "distinct_event_skeletons": diversity_report.distinct_event_skeletons,
            "distinct_domain_skeletons": diversity_report.distinct_domain_skeletons,
            "max_cluster_size": diversity_report.max_cluster_size,
            "mean_cluster_size": diversity_report.mean_cluster_size,
        },
    }
    summary_path = output_dir / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="udlab-gen",
        description="Generate, seal, verify, and probe udlab release-1 cognitive-empathy scenarios.",
    )
    parser.add_argument("--count", type=int, default=300, help="Number of scenarios to generate (default: 300).")
    parser.add_argument("--seed", type=int, default=20261030, help="Base deterministic generation seed.")
    parser.add_argument(
        "--private-seed",
        type=str,
        default=None,
        help="Optional 256-bit hex private shuffle seed (if omitted, reads UDLAB_PRIVATE_SEED or generates fresh secrets.token_hex(32)).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/r1_300"),
        help="Output directory containing part_public/ and sealed_store/.",
    )
    parser.add_argument(
        "--probe-only",
        type=Path,
        default=None,
        help="Run the shortcut probe directly on an existing sealed JSONL file and exit.",
    )
    args = parser.parse_args(argv)

    if args.probe_only is not None:
        result = run_probe_only(args.probe_only)
        probe_passed = bool(result["probe_report"]["passed"])  # type: ignore[index]
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if probe_passed else 1

    resolved_private_seed = args.private_seed or os.environ.get("UDLAB_PRIVATE_SEED") or None
    summary = build_and_probe_dataset(
        count=args.count,
        base_seed=args.seed,
        private_seed_hex=resolved_private_seed,
        output_dir=args.out_dir,
    )
    probe_passed = bool(summary["probe_report"]["passed"])  # type: ignore[index]
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if probe_passed else 1


if __name__ == "__main__":
    sys.exit(main())
