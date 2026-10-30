"""CLI entrypoint for the udlab release-1 scenario generator, sealing, and shortcut probe."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Sequence

from udlab_generator.diversity import count_template_diversity
from udlab_generator.probe import run_shortcut_probe
from udlab_generator.render import MockWriter
from udlab_generator.sealing import verify_sealed_part, write_sealed_dataset
from udlab_generator.world import generate_world, make_balanced_configs


def build_and_probe_dataset(
    *,
    count: int = 300,
    base_seed: int = 20261030,
    private_seed_hex: str = "a94f3c82e1b07d65432198fedcba0123456789abcdef0123456789abcdef0123",
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

    summary = {
        "num_scenarios": receipt.num_scenarios,
        "num_items": receipt.num_items,
        "items_sha256": receipt.items_sha256,
        "part_json_sha256": receipt.part_json_sha256,
        "keys_sealed_sha256": receipt.keys_sealed_sha256,
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
        default="a94f3c82e1b07d65432198fedcba0123456789abcdef0123456789abcdef0123",
        help="256-bit hex private shuffle seed stored only in sealed_store.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/r1_300"),
        help="Output directory containing part_public/ and sealed_store/.",
    )
    args = parser.parse_args(argv)

    summary = build_and_probe_dataset(
        count=args.count,
        base_seed=args.seed,
        private_seed_hex=args.private_seed,
        output_dir=args.out_dir,
    )
    probe_passed = bool(summary["probe_report"]["passed"])  # type: ignore[index]
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if probe_passed else 1


if __name__ == "__main__":
    sys.exit(main())
