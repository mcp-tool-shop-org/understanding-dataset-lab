"""CLI entrypoint for the udlab release-1 scenario generator, writer pass, sealing, and shortcut probe."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
import time
from typing import Any

from udlab_generator.diversity import count_template_diversity
from udlab_generator.probe import run_shortcut_probe
from udlab_generator.render import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_VRAM_HALT_MIB,
    DEFAULT_WRITER_MAX_NUM_CTX,
    DEFAULT_WRITER_MODEL_PIN,
    DEFAULT_WRITER_NUM_PREDICT,
    GENERATIONS_FILENAME,
    NUM_CTX_MARGIN_TOKENS,
    NUM_CTX_STEP_TOKENS,
    GenerationCallRecord,
    MockWriter,
    OllamaHttpTransport,
    OllamaWriter,
    RenderedScenario,
    WriterCheckpointManager,
    compute_num_ctx,
    estimate_prompt_tokens,
    kill_pid_tree,
)
from udlab_generator.sealing import (
    verify_sealed_part,
    verify_written_dataset_recounts_and_prose,
    write_sealed_dataset,
)
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


def verify_written_dataset(output_dir: Path) -> dict[str, object]:
    """Post-write verification command (`--verify-written <dir>`):

    1. Runs canonical `verify_sealed_part`, prose checks (a)-(d), independent B2 derivation
       from the stated rule sentence alone, and all independent recounts (`P, K, B1, B2, X, D, S, U`).
    2. Runs `--probe-only` (`run_probe_only`) on the written `sealed_store/keys.sealed.jsonl`.
    """
    public_dir = output_dir / "part_public"
    sealed_dir = output_dir / "sealed_store"
    keys_sealed_path = sealed_dir / "keys.sealed.jsonl"
    generations_path = output_dir / GENERATIONS_FILENAME

    recount_report = verify_written_dataset_recounts_and_prose(public_dir, sealed_dir)
    probe_only_result = run_probe_only(keys_sealed_path)

    generations_sha256: str | None = None
    generations_count = 0
    if generations_path.is_file():
        gen_bytes = generations_path.read_bytes()
        generations_sha256 = hashlib.sha256(gen_bytes).hexdigest()
        generations_count = sum(1 for line in gen_bytes.decode("utf-8").splitlines() if line.strip())

    result: dict[str, object] = {
        "output_dir": str(output_dir),
        "keys_sealed_path": str(keys_sealed_path),
        "generations_path": str(generations_path) if generations_path.is_file() else None,
        "generations_sha256": generations_sha256,
        "generations_count": generations_count,
        "recount_verification": recount_report,
        "probe_only": probe_only_result,
        "passed": bool(
            recount_report["independent_b2_verified"]
            and recount_report["independent_recounts_verified"]
            and recount_report["prose_checks_passed"]
            and probe_only_result["probe_report"]["passed"]  # type: ignore[index]
        ),
    }
    verify_report_path = output_dir / "verify_written_report.json"
    verify_report_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def build_and_probe_dataset(
    *,
    count: int = 300,
    base_seed: int = 20261030,
    private_seed_hex: str | None = None,
    output_dir: Path,
    writer_kind: str = "mock",
    vram_halt_mib: int = DEFAULT_VRAM_HALT_MIB,
    num_predict: int = DEFAULT_WRITER_NUM_PREDICT,
    max_num_ctx: int = DEFAULT_WRITER_MAX_NUM_CTX,
    grant_id: str | None = None,
    resume: bool = False,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    start_owned_server: bool | None = None,
    http_transport: OllamaHttpTransport | None = None,
    vram_probe: Callable[[], int] | None = None,
    halt_pid: int | None = None,
    pid_killer: Callable[[int], None] = kill_pid_tree,
    popen_factory: Callable[..., Any] | None = None,
) -> dict[str, object]:
    """Generate `count` scenarios, render via `mock` or `ollama` writer with checkpointing, seal, verify, and probe."""
    if writer_kind not in ("mock", "ollama"):
        raise ValueError(f"writer_kind must be 'mock' or 'ollama', got {writer_kind!r}.")

    configs = make_balanced_configs(count, base_seed=base_seed)
    worlds = [generate_world(cfg) for cfg in configs]

    output_dir.mkdir(parents=True, exist_ok=True)
    public_dir = output_dir / "part_public"
    sealed_dir = output_dir / "sealed_store"
    generations_path = output_dir / GENERATIONS_FILENAME

    initial_private_seed = (
        private_seed_hex.strip().lower()
        if private_seed_hex
        else secrets.token_hex(32)
    )
    checkpoint = WriterCheckpointManager(
        output_dir=output_dir,
        count=count,
        base_seed=base_seed,
        private_seed_hex=initial_private_seed,
        writer_kind=writer_kind,
        writer_pin=DEFAULT_WRITER_MODEL_PIN,
    )

    if resume:
        restored_pairs, restored_calls, resolved_private_seed = checkpoint.load_and_verify(
            worlds=worlds
        )
        if private_seed_hex and private_seed_hex.strip().lower() != resolved_private_seed:
            raise ValueError(
                "Explicit --private-seed differs from the checkpoint's sealed private_seed_hex."
            )
    else:
        restored_pairs = []
        restored_calls = []
        resolved_private_seed = initial_private_seed
        if generations_path.exists():
            generations_path.unlink()
        checkpoint.initialize_fresh()

    rendered_pairs: list[tuple[RenderedScenario, RenderedScenario]] = list(restored_pairs)
    completed_count = len(rendered_pairs)

    if writer_kind == "ollama":
        should_own_server = (
            start_owned_server
            if start_owned_server is not None
            else (http_transport is None)
        )
        ollama_writer = OllamaWriter(
            model_pin=DEFAULT_WRITER_MODEL_PIN,
            base_url=ollama_base_url,
            num_predict=num_predict,
            max_num_ctx=max_num_ctx,
            vram_halt_mib=vram_halt_mib,
            grant_id=grant_id,
            http_transport=http_transport,
            vram_probe=vram_probe,
            halt_pid=halt_pid,
            pid_killer=pid_killer,
            generations_path=generations_path,
            start_owned_server=should_own_server,
            popen_factory=popen_factory,
        )
        if resume:
            ollama_writer.restore_call_records(restored_calls)
        try:
            for idx in range(completed_count, len(worlds)):
                w = worlds[idx]
                prev_calls = len(ollama_writer.call_records)
                pair = ollama_writer.render_scenario_pair(w)
                new_calls = ollama_writer.call_records[prev_calls:]
                rendered_pairs.append(pair)
                checkpoint.append_scenario(
                    scenario_index=idx,
                    pair=pair,
                    call_records=new_calls,
                )
        finally:
            ollama_writer.close()
        peak_vram_mib = ollama_writer.peak_vram_mib
        effective_grant_id = ollama_writer.grant_id
        server_info = (
            ollama_writer.owned_server.to_dict()
            if ollama_writer.owned_server is not None
            else None
        )
    else:
        mock_writer = MockWriter()
        all_mock_calls: list[GenerationCallRecord] = list(restored_calls)
        if resume and generations_path is not None:
            lines = "".join(
                json.dumps(r.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
                for r in all_mock_calls
            )
            generations_path.write_text(lines, encoding="utf-8", newline="\n")

        for idx in range(completed_count, len(worlds)):
            w = worlds[idx]
            t0 = time.perf_counter()
            pair = mock_writer.render_scenario_pair(w)
            elapsed = max(0.0, time.perf_counter() - t0)
            scen_calls: list[GenerationCallRecord] = []
            for r_idx, rend in enumerate(pair):
                call_index = len(all_mock_calls) + 1
                call_id = f"{w.scenario_id}:r{r_idx}:call_{call_index:04d}"
                messages = [{"role": "user", "content": rend.writer_prompt}]
                prompt_sha = hashlib.sha256(
                    json.dumps(messages, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                p_est = estimate_prompt_tokens(messages)
                n_ctx = compute_num_ctx(
                    p_est,
                    num_predict,
                    margin=NUM_CTX_MARGIN_TOKENS,
                    step=NUM_CTX_STEP_TOKENS,
                )
                rec = GenerationCallRecord(
                    call_id=call_id,
                    call_index=call_index,
                    scenario_id=w.scenario_id,
                    rendering_index=r_idx,
                    focal_agent=rend.focal_agent,
                    operation="render_scenario_mock",
                    model=DEFAULT_WRITER_MODEL_PIN.ollama_tag,
                    manifest_digest=DEFAULT_WRITER_MODEL_PIN.manifest_digest,
                    model_layer_digest=DEFAULT_WRITER_MODEL_PIN.model_layer_digest,
                    licence_id=DEFAULT_WRITER_MODEL_PIN.licence_id,
                    prompt_sha256=prompt_sha,
                    prompt_tokens_estimate=p_est,
                    prompt_eval_count=p_est,
                    eval_count=len(rend.prose.split()),
                    thinking_words=0,
                    num_predict=num_predict,
                    num_ctx=n_ctx,
                    max_num_ctx=max_num_ctx,
                    think=DEFAULT_WRITER_MODEL_PIN.think,
                    temperature=0.0,
                    done_reason="stop",
                    seconds=round(elapsed / 2.0, 6),
                    vram_peak_mib=0,
                    vram_halt_mib=vram_halt_mib,
                    grant_id="cpu-mock",
                )
                all_mock_calls.append(rec)
                scen_calls.append(rec)
                with generations_path.open("a", encoding="utf-8", newline="\n") as fh:
                    fh.write(json.dumps(rec.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")

            rendered_pairs.append(pair)
            checkpoint.append_scenario(
                scenario_index=idx,
                pair=pair,
                call_records=scen_calls,
            )
        peak_vram_mib = 0
        effective_grant_id = "cpu-mock"
        server_info = None

    receipt = write_sealed_dataset(
        rendered_pairs=rendered_pairs,
        public_dir=public_dir,
        sealed_dir=sealed_dir,
        private_seed_hex=resolved_private_seed,
    )
    recount_verification = verify_written_dataset_recounts_and_prose(public_dir, sealed_dir)

    sealed_rows = [
        json.loads(line)
        for line in receipt.keys_sealed_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    probe_report = run_shortcut_probe(sealed_rows)
    diversity_report = count_template_diversity(worlds)
    d_distribution = _summarize_d_key_patterns(sealed_rows)

    generations_bytes = generations_path.read_bytes() if generations_path.is_file() else b""
    generations_sha256 = hashlib.sha256(generations_bytes).hexdigest()

    summary: dict[str, object] = {
        "writer": writer_kind,
        "writer_pin": DEFAULT_WRITER_MODEL_PIN.to_dict(),
        "grant_id": effective_grant_id,
        "vram_halt_mib": vram_halt_mib,
        "vram_peak_mib": peak_vram_mib,
        "ollama_server": server_info,
        "num_scenarios": receipt.num_scenarios,
        "num_items": receipt.num_items,
        "items_sha256": receipt.items_sha256,
        "part_json_sha256": receipt.part_json_sha256,
        "keys_sealed_sha256": receipt.keys_sealed_sha256,
        "generations_file": GENERATIONS_FILENAME,
        "generations_sha256": generations_sha256,
        "checkpoint_chain_head_sha256": checkpoint.chain_head_sha256,
        "recount_verification": recount_verification,
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
        description="Generate, write, seal, verify, and probe udlab release-1 cognitive-empathy scenarios.",
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
        "--write",
        action="store_true",
        help="Run the scenario writer pass (paired with --writer ollama|mock).",
    )
    parser.add_argument(
        "--writer",
        choices=("ollama", "mock"),
        default="mock",
        help="Writer adapter to use: 'ollama' (pinned qwen3.6:35b) or 'mock' (CPU template, default).",
    )
    parser.add_argument(
        "--vram-halt-mib",
        type=int,
        default=DEFAULT_VRAM_HALT_MIB,
        help=f"VRAM ceiling in MiB above which the owned Ollama server PID tree is killed (default: {DEFAULT_VRAM_HALT_MIB}).",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=DEFAULT_WRITER_NUM_PREDICT,
        help=f"Total num_predict budget per writer call (default: {DEFAULT_WRITER_NUM_PREDICT}).",
    )
    parser.add_argument(
        "--max-num-ctx",
        type=int,
        default=DEFAULT_WRITER_MAX_NUM_CTX,
        help=f"Maximum allowed num_ctx for writer calls (default: {DEFAULT_WRITER_MAX_NUM_CTX}).",
    )
    parser.add_argument(
        "--grant-id",
        type=str,
        default=None,
        help="Publisher GPU grant ID required when running --writer ollama.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted writer build from the verified per-scenario sealed checkpoint.",
    )
    parser.add_argument(
        "--verify-written",
        type=Path,
        default=None,
        help="Run post-write verification (independent B2 derivation, all recounts, prose checks, and --probe-only) on a written output directory.",
    )
    parser.add_argument(
        "--probe-only",
        type=Path,
        default=None,
        help="Run the shortcut probe directly on an existing sealed JSONL file and exit.",
    )
    args = parser.parse_args(argv)

    if args.verify_written is not None:
        verify_res = verify_written_dataset(args.verify_written)
        passed = bool(verify_res["passed"])
        print(json.dumps(verify_res, indent=2, sort_keys=True))
        return 0 if passed else 1

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
        writer_kind=args.writer,
        vram_halt_mib=args.vram_halt_mib,
        num_predict=args.num_predict,
        max_num_ctx=args.max_num_ctx,
        grant_id=args.grant_id,
        resume=args.resume,
    )
    probe_passed = bool(summary["probe_report"]["passed"])  # type: ignore[index]
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if probe_passed else 1


if __name__ == "__main__":
    sys.exit(main())
