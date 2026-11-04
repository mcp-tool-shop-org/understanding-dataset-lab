"""Shared pytest configuration: the `slow` marker.

Each test listed here runs the full mock Stage 5 pipeline (or the registered Gate 0 self-test) through the real
prose checker, and takes 44 s to 11 min on one core. Their time goes to the checker itself, not to sleeps or
waits, so mocking would stop them testing what they test. CI runs `-m "not slow"` on pull requests and the full
suite on every push to main and on manual dispatch.
"""

from __future__ import annotations

import pytest

SLOW_TESTS: frozenset[str] = frozenset(
    {
        # The registered Gate 0 self-test: 270 scenarios x 2 renderings.
        "tests/test_stage5_templates.py::test_gate0_template_self_test_270_generated_scenarios",
        # Full mock Stage 5 runs over a seed bank.
        "tests/test_stage5_pipeline.py::test_1_full_mock_run_schema_and_summary",
        "tests/test_stage5_pipeline.py::test_2_s_ge_b_stop_and_resume",
        "tests/test_stage5_pipeline.py::test_3_s_passes_where_b_failed",
        "tests/test_stage5_pipeline.py::test_4_context_cap_and_4a_cap_decision_per_rendering",
        "tests/test_stage5_pipeline.py::test_5_plan_seed_draws_write_order_and_resume_refusal",
        "tests/test_stage5_pipeline.py::test_6_rewrite_input_rung_sample_rule_boundary_and_short_refusal",
        "tests/test_stage5_pipeline.py::test_8_no_prose_in_record_stdout_or_stderr",
        "tests/test_stage5_pipeline.py::test_11_seed_bank_input_matches_paired_seed_bank_and_dry",
        "tests/test_stage5_pipeline.py::test_13_addendum2_item2_both_seeds_drawn_at_run_start_and_zero_on_resume",
        "tests/test_stage5_pipeline.py::test_14_addendum2_item3_resume_after_s_lt_b_refuses_until_reset",
        "tests/test_stage5_pipeline.py::test_15_addendum2_item4_redo_b_sampled_by_rung_refuses_unless_redo_rung",
        "tests/test_stage5_pipeline.py::test_16_addendum2_item5_over_cap_excluded_from_rung_and_zero_reads_on_resume",
        "tests/test_stage5_pipeline.py::test_17_addendum2_item6_mid_rung_crash_call_45_and_resume",
        "tests/test_stage5_pipeline.py::test_18_addendum2_item7_canonical_rung_pool_and_mode_order_invariance",
        "tests/test_stage5_pipeline.py::test_19_addendum2_item8_rewrite_input_enum_and_canary_test",
        "tests/test_stage5_pipeline.py::test_24_addendum2_item13_rung_runs_on_post_swap_scene",
        "tests/test_stage5_pipeline.py::test_28_owned_server_single_url_closed_once_and_model_endpoint",
        "tests/test_stage5_pipeline.py::test_29_owned_server_closed_once_on_mid_rewrite_error",
        "tests/test_stage5_pipeline.py::test_30_step_timings_in_record_and_mirrored_in_status_with_fake_clock",
    }
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: runs the full mock Stage 5 pipeline; CI runs it on main and dispatch")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    collected = {item.nodeid for item in items}
    for item in items:
        if item.nodeid in SLOW_TESTS:
            item.add_marker(pytest.mark.slow)
    # A renamed or removed slow test must not drop silently out of the list: when the whole suite is collected
    # (no node ids, files or -k given), every listed test must be found.
    whole_suite = not config.option.keyword and all("::" not in arg and not arg.endswith(".py") for arg in config.args)
    missing = sorted(SLOW_TESTS - collected)
    if whole_suite and missing:
        raise pytest.UsageError(f"tests/conftest.py SLOW_TESTS lists tests that were not collected: {missing}")
