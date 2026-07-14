from __future__ import annotations

import pandas as pd
import pytest

from better_router_adaptive.policies import (
    PolicyError,
    apply_rules_policy,
    evaluate_selection,
    fit_better_rules_proxy,
    fixed_arm_selection,
    oracle_selection,
)

MODEL_ARMS = ("arm-fast", "arm-balanced", "arm-reasoning", "arm-premium")


def _utility_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    utilities = {
        ("p-1", "arm-fast"): 0.30,
        ("p-1", "arm-balanced"): 0.55,
        ("p-1", "arm-reasoning"): 0.50,
        ("p-1", "arm-premium"): 0.40,
        ("p-2", "arm-fast"): 0.60,
        ("p-2", "arm-balanced"): 0.20,
        ("p-2", "arm-reasoning"): 0.35,
        ("p-2", "arm-premium"): 0.45,
    }
    for (prompt_id, model_id), utility in utilities.items():
        task_group = "coding" if prompt_id == "p-1" else "mathematics"
        rows.append(
            {
                "prompt_id": prompt_id,
                "task_group": task_group,
                "model_id": model_id,
                "quality": 0.7,
                "cost_usd": 0.001,
                "success": True,
                "utility": utility,
            }
        )
    return pd.DataFrame(rows)


def test_oracle_selects_the_highest_utility_arm_per_prompt() -> None:
    selection = oracle_selection(_utility_frame())

    chosen = dict(zip(selection["prompt_id"], selection["model_id"], strict=True))
    assert chosen == {"p-1": "arm-balanced", "p-2": "arm-fast"}


def test_oracle_breaks_utility_ties_deterministically_by_model_id() -> None:
    frame = _utility_frame()
    frame.loc[(frame["prompt_id"] == "p-1") & (frame["model_id"] == "arm-reasoning"), "utility"] = (
        0.55
    )

    selection = oracle_selection(frame)

    chosen = dict(zip(selection["prompt_id"], selection["model_id"], strict=True))
    assert chosen["p-1"] == "arm-balanced"


def test_oracle_dominates_every_other_policy() -> None:
    frame = _utility_frame()
    oracle = evaluate_selection(oracle_selection(frame), policy="oracle")
    for arm in MODEL_ARMS:
        fixed = evaluate_selection(fixed_arm_selection(frame, arm), policy=f"fixed:{arm}")
        assert oracle.mean_utility >= fixed.mean_utility


def test_rules_proxy_maps_each_task_group_to_the_best_mean_arm() -> None:
    rules = fit_better_rules_proxy(_utility_frame())

    assert rules == {"coding": "arm-balanced", "mathematics": "arm-fast"}


def test_rules_proxy_breaks_mean_utility_ties_by_model_id() -> None:
    frame = _utility_frame()
    frame.loc[
        (frame["task_group"] == "coding") & (frame["model_id"] == "arm-reasoning"), "utility"
    ] = 0.55

    rules = fit_better_rules_proxy(frame)

    assert rules["coding"] == "arm-balanced"


def test_apply_rules_policy_selects_one_row_per_prompt() -> None:
    frame = _utility_frame()
    rules = fit_better_rules_proxy(frame)

    selection = apply_rules_policy(frame, rules)

    assert len(selection) == frame["prompt_id"].nunique()
    chosen = dict(zip(selection["prompt_id"], selection["model_id"], strict=True))
    assert chosen == {"p-1": "arm-balanced", "p-2": "arm-fast"}


def test_apply_rules_policy_rejects_task_groups_without_rules() -> None:
    with pytest.raises(PolicyError, match="no rule for task groups"):
        apply_rules_policy(_utility_frame(), {"coding": "arm-fast"})


def test_fixed_arm_selection_requires_the_arm_for_every_prompt() -> None:
    frame = _utility_frame()
    incomplete = frame[~((frame["prompt_id"] == "p-2") & (frame["model_id"] == "arm-premium"))]

    with pytest.raises(PolicyError, match="missing for some prompts"):
        fixed_arm_selection(incomplete, "arm-premium")


def test_evaluate_selection_summarizes_utility_quality_cost_and_errors() -> None:
    frame = _utility_frame()
    frame.loc[frame["prompt_id"] == "p-2", "success"] = False

    metrics = evaluate_selection(oracle_selection(frame), policy="oracle")

    assert metrics.policy == "oracle"
    assert metrics.prompts == 2
    assert metrics.mean_utility == pytest.approx((0.55 + 0.60) / 2)
    assert metrics.mean_quality == pytest.approx(0.7)
    assert metrics.error_rate == pytest.approx(0.5)


def test_evaluate_selection_rejects_duplicated_prompts() -> None:
    frame = _utility_frame()

    with pytest.raises(PolicyError, match="more than one arm"):
        evaluate_selection(frame, policy="broken")
