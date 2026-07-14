from __future__ import annotations

import pandas as pd
import pytest

from better_router_adaptive.config import RewardWeights
from better_router_adaptive.utility import (
    UTILITY_COLUMN,
    UtilityError,
    compute_normalization_stats,
    compute_utility,
)

WEIGHTS = RewardWeights(quality=0.65, cost=0.20, latency=0.10, error=0.05)


def _train_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "quality": [0.2, 0.8, 0.5, 1.0],
            "cost_usd": [0.001, 0.005, 0.003, 0.009],
            "latency_ms": [100.0, 300.0, 200.0, 500.0],
            "success": [True, True, False, True],
        }
    )


def test_normalization_stats_use_min_and_max_of_training_split_only() -> None:
    stats = compute_normalization_stats(_train_frame())

    assert stats.cost_usd.enabled
    assert stats.cost_usd.minimum == pytest.approx(0.001)
    assert stats.cost_usd.maximum == pytest.approx(0.009)
    assert stats.latency_ms.minimum == pytest.approx(100.0)
    assert stats.latency_ms.maximum == pytest.approx(500.0)


def test_out_of_range_evaluation_values_are_clipped_not_refitted() -> None:
    stats = compute_normalization_stats(_train_frame())
    evaluation = pd.DataFrame(
        {
            "quality": [1.0, 1.0],
            "cost_usd": [0.000001, 99.0],
            "latency_ms": [1.0, 9999.0],
            "success": [True, True],
        }
    )

    result = compute_utility(evaluation, weights=WEIGHTS, stats=stats)

    assert result["cost_usd_norm"].tolist() == [0.0, 1.0]
    assert result["latency_ms_norm"].tolist() == [0.0, 1.0]


def test_utility_formula_matches_the_locked_definition() -> None:
    stats = compute_normalization_stats(_train_frame())
    frame = pd.DataFrame(
        {
            "quality": [0.8],
            "cost_usd": [0.005],
            "latency_ms": [300.0],
            "success": [False],
        }
    )

    result = compute_utility(frame, weights=WEIGHTS, stats=stats)

    cost_norm = (0.005 - 0.001) / (0.009 - 0.001)
    latency_norm = (300.0 - 100.0) / (500.0 - 100.0)
    expected = 0.65 * 0.8 - 0.20 * cost_norm - 0.10 * latency_norm - 0.05 * 1.0
    assert result[UTILITY_COLUMN].iloc[0] == pytest.approx(expected)


def test_fully_null_training_column_disables_the_term_explicitly() -> None:
    train = _train_frame().assign(latency_ms=float("nan"))
    stats = compute_normalization_stats(train)

    assert not stats.latency_ms.enabled

    result = compute_utility(train, weights=WEIGHTS, stats=stats)
    assert (result["latency_ms_norm"] == 0.0).all()


def test_partially_null_training_column_is_rejected() -> None:
    train = _train_frame()
    train.loc[0, "latency_ms"] = None

    with pytest.raises(UtilityError, match="refusing silent imputation"):
        compute_normalization_stats(train)


def test_disabled_column_with_values_at_evaluation_time_is_rejected() -> None:
    stats = compute_normalization_stats(_train_frame().assign(latency_ms=float("nan")))

    with pytest.raises(UtilityError, match="disabled by the training split"):
        compute_utility(_train_frame(), weights=WEIGHTS, stats=stats)


def test_constant_training_column_normalizes_to_zero() -> None:
    train = _train_frame().assign(cost_usd=0.002)
    stats = compute_normalization_stats(train)

    result = compute_utility(train, weights=WEIGHTS, stats=stats)

    assert (result["cost_usd_norm"] == 0.0).all()


def test_error_indicator_comes_from_the_success_flag() -> None:
    stats = compute_normalization_stats(_train_frame())

    result = compute_utility(_train_frame(), weights=WEIGHTS, stats=stats)

    assert result["error_indicator"].tolist() == [0.0, 0.0, 1.0, 0.0]


def test_non_boolean_success_is_rejected() -> None:
    stats = compute_normalization_stats(_train_frame())
    frame = _train_frame().assign(success=pd.Series(["yes", "no", "yes", "no"]))

    with pytest.raises(UtilityError, match="success must be boolean"):
        compute_utility(frame, weights=WEIGHTS, stats=stats)


def test_empty_training_split_is_rejected() -> None:
    with pytest.raises(UtilityError, match="empty training split"):
        compute_normalization_stats(_train_frame().iloc[:0])
