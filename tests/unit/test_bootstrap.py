from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from better_router_adaptive.evaluate import paired_bootstrap_summary
from better_router_adaptive.routers import RouterError


def _prompt_table() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 200
    baseline = rng.normal(0.5, 0.05, size=n)
    return pd.DataFrame(
        {
            "better-rules-proxy": baseline,
            "constant-better": baseline + 0.1,
            "constant-equal": baseline,
        },
        index=pd.Index([f"p-{i}" for i in range(n)], name="prompt_id"),
    )


def test_bootstrap_recovers_a_constant_paired_difference_exactly() -> None:
    table = _prompt_table()

    summary = paired_bootstrap_summary(
        table, baseline="better-rules-proxy", samples=500, rng=np.random.default_rng(42)
    ).set_index("policy")

    better = summary.loc["constant-better"]
    assert better["diff_vs_baseline"] == pytest.approx(0.1)
    # A constant +0.1 shift is invariant under paired resampling.
    assert better["diff_ci95_low"] == pytest.approx(0.1)
    assert better["diff_ci95_high"] == pytest.approx(0.1)

    equal = summary.loc["constant-equal"]
    assert equal["diff_vs_baseline"] == pytest.approx(0.0)
    assert equal["diff_ci95_low"] == pytest.approx(0.0)
    assert equal["diff_ci95_high"] == pytest.approx(0.0)


def test_bootstrap_confidence_interval_contains_the_point_estimate() -> None:
    table = _prompt_table()

    summary = paired_bootstrap_summary(
        table, baseline="better-rules-proxy", samples=500, rng=np.random.default_rng(42)
    )

    for record in summary.to_dict(orient="records"):
        assert record["ci95_low"] <= record["mean_utility"] <= record["ci95_high"]
        assert record["ci95_low"] < record["ci95_high"]


def test_bootstrap_is_deterministic_for_the_same_generator_seed() -> None:
    table = _prompt_table()

    first = paired_bootstrap_summary(
        table, baseline="better-rules-proxy", samples=200, rng=np.random.default_rng(1)
    )
    second = paired_bootstrap_summary(
        table, baseline="better-rules-proxy", samples=200, rng=np.random.default_rng(1)
    )

    pd.testing.assert_frame_equal(first, second)


def test_bootstrap_rejects_missing_baseline_and_empty_tables() -> None:
    table = _prompt_table()

    with pytest.raises(RouterError, match="baseline policy"):
        paired_bootstrap_summary(
            table, baseline="missing", samples=10, rng=np.random.default_rng(0)
        )
    with pytest.raises(RouterError, match="empty utilities table"):
        paired_bootstrap_summary(
            table.iloc[:0],
            baseline="better-rules-proxy",
            samples=10,
            rng=np.random.default_rng(0),
        )
