from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.data_access import PublicResultsError, load_public_results


def _write_valid_results(root: Path) -> Path:
    base = root / "artifacts" / "public" / "step7-real"
    base.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "policy": "xgboost",
                "prompts": 20,
                "mean_utility": 0.49,
                "ci95_low": 0.48,
                "ci95_high": 0.50,
                "diff_vs_baseline": 0.0,
                "diff_ci95_low": -0.01,
                "diff_ci95_high": 0.01,
            },
            {
                "policy": "better-rules-proxy",
                "prompts": 20,
                "mean_utility": 0.48,
                "ci95_low": 0.47,
                "ci95_high": 0.49,
                "diff_vs_baseline": 0.0,
                "diff_ci95_low": 0.0,
                "diff_ci95_high": 0.0,
            },
        ]
    ).to_csv(base / "evaluation_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "seed": 123,
                "split": "test",
                "selected_xgboost": '{"max_depth": 2}',
                "selected_linucb_alpha": 0.5,
                "policy": "xgboost",
                "prompts": 10,
                "mean_utility": 0.49,
                "mean_quality": 0.78,
                "mean_cost_usd": 0.003,
                "error_rate": 0.0,
            },
            {
                "seed": 42,
                "split": "test",
                "selected_xgboost": '{"max_depth": 2}',
                "selected_linucb_alpha": 0.5,
                "policy": "better-rules-proxy",
                "prompts": 10,
                "mean_utility": 0.48,
                "mean_quality": 0.77,
                "mean_cost_usd": 0.003,
                "error_rate": 0.0,
            },
        ]
    ).to_csv(base / "evaluation_per_seed.csv", index=False)
    return root


def test_load_public_results_returns_sorted_copies(tmp_path: Path) -> None:
    root = _write_valid_results(tmp_path)

    results = load_public_results(root)

    assert list(results.summary["policy"]) == ["better-rules-proxy", "xgboost"]
    assert list(results.per_seed["seed"]) == [42, 123]
    results.summary.loc[0, "policy"] = "mutated"
    reloaded = load_public_results(root)
    assert list(reloaded.summary["policy"]) == ["better-rules-proxy", "xgboost"]


def test_load_public_results_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(PublicResultsError, match=r"evaluation_summary\.csv"):
        load_public_results(tmp_path)


def test_load_public_results_rejects_missing_columns(tmp_path: Path) -> None:
    root = _write_valid_results(tmp_path)
    base = root / "artifacts" / "public" / "step7-real"
    pd.DataFrame({"policy": ["xgboost"]}).to_csv(base / "evaluation_summary.csv", index=False)

    with pytest.raises(PublicResultsError, match="missing columns"):
        load_public_results(root)


def test_load_public_results_rejects_empty_csv(tmp_path: Path) -> None:
    root = _write_valid_results(tmp_path)
    base = root / "artifacts" / "public" / "step7-real"
    pd.DataFrame(
        columns=[
            "policy",
            "prompts",
            "mean_utility",
            "ci95_low",
            "ci95_high",
            "diff_vs_baseline",
            "diff_ci95_low",
            "diff_ci95_high",
        ]
    ).to_csv(base / "evaluation_summary.csv", index=False)

    with pytest.raises(PublicResultsError, match="contains no rows"):
        load_public_results(root)
