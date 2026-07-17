"""Validated loading of the public aggregate research results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

SUMMARY_COLUMNS: Final = (
    "policy",
    "prompts",
    "mean_utility",
    "ci95_low",
    "ci95_high",
    "diff_vs_baseline",
    "diff_ci95_low",
    "diff_ci95_high",
)
PER_SEED_COLUMNS: Final = (
    "seed",
    "split",
    "selected_xgboost",
    "selected_linucb_alpha",
    "policy",
    "prompts",
    "mean_utility",
    "mean_quality",
    "mean_cost_usd",
    "error_rate",
)


class PublicResultsError(ValueError):
    """Raised when public aggregate result files are unavailable or invalid."""


@dataclass(frozen=True, slots=True)
class PublicResults:
    """Validated public aggregate results used by the Streamlit dashboard."""

    summary: pd.DataFrame
    per_seed: pd.DataFrame


def _read_required_csv(path: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise PublicResultsError(f"required result file not found: {path.name}") from exc
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise PublicResultsError(f"could not read result file {path.name}: {exc}") from exc

    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise PublicResultsError(
            f"result file {path.name} is missing columns: {', '.join(missing)}"
        )
    if frame.empty:
        raise PublicResultsError(f"result file {path.name} contains no rows")
    return frame.loc[:, list(required_columns)].copy()


def load_public_results(root: Path) -> PublicResults:
    """Load and validate aggregate Step 7 result files from ``root``."""

    base = root / "artifacts" / "public" / "step7-real"
    summary = _read_required_csv(base / "evaluation_summary.csv", SUMMARY_COLUMNS)
    per_seed = _read_required_csv(base / "evaluation_per_seed.csv", PER_SEED_COLUMNS)
    return PublicResults(
        summary=summary.sort_values("policy", kind="mergesort").reset_index(drop=True),
        per_seed=per_seed.sort_values(["seed", "policy"], kind="mergesort").reset_index(drop=True),
    )
