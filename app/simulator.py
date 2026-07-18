"""Didactic routing simulator backed by the project's real router classes.

The simulator intentionally trains on the original synthetic test fixture. It
is not experimental evidence and must not be interpreted as a RouterBench or
production routing result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
import pandas as pd

from better_router_adaptive.config import load_experiment_config
from better_router_adaptive.features import build_prompt_features, feature_column_names
from better_router_adaptive.routers import (
    FeatureScaler,
    LinUCBRouter,
    XGBoostConfig,
    XGBoostRouter,
)
from better_router_adaptive.utility import compute_normalization_stats, compute_utility

_MIN_PROMPT_LENGTH: Final = 5
_MAX_PROMPT_LENGTH: Final = 2_000
_REQUIRED_FIXTURE_COLUMNS: Final = (
    "prompt_id",
    "prompt_text",
    "task_group",
    "model_id",
    "quality",
    "cost_usd",
    "latency_ms",
    "success",
)
_ARM_LABELS: Final = {
    "arm-fast": "Rápido",
    "arm-balanced": "Equilibrado",
    "arm-reasoning": "Razonamiento",
    "arm-premium": "Premium",
}


class SimulationError(ValueError):
    """Raised when the didactic simulator cannot train or route safely."""


@dataclass(slots=True)
class DemoBundle:
    """Trained in-memory routers and preprocessing state for the synthetic demo."""

    task_groups: tuple[str, ...]
    arms: tuple[str, ...]
    scaler: FeatureScaler
    xgboost: XGBoostRouter
    linucb: LinUCBRouter


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """One deterministic didactic routing result for a user-entered prompt."""

    xgboost_arm: str
    linucb_arm: str
    prompt_char_count: int
    prompt_word_count: int
    prompt_avg_word_length: float


def display_arm(arm: str) -> str:
    """Return a didactic display name without implying a production model."""

    return _ARM_LABELS.get(arm, arm)


def _load_fixture(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise SimulationError(f"synthetic fixture not found: {path}") from exc
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise SimulationError(f"could not read synthetic fixture: {exc}") from exc

    missing = [column for column in _REQUIRED_FIXTURE_COLUMNS if column not in frame.columns]
    if missing:
        raise SimulationError(f"synthetic fixture is missing columns: {', '.join(missing)}")
    if frame.empty:
        raise SimulationError("synthetic fixture contains no rows")

    success = pd.to_numeric(frame["success"], errors="coerce")
    if bool(success.isna().any()) or not bool(success.isin([0, 1]).all()):
        raise SimulationError("synthetic fixture success values must be 0 or 1")
    result = frame.copy()
    result["success"] = success.astype(bool)
    return result


def train_demo_bundle(root: Path) -> DemoBundle:
    """Train both real router implementations on the original synthetic fixture."""

    config = load_experiment_config(root / "config" / "experiment.yaml")
    frame = _load_fixture(root / "tests" / "fixtures" / "routerbench_sample.csv")

    arms = tuple(frame["model_id"].astype(str).drop_duplicates().tolist())
    if len(arms) != 4 or len(set(arms)) != 4:
        raise SimulationError("synthetic fixture must contain exactly four unique arms")

    prompt_features = build_prompt_features(frame, task_groups=config.task_groups)
    feature_columns = feature_column_names(config.task_groups)
    scaler = FeatureScaler.fit(prompt_features, feature_columns)
    contexts = scaler.transform(prompt_features)

    normalization = compute_normalization_stats(frame)
    utility_frame = compute_utility(
        frame,
        weights=config.reward_weights,
        stats=normalization,
    )
    utility_table = utility_frame.pivot(
        index="prompt_id",
        columns="model_id",
        values="utility",
    ).reindex(index=prompt_features["prompt_id"].tolist(), columns=list(arms))
    if bool(utility_table.isna().any(axis=None)):
        raise SimulationError("synthetic fixture does not contain every arm for every prompt")
    utilities = utility_table.to_numpy(dtype=np.float64)

    xgboost = XGBoostRouter(
        arms,
        config=XGBoostConfig(max_depth=2, n_estimators=50, learning_rate=0.1),
        seed=42,
    )
    xgboost.fit(contexts, utilities)

    linucb = LinUCBRouter(arms, alpha=0.5)
    linucb.replay(contexts, utilities)

    return DemoBundle(
        task_groups=config.task_groups,
        arms=arms,
        scaler=scaler,
        xgboost=xgboost,
        linucb=linucb,
    )


def simulate_prompt(
    bundle: DemoBundle,
    *,
    prompt_text: str,
    task_group: str,
) -> SimulationResult:
    """Route one prompt with both routers using only pre-inference features."""

    cleaned = prompt_text.strip()
    if not _MIN_PROMPT_LENGTH <= len(cleaned) <= _MAX_PROMPT_LENGTH:
        raise SimulationError("prompt length must be between 5 and 2000 characters")
    if task_group not in bundle.task_groups:
        raise SimulationError(f"unknown task group: {task_group}")

    input_frame = pd.DataFrame(
        [
            {
                "prompt_id": "streamlit-user-query",
                "prompt_text": cleaned,
                "task_group": task_group,
            }
        ]
    )
    features = build_prompt_features(input_frame, task_groups=bundle.task_groups)
    contexts = bundle.scaler.transform(features)
    char_count = cast(int | float, features.at[0, "prompt_char_count"])
    word_count = cast(int | float, features.at[0, "prompt_word_count"])
    average_word_length = cast(int | float, features.at[0, "prompt_avg_word_length"])

    return SimulationResult(
        xgboost_arm=bundle.xgboost.route(contexts)[0],
        linucb_arm=bundle.linucb.route(contexts)[0],
        prompt_char_count=int(char_count),
        prompt_word_count=int(word_count),
        prompt_avg_word_length=float(average_word_length),
    )
