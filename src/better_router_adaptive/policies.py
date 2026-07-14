"""Reference routing policies: offline oracle, Better Rules Proxy, fixed arms.

Every policy consumes a long-format dataframe that already carries the
``utility`` column and returns one selected row per prompt. The oracle is the
offline upper bound (it sees realized utilities); the Better Rules Proxy is a
deterministic table fitted on training aggregates only; fixed-arm policies
route every prompt to the same model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

from better_router_adaptive.utility import UTILITY_COLUMN

ORACLE_POLICY: Final = "oracle"
RULES_POLICY: Final = "better-rules-proxy"

_SELECTION_SORT: Final = ["prompt_id", UTILITY_COLUMN, "model_id"]


class PolicyError(ValueError):
    """Raised when a policy cannot produce a complete, deterministic selection."""


@dataclass(frozen=True, slots=True)
class PolicyMetrics:
    """Aggregate outcome of one policy on one set of prompts."""

    policy: str
    prompts: int
    mean_utility: float
    mean_quality: float
    mean_cost_usd: float
    error_rate: float


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise PolicyError(f"missing required columns: {', '.join(missing)}")


def oracle_selection(frame: pd.DataFrame) -> pd.DataFrame:
    """Select, for every prompt, the arm with the highest realized utility.

    Ties are broken by ascending ``model_id`` so the selection is fully
    deterministic. This is an offline upper bound, not a deployable policy.
    """

    _require_columns(frame, ("prompt_id", "model_id", UTILITY_COLUMN))
    ordered = frame.sort_values(_SELECTION_SORT, ascending=[True, False, True], kind="mergesort")
    return ordered.drop_duplicates("prompt_id", keep="first").reset_index(drop=True)


def fit_better_rules_proxy(train_frame: pd.DataFrame) -> dict[str, str]:
    """Fit the deterministic rules table: task group -> best mean-utility arm.

    Only training aggregates are used, mirroring how Better Router applies a
    fixed weighted preference per task category. Ties are broken by ascending
    ``model_id``.
    """

    _require_columns(train_frame, ("task_group", "model_id", UTILITY_COLUMN))
    if train_frame.empty:
        raise PolicyError("cannot fit the rules proxy on an empty training split")

    means = (
        train_frame.groupby(["task_group", "model_id"], sort=True)[UTILITY_COLUMN]
        .mean()
        .reset_index(name="mean_utility")
    )
    ordered = means.sort_values(
        ["task_group", "mean_utility", "model_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    best = ordered.drop_duplicates("task_group", keep="first")
    return {
        str(task_group): str(model_id)
        for task_group, model_id in zip(best["task_group"], best["model_id"], strict=True)
    }


def apply_rules_policy(frame: pd.DataFrame, rules: dict[str, str]) -> pd.DataFrame:
    """Route every prompt to the arm chosen by its task group rule."""

    _require_columns(frame, ("prompt_id", "task_group", "model_id", UTILITY_COLUMN))
    unknown = sorted(set(frame["task_group"].astype(str)) - set(rules))
    if unknown:
        raise PolicyError(f"no rule for task groups: {', '.join(unknown)}")

    chosen_arm = frame["task_group"].astype(str).map(rules)
    selection = frame.loc[frame["model_id"].astype(str) == chosen_arm].copy()

    expected_prompts = frame["prompt_id"].nunique()
    if selection["prompt_id"].nunique() != expected_prompts:
        raise PolicyError("rules policy could not select an arm for every prompt")
    return selection.sort_values("prompt_id", kind="mergesort").reset_index(drop=True)


def fixed_arm_selection(frame: pd.DataFrame, model_id: str) -> pd.DataFrame:
    """Route every prompt to one fixed arm (reference policy)."""

    _require_columns(frame, ("prompt_id", "model_id", UTILITY_COLUMN))
    selection = frame.loc[frame["model_id"].astype(str) == model_id].copy()
    if selection["prompt_id"].nunique() != frame["prompt_id"].nunique():
        raise PolicyError(f"arm {model_id!r} is missing for some prompts")
    return selection.sort_values("prompt_id", kind="mergesort").reset_index(drop=True)


def evaluate_selection(selection: pd.DataFrame, *, policy: str) -> PolicyMetrics:
    """Summarize one per-prompt selection into auditable aggregate metrics."""

    _require_columns(selection, ("prompt_id", UTILITY_COLUMN, "quality", "cost_usd", "success"))
    if selection.empty:
        raise PolicyError(f"policy {policy!r} produced an empty selection")
    if bool(selection["prompt_id"].duplicated().any()):
        raise PolicyError(f"policy {policy!r} selected more than one arm for a prompt")

    return PolicyMetrics(
        policy=policy,
        prompts=int(selection["prompt_id"].nunique()),
        mean_utility=float(selection[UTILITY_COLUMN].mean()),
        mean_quality=float(pd.to_numeric(selection["quality"]).mean()),
        mean_cost_usd=float(pd.to_numeric(selection["cost_usd"]).mean()),
        error_rate=float((~selection["success"].astype(bool)).mean()),
    )
