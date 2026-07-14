"""Train-anchored utility computation for routing policies.

The utility of one (prompt, model) outcome is::

    U = w_quality * quality
        - w_cost * normalized_cost
        - w_latency * normalized_latency
        - w_error * error_indicator

Cost and latency are normalized with min-max statistics computed **only on the
training split**, then applied unchanged to validation and test. Values that
fall outside the training range are clipped to [0, 1], so evaluation splits
can never influence the scale they are judged with.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Final

import pandas as pd

from better_router_adaptive.config import RewardWeights

NORMALIZED_COLUMNS: Final = ("cost_usd", "latency_ms")
UTILITY_COLUMN: Final = "utility"


class UtilityError(ValueError):
    """Raised when utility terms cannot be computed without silent repairs."""


@dataclass(frozen=True, slots=True)
class ColumnNormalization:
    """Min-max anchors for one cost-like column, fitted on the training split.

    ``enabled`` is ``False`` when the column is entirely null in training, in
    which case the corresponding utility term is disabled explicitly instead
    of being imputed.
    """

    enabled: bool
    minimum: float | None
    maximum: float | None


@dataclass(frozen=True, slots=True)
class NormalizationStats:
    """Train-split normalization anchors for every cost-like column."""

    cost_usd: ColumnNormalization
    latency_ms: ColumnNormalization

    def column(self, name: str) -> ColumnNormalization:
        if name == "cost_usd":
            return self.cost_usd
        if name == "latency_ms":
            return self.latency_ms
        raise UtilityError(f"unknown normalized column: {name}")

    def as_dict(self) -> dict[str, dict[str, object]]:
        return {"cost_usd": asdict(self.cost_usd), "latency_ms": asdict(self.latency_ms)}


def _fit_column(values: pd.Series[Any]) -> ColumnNormalization:
    numeric = pd.to_numeric(values, errors="coerce")
    non_null = numeric.dropna()
    if non_null.empty:
        return ColumnNormalization(enabled=False, minimum=None, maximum=None)
    if len(non_null) != len(numeric):
        raise UtilityError(
            "training split mixes null and non-null values; refusing silent imputation"
        )
    return ColumnNormalization(
        enabled=True, minimum=float(non_null.min()), maximum=float(non_null.max())
    )


def compute_normalization_stats(train_frame: pd.DataFrame) -> NormalizationStats:
    """Fit min-max anchors for cost and latency using the training split only."""

    if train_frame.empty:
        raise UtilityError("cannot fit normalization statistics on an empty training split")
    missing = [column for column in NORMALIZED_COLUMNS if column not in train_frame.columns]
    if missing:
        raise UtilityError(f"missing columns for normalization: {', '.join(missing)}")
    return NormalizationStats(
        cost_usd=_fit_column(train_frame["cost_usd"]),
        latency_ms=_fit_column(train_frame["latency_ms"]),
    )


def _normalize_column(
    frame: pd.DataFrame, column: str, anchors: ColumnNormalization
) -> pd.Series[float]:
    values = pd.to_numeric(frame[column], errors="coerce")
    if not anchors.enabled:
        if bool(values.notna().any()):
            raise UtilityError(
                f"{column} was disabled by the training split but contains values here"
            )
        return pd.Series(0.0, index=frame.index, name=f"{column}_norm")
    if bool(values.isna().any()):
        raise UtilityError(f"{column} contains null values; refusing silent imputation")
    assert anchors.minimum is not None and anchors.maximum is not None
    span = anchors.maximum - anchors.minimum
    if span == 0:
        normalized = pd.Series(0.0, index=frame.index)
    else:
        normalized = ((values - anchors.minimum) / span).clip(lower=0.0, upper=1.0)
    return normalized.rename(f"{column}_norm")


def compute_utility(
    frame: pd.DataFrame, *, weights: RewardWeights, stats: NormalizationStats
) -> pd.DataFrame:
    """Return a copy of ``frame`` with normalized terms and the utility column."""

    required = ("quality", "success", *NORMALIZED_COLUMNS)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise UtilityError(f"missing columns for utility: {', '.join(missing)}")

    result = frame.copy()
    quality = pd.to_numeric(result["quality"], errors="coerce")
    if bool(quality.isna().any()):
        raise UtilityError("quality contains null values")

    success = result["success"]
    if success.dtype != bool:
        raise UtilityError("success must be boolean before computing utility")
    error_indicator = (~success).astype(float)

    cost_norm = _normalize_column(result, "cost_usd", stats.cost_usd)
    latency_norm = _normalize_column(result, "latency_ms", stats.latency_ms)

    result["cost_usd_norm"] = cost_norm
    result["latency_ms_norm"] = latency_norm
    result["error_indicator"] = error_indicator
    result[UTILITY_COLUMN] = (
        weights.quality * quality
        - weights.cost * cost_norm
        - weights.latency * latency_norm
        - weights.error * error_indicator
    )
    return result
