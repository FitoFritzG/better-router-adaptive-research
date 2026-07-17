"""Cascade action space for the EvoCascade router (Step 6 extension).

A cascade action calls a cheap arm first and escalates to a strictly more
expensive arm only when the first answer is rejected. The design adapts two
ideas from Sakana AI's model-coordination papers to offline replay routing:
the recursive "retry with a stronger setup" topologies of the RL Conductor
(arXiv:2512.04388) and the verifier role of Trinity (arXiv:2512.04695).

Escalation trigger (documented modelling assumption): the verifier is
simulated as *ideal* during replay — it rejects the first answer when the
call failed (``success = False``) or scored zero quality on the benchmark
(``quality == 0``). In deployment this corresponds to an automatic check
(unit tests, format validation, a lightweight judge), which RouterBench
cannot execute; the simulated verifier is therefore an explicit upper bound
on verifier accuracy, never a deployable signal. Quality values above zero
are never observed by the policy, so pre-inference feature guarantees from
Step 4 remain intact: the learned weights only ever see prompt features.

Accounting is pessimistic and train-anchored: an escalated action pays the
cost and latency of *both* calls (re-normalized with the Step 5 train
anchors, clipped to [0, 1]) and inherits quality, success, and the error
indicator from the second call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from better_router_adaptive.config import RewardWeights
from better_router_adaptive.utility import UTILITY_COLUMN, ColumnNormalization, NormalizationStats

_FloatArray = npt.NDArray[np.float64]
_BoolArray = npt.NDArray[np.bool_]

CASCADE_SEPARATOR: Final = ">"


class CascadeError(ValueError):
    """Raised when a cascade action space or its outcomes cannot be built."""


@dataclass(frozen=True, slots=True)
class ActionSpace:
    """Single-arm actions plus escalation pairs ``(first, second)`` by index.

    The first ``len(arms)`` actions route the prompt to one arm; every pair
    ``(i, j)`` appends the cascade "call ``arms[i]``, escalate to ``arms[j]``
    on rejection". Pairs only escalate towards strictly more expensive arms
    (by mean training cost), so the space stays small and interpretable.
    """

    arms: tuple[str, ...]
    pairs: tuple[tuple[int, int], ...]

    @property
    def labels(self) -> tuple[str, ...]:
        cascades = tuple(
            f"{self.arms[first]}{CASCADE_SEPARATOR}{self.arms[second]}"
            for first, second in self.pairs
        )
        return self.arms + cascades

    @property
    def size(self) -> int:
        return len(self.arms) + len(self.pairs)


def build_action_space(arms: tuple[str, ...], mean_train_cost: _FloatArray) -> ActionSpace:
    """Build the action space; escalation targets must cost strictly more.

    ``mean_train_cost`` holds the mean training-split cost of every arm in
    the same order as ``arms``; only training aggregates are used, so the
    action space itself cannot leak evaluation information.
    """

    if not arms or len(arms) != len(set(arms)):
        raise CascadeError("arms must be non-empty and unique")
    if mean_train_cost.shape != (len(arms),):
        raise CascadeError("mean_train_cost must hold one value per arm")
    pairs = tuple(
        (first, second)
        for first in range(len(arms))
        for second in range(len(arms))
        if first != second and mean_train_cost[second] > mean_train_cost[first]
    )
    return ActionSpace(arms=arms, pairs=pairs)


@dataclass(frozen=True, slots=True)
class ActionOutcomes:
    """Realized per-prompt outcomes for every action, aligned row by row."""

    labels: tuple[str, ...]
    quality: _FloatArray
    cost_usd: _FloatArray
    success: _BoolArray
    utility: _FloatArray
    escalated: _BoolArray

    def restrict(self, count: int) -> ActionOutcomes:
        """Return a view limited to the first ``count`` actions (arms first)."""

        if not 0 < count <= len(self.labels):
            raise CascadeError(f"cannot restrict {len(self.labels)} actions to {count}")
        return ActionOutcomes(
            labels=self.labels[:count],
            quality=self.quality[:, :count],
            cost_usd=self.cost_usd[:, :count],
            success=self.success[:, :count],
            utility=self.utility[:, :count],
            escalated=self.escalated[:, :count],
        )


def _normalized_term(values: _FloatArray, anchors: ColumnNormalization, column: str) -> _FloatArray:
    if not anchors.enabled:
        if bool(np.isfinite(values).any()):
            raise CascadeError(
                f"{column} was disabled by the training split but contains values here"
            )
        return np.zeros_like(values)
    if bool(np.isnan(values).any()):
        raise CascadeError(f"{column} contains null values; refusing silent imputation")
    assert anchors.minimum is not None and anchors.maximum is not None
    span = anchors.maximum - anchors.minimum
    if span == 0:
        return np.zeros_like(values)
    normalized: _FloatArray = np.clip((values - anchors.minimum) / span, 0.0, 1.0)
    return normalized


def compute_action_outcomes(
    space: ActionSpace,
    *,
    quality: _FloatArray,
    cost_usd: _FloatArray,
    latency_ms: _FloatArray,
    success: _BoolArray,
    weights: RewardWeights,
    stats: NormalizationStats,
) -> ActionOutcomes:
    """Compute quality, cost, success, and utility for every action.

    All inputs are ``(n_prompts, n_arms)`` wide matrices aligned with
    ``space.arms``. Single-arm actions reproduce the Step 5 utility exactly;
    cascade actions apply the simulated-verifier escalation rule documented
    in the module docstring.
    """

    n_arms = len(space.arms)
    for name, matrix in (
        ("quality", quality),
        ("cost_usd", cost_usd),
        ("latency_ms", latency_ms),
        ("success", success),
    ):
        if matrix.ndim != 2 or matrix.shape[1] != n_arms or matrix.shape[0] != quality.shape[0]:
            raise CascadeError(f"{name} must be a (n_prompts, n_arms) matrix")

    rejected = (quality == 0.0) | ~success

    action_quality = [quality[:, index] for index in range(n_arms)]
    action_cost = [cost_usd[:, index] for index in range(n_arms)]
    action_latency = [latency_ms[:, index] for index in range(n_arms)]
    action_success = [success[:, index] for index in range(n_arms)]
    action_escalated = [np.zeros(quality.shape[0], dtype=np.bool_) for _ in range(n_arms)]

    for first, second in space.pairs:
        escalate = rejected[:, first]
        action_quality.append(np.where(escalate, quality[:, second], quality[:, first]))
        action_cost.append(cost_usd[:, first] + np.where(escalate, cost_usd[:, second], 0.0))
        action_latency.append(latency_ms[:, first] + np.where(escalate, latency_ms[:, second], 0.0))
        action_success.append(np.where(escalate, success[:, second], success[:, first]))
        action_escalated.append(escalate)

    quality_matrix = np.column_stack(action_quality)
    cost_matrix = np.column_stack(action_cost)
    latency_matrix = np.column_stack(action_latency)
    success_matrix = np.column_stack(action_success).astype(np.bool_)
    escalated_matrix = np.column_stack(action_escalated).astype(np.bool_)

    utility_matrix = (
        weights.quality * quality_matrix
        - weights.cost * _normalized_term(cost_matrix, stats.cost_usd, "cost_usd")
        - weights.latency * _normalized_term(latency_matrix, stats.latency_ms, "latency_ms")
        - weights.error * (~success_matrix).astype(np.float64)
    )
    return ActionOutcomes(
        labels=space.labels,
        quality=quality_matrix,
        cost_usd=cost_matrix,
        success=success_matrix,
        utility=utility_matrix,
        escalated=escalated_matrix,
    )


def cascade_selection_frame(
    outcomes: ActionOutcomes, prompt_ids: list[str], action_indices: list[int]
) -> pd.DataFrame:
    """Materialize one selected action per prompt as an auditable selection.

    The frame carries the same outcome columns the reference policies expose,
    so ``evaluate_selection`` and the bootstrap treat cascade policies exactly
    like single-arm ones. ``model_id`` holds the action label, which for
    cascades reads ``first>second``.
    """

    if len(prompt_ids) != len(action_indices) or not prompt_ids:
        raise CascadeError("prompt_ids and action_indices must be non-empty and aligned")
    if any(not 0 <= index < len(outcomes.labels) for index in action_indices):
        raise CascadeError("action index out of range for the outcome table")
    rows = np.arange(len(prompt_ids))
    columns = np.asarray(action_indices, dtype=np.int64)
    return pd.DataFrame(
        {
            "prompt_id": prompt_ids,
            "model_id": [outcomes.labels[index] for index in action_indices],
            "quality": outcomes.quality[rows, columns],
            "cost_usd": outcomes.cost_usd[rows, columns],
            "success": outcomes.success[rows, columns],
            "escalated": outcomes.escalated[rows, columns],
            UTILITY_COLUMN: outcomes.utility[rows, columns],
        }
    )
