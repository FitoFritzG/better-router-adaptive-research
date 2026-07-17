from __future__ import annotations

import numpy as np
import pytest

from better_router_adaptive.cascade import (
    ActionOutcomes,
    ActionSpace,
    CascadeError,
    build_action_space,
    cascade_selection_frame,
    compute_action_outcomes,
)
from better_router_adaptive.config import RewardWeights
from better_router_adaptive.evocascade import (
    EVOCASCADE_POLICY,
    EvoCascadeRouter,
    SepCMAES,
)
from better_router_adaptive.routers import RouterError
from better_router_adaptive.utility import ColumnNormalization, NormalizationStats

WEIGHTS = RewardWeights(quality=0.65, cost=0.20, latency=0.10, error=0.05)


def _stats(cost: np.ndarray, latency: np.ndarray) -> NormalizationStats:
    return NormalizationStats(
        cost_usd=ColumnNormalization(True, float(cost.min()), float(cost.max())),
        latency_ms=ColumnNormalization(
            True, float(latency.min()), float(latency.max())
        ),
    )


def _scenario(n: int = 240) -> tuple[np.ndarray, ActionSpace, ActionOutcomes]:
    rng = np.random.default_rng(7)
    contexts = rng.random((n, 2))
    arms = ("cheap", "strong")
    mean_cost = np.array([0.02, 1.0])
    cost = np.tile(mean_cost, (n, 1))
    latency = cost * 1000.0
    quality = np.tile(np.array([0.7, 0.9]), (n, 1))
    success = np.ones((n, 2), dtype=np.bool_)
    failed = rng.random(n) < 0.15
    quality[failed, 0] = 0.0
    success[failed, 0] = False
    space = build_action_space(arms, mean_cost)
    outcomes = compute_action_outcomes(
        space,
        quality=quality,
        cost_usd=cost,
        latency_ms=latency,
        success=success,
        weights=WEIGHTS,
        stats=_stats(cost, latency),
    )
    return contexts, space, outcomes


def test_policy_name_declares_the_ideal_verifier_assumption() -> None:
    assert EVOCASCADE_POLICY == "evocascade-ideal-verifier"


def test_action_space_only_escalates_to_more_expensive_arms() -> None:
    space = build_action_space(("a", "b", "c"), np.array([0.1, 0.4, 0.9]))
    assert space.labels == ("a", "b", "c", "a>b", "a>c", "b>c")
    with pytest.raises(CascadeError, match="non-empty and unique"):
        build_action_space(("x", "x"), np.array([0.1, 0.2]))


def test_ideal_verifier_escalates_only_rejected_first_answers() -> None:
    space = build_action_space(("a", "b"), np.array([0.1, 1.0]))
    quality = np.array([[0.0, 0.9], [0.6, 0.9]])
    cost = np.array([[0.1, 1.0], [0.1, 1.0]])
    latency = cost * 1000.0
    success = np.array([[False, True], [True, True]])
    outcomes = compute_action_outcomes(
        space,
        quality=quality,
        cost_usd=cost,
        latency_ms=latency,
        success=success,
        weights=WEIGHTS,
        stats=_stats(cost, latency),
    )
    cascade = space.labels.index("a>b")
    assert outcomes.escalated[:, cascade].tolist() == [True, False]
    assert outcomes.quality[0, cascade] == 0.9
    assert outcomes.cost_usd[0, cascade] == pytest.approx(1.1)
    assert outcomes.cost_usd[1, cascade] == pytest.approx(0.1)


def test_action_outcomes_validate_shapes_and_nulls() -> None:
    space = build_action_space(("a", "b"), np.array([0.1, 1.0]))
    with pytest.raises(CascadeError, match="n_prompts, n_arms"):
        compute_action_outcomes(
            space,
            quality=np.zeros((4, 1)),
            cost_usd=np.zeros((4, 2)),
            latency_ms=np.zeros((4, 2)),
            success=np.ones((4, 2), dtype=bool),
            weights=WEIGHTS,
            stats=_stats(np.zeros((4, 2)), np.zeros((4, 2))),
        )
    with pytest.raises(CascadeError, match="null values"):
        compute_action_outcomes(
            space,
            quality=np.array([[0.5, 0.6]]),
            cost_usd=np.array([[0.1, np.nan]]),
            latency_ms=np.array([[100.0, 200.0]]),
            success=np.array([[True, True]]),
            weights=WEIGHTS,
            stats=_stats(np.array([[0.1, 1.0]]), np.array([[100.0, 200.0]])),
        )


def test_selection_frame_is_auditable() -> None:
    _, space, outcomes = _scenario(5)
    prompt_ids = [str(index) for index in range(5)]
    frame = cascade_selection_frame(outcomes, prompt_ids, [0, 1, 2, 0, 1])
    assert list(frame["prompt_id"]) == prompt_ids
    assert {"quality", "cost_usd", "success", "escalated", "utility"} <= set(frame)
    with pytest.raises(CascadeError, match="out of range"):
        cascade_selection_frame(outcomes, prompt_ids, [0, 1, 2, 0, space.size])


def test_sep_cmaes_converges_on_a_quadratic_problem() -> None:
    target = np.array([1.5, -0.5, 0.25])

    def objective(population: np.ndarray) -> np.ndarray:
        return np.asarray(((population - target) ** 2).sum(axis=1))

    solution = SepCMAES(target.size, sigma=1.0, seed=42).minimize(objective, 300)
    assert float(np.linalg.norm(solution - target)) < 1e-3


def test_evocascade_learns_the_dominant_ideal_cascade() -> None:
    contexts, space, outcomes = _scenario()
    cascade_index = space.labels.index("cheap>strong")
    assert (
        outcomes.utility[:, cascade_index].mean()
        > outcomes.utility[:, :2].mean(axis=0).max()
    )
    router = EvoCascadeRouter(space, sigma=1.0, iterations=120, seed=11)
    router.fit(contexts, outcomes)
    actions = router.route(contexts)
    rows = np.arange(len(actions))
    achieved = float(outcomes.utility[rows, np.asarray(actions)].mean())
    assert achieved >= float(outcomes.utility[:, :2].mean(axis=0).max())
    assert actions.count(cascade_index) > len(actions) // 2


def test_evocascade_rejects_invalid_state() -> None:
    contexts, space, outcomes = _scenario(30)
    with pytest.raises(RouterError, match="sigma must be positive"):
        EvoCascadeRouter(space, sigma=0.0, iterations=10, seed=0)
    router = EvoCascadeRouter(space, sigma=1.0, iterations=10, seed=0)
    with pytest.raises(RouterError, match="fitted before routing"):
        router.route(contexts)
    with pytest.raises(RouterError, match="training outcomes must be"):
        router.fit(np.zeros((5, 2)), outcomes)
