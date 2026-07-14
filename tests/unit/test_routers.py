from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from better_router_adaptive.routers import (
    XGBOOST_GRID,
    FeatureScaler,
    LinUCBRouter,
    RouterError,
    XGBoostConfig,
    XGBoostRouter,
)

ARMS = ("arm-a", "arm-b")


def _separable_problem(n_prompts: int) -> tuple[np.ndarray, np.ndarray]:
    """Half the prompts prefer arm-a, the other half arm-b, via one feature."""

    rng = np.random.default_rng(0)
    indicator = np.arange(n_prompts) % 2
    contexts = np.column_stack([indicator.astype(np.float64)])
    noise = rng.normal(0.0, 0.01, size=n_prompts)
    utility_a = np.where(indicator == 0, 0.8, 0.2) + noise
    utility_b = np.where(indicator == 0, 0.2, 0.8) - noise
    return contexts, np.column_stack([utility_a, utility_b])


def test_feature_scaler_is_anchored_to_the_training_frame() -> None:
    train = pd.DataFrame({"f1": [0.0, 10.0], "f2": [1.0, 1.0]})
    scaler = FeatureScaler.fit(train, ("f1", "f2"))

    evaluation = pd.DataFrame({"f1": [-5.0, 5.0, 20.0], "f2": [1.0, 3.0, 0.0]})
    transformed = scaler.transform(evaluation)

    assert transformed[:, 0].tolist() == [0.0, 0.5, 1.0]
    assert transformed[:, 1].tolist() == [0.0, 0.0, 0.0]


def test_feature_scaler_rejects_missing_columns() -> None:
    scaler = FeatureScaler.fit(pd.DataFrame({"f1": [0.0, 1.0]}), ("f1",))

    with pytest.raises(RouterError, match="missing feature columns"):
        scaler.transform(pd.DataFrame({"other": [1.0]}))


def test_xgboost_router_learns_a_separable_routing_rule() -> None:
    contexts, utilities = _separable_problem(n_prompts=40)
    router = XGBoostRouter(ARMS, config=XGBoostConfig(2, 50, 0.3), seed=42)
    router.fit(contexts, utilities)

    chosen = router.route(contexts)

    expected = ["arm-a" if int(row[0]) == 0 else "arm-b" for row in contexts]
    assert chosen == expected


def test_xgboost_router_is_deterministic_per_seed() -> None:
    contexts, utilities = _separable_problem(n_prompts=40)

    first = XGBoostRouter(ARMS, config=XGBoostConfig(3, 150, 0.1), seed=42)
    first.fit(contexts, utilities)
    second = XGBoostRouter(ARMS, config=XGBoostConfig(3, 150, 0.1), seed=42)
    second.fit(contexts, utilities)

    assert np.array_equal(first.predict_utilities(contexts), second.predict_utilities(contexts))


def test_xgboost_router_requires_fit_before_routing() -> None:
    router = XGBoostRouter(ARMS, config=XGBoostConfig(2, 50, 0.1), seed=42)

    with pytest.raises(RouterError, match="fitted before predicting"):
        router.route(np.zeros((1, 1)))


def test_xgboost_router_rejects_misshaped_utilities() -> None:
    router = XGBoostRouter(ARMS, config=XGBoostConfig(2, 50, 0.1), seed=42)

    with pytest.raises(RouterError, match="n_prompts, n_arms"):
        router.fit(np.zeros((4, 1)), np.zeros((4, 3)))


def test_xgboost_grid_is_locked_and_unique() -> None:
    assert len(XGBOOST_GRID) == 8
    assert len(set(XGBOOST_GRID)) == 8


def test_linucb_learns_the_separable_rule_after_replay() -> None:
    contexts, rewards = _separable_problem(n_prompts=200)
    router = LinUCBRouter(ARMS, alpha=0.5)

    regrets = router.replay(contexts, rewards)
    chosen = router.route(contexts)

    expected = ["arm-a" if int(row[0]) == 0 else "arm-b" for row in contexts]
    assert chosen == expected
    assert (regrets >= -1e-12).all()
    late_regret = regrets[150:].mean()
    early_regret = regrets[:50].mean()
    assert late_regret <= early_regret


def test_linucb_replay_is_deterministic() -> None:
    contexts, rewards = _separable_problem(n_prompts=100)

    first = LinUCBRouter(ARMS, alpha=1.0)
    second = LinUCBRouter(ARMS, alpha=1.0)

    assert np.array_equal(first.replay(contexts, rewards), second.replay(contexts, rewards))


def test_linucb_requires_replay_before_routing() -> None:
    router = LinUCBRouter(ARMS, alpha=1.0)

    with pytest.raises(RouterError, match="replay a training stream"):
        router.route(np.zeros((1, 1)))


def test_linucb_rejects_dimension_changes_and_bad_parameters() -> None:
    contexts, rewards = _separable_problem(n_prompts=10)
    router = LinUCBRouter(ARMS, alpha=1.0)
    router.replay(contexts, rewards)

    with pytest.raises(RouterError, match="context dimension"):
        router.route(np.zeros((1, 5)))
    with pytest.raises(RouterError, match="alpha must be non-negative"):
        LinUCBRouter(ARMS, alpha=-0.1)
    with pytest.raises(RouterError, match="rewards matrix"):
        LinUCBRouter(ARMS, alpha=1.0).replay(np.zeros((4, 1)), np.zeros((4, 3)))


def test_xgboost_router_limits_native_threads_during_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    import better_router_adaptive.routers as routers_module

    active = {"value": False}
    fitted_inside_limit: list[bool] = []

    class LimitContext:
        def __enter__(self) -> None:
            active["value"] = True

        def __exit__(self, *_args: object) -> None:
            active["value"] = False

    class FakeRegressor:
        def __init__(self, **_kwargs: object) -> None:
            self._Booster = object()

        def fit(self, _contexts: np.ndarray, _utilities: np.ndarray) -> None:
            fitted_inside_limit.append(active["value"])

        def predict(self, contexts: np.ndarray) -> np.ndarray:
            return np.zeros(contexts.shape[0])

    def fake_limits(*, limits: int) -> LimitContext:
        assert limits == 1
        return LimitContext()

    monkeypatch.setattr(routers_module, "threadpool_limits", fake_limits)
    monkeypatch.setattr(routers_module, "XGBRegressor", FakeRegressor)

    router = routers_module.XGBoostRouter(
        ARMS, config=routers_module.XGBoostConfig(2, 50, 0.1), seed=42
    )
    router.fit(np.zeros((4, 1)), np.zeros((4, 2)))

    assert fitted_inside_limit == [True, True]


def test_xgboost_router_close_releases_native_boosters(monkeypatch: pytest.MonkeyPatch) -> None:
    import better_router_adaptive.routers as routers_module

    created: list[object] = []

    class FakeRegressor:
        def __init__(self, **_kwargs: object) -> None:
            self._Booster: object | None = object()
            created.append(self)

        def fit(self, _contexts: np.ndarray, _utilities: np.ndarray) -> None:
            return None

        def predict(self, contexts: np.ndarray) -> np.ndarray:
            return np.zeros(contexts.shape[0])

    monkeypatch.setattr(routers_module, "XGBRegressor", FakeRegressor)

    router = routers_module.XGBoostRouter(
        ARMS, config=routers_module.XGBoostConfig(2, 50, 0.1), seed=42
    )
    router.fit(np.zeros((4, 1)), np.zeros((4, 2)))
    router.close()

    assert router._models == {}
    assert all(model._Booster is None for model in created)  # type: ignore[attr-defined]


def test_package_sets_safe_native_thread_defaults_in_clean_process(tmp_path: Path) -> None:
    script = tmp_path / "check_thread_defaults.py"
    script.write_text(
        """
import json
import os

for key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.pop(key, None)

import better_router_adaptive  # noqa: F401, E402

print(json.dumps({
    key: os.environ.get(key)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
}, sort_keys=True))
""".strip()
        + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(script)],
        check=True,
        capture_output=True,
        text=True,
        env={
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            }
        },
    )
    assert json.loads(completed.stdout) == {
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
