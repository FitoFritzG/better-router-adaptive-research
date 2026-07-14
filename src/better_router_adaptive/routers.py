"""Learned routing policies: per-arm XGBoost regressors and disjoint LinUCB.

Both routers consume the pre-inference feature matrix from Step 4, scaled with
train-anchored min-max statistics, and are rewarded with the Step 5 utility.
The XGBoost router is an offline supervised policy; LinUCB is an online
contextual bandit trained by prequential replay (choose, observe, update)
over a seeded shuffle of the training prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]
from xgboost import XGBRegressor

_FloatArray = npt.NDArray[np.float64]


class RouterError(ValueError):
    """Raised when a learned router cannot be trained or applied safely."""


@dataclass(frozen=True, slots=True)
class FeatureScaler:
    """Train-anchored min-max scaler for the routing feature columns."""

    columns: tuple[str, ...]
    minimums: tuple[float, ...]
    maximums: tuple[float, ...]

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: tuple[str, ...]) -> FeatureScaler:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise RouterError(f"missing feature columns: {', '.join(missing)}")
        if frame.empty:
            raise RouterError("cannot fit the feature scaler on an empty training split")
        minimums = tuple(float(frame[column].min()) for column in columns)
        maximums = tuple(float(frame[column].max()) for column in columns)
        return cls(columns=columns, minimums=minimums, maximums=maximums)

    def transform(self, frame: pd.DataFrame) -> _FloatArray:
        missing = [column for column in self.columns if column not in frame.columns]
        if missing:
            raise RouterError(f"missing feature columns: {', '.join(missing)}")
        scaled: list[_FloatArray] = []
        for column, minimum, maximum in zip(
            self.columns, self.minimums, self.maximums, strict=True
        ):
            values = frame[column].to_numpy(dtype=np.float64)
            span = maximum - minimum
            if span == 0:
                scaled.append(np.zeros_like(values))
            else:
                scaled.append(np.clip((values - minimum) / span, 0.0, 1.0))
        return np.column_stack(scaled)


@dataclass(frozen=True, slots=True)
class XGBoostConfig:
    """One locked hyperparameter combination for the XGBoost router."""

    max_depth: int
    n_estimators: int
    learning_rate: float


# Locked search grid; the winning combination is selected on the validation
# split only, never on test.
XGBOOST_GRID: Final = tuple(
    XGBoostConfig(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)
    for max_depth in (2, 3)
    for n_estimators in (50, 150)
    for learning_rate in (0.1, 0.3)
)


class XGBoostRouter:
    """Supervised router: one utility regressor per arm, greedy argmax routing."""

    def __init__(self, arms: tuple[str, ...], *, config: XGBoostConfig, seed: int) -> None:
        if len(arms) != len(set(arms)):
            raise RouterError("arms must be unique")
        self.arms = arms
        self.config = config
        self.seed = seed
        self._models: dict[str, XGBRegressor] = {}

    def fit(self, contexts: _FloatArray, utilities: _FloatArray) -> None:
        """Train one regressor per arm on (context -> realized utility)."""

        if utilities.shape != (contexts.shape[0], len(self.arms)):
            raise RouterError("utilities matrix must be (n_prompts, n_arms)")
        self.close()
        with threadpool_limits(limits=1):
            for index, arm in enumerate(self.arms):
                model = XGBRegressor(
                    max_depth=self.config.max_depth,
                    n_estimators=self.config.n_estimators,
                    learning_rate=self.config.learning_rate,
                    objective="reg:squarederror",
                    random_state=self.seed,
                    n_jobs=1,
                    tree_method="hist",
                    verbosity=0,
                )
                model.fit(contexts, utilities[:, index])
                self._models[arm] = model

    def close(self) -> None:
        """Release native XGBoost booster handles held by this router."""

        for model in self._models.values():
            if model._Booster is not None:
                model._Booster = None  # type: ignore[assignment]
        self._models.clear()

    def predict_utilities(self, contexts: _FloatArray) -> _FloatArray:
        if set(self._models) != set(self.arms):
            raise RouterError("router must be fitted before predicting")
        predictions = np.column_stack([self._models[arm].predict(contexts) for arm in self.arms])
        return predictions.astype(np.float64)

    def route(self, contexts: _FloatArray) -> list[str]:
        """Pick the arm with the highest predicted utility (ties: first arm)."""

        choices = np.argmax(self.predict_utilities(contexts), axis=1)
        return [self.arms[int(choice)] for choice in choices]


class LinUCBRouter:
    """Disjoint LinUCB contextual bandit with an intercept term per arm."""

    def __init__(self, arms: tuple[str, ...], *, alpha: float, ridge: float = 1.0) -> None:
        if len(arms) != len(set(arms)):
            raise RouterError("arms must be unique")
        if alpha < 0:
            raise RouterError("alpha must be non-negative")
        if ridge <= 0:
            raise RouterError("ridge must be positive")
        self.arms = arms
        self.alpha = alpha
        self.ridge = ridge
        self._dimension: int | None = None
        self._a_matrices: list[_FloatArray] = []
        self._b_vectors: list[_FloatArray] = []

    @staticmethod
    def _with_intercept(contexts: _FloatArray) -> _FloatArray:
        ones = np.ones((contexts.shape[0], 1), dtype=np.float64)
        return np.hstack([ones, contexts])

    def _initialize(self, dimension: int) -> None:
        self._dimension = dimension
        self._a_matrices = [np.eye(dimension, dtype=np.float64) * self.ridge for _ in self.arms]
        self._b_vectors = [np.zeros(dimension, dtype=np.float64) for _ in self.arms]

    def _scores(self, x: _FloatArray, *, explore: bool) -> _FloatArray:
        scores = np.empty(len(self.arms), dtype=np.float64)
        for index in range(len(self.arms)):
            a_matrix = self._a_matrices[index]
            theta = np.linalg.solve(a_matrix, self._b_vectors[index])
            estimate = float(theta @ x)
            if explore:
                width = float(np.sqrt(x @ np.linalg.solve(a_matrix, x)))
                estimate += self.alpha * width
            scores[index] = estimate
        return scores

    def replay(self, contexts: _FloatArray, rewards: _FloatArray) -> _FloatArray:
        """Prequential replay: choose with UCB, observe, update; returns regrets.

        ``rewards`` holds the realized utility of every arm for each prompt in
        stream order, which RouterBench makes possible because all arms were
        executed offline. The per-step regret is measured against the best
        realized arm of that prompt.
        """

        if rewards.shape != (contexts.shape[0], len(self.arms)):
            raise RouterError("rewards matrix must be (n_prompts, n_arms)")
        augmented = self._with_intercept(contexts)
        self._initialize(augmented.shape[1])

        regrets = np.empty(augmented.shape[0], dtype=np.float64)
        for step in range(augmented.shape[0]):
            x = augmented[step]
            choice = int(np.argmax(self._scores(x, explore=True)))
            reward = float(rewards[step, choice])
            self._a_matrices[choice] += np.outer(x, x)
            self._b_vectors[choice] += reward * x
            regrets[step] = float(rewards[step].max()) - reward
        return regrets

    def route(self, contexts: _FloatArray) -> list[str]:
        """Greedy exploitation with the learned estimates (no update, no bonus)."""

        if self._dimension is None:
            raise RouterError("router must replay a training stream before routing")
        augmented = self._with_intercept(contexts)
        if augmented.shape[1] != self._dimension:
            raise RouterError("context dimension differs from the training dimension")
        choices = [
            int(np.argmax(self._scores(augmented[row], explore=False)))
            for row in range(augmented.shape[0])
        ]
        return [self.arms[choice] for choice in choices]
