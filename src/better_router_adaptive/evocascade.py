"""EvoCascade policy evaluated with an ideal simulated verifier.

The learned policy consumes only pre-inference features. Its cascade outcomes,
however, use benchmark ground truth to simulate a perfect verifier that
escalates exactly when the first result has zero quality or a failed call.
Consequently, the reported result is an upper-bound experiment, not directly
deployable routing performance.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import numpy as np
import numpy.typing as npt

from better_router_adaptive.cascade import ActionOutcomes, ActionSpace
from better_router_adaptive.routers import RouterError

_FloatArray = npt.NDArray[np.float64]
EVOCASCADE_POLICY: Final = "evocascade-ideal-verifier"
EVOCASCADE_SIGMA_GRID: Final = (0.5, 1.5)
EVOCASCADE_ITERATIONS: Final = 160


class SepCMAES:
    """Separable CMA-ES black-box minimizer with diagonal covariance."""

    def __init__(
        self, dimension: int, *, sigma: float, seed: int, population: int | None = None
    ) -> None:
        if dimension <= 0:
            raise RouterError("search dimension must be positive")
        if sigma <= 0:
            raise RouterError("initial sigma must be positive")
        n = float(dimension)
        self.dimension = dimension
        self.sigma0 = float(sigma)
        self._rng = np.random.default_rng(seed)
        self.population = population if population is not None else 4 + int(3 * np.log(n))
        if self.population < 2:
            raise RouterError("population must contain at least two candidates")
        self.mu = self.population // 2
        raw_weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_weights / raw_weights.sum()
        self.mu_eff = float(1.0 / np.sum(self.weights**2))
        self.c_sigma = (self.mu_eff + 2) / (n + self.mu_eff + 5)
        self.d_sigma = (
            1
            + 2 * max(0.0, np.sqrt((self.mu_eff - 1) / (n + 1)) - 1)
            + self.c_sigma
        )
        self.c_c = (4 + self.mu_eff / n) / (n + 4 + 2 * self.mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + self.mu_eff)
        c_mu = min(
            1 - c_1,
            2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((n + 2) ** 2 + self.mu_eff),
        )
        separable_factor = (n + 2) / 3
        self.c_1 = min(1.0, c_1 * separable_factor)
        self.c_mu = min(1 - self.c_1, c_mu * separable_factor)
        self.chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

    def minimize(
        self, objective: Callable[[_FloatArray], _FloatArray], iterations: int
    ) -> _FloatArray:
        """Run the requested generations and return the best sampled solution."""

        if iterations <= 0:
            raise RouterError("iterations must be positive")
        n = self.dimension
        mean = np.zeros(n, dtype=np.float64)
        sigma = self.sigma0
        variances = np.ones(n, dtype=np.float64)
        path_sigma = np.zeros(n, dtype=np.float64)
        path_c = np.zeros(n, dtype=np.float64)
        best_solution = mean.copy()
        best_fitness = np.inf

        for generation in range(1, iterations + 1):
            std = np.sqrt(variances)
            noise = self._rng.standard_normal((self.population, n))
            steps = noise * std
            candidates = mean + sigma * steps
            fitness = np.asarray(objective(candidates), dtype=np.float64)
            if fitness.shape != (self.population,):
                raise RouterError("objective must return one fitness per candidate")
            order = np.argsort(fitness, kind="mergesort")
            if fitness[order[0]] < best_fitness:
                best_fitness = float(fitness[order[0]])
                best_solution = candidates[order[0]].copy()

            selected_steps = steps[order[: self.mu]]
            step_mean = self.weights @ selected_steps
            mean = mean + sigma * step_mean
            path_sigma = (1 - self.c_sigma) * path_sigma + np.sqrt(
                self.c_sigma * (2 - self.c_sigma) * self.mu_eff
            ) * (step_mean / std)
            path_sigma_norm = float(np.linalg.norm(path_sigma))
            expected = path_sigma_norm / np.sqrt(1 - (1 - self.c_sigma) ** (2 * generation))
            h_sigma = 1.0 if expected / self.chi_n < 1.4 + 2 / (n + 1) else 0.0
            path_c = (1 - self.c_c) * path_c + h_sigma * np.sqrt(
                self.c_c * (2 - self.c_c) * self.mu_eff
            ) * step_mean
            delta_h_sigma = (1 - h_sigma) * self.c_c * (2 - self.c_c)
            rank_mu = self.weights @ (selected_steps**2)
            variances = (
                (1 - self.c_1 - self.c_mu) * variances
                + self.c_1 * (path_c**2 + delta_h_sigma * variances)
                + self.c_mu * rank_mu
            )
            variances = np.clip(variances, 1e-20, 1e20)
            sigma *= float(
                np.exp((self.c_sigma / self.d_sigma) * (path_sigma_norm / self.chi_n - 1))
            )
        return best_solution


class EvoCascadeRouter:
    """Linear cascade policy optimized directly with separable CMA-ES."""

    def __init__(self, space: ActionSpace, *, sigma: float, iterations: int, seed: int) -> None:
        if iterations <= 0:
            raise RouterError("iterations must be positive")
        if sigma <= 0:
            raise RouterError("sigma must be positive")
        self.space = space
        self.sigma = float(sigma)
        self.iterations = iterations
        self.seed = seed
        self._weights: _FloatArray | None = None
        self._dimension: int | None = None

    @staticmethod
    def _augment(contexts: _FloatArray) -> _FloatArray:
        return np.hstack([np.ones((contexts.shape[0], 1), dtype=np.float64), contexts])

    def fit(self, contexts: _FloatArray, outcomes: ActionOutcomes) -> None:
        """Evolve weights to maximize mean ideal-verifier cascade utility."""

        n_actions = self.space.size
        if outcomes.utility.shape != (contexts.shape[0], n_actions):
            raise RouterError("training outcomes must be (n_prompts, n_actions)")
        augmented = self._augment(contexts)
        dimension = augmented.shape[1]
        rows = np.arange(outcomes.utility.shape[0])

        def objective(population: _FloatArray) -> _FloatArray:
            fitness = np.empty(population.shape[0], dtype=np.float64)
            for index in range(population.shape[0]):
                weights = population[index].reshape(dimension, n_actions)
                actions = np.argmax(augmented @ weights, axis=1)
                fitness[index] = -float(outcomes.utility[rows, actions].mean())
            return fitness

        optimizer = SepCMAES(dimension * n_actions, sigma=self.sigma, seed=self.seed)
        best = optimizer.minimize(objective, self.iterations)
        self._weights = best.reshape(dimension, n_actions)
        self._dimension = dimension

    def route(self, contexts: _FloatArray) -> list[int]:
        """Return one action index per prompt."""

        if self._weights is None or self._dimension is None:
            raise RouterError("router must be fitted before routing")
        augmented = self._augment(contexts)
        if augmented.shape[1] != self._dimension:
            raise RouterError("context dimension differs from the training dimension")
        return [int(choice) for choice in np.argmax(augmented @ self._weights, axis=1)]
