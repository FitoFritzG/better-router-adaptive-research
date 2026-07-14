"""Reproducible research package for adaptive multi-LLM routing."""

from __future__ import annotations

import os

# Keep scientific runs deterministic and prevent native BLAS/OpenMP
# oversubscription. Explicit user settings take precedence.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

from .config import (  # noqa: E402
    ExperimentConfig,
    ModelConfig,
    RewardWeights,
    SplitRatios,
    load_experiment_config,
)

__all__ = [
    "ExperimentConfig",
    "ModelConfig",
    "RewardWeights",
    "SplitRatios",
    "load_experiment_config",
]
__version__ = "0.2.0"
