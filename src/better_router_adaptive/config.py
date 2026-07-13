"""Strict experiment configuration loading and validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_TOLERANCE = 1e-9
_REQUIRED_WEIGHT_KEYS = ("quality", "cost", "latency", "error")
_REQUIRED_SPLIT_KEYS = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class RewardWeights:
    """Weights used to combine quality, cost, latency, and failure penalties."""

    quality: float
    cost: float
    latency: float
    error: float

    def __post_init__(self) -> None:
        values = (self.quality, self.cost, self.latency, self.error)
        if any(value < 0 for value in values):
            raise ValueError("reward weights must be non-negative")
        if abs(sum(values) - 1.0) > _TOLERANCE:
            raise ValueError("reward weights must sum to 1.0")


@dataclass(frozen=True, slots=True)
class SplitRatios:
    """Prompt-level train, validation, and test split ratios."""

    train: float
    validation: float
    test: float

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(value <= 0 or value >= 1 for value in values):
            raise ValueError("split ratios must be strictly between 0 and 1")
        if abs(sum(values) - 1.0) > _TOLERANCE:
            raise ValueError("split ratios must sum to 1.0")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """One candidate model arm exposed to the routing experiment."""

    model_id: str
    provider: str
    display_name: str | None = None

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model ID must not be empty")
        if not self.provider.strip():
            raise ValueError(f"provider must not be empty for model {self.model_id}")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Fully validated immutable configuration for one experiment family."""

    project_name: str
    project_version: str
    task_groups: tuple[str, ...]
    models: tuple[ModelConfig, ...]
    seeds: tuple[int, ...]
    reward_weights: RewardWeights
    split: SplitRatios

    def __post_init__(self) -> None:
        if not self.project_name.strip():
            raise ValueError("project name must not be empty")
        if not self.project_version.strip():
            raise ValueError("project version must not be empty")
        if len(self.task_groups) != 4:
            raise ValueError("exactly four task groups are required")
        if len(set(self.task_groups)) != len(self.task_groups):
            raise ValueError("task groups must be unique")
        if len(self.models) != 4:
            raise ValueError("exactly four model arms are required")
        model_ids = tuple(model.model_id for model in self.models)
        if len(set(model_ids)) != len(model_ids):
            raise ValueError("model IDs must be unique")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in self.seeds
        ):
            raise ValueError("seeds must be non-negative integers")


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return raw


def _require_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _require_string(parent: Mapping[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _require_string_sequence(parent: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = parent.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{key} must be a sequence of strings")
    result = tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    if len(result) != len(value):
        raise ValueError(f"{key} must contain only non-empty strings")
    return result


def _require_int_sequence(parent: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = parent.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{key} must be a sequence of integers")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{key} must contain only integers")
        result.append(item)
    return tuple(result)


def _require_numeric_fields(
    parent: Mapping[str, Any], keys: tuple[str, ...], section: str
) -> dict[str, float]:
    missing = [key for key in keys if key not in parent]
    if missing:
        raise ValueError(f"missing {section} fields: {', '.join(missing)}")
    unexpected = sorted(set(parent) - set(keys))
    if unexpected:
        raise ValueError(f"unknown {section} fields: {', '.join(unexpected)}")
    converted: dict[str, float] = {}
    for key in keys:
        value = parent[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{section}.{key} must be numeric")
        converted[key] = float(value)
    return converted


def _load_model_registry(path: Path) -> dict[str, ModelConfig]:
    raw = _load_yaml_mapping(path)
    entries = raw.get("models")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise ValueError("models must be a sequence")

    registry: dict[str, ModelConfig] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"models[{index}] must be a mapping")
        model_id = _require_string(entry, "id")
        if model_id in registry:
            raise ValueError(f"duplicate model ID in registry: {model_id}")
        display_name_value = entry.get("display_name")
        if display_name_value is not None and not isinstance(display_name_value, str):
            raise ValueError(f"models[{index}].display_name must be a string")
        display_name = display_name_value.strip() if display_name_value is not None else None
        registry[model_id] = ModelConfig(
            model_id=model_id,
            provider=_require_string(entry, "provider"),
            display_name=display_name or None,
        )
    return registry


def load_experiment_config(
    path: Path | str, *, models_path: Path | str | None = None
) -> ExperimentConfig:
    """Load a YAML experiment file and return an immutable validated configuration.

    When ``models_path`` is provided, every selected model arm must exist in that
    registry. Without a registry, model IDs are retained with provider
    ``"unspecified"`` so temporary unit-test configurations remain self-contained.
    """

    experiment_path = Path(path)
    raw = _load_yaml_mapping(experiment_path)
    project = _require_mapping(raw, "project")
    experiment = _require_mapping(raw, "experiment")

    task_groups = _require_string_sequence(experiment, "task_groups")
    model_ids = _require_string_sequence(experiment, "model_arms")
    seeds = _require_int_sequence(experiment, "seeds")

    weights_values = _require_numeric_fields(
        _require_mapping(experiment, "reward_weights"),
        _REQUIRED_WEIGHT_KEYS,
        "reward_weights",
    )
    split_values = _require_numeric_fields(
        _require_mapping(experiment, "split"), _REQUIRED_SPLIT_KEYS, "split"
    )

    if models_path is not None:
        registry = _load_model_registry(Path(models_path))
        unknown = sorted(set(model_ids) - set(registry))
        if unknown:
            raise ValueError(f"unknown model IDs: {', '.join(unknown)}")
        models = tuple(registry[model_id] for model_id in model_ids)
    else:
        models = tuple(
            ModelConfig(model_id=model_id, provider="unspecified") for model_id in model_ids
        )

    return ExperimentConfig(
        project_name=_require_string(project, "name"),
        project_version=_require_string(project, "version"),
        task_groups=task_groups,
        models=models,
        seeds=seeds,
        reward_weights=RewardWeights(**weights_values),
        split=SplitRatios(**split_values),
    )
