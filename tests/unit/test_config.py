from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from better_router_adaptive.config import load_experiment_config


def write_config(tmp_path: Path, *, payload: dict[str, Any]) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def valid_payload() -> dict[str, Any]:
    return {
        "project": {"name": "better-router-adaptive", "version": "0.1.0"},
        "experiment": {
            "task_groups": ["coding", "mathematics", "reasoning", "general"],
            "model_arms": ["model-a", "model-b", "model-c", "model-d"],
            "seeds": [42, 123, 2026, 31415, 271828],
            "reward_weights": {
                "quality": 0.65,
                "cost": 0.20,
                "latency": 0.10,
                "error": 0.05,
            },
            "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
        },
    }


def test_loads_valid_configuration(tmp_path: Path) -> None:
    config = load_experiment_config(write_config(tmp_path, payload=valid_payload()))

    assert config.project_name == "better-router-adaptive"
    assert config.project_version == "0.1.0"
    assert config.task_groups == ("coding", "mathematics", "reasoning", "general")
    assert tuple(model.model_id for model in config.models) == (
        "model-a",
        "model-b",
        "model-c",
        "model-d",
    )
    assert config.seeds == (42, 123, 2026, 31415, 271828)
    assert config.reward_weights.quality == pytest.approx(0.65)
    assert config.split.train == pytest.approx(0.70)


def test_reward_weights_must_sum_to_one(tmp_path: Path) -> None:
    payload = valid_payload()
    payload["experiment"]["reward_weights"]["quality"] = 0.80

    with pytest.raises(ValueError, match=r"reward weights must sum to 1\.0"):
        load_experiment_config(write_config(tmp_path, payload=payload))


def test_split_ratios_must_sum_to_one(tmp_path: Path) -> None:
    payload = valid_payload()
    payload["experiment"]["split"]["test"] = 0.20

    with pytest.raises(ValueError, match=r"split ratios must sum to 1\.0"):
        load_experiment_config(write_config(tmp_path, payload=payload))


def test_exactly_four_model_arms_are_required(tmp_path: Path) -> None:
    payload = valid_payload()
    payload["experiment"]["model_arms"] = ["model-a", "model-b", "model-c"]

    with pytest.raises(ValueError, match="exactly four model arms"):
        load_experiment_config(write_config(tmp_path, payload=payload))


def test_duplicate_model_ids_fail(tmp_path: Path) -> None:
    payload = valid_payload()
    payload["experiment"]["model_arms"] = ["model-a", "model-b", "model-b", "model-d"]

    with pytest.raises(ValueError, match="model IDs must be unique"):
        load_experiment_config(write_config(tmp_path, payload=payload))


def test_duplicate_seeds_fail(tmp_path: Path) -> None:
    payload = valid_payload()
    payload["experiment"]["seeds"] = [42, 42, 2026, 31415, 271828]

    with pytest.raises(ValueError, match="seeds must be unique"):
        load_experiment_config(write_config(tmp_path, payload=payload))


def test_negative_reward_weight_fails(tmp_path: Path) -> None:
    payload = valid_payload()
    payload["experiment"]["reward_weights"] = {
        "quality": 0.75,
        "cost": 0.20,
        "latency": 0.10,
        "error": -0.05,
    }

    with pytest.raises(ValueError, match="reward weights must be non-negative"):
        load_experiment_config(write_config(tmp_path, payload=payload))


def test_unknown_model_id_fails_when_registry_is_provided(tmp_path: Path) -> None:
    payload = valid_payload()
    experiment_path = write_config(tmp_path, payload=payload)
    models_path = tmp_path / "models.yaml"
    models_path.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {"id": "model-a", "provider": "test"},
                    {"id": "model-b", "provider": "test"},
                    {"id": "model-c", "provider": "test"},
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown model IDs: model-d"):
        load_experiment_config(experiment_path, models_path=models_path)


def test_non_string_display_name_is_rejected(tmp_path: Path) -> None:
    payload = valid_payload()
    experiment_path = write_config(tmp_path, payload=payload)
    models_path = tmp_path / "models.yaml"
    models_path.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {"id": "model-a", "provider": "test", "display_name": 123},
                    {"id": "model-b", "provider": "test"},
                    {"id": "model-c", "provider": "test"},
                    {"id": "model-d", "provider": "test"},
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"display_name must be a string"):
        load_experiment_config(experiment_path, models_path=models_path)


def test_missing_configuration_file_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="configuration file not found"):
        load_experiment_config(tmp_path / "missing.yaml")


def test_configuration_root_must_be_mapping(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="configuration root must be a mapping"):
        load_experiment_config(path)


@pytest.mark.parametrize("key", ["name", "version"])
def test_project_strings_must_be_non_empty(tmp_path: Path, key: str) -> None:
    payload = valid_payload()
    payload["project"][key] = " "

    with pytest.raises(ValueError, match=rf"{key} must be a non-empty string"):
        load_experiment_config(write_config(tmp_path, payload=payload))


def test_task_groups_must_be_unique(tmp_path: Path) -> None:
    payload = valid_payload()
    payload["experiment"]["task_groups"] = ["coding", "coding", "reasoning", "general"]

    with pytest.raises(ValueError, match="task groups must be unique"):
        load_experiment_config(write_config(tmp_path, payload=payload))


def test_exactly_four_task_groups_are_required(tmp_path: Path) -> None:
    payload = valid_payload()
    payload["experiment"]["task_groups"] = ["coding", "reasoning", "general"]

    with pytest.raises(ValueError, match="exactly four task groups"):
        load_experiment_config(write_config(tmp_path, payload=payload))


@pytest.mark.parametrize("seed", [-1, True, "42"])
def test_seeds_must_be_non_negative_integers(tmp_path: Path, seed: object) -> None:
    payload = valid_payload()
    payload["experiment"]["seeds"] = [seed]

    with pytest.raises(ValueError, match="seeds must"):
        load_experiment_config(write_config(tmp_path, payload=payload))


def test_at_least_one_seed_is_required(tmp_path: Path) -> None:
    payload = valid_payload()
    payload["experiment"]["seeds"] = []

    with pytest.raises(ValueError, match="at least one seed"):
        load_experiment_config(write_config(tmp_path, payload=payload))


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.1])
def test_split_ratios_must_be_strict_probabilities(tmp_path: Path, value: float) -> None:
    payload = valid_payload()
    payload["experiment"]["split"] = {
        "train": value,
        "validation": 0.5,
        "test": 0.5 - value,
    }

    with pytest.raises(ValueError, match="split ratios must be strictly between 0 and 1"):
        load_experiment_config(write_config(tmp_path, payload=payload))


def test_missing_reward_field_fails(tmp_path: Path) -> None:
    payload = valid_payload()
    del payload["experiment"]["reward_weights"]["error"]

    with pytest.raises(ValueError, match="missing reward_weights fields: error"):
        load_experiment_config(write_config(tmp_path, payload=payload))


def test_unknown_reward_field_fails(tmp_path: Path) -> None:
    payload = valid_payload()
    payload["experiment"]["reward_weights"]["throughput"] = 0.0

    with pytest.raises(ValueError, match="unknown reward_weights fields: throughput"):
        load_experiment_config(write_config(tmp_path, payload=payload))


def test_non_numeric_reward_field_fails(tmp_path: Path) -> None:
    payload = valid_payload()
    payload["experiment"]["reward_weights"]["quality"] = "high"

    with pytest.raises(ValueError, match=r"reward_weights\.quality must be numeric"):
        load_experiment_config(write_config(tmp_path, payload=payload))


def test_duplicate_model_id_in_registry_fails(tmp_path: Path) -> None:
    payload = valid_payload()
    experiment_path = write_config(tmp_path, payload=payload)
    models_path = tmp_path / "models.yaml"
    models_path.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {"id": "model-a", "provider": "test"},
                    {"id": "model-a", "provider": "test"},
                    {"id": "model-c", "provider": "test"},
                    {"id": "model-d", "provider": "test"},
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate model ID in registry: model-a"):
        load_experiment_config(experiment_path, models_path=models_path)


def test_committed_configuration_and_registry_load() -> None:
    config = load_experiment_config(
        Path("config/experiment.yaml"), models_path=Path("config/models.yaml")
    )

    assert len(config.models) == 4
    assert all(model.provider == "benchmark" for model in config.models)
