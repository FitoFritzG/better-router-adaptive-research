from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from better_router_adaptive.config import load_experiment_config
from better_router_adaptive.data.schema import CANONICAL_COLUMNS
from better_router_adaptive.features import OUTCOME_COLUMNS
from better_router_adaptive.prepare import run_prepare_pipeline

MODEL_ARMS = ("arm-fast", "arm-balanced", "arm-reasoning", "arm-premium")
TASK_GROUPS = ("coding", "mathematics", "reasoning", "general")
PROMPTS_PER_GROUP = 10


def _synthetic_clean_canonical() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group in TASK_GROUPS:
        for prompt_index in range(PROMPTS_PER_GROUP):
            prompt_id = f"{group}-{prompt_index:03d}"
            for model_index, model_id in enumerate(MODEL_ARMS, start=1):
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "prompt_text": f"Consulta sintética {prompt_index} del grupo {group}",
                        "dataset": f"dataset-{group}",
                        "task_group": group,
                        "model_id": model_id,
                        "quality": min(1.0, 0.4 + 0.1 * model_index + 0.01 * prompt_index),
                        "input_tokens": 25 + prompt_index,
                        "output_tokens": 40 * model_index,
                        "cost_usd": 0.0005 * model_index,
                        "latency_ms": 120.0 * model_index,
                        "success": True,
                        "data_origin": "synthetic_test_fixture_not_experimental_evidence",
                    }
                )
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def test_step4_pipeline_generates_features_splits_and_manifest(tmp_path: Path) -> None:
    config = load_experiment_config(
        Path("tests/fixtures/experiment.yaml"), models_path=Path("tests/fixtures/models.yaml")
    )
    input_path = tmp_path / "clean.csv"
    _synthetic_clean_canonical().to_csv(input_path, index=False, lineterminator="\n")

    output = run_prepare_pipeline(
        input_path,
        tmp_path / "step4",
        config=config,
        seed=42,
        evidence_label="DEMO SINTÉTICA — NO ES RESULTADO EXPERIMENTAL",
    )

    with_split = pd.read_csv(output)
    assert len(with_split) == PROMPTS_PER_GROUP * len(TASK_GROUPS) * len(MODEL_ARMS)
    assert (with_split.groupby("prompt_id")["split"].nunique() == 1).all()
    assert set(with_split["split"]) == {"train", "validation", "test"}

    features = pd.read_csv(tmp_path / "step4" / "prompt_features.csv.gz")
    assert len(features) == PROMPTS_PER_GROUP * len(TASK_GROUPS)
    assert not set(features.columns) & set(OUTCOME_COLUMNS)

    assignment = pd.read_csv(tmp_path / "step4" / "split_assignment.csv")
    assert not assignment["prompt_id"].duplicated().any()
    train_prompts = set(assignment.loc[assignment["split"] == "train", "prompt_id"])
    test_prompts = set(assignment.loc[assignment["split"] == "test", "prompt_id"])
    assert not train_prompts & test_prompts

    manifest = json.loads((tmp_path / "step4" / "step4_manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 42
    assert manifest["grouping_key"] == "prompt_id"
    assert manifest["prompts"] == PROMPTS_PER_GROUP * len(TASK_GROUPS)
    assert manifest["prompts_per_split"] == {"train": 28, "validation": 8, "test": 4}
    assert manifest["excluded_outcome_columns"] == list(OUTCOME_COLUMNS)
    assert manifest["evidence_label"].startswith("DEMO SINTÉTICA")


def test_step4_pipeline_is_deterministic_for_the_same_seed(tmp_path: Path) -> None:
    config = load_experiment_config(
        Path("tests/fixtures/experiment.yaml"), models_path=Path("tests/fixtures/models.yaml")
    )
    input_path = tmp_path / "clean.csv"
    _synthetic_clean_canonical().to_csv(input_path, index=False, lineterminator="\n")

    first = run_prepare_pipeline(
        input_path, tmp_path / "a", config=config, seed=2026, evidence_label="DEMO"
    )
    second = run_prepare_pipeline(
        input_path, tmp_path / "b", config=config, seed=2026, evidence_label="DEMO"
    )

    pd.testing.assert_frame_equal(pd.read_csv(first), pd.read_csv(second))


def test_step4_pipeline_rejects_seeds_outside_the_locked_protocol(tmp_path: Path) -> None:
    config = load_experiment_config(
        Path("tests/fixtures/experiment.yaml"), models_path=Path("tests/fixtures/models.yaml")
    )
    input_path = tmp_path / "clean.csv"
    _synthetic_clean_canonical().to_csv(input_path, index=False, lineterminator="\n")

    with pytest.raises(ValueError, match="locked experiment seeds"):
        run_prepare_pipeline(
            input_path, tmp_path / "step4", config=config, seed=7, evidence_label="DEMO"
        )
