from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from better_router_adaptive.baselines import run_baselines_pipeline
from better_router_adaptive.config import load_experiment_config
from better_router_adaptive.data.schema import CANONICAL_COLUMNS
from better_router_adaptive.prepare import run_prepare_pipeline

MODEL_ARMS = ("arm-fast", "arm-balanced", "arm-reasoning", "arm-premium")
TASK_GROUPS = ("coding", "mathematics", "reasoning", "general")
PROMPTS_PER_GROUP = 10


def _synthetic_clean_canonical() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_index, group in enumerate(TASK_GROUPS):
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
                        "quality": min(
                            1.0, 0.3 + 0.1 * ((model_index + group_index) % 4) + 0.05 * model_index
                        ),
                        "input_tokens": 25 + prompt_index,
                        "output_tokens": 40 * model_index,
                        "cost_usd": 0.0005 * model_index,
                        "latency_ms": 120.0 * model_index,
                        "success": prompt_index != 0 or model_index != 1,
                        "data_origin": "synthetic_test_fixture_not_experimental_evidence",
                    }
                )
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def _prepare_split_dataset(tmp_path: Path) -> Path:
    config = load_experiment_config(
        Path("tests/fixtures/experiment.yaml"), models_path=Path("tests/fixtures/models.yaml")
    )
    input_path = tmp_path / "clean.csv"
    _synthetic_clean_canonical().to_csv(input_path, index=False, lineterminator="\n")
    run_prepare_pipeline(
        input_path, tmp_path / "step4", config=config, seed=42, evidence_label="DEMO"
    )
    return tmp_path / "step4" / "routerbench_canonical_split.csv.gz"


def test_step5_pipeline_generates_utilities_baselines_and_manifest(tmp_path: Path) -> None:
    config = load_experiment_config(
        Path("tests/fixtures/experiment.yaml"), models_path=Path("tests/fixtures/models.yaml")
    )
    split_dataset = _prepare_split_dataset(tmp_path)

    results_path = run_baselines_pipeline(
        split_dataset,
        tmp_path / "step5",
        config=config,
        evidence_label="DEMO SINTÉTICA — NO ES RESULTADO EXPERIMENTAL",
    )

    results = pd.read_csv(results_path)
    expected_policies = {"oracle", "better-rules-proxy", *(f"fixed:{arm}" for arm in MODEL_ARMS)}
    assert set(results["policy"]) == expected_policies
    assert set(results["split"]) == {"train", "validation", "test"}

    for split in ("train", "validation", "test"):
        subset = results[results["split"] == split].set_index("policy")
        oracle_utility = float(subset.loc["oracle", "mean_utility"])  # type: ignore[arg-type]
        for policy in expected_policies - {"oracle"}:
            other_utility = float(subset.loc[policy, "mean_utility"])  # type: ignore[arg-type]
            assert oracle_utility >= other_utility - 1e-12

    utilities = pd.read_csv(tmp_path / "step5" / "utility_dataset.csv.gz")
    assert "utility" in utilities.columns
    assert len(utilities) == PROMPTS_PER_GROUP * len(TASK_GROUPS) * len(MODEL_ARMS)

    stats = json.loads(
        (tmp_path / "step5" / "normalization_stats.json").read_text(encoding="utf-8")
    )
    assert stats["cost_usd"]["enabled"] is True

    rules = json.loads((tmp_path / "step5" / "better_rules_proxy.json").read_text(encoding="utf-8"))
    assert set(rules) == set(TASK_GROUPS)
    assert set(rules.values()) <= set(MODEL_ARMS)

    manifest = json.loads((tmp_path / "step5" / "step5_manifest.json").read_text(encoding="utf-8"))
    assert manifest["normalization_anchor_split"] == "train"
    assert manifest["reward_weights"] == {
        "quality": 0.65,
        "cost": 0.20,
        "latency": 0.10,
        "error": 0.05,
    }
    assert manifest["evidence_label"].startswith("DEMO SINTÉTICA")


def test_step5_normalization_ignores_extreme_values_outside_train(tmp_path: Path) -> None:
    config = load_experiment_config(
        Path("tests/fixtures/experiment.yaml"), models_path=Path("tests/fixtures/models.yaml")
    )
    split_dataset = _prepare_split_dataset(tmp_path)

    frame = pd.read_csv(split_dataset)
    train_max = frame.loc[frame["split"] == "train", "cost_usd"].max()
    outlier_index = frame.index[frame["split"] == "test"][0]
    frame.loc[outlier_index, "cost_usd"] = 999.0
    tampered = tmp_path / "tampered.csv.gz"
    frame.to_csv(tampered, index=False, lineterminator="\n", compression="gzip")

    run_baselines_pipeline(tampered, tmp_path / "step5", config=config, evidence_label="DEMO")

    stats = json.loads(
        (tmp_path / "step5" / "normalization_stats.json").read_text(encoding="utf-8")
    )
    assert stats["cost_usd"]["maximum"] == pytest.approx(float(train_max))

    utilities = pd.read_csv(tmp_path / "step5" / "utility_dataset.csv.gz")
    outlier_norm = utilities.loc[utilities["cost_usd"] == 999.0, "cost_usd_norm"]
    assert outlier_norm.tolist() == [1.0]


def test_step5_rejects_inputs_without_split_column(tmp_path: Path) -> None:
    config = load_experiment_config(
        Path("tests/fixtures/experiment.yaml"), models_path=Path("tests/fixtures/models.yaml")
    )
    input_path = tmp_path / "clean.csv"
    _synthetic_clean_canonical().to_csv(input_path, index=False, lineterminator="\n")

    with pytest.raises(ValueError, match="run the Step 4 pipeline first"):
        run_baselines_pipeline(input_path, tmp_path / "step5", config=config, evidence_label="X")
