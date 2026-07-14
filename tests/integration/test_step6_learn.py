from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from better_router_adaptive.baselines import run_baselines_pipeline
from better_router_adaptive.config import load_experiment_config
from better_router_adaptive.data.schema import CANONICAL_COLUMNS
from better_router_adaptive.learn import LINUCB_ALPHA_GRID, run_learn_pipeline
from better_router_adaptive.prepare import run_prepare_pipeline

MODEL_ARMS = ("arm-fast", "arm-balanced", "arm-reasoning", "arm-premium")
TASK_GROUPS = ("coding", "mathematics", "reasoning", "general")
PROMPTS_PER_GROUP = 12

# Deterministic best arm per group so learned routers have signal to recover.
BEST_ARM_BY_GROUP = {
    "coding": "arm-reasoning",
    "mathematics": "arm-premium",
    "reasoning": "arm-balanced",
    "general": "arm-fast",
}


def _synthetic_clean_canonical() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group in TASK_GROUPS:
        for prompt_index in range(PROMPTS_PER_GROUP):
            prompt_id = f"{group}-{prompt_index:03d}"
            for model_index, model_id in enumerate(MODEL_ARMS, start=1):
                is_best = model_id == BEST_ARM_BY_GROUP[group]
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "prompt_text": f"Consulta sintética {prompt_index} del grupo {group}",
                        "dataset": f"dataset-{group}",
                        "task_group": group,
                        "model_id": model_id,
                        "quality": 0.9 if is_best else 0.4 + 0.02 * model_index,
                        "input_tokens": 25 + prompt_index,
                        "output_tokens": 40 * model_index,
                        "cost_usd": 0.0005 * model_index,
                        "latency_ms": 120.0 * model_index,
                        "success": True,
                        "data_origin": "synthetic_test_fixture_not_experimental_evidence",
                    }
                )
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def _prepare_inputs(tmp_path: Path) -> tuple[Path, Path]:
    config = load_experiment_config(
        Path("tests/fixtures/experiment.yaml"), models_path=Path("tests/fixtures/models.yaml")
    )
    input_path = tmp_path / "clean.csv"
    _synthetic_clean_canonical().to_csv(input_path, index=False, lineterminator="\n")
    run_prepare_pipeline(
        input_path, tmp_path / "step4", config=config, seed=42, evidence_label="DEMO"
    )
    run_baselines_pipeline(
        tmp_path / "step4" / "routerbench_canonical_split.csv.gz",
        tmp_path / "step5",
        config=config,
        evidence_label="DEMO",
    )
    return (
        tmp_path / "step5" / "utility_dataset.csv.gz",
        tmp_path / "step4" / "prompt_features.csv.gz",
    )


def test_step6_pipeline_trains_routers_and_writes_auditable_outputs(tmp_path: Path) -> None:
    config = load_experiment_config(
        Path("tests/fixtures/experiment.yaml"), models_path=Path("tests/fixtures/models.yaml")
    )
    utility_path, features_path = _prepare_inputs(tmp_path)

    results_path = run_learn_pipeline(
        utility_path,
        features_path,
        tmp_path / "step6",
        config=config,
        seed=42,
        evidence_label="DEMO SINTÉTICA — NO ES RESULTADO EXPERIMENTAL",
    )

    results = pd.read_csv(results_path)
    assert set(results["policy"]) == {"xgboost", "linucb"}
    assert set(results["split"]) == {"train", "validation", "test"}
    assert len(results) == 6

    # The task-group signal is deterministic, so both routers must beat the
    # worst fixed arm on test and stay below the oracle.
    baseline = pd.read_csv(tmp_path / "step5" / "baseline_results.csv")
    test_baseline = baseline[baseline["split"] == "test"].set_index("policy")
    oracle = float(test_baseline.loc["oracle", "mean_utility"])  # type: ignore[arg-type]
    fixed_utilities = [
        float(test_baseline.loc[f"fixed:{arm}", "mean_utility"])  # type: ignore[arg-type]
        for arm in MODEL_ARMS
    ]
    for policy in ("xgboost", "linucb"):
        row = results[(results["policy"] == policy) & (results["split"] == "test")]
        mean_utility = float(row["mean_utility"].iloc[0])
        assert mean_utility <= oracle + 1e-9
        assert mean_utility >= min(fixed_utilities) - 1e-9

    regret = pd.read_csv(tmp_path / "step6" / "linucb_regret.csv")
    assert (regret["regret"] >= -1e-9).all()
    assert regret["cumulative_regret"].iloc[-1] == pytest.approx(regret["regret"].sum())
    assert (np.diff(regret["cumulative_regret"].to_numpy()) >= -1e-9).all()

    xgboost_search = json.loads(
        (tmp_path / "step6" / "xgboost_search.json").read_text(encoding="utf-8")
    )
    assert len(xgboost_search["grid"]) == 8
    assert xgboost_search["selected"] in [
        {key: record[key] for key in ("max_depth", "n_estimators", "learning_rate")}
        for record in xgboost_search["grid"]
    ]

    linucb_search = json.loads(
        (tmp_path / "step6" / "linucb_search.json").read_text(encoding="utf-8")
    )
    assert len(linucb_search["grid"]) == len(LINUCB_ALPHA_GRID)
    assert linucb_search["selected_alpha"] in LINUCB_ALPHA_GRID

    manifest = json.loads((tmp_path / "step6" / "step6_manifest.json").read_text(encoding="utf-8"))
    assert manifest["hyperparameter_selection_split"] == "validation"
    assert manifest["seed"] == 42
    assert manifest["evidence_label"].startswith("DEMO SINTÉTICA")


def test_step6_pipeline_is_deterministic_for_the_same_seed(tmp_path: Path) -> None:
    config = load_experiment_config(
        Path("tests/fixtures/experiment.yaml"), models_path=Path("tests/fixtures/models.yaml")
    )
    utility_path, features_path = _prepare_inputs(tmp_path)

    first = run_learn_pipeline(
        utility_path,
        features_path,
        tmp_path / "a",
        config=config,
        seed=123,
        evidence_label="DEMO",
    )
    second = run_learn_pipeline(
        utility_path,
        features_path,
        tmp_path / "b",
        config=config,
        seed=123,
        evidence_label="DEMO",
    )

    pd.testing.assert_frame_equal(pd.read_csv(first), pd.read_csv(second))


def test_step6_rejects_seeds_outside_the_locked_protocol(tmp_path: Path) -> None:
    config = load_experiment_config(
        Path("tests/fixtures/experiment.yaml"), models_path=Path("tests/fixtures/models.yaml")
    )
    utility_path, features_path = _prepare_inputs(tmp_path)

    with pytest.raises(ValueError, match="locked experiment seeds"):
        run_learn_pipeline(
            utility_path,
            features_path,
            tmp_path / "step6",
            config=config,
            seed=99,
            evidence_label="X",
        )
