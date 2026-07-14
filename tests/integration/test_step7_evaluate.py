from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pytest import MonkeyPatch

import better_router_adaptive.evaluate as evaluate_module
from better_router_adaptive.config import load_experiment_config
from better_router_adaptive.data.schema import CANONICAL_COLUMNS
from better_router_adaptive.evaluate import run_evaluation_pipeline

MODEL_ARMS = ("arm-fast", "arm-balanced", "arm-reasoning", "arm-premium")
TASK_GROUPS = ("coding", "mathematics", "reasoning", "general")
PROMPTS_PER_GROUP = 12

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


def test_step7_pipeline_evaluates_all_policies_across_all_seeds(tmp_path: Path) -> None:
    config = load_experiment_config(
        Path("tests/fixtures/experiment.yaml"), models_path=Path("tests/fixtures/models.yaml")
    )
    input_path = tmp_path / "clean.csv"
    _synthetic_clean_canonical().to_csv(input_path, index=False, lineterminator="\n")

    summary_path = run_evaluation_pipeline(
        input_path,
        tmp_path / "step7",
        config=config,
        bootstrap_samples=200,
        evidence_label="DEMO SINTÉTICA — NO ES RESULTADO EXPERIMENTAL",
    )

    expected_policies = {
        "oracle",
        "better-rules-proxy",
        "xgboost",
        "linucb",
        *(f"fixed:{arm}" for arm in MODEL_ARMS),
    }

    per_seed = pd.read_csv(tmp_path / "step7" / "evaluation_per_seed.csv")
    assert set(per_seed["seed"]) == set(config.seeds)
    assert set(per_seed["policy"]) == expected_policies
    assert len(per_seed) == len(config.seeds) * len(expected_policies)

    summary = pd.read_csv(summary_path).set_index("policy")
    assert set(summary.index) == expected_policies

    oracle = summary.loc["oracle"]
    for policy in expected_policies:
        other = float(summary.loc[policy, "mean_utility"])  # type: ignore[arg-type]
        assert float(oracle["mean_utility"]) >= other - 1e-12
    # The oracle can never lose to the rules baseline on paired resamples.
    assert float(oracle["diff_ci95_low"]) >= -1e-12
    baseline_row = summary.loc["better-rules-proxy"]
    assert float(baseline_row["diff_vs_baseline"]) == 0.0

    regret = pd.read_csv(tmp_path / "step7" / "linucb_regret_curves.csv")
    assert set(regret["seed"]) == set(config.seeds)

    figures = tmp_path / "step7" / "figures"
    assert (figures / "comparacion_politicas.svg").is_file()
    assert (figures / "regret_linucb.svg").is_file()
    assert (figures / "frontera_calidad_costo.svg").is_file()

    manifest = json.loads((tmp_path / "step7" / "step7_manifest.json").read_text(encoding="utf-8"))
    assert manifest["bootstrap_unit"] == "prompt_id"
    assert manifest["bootstrap_baseline"] == "better-rules-proxy"
    assert manifest["seeds"] == list(config.seeds)
    assert manifest["evidence_label"].startswith("DEMO SINTÉTICA")


def test_step7_evaluates_seeds_outside_the_parent_process(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    config = load_experiment_config(
        Path("tests/fixtures/experiment.yaml"), models_path=Path("tests/fixtures/models.yaml")
    )
    input_path = tmp_path / "clean.csv"
    _synthetic_clean_canonical().to_csv(input_path, index=False, lineterminator="\n")

    def fail_if_called_in_parent(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("seed evaluation must run in an isolated child process")

    monkeypatch.setattr(evaluate_module, "_evaluate_one_seed", fail_if_called_in_parent)

    summary_path = run_evaluation_pipeline(
        input_path,
        tmp_path / "isolated-step7",
        config=config,
        bootstrap_samples=20,
        evidence_label="DEMO SINTÉTICA — AISLAMIENTO DE PROCESOS",
    )
    assert summary_path.is_file()
