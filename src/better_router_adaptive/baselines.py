"""End-to-end Step 5 pipeline: utility dataset, Better Rules Proxy, and oracle."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from better_router_adaptive.config import ExperimentConfig, load_experiment_config
from better_router_adaptive.policies import (
    ORACLE_POLICY,
    RULES_POLICY,
    PolicyMetrics,
    apply_rules_policy,
    evaluate_selection,
    fit_better_rules_proxy,
    fixed_arm_selection,
    oracle_selection,
)
from better_router_adaptive.split import SPLIT_COLUMN, SPLIT_LABELS, assert_split_integrity
from better_router_adaptive.utility import (
    compute_normalization_stats,
    compute_utility,
)

_MANIFEST_SCHEMA_VERSION = "1.0.0"


def _write_atomic_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.part")
    frame.to_csv(temporary, index=False, lineterminator="\n", compression="gzip")
    temporary.replace(path)


def _load_split_dataset(input_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(input_path)
    if SPLIT_COLUMN not in frame.columns:
        raise ValueError(
            f"input is missing the {SPLIT_COLUMN!r} column; run the Step 4 pipeline first"
        )
    frame["success"] = frame["success"].astype(bool)
    assignment = frame.loc[:, ["prompt_id", "task_group", SPLIT_COLUMN]].drop_duplicates()
    assert_split_integrity(assignment)
    return frame


def _evaluate_policies_per_split(
    utility_frame: pd.DataFrame,
    *,
    rules: dict[str, str],
    model_arms: tuple[str, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in SPLIT_LABELS:
        subset = utility_frame.loc[utility_frame[SPLIT_COLUMN] == split]
        evaluations: list[PolicyMetrics] = [
            evaluate_selection(oracle_selection(subset), policy=ORACLE_POLICY),
            evaluate_selection(apply_rules_policy(subset, rules), policy=RULES_POLICY),
        ]
        evaluations.extend(
            evaluate_selection(fixed_arm_selection(subset, arm), policy=f"fixed:{arm}")
            for arm in model_arms
        )
        rows.extend({"split": split, **asdict(metrics)} for metrics in evaluations)
    return rows


def run_baselines_pipeline(
    input_path: Path,
    output_directory: Path,
    *,
    config: ExperimentConfig,
    evidence_label: str,
) -> Path:
    """Compute train-anchored utilities and evaluate the reference policies."""

    frame = _load_split_dataset(input_path)
    train = frame.loc[frame[SPLIT_COLUMN] == "train"]
    stats = compute_normalization_stats(train)
    utility_frame = compute_utility(frame, weights=config.reward_weights, stats=stats)

    rules = fit_better_rules_proxy(utility_frame.loc[utility_frame[SPLIT_COLUMN] == "train"])
    model_arms = tuple(model.model_id for model in config.models)
    results = _evaluate_policies_per_split(utility_frame, rules=rules, model_arms=model_arms)

    output_directory.mkdir(parents=True, exist_ok=True)
    utility_path = output_directory / "utility_dataset.csv.gz"
    _write_atomic_csv_gz(utility_frame, utility_path)

    results_frame = pd.DataFrame(results)
    results_path = output_directory / "baseline_results.csv"
    results_frame.to_csv(results_path, index=False, lineterminator="\n")

    (output_directory / "normalization_stats.json").write_text(
        json.dumps(stats.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_directory / "better_rules_proxy.json").write_text(
        json.dumps(rules, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "step": "step5_utility_and_baselines",
        "reward_weights": {
            "quality": config.reward_weights.quality,
            "cost": config.reward_weights.cost,
            "latency": config.reward_weights.latency,
            "error": config.reward_weights.error,
        },
        "normalization_anchor_split": "train",
        "normalization_stats": stats.as_dict(),
        "better_rules_proxy": rules,
        "policies": [ORACLE_POLICY, RULES_POLICY, *(f"fixed:{arm}" for arm in model_arms)],
        "rows": len(utility_frame),
        "prompts": int(utility_frame["prompt_id"].nunique()),
        "evidence_label": evidence_label,
    }
    (output_directory / "step5_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return results_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute train-anchored utilities and evaluate baseline routing policies."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/experiment.yaml"))
    parser.add_argument("--models", type=Path, default=Path("config/models.yaml"))
    parser.add_argument("--evidence-label", default="DATOS EXPERIMENTALES")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config = load_experiment_config(args.config, models_path=args.models)
    output = run_baselines_pipeline(
        args.input,
        args.output_directory,
        config=config,
        evidence_label=args.evidence_label,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
