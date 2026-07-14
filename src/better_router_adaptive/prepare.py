"""End-to-end Step 4 pipeline: leakage-free features and grouped splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from better_router_adaptive.config import ExperimentConfig, load_experiment_config
from better_router_adaptive.data.schema import validate_canonical_dataframe
from better_router_adaptive.features import (
    OUTCOME_COLUMNS,
    assert_no_outcome_columns,
    build_prompt_features,
    feature_column_names,
)
from better_router_adaptive.split import (
    SPLIT_COLUMN,
    SPLIT_LABELS,
    assign_prompt_splits,
    attach_split_column,
)

_MANIFEST_SCHEMA_VERSION = "1.0.0"


def _write_atomic_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.part")
    frame.to_csv(temporary, index=False, lineterminator="\n", compression="gzip")
    temporary.replace(path)


def _split_summary(assignment: pd.DataFrame) -> dict[str, object]:
    prompt_counts = assignment[SPLIT_COLUMN].value_counts()
    by_group = (
        assignment.groupby(["task_group", SPLIT_COLUMN], sort=True)
        .size()
        .reset_index(name="prompts")
    )
    return {
        "prompts_per_split": {label: int(prompt_counts.get(label, 0)) for label in SPLIT_LABELS},
        "prompts_per_task_group_and_split": {
            f"{task_group}/{split}": int(prompts)
            for task_group, split, prompts in zip(
                by_group["task_group"], by_group[SPLIT_COLUMN], by_group["prompts"], strict=True
            )
        },
    }


def run_prepare_pipeline(
    input_path: Path,
    output_directory: Path,
    *,
    config: ExperimentConfig,
    seed: int,
    evidence_label: str,
) -> Path:
    """Build features and grouped splits from a cleaned canonical dataset."""

    if seed not in config.seeds:
        raise ValueError(f"seed {seed} is not part of the locked experiment seeds {config.seeds}")

    frame = pd.read_csv(input_path)
    frame["success"] = frame["success"].astype(bool)
    validate_canonical_dataframe(
        frame,
        required_model_arms=tuple(model.model_id for model in config.models),
        allowed_task_groups=config.task_groups,
    )

    features = build_prompt_features(frame, task_groups=config.task_groups)
    assignment = assign_prompt_splits(
        features.loc[:, ["prompt_id", "task_group"]], ratios=config.split, seed=seed
    )
    with_split = attach_split_column(frame, assignment)

    output_directory.mkdir(parents=True, exist_ok=True)
    features_with_split = features.merge(
        assignment.loc[:, ["prompt_id", SPLIT_COLUMN]], on="prompt_id", how="left"
    )
    assert_no_outcome_columns(features_with_split)

    features_path = output_directory / "prompt_features.csv.gz"
    _write_atomic_csv_gz(features_with_split, features_path)

    assignment_path = output_directory / "split_assignment.csv"
    assignment.to_csv(assignment_path, index=False, lineterminator="\n")

    canonical_split_path = output_directory / "routerbench_canonical_split.csv.gz"
    _write_atomic_csv_gz(with_split, canonical_split_path)

    manifest = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "step": "step4_features_and_splits",
        "seed": seed,
        "target_split_ratios": {
            "train": config.split.train,
            "validation": config.split.validation,
            "test": config.split.test,
        },
        "grouping_key": "prompt_id",
        "stratification_key": "task_group",
        "feature_columns": list(feature_column_names(config.task_groups)),
        "excluded_outcome_columns": list(OUTCOME_COLUMNS),
        "rows": len(with_split),
        "prompts": int(assignment["prompt_id"].nunique()),
        "evidence_label": evidence_label,
        **_split_summary(assignment),
    }
    (output_directory / "step4_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return canonical_split_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build pre-inference features and prompt-grouped 70/15/15 splits."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/experiment.yaml"))
    parser.add_argument("--models", type=Path, default=Path("config/models.yaml"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--evidence-label", default="DATOS EXPERIMENTALES")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config = load_experiment_config(args.config, models_path=args.models)
    output = run_prepare_pipeline(
        args.input,
        args.output_directory,
        config=config,
        seed=args.seed,
        evidence_label=args.evidence_label,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
