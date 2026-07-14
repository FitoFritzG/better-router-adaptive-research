"""End-to-end Step 6 pipeline: train and evaluate the learned routers.

Hyperparameters (XGBoost grid, LinUCB alpha) are selected exclusively on the
validation split; the test split is only touched by the final frozen policies.
The LinUCB stream order is a seeded shuffle of the training prompts, so every
run is reproducible from the locked protocol seeds.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from better_router_adaptive.config import ExperimentConfig, load_experiment_config
from better_router_adaptive.features import feature_column_names
from better_router_adaptive.policies import evaluate_selection
from better_router_adaptive.routers import (
    XGBOOST_GRID,
    FeatureScaler,
    LinUCBRouter,
    RouterError,
    XGBoostConfig,
    XGBoostRouter,
)
from better_router_adaptive.split import SPLIT_COLUMN, SPLIT_LABELS
from better_router_adaptive.utility import UTILITY_COLUMN

XGBOOST_POLICY: Final = "xgboost"
LINUCB_POLICY: Final = "linucb"
LINUCB_ALPHA_GRID: Final = (0.1, 0.5, 1.0, 2.0)
_MANIFEST_SCHEMA_VERSION = "1.0.0"


def _read_table(path: Path, *, required: tuple[str, ...], label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RouterError(f"{label} is missing columns: {', '.join(missing)}")
    return frame


class SplitData:
    """Aligned prompts, scaled contexts, and realized rewards for one split."""

    def __init__(
        self,
        utility_frame: pd.DataFrame,
        features: pd.DataFrame,
        *,
        split: str,
        arms: tuple[str, ...],
        scaler: FeatureScaler,
    ) -> None:
        subset = features.loc[features[SPLIT_COLUMN] == split].sort_values(
            "prompt_id", kind="mergesort"
        )
        self.prompt_ids: list[str] = subset["prompt_id"].astype(str).tolist()
        self.contexts = scaler.transform(subset)

        pivot = (
            utility_frame.loc[utility_frame[SPLIT_COLUMN] == split]
            .pivot(index="prompt_id", columns="model_id", values=UTILITY_COLUMN)
            .reindex(index=self.prompt_ids, columns=list(arms))
        )
        if bool(pivot.isna().any(axis=None)):
            raise RouterError(f"split {split!r} lacks utilities for some prompt/arm pairs")
        self.rewards = pivot.to_numpy(dtype=np.float64)

    def realized_mean_utility(self, chosen_indices: list[int]) -> float:
        rows = np.arange(len(chosen_indices))
        return float(self.rewards[rows, np.array(chosen_indices)].mean())


def chosen_indices(arms: tuple[str, ...], chosen_arms: list[str]) -> list[int]:
    positions = {arm: index for index, arm in enumerate(arms)}
    return [positions[arm] for arm in chosen_arms]


def selection_metrics(
    utility_frame: pd.DataFrame,
    *,
    split: str,
    prompt_ids: list[str],
    chosen_arms: list[str],
    policy: str,
) -> dict[str, object]:
    choices = pd.DataFrame({"prompt_id": prompt_ids, "model_id": chosen_arms})
    subset = utility_frame.loc[utility_frame[SPLIT_COLUMN] == split]
    selection = subset.merge(choices, on=["prompt_id", "model_id"], how="inner")
    if len(selection) != len(prompt_ids):
        raise RouterError(f"policy {policy!r} produced choices without matching outcomes")
    metrics = evaluate_selection(selection, policy=policy)
    return {"split": split, **asdict(metrics)}


def tune_xgboost(
    train: SplitData, validation: SplitData, *, arms: tuple[str, ...], seed: int
) -> tuple[XGBoostConfig, list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    best_config: XGBoostConfig | None = None
    best_score = -np.inf
    for config in XGBOOST_GRID:
        router = XGBoostRouter(arms, config=config, seed=seed)
        try:
            router.fit(train.contexts, train.rewards)
            chosen = chosen_indices(arms, router.route(validation.contexts))
            score = validation.realized_mean_utility(chosen)
        finally:
            router.close()
        records.append({**asdict(config), "validation_mean_utility": score})
        if score > best_score:
            best_score = score
            best_config = config
    assert best_config is not None
    return best_config, records


def tune_linucb(
    train: SplitData,
    validation: SplitData,
    *,
    arms: tuple[str, ...],
    stream_order: npt.NDArray[np.int64],
) -> tuple[float, list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    best_alpha: float | None = None
    best_score = -np.inf
    for alpha in LINUCB_ALPHA_GRID:
        router = LinUCBRouter(arms, alpha=alpha)
        router.replay(train.contexts[stream_order], train.rewards[stream_order])
        chosen = chosen_indices(arms, router.route(validation.contexts))
        score = validation.realized_mean_utility(chosen)
        records.append({"alpha": alpha, "validation_mean_utility": score})
        if score > best_score:
            best_score = score
            best_alpha = alpha
    assert best_alpha is not None
    return best_alpha, records


def run_learn_pipeline(
    utility_path: Path,
    features_path: Path,
    output_directory: Path,
    *,
    config: ExperimentConfig,
    seed: int,
    evidence_label: str,
) -> Path:
    """Train both learned routers and write auditable per-split results."""

    if seed not in config.seeds:
        raise ValueError(f"seed {seed} is not part of the locked experiment seeds {config.seeds}")

    arms = tuple(sorted(model.model_id for model in config.models))
    feature_columns = feature_column_names(config.task_groups)

    utility_frame = _read_table(
        utility_path,
        required=("prompt_id", "model_id", "task_group", SPLIT_COLUMN, UTILITY_COLUMN),
        label="utility dataset",
    )
    features = _read_table(
        features_path,
        required=("prompt_id", SPLIT_COLUMN, *feature_columns),
        label="feature table",
    )
    if set(features["prompt_id"].astype(str)) != set(utility_frame["prompt_id"].astype(str)):
        raise RouterError("feature table and utility dataset cover different prompts")

    scaler = FeatureScaler.fit(features.loc[features[SPLIT_COLUMN] == "train"], feature_columns)
    splits = {
        split: SplitData(utility_frame, features, split=split, arms=arms, scaler=scaler)
        for split in SPLIT_LABELS
    }
    train, validation = splits["train"], splits["validation"]

    rng = np.random.default_rng(seed)
    stream_order = rng.permutation(len(train.prompt_ids))

    xgboost_config, xgboost_records = tune_xgboost(train, validation, arms=arms, seed=seed)
    xgboost_router = XGBoostRouter(arms, config=xgboost_config, seed=seed)
    xgboost_router.fit(train.contexts, train.rewards)

    linucb_alpha, linucb_records = tune_linucb(
        train, validation, arms=arms, stream_order=stream_order
    )
    linucb_router = LinUCBRouter(arms, alpha=linucb_alpha)
    regrets = linucb_router.replay(train.contexts[stream_order], train.rewards[stream_order])

    results: list[dict[str, object]] = []
    try:
        for split in SPLIT_LABELS:
            data = splits[split]
            for policy, router in (
                (XGBOOST_POLICY, xgboost_router),
                (LINUCB_POLICY, linucb_router),
            ):
                results.append(
                    selection_metrics(
                        utility_frame,
                        split=split,
                        prompt_ids=data.prompt_ids,
                        chosen_arms=router.route(data.contexts),
                        policy=policy,
                    )
                )
    finally:
        xgboost_router.close()

    output_directory.mkdir(parents=True, exist_ok=True)
    results_path = output_directory / "router_results.csv"
    pd.DataFrame(results).to_csv(results_path, index=False, lineterminator="\n")

    regret_frame = pd.DataFrame(
        {
            "step": np.arange(1, len(regrets) + 1),
            "prompt_id": [train.prompt_ids[int(index)] for index in stream_order],
            "regret": regrets,
            "cumulative_regret": np.cumsum(regrets),
        }
    )
    regret_frame.to_csv(output_directory / "linucb_regret.csv", index=False, lineterminator="\n")

    (output_directory / "xgboost_search.json").write_text(
        json.dumps(
            {"grid": xgboost_records, "selected": asdict(xgboost_config)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_directory / "linucb_search.json").write_text(
        json.dumps(
            {"grid": linucb_records, "selected_alpha": linucb_alpha}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "step": "step6_learned_routers",
        "seed": seed,
        "arms": list(arms),
        "feature_columns": list(feature_columns),
        "policies": [XGBOOST_POLICY, LINUCB_POLICY],
        "hyperparameter_selection_split": "validation",
        "xgboost_selected": asdict(xgboost_config),
        "linucb_selected_alpha": linucb_alpha,
        "linucb_stream": "seeded shuffle of training prompts",
        "prompts": int(features["prompt_id"].nunique()),
        "evidence_label": evidence_label,
    }
    (output_directory / "step6_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return results_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the XGBoost and LinUCB routers and evaluate them per split."
    )
    parser.add_argument("--utility-dataset", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/experiment.yaml"))
    parser.add_argument("--models", type=Path, default=Path("config/models.yaml"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--evidence-label", default="DATOS EXPERIMENTALES")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config = load_experiment_config(args.config, models_path=args.models)
    output = run_learn_pipeline(
        args.utility_dataset,
        args.features,
        args.output_directory,
        config=config,
        seed=args.seed,
        evidence_label=args.evidence_label,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
