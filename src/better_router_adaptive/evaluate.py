"""End-to-end Step 7 pipeline: multi-seed evaluation with paired bootstrap.

For every locked protocol seed the pipeline re-derives the prompt-grouped
split, the train-anchored utility, the reference policies, and both learned
routers, then evaluates everything on that seed's test split. Confidence
intervals come from a paired bootstrap that resamples ``prompt_id`` clusters,
so every policy is always compared on exactly the same resampled prompts.
All tables and figures are generated from artifacts, never written by hand.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from better_router_adaptive.config import ExperimentConfig, load_experiment_config
from better_router_adaptive.data.schema import validate_canonical_dataframe
from better_router_adaptive.features import build_prompt_features
from better_router_adaptive.learn import (
    LINUCB_POLICY,
    XGBOOST_POLICY,
    SplitData,
    tune_linucb,
    tune_xgboost,
)
from better_router_adaptive.policies import (
    ORACLE_POLICY,
    RULES_POLICY,
    apply_rules_policy,
    evaluate_selection,
    fit_better_rules_proxy,
    fixed_arm_selection,
    oracle_selection,
)
from better_router_adaptive.routers import (
    FeatureScaler,
    LinUCBRouter,
    RouterError,
    XGBoostRouter,
)
from better_router_adaptive.split import (
    SPLIT_COLUMN,
    assign_prompt_splits,
    attach_split_column,
)
from better_router_adaptive.utility import (
    UTILITY_COLUMN,
    compute_normalization_stats,
    compute_utility,
)

_MANIFEST_SCHEMA_VERSION = "1.0.0"
DEFAULT_BOOTSTRAP_SAMPLES: Final = 2000


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def paired_bootstrap_summary(
    prompt_utilities: pd.DataFrame,
    *,
    baseline: str,
    samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Paired bootstrap over prompt-level utilities, one column per policy.

    ``prompt_utilities`` must be indexed by ``prompt_id`` with one utility
    column per policy. Every bootstrap resample draws prompt IDs with
    replacement and evaluates all policies on the same resampled prompts,
    which is what makes the intervals paired.
    """

    if baseline not in prompt_utilities.columns:
        raise RouterError(f"baseline policy {baseline!r} missing from utilities table")
    if prompt_utilities.empty:
        raise RouterError("cannot bootstrap an empty utilities table")

    values = prompt_utilities.to_numpy(dtype=np.float64)
    policies = list(prompt_utilities.columns)
    n_prompts = values.shape[0]
    baseline_position = policies.index(baseline)

    # Resample in batches to keep memory flat on the real dataset
    # (36k prompts x 2000 samples would not fit as one gathered array).
    batch_size = max(1, min(samples, 50_000_000 // max(1, n_prompts * len(policies))))
    mean_batches: list[npt.NDArray[np.float64]] = []
    remaining = samples
    while remaining > 0:
        batch = min(batch_size, remaining)
        indices = rng.integers(0, n_prompts, size=(batch, n_prompts))
        mean_batches.append(values[indices].mean(axis=1))
        remaining -= batch
    resampled_means = np.concatenate(mean_batches, axis=0)  # (samples, n_policies)
    differences = resampled_means - resampled_means[:, [baseline_position]]

    rows: list[dict[str, object]] = []
    for position, policy in enumerate(policies):
        rows.append(
            {
                "policy": policy,
                "prompts": n_prompts,
                "mean_utility": float(values[:, position].mean()),
                "ci95_low": float(np.percentile(resampled_means[:, position], 2.5)),
                "ci95_high": float(np.percentile(resampled_means[:, position], 97.5)),
                "diff_vs_baseline": float(
                    values[:, position].mean() - values[:, baseline_position].mean()
                ),
                "diff_ci95_low": float(np.percentile(differences[:, position], 2.5)),
                "diff_ci95_high": float(np.percentile(differences[:, position], 97.5)),
            }
        )
    return pd.DataFrame(rows)


def _evaluate_one_seed(
    frame: pd.DataFrame,
    *,
    config: ExperimentConfig,
    seed: int,
) -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    """Run steps 4-6 in memory for one seed; return metrics, utilities, regret."""

    arms = tuple(sorted(model.model_id for model in config.models))

    features = build_prompt_features(frame, task_groups=config.task_groups)
    assignment = assign_prompt_splits(
        features.loc[:, ["prompt_id", "task_group"]], ratios=config.split, seed=seed
    )
    with_split = attach_split_column(frame, assignment)
    features_split = features.merge(
        assignment.loc[:, ["prompt_id", SPLIT_COLUMN]], on="prompt_id", how="left"
    )

    stats = compute_normalization_stats(with_split.loc[with_split[SPLIT_COLUMN] == "train"])
    utility_frame = compute_utility(with_split, weights=config.reward_weights, stats=stats)

    feature_columns = tuple(
        column
        for column in features_split.columns
        if column not in ("prompt_id", "task_group", SPLIT_COLUMN)
    )
    scaler = FeatureScaler.fit(
        features_split.loc[features_split[SPLIT_COLUMN] == "train"], feature_columns
    )
    splits = {
        split: SplitData(utility_frame, features_split, split=split, arms=arms, scaler=scaler)
        for split in ("train", "validation", "test")
    }
    train, validation, test = splits["train"], splits["validation"], splits["test"]

    rules = fit_better_rules_proxy(utility_frame.loc[utility_frame[SPLIT_COLUMN] == "train"])

    rng = np.random.default_rng(seed)
    stream_order = rng.permutation(len(train.prompt_ids))

    xgboost_config, _ = tune_xgboost(train, validation, arms=arms, seed=seed)
    xgboost_router = XGBoostRouter(arms, config=xgboost_config, seed=seed)
    xgboost_router.fit(train.contexts, train.rewards)

    linucb_alpha, _ = tune_linucb(train, validation, arms=arms, stream_order=stream_order)
    linucb_router = LinUCBRouter(arms, alpha=linucb_alpha)
    regrets = linucb_router.replay(train.contexts[stream_order], train.rewards[stream_order])

    test_subset = utility_frame.loc[utility_frame[SPLIT_COLUMN] == "test"]
    selections: dict[str, pd.DataFrame] = {
        ORACLE_POLICY: oracle_selection(test_subset),
        RULES_POLICY: apply_rules_policy(test_subset, rules),
    }
    for arm in arms:
        selections[f"fixed:{arm}"] = fixed_arm_selection(test_subset, arm)
    try:
        for policy, router in (
            (XGBOOST_POLICY, xgboost_router),
            (LINUCB_POLICY, linucb_router),
        ):
            choices = pd.DataFrame(
                {"prompt_id": test.prompt_ids, "model_id": router.route(test.contexts)}
            )
            selection = test_subset.merge(choices, on=["prompt_id", "model_id"], how="inner")
            if len(selection) != len(test.prompt_ids):
                raise RouterError(f"policy {policy!r} produced choices without matching outcomes")
            selections[policy] = selection
    finally:
        xgboost_router.close()

    metric_rows: list[dict[str, object]] = []
    utility_records: list[pd.DataFrame] = []
    for policy, selection in selections.items():
        metrics = evaluate_selection(selection, policy=policy)
        metric_rows.append(
            {
                "seed": seed,
                "split": "test",
                "selected_xgboost": json.dumps(asdict(xgboost_config)),
                "selected_linucb_alpha": linucb_alpha,
                **asdict(metrics),
            }
        )
        utility_records.append(
            pd.DataFrame(
                {
                    "prompt_id": selection["prompt_id"].astype(str),
                    "seed": seed,
                    "policy": policy,
                    UTILITY_COLUMN: selection[UTILITY_COLUMN].astype(float),
                }
            )
        )

    regret_frame = pd.DataFrame(
        {
            "seed": seed,
            "step": np.arange(1, len(regrets) + 1),
            "cumulative_regret": np.cumsum(regrets),
        }
    )
    return metric_rows, pd.concat(utility_records, ignore_index=True), regret_frame


def _read_validated_frame(input_path: Path, config: ExperimentConfig) -> pd.DataFrame:
    frame = pd.read_csv(input_path)
    frame["success"] = frame["success"].astype(bool)
    validate_canonical_dataframe(
        frame,
        required_model_arms=tuple(model.model_id for model in config.models),
        allowed_task_groups=config.task_groups,
    )
    return frame


def _evaluate_seed_worker(
    input_path: str,
    config: ExperimentConfig,
    seed: int,
    output_directory: str,
) -> None:
    """Evaluate one seed in a fresh process and serialize intermediate tables."""

    frame = _read_validated_frame(Path(input_path), config)
    metric_rows, utilities, regret = _evaluate_one_seed(frame, config=config, seed=seed)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(destination / "metrics.csv", index=False, lineterminator="\n")
    utilities.to_csv(
        destination / "utilities.csv.gz", index=False, compression="gzip", lineterminator="\n"
    )
    regret.to_csv(
        destination / "regret.csv.gz", index=False, compression="gzip", lineterminator="\n"
    )


def _collect_seed_evaluations(
    input_path: Path,
    *,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run each locked seed in a spawned process to isolate native ML state."""

    metric_frames: list[pd.DataFrame] = []
    utility_frames: list[pd.DataFrame] = []
    regret_frames: list[pd.DataFrame] = []
    context = mp.get_context("spawn")

    with tempfile.TemporaryDirectory(prefix="better-router-adaptive-seeds-") as temporary:
        temporary_root = Path(temporary)
        for seed in config.seeds:
            seed_directory = temporary_root / f"seed-{seed}"
            process = context.Process(
                target=_evaluate_seed_worker,
                args=(str(input_path), config, seed, str(seed_directory)),
                name=f"better-router-seed-{seed}",
            )
            process.start()
            process.join()
            if process.exitcode != 0:
                raise RouterError(
                    f"isolated evaluation failed for seed {seed} with exit code {process.exitcode}"
                )
            metric_frames.append(pd.read_csv(seed_directory / "metrics.csv"))
            utility_frames.append(pd.read_csv(seed_directory / "utilities.csv.gz"))
            regret_frames.append(pd.read_csv(seed_directory / "regret.csv.gz"))

    return (
        pd.concat(metric_frames, ignore_index=True),
        pd.concat(utility_frames, ignore_index=True),
        pd.concat(regret_frames, ignore_index=True),
    )


def _generate_figures(
    summary: pd.DataFrame,
    per_seed: pd.DataFrame,
    regret_curves: pd.DataFrame,
    directory: Path,
    *,
    evidence_label: str,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)

    ordered = summary.sort_values("mean_utility", ascending=False)
    errors = np.array(
        [
            ordered["mean_utility"] - ordered["ci95_low"],
            ordered["ci95_high"] - ordered["mean_utility"],
        ]
    )
    plt.figure(figsize=(8, 4.5))
    plt.bar(ordered["policy"], ordered["mean_utility"], yerr=errors, capsize=4)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Utilidad media en test")
    plt.title(f"Comparación de políticas (IC 95 % bootstrap) — {evidence_label}")
    _save_figure(directory / "comparacion_politicas.svg")

    plt.figure(figsize=(8, 4.5))
    for seed, group in regret_curves.groupby("seed", sort=True):
        plt.plot(group["step"], group["cumulative_regret"], label=f"semilla {seed}")
    plt.xlabel("Paso del replay prequential")
    plt.ylabel("Regret acumulado")
    plt.title(f"LinUCB: regret acumulado — {evidence_label}")
    plt.legend()
    _save_figure(directory / "regret_linucb.svg")

    frontier = (
        per_seed.groupby("policy", as_index=False)
        .agg(mean_quality=("mean_quality", "mean"), mean_cost_usd=("mean_cost_usd", "mean"))
        .sort_values("policy")
    )
    plt.figure(figsize=(8, 4.5))
    plt.xscale("log")
    plt.scatter(frontier["mean_cost_usd"], frontier["mean_quality"])
    for record in frontier.to_dict(orient="records"):
        plt.annotate(
            str(record["policy"]),
            (float(str(record["mean_cost_usd"])), float(str(record["mean_quality"]))),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
        )
    plt.margins(x=0.3, y=0.2)
    plt.xlabel("Costo medio por consulta [USD] (escala log)")
    plt.ylabel("Calidad media")
    plt.title(f"Frontera calidad-costo en test — {evidence_label}")
    _save_figure(directory / "frontera_calidad_costo.svg")


def run_evaluation_pipeline(
    input_path: Path,
    output_directory: Path,
    *,
    config: ExperimentConfig,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    evidence_label: str,
) -> Path:
    """Evaluate every policy across the locked seeds and write the report tables."""

    # Validate once in the parent, then evaluate every seed in a fresh spawned
    # process. This prevents accumulation of native XGBoost/OpenMP state across
    # the five complete training runs.
    _read_validated_frame(input_path, config)
    per_seed, pooled, regret_curves = _collect_seed_evaluations(
        input_path,
        config=config,
    )

    # Prompt-level utilities: average over the seeds where the prompt fell in
    # test, then bootstrap prompt IDs so every policy sees the same resample.
    prompt_utilities = pooled.pivot_table(
        index="prompt_id", columns="policy", values=UTILITY_COLUMN, aggfunc="mean"
    )
    if bool(prompt_utilities.isna().any(axis=None)):
        raise RouterError("some prompts lack utilities for some policies")

    bootstrap_rng = np.random.default_rng(config.seeds[0])
    summary = paired_bootstrap_summary(
        prompt_utilities,
        baseline=RULES_POLICY,
        samples=bootstrap_samples,
        rng=bootstrap_rng,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output_directory / "evaluation_per_seed.csv", index=False, lineterminator="\n")
    summary_path = output_directory / "evaluation_summary.csv"
    summary.to_csv(summary_path, index=False, lineterminator="\n")
    regret_curves.to_csv(
        output_directory / "linucb_regret_curves.csv", index=False, lineterminator="\n"
    )

    _generate_figures(
        summary,
        per_seed,
        regret_curves,
        output_directory / "figures",
        evidence_label=evidence_label,
    )

    manifest = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "step": "step7_evaluation",
        "seeds": list(config.seeds),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_unit": "prompt_id",
        "bootstrap_baseline": RULES_POLICY,
        "policies": sorted(set(per_seed["policy"].astype(str))),
        "test_prompts_pooled": int(prompt_utilities.shape[0]),
        "evidence_label": evidence_label,
    }
    (output_directory / "step7_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-seed evaluation with paired bootstrap confidence intervals."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/experiment.yaml"))
    parser.add_argument("--models", type=Path, default=Path("config/models.yaml"))
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--evidence-label", default="DATOS EXPERIMENTALES")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config = load_experiment_config(args.config, models_path=args.models)
    output = run_evaluation_pipeline(
        args.input,
        args.output_directory,
        config=config,
        bootstrap_samples=args.bootstrap_samples,
        evidence_label=args.evidence_label,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
