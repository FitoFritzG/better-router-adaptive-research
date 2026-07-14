"""Deterministic prompt-level train/validation/test splits without leakage."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from better_router_adaptive.config import SplitRatios

SPLIT_LABELS: Final = ("train", "validation", "test")
SPLIT_COLUMN: Final = "split"

# A stratum needs at least one prompt per split so every task group is
# represented in train, validation, and test.
_MINIMUM_GROUP_PROMPTS: Final = len(SPLIT_LABELS)


class SplitError(ValueError):
    """Raised when a safe grouped split cannot be produced."""


def _allocate_split_counts(total: int, ratios: SplitRatios) -> dict[str, int]:
    """Split ``total`` prompts into per-split counts, each split non-empty.

    Uses floor-with-minimum-one allocation followed by deterministic
    largest-remainder distribution, so realized counts follow the target
    ratios as closely as an integer partition allows.
    """

    if total < _MINIMUM_GROUP_PROMPTS:
        raise SplitError(
            f"a stratum with {total} prompts cannot cover the {len(SPLIT_LABELS)} splits"
        )

    targets = dict(
        zip(
            SPLIT_LABELS,
            (ratios.train * total, ratios.validation * total, ratios.test * total),
            strict=True,
        )
    )
    counts = {label: max(1, int(np.floor(target))) for label, target in targets.items()}

    while sum(counts.values()) > total:
        reducible = [label for label in SPLIT_LABELS if counts[label] > 1]
        label = max(reducible, key=lambda name: (counts[name] - targets[name], name))
        counts[label] -= 1

    while sum(counts.values()) < total:
        label = max(SPLIT_LABELS, key=lambda name: (targets[name] - counts[name],))
        counts[label] += 1

    return counts


def assign_prompt_splits(prompts: pd.DataFrame, *, ratios: SplitRatios, seed: int) -> pd.DataFrame:
    """Assign every prompt to exactly one split, stratified by task group.

    The unit of assignment is ``prompt_id``: all rows of a prompt land in the
    same split, which prevents outcome leakage across partitions. The
    assignment is fully determined by ``seed`` and the sorted prompt IDs.
    """

    for column in ("prompt_id", "task_group"):
        if column not in prompts.columns:
            raise SplitError(f"missing required column: {column}")

    working = prompts.loc[:, ["prompt_id", "task_group"]].copy()
    working["prompt_id"] = working["prompt_id"].astype(str)
    working["task_group"] = working["task_group"].astype(str)
    if bool(working["prompt_id"].duplicated().any()):
        duplicated = working.loc[working["prompt_id"].duplicated(), "prompt_id"]
        preview = ", ".join(duplicated.head(5))
        raise SplitError(f"duplicated prompt IDs in split input: {preview}")

    rng = np.random.default_rng(seed)
    assignments: list[pd.DataFrame] = []
    for task_group in sorted(working["task_group"].unique()):
        group_ids = np.sort(
            working.loc[working["task_group"] == task_group, "prompt_id"].to_numpy()
        )
        shuffled = group_ids[rng.permutation(len(group_ids))]
        counts = _allocate_split_counts(len(shuffled), ratios)

        start = 0
        for label in SPLIT_LABELS:
            end = start + counts[label]
            assignments.append(
                pd.DataFrame(
                    {
                        "prompt_id": shuffled[start:end],
                        "task_group": task_group,
                        SPLIT_COLUMN: label,
                    }
                )
            )
            start = end

    assignment = pd.concat(assignments, ignore_index=True)
    assignment = assignment.sort_values("prompt_id", kind="mergesort").reset_index(drop=True)
    assert_split_integrity(assignment)
    return assignment


def assert_split_integrity(assignment: pd.DataFrame) -> None:
    """Verify that splits are disjoint, exhaustive, and correctly labeled."""

    if bool(assignment["prompt_id"].duplicated().any()):
        raise SplitError("a prompt ID is assigned to more than one split")
    unexpected = sorted(set(assignment[SPLIT_COLUMN].astype(str)) - set(SPLIT_LABELS))
    if unexpected:
        raise SplitError(f"unexpected split labels: {', '.join(unexpected)}")
    missing = sorted(set(SPLIT_LABELS) - set(assignment[SPLIT_COLUMN].astype(str)))
    if missing:
        raise SplitError(f"empty splits: {', '.join(missing)}")


def attach_split_column(frame: pd.DataFrame, assignment: pd.DataFrame) -> pd.DataFrame:
    """Return the canonical dataframe with the split of each prompt attached."""

    frame_ids = set(frame["prompt_id"].astype(str))
    assignment_ids = set(assignment["prompt_id"].astype(str))
    unassigned = sorted(frame_ids - assignment_ids)
    if unassigned:
        raise SplitError(f"prompts without split assignment: {', '.join(unassigned[:5])}")
    unknown = sorted(assignment_ids - frame_ids)
    if unknown:
        raise SplitError(f"assignment contains unknown prompts: {', '.join(unknown[:5])}")

    merged = frame.merge(assignment.loc[:, ["prompt_id", SPLIT_COLUMN]], on="prompt_id", how="left")
    return merged.sort_values(["prompt_id", "model_id"], kind="mergesort").reset_index(drop=True)
