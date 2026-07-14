from __future__ import annotations

import pandas as pd
import pytest

from better_router_adaptive.config import SplitRatios
from better_router_adaptive.split import (
    SPLIT_COLUMN,
    SPLIT_LABELS,
    SplitError,
    assert_split_integrity,
    assign_prompt_splits,
    attach_split_column,
)

TASK_GROUPS = ("coding", "mathematics", "reasoning", "general")
RATIOS = SplitRatios(train=0.70, validation=0.15, test=0.15)


def _prompt_table(prompts_per_group: int) -> pd.DataFrame:
    rows = [
        {"prompt_id": f"{group}-{index:03d}", "task_group": group}
        for group in TASK_GROUPS
        for index in range(prompts_per_group)
    ]
    return pd.DataFrame(rows)


def test_assign_prompt_splits_is_disjoint_and_exhaustive() -> None:
    prompts = _prompt_table(prompts_per_group=20)

    assignment = assign_prompt_splits(prompts, ratios=RATIOS, seed=42)

    assert len(assignment) == len(prompts)
    assert not assignment["prompt_id"].duplicated().any()
    assert set(assignment["prompt_id"]) == set(prompts["prompt_id"])
    assert set(assignment[SPLIT_COLUMN]) == set(SPLIT_LABELS)


def test_assign_prompt_splits_follows_target_ratios_per_task_group() -> None:
    assignment = assign_prompt_splits(_prompt_table(prompts_per_group=20), ratios=RATIOS, seed=42)

    counts = assignment.groupby(["task_group", SPLIT_COLUMN], sort=True).size()
    for group in TASK_GROUPS:
        assert counts[(group, "train")] == 14
        assert counts[(group, "validation")] == 3
        assert counts[(group, "test")] == 3


def test_assign_prompt_splits_is_deterministic_per_seed() -> None:
    prompts = _prompt_table(prompts_per_group=20)

    first = assign_prompt_splits(prompts, ratios=RATIOS, seed=42)
    second = assign_prompt_splits(prompts, ratios=RATIOS, seed=42)
    different_seed = assign_prompt_splits(prompts, ratios=RATIOS, seed=123)

    pd.testing.assert_frame_equal(first, second)
    assert not first[SPLIT_COLUMN].equals(different_seed[SPLIT_COLUMN])


def test_assign_prompt_splits_keeps_every_split_non_empty_in_small_groups() -> None:
    assignment = assign_prompt_splits(_prompt_table(prompts_per_group=3), ratios=RATIOS, seed=42)

    counts = assignment.groupby(["task_group", SPLIT_COLUMN], sort=True).size()
    for group in TASK_GROUPS:
        for label in SPLIT_LABELS:
            assert counts[(group, label)] == 1


def test_assign_prompt_splits_rejects_groups_below_the_minimum() -> None:
    with pytest.raises(SplitError, match="cannot cover"):
        assign_prompt_splits(_prompt_table(prompts_per_group=2), ratios=RATIOS, seed=42)


def test_assign_prompt_splits_rejects_duplicated_prompt_ids() -> None:
    prompts = pd.concat([_prompt_table(3), _prompt_table(3)], ignore_index=True)

    with pytest.raises(SplitError, match="duplicated prompt IDs"):
        assign_prompt_splits(prompts, ratios=RATIOS, seed=42)


def test_assign_prompt_splits_rejects_missing_columns() -> None:
    with pytest.raises(SplitError, match="missing required column"):
        assign_prompt_splits(pd.DataFrame({"prompt_id": ["p-1"]}), ratios=RATIOS, seed=42)


def test_attach_split_column_places_all_rows_of_a_prompt_in_one_split() -> None:
    prompts = _prompt_table(prompts_per_group=5)
    assignment = assign_prompt_splits(prompts, ratios=RATIOS, seed=42)
    canonical = pd.DataFrame(
        {
            "prompt_id": prompts["prompt_id"].repeat(2).reset_index(drop=True),
            "model_id": ["arm-fast", "arm-premium"] * len(prompts),
        }
    )

    with_split = attach_split_column(canonical, assignment)

    assert len(with_split) == len(canonical)
    splits_per_prompt = with_split.groupby("prompt_id")[SPLIT_COLUMN].nunique()
    assert (splits_per_prompt == 1).all()


def test_attach_split_column_rejects_unassigned_and_unknown_prompts() -> None:
    prompts = _prompt_table(prompts_per_group=3)
    assignment = assign_prompt_splits(prompts, ratios=RATIOS, seed=42)
    canonical = pd.DataFrame({"prompt_id": ["nunca-asignado"], "model_id": ["arm-fast"]})

    with pytest.raises(SplitError, match="without split assignment"):
        attach_split_column(canonical, assignment)

    smaller = pd.DataFrame({"prompt_id": prompts["prompt_id"].iloc[:1], "model_id": ["arm-fast"]})
    with pytest.raises(SplitError, match="unknown prompts"):
        attach_split_column(smaller, assignment)


def test_assert_split_integrity_detects_violations() -> None:
    duplicated = pd.DataFrame(
        {
            "prompt_id": ["p-1", "p-1", "p-2"],
            "task_group": ["coding"] * 3,
            SPLIT_COLUMN: ["train", "test", "validation"],
        }
    )
    with pytest.raises(SplitError, match="more than one split"):
        assert_split_integrity(duplicated)

    mislabeled = pd.DataFrame(
        {
            "prompt_id": ["p-1", "p-2", "p-3"],
            "task_group": ["coding"] * 3,
            SPLIT_COLUMN: ["train", "validation", "holdout"],
        }
    )
    with pytest.raises(SplitError, match="unexpected split labels"):
        assert_split_integrity(mislabeled)

    incomplete = pd.DataFrame(
        {
            "prompt_id": ["p-1", "p-2"],
            "task_group": ["coding"] * 2,
            SPLIT_COLUMN: ["train", "validation"],
        }
    )
    with pytest.raises(SplitError, match="empty splits"):
        assert_split_integrity(incomplete)
