"""Pre-inference prompt features with explicit information-leakage guards."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import pandas as pd

# Columns whose values only exist after running a model on the prompt. None of
# them may ever appear in a feature table used to train or evaluate a router.
# ``input_tokens`` is excluded conservatively: token counts depend on each
# model's tokenizer, so they are treated as an outcome of the model choice.
OUTCOME_COLUMNS: Final = (
    "model_id",
    "quality",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "latency_ms",
    "success",
)

_REQUIRED_INPUT_COLUMNS: Final = ("prompt_id", "prompt_text", "task_group")
_NUMERIC_FEATURE_COLUMNS: Final = (
    "prompt_char_count",
    "prompt_word_count",
    "prompt_avg_word_length",
)


class FeatureError(ValueError):
    """Raised when a feature table cannot be built safely."""


class LeakageError(FeatureError):
    """Raised when post-inference outcome columns reach a feature table."""


def feature_column_names(task_groups: Sequence[str]) -> tuple[str, ...]:
    """Return the exact model-facing feature columns for the given task groups."""

    one_hot = tuple(f"task_group_{group}" for group in task_groups)
    return _NUMERIC_FEATURE_COLUMNS + one_hot


def assert_no_outcome_columns(frame: pd.DataFrame) -> None:
    """Reject any dataframe that carries post-inference outcome columns."""

    leaked = [column for column in frame.columns if column in OUTCOME_COLUMNS]
    if leaked:
        raise LeakageError(f"outcome columns leaked into feature table: {', '.join(leaked)}")


def build_prompt_features(frame: pd.DataFrame, *, task_groups: Sequence[str]) -> pd.DataFrame:
    """Build one feature row per prompt using only pre-inference information.

    The input is a canonical long-format dataframe. Every feature is derived
    from ``prompt_text`` and ``task_group``, both known before any model runs,
    so the resulting table is safe for training and for online routing alike.
    """

    missing = [column for column in _REQUIRED_INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise FeatureError(f"missing required columns: {', '.join(missing)}")

    working = frame.loc[:, list(_REQUIRED_INPUT_COLUMNS)].copy()
    for column in _REQUIRED_INPUT_COLUMNS:
        working[column] = working[column].astype("string").str.strip()
        blank = working[column].isna() | working[column].eq("")
        if bool(blank.any()):
            raise FeatureError(f"column {column!r} contains null or blank values")

    metadata_counts = working.groupby("prompt_id", sort=False)[
        ["prompt_text", "task_group"]
    ].nunique(dropna=False)
    if bool((metadata_counts > 1).any(axis=None)):
        inconsistent = metadata_counts[(metadata_counts > 1).any(axis=1)]
        preview = ", ".join(str(item) for item in inconsistent.index[:5])
        raise FeatureError(f"inconsistent prompt metadata for: {preview}")

    unexpected = sorted(set(working["task_group"]) - set(task_groups))
    if unexpected:
        raise FeatureError(f"unexpected task groups: {', '.join(unexpected)}")

    prompts = working.drop_duplicates("prompt_id", ignore_index=True)
    char_count = prompts["prompt_text"].str.len().astype("int64")
    word_count = prompts["prompt_text"].str.split().str.len().astype("int64")

    features = pd.DataFrame(
        {
            "prompt_id": prompts["prompt_id"].astype(str),
            "task_group": prompts["task_group"].astype(str),
            "prompt_char_count": char_count,
            "prompt_word_count": word_count,
            "prompt_avg_word_length": (char_count / word_count).astype("float64"),
        }
    )
    for group in task_groups:
        features[f"task_group_{group}"] = (features["task_group"] == group).astype("int64")

    features = features.sort_values("prompt_id", kind="mergesort").reset_index(drop=True)
    assert_no_outcome_columns(features)
    return features
