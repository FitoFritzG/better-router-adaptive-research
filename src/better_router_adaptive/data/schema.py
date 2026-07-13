"""Canonical long-format schema for multi-LLM routing experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

CANONICAL_COLUMNS = (
    "prompt_id",
    "prompt_text",
    "dataset",
    "task_group",
    "model_id",
    "quality",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "latency_ms",
    "success",
    "data_origin",
)

_REQUIRED_STRING_COLUMNS = (
    "prompt_id",
    "prompt_text",
    "dataset",
    "task_group",
    "model_id",
    "data_origin",
)
_NUMERIC_COLUMNS = (
    "quality",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "latency_ms",
)
_PROMPT_METADATA_COLUMNS = ("prompt_text", "dataset", "task_group")


class SchemaError(ValueError):
    """Raised when a dataframe violates the canonical research schema."""


def _conflicting_duplicate_keys(frame: pd.DataFrame) -> list[tuple[str, str]]:
    duplicate_rows = frame[frame.duplicated(["prompt_id", "model_id"], keep=False)]
    conflicts: list[tuple[str, str]] = []
    for key, group in duplicate_rows.groupby(["prompt_id", "model_id"], sort=True):
        if len(group.drop_duplicates()) > 1:
            conflicts.append((str(key[0]), str(key[1])))
    return conflicts


def validate_canonical_dataframe(
    frame: pd.DataFrame,
    *,
    required_model_arms: Sequence[str] | None = None,
    allowed_task_groups: Sequence[str] | None = None,
) -> None:
    """Validate canonical columns, ranges, key uniqueness, and prompt consistency."""

    missing = [column for column in CANONICAL_COLUMNS if column not in frame.columns]
    if missing:
        raise SchemaError(f"missing canonical columns: {', '.join(missing)}")

    conflicts = _conflicting_duplicate_keys(frame)
    if conflicts:
        preview = ", ".join(f"{prompt}/{model}" for prompt, model in conflicts[:5])
        raise SchemaError(f"conflicting duplicate prompt/model keys: {preview}")

    for column in _REQUIRED_STRING_COLUMNS:
        values = frame[column]
        invalid = values.isna() | values.astype("string").str.strip().eq("")
        if bool(invalid.any()):
            raise SchemaError(f"column {column!r} contains null or blank values")

    numeric: dict[str, pd.Series[Any]] = {}
    for column in _NUMERIC_COLUMNS:
        numeric[column] = pd.to_numeric(frame[column], errors="coerce")

    quality = numeric["quality"]
    if bool(quality.isna().any()) or bool((~np.isfinite(quality)).any()):
        raise SchemaError("quality must contain finite numeric values")
    if bool(((quality < 0) | (quality > 1)).any()):
        raise SchemaError("quality must be within [0, 1]")

    for column in ("input_tokens", "output_tokens"):
        values = numeric[column]
        non_null = values.dropna()
        if bool((~np.isfinite(non_null)).any()) or bool((non_null < 0).any()):
            raise SchemaError(f"{column} must be null or a finite non-negative value")
        if bool(((non_null % 1) != 0).any()):
            raise SchemaError(f"{column} must contain whole token counts")

    for column in ("cost_usd", "latency_ms"):
        values = numeric[column].dropna()
        if bool((~np.isfinite(values)).any()) or bool((values < 0).any()):
            raise SchemaError(f"{column} must be null or a finite non-negative value")

    success = frame["success"]
    valid_success = success.map(lambda value: isinstance(value, (bool, np.bool_))).all()
    if bool(success.isna().any()) or not valid_success:
        raise SchemaError("success must contain boolean values")

    metadata_columns = list(_PROMPT_METADATA_COLUMNS)
    metadata_counts = frame.groupby("prompt_id", sort=False)[metadata_columns].nunique(dropna=False)
    if bool((metadata_counts > 1).any(axis=None)):
        raise SchemaError("inconsistent prompt metadata within a prompt_id")

    if allowed_task_groups is not None:
        unexpected = sorted(set(frame["task_group"].astype(str)) - set(allowed_task_groups))
        if unexpected:
            raise SchemaError(f"unexpected task groups: {', '.join(unexpected)}")

    if required_model_arms is not None:
        expected = set(required_model_arms)
        incomplete = [
            str(prompt_id)
            for prompt_id, models in frame.groupby("prompt_id", sort=True)["model_id"]
            if set(models.astype(str)) != expected
        ]
        if incomplete:
            preview = ", ".join(incomplete[:5])
            raise SchemaError(f"prompt groups do not contain the required model arms: {preview}")
