"""Deterministic cleaning for canonical routing outcomes."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from better_router_adaptive.data.schema import (
    CANONICAL_COLUMNS,
    SchemaError,
    validate_canonical_dataframe,
)

_STRING_COLUMNS: Final = (
    "prompt_id",
    "prompt_text",
    "dataset",
    "task_group",
    "model_id",
    "data_origin",
)
_NUMERIC_COLUMNS: Final = (
    "quality",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "latency_ms",
)
_TRUE_STRINGS: Final = {"1", "true", "yes", "y"}
_FALSE_STRINGS: Final = {"0", "false", "no", "n"}


class CleaningError(ValueError):
    """Raised when cleaning cannot resolve an integrity conflict safely."""


@dataclass(frozen=True, slots=True)
class CleaningReport:
    """Audit counts produced by one deterministic cleaning run."""

    rows_before: int
    exact_duplicates_removed: int
    invalid_rows_removed: int
    incomplete_prompt_rows_removed: int
    rows_after: int
    prompt_groups_before: int
    prompt_groups_after: int


def _normalize_success(value: object) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    return None


def _assert_no_conflicting_duplicates(frame: pd.DataFrame) -> None:
    duplicates = frame[frame.duplicated(["prompt_id", "model_id"], keep=False)]
    for key, group in duplicates.groupby(["prompt_id", "model_id"], sort=True):
        if len(group.drop_duplicates()) > 1:
            raise CleaningError(f"conflicting duplicate prompt/model key: {key[0]}/{key[1]}")


def _assert_prompt_metadata_consistency(frame: pd.DataFrame) -> None:
    columns = ["prompt_text", "dataset", "task_group"]
    counts = frame.groupby("prompt_id", sort=False)[columns].nunique(dropna=False)
    inconsistent = counts[(counts > 1).any(axis=1)]
    if not inconsistent.empty:
        preview = ", ".join(str(item) for item in inconsistent.index[:5])
        raise CleaningError(f"inconsistent prompt metadata for: {preview}")


def clean_canonical_dataframe(
    frame: pd.DataFrame,
    *,
    required_model_arms: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean a canonical dataframe without silently repairing target values."""

    missing = [column for column in CANONICAL_COLUMNS if column not in frame.columns]
    if missing:
        raise CleaningError(f"missing canonical columns: {', '.join(missing)}")

    working = frame.loc[:, list(CANONICAL_COLUMNS)].copy()
    rows_before = len(working)
    prompt_groups_before = int(working["prompt_id"].nunique(dropna=True))

    before_dedup = len(working)
    working = working.drop_duplicates(ignore_index=True)
    exact_duplicates_removed = before_dedup - len(working)
    _assert_no_conflicting_duplicates(working)

    for column in _STRING_COLUMNS:
        working[column] = working[column].astype("string").str.strip()

    for column in _NUMERIC_COLUMNS:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    working["success"] = working["success"].map(_normalize_success).astype("boolean")

    invalid = pd.Series(False, index=working.index)
    for column in _STRING_COLUMNS:
        invalid |= working[column].isna() | working[column].eq("")

    quality = working["quality"]
    invalid |= quality.isna() | ~np.isfinite(quality) | (quality < 0) | (quality > 1)

    for column in ("input_tokens", "output_tokens"):
        values = working[column]
        non_null = values.notna()
        invalid |= non_null & (~np.isfinite(values) | (values < 0) | ((values % 1) != 0))

    for column in ("cost_usd", "latency_ms"):
        values = working[column]
        non_null = values.notna()
        invalid |= non_null & (~np.isfinite(values) | (values < 0))

    invalid |= working["success"].isna()
    invalid_rows_removed = int(invalid.sum())
    working = working.loc[~invalid].copy()
    working["success"] = working["success"].astype(bool)

    _assert_prompt_metadata_consistency(working)

    incomplete_prompt_rows_removed = 0
    if required_model_arms is not None:
        expected = set(required_model_arms)
        model_sets = working.groupby("prompt_id", sort=True)["model_id"].agg(
            lambda values: set(values.astype(str))
        )
        incomplete_ids = model_sets[model_sets != expected].index
        incomplete_mask = working["prompt_id"].isin(incomplete_ids)
        incomplete_prompt_rows_removed = int(incomplete_mask.sum())
        working = working.loc[~incomplete_mask].copy()

    working["input_tokens"] = working["input_tokens"].astype("Int64")
    working["output_tokens"] = working["output_tokens"].astype("Int64")
    working = working.sort_values(["prompt_id", "model_id"], kind="mergesort").reset_index(
        drop=True
    )

    try:
        validate_canonical_dataframe(working, required_model_arms=required_model_arms)
    except SchemaError as exc:
        raise CleaningError(str(exc)) from exc

    report = CleaningReport(
        rows_before=rows_before,
        exact_duplicates_removed=exact_duplicates_removed,
        invalid_rows_removed=invalid_rows_removed,
        incomplete_prompt_rows_removed=incomplete_prompt_rows_removed,
        rows_after=len(working),
        prompt_groups_before=prompt_groups_before,
        prompt_groups_after=int(working["prompt_id"].nunique()),
    )
    return working, report


def write_cleaning_report(report: CleaningReport, directory: Path) -> tuple[Path, Path]:
    """Write the cleaning audit in JSON and two-column CSV formats."""

    directory.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    json_path = directory / "data_quality_report.json"
    csv_path = directory / "data_quality_report.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["metric", "value"])
        writer.writerows(payload.items())
    return json_path, csv_path
