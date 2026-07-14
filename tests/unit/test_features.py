from __future__ import annotations

import pandas as pd
import pytest

from better_router_adaptive.data.schema import CANONICAL_COLUMNS
from better_router_adaptive.features import (
    OUTCOME_COLUMNS,
    FeatureError,
    LeakageError,
    assert_no_outcome_columns,
    build_prompt_features,
    feature_column_names,
)

MODEL_ARMS = ("arm-fast", "arm-balanced", "arm-reasoning", "arm-premium")
TASK_GROUPS = ("coding", "mathematics", "reasoning", "general")


def _canonical_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for prompt_index, task_group in enumerate(TASK_GROUPS, start=1):
        for model_index, model_id in enumerate(MODEL_ARMS, start=1):
            rows.append(
                {
                    "prompt_id": f"p-{prompt_index}",
                    "prompt_text": f"Texto del prompt número {prompt_index}",
                    "dataset": f"dataset-{task_group}",
                    "task_group": task_group,
                    "model_id": model_id,
                    "quality": 0.5 + model_index * 0.1,
                    "input_tokens": 20 + prompt_index,
                    "output_tokens": 30 + model_index,
                    "cost_usd": 0.001 * model_index,
                    "latency_ms": 100.0 * model_index,
                    "success": True,
                    "data_origin": "synthetic_test_fixture_not_experimental_evidence",
                }
            )
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def test_build_prompt_features_produces_one_leakage_free_row_per_prompt() -> None:
    features = build_prompt_features(_canonical_frame(), task_groups=TASK_GROUPS)

    assert list(features["prompt_id"]) == ["p-1", "p-2", "p-3", "p-4"]
    expected_columns = ("prompt_id", "task_group", *feature_column_names(TASK_GROUPS))
    assert tuple(features.columns) == expected_columns
    assert not set(features.columns) & set(OUTCOME_COLUMNS)


def test_build_prompt_features_computes_text_statistics() -> None:
    features = build_prompt_features(_canonical_frame(), task_groups=TASK_GROUPS)

    first = features.iloc[0]
    text = "Texto del prompt número 1"
    assert first["prompt_char_count"] == len(text)
    assert first["prompt_word_count"] == len(text.split())
    assert first["prompt_avg_word_length"] == pytest.approx(len(text) / len(text.split()))


def test_build_prompt_features_encodes_task_group_as_one_hot() -> None:
    features = build_prompt_features(_canonical_frame(), task_groups=TASK_GROUPS)

    one_hot_columns = [f"task_group_{group}" for group in TASK_GROUPS]
    assert (features[one_hot_columns].sum(axis=1) == 1).all()
    for group in TASK_GROUPS:
        row = features[features["task_group"] == group].iloc[0]
        assert row[f"task_group_{group}"] == 1


def test_build_prompt_features_is_deterministic() -> None:
    frame = _canonical_frame()

    first = build_prompt_features(frame, task_groups=TASK_GROUPS)
    second = build_prompt_features(frame, task_groups=TASK_GROUPS)

    pd.testing.assert_frame_equal(first, second)


def test_build_prompt_features_rejects_missing_columns() -> None:
    frame = _canonical_frame().drop(columns=["prompt_text"])

    with pytest.raises(FeatureError, match="missing required columns"):
        build_prompt_features(frame, task_groups=TASK_GROUPS)


def test_build_prompt_features_rejects_blank_prompt_text() -> None:
    frame = _canonical_frame()
    frame.loc[0, "prompt_text"] = "   "

    with pytest.raises(FeatureError, match="null or blank"):
        build_prompt_features(frame, task_groups=TASK_GROUPS)


def test_build_prompt_features_rejects_inconsistent_prompt_metadata() -> None:
    frame = _canonical_frame()
    frame.loc[0, "prompt_text"] = "Texto distinto para la misma clave"

    with pytest.raises(FeatureError, match="inconsistent prompt metadata"):
        build_prompt_features(frame, task_groups=TASK_GROUPS)


def test_build_prompt_features_rejects_unexpected_task_groups() -> None:
    frame = _canonical_frame()

    with pytest.raises(FeatureError, match="unexpected task groups"):
        build_prompt_features(frame, task_groups=("coding", "mathematics"))


def test_assert_no_outcome_columns_detects_leaked_outcomes() -> None:
    contaminated = pd.DataFrame({"prompt_id": ["p-1"], "quality": [0.9], "cost_usd": [0.1]})

    with pytest.raises(LeakageError, match="quality, cost_usd"):
        assert_no_outcome_columns(contaminated)


def test_feature_column_names_lists_numeric_and_one_hot_columns() -> None:
    columns = feature_column_names(("coding", "general"))

    assert columns == (
        "prompt_char_count",
        "prompt_word_count",
        "prompt_avg_word_length",
        "task_group_coding",
        "task_group_general",
    )
