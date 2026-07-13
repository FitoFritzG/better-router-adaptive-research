from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from better_router_adaptive.data.clean import (
    CleaningError,
    clean_canonical_dataframe,
    write_cleaning_report,
)
from better_router_adaptive.data.convert import (
    ConversionError,
    convert_routerbench_wide_dataframe,
    convert_verified_pickle,
)
from better_router_adaptive.data.profile import generate_profile_assets
from better_router_adaptive.data.schema import (
    CANONICAL_COLUMNS,
    SchemaError,
    validate_canonical_dataframe,
)

MODEL_ARMS = ("arm-fast", "arm-balanced", "arm-reasoning", "arm-premium")
TASK_GROUPS = ("coding", "mathematics", "reasoning", "general")


def _valid_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for prompt_index, task_group in enumerate(TASK_GROUPS, start=1):
        for model_index, model_id in enumerate(MODEL_ARMS, start=1):
            rows.append(
                {
                    "prompt_id": f"p-{prompt_index}",
                    "prompt_text": f"Prompt {prompt_index}",
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


def test_validate_canonical_dataframe_accepts_valid_long_format() -> None:
    frame = _valid_rows()

    validate_canonical_dataframe(
        frame,
        required_model_arms=MODEL_ARMS,
        allowed_task_groups=TASK_GROUPS,
    )


def test_validate_canonical_dataframe_rejects_conflicting_duplicate_key() -> None:
    frame = _valid_rows()
    duplicate = frame.iloc[[0]].copy()
    duplicate.loc[:, "quality"] = 0.01
    frame = pd.concat([frame, duplicate], ignore_index=True)

    with pytest.raises(SchemaError, match="conflicting duplicate"):
        validate_canonical_dataframe(frame)


def test_cleaning_removes_invalid_rows_exact_duplicates_and_incomplete_prompts() -> None:
    frame = _valid_rows()
    exact_duplicate = frame.iloc[[0]].copy()
    invalid_prompt = frame[frame["prompt_id"] == "p-1"].copy()
    invalid_prompt.loc[invalid_prompt["model_id"] == "arm-balanced", "cost_usd"] = -1.0
    incomplete = frame[frame["prompt_id"] == "p-4"].iloc[:3].copy()
    frame = pd.concat(
        [
            invalid_prompt,
            frame[frame["prompt_id"].isin(["p-2", "p-3"])],
            incomplete,
            exact_duplicate,
        ],
        ignore_index=True,
    )

    cleaned, report = clean_canonical_dataframe(frame, required_model_arms=MODEL_ARMS)

    assert len(cleaned) == 8
    assert set(cleaned["prompt_id"]) == {"p-2", "p-3"}
    assert report.rows_before == 16
    assert report.exact_duplicates_removed == 1
    assert report.invalid_rows_removed == 1
    assert report.incomplete_prompt_rows_removed == 6
    assert report.rows_after == 8


def test_cleaning_rejects_prompt_metadata_conflict() -> None:
    frame = _valid_rows()
    frame.loc[1, "prompt_text"] = "Different prompt text"

    with pytest.raises(CleaningError, match="inconsistent prompt metadata"):
        clean_canonical_dataframe(frame, required_model_arms=MODEL_ARMS)


def test_cleaning_report_is_written_as_json_and_csv(tmp_path: Path) -> None:
    cleaned, report = clean_canonical_dataframe(_valid_rows(), required_model_arms=MODEL_ARMS)

    json_path, csv_path = write_cleaning_report(report, tmp_path)

    assert cleaned.shape == (16, len(CANONICAL_COLUMNS))
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["rows_before"] == 16
    assert payload["rows_after"] == 16
    assert csv_path.read_text(encoding="utf-8").startswith("metric,value\n")


def test_routerbench_wide_dataframe_converts_to_canonical_long_format() -> None:
    wide = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "prompt": ["Solve 2+2", "Write a loop"],
            "eval_name": ["gsm8k", "mbpp"],
            "model-a": [1.0, 0.0],
            "model-b": [0.5, 1.0],
            "model-a|total_cost": [0.001, 0.0012],
            "model-b|total_cost": [0.004, 0.0042],
            "model-a|model_response": ["4", "bad"],
            "model-b|model_response": ["four", "for i in range(3): pass"],
        }
    )

    long_frame = convert_routerbench_wide_dataframe(wide)

    assert tuple(long_frame.columns) == CANONICAL_COLUMNS
    assert len(long_frame) == 4
    assert set(long_frame["model_id"]) == {"model-a", "model-b"}
    assert set(long_frame["task_group"]) == {"mathematics", "coding"}
    assert long_frame["latency_ms"].isna().all()
    assert long_frame["input_tokens"].isna().all()
    assert long_frame["output_tokens"].isna().all()
    assert long_frame["success"].all()
    assert "model_response" not in long_frame.columns


def test_verified_pickle_conversion_rejects_wrong_checksum_before_loading(tmp_path: Path) -> None:
    source = tmp_path / "input.pkl"
    output = tmp_path / "output.csv.gz"
    _valid_rows().to_pickle(source)

    with pytest.raises(ConversionError, match="checksum mismatch"):
        convert_verified_pickle(source, output, expected_sha256="0" * 64)

    assert not output.exists()


def test_verified_pickle_conversion_writes_non_executable_csv(tmp_path: Path) -> None:
    source = tmp_path / "input.pkl"
    output = tmp_path / "output.csv.gz"
    frame = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "prompt": ["Solve 2+2"],
            "eval_name": ["gsm8k"],
            "model-a": [1.0],
            "model-b": [0.5],
            "model-a|total_cost": [0.001],
            "model-b|total_cost": [0.004],
        }
    )
    frame.to_pickle(source)
    expected = hashlib.sha256(source.read_bytes()).hexdigest()

    result = convert_verified_pickle(source, output, expected_sha256=expected, timeout_seconds=30)

    assert result == output
    converted = pd.read_csv(output)
    assert len(converted) == 2
    assert tuple(converted.columns) == CANONICAL_COLUMNS
    assert not list(tmp_path.glob("*.part"))


def test_profile_assets_are_generated_and_marked_as_synthetic(tmp_path: Path) -> None:
    output = generate_profile_assets(_valid_rows(), tmp_path, evidence_label="DEMO SINTÉTICA")

    assert {path.name for path in output} == {
        "distribucion_tareas.svg",
        "calidad_costo_modelos.svg",
        "latencia_modelos.svg",
        "resumen_perfil.json",
    }
    summary = json.loads((tmp_path / "resumen_perfil.json").read_text(encoding="utf-8"))
    assert summary["evidence_label"] == "DEMO SINTÉTICA"
    assert summary["rows"] == 16
    assert all(path.stat().st_size > 0 for path in output)
