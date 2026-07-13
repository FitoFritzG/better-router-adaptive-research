from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from better_router_adaptive.data import pickle_worker, pipeline
from better_router_adaptive.data.clean import CleaningError, clean_canonical_dataframe
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

ARMS = ("arm-fast", "arm-balanced", "arm-reasoning", "arm-premium")


def _canonical_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_id in ARMS:
        rows.append(
            {
                "prompt_id": "p-1",
                "prompt_text": "Prompt",
                "dataset": "synthetic",
                "task_group": "general",
                "model_id": model_id,
                "quality": 0.8,
                "input_tokens": 10,
                "output_tokens": 20,
                "cost_usd": 0.001,
                "latency_ms": 100.0,
                "success": True,
                "data_origin": "synthetic",
            }
        )
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def _wide_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["s1"],
            "prompt": ["Solve"],
            "eval_name": ["gsm8k"],
            "model-a": [1.0],
            "model-a|total_cost": [0.001],
        }
    )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("prompt_text", "", "null or blank"),
        ("quality", 1.2, "within"),
        ("input_tokens", 1.5, "whole token"),
        ("cost_usd", -0.1, "non-negative"),
        ("success", "yes", "boolean"),
    ],
)
def test_schema_rejects_invalid_fields(column: str, value: object, message: str) -> None:
    frame = _canonical_frame()
    remaining = frame[column].iloc[1:].tolist()
    frame[column] = [value, *remaining]

    with pytest.raises(SchemaError, match=message):
        validate_canonical_dataframe(frame)


def test_schema_rejects_missing_column_unexpected_group_and_incomplete_arms() -> None:
    frame = _canonical_frame().drop(columns=["dataset"])
    with pytest.raises(SchemaError, match="missing canonical"):
        validate_canonical_dataframe(frame)

    frame = _canonical_frame()
    frame.loc[:, "task_group"] = "other"
    with pytest.raises(SchemaError, match="unexpected task groups"):
        validate_canonical_dataframe(frame, allowed_task_groups=("general",))

    with pytest.raises(SchemaError, match="required model arms"):
        validate_canonical_dataframe(_canonical_frame().iloc[:3], required_model_arms=ARMS)


def test_cleaning_rejects_missing_columns_and_normalizes_success_values() -> None:
    with pytest.raises(CleaningError, match="missing canonical"):
        clean_canonical_dataframe(_canonical_frame().drop(columns=["quality"]))

    frame = _canonical_frame()
    frame["success"] = frame["success"].astype(object)
    frame.loc[0, "success"] = "yes"
    frame.loc[1, "success"] = 1
    frame.loc[2, "success"] = "no"
    frame.loc[3, "success"] = 0
    cleaned, _ = clean_canonical_dataframe(frame, required_model_arms=ARMS)
    success_by_model = dict(zip(cleaned["model_id"], cleaned["success"], strict=True))
    assert success_by_model == {
        "arm-fast": True,
        "arm-balanced": True,
        "arm-reasoning": False,
        "arm-premium": False,
    }


def test_routerbench_conversion_rejects_missing_and_undetectable_model_columns() -> None:
    with pytest.raises(ConversionError, match="missing RouterBench"):
        convert_routerbench_wide_dataframe(pd.DataFrame({"sample_id": ["s1"]}))

    no_models = pd.DataFrame({"sample_id": ["s1"], "prompt": ["x"], "eval_name": ["mmlu"]})
    with pytest.raises(ConversionError, match="no RouterBench model"):
        convert_routerbench_wide_dataframe(no_models)


def test_routerbench_conversion_handles_missing_cost_and_general_task() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "prompt": ["Hello"],
            "eval_name": ["mt_bench"],
            "model-a": [0.7],
            "model-a|model_response": ["response"],
        }
    )
    converted = convert_routerbench_wide_dataframe(frame)
    assert converted.loc[0, "task_group"] == "general"
    assert pd.isna(converted.loc[0, "cost_usd"])


def test_verified_pickle_rejects_missing_source_and_bad_destination(tmp_path: Path) -> None:
    with pytest.raises(ConversionError, match="does not exist"):
        convert_verified_pickle(tmp_path / "missing.pkl", tmp_path / "out.csv", expected_sha256="0")

    source = tmp_path / "input.pkl"
    _wide_frame().to_pickle(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ConversionError, match="destination"):
        convert_verified_pickle(source, tmp_path / "out.parquet", expected_sha256=digest)


def test_verified_pickle_reports_timeout_and_worker_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.pkl"
    _wide_frame().to_pickle(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    def timeout(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="worker", timeout=1)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(ConversionError, match="timed out"):
        convert_verified_pickle(source, tmp_path / "out.csv", expected_sha256=digest)

    def failure(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=["worker"], returncode=2, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", failure)
    with pytest.raises(ConversionError, match="boom"):
        convert_verified_pickle(source, tmp_path / "out.csv", expected_sha256=digest)


def test_pickle_worker_main_converts_dataframe_and_rejects_other_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.pkl"
    output = tmp_path / "out.csv"
    _wide_frame().to_pickle(source)
    monkeypatch.setattr(sys, "argv", ["worker", "--input", str(source), "--output", str(output)])
    assert pickle_worker.main() == 0
    assert output.is_file()

    bad_source = tmp_path / "bad.pkl"
    pd.to_pickle([1, 2, 3], bad_source)
    monkeypatch.setattr(
        sys,
        "argv",
        ["worker", "--input", str(bad_source), "--output", str(tmp_path / "bad.csv")],
    )
    with pytest.raises(ConversionError, match="does not contain"):
        pickle_worker.main()


def test_pipeline_main_and_profile_without_latency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input.csv"
    _canonical_frame().to_csv(input_path, index=False)
    output_directory = tmp_path / "output"
    argv = ["pipeline", "--input", str(input_path), "--output-directory", str(output_directory)]
    for arm in ARMS:
        argv.extend(["--model-arm", arm])
    monkeypatch.setattr(sys, "argv", argv)
    assert pipeline.main() == 0
    assert (output_directory / "routerbench_canonical_clean.csv.gz").is_file()

    frame = _canonical_frame()
    frame.loc[:, "latency_ms"] = float("nan")
    assets = generate_profile_assets(frame, tmp_path / "no-latency", evidence_label="TEST")
    assert (tmp_path / "no-latency" / "latencia_modelos.svg") in assets
