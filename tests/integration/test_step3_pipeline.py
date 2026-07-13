from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from better_router_adaptive.data.pipeline import run_data_pipeline
from better_router_adaptive.data.schema import CANONICAL_COLUMNS


def test_step3_pipeline_generates_clean_data_reports_and_figures(tmp_path: Path) -> None:
    output = run_data_pipeline(
        Path("tests/fixtures/routerbench_sample.csv"),
        tmp_path,
        required_model_arms=("arm-fast", "arm-balanced", "arm-reasoning", "arm-premium"),
        evidence_label="DEMO SINTÉTICA — NO ES RESULTADO EXPERIMENTAL",
    )

    cleaned = pd.read_csv(output)
    assert tuple(cleaned.columns) == CANONICAL_COLUMNS
    assert len(cleaned) == 48
    assert cleaned["prompt_id"].nunique() == 12
    assert (tmp_path / "data_quality_report.json").is_file()
    assert (tmp_path / "data_quality_report.csv").is_file()
    assert (tmp_path / "schema_manifest.json").is_file()
    assert (tmp_path / "figures" / "distribucion_tareas.svg").is_file()
    assert (tmp_path / "figures" / "calidad_costo_modelos.svg").is_file()
    assert (tmp_path / "figures" / "latencia_modelos.svg").is_file()

    manifest = json.loads((tmp_path / "schema_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["rows"] == 48
    assert manifest["prompts"] == 12
    assert manifest["evidence_label"].startswith("DEMO SINTÉTICA")
