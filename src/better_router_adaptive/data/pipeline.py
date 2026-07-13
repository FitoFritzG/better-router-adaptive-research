"""End-to-end Step 3 pipeline for canonical cleaning and documentation assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from better_router_adaptive.data.clean import clean_canonical_dataframe, write_cleaning_report
from better_router_adaptive.data.profile import generate_profile_assets
from better_router_adaptive.data.schema import CANONICAL_COLUMNS


def run_data_pipeline(
    input_path: Path,
    output_directory: Path,
    *,
    required_model_arms: tuple[str, ...],
    evidence_label: str,
) -> Path:
    """Clean canonical CSV input and write audited outputs atomically enough for research use."""

    frame = pd.read_csv(input_path)
    cleaned, report = clean_canonical_dataframe(frame, required_model_arms=required_model_arms)

    output_directory.mkdir(parents=True, exist_ok=True)
    processed_path = output_directory / "routerbench_canonical_clean.csv.gz"
    temporary_path = output_directory / ".routerbench_canonical_clean.csv.gz.part"
    cleaned.to_csv(temporary_path, index=False, lineterminator="\n", compression="gzip")
    temporary_path.replace(processed_path)

    write_cleaning_report(report, output_directory)
    generate_profile_assets(cleaned, output_directory / "figures", evidence_label=evidence_label)
    schema_manifest = {
        "schema_version": "1.0.0",
        "format": "canonical_long",
        "columns": list(CANONICAL_COLUMNS),
        "primary_key": ["prompt_id", "model_id"],
        "rows": len(cleaned),
        "prompts": int(cleaned["prompt_id"].nunique()),
        "model_arms": list(required_model_arms),
        "evidence_label": evidence_label,
    }
    (output_directory / "schema_manifest.json").write_text(
        json.dumps(schema_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return processed_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean canonical routing outcomes and audit them.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model-arm", action="append", required=True)
    parser.add_argument("--evidence-label", default="DATOS EXPERIMENTALES")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    output = run_data_pipeline(
        args.input,
        args.output_directory,
        required_model_arms=tuple(args.model_arm),
        evidence_label=args.evidence_label,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
