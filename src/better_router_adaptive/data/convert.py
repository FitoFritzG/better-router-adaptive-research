"""Controlled conversion of RouterBench wide data to canonical non-executable CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Final

import pandas as pd

from better_router_adaptive.data.schema import CANONICAL_COLUMNS

_FIXED_ROUTERBENCH_COLUMNS: Final = {"sample_id", "prompt", "eval_name", "idx"}
_TASK_PATTERNS: Final[dict[str, tuple[str, ...]]] = {
    "coding": ("mbpp", "human_eval", "humaneval", "code"),
    "mathematics": ("gsm", "math", "arithmetic"),
    "reasoning": ("mmlu", "hellaswag", "winogrande", "arc", "truthful"),
}


class ConversionError(ValueError):
    """Raised when source verification or conversion fails."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_group(eval_name: object) -> str:
    normalized = str(eval_name).strip().lower().replace("-", "_")
    for task_group, patterns in _TASK_PATTERNS.items():
        if any(pattern in normalized for pattern in patterns):
            return task_group
    return "general"


def _model_columns(frame: pd.DataFrame) -> list[str]:
    prefixed_models = {
        column.split("|", maxsplit=1)[0]
        for column in frame.columns.astype(str)
        if column.endswith("|total_cost") or column.endswith("|model_response")
    }
    models = sorted(
        model
        for model in prefixed_models
        if model in frame.columns and model not in _FIXED_ROUTERBENCH_COLUMNS
    )
    if not models:
        raise ConversionError(
            "no RouterBench model columns detected; expected performance columns with "
            "matching '<model>|total_cost' or '<model>|model_response' columns"
        )
    return models


def convert_routerbench_wide_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert RouterBench's wide dataframe into one row per prompt/model outcome."""

    required = {"sample_id", "prompt", "eval_name"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ConversionError(f"missing RouterBench columns: {', '.join(missing)}")

    converted: list[pd.DataFrame] = []
    for model_id in _model_columns(frame):
        quality = pd.to_numeric(frame[model_id], errors="coerce")
        cost_column = f"{model_id}|total_cost"
        cost = (
            pd.to_numeric(frame[cost_column], errors="coerce")
            if cost_column in frame.columns
            else pd.Series(float("nan"), index=frame.index)
        )
        converted.append(
            pd.DataFrame(
                {
                    "prompt_id": frame["eval_name"].astype(str)
                    + ":"
                    + frame["sample_id"].astype(str),
                    "prompt_text": frame["prompt"].astype(str),
                    "dataset": frame["eval_name"].astype(str),
                    "task_group": frame["eval_name"].map(_task_group),
                    "model_id": model_id,
                    "quality": quality,
                    "input_tokens": pd.Series(pd.NA, index=frame.index, dtype="Int64"),
                    "output_tokens": pd.Series(pd.NA, index=frame.index, dtype="Int64"),
                    "cost_usd": cost,
                    "latency_ms": pd.Series(float("nan"), index=frame.index),
                    "success": quality.notna(),
                    "data_origin": "routerbench_0shot_pinned",
                }
            )
        )

    result = pd.concat(converted, ignore_index=True)
    return result.loc[:, list(CANONICAL_COLUMNS)]


def _sanitized_environment(source_root: Path, temporary_home: Path) -> dict[str, str]:
    environment = {
        "PYTHONPATH": str(source_root),
        "PYTHONNOUSERSITE": "1",
        "PYTHONHASHSEED": "0",
        "HOME": str(temporary_home),
        "TMPDIR": str(temporary_home),
    }
    for key in ("PATH", "SYSTEMROOT", "WINDIR", "LD_LIBRARY_PATH"):
        value = os.environ.get(key)
        if value:
            environment[key] = value
    return environment


def convert_verified_pickle(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    timeout_seconds: int = 180,
) -> Path:
    """Verify a pinned pickle, convert it in a child process, and atomically publish CSV.

    The checksum gate and secret-free subprocess reduce risk but do not make
    pickle a safe format. Execute real-data conversion in a disposable machine
    or container without credentials or access to production systems.
    """

    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        raise ConversionError(f"source file does not exist: {source}")
    actual_sha256 = _sha256_file(source)
    if actual_sha256.lower() != expected_sha256.lower():
        raise ConversionError(
            f"source checksum mismatch (expected {expected_sha256}, got {actual_sha256})"
        )
    if destination.suffix not in {".csv", ".gz"}:
        raise ConversionError("destination must be .csv or .csv.gz")

    destination.parent.mkdir(parents=True, exist_ok=True)
    package_source_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="brar-pickle-conversion-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        temporary_output = temporary_root / (
            "canonical.csv.gz" if destination.suffix == ".gz" else "canonical.csv"
        )
        command = [
            sys.executable,
            "-m",
            "better_router_adaptive.data.pickle_worker",
            "--input",
            str(source.resolve()),
            "--output",
            str(temporary_output),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=temporary_root,
                env=_sanitized_environment(package_source_root, temporary_root),
            )
        except subprocess.TimeoutExpired as exc:
            raise ConversionError(f"pickle conversion timed out after {timeout_seconds}s") from exc
        if completed.returncode != 0:
            details = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
            raise ConversionError(f"pickle conversion failed: {details}")
        if not temporary_output.is_file():
            raise ConversionError("pickle converter did not produce an output file")

        publish_temp = destination.with_name(f".{destination.name}.part")
        publish_temp.write_bytes(temporary_output.read_bytes())
        os.replace(publish_temp, destination)

    manifest = {
        "source_file": source.name,
        "source_sha256": actual_sha256,
        "output_file": destination.name,
        "output_sha256": _sha256_file(destination),
        "serialization": "csv_gzip" if destination.suffix == ".gz" else "csv",
        "pickle_policy": "verified_checksum_child_process_no_secrets_not_a_security_sandbox",
    }
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(manifest_text, encoding="utf-8")
    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a checksum-pinned RouterBench pickle to canonical CSV."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = convert_verified_pickle(
        args.input,
        args.output,
        expected_sha256=args.expected_sha256,
        timeout_seconds=args.timeout_seconds,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
