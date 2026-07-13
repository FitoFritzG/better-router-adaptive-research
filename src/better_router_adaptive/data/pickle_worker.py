"""Child-process worker for converting a verified RouterBench pickle."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from better_router_adaptive.data.convert import ConversionError, convert_routerbench_wide_dataframe


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    loaded = pd.read_pickle(args.input)
    if not isinstance(loaded, pd.DataFrame):
        raise ConversionError("verified pickle does not contain a pandas DataFrame")
    converted = convert_routerbench_wide_dataframe(loaded)
    converted.to_csv(args.output, index=False, lineterminator="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
