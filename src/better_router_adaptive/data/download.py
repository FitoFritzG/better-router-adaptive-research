"""Verified acquisition of the pinned RouterBench 0-shot artifact.

The module downloads but never deserializes the upstream pickle file. Pickle
payloads can execute code during loading, so conversion is intentionally left
to a later, isolated and reviewed pipeline stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

ROUTERBENCH_DATASET_REVISION = "a4dcf98b60f1faf85c572ee5f20cc0069ca0501a"
ROUTERBENCH_0SHOT_FILENAME = "routerbench_0shot.pkl"
ROUTERBENCH_0SHOT_SHA256 = "ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d"
ROUTERBENCH_0SHOT_BYTES = 99_567_659
ROUTERBENCH_0SHOT_URL = (
    "https://huggingface.co/datasets/withmartian/routerbench/resolve/"
    f"{ROUTERBENCH_DATASET_REVISION}/{ROUTERBENCH_0SHOT_FILENAME}?download=true"
)
ROUTERBENCH_DATASET_CARD = "https://huggingface.co/datasets/withmartian/routerbench"
ROUTERBENCH_CODE_REPOSITORY = "https://github.com/withmartian/routerbench"
ROUTERBENCH_PAPER = "https://arxiv.org/abs/2403.12031"
ROUTERBENCH_DATASET_DOI = "10.57967/hf/1996"
_MANIFEST_FILENAME = "download_manifest.json"
_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of ``path``."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_payload(destination: Path, *, reused_existing: bool) -> dict[str, Any]:
    return {
        "dataset": "RouterBench",
        "selected_file": ROUTERBENCH_0SHOT_FILENAME,
        "source_url": ROUTERBENCH_0SHOT_URL,
        "dataset_revision": ROUTERBENCH_DATASET_REVISION,
        "dataset_card": ROUTERBENCH_DATASET_CARD,
        "code_repository": ROUTERBENCH_CODE_REPOSITORY,
        "paper": ROUTERBENCH_PAPER,
        "dataset_doi": ROUTERBENCH_DATASET_DOI,
        "license_status": "not_declared_on_dataset_card",
        "redistribution_policy": "do_not_commit_or_redistribute_upstream_dataset",
        "serialization_warning": "pickle_not_deserialized_by_download_stage",
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "byte_count": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "expected_sha256": ROUTERBENCH_0SHOT_SHA256,
        "expected_byte_count": ROUTERBENCH_0SHOT_BYTES,
        "reused_existing": reused_existing,
    }


def _write_manifest_atomic(destination: Path, *, reused_existing: bool) -> None:
    manifest_path = destination.parent / _MANIFEST_FILENAME
    payload = json.dumps(
        _manifest_payload(destination, reused_existing=reused_existing),
        indent=2,
        sort_keys=True,
    )
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{_MANIFEST_FILENAME}.", suffix=".part", dir=manifest_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, manifest_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _download_to_temporary_file(destination: Path) -> Path:
    request = Request(
        ROUTERBENCH_0SHOT_URL,
        headers={"User-Agent": "better-router-adaptive-research/0.1.0"},
    )
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{destination.name}.", suffix=".part", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as output, urlopen(request, timeout=120) as response:
            while chunk := response.read(_CHUNK_SIZE):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        return temporary_path
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def download_routerbench(destination: Path, force: bool = False) -> Path:
    """Download the pinned RouterBench 0-shot artifact to ``destination``.

    Existing files are reused only when their SHA-256 matches the pinned
    upstream checksum. ``force=True`` permits replacement of a mismatched
    existing file, but the old file remains intact until a complete verified
    download is ready for an atomic rename.
    """

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        existing_digest = sha256_file(destination)
        if existing_digest == ROUTERBENCH_0SHOT_SHA256 and not force:
            _write_manifest_atomic(destination, reused_existing=True)
            return destination
        if not force:
            raise ValueError(
                "existing file checksum mismatch; pass force=True to replace it "
                f"(expected {ROUTERBENCH_0SHOT_SHA256}, got {existing_digest})"
            )

    temporary_path = _download_to_temporary_file(destination)
    try:
        downloaded_digest = sha256_file(temporary_path)
        if downloaded_digest != ROUTERBENCH_0SHOT_SHA256:
            raise ValueError(
                "downloaded file checksum mismatch "
                f"(expected {ROUTERBENCH_0SHOT_SHA256}, got {downloaded_digest})"
            )
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    _write_manifest_atomic(destination, reused_existing=False)
    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and verify the pinned RouterBench 0-shot artifact."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/raw") / ROUTERBENCH_0SHOT_FILENAME,
        help="Final file path (default: data/raw/routerbench_0shot.pkl).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing file after a new download passes checksum verification.",
    )
    return parser


def main() -> int:
    """CLI entry point."""

    args = _build_parser().parse_args()
    path = download_routerbench(args.destination, force=args.force)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
