from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import BinaryIO

import pytest

from better_router_adaptive.data import download as download_module
from better_router_adaptive.data.download import download_routerbench, sha256_file


class BytesResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0

    def __enter__(self) -> BytesResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._payload) - self._offset
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"routerbench")

    assert sha256_file(path) == digest(b"routerbench")


def test_existing_verified_file_is_reused_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "routerbench_0shot.pkl"
    payload = b"verified-existing-data"
    destination.write_bytes(payload)

    def fail_urlopen(*_args: object, **_kwargs: object) -> BinaryIO:
        raise AssertionError("network must not be called for a verified existing file")

    monkeypatch.setattr(download_module, "urlopen", fail_urlopen)
    monkeypatch.setattr(download_module, "ROUTERBENCH_0SHOT_SHA256", digest(payload))

    result = download_routerbench(destination)

    assert result == destination
    assert destination.read_bytes() == payload
    manifest = json.loads((tmp_path / "download_manifest.json").read_text(encoding="utf-8"))
    assert manifest["reused_existing"] is True


def test_existing_mismatched_file_raises_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "routerbench_0shot.pkl"
    destination.write_bytes(b"wrong")
    monkeypatch.setattr(download_module, "ROUTERBENCH_0SHOT_SHA256", digest(b"expected"))

    with pytest.raises(ValueError, match="existing file checksum mismatch"):
        download_routerbench(destination)


def test_force_replaces_mismatched_file_and_writes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "routerbench_0shot.pkl"
    destination.write_bytes(b"stale")
    payload = b"fresh-routerbench-data"

    monkeypatch.setattr(download_module, "ROUTERBENCH_0SHOT_SHA256", digest(payload))
    monkeypatch.setattr(
        download_module,
        "urlopen",
        lambda *_args, **_kwargs: BytesResponse(payload),
    )

    result = download_routerbench(destination, force=True)

    assert result == destination
    assert destination.read_bytes() == payload
    manifest = json.loads((tmp_path / "download_manifest.json").read_text(encoding="utf-8"))
    assert manifest["selected_file"] == "routerbench_0shot.pkl"
    assert manifest["sha256"] == digest(payload)
    assert manifest["byte_count"] == len(payload)
    assert manifest["dataset_revision"] == download_module.ROUTERBENCH_DATASET_REVISION
    assert manifest["license_status"] == "not_declared_on_dataset_card"


def test_failed_download_never_replaces_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "routerbench_0shot.pkl"
    destination.write_bytes(b"original")
    monkeypatch.setattr(download_module, "ROUTERBENCH_0SHOT_SHA256", digest(b"expected"))
    monkeypatch.setattr(
        download_module,
        "urlopen",
        lambda *_args, **_kwargs: BytesResponse(b"corrupt"),
    )

    with pytest.raises(ValueError, match="downloaded file checksum mismatch"):
        download_routerbench(destination, force=True)

    assert destination.read_bytes() == b"original"
    assert not list(tmp_path.glob("*.part"))


def test_network_failure_removes_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "routerbench_0shot.pkl"

    def raise_network_error(*_args: object, **_kwargs: object) -> BinaryIO:
        raise OSError("simulated network failure")

    monkeypatch.setattr(download_module, "urlopen", raise_network_error)

    with pytest.raises(OSError, match="simulated network failure"):
        download_routerbench(destination)

    assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))


def test_synthetic_fixture_has_declared_shape_and_no_experimental_claims() -> None:
    fixture_path = Path("tests/fixtures/routerbench_sample.csv")
    with fixture_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = set(reader.fieldnames or [])

    required_columns = {
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
    }
    assert required_columns.issubset(columns)
    assert 40 <= len(rows) <= 80
    assert len({row["prompt_id"] for row in rows}) >= 4
    assert {row["task_group"] for row in rows} == {
        "coding",
        "mathematics",
        "reasoning",
        "general",
    }
    assert {row["model_id"] for row in rows} == {
        "arm-fast",
        "arm-balanced",
        "arm-reasoning",
        "arm-premium",
    }
    assert {row["data_origin"] for row in rows} == {
        "synthetic_test_fixture_not_experimental_evidence"
    }
    models_by_prompt: dict[str, set[str]] = {}
    for row in rows:
        models_by_prompt.setdefault(row["prompt_id"], set()).add(row["model_id"])
    assert all(len(model_ids) == 4 for model_ids in models_by_prompt.values())


def test_official_artifact_constants_are_pinned() -> None:
    assert download_module.ROUTERBENCH_DATASET_REVISION == (
        "a4dcf98b60f1faf85c572ee5f20cc0069ca0501a"
    )
    assert download_module.ROUTERBENCH_0SHOT_BYTES == 99_567_659
    assert download_module.ROUTERBENCH_0SHOT_SHA256 == (
        "ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d"
    )
    assert download_module.ROUTERBENCH_DATASET_DOI == "10.57967/hf/1996"
