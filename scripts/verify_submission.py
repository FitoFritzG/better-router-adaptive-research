from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REQUIRED_PATHS = (
    "README.md",
    "ENTREGA.md",
    "pyproject.toml",
    "requirements.txt",
    "config/experiment.yaml",
    "config/models.yaml",
    "src/better_router_adaptive/routers.py",
    "src/better_router_adaptive/learn.py",
    "src/better_router_adaptive/evocascade.py",
    "src/better_router_adaptive/data/pipeline.py",
    "tests/unit/test_routers.py",
    "tests/unit/test_evocascade.py",
    "artifacts/public/step7-real/evaluation_summary.csv",
    "artifacts/public/evocascade-ideal-verifier/evaluation_summary.csv",
    "docs/METHODOLOGY.md",
    "docs/RESULTS.md",
    "docs/ASSIGNMENT_READINESS.md",
)

FORBIDDEN_PARTS = {".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
FORBIDDEN_NAMES = {".env", ".env.local", "secrets.json", "credentials.json"}
FORBIDDEN_SUFFIXES = {".pkl", ".pickle", ".key", ".pem"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path, poster: Path) -> dict[str, object]:
    missing = [relative for relative in REQUIRED_PATHS if not (root / relative).is_file()]
    forbidden: list[str] = []
    for path in root.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            forbidden.append(str(relative))
        if path.name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            forbidden.append(str(relative))
        if relative.parts[:2] in {("data", "raw"), ("data", "interim")} and path.name != ".gitkeep":
            forbidden.append(str(relative))
        if relative.parts[:2] == ("data", "processed") and path.name != ".gitkeep":
            forbidden.append(str(relative))

    poster_ok = poster.is_file() and poster.stat().st_size > 10_000
    if poster_ok:
        with poster.open("rb") as handle:
            poster_ok = handle.read(5) == b"%PDF-"

    routers_text = (root / "src/better_router_adaptive/routers.py").read_text(encoding="utf-8")
    algorithms_ok = "XGBoostRouter" in routers_text and "LinUCBRouter" in routers_text

    report = {
        "status": "PASS" if not missing and not forbidden and poster_ok and algorithms_ok else "FAIL",
        "missing": missing,
        "forbidden": sorted(set(forbidden)),
        "poster": str(poster.relative_to(root)),
        "poster_sha256": sha256(poster) if poster_ok else None,
        "poster_bytes": poster.stat().st_size if poster.exists() else 0,
        "two_required_algorithms_detected": algorithms_ok,
    }
    if report["status"] != "PASS":
        raise SystemExit(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the final academic submission tree.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--poster",
        type=Path,
        default=Path("paper/poster/poster_evocascade_ideal.pdf"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    poster = args.poster if args.poster.is_absolute() else root / args.poster
    verify(root, poster.resolve())


if __name__ == "__main__":
    main()
