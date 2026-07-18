"""Validate that the public repository contains the mandatory academic deliverables.

The verifier checks repository structure and tracked files only. It intentionally
allows a local ``.venv`` created by the evaluator, while rejecting forbidden
artifacts if they were committed to Git.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = (
    "README.md",
    "ENTREGA.md",
    "EVALUACION_PROFESOR.md",
    "pyproject.toml",
    "requirements.txt",
    "Makefile",
    "config/experiment.yaml",
    "config/models.yaml",
    "src/better_router_adaptive",
    "tests/unit",
    "tests/integration",
    "artifacts/public/step7-real",
    "docs/METHODOLOGY.md",
    "docs/RESULTS.md",
    "docs/REPRODUCIBILITY.md",
)

FORBIDDEN_TRACKED_PREFIXES = (
    ".venv/",
    "venv/",
    "data/raw/",
    "data/interim/",
    "data/processed/",
    "__pycache__/",
    ".superpowers/",
    "docs/superpowers/",
    "docs/reviews/",
    "docs/evidence/",
)

FORBIDDEN_TRACKED_NAMES = {
    ".env",
    ".env.local",
    "secrets.json",
    "credentials.json",
    ".github/workflows/finalize-academic-delivery.yml",
}

ALLOWED_EMPTY_DIRECTORY_MARKERS = {
    "data/raw/.gitkeep",
    "data/interim/.gitkeep",
    "data/processed/.gitkeep",
}


def tracked_files() -> tuple[str, ...]:
    """Return Git-tracked paths, failing closed when Git cannot be queried."""

    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {completed.stderr.strip()}")
    return tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())


def verify_required_paths() -> list[str]:
    """Return missing mandatory files or directories."""

    return [relative for relative in REQUIRED_PATHS if not (ROOT / relative).exists()]


def verify_tracked_files(paths: tuple[str, ...]) -> list[str]:
    """Return tracked paths that must not be part of the public submission."""

    violations: list[str] = []
    for path in paths:
        normalized = path.replace("\\", "/")
        if normalized in ALLOWED_EMPTY_DIRECTORY_MARKERS:
            continue
        if normalized in FORBIDDEN_TRACKED_NAMES:
            violations.append(path)
            continue
        if any(normalized.startswith(prefix) for prefix in FORBIDDEN_TRACKED_PREFIXES):
            violations.append(path)
    return sorted(violations)


def main() -> int:
    """Run all structural checks and return a process-compatible status code."""

    missing = verify_required_paths()
    try:
        tracked = tracked_files()
    except RuntimeError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    forbidden = verify_tracked_files(tracked)

    if missing:
        print("[FAIL] Faltan archivos obligatorios:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)

    if forbidden:
        print("[FAIL] Hay archivos prohibidos versionados:", file=sys.stderr)
        for path in forbidden:
            print(f"  - {path}", file=sys.stderr)

    if missing or forbidden:
        return 1

    print("[PASS] Estructura obligatoria presente.")
    print(f"[PASS] {len(tracked)} archivos versionados revisados.")
    print("[PASS] No se detectaron datasets restringidos, secretos ni artefactos internos.")
    print("[PASS] El código obligatorio está listo para la evaluación técnica.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
