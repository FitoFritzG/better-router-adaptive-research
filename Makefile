.PHONY: install test lint format typecheck build smoke verify

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest -q --cov=better_router_adaptive --cov-branch --cov-report=term-missing --cov-fail-under=85

lint:
	ruff check .

format:
	ruff format --check .

typecheck:
	mypy src tests

build:
	python -m build

smoke:
	python -m better_router_adaptive.prepare --help
	python -m better_router_adaptive.baselines --help
	python -m better_router_adaptive.learn --help
	python -m better_router_adaptive.evaluate --help

verify: test lint format typecheck build smoke
