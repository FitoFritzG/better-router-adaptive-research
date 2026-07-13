.PHONY: install test lint typecheck verify

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest -v

lint:
	ruff check .

typecheck:
	mypy src tests

verify: test lint typecheck
