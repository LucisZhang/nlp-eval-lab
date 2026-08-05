.PHONY: setup test lint data

setup:
	uv sync --frozen

test:
	uv run pytest

lint:
	uv run ruff check

data:
	@echo "ERROR: 'make data' not implemented yet — Phase 0 (snapshot freeze, ingest, splits) incomplete." >&2
	@exit 1
