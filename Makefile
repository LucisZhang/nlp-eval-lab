.PHONY: setup test lint data snapshot ingest

setup:
	uv sync --frozen

test:
	uv run pytest

lint:
	uv run ruff check

snapshot:
	@if [ ! -f data/raw/complaints.csv.zip ]; then \
		mkdir -p data/raw; \
		curl -sSL --retry 3 --continue-at - -o data/raw/complaints.csv.zip https://files.consumerfinance.gov/ccdb/complaints.csv.zip; \
	fi
	uv run python -m triage_lab.snapshot verify

ingest: snapshot
	uv run python -m triage_lab.ingest

data: snapshot ingest
	@echo "ERROR: 'make data' incomplete — remaining Phase 0 steps (taxonomy map, dedup, temporal splits, datasheet) not implemented yet." >&2
	@exit 1
