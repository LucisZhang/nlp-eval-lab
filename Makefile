.PHONY: setup test lint data snapshot ingest taxonomy dedup splits datasheet tier-a tier-b-data

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

taxonomy: ingest
	uv run python -m triage_lab.taxonomy

dedup: taxonomy
	uv run python -m triage_lab.dedup

splits: dedup
	uv run python -m triage_lab.splits

datasheet: splits
	uv run python -m triage_lab.datasheet

data: snapshot ingest taxonomy dedup splits datasheet

# Tier A eval ladder: three CAL iteration runs (single delta between consecutive
# rungs), then the two final TEST-IID reported runs. Appends to results/runs.jsonl.
tier-a:
	uv run python -m triage_lab.harness configs/tier_a_logreg_word_cal.yaml
	uv run python -m triage_lab.harness configs/tier_a_logreg_wordchar_cal.yaml
	uv run python -m triage_lab.harness configs/tier_a_cnb_wordchar_cal.yaml
	uv run python -m triage_lab.harness configs/tier_a_logreg_test_iid.yaml
	uv run python -m triage_lab.harness configs/tier_a_cnb_test_iid.yaml

# Tier B training kit: materialize the TRAIN + CAL parquets + provenance manifest the
# cloud GPU box needs (see docs/TIER_B_RUNBOOK.md). Deterministic, byte-identical.
tier-b-data:
	uv run python scripts/export_tier_b_data.py
