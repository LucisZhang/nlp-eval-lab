.PHONY: setup test lint data snapshot ingest taxonomy dedup splits datasheet tier-a tier-b-data tier-b-bundle tier-c-prompt tier-c-exemplars-verify

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

# Tier B Colab bundle: one tarball with the flat upload set (kit + script + requirements
# + the four configs). Prints its path, size, and sha256. (gzip embeds a timestamp, so the
# sha identifies a given build, not a reproducible one.)
tier-b-bundle: tier-b-data
	@stage=$$(mktemp -d) && \
	cp -R data/tier_b_kit $$stage/tier_b_kit && \
	cp scripts/train_tier_b.py requirements-tierb-colab.txt $$stage/ && \
	cp configs/tier_b1_modernbert_sa.yaml configs/tier_b1_modernbert_sb.yaml \
	   configs/tier_b1_modernbert_sc.yaml configs/tier_b2_distilbert_s0.yaml $$stage/ && \
	tar -C $$stage -czf data/tier_b_colab_bundle.tar.gz . && \
	rm -rf $$stage && \
	echo "bundle_path: data/tier_b_colab_bundle.tar.gz" && \
	echo "size_bytes: $$(wc -c < data/tier_b_colab_bundle.tar.gz | tr -d ' ')" && \
	shasum -a 256 data/tier_b_colab_bundle.tar.gz


# Tier C prompt v1: print the frozen per-file + bundle content hashes (the prompt
# identity every Tier C run record carries; CLAUDE.md rule 4).
tier-c-prompt:
	uv run python -m triage_lab.tier_c_prompt --version v1

# Freeze gate: regenerate the frozen few-shot exemplars from data/ and fail loud on
# any byte difference (TRAIN split or selection logic drift).
tier-c-exemplars-verify:
	uv run python -m triage_lab.tier_c_prompt --verify-exemplars
