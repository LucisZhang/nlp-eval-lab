.PHONY: setup test lint data snapshot ingest taxonomy dedup splits datasheet tier-a tier-b-data tier-b-bundle tier-c-prompt tier-c-exemplars-verify tier-c-smoke preds risk-coverage cost-model thresholds router-sim frontier

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

# Tier C SMOKE + cost gate: 25 CAL rows through Claude Haiku 4.5 via OpenRouter (few-shot
# first, then zero-shot). Makes LIVE calls and spends real (tiny) money — needs the `tierc`
# dep group + OPENROUTER_API_KEY in .env. --no-append: these garbage-at-n=25 records never
# enter results/runs.jsonl; the raw per-call receipts land under results/tier_c_raw/.
tier-c-smoke:
	uv run --extra tierc python -m triage_lab.harness configs/tier_c_haiku_smoke_fewshot_cal.yaml
	uv run --extra tierc python -m triage_lab.harness configs/tier_c_haiku_smoke_zeroshot_cal.yaml

# Per-example prediction artifacts (Phase 4): regenerate data/preds/<run_id>.parquet for
# every non-smoke run and verify each against its logged point metrics (✓/✗, nonzero exit
# on mismatch). Artifacts are gitignored (data/) and regenerable; runs.jsonl is untouched.
# The slow LogReg re-fit is isolated with --only for a separate background launch.
preds:
	uv run python -m triage_lab.predictions --all

# Risk-coverage evidence JSONs (Phase 4): committed, deterministic operating-point tables
# + CI'd AURC / acc@coverage summaries, one per artifact, for the router phase and demo.
risk-coverage:
	uv run python -m triage_lab.risk_coverage --all

# Business cost model (Phase 4): expected cost per 1,000 complaints with bootstrap CIs for
# every single-tier policy, under configs/cost_model_v1.yaml (hash bound into each output).
# Tier C API cost is joined from committed per-call receipts and cross-checked against the
# run record's logged cost_usd at 1e-6 — a mismatch fails loud.
cost-model:
	uv run python -m triage_lab.cost_model --all

# Threshold optimization on CAL (Phase 4 task 3): sweeps the Tier A confidence gate over
# every distinct p_max, prices each operating point with the cost model, and writes the
# argmin + reference policies + a (c_misroute x c_human) sensitivity grid to
# results/thresholds/. Offline and CAL-only — no API calls, no TEST-* artifact is opened.
# Both derivations, deterministically: v1 (raw-CAL tau*, kept as the documented
# calibration-mismatch lesson) then v2 (tau* fit in the deployment calibration space,
# primary for reported numbers). Each writes its own files; v1's regenerate byte-identical.
thresholds:
	uv run python -m triage_lab.threshold_opt --derivation v1-raw
	uv run python -m triage_lab.threshold_opt --derivation v2-isocal

# Router simulator on TEST-IID (Phase 4 task 4): applies the CAL-fit tau* constants from
# results/thresholds/ to the frozen TEST-IID artifacts, prices every policy, and reports
# paired deltas + McNemar against each baseline. Offline (no model runs, no API calls);
# TEST-* artifacts are read here because this IS the final reported evaluation.
router-sim: thresholds
	uv run python -m triage_lab.router_sim --op-version v1-raw
	uv run python -m triage_lab.router_sim --op-version v2-isocal

# Frontier claims (Phase 4, partial): the two §4.2 headline claims with paired bootstrap
# CIs against the v2 operating points, the dominance census, and explicit pending_tier_b
# placeholders for every Tier B frontier slot.
frontier: router-sim
	uv run python -m triage_lab.frontier --op-version v2-isocal
