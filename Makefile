.PHONY: setup test lint data snapshot ingest taxonomy dedup splits datasheet tier-a tier-b-data tier-b-bundle tier-c-prompt tier-c-exemplars-verify tier-c-smoke preds risk-coverage cost-model thresholds router-sim frontier prior-shift oov perturb perturb-tier-c perturb-report

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

# Prior-shift decomposition (Phase 5 task 2): how much of each tier's yearly macro-F1 drop
# vs 2023 is the class-mix change alone (Path P reweighted-F1 counterfactual, primary) vs
# within-class drift, with the interaction, the Path Q / Shapley / ANOVA sensitivities, the
# exact accuracy decomposition, and per-class attribution. Offline; reads only the frozen
# per-example artifacts, so it REQUIRES `make preds` to have been run first. Deliberately
# standalone (not a prerequisite of anything): it is an analysis of existing runs and
# appends nothing to results/runs.jsonl.
prior-shift:
	uv run python -m triage_lab.prior_shift --all

# OOV / covariate tracking (Phase 5 task 4): yearly OOV rate against the frozen TRAIN
# vocabulary under two definitions — model-vocab (post min_df/max_features pruning, the
# vocabulary Tier A actually has) and corpus-novelty (unpruned, true lexical novelty) — plus
# the TF-IDF-space centroid cosine distance to TRAIN, all with document-bootstrap CIs. Reads
# only the frozen split parquets (sha256-gated) and refits the tier_a FeatureUnion on TRAIN
# from scratch, so it needs no artifacts and appends nothing to results/runs.jsonl. Slow by
# construction (a 300k-doc word+char TF-IDF fit, then one X^T C pass per slice); TRAIN is
# emitted as a slice too, as the pruning/noise-floor baseline the drift numbers are read
# against. Deliberately standalone, like prior-shift.
oov:
	uv run python -m triage_lab.oov --all

# Perturbation robustness (Phase 5 task 4): the 16 Tier A TEST-IID eval runs behind the
# typo / OCR / case exhibit — the clean word-only sensitivity baseline first, then 15
# perturbed runs (2 finals x 3 families x 2 rates, plus the word-only arm at 0.10). Each is
# a full TRAIN fit + CAL isotonic calibration with ONLY the eval narratives rewritten, so
# this is genuinely 16 model fits and takes hours; run it in the background. APPENDS to
# results/runs.jsonl and auto-writes data/preds/<run_id>.parquet.
#
# The 5 `case` runs are expected to reproduce their clean baseline EXACTLY (both TF-IDF
# blocks lowercase), and are run as the end-to-end control on the perturbation plumbing —
# they are the first thing to cut if wall-clock becomes the binding constraint.
perturb:
	uv run python -m triage_lab.harness configs/tier_a_logreg_word_test_iid.yaml
	uv run python -m triage_lab.harness configs/tier_a_logreg_test_iid_perturb_typo_05.yaml
	uv run python -m triage_lab.harness configs/tier_a_logreg_test_iid_perturb_typo_10.yaml
	uv run python -m triage_lab.harness configs/tier_a_logreg_test_iid_perturb_ocr_05.yaml
	uv run python -m triage_lab.harness configs/tier_a_logreg_test_iid_perturb_ocr_10.yaml
	uv run python -m triage_lab.harness configs/tier_a_logreg_test_iid_perturb_case_05.yaml
	uv run python -m triage_lab.harness configs/tier_a_logreg_test_iid_perturb_case_10.yaml
	uv run python -m triage_lab.harness configs/tier_a_cnb_test_iid_perturb_typo_05.yaml
	uv run python -m triage_lab.harness configs/tier_a_cnb_test_iid_perturb_typo_10.yaml
	uv run python -m triage_lab.harness configs/tier_a_cnb_test_iid_perturb_ocr_05.yaml
	uv run python -m triage_lab.harness configs/tier_a_cnb_test_iid_perturb_ocr_10.yaml
	uv run python -m triage_lab.harness configs/tier_a_cnb_test_iid_perturb_case_05.yaml
	uv run python -m triage_lab.harness configs/tier_a_cnb_test_iid_perturb_case_10.yaml
	uv run python -m triage_lab.harness configs/tier_a_logreg_word_test_iid_perturb_typo_10.yaml
	uv run python -m triage_lab.harness configs/tier_a_logreg_word_test_iid_perturb_ocr_10.yaml
	uv run python -m triage_lab.harness configs/tier_a_logreg_word_test_iid_perturb_case_10.yaml

# Tier C perturbation arm (Phase 5 task 4, owner-approved 2026-08-09). Haiku 4.5 zero-shot,
# TEST-IID, the 1,500-row paired subset (cap_seed 20260806 -> byte-identical subset of the
# clean 5,000-row run 70a1b0c4, and identical to the Sonnet paired rows), one run per family
# at rate 0.10. Makes LIVE OpenRouter calls and spends REAL money — needs the `tierc` dep
# group + OPENROUTER_API_KEY in .env. Nominal projection is ~$1.97/family at the clean run's
# measured $1.315/1k calls; actual will be HIGHER for typo/ocr (noisy text tokenizes into
# more prompt tokens) and is read from the per-call receipts, never from this estimate.
perturb-tier-c:
	uv run --extra tierc python -m triage_lab.harness configs/tier_c_haiku_zeroshot_test_iid_perturb_typo_10.yaml
	uv run --extra tierc python -m triage_lab.harness configs/tier_c_haiku_zeroshot_test_iid_perturb_ocr_10.yaml
	uv run --extra tierc python -m triage_lab.harness configs/tier_c_haiku_zeroshot_test_iid_perturb_case_10.yaml

# Perturbed-vs-clean paired bootstrap deltas on identical TEST-IID rows, from the frozen
# per-example artifacts the runs above wrote. Offline; appends nothing to runs.jsonl.
# Exits nonzero (naming the cells) if any run in `make perturb` is still missing.
perturb-report:
	uv run python -m triage_lab.perturb_report --all
