# STATUS.md — Execution State

**Single source of truth for _execution state_.** Scope, architecture, metrics, and phase
acceptance criteria live in **`UPGRADE_PLAN.md`** (untouched by this file). When the two ever
appear to disagree, UPGRADE_PLAN.md defines _what_ and _why_; STATUS.md tracks _where we are_ and
_what's next_.

- **Owner decision (2026-08-06):** The A6000 is unavailable until the weekend, so **all Tier B
  training and eval is BLOCKED-until-weekend.** Work has been reordered to make maximal progress on
  the tiers and infrastructure that do not need the GPU.
- Last updated: 2026-08-10.

---

## (a) Task table

Status legend: **done** · **in-progress** · **blocked** (BLOCKED-until-weekend) · **pending**.
Evidence links are commit SHAs or EXPERIMENT_LOG.md entries (done tasks only).

### Phase 0 — Repo + data engineering
| Task | Status | Evidence |
|---|---|---|
| Repo scaffold, uv lock, CI skeleton | done | `d290240` |
| CFPB snapshot freeze (download date + sha256) | done | `6db4bc0` |
| DuckDB ingest → deterministic narratives.parquet | done | `fabe13c` |
| Taxonomy harmonization map | done | `72a7317` |
| MinHash dedup | done | `e60bc69` |
| Temporal splits (TRAIN/CAL/TEST-IID/TEST-DRIFT/TEST-POSTCUTOFF) + leakage check | done | `c9786db` |
| Datasheet (deterministic generator wired into `make data`) | done | `5be8736` |

### Phase 1 — Eval harness + Tier A
| Task | Status | Evidence |
|---|---|---|
| Eval harness (YAML→JSONL, bootstrap CIs, paired tests) + metric unit tests | done | `c61b464` |
| Tier A LogReg + ComplementNB (word/char), calibrated, CAL ladder | done | `9750d53`; EXPERIMENT_LOG 2026-08-05 rungs 1–3 |
| Tier A TEST-IID finals (LogReg + CNB, isotonic) | done | `9750d53`; EXPERIMENT_LOG 2026-08-05 finals |
| ≥3 logged runs in EXPERIMENT_LOG.md | done | 5 runs logged (2026-08-05) |

### Phase 2 — Tier B fine-tuning  🔒 BLOCKED-until-weekend (needs A6000)
| Task | Status | Evidence |
|---|---|---|
| Tier B training kit (runner, cloud training script, data export, ONNX parity harness) | done | `ba13a08`, `0dfa94b` |
| Tier B tokenization OOM fix (chunked + int32) | done | `5a7d641` |
| Tier B training preemptible + resumable | done | `334a47c` |
| ModernBERT-base fine-tune ×3 seeds | **blocked** | — needs GPU |
| DistilBERT fine-tune | **blocked** | — needs GPU |
| Temperature scaling + TEST-IID eval (both points, CIs) | **blocked** | — needs trained models |
| B1-vs-A delta, B1-vs-B2 intra-tier delta, seed variance | **blocked** | — needs trained models |
| DistilBERT int8 ONNX export + parity check (≥99%) | **blocked** | — needs trained model |

> **Kit is ready to launch the moment the A6000 is free.** Only the GPU-bound train/eval steps remain.

### Phase 3 — Tier C LLM  ✅ **COMPLETE 2026-08-07** (acceptance closed — EXPERIMENT_LOG step 6)
| Task | Status | Evidence |
|---|---|---|
| Tier C prompt + structured-output schema (versioned, content-hashed) | done | EXPERIMENT_LOG 2026-08-06; `prompts/tier_c/v1/` bundle `f6777a96…` |
| Zero-shot vs few-shot ablation on CAL (Haiku) | done | EXPERIMENT_LOG 2026-08-07; runs `c7598f84…`, `3f310951…`; paired CIs include zero (no few-shot gain on CAL) |
| Smoke run + cost approval gate (see execution order) | done (approval granted 2026-08-06 — see §b) | EXPERIMENT_LOG 2026-08-06 (step 2); runs `e22fba2a…`, `77cbd36f…`; measured $0.002656/call few-shot, ~$48.5 projected total |
| Haiku 4.5 full eval — TEST-IID + TEST-POSTCUTOFF (**zero-shot** per §4.2 amendment 2026-08-07) | done | EXPERIMENT_LOG 2026-08-07 (step 4); runs `70a1b0c4…`, `82af4e01…`; IID macro-F1 0.7697, POSTCUTOFF −0.0443 delta; $1.32–1.37/1k |
| Sonnet 5 subsample (**zero-shot** per §4.2 amendment 2026-08-07) | done | EXPERIMENT_LOG 2026-08-07 (step 5); runs `e1503146…`, `d1c42d7d…`; IID macro-F1 0.7418, POSTCUTOFF 0.7876; $3.66–3.85/1k; 1,500-row subsets pair with Haiku rows (verified) |
| Both Tier C points with CIs on both slices; measured $/1k + p50/p95 latency; Haiku-vs-Sonnet paired delta; contamination delta; raw API logs retained | done | EXPERIMENT_LOG 2026-08-07 (step 6); paired Sonnet−Haiku: IID ≈0 (McNemar p=1.00), POSTCUTOFF +0.0553 acc / +0.0458 macro-F1 (p=2e-10); sensitivity excl. fallback rows agrees; fallback asymmetry 0 vs 12/37 reported. **Latency labeling (step 7 audit):** all Tier C p50/p95 figures are client-side wall-clock through OpenRouter with **no provider pinning** — served ≈99.8% by Amazon Bedrock (rest Anthropic/Azure; per-call provider in receipts) at `max_concurrency: 8`; they characterize the OpenRouter→Bedrock route, not the Anthropic first-party API |

### Phase 4 — Calibration + router  ✅ **COMPLETE 2026-08-08 in owner-approved partial form** (Tier B backfill pending GPU)
| Task | Status | Evidence |
|---|---|---|
| Risk-coverage machinery (+ per-example prediction artifacts, all runs) | done | `182d425`; EXPERIMENT_LOG 2026-08-07 Phase 4 task 1 (row was stale — completed previous session, flipped 2026-08-07) |
| Cost-model implementation | done | EXPERIMENT_LOG 2026-08-07 Phase 4 task 2; `configs/cost_model_v1.yaml` `f76ad15a…`; `results/cost_model/` |
| Threshold optimization on CAL (per owner-amended §4.2: τ at A/B only, Tier C terminal, parse-failure→human; amendment recorded in UPGRADE_PLAN §4.2 + EXPERIMENT_LOG) | done | EXPERIMENT_LOG 2026-08-07 Phase 4 task 3; `results/thresholds/`; carry-forwards for task 4: cross-family delta directional-only at n=1,500; τ→TEST transfer rule (calibration-space) open; parse-fail arm empty on Haiku, fires on Sonnet; `tier_a_logreg_test_iid.yaml` "winning CAL rung" comment unsupported by runs.jsonl |
| Router simulator vs all policies (TEST-IID, owner decisions 1–4 of 2026-08-07 applied) | done | EXPERIMENT_LOG 2026-08-08 Phase 4 task 4; `results/router_sim/`; A→human dominates a_only + a_only_cnb (paired CIs ✓); Haiku-terminal headline dominates c_only only; cross-family delta directional-only (owner decision 1 verdict) |
| Two headline frontier claims with CIs; router dominates ≥2 single tiers (or honest diagnosis) | done (partial form, owner-approved 2026-08-08) — v2 isocal thresholds primary (run `40513354…`; coverage gap closed ~10×); CLAIM 2 a_to_human CERTIFIED on full TEST-IID (−$60.60/1k ✓, macro-F1 +0.0870 ✓); a_to_c claims NOT ESTABLISHED at n=5,000 (honest diagnosis); dominance ≥2 met by a_to_human; cross-family resolved against the cascade (+47.74 ✓); v1 retained as calibration-mismatch lesson; Tier B frontier points **pending Tier B** | EXPERIMENT_LOG 2026-08-08 Phase 4 task 5; `results/frontier/` |

### Phase 5 — Drift protocol  ▶ **NEXT (per §b order — available tiers only)**
| Task | Status | Evidence |
|---|---|---|
| Rolling yearly evals (2023–2026) | **Tier A done** (2026-08-08); **Tier C done** (2026-08-09, owner-approved; 8 runs, $30.48 actual vs $30.94 projected, cumulative $67.38 of ≈$75); Tier B rows **pending Tier B** | EXPERIMENT_LOG 2026-08-08 Phase 5 task 1 (Tier A: macro-F1 0.758→0.748→0.730→0.666, 2026-H1 cliff = credit_reporting prior shift) + 2026-08-09 tasks 2a/2b (Tier C: Haiku/Sonnet tied 2023–25 paired CIs ∋ 0; 2026-H1 Sonnet +0.054 F1 paired CI excl. 0, p≈1e-15; Tier C degrades slower than Tier A — 2026-H1: A 0.666 / Haiku 0.728 / Sonnet 0.782) |
| Prior-shift decomposition | **done for available tiers** (2026-08-09; Tier B rows pending Tier B) | EXPERIMENT_LOG 2026-08-09 Phase 5 task 3; `results/prior_shift/` (12 decompositions + summary.json); Tier A 2026-H1: 4.2 of 9.2 pts prior-shift alone [3.9, 4.5], credit_reporting = 7.6 pts; LLMs pay the prior penalty (Haiku +3.0 [1.4, 4.4]) but within-class holds (≈0/negative) |
| OOV tracking | **done** (2026-08-09; model-free, $0; dense-encoder centroid variant pending Tier B) | EXPERIMENT_LOG 2026-08-09 Phase 5 task 4; `results/oov/` (6 slices + summary.json); OOV hypothesis **refuted**: model-vocab token OOV 0.545%→0.773% TRAIN→2026-H1 (+0.23 pp), 99.2% of 2026-H1 token mass still in-vocab; TF-IDF centroid distance peaks at 2025 and *falls* at 2026-H1 (CIs disjoint) — lexical drift ruled out as cause of Tier A cliff, prior shift stands; types-vs-tokens gap (53.4% types vs 0.77% tokens OOV at 2026-H1) = grown-up CoNLL finding |
| Perturbation robustness | **done for available tiers** (Tier A 2026-08-09, 16 runs, $0; Tier C 2026-08-10 owner-approved option A, 4,500 calls, $6.287 actual vs $5.92 nominal / ≈$6.8 ceiling — cumulative ≈$73.67 of ≈$75). Tier A: typo −4.4 F1 pts @ 0.10, ocr ~−1, case = exact structural zero; char-n-gram shield directional (word-only −6.6). Tier C Haiku @ 0.10: typo −3.1 pts (CI excl. 0, smallest of any tier), ocr and case ∅ (CIs ∋ 0) despite case's largest token inflation (+8.8%); perturbation = +3–9% Tier C serving-cost tax. Cross-tier: LLM ≳ word+char > word-only. Tier B rows **pending Tier B** | EXPERIMENT_LOG 2026-08-09 task 5 + 2026-08-10 task 5b; `results/perturbation/summary.json` (18 rows); Tier A runs `4bbd26c4…`–`3f5eff59…`, Tier C runs `182774b6…`/`237bffae…`/`c7e53e2a…` |
| Novel-class probe (stretch) | pending | — |
| Drift charts from results log; escalation-rate-over-time; evidence classes labeled | pending | — |

### Phase 6 — Demo + case study
| Task | Status | Evidence |
|---|---|---|
| Static demo **site scaffold** (triage playground, frontier plot, policy builder, drift timeline, calibration panel, receipts drawer) | pending — scaffold authorized now; Tier B panels **pending Tier B** | — |
| Case study page (verification + "does not prove" sections) | pending | — |
| Provenance links to coursework seeds | pending | — |
| `make reproduce-headline` | pending | — |
| Demo fully static/offline; every number traces to a results record; reproduce-headline on clean machine | pending | — |

### Phase 7 — PII cross-check tie-in (optional stretch)
| Task | Status | Evidence |
|---|---|---|
| Privacy-preflight detection on CFPB narratives; residual-PII rate; two-case-study link | pending — cut freely if time-boxed | — |

---

## (b) Authorized execution order (going forward)

Tier B is **BLOCKED-until-weekend**; do the GPU-free work first, in this exact order:

1. **Phase 3 — Tier C** (do first)
   1. Version + hash the Tier C prompt and structured-output schema.
   2. ~~**Smoke run** (tiny subsample) → measure real per-call token cost~~ **done 2026-08-06**
      → **cost approval GRANTED by owner 2026-08-06** for ≈$48.5 total and the proposed
      subsample sizes (ablation 1,500/arm; Haiku TEST-IID + TEST-POSTCUTOFF 5,000 each;
      Sonnet 5 paired 1,500 × 2 slices — see EXPERIMENT_LOG 2026-08-06 step 2).
   3. ~~Zero-shot vs few-shot ablation on CAL~~ **done 2026-08-07** (no few-shot gain; paired
      CIs include zero → owner amended UPGRADE_PLAN §4.2: **all Tier C finals run zero-shot**,
      few-shot config archived for possible drift-slice probes). ~~Then full Haiku 4.5
      (TEST-IID + TEST-POSTCUTOFF, zero-shot) and Sonnet 5 zero-shot subsample.~~
      **All done 2026-08-07; Phase 3 acceptance closed (EXPERIMENT_LOG step 6). Next: Phase 4.**
2. **Phase 4 — Calibration + router infra**, built against **Tier A outputs** (and Tier C once
   logged). Stand up the cost model, risk-coverage, threshold optimization, and router simulator.
   Any frontier claim requiring Tier B points is **pending Tier B**.
3. **Phase 5 — Drift** for **available tiers only** (Tier A, then Tier C). Tier B drift rows are
   **pending Tier B**.
4. **Phase 6 — Site scaffold.** Build the static demo structure and wire in available exhibits.
   Any panel/number sourced from Tier B is **pending Tier B**.

**Tier B (training + eval + ONNX parity): BLOCKED-until-weekend.** Launch the existing training kit
the moment the A6000 is available; then backfill every item marked **pending Tier B** above
(Phase 4 frontier points, Phase 5 drift rows, Phase 6 panels).

---

## (c) Session protocol

Each future session does **exactly one task** from the authorized execution order (§b), and no more:

1. **Pick** the next single task from §b (respect the ordering; skip anything BLOCKED-until-weekend).
2. **Restate** that task's ✅ Accept criteria from UPGRADE_PLAN.md before implementing.
3. **Implement** the one task (single-variable discipline; one config delta per run).
4. **Prove** the acceptance criteria by running the proving command; capture metrics + CIs.
5. **Append** a dated hypothesis → result → verdict entry to `EXPERIMENT_LOG.md`, including the
   exact reproduction command (CLAUDE.md rule 5). `results/*.jsonl` stays append-only.
6. **Update `STATUS.md`** — flip the task's status and add its evidence link.
7. **Commit** as `phaseN: <task name>`.
8. **Stop.** Do not start the next task in the same session.

**Guardrails (from CLAUDE.md):** splits/seeds/exemplars are frozen; TEST-* slices are touched only
for final reported runs (iterate on CAL); Tier C prompts are versioned + hashed and go through
OpenRouter with cost from real token usage; `docs/seed-evidence/` is read-only; `UPGRADE_PLAN.md`
is not edited by execution sessions.
