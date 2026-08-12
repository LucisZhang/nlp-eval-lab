# STATUS.md — Execution State

**Single source of truth for _execution state_.** Scope, architecture, metrics, and phase
acceptance criteria live in **`UPGRADE_PLAN.md`** (untouched by this file). When the two ever
appear to disagree, UPGRADE_PLAN.md defines _what_ and _why_; STATUS.md tracks _where we are_ and
_what's next_.

- **Owner decision (2026-08-06):** The A6000 is unavailable until the weekend, so **all Tier B
  training and eval is BLOCKED-until-weekend.** Work has been reordered to make maximal progress on
  the tiers and infrastructure that do not need the GPU.
- **Owner decision (2026-08-10):** the real Tier B training-results bundle arrived and its ingest
  took priority over the remaining Phase 6 tasks. **Ingest + validation complete — Tier B training
  is DONE and the block is lifted.** Checkpoints sit at the canonical `data/checkpoints/tier_b*`
  paths; next sessions run the Tier B **eval backfill** (harness TEST-IID finals + temperature
  scaling, then the pending Phase 4/5/6 Tier B slots), one task per session per §c.
- **Owner decision (2026-08-12, execute in the Phase 6 panels task — not before):**
  `headline_router` repoints from `a_to_c_parsefail_human` to **`a_to_b`** — the only certified
  two-axis win (cost CI excl. 0 AND F1 CI excl. 0), dominating 3 baselines. The Haiku cascade
  stays as the LLM-cascade **contrast exhibit**; the drift-chapter Sonnet-terminal variant is
  unchanged. The panels session picks this up alongside the pending `make demo-data` regen.
- **Owner decision (2026-08-12, Phase 5 Tier B sub-slots):** three of the four remaining Tier B
  sub-slots are **descoped by owner** — (1) B1 yearly drift series (~8 h MPS for a model B2
  dominates on every TEST-IID metric; no narrative value), (2) Tier B perturbation grid
  (cross-tier robustness ranking already measured and bracketed), (3) OOV dense-encoder centroid
  variant (OOV hypothesis already refuted model-free; re-probing a closed question). Slots stay
  labeled below, not deleted. The fourth — **Tier B2 prior-shift decomposition** — was approved
  and is **done** (see Phase 5 table + EXPERIMENT_LOG 2026-08-12 task 5).
- Last updated: 2026-08-12.

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

### Phase 2 — Tier B fine-tuning  ✅ **COMPLETE 2026-08-11** (ONNX export + parity closed the phase)
| Task | Status | Evidence |
|---|---|---|
| Tier B training kit (runner, cloud training script, data export, ONNX parity harness) | done | `ba13a08`, `0dfa94b` |
| Tier B tokenization OOM fix (chunked + int32) | done | `5a7d641` |
| Tier B training preemptible + resumable | done | `334a47c` |
| ModernBERT-base fine-tune ×3 seeds | done (A6000 bf16, 2026-08-07; ingested + validated 2026-08-10) | EXPERIMENT_LOG 2026-08-10 (bundle `c47f927e…`; 51/51 manifest re-hash, configs byte-identical, seeds/precision verified; final CAL macro-F1 0.7856/0.7874/0.7865) |
| DistilBERT fine-tune | done (A6000 bf16, 2026-08-07; ingested + validated 2026-08-10) | EXPERIMENT_LOG 2026-08-10 (same bundle; final CAL macro-F1 0.7923 — edges B1 on training-script numbers, re-examine under harness) |
| Temperature scaling + TEST-IID eval (both points, CIs) | done (2026-08-10, MPS local) | EXPERIMENT_LOG 2026-08-10 (eval backfill task 1); runs `8071d31d…`/`adb96307…`/`a523049a…`/`5517ebf1…`; B1 macro-F1 0.7878/0.7878/0.7863, B2 **0.7950** [0.7909, 0.7988], full TEST-IID n=104,443 |
| B1-vs-A delta, B1-vs-B2 intra-tier delta, seed variance | done (2026-08-10) | EXPERIMENT_LOG 2026-08-10; `results/tier_b_compare/summary.json`; B1−A +0.026..+0.027 (CIs excl. 0 ✓); **B1−B2 negative all seeds** (−0.0072..−0.0088, CIs excl. 0, McNemar p≤6e-8) — pre-registered surprise CONFIRMED: B2 DistilBERT is the top Tier B point; seed sd 0.0009 (ddof=1) |
| DistilBERT int8 ONNX export + parity check (≥99%) | done (2026-08-11) | EXPERIMENT_LOG 2026-08-11; `results/onnx_parity/tier_b2_s0_parity.json`; int8-vs-fp32 agreement **0.9944** ✓ (per-channel QInt8; stock per-tensor missed at 0.9824 — measured + logged, config fixed, no threshold relaxed), macro-F1 Δ +0.0016, fp32-ONNX-vs-PyTorch exact 1.0000; artifact = demo implementation only, official numbers stay harness fp32 runs.jsonl |

> **Checkpoints validated + placed at `data/checkpoints/tier_b{1_sa,1_sb,1_sc,2_s0}/`** (chain of
> custody: bundle manifest ↔ frozen configs ↔ frozen kit manifest ↔ CFPB input sha, all verified;
> receipts at `data/checkpoints/incoming/receipts_20260807T191924Z/`). Only local eval/ONNX steps remain.

### Phase 3 — Tier C LLM  ✅ **COMPLETE 2026-08-07** (acceptance closed — EXPERIMENT_LOG step 6)
| Task | Status | Evidence |
|---|---|---|
| Tier C prompt + structured-output schema (versioned, content-hashed) | done | EXPERIMENT_LOG 2026-08-06; `prompts/tier_c/v1/` bundle `f6777a96…` |
| Zero-shot vs few-shot ablation on CAL (Haiku) | done | EXPERIMENT_LOG 2026-08-07; runs `c7598f84…`, `3f310951…`; paired CIs include zero (no few-shot gain on CAL) |
| Smoke run + cost approval gate (see execution order) | done (approval granted 2026-08-06 — see §b) | EXPERIMENT_LOG 2026-08-06 (step 2); runs `e22fba2a…`, `77cbd36f…`; measured $0.002656/call few-shot, ~$48.5 projected total |
| Haiku 4.5 full eval — TEST-IID + TEST-POSTCUTOFF (**zero-shot** per §4.2 amendment 2026-08-07) | done | EXPERIMENT_LOG 2026-08-07 (step 4); runs `70a1b0c4…`, `82af4e01…`; IID macro-F1 0.7697, POSTCUTOFF −0.0443 delta; $1.32–1.37/1k |
| Sonnet 5 subsample (**zero-shot** per §4.2 amendment 2026-08-07) | done | EXPERIMENT_LOG 2026-08-07 (step 5); runs `e1503146…`, `d1c42d7d…`; IID macro-F1 0.7418, POSTCUTOFF 0.7876; $3.66–3.85/1k; 1,500-row subsets pair with Haiku rows (verified) |
| Both Tier C points with CIs on both slices; measured $/1k + p50/p95 latency; Haiku-vs-Sonnet paired delta; contamination delta; raw API logs retained | done | EXPERIMENT_LOG 2026-08-07 (step 6); paired Sonnet−Haiku: IID ≈0 (McNemar p=1.00), POSTCUTOFF +0.0553 acc / +0.0458 macro-F1 (p=2e-10); sensitivity excl. fallback rows agrees; fallback asymmetry 0 vs 12/37 reported. **Latency labeling (step 7 audit):** all Tier C p50/p95 figures are client-side wall-clock through OpenRouter with **no provider pinning** — served ≈99.8% by Amazon Bedrock (rest Anthropic/Azure; per-call provider in receipts) at `max_concurrency: 8`; they characterize the OpenRouter→Bedrock route, not the Anthropic first-party API |

### Phase 4 — Calibration + router  ✅ **COMPLETE 2026-08-11** (Tier B frontier points backfilled — no pending slots remain)
| Task | Status | Evidence |
|---|---|---|
| Risk-coverage machinery (+ per-example prediction artifacts, all runs) | done | `182d425`; EXPERIMENT_LOG 2026-08-07 Phase 4 task 1 (row was stale — completed previous session, flipped 2026-08-07) |
| Cost-model implementation | done | EXPERIMENT_LOG 2026-08-07 Phase 4 task 2; `configs/cost_model_v1.yaml` `f76ad15a…`; `results/cost_model/` |
| Threshold optimization on CAL (per owner-amended §4.2: τ at A/B only, Tier C terminal, parse-failure→human; amendment recorded in UPGRADE_PLAN §4.2 + EXPERIMENT_LOG) | done | EXPERIMENT_LOG 2026-08-07 Phase 4 task 3; `results/thresholds/`; carry-forwards for task 4: cross-family delta directional-only at n=1,500; τ→TEST transfer rule (calibration-space) open; parse-fail arm empty on Haiku, fires on Sonnet; `tier_a_logreg_test_iid.yaml` "winning CAL rung" comment unsupported by runs.jsonl |
| Router simulator vs all policies (TEST-IID, owner decisions 1–4 of 2026-08-07 applied) | done | EXPERIMENT_LOG 2026-08-08 Phase 4 task 4; `results/router_sim/`; A→human dominates a_only + a_only_cnb (paired CIs ✓); Haiku-terminal headline dominates c_only only; cross-family delta directional-only (owner decision 1 verdict) |
| Two headline frontier claims with CIs; router dominates ≥2 single tiers (or honest diagnosis) | done (partial form, owner-approved 2026-08-08) — v2 isocal thresholds primary (run `40513354…`; coverage gap closed ~10×); CLAIM 2 a_to_human CERTIFIED on full TEST-IID (−$60.60/1k ✓, macro-F1 +0.0870 ✓); a_to_c claims NOT ESTABLISHED at n=5,000 (honest diagnosis); dominance ≥2 met by a_to_human; cross-family resolved against the cascade (+47.74 ✓); v1 retained as calibration-mismatch lesson; Tier B frontier points **pending Tier B** | EXPERIMENT_LOG 2026-08-08 Phase 4 task 5; `results/frontier/` |
| Tier B frontier points backfill (b1_only ×3 / b2_only / a_to_b / a_to_b_to_c) | done (2026-08-11) — B2 CAL artifact run `aa89db57…` (n=86,972); `configs/cost_model_v2.yaml` (v1 + tier_b1/tier_b2 estimated amortized pricing, params byte-identical; sha `2c969255…`); (τ_A, τ_B) joint 2-D fit on CAL; `results/frontier/frontier__opv2__cost-2c969255.json` pending **[]**, pre-Tier-B claims reproduce byte-identically. **Dominance shifted:** `a_to_b` new best full-slice row (beats a_only/b1_only/b2_only, cost CIs excl. 0); `a_to_b_to_c` 786.10/1k = cheapest Phase 4 policy; incumbents a_to_human + a_to_c both now fail to beat b2_only. **Verdict: "gate not LLM" restated — what pays is routing to cheap capacity; gate certified two-axis in front of B2 (−$21.49/1k AND +0.0025 F1) but worth ~4× less than in front of A; C rung still not established (n=5,000).** Acceptance "dominates ≥2 single tiers" now met by a_to_b at 3; `headline_router` repoint = open owner decision | EXPERIMENT_LOG 2026-08-11 Phase 4 backfill; `make tier-b-frontier` |

### Phase 5 — Drift protocol  ✅ **COMPLETE 2026-08-12** (B2 prior-shift closed the last approved slot; B1 yearly / Tier B perturbation / OOV-centroid = descoped by owner; novel-class probe = unstarted stretch)
| Task | Status | Evidence |
|---|---|---|
| Rolling yearly evals (2023–2026) | **Tier A done** (2026-08-08); **Tier C done** (2026-08-09, owner-approved; 8 runs, $30.48 actual vs $30.94 projected, cumulative $67.38 of ≈$75); **Tier B2 done** (2026-08-12; runs `7224d2c1…`/`eed4b95c…`/`87a5305b…`/`59a81153…`, $0 MPS; macro-F1 0.790→0.774→0.760→**0.726** — tracks the LLMs on aggregate (2026-H1: A 0.666 / B2 0.726 / Haiku 0.728 / Sonnet 0.782) but inherits Tier A's class-level collapse, credit_reporting F1 0.899→0.285; T=1.3192 identical all slices); **B1 yearly series = descoped by owner (2026-08-12)** — ~8 h MPS for a model B2 dominates on every TEST-IID metric; no narrative value; slot retained as a label, do not run | EXPERIMENT_LOG 2026-08-08 Phase 5 task 1 (Tier A: macro-F1 0.758→0.748→0.730→0.666, 2026-H1 cliff = credit_reporting prior shift) + 2026-08-09 tasks 2a/2b (Tier C: Haiku/Sonnet tied 2023–25 paired CIs ∋ 0; 2026-H1 Sonnet +0.054 F1 paired CI excl. 0, p≈1e-15; Tier C degrades slower than Tier A — 2026-H1: A 0.666 / Haiku 0.728 / Sonnet 0.782) |
| Prior-shift decomposition | **done for all evaluated tiers** (A + C 2026-08-09; **B2 added 2026-08-12**, $0 derivation-only, native scope; pre-existing A/C rows reproduce bit-identically) | EXPERIMENT_LOG 2026-08-09 Phase 5 task 3 + 2026-08-12 task 5; `results/prior_shift/` (15 decompositions + summary.json); Tier A 2026-H1: 4.2 of 9.2 pts prior-shift alone [3.9, 4.5], credit_reporting = 7.6 pts; LLMs pay the prior penalty (Haiku +3.0 [1.4, 4.4]) but within-class holds (≈0/negative); **B2 2026-H1: total +0.0639 = prior +0.0303 [+0.0266, +0.0337] + within +0.0336 [+0.0234, +0.0451] — pre-registered "LLM-like near-zero within" REFUTED: B2 is Tier-A-shaped (share_prior 0.4744 vs A 0.4549, CIs overlap), same wound at ~70% magnitude; credit_reporting = 6.9 of 6.4 pts; fine-tuning bought loss magnitude, not loss shape** |
| OOV tracking | **done** (2026-08-09; model-free, $0; dense-encoder centroid variant **descoped by owner 2026-08-12** — OOV hypothesis already refuted model-free, closed question) | EXPERIMENT_LOG 2026-08-09 Phase 5 task 4; `results/oov/` (6 slices + summary.json); OOV hypothesis **refuted**: model-vocab token OOV 0.545%→0.773% TRAIN→2026-H1 (+0.23 pp), 99.2% of 2026-H1 token mass still in-vocab; TF-IDF centroid distance peaks at 2025 and *falls* at 2026-H1 (CIs disjoint) — lexical drift ruled out as cause of Tier A cliff, prior shift stands; types-vs-tokens gap (53.4% types vs 0.77% tokens OOV at 2026-H1) = grown-up CoNLL finding |
| Perturbation robustness | **done for available tiers** (Tier A 2026-08-09, 16 runs, $0; Tier C 2026-08-10 owner-approved option A, 4,500 calls, $6.287 actual vs $5.92 nominal / ≈$6.8 ceiling — cumulative ≈$73.67 of ≈$75). Tier A: typo −4.4 F1 pts @ 0.10, ocr ~−1, case = exact structural zero; char-n-gram shield directional (word-only −6.6). Tier C Haiku @ 0.10: typo −3.1 pts (CI excl. 0, smallest of any tier), ocr and case ∅ (CIs ∋ 0) despite case's largest token inflation (+8.8%); perturbation = +3–9% Tier C serving-cost tax. Cross-tier: LLM ≳ word+char > word-only. Tier B rows **descoped by owner 2026-08-12** — ranking already measured and bracketed; a B2 subword point would refine ordering, not change any claim | EXPERIMENT_LOG 2026-08-09 task 5 + 2026-08-10 task 5b; `results/perturbation/summary.json` (18 rows); Tier A runs `4bbd26c4…`–`3f5eff59…`, Tier C runs `182774b6…`/`237bffae…`/`c7e53e2a…` |
| Novel-class probe (stretch) | pending | — |
| Drift charts from results log; escalation-rate-over-time; evidence classes labeled | **done for A + B2 + C** (2026-08-10 available tiers; **Tier B2 series + a_to_b escalation arms added 2026-08-12**, $0 derivation-only: tier_b2 on macro-F1 + ECE charts; a_to_b at frozen τ_A 0.6449/0.7981 (full_cal/paired, replay-verified, B2-terminal — no B→human arm exists in the frozen family, none invented); escalate-to-B2 rate quasi-flat 0.296–0.329 2022→2025 then **0.4849 [0.4780, 0.4921]** at 2026-H1 (+47% rel.); frozen-τ cascade slightly trails b2_only at the cliff (acc 0.7550 vs 0.7584) — τ-staleness finding; module cost config v1→v2, pre-existing τ byte-identical; TEST-IID arm reproduces frozen frontier point bit-for-bit; B1 series = explicit pending slot). Escalation self-adjusts late and abruptly: a_to_human flat at CAL op point 2022→2025 (0.095–0.103 vs 0.0994) then **0.1674 [0.1623, 0.1724]** at 2026-H1 (+68% rel.); selective gate worth ~5 acc pts at the cliff (answered 0.7436 vs full 0.6918); a_to_c parse-fail→human ≤0.27% everywhere (Haiku exactly 0) | EXPERIMENT_LOG 2026-08-10 Phase 5 task 6; `results/drift/summary.json` + `results/drift/charts/` (3 SVGs, evidence-class footnotes); `make drift-charts` |

### Phase 6 — Demo + case study  ▶ **IN PROGRESS (scaffold done 2026-08-10; Tier B panels pending Tier B)**
| Task | Status | Evidence |
|---|---|---|
| Static demo **site scaffold** (triage playground, frontier plot, policy builder, drift timeline, calibration panel, receipts drawer) | **done** (2026-08-10, $0, no new API calls, runs.jsonl untouched). `demo/` static site (vanilla JS, no external deps) + `demo/data/` (9 committed JSONs) built deterministically by `make demo-data` (`src/triage_lab/demo_build.py`, contract in `demo/DATA_CONTRACT.md`); curated set n=200 FROZEN (seed 20260806, pool = 1,500 Haiku∩Sonnet TEST-IID receipt ids, narratives from frozen splits); traceability test-enforced (`tests/test_demo_build.py`, 43 passed; CI-safe subset without `data/`); browser-verified light+dark, zero console errors; all Tier B panels explicit **pending Tier B** slot placeholders. Live ONNX inference deferred to a later Phase 6 task | EXPERIMENT_LOG 2026-08-10 Phase 6 task 1 |
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
4. ~~**Phase 6 — Site scaffold.** Build the static demo structure and wire in available exhibits.~~
   **Done 2026-08-10** (EXPERIMENT_LOG Phase 6 task 1). Any panel/number sourced from Tier B is
   **pending Tier B**. Remaining Phase 6 tasks, in order: live in-browser Tier A inference +
   case study page (verification + "does not prove"), provenance links, `make reproduce-headline`.

**Tier B training: DONE (ingested + validated 2026-08-10).** Backfill order for the coming
sessions (one task per session): ~~**(1)** Tier B harness finals — the four configs on TEST-IID with
temperature scaling (runbook §6), then B1-vs-A / B1-vs-B2 paired deltas + seed variance~~
**done 2026-08-10** (EXPERIMENT_LOG eval backfill task 1; headline: B2 DistilBERT 0.7950 tops all
B1 seeds, paired CIs excl. 0 — the pre-registered surprise held under the frozen protocol);
~~**(2)** DistilBERT int8 ONNX export + parity (runbook §7)~~ **done 2026-08-11**
(EXPERIMENT_LOG 2026-08-11; agreement 0.9944 ✓ with per-channel QInt8 — Phase 2 fully closed);
**(3)** every item marked **pending Tier B** above — ~~Phase 4 frontier points~~ **done
2026-08-11** (EXPERIMENT_LOG Phase 4 backfill; dominance table shifted — a_to_b is the new
best full-slice row); ~~Phase 5 drift rows~~ — **B2 yearly evals + drift charts done
2026-08-12** (EXPERIMENT_LOG 2026-08-12; Phase 5 Tier B sub-slots resolved by owner 2026-08-12:
**B2 prior-shift decomposition done** (EXPERIMENT_LOG 2026-08-12 task 5 — B2 Tier-A-shaped,
not LLM-shaped, within-class term +0.0336 CI excl. 0), B1 yearly series / Tier B perturbation
rows / OOV dense-encoder centroid variant all **descoped by owner** with rationale recorded);
then Phase 6 panels — **next** (`make demo-data` regen belongs there — demo/data stale at 46
of now-55 runs, two known failing demo tests until then — plus the 2026-08-12 owner decision:
`headline_router` repoint to `a_to_b`, Haiku cascade kept as the LLM-cascade contrast exhibit).

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
