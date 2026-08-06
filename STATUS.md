# STATUS.md — Execution State

**Single source of truth for _execution state_.** Scope, architecture, metrics, and phase
acceptance criteria live in **`UPGRADE_PLAN.md`** (untouched by this file). When the two ever
appear to disagree, UPGRADE_PLAN.md defines _what_ and _why_; STATUS.md tracks _where we are_ and
_what's next_.

- **Owner decision (2026-08-06):** The A6000 is unavailable until the weekend, so **all Tier B
  training and eval is BLOCKED-until-weekend.** Work has been reordered to make maximal progress on
  the tiers and infrastructure that do not need the GPU.
- Last updated: 2026-08-06.

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

### Phase 3 — Tier C LLM  ▶ **NEXT (first authorized task)**
| Task | Status | Evidence |
|---|---|---|
| Tier C prompt + structured-output schema (versioned, content-hashed) | pending | — |
| Zero-shot vs few-shot ablation on CAL (Haiku) | pending | — |
| Smoke run + cost approval gate (see execution order) | pending | — |
| Haiku 4.5 full eval — TEST-IID + TEST-POSTCUTOFF | pending | — |
| Sonnet 5 few-shot subsample | pending | — |
| Both Tier C points with CIs on both slices; measured $/1k + p50/p95 latency; Haiku-vs-Sonnet paired delta; contamination delta; raw API logs retained | pending | — |

### Phase 4 — Calibration + router
| Task | Status | Evidence |
|---|---|---|
| Risk-coverage machinery | pending | — |
| Cost-model implementation | pending | — |
| Threshold optimization on CAL | pending | — |
| Router simulator vs all policies | pending | — |
| Two headline frontier claims with CIs; router dominates ≥2 single tiers (or honest diagnosis) | pending — **partial**: build infra against Tier A (+ Tier C when ready) now; Tier B frontier points **pending Tier B** | — |

### Phase 5 — Drift protocol
| Task | Status | Evidence |
|---|---|---|
| Rolling yearly evals (2023–2026) | pending — run for **available tiers** (A, then C); Tier B rows **pending Tier B** | — |
| Prior-shift decomposition | pending | — |
| OOV tracking | pending | — |
| Perturbation robustness | pending | — |
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
   2. **Smoke run** (tiny subsample) → measure real per-call token cost → **stop and get cost
      approval before any full run.** (CLAUDE.md rule 6: OpenRouter only; cost from actual
      per-call token usage; record upstream provider per call.)
   3. After approval: zero-shot vs few-shot ablation on CAL, then full Haiku 4.5 (TEST-IID +
      TEST-POSTCUTOFF) and Sonnet 5 few-shot subsample.
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
