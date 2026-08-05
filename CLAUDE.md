# CLAUDE.md — Triage Router Lab

## What this repo is

A three-tier consumer-complaint triage evaluation lab built on the CFPB Consumer Complaint Database: **Tier A** (TF-IDF + calibrated linear models), **Tier B** (fine-tuned transformers — ModernBERT-base headline, DistilBERT deployment point), **Tier C** (Claude LLMs, few-shot with structured output), combined into a **confidence-cascade router** optimized against an explicit business cost model, and stress-tested against measured 2015→2026 distribution drift.

**`UPGRADE_PLAN.md` is the single source of truth** for scope, architecture, metrics, phase acceptance criteria, and cut-lines. Read the relevant section before starting any phase task.

## Repo layout (UPGRADE_PLAN.md §4.3)

| Path | Purpose |
|---|---|
| `data/` | Datasets, snapshots, splits — **gitignored**, reproduced via `make data` |
| `src/triage_lab/` | All library code (ingest, models, eval harness, router) |
| `configs/` | One YAML per experiment run (model, data slice, seed, prompt hash) |
| `results/` | Append-only JSONL run log — **committed** |
| `EXPERIMENT_LOG.md` | Dated hypothesis → result → verdict log (incl. failed hypotheses) |
| `demo/` | Static demo site (precomputed JSON + in-browser ONNX) |
| `seeds/` | Ported coursework exhibits (optional Tier-0 NB port) |
| `docs/seed-evidence/` | Coursework provenance evidence — **read-only** |

## Hard rules

1. **`docs/seed-evidence/` is read-only.** It is markdown citation evidence copied from the coursework archive, used only for the case study's provenance section (UPGRADE_PLAN.md Appendix A). NEVER edit, refactor, execute, or "clean up" anything in it. It is not code to run.
2. **Splits and seed lists freeze once generated.** Temporal splits (TRAIN / CAL / TEST-IID / TEST-DRIFT / TEST-POSTCUTOFF), random seed lists, and few-shot exemplar selections are frozen the moment they are first materialized. `make data` must reproduce them byte-identically from the frozen snapshot. NEVER re-cut a split or reselect exemplars to improve a number. TEST-* slices are touched only for final reported runs; all iteration happens on CAL.
3. **`results/*.jsonl` is append-only.** NEVER edit or delete an existing record. A correction is a new record that references the superseded run id. Every record carries: metrics + CIs, git SHA, dataset snapshot hash, config/prompt hash, wall-clock, and cost.
4. **Tier C prompts are versioned.** Every prompt template lives in a versioned file with a content hash; every Tier C run record includes that hash. Changing a prompt in any way = a new version, never an in-place edit.
5. **Every portfolio number has a reproduction command.** Any metric destined for the case study / portfolio site must have its exact reproduction command recorded in `EXPERIMENT_LOG.md` alongside the hypothesis → result → verdict entry. No command, no claim.
6. **Tier C goes through OpenRouter, not the Anthropic API.** Use an OpenAI-compatible client with `base_url="https://openrouter.ai/api/v1"` and key `OPENROUTER_API_KEY` from `.env` (never committed; `.claude/settings.json` denies reading it — do not try). Cost figures MUST come from actual per-call token usage × published per-MTok prices — never estimated from character counts or averages. Every logged call MUST record which upstream provider served it (OpenRouter response metadata).
7. **CFPB data only.** The dataset is the CFPB Consumer Complaint Database (US-government work, public domain). The coursework datasets (Reuters, CoNLL-2003) are research-licensed and were deliberately excluded — they must NEVER enter this repo in any form.

## Workflow conventions

- Python 3.12, `uv` with a committed lockfile; `uv sync --frozen` is the only supported install path.
- DuckDB for ingest and split materialization.
- Eval harness: one YAML config per run → one JSONL record appended to `results/`. Every headline number carries a 95% bootstrap CI (n=1,000, fixed seed); comparison claims require the paired CI to exclude zero.
- Single-variable experiment discipline: one config delta per run, logged in `EXPERIMENT_LOG.md`.
- Each phase task follows: restate the phase's ✅ Accept criteria → implement → run the proving command → commit as `phaseN: <task name>`.

## Phase roadmap (condensed from UPGRADE_PLAN.md §8 — see there for full ✅ Accept lines)

| Phase | Scope | Accept (gist) |
|---|---|---|
| 0 | Repo, uv lock, CI; CFPB snapshot freeze; ingest, taxonomy map, dedup, temporal splits, datasheet | `make data` byte-identical; datasheet complete; cross-split leakage check green in CI |
| 1 | Eval harness (YAML→JSONL, bootstrap CIs) + Tier A | Harness tests green; Tier A CI'd on TEST-IID; ≥3 logged runs |
| 2 | Tier B fine-tunes (ModernBERT ×3 seeds, DistilBERT) + ONNX export | Both points CI'd; B1-vs-A and B1-vs-B2 deltas; ONNX parity ≥99% |
| 3 | Tier C via OpenRouter (Haiku 4.5 full budget, Sonnet 5 subsample) | Both points CI'd on TEST-IID + POSTCUTOFF; measured $/1k, latency; paired deltas |
| 4 | Calibration + router (cost model, threshold optimization) | Headline frontier claims with CIs; router dominates ≥2 single tiers (or honest diagnosis) |
| 5 | Drift protocol (yearly evals, prior-shift, OOV, perturbations) | Drift charts from results log; escalation-rate-over-time; evidence classes labeled |
| 6 | Static demo + case study + `make reproduce-headline` | Demo fully static; every number traces to a results record |
| 7 | (Stretch) PII cross-check tie-in | Cut freely |
