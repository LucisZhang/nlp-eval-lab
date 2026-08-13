# demo/data contract — Phase 6 site scaffold

Normative contract between `src/triage_lab/demo_build.py` (producer) and the static site
(`demo/index.html` + `demo/assets/`, consumer). Both sides follow this file exactly.
All files live in `demo/data/` and are **committed**. The build is **deterministic**:
no wall-clock timestamps; the only timestamps are those copied from results records.

Global conventions:

- Every metric object is `{"point": float, "ci_lo": float, "ci_hi": float}` (95% bootstrap,
  copied — never recomputed — from the source results record) or `{"point": float}` when the
  source has no CI.
- Every number-bearing object carries provenance: `"run_id"` (full 64-char id) and/or
  `"source"` (repo-relative path of the artifact it was copied from). The frontend renders a
  clickable run-id chip everywhere a number appears; the chip opens the receipts drawer.
- Pending slots are real objects: `{"pending": true, "slot": "<key>", "label": "..."}`
  — never omitted keys. **Tier B backfilled 2026-08-12**: every scaffold-era pending slot is
  now real data except the Tier B1 yearly drift series (descoped by owner 2026-08-12; slot
  retained, labeled).
- **Headline router = `a_to_b`** (owner decision 2026-08-12): the certified two-axis win on
  the full slice. `a_to_c_parsefail_human` (rendered as key `a_to_c_haiku`) stays as the
  LLM-cascade contrast exhibit. Router points/policies carry a boolean `"headline"` field.
- The payload builds under `configs/cost_model_v2.yaml` (the generation that prices Tier B);
  the build hard-fails under a config that does not price Tier B.
- `evidence_class` ∈ {"measured", "estimated", "projected", "derived"} on every exhibit.

## Files

### 1. `meta.json`
`{"schema_version": "demo-v1", "git_sha": "<HEAD at build>", "snapshot_sha256": "<from run records>",
"op_version": "<primary router op version, v2 isocal>", "cost_model": {"path": "configs/cost_model_v2.yaml",
"sha256": "..."}, "evidence_classes": {...legend...},
"headline_router": {"policy": "a_to_b", "evaluation_set": "full_test_iid", "note": "<owner decision 2026-08-12>"},
"pending_tier_b": ["tier_b1"]}` — the one remaining slot is the drift panel's B1 yearly series.

### 2. `runs_index.json`
`{"<run_id>": {"config_name", "config_path", "config_sha256", "slice", "timestamp_utc",
"git_sha", "dataset": {...as logged...}, "metrics": {...verbatim from runs.jsonl...},
"cost_usd", "wall_clock_seconds", "tier": "A|C", "model_label": "<human label>",
"extra": {...verbatim if present...}}}` — one entry per non-header record in `results/runs.jsonl`.

### 3. `frontier.json`
```
{"claims": {...verbatim primary frontier claims JSON..., "source": "results/frontier/<file>"},
 "points": [{"key": "tier_a_logreg", "label": "Tier A — TF-IDF LogReg", "kind": "single|router",
   "run_id": "...", "cost_model_source": "results/cost_model/<run_id>.json",
   "cost_per_1k_usd": {point, ci_lo, ci_hi}, "macro_f1": {point, ci_lo, ci_hi},
   "evidence_class": "measured"}...],
 "headline_note": "<owner decision 2026-08-12>",
 "pending_points": []}
```
Points (10 single + router): Tier A LogReg, Tier A CNB, Tier B1 ModernBERT ×3 seeds
(`tier_b1_sa/sb/sc` — seed variance is the exhibit, never a mean), Tier B2 DistilBERT
(`tier_b2`), Haiku (TEST-IID), Sonnet (TEST-IID subsample), and routers `a_to_human`,
`a_to_b` (**`"headline": true`**), `a_to_c_haiku` (LLM-cascade contrast), `a_to_b_to_c`.
Router points carry `"headline"` (bool). Cost axis = `expected_cost_per_1k.total` for
router policies and the api/compute measured cost for single tiers (state which in a
`"cost_basis"` string field). Tier B single-tier costs are the cost config's
amortized-compute ESTIMATE priced through the same cost_model artifacts as every tier.

### 4. `policies.json`
```
{"op_version": "...", "cost_defaults": {"c_misroute": ..., "c_human": ..., "source": "configs/cost_model_v2.yaml", "sha256": "..."},
 "headline_note": "<owner decision 2026-08-12>",
 "policies": [{"key": "a_to_human", "label": "...", "headline": bool, "tau": {...frozen tau + source...},
   "run_refs": [...], "n": ..., "rates": {"answered_a": f, "escalated": f, "human": f},
   "p_error_machine": f, "api_cost_per_1k_usd": {...}, "macro_f1_system": {point,ci_lo,ci_hi},
   "accuracy_system": {point,ci_lo,ci_hi}, "expected_cost_per_1k": {...at defaults, with CI, verbatim from cost_model/router_sim...},
   "evidence_class": "measured", "source": "results/router_sim/<file>"}...],
 "tau_sweep_a_to_human": {"slice": "cal", "calibration": "isotonic", "run_id": "<cal isocal run>",
   "grid": [{"tau": f, "coverage": f, "acc_answered": f, "misroute_rate_answered": f, "human_rate": f}...],
   "note": "sweep computed from the frozen CAL per-example artifact; enables live threshold re-solve for the A->human arm only",
   "evidence_class": "derived"},
 "frozen_tau_note": "a_to_c, a_to_b and a_to_b_to_c taus are frozen from Phase 4 CAL optimization; ..."}
```
Four policies: `a_to_human`, `a_to_b` (**`"headline": true`**, full slice, B2-terminal,
human rate structurally 0), `a_to_c_haiku` (LLM-cascade contrast, paired subset),
`a_to_b_to_c` (paired subset; additionally carries `"tau_b"` — the frozen second gate
`{value, cal_tau_b_star, cal_coverage_b_marginal}`; single-gate policies have no `tau_b`
key). Slider re-solve semantics (frontend): expected cost is linear in (c_misroute,
c_human, api price scale) given each policy's measured rates — recompute client-side from
`rates`, `p_error_machine`, `api_cost_per_1k_usd`; re-solve tau only for `a_to_human` via
the sweep grid. The api multiplier scales Tier B's amortized compute charge together with
real API spend (both live in `api_cost_per_1k_usd`).

### 5. `drift.json`
Verbatim copy of `results/drift/summary.json` under `"summary"` (which carries the
tier_b2 yearly series and the two `a_to_b` escalation arms since 2026-08-12), plus:
`"annotations": [{"x": "2023-04", "label": "CFPB taxonomy consolidation"}, {"x": "2026-H1", "label": "credit_reporting prior-shift cliff"}]`,
`"pending_series": [{"pending": true, "slot": "tier_b1", "label": "Tier B1 yearly drift series — descoped by owner 2026-08-12 ..."}]`,
`"source": "results/drift/summary.json"`. The frontend keys escalation/policy line series
by the full arm identity (policy + terminal model + tau-fit dataset), never by policy
alone — the same policy legitimately ships more than one arm.

### 6. `calibration.json`
```
{"exhibits": [{"key": "tier_a_logreg_raw", "label": "...", "run_id": "...", "slice": "...",
   "calibration": "raw|isotonic", "bins": [{"lo": f, "hi": f, "n": int, "conf_mean": f, "acc": f}... 15 bins],
   "ece": {point,ci_lo,ci_hi}, "brier": {point,ci_lo,ci_hi}, "evidence_class": "measured (bins derived from frozen per-example artifact)"}...],
 "tier_c_note": {"text": "Tier C structured output emits a degenerate one-hot p_max — no calibration signal to plot; parse-failure is its only self-signal (UPGRADE_PLAN §4.2 amendment).", "run_ids": [...]},
 "pending": []}
```
Seven exhibits: the three Tier A ones (raw CAL / isotonic CAL / isotonic TEST-IID) plus
the four Tier B TEST-IID finals (`tier_b1_sa/sb/sc`, `tier_b2`) with
`"calibration": "temperature"` (fit on CAL). Bins computed from
`data/preds/<run_id>.parquet` (p_max vs correctness, 15 equal-width bins); ECE/Brier
copied from the logged run metrics; bins must reproduce the logged ECE to 1e-9 (hard gate).

### 7. `samples.json` + 8. `curated_ids.json`
`curated_ids.json`: `{"version": "v1", "seed": 20260806, "method": "class-stratified proportional (largest remainder, min 1/class), ids sorted before draw",
"pool": "TEST-IID complaint_ids scored by BOTH Haiku and Sonnet finals (paired 1,500)", "n": 200, "complaint_ids": [ints, sorted]}`
**Frozen once committed**: on rebuild, `demo_build.py` regenerates the selection and hard-fails
if it differs from the committed file (same rule as exemplars).

`samples.json`:
```
{"selection": {...copied from curated_ids.json..., "narrative_source": "frozen split parquet (CFPB, US-gov public domain)"},
 "samples": [{"complaint_id": int, "narrative": "<full text>", "y_true": "<label>",
   "tiers": {
     "tier_a_logreg": {"label": "...", "p_max": f, "correct": bool, "run_id": "..."},
     "tier_b1": {"label": "...", "p_max": f, "correct": bool, "run_id": "..."},  // seed sa; see samples.json tier_b1_note
     "tier_b2": {...same shape...},
     "haiku": {"label": "...", "correct": bool, "cost_usd": f, "latency_ms": f, "provider": "...",
               "prompt_tokens": int, "completion_tokens": int, "parse_failed": bool, "run_id": "..."},
     "sonnet": {...same shape...}},
   "router": {"op_version": "...", "policy": "a_to_b", "tau": f,
              "path": ["A","answered"] | ["A","escalated","B2","answered"],
              "note": "decision recomputed with the frozen Phase 4 op (reuse router_sim logic)"}}]}
```
The per-sample router path is the HEADLINE router (`a_to_b`, owner decision 2026-08-12):
B2-terminal, so no human arm exists and every path ends "answered". The replay gate is
exact: `np.where(p_max_A >= tau, A_label, B2_label)` must reproduce router_sim's
`policy.y_pred` vector or the build fails. Tier responses come ONLY from committed
receipts / frozen preds artifacts — no API calls.

### 9. `receipts.json`
Per Tier C run: `{"run_id", "config_name", "model", "raw_log_path", "receipts_sha256" (if logged),
"n_calls", "total_cost_usd", "provider_mix": {"<provider>": count}, "token_totals": {"prompt": int, "completion": int},
"parse_failures": int}` — aggregated from `results/tier_c_raw/**/calls.jsonl`. Plus
`{"repro": {"results_log": "results/runs.jsonl", "note": "append-only; corrections reference superseded run ids"}}`.
