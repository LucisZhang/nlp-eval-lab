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
- Pending Tier B slots are real objects: `{"pending": true, "slot": "<key>", "label": "..."}`
  — never omitted keys. The backfill will replace them in place.
- `evidence_class` ∈ {"measured", "estimated", "projected", "derived"} on every exhibit.

## Files

### 1. `meta.json`
`{"schema_version": "demo-v1", "git_sha": "<HEAD at build>", "snapshot_sha256": "<from run records>",
"op_version": "<primary router op version, v2 isocal>", "cost_model": {"path": "configs/cost_model_v1.yaml",
"sha256": "..."}, "evidence_classes": {...legend...}, "pending_tier_b": ["<slot keys>"]}`

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
 "pending_points": [{"pending": true, "slot": "tier_b1_modernbert", "label": "Tier B1 — ModernBERT-base (3 seeds)"},
                    {"pending": true, "slot": "tier_b2_distilbert", "label": "Tier B2 — DistilBERT int8 ONNX"},
                    {"pending": true, "slot": "router_a_b_c", "label": "Router A→B→C"}]}
```
Points: Tier A LogReg, Tier A CNB, Haiku (TEST-IID), Sonnet (TEST-IID subsample), router
`a_to_human`, router `a_to_c_haiku` — whichever have both a cost_model artifact and a
macro-F1 with CI. Cost axis = `expected_cost_per_1k.total` for router policies and the
api/compute measured cost for single tiers (state which in a `"cost_basis"` string field).

### 4. `policies.json`
```
{"op_version": "...", "cost_defaults": {"c_misroute": ..., "c_human": ..., "source": "configs/cost_model_v1.yaml", "sha256": "..."},
 "policies": [{"key": "a_to_human", "label": "...", "tau": {...frozen tau + source...},
   "run_refs": [...], "n": ..., "rates": {"answered_a": f, "escalated": f, "human": f},
   "p_error_machine": f, "api_cost_per_1k_usd": {...}, "macro_f1_system": {point,ci_lo,ci_hi},
   "accuracy_system": {point,ci_lo,ci_hi}, "expected_cost_per_1k": {...at defaults, with CI, verbatim from cost_model/router_sim...},
   "evidence_class": "measured", "source": "results/router_sim/<file>"}...],
 "tau_sweep_a_to_human": {"slice": "cal", "calibration": "isotonic", "run_id": "<cal isocal run>",
   "grid": [{"tau": f, "coverage": f, "acc_answered": f, "misroute_rate_answered": f, "human_rate": f}...],
   "note": "sweep computed from the frozen CAL per-example artifact; enables live threshold re-solve for the A->human arm only",
   "evidence_class": "derived"},
 "frozen_tau_note": "a_to_c tau is frozen from Phase 4 CAL optimization; the demo does not re-solve it (Haiku scored only the paired subset on CAL)."}
```
Slider re-solve semantics (frontend): expected cost is linear in (c_misroute, c_human, api
price scale) given each policy's measured rates — recompute client-side from `rates`,
`p_error_machine`, `api_cost_per_1k_usd`; re-solve tau only for `a_to_human` via the sweep grid.

### 5. `drift.json`
Verbatim copy of `results/drift/summary.json` under `"summary"`, plus:
`"annotations": [{"x": "2023-04", "label": "CFPB taxonomy consolidation"}, {"x": "2026-H1", "label": "credit_reporting prior-shift cliff"}]`,
`"pending_series": [{"pending": true, "slot": "tier_b1"}, {"pending": true, "slot": "tier_b2"}]`,
`"source": "results/drift/summary.json"`.

### 6. `calibration.json`
```
{"exhibits": [{"key": "tier_a_logreg_raw", "label": "...", "run_id": "...", "slice": "...",
   "calibration": "raw|isotonic", "bins": [{"lo": f, "hi": f, "n": int, "conf_mean": f, "acc": f}... 15 bins],
   "ece": {point,ci_lo,ci_hi}, "brier": {point,ci_lo,ci_hi}, "evidence_class": "measured (bins derived from frozen per-example artifact)"}...],
 "tier_c_note": {"text": "Tier C structured output emits a degenerate one-hot p_max — no calibration signal to plot; parse-failure is its only self-signal (UPGRADE_PLAN §4.2 amendment).", "run_ids": [...]},
 "pending": [{"pending": true, "slot": "tier_b1_temp_scaling"}, {"pending": true, "slot": "tier_b2_temp_scaling"}]}
```
Bins computed from `data/preds/<run_id>.parquet` (p_max vs correctness, 15 equal-width bins);
ECE/Brier copied from the logged run metrics.

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
     "tier_b1": {"pending": true, "slot": "tier_b1"}, "tier_b2": {"pending": true, "slot": "tier_b2"},
     "haiku": {"label": "...", "correct": bool, "cost_usd": f, "latency_ms": f, "provider": "...",
               "prompt_tokens": int, "completion_tokens": int, "parse_failed": bool, "run_id": "..."},
     "sonnet": {...same shape...}},
   "router": {"op_version": "...", "policy": "a_to_c_haiku", "tau": f,
              "path": ["A","answered"] | ["A","escalated","C","answered"] | ["A","escalated","C","human"],
              "note": "decision recomputed with the frozen Phase 4 op (reuse router_sim logic)"}}]}
```
Tier responses come ONLY from committed receipts / frozen preds artifacts — no API calls.

### 9. `receipts.json`
Per Tier C run: `{"run_id", "config_name", "model", "raw_log_path", "receipts_sha256" (if logged),
"n_calls", "total_cost_usd", "provider_mix": {"<provider>": count}, "token_totals": {"prompt": int, "completion": int},
"parse_failures": int}` — aggregated from `results/tier_c_raw/**/calls.jsonl`. Plus
`{"repro": {"results_log": "results/runs.jsonl", "note": "append-only; corrections reference superseded run ids"}}`.
