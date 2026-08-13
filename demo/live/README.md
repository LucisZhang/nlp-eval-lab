# demo/live/

Assets for live, in-browser (WebAssembly) inference used by the Phase 6 static
demo's "Live in-browser inference" section (see `demo/index.html` /
`demo/assets/app.js`, engine module `demo/assets/live.js`).

## Contents

- `tier_a/tier_a_live.json` — exported Tier A (TF-IDF + calibrated linear
  model) weights, loaded and run entirely client-side by `loadTierA()`.
- `tier_b2/model.int8.onnx` + `tier_b2/tokenizer.json` +
  `tier_b2/tokenizer_config.json` — the DistilBERT int8 ONNX export and its
  tokenizer, run client-side via `onnxruntime-web` (vendored in
  `demo/vendor/ort/`) by `loadTierB2()`.
- `agreement.html` — a standalone harness (not part of the main demo nav)
  used to measure browser-engine-vs-official-Python agreement on the curated
  ~200-sample set.
- `agreement_report.json` — the harness's output: label agreement rates for
  Tier A and Tier B2 (browser int8 vs. the frozen Python harness, and for
  Tier B2 also vs. Python int8). The main demo fetches this lazily and
  renders the actual rates inline in its disclosure note; if the file is
  absent, the demo shows "agreement report pending" instead of failing.

## The approximate-implementation principle

Everything under `demo/live/` is a **best-effort, re-implemented, in-browser
approximation** of the corresponding tier — not the official measurement.
Differences can arise from int8 quantization, a from-scratch JS/WASM
inference pipeline, tokenizer edge cases, and floating-point nondeterminism
across browsers/devices. The demo UI is required to disclose this everywhere
a live prediction is shown, and to point back at the authoritative source:

- **Official numbers** are the frozen, append-only harness records in
  `results/runs.jsonl`, exposed in the demo via the receipts drawer
  (`demo/data/runs_index.json`, `demo/data/receipts.json`).
- Live in-browser predictions are for interactive, qualitative
  demonstration only. They are never used to compute or override any
  headline metric.

## `demo/data/` is untouched

Live inference reads only from `demo/live/`. It never reads, writes, or
otherwise touches the precomputed panel data contract files in
`demo/data/` (`samples.json`, `frontier.json`, `policies.json`, `drift.json`,
`calibration.json`, `runs_index.json`, `receipts.json`). Those remain the
sole source of truth for every non-live panel and for the precomputed
per-tier cards shown alongside live results in the triage playground.
