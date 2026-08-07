# EXPERIMENT_LOG

Dated hypothesis → result → verdict log. One entry per run; single config delta
between consecutive ladder rungs. All iteration on CAL; TEST-* only for final
reported runs. Every portfolio-bound number carries its reproduction command
(CLAUDE.md rule 5). CIs are 95% bootstrap (n=1,000, seed=20260805).

---

## 2026-08-05 — Tier A rung 1: LogReg, word TF-IDF, CAL

- **Config:** `configs/tier_a_logreg_word_cal.yaml` (run `d35c23d2`, split=cal)
- **Hypothesis:** TF-IDF word 1–2-grams + class-weighted LogisticRegression is a
  strong classical baseline for CFPB product triage.
- **Result:** macro-F1 **0.7535** [0.7491, 0.7576]
- **Verdict:** Baseline established; anchor for the ladder.
- **Repro:** `uv run python -m triage_lab.harness configs/tier_a_logreg_word_cal.yaml`

## 2026-08-05 — Tier A rung 2: + char_wb 3–5-grams, CAL

- **Config:** `configs/tier_a_logreg_wordchar_cal.yaml` (run `abcadd53`, split=cal)
- **Single delta vs rung 1:** `features.char.enabled: false → true`
- **Hypothesis:** char n-grams (typo/OOV robustness) improve macro-F1 over
  word-only features.
- **Result:** macro-F1 **0.7466** [0.7425, 0.7505] — point estimate *below*
  rung 1; the two CIs overlap slightly. No paired test was run, so no
  directional claim is made (paired CI excluding zero required for claims).
- **Verdict:** **Hypothesis not supported on CAL.** Char n-grams did not improve
  IID macro-F1. They are retained in the frozen final config on the expectation
  of robustness value under drift/perturbation (to be tested in Phase 5); if
  Phase 5 shows no robustness benefit either, the word-only variant is the
  better Tier A point.
- **Repro:** `uv run python -m triage_lab.harness configs/tier_a_logreg_wordchar_cal.yaml`

## 2026-08-05 — Tier A rung 3: ComplementNB, word+char, CAL

- **Config:** `configs/tier_a_cnb_wordchar_cal.yaml` (run `c78c9d07`, split=cal)
- **Single delta vs rung 2:** `model.family: logreg → complement_nb`
  (with family-specific params: alpha=0.3, norm=false)
- **Hypothesis:** ComplementNB, designed for imbalanced text classification,
  is competitive with LogReg at a fraction of the training cost.
- **Result:** macro-F1 **0.6716** [0.6674, 0.6759] — CIs disjoint from rung 2
  by a wide margin (≈ −7.5 points vs LogReg on identical features).
- **Verdict:** **Not competitive on macro-F1.** LogReg is the primary Tier A
  family; CNB is kept as the cheap secondary point for the cost/quality
  frontier (Phase 4 router), not as a quality contender.
- **Repro:** `uv run python -m triage_lab.harness configs/tier_a_cnb_wordchar_cal.yaml`

## 2026-08-05 — Tier A FINAL: LogReg word+char + isotonic, TEST-IID

- **Config:** `configs/tier_a_logreg_test_iid.yaml` (run `8e4d6345`, split=test_iid)
- **Deltas vs rung 2 (frozen final config):** `calibration: none → isotonic`
  (fit on CAL), `split: cal → test_iid`
- **Hypothesis:** the frozen LogReg config holds ≈0.75 macro-F1 on the held-out
  IID test slice; isotonic calibration improves probability quality.
- **Result:** macro-F1 **0.7605** [0.7564, 0.7643]; accuracy 0.8444
  [0.8423, 0.8465]; ECE 0.1059 [0.1040, 0.1079]; Brier 0.2549; AURC 0.0500.
- **Verdict:** **Tier A headline point established.** Discrimination holds up on
  TEST-IID. Caveat logged honestly: even after isotonic calibration ECE remains
  high (0.106) — a Phase 4 calibration work item, not a Phase 1 blocker. SAGA
  emitted a max_iter=200 ConvergenceWarning (frozen config value; revisit only
  via a new config, never in place).
- **Repro:** `uv run python -m triage_lab.harness configs/tier_a_logreg_test_iid.yaml`

## 2026-08-05 — Tier A FINAL: ComplementNB word+char + isotonic, TEST-IID

- **Config:** `configs/tier_a_cnb_test_iid.yaml` (run `c20cd14a`, split=test_iid)
- **Single delta vs LogReg final:** `model.family: logreg → complement_nb`
- **Hypothesis:** CNB lands well below LogReg on macro-F1 but remains a viable
  ultra-cheap point for the Phase 4 cost/quality frontier.
- **Result:** macro-F1 **0.7265** [0.7222, 0.7308]; accuracy 0.8279
  [0.8256, 0.8300]; ECE **0.0250** [0.0230, 0.0270]; Brier 0.2605; AURC 0.0589.
- **Verdict:** As hypothesized on discrimination (CIs disjoint from LogReg's,
  ≈ −3.4 points; no paired delta run, so stated as an observation, not a formal
  claim). Surprise worth carrying forward: post-isotonic CNB is far *better
  calibrated* than post-isotonic LogReg (ECE 0.025 vs 0.106), though LogReg
  still wins selective-risk (AURC 0.050 vs 0.059) — relevant to Phase 4
  confidence-cascade thresholds.
- **Repro:** `uv run python -m triage_lab.harness configs/tier_a_cnb_test_iid.yaml`

## 2026-08-06 — Phase 3 step 1: Tier C prompt v1 + structured-output schema (versioned, content-hashed)

- **Artifacts:** `prompts/tier_c/v1/{prompt.yaml, schema.json, exemplars.json}` (frozen;
  any change = new version dir), loader `src/triage_lab/tier_c_prompt.py`, tests
  `tests/test_tier_c_prompt.py` (14), Makefile targets `tier-c-prompt` /
  `tier-c-exemplars-verify`. No API calls made; no configs or TEST-* slices touched.
- **Hypothesis:** a single parameterized prompt version (k=0 zero-shot / k=9 few-shot via
  `num_exemplars`) with a strict JSON-Schema label enum generated from `taxonomy.py`, and
  one seeded exemplar per harmonized class from TRAIN (seed 20260806, 200–1200 normalized
  chars, candidates sorted by complaint_id), satisfies hard rules 2 & 4: content-hashed,
  frozen, byte-reproducible from the frozen snapshot, zero leakage into CAL/TEST-*.
- **Result:** hashes — prompt.yaml `52f82550…`, schema.json `f533bb73…`, exemplars.json
  `809db598…`, **bundle_sha256 `f6777a96bc58f546b48d5f85ba47d683558c88504baab3333dbcd80e6b260fbe`**
  (this is the `prompt_hash` every Tier C run record must carry). 9 exemplars, one per
  class, all verified present in TRAIN (`train_sha256 939186e7…` integrity-gated) and
  absent from CAL/TEST-IID/TEST-DRIFT/TEST-POSTCUTOFF. `--verify-exemplars`: byte-identical
  regeneration OK. Full suite 117 passed; ruff clean. `--generate-exemplars` refuses to
  overwrite (freeze enforced in code, not just convention).
- **Verdict:** **Prompt/schema versioning infrastructure accepted.** k landed at 9 (one per
  class) against the plan's "k≈10" — stratified coverage beats a round number. Next task
  (per STATUS.md §b): smoke run on a tiny subsample → real per-call token cost → stop for
  cost approval before any full run.
- **Repro:** `make tier-c-prompt && make tier-c-exemplars-verify && uv run pytest -q tests/test_tier_c_prompt.py`

## 2026-08-06 — Phase 3 step 2: Tier C smoke run + measured per-call cost (cost-approval gate)

- **Artifacts:** OpenRouter runner `src/triage_lab/tier_c.py` (registered as `tier_c`; lazy
  `openai` import, optional dep group `tierc`), smoke configs
  `configs/tier_c_haiku_smoke_{fewshot,zeroshot}_cal.yaml` (25 CAL rows, seeded
  `eval_rows_cap`, only delta = `prompt.num_exemplars` 9 vs 0), tests `tests/test_tier_c.py`
  (25, network-free), Makefile target `tier-c-smoke`, per-call receipts under
  `results/tier_c_raw/`.
- **Hypothesis:** the frozen v1 prompt bundle (`f6777a96…`) + strict JSON-Schema structured
  output works end-to-end against `anthropic/claude-haiku-4.5` via OpenRouter, and real
  per-call token cost can be measured from actual usage × published per-MTok prices
  (CLAUDE.md rule 6) to gate the full-run budget.
- **Result:** two runs appended to `results/runs.jsonl`, both n=25 CAL, 0 parse failures,
  upstream provider Amazon Bedrock 25/25 on both, computed cost == OpenRouter-reported cost
  to the microdollar on both:
  - few-shot k=9 (`run_id e22fba2a…`): accuracy 0.80, 65,205 prompt + 238 completion tokens,
    **$0.066395 total = $0.002656/call**, latency p50 2.0 s / p95 8.4 s.
  - zero-shot k=0 (`run_id 77cbd36f…`): accuracy 0.76, 32,430 + 263 tokens,
    **$0.033745 total = $0.001350/call**, p50 1.6 s / p95 6.7 s.
  - Pricing snapshots (published, from `GET /models`): Haiku 4.5 $1/$5 per MTok;
    Sonnet 5 (`anthropic/claude-sonnet-5`, preflight only, no calls) $2/$10 per MTok.
  - Smoke macro-F1 (0.46 / 0.38) is not a quality claim: n=25 leaves several of 9 classes
    with zero support (hence the benign `metrics.py` zero-division RuntimeWarning); the
    smoke exists to measure cost, and accuracy 0.80 merely confirms the pipeline is sane.
- **Projection for approval (few-shot $0.002656/call Haiku, ≈$0.005312/call Sonnet):**
  ablation on CAL 1,500/arm ≈ $6.0; Haiku few-shot TEST-IID n=5,000 ≈ $13.3 and
  TEST-POSTCUTOFF n=5,000 ≈ $13.3; Sonnet 5 few-shot paired n=1,500 × 2 slices ≈ $15.9.
  **Total ≈ $48.5** (+ retry contingency < $55). Subsample sizes are proposals, not yet frozen.
- **Verdict:** **Smoke accepted; full runs NOT started.** Per STATUS.md §b the session stops
  here — awaiting owner cost approval (and subsample-size sign-off) before the ablation and
  any TEST-* run.
- **Repro:** `make tier-c-smoke` (live API; appends two new records with fresh run_ids;
  receipts land in `results/tier_c_raw/<config name>/<UTC ts>/calls.jsonl`)

## 2026-08-07 — Phase 3 step 3: zero-shot vs few-shot ablation on CAL (Haiku 4.5)

- **Configs:** `configs/tier_c_haiku_ablation_{fewshot,zeroshot}_cal.yaml` — both arms the
  identical seeded 1,500-row CAL subsample (`eval_rows_cap: 1500`, `cap_seed: 20260806`, so
  the pairing is exact); **single inter-arm delta: `prompt.num_exemplars` 9 vs 0.** Runner
  gained `model.params.max_concurrency` (thread pool, id-order-aligned aggregation,
  lock-guarded receipts) + read-only paired-compare CLI `triage_lab.tier_c_compare`; suite
  150 passed, ruff clean. Runs executed 2026-08-06 UTC (~16:00Z).
- **Hypothesis:** k=9 few-shot exemplars improve Haiku 4.5 macro-F1 over zero-shot on CAL
  enough to justify 2× per-call cost.
- **Result:** (both runs appended to `results/runs.jsonl`; 0 parse failures / 3,000 calls;
  provider Amazon Bedrock 3,000/3,000; computed cost == OpenRouter-reported both arms)
  - few-shot k=9 (`run_id c7598f84…`): macro-F1 **0.7674** [0.7351, 0.7986], accuracy
    0.8413; 3,848,973 prompt + 14,561 completion tokens; **$3.9218** ($0.002615/call);
    p50 1.39 s / p95 2.36 s.
  - zero-shot k=0 (`run_id 3f310951…`): macro-F1 **0.7567** [0.7217, 0.7891], accuracy
    0.8360; 1,882,473 + 16,091 tokens; **$1.9629** ($0.001309/call); p50 1.37 s / p95 2.33 s.
  - **Paired few−zero deltas (n=1,500):** accuracy **+0.0053 [−0.0060, +0.0167]**, macro-F1
    **+0.0107 [−0.0054, +0.0266]** — both CIs include zero; McNemar b=40, c=32, p=0.41.
- **Verdict:** **Hypothesis not supported on CAL** — no paired CI excludes zero, so no
  directional claim; few-shot buys no measurable CAL quality at 2× cost. Per UPGRADE_PLAN
  §4.2 the C1 definition stays **few-shot** for the TEST-IID/POSTCUTOFF finals (plan is not
  edited by execution sessions), and the ablation stands as the §6.2 reference pair. Noted
  for Phase 4: zero-shot Haiku is a statistically indistinguishable half-cost point the
  router frontier should include. Owner may optionally switch the TEST finals to zero-shot
  (halves the remaining Haiku budget ≈$26.6→$13.5) — that is a plan deviation requiring
  sign-off; default remains few-shot. Incidental: Haiku CAL macro-F1 (both arms) already
  sits at/above Tier A's CAL 0.7534, consistent CIs pending the TEST finals.
- **Repro:** arms:
  `uv run --extra tierc python -m triage_lab.harness configs/tier_c_haiku_ablation_fewshot_cal.yaml`
  then same with `..._zeroshot_...`; paired comparison:
  `uv run python -m triage_lab.tier_c_compare results/tier_c_raw/tier_c_haiku_ablation_fewshot_cal/20260806T160011Z/calls.jsonl results/tier_c_raw/tier_c_haiku_ablation_zeroshot_cal/20260806T160510Z/calls.jsonl --split cal`

## 2026-08-07 — Phase 3 step 4: Haiku 4.5 zero-shot FINALS on TEST-IID + TEST-POSTCUTOFF

- **Configs:** `configs/tier_c_haiku_zeroshot_{test_iid,test_postcutoff}.yaml` — first and
  only Tier C touch of TEST-*; zero-shot per the §4.2 amendment (owner-approved 2026-08-07,
  commit `9812499`); the two configs differ only in `model.name`/`data.split`; n=5,000
  seeded subsample each (`cap_seed: 20260806`, kept identical to the ablation so the Sonnet 5
  subsample pairs on the same rows); prompt bundle `f6777a96…`, k=0.
- **Hypothesis:** the frozen zero-shot Haiku config holds ≈0.76 macro-F1 (its CAL level) on
  TEST-IID; the TEST-POSTCUTOFF delta (contamination defense, §6.3.5) is reported whatever
  it shows.
- **Result:** (0 parse failures / 10,000 calls; computed cost == OpenRouter-reported both
  runs; providers per call: IID Bedrock 4,984 + Anthropic 16, POSTCUTOFF Bedrock 4,998 +
  Anthropic 2)
  - TEST-IID (`run_id 70a1b0c4…`): macro-F1 **0.7697** [0.7499, 0.7886], accuracy **0.8474**
    [0.8368, 0.8570]; **$6.5751** = **$1.315/1k complaints**; p50 1.38 s / p95 2.52 s;
    6,307,301 prompt + 53,557 completion tokens.
  - TEST-POSTCUTOFF (`run_id 82af4e01…`): macro-F1 **0.7254** [0.7122, 0.7396], accuracy
    **0.7374** [0.7246, 0.7494]; **$6.8521** = **$1.370/1k**; p50 1.43 s / p95 2.36 s.
  - **Contamination/drift delta (POSTCUTOFF − IID, different rows → not paired; per-slice
    CIs disjoint on both metrics):** macro-F1 **−0.0443**, accuracy **−0.1100**. This
    conflates post-training-cutoff contamination defense with genuine 2026 distribution
    drift; Phase 5's yearly TEST-DRIFT slices will decompose it. Reported as measured.
  - vs Tier A on TEST-IID (run `8e4d6345`, macro-F1 0.7605 [0.7564, 0.7643]): Haiku's point
    is +0.009 with overlapping CIs and different eval rows (full slice vs 5k subsample) —
    **no cross-tier claim**; the Phase 4 frontier will handle cross-tier comparisons properly.
  - ECE/Brier/AURC for Tier C remain degenerate one-hot artifacts (see step 2 note), logged
    but not quality claims.
- **Verdict:** **Both Haiku final points established and CI'd.** IID held slightly above the
  CAL level (0.770 vs 0.757). The −4.4pt macro-F1 / −11.0pt accuracy POSTCUTOFF drop is the
  honest headline of the contamination defense. Task spend $13.43 (projected ~$13.5 for
  zero-shot); cumulative Tier C spend $19.41 of the approved ≈$48.5 envelope (zero-shot
  switch leaves ample room for Sonnet 5, the remaining Phase 3 task).
- **Repro:**
  `uv run --extra tierc python -m triage_lab.harness configs/tier_c_haiku_zeroshot_test_iid.yaml`
  then same with `..._test_postcutoff.yaml` (live API; receipts under `results/tier_c_raw/`).

## 2026-08-07 — Phase 3 step 5: Sonnet 5 zero-shot paired subsample on TEST-IID + TEST-POSTCUTOFF

- **Configs:** `configs/tier_c_sonnet_zeroshot_{test_iid,test_postcutoff}.yaml` — mirror the
  Haiku final configs with exactly three deltas each (`model.name`, `model.slug:
  anthropic/claude-sonnet-5`, `eval_rows_cap: 1500`); `cap_seed: 20260806` unchanged, so each
  slice's 1,500 rows are a strict subset of the corresponding Haiku 5,000 (verified on the
  receipt id sets: 1,500/1,500 overlap on both slices) — the Haiku-vs-Sonnet delta can be
  paired on these rows. Zero-shot (k=0) per the §4.2 amendment; frozen v1 bundle `f6777a96…`;
  n=1,500 × 2 slices is the owner-approved sizing (2026-08-06 gate).
- **Hypothesis:** Sonnet 5 zero-shot beats Haiku 4.5 on macro-F1 on both slices, at ≈2–3× the
  $/1k (OpenRouter list $2/MTok prompt + $10/MTok completion vs Haiku's $1 + $5).
- **Result:** (computed cost == OpenRouter-reported both runs; providers per call: IID
  Bedrock 1,495 + Azure 5, POSTCUTOFF Bedrock 1,493 + Azure 7)
  - TEST-IID (`run_id e1503146…`): macro-F1 **0.7418** [0.7015, 0.7730], accuracy **0.8413**
    [0.8220, 0.8593]; **$5.4881** = **$3.659/1k complaints**; p50 3.17 s / p95 5.92 s;
    2,635,652 prompt + 21,677 completion tokens; **12 parse failures (0.8%)**.
  - TEST-POSTCUTOFF (`run_id d1c42d7d…`): macro-F1 **0.7876** [0.7672, 0.8085], accuracy
    **0.8013** [0.7813, 0.8220]; **$5.7795** = **$3.853/1k**; p50 3.12 s / p95 5.27 s;
    2,779,838 prompt + 21,979 completion tokens; **37 parse failures (2.5%)**.
  - **Parse failures:** all `finish_reason: "length"` — Sonnet 5 hit the shared
    `max_tokens: 64` completion cap with mostly empty visible content (consistent with
    internal reasoning-token burn; Haiku had 0/10,000). Per the frozen protocol these resolve
    to the fallback label and are kept as measured: config parity with the Haiku finals takes
    precedence, and any params/prompt change is a new version, not an in-place fix.
  - **Cross-slice observation (different rows → NOT paired; n=1,500 CIs are wide):** Sonnet's
    POSTCUTOFF−IID is macro-F1 **+0.0459** / accuracy **−0.0400** — macro-F1 moves in the
    opposite direction to Haiku's −0.0443, while accuracy drops like Haiku's (−0.1100). The
    2.5%-vs-0.8% parse-failure asymmetry confounds the accuracy side. No drift claim from
    this; Phase 5 decomposes it.
  - vs Haiku same-slice points: on IID Sonnet's 0.7418 sits *below* Haiku's 0.7697 with
    overlapping CIs (different n); on POSTCUTOFF Sonnet's 0.7876 sits above Haiku's 0.7254
    with disjoint per-slice CIs. The honest comparison is the paired same-rows delta — next
    task. Note for it: `triage_lab.tier_c_compare` fails loud on unequal id sets, so the
    Haiku receipts must first be filtered to the shared 1,500 ids (or the tool gains an
    explicit intersection mode).
  - ECE/Brier/AURC remain degenerate one-hot artifacts (see step 2 note), logged only.
- **Verdict:** **Both Sonnet 5 points established and CI'd; pairing verified.** The headline
  surprise: no visible Sonnet-over-Haiku gain on TEST-IID at 2.8× the $/1k — judgment
  reserved for the paired delta. Task spend $11.27 (projected ≈$8 from list prices — actual
  Sonnet prompt-token counts ran ~40% higher per call than Haiku's for identical prompts);
  cumulative Tier C spend $30.68 of the approved ≈$48.5 envelope.
- **Repro:**
  `uv run --extra tierc python -m triage_lab.harness configs/tier_c_sonnet_zeroshot_test_iid.yaml`
  then same with `..._test_postcutoff.yaml` (live API; receipts under `results/tier_c_raw/`).

## 2026-08-07 — Phase 3 step 6 (acceptance): paired Haiku-vs-Sonnet deltas + Phase 3 close-out

- **Method:** `triage_lab.tier_c_compare` (frozen read-only tool; no API calls, no new runs)
  on the shared 1,500 ids per slice. The tool requires identical id sets, so the Haiku
  receipts are first filtered to the Sonnet id set (copies under a temp dir; committed
  receipts untouched). **A = Sonnet, B = Haiku**, so every delta below reads Sonnet − Haiku
  (the tool's `arm_a_role`/`arm_b_role` strings still say "few_shot"/"zero_shot" — that is a
  CAL-ablation naming convention; the receipt paths in each report identify the real arms).
  Fallback-labeled calls enter exactly as the runner scored them (frozen protocol).
- **Hypothesis:** the paired same-rows delta confirms the step 5 impression — no Sonnet gain
  on TEST-IID, a real Sonnet gain on TEST-POSTCUTOFF.
- **Result — PRIMARY (frozen protocol, all 1,500 shared rows per slice):**
  - TEST-IID: accuracy **−0.0007** [−0.0167, +0.0140], macro-F1 **−0.0073** [−0.0427,
    +0.0263]; McNemar b=68 / c=69 (137 discordant), **p=1.00**. Sonnet and Haiku are
    statistically indistinguishable on IID at **2.8× the $/1k** (Sonnet $3.66 vs Haiku
    $1.32) and ~2.3× the p50 latency (3.17 s vs 1.38 s).
  - TEST-POSTCUTOFF: accuracy **+0.0553** [+0.0393, +0.0734], macro-F1 **+0.0458**
    [+0.0282, +0.0663]; McNemar b=128 / c=45 (173 discordant), **p=2.0e-10**. Sonnet is
    decisively better on post-cutoff data; the capability gap only appears off the models'
    (shared) training distribution.
- **Result — SENSITIVITY (excludes Sonnet's fallback-labeled rows; NOT the headline — the
  frozen-protocol numbers above stay primary):** IID n=1,488: accuracy +0.0007 [−0.0148,
  +0.0155], macro-F1 −0.0066 [−0.0419, +0.0291], p=1.00 — conclusion unchanged. POSTCUTOFF
  n=1,463: accuracy +0.0622 [+0.0458, +0.0793], macro-F1 +0.0499 [+0.0318, +0.0692],
  p=2.0e-13 — slightly larger, same conclusion. **Neither slice's verdict is driven by the
  parse-failure handling.**
- **Fallback asymmetry (substantive finding, not a bug):** Haiku produced valid structured
  output on **10,000/10,000** final calls; Sonnet failed to answer **12/1,500 (0.8%)** on IID
  and **37/1,500 (2.5%)** on POSTCUTOFF — every failure `finish_reason: "length"`, i.e. the
  64-token completion budget (an ops-style latency/cost constraint both models share) was
  consumed by Sonnet's internal reasoning before any parseable JSON was emitted, and the
  frozen protocol scored the call as the fallback label. Under a production output-token
  budget, **the stronger model silently fails to answer up to 2.5% of the time while the
  cheaper model never does** — and the failure rate rose on the drifted slice, exactly where
  escalation to the stronger model is most valuable. Router implication for Phase 4: a
  structured-output parse failure is itself a signal (abstain/escalate/retry-with-bigger-
  budget), and completion budget is a per-model config dimension, not a shared constant.
- **Contamination delta recap (per-model, cross-slice, different rows → not paired):** Haiku
  macro-F1 −0.0443 / accuracy −0.1100 (step 4, n=5,000/slice); Sonnet macro-F1 +0.0459 /
  accuracy −0.0400 (step 5, n=1,500/slice, confounded by the 0.8%→2.5% fallback shift). The
  paired same-rows view above is the clean statement: the Sonnet−Haiku gap moves from ≈0 on
  IID to ≈+5 pts on POSTCUTOFF.
- **Phase 3 ✅ Accept checklist:** both Tier C points with CIs on both slices ✓ (runs
  `70a1b0c4…`, `82af4e01…`, `e1503146…`, `d1c42d7d…`); measured $/1k ✓ (Haiku $1.32–1.37,
  Sonnet $3.66–3.85) and p50/p95 latency ✓ per model; Haiku-vs-Sonnet paired delta ✓ (this
  entry); contamination delta ✓ (steps 4–5 + recap above); raw API logs retained ✓
  (committed under `results/tier_c_raw/`, commits `f67cf6a`, `67af2d2`).
- **Verdict:** **Phase 3 acceptance satisfied — phase complete.** Headline: zero-shot Sonnet 5
  buys nothing over Haiku 4.5 on TEST-IID at 2.8× the cost, but is worth ≈+5.5 pts accuracy
  (paired, p=2e-10) on TEST-POSTCUTOFF — while also silently failing 2.5% of POSTCUTOFF
  calls under the shared 64-token output budget. Total Tier C spend $30.68 of the approved
  ≈$48.5 (this step: $0, no API calls).
- **Repro** (read-only; regenerates the four reports from committed receipts):
  ```
  python3 - <<'EOF'
  import json, pathlib
  base = pathlib.Path("results/tier_c_raw"); out = pathlib.Path("/tmp/tierc_cmp"); out.mkdir(exist_ok=True)
  for tag, h, s in (("iid", "tier_c_haiku_zeroshot_test_iid/20260807T004109Z",
                     "tier_c_sonnet_zeroshot_test_iid/20260807T015725Z"),
                    ("pc", "tier_c_haiku_zeroshot_test_postcutoff/20260807T005820Z",
                     "tier_c_sonnet_zeroshot_test_postcutoff/20260807T020907Z")):
      srec = [json.loads(l) for l in open(base / s / "calls.jsonl") if l.strip()]
      shared = {r["complaint_id"] for r in srec}
      ok = {r["complaint_id"] for r in srec if not r.get("parse_failed")}
      hl = [l for l in open(base / h / "calls.jsonl") if l.strip()]
      (out / f"haiku_{tag}_shared.jsonl").write_text("".join(l for l in hl if json.loads(l)["complaint_id"] in shared))
      (out / f"haiku_{tag}_ok.jsonl").write_text("".join(l for l in hl if json.loads(l)["complaint_id"] in ok))
      (out / f"sonnet_{tag}_ok.jsonl").write_text("".join(json.dumps(r) + "\n" for r in srec if not r.get("parse_failed")))
  EOF
  uv run python -m triage_lab.tier_c_compare results/tier_c_raw/tier_c_sonnet_zeroshot_test_iid/20260807T015725Z/calls.jsonl /tmp/tierc_cmp/haiku_iid_shared.jsonl --split test_iid
  uv run python -m triage_lab.tier_c_compare results/tier_c_raw/tier_c_sonnet_zeroshot_test_postcutoff/20260807T020907Z/calls.jsonl /tmp/tierc_cmp/haiku_pc_shared.jsonl --split test_postcutoff
  uv run python -m triage_lab.tier_c_compare /tmp/tierc_cmp/sonnet_iid_ok.jsonl /tmp/tierc_cmp/haiku_iid_ok.jsonl --split test_iid
  uv run python -m triage_lab.tier_c_compare /tmp/tierc_cmp/sonnet_pc_ok.jsonl /tmp/tierc_cmp/haiku_pc_ok.jsonl --split test_postcutoff
  ```

## 2026-08-07 — Phase 3 step 7a: provider-routing audit + latency-exhibit relabel (no API calls)

- **Method:** code + receipts audit, prompted by the owner's observation that recent Tier C
  runs were all served by Amazon Bedrock despite an assumed pin to Anthropic. Read the
  request construction (`src/triage_lab/tier_c.py::_create_completion`) and aggregated the
  per-call `provider` field across every committed Tier C receipt.
- **Result:**
  - **Provider pinning is NOT in effect and never was.** The request body sends only
    `model/messages/temperature/max_tokens/response_format` plus
    `extra_body={"usage": {"include": true}}` — no OpenRouter `provider` routing preference
    (`order`/`only`/`allow_fallbacks`) is sent anywhere, and no config exposes one.
    OpenRouter free-routes every call.
  - Aggregate across all 8 committed Tier C runs (16,050 calls): **Amazon Bedrock 16,020
    (99.81%)**, Anthropic 18 (0.11%, all Haiku finals), Azure 12 (0.07%, all Sonnet finals).
  - **Latency relabel:** all Phase 3 p50/p95 latency figures (steps 2–6) are client-side
    wall-clock through the OpenRouter proxy on this un-pinned, ≈99.8%-Bedrock route at
    `max_concurrency: 8`. They characterize **Claude via OpenRouter→Bedrock**, not the
    Anthropic first-party API; the Haiku-vs-Sonnet latency ratio (~2.3× p50) is contingent
    on OpenRouter's routing. STATUS.md's Phase 3 acceptance row now carries this labeling;
    upstream region is not exposed by OpenRouter, so per UPGRADE_PLAN §7 the latency method
    statement is "provider recorded per call; region unknown".
  - Prior log entries are correct as written (each already reported its provider histogram);
    nothing is retro-edited.
- **Verdict:** **No mislogged data — the gap was labeling, now fixed.** If provider-controlled
  latency is ever wanted, sending an OpenRouter `provider` preference is a params change =
  new config/run version, never an in-place edit; not done here.
- **Repro (read-only):**
  `grep -n "extra_body\|provider" src/triage_lab/tier_c.py` (no routing preference), and
  `python3 -c "import json,glob;h={};[h.update({r['provider']:h.get(r['provider'],0)+1}) for f in glob.glob('results/tier_c_raw/*/*/calls.jsonl') for r in map(json.loads,open(f))];print(h)"`

## 2026-08-07 — Phase 3 step 7b: v2-params PROBE — Sonnet 5 zero-shot, max_tokens 256, POSTCUTOFF only

- **Config:** `configs/tier_c_sonnet_zeroshot_v2params_test_postcutoff.yaml` — identical to
  the v1 POSTCUTOFF config except **`max_tokens: 64 → 256`** (single delta; same frozen v1
  prompt bundle `f6777a96…`, same 1,500 shared rows via `cap_seed: 20260806`). Owner-approved
  2026-08-07 (~$6). **PROBE: the v1 run (`d1c42d7d…`) stays PRIMARY in the results log;
  this run exists only to bound the truncation artifact.**
- **Hypothesis:** step 6 found Sonnet's +5.5-pt paired POSTCUTOFF advantage was earned
  despite 37/1,500 length-truncated calls resolving to fallback labels; lifting the
  completion budget recovers those rows and widens the advantage — i.e. v1 understates
  Sonnet on drifted data.
- **Result** (`run_id 1f2b8f2a…`; computed cost == OpenRouter-reported; providers Bedrock
  1,495 + Azure 5 — un-pinned routing per step 7a):
  - macro-F1 **0.7960** [0.7763, 0.8158], accuracy **0.8087** [0.7886, 0.8280];
    **$5.8055 = $3.870/1k**; p50 3.32 s / p95 5.79 s (OpenRouter→Bedrock route); prompt
    tokens 2,779,838 (byte-identical prompts to v1, same rows), completion 24,582.
  - **Parse failures 37 → 3 (0.2%)**, all three still `finish_reason: "length"` at 256
    tokens (ids 21021115, 21069060, 21714801) — a larger budget shrinks but does not
    eliminate reasoning-token burn.
  - **v2 − v1 paired (same 1,500 rows):** accuracy **+0.0073** [−0.0020, +0.0180], macro-F1
    **+0.0083** [−0.0021, +0.0192]; McNemar b=33 / c=22 (55 discordant), **p=0.18** — the
    paired CI includes zero. The 55 discordant rows exceed the 37 recovered fallback rows:
    temperature-0 answers still vary run-to-run across the un-pinned provider route, so part
    of the movement is route/rerun noise, not budget.
  - **v2 − Haiku paired (same rows, Haiku receipts filtered to the shared 1,500):** accuracy
    **+0.0627** [+0.0460, +0.0800], macro-F1 **+0.0541** [+0.0350, +0.0750]; McNemar
    b=130 / c=36, **p=1.1e-13** — vs the v1 primary +0.0553 / +0.0458 (p=2.0e-10), and
    consistent with the step 6 sensitivity view (+0.0622 excluding fallback rows).
- **Verdict:** **v1 understates Sonnet's POSTCUTOFF advantage by at most ≈1 accuracy point,
  and not statistically significantly (v2−v1 CI includes zero).** The primary +5.5-pt
  paired-advantage claim stands as reported; the probe bounds the 64-token truncation
  artifact rather than revising the headline. Step 6's router implication is refined:
  parse-failure-as-escalation-signal stands, but the completion budget's direct accuracy
  cost on this task is ≤1 pt, and even 256 tokens leaves a 0.2% silent-failure tail.
  Task spend $5.81; cumulative Tier C spend $36.49 of the approved ≈$48.5 envelope.
- **Repro:**
  `uv run --extra tierc python -m triage_lab.harness configs/tier_c_sonnet_zeroshot_v2params_test_postcutoff.yaml`
  (live API); comparisons (read-only):
  `uv run python -m triage_lab.tier_c_compare results/tier_c_raw/tier_c_sonnet_zeroshot_v2params_test_postcutoff/20260807T040553Z/calls.jsonl results/tier_c_raw/tier_c_sonnet_zeroshot_test_postcutoff/20260807T020907Z/calls.jsonl --split test_postcutoff`
  then the same v2 receipts vs the Haiku POSTCUTOFF receipts filtered to the shared 1,500
  ids (filter snippet as in step 6's repro block).

## 2026-08-07 — Phase 4 task 1: per-example prediction artifacts + risk-coverage evidence (all runs)

- **Method:** every eval run now auto-persists a per-example Parquet artifact
  (`data/preds/<run_id>.parquet`, schema `preds-v2`, DuckDB-only I/O) carrying
  `complaint_id / y_true / y_pred / p_max / prob::<label>…` plus bound provenance
  (run_id, config_sha256, split + split_sha256, class_labels, git_sha, dataset
  input_sha256, and prompt_bundle_sha256 for Tier C). A backfill CLI
  (`python -m triage_lab.predictions`) regenerated artifacts for all **12 non-smoke
  historical runs** (Tier A: deterministic offline refit, appends nothing; Tier C:
  offline reconstruction from committed receipts reusing the runner's own subset
  selection and parse/fallback code paths; the 2 smoke runs are `skip-smoke` by
  design). `results/risk_coverage/<run_id>.json` (committed, derived, regenerable —
  not covered by the runs.jsonl append-only rule) holds per-run threshold tables and
  bootstrap-CI'd AURC / acc@coverage summaries.
- **Verification gate:** every artifact must reproduce the run's logged point metrics
  (accuracy, macro-F1, AURC, acc@cov::{0.50,0.80,0.90,0.95}) at 1e-9 **and** pass
  structural checks: id uniqueness/non-null, id membership in the frozen split,
  row-by-row `y_true` agreement with the frozen split (catches wrong-ID row mapping,
  which aggregate metrics cannot), `p_max == max(probs)`, `y_pred == argmax(probs)`
  (Tier A/B) or exact one-hot (Tier C). The gate was hardened after an external Codex
  review (2026-08-07) flagged the original aggregate-only gate as a blocker; the same
  review drove backfill hash validation (recomputed config hash and Tier C prompt
  bundle hash must equal the record's — hard fail, never silently stamp historical
  hashes onto data from different inputs), a duplicate-receipt hard error, and a
  coverage-domain guard (acc@coverage requires 0 < c ≤ 1). All 12 artifacts pass:
  every logged metric reproduces at **abs delta 0.00e+00**.
- **Result** (point estimates; AURC with 95% bootstrap CI, n=1,000, fixed seed):

  | split | config | run | n | AURC [95% CI] | acc@50% | acc@80% | acc@90% | acc@95% |
  |---|---|---|---|---|---|---|---|---|
  | cal | tier_a_cnb_wordchar_cal | c78c9d07 | 86,972 | 0.0740 [0.0722, 0.0759] | 0.9374 | 0.8771 | 0.8471 | 0.8292 |
  | cal | tier_a_logreg_word_cal | d35c23d2 | 86,972 | 0.0664 [0.0645, 0.0681] | 0.9458 | 0.8962 | 0.8662 | 0.8486 |
  | cal | tier_a_logreg_wordchar_cal | abcadd53 | 86,972 | 0.0678 [0.0660, 0.0696] | 0.9447 | 0.8921 | 0.8601 | 0.8420 |
  | cal | tier_c_haiku_ablation_fewshot_cal | c7598f84 | 1,500 | 0.1744 [0.1351, 0.1852] | 0.8320 | 0.8350 | 0.8378 | 0.8407 |
  | cal | tier_c_haiku_ablation_zeroshot_cal | 3f310951 | 1,500 | 0.1734 [0.1390, 0.1915] | 0.8227 | 0.8333 | 0.8341 | 0.8351 |
  | test_iid | tier_a_cnb_test_iid | c20cd14a | 104,443 | 0.0589 [0.0575, 0.0603] | 0.9543 | 0.8991 | 0.8678 | 0.8502 |
  | test_iid | tier_a_logreg_test_iid | 8e4d6345 | 104,443 | 0.0500 [0.0487, 0.0512] | 0.9638 | 0.9147 | 0.8848 | 0.8668 |
  | test_iid | tier_c_haiku_zeroshot_test_iid | 70a1b0c4 | 5,000 | 0.1495 [0.1393, 0.1663] | 0.8548 | 0.8470 | 0.8493 | 0.8488 |
  | test_iid | tier_c_sonnet_zeroshot_test_iid | e1503146 | 1,500 | 0.1619 [0.1333, 0.1860] | 0.8373 | 0.8408 | 0.8452 | 0.8421 |
  | test_postcutoff | tier_c_haiku_zeroshot_test_postcutoff | 82af4e01 | 5,000 | 0.2713 [0.2461, 0.2794] | 0.7232 | 0.7348 | 0.7349 | 0.7373 |
  | test_postcutoff | tier_c_sonnet_zeroshot_test_postcutoff | d1c42d7d | 1,500 | 0.1924 [0.1711, 0.2265] | 0.7933 | 0.8083 | 0.7985 | 0.7986 |
  | test_postcutoff | tier_c_sonnet_zeroshot_v2params_test_postcutoff | 1f2b8f2a | 1,500 | 0.1870 [0.1645, 0.2206] | 0.7973 | 0.8158 | 0.8067 | 0.8077 |

  (n = the run's own eval slice: Tier A full splits; Tier C its frozen 5,000/1,500-row
  subsets. Cross-run AURC comparisons are only like-for-like within a split at equal n.)
- **Findings:**
  - **Tier A confidence genuinely ranks errors.** LogReg TEST-IID walks 0.8668 @ 95%
    coverage → 0.9638 @ 50%; AURC 0.0500 [0.0487, 0.0512]. LogReg beats CNB on AURC
    on both CAL and TEST-IID — consistent with the Phase 1 accuracy ordering.
  - **Tier C self-confidence is degenerate by construction** (structured-output runner
    logs a one-hot; `p_max ≡ 1.0`, threshold table has a single row): acc-vs-coverage
    is flat (Haiku TEST-IID 0.847–0.855 across all four coverages) and AURC reduces to
    a rescaled error rate with no ranking power. **A confidence-cascade router cannot
    use Tier C p_max as an escalation signal** — Tier C escalation must come from other
    signals (e.g. parse-failure, per Phase 3 step 6). Structural property, not a bug;
    logged here so Phase 4 router design starts from it.
  - Drift echo at the confidence level: Haiku AURC 0.1495 (IID) → 0.2713 (POSTCUTOFF).
- **Verdict:** substrate for Phase 4 calibration + router is in place — one frozen,
  self-describing per-example source of truth per run, gate-verified bit-identical to
  the logged metrics, with committed risk-coverage evidence JSONs downstream numbers
  can be traced to.
- **Repro:** `make preds` (add `--force` to regenerate over existing artifacts:
  `uv run python -m triage_lab.predictions --all --force`), then `make risk-coverage`;
  tests: `uv run pytest tests/test_predictions.py tests/test_risk_coverage.py -q`.

## 2026-08-07 — Phase 4 task 2: cost-model implementation (single-tier baseline costs, all runs)

- **Hypothesis:** the UPGRADE_PLAN §4.2 business cost model
  (`cost = c_misroute · P(error) + c_api · E[tokens] + c_human · P(human)`; defaults
  c_misroute $6.00, c_human $2.50, both ESTIMATED; API cost MEASURED) can be implemented
  against the frozen per-example artifacts with Tier C API cost joined per-example from
  committed receipts, reproducing each run's logged `cost_usd` exactly; expectation going
  in: measured API cost visibly separates the tiers on the cost axis.
- **Method:** new `src/triage_lab/cost_model.py` + versioned parameter file
  `configs/cost_model_v1.yaml` (sha256 `f76ad15a8745…`, bound into every output).
  Per-example cost = `c_misroute·1{answered ∧ wrong} + api_cost_usd + c_human·1{human}`,
  where `api_cost_usd` is **incurred spend charged unconditionally** — a policy that
  defers to human before any paid call passes 0.0 for that row, and a cascade that pays
  for an LLM call and then defers still carries that spend. Human-assigned rows are
  assumed correctly resolved (assumption recorded in every output JSON). Expected cost per 1,000 complaints with 95%
  bootstrap CI (frozen constants: n=1,000, seed 20260805, percentile), total + per-component
  from the same resamples. Tier A api_cost = $0 (amortized CPU inference <1 ms/example,
  evidence class ESTIMATED, per §6.1); Tier C api_cost joined from the run's
  `calls.jsonl` receipts on `complaint_id` (`computed_cost_usd` = real tokens × published
  prices, evidence class MEASURED). Hard-fail gates: missing/duplicate receipt, null or
  negative cost, and Σ(joined per-example costs) vs the run record's logged `cost_usd`
  beyond 1e-6 (`cost_sum_check` in each JSON). Outputs
  `results/cost_model/<run_id>.json` — committed, derived, regenerable (same class as
  `results/risk_coverage/`; runs.jsonl untouched). Tier detection comes from the cost
  config's `api_cost` keys: Tier B is deliberately unpriced in v1, so Tier B artifacts
  will fail loud until a v2 config prices GPU inference (never silently $0).
  An external Codex review (2026-08-07, same-day) was run on the diff before commit: it
  confirmed the implementer-flagged cascade seam as a blocker (the first draft zeroed
  api cost on human-deferred rows, per a too-literal reading of §4.2 — fixed to the
  incurred-spend semantics above; numerically inert for these all-answered baselines,
  verified by leaf-level diff: 0 numeric changes across all 12 outputs) and drove
  hardening: per-receipt token/slug/pricing-snapshot recomputation gates (all 16,500
  committed receipts re-derive at abs delta 0.0 vs tol 1e-12), a join-keyed-on-
  complaint_id test (sum- and recompute-invisible permutation failure mode),
  artifact-vs-run-record provenance gate (run_id, config/split hashes,
  prompt_bundle_sha256), finite-parameter config validation, empty `--all` = hard
  failure, and atomic score-all-then-write batch output. Outputs carry
  `schema_version: "cost-v1"`; the shared-index resampling that makes component CI
  bands decompose the total band is a public, contract-tested API
  (`resample_means_per_1k`).
- **Result:** 36 new tests; full suite 226 passed. `--all` scored all 12 non-smoke
  artifacts; all 7 Tier C runs pass `cost_sum_check` (max abs delta 6.2e-15 vs tol 1e-6).
  Policy `single_tier_all_answered`, cost config v1:

  | config | split | n | acc | total $/1k [95% CI] | misroute/1k | api/1k |
  |---|---|---|---|---|---|---|
  | tier_a_cnb_wordchar_cal | cal | 86,972 | 0.8063 | 1162.24 [1146.23, 1177.98] | 1162.24 | 0.00 |
  | tier_a_logreg_word_cal | cal | 86,972 | 0.8264 | 1041.78 [1026.88, 1056.48] | 1041.78 | 0.00 |
  | tier_a_logreg_wordchar_cal | cal | 86,972 | 0.8193 | 1084.14 [1068.68, 1099.12] | 1084.14 | 0.00 |
  | tier_c_haiku_ablation_fewshot_cal | cal | 1,500 | 0.8413 | 954.61 [850.61, 1070.62] | 952.00 | 2.61 |
  | tier_c_haiku_ablation_zeroshot_cal | cal | 1,500 | 0.8360 | 985.31 [877.31, 1097.42] | 984.00 | 1.31 |
  | tier_a_cnb_test_iid | test_iid | 104,443 | 0.8279 | 1032.39 [1019.75, 1046.19] | 1032.39 | 0.00 |
  | tier_a_logreg_test_iid | test_iid | 104,443 | 0.8444 | 933.41 [920.94, 946.45] | 933.41 | 0.00 |
  | tier_c_haiku_zeroshot_test_iid | test_iid | 5,000 | 0.8474 | 916.92 [859.32, 980.51] | 915.60 | 1.32 |
  | tier_c_sonnet_zeroshot_test_iid | test_iid | 1,500 | 0.8413 | 955.66 [847.66, 1071.63] | 952.00 | 3.66 |
  | tier_c_haiku_zeroshot_test_postcutoff | test_postcutoff | 5,000 | 0.7374 | 1576.97 [1504.94, 1653.80] | 1575.60 | 1.37 |
  | tier_c_sonnet_zeroshot_test_postcutoff | test_postcutoff | 1,500 | 0.8013 | 1195.85 [1071.72, 1315.85] | 1192.00 | 3.85 |
  | tier_c_sonnet_zeroshot_v2params_test_postcutoff | test_postcutoff | 1,500 | 0.8087 | 1151.87 [1035.88, 1272.03] | 1148.00 | 3.87 |

  (human/1k = $0 by construction for these all-answered baselines; the component is still
  bootstrapped so router rows will be schema-comparable. TEST-* rows are derived analyses
  of already-final frozen runs, not new evaluations; cross-run cost comparisons are
  like-for-like only within a split at equal n — the Tier A vs Tier C TEST-IID rows differ
  in n and are NOT paired.)
- **Findings:**
  - **HYPOTHESIS PARTLY WRONG (the interesting part):** at the v1 defaults, measured API
    cost is $1.31–$3.87 per 1k — 0.1–0.4% of total — so misroute dominates by ~3 orders
    of magnitude and `total/1k ≈ 6000 × (1 − acc)`. The single-tier cost ranking is
    effectively the accuracy ranking wearing a dollar sign: all-Haiku on TEST-IID
    ($916.92/1k, n=5,000) comes out cheaper than all-LogReg ($933.41/1k, n=104,443) at
    these defaults (unpaired; CIs overlap). Any "router beats every tier on cost" claim
    at c_misroute=$6.00 is an accuracy claim in disguise; the real economic tension is in
    the human-queue arm (c_human $2.50 vs c_misroute $6.00) and in cost-parameter
    sensitivity — the frontier chapter must expose that sensitivity rather than assert
    one operating point.
  - **Cascade accounting seam (found, fixed):** a literal per-term reading of §4.2 would
    drop already-paid API spend on human-deferred rows and undercount every
    paid-call-then-defer route. Flagged by the implementer, confirmed blocker by the
    Codex review, fixed as incurred-spend semantics (above) before first commit; the
    router simulator (task 4) must pass accumulated per-stage spend per row.
- **Verdict:** cost-model machinery in place and receipt-verified; §4.2 implemented
  verbatim with parameters versioned + hashed for the demo's user-adjustable sliders.
  The misroute-dominance finding reframes the router's headline: the frontier story at
  default parameters is driven by accuracy and the human-queue trade-off, not raw API
  spend — carry into threshold optimization (task 3) and the frontier claims.
- **Repro:** `make cost-model` (= `uv run python -m triage_lab.cost_model --all`);
  tests: `uv run pytest tests/test_cost_model.py -q`.
