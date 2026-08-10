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

## 2026-08-07 — Phase 4 task 3: threshold optimization on CAL (owner-amended §4.2 cascade)

- **Owner amendment (recorded, 2026-08-07, blockquote added to UPGRADE_PLAN §4.2):** the
  cascade carries confidence thresholds at **Tier A (and Tier B when it lands) only**;
  **Tier C is an unconditional terminal stop** once escalated (no τ_C, no Tier C
  self-confidence); **parse-failure is the only Tier C→human signal**. Rationale: the
  lab's own evidence (Phase 4 task 1) shows the frontier tier emits no usable confidence
  (structured-output p_max is a degenerate one-hot); its one reliable self-signal is
  failure to answer, so the router encodes exactly that. **Considered-and-deferred
  alternative (owner-directed, NOT implemented):** the §4.2 3-sample self-consistency
  proxy on the escalated subset — it would require temperature>0 (a new Tier C protocol
  version) and new API spend for a signal of questionable signal-to-noise given the 55
  discordant rows observed between temperature-0 reruns of identical prompts (step 7b).
- **Hypothesis:** a Tier A confidence gate lowers expected cost vs answer-everything on
  CAL at cost-model v1 defaults, and the escalate-to-C arm beats the send-to-human arm
  at those defaults.
- **Method:** `src/triage_lab/threshold_opt.py` sweeps τ_A over the CAL artifact's
  distinct p_max values (answer-nothing point included in the argmin space) for two
  policy families: `a_to_human` (full CAL, n=86,972) and `a_to_c_parsefail_human`
  (paired 1,500-row subset of the Tier C zero-shot CAL run `3f310951…`; escalated rows
  charged their MEASURED per-receipt cost; parse-failed rows → human with incurred spend
  still charged, overriding the artifact's fallback label), plus `a_only` / `c_only` /
  `all_human` references through the same machinery. Objective = expected cost/1k under
  `configs/cost_model_v1.yaml` (`f76ad15a…`). Full bootstrap CIs (frozen constants,
  shared indices) at operating points; **paired deltas** (τ* vs references;
  cross-family on identical rows) per the no-claim-without-paired-CI rule; 6×6
  c_misroute × c_human sensitivity grid (defaults an exact cell; evidence class:
  estimated parameters, measured predictions). PRIMARY Tier A rung =
  `tier_a_logreg_wordchar_cal` (`abcadd53…`) — byte-matched in family/params/features/
  seed to the TEST-IID final `tier_a_logreg_test_iid`; all three CAL rungs swept.
  **p_max-space note:** CAL rungs are `calibration: none`, the TEST final is isotonic —
  every operating point is recorded both as τ* (in the CAL artifact's own p_max space)
  and as target coverage; the τ→TEST transfer rule is a task-4 decision. Pre-commit
  Codex review (no blockers) drove: exact-float64 τ serialization (the 10-dp-rounded τ
  mis-replayed 6/9 files by one boundary row — now 0/9, regression-tested against every
  shipped operating point), the full `predictions.verify_artifact` gate on every loaded
  artifact, receipts sha256 bound into provenance (thresholds + cost_model outputs), and
  an atomicity proof test for batch publish.
- **Result** (PRIMARY rung, v1 defaults; `results/thresholds/`, summary
  `summary__cost-f76ad15a.json`; 35 module tests, full suite 261 passed):

  | policy | data | point | τ* | cov_A | esc% | human% | cost/1k [95% CI] |
  |---|---|---|---|---|---|---|---|
  | a_to_human | full CAL | τ* | 0.558766 | 0.8689 | 13.11 | 13.11 | 998.16 [985.08, 1010.75] |
  | a_to_human | full CAL | a_only | — | 1.0000 | 0 | 0 | 1084.14 [1068.68, 1099.12] |
  | a_to_human | full CAL | all_human | — | 0 | 100 | 100 | 2500.00 [2500.00, 2500.00] |
  | a_to_c_parsefail_human | paired 1,500 | τ* | 0.702020 | 0.7793 | 22.07 | 0.00 | 896.29 [788.30, 1012.29] |
  | a_to_c_parsefail_human | paired 1,500 | a_only | — | 1.0000 | 0 | 0 | 1048.00 [935.90, 1168.00] |
  | a_to_c_parsefail_human | paired 1,500 | c_only | — | 0 | 100 | 0 | 985.31 [877.31, 1097.42] |

  Paired deltas (negative favors τ*): a_to_human τ*−a_only **−85.98 [−93.90, −79.27]**
  (full CAL); a_to_c τ*−a_only **−151.71 [−231.70, −71.70]**; a_to_c τ*−c_only
  **−89.02 [−165.14, −9.02]** — all exclude zero. Cross-checks: a_only full-CAL and
  c_only reproduce the task-2 numbers exactly from an independent code path.
- **Findings:**
  - **A Tier A confidence gate pays for itself everywhere:** both thresholded families
    beat `a_only` in every cell of the 6×6 grid (32× price range), and at defaults the
    paired CIs exclude zero.
  - **Cross-family claim is directional only:** a_to_c@τ* − a_to_human@τ* on identical
    rows = **−64.04 [−129.06, +4.65]** — CI includes zero at n=1,500. The sensitivity
    grid favors escalate-to-C in every cell at defaults, with the family boundary at
    c_human/c_misroute ≈ 0.344 (v1 defaults sit at 0.4167, C side); but per the
    no-claim rule this is not yet a claim. Task 4 needs the larger paired TEST slice or
    must report it as directional.
  - **a_to_c's τ* is price-insensitive:** coverage 0.7793 in all 36 cells (stable even
    at c_misroute $240) — API spend is too small to move the boundary; the gate sits
    where Tier C's accuracy overtakes Tier A's confidence-conditional accuracy.
    a_to_human's τ* swings coverage 0.076→1.000 across the same grid.
  - **Parse-failure arm is empty on Haiku:** 0/1,500 parse failures in the CAL
    zero-shot receipts (and 0 in every Haiku run); the Tier C→human path is exercised
    only synthetically in tests here. It does fire on Sonnet (12/37/3 on
    IID/POSTCUTOFF/v2params) — a Haiku-terminal cascade's c_human sensitivity is
    vacuous, a Sonnet-terminal one is not. Carry to task 4.
  - **Pre-existing discrepancy (flagged, not fixed):** `tier_a_logreg_test_iid.yaml`'s
    header comment calls wordchar "the winning CAL rung", but on CAL it loses to
    word-only on all three logged metrics (acc 0.8193 vs 0.8264, macro-F1 0.7466 vs
    0.7535, AURC 0.0664 word vs 0.0678 wordchar), and word-only's τ* is cheaper here
    ($970.35 vs $998.16/1k). Feature-match to the shipping TEST final still makes
    wordchar PRIMARY for router work; the comment's claim is unsupported by
    results/runs.jsonl. Owner decision needed if Tier A model choice is ever revisited;
    TEST finals stay frozen.
  - Selection caveat: τ* is chosen on the same CAL data it is scored on (by design —
    CAL is the threshold-fitting split); unbiased evaluation happens on TEST in task 4.
- **Verdict:** hypothesis confirmed with paired CIs for the gate-vs-a_only claims;
  escalate-to-C superiority at defaults is a point-estimate/sensitivity finding, not
  yet a CI-backed claim. Threshold machinery, sensitivity exhibits, and operating
  points are committed and replayable (exact-τ regression suite).
- **Repro:** `make thresholds` (= `uv run python -m triage_lab.threshold_opt --all`);
  tests: `uv run pytest tests/test_threshold_opt.py -q`.

## 2026-08-08 — Phase 4 task 4: router simulator on TEST-IID (all available policies)

- **Owner decisions (2026-08-07, binding for this task):** (1) cross-family A→C vs
  A→human re-evaluated on the full n=5,000 Haiku TEST-IID paired rows, offline; if the
  CI still includes zero → directional-only, no additional samples purchased. (2)
  Headline router = **Haiku-terminal** cascade; its empty parse-failure→human arm is a
  reported robustness fact; c_human sensitivity covered by the A→human rung;
  Sonnet-terminal deferred to the drift chapter. (3) Frozen Tier A TEST final stays;
  the erroneous "winning CAL rung" header comment in `tier_a_logreg_test_iid.yaml` is
  fixed as a documentation correction (see registry below). (4) τ→TEST transfer:
  **threshold transfer primary** (CAL-fit τ* deployed as a fixed constant, realized
  TEST coverage reported — production semantics); coverage-matched transfer secondary
  sensitivity only.
- **Config documentation correction (registry):** the comment fix changes the file
  hash, so it is registered explicitly in `predictions.CONFIG_DOC_CORRECTIONS`
  (immutable MappingProxyType): run `8e4d6345…`, recorded `b22be1e96376…` → corrected
  `0813065e47c7…`, announced on stdout on every use; any other run/hash mismatch
  hard-fails exactly as before. Prose-only nature proven two ways: parsed-object
  equality of old-vs-new YAML, and comment-stripping invariance. Force-regeneration of
  the affected run reproduces all 7 logged metrics at abs delta 0.00e+00.
  **Footnote (owner decision 3):** on CAL, word-only LogReg beat word+char on all
  three logged metrics (acc 0.8264 vs 0.8193, macro-F1 0.7535 vs 0.7466, AURC 0.0664
  vs 0.0678) and its CAL-fit gate was cheaper ($970.35 vs $998.16/1k). The frozen
  TEST final remains word+char (its documented rationale is now feature-match to the
  frozen final + Phase 5 robustness expectation, not "winning CAL rung"); the model
  choice is NOT reopened.
- **Hypothesis:** deploying the CAL-fit τ* constants on TEST-IID yields a router that
  dominates ≥2 single-tier policies on cost (phase-accept criterion), with the
  Haiku-terminal cascade as the headline.
- **Method:** `src/triage_lab/router_sim.py` consumes only frozen artifacts (no new
  API calls): Tier A TEST-IID artifacts (`8e4d6345` LogReg isotonic, `c20cd14a` CNB),
  the Haiku TEST-IID artifact + receipts (`70a1b0c4`, 5,000 rows), CAL-fit operating
  points loaded from `results/thresholds/` primary-rung files (validated: bound to the
  CAL run record via runs.jsonl, finite/in-range τ and coverage, count consistency,
  duplicate files hard-fail), cost model v1. Policies — full TEST-IID (n=104,443):
  a_only, a_only_cnb, a_to_human@τ*, all_human; paired 5,000 subset: those plus
  c_only and a_to_c_parsefail_human@τ* (measured per-receipt costs, incurred-spend
  semantics; 0/5,000 Haiku parse failures → human arm empty, as expected). System
  metrics count human rows as correct (assumption documented in outputs);
  answered-only metrics reported alongside. Paired deltas (cost + accuracy, shared
  frozen resample indices) and McNemar on shared machine rows. Two pre-commit Codex
  reviews drove: the threshold-artifact validation gate above; a McNemar exact-tail
  rewrite (old code hit OverflowError on ~6,000-digit binomial tails — no full-TEST
  McNemar was computable at all — and after the first fix still silently underflowed
  to a fabricated p=0.0 for n>1074; now exponent-bound-routed with boundary
  regressions at n∈{1073..1076} and 196-case parity vs the old formula); the
  coverage-matched "nearest achievable" contract (matching error 6.67e-05 vs quantum
  2e-04 on real data); and the immutable correction registry.
- **Result** (threshold transfer PRIMARY; `results/router_sim/`; 45 module tests,
  full suite 313 passed):

  Full TEST-IID (n=104,443):

  | policy | cov_machine | human% | acc_mach | acc_sys | mF1_ans | mF1_sys | cost/1k [95% CI] |
  |---|---|---|---|---|---|---|---|
  | a_only (LogReg) | 1.0000 | 0 | 0.8444 | 0.8444 | 0.7605 | 0.7605 | 933.41 [920.94, 946.45] |
  | a_only_cnb | 1.0000 | 0 | 0.8279 | 0.8279 | 0.7265 | 0.7265 | 1032.39 [1019.75, 1046.19] |
  | **a_to_human@τ*** | 0.8206 | 17.94 | 0.9089 | 0.9253 | 0.8405 | 0.8916 | **896.89 [886.69, 906.94]** |
  | all_human | 0 | 100 | — | 1.0000 | — | 1.0000 | 2500.00 |

  Paired subset (n=5,000):

  | policy | cov_A | human% | acc_mach | mF1_ans | cost/1k [95% CI] |
  |---|---|---|---|---|---|
  | a_only | 1.0000 | 0 | 0.8448 | 0.7652 | 931.20 [870.00, 992.40] |
  | c_only (Haiku) | — | 0 | 0.8474 | 0.7697 | 916.92 [859.32, 980.51] |
  | a_to_human@τ* | 0.7786 | 22.14 | 0.9253 | 0.8603 | 902.70 [857.58, 949.00] |
  | **a_to_c@τ*** (headline) | 0.6224 | 0 | 0.8526 | 0.7749 | **884.89 [827.27, 946.09]** |

  Paired deltas (✓ = CI excludes zero): full a_to_human−a_only **−36.52
  [−44.51, −28.91] ✓**; full a_to_human−a_only_cnb **−135.50 [−145.74, −125.16] ✓**;
  paired a_to_c−c_only **−32.02 [−53.64, −10.43] ✓** (McNemar b=52/c=26, p=0.0043);
  paired a_to_c−a_only −46.31 [−95.53, **+6.54**] · (McNemar p=0.078); paired
  **a_to_c−a_to_human −17.81 [−63.02, +27.20] · → owner decision 1 verdict:
  DIRECTIONAL ONLY**, under both transfer modes.
- **Findings:**
  - **Phase-accept groundwork, stated honestly:** the criterion "router dominates ≥2
    single-tier policies" is met by the **confidence-gated A→human router on full
    TEST-IID** (beats a_only and a_only_cnb, both paired CIs excluding zero) — not by
    the LLM cascade specifically. The Haiku-terminal headline dominates **one** model
    baseline (c_only). `all_human` is deliberately excluded from dominance counting
    (at c_human=$2.50 any machine policy beats $2,500/1k arithmetically — counting it
    would inflate the claim).
  - **Threshold transfer loses 5–16 points of realized coverage** (a_to_human full:
    0.8689 target → 0.8206 realized; a_to_c: 0.7793 → 0.6224): CAL rungs emit raw
    p_max, the TEST final emits isotonic-compressed p_max, so the same constant
    answers fewer rows. This is not cosmetic — under the coverage-matched SECONDARY,
    a_to_c would have dominated a_only too (**−83.71 [−129.34, −36.92] ✓**) and
    a_to_human−a_only on the subset flips significant (−67.10 ✓). The intended
    operating point dominates; the raw-space constant shipped into isotonic space
    does not. Calibration-space alignment for threshold fitting is an open item for
    the frontier-claims task / owner.
  - a_to_c's machine accuracy 0.8526 > both a_only 0.8448 and c_only 0.8474 at 100%
    machine coverage — the cascade routes hard rows to the slightly-better model and
    keeps easy rows on the free one; but at n=5,000 its cost edge over a_only is not
    yet a claim (CI +6.54 upper bound).
  - McNemar on a_to_c−a_to_human's 3,893 shared machine rows favors a_to_human
    (p=0.0024) — expected: those are the easy rows both answer; the policies differ
    on what happens to the hard 22%.
- **Verdict:** router simulator complete and gate-hardened; hypothesis **partially
  confirmed** — a confidence-gated router beats ≥2 single tiers with paired CIs, but
  the specific Haiku-terminal headline's a_only edge is directional pending either the
  calibration-space fix or Tier B. Honest diagnosis (the lab's credibility rule) is
  the deliverable: the shortfall is attributable to the raw→isotonic threshold
  transfer, measured and quantified above.
- **Repro:** `make router-sim` (= `uv run python -m triage_lab.router_sim --all`);
  tests: `uv run pytest tests/test_router_sim.py -q`.

## 2026-08-08 — Phase 4 task 5: calibrated-threshold v2 (isocal) + frontier claims (partial)

- **Owner decision (2026-08-08):** calibration-space alignment approved as a **v2
  threshold derivation** — τ fit on isotonic-calibrated CAL confidences (the same
  calibration the deployment artifact applies), CAL-only fitting, single TEST
  evaluation, versioned alongside v1. **v2 is the primary operating point for the
  frontier exhibits; v1 is retained untouched** as the case-study
  calibration-space-mismatch lesson (diagnosis, coverage shortfall, fix). Then the
  frontier-claims task in partial form: Tier A/C claims now, every Tier B point
  `pending_tier_b`.
- **v2 derivation run:** `configs/tier_a_logreg_wordchar_isocal_cal.yaml`
  (sha `6485a765…`), parsed-object delta vs the v1 rung exactly
  {`calibration: none→isotonic`, `model.name` (identifier)}; same
  `CalibratedClassifierCV(FrozenEstimator, isotonic)` path as the TEST final, applied
  to CAL. Run `40513354…` appended to runs.jsonl (the task's single append); artifact
  gate ✓ 7/7 metrics at abs delta 0. CAL metrics: acc 0.8341 (was 0.8193), macro-F1
  0.7611, AURC 0.0574 (was 0.0678) — but **ECE 0.0225 → 0.1024, 4.5× worse**: per-class
  isotonic + renormalization sharpens ranking while degrading max-prob ECE.
  Stated plainly: calibration-space *alignment* is not better *calibration*; the point
  is that τ now lives in the deployed model's confidence space. In-sample caveat
  (calibrator fit on CAL, applied to CAL) documented — CAL is the calibration split by
  design.
- **Result 1 — alignment closed the transfer gap ~10×** (threshold transfer, target →
  realized TEST coverage): a_to_human full 0.9006 → 0.9053 (**+0.0047**; v1 was
  −0.0483); a_to_c paired 0.9600 → 0.9710 (**+0.0110**; v1 was −0.1569). v1's
  systematically negative gaps were the space mismatch; v2's small positive residuals
  are CAL-vs-TEST distribution difference + calibrator in-sample optimism.
- **Result 2 — v2 router on TEST (`results/router_sim/*__opv2__*`):** full TEST-IID:
  a_to_human@τ*v2 = cov 0.9053, 9.47% human, **$872.81/1k [861.64, 883.88]**, system
  macro-F1 0.8475; paired deltas −60.60 [−66.79, −54.88] vs a_only ✓ and −159.58 vs
  a_only_cnb ✓. Paired 5,000: a_to_c@τ*v2 = cov_A 0.9710, $908.44 [849.63, 966.04],
  machine acc 0.8486; vs a_only **−22.76 [−41.96, −2.37] ✓**; vs c_only −8.48
  [−61.27, +39.56] ·; **vs a_to_human +47.74 [+22.53, +74.74] ✓ — the v1
  directional-only cross-family question resolves AGAINST the cascade**: with aligned
  thresholds, escalating to the human queue is significantly cheaper than escalating
  to Haiku at v1 cost parameters. Dominance census (model baselines only): a_to_human
  beats 2 (full); a_to_c beats 1 (a_only, paired).
- **Result 3 — the two §4.2 claims (`results/frontier/frontier__opv2__cost-f76ad15a.json`):**
  - **CERTIFIED — CLAIM 2, a_to_human, full TEST-IID (n=104,443):** "At significantly
    LOWER cost than the all-linear policy (paired cost delta −$60.60/1k
    [−66.79, −54.88], a 6.49% [5.90, 7.11] reduction), the a_to_human router raises
    system macro-F1 by **+0.0870 [0.0841, 0.0899]**." Caveat travels with it:
    macro_f1_system credits the 9.47% human-routed rows as correct; answered-only
    macro-F1 0.8110 vs 0.7605 is reported unpaired (different row sets).
  - **NOT ESTABLISHED — CLAIM 1 (a_to_c vs c_only, n=5,000):** accuracy +0.0012
    [−0.0068, +0.0100], pct cost reduction +0.92% [−4.45, +6.35] — directional only.
  - **NOT ESTABLISHED — CLAIM 2 (a_to_c vs a_only, n=5,000):** cost favorable and
    significant (−22.76 ✓, 2.44% [0.26, 4.48]) but macro-F1 +0.0070 [−0.0027, +0.0165]
    spans zero — directional only.
  - CLAIM 1 on full TEST: `not_applicable` (no all-LLM baseline on the full slice —
    Haiku covers a frozen 5,000-row subsample); recorded with reason.
  - `pending_tier_b` placeholders (b1_only, b2_only, a_to_b, a_to_b_to_c) are
    data-shaped status entries with no numeric fields.
- **Statistical-gate fix (Codex review blocker, pre-commit):** the first frontier
  draft passed a claim gate on "point favorable OR CI includes zero" — an invalid
  non-inferiority test (failure to establish inferiority ≠ non-inferiority). Replaced
  with three-way per-metric classification on the favorable CI bound at margin 0
  (favorable_significant / not_established / adverse_significant; degenerate [0,0]
  bands read not_established); certification requires all gated metrics
  favorable_significant; adverse results emit explicitly adverse diagnosis strings.
  Numeric values everywhere invariant under the fix; CLAIM 1's accuracy gate
  downgraded from "pass" to not_established as required. Further hardening from the
  same review: τ* replay-on-load against the verified CAL artifact (tampered-file
  tests), git_sha + input_sha256 added to the provenance gate and serialized into all
  router/frontier input blocks (all 13 artifacts verified consistent first; the two
  committed v1 router JSONs gained exactly 4 provenance string lines — documented
  carve-out, zero numeric change), risk_coverage CLI routed through the verified
  loader (13/13 byte-identical regeneration), y_true equality in pair alignment.
- **Verdict:** Phase 4 acceptance in its owner-approved partial form is **met**: two
  §4.2 headline claims computed with paired CIs — one CERTIFIED (a_to_human), two
  honest not-established diagnoses for the LLM cascade at n=5,000 — and the router
  dominates ≥2 single-tier policies (a_to_human vs a_only + a_only_cnb, paired CIs
  excluding zero). The v1→v2 pair is the case-study exhibit: a measured
  calibration-space-mismatch lesson with its measured fix. Tier B backfill (B1/B2
  points, A→B, A→B→C) remains pending GPU.
- **Repro:** `uv run python -m triage_lab.harness configs/tier_a_logreg_wordchar_isocal_cal.yaml`
  (offline refit; run already logged), then `make frontier` (regenerates thresholds
  v1+v2, router-sim v1+v2, frontier v2); tests:
  `uv run pytest tests/test_frontier.py tests/test_router_sim.py tests/test_threshold_opt.py -q`.

## 2026-08-08 — Phase 5 task 1: Tier A rolling yearly evals (TEST-DRIFT 2023–2026H1)

- **Context:** Tier B backfill deferred (incoming bundle was the outgoing kit by
  mistake; real checkpoints still on the A6000). Per STATUS §b order, Phase 5 drift
  proceeds for available tiers — Tier A this task, Tier C (Haiku headline +
  Sonnet-terminal as the drift-chapter variant, per Phase 4 task 4 owner decision 2)
  in following sessions.
- **Hypothesis:** the frozen Tier A model (TF-IDF word+char LogReg, fit TRAIN,
  isotonic-calibrated on CAL; TEST-IID reference run `8e4d6345…`, macro-F1 0.7605
  [0.7564, 0.7643]) degrades monotonically across the frozen yearly TEST-DRIFT
  slices, with the 2026-H1 slice worst given its known class-mix inversion
  (credit_reporting 585 rows vs ~10k in 2023–25; splits_stats.yaml).
- **Configs:** four clones of `tier_a_logreg_test_iid.yaml`
  (`tier_a_logreg_test_drift_{2023,2024,2025,2026h1}.yaml`); diff-verified delta per
  config = {`data.split`, `model.name` (identifier), comments} — single-variable
  discipline (split only). One harness run each; four appends to runs.jsonl (git diff
  confirms 4 insertions, 0 modifications).
- **Result (evidence class: measured, frozen protocol; 95% bootstrap CI n=1,000):**

  | split | run | macro-F1 | accuracy | ECE | AURC |
  |---|---|---|---|---|---|
  | test_iid (2022-H2, ref) | `8e4d6345…` | 0.7605 [0.7564, 0.7643] | 0.8444 | 0.1059 | 0.0500 |
  | test_drift_2023 | `389d3e69…` | 0.7579 [0.7496, 0.7663] | 0.8338 | 0.1039 | 0.0617 |
  | test_drift_2024 | `c1c810e2…` | 0.7478 [0.7376, 0.7569] | 0.8304 | 0.0970 | 0.0675 |
  | test_drift_2025 | `cef3a58c…` | 0.7295 [0.7205, 0.7386] | 0.8108 | 0.0855 | 0.0757 |
  | test_drift_2026h1 | `2ae5d8ea…` | 0.6656 [0.6585, 0.6729] | 0.6918 | 0.0422 | 0.1888 |

- **Findings:** (1) monotone macro-F1 decay 2023→2025 (−0.028 over two years; 2025 CI
  [0.7205, 0.7386] excludes the 2023 point), then a cliff at 2026-H1 (−0.064 vs
  2025). (2) The cliff is localized: per-class F1 for credit_reporting collapses
  0.887 (2025) → 0.215 (2026-H1) while mortgage/card/debt_collection hold or
  improve (0.882/0.777/0.725) — consistent with prior shift (TRAIN-era
  credit_reporting prior ≫ 2026-H1 share ⇒ precision collapse on the shrunken
  class), to be decomposed in the prior-shift task before claiming within-class
  drift. (3) Selective-risk degrades faster than raw accuracy (AURC 0.050 → 0.189,
  3.8×) while max-prob ECE *improves* (0.1059 → 0.0422) — confidence stays
  usable for ranking-threshold routing but the risk-coverage curve moves
  substantially; escalation-rate-over-time computation must use these per-year
  prediction artifacts (`extra.predictions_path` present in all four records).
- **Verdict:** hypothesis supported; degradation measured and logged. The 2026-H1
  credit_reporting collapse is a finding, not a bug — it is the drift chapter's
  motivating exhibit and feeds tasks 2 (prior-shift decomposition) and the
  escalation-rate chart.
- **Repro:** `uv run python -m triage_lab.harness configs/tier_a_logreg_test_drift_2023.yaml`
  (and `_2024`, `_2025`, `_2026h1`); ~29 min each on M-series CPU (full TRAIN refit
  per run, deterministic seed 20260805).

## 2026-08-09 — Phase 5 task 2a: Tier C drift smoke (20 calls/slice/model) + cost projection

- **Owner decision (2026-08-09):** Tier C envelope raised ≈$48.5 → ≈$75. Approved
  scope: Haiku headline on all four drift slices at n=1,500/slice + Sonnet-terminal
  drift variant paired on the same 1,500 rows/slice (frozen zero-shot prompt v1
  `f6777a96…`, canonical params — Sonnet v1 max_tokens 64, NOT the v2params probe;
  cap_seed 20260806 everywhere ⇒ identical row sets per slice). Gate: per-slice
  smoke (20 calls) with projected total, STOP for owner sign-off before full runs.
- **Method:** 16 configs created — 8 full-run
  (`tier_c_{haiku,sonnet}_zeroshot_test_drift_{2023,2024,2025,2026h1}.yaml`; Haiku
  delta vs frozen TEST-IID final = {split, eval_rows_cap 5000→1500 owner-approved
  pairing sizing, name}; Sonnet delta = {split, name} only) and 8 smoke clones
  (eval_rows_cap: 20 = seeded first-20 subset of each slice, owner-authorized TEST
  touch). Smokes run with `--no-append` (no smoke record enters runs.jsonl);
  receipts under `results/tier_c_raw/tier_c_*_smoke_zeroshot_test_drift_*/`. Both
  models smoked per slice deliberately — Phase 3 step 5 showed Haiku→Sonnet
  extrapolation under-projects ~40% (Sonnet prompt tokenization runs heavier).
- **Smoke results (measured; all 8×20 calls, provider Amazon Bedrock 160/160;
  computed cost = OpenRouter-reported cost on every receipt):** parse failures
  0/140 except sonnet 2024: 1/20 (consistent with the Phase 4 finding that the
  parse-failure→human arm fires on Sonnet, not Haiku). Mean $/call — Haiku:
  0.001430 / 0.001339 / 0.001395 / 0.001264 (2023/24/25/26h1); Sonnet: 0.003960 /
  0.003775 / 0.003930 / 0.003534. Smoke spend **$0.41**.
- **Projection (mean $/call × 1,500, evidence class: measured-smoke projection):**

  | slice | Haiku n=1500 | Sonnet n=1500 |
  |---|---|---|
  | 2023 | $2.15 | $5.94 |
  | 2024 | $2.01 | $5.66 |
  | 2025 | $2.09 | $5.90 |
  | 2026h1 | $1.90 | $5.30 |
  | **subtotal** | **$8.14** | **$22.80** |

  **Projected full-run total ≈ $30.94** (+10% retry contingency ⇒ ceiling ≈ $34.0).
  Cumulative Tier C spend: $36.49 prior + $0.41 smoke = **$36.90 of ≈$75**;
  if full runs land on projection, cumulative ≈ **$67.84** (contingency ceiling
  ≈ $70.9) — inside the raised envelope with ≈$4–7 headroom.
- **Verdict: smoke accepted; full runs NOT started.** Awaiting owner sign-off on
  the projection per the 2026-08-09 gate. On approval, the 8 full runs append to
  runs.jsonl (single append each) and Phase 5 task 2 (Tier C yearly rows +
  paired Haiku-vs-Sonnet deltas per slice) proceeds.
- **Repro (smoke):** `uv run --extra tierc python -m triage_lab.harness
  configs/tier_c_haiku_smoke_zeroshot_test_drift_2023.yaml --no-append` (and the
  other 7); receipts are the audit artifact.

## 2026-08-09 — Phase 5 task 2b: Tier C rolling yearly evals (owner-approved full runs)

- **Gate cleared:** owner approved the task-2a projection (2026-08-09); the eight
  staged configs ran unchanged — Haiku + Sonnet zero-shot (prompt v1 `f6777a96…`,
  canonical params) on each drift slice, n=1,500, cap_seed 20260806. Eight appends
  to runs.jsonl (git diff: 8 insertions, 0 modifications). Receipt-verified pairing:
  Haiku∩Sonnet complaint_id overlap **1500/1500 on all four slices**.
- **Result (measured, frozen protocol; macro-F1 [95% CI]):**

  | slice | Haiku (run) | Sonnet (run) | Tier A (task 1, full slice) |
  |---|---|---|---|
  | 2023 | 0.7357 [0.6994, 0.7692] (`cf190ce2…`) | 0.7544 [0.7188, 0.7863] (`c82988bd…`) | 0.7579 |
  | 2024 | 0.7617 [0.7274, 0.7918] (`b83b2d0d…`) | 0.7795 [0.7462, 0.8075] (`9d54ec58…`) | 0.7478 |
  | 2025 | 0.7639 [0.7296, 0.7903] (`446fa236…`) | 0.7634 [0.7273, 0.7906] (`daa58725…`) | 0.7295 |
  | 2026h1 | 0.7278 [0.7048, 0.7508] (`aa2bc9a0…`) | 0.7821 [0.7587, 0.8046] (`55cffcbe…`) | 0.6656 |

  (Tier A column is the full-20k-slice runs from task 1 — context, not a paired
  comparison; cross-tier paired deltas on the common 1,500 rows are a follow-on
  computation from the prediction artifacts.)
- **Paired Sonnet−Haiku deltas per slice** (tier_c_compare, paired bootstrap +
  McNemar on the identical 1,500 rows): 2023 ΔF1 +0.019 [−0.008, +0.045] (p=.31),
  2024 +0.018 [−0.006, +0.043] (p=.54), 2025 −0.001 [−0.026, +0.026] (p=.46) — all
  CIs include zero, models statistically tied in-distribution. **2026h1: ΔF1
  +0.054 [+0.036, +0.073], Δacc +0.074 [+0.056, +0.091], McNemar p≈1e-15,
  discordants 155:44** — the Sonnet advantage concentrates exactly where drift is
  worst, echoing the Phase 3 POSTCUTOFF paired finding.
- **Findings:** (1) Tier C degrades far slower than Tier A across the drift years —
  at 2026-H1, Tier A 0.666 vs Haiku 0.728 vs Sonnet 0.782; Sonnet's 2026-H1 point
  is statistically indistinguishable from its own 2023 point (CIs overlap
  broadly), i.e. near-flat over the horizon. (2) Parse failures: Haiku 0/6,000;
  Sonnet 19/10/16/26 per slice (1.3–1.7%) — stable Sonnet-only failure arm, rate
  roughly year-invariant. (3) Haiku ECE tracks 1−accuracy (degenerate one-hot
  self-confidence, known from Phase 4 task 1) — Tier C yearly ECE is not
  independently meaningful. (4) Providers recorded per receipt: Amazon Bedrock
  for all Sonnet calls; Haiku 2025/2026h1 partially served by Anthropic (7 and
  24 calls).
- **Cost (measured from receipts):** task spend **$30.48** (projected $30.94;
  −1.5% error). Per-run $1.97–2.04 Haiku, $5.50–5.71 Sonnet. Cumulative Tier C
  spend **$67.38 of the ≈$75 envelope** (smoke included).
- **Verdict:** hypothesis half-confirmed, half-overturned in an interesting way —
  Tier C does not show Tier A's monotone decay; the LLM tier is drift-robust
  in-distribution-tied but pulls decisively ahead at the 2026-H1 shift, and the
  spread by model (Haiku dips at 2026-H1, Sonnet doesn't) makes the
  Sonnet-terminal drift variant the natural escalation story for the router
  chapter.
- **Repro:** `uv run --extra tierc python -m triage_lab.harness
  configs/tier_c_haiku_zeroshot_test_drift_2023.yaml` (and the other 7 configs);
  paired deltas: `uv run --extra tierc python -m triage_lab.tier_c_compare
  results/tier_c_raw/tier_c_sonnet_zeroshot_test_drift_2023/<ts>/calls.jsonl
  results/tier_c_raw/tier_c_haiku_zeroshot_test_drift_2023/<ts>/calls.jsonl
  --split test_drift_2023` (and per-slice analogues).

## 2026-08-09 — Phase 5 task 3: prior-shift decomposition (reweighted-F1 counterfactual, Tiers A + C)

- **Context:** UPGRADE_PLAN §6.3.2 — "show how much yearly degradation is explained
  by class-mix change alone (reweighted-F1 counterfactual) vs within-class drift."
  Post-hoc analysis of the frozen per-example prediction artifacts from tasks 1/2b
  (12 yearly runs, `data/preds/<run_id>.parquet`); zero new model calls, $0 spend.
  Methodology designed by two independent passes (deep-reasoner + Codex) and
  synthesized; both converged on the four-cell counterfactual with exact per-replicate
  additivity.
- **Hypothesis:** Tier A's 2026-H1 cliff (task 1) is substantially explained by the
  class-mix change alone (credit_reporting 47.2% → 2.9% of the slice), with a
  prior-shift component whose CI excludes zero; Tier C's flat curves reflect
  within-class stability (within-class component ≈ 0 or negative), i.e. the LLMs
  survive drifted data because their per-class behavior does not degrade, not because
  they are exempt from the mix change.
- **Method (new module `src/triage_lab/prior_shift.py`; 43 unit tests):** every
  macro-F1 is an exact function of the class prior π and the row-normalized confusion
  C (per-class recall = C[k,k] is invariant to true-class reweighting; all prior
  sensitivity flows through precision). Four cells: A = F1(π_2023, C_2023),
  B = F1(π_Y, C_2023), C = F1(π_2023, C_Y), D = F1(π_Y, C_Y); total Δ = A − D.
  **Primary = Path P (prior-first, reference-anchored):** prior = A − B,
  within = B − D. Always-stored labeled sensitivities: Path Q, Shapley average,
  ANOVA main effects + interaction I = B + C − A − D, and the two-path prior bracket
  (mandatory — the paths differ by up to 3.8 F1 points, so no single path is ever
  quoted alone). Bootstrap: n=1,000, seed 20260805, both slices resampled
  independently per replicate (ref-then-year draw order), components sum exactly to
  the total on every replicate (asserted ≤1e-12); `ref_fixed` variant stored as
  sensitivity. Guards: identity gates (cells A and D reproduce the logged
  `macro_f1`/`accuracy`/`balanced_accuracy` of runs.jsonl at ≤1e-12, hard fail);
  Kish n_eff + max/min weight diagnostics instead of weight caps; share = prior/total
  emitted only when the total CI excludes 0 AND |Δ| ≥ 0.02 (else null + reason);
  empty-class-replicate counter with ci_valid rule. Complements: exact linear
  accuracy decomposition (validator), balanced-accuracy delta (prior-free anchor),
  exact additive per-class contributions, one-at-a-time credit_reporting prior
  counterfactual (labeled non-additive). Tier C: own-subsample π primary (preserves
  the D == logged-macro-F1 traceability identity), full-slice π stored as
  lower-variance sensitivity with π-deviation diagnostics. Extra scope:
  `tier_a__<year>__paired_subsample` restricts Tier A to the exact 1,500 Tier C rows
  so the cross-tier claim is row-paired.
- **Result (evidence class: measured, frozen artifacts, post-hoc decomposition;
  sign convention: positive = degradation vs 2023; 95% bootstrap CI n=1,000):**

  | tier | year | total Δ | prior (Path P) | within (Path P) | interaction | share_prior |
  |---|---|---|---|---|---|---|
  | tier_a | 2024 | +0.0101 [−0.0027, +0.0232] | +0.0075 [+0.0052, +0.0098] | +0.0026 | −0.0017 | suppressed (total ∋ 0) |
  | tier_a | 2025 | +0.0284 [+0.0157, +0.0406] | +0.0099 [+0.0074, +0.0125] | +0.0185 | −0.0075 | 0.348 |
  | tier_a | **2026h1** | **+0.0924 [+0.0802, +0.1042]** | **+0.0420 [+0.0385, +0.0454]** | **+0.0503 [+0.0385, +0.0625]** | **−0.0382** | **0.455 [0.395, 0.524]** |
  | tier_c_haiku | 2026h1 | +0.0078 [−0.0349, +0.0487] | +0.0305 [+0.0145, +0.0438] | −0.0227 | −0.0119 | suppressed |
  | tier_c_sonnet | 2026h1 | −0.0277 [−0.0710, +0.0124] | +0.0009 [−0.0177, +0.0167] | −0.0287 | −0.0179 | suppressed |
  | tier_a (paired 1.5k rows) | 2026h1 | +0.0690 [+0.0236, +0.1087] | +0.0356 [+0.0199, +0.0498] | +0.0333 | −0.0374 | 0.517 (do not quote; CI [0.24, 1.55]) |

  (Tier C 2024/2025 rows: all total CIs ∋ 0, shares suppressed; Haiku/Sonnet 2025
  prior terms +0.0162/+0.0157 marginally exclude 0. Full grid incl. Path Q, Shapley,
  ANOVA, ref_fixed, and full-slice-π sensitivities: `results/prior_shift/`.)
- **Findings:** (1) **Tier A 2026-H1: 4.2 of the 9.2-point macro-F1 drop
  (CI [3.9, 4.5]) is the class-mix change alone**, holding 2023 per-class behavior
  fixed; 5.0 points is within-class drift. The effects are sub-additive by 3.8
  points because both hit the same class — the two decomposition paths bracket the
  prior contribution at [0.4, 4.2] points, so the interaction is part of the claim,
  not a footnote. (2) Per-class attribution: **credit_reporting alone contributes
  7.6 of the 9.2 points** (prior +6.55, within +1.06; its F1 0.900 → 0.310 under
  the mix change alone → 0.215 realized). One-at-a-time counterfactual: moving only
  credit_reporting's share 47.2% → 2.9% (others renormalized, 2023 behavior fixed)
  costs 4.0 points [3.7, 4.3] — 44% of the drop (non-additive diagnostic).
  (3) **Prior-free anchor (balanced accuracy, no counterfactual needed):** Tier A
  −7.2 points [−8.4, −5.9] 2023→2026-H1 vs Haiku +0.8 and Sonnet +2.7 — Tier A's
  per-class behavior genuinely collapsed; the LLMs' did not. (4) **The
  metric-structural prior penalty reaches the LLMs too** — Haiku's 2026-H1 prior
  component +3.0 points [+1.4, +4.4] excludes zero — but their within-class terms
  are ≈0/negative (Haiku −2.3, Sonnet −2.9), which is precisely what saves them.
  Case-study sentence: *the free model dies on drifted data because prior collapse
  and within-class collapse compound on its dominant class; the LLMs pay the same
  prior penalty but their within-class behavior holds, netting ≈0 degradation.*
  (5) Cross-tier on identical rows (paired_subsample): Tier A still degrades +6.9
  points [+2.4, +10.9] on the exact 1,500 rows where Haiku is +0.8 ∋ 0 — the
  comparison is not a sampling artifact. Caveats: paired scope is only sound for
  2026-H1 (small-Δ years flip sign at n=1,500); quote share ratios from native
  Tier A rows only; accuracy decomposition (exact) corroborates: prior +9.1 /
  within +9.8 / interaction −4.7 accuracy points. (6) Amendments recorded:
  `model_id` resolves model slug/runner (per-run `model.name` embeds the year);
  interaction sign identity is I = (C−D) − (A−B) (spec prose initially inverted;
  implementation and tests pin the correct sign).
- **Verdict:** hypothesis supported and quantified. The decomposition licenses the
  drift chapter's central causal-shaped claim with CIs on every component, and the
  suppression gates prevent the two dishonest numbers this analysis could have
  produced (a single-path prior share, and a share ratio on a ≈0 denominator).
- **Repro:** `make prior-shift` (= `uv run python -m triage_lab.prior_shift --all`);
  deterministic (seed 20260805), ~6 s, byte-identical output modulo `generated_at`;
  requires `make preds` artifacts + `data/splits/splits_stats.yaml`. Tests:
  `uv run pytest tests/test_prior_shift.py -q` (43 passed; full suite 399 passed,
  1 skipped).

## 2026-08-09 — Phase 5 task 4: OOV / covariate tracking vs the frozen TRAIN vocabulary

- **Context:** UPGRADE_PLAN §6.3.3 — "yearly OOV rate against the TRAIN vocabulary and
  embedding-centroid distance — the grown-up version of the CoNLL dev/test OOV finding
  in the seed." Model-free covariate-shift exhibit: no prediction is read, no run
  record is opened, nothing appended to `results/runs.jsonl`. Refits the tier_a
  FeatureUnion on the frozen TRAIN split from scratch (vocabularies are not persisted
  anywhere), then measures every eval slice against it. Zero API spend.
- **Hypothesis:** if Tier A's 2026-H1 macro-F1 cliff (task 1: −9.2 points) were driven
  by *covariate* shift, the yearly OOV rate and the TRAIN-centroid distance should rise
  monotonically and the 2026-H1 jump should be the largest — a lexical explanation
  competing with task 3's prior-shift explanation.
- **Method (new module `src/triage_lab/oov.py`; 33 unit tests):** two OOV definitions,
  both computed with the *fitted vectorizer's own* `build_analyzer()` (unigram level =
  a `clone` with `ngram_range=(1,1)`, so lowercasing/token_pattern/preprocessing are
  byte-for-byte the model's; a test pins that the (1,2) analyzer's output is the (1,1)
  output followed by the bigrams). **model_vocab** (primary) = occurrences whose
  unigram is not a key of the fitted word `TfidfVectorizer.vocabulary_`, i.e. after
  `min_df=5` / `max_features=150000` pruning — nonzero on TRAIN itself, which is why
  TRAIN is emitted as a slice and is the baseline the drift numbers are read against.
  **corpus_novelty** = occurrences whose unigram never appears anywhere in TRAIN
  (unpruned; 0 on TRAIN by construction). Token-level is the headline; type-level is
  reported as **point estimates only** (a document bootstrap replicate holds ~63.2% of
  the distinct documents, so an interval around a distinct-type count describes the
  bootstrap's combinatorics, not the estimand — stated in `methods_notes`). Secondary:
  the (1,2)-gram-level model_vocab rate. **`tfidf_centroid_cosine_distance`** =
  `1 − cos(µ_slice, µ_TRAIN)` in the frozen word+char_wb FeatureUnion space, rows
  renormalized to unit L2 after the hstack (sklearn L2-normalizes each block
  separately, so raw rows have norm √2 and an empty-word-block document would be
  down-weighted); cosine clipped to [−1,1] so the TRAIN self-distance is an exact 0
  rather than −2.2e−16. Bootstrap: n=1,000, seed 20260805, resampling **documents**,
  one weight vector per replicate driving every statistic of the slice; the TRAIN side
  (vocabulary, idf, centroid, unpruned type set) is held **fixed** — it is a frozen
  model artifact, not a sample, and cannot be resampled without refitting the
  vectorizer and changing the estimand. Weights are materialized as a `uint16` count
  matrix because every statistic is a ratio of linear functionals of the rows, so the
  whole centroid bootstrap collapses to one `Xᵀ C` sparse×dense product accumulated
  over fixed document chunks. Guards: split-parquet sha256 gate on every slice
  (CLAUDE.md #2), and a **feature-block parity gate** asserting the four yearly Tier A
  drift configs plus the CAL rung declare byte-identical `features` blocks — that
  assertion is what makes "the TRAIN vocabulary" a single well-defined object.
- **Result (evidence class: measured; 95% bootstrap percentile CI, n=1,000):**

  | slice | n_docs | n_tokens | model-vocab OOV (token) | corpus-novelty OOV (token) | TF-IDF centroid cos-dist |
  |---|---|---|---|---|---|
  | train (reference) | 300,000 | 55,547,546 | 0.545% [0.540, 0.550] | 0.000% (structural) | 0.000000 [0.000020, 0.000023] |
  | test_iid (2022-H2) | 104,443 | 18,871,498 | 0.561% [0.551, 0.570] | 0.129% [0.126, 0.133] | 0.053578 [0.052716, 0.054579] |
  | test_drift_2023 | 20,000 | 3,706,384 | 0.762% [0.738, 0.789] | 0.140% [0.131, 0.150] | 0.031173 [0.030226, 0.032882] |
  | test_drift_2024 | 20,000 | 3,794,437 | 0.695% [0.673, 0.717] | 0.136% [0.128, 0.145] | 0.052102 [0.050811, 0.053950] |
  | test_drift_2025 | 20,000 | 3,817,701 | 0.713% [0.689, 0.739] | 0.183% [0.172, 0.193] | 0.092611 [0.090854, 0.095114] |
  | test_drift_2026h1 | 20,000 | 4,377,819 | 0.773% [0.751, 0.796] | 0.192% [0.181, 0.203] | 0.081474 [0.080100, 0.083270] |

  Type-level (point only) and the (1,2)-gram companion, same slice order:
  model-vocab TYPE OOV 86.9 / 73.5 / 50.5 / 49.6 / 51.5 / 53.4%; corpus-novelty TYPE
  OOV 0.0 / 29.5 / 14.2 / 15.3 / 20.4 / 20.2%; (1,2)-gram token OOV 7.56 / 7.43 /
  8.04 / 8.20 / 9.47 / 11.06%. Vocabulary: **13,471 of 102,636 distinct TRAIN unigram
  types survive pruning** (the other 136,529 of the 150,000 word features are bigrams).
- **Findings:** (1) **The OOV hypothesis is refuted, and that is the finding.**
  Model-vocab OOV rises only 0.545% → 0.773% of token mass TRAIN → 2026-H1 (+0.23 pp),
  and true novelty only 0.00% → 0.19%. **99.2% of 2026-H1 token occurrences are still
  in the vocabulary Tier A was fitted on.** A 0.2-pp shift in representable token mass
  cannot produce a 9.2-point macro-F1 collapse; task 3's prior shift (4.2 pts) +
  within-class drift (5.0 pts) remains the explanation. (2) **Neither covariate signal
  tracks the performance cliff.** Model-vocab OOV is not even monotone in time
  (2023 0.762% > 2025 0.713% > 2024 0.695%), and the centroid distance *peaks at 2025*
  (0.0926) and then *falls* at 2026-H1 (0.0815, CIs disjoint) — the exact year the
  macro-F1 cliff happens. Text-space distance to TRAIN and Tier A's failure are
  decoupled. (3) **The types-vs-tokens gap is the grown-up CoNLL finding.** 86.9% of
  TRAIN's own unigram *types* are pruned out of the model vocabulary, yet those types
  carry only 0.545% of TRAIN's token *occurrences*; on 2026-H1, 53.4% of types are OOV
  against 0.77% of tokens. Reporting type-level OOV alone would have overstated the
  problem ~70×. (4) The (1,2)-gram rate is 14× the unigram rate (11.06% at 2026-H1,
  and already 7.56% on TRAIN itself) — n-gram sparsity, not drift, dominates it; it is
  kept as a labelled secondary for exactly that reason. (5) Caveat recorded in every
  output: the centroid distance is an *aggregate* covariate statistic absorbing class-mix
  change and within-class lexical change together (each eval slice is a natural-mix
  Hamilton-apportioned sample of its period while TRAIN is class_year-stratified), which
  is also why test_iid (0.0536) sits *further* from TRAIN than test_drift_2023 (0.0312).
  Only the four drift slices are like-for-like with each other.
- **Deviations / limits:** the centroid lives in TF-IDF space, not a dense-encoder
  space (named `tfidf_centroid_cosine_distance` and flagged `dense_encoder_note`;
  a sentence-encoder version is pending Tier B). No size-matched null was measured:
  the TRAIN interval is the noise floor at n=300,000 and scales as 1/√n, so at the
  20,000-row slice size it is ≈3.9× larger — still two-plus orders of magnitude below
  every measured drift distance, so no conclusion turns on it (`centroid.ci_note`).
- **Verdict:** hypothesis refuted, cleanly and with CIs. This is the negative control
  the drift chapter needs: it rules out lexical drift as the cause of Tier A's collapse
  and leaves the prior-shift decomposition as the standing explanation.
- **Repro:** `make oov` (= `uv run python -m triage_lab.oov --all`); deterministic
  (seed 20260805), ~10.5 min wall-clock, ~7.7 GB peak RSS (a 300k-document word+char
  TF-IDF fit plus one `Xᵀ C` pass per slice), byte-identical output modulo
  `generated_at` (verified by two consecutive runs). Requires only
  `data/splits/*.parquet` + `data/splits/splits_stats.yaml` — no `make preds`
  artifacts. Outputs: `results/oov/{train,test_iid,test_drift_*}.json` + `summary.json`.
  Tests: `uv run pytest tests/test_oov.py -q` (33 passed; full suite 432 passed,
  1 skipped).

## 2026-08-09 — Phase 5 task 5: perturbation robustness on TEST-IID (Tier A full grid; Tier C pending cost approval)

- **Context:** UPGRADE_PLAN §6.3.4 — "typo/OCR-noise/case-mangling perturbations at fixed
  rates on TEST-IID; report per-tier deltas. (Char-n-gram TF-IDF vs subword vs LLM is a
  genuinely interesting comparison here.)" This session runs the **Tier A** grid only
  ($0 API spend). The Tier C arm needs owner cost approval (projection below, per the
  session budget gate); Tier B rows pending GPU. Splits frozen; perturbation is applied
  to the eval slice's `narrative` only — TRAIN fitting and CAL calibration stay clean.
- **Hypothesis:** the Phase 1 rung-2 rationale claimed the `char_wb` 3–5-gram block buys
  typo/OOV robustness. Prediction: word+char Tier A degrades under character noise but
  less than a word-only variant; case-mangling is a structural no-op for Tier A
  (both TF-IDF blocks set `lowercase=True`).
- **Method (new modules `src/triage_lab/perturb.py` + `perturb_report.py`; 54 unit tests):**
  three families — **typo** (QWERTY-adjacency ops over all non-whitespace chars, frozen op
  mix sub .40 / del .20 / transpose .20 / insert .20, adjacency derived from 4-row layout
  geometry), **ocr** (fixed ASCII confusion table l↔1, O↔0, S↔5, B↔8, e↔c, rn↔m, cl↔d;
  greedy longest-match non-overlapping site scan computed *before* any random draw, so the
  site set is a pure function of the text), **case** (flip alphabetic case). `rate` =
  independent per-**site** perturbation probability over the family's eligible set — a
  protocol constant, not a realized changed-char fraction; the three families are therefore
  NOT comparable at equal nominal rate (documented in the module). Determinism: per-document
  RNG keyed `blake2b(f"{seed}:{family}:{rate}:{complaint_id}")` (the splits.py convention),
  frozen seed 20260805 — output is independent of row order and batch size, and the 0.05 /
  0.10 arms are independent draws (rate in the key ⇒ no dose-response claim across rates).
  Harness carries an optional `data.perturbation` block, validates it pre-runner, and
  **hard-fails any run whose runner does not echo the applied block back** — a tier that
  silently ignored the perturbation cannot emit a clean record under a perturbed config
  hash. Report joins per-example artifacts (`data/preds/<run_id>.parquet`, provenance-gated
  via `cost_model.load_artifact_verified`) perturbed-vs-clean on `complaint_id` and reuses
  `harness.paired_bootstrap_delta` (n=1,000, seed 20260805). Clean baselines: existing
  TEST-IID finals `8e4d6345` (logreg word+char) / `c20cd14a` (cnb) + new word-only clean
  run `8dacc1b9`.
- **Result (evidence class: measured; n=104,443; paired 95% bootstrap CIs; delta = perturbed − clean macro-F1):**

  | model | family | rate 0.05 | rate 0.10 |
  |---|---|---|---|
  | logreg word+char (clean 0.7605) | typo | −0.0168 [−0.0194, −0.0142] ✓ | −0.0436 [−0.0469, −0.0403] ✓ |
  | logreg word+char | ocr | −0.0056 [−0.0077, −0.0038] ✓ | −0.0095 [−0.0116, −0.0072] ✓ |
  | logreg word+char | case | +0.0000 [0, 0] (structural) | +0.0000 [0, 0] (structural) |
  | cnb word+char (clean 0.7265) | typo | −0.0128 [−0.0156, −0.0101] ✓ | −0.0434 [−0.0468, −0.0401] ✓ |
  | cnb word+char | ocr | −0.0050 [−0.0068, −0.0033] ✓ | −0.0103 [−0.0127, −0.0078] ✓ |
  | cnb word+char | case | +0.0000 [0, 0] (structural) | +0.0000 [0, 0] (structural) |
  | logreg word-only, sensitivity arm (clean 0.7676) | typo | — | −0.0661 [−0.0697, −0.0626] ✓ |
  | logreg word-only | ocr | — | −0.0167 [−0.0191, −0.0145] ✓ |
  | logreg word-only | case | — | +0.0000 [0, 0] (structural) |

  ✓ = paired CI excludes zero. Accuracy deltas in `results/perturbation/summary.json`.
- **Findings:** (1) **Typo is the damaging family** for Tier A: −4.4 macro-F1 points at
  rate 0.10 for both models; OCR noise costs only ~1 point at the same nominal rate.
  (2) **Case-mangling is an exact structural zero for Tier A** — `lowercase=True`
  annihilates it before featurization; the delta is bit-exact 0.0 with CI [0, 0]
  (pinned by an end-to-end unit test *and* measured here as the plumbing control). This
  is precisely the family where Tier B/C subword/LLM tokenizers CAN be hurt — the
  cross-tier exhibit's most interesting cell is one Tier A cannot occupy.
  (3) **The char-n-gram shield is real (point estimates):** at typo 0.10 the word-only
  model loses −0.0661 vs word+char's −0.0436 — char n-grams absorb ~34% of the typo
  damage (ocr 0.10: −0.0167 vs −0.0095, ~43%). No CI is attached to this
  difference-of-differences by design: two independently-bootstrapped paired deltas do
  not compose into an honest interval, and a joint 4-artifact bootstrap was deliberately
  not built (`methods_notes.char_shield`). Directional-only, same label discipline as the
  Phase 4 cross-family delta. (4) **Honest observation, not a claim:** the word-only
  clean TEST-IID macro-F1 (0.7676) is *higher* than the shipped word+char final (0.7605,
  point estimates; no paired delta computed). The word+char choice was made on the CAL
  ladder (Phase 1) and is kept — but under this grid the word-only model is better clean
  and markedly worse under noise, i.e. the char block trades ~0.7 clean points for typo
  robustness. Flagged for the drift chapter as a labelled sensitivity finding.
- **Deviations / limits:** per-site rate semantics means realized changed-char fractions
  differ across families at equal nominal rate; the realized fraction is computable from
  the frozen perturbation function but not yet emitted — required before any cross-tier
  comparability chart (flagged). The 0.05/0.10 arms are independent draws — no
  dose-response claim. `case` rows are labelled structural-zero in every report row.
- **Tier C projection (NO API calls made — awaiting owner approval per budget gate;
  remaining OpenRouter envelope ≈ $7.6):** measured Haiku 4.5 zero-shot TEST-IID cost is
  $0.001315/call (Phase 3 step 4, run `70a1b0c4`). Mirroring the full Tier A grid
  (3 families × 2 rates × 5,000 rows) = 30,000 calls ≈ **$39.5 — over the envelope,
  not proposed.** Proposed subsample options (all rate 0.10 only, on the frozen
  1,500-row paired subset whose clean Haiku predictions already exist as a strict subset
  of the clean 5,000-row run — zero additional clean calls):
  **A (recommended):** all 3 families × 1,500 = 4,500 calls ≈ $5.92 nominal, ≈$6.8 with
  a +15% margin for perturbed-text token inflation (typos fragment subwords). Leaves
  ≥$0.8 headroom. **B:** typo + case only × 1,500 = 3,000 calls ≈ $3.95 (≤$4.5 with
  margin) — drops OCR, the smallest Tier A effect. **C:** typo only × 5,000 = 5,000
  calls ≈ $6.58 nominal but ≈$7.5 with margin — tightest CI, single family, risks the
  envelope; not recommended. Sonnet excluded at ~2.8× Haiku's per-call cost.
- **Verdict:** hypothesis confirmed with CIs on both legs — Tier A word+char degrades
  under character noise (typo ≫ ocr), the char block demonstrably buys robustness
  (directional), and the case no-op prediction held bit-exactly. Tier A rows of the
  §6.3.4 exhibit are done; Tier C arm awaits cost approval; Tier B pending GPU.
- **Repro:** `make perturb` (16 runs: 1 clean word-only + 15 perturbed; sequential,
  ~8.6 h summed wall-clock, $0) then `make perturb-report`
  (= `uv run python -m triage_lab.perturb_report --all`) →
  `results/perturbation/summary.json` (15 rows, 0 missing). Run ids: logreg word+char
  perturb `4bbd26c4`/`0c9308be` (typo 05/10), `cd8be5b7`/`acb00456` (ocr), `5e5e030c`/
  `96a13428` (case); cnb `6b6c7825`/`61f786be`, `169d5c91`/`84c34754`, `9882794f`/
  `3ddd0a7f`; word-only clean `8dacc1b9`, perturbed `dcfb1bee`/`e94b4e51`/`3f5eff59`.
  Tests: `uv run pytest tests/test_perturb.py -q` (54 passed; full suite 486 passed,
  1 skipped).

## 2026-08-10 — Phase 5 task 5b: perturbation robustness — Tier C arm (owner-approved option A)

- **Context:** owner approved option A of the task-5 projection (2026-08-09): Haiku 4.5
  zero-shot, TEST-IID, typo/ocr/case at rate 0.10 only, on the frozen 1,500-row paired
  subset — ceiling ≈$6.8 incl. +15% token-inflation margin, with a guard: pause between
  families and report if projected total breaches the ≈$7.6 remaining envelope (owner
  offered top-up). Guard evaluated after each family; never triggered. Runs executed
  2026-08-09 UTC.
- **Hypothesis:** subword tokenization degrades more gracefully than word-level TF-IDF
  under character noise (the §6.3.4 cross-tier question), and **case** — structurally
  unmeasurable for Tier A (task 5) — is the family where an LLM *could* pay a real
  penalty, since case-mangling breaks tokenizer merges.
- **Method:** `tier_c.py` now applies `data.perturbation` via `perturb.apply_spec`
  **after** `subsample_eval`, keyed by `complaint_id` — row selection is computed on
  unperturbed data, so the 1,500 rows are byte-identical to the clean run's subset
  (asserted; also field-by-field config parity with `tier_c_sonnet_zeroshot_test_iid.yaml`
  pinned by test). Prompt template/exemplars untouched (zero-shot; bundle
  `f6777a96…` unchanged). `perturb_report.py` gained a `tier_c_haiku` arm: containment
  join of each perturbed run against clean run `70a1b0c4`'s per-example artifact
  (1,500 ⊂ 5,000, n=1500 matched asserted); structural-zero labelling explicitly NOT
  applied to tier_c case rows. 16 new tests (70 total in `tests/test_perturb.py`;
  full suite 502 passed, 1 skipped).
- **Result (evidence class: measured; n=1,500 paired rows; clean Haiku on same rows
  macro-F1 0.7491, accuracy 0.8420; delta = perturbed − clean, paired 95% CI):**

  | family @ 0.10 | perturbed F1 | ΔF1 | Δacc | CI≠0 |
  |---|---|---|---|---|
  | typo | 0.7181 | −0.0310 [−0.0480, −0.0161] | −0.0193 [−0.0293, −0.0093] | yes |
  | ocr | 0.7606 | +0.0115 [−0.0081, +0.0317] | −0.0013 [−0.0100, +0.0060] | no |
  | case | 0.7521 | +0.0031 [−0.0188, +0.0258] | +0.0013 [−0.0073, +0.0093] | no |

  Parse failures 0/4,500; provider 100% Amazon Bedrock; completion tokens pinned
  (~10.7/call, all three runs); latency p50 ≈1.4 s / p95 ≈2.3 s (typo run; OpenRouter→
  Bedrock route caveat as per Phase 3 step 7 audit).
- **Per-family cost, actual vs projection (per owner instruction):** honest clean
  baseline = clean run receipts restricted to the *same 1,500 rows*: **$1.9725**
  (the $1.97/family nominal was near-exact). Pricing snapshot identical to the clean
  run ($1/$5 per MTok), so dollar deltas are pure token deltas. **typo $2.1216**
  (prompt tokens +7.9%), **ocr $2.0259** (+2.8%), **case $2.1394** (+8.8% — the
  *largest* inflation). Total **$6.287** vs approved ≈$6.8 ceiling; computed cost
  matches `openrouter_reported_cost_usd` per run. Cumulative project spend
  ≈$73.67 of ≈$75 (≈$1.3 remaining).
- **Findings:** (1) **Typo is the only family that hurts Haiku** (−3.1 F1 points,
  CI excludes zero), and its point loss is smaller than every Tier A arm at the same
  nominal rate (word+char −4.4, word-only −6.6; cross-arm comparison directional-only —
  different n, no joint CI, same discipline as task 5's char-shield note).
  (2) **OCR noise: no measurable effect** on Haiku (ΔF1 CI spans zero) where Tier A
  pays ~−1 point — the confusion-table corruptions that break TF-IDF vocabulary lookups
  are evidently recoverable in context by the LLM. (3) **Case-mangling: no accuracy
  effect despite the largest tokenizer disruption** (+8.8% prompt tokens). The one cell
  Tier A cannot occupy turns out to be a robustness win, not a vulnerability — the
  §6.3.4 cross-tier exhibit now reads: LLM ≳ word+char TF-IDF > word-only TF-IDF under
  character noise (Tier B cell pending GPU). (4) **Perturbation is a serving-cost tax
  for Tier C only:** +3–9% per-call cost purely from token inflation, invisible to
  Tier A economics. A router implication worth one sentence in the case study: noisy
  inputs make escalation *more* expensive exactly when Tier A is *least* reliable.
- **Deviations / limits:** n=1,500 CIs are ~6× wider than the Tier A cells (±0.016 vs
  ±0.003); a Haiku "shield" claim vs Tier A is directional-only. Rate 0.10 only — no
  0.05 rung, no dose-response. Sonnet excluded by cost. Realized changed-char fraction
  still not emitted (carried over from task 5).
- **Verdict:** hypothesis confirmed — the LLM degrades slowest under character noise,
  and the case-family risk did not materialize. §6.3.4 is now complete for all
  available tiers; Tier B rows pending GPU.
- **Repro:** `make perturb-tier-c` (3 runs, 4,500 calls, $6.287 measured) then
  `make perturb-report` → `results/perturbation/summary.json` (18 rows: 15 Tier A +
  3 Tier C, 0 missing). Run ids: typo `182774b6`, ocr `237bffae`, case `c7e53e2a`;
  clean baseline `70a1b0c4` (Phase 3). Raw receipts under
  `results/tier_c_raw/tier_c_haiku_zeroshot_test_iid_perturb_*_10/`.
  Tests: `uv run pytest tests/test_perturb.py -q` (70 passed).

## 2026-08-10 — Phase 5 task 6: drift charts + escalation-rate-over-time rollup ($0, derivation-only)

- **Context:** last pending non-stretch Phase 5 item; owner constraint: API budget
  effectively exhausted (≈$1.3 left), so this task is **$0 by construction** — every
  number derives from `results/runs.jsonl`, the frozen v2-isocal thresholds
  (`results/thresholds/summary__v2-isocal__cost-f76ad15a.json`), committed Tier C
  receipts, and the verified per-example artifacts (`data/preds/`). **No new
  runs.jsonl records** (append-only log untouched; derivation outputs only, under
  `results/drift/`).
- **Hypothesis (§6.3.1's explicit question):** the router's escalation rate
  self-adjusts as confidence drops under drift — i.e. the frozen CAL-derived τ*
  produces a rising escalation rate as the 2026-H1 prior shift erodes Tier A
  confidence, rather than silently answering at degraded quality.
- **Method:** new `src/triage_lab/drift_charts.py` (+ `make drift-charts`, matplotlib
  behind a new optional `charts` extra mirroring `tierc`; core `uv sync --frozen`
  unchanged). Logged metrics (macro-F1/ECE/accuracy + CIs) are **copied** from
  runs.jsonl, never recomputed. Escalation series computed fresh by applying the
  frozen v2-isocal τ* to yearly `p_max` artifacts — calibration-space-consistent by
  construction (yearly Tier A configs are isotonic-calibrated, byte-matching the
  derivation rung's feature/param config; the v1 raw-vs-isocal mismatch cannot recur).
  τ* values **loaded** from the thresholds file (family-matched: `a_to_human`
  full_cal 0.500519 / paired 0.496807; `a_to_c_parsefail_human` paired 0.421712),
  never transcribed; τ exempted from JSON rounding (`NO_ROUND_KEYS`) per the
  `threshold_opt._tau_json` lesson. Parse-fail rows recovered per the Phase 4
  convention (`cost_model.join_parse_failed` off committed receipts, joined on
  `complaint_id`; escalated parse-fail → human, Tier C fallback label discarded, not
  scored); receipt-joined counts cross-checked against each run's logged
  `extra.parse_failures`. Paired arms: Sonnet artifact ids per slice (exactly 1,500,
  asserted; identical to Haiku's on all four drift years, ⊂ Haiku's 5,000 at
  test_iid). Bootstrap CIs n=1,000, percentile, seed 20260805 (harness convention).
- **Result — escalation-rate-over-time (evidence class: measured, derived from frozen
  artifacts under frozen τ; a_to_human on full slices, τ=0.500519, CAL operating
  point 0.0994):**

  | slice | n | escalation [95% CI] | answered-set acc | answered-set macro-F1 |
  |---|---|---|---|---|
  | 2022-H2 | 104,443 | 0.0947 [0.0930, 0.0964] | 0.8829 | 0.8110 |
  | 2023 | 20,000 | 0.1032 [0.0992, 0.1076] | 0.8742 | 0.8077 |
  | 2024 | 20,000 | 0.0993 [0.0951, 0.1033] | 0.8707 | 0.8076 |
  | 2025 | 20,000 | 0.1003 [0.0961, 0.1047] | 0.8479 | 0.7776 |
  | 2026-H1 | 20,000 | **0.1674 [0.1623, 0.1724]** | 0.7436 | 0.7178 |

  `a_to_c_parsefail_human` (paired n=1,500, τ=0.4217, CAL point 0.0400): escalated→C
  0.0320 / 0.0293 / 0.0380 / 0.0300 / **0.0513 [0.0407, 0.0627]**; escalation is
  identical across terminal models by construction (pure Tier A property — built-in
  consistency check). Parse-fail→human: Haiku 0 on every slice; Sonnet 1/4/1/0/4 of
  12/19/10/16/26 slice-level parse-fails (human arm ≤0.27% of the paired subset).
  Charts (3 SVGs under `results/drift/charts/`): macro-F1-over-time (all available
  tiers, CI whiskers, taxonomy annotation), ECE-over-time (Tier A only — Tier C
  structured-output p_max is a degenerate one-hot, ECE not meaningful, stated on the
  chart), escalation-over-time (all three arms vs CAL reference lines). Every chart
  carries an evidence-class footnote and support-size labels; `summary.json` carries
  a full `evidence_class` block (Tier B series labeled "pending").
- **Findings:** (1) **Hypothesis confirmed, with a sharper shape than expected:** the
  frozen gate holds *flat at the CAL operating point for three straight years* (0.095–
  0.103 vs 0.0994) and then jumps **+68% relative in 2026-H1** — escalation
  self-adjusts late and abruptly, co-timed with the credit_reporting prior shift
  (task 3), not gradually. The router's confidence signal sees the same cliff the
  F1 curve does. (2) The selective gate is worth ~5 accuracy points at the cliff:
  answered-set accuracy 0.7436 vs 0.6918 full-slice Tier A, bought with a 16.7%
  human/escalation load. (3) The a_to_c arm's Tier C human-fallback stays ≤0.27%
  everywhere — parse-failure is a stable, near-silent escape hatch, and Haiku's is
  exactly zero on all 7,500 paired rows (5 slices × 1,500). (4) 2026-H1 is the only
  slice where every series moves together: Tier A F1 cliff (task 1), prior-shift
  decomposition (task 3), escalation-rate jump (this task) — one drift event, three
  independent measurements.
- **Deviations / limits:** (a) orchestrator's brief mis-assigned the a_to_human
  paired τ (0.4968) to the a_to_c family; implementation caught it and used the
  family-matched τ* loaded from the thresholds file — transplanting an argmin across
  families is the same error class as the v1 calibration mismatch. (b) The a_to_c
  2022-H2 point is on the 1,500 paired ids (consistent timeline support), NOT the
  5,000-row support Phase 4's router_sim reported — the two numbers intentionally
  differ; support sizes are labeled on the chart and recorded per row. (c) The 2023
  taxonomy consolidation was announced 2023-04 but first observed in the snapshot at
  2023-08, so the 2023 slice *straddles* it; charts mark the test_iid|2023 boundary
  and shade the 2023 bucket rather than drawing a falsely precise dated line.
  (d) `accuracy_system`/`macro_f1_system` credit human-routed rows as correct — the
  cost model's P(error|human)=0 ASSUMPTION, labeled as such; `*_answered` views carry
  no assumption. (e) CI note: `uv sync --frozen --extra charts` alone uninstalls the
  tierb/tierc extras (4 tests then fail on missing torch/openai); full suite needs
  all extras — 502 passed, 1 skipped with `--extra charts --extra tierb --extra
  tierc`. (f) Determinism: with `--generated-at` pinned, `summary.json` + all SVGs
  are byte-identical across runs; unpinned, only the `generated_at` line differs
  (SVGs byte-identical regardless).
- **Verdict:** Phase 5 acceptance line closed for available tiers — drift charts
  rendered from the results log ✓, escalation-rate-over-time computed ✓, every
  chart's evidence class labeled ✓. Tier B series/rows pending GPU; novel-class
  probe remains the only (stretch) Phase 5 item. Cost: **$0.000** (no API calls).
- **Repro:** `make drift-charts` (equivalently
  `uv run --extra charts python -m triage_lab.drift_charts --all`); prerequisites
  `make preds` and `make thresholds`. Outputs: `results/drift/summary.json`
  (15 logged rows + 20 escalation rows) and `results/drift/charts/{macro_f1,ece,
  escalation}_over_time.svg`. Source run ids: Tier A `8e4d6345`/`389d3e69`/
  `c1c810e2`/`cef3a58c`/`2ae5d8ea`; Haiku `70a1b0c4`/`cf190ce2`/`b83b2d0d`/
  `446fa236`/`aa2bc9a0`; Sonnet `e1503146`/`c82988bd`/`9d54ec58`/`daa58725`/
  `55cffcbe`; thresholds run `40513354`, cost config `f76ad15a`.

## 2026-08-10 — Phase 6 task 1: static demo site scaffold ($0, derivation-only)

- **Hypothesis:** All six §7 demo panels (triage playground, cost-quality frontier,
  router policy builder, drift timeline, calibration panel, receipts drawer) can be
  scaffolded as a fully static site whose every number is derived from committed
  results artifacts — including the curated-sample Tier C responses, drawn entirely
  from existing committed receipts (owner's $0 constraint: **no new API calls**) —
  with Tier B rendered as explicit pending placeholders wired to backfill slots.
- **Result:** Confirmed. New `demo/` site: `index.html` + `assets/app.js` +
  `assets/styles.css` (vanilla JS, zero external dependencies — the only `http://`
  string in the bundle is the SVG XML namespace constant) + `demo/data/` (9 committed
  JSON payloads, 965 KB total) built deterministically by new
  `src/triage_lab/demo_build.py` (`make demo-data`); schema contract in
  `demo/DATA_CONTRACT.md`. Curated sample set: n=200, seed 20260806, class-stratified
  (largest remainder, min 1/class; credit_reporting 101 … vehicle_loan 3) from the
  pool of 1,500 TEST-IID complaint_ids scored by BOTH Haiku (`70a1b0c4`) and Sonnet
  (`e1503146`) finals; pool_sha256 `3b31a165…`; selection FROZEN in
  `demo/data/curated_ids.json` (rebuild hard-fails on mismatch, exemplar-style).
  Narratives joined from the frozen split parquet (CFPB, US-gov public domain);
  Tier C labels/cost/latency/provider/tokens per sample copied from committed
  receipts only; per-sample router path recomputed with the frozen v2-isocal op via
  `router_sim` logic (τ_a_to_human 0.5005, τ_a_to_c 0.4217, loaded from
  `results/thresholds/`, never transcribed) and cross-checked against router_sim's
  own to_human vector (hard gate). Calibration bins (15) derived from frozen
  per-example artifacts must reproduce each run's logged ECE to 1e-9 (hard gate).
  Router paths over the 200: 187 answered-at-A, 13 escalated→C-answered, 0 →human
  (Haiku parse-fail = 0 on TEST-IID). Verified in-browser (local static server):
  all six panels render, receipts drawer opens from any run chip, policy-builder
  client re-solve reproduces the measured $872.81/1k at default sliders exactly and
  switches to "derived (client re-solve)" labeling off-defaults; light + dark themes;
  zero console errors. One integration fix during verification: app.js read
  `expected_cost_per_1k` as a flat metric — it is a breakdown object; now reads
  `.total`.
- **Findings:** (1) The owner's $0 gate closes with no gap: committed receipts +
  frozen artifacts fully determine the curated set; the only data not in git
  (narratives) is public-domain and reproduced byte-identically by `make data`.
  (2) Traceability is testable, not aspirational: `tests/test_demo_build.py`
  (43 tests) asserts every run_id under `demo/data/` exists in `results/runs.jsonl`
  and every copied metric equals its source record; receipt-consistency and
  pool-freeze tests run WITHOUT `data/` (CI-safe: 39 passed, 4 skipped in a no-data
  repo copy; full local: 43 passed). (3) Rebuilds are byte-identical (no wall-clock
  in outputs). (4) `.gitignore`'s bare `data/` pattern silently ignored `demo/data/`;
  added `!demo/data/` negation (top-level dataset dir coverage unchanged —
  `data/preds/*` still ignored).
- **Deviations / limits:** builder deviations from DATA_CONTRACT.md are enumerated
  in that file's terms: frontier cost axis uses `expected_cost_per_1k.total` with
  `cost_basis` labels (api-only would park Tier A at $0 — meaningless frontier axis);
  router frontier points carry point-only macro-F1 (router_sim logs no CI on levels,
  only on paired deltas); no raw-probability TEST-IID calibration exhibit exists
  (all TEST-IID finals are isotonic) — three honestly-labeled exhibits emitted
  instead; τ sweep downsampled to 256 grid points. Live in-browser ONNX inference,
  case study page, provenance links, and `make reproduce-headline` are LATER Phase 6
  tasks — the playground states this. All Tier B panels/series/slots render as
  explicit "pending Tier B" placeholders keyed for backfill
  (`tier_b1_modernbert`, `tier_b2_distilbert`, `router_a_b_c`, `tier_b1`, `tier_b2`,
  `tier_b1_temp_scaling`, `tier_b2_temp_scaling`).
- **Verdict:** Phase 6 scaffold task closed. Site scaffold static/offline ✓, six
  panels wired to available exhibits ✓, every displayed number traces to a results
  record (test-enforced) ✓, Tier B pending placeholders ✓, curated set from
  committed receipts only ✓. Cost: **$0.000** (no API calls; `results/runs.jsonl`
  untouched — derivation-only).
- **Repro:** `make demo-data` (equivalently
  `uv run python -m triage_lab.demo_build --all`; prerequisite `make preds` for the
  gitignored per-example artifacts) then
  `uv run pytest tests/test_demo_build.py -q` (43 passed) and any static server on
  `demo/` (e.g. `python3 -m http.server -d demo`). Source run ids: Tier A
  `8e4d6345`/`c20cd14a`/`abcadd53`/`40513354`; Tier C `70a1b0c4`/`82af4e01`/
  `e1503146`/`d1c42d7d`; router artifacts opv2 cost config `f76ad15a`.

## 2026-08-10 — Phase 2: Tier B training-results bundle ingest + validation

- **Context:** The real training-results bundle arrived from the shared A6000
  (`data/checkpoints/incoming/tier_b_training_results_20260807T191924Z.tar.gz`,
  1,917,787,594 bytes, sha256 `c47f927e333752e014259f453e820c3a890a4a358dd0f02f577094647fbd000e`).
  The 2026-08-08 incident (outgoing kit shipped back by mistake) does not recur: this
  archive contains the four trained checkpoints. Owner decision (2026-08-10): ingest
  takes priority over remaining Phase 6 tasks; eval backfill follows in later sessions.
- **Hypothesis:** the bundle contains all four completed runs
  (B1 ModernBERT s{a,b,c} + B2 DistilBERT s0), trained under the frozen configs,
  seeds, and data kit, with an intact chain of custody.
- **Validation (all fail-loud; `scripts/validate_tier_b_bundle.py`):**
  1. `gzip -t` clean; 52 tar entries, extracted to a staging dir.
  2. Bundle `manifest.json` re-hash: **51/51 files** sha256+size match; **zero
     unlisted files** in the archive.
  3. Chain of custody closed on four anchors: bundled `metadata/configs/*.yaml`
     **byte-identical** to frozen `configs/` (and manifest job `config_sha256`s match
     the frozen files); bundled `metadata/train_tier_b.py` byte-identical to
     `scripts/train_tier_b.py` (`5764a3e1…`); bundle `input_bundle.sha256` ==
     local `data/tier_b_colab_bundle.tar.gz` (`87e3ddab…`); bundle
     `data_manifest_sha256` == frozen `data/tier_b_kit/manifest.json` (`91befff3…`),
     whose `input_sha256` (`170f66cd…`) equals every run's `data_input_sha256`.
  4. Per-run `training_meta.json` / `status.json`: seeds exactly the frozen list
     (sa 20260805, sb 20260806, sc 20260807, s0 20260805); base models correct;
     precision **bf16** on `cuda:NVIDIA RTX A6000` (runbook auto-selection);
     n_train 300,000 / n_cal 86,972 == kit manifest; all four
     `exit_status=completed` at step 28125/28125 (= 300k×3/32, effective batch 32 ✓).
  5. Weights sanity: valid safetensors headers, fp32 tensors, 9-class heads matching
     the frozen label order; the three B1 `model.safetensors` have **distinct** shas
     (not copies). Post-placement re-hash at canonical paths: **32/32 files** clean.
- **Result — per-run training summary (final CAL eval from `training_log.jsonl`;
  evidence class: training-script numbers — uncalibrated argmax, no CIs, NOT harness
  records; `results/runs.jsonl` untouched):**

  | run | seed | final CAL macro-F1 | final CAL acc | epoch-2 CAL macro-F1 | train wall-clock | truncation |
  |---|---|---|---|---|---|---|
  | tier_b1_sa | 20260805 | 0.7856 | 0.8468 | 0.7974 | 8,054 s | 0.32475 |
  | tier_b1_sb | 20260806 | 0.7874 | 0.8481 | 0.8005 | 8,143 s | 0.32475 |
  | tier_b1_sc | 20260807 | 0.7865 | 0.8474 | 0.7942 | 8,190 s | 0.32475 |
  | tier_b2_s0 | 20260805 | 0.7923 | 0.8536 | 0.7934 | 2,238 s | 0.32825 |

- **Findings:** (1) All three B1 seeds peak at **epoch 2** and dip at epoch 3 with
  eval-loss rising 0.42–0.44 → 0.57–0.63 — mild late overfit; the frozen protocol
  ships the final-epoch checkpoint, so the epoch-3 weights stand (no epoch
  cherry-picking; noted as context for TEST-IID numbers). (2) **B2 DistilBERT's final
  CAL macro-F1 (0.7923) edges all three B1 finals (0.7856–0.7874)** on these
  training-script numbers — pre-registered surprise to re-examine under the harness
  (temperature scaling, CIs) before any claim. (3) B1 seed spread on final CAL
  macro-F1 is tight: 0.7856–0.7874 (range 0.0018). (4) Truncation rate at
  max_seq_length 256 is ~32.5% of TRAIN rows for both tokenizers — matches the
  runbook's long-tail expectation. (5) Bundle also carries supervisor machinery not
  in the original kit (`tier_b_supervisor.py`, `run_train_py310.py`, per-attempt
  logs) — retained as receipts, not repo code.
- **Placement:** checkpoints moved to the canonical config targets
  `data/checkpoints/tier_b{1_sa,1_sb,1_sc,2_s0}/` (gitignored); bundle receipts
  (manifest, supervisor metadata, attempt logs) retained at
  `data/checkpoints/incoming/receipts_20260807T191924Z/`; the original tarball is
  retained unmodified.
- **Verdict:** Ingest ACCEPTED — zero mismatches across all five check groups.
  Phase 2 training rows are done; Tier B is **unblocked** for the eval backfill
  (harness TEST-IID runs + temperature scaling, then the pending Phase 4/5/6
  Tier B slots, per STATUS.md §b).
- **Repro:** `shasum -a 256 data/checkpoints/incoming/tier_b_training_results_20260807T191924Z.tar.gz`
  (expect `c47f927e…`); then
  `mkdir -p /tmp/tb_check && tar -xzf data/checkpoints/incoming/tier_b_training_results_20260807T191924Z.tar.gz -C /tmp/tb_check && uv run python scripts/validate_tier_b_bundle.py /tmp/tb_check`
  (expect `ALL CHECKS PASSED`); final CAL rows:
  `grep '"eval_macro_f1"' data/checkpoints/tier_b1_sa/training_log.jsonl | tail -1`
  (and likewise for sb/sc/s0).

## 2026-08-10 — Phase 2: Tier B harness TEST-IID finals + paired deltas (eval backfill task 1)

- **Context:** first Tier B eval-backfill session (STATUS.md §b item 1; runbook §6).
  Checkpoints are the 2026-08-10-validated set at `data/checkpoints/tier_b{1_sa,1_sb,1_sc,2_s0}/`.
- **Accept criteria restated (UPGRADE_PLAN §8 Phase 2, eval portion — ONNX parity is
  the next session's task):** both Tier B points evaluated with CIs; B1 beats Tier A
  on macro-F1 with CI excluding zero *(expected; if not, that is the finding)*;
  B1-vs-B2 delta reported (intra-tier trade-off exhibit); seed variance reported;
  temperature scaling fit on CAL.
- **Hypothesis (two-part):** (1) plan expectation — B1 ModernBERT beats Tier A LogReg
  on TEST-IID macro-F1 with paired CI excluding zero. (2) The **pre-registered
  surprise** from the ingest entry above — B2 DistilBERT's final CAL macro-F1 edging
  all three B1 seeds — either survives or dissolves under the frozen harness protocol
  (temperature scaling on CAL, full TEST-IID, bootstrap CIs).
- **Method:** four harness runs (one per frozen config; single-variable: only the four
  final-eval configs, nothing else changed), temperature fit on CAL (n=86,972), eval
  on **full TEST-IID (n=104,443)**, MPS inference, git sha `25c3b0b`; bootstrap
  n=1,000 seed 20260805. Paired comparisons via `harness.paired_bootstrap_delta` +
  exact McNemar on complaint_id-aligned prediction artifacts (new
  `scripts/compare_tier_b.py` → `results/tier_b_compare/summary.json`; script
  smoke-validated on Tier A artifacts: self-comparison → delta exactly 0, 0 discordant
  pairs; LogReg-vs-CNB reproduces the known +0.0340). Tier A baseline = run
  `8e4d6345…` (`tier_a_logreg_test_iid`, macro-F1 0.7605 [0.7564, 0.7643]).
- **Result — final TEST-IID records (all appended to `results/runs.jsonl`):**

  | run | run_id | macro-F1 [95% CI] | acc | ECE | fitted T | wall-clock |
  |---|---|---|---|---|---|---|
  | tier_b1_sa | `8071d31d…` | 0.7878 [0.7836, 0.7918] | 0.8562 | 0.0061 | 1.775 | 13,225 s |
  | tier_b1_sb | `adb96307…` | 0.7878 [0.7838, 0.7914] | 0.8557 | 0.0051 | 1.953 | 10,752 s |
  | tier_b1_sc | `a523049a…` | 0.7863 [0.7821, 0.7898] | 0.8553 | 0.0076 | 1.750 | 15,155 s |
  | tier_b2_s0 | `5517ebf1…` | **0.7950 [0.7909, 0.7988]** | 0.8609 | 0.0092 | 1.319 | 4,230 s |

  **B1 seed variance (macro-F1):** mean **0.7873 ± 0.0009** (sd, ddof=1), range
  0.0016 (0.7863–0.7878) — as tight as the training-script CAL spread.

  **Paired deltas (macro-F1, paired bootstrap; McNemar exact, n=104,443):**

  | comparison | Δ macro-F1 [95% CI] | Δ acc [95% CI] | McNemar p |
  |---|---|---|---|
  | B1 sa − A logreg | +0.0273 [+0.0234, +0.0312] | +0.0118 [+0.0097, +0.0138] | 1.3e-30 |
  | B1 sb − A logreg | +0.0273 [+0.0235, +0.0311] | +0.0113 [+0.0093, +0.0132] | 7.3e-28 |
  | B1 sc − A logreg | +0.0258 [+0.0220, +0.0297] | +0.0109 [+0.0089, +0.0130] | 2.3e-26 |
  | B1 sa − B2 | **−0.0072 [−0.0106, −0.0038]** | −0.0047 [−0.0063, −0.0029] | 6.3e-08 |
  | B1 sb − B2 | **−0.0072 [−0.0106, −0.0040]** | −0.0052 [−0.0069, −0.0036] | 2.4e-09 |
  | B1 sc − B2 | **−0.0088 [−0.0119, −0.0054]** | −0.0056 [−0.0073, −0.0039] | 7.1e-11 |
  | B2 − A logreg | +0.0345 [+0.0310, +0.0382] | +0.0165 [+0.0147, +0.0184] | 7.4e-65 |

- **Findings:** (1) **B1 beats Tier A** — every seed's paired CI excludes zero
  (+0.026 to +0.027 macro-F1); the plan's expected direction, now certified.
  (2) **The pre-registered surprise is CONFIRMED on the frozen protocol: B2
  DistilBERT beats B1 ModernBERT** — all three per-seed B1−B2 deltas are negative
  with CIs excluding zero (−0.0072 to −0.0088) and McNemar p ≤ 6.3e-08. The
  "headline accuracy model" is not the top Tier B point; the deployment point is,
  and it also beats Tier A by +0.0345. (3) Fitted temperatures are all > 1
  (B1 1.75–1.95, B2 1.32) — overconfident raw heads, consistent with the epoch-3
  eval-loss rise seen in the training logs; post-scaling ECE lands at 0.005–0.009.
  (4) B2 is also better calibrated in raw form (smallest T) and dominates on
  AURC (0.0365 vs B1's 0.0420–0.0460), which matters for router thresholds.
- **Interpretation caution (protocol-scoped claim):** "B2 > B1" is a claim about
  *these frozen recipes* (max_seq_length 256, 3 epochs, published-recipe LRs,
  final-epoch checkpoint) on *this task*, not about the architectures. The B1
  training curves peak at epoch 2 then overfit; the frozen protocol ships the
  final epoch, and no re-training or checkpoint cherry-picking is permitted.
  The honest headline: under this protocol the 66M deployment model is the best
  Tier B point. That is the finding, exactly as the Phase 2 accept line anticipates.
- **Verdict:** ACCEPTED — Phase 2 eval-side criteria all met: both points CI'd on
  TEST-IID ✓; B1-vs-A CI excludes zero ✓; B1-vs-B2 delta reported (sign inverted —
  the finding) ✓; seed variance reported ✓; temperature scaling on CAL ✓.
  Remaining Phase 2 task: DistilBERT int8 ONNX export + parity (next session).
- **Repro:**
  `uv run python -m triage_lab.harness configs/tier_b1_modernbert_sa.yaml` (and
  `…_sb.yaml`, `…_sc.yaml`, `configs/tier_b2_distilbert_s0.yaml`) — one record each;
  then `uv run python scripts/compare_tier_b.py` (defaults select the latest run per
  config + the `8e4d6345…` Tier A baseline; expect the two tables above; writes
  `results/tier_b_compare/summary.json`).
