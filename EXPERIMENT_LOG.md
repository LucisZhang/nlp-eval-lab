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
