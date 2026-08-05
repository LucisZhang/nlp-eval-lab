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
