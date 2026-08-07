# UPGRADE PLAN — Triage Router Lab

**A three-tier text-triage evaluation lab with a routing-policy centerpiece, built from two NLP coursework seeds.**

This document is self-contained. It was produced after a first-hand audit of every file in this directory (`.../Natural Language Processing/Project/`). It is written for a coding agent with **no access to the conversation that produced it**. Follow it as the single source of truth.

> **Hard constraint for the implementing agent:** Do NOT modify, move, or delete anything in this coursework directory. It is an archive and lives in OneDrive. Build the new project in a **fresh repository outside this folder** (suggested: `~/Projects/triage-router-lab`). Files here are referenced read-only, as seed artifacts and evidence.

---

## 1. Project goal and target narrative

### 1.1 Who this is for

Xiangguo Zhang's portfolio site (https://xiangguozhang.com) is a verification-first "systems portfolio" targeting Data Analytics / Data Engineering / AI Application Engineering roles. Every case study has: hard measured metrics, an interactive demo, a "How this was verified" section, and a "What this does not prove" section with evidence-class labels (measured / estimated / projected). Existing case studies: Release Guardian (LangGraph release gate), RAG Quality Lab (RAG evaluation, 11,309 docs), Privacy Preflight (browser-local redaction + OCR), Margin Control Tower (Olist margin analytics), Streaming Reliability Lab (CDC → Flink → Iceberg failure injection), Credit Policy Lab (Lending Club risk policy).

Known portfolio gaps this project must attack: **(a) no model training / fine-tuning anywhere in the portfolio, (b) weak experimentation/measurement-rigor signal beyond offline eval, (c)** — explicitly out of scope here — production cloud deployment (a different project should own that).

### 1.2 The one-sentence pitch

> **"Before you pay for an LLM, know your frontier."** A consumer-complaint triage system evaluated across three model tiers — a classical linear baseline, a fine-tuned transformer, and a few-shot LLM — measured on quality, calibration, cost, and latency, then combined into a confidence-based **router** that beats any single tier on the cost-quality frontier, and stress-tested against **ten years of real distribution drift**.

### 1.3 Why this makes a hiring manager stop

1. **It answers the question every AI team is actually arguing about in 2026**: "do we fine-tune a small model or call an LLM?" — and answers it with measured frontier curves, not vibes. Model routing / LLM cascades are a hot, current topic; almost no junior portfolio treats model selection as an economic decision.
2. **It fills the fine-tuning gap** with a real training run (transformer fine-tuning, training curves, seeds, ablations), not a `model.fit()` toy.
3. **It uses real, messy, public-domain business data with genuine temporal drift** (CFPB complaints, 2015→2026), so the drift chapter is *measured history*, not synthetic noise.
4. **It inherits and upgrades a documented experimental discipline** — the coursework seed (CoNLL NER MEMM) contains a genuinely rigorous ablation log (5 leaderboard submissions, single-variable comparisons, harmful-feature removal, per-class error analysis, an explicit "we couldn't afford bootstrap CIs under competition time limits" admission). The new lab's story: *"every lesson from that log, done properly this time — every headline number now carries a 95% CI."*
5. **It cross-links the portfolio**: evaluation DNA from RAG Quality Lab, finance domain from Credit Policy Lab, in-browser inference DNA from Privacy Preflight.

### 1.4 Naming

Primary: **Triage Router Lab**. Alternates if the site prefers: *Complaint Triage Lab*, *Model Frontier Lab*. Keep the site's existing "\<Domain\> \<Noun\> Lab" pattern.

---

## 2. Verdict on the reference hypothesis

A prior advisory session proposed: merge the two seeds into a "text triage / routing evaluation lab" — classical baseline vs fine-tuned small model vs LLM few-shot, evaluated on accuracy, calibration, cost, latency, drift; original code disposable; reuse the experimental discipline.

**Verdict: ADOPT the shape, with three material upgrades**, after weighing alternatives:

| Option considered | Why not (or why yes) |
|---|---|
| **A. Reference hypothesis as-is** (generic ticket/intent triage on Banking77 or CLINC150) | Right shape, but generic intent datasets have **no natural drift axis** (drift would have to be synthetic = weaker evidence class) and weak business realism. Adopted the shape, replaced the substrate. |
| **B. Three-tier NER/PII extraction lab** (upgrade the MEMM seed directly: MEMM/CRF vs fine-tuned token classifier vs LLM extraction) | Fills the training gap and ties to Privacy Preflight, but: span-level eval is illegible to non-NLP hiring managers, the demo is less business-native, CoNLL-2003 **cannot be redistributed** (Reuters research license), and the portfolio already owns the redaction story via Privacy Preflight — overlap, not new surface. Rejected as the core; kept as an explicit stretch tie-in (§9, Phase 7). |
| **C. Two-track mega-lab (classification + NER)** | Scope kill. A handed-off agent will ship neither track to case-study quality. Rejected. |
| **D. Chosen: triage lab on CFPB complaints + routing-policy centerpiece + measured temporal drift** | Real business framing (route complaints to the right team), US-government public-domain data (safe to redistribute samples), timestamps 2015→2026 giving *measured* covariate/prior/label drift including a real 2023 taxonomy change, and an out-of-the-benchmark differentiator: the router as a business decision system. |

The three upgrades over the reference hypothesis:

1. **Substrate**: CFPB Consumer Complaint Database instead of a static intent benchmark → drift becomes *measured*, licensing becomes trivial, domain links to Credit Policy Lab.
2. **Centerpiece**: not "three models compared" (a benchmark table) but a **confidence-cascade router** with an explicit business cost model, risk-coverage curves, and a demonstration that the router dominates every single tier on the cost-quality frontier. This is the thing that reads as *systems engineering*, not coursework.
3. **Contamination defense**: LLM-tier results are reported on a **post-knowledge-cutoff slice** (complaints received after the LLM's training cutoff) alongside the full test set, pre-empting the obvious "the LLM memorized public data" objection. Almost nobody does this; it is exactly the kind of verification-first move the site is built on.

---

## 3. Ground-truth asset inventory (first-hand audit)

All paths relative to this directory. Audit performed 2026-08-02 by reading source files, reports, notebooks, and score records directly.

### 3.1 What exists and its quality

**`Project Task2/` — Naive Bayes Reuters classification (solo)**
- `Starter Code/naive-bayes.py` (537 lines): complete from-scratch multinomial NB pipeline — tokenization, hand-implemented Porter stemmer fallback, per-class word counting, frequency-based feature selection, Laplace smoothing, log-space classification, macro-F1 evaluation. Clean, argparse-driven, dependency-free. Genuinely from scratch.
- `Starter Code/report.md`: full report. **Verified numbers**: train 5,787 docs / test 2,298 docs, 5 classes (crude, grain, money-fx, acq, earn); macro-F1 **0.96477** at the required 10,000 features; feature-count sweep 3k/5k/10k/15k → 0.96367 / 0.96525 / 0.96477 / 0.96585. Report openly acknowledges AI assistance.
- Data: `train.json`, `test.json` (Reuters-derived; **do not redistribute** — Reuters corpus licensing is research-only).
- Output artifacts for all sweep settings present (`classification_result_*.txt`, `word_dict_*.txt`, etc.).

**`Project3 (Group)/` — CoNLL-2003 MEMM NER (group; user led modeling)**
- `NER_MEMM_Project_Session_Summary.md` (379 lines): the crown-jewel artifact. A complete, dated iteration log: Phase 1 (dev macro-F1 0.8322) → Phase 2 (0.8535) → Phase 3 (0.8553) → Phase 3-Optimized (0.8733); five Kaggle submissions (v1 0.83522, v2 0.82335, v3 0.83122, v4 0.83239) each with a stated hypothesis and a stated verdict (including "HYPOTHESIS WRONG" on v2); harmful-feature identification (`no_gaz`, `pl|s3` removed with mechanistic explanations); the finding that MAX_ITER (convergence), not features, was the biggest lever; documented OOM failure and fix; documented dev→test drift (~0.04 F1, temporal OOV gap — a real distribution-shift observation).
- `Starter Code/notebooks/ner-phase4-v5.ipynb` … `v7.ipynb`: post-summary iterations. **v5's public score 0.83615 is confirmed** in v7's docstring ("v5 improved public LB only slightly (0.83615)"). v6/v7 exist as code; no recorded scores in this archive.
- `NER Score.docx`: starter-code baseline dev report — **macro-F1 0.8136** (this is the origin of the "0.8136 → 0.8733" claim: 0.8136 is the *starter baseline*, not the user's Phase 1).
- `NER_Feature_Engineering_Consolidated_Reference.md` (837 lines): a synthesis of 5 AI research sources into a consensus-ranked feature catalog with citations (Klein 2003, Curran & Clark 2003, Chieu & Ng 2003, Ratinov & Roth 2009). Methodologically interesting (structured multi-source AI research), content NER-specific.
- `Pre/Speaker_Notes_and_QA-2.md`: bilingual presentation script + 10 deep Q&As (sparsity-control arithmetic 9×4=36 vs 9×5000; macro-vs-micro F1 under 83% O-class; GIS convergence; honest answer on statistical noise and lack of bootstrap CIs).
- Data: `Starter Code/data/*.csv` — CoNLL-2003 (**do not redistribute**; Reuters/RCV1 research license).
- `NLP/` folder: raw outputs from 5 AI research assistants + `memm_features.py` (a prior reference implementation, never the submitted code).

**`Project Task1/` — Books-to-Scrape spider (solo)**
- `project1/spider-url.py`, `spider-books.py`, `preprocess.py`, `report.md`: a competent but small crawl-and-preprocess pipeline (200 books). Nothing here is load-bearing for the new project.

### 3.2 Accuracy audit of the owner's claims

| Claim | Verdict |
|---|---|
| NB macro-F1 ~0.965 | **Accurate** (0.96477 at required setting; 0.96585 best). |
| NER dev macro-F1 0.8136 → 0.8733 | **Accurate with nuance**: 0.8136 is the provided starter baseline; the user's own first phase was 0.8322. Phrase publicly as "starter baseline 0.8136 → 0.8733 through documented feature iteration." |
| Kaggle public 0.83615 | **Accurate** (v5; confirmed in v7 docstring). |
| "I led the modeling, teammate did frontend" | **Not verifiable from this archive** (group PPT exists; no frontend artifacts here). Do not put this claim under "measured" evidence on the site; keep it as biographical context only. |
| General caveat | Both reports acknowledge AI assistance; the NER feature research was explicitly AI-source-synthesized. The portfolio narrative must therefore rest on **the new lab's independently reproducible evidence**, with coursework framed as "seed / provenance," never as verified metrics. |

### 3.3 Reuse vs discard

| Asset (exact path) | Decision | How |
|---|---|---|
| `Project3 (Group)/NER_MEMM_Project_Session_Summary.md` | **REUSE (as evidence artifact)** | Link/screenshot in the case study's "provenance" section as proof of prior experimental discipline; import its lesson list (convergence as hyperparameter, harmful-feature removal, single-variable submissions, dev/test drift) as the lab's stated methodology principles. |
| `Project3 (Group)/Pre/Speaker_Notes_and_QA-2.md` (esp. Q10 on statistical noise) | **REUSE (narrative)** | The "coursework couldn't afford bootstrap CIs; this lab CIs everything" arc comes from here. |
| `Project Task2/Starter Code/naive-bayes.py` | **REUSE (optional Tier-0 exhibit)** | Port unchanged into the new repo under `seeds/` as a from-scratch-NB curiosity; the *evaluated* classical tier uses sklearn (see §4). Cut first if time-boxed. |
| `Project Task2/Starter Code/report.md` | REUSE (provenance link only) | |
| `Project3 (Group)/NER_Feature_Engineering_Consolidated_Reference.md` | DISCARD from case study (optionally one screenshot as "how I run structured research") | NER-specific content is irrelevant to the new task. |
| `Project3 (Group)/Starter Code/notebooks/*.ipynb`, `NLP/memm_features.py` | DISCARD (archive) | MEMM code is dead-end tech for this lab. |
| `Project Task2` + `Project3` **datasets** (Reuters JSON, CoNLL CSVs) | **DISCARD — never redistribute** | Research-only licenses. The new lab must not ship them. |
| `Project Task1/` entirely | DISCARD | No reusable assets beyond generic scraping experience. |

---

## 4. Target architecture and tech choices

### 4.1 System overview

```
CFPB complaints.csv.zip  ──▶  ingest (DuckDB) ──▶ harmonize taxonomy ──▶ dedup ──▶ temporal splits
                                                                                       │
                        ┌──────────────────────────────────────────────────────────────┤
                        ▼                          ▼                          ▼
                 Tier A: TF-IDF +           Tier B: fine-tuned         Tier C: LLM few-shot
                 linear model                transformer               (Claude Haiku 4.5)
                 (calibrated)                (calibrated)              (structured output)
                        │                          │                          │
                        └──────────────┬───────────┴──────────────────────────┘
                                       ▼
                          Unified eval harness (YAML-config runs, append-only results log)
                          metrics: macro-F1, per-class F1, ECE, Brier, AURC, $/1k, p50/p95 latency
                          all with bootstrap 95% CIs
                                       ▼
                          Router: confidence cascade A→B→C→human, thresholds optimized
                          against an explicit business cost model
                                       ▼
                          Drift protocol: rolling yearly eval 2022→2026, prior-shift and
                          OOV tracking, post-LLM-cutoff contamination-safe slice
                                       ▼
                          Static demo (precomputed JSON + in-browser ONNX) + case study
```

### 4.2 Tier choices and rationale

**Tier A — classical baseline: TF-IDF (word 1–2-grams + char 3–5-grams) → Logistic Regression** (sklearn, SAGA, class-weighted), with **Complement NB** as a secondary point (the explicit heir of the Task2 seed). Rationale: linear TF-IDF is the honest strong baseline every serious eval paper uses; it is nearly free at inference (<1 ms, ~$0), which anchors the cost axis of the frontier. Calibrate with isotonic regression on the calibration split.

**Tier B — fine-tuned transformers, two required frontier points.** **B1: `answerdotai/ModernBERT-base` (149M, Apache-2.0) — the headline accuracy model** and the job-market-signal choice (current-generation encoder, shows the candidate tracks the field, not 2019 tutorials). **B2: `distilbert-base-uncased` (66M) — the deployment point**: int8-quantized ONNX (~65 MB) runs **in the browser via transformers.js**, powering the live demo (rhymes with Privacy Preflight's browser-local story) and anchoring a cheap-inference point on the frontier. Reporting both turns Tier B into its own mini cost-quality trade-off — a stronger story than one model. Training: HF `Trainer`/`accelerate`, fp16/bf16, 2–4 epochs on up to ~300k narratives, 3 seeds for the headline (B1) config, 1–3 seeds for B2. Runs on Apple-silicon MPS (hours) or a free/cheap cloud T4/A10 (Colab or a rented GPU — either is fine; record which, it goes in "How this was verified"). Calibrate both with temperature scaling.

**Tier C — LLM few-shot, two required frontier points.** **C1: Claude Haiku 4.5 (`claude-haiku-4-5-20251001`)** with a structured-output/tool-use schema forcing one of the harmonized labels, k≈10 few-shot exemplars, fixed prompt hash — the realistic "just call an API" triage candidate. **C2: Claude Sonnet 5 (`claude-sonnet-5`)** on a budgeted subsample (same prompt, same schema) to extend the frontier's upper-right corner and answer the question a hiring manager will actually ask: *"does paying ~4× more per token buy anything on this task?"* Rationale: measuring both honestly (including where they lose to a $0.0001 linear model) *is the story*. Confidence signal for routing: self-reported confidence is untrustworthy → use (a) logit-free proxy: agreement between Tier C and Tier B, and (b) for router purposes, primarily use Tiers A/B calibrated confidence to decide *whether to escalate to* C; report Tier C selective accuracy via answer-consistency (3-sample self-consistency on the escalated subset only, cost-tracked).

> **AMENDMENT (2026-08-07, owner-approved, evidence-based).** All Tier C **finals run zero-shot
> (`num_exemplars: 0`)**, for both C1 (Haiku 4.5) and C2 (Sonnet 5) — the two models must share
> the prompt config so the Haiku-vs-Sonnet comparison stays paired and clean. Basis: the CAL
> ablation (commit `84d79d6`, EXPERIMENT_LOG 2026-08-07; runs `c7598f84…` vs `3f310951…`,
> n=1,500 paired) found no few-shot gain — paired few−zero deltas accuracy **+0.0053
> [−0.0060, +0.0167]**, macro-F1 **+0.0107 [−0.0054, +0.0266]**, McNemar p=0.41 — at 2× the
> per-call cost. The few-shot configuration is **archived, not deleted**: the frozen v1 bundle
> (`prompts/tier_c/v1/`, bundle `f6777a96…`) retains `exemplars.json` and the `num_exemplars`
> parameter, so a future probe of whether exemplars matter on the 2025/2026 drift slices needs
> only a config flip. "k≈10 few-shot" above is superseded for finals accordingly.

**Router — the centerpiece.** Confidence cascade: Tier A answers if `p_max ≥ τ_A`; else escalate to Tier B; if `p_max ≥ τ_B` answer; else escalate to Tier C; below `τ_C`-proxy → human queue. Thresholds (τ_A, τ_B) chosen on the calibration split to minimize expected cost under an explicit, parameterized business cost model: `cost = c_misroute · P(error) + c_api · E[tokens] + c_human · P(human)` with defaults documented and user-adjustable in the demo (e.g., misroute = $6 handling delay, human review = $2.50, API cost measured). Deliverable claim shape: *"At equal accuracy to the all-LLM policy, the router cuts cost per 1,000 complaints by X% (measured); at equal cost to the all-linear policy, it raises macro-F1 by Y points (measured)."*

### 4.3 Engineering stack

- **Python 3.12, `uv` for locked environments.** Repo layout: `data/` (gitignored) · `src/triage_lab/` (ingest, models, eval, router) · `configs/*.yaml` (one per experiment) · `results/` (append-only JSONL run log, committed) · `EXPERIMENT_LOG.md` (dated, hypothesis→result→verdict format copied from the NER session summary) · `demo/` (static site) · `seeds/` (ported coursework exhibits).
- **DuckDB** for ingest/splitting (portfolio consistency with the DE case studies).
- **Eval harness**: every run = one YAML (model, data slice, seed, prompt hash) → one JSONL record (metrics + CIs + git SHA + dataset snapshot hash + wall-clock + cost). A `make reproduce-headline` target re-runs the headline eval end-to-end on the frozen snapshot.
- **Testing/CI**: pytest for harness logic (metric math vs sklearn reference, bootstrap determinism, router simulator); a GitHub Actions smoke job that runs Tier A end-to-end on a 2k-row fixture.

---

## 5. Dataset and licensing plan

**Primary: CFPB Consumer Complaint Database.**
- Source: https://www.consumerfinance.gov/data-research/consumer-complaints/ — bulk CSV at `https://files.consumerfinance.gov/ccdb/complaints.csv.zip` (multi-GB; also a Socrata API). Record the download date and SHA-256 of the snapshot; freeze it.
- License: **U.S. Government work — public domain (17 U.S.C. §105).** Redistribution of samples in the demo is safe. Narratives are opt-in and already scrubbed by CFPB (PII masked as `XXXX`) — note this in the datasheet; do not claim the lab performed redaction.
- Task: predict **Product** (the routing target) from `Consumer complaint narrative`. Narratives exist from 2015-06 onward; millions of rows through 2026.
- **Taxonomy harmonization (required, and a feature of the story):** CFPB renamed/merged product categories over the years (notably the 2023-04 consolidation of credit-reporting categories). Build an explicit `taxonomy_map.yaml` collapsing historical products into ~9–11 stable routing classes; document every mapping decision in the datasheet. The taxonomy change itself becomes a measured "label drift" exhibit.
- **Dedup (required):** CFPB narratives contain mass near-duplicate template complaints (credit-report disputes especially). Near-dup removal via MinHash-LSH (datasketch) or embedding-cosine threshold; report the dedup rate; dedup **before** splitting to kill train/test leakage.
- **Class imbalance:** credit-reporting complaints dominate recent years (a measured *prior shift* — plot it; it is part of the drift chapter). Use class weights / stratified subsampling for training; never for test sets (test must reflect reality).

**Splits (temporal, the drift protocol's backbone):**
- TRAIN: 2015-07 → 2021-12 (subsample to a budgeted size, e.g. 300k, stratified).
- CAL: 2022-H1 (calibration + router threshold fitting).
- TEST-IID: 2022-H2 (the "in-period" headline test).
- TEST-DRIFT: rolling yearly slices 2023, 2024, 2025, 2026-H1 (frozen, ~20k each, never touched during development).
- TEST-POSTCUTOFF: complaints received after Tier C's model knowledge cutoff (verify the exact cutoff for `claude-haiku-4-5-20251001` at implementation time and document it) — the contamination-safe LLM slice.

**Fallback / robustness extension (only if CFPB fails unexpectedly):** Banking77 (CC-BY-4.0) + CLINC150 (CC-BY-3.0, includes out-of-scope class). If used as fallback, drift becomes synthetic-only — accept the weaker evidence class and label it accordingly.

---

## 6. Evaluation design

### 6.1 Metrics (all reported with 95% bootstrap CIs, n=1,000 resamples, fixed seed)

| Axis | Metrics |
|---|---|
| Quality | macro-F1 (headline), per-class F1, balanced accuracy, confusion matrices |
| Calibration | ECE (15-bin), Brier score, reliability diagrams (per tier, pre/post calibration) |
| Selective prediction | risk-coverage curves, AURC, accuracy@coverage∈{50,80,90,95%} |
| Cost | measured $ per 1,000 complaints per policy (API receipts for Tier C; amortized compute note for A/B) |
| Latency | p50/p95 per tier; method documented (n≥200 timed calls, warm vs cold stated, hardware/API region stated) |
| Comparisons | paired bootstrap deltas between tiers/policies; McNemar's test on headline pairs; a difference is only claimed where CI excludes zero |

### 6.2 Baselines and reference points

Majority-class and random-stratified floors; Tier A ComplementNB (seed homage); Tier A LogReg; Tier B1 ModernBERT-base (3 seeds, report mean±sd — headline); Tier B2 DistilBERT (deployment point); Tier C1 Haiku 4.5 zero-shot and few-shot (prompt ablation: 0 vs 10 exemplars); Tier C2 Sonnet 5 few-shot (subsample); router policies (A-only, B-only, C-only, A→B, A→B→C, cost-optimal).

### 6.3 Drift / robustness protocol

1. **Temporal degradation (measured):** every tier + router evaluated on each rolling yearly slice; plot macro-F1 and ECE vs time; annotate the 2023 taxonomy event. Key question the chart answers: *which tier degrades slowest, and does the router's escalation rate self-adjust as confidence drops?* (Report escalation-rate-over-time — an underrated systems insight.)
2. **Prior shift decomposition (measured):** show how much yearly degradation is explained by class-mix change alone (reweighted-F1 counterfactual) vs within-class drift.
3. **OOV / covariate tracking (measured):** yearly OOV rate against the TRAIN vocabulary and embedding-centroid distance — the grown-up version of the CoNLL dev/test OOV finding in the seed.
4. **Perturbation robustness (measured, cut-line candidate):** typo/OCR-noise/case-mangling perturbations at fixed rates on TEST-IID; report per-tier deltas. (Char-n-gram TF-IDF vs subword vs LLM is a genuinely interesting comparison here.)
5. **Contamination defense (measured):** Tier C on TEST-POSTCUTOFF vs TEST-IID; report the delta explicitly, whatever it shows.
6. **Novel-class probe (measured, stretch):** hold one small product class out of TRAIN entirely; measure where its complaints land and whether calibrated confidence flags them (link to selective-prediction story).

### 6.4 Statistical honesty rules (inherited from the seed, now enforced)

Single-variable experiment discipline (one config delta per run, logged in `EXPERIMENT_LOG.md` as hypothesis → result → verdict, including failed hypotheses); no headline claim without a CI; dev/test hygiene (TEST-* slices touched only for final reported runs; iteration happens on CAL).

---

## 7. Interactive demo spec

Static, self-contained, hostable alongside the existing site. No server. Precomputed JSON from the results log + in-browser ONNX for live inference.

1. **Triage playground.** User picks a real sample complaint (from the redistributable CFPB set) or pastes text. Tier A (TF-IDF+LogReg compiled to ONNX or reimplemented in JS — it is small) and Tier B2 (DistilBERT int8 ONNX via transformers.js, lazy-loaded with a size warning) run **live in the browser**; Tier B1 (ModernBERT) and both Tier C models show precomputed responses for the curated sample set (N≈200), with an optional bring-your-own-key mode for Tier C. Display: per-tier label + calibrated confidence + cost + latency, then the **router's decision path animated** (answered at A / escalated to B / escalated to C / sent to human).
2. **Cost-quality frontier.** Interactive scatter: x = $ per 1k complaints (log), y = macro-F1, one point per tier/policy with CI whiskers; router policies visibly dominating the single-tier points.
3. **Router policy builder.** Sliders for `c_misroute`, `c_human`, API price; the demo re-solves thresholds from precomputed risk-coverage tables and live-updates the frontier point, escalation mix, and expected cost. This is the "makes them want to talk" widget.
4. **Drift timeline.** Per-tier macro-F1 by year 2022→2026 with the taxonomy-change annotation and escalation-rate overlay.
5. **Calibration panel.** Reliability diagrams pre/post calibration, per tier.
6. **Receipts drawer.** Every displayed number links to its results-log record (run config hash, dataset snapshot hash, git SHA).

---

## 8. Phased work plan with acceptance criteria

**Phase 0 — Repo + data engineering (est. 2–3 days).** New repo, uv lock, CI skeleton. Download + freeze CFPB snapshot (hash recorded). DuckDB ingest; taxonomy harmonization map; MinHash dedup; temporal splits materialized; datasheet written (source, license, scrubbing note, dedup rate, class×year matrix).
✅ *Accept:* `make data` reproduces splits from the frozen snapshot byte-identically; datasheet complete; leakage check (no near-dup pairs across split boundaries at the chosen threshold) passes in CI on a fixture.

**Phase 1 — Eval harness + Tier A (est. 2–3 days).** Harness (YAML→JSONL, bootstrap CIs, paired tests), metric unit tests vs sklearn reference; TF-IDF LogReg + ComplementNB trained, calibrated, evaluated on TEST-IID.
✅ *Accept:* harness tests green; Tier A macro-F1 with CI on TEST-IID recorded in results log; `EXPERIMENT_LOG.md` begun with ≥3 logged single-variable runs.

**Phase 2 — Tier B fine-tuning (est. 4–6 days).** ModernBERT-base fine-tune (3 seeds, headline) and DistilBERT fine-tune (deployment point), temperature scaling, TEST-IID eval; training curves + configs archived. Export DistilBERT int8 ONNX and verify parity (agreement ≥99% with the PyTorch model on 5k samples).
✅ *Accept:* both Tier B points evaluated with CIs; B1 beats Tier A on macro-F1 with CI excluding zero (expected; if not, that *is* the finding — report it); B1-vs-B2 delta reported (the intra-tier trade-off exhibit); seed variance reported; ONNX parity check logged.

**Phase 3 — Tier C LLM (est. 3–4 days).** Prompt + structured-output schema (versioned, hashed); zero-shot vs few-shot ablation on CAL (Haiku); Haiku 4.5 full eval on TEST-IID subsample (size budgeted, CI-honest) + TEST-POSTCUTOFF; Sonnet 5 few-shot on a smaller budgeted subsample of the same slices (paired sampling so the Haiku-vs-Sonnet delta gets a paired CI); raw API usage logs retained as cost receipts.
✅ *Accept:* both Tier C points with CIs on both slices; measured $/1k and p50/p95 latency per model; Haiku-vs-Sonnet paired delta reported; contamination delta reported.

**Phase 4 — Calibration + router (est. 3–4 days).** Risk-coverage machinery; cost model; threshold optimization on CAL; router simulator evaluated on TEST-IID against all single-tier and naive-cascade policies.
✅ *Accept:* the two headline frontier claims (§4.2) computed with CIs; router dominates ≥2 single-tier policies on the frontier (if it doesn't, diagnose and report honestly — the lab's credibility rule).

**Phase 5 — Drift protocol (est. 3–4 days).** Rolling yearly evals for all tiers + router; prior-shift decomposition; OOV tracking; perturbation suite; novel-class probe (stretch).
✅ *Accept:* drift chapter charts rendered from the results log; escalation-rate-over-time computed; every chart's evidence class labeled.

**Phase 6 — Demo + case study (est. 4–6 days).** Static demo per §7; case study page with "How this was verified" and "What this does not prove" (content in §9); provenance section linking the coursework seeds; `make reproduce-headline` verified on a clean machine.
✅ *Accept:* demo runs fully offline/static; every displayed number traces to a results-log record; reproduce-headline succeeds from the frozen snapshot.

**Phase 7 (optional stretch, only if everything above ships) — PII cross-check tie-in.** Small exhibit: run Privacy Preflight-style detection over CFPB's pre-scrubbed narratives to measure residual-PII rate on a sample; links the two case studies. Cut freely.

Total realistic effort: **~4 weeks part-time** (the full-signal configuration — both Tier B models and both Tier C models — is the default, per the owner's explicit preference for job-market signal over minimal scope).

---

## 9. Site evidence sections (write these from the artifacts above)

**"How this was verified" must include:** frozen dataset snapshot (URL, download date, SHA-256) + deterministic `make data`; locked environment (uv lock, hardware/GPU documented); one-YAML-per-run configs and append-only JSONL results with git SHAs; bootstrap CIs on every headline number, paired tests for every comparison claim; raw LLM API usage logs as cost receipts; latency methodology (n, warm/cold, region/hardware); ONNX-vs-PyTorch parity check for the in-browser model; CI smoke eval; `make reproduce-headline` instructions for third parties; dev/test hygiene statement (TEST-* frozen, iteration on CAL only) with `EXPERIMENT_LOG.md` published including failed hypotheses.

**"What this does not prove" must include:** not a production serving system (no SLA/throughput/uptime claims; latency is bench-measured, not fleet-measured); single-domain, English-only finance complaints — generalization to other triage domains **not demonstrated**; Tier C costs are point-in-time prices for specific models — forward cost claims are **projected**; business cost-model parameters (misroute/human costs) are **estimated** defaults, with sensitivity exposed in the demo rather than asserted; possible LLM contamination on pre-cutoff data — mitigated but not eliminated by the post-cutoff slice; CFPB narratives are opt-in and CFPB-scrubbed — selection bias vs the full complaint population, and scrubbing artifacts (`XXXX`) are in-distribution quirks; no online learning / human-in-the-loop feedback measured; **coursework seed scores (NB 0.965, NER 0.8136→0.8733, Kaggle 0.83615) are self-reported class results — provenance, not evidence**; the group NER project was collaborative and role attribution is biographical context, not a measured claim.

---

## 10. Risks and cut-lines

| Risk | Mitigation |
|---|---|
| CFPB bulk file size / schema surprises | Ingest via DuckDB streaming; work from a narratives-only projection; freeze early. |
| Extreme class skew (credit-reporting share) makes macro-F1 volatile for tiny classes | Harmonized taxonomy keeps classes ≥ a support floor; CIs surface volatility honestly; consider merging sub-floor classes and documenting it. |
| LLM eval cost balloons | Budget the Tier C sample (e.g., 5–10k TEST-IID + full POSTCUTOFF slice); CIs make sampling defensible. |
| Fine-tuning compute limits | Both Tier B models with a 300k-row cap train on MPS/T4 in hours each; if compute is truly blocked, drop to 1 seed for B2 before dropping the model. |
| Router fails to dominate single tiers | Still publishable — the frontier chart and honest diagnosis are the product; the site's credibility model rewards this. |
| Taxonomy harmonization judgment calls attacked | Every mapping in `taxonomy_map.yaml` with rationale; sensitivity run with the alternative mapping (stretch). |
| Model deprecation (Tier C) before publication | Record model ID + date; the frontier is a method, not a leaderboard — state this. |

**Cut order if scope must shrink (cut from the top).** The owner has explicitly chosen the full-signal configuration, so the second frontier points (ModernBERT B1, Sonnet C2) are **core scope, not stretch** — cut them only as a last resort, after everything else below: 1) Phase 7 PII tie-in → 2) novel-class probe → 3) perturbation suite → 4) in-browser Tier B (fall back to precomputed-only demo; keep live Tier A, it's tiny) → 5) prior-shift decomposition (keep the raw drift curves) → 6) reduce Sonnet C2 to a CAL-only exhibit and B1 to a single seed (still report both). **Never cut:** the three tiers, temporal drift curves, calibration, the router + frontier chart, CIs, and the two verification sections — they are the identity of the piece.

---

## Appendix A — Key numbers from the seeds (for the provenance section)

- Naive Bayes (Reuters 5-class, from scratch, solo): macro-F1 0.96477 (10k features), sweep 0.96367–0.96585; 5,787 train / 2,298 test docs. Source: `Project Task2/Starter Code/report.md`.
- MEMM NER (CoNLL-2003, group; user-led modeling): starter baseline dev macro-F1 0.8136 → 0.8733 via 4 documented feature phases; 5 Kaggle submissions 0.82335–0.83615 (best = v5); key documented lessons: convergence (MAX_ITER) as the dominant lever, harmful-feature removal (`no_gaz`, `pl|s3`), train+dev > train-only, ~0.04 structural dev→test drift from temporal OOV. Sources: `Project3 (Group)/NER_MEMM_Project_Session_Summary.md`, `NER Score.docx`, `Starter Code/notebooks/ner-phase4-v7.ipynb` (docstring confirms 0.83615), `Pre/Speaker_Notes_and_QA-2.md`.
- Neither seed's dataset may be redistributed (Reuters / CoNLL-2003 research licenses). The new lab's data (CFPB) is U.S. public domain.
