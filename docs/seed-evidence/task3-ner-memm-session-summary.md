# CoNLL-2003 NER MEMM Project — Complete Session Summary

## 1. Task Definition

```yaml
Task: Token-level Named Entity Recognition (NER)
Dataset: CoNLL-2003 English news text
Model: MEMM (Maximum Entropy Markov Model) with NLTK MaxentClassifier
Decoding: Greedy (left-to-right, no Viterbi)
Labels: [B-LOC, I-LOC, B-ORG, I-ORG, B-PER, I-PER, B-MISC, I-MISC, O]
Metric: Macro F1 over all 9 labels
Constraint: Only modify features(), __init__, and helper functions. Model architecture/training loop/decoding logic are FIXED.
Platform: Kaggle (no internet at runtime, 16GB RAM, 6hr time limit)
Data Path: /kaggle/input/mem-classification/
```

## 2. Data Profile

```yaml
Train: 204,567 tokens, ~14,000 sentences
Dev: 51,578 tokens, ~3,400 sentences (official CoNLL-2003 dev split)
Test: 46,666 tokens, ~3,684 sentences (official CoNLL-2003 test split)
NaN tokens: train=6, dev=5, test=8 (must handle with pd.isna())
Special tokens: "-DOCSTART-" marks document boundaries
Label distribution (train): O=170524 (83.4%), B-PER=6600, B-LOC=7140, B-ORG=6321, B-MISC=3438, I-PER=4528, I-ORG=3704, I-LOC=1157, I-MISC=1155
Class imbalance: O dominates; I-LOC and I-MISC are severely underrepresented
Submission format: CSV with columns [id, label], 46666 rows, ids 0-46665
```

## 3. Key Technical Insight: CoNLL-2003 Dev vs Test Gap

```
CRITICAL FINDING: Dev set and test set come from DIFFERENT time periods of news.
Test set has significantly higher OOV (out-of-vocabulary) rate than dev set.
Empirically observed gap: dev macro F1 ~ 0.87 → test public score ~ 0.83-0.84
This ~0.04 gap is a KNOWN property of CoNLL-2003, NOT a bug in our features.
MEMM + greedy decoding theoretical ceiling: dev ~0.92-0.93, test ~0.87-0.89
```

## 4. Development Phases and Results

### Phase 1: Tier 1 Features Only
```yaml
Dev Macro F1: 0.8322
MAX_ITER: 10
Features implemented:
  - Word identity: raw (w=) + lowercased (wl=)
  - Orthographic flags: is_title, is_upper, is_lower, is_mixed
  - Collapsed word shape (shape=)
  - Suffixes/Prefixes length 2,3,4
  - Digit/punct flags: has_digit, all_digit, has_hyphen, has_period
  - Context ±1: raw word, lowercased, shape, titlecase flag
  - Context ±2: lowercased, shape (reduced per Conflict #7 in reference doc)
  - prev_label (raw) + prev_label × word_shape conjunction
  - BOS / EOS markers
Key observations:
  - Precision very high (0.93) but recall low (0.76) → model too conservative
  - I-ORG recall 0.55, I-MISC recall 0.55 → worst performers
  - LEICESTERSHIRE (all-caps BOS) missed as B-ORG → BOS+uppercase bias toward O
```

### Phase 2: +Tier 2 Features
```yaml
Dev Macro F1: 0.8535 (+0.021 over Phase 1)
MAX_ITER: 15 (increased from 10 due to larger feature space)
New features added:
  - Context titlecase flags ±2
  - Bigram features (raw words): bi=w-1|w0, bi=w0|w+1
  - prev_label × is_title conjunction
  - prev_label × is_upper conjunction
  - Coarse prev label: prev_B, prev_I, prev_O + entity type (pBt=, pIt=)
  - Training-data gazetteers (PER/LOC/ORG/MISC sets + ambiguity flag + no_gaz)
  - Word frequency bucket (rare/uncommon/common/very_common)
  - PER trigger words (Mr., Dr., President, ...)
  - LOC trigger pattern (spatial preposition + titlecase)
  - ORG suffix triggers (Inc., Corp., Ltd., ...)
  - BOS + titlecase interaction (BOS_tit)
  - Word length bucket (wlen=s/m/l)
Key observations:
  - LEICESTERSHIRE now correctly recognized (gazetteers working)
  - I-ORG recall 0.55→0.61, overall recall 0.76→0.79
  - Still precision >> recall pattern
```

### Phase 3 (Original): +Tier 3+4 Features
```yaml
Dev Macro F1: 0.8553 (+0.002 over Phase 2, negligible)
MAX_ITER: 15
New features added:
  - DOCSTART fast exit
  - Capitalization run (crun2/3/4)
  - prev_label × suffix_3 (pl|s3=)
  - has_apostrophe, is_punct
  - TitleCase mid-sentence (tit_mid)
  - Sentence length bucket (slen=)
  - Context gazetteers ±1
  - Lookahead proxy-tags (ptag+1, ptag+2)
  - Entity-internal function words
Problem diagnosed: MAX_ITER=15 insufficient for larger feature space + some features harmful
```

### Phase 3 (Optimized): Feature Pruning + More Iterations
```yaml
Dev Macro F1: 0.8733 (+0.018 over Phase 3 original, +0.020 over Phase 2)
MAX_ITER: 25 (increased from 15 — THIS WAS THE BIGGEST SINGLE IMPROVEMENT)
Changes from Phase 3 original:
  REMOVED: no_gaz (reinforced O-class on OOV entities)
  REMOVED: pl|s3 (prev_label × suffix_3, too sparse: 9 labels × thousands suffixes)
  ADDED: pl|freq (prev_label × freq_bucket, only 9×4=36 combinations, low sparsity)
  ADDED: pl|gaz + pl_gaz_match (prev_label × gazetteer match for I-tag continuation)
  ADDED: reporting verb features (said/told/says signal entity boundary)
Key insight: MAX_ITER 15→25 was the dominant factor, not feature changes.
  The model simply hadn't converged at 15 iterations with ~60 features per token.
Per-label improvements:
  - I-ORG recall: 0.62→0.71 (+9%, largest single-label gain)
  - B-PER recall: 0.86→0.89
  - I-PER recall: 0.93→0.95
  - B-ORG recall: 0.78→0.82
```

## 5. Submission Experiments on Test Set

### v1 (Phase 3 Optimized features, train+dev, iter=25)
```yaml
Public Score: 0.83522 ← BEST
Training data: train + dev combined (256K tokens)
MAX_ITER: 25
Features: Phase 3 Optimized full set (including proxy-tags)
Train accuracy at convergence: 0.998
Training time: 160.8 min
Note: First submission attempt with train+dev iter=35 FAILED (OOM after 4hrs)
  → Fixed by removing redundant dev evaluation step and adding gc.collect()
```

### v2 (Phase 3 Optimized features, train-only, iter=20)
```yaml
Public Score: 0.82335 ← WORST
Training data: train only (204K tokens)
MAX_ITER: 20
Hypothesis tested: "v1 overfit because train+dev bloated gazetteers"
Result: HYPOTHESIS WRONG. Less data + fewer iterations = worse, not better.
Key learning: train+dev > train-only on public test
```

### v3 (Reduced sparsity features, train+dev, iter=30)
```yaml
Public Score: 0.83122
Training data: train + dev combined
MAX_ITER: 30
Changes from v1:
  REMOVED: raw context words w±1= (kept lowercased only)
  CHANGED: bigrams to use lowercased words
Result: Score dropped 0.004 vs v1
Key learning: Raw context word identity (case-sensitive) IS valuable, don't remove
```

### v4 (v1 features minus proxy-tags, train+dev, iter=25)
```yaml
Public Score: 0.83239
Training data: train + dev combined
MAX_ITER: 25
Only change from v1: REMOVED proxy-tag features (ptag+1, ptag+2)
Rationale: proxy-tags default to "O" for OOV words → hurts entity recall on test
Result: Score dropped 0.003 vs v1, but may be better on private split if more OOV
```

### Summary Table
```
| Version | Train Data  | Iter | Key Change vs v1           | Public Score |
|---------|-------------|------|----------------------------|--------------|
| v1      | train+dev   | 25   | baseline (best)            | 0.83522      |
| v2      | train-only  | 20   | less data, fewer iter      | 0.82335      |
| v3      | train+dev   | 30   | no raw w±1, lower bigrams  | 0.83122      |
| v4      | train+dev   | 25   | no proxy-tags              | 0.83239      |
```

## 6. Final Feature Set (v1 — Best Submission)

```python
"""
Complete feature inventory for the best-performing model (v1, public=0.8352):

BLOCK 1 — Word Identity [T1, 5/5 consensus]:
  w={raw_word}                    # case-sensitive current word
  wl={lowercased_word}            # case-insensitive current word

BLOCK 2 — Orthographic Flags [T1, 5/5]:
  is_title                        # first char upper, rest not all upper
  is_upper                        # all uppercase, length > 1
  is_lower                        # all lowercase
  is_mixed                        # has alpha but not title/upper/lower

BLOCK 3 — Word Shape [T1, 5/5]:
  shape={collapsed_shape}         # e.g., "Washington"→"Xx", "U.S."→"X.X.", "2024"→"d"

BLOCK 4 — Affixes [T1, 5/5]:
  suf2={last_2_chars}             # suffix length 2
  suf3={last_3_chars}             # suffix length 3
  suf4={last_4_chars}             # suffix length 4
  pre2={first_2_chars}            # prefix length 2
  pre3={first_3_chars}            # prefix length 3
  pre4={first_4_chars}            # prefix length 4

BLOCK 5 — Digit/Punct Flags [T1+T3]:
  has_digit                       # contains any digit
  all_digit                       # entirely digits
  has_hyphen                      # contains '-'
  has_period                      # contains '.'
  has_apos                        # contains "'" [T3]
  is_punct                        # non-alphanumeric only [T3]

BLOCK 6 — Context Window [T1+T2]:
  # ±1 positions: RICH features
  w{±1}={raw_word}                # raw identity (case-sensitive)
  wl{±1}={lowercased}             # lowercased identity
  sh{±1}={shape}                  # collapsed shape
  tit{±1}                         # titlecase flag (if applicable)
  # ±2 positions: REDUCED features
  wl{±2}={lowercased}             # lowercased only
  sh{±2}={shape}                  # shape
  tit{±2}                         # titlecase flag [T2]

BLOCK 7 — Bigrams [T2, 4/5]:
  bi={w-1}|{w0}                   # left bigram (raw words)
  bi={w0}|{w+1}                   # right bigram (raw words)

BLOCK 8 — Previous Label + Conjunctions [T1+T2]:
  prev_label={label}              # raw previous label
  pl|sh={prev_label}|{shape}      # × word shape [T1, 4/5]
  pl|tit={prev_label}             # × is_title [T2, 3/5]
  pl|up={prev_label}              # × is_upper [T2, 2/5]
  prev_B / prev_I / prev_O        # coarse transition type [T2]
  pBt={entity_type}               # previous B-entity type [T2]
  pIt={entity_type}               # previous I-entity type [T2]
  pl|freq={prev_label}|{bucket}   # × frequency bucket [NEW, 36 combos]

BLOCK 9 — Gazetteers [T2+T3]:
  gaz_{PER|LOC|ORG|MISC}          # word in entity-type gazetteer
  in_gaz                          # in any gazetteer
  gaz_amb                         # in multiple gazetteers
  cgaz{±1}_{type}                 # context word ±1 in gazetteer [T3]
  freq={r|u|c|v}                  # word frequency bucket [T2]
  pl_gaz_match                    # prev entity type matches current gaz [NEW]
  pl|gaz={prev_label}|{type}      # prev_label × gazetteer type [NEW]
  # NOTE: no_gaz was REMOVED (reinforced O-bias on OOV)

BLOCK 10 — Trigger Words [T2]:
  per_trig                        # prev word in PER trigger set
  loc_trig                        # spatial preposition + titlecase
  org_trig_n                      # next word in ORG suffix set
  org_suf_w                       # current word in ORG suffix set
  next_report_verb                # next word is said/told/says/... [NEW]
  after_report_verb_tit           # prev word is report verb + current titlecase [NEW]

BLOCK 11 — Positional [T1+T2+T3]:
  BOS                             # sentence start
  BOS_tit                         # BOS + titlecase [T2]
  EOS                             # sentence end
  tit_mid                         # titlecase mid-sentence (pos>0) [T3]
  wlen={s|m|l}                    # word length bucket [T2]
  slen={vs|s|m|l}                 # sentence length bucket [T3]

BLOCK 12 — Capitalization Run [T3]:
  crun2                           # 2+ consecutive titlecase words
  crun3                           # 3+ consecutive titlecase words
  crun4                           # 4+ consecutive titlecase words

BLOCK 13 — Lookahead Proxy-Tags [T4]:
  ptag{+1}={predicted_label}      # most common training label for w+1
  ptag{+2}={predicted_label}      # most common training label for w+2
  # NOTE: defaults to "O" for OOV words → may hurt test generalization

BLOCK 14 — Entity-Internal Function Words [T4]:
  ent_func                        # "of"/"and"/"the"/etc. after B-/I- label
  ent_func|{entity_type}          # × entity type

SPECIAL: DOCSTART [T3]:
  When token == "-DOCSTART-": emit only {w=, wl=, is_docstart, prev_label} and RETURN
"""
```

## 7. Critical Lessons Learned

### 7.1 MAX_ITER is the Most Important Hyperparameter
```
iter=10-15: Insufficient for ~60 features. Model underfits.
iter=20-25: Sweet spot for dev evaluation. 
iter=25: Best test score achieved (v1).
iter=30: Marginal or negative returns (v3 score dropped).
Key insight: When adding features, ALWAYS increase iterations proportionally.
  Phase 1 (31 features): iter=10 OK
  Phase 2 (50 features): iter=15 needed
  Phase 3 (60 features): iter=25 needed
```

### 7.2 Feature Removal Matters More Than Feature Addition
```
HARMFUL features identified and removed:
  - no_gaz: Told model "this word isn't in any gazetteer" → reinforced O for OOV entities
  - pl|s3 (prev_label × suffix_3): 9 × thousands = extreme sparsity, no learnable signal

NEUTRAL features (no measurable impact):
  - Capitalization run (crun2/3/4)
  - Sentence length bucket
  - Position quartile (never implemented, 1/5 consensus)

VALUABLE features confirmed:
  - Raw context words w±1= (removing them in v3 hurt score)
  - Raw bigrams (lowercasing them in v3 hurt score)
  - Gazetteers + ambiguity (LEICESTERSHIRE fix)
  - pl|freq (low-sparsity prev_label interaction)
```

### 7.3 Train+Dev > Train-Only for Final Submission
```
Empirically proven: v1 (train+dev, 0.8352) > v2 (train-only, 0.8234)
The extra 25% data from dev helps more than it hurts via overfitting.
```

### 7.4 Kaggle Infrastructure Gotchas
```
- OOM crash: Training TWO models sequentially without gc.collect() exceeded 16GB
  Fix: Single training run + aggressive gc.collect() after every major step
- GPU not needed: NLTK MaxentClassifier is CPU-only. Select "No Accelerator" for more RAM.
- Time limit: 6 hours. iter=25 on 256K samples ≈ 160 min (safe). iter=35 ≈ risky.
```

### 7.5 MEMM Structural Limitations
```
- Label bias problem: Greedy decoding means once O is predicted, I-tags can't follow
- Recall << Precision pattern: Model is systematically conservative (precision ~0.94, recall ~0.83)
- CoNLL-2003 test gap: ~0.04 F1 drop from dev→test is structural, not fixable with features
- To go beyond ~0.85 on test: Need CRF (Viterbi decoding) or neural models
```

## 8. Final Submission Selection

```yaml
Primary submission: v1 (public score: 0.83522)
  Config: train+dev, MAX_ITER=25, full Phase 3 Optimized features including proxy-tags
  Rationale: Highest observed public score

Secondary submission: v4 (public score: 0.83239)
  Config: train+dev, MAX_ITER=25, same as v1 but WITHOUT proxy-tags
  Rationale: Better OOV robustness; may outperform v1 if private split has higher OOV rate

Selection logic: v1 and v4 differ by exactly ONE feature (proxy-tags).
  v1 = better for known words, v4 = better for unknown words.
  Together they hedge against unknown private split characteristics.
```

## 9. Files Inventory

```yaml
Reference documents:
  - NER_Feature_Engineering_Consolidated_Reference.md  # Master feature guide (5 AI sources synthesized)
  - competition_text.md                                 # Task description
  - Project_3.docx                                      # Project constraints
  - NER-starter-code.ipynb                              # Baseline MEMM code
  - memm_features.py                                    # Prior research implementation

Data files:
  - train.csv (204,567 rows: sentence_id, token_idx, token, label)
  - dev.csv   (51,578 rows: same schema)
  - test.csv  (46,666 rows: id, sentence_id, token_idx, token)
  - sample_submission.csv (46,666 rows: id, label)

Code outputs:
  - phase1_notebook.py      # Tier 1 features, dev eval
  - phase2_notebook.py      # +Tier 2 features, dev eval
  - phase3_notebook.py      # +Tier 3/4 features (original, before optimization)
  - phase3_optimized.py     # Optimized: removed harmful features, iter=25
  - phase4_final.py         # FAILED: OOM (trained 2 models)
  - phase4_streamlined.py   # v1: train+dev, iter=25 → public 0.8352
  - phase4_v2_train_only.py # v2: train-only, iter=20 → public 0.8234
  - phase4_v3_final.py      # v3: reduced sparsity, iter=30 → public 0.8312
  - phase4_v4.py            # v4: v1 minus proxy-tags → public 0.8324
```
