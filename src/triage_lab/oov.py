"""OOV / covariate tracking against the frozen TRAIN vocabulary (Phase 5, UPGRADE_PLAN §6.3.3).

The grown-up version of the seed's CoNLL dev/test OOV finding: instead of one dev-vs-test
number on a research corpus, this measures, per rolling yearly slice, *how far the incoming
text has moved away from the vocabulary and the feature-space centroid that Tier A was
actually fitted on*. It is a **covariate**-shift exhibit and deliberately model-free: no
prediction is read, no run record is opened, nothing is appended to ``results/runs.jsonl``.
It is an analysis of frozen data, so its outputs are committed derived JSON under
``results/oov/``.

**Two OOV definitions, because they answer different questions.** Reporting one alone is
the classic way to get this wrong:

- ``model_vocab`` (**primary**): a token is OOV if its string is not a key of the fitted
  word ``TfidfVectorizer.vocabulary_`` -- the vocabulary the model *actually has*, after
  ``min_df=5`` and ``max_features=150000`` pruning. This is the operationally relevant
  rate: an OOV token here contributes literally nothing to the word block of the Tier A
  feature vector. It is nonzero even on TRAIN itself (pruning), which is exactly why the
  TRAIN row is emitted as the baseline any drift number must be read against.
- ``corpus_novelty``: a token is OOV if its unigram never appears **anywhere** in TRAIN, at
  any frequency. No pruning, so this is true lexical novelty, and it is 0 on TRAIN by
  construction. The gap between the two definitions is the pruning artifact; the
  ``corpus_novelty`` *trend* is the part that no amount of re-tuning ``min_df`` can fix.

Both are computed with the *fitted vectorizer's own analyzer* (``build_analyzer()`` on a
``clone`` with ``ngram_range=(1, 1)``), never a hand-rolled tokenizer, so lowercasing, the
token pattern and the accent/preprocessing chain are byte-for-byte what the model sees.
``tests/test_oov.py`` pins that parity, including that the (1, 2) analyzer's output is the
(1, 1) output followed by the bigrams.

**Token-level is the headline; type-level is a companion.** The token-level (occurrence-
weighted) rate is what determines how much of a document's mass the model can represent.
The type-level (distinct-string) rate answers the vocabulary-coverage question and is
always much larger, because novel tokens are rare by construction. Type-level rates are
emitted as **point estimates only**: under a document bootstrap the number of distinct
types is itself a random quantity dominated by the ~63.2% distinct-document rate of the
resample, so a percentile interval around it would describe the bootstrap's own
combinatorics rather than sampling error in the estimand. This is stated in
``methods_notes`` in every output file rather than silently omitted.

**Centroid distance.** ``tfidf_centroid_cosine_distance`` is ``1 - cos(mu_slice, mu_TRAIN)``
in the frozen Tier A feature space: the ``tier_a.build_features`` FeatureUnion (word 1-2gram
+ char_wb 3-5gram TF-IDF) fitted on TRAIN, rows renormalised to unit L2 after the hstack --
sklearn L2-normalises each block separately, so raw FeatureUnion rows have norm sqrt(2) and
a document with an empty word block would otherwise count for less than its neighbours.
The centroid is the plain mean of those unit rows, so every document weighs the same. The
field is named for the space it lives in on purpose: this is a **TF-IDF-space** centroid,
the honest GPU-free "embedding" available at Phase 5 (see ``dense_encoder_note``).

**Bootstrap.** 95% percentile intervals, ``harness.N_RESAMPLES`` replicates at
``harness.BOOTSTRAP_SEED``, resampling **documents** within the slice. One index draw per
replicate feeds *every* statistic for that slice, so the OOV rates and the centroid distance
move together across replicates. The TRAIN side -- vocabulary, idf weights, centroid, and
the unpruned unigram type set -- is held **fixed**: it is a frozen model artifact, not a
sample, and it cannot be resampled without refitting the vectorizer (which would change the
vocabulary and therefore the estimand). TRAIN is nevertheless evaluated as a slice: its
centroid distance is a structural zero whose bootstrap interval is the pure resampling
noise floor at ``n = 300,000`` (read it as "what does no drift look like", not as a
size-matched null for the 20,000-row drift slices).

The replicate weights are materialised as a ``uint16`` count matrix rather than index
vectors, because every statistic here is a ratio of *linear* functionals of the rows::

    rate_b = (o . c_b) / (t . c_b)                     token-weighted OOV
    mu_b   = X^T c_b / n                               bootstrap centroid

so the whole centroid bootstrap collapses to one sparse-times-dense product
``X^T C`` accumulated over document chunks -- ``nnz x B`` flops with no per-replicate
re-transform. ``CHUNK_DOCS`` and ``BOOT_BLOCK`` bound peak memory; both are frozen module
constants and not CLI flags, because float summation order (and therefore the last digits
of every emitted number) depends on them.

**Provenance gate.** The feature block is read from a real Tier A config, not hardcoded, and
``FEATURE_PARITY_CONFIGS`` -- the four yearly Tier A drift runs plus the CAL rung they match
-- are asserted to declare byte-identical ``features`` blocks. That assertion is the claim
that "the TRAIN vocabulary" is well defined: one vocabulary underlies all four yearly Tier A
points, so one OOV curve describes all of them. Every split parquet is checked against its
frozen ``splits_stats.yaml`` sha256 before it is read (CLAUDE.md rule 2).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import normalize

from triage_lab import harness, tier_a
from triage_lab.snapshot import sha256_file

REPO_ROOT = harness.REPO_ROOT
DEFAULT_SPLITS_DIR = tier_a.DEFAULT_SPLITS_DIR
DEFAULT_SPLITS_STATS_PATH = harness.DEFAULT_SPLITS_STATS_PATH
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "oov"

SCHEMA_VERSION = 1
JSON_ROUND = 10
EVIDENCE_CLASS = "measured"

# Frozen protocol constants. Changing any of these changes the reported digits, so none of
# them is a CLI flag (same discipline as prior_shift.py).
REFERENCE_SLICE = "train"
SLICE_ORDER: tuple[str, ...] = (
    "train",
    "test_iid",
    "test_drift_2023",
    "test_drift_2024",
    "test_drift_2025",
    "test_drift_2026h1",
)
# Documents per transform chunk, and bootstrap replicates per dense column block. These
# bound peak RSS (the d x B accumulator dominates) and fix the float summation order.
CHUNK_DOCS = 20_000
BOOT_BLOCK = 250

# The Tier A config whose `features` block defines "the TRAIN vocabulary", plus the runs
# whose feature blocks must match it for that phrase to be well defined.
DEFAULT_FEATURE_CONFIG = REPO_ROOT / "configs" / "tier_a_logreg_test_drift_2023.yaml"
FEATURE_PARITY_CONFIGS: tuple[str, ...] = (
    "tier_a_logreg_test_drift_2023.yaml",
    "tier_a_logreg_test_drift_2024.yaml",
    "tier_a_logreg_test_drift_2025.yaml",
    "tier_a_logreg_test_drift_2026h1.yaml",
    "tier_a_logreg_wordchar_cal.yaml",
)

# Metric ids. `::token` / `::type` is the level; the prefix is the OOV definition.
METRIC_MODEL_TOKEN = "model_vocab_oov_token_rate"
METRIC_MODEL_TYPE = "model_vocab_oov_type_rate"
METRIC_NOVEL_TOKEN = "corpus_novelty_oov_token_rate"
METRIC_NOVEL_TYPE = "corpus_novelty_oov_type_rate"
METRIC_MODEL_NGRAM = "model_vocab_oov_ngram12_token_rate"
METRIC_CENTROID = "tfidf_centroid_cosine_distance"

BOOTSTRAPPED_METRICS: tuple[str, ...] = (
    METRIC_MODEL_TOKEN,
    METRIC_NOVEL_TOKEN,
    METRIC_MODEL_NGRAM,
    METRIC_CENTROID,
)
POINT_ONLY_METRICS: tuple[str, ...] = (METRIC_MODEL_TYPE, METRIC_NOVEL_TYPE)

DENSE_ENCODER_NOTE = (
    "TF-IDF-space centroid, not a dense-encoder embedding: this is the frozen Tier A "
    "FeatureUnion (word 1-2gram + char_wb 3-5gram) fitted on TRAIN. A dense sentence-encoder "
    "version of the same exhibit is pending Tier B; until it exists, no claim here should be "
    "read as being about semantic drift, only about lexical/character-n-gram drift."
)
TYPE_CI_NOTE = (
    "Type-level rates are point estimates with no CI. A document bootstrap replicate "
    "contains only ~63.2% of the distinct documents, so the number of distinct token TYPES "
    "it exhibits is driven by the resample's own combinatorics rather than by sampling error "
    "in the estimand; a percentile interval around it would be a statement about the "
    "bootstrap, not about the slice. Token-level rates are the headline and are CI'd."
)
CENTROID_CI_NOTE = (
    "A cosine distance is non-negative, so its bootstrap distribution is biased UPWARD by "
    "resampling noise and the interval need not bracket the point estimate -- most visibly "
    "on TRAIN, where the point is a structural 0 and the interval sits at the pure noise "
    "floor. That floor scales as 1/sqrt(n_docs), so at the 20,000-row drift-slice size it is "
    "approximately sqrt(300000/20000) = 3.9x the TRAIN figure. That factor is an analytic "
    "extrapolation, NOT a measurement: no size-matched null (n = 20,000 drawn from TRAIN) was "
    "computed, because even after the correction the floor stays two-plus orders of magnitude "
    "below every measured drift distance and no conclusion turns on its exact value. Measure "
    "it if a future claim ever rests on a centroid distance below ~0.001."
)
CENTROID_CONFOUND_NOTE = (
    "The centroid distance is an AGGREGATE covariate statistic: it absorbs the class-mix "
    "change and the within-class lexical change together, and does not separate them. Every "
    "eval slice is a natural-mix (Hamilton-apportioned) sample of its own period while TRAIN "
    "is class_year-stratified, so part of every distance here -- including test_iid's -- is "
    "prior shift, which results/prior_shift/ measures separately for macro-F1. Do not read "
    "this number as within-class text drift; the year-over-year comparison among the four "
    "drift slices is the like-for-like one, since they share a sampling scheme."
)
TRAIN_REFERENCE_NOTE = (
    "The TRAIN side (vocabulary, idf weights, centroid, unpruned unigram type set) is held "
    "fixed across bootstrap replicates: it is a frozen model artifact, not a sample, and it "
    "cannot be resampled without refitting the vectorizer, which would change the vocabulary "
    "and hence the estimand. Only the slice's documents are resampled."
)


# ---------------------------------------------------------------------------
# Split loading + integrity
# ---------------------------------------------------------------------------

def split_path(split: str, splits_dir=DEFAULT_SPLITS_DIR) -> Path:
    return Path(splits_dir) / f"{split}.parquet"


def verify_split_integrity(
    split: str, splits_dir=DEFAULT_SPLITS_DIR, splits_stats_path=DEFAULT_SPLITS_STATS_PATH
) -> dict:
    """Fail loud if a split parquet drifts from the frozen splits_stats.yaml (CLAUDE.md #2).

    Same gate as ``tier_a._verify_integrity``, duplicated rather than imported only because
    it returns the dataset block this module embeds in its output provenance.
    """
    info = harness.dataset_info(split, splits_stats_path)
    path = split_path(split, splits_dir)
    actual = sha256_file(path)
    if actual != info["split_sha256"]:
        raise ValueError(
            f"integrity check failed for split {split!r}: parquet sha256 {actual} "
            f"!= frozen splits_stats.yaml {info['split_sha256']}"
        )
    return info


def load_texts(split: str, splits_dir=DEFAULT_SPLITS_DIR) -> list[str]:
    """Narratives of `split`, ordered by complaint_id -- the exact order tier_a reads."""
    texts, _ = tier_a.load_split_frame(
        split_path(split, splits_dir),
        tier_a._DEFAULT_TEXT_COLUMN,
        tier_a._DEFAULT_LABEL_COLUMN,
        tier_a._DEFAULT_ORDER_COLUMN,
    )
    return texts


# ---------------------------------------------------------------------------
# Feature config resolution (provenance gate)
# ---------------------------------------------------------------------------

def resolve_feature_config(
    config_path=DEFAULT_FEATURE_CONFIG, parity_configs=FEATURE_PARITY_CONFIGS
) -> dict:
    """Read the `features` block from a Tier A config and assert the parity set matches.

    "The TRAIN vocabulary" is only a well-defined object if every Tier A run whose drift
    this exhibit explains was fitted with the same feature spec. That is checked here rather
    than assumed; a mismatch is a hard error, not a warning.
    """
    config_path = Path(config_path)
    config = harness.load_config(config_path)
    features = config.get("features", {})
    mismatched = []
    for name in parity_configs:
        other_path = config_path.parent / name
        if other_path == config_path or not other_path.exists():
            continue
        other = harness.load_config(other_path).get("features", {})
        if other != features:
            mismatched.append(name)
    if mismatched:
        raise ValueError(
            f"feature-block parity failed: {sorted(mismatched)} declare a different "
            f"`features` block than {config_path.name}. 'The TRAIN vocabulary' is then not "
            "a single object and one OOV curve cannot describe all the Tier A yearly runs"
        )
    return {
        "features": features,
        "config_path": str(config_path.relative_to(REPO_ROOT))
        if config_path.is_absolute() and config_path.is_relative_to(REPO_ROOT)
        else str(config_path),
        "config_sha256": harness.config_sha256(config_path),
        "parity_configs": sorted(parity_configs),
    }


# ---------------------------------------------------------------------------
# The frozen TRAIN reference
# ---------------------------------------------------------------------------

@dataclass
class Reference:
    """Everything derived from TRAIN that later slices are measured against."""

    features: object                  # fitted tier_a FeatureUnion (word + char)
    word_vec: object                  # the fitted word TfidfVectorizer
    uni_analyzer: object              # callable text -> list[str] of unigrams
    ngram_analyzer: object            # callable text -> list[str] of 1- and 2-grams
    model_vocab: dict                 # word_vec.vocabulary_ (post-pruning)
    train_types: set                  # unpruned TRAIN unigram types
    n_word_features: int
    n_char_features: int
    n_train_types_in_vocab: int = 0    # TRAIN unigram types that survive pruning
    centroid: np.ndarray | None = None    # filled once TRAIN has been transformed


def _optional_transformer(feature_union, name: str):
    for key, transformer in feature_union.transformer_list:
        if key == name:
            return transformer
    return None


def _named_transformer(feature_union, name: str):
    transformer = _optional_transformer(feature_union, name)
    if transformer is None:
        raise ValueError(
            f"the Tier A FeatureUnion has no {name!r} block; this exhibit measures OOV "
            "against the word vocabulary and needs the word block enabled"
        )
    return transformer


def unigram_analyzer(word_vec):
    """`build_analyzer()` of a clone of the fitted word vectorizer with ngram_range=(1, 1).

    Cloning rather than re-constructing guarantees the preprocessing chain (lowercase,
    token_pattern, strip_accents, stop_words) is identical to the fitted model's; only the
    n-gram assembly step differs. ``clone`` drops fitted state, which is fine -- the
    analyzer never consults the vocabulary.
    """
    uni = clone(word_vec)
    uni.set_params(ngram_range=(1, 1))
    return uni.build_analyzer()


def build_reference(train_texts, features_cfg: dict) -> Reference:
    """Fit the Tier A FeatureUnion on TRAIN and collect the unpruned unigram type set."""
    feature_union = tier_a.build_features(features_cfg)
    feature_union.fit(train_texts)
    word_vec = _named_transformer(feature_union, "word")
    # The char block is optional (tier_a supports the word-only ablation); when it is off,
    # the "TF-IDF space" is just the word block and n_char_features is 0.
    char_vec = _optional_transformer(feature_union, "char")
    uni = unigram_analyzer(word_vec)

    train_types: set[str] = set()
    for text in train_texts:
        train_types.update(uni(text))

    return Reference(
        features=feature_union,
        word_vec=word_vec,
        uni_analyzer=uni,
        ngram_analyzer=word_vec.build_analyzer(),
        model_vocab=word_vec.vocabulary_,
        train_types=train_types,
        n_word_features=len(word_vec.vocabulary_),
        n_char_features=0 if char_vec is None else len(char_vec.vocabulary_),
        # The word vocabulary mixes unigrams and bigrams, so it is not comparable to a
        # unigram type count. The intersection is: every unigram in the vocabulary came
        # from TRAIN, so |train_types & vocab| is exactly "unigram types that survived
        # pruning" -- the denominator both headline rates are actually measured against.
        n_train_types_in_vocab=len(train_types & word_vec.vocabulary_.keys()),
    )


# ---------------------------------------------------------------------------
# Per-document token statistics
# ---------------------------------------------------------------------------

@dataclass
class TokenStats:
    """Per-document counts (the linear functionals the bootstrap resamples) + type sets."""

    n_tokens: np.ndarray        # unigram occurrences per document
    n_oov_model: np.ndarray     # occurrences whose unigram is not in the fitted vocabulary
    n_oov_novel: np.ndarray     # occurrences whose unigram never appears in TRAIN
    n_ngrams: np.ndarray        # (1, 2)-gram occurrences per document
    n_oov_ngram: np.ndarray     # (1, 2)-gram occurrences not in the fitted vocabulary
    n_types: int
    n_oov_model_types: int
    n_oov_novel_types: int


def token_stats(texts, ref: Reference) -> TokenStats:
    """Count OOV occurrences per document under both definitions, plus slice-level types.

    ``sum(map(dict.__contains__, toks))`` keeps the hot loop in C; the arithmetic is
    ``occurrences - in_vocab_occurrences`` so a repeated OOV token counts once per
    occurrence, which is what "token-level" means.
    """
    n = len(texts)
    n_tokens = np.zeros(n, dtype=np.int64)
    n_oov_model = np.zeros(n, dtype=np.int64)
    n_oov_novel = np.zeros(n, dtype=np.int64)
    n_ngrams = np.zeros(n, dtype=np.int64)
    n_oov_ngram = np.zeros(n, dtype=np.int64)

    vocab_has = ref.model_vocab.__contains__
    train_has = ref.train_types.__contains__
    uni, ngram = ref.uni_analyzer, ref.ngram_analyzer
    slice_types: set[str] = set()

    for i, text in enumerate(texts):
        toks = uni(text)
        n_tokens[i] = len(toks)
        n_oov_model[i] = len(toks) - sum(map(vocab_has, toks))
        n_oov_novel[i] = len(toks) - sum(map(train_has, toks))
        slice_types.update(toks)
        grams = ngram(text)
        n_ngrams[i] = len(grams)
        n_oov_ngram[i] = len(grams) - sum(map(vocab_has, grams))

    return TokenStats(
        n_tokens=n_tokens,
        n_oov_model=n_oov_model,
        n_oov_novel=n_oov_novel,
        n_ngrams=n_ngrams,
        n_oov_ngram=n_oov_ngram,
        n_types=len(slice_types),
        n_oov_model_types=sum(1 for t in slice_types if t not in ref.model_vocab),
        n_oov_novel_types=sum(1 for t in slice_types if t not in ref.train_types),
    )


def _ratio(numer: float, denom: float) -> float:
    """0/0 -> 0.0 (an empty slice has no OOV), never a NaN in a committed JSON."""
    return float(numer) / float(denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Bootstrap weights
# ---------------------------------------------------------------------------

def bootstrap_counts(
    n: int, n_resamples: int = harness.N_RESAMPLES, seed: int = harness.BOOTSTRAP_SEED
) -> np.ndarray:
    """(n, B) uint16 multiplicities of an i.i.d.-with-replacement document bootstrap.

    One ``default_rng(seed)`` stream, one ``rng.integers(0, n, size=n)`` draw per replicate,
    in order -- so the stream depends only on ``(seed, n)`` and the SAME weights drive every
    statistic of the slice. Counts, not index vectors, because every statistic here is a
    ratio of linear functionals of the rows; ``uint16`` because the maximum multiplicity of
    one document in a size-n resample is a handful even at n = 300,000 (checked below).
    """
    rng = np.random.default_rng(seed)
    counts = np.empty((n, n_resamples), dtype=np.uint16)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        col = np.bincount(idx, minlength=n)
        if col.max() > np.iinfo(np.uint16).max:
            raise ValueError("bootstrap multiplicity overflows uint16")
        counts[:, b] = col.astype(np.uint16)
    return counts


def weighted_ratio_replicates(
    numer: np.ndarray, denom: np.ndarray, counts: np.ndarray, block: int = BOOT_BLOCK
) -> np.ndarray:
    """Per-replicate ``(numer . c_b) / (denom . c_b)``, evaluated in column blocks.

    Blocking exists only to bound the float64 cast of the uint16 count matrix; the result is
    independent of ``block`` because each column is an independent dot product.
    """
    n_resamples = counts.shape[1]
    out = np.empty(n_resamples, dtype=np.float64)
    numer = numer.astype(np.float64)
    denom = denom.astype(np.float64)
    for b0 in range(0, n_resamples, block):
        chunk = counts[:, b0 : b0 + block].astype(np.float64)
        num_b = numer @ chunk
        den_b = denom @ chunk
        out[b0 : b0 + block] = np.where(den_b > 0, num_b / np.where(den_b > 0, den_b, 1.0), 0.0)
    return out


# ---------------------------------------------------------------------------
# Centroid + its bootstrap in the frozen TF-IDF space
# ---------------------------------------------------------------------------

@dataclass
class CentroidResult:
    centroid: np.ndarray
    distance: float
    replicates: np.ndarray | None
    n_zero_rows: int
    centroid_l2: float


def centroid_pass(
    texts,
    ref: Reference,
    counts: np.ndarray | None,
    reference_centroid: np.ndarray | None,
    chunk_docs: int = CHUNK_DOCS,
    block: int = BOOT_BLOCK,
) -> CentroidResult:
    """Slice centroid, its cosine distance to `reference_centroid`, and its bootstrap.

    One streaming pass over the documents in fixed ``chunk_docs`` blocks: each chunk is
    transformed by the frozen FeatureUnion, its rows renormalised to unit L2 (sklearn
    normalises the word and char blocks separately, so raw hstacked rows have norm sqrt(2)
    and a document with an empty word block would otherwise be down-weighted), then folded
    into two accumulators -- the plain column sum (the centroid) and ``X^T C`` (every
    bootstrap centroid at once). The chunked accumulation is why ``chunk_docs`` is frozen:
    it fixes the float summation order.

    ``reference_centroid=None`` means "this slice IS the reference" and the distance is
    computed against the slice's own centroid, i.e. a structural zero.
    """
    n = len(texts)
    n_features = ref.n_word_features + ref.n_char_features
    total = np.zeros(n_features, dtype=np.float64)
    n_resamples = 0 if counts is None else counts.shape[1]
    boot_sum = (
        np.zeros((n_features, n_resamples), dtype=np.float64) if n_resamples else None
    )
    n_zero_rows = 0

    for start in range(0, n, chunk_docs):
        stop = min(start + chunk_docs, n)
        mat = ref.features.transform(texts[start:stop]).tocsr()
        if mat.dtype != np.float64:
            mat = mat.astype(np.float64)
        # TF-IDF never stores an explicit zero, so "no stored entry in this row" is exactly
        # "this document has no representable feature" -- cheaper and exact vs a norm pass.
        n_zero_rows += int(np.sum(np.diff(mat.indptr) == 0))
        mat = normalize(mat, norm="l2", axis=1, copy=False)
        total += np.asarray(mat.sum(axis=0)).ravel()
        if boot_sum is not None:
            mat_t = mat.T.tocsr()
            for b0 in range(0, n_resamples, block):
                weights = np.ascontiguousarray(
                    counts[start:stop, b0 : b0 + block], dtype=np.float64
                )
                boot_sum[:, b0 : b0 + block] += mat_t @ weights
        del mat

    centroid = total / n if n > 0 else total
    target = centroid if reference_centroid is None else reference_centroid
    distance = cosine_distance(centroid, target)

    replicates = None
    if boot_sum is not None:
        # ||mu_b|| accumulated in feature blocks so the squaring never allocates a second
        # d x B array.
        sq = np.zeros(n_resamples, dtype=np.float64)
        for f0 in range(0, n_features, 50_000):
            sq += np.einsum("ij,ij->j", boot_sum[f0 : f0 + 50_000], boot_sum[f0 : f0 + 50_000])
        norms = np.sqrt(sq)
        dots = target @ boot_sum
        target_norm = float(np.linalg.norm(target))
        denom = norms * target_norm
        cos = np.where(denom > 0, dots / np.where(denom > 0, denom, 1.0), 0.0)
        replicates = 1.0 - np.clip(cos, -1.0, 1.0)
        del boot_sum

    return CentroidResult(
        centroid=centroid,
        distance=distance,
        replicates=replicates,
        n_zero_rows=n_zero_rows,
        centroid_l2=float(np.linalg.norm(centroid)),
    )


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cos(a, b); 0 when either vector is the origin (no direction to disagree about).

    The cosine is clipped into [-1, 1] before the subtraction, and distances below 1e-12
    snap to exactly +0.0. Both guards exist because float rounding lands the self-cosine on
    either side of 1 depending on the BLAS: on macOS/Accelerate the 300,000-document TRAIN
    centroid comes out at cos > 1 (a *negative* distance, -2.2e-16, in a JSON that claims a
    structural zero -- the clip fixes that); on Linux/OpenBLAS it comes out at cos < 1
    (+3.3e-16, which the clip cannot touch -- the snap fixes that). The quantity is
    mathematically in [0, 2] and every measured drift distance in the committed artifacts is
    >= ~1e-3, four orders above the snap threshold, so the snap can only ever absorb rounding
    noise.
    """
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    d = 1.0 - min(1.0, max(-1.0, float(a @ b) / (na * nb)))
    return 0.0 if d < 1e-12 else float(d)


# ---------------------------------------------------------------------------
# CI blocks
# ---------------------------------------------------------------------------

def _percentile_ci(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(values, [harness.CI_LOWER_PCT, harness.CI_UPPER_PCT])
    return float(lo), float(hi)


def ci_block(point: float, replicates: np.ndarray) -> dict:
    lo, hi = _percentile_ci(replicates)
    return {"point": float(point), "ci_lo": lo, "ci_hi": hi, "ci_kind": "bootstrap_percentile"}


def point_block(point: float, reason: str) -> dict:
    return {"point": float(point), "ci_lo": None, "ci_hi": None, "ci_kind": None,
            "no_ci_reason": reason}


# ---------------------------------------------------------------------------
# Per-slice assembly
# ---------------------------------------------------------------------------

def build_slice(
    slice_name: str,
    ref: Reference,
    *,
    splits_dir=DEFAULT_SPLITS_DIR,
    splits_stats_path=DEFAULT_SPLITS_STATS_PATH,
    feature_provenance: dict | None = None,
    texts=None,
    n_resamples: int = harness.N_RESAMPLES,
    seed: int = harness.BOOTSTRAP_SEED,
    chunk_docs: int = CHUNK_DOCS,
    generated_at: str | None = None,
    git_sha: str | None = None,
) -> dict:
    """Full JSON object for one slice. `ref.centroid` must be set unless this IS the reference."""
    is_reference = slice_name == REFERENCE_SLICE
    dataset = verify_split_integrity(slice_name, splits_dir, splits_stats_path)
    if texts is None:
        texts = load_texts(slice_name, splits_dir)
    n_docs = len(texts)

    stats = token_stats(texts, ref)
    counts = bootstrap_counts(n_docs, n_resamples=n_resamples, seed=seed)

    reference_centroid = None if is_reference else ref.centroid
    if not is_reference and reference_centroid is None:
        raise ValueError(
            f"slice {slice_name!r} needs the TRAIN centroid, which has not been computed; "
            "the reference slice must be processed first"
        )
    cent = centroid_pass(texts, ref, counts, reference_centroid, chunk_docs=chunk_docs)
    if is_reference:
        ref.centroid = cent.centroid

    total_tokens = int(stats.n_tokens.sum())
    total_ngrams = int(stats.n_ngrams.sum())
    metrics = {
        METRIC_MODEL_TOKEN: ci_block(
            _ratio(stats.n_oov_model.sum(), total_tokens),
            weighted_ratio_replicates(stats.n_oov_model, stats.n_tokens, counts),
        ),
        METRIC_NOVEL_TOKEN: ci_block(
            _ratio(stats.n_oov_novel.sum(), total_tokens),
            weighted_ratio_replicates(stats.n_oov_novel, stats.n_tokens, counts),
        ),
        METRIC_MODEL_NGRAM: ci_block(
            _ratio(stats.n_oov_ngram.sum(), total_ngrams),
            weighted_ratio_replicates(stats.n_oov_ngram, stats.n_ngrams, counts),
        ),
        METRIC_CENTROID: ci_block(cent.distance, cent.replicates),
        METRIC_MODEL_TYPE: point_block(
            _ratio(stats.n_oov_model_types, stats.n_types), "type_level_under_document_bootstrap"
        ),
        METRIC_NOVEL_TYPE: point_block(
            _ratio(stats.n_oov_novel_types, stats.n_types), "type_level_under_document_bootstrap"
        ),
    }

    structural = {}
    if is_reference:
        structural[METRIC_NOVEL_TOKEN] = (
            "exactly 0 by construction: corpus novelty is defined against TRAIN itself"
        )
        structural[METRIC_NOVEL_TYPE] = structural[METRIC_NOVEL_TOKEN]
        structural[METRIC_CENTROID] = (
            "exactly 0 by construction (self-distance); the interval is the pure "
            f"resampling noise floor at n = {n_docs}, NOT a size-matched null for the "
            "20,000-row drift slices -- see centroid.ci_note"
        )
        structural[METRIC_MODEL_TOKEN] = (
            "nonzero on TRAIN because of min_df / max_features pruning; this is the "
            "baseline every drift-slice model-vocab rate must be read against"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "git_sha": git_sha if git_sha is not None else harness._git_sha(),
        "evidence_class": EVIDENCE_CLASS,
        "repro_command": f"uv run python -m triage_lab.oov --slice {slice_name}",
        "analysis": "oov_covariate_tracking",
        "slice": slice_name,
        "is_reference_slice": is_reference,
        "reference_slice": REFERENCE_SLICE,
        "dataset": dataset,
        "counts": {
            "n_docs": n_docs,
            "n_tokens": total_tokens,
            "n_types": stats.n_types,
            "n_ngrams_12": total_ngrams,
            "n_oov_model_tokens": int(stats.n_oov_model.sum()),
            "n_oov_novel_tokens": int(stats.n_oov_novel.sum()),
            "n_oov_model_types": stats.n_oov_model_types,
            "n_oov_novel_types": stats.n_oov_novel_types,
            "n_oov_model_ngrams_12": int(stats.n_oov_ngram.sum()),
            "n_docs_zero_feature_row": cent.n_zero_rows,
        },
        "metrics": metrics,
        "structural_values": structural,
        "vocabulary": {
            "source_split": REFERENCE_SLICE,
            "n_word_features": ref.n_word_features,
            "n_char_features": ref.n_char_features,
            "n_features": ref.n_word_features + ref.n_char_features,
            "n_train_unigram_types_unpruned": len(ref.train_types),
            "n_train_unigram_types_in_model_vocab": ref.n_train_types_in_vocab,
            "pruning_note": (
                f"{ref.n_train_types_in_vocab} of {len(ref.train_types)} distinct TRAIN "
                f"unigram types survive min_df / max_features into the word vocabulary; the "
                f"other {len(ref.model_vocab) - ref.n_train_types_in_vocab} of that "
                f"vocabulary's {len(ref.model_vocab)} features are bigrams, which is why the "
                "feature count is not comparable to a unigram type count. The gap between "
                "the model_vocab and corpus_novelty rates is exactly this pruning"
            ),
        },
        "centroid": {
            "space": "tier_a_tfidf_featureunion_word1_2_char_wb3_5",
            "row_normalization": "unit L2 after hstack (blocks are L2-normalised separately "
                                 "by sklearn, giving raw rows norm sqrt(2))",
            "aggregation": "unweighted mean of unit rows",
            "slice_centroid_l2": cent.centroid_l2,
            "dense_encoder_note": DENSE_ENCODER_NOTE,
            "ci_note": CENTROID_CI_NOTE,
            "confounding_note": CENTROID_CONFOUND_NOTE,
        },
        "provenance": {
            **(feature_provenance or {}),
            "chunk_docs": chunk_docs,
            "boot_block": BOOT_BLOCK,
        },
        "bootstrap": {
            "n_resamples": n_resamples,
            "seed": seed,
            "method": harness.BOOTSTRAP_METHOD,
            "ci_pct": [harness.CI_LOWER_PCT, harness.CI_UPPER_PCT],
            "resample": "slice_documents_with_replacement",
            "joint_draw": "one weight vector per replicate drives every statistic of the slice",
            "train_reference_note": TRAIN_REFERENCE_NOTE,
        },
        "methods_notes": {
            "model_vocab": (
                "OOV against the fitted word TfidfVectorizer.vocabulary_ (post min_df / "
                "max_features pruning) -- the vocabulary the model actually has"
            ),
            "corpus_novelty": (
                "OOV against every unigram type appearing anywhere in TRAIN, unpruned -- "
                "true lexical novelty, disentangled from the pruning artifact"
            ),
            "tokenization": (
                "the fitted vectorizer's own build_analyzer(); the unigram analyzer is a "
                "clone with ngram_range=(1, 1), so lowercasing / token_pattern / "
                "preprocessing are byte-for-byte the model's"
            ),
            "token_vs_type": TYPE_CI_NOTE,
            "headline": (
                f"{METRIC_MODEL_TOKEN} is the primary Tier-A-relevant rate; "
                f"{METRIC_NOVEL_TOKEN} is the pruning-free novelty rate; "
                f"{METRIC_MODEL_NGRAM} is a secondary (1, 2)-gram-level companion"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Flat summary rows (drift-chart input)
# ---------------------------------------------------------------------------

def summary_rows(obj: dict) -> list[dict]:
    """One flat row per (slice, metric)."""
    rows = []
    for metric in (*BOOTSTRAPPED_METRICS, *POINT_ONLY_METRICS):
        block = obj["metrics"][metric]
        rows.append({
            "slice": obj["slice"],
            "metric": metric,
            "point": block["point"],
            "ci_lo": block["ci_lo"],
            "ci_hi": block["ci_hi"],
            "ci_kind": block["ci_kind"],
            "n_docs": obj["counts"]["n_docs"],
            "n_tokens": obj["counts"]["n_tokens"],
            "is_reference_slice": obj["is_reference_slice"],
            "structural_note": obj["structural_values"].get(metric),
        })
    return rows


# ---------------------------------------------------------------------------
# Deterministic JSON output
# ---------------------------------------------------------------------------

def _round_tree(value):
    """Round floats to JSON_ROUND; NaN/inf -> None (valid JSON, honest about undefined)."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _round_tree(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_round_tree(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, JSON_ROUND)
    return value


def write_json(obj: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_round_tree(obj), sort_keys=True, indent=2) + "\n")
    return path


def output_name(slice_name: str) -> str:
    return f"{slice_name}.json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def select_slices(selected) -> list[str]:
    """Requested slices in frozen SLICE_ORDER; the reference is always processed first."""
    if not selected:
        return list(SLICE_ORDER)
    wanted = set(selected)
    return [s for s in SLICE_ORDER if s in wanted]


def run(
    slices,
    *,
    splits_dir=DEFAULT_SPLITS_DIR,
    splits_stats_path=DEFAULT_SPLITS_STATS_PATH,
    out_dir=DEFAULT_OUT_DIR,
    feature_config=DEFAULT_FEATURE_CONFIG,
    write_summary: bool = False,
    n_resamples: int = harness.N_RESAMPLES,
    seed: int = harness.BOOTSTRAP_SEED,
    chunk_docs: int = CHUNK_DOCS,
    log=print,
) -> list[dict]:
    """Fit the TRAIN reference once, then emit one JSON per requested slice."""
    provenance = resolve_feature_config(feature_config)
    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    git_sha = harness._git_sha()

    verify_split_integrity(REFERENCE_SLICE, splits_dir, splits_stats_path)
    train_texts = load_texts(REFERENCE_SLICE, splits_dir)
    log(f"[reference] fitting the Tier A FeatureUnion on {len(train_texts)} TRAIN documents")
    ref = build_reference(train_texts, provenance["features"])
    log(
        f"[reference] word features={ref.n_word_features} char features={ref.n_char_features} "
        f"unpruned TRAIN unigram types={len(ref.train_types)}"
    )

    wanted = select_slices(slices)
    objs = []
    all_rows: list[dict] = []

    def emit(obj) -> None:
        out_path = write_json(obj, Path(out_dir) / output_name(obj["slice"]))
        objs.append(obj)
        all_rows.extend(summary_rows(obj))
        _log_slice(obj, out_path, log)

    # TRAIN always comes first: it defines the centroid every other slice is measured
    # against. When TRAIN is not itself requested, only the centroid is computed -- its
    # bootstrap is by far the most expensive single item here (n = 300,000) and nothing
    # else needs it.
    if REFERENCE_SLICE in wanted:
        emit(build_slice(
            REFERENCE_SLICE, ref,
            splits_dir=splits_dir, splits_stats_path=splits_stats_path,
            feature_provenance=provenance, texts=train_texts,
            n_resamples=n_resamples, seed=seed, chunk_docs=chunk_docs,
            generated_at=generated_at, git_sha=git_sha,
        ))
    else:
        ref.centroid = centroid_pass(
            train_texts, ref, None, None, chunk_docs=chunk_docs
        ).centroid
        log(f"[{REFERENCE_SLICE:17s}] reference centroid only (not requested for output)")
    del train_texts

    for name in wanted:
        if name == REFERENCE_SLICE:
            continue
        emit(build_slice(
            name, ref,
            splits_dir=splits_dir, splits_stats_path=splits_stats_path,
            feature_provenance=provenance,
            n_resamples=n_resamples, seed=seed, chunk_docs=chunk_docs,
            generated_at=generated_at, git_sha=git_sha,
        ))

    if write_summary:
        summary_path = write_json(
            {
                "schema_version": SCHEMA_VERSION,
                "generated_at": generated_at,
                "git_sha": git_sha,
                "evidence_class": EVIDENCE_CLASS,
                "analysis": "oov_covariate_tracking",
                "reference_slice": REFERENCE_SLICE,
                "slice_order": list(SLICE_ORDER),
                "repro_command": "uv run python -m triage_lab.oov --all",
                "vocabulary": objs[0]["vocabulary"] if objs else {},
                "provenance": objs[0]["provenance"] if objs else {},
                "methods_notes": objs[0]["methods_notes"] if objs else {},
                "rows": all_rows,
            },
            Path(out_dir) / "summary.json",
        )
        log(f"summary: {len(all_rows)} rows -> {summary_path}")
    return objs


def _log_slice(obj: dict, out_path, log) -> None:
    m = obj["metrics"]
    model, novel, cent = m[METRIC_MODEL_TOKEN], m[METRIC_NOVEL_TOKEN], m[METRIC_CENTROID]
    log(
        f"[{obj['slice']:17s}] n_docs={obj['counts']['n_docs']:>7d} "
        f"n_tokens={obj['counts']['n_tokens']:>11d}  "
        f"model_vocab_oov={model['point'] * 100:6.3f}% "
        f"[{model['ci_lo'] * 100:.3f},{model['ci_hi'] * 100:.3f}]  "
        f"novelty_oov={novel['point'] * 100:6.3f}% "
        f"[{novel['ci_lo'] * 100:.3f},{novel['ci_hi'] * 100:.3f}]  "
        f"centroid_dist={cent['point']:.6f} "
        f"[{cent['ci_lo']:.6f},{cent['ci_hi']:.6f}] -> {out_path}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m triage_lab.oov")
    parser.add_argument("--all", action="store_true",
                        help="every slice in the frozen order (train + test_iid + 4 drift years)")
    parser.add_argument("--slice", action="append", choices=list(SLICE_ORDER),
                        help="restrict to this slice (repeatable); TRAIN is fitted regardless, "
                             "because it defines the reference vocabulary and centroid")
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--splits-stats", type=Path, default=DEFAULT_SPLITS_STATS_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--feature-config", type=Path, default=DEFAULT_FEATURE_CONFIG)
    args = parser.parse_args(argv)

    if not args.all and not args.slice:
        parser.error("give --all or at least one --slice")

    slices = list(SLICE_ORDER) if args.all else args.slice
    run(
        slices,
        splits_dir=args.splits_dir,
        splits_stats_path=args.splits_stats,
        out_dir=args.out_dir,
        feature_config=args.feature_config,
        write_summary=bool(args.all),
    )
    if not args.all:
        print("summary.json not rewritten (partial selection; use --all)")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
