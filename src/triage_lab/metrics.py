"""Pure-numpy classification / calibration / selective-prediction metrics.

Runtime policy: this module has **no runtime dependency on scikit-learn**. Every
metric is implemented from confusion-matrix / probability primitives in numpy so
that (a) the exact convention behind each headline number is auditable in one
place, (b) results are byte-stable across environments (no sklearn version drift),
and (c) the metric path stays cheap enough to call ~1000× inside the bootstrap.
scikit-learn appears only as the *reference oracle in the unit tests*
(`tests/test_metrics.py`), never here. (UPGRADE_PLAN.md §6.1: "metric unit tests
vs sklearn reference".)

Two API layers, one implementation:

- **Label-level** functions (`macro_f1`, `brier_score`, ...) take `y_true` /
  `y_pred` as arrays of *labels* drawn from an ordered `class_labels` list, plus a
  `probs` matrix whose columns are aligned to that list. These are the ergonomic,
  test-facing entry points.
- **Code-level** `*_from_codes` functions take pre-encoded integer class codes
  (0..K-1). The bootstrap in `harness.py` encodes `y_true`/`y_pred` to codes *once*
  and calls these directly, so resampling never re-hashes labels.

Conventions (documented because portfolio numbers depend on them):

- **Confusion matrix**: rows = true class, cols = predicted class, ordered by
  `class_labels` (sklearn's `confusion_matrix` orientation).
- **F1** (per-class and macro): `2*TP / (2*TP + FP + FN)`; a class whose
  denominator is 0 (absent from both y_true and y_pred) contributes F1 = 0.
  macro-F1 averages per-class F1 over *all* `class_labels`. This matches
  `sklearn.f1_score(average="macro", labels=class_labels, zero_division=0)`.
- **Balanced accuracy**: mean of per-class recall over classes with non-zero true
  support (matches `sklearn.balanced_accuracy_score`, which drops classes absent
  from y_true).
- **Probability-based metrics** (ECE / Brier / risk-coverage / AURC /
  accuracy@coverage) derive the predicted class from `argmax(probs, axis=1)`
  (numpy tie-break: lowest column index) and the confidence from
  `max(probs, axis=1)`. They intentionally ignore the passed-in `y_pred`, so the
  confidence and the prediction it scores are always self-consistent.
- **ECE**: 15 equal-width bins on [0, 1]; sample of confidence `p` lands in bin
  `min(floor(p * n_bins), n_bins - 1)` (lower-closed intervals `[i/n, (i+1)/n)`,
  with `p = 1.0` folded into the top bin). ECE = Σ (n_bin/N) · |acc_bin - conf_bin|
  over non-empty bins.
- **Brier (multiclass, sum-over-classes convention)**: mean over samples of
  Σ_k (p_k - y_k)² with y one-hot. Range [0, 2]; documented explicitly because the
  alternative (dividing by K) is also in circulation.
- **Selective prediction**: samples ranked by confidence *descending*, ties broken
  by ascending original index (stable argsort on the negated confidence). The
  risk-coverage curve reports, at coverage k/N, the error rate over the top-k most
  confident samples. AURC = mean of that risk over k = 1..N (empirical
  Geifman–El-Yaniv AURC). `accuracy_at_coverage(c)` accepts the top
  `ceil(c * N)` samples (clamped to [1, N]).
"""

from __future__ import annotations

import numpy as np

DEFAULT_N_BINS = 15
DEFAULT_COVERAGES: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95)


# ---------------------------------------------------------------------------
# Label <-> integer-code encoding
# ---------------------------------------------------------------------------

def encode_labels(y, class_labels) -> np.ndarray:
    """Map an array of labels to integer class codes (0..K-1) via `class_labels`."""
    lookup = {label: i for i, label in enumerate(class_labels)}
    try:
        return np.fromiter((lookup[v] for v in y), dtype=np.int64, count=len(y))
    except KeyError as exc:  # a label outside the declared class set is a bug, fail loud
        raise ValueError(f"label {exc.args[0]!r} not in class_labels {list(class_labels)}") from exc


def _as_probs(probs) -> np.ndarray:
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"probs must be 2-D (N, K); got shape {arr.shape}")
    return arr


# ---------------------------------------------------------------------------
# Confusion matrix and F1 family (code level)
# ---------------------------------------------------------------------------

def confusion_from_codes(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """(K, K) int64 matrix; rows = true code, cols = predicted code."""
    m = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(m, (y_true, y_pred), 1)
    return m


def per_class_f1_from_codes(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = confusion_from_codes(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2.0 * tp + fp + fn
    return np.where(denom > 0, 2.0 * tp / denom, 0.0)


def macro_f1_from_codes(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    return float(per_class_f1_from_codes(y_true, y_pred, n_classes).mean())


def balanced_accuracy_from_codes(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> float:
    cm = confusion_from_codes(y_true, y_pred, n_classes)
    support = cm.sum(axis=1)
    present = support > 0
    if not present.any():
        return 0.0
    recall = np.diag(cm)[present] / support[present]
    return float(recall.mean())


def accuracy_from_codes(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float((y_true == y_pred).mean())


# ---------------------------------------------------------------------------
# Calibration (code level)
# ---------------------------------------------------------------------------

def expected_calibration_error_from_codes(
    true_idx: np.ndarray, probs: np.ndarray, n_bins: int = DEFAULT_N_BINS
) -> float:
    probs = _as_probs(probs)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == true_idx).astype(np.float64)
    n = len(true_idx)
    if n == 0:
        return 0.0
    bin_idx = np.minimum((conf * n_bins).astype(np.int64), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        acc_b = correct[mask].mean()
        conf_b = conf[mask].mean()
        ece += (count / n) * abs(acc_b - conf_b)
    return float(ece)


def brier_score_from_codes(true_idx: np.ndarray, probs: np.ndarray) -> float:
    probs = _as_probs(probs)
    n, k = probs.shape
    if n == 0:
        return 0.0
    onehot = np.zeros((n, k), dtype=np.float64)
    onehot[np.arange(n), true_idx] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


# ---------------------------------------------------------------------------
# Selective prediction (code level)
# ---------------------------------------------------------------------------

def _confidence_order(probs: np.ndarray) -> np.ndarray:
    """Indices sorted by confidence descending; ties -> ascending original index."""
    conf = probs.max(axis=1)
    # Stable sort on the negated confidence keeps original (ascending) order on ties.
    return np.argsort(-conf, kind="stable")


def risk_coverage_curve_from_codes(
    true_idx: np.ndarray, probs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (coverages, risks): coverage k/N and error-rate over top-k confident."""
    probs = _as_probs(probs)
    n = len(true_idx)
    if n == 0:
        return np.array([]), np.array([])
    pred = probs.argmax(axis=1)
    correct = (pred == true_idx).astype(np.float64)
    order = _confidence_order(probs)
    errors = 1.0 - correct[order]
    k = np.arange(1, n + 1, dtype=np.float64)
    risks = np.cumsum(errors) / k
    coverages = k / n
    return coverages, risks


def aurc_from_codes(true_idx: np.ndarray, probs: np.ndarray) -> float:
    _, risks = risk_coverage_curve_from_codes(true_idx, probs)
    if risks.size == 0:
        return 0.0
    return float(risks.mean())


def accuracy_at_coverage_from_codes(
    true_idx: np.ndarray,
    probs: np.ndarray,
    coverages: tuple[float, ...] = DEFAULT_COVERAGES,
) -> dict[str, float]:
    probs = _as_probs(probs)
    n = len(true_idx)
    pred = probs.argmax(axis=1)
    correct = (pred == true_idx).astype(np.float64)
    order = _confidence_order(probs)
    out: dict[str, float] = {}
    for c in coverages:
        key = f"{c:.2f}"
        if n == 0:
            out[key] = 0.0
            continue
        n_accept = int(np.ceil(c * n))
        n_accept = min(max(n_accept, 1), n)
        out[key] = float(correct[order[:n_accept]].mean())
    return out


# ---------------------------------------------------------------------------
# Label-level public API (thin wrappers that encode then delegate)
# ---------------------------------------------------------------------------

def confusion_matrix(y_true, y_pred, class_labels) -> np.ndarray:
    k = len(class_labels)
    return confusion_from_codes(
        encode_labels(y_true, class_labels), encode_labels(y_pred, class_labels), k
    )


def per_class_f1(y_true, y_pred, class_labels) -> np.ndarray:
    k = len(class_labels)
    return per_class_f1_from_codes(
        encode_labels(y_true, class_labels), encode_labels(y_pred, class_labels), k
    )


def macro_f1(y_true, y_pred, class_labels) -> float:
    k = len(class_labels)
    return macro_f1_from_codes(
        encode_labels(y_true, class_labels), encode_labels(y_pred, class_labels), k
    )


def balanced_accuracy(y_true, y_pred, class_labels) -> float:
    k = len(class_labels)
    return balanced_accuracy_from_codes(
        encode_labels(y_true, class_labels), encode_labels(y_pred, class_labels), k
    )


def accuracy(y_true, y_pred, class_labels) -> float:
    return accuracy_from_codes(
        encode_labels(y_true, class_labels), encode_labels(y_pred, class_labels)
    )


def expected_calibration_error(
    y_true, probs, class_labels, n_bins: int = DEFAULT_N_BINS
) -> float:
    return expected_calibration_error_from_codes(
        encode_labels(y_true, class_labels), probs, n_bins
    )


def brier_score(y_true, probs, class_labels) -> float:
    return brier_score_from_codes(encode_labels(y_true, class_labels), probs)


def risk_coverage_curve(y_true, probs, class_labels) -> tuple[np.ndarray, np.ndarray]:
    return risk_coverage_curve_from_codes(encode_labels(y_true, class_labels), probs)


def aurc(y_true, probs, class_labels) -> float:
    return aurc_from_codes(encode_labels(y_true, class_labels), probs)


def accuracy_at_coverage(
    y_true, probs, class_labels, coverages: tuple[float, ...] = DEFAULT_COVERAGES
) -> dict[str, float]:
    return accuracy_at_coverage_from_codes(
        encode_labels(y_true, class_labels), probs, coverages
    )
