"""Metric unit tests: pure-numpy implementations vs sklearn / hand-computed refs.

sklearn is the *reference oracle* here only (UPGRADE_PLAN.md §6.1); the runtime path
in `triage_lab.metrics` never imports it.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
)
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
)

from triage_lab import metrics

# ---------------------------------------------------------------------------
# A multiclass fixture with imbalance and a couple of confusions.
# ---------------------------------------------------------------------------

LABELS = ["card", "credit_reporting", "debt_collection", "mortgage"]

Y_TRUE = [
    "card", "card", "credit_reporting", "credit_reporting", "credit_reporting",
    "debt_collection", "debt_collection", "mortgage", "mortgage", "mortgage",
    "card", "credit_reporting", "debt_collection", "mortgage", "card",
]
Y_PRED = [
    "card", "credit_reporting", "credit_reporting", "credit_reporting", "debt_collection",
    "debt_collection", "mortgage", "mortgage", "mortgage", "card",
    "card", "credit_reporting", "debt_collection", "mortgage", "card",
]


def test_confusion_matrix_matches_sklearn():
    got = metrics.confusion_matrix(Y_TRUE, Y_PRED, LABELS)
    want = sk_confusion_matrix(Y_TRUE, Y_PRED, labels=LABELS)
    assert np.array_equal(got, want)


def test_per_class_f1_matches_sklearn():
    got = metrics.per_class_f1(Y_TRUE, Y_PRED, LABELS)
    want = f1_score(Y_TRUE, Y_PRED, labels=LABELS, average=None, zero_division=0)
    assert np.allclose(got, want)


def test_macro_f1_matches_sklearn():
    got = metrics.macro_f1(Y_TRUE, Y_PRED, LABELS)
    want = f1_score(Y_TRUE, Y_PRED, labels=LABELS, average="macro", zero_division=0)
    assert got == pytest.approx(want)


def test_balanced_accuracy_matches_sklearn():
    got = metrics.balanced_accuracy(Y_TRUE, Y_PRED, LABELS)
    want = balanced_accuracy_score(Y_TRUE, Y_PRED)
    assert got == pytest.approx(want)


# ---------------------------------------------------------------------------
# Degenerate class: a class present in y_true but never predicted.
# ---------------------------------------------------------------------------

def test_degenerate_absent_predicted_class():
    labels = ["x", "y", "z"]
    y_true = ["x", "y", "z", "z", "x", "y"]
    y_pred = ["x", "y", "y", "x", "x", "y"]  # "z" never predicted
    # per-class + macro F1
    got_pc = metrics.per_class_f1(y_true, y_pred, labels)
    want_pc = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    assert np.allclose(got_pc, want_pc)
    assert got_pc[labels.index("z")] == 0.0  # TP=0 -> F1 0
    assert metrics.macro_f1(y_true, y_pred, labels) == pytest.approx(
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    # balanced accuracy: z has true support, recall 0, still averaged in
    assert metrics.balanced_accuracy(y_true, y_pred, labels) == pytest.approx(
        balanced_accuracy_score(y_true, y_pred)
    )
    # confusion matrix still square over all labels
    got_cm = metrics.confusion_matrix(y_true, y_pred, labels)
    assert got_cm.shape == (3, 3)
    assert np.array_equal(got_cm, sk_confusion_matrix(y_true, y_pred, labels=labels))


# ---------------------------------------------------------------------------
# Calibration: Brier (hand) and ECE (hand, n_bins=2).
# ---------------------------------------------------------------------------

CAL_LABELS = ["a", "b", "c"]
# sample -> (probs row, true label). Confidences: 0.6, 0.8, 0.9, 0.4.
CAL_PROBS = np.array(
    [
        [0.60, 0.25, 0.15],  # pred a, conf 0.6, true a  -> correct
        [0.80, 0.10, 0.10],  # pred a, conf 0.8, true b  -> wrong
        [0.05, 0.05, 0.90],  # pred c, conf 0.9, true c  -> correct
        [0.40, 0.35, 0.25],  # pred a, conf 0.4, true a  -> correct
    ]
)
CAL_TRUE = ["a", "b", "c", "a"]


def test_brier_hand_computed():
    # sum-over-classes convention, mean over samples.
    # s1: 0.245  s2: 1.46  s3: 0.015  s4: 0.545  -> mean 0.56625
    got = metrics.brier_score(CAL_TRUE, CAL_PROBS, CAL_LABELS)
    assert got == pytest.approx(0.56625)


def test_ece_hand_computed_two_bins():
    # bins edges {0, 0.5, 1}. bin0 = {s4 conf0.4, acc1.0} -> 0.25*|1-0.4|=0.15
    # bin1 = {s1,s2,s3 conf .6,.8,.9 mean .766..., acc 2/3} -> 0.75*|0.6667-0.7667|=0.075
    got = metrics.expected_calibration_error(CAL_TRUE, CAL_PROBS, CAL_LABELS, n_bins=2)
    assert got == pytest.approx(0.225)


# ---------------------------------------------------------------------------
# Selective prediction: risk-coverage / AURC / accuracy@coverage (hand).
# ---------------------------------------------------------------------------

SEL_LABELS = ["neg", "pos"]
# Confidences (desc): 0.9 correct, 0.8 wrong, 0.7 correct, 0.6 correct, 0.55 wrong.
SEL_PROBS = np.array(
    [
        [0.90, 0.10],  # pred neg, conf 0.9, true neg -> correct
        [0.80, 0.20],  # pred neg, conf 0.8, true pos -> wrong
        [0.30, 0.70],  # pred pos, conf 0.7, true pos -> correct
        [0.60, 0.40],  # pred neg, conf 0.6, true neg -> correct
        [0.45, 0.55],  # pred pos, conf 0.55, true neg -> wrong
    ]
)
SEL_TRUE = ["neg", "pos", "pos", "neg", "neg"]


def test_risk_coverage_curve_hand():
    cov, risk = metrics.risk_coverage_curve(SEL_TRUE, SEL_PROBS, SEL_LABELS)
    assert np.allclose(cov, [1 / 5, 2 / 5, 3 / 5, 4 / 5, 5 / 5])
    assert np.allclose(risk, [0.0, 1 / 2, 1 / 3, 1 / 4, 2 / 5])


def test_aurc_hand():
    got = metrics.aurc(SEL_TRUE, SEL_PROBS, SEL_LABELS)
    want = np.mean([0.0, 1 / 2, 1 / 3, 1 / 4, 2 / 5])
    assert got == pytest.approx(want)


def test_accuracy_at_coverage_hand():
    got = metrics.accuracy_at_coverage(SEL_TRUE, SEL_PROBS, SEL_LABELS)
    # c=0.50 -> ceil(2.5)=3 top -> [correct,wrong,correct] = 2/3
    # c=0.80 -> ceil(4.0)=4 top -> 3/4 ; c=0.90 -> ceil(4.5)=5 -> 3/5 ; c=0.95 -> 5 -> 3/5
    assert got["0.50"] == pytest.approx(2 / 3)
    assert got["0.80"] == pytest.approx(3 / 4)
    assert got["0.90"] == pytest.approx(3 / 5)
    assert got["0.95"] == pytest.approx(3 / 5)


def test_confidence_tiebreak_is_stable_by_index():
    # Two samples share confidence 0.7; stable order keeps the earlier index first.
    labels = ["neg", "pos"]
    probs = np.array([[0.30, 0.70], [0.70, 0.30]])  # both conf 0.7
    true = ["pos", "neg"]  # both correct
    _cov, risk = metrics.risk_coverage_curve(true, probs, labels)
    assert np.allclose(risk, [0.0, 0.0])  # both correct regardless, sanity
    # accuracy@coverage 0.5 accepts exactly the first-by-index of the tie
    labels2 = ["neg", "pos"]
    probs2 = np.array([[0.30, 0.70], [0.70, 0.30]])
    true2 = ["neg", "neg"]  # index0 wrong (pred pos), index1 correct
    acc = metrics.accuracy_at_coverage(true2, probs2, labels2, coverages=(0.5,))
    # ceil(0.5*2)=1 -> accept index0 (tie -> lowest index) which is wrong -> 0.0
    assert acc["0.50"] == 0.0
