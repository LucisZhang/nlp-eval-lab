"""Drift-rollup tests for the Tier B2 rung: config resolution, the one-gate a_to_b arm's
row-level semantics (escalate-to-B2 rate, structurally-empty human arm, id pairing, label-
space and answer-key guards), the arm-row schema staying uniform across policy families,
and the frozen tier-B1 pending slot.

Everything here is built from synthetic artifacts: the arms are pure functions of
(p_max, tau, labels), so no committed parquet is needed to pin their behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from triage_lab import drift_charts, predictions, threshold_opt

LABELS = ["card", "debt_collection", "mortgage"]


def _artifact(ids, y_true, y_pred, p_max, class_labels=LABELS):
    """A minimal PredictionsArtifact; `probs` is unused by the arm builders."""
    n = len(ids)
    return predictions.PredictionsArtifact(
        complaint_id=np.asarray(ids, dtype=np.int64),
        y_true=np.asarray(y_true, dtype=object),
        y_pred=np.asarray(y_pred, dtype=object),
        p_max=np.asarray(p_max, dtype=np.float64),
        probs=np.zeros((n, len(class_labels)), dtype=np.float64),
        class_labels=list(class_labels),
    )


def _record(run_id):
    return {"run_id": run_id}


# ---------------------------------------------------------------------------
# (a) config resolution + registration of the tier
# ---------------------------------------------------------------------------

def test_tier_b2_config_stems_are_the_four_yearly_runs_plus_the_iid_final():
    assert drift_charts.config_stem("tier_b2", "test_iid") == "tier_b2_distilbert_s0"
    assert [drift_charts.config_stem("tier_b2", s) for s in drift_charts.SLICE_ORDER[1:]] == [
        "tier_b2_distilbert_s0_test_drift_2023",
        "tier_b2_distilbert_s0_test_drift_2024",
        "tier_b2_distilbert_s0_test_drift_2025",
        "tier_b2_distilbert_s0_test_drift_2026h1",
    ]


def test_tier_b2_is_charted_and_tier_c_stays_out_of_the_ece_chart():
    assert "tier_b2" in drift_charts.TIER_ORDER
    assert drift_charts.CASCADE_TIER in drift_charts.TIER_CONFIGS
    assert set(drift_charts.ECE_TIERS) == {"tier_a", "tier_b2"}
    assert not set(drift_charts.ECE_TIERS) & set(drift_charts.TERMINAL_MODELS)
    # every tier and arm the summary orders must have a chart style
    assert set(drift_charts.TIER_ORDER) <= set(drift_charts.SERIES_STYLE)
    assert set(drift_charts.ARM_ORDER) <= set(drift_charts.ARM_THRESHOLD_KEYS)
    assert set(drift_charts.ARM_ORDER) == set(drift_charts.ARM_HLINE_STYLE)


def test_tier_b1_is_still_an_explicit_pending_slot_and_b2_is_not():
    pending = drift_charts.TIER_B_PENDING
    assert pending["tiers"] == ["tier_b1_modernbert"]
    assert pending["landed"] == ["tier_b2_distilbert"]
    assert pending["evidence_class"] == "pending"
    assert drift_charts.EVIDENCE_CLASS["series.logged.tier_b2"] == "measured"
    assert drift_charts.EVIDENCE_CLASS["series.logged.tier_b1"] == "pending"


# ---------------------------------------------------------------------------
# (b) the a_to_b arm's row-level semantics
# ---------------------------------------------------------------------------

def _ab_arm(tau, *, index_a=slice(None), index_b=None, art_b=None):
    art_a = _artifact([10, 11, 12, 13],
                      ["card", "card", "mortgage", "mortgage"],
                      ["card", "debt_collection", "mortgage", "card"],
                      [0.9, 0.8, 0.4, 0.3])
    if art_b is None:
        art_b = _artifact([10, 11, 12, 13],
                          ["card", "card", "mortgage", "mortgage"],
                          ["mortgage", "mortgage", "mortgage", "mortgage"],
                          [0.7, 0.7, 0.7, 0.7])
    if index_b is None:
        index_b = threshold_opt.restrict_to_ids(
            art_b, np.asarray(art_a.complaint_id)[index_a])
    return drift_charts.build_a_to_b_arm(
        "test_drift_2023", dataset=threshold_opt.DATASET_FULL_CAL, tau=tau,
        art_a=art_a, record_a=_record("a" * 64), index_a=index_a,
        art_b=art_b, record_b=_record("b" * 64), index_b=index_b,
        n_resamples=16, seed=7)


def test_escalation_rate_is_the_escalate_to_b_rate_and_the_human_arm_is_empty():
    row = _ab_arm(0.5)          # rows 0,1 answered by A; rows 2,3 go to B2
    assert row["policy"] == threshold_opt.FAMILY_A_TO_B
    assert row["terminal_model"] == "tier_b2"
    assert (row["n_answered_a"], row["n_escalated"]) == (2, 2)
    assert row["escalation_rate"]["point"] == pytest.approx(0.5)
    # tier_b_terminal: no row can reach a human, so the rate is 0.0 with a degenerate CI
    assert row["n_to_human"] == 0
    assert (row["human_rate"]["point"], row["human_rate"]["ci_lo"],
            row["human_rate"]["ci_hi"]) == (0.0, 0.0, 0.0)
    assert row["coverage_machine"]["point"] == 1.0
    # A gets 1 of its 2 answered rows right, B2 gets 2 of its 2 -> 3/4, and with no human
    # arm the *_system view (which credits human rows) cannot differ from the answered set
    assert row["accuracy_machine"]["point"] == pytest.approx(0.75)
    assert row["accuracy_system"]["point"] == row["accuracy_machine"]["point"]
    assert row["macro_f1_system"]["point"] == row["macro_f1_answered"]["point"]


def test_tau_1_sends_everything_to_b2_and_tau_0_sends_nothing():
    assert _ab_arm(1.0)["escalation_rate"]["point"] == pytest.approx(1.0)
    all_b = _ab_arm(1.0)
    assert all_b["accuracy_machine"]["point"] == pytest.approx(0.5)   # B2's own accuracy
    none_b = _ab_arm(0.0)
    assert none_b["escalation_rate"]["point"] == 0.0
    assert none_b["accuracy_machine"]["point"] == pytest.approx(0.5)  # A's own accuracy


def test_rows_are_paired_on_complaint_id_not_on_row_order():
    """A Tier B artifact in a different row order must give the SAME numbers."""
    straight = _ab_arm(0.5)
    shuffled = _artifact([13, 11, 10, 12],
                         ["mortgage", "card", "card", "mortgage"],
                         ["mortgage", "mortgage", "mortgage", "mortgage"],
                         [0.7, 0.7, 0.7, 0.7])
    assert _ab_arm(0.5, art_b=shuffled)["accuracy_machine"] == straight["accuracy_machine"]


def test_a_mismatched_answer_key_is_a_hard_failure():
    wrong_truth = _artifact([10, 11, 12, 13],
                            ["card", "card", "card", "mortgage"],   # row 2 disagrees
                            ["mortgage"] * 4, [0.7] * 4)
    with pytest.raises(ValueError, match="disagree on y_true"):
        _ab_arm(0.5, art_b=wrong_truth)


def test_a_mismatched_label_space_is_a_hard_failure():
    other_space = _artifact([10, 11, 12, 13], ["card"] * 4, ["card"] * 4, [0.7] * 4,
                            class_labels=["mortgage", "card", "debt_collection"])
    with pytest.raises(ValueError, match="different class label orders"):
        _ab_arm(0.5, art_b=other_space)


# ---------------------------------------------------------------------------
# (c) the escalation series stays one uniform schema across policy families
# ---------------------------------------------------------------------------

def test_a_to_b_rows_carry_exactly_the_a_to_human_row_schema():
    art_a = _artifact([10, 11, 12, 13],
                      ["card", "debt_collection", "mortgage", "card"],
                      ["card", "debt_collection", "mortgage", "mortgage"],
                      [0.9, 0.8, 0.4, 0.3])
    human = drift_charts.build_a_to_human_arm(
        "test_drift_2023", dataset=threshold_opt.DATASET_FULL_CAL, tau=0.5, art_a=art_a,
        record_a=_record("a" * 64), index=slice(None), n_resamples=16, seed=7)
    assert set(_ab_arm(0.5)) == set(human)
    assert set(drift_charts.BOOTSTRAPPED_KEYS) <= set(human)
