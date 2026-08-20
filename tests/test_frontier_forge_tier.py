"""The external forged tier remains a hash-pinned, scenario-only CAL calculation."""

from __future__ import annotations

import json

import pytest

from triage_lab import frontier_forge_tier


def test_committed_scenario_reproduces_exactly():
    result = frontier_forge_tier.build_result()
    committed = json.loads(frontier_forge_tier.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert result == committed


def test_scope_guard_prevents_a_false_certified_claim():
    result = frontier_forge_tier.build_result()
    assert result["status"] == "scenario_only_not_certified"
    assert result["claim_gate"]["certified"] is False
    assert result["claim_gate"]["replaces_existing_headline"] is False
    reasons = " ".join(result["claim_gate"]["reasons"])
    assert "not a narrative-only product classifier" in reasons
    assert "No joint per-row" in reasons
    assert "20 serving requests" in reasons
    assert "CAL only" in reasons


def test_selected_point_and_cost_components_are_pinned():
    result = frontier_forge_tier.build_result()
    selected = result["optimization"]["selected"]
    assert selected["tau"] == pytest.approx(0.8483569229)
    assert selected["tier_a_coverage"] == pytest.approx(0.325185117)
    assert selected["frontier_forge_escalation_rate"] == pytest.approx(0.674814883)
    assert selected["cost_per_1k"]["total_usd"] == pytest.approx(254.681803452)
    assert result["optimization"]["candidate_count"] == 513


def test_failure_uncertainty_materially_moves_the_scenario():
    result = frontier_forge_tier.build_result()
    sensitivity = {
        item["label"]: item for item in result["optimization"]["failure_rate_sensitivity"]
    }
    assert sensitivity["service_wilson95_low"]["tier_a_coverage"] == pytest.approx(
        0.0332635791
    )
    assert sensitivity["service_wilson95_high"]["tier_a_coverage"] == pytest.approx(
        0.7320631928
    )
    assert sensitivity["service_wilson95_low"]["cost_per_1k_usd"] < 60
    assert sensitivity["service_wilson95_high"]["cost_per_1k_usd"] > 700


def test_handoff_and_cal_membership_are_hash_pinned():
    result = frontier_forge_tier.build_result()
    assert (
        result["inputs"]["frontier_forge_handoff"]["sha256"]
        == frontier_forge_tier.PINNED_HANDOFF_SHA256
    )
    assert (
        result["inputs"]["tier_a_cal_risk_curve"]["split_sha256"]
        == "d7c24d6db05f337ac3fca1922b16c4324775e2007833af1eae684ddc37533c94"
    )


def test_wilson_interval_rejects_invalid_samples():
    with pytest.raises(ValueError, match="binomial"):
        frontier_forge_tier.wilson_interval(2, 1)
