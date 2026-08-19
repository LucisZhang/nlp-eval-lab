"""CAL-only cost scenario for the external Frontier Forge R1b terminal tier.

This module deliberately does *not* turn a cross-task benchmark into a certified router
claim. Frontier Forge R1b consumes source-product/source-company metadata and produces a
structured action, while this repository predicts product class from complaint narrative.
There are no joint per-row CAL predictions. The only honest integration available from the
committed evidence is therefore a scenario: keep Tier A's measured CAL risk curve, assume
the external terminal's observed 4-QPS failure rate transfers independently to escalated
rows, and re-price every committed CAL threshold under the existing misroute cost.

The result is useful as a deployment hypothesis and explicitly ineligible to replace the
certified TEST-IID A->B headline. A real integration needs a narrative-only adapter, joint
per-row CAL predictions, a CAL-only threshold fit, and a once-only TEST evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from triage_lab import harness

REPO_ROOT = harness.REPO_ROOT
DEFAULT_HANDOFF = REPO_ROOT / "configs/external_tiers/frontier_forge_r1b_v1.json"
DEFAULT_RISK_CURVE = (
    REPO_ROOT
    / "results/risk_coverage/40513354503c8e8cec48666e2f62bb1e0cb5c04352f8c87965caf8b0c3cab0fc.json"
)
DEFAULT_COST_CONFIG = REPO_ROOT / "configs/cost_model_v2.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "results/external_tiers/frontier_forge_r1b_cal_scenario.json"

PINNED_HANDOFF_SHA256 = "5038a91ab883116ad8607bf442e0e73016d1fb42ef44623c37593a2a90253b0b"
SCHEMA_VERSION = "frontier-forge-cal-scenario-v1"
JSON_ROUND = 10
Z_95 = 1.959963984540054


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, JSON_ROUND)
    if isinstance(value, list):
        return [_round(item) for item in value]
    if isinstance(value, dict):
        return {key: _round(item) for key, item in value.items()}
    return value


def wilson_interval(failures: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial failure rate."""
    if total <= 0 or failures < 0 or failures > total:
        raise ValueError("failures and total do not define a binomial sample")
    p_hat = failures / total
    denominator = 1.0 + z**2 / total
    center = (p_hat + z**2 / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(p_hat * (1.0 - p_hat) / total + z**2 / (4.0 * total**2))
        / denominator
    )
    return center - half_width, center + half_width


def _validate_inputs(
    handoff: dict[str, Any], risk_curve: dict[str, Any], cost_config: dict[str, Any]
) -> None:
    if handoff.get("schema_version") != "frontier-forge-cascade-handoff-v1":
        raise ValueError("unsupported Frontier Forge handoff schema")
    contract = handoff.get("integration_contract", {})
    if contract.get("status") != "scenario-only-without-joint-cal-predictions":
        raise ValueError("handoff does not carry the required scenario-only guard")
    warning = str(contract.get("warning", ""))
    for phrase in ("source_product", "no joint per-row CAL predictions", "not a certified"):
        if phrase not in warning:
            raise ValueError(f"handoff warning lost required scope phrase: {phrase}")

    shared = handoff.get("shared_frozen_cal", {})
    if risk_curve.get("split") != "cal":
        raise ValueError("risk curve is not CAL")
    if shared.get("rows") != risk_curve.get("n_examples"):
        raise ValueError("handoff and risk curve disagree on CAL row count")
    if shared.get("source_split_sha256") != risk_curve.get("split_sha256"):
        raise ValueError("handoff and risk curve disagree on frozen CAL membership")

    service = handoff.get("service_profile", {})
    observed_failure = 1.0 - float(service.get("task_success"))
    scenario_failure = float(contract.get("terminal_failure_rate_for_cal_scenario"))
    if not math.isclose(observed_failure, scenario_failure, abs_tol=1e-12):
        raise ValueError("scenario failure rate is not the observed serving failure rate")
    service_cost = float(service.get("cost_per_1k_successful_tasks_usd")) / 1000.0
    scenario_cost = float(contract.get("terminal_cost_per_request_usd_conservative"))
    if not math.isclose(service_cost, scenario_cost, abs_tol=1e-15):
        raise ValueError("scenario request cost is not the measured serving cost")
    if service.get("stable") is not True or int(service.get("requests")) != 20:
        raise ValueError("expected the pinned stable 20-request 4-QPS serving point")

    params = cost_config.get("params", {})
    if float(params.get("c_misroute_usd")) <= 0:
        raise ValueError("cost config must define a positive misroute cost")


def _candidate(
    *,
    row: dict[str, Any],
    n_examples: int,
    terminal_failure_rate: float,
    terminal_cost_per_request_usd: float,
    c_misroute_usd: float,
) -> dict[str, Any]:
    n_covered = int(row["n_covered"])
    if n_covered < 0 or n_covered > n_examples:
        raise ValueError("threshold row has an invalid covered count")
    coverage = n_covered / n_examples
    if not math.isclose(coverage, float(row["coverage"]), abs_tol=5e-10):
        raise ValueError("threshold row coverage does not reproduce from counts")

    risk = 0.0 if n_covered == 0 else float(row["selective_risk"])
    reported_errors = n_covered * risk
    tier_a_errors = round(reported_errors)
    if not math.isclose(reported_errors, tier_a_errors, abs_tol=5e-6):
        raise ValueError("rounded risk no longer resolves to an integer error count")
    n_escalated = n_examples - n_covered
    expected_terminal_failures = n_escalated * terminal_failure_rate

    tier_a_misroute = tier_a_errors / n_examples * c_misroute_usd * 1000.0
    terminal_misroute = (
        expected_terminal_failures / n_examples * c_misroute_usd * 1000.0
    )
    terminal_compute = (
        n_escalated / n_examples * terminal_cost_per_request_usd * 1000.0
    )
    total = tier_a_misroute + terminal_misroute + terminal_compute
    return {
        "tau": row.get("tau"),
        "n_answered_tier_a": n_covered,
        "tier_a_coverage": coverage,
        "n_escalated_frontier_forge": n_escalated,
        "frontier_forge_escalation_rate": n_escalated / n_examples,
        "tier_a_selective_risk": risk if n_covered else None,
        "tier_a_errors": tier_a_errors,
        "expected_frontier_forge_failures": expected_terminal_failures,
        "cost_per_1k": {
            "tier_a_misroute_usd": tier_a_misroute,
            "frontier_forge_misroute_usd": terminal_misroute,
            "frontier_forge_compute_usd": terminal_compute,
            "total_usd": total,
        },
    }


def _select(
    rows: list[dict[str, Any]],
    *,
    n_examples: int,
    terminal_failure_rate: float,
    terminal_cost_per_request_usd: float,
    c_misroute_usd: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = [
        _candidate(
            row=row,
            n_examples=n_examples,
            terminal_failure_rate=terminal_failure_rate,
            terminal_cost_per_request_usd=terminal_cost_per_request_usd,
            c_misroute_usd=c_misroute_usd,
        )
        for row in rows
    ]
    selected = min(
        candidates,
        key=lambda item: (
            item["cost_per_1k"]["total_usd"],
            -item["tier_a_coverage"],
        ),
    )
    return selected, candidates


def build_result(
    *,
    handoff_path: Path = DEFAULT_HANDOFF,
    risk_curve_path: Path = DEFAULT_RISK_CURVE,
    cost_config_path: Path = DEFAULT_COST_CONFIG,
) -> dict[str, Any]:
    if sha256_file(handoff_path) != PINNED_HANDOFF_SHA256:
        raise ValueError("Frontier Forge handoff hash differs from the reviewed Phase 6 handoff")
    handoff = _load_json(handoff_path)
    risk_curve = _load_json(risk_curve_path)
    cost_config = yaml.safe_load(cost_config_path.read_text(encoding="utf-8"))
    if not isinstance(cost_config, dict):
        raise TypeError("cost config must be a mapping")
    _validate_inputs(handoff, risk_curve, cost_config)

    contract = handoff["integration_contract"]
    service = handoff["service_profile"]
    n_examples = int(risk_curve["n_examples"])
    terminal_failure_rate = float(contract["terminal_failure_rate_for_cal_scenario"])
    terminal_cost = float(contract["terminal_cost_per_request_usd_conservative"])
    c_misroute = float(cost_config["params"]["c_misroute_usd"])
    rows = [
        {"tau": None, "n_covered": 0, "coverage": 0.0, "selective_risk": None},
        *risk_curve["threshold_table"],
    ]
    selected, candidates = _select(
        rows,
        n_examples=n_examples,
        terminal_failure_rate=terminal_failure_rate,
        terminal_cost_per_request_usd=terminal_cost,
        c_misroute_usd=c_misroute,
    )
    tier_a_only = candidates[-1]
    forge_only = candidates[0]

    requests = int(service["requests"])
    failures = round(requests * terminal_failure_rate)
    wilson_lo, wilson_hi = wilson_interval(failures, requests)
    sensitivity = []
    for label, rate in (
        ("service_wilson95_low", wilson_lo),
        ("observed_point", terminal_failure_rate),
        ("service_wilson95_high", wilson_hi),
    ):
        point, _ = _select(
            rows,
            n_examples=n_examples,
            terminal_failure_rate=rate,
            terminal_cost_per_request_usd=terminal_cost,
            c_misroute_usd=c_misroute,
        )
        sensitivity.append(
            {
                "label": label,
                "assumed_terminal_failure_rate": rate,
                "selected_tau": point["tau"],
                "tier_a_coverage": point["tier_a_coverage"],
                "cost_per_1k_usd": point["cost_per_1k"]["total_usd"],
            }
        )

    total = selected["cost_per_1k"]["total_usd"]
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "scenario_only_not_certified",
        "headline": (
            "CAL-only cross-task scenario: the committed 512-point Tier A risk grid "
            f"selects tau={selected['tau']:.10f}, answers "
            f"{selected['tier_a_coverage']:.2%} in Tier A, and models "
            f"${total:.2f}/1k. This is not a classifier result or a certified cascade claim."
        ),
        "claim_gate": {
            "certified": False,
            "replaces_existing_headline": False,
            "reasons": [
                "Frontier Forge R1b solves structured ticket/action policy with source metadata; it is not a narrative-only product classifier.",
                "No joint per-row Frontier Forge predictions exist on this CAL split.",
                "The terminal failure rate is transferred from only 20 serving requests on a different task contract.",
                "The threshold is selected and priced on CAL only; no TEST result is produced.",
            ],
        },
        "inputs": {
            "frontier_forge_handoff": {
                "path": str(handoff_path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(handoff_path),
                "source_manifest_sha256": handoff["source_manifest_sha256"],
                "source_repository": handoff["source_repository"],
            },
            "tier_a_cal_risk_curve": {
                "path": str(risk_curve_path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(risk_curve_path),
                "run_id": risk_curve["run_id"],
                "split_sha256": risk_curve["split_sha256"],
                "n_examples": n_examples,
            },
            "cost_config": {
                "path": str(cost_config_path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(cost_config_path),
                "version": cost_config["version"],
                "c_misroute_usd": c_misroute,
                "c_misroute_evidence_class": cost_config["evidence_class"][
                    "params.c_misroute_usd"
                ],
            },
        },
        "frontier_forge_assumption": {
            "quality_run_id": handoff["quality"]["run_id"],
            "serving_run_id": service["run_id"],
            "artifact_sha256": service["artifact_sha256"],
            "serving_requests": requests,
            "observed_successes": requests - failures,
            "observed_failures": failures,
            "terminal_failure_rate": terminal_failure_rate,
            "within_task_wilson95_failure_rate": [wilson_lo, wilson_hi],
            "terminal_cost_per_request_usd": terminal_cost,
            "terminal_cost_evidence": "measured throughput at 4 QPS x owner-supplied GPU-hour rate",
            "transfer_warning": contract["warning"],
        },
        "optimization": {
            "domain": "CAL only",
            "grid": "coverage-zero endpoint plus the committed downsampled Tier A risk-coverage thresholds",
            "candidate_count": len(candidates),
            "tie_break": "lowest modeled cost, then largest Tier A coverage",
            "formula": (
                "1000 * (tier_a_errors + expected_frontier_forge_failures) / n * "
                "c_misroute + 1000 * n_escalated / n * frontier_forge_cost_per_request"
            ),
            "selected": selected,
            "references": {
                "tier_a_only": tier_a_only,
                "frontier_forge_only": forge_only,
                "selected_minus_tier_a_only_usd_per_1k": (
                    total - tier_a_only["cost_per_1k"]["total_usd"]
                ),
                "selected_minus_frontier_forge_only_usd_per_1k": (
                    total - forge_only["cost_per_1k"]["total_usd"]
                ),
            },
            "failure_rate_sensitivity": sensitivity,
        },
    }
    return _round(result)


def write_result(result: dict[str, Any], output: Path = DEFAULT_OUTPUT) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    output.write_text(text, encoding="utf-8")


def check_result(result: dict[str, Any], output: Path = DEFAULT_OUTPUT) -> None:
    if not output.is_file():
        raise FileNotFoundError(f"missing committed scenario result: {output}")
    committed = _load_json(output)
    if committed != result:
        raise RuntimeError(f"scenario result differs from committed artifact: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the deterministic result")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_result()
    if args.write:
        write_result(result, args.output)
    else:
        check_result(result, args.output)
    print(result["headline"])


if __name__ == "__main__":
    main()
