"""Frontier claims (Phase 4, partial — Tier B slots are declared pending).

UPGRADE_PLAN §4.2 names two headline claims, and this module is the only place either is
allowed to be computed:

    CLAIM 1 (vs all-LLM)    "At equal accuracy to the all-LLM policy, the router cuts cost
                             per 1,000 complaints by X% (measured)."
    CLAIM 2 (vs all-linear) "At equal cost to the all-linear policy, it raises macro-F1 by
                             Y points (measured)."

Both are *comparative* claims, so both are computed as PAIRED bootstraps on identical rows
with the frozen constants (`N_RESAMPLES`, `BOOTSTRAP_SEED`, 2.5/97.5 percentiles) and one
shared index draw per replicate, and both carry an explicit gate:

- Claim 1's "at equal accuracy" is a *non-inferiority* condition, not an equality: it holds
  if the paired accuracy delta is non-negative, or if its CI includes zero (statistically
  compatible with equality). Which of the two applies is stated in the output rather than
  collapsed into a pass/fail bit.
- Claim 2's "at equal cost" is the mirror condition on cost.

If a gate fails, the claim is reported as an honest diagnosis with the numbers that failed
it. Operating points are never re-picked to make a claim land — they come from the
committed threshold files, and this module cannot choose them.

**Which operating points.** v2 (`opv2`): thresholds derived on the isotonic-calibrated CAL
rung, i.e. in the same probability space as the TEST-IID deployment artifact. The v1
raw-space thresholds remain in the repo as the documented calibration-mismatch lesson but
are not the claim basis. Both routers appear as exhibits: the Haiku-terminal cascade
(`a_to_c_parsefail_human`, paired 5,000 rows) and the confidence-gated
`a_to_human` (full 104,443 rows).

**Macro-F1 views.** `macro_f1_system` credits human-routed rows as correct (`y_pred :=
y_true`) — the cost model's assumption — and is defined on all rows, so it can be paired.
`macro_f1_answered` is defined only over machine-answered rows, which differ between two
policies with different coverage; a paired CI for it is computed ONLY when both policies
answer exactly the same rows, and otherwise the two unpaired point values are reported with
that fact stated. Every entry carries `human_credit` counts so a reader can see which view
is load-bearing for which policy (`a_to_c` has no human rows on TEST-IID — Haiku logged
zero parse failures; `a_to_human` routes ~18-22%).

**Tier B is not evaluated here.** Every Tier B frontier slot (B1/B2 single-tier points, the
A→B cascade, the A→B→C cascade) is emitted as an explicit `pending_tier_b` placeholder
rather than omitted, so the exhibit shows its own holes instead of implying the frontier is
complete.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from triage_lab import cost_model, harness, metrics, router_sim

REPO_ROOT = harness.REPO_ROOT
DEFAULT_FRONTIER_DIR = REPO_ROOT / "results" / "frontier"

SCHEMA_VERSION = "frontier-v1"

CLAIM_VS_ALL_LLM = "claim_1_vs_all_llm"
CLAIM_VS_ALL_LINEAR = "claim_2_vs_all_linear"

ROUTER_CASCADE = "a_to_c_parsefail_human"
ROUTER_GATED = "a_to_human"

_round = cost_model._round

# (evaluation_set, router, {claim: baseline}) — the exhibits this module reports.
EXHIBITS = (
    (router_sim.EVAL_PAIRED, ROUTER_CASCADE,
     {CLAIM_VS_ALL_LLM: "c_only", CLAIM_VS_ALL_LINEAR: "a_only"}),
    (router_sim.EVAL_FULL, ROUTER_GATED,
     {CLAIM_VS_ALL_LINEAR: "a_only", CLAIM_VS_ALL_LLM: None}),
)

# Frontier points that need Tier B before they exist. Declared, never silently omitted.
PENDING_TIER_B = (
    {"point": "b1_only", "description": "ModernBERT-base single-tier point (3 seeds)"},
    {"point": "b2_only", "description": "DistilBERT deployment single-tier point"},
    {"point": "a_to_b", "description": "Tier A gate -> Tier B terminal cascade"},
    {"point": "a_to_b_to_c",
     "description": "Tier A gate -> Tier B gate -> Tier C terminal cascade (full §4.2)"},
)
PENDING_STATUS = "pending_tier_b"

EVIDENCE_CLASSES = {
    "predictions": "measured (frozen prediction artifacts, gate-verified)",
    "tier_c_api_cost": "measured (per-call receipts: real tokens x published prices)",
    "tier_a_api_cost": "estimated (amortized_zero: CPU inference charged $0 by decision)",
    "c_misroute_usd": "estimated (business default, UPGRADE_PLAN §4.2)",
    "c_human_usd": "estimated (business default, UPGRADE_PLAN §4.2)",
    "human_resolution_correctness": "assumed (P(error|human) = 0)",
}

# ---------------------------------------------------------------------------
# Gate classification
# ---------------------------------------------------------------------------
# A claim is certified only when every metric it depends on is FAVORABLE AND SIGNIFICANT,
# decided by the CI bound facing zero — never by the point estimate's sign. The earlier
# formulation ("point favourable OR CI includes zero") was not a test at all: it certified
# on a coin-flip point estimate whose interval spanned zero, and it treated "we could not
# measure a difference" as evidence of equality. Three outcomes, no fourth:
#
#   favorable_significant : the favourable CI bound clears zero (ci_lo > 0 for a
#                           higher-is-better delta, ci_hi < 0 for lower-is-better).
#   not_established       : the interval spans zero — the honest "we did not measure it".
#   adverse_significant   : the whole interval is on the unfavourable side.
#
# Strict inequalities on purpose: a system compared against itself produces a degenerate
# [0, 0] band, which must read as not_established, not as a certified win.
CLASS_FAVORABLE = "favorable_significant"
CLASS_NOT_ESTABLISHED = "not_established"
CLASS_ADVERSE = "adverse_significant"

HIGHER_IS_BETTER = "higher_is_better"
LOWER_IS_BETTER = "lower_is_better"

# Which metrics each claim is certified on. Both must be favorable_significant.
CLAIM_GATED_METRICS = {
    CLAIM_VS_ALL_LLM: (
        ("delta_accuracy_system", HIGHER_IS_BETTER),
        ("pct_cost_reduction", HIGHER_IS_BETTER),
    ),
    CLAIM_VS_ALL_LINEAR: (
        ("delta_cost_per_1k", LOWER_IS_BETTER),
        ("delta_macro_f1_system", HIGHER_IS_BETTER),
    ),
}

PAIRING_NOTE = (
    "Every delta is router - baseline on identical rows, with one shared bootstrap index "
    "draw per replicate (frozen seed), so cost, accuracy and macro-F1 deltas for a given "
    "comparison are read off the SAME resamples. Marginal per-policy CIs overlap even "
    "under strict dominance and are never used for a comparison."
)

ANSWERED_F1_NOTE = (
    "macro_f1_answered is defined over machine-answered rows only. Two policies with "
    "different coverage answer different rows, so no paired CI exists for their "
    "difference; in that case the two unpaired point values are reported and paired=false."
)

GATE_NOTE = (
    "Each gated metric is classified from the CI bound facing zero at margin 0: "
    "favorable_significant (favourable bound strictly clears zero), not_established (CI "
    "spans zero), adverse_significant (whole CI unfavourable). A claim is CERTIFIED only "
    "when every gated metric is favorable_significant; anything else is reported as an "
    "honest diagnosis carrying the per-metric classification. Note this is STRICTER than "
    "UPGRADE_PLAN §4.2's 'at equal accuracy/cost' phrasing: a certified claim here has "
    "demonstrated a significant improvement on both axes, not parity on one of them."
)

# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------

def _ci(point: float, replicates: np.ndarray) -> dict:
    lo, hi = np.percentile(replicates, [harness.CI_LOWER_PCT, harness.CI_UPPER_PCT])
    return {
        "point": _round(float(point)),
        "ci_lo": _round(float(lo)),
        "ci_hi": _round(float(hi)),
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }


def paired_cost_claim(router, baseline, cfg: cost_model.CostConfig, *,
                      n_resamples: int = harness.N_RESAMPLES,
                      seed: int = harness.BOOTSTRAP_SEED) -> dict:
    """Cost/1k delta AND percent cost reduction, off the same resamples.

    The percentage is bootstrapped as a per-replicate RATIO of the two resampled means,
    not derived from the two marginal point estimates: a ratio of means is not the mean of
    ratios, and a CI built by dividing two independent intervals would be both wrong and
    wider than the truth.
    """
    per_router = cost_model.per_example_cost(
        router.correct_for_cost, router.api_cost_usd, router.to_human,
        c_misroute=cfg.c_misroute_usd, c_human=cfg.c_human_usd)
    per_baseline = cost_model.per_example_cost(
        baseline.correct_for_cost, baseline.api_cost_usd, baseline.to_human,
        c_misroute=cfg.c_misroute_usd, c_human=cfg.c_human_usd)

    reps = cost_model.resample_means(
        {"router": per_router, "baseline": per_baseline},
        scale=cost_model.PER_N_COMPLAINTS, n_resamples=n_resamples, seed=seed)
    mean_router = float(per_router.mean()) * cost_model.PER_N_COMPLAINTS
    mean_baseline = float(per_baseline.mean()) * cost_model.PER_N_COMPLAINTS
    if mean_baseline <= 0.0:
        raise ValueError(
            "the baseline costs $0 per 1,000 complaints; a percent cost reduction against "
            "a free baseline is undefined"
        )
    n_degenerate = int(np.count_nonzero(reps["baseline"] <= 0.0))
    if n_degenerate:
        # Only reachable on a tiny slice whose baseline has a handful of errors, where a
        # resample can miss all of them. Refusing beats silently dropping replicates (which
        # would narrow the interval) or emitting inf.
        raise ValueError(
            f"{n_degenerate}/{n_resamples} bootstrap resamples give a zero-cost baseline, "
            "so the percent-reduction ratio is undefined on them; this slice is too small "
            "or too easy to support a percentage claim"
        )

    delta_reps = reps["router"] - reps["baseline"]
    pct_reps = 100.0 * (reps["baseline"] - reps["router"]) / reps["baseline"]
    return {
        "cost_per_1k_router": _round(mean_router),
        "cost_per_1k_baseline": _round(mean_baseline),
        "delta_cost_per_1k": _ci(mean_router - mean_baseline, delta_reps),
        "pct_cost_reduction": _ci(
            100.0 * (mean_baseline - mean_router) / mean_baseline, pct_reps),
    }


def paired_accuracy_delta(router, baseline, *, n_resamples: int = harness.N_RESAMPLES,
                          seed: int = harness.BOOTSTRAP_SEED) -> dict:
    """System-accuracy delta (human rows credited), paired on identical rows."""
    a = router.correct_for_cost.astype(np.float64)
    b = baseline.correct_for_cost.astype(np.float64)
    reps = cost_model.resample_means({"a": a, "b": b}, scale=1.0,
                                     n_resamples=n_resamples, seed=seed)
    return _ci(float(a.mean() - b.mean()), reps["a"] - reps["b"])


def _zero_probs(n: int, k: int) -> np.ndarray:
    """Placeholder probability block for label-only paired metrics.

    `harness.paired_bootstrap_delta` indexes `probs` for every metric even though the
    label-only resolvers (macro_f1, accuracy) ignore it. Passing zeros keeps the shared
    RNG contract intact without inventing probabilities the policies do not have: a
    routed system emits a label, not a calibrated distribution.
    """
    return np.zeros((n, k), dtype=np.float64)


def paired_macro_f1_delta(y_true, pred_a, pred_b, class_labels, *,
                          n_resamples: int = harness.N_RESAMPLES,
                          seed: int = harness.BOOTSTRAP_SEED) -> dict:
    """Paired macro-F1 delta via the harness's own paired bootstrap (per-replicate refit).

    Reused rather than reimplemented: `harness.paired_bootstrap_delta` already draws one
    `integers(0, n, size=n)` per replicate from `default_rng(seed)` and recomputes the
    metric on both systems from those indices — the same contract the cost bootstrap uses,
    so for a given comparison the macro-F1 and cost replicates share index draws.
    """
    n, k = len(y_true), len(class_labels)
    out = harness.paired_bootstrap_delta(
        y_true, pred_a, pred_b, _zero_probs(n, k), _zero_probs(n, k),
        "macro_f1", class_labels, n_resamples=n_resamples, seed=seed,
    )
    return {
        "point": _round(out["delta"]),
        "ci_lo": _round(out["ci_lo"]),
        "ci_hi": _round(out["ci_hi"]),
        "excludes_zero": bool(out["ci_lo"] > 0.0 or out["ci_hi"] < 0.0),
    }


# ---------------------------------------------------------------------------
# Claim assembly
# ---------------------------------------------------------------------------

def classify(band: dict, direction: str) -> str:
    """Three-way CI classification of one delta. See the CLASS_* constants."""
    if direction not in (HIGHER_IS_BETTER, LOWER_IS_BETTER):
        raise ValueError(f"unknown direction {direction!r}")
    lo, hi = band["ci_lo"], band["ci_hi"]
    if direction == HIGHER_IS_BETTER:
        if lo > 0.0:
            return CLASS_FAVORABLE
        if hi < 0.0:
            return CLASS_ADVERSE
    else:
        if hi < 0.0:
            return CLASS_FAVORABLE
        if lo > 0.0:
            return CLASS_ADVERSE
    return CLASS_NOT_ESTABLISHED


def build_gate(claim: str, bands: dict) -> dict:
    """Classify every metric this claim is certified on, and decide certification."""
    gated = {}
    for name, direction in CLAIM_GATED_METRICS[claim]:
        band = bands[name]
        gated[name] = {
            "direction": direction,
            "classification": classify(band, direction),
            "point": band["point"],
            "ci_lo": band["ci_lo"],
            "ci_hi": band["ci_hi"],
        }
    classifications = {m["classification"] for m in gated.values()}
    return {
        "criterion": GATE_NOTE,
        "metrics": gated,
        "certified": classifications == {CLASS_FAVORABLE},
        "any_adverse": CLASS_ADVERSE in classifications,
    }


def build_claim(claim: str, router, baseline, cfg: cost_model.CostConfig, class_labels, *,
                evaluation_set: str, n_resamples: int = harness.N_RESAMPLES,
                seed: int = harness.BOOTSTRAP_SEED) -> dict:
    """One claim: cost + accuracy + macro-F1 deltas, its gate, and its verdict sentence."""
    router_sim._require_aligned(router, baseline)
    cost = paired_cost_claim(router, baseline, cfg, n_resamples=n_resamples, seed=seed)
    accuracy = paired_accuracy_delta(router, baseline, n_resamples=n_resamples, seed=seed)
    f1_system = paired_macro_f1_delta(
        router.y_true, router.y_pred_system, baseline.y_pred_system, class_labels,
        n_resamples=n_resamples, seed=seed)

    same_answered = np.array_equal(router.machine, baseline.machine)
    if same_answered and int(router.machine.sum()):
        mask = router.machine
        f1_answered = {
            "paired": True,
            **paired_macro_f1_delta(router.y_true[mask], router.y_pred[mask],
                                    baseline.y_pred[mask], class_labels,
                                    n_resamples=n_resamples, seed=seed),
        }
    else:
        f1_answered = {
            "paired": False,
            "note": ANSWERED_F1_NOTE,
            "router_macro_f1_answered": _round(metrics.macro_f1(
                router.y_true[router.machine], router.y_pred[router.machine],
                class_labels)) if int(router.machine.sum()) else None,
            "baseline_macro_f1_answered": _round(metrics.macro_f1(
                baseline.y_true[baseline.machine], baseline.y_pred[baseline.machine],
                class_labels)) if int(baseline.machine.sum()) else None,
        }

    bands = {
        "delta_accuracy_system": accuracy,
        "delta_macro_f1_system": f1_system,
        "delta_cost_per_1k": cost["delta_cost_per_1k"],
        "pct_cost_reduction": cost["pct_cost_reduction"],
    }
    gate = build_gate(claim, bands)
    mcnemar_rows = router.machine & baseline.machine
    n_both = int(mcnemar_rows.sum())
    mcnemar = harness.mcnemar(router.y_true[mcnemar_rows], router.y_pred[mcnemar_rows],
                              baseline.y_pred[mcnemar_rows]) if n_both else {
        "b": 0, "c": 0, "n_discordant": 0, "p_value": 1.0}

    return {
        "claim": claim,
        "evaluation_set": evaluation_set,
        "router": router.name,
        "baseline": baseline.name,
        "n_examples": len(router),
        "tau": router.gate.get("tau"),
        "coverage_a": _round(float(router.gate.get("coverage_a", np.nan))),
        **cost,
        "delta_accuracy_system": accuracy,
        "delta_macro_f1_system": f1_system,
        "macro_f1_answered": f1_answered,
        "mcnemar_machine_rows": {
            **{k: (float(v) if k == "p_value" else int(v)) for k, v in mcnemar.items()},
            "n_both_machine": n_both,
        },
        "gate": gate,
        "human_credit": {
            "router_n_to_human": int(router.to_human.sum()),
            "baseline_n_to_human": int(baseline.to_human.sum()),
            "note": router_sim.HUMAN_CREDIT_NOTE,
        },
        "verdict": _verdict_sentence(claim, router, baseline, bands, gate),
    }


_METRIC_PHRASE = {
    "delta_accuracy_system": "system accuracy",
    "delta_macro_f1_system": "system macro-F1",
    "delta_cost_per_1k": "cost per 1,000 complaints",
    "pct_cost_reduction": "percent cost reduction",
}


def _band_text(name: str, bands: dict) -> str:
    b = bands[name]
    unit = "%" if name == "pct_cost_reduction" else ""
    return (f"{_METRIC_PHRASE[name]} {b['point']:+.4f}{unit} "
            f"[{b['ci_lo']:.4f}, {b['ci_hi']:.4f}]")


def _verdict_sentence(claim, router, baseline, bands, gate) -> str:
    """The quotable claim, or the diagnosis — direction taken from the CLASSIFICATION.

    A favourable sentence ("cuts cost by X%", "raises macro-F1 by Y") is emitted only when
    every gated metric is favorable_significant. That is what stops a negative point
    estimate from being narrated as a "cut of -10%": an adverse result gets an adverse
    sentence and a spanning interval gets a directional-only one.
    """
    where = f"n={len(router)} {router.evaluation_set}"
    adverse = [n for n, m in gate["metrics"].items()
               if m["classification"] == CLASS_ADVERSE]
    unresolved = [n for n, m in gate["metrics"].items()
                  if m["classification"] == CLASS_NOT_ESTABLISHED]

    if adverse:
        detail = "; ".join(_band_text(n, bands) for n in adverse)
        return (
            f"ADVERSE ({where}): {router.name} is significantly WORSE than "
            f"{baseline.name} on {detail}. No {claim} claim can be made."
        )
    if unresolved:
        detail = "; ".join(_band_text(n, bands) for n in unresolved)
        settled = "; ".join(
            _band_text(n, bands) for n, m in gate["metrics"].items()
            if m["classification"] == CLASS_FAVORABLE)
        prefix = f"favourable and significant on {settled}; " if settled else ""
        return (
            f"NOT ESTABLISHED ({where}): {prefix}but {detail} — the interval spans zero, "
            f"so this is directional only, not a {claim} claim."
        )
    if claim == CLAIM_VS_ALL_LLM:
        pct, acc = bands["pct_cost_reduction"], bands["delta_accuracy_system"]
        return (
            f"At significantly HIGHER accuracy than the all-LLM policy (paired system "
            f"accuracy delta {acc['point']:+.4f} [{acc['ci_lo']:.4f}, {acc['ci_hi']:.4f}]), "
            f"the {router.name} router cuts cost per 1,000 complaints by "
            f"{pct['point']:.2f}% [{pct['ci_lo']:.2f}%, {pct['ci_hi']:.2f}%] — measured on "
            f"{where}."
        )
    cost, f1 = bands["delta_cost_per_1k"], bands["delta_macro_f1_system"]
    return (
        f"At significantly LOWER cost than the all-linear policy (paired cost delta "
        f"${cost['point']:+.2f}/1k [{cost['ci_lo']:.2f}, {cost['ci_hi']:.2f}]), the "
        f"{router.name} router raises system macro-F1 by {f1['point']:+.4f} "
        f"[{f1['ci_lo']:.4f}, {f1['ci_hi']:.4f}] — measured on {where}."
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_frontier(cfg: cost_model.CostConfig, *,
                   preds_dir=router_sim.DEFAULT_PREDS_DIR,
                   results_path=router_sim.DEFAULT_RESULTS_PATH,
                   thresholds_dir=router_sim.DEFAULT_THRESHOLDS_DIR,
                   op_version: str = router_sim.OP_V2,
                   n_resamples: int = harness.N_RESAMPLES,
                   seed: int = harness.BOOTSTRAP_SEED) -> dict:
    """Both claims for both routers, the dominance census, and the Tier B placeholders."""
    inputs = router_sim.load_test_inputs(preds_dir, results_path)
    cal = router_sim.load_cal_thresholds(thresholds_dir, cost_sha256=cfg.sha256,
                                         results_path=results_path,
                                         derivation=op_version, cost_config=cfg,
                                         preds_dir=preds_dir)
    class_labels = list(inputs.art_a.class_labels)
    builders = {router_sim.EVAL_FULL: router_sim.build_full_policies,
                router_sim.EVAL_PAIRED: router_sim.build_paired_policies}

    claims = []
    for evaluation_set, router_name, baselines in EXHIBITS:
        policies = {p.name: p for p in builders[evaluation_set](inputs, cal)}
        router = policies[router_name]
        for claim, baseline_name in baselines.items():
            if baseline_name is None:
                claims.append({
                    "claim": claim,
                    "evaluation_set": evaluation_set,
                    "router": router_name,
                    "baseline": None,
                    "status": "not_applicable",
                    "reason": (
                        "no Tier C policy is defined on the full TEST-IID slice: the Haiku "
                        "run covers a frozen 5,000-row subsample, so an all-LLM baseline "
                        "exists only on the paired subset"
                    ),
                })
                continue
            claims.append(build_claim(
                claim, router, policies[baseline_name], cfg, class_labels,
                evaluation_set=evaluation_set, n_resamples=n_resamples, seed=seed))

    evaluations = router_sim.build_all(
        cfg, preds_dir=preds_dir, results_path=results_path,
        thresholds_dir=thresholds_dir, n_resamples=n_resamples, seed=seed,
        op_version=op_version)
    census = {
        f"{name}/{a}": router_sim._dominance_row(ev, a)
        for name, ev in evaluations.items()
        for a in dict.fromkeys(d["a"] for d in ev["paired_deltas"])
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "operating_point_version": op_version,
        "cost_config": cost_model.config_block(cfg),
        "thresholds": {
            f"{family}/{dataset}": {
                "tau_star": entry.tau_star,
                "cal_target_coverage_a": _round(entry.target_coverage_a),
                "tier_a_cal_run_id": entry.tier_a_run_id,
                "tier_a_cal_config": entry.tier_a_config,
                "source_file": entry.source_file,
                "source_sha256": entry.source_sha256,
            }
            for (family, dataset), entry in sorted(cal.items())
        },
        "inputs": inputs.blocks,
        "claims": claims,
        "dominance_census": {
            "criterion": (
                "router cheaper than the baseline with a paired bootstrap CI on the "
                "cost/1k difference excluding zero"
            ),
            "model_baselines": sorted(router_sim.MODEL_BASELINES),
            "note": router_sim.DOMINANCE_NOTE,
            "by_router": census,
        },
        "pending": [
            {**slot, "status": PENDING_STATUS,
             "blocked_on": "Tier B fine-tunes are not yet trained (UPGRADE_PLAN Phase 2)"}
            for slot in PENDING_TIER_B
        ],
        "evidence_classes": EVIDENCE_CLASSES,
        "bootstrap": {
            "n_resamples": int(n_resamples),
            "seed": int(seed),
            "method": (
                f"percentile [{harness.CI_LOWER_PCT}, {harness.CI_UPPER_PCT}] over "
                "resampled example indices; one integers(0, n, size=n) draw per replicate, "
                "shared across the two systems of a comparison"
            ),
        },
        "notes": {
            "pairing": PAIRING_NOTE,
            "gate": GATE_NOTE,
            "answered_macro_f1": ANSWERED_F1_NOTE,
            "human_credit": router_sim.HUMAN_CREDIT_NOTE,
            "transfer": router_sim.TRANSFER_NOTE,
            "parse_failures": router_sim.PARSE_FAIL_NOTE,
        },
    }


def result_filename(cfg: cost_model.CostConfig, op_version: str) -> str:
    suffix = router_sim.OP_VERSIONS[op_version]["suffix"] or "__opv1"
    return f"frontier{suffix}__cost-{cfg.sha256[:8]}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m triage_lab.frontier")
    parser.add_argument("--preds-dir", type=Path, default=router_sim.DEFAULT_PREDS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FRONTIER_DIR)
    parser.add_argument("--cost-config", type=Path, default=cost_model.DEFAULT_COST_CONFIG)
    parser.add_argument("--results", type=Path, default=router_sim.DEFAULT_RESULTS_PATH)
    parser.add_argument("--thresholds-dir", type=Path,
                        default=router_sim.DEFAULT_THRESHOLDS_DIR)
    parser.add_argument("--op-version", choices=sorted(router_sim.OP_VERSIONS),
                        default=router_sim.OP_V2)
    args = parser.parse_args(argv)

    cfg = cost_model.load_cost_config(args.cost_config)
    obj = build_frontier(cfg, preds_dir=args.preds_dir, results_path=args.results,
                         thresholds_dir=args.thresholds_dir, op_version=args.op_version)
    path = cost_model.write_result_json(
        obj, args.out_dir / result_filename(cfg, args.op_version))

    print(f"frontier ({args.op_version}) -> {path}")
    for claim in obj["claims"]:
        if claim.get("status") == "not_applicable":
            print(f"\n[{claim['claim']}] {claim['router']} vs (none): "
                  f"{claim['reason'][:70]}...")
            continue
        print(f"\n[{claim['claim']}] {claim['router']} vs {claim['baseline']} "
              f"({claim['evaluation_set']}, n={claim['n_examples']})")
        print(f"  cost/1k {claim['cost_per_1k_router']:.2f} vs "
              f"{claim['cost_per_1k_baseline']:.2f}  "
              f"delta {claim['delta_cost_per_1k']['point']:+.2f} "
              f"[{claim['delta_cost_per_1k']['ci_lo']:.2f}, "
              f"{claim['delta_cost_per_1k']['ci_hi']:.2f}]  "
              f"pct {claim['pct_cost_reduction']['point']:+.2f}% "
              f"[{claim['pct_cost_reduction']['ci_lo']:.2f}, "
              f"{claim['pct_cost_reduction']['ci_hi']:.2f}]")
        print(f"  acc {claim['delta_accuracy_system']['point']:+.4f} "
              f"[{claim['delta_accuracy_system']['ci_lo']:.4f}, "
              f"{claim['delta_accuracy_system']['ci_hi']:.4f}]  "
              f"macroF1_sys {claim['delta_macro_f1_system']['point']:+.4f} "
              f"[{claim['delta_macro_f1_system']['ci_lo']:.4f}, "
              f"{claim['delta_macro_f1_system']['ci_hi']:.4f}]  "
              f"certified={claim['gate']['certified']}")
        for name, m in claim["gate"]["metrics"].items():
            print(f"    gate {name:24s} {m['classification']}")
        print(f"  {claim['verdict']}")
    print(f"\npending Tier B slots: {[p['point'] for p in obj['pending']]}")
    return 0


if __name__ == "__main__":
    import sys

    from triage_lab.frontier import main as _main

    sys.exit(_main())
