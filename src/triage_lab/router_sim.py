"""Router simulator on TEST-IID (Phase 4 task 4).

Task 3 chose the Tier A confidence gate on CAL. This module spends that choice: it applies
the CAL-fit thresholds to the frozen TEST-IID artifacts, prices every policy under
``configs/cost_model_v1.yaml``, and reports paired comparisons against each single-tier
baseline. It is the final reported evaluation of Phase 4 — and the first and only time the
router's thresholds meet TEST data.

Offline throughout: no model is run and no API is called. Every number comes from frozen
prediction artifacts plus the committed Tier C receipts. `results/runs.jsonl` is read-only.

**Owner decisions this module implements (2026-08-07), binding:**

- The headline router is the **Haiku-terminal cascade** ``a_to_c_parsefail_human``: Tier A
  answers above tau, otherwise Tier C answers and its verdict is final, except that a
  receipt marked ``parse_failed`` routes to a human. Haiku logged zero parse failures on
  TEST-IID, so that arm is empty here — reported as a robustness fact, not hidden. The
  ``c_human`` sensitivity therefore lives entirely in the ``a_to_human`` rung.
- No Sonnet-terminal variant (deferred to the drift chapter).
- **Threshold transfer is PRIMARY**: the CAL tau* constants are applied directly to the
  TEST artifact's ``p_max`` and the *realized* coverage is whatever it turns out to be.
  Coverage-matched transfer (pick tau on TEST to hit the CAL target coverage) appears only
  as a clearly-labeled secondary block.

**The p_max-space caveat that makes the above non-trivial** (``TRANSFER_NOTE``): the CAL
rungs are ``calibration: none`` (raw probabilities) while the TEST-IID final config is
isotonic-calibrated on CAL. A tau fit in the raw space and applied in the calibrated space
is not the same operating point, and the realized-vs-target coverage gap in the output is
the direct measurement of that mismatch. That is why both transfer modes are reported: the
primary is the honest "what happens if you ship the constant", the secondary is "what the
intended operating point would have cost".

**Accuracy accounting.** Rows sent to a human are counted correct — the cost model's
``P(error|human)=0`` assumption — which flatters any policy with a human arm. So every
policy reports BOTH: ``accuracy_system`` / ``macro_f1_system`` (human rows credited,
``y_pred := y_true``) and ``accuracy_machine`` / ``macro_f1_answered`` (machine-answered
rows only, no assumption). A claim that moves between those two views is a claim about the
assumption, not about the router.

**Comparisons** are paired on identical rows: cost/1k deltas and system-accuracy deltas
with the frozen shared-index bootstrap, plus exact McNemar on the subset of rows where
BOTH policies answered by machine (the only rows where two label decisions can disagree).
Sign convention is fixed and stated in the output: ``delta = A - B``, so a NEGATIVE cost
delta and a POSITIVE accuracy delta both favor A.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from triage_lab import cost_model, harness, metrics, predictions, threshold_opt

REPO_ROOT = harness.REPO_ROOT
DEFAULT_PREDS_DIR = harness.DEFAULT_PREDS_DIR
DEFAULT_RESULTS_PATH = harness.DEFAULT_RESULTS_PATH
DEFAULT_ROUTER_DIR = REPO_ROOT / "results" / "router_sim"
DEFAULT_THRESHOLDS_DIR = threshold_opt.DEFAULT_THRESHOLDS_DIR

SCHEMA_VERSION = "router-sim-v1"

# The frozen TEST-IID runs this simulator consumes, by config stem.
TIER_A_TEST_CONFIG = "tier_a_logreg_test_iid"
TIER_A_CNB_TEST_CONFIG = "tier_a_cnb_test_iid"
TIER_C_TEST_CONFIG = "tier_c_haiku_zeroshot_test_iid"
ALLOWED_SPLITS = {"test_iid"}

EVAL_FULL = "full_test_iid"
EVAL_PAIRED = "paired_subset"

TRANSFER_PRIMARY = "threshold_transfer"
TRANSFER_SECONDARY = "coverage_matched"

# The exact CAL threshold constants the router expects to find. Stated as a set so a
# missing file fails at load time with a readable message instead of as a KeyError inside
# a policy builder, and so an unexpected extra key cannot be silently ignored.
# Operating-point versions: which CAL derivation the router's tau* constants come from.
# v1 = raw-CAL thresholds crossing into the calibrated TEST space (kept as the documented
# lesson); v2 = thresholds derived in the deployment calibration space. Output files are
# suffixed per version so v1 evidence is never rewritten.
OP_V1 = threshold_opt.DERIVATION_V1
OP_V2 = threshold_opt.DERIVATION_V2
OP_VERSIONS: dict[str, dict] = {
    OP_V1: {"tier_a_cal_config": threshold_opt.PRIMARY_TIER_A_CONFIG, "suffix": ""},
    OP_V2: {"tier_a_cal_config": threshold_opt.V2_TIER_A_CAL_CONFIG, "suffix": "__opv2"},
}

EXPECTED_THRESHOLD_KEYS = frozenset({
    (threshold_opt.FAMILY_A_TO_HUMAN, threshold_opt.DATASET_FULL_CAL),
    (threshold_opt.FAMILY_A_TO_HUMAN, threshold_opt.DATASET_PAIRED),
    (threshold_opt.FAMILY_A_TO_C, threshold_opt.DATASET_PAIRED),
})

# `target_coverage_a` is written rounded to JSON_ROUND (10 dp), so the count check has to
# tolerate that rounding and nothing more.
THRESHOLD_COUNT_TOL = 1e-9

# The published objective is rounded to JSON_ROUND (10 dp), so the replay compares against
# that rounding and nothing looser.
REPLAY_COST_TOL = 1e-9

JSON_ROUND = cost_model.JSON_ROUND
_round = cost_model._round
_round_ci = cost_model._round_ci

TRANSFER_NOTE = (
    "PRIMARY transfer is THRESHOLD TRANSFER: the CAL-fit tau* constant is applied "
    "unchanged to the TEST artifact's p_max and the realized coverage is reported as "
    "whatever it turns out to be. This is what shipping the constant actually does. The "
    "CAL rungs are calibration: none (raw probabilities) while tier_a_logreg_test_iid is "
    "calibration: isotonic (fit on CAL), so the tau crosses a probability-space boundary; "
    "the realized-vs-target coverage gap below is the direct measurement of that mismatch. "
    "The coverage-matched block (tau re-chosen on TEST p_max to hit the CAL target "
    "coverage) is SECONDARY sensitivity only — it uses TEST data to pick tau and so is not "
    "a deployable procedure without a held-out calibration slice."
)

HUMAN_CREDIT_NOTE = (
    "accuracy_system and macro_f1_system credit every human-routed row as correct "
    "(y_pred := y_true), matching the cost model's P(error|human)=0 assumption. That "
    "flatters any policy with a human arm, so accuracy_machine and macro_f1_answered are "
    "reported alongside over machine-answered rows only, where no assumption applies. "
    "Compare policies with different human rates using both."
)

SIGN_CONVENTION_NOTE = (
    "Every paired delta is A - B on identical rows: a NEGATIVE cost_per_1k delta means A "
    "is cheaper, a POSITIVE accuracy delta means A is more accurate. A comparison claim "
    "requires the paired CI to exclude zero (CLAUDE.md); marginal per-policy CIs overlap "
    "even under strict dominance and must not be used for comparisons."
)

MCNEMAR_NOTE = (
    "McNemar is computed only over rows where BOTH policies answered by machine — the "
    "rows on which two label decisions can actually disagree. Rows either policy sent to "
    "a human are excluded and counted in n_excluded_human, because their 'correctness' is "
    "an assumption rather than a prediction."
)

# Baselines that are actual competing SYSTEMS. `all_human` is excluded on purpose: it is
# the $2,500/1k ceiling and beating it is arithmetic, not evidence. A "router dominates N
# single tiers" claim that counts it is inflated, so the phase-acceptance count is reported
# over model baselines only, with the full list alongside.
MODEL_BASELINES = frozenset({"a_only", "a_only_cnb", "c_only"})

DOMINANCE_NOTE = (
    "`dominated_model_baselines` counts only competing model systems (a_only, a_only_cnb, "
    "c_only). Beating `all_human` is reported but never counted toward a dominance claim: "
    "at c_human=$2.50 the all-human policy costs $2,500/1k and any machine policy beats it "
    "by construction, so counting it would inflate the claim."
)

COVERAGE_MATCH_NOTE = (
    "Coverage matching hits the NEAREST ACHIEVABLE coverage, not the requested one: "
    "coverage on a finite slice is quantised to multiples of 1/n, and ties in p_max cannot "
    "be split by a `p_max >= tau` gate. Each entry therefore reports the realized coverage "
    "and abs_coverage_error next to the target; read the realized value, never the target."
)

PARSE_FAIL_NOTE = (
    "The Tier C -> human arm fires only on receipt parse_failed. Haiku 4.5 logged ZERO "
    "parse failures across all 5,000 TEST-IID calls, so the headline cascade's human arm "
    "is empty on this slice: its human_rate is 0.0 by measurement, not by construction. "
    "The c_human parameter therefore has no effect on the Haiku cascade here; the "
    "a_to_human policies are where c_human sensitivity is exercised."
)


# ---------------------------------------------------------------------------
# CAL threshold constants (loaded, never hardcoded)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalThreshold:
    """A tau* constant fit on CAL, with the identity of the file it came from."""

    policy_family: str
    dataset: str
    tau_star: float
    target_coverage_a: float
    n_answered_at_tau_star: int
    tier_a_config: str
    tier_a_run_id: str
    source_file: str
    source_sha256: str


def _check_threshold_object(obj: dict, path, *, cal_record: dict,
                            cost_sha256: str | None) -> None:
    """Refuse a threshold file unless it is internally sound AND bound to the CAL run.

    A tau* is a constant this module copies out of a JSON file and applies to TEST data;
    nothing downstream can tell a good constant from a stale or foreign one. So the file
    has to prove three things before its number is used:

    1. **Whose tau is this?** The file's recorded Tier A run_id, config hash and split must
       match the CAL rung's record in `results/runs.jsonl`, and its cost-config hash must
       match the prices the router is running under. A tau fit on a different model, a
       different data slice, or different prices is not this router's tau.
    2. **Is it a threshold at all?** Finite and inside (0, 1] — the range a `p_max` can
       occupy. NaN would make `p_max >= tau` False for every row (silently answering
       nothing) while still looking like a number in the output.
    3. **Does the file agree with itself?** `n_answered_at_tau_star / n_examples` must
       reproduce `target_coverage_a` — the cheapest possible check that the file was
       written by the code that computed it and not hand-edited afterwards.
    """
    tier_a = obj.get("inputs", {}).get("tier_a", {})
    expected = {
        "run_id": cal_record["run_id"],
        "config_sha256": cal_record.get("config_sha256", ""),
        "split": (cal_record.get("dataset") or {}).get("split", ""),
    }
    mismatched = {k: (tier_a.get(k), v) for k, v in expected.items() if tier_a.get(k) != v}
    if mismatched:
        detail = "; ".join(f"{k}: file {got!r} vs record {want!r}"
                           for k, (got, want) in mismatched.items())
        raise ValueError(
            f"threshold file {path.name} is not bound to the CAL run it claims — {detail}. "
            "A tau* may only be transferred from the run record that produced it"
        )
    if cost_sha256 is not None and obj.get("cost_config", {}).get("sha256") != cost_sha256:
        raise ValueError(
            f"threshold file {path.name} was fit under cost config "
            f"{obj.get('cost_config', {}).get('sha256', '?')[:12]} but the router is "
            f"running under {cost_sha256[:12]}; a tau* is only meaningful with the prices "
            "it minimized — regenerate `make thresholds` first"
        )
    if obj.get("tau_star") is None:
        raise ValueError(
            f"threshold file {path.name} has a null tau* (the cost-minimizing policy "
            "answers nothing); there is no constant to transfer"
        )
    tau = float(obj["tau_star"])
    if not math.isfinite(tau) or not (0.0 < tau <= 1.0):
        raise ValueError(
            f"threshold file {path.name} has tau_star {obj['tau_star']!r}, which is not a "
            "finite probability threshold in (0, 1]"
        )
    coverage = float(obj["target_coverage_a"])
    if not math.isfinite(coverage) or not (0.0 < coverage <= 1.0):
        raise ValueError(
            f"threshold file {path.name} has target_coverage_a "
            f"{obj['target_coverage_a']!r}, which is not a finite coverage in (0, 1]"
        )
    n_examples = int(obj["n_examples"])
    n_answered = int(obj["n_answered_at_tau_star"])
    if n_examples <= 0 or not (0 <= n_answered <= n_examples):
        raise ValueError(
            f"threshold file {path.name} has inconsistent counts: "
            f"n_answered_at_tau_star={n_answered}, n_examples={n_examples}"
        )
    implied = n_answered / n_examples
    if abs(implied - coverage) > THRESHOLD_COUNT_TOL:
        raise ValueError(
            f"threshold file {path.name} disagrees with itself: {n_answered}/{n_examples} "
            f"= {implied!r} but target_coverage_a is {coverage!r} "
            f"(|delta| > {THRESHOLD_COUNT_TOL:g}); the file was not written by the code "
            "that computed it"
        )


def _replay_threshold(obj: dict, path, entry_key, *, cal_record, cost_config,
                      preds_dir, results_path) -> None:
    """Rebuild the CAL policy at the stored tau* and demand the stored numbers back.

    The metadata checks prove the file DESCRIBES the right run; this proves the file's
    numbers were actually produced by that run. Everything downstream is a transcription
    of `tau_star`, so a file whose tau no longer selects the rows it claims — because the
    artifact was regenerated, or a value was hand-edited — would silently move every
    router operating point while every hash still matched.

    Recomputed from the gate-verified CAL artifact: the answered count at tau*, and the
    objective (expected cost per 1,000) the sweep minimized.
    """
    family, dataset = entry_key
    art_a = cost_model.load_artifact_verified(cal_record, preds_dir,
                                              allowed_splits={"cal"})
    records = predictions.records_by_config(results_path)
    record_c = records[threshold_opt.TIER_C_CAL_CONFIG]
    art_c = cost_model.load_artifact_verified(record_c, preds_dir,
                                              allowed_splits={"cal"})
    if dataset == threshold_opt.DATASET_FULL_CAL:
        index = None
    else:
        index = threshold_opt.restrict_to_ids(art_a, art_c.complaint_id)

    if family == threshold_opt.FAMILY_A_TO_HUMAN:
        policy = threshold_opt.build_a_to_human(
            art_a, cal_record, obj["inputs"]["tier_a"]["config_name"],
            dataset=dataset, index=index)
    else:
        api, parse_failed, check, _ = threshold_opt.load_tier_c_arm_inputs(art_c, record_c)
        policy = threshold_opt.build_a_to_c(
            art_a, cal_record, obj["inputs"]["tier_a"]["config_name"], art_c, record_c,
            threshold_opt.TIER_C_CAL_CONFIG, api_cost_usd=api,
            parse_failed=parse_failed, cost_sum_check=check)

    tau = float(obj["tau_star"])
    n_answered = int(np.count_nonzero(np.asarray(policy.p_max, dtype=np.float64) >= tau))
    if n_answered != int(obj["n_answered_at_tau_star"]):
        raise ValueError(
            f"threshold file {path.name} does not replay: tau_star {tau!r} selects "
            f"{n_answered} row(s) on the CAL artifact but the file records "
            f"{obj['n_answered_at_tau_star']}"
        )
    if len(policy) != int(obj["n_examples"]):
        raise ValueError(
            f"threshold file {path.name} does not replay: the CAL policy has "
            f"{len(policy)} row(s), the file records {obj['n_examples']}"
        )
    recomputed = threshold_opt.cost_at(
        policy, tau, c_misroute=cost_config.c_misroute_usd,
        c_human=cost_config.c_human_usd)["total"]
    published = obj["operating_points"]["tau_star"]["expected_cost_per_1k"]["total"]["point"]
    if abs(recomputed - published) > REPLAY_COST_TOL:
        raise ValueError(
            f"threshold file {path.name} does not replay: the objective at tau_star "
            f"recomputes to {recomputed!r} but the file records {published!r} "
            f"(|delta| > {REPLAY_COST_TOL:g})"
        )


def load_cal_thresholds(thresholds_dir=DEFAULT_THRESHOLDS_DIR, *,
                        cost_sha256: str | None = None,
                        tier_a_config: str | None = None,
                        results_path=DEFAULT_RESULTS_PATH,
                        derivation: str = OP_V1,
                        cost_config: cost_model.CostConfig | None = None,
                        preds_dir=DEFAULT_PREDS_DIR,
                        verify_replay: bool = True,
                        ) -> dict[tuple[str, str], CalThreshold]:
    """Read + validate the primary rung's tau* constants out of `results/thresholds/`.

    Constants are LOADED, never transcribed: a number typed into this module would be a
    second source of truth that no gate compares against the file it came from. Each file
    is validated by `_check_threshold_object` against the CAL rung's own run record before
    its tau is admitted, and the file's sha256 is carried into the router outputs.

    Two files claiming the same (policy_family, dataset) is a HARD FAILURE, not
    last-one-wins: they would differ in something the key does not name (a different Tier A
    run, a regeneration under different prices), and silently keeping whichever sorted last
    would make every router number depend on file order. The exact expected key set is
    required too, so a missing file is caught here rather than as a KeyError deep inside a
    policy builder.
    """
    thresholds_dir = Path(thresholds_dir)
    tier_a_config = tier_a_config or OP_VERSIONS[derivation]["tier_a_cal_config"]
    cal_records = predictions.records_by_config(results_path)
    if tier_a_config not in cal_records:
        raise ValueError(
            f"no run record for the CAL rung {tier_a_config!r} in {results_path}; the "
            "thresholds cannot be bound to the run that produced them"
        )
    cal_record = cal_records[tier_a_config]

    out: dict[tuple[str, str], CalThreshold] = {}
    sources: dict[tuple[str, str], str] = {}
    for path in sorted(thresholds_dir.glob("*__cost-*.json")):
        if path.name.startswith("summary__"):
            continue
        obj = json.loads(path.read_text())
        tier_a = obj.get("inputs", {}).get("tier_a", {})
        # Files written before the derivation field are v1 by definition, which is why the
        # default matters: it is what keeps v1 evidence readable without rewriting it.
        if obj.get("derivation", threshold_opt.DERIVATION_V1) != derivation:
            continue
        if tier_a.get("config_name") != tier_a_config or not obj.get("is_primary"):
            continue
        _check_threshold_object(obj, path, cal_record=cal_record, cost_sha256=cost_sha256)
        key = (obj["policy_family"], obj["dataset"])
        if verify_replay:
            if cost_config is None:
                raise ValueError(
                    "verify_replay needs the cost config whose prices the tau* minimized; "
                    "pass cost_config=..., or verify_replay=False to check metadata only"
                )
            _replay_threshold(obj, path, key, cal_record=cal_record,
                              cost_config=cost_config, preds_dir=preds_dir,
                              results_path=results_path)
        if key in out:
            raise ValueError(
                f"two primary threshold files claim {key}: {sources[key]} and "
                f"{path.name}; refusing to pick one by file order — remove the stale file"
            )
        sources[key] = path.name
        out[key] = CalThreshold(
            policy_family=obj["policy_family"],
            dataset=obj["dataset"],
            tau_star=float(obj["tau_star"]),
            target_coverage_a=float(obj["target_coverage_a"]),
            n_answered_at_tau_star=int(obj["n_answered_at_tau_star"]),
            tier_a_config=tier_a["config_name"],
            tier_a_run_id=tier_a["run_id"],
            source_file=path.name,
            source_sha256=cost_model.sha256_file(path),
        )
    if not out:
        raise ValueError(
            f"no primary-rung {derivation} threshold files for {tier_a_config!r} under "
            f"{thresholds_dir}; run `make thresholds` first"
        )
    missing = EXPECTED_THRESHOLD_KEYS - set(out)
    unexpected = set(out) - EXPECTED_THRESHOLD_KEYS
    if missing or unexpected:
        raise ValueError(
            f"the {derivation} threshold set under {thresholds_dir} is not the expected "
            f"one — missing {sorted(missing)}, unexpected {sorted(unexpected)}; each "
            f"derivation needs exactly {sorted(EXPECTED_THRESHOLD_KEYS)}"
        )
    return out


def coverage_matched_tau(p_max, target_coverage: float) -> float:
    """Tau on THIS p_max giving the NEAREST ACHIEVABLE coverage to `target_coverage`.

    Not "at least the target": coverage on a finite slice is quantised to multiples of
    1/n, so the requested figure is generally unreachable and `round()` picks the nearer
    of the two neighbouring achievable values — which may be BELOW the target (n=5,
    target 0.5 -> k=2 -> 0.4). Ties in `p_max` can also push the realized coverage above
    the requested one, because the gate `p_max >= tau` cannot split a tie block.

    Both effects mean the realized coverage must be read from the output rather than
    assumed: every secondary block reports the realized coverage and the absolute matching
    error alongside the target. Secondary-only — choosing tau from the evaluation slice is
    not a deployable procedure, it is a "what was the intended operating point worth" probe.
    """
    p_max = np.asarray(p_max, dtype=np.float64)
    n = len(p_max)
    k = round(float(target_coverage) * n)
    if k <= 0:
        return math.inf
    if k >= n:
        return float(p_max.min())
    return float(np.sort(p_max)[::-1][k - 1])


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RouterPolicy:
    """One evaluated policy: per-row outcome, spend, and how the gate was set."""

    name: str
    evaluation_set: str
    ids: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray        # machine label; ignored where to_human
    to_human: np.ndarray
    api_cost_usd: np.ndarray
    gate: dict
    inputs: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def machine(self) -> np.ndarray:
        return ~self.to_human

    @property
    def correct_machine(self) -> np.ndarray:
        """Per-row correctness of the machine answer (meaningless where to_human)."""
        return self.y_pred == self.y_true

    @property
    def correct_for_cost(self) -> np.ndarray:
        """Correctness as the cost model sees it: human rows count as resolved."""
        return np.where(self.to_human, True, self.correct_machine)

    @property
    def y_pred_system(self) -> np.ndarray:
        """System-level labels with human rows credited (y_pred := y_true)."""
        return np.where(self.to_human, self.y_true, self.y_pred)


def _gate(kind: str, *, model: str, tau: float | None = None, source: dict | None = None,
          transfer: str | None = None) -> dict:
    return {
        "kind": kind,
        "gate_model": model,
        "tau": None if tau is None or not math.isfinite(tau) else float(tau),
        "transfer_mode": transfer,
        "tau_source": source or {},
    }


def single_tier_policy(name, art, *, evaluation_set, api_cost_usd=None, to_human=None,
                       gate, inputs, index=None) -> RouterPolicy:
    """A tier answering every row (optionally with its own parse-fail human arm)."""
    idx = slice(None) if index is None else index
    ids = np.asarray(art.complaint_id[idx], dtype=np.int64)
    n = len(ids)
    return RouterPolicy(
        name=name,
        evaluation_set=evaluation_set,
        ids=ids,
        y_true=np.asarray(art.y_true[idx], dtype=object),
        y_pred=np.asarray(art.y_pred[idx], dtype=object),
        to_human=(np.zeros(n, dtype=bool) if to_human is None
                  else np.asarray(to_human, dtype=bool)),
        api_cost_usd=(np.zeros(n, dtype=np.float64) if api_cost_usd is None
                      else np.asarray(api_cost_usd, dtype=np.float64)),
        gate=gate,
        inputs=inputs,
    )


def all_human_policy(ids, y_true, *, evaluation_set, inputs) -> RouterPolicy:
    """Every complaint to a human: the ceiling the router has to undercut."""
    n = len(ids)
    return RouterPolicy(
        name="all_human",
        evaluation_set=evaluation_set,
        ids=np.asarray(ids, dtype=np.int64),
        y_true=np.asarray(y_true, dtype=object),
        y_pred=np.asarray(y_true, dtype=object),  # never used: every row is to_human
        to_human=np.ones(n, dtype=bool),
        api_cost_usd=np.zeros(n, dtype=np.float64),
        gate=_gate("no_machine", model="none"),
        inputs=inputs,
    )


def a_to_human_policy(art_a, tau, *, evaluation_set, gate, inputs,
                      index=None) -> RouterPolicy:
    """Tier A answers above tau; everything else goes to a human (no API spend)."""
    idx = slice(None) if index is None else index
    p_max = np.asarray(art_a.p_max[idx], dtype=np.float64)
    answered = p_max >= tau
    n = len(p_max)
    return RouterPolicy(
        name="a_to_human",
        evaluation_set=evaluation_set,
        ids=np.asarray(art_a.complaint_id[idx], dtype=np.int64),
        y_true=np.asarray(art_a.y_true[idx], dtype=object),
        y_pred=np.asarray(art_a.y_pred[idx], dtype=object),
        to_human=~answered,
        api_cost_usd=np.zeros(n, dtype=np.float64),
        gate=gate,
        inputs=inputs,
    )


def a_to_c_policy(art_a, index, art_c, *, tau, api_cost_usd, parse_failed, gate,
                  inputs) -> RouterPolicy:
    """Tier A answers above tau; below it Tier C answers TERMINALLY.

    Escalated rows are charged their measured per-call cost whatever happens next
    (incurred spend). A receipt marked `parse_failed` routes that row to a human and its
    Tier C `y_pred` — the fallback label — is discarded rather than scored: under this
    policy no model answered that complaint.
    """
    p_max = np.asarray(art_a.p_max[index], dtype=np.float64)
    answered_by_a = p_max >= tau
    parse_failed = np.asarray(parse_failed, dtype=bool)
    api = np.where(answered_by_a, 0.0, np.asarray(api_cost_usd, dtype=np.float64))
    y_pred = np.where(answered_by_a, np.asarray(art_a.y_pred[index], dtype=object),
                      np.asarray(art_c.y_pred, dtype=object))
    return RouterPolicy(
        name="a_to_c_parsefail_human",
        evaluation_set=EVAL_PAIRED,
        ids=np.asarray(art_c.complaint_id, dtype=np.int64),
        y_true=np.asarray(art_c.y_true, dtype=object),
        y_pred=y_pred,
        to_human=(~answered_by_a) & parse_failed,
        api_cost_usd=api,
        gate=gate,
        inputs=inputs,
    )


# ---------------------------------------------------------------------------
# Per-policy metrics
# ---------------------------------------------------------------------------

def evaluate_policy(policy: RouterPolicy, cfg: cost_model.CostConfig, class_labels, *,
                    n_resamples: int = harness.N_RESAMPLES,
                    seed: int = harness.BOOTSTRAP_SEED) -> dict:
    """Cost (with CI), routing mix, and both accuracy views for one policy."""
    n = len(policy)
    machine = policy.machine
    n_machine = int(machine.sum())
    n_human = int(policy.to_human.sum())

    bands = cost_model.bootstrap_cost(
        policy.correct_for_cost, policy.api_cost_usd, policy.to_human,
        c_misroute=cfg.c_misroute_usd, c_human=cfg.c_human_usd,
        n_resamples=n_resamples, seed=seed,
    )
    correct = policy.correct_machine
    labels = list(class_labels)
    return {
        "policy": policy.name,
        "evaluation_set": policy.evaluation_set,
        "n_examples": n,
        "gate": policy.gate,
        "routing": {
            "n_answered_machine": n_machine,
            "n_to_human": n_human,
            "coverage_machine": _round(n_machine / n),
            "human_rate": _round(n_human / n),
            "coverage_a": _round(float(policy.gate.get("coverage_a", np.nan))),
            "escalation_rate": _round(float(policy.gate.get("escalation_rate", np.nan))),
        },
        "expected_cost_per_1k": {k: _round_ci(v) for k, v in bands.items()},
        "api_cost_usd_total": _round(float(policy.api_cost_usd.sum())),
        "accuracy_machine": _round(float(correct[machine].mean())) if n_machine else None,
        "macro_f1_answered": (
            _round(metrics.macro_f1(policy.y_true[machine], policy.y_pred[machine], labels))
            if n_machine else None
        ),
        "accuracy_system": _round(float(policy.correct_for_cost.mean())),
        "macro_f1_system": _round(
            metrics.macro_f1(policy.y_true, policy.y_pred_system, labels)),
    }


# ---------------------------------------------------------------------------
# Paired comparisons
# ---------------------------------------------------------------------------

def _require_aligned(a: RouterPolicy, b: RouterPolicy) -> None:
    """Two policies may only be paired if they describe the same rows, in the same order.

    Ids alone are not enough: two artifacts can carry identical id vectors while disagreeing
    on the ground truth (a split re-cut, a relabelled taxonomy), and a paired delta computed
    across that disagreement would silently compare each system against a different answer
    key. Length is checked first so the error names the real problem instead of surfacing as
    a broadcast failure.
    """
    if len(a) != len(b):
        raise ValueError(
            f"cannot pair {a.name} (n={len(a)}) with {b.name} (n={len(b)}): different "
            "numbers of rows"
        )
    if not np.array_equal(a.ids, b.ids):
        raise ValueError(
            f"cannot pair {a.name} with {b.name}: their ids differ, so a per-example "
            "difference would compare different complaints"
        )
    if not np.array_equal(a.y_true, b.y_true):
        n_bad = int(np.count_nonzero(np.asarray(a.y_true) != np.asarray(b.y_true)))
        raise ValueError(
            f"cannot pair {a.name} with {b.name}: they disagree on y_true for {n_bad} "
            "row(s) despite identical ids — the two are not scored against the same "
            "ground truth"
        )


def _band(values, *, scale, n_resamples, seed) -> dict:
    reps = cost_model.resample_means(
        {"d": np.asarray(values, dtype=np.float64)}, scale=scale,
        n_resamples=n_resamples, seed=seed)["d"]
    lo, hi = np.percentile(reps, [harness.CI_LOWER_PCT, harness.CI_UPPER_PCT])
    point = float(np.mean(values)) * scale
    return {
        "point": _round(point),
        "ci_lo": _round(float(lo)),
        "ci_hi": _round(float(hi)),
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }


def paired_comparison(a: RouterPolicy, b: RouterPolicy, cfg: cost_model.CostConfig, *,
                      n_resamples: int = harness.N_RESAMPLES,
                      seed: int = harness.BOOTSTRAP_SEED) -> dict:
    """A - B on identical rows: cost/1k, system accuracy, and machine-vs-machine McNemar."""
    _require_aligned(a, b)
    per = {}
    for policy in (a, b):
        per[policy.name] = cost_model.per_example_cost(
            policy.correct_for_cost, policy.api_cost_usd, policy.to_human,
            c_misroute=cfg.c_misroute_usd, c_human=cfg.c_human_usd)
    cost_delta = _band(per[a.name] - per[b.name], scale=cost_model.PER_N_COMPLAINTS,
                       n_resamples=n_resamples, seed=seed)
    acc_delta = _band(
        a.correct_for_cost.astype(np.float64) - b.correct_for_cost.astype(np.float64),
        scale=1.0, n_resamples=n_resamples, seed=seed)

    both_machine = a.machine & b.machine
    n_both = int(both_machine.sum())
    mcnemar = harness.mcnemar(a.y_true[both_machine], a.y_pred[both_machine],
                              b.y_pred[both_machine]) if n_both else {
        "b": 0, "c": 0, "n_discordant": 0, "p_value": 1.0}
    return {
        "a": a.name,
        "b": b.name,
        "evaluation_set": a.evaluation_set,
        "n_examples": len(a),
        "delta_cost_per_1k": cost_delta,
        "delta_accuracy_system": acc_delta,
        "cheaper": a.name if cost_delta["point"] < 0 else (
            b.name if cost_delta["point"] > 0 else "neither"),
        "mcnemar_machine_rows": {
            **{k: (float(v) if k == "p_value" else int(v)) for k, v in mcnemar.items()},
            "n_both_machine": n_both,
            "n_excluded_human": len(a) - n_both,
            "note": MCNEMAR_NOTE,
        },
    }


# ---------------------------------------------------------------------------
# Building the two evaluation sets
# ---------------------------------------------------------------------------

def _artifact_block(record, art, config_name) -> dict:
    """Everything that identifies the run behind an artifact, serialized into outputs.

    Includes `git_sha` and `input_sha256`: a derived number should name the code and the
    data snapshot it came from, not just the config and the split.
    """
    return {
        "run_id": record["run_id"],
        "config_name": config_name,
        "config_sha256": art.provenance.get("config_sha256", ""),
        "git_sha": art.provenance.get("git_sha", ""),
        "split": art.provenance.get("split", ""),
        "split_sha256": art.provenance.get("split_sha256", ""),
        "input_sha256": art.provenance.get("input_sha256", ""),
        "n_examples": len(art),
    }


@dataclass
class TestInputs:
    """Every frozen TEST-IID input the simulator consumes, already gate-verified."""

    records: dict
    art_a: object
    art_cnb: object
    art_c: object
    index_paired: np.ndarray
    api_cost_usd: np.ndarray
    parse_failed: np.ndarray
    cost_sum_check: dict
    receipts_sha256: str
    blocks: dict


def load_test_inputs(preds_dir=DEFAULT_PREDS_DIR,
                     results_path=DEFAULT_RESULTS_PATH) -> TestInputs:
    """Load + verify the three TEST-IID artifacts and the Haiku receipts they pair with."""
    records = predictions.records_by_config(results_path)
    missing = [c for c in (TIER_A_TEST_CONFIG, TIER_A_CNB_TEST_CONFIG, TIER_C_TEST_CONFIG)
               if c not in records]
    if missing:
        raise ValueError(f"no run record for config(s) {missing} in {results_path}")

    def load(config_name):
        return cost_model.load_artifact_verified(
            records[config_name], preds_dir, allowed_splits=ALLOWED_SPLITS)

    art_a = load(TIER_A_TEST_CONFIG)
    art_cnb = load(TIER_A_CNB_TEST_CONFIG)
    art_c = load(TIER_C_TEST_CONFIG)

    record_c = records[TIER_C_TEST_CONFIG]
    raw_log_path = (record_c.get("extra") or {}).get("raw_log_path")
    if not raw_log_path:
        raise ValueError(
            f"tier_c run {record_c['run_id'][:8]} has no extra.raw_log_path; there are no "
            "measured per-call costs to escalate against"
        )
    api = cost_model.join_receipt_costs(art_c.complaint_id, raw_log_path, record=record_c)
    parse_failed = cost_model.join_parse_failed(art_c.complaint_id, raw_log_path,
                                                record=record_c)
    check = cost_model.check_cost_sum(api, record_c)
    if not check["ok"]:
        raise ValueError(
            f"cost_sum_check FAILED for {record_c['run_id'][:8]}: joined "
            f"${check['joined_cost_usd']:.6f} vs logged ${check['logged_cost_usd']:.6f}"
        )

    index = threshold_opt.restrict_to_ids(art_a, art_c.complaint_id)
    if not np.array_equal(art_a.y_true[index], art_c.y_true):
        raise ValueError(
            "Tier A and Tier C TEST-IID artifacts disagree on y_true for the paired ids; "
            "the two artifacts are not describing the same rows"
        )
    if not np.array_equal(np.asarray(art_a.class_labels), np.asarray(art_c.class_labels)):
        raise ValueError("Tier A and Tier C artifacts use different class label orders")

    blocks = {
        "tier_a_logreg": _artifact_block(records[TIER_A_TEST_CONFIG], art_a,
                                         TIER_A_TEST_CONFIG),
        "tier_a_cnb": _artifact_block(records[TIER_A_CNB_TEST_CONFIG], art_cnb,
                                      TIER_A_CNB_TEST_CONFIG),
        "tier_c_haiku": {
            **_artifact_block(record_c, art_c, TIER_C_TEST_CONFIG),
            "raw_log_path": str(raw_log_path),
            "receipts_sha256": cost_model.receipts_sha256(raw_log_path),
            "prompt_bundle_sha256": art_c.provenance.get("prompt_bundle_sha256", ""),
            "model_slug": (record_c.get("extra") or {}).get("model_slug", ""),
            "logged_cost_usd": record_c.get("cost_usd"),
            "cost_sum_check": {k: (_round(v) if isinstance(v, float) else v)
                               for k, v in check.items()},
            "n_parse_failed": int(np.count_nonzero(parse_failed)),
        },
    }
    return TestInputs(
        records=records, art_a=art_a, art_cnb=art_cnb, art_c=art_c,
        index_paired=index, api_cost_usd=api, parse_failed=parse_failed,
        cost_sum_check=check, receipts_sha256=blocks["tier_c_haiku"]["receipts_sha256"],
        blocks=blocks,
    )


def _with_rates(gate: dict, p_max, tau) -> dict:
    answered = np.asarray(p_max, dtype=np.float64) >= tau
    n = len(answered)
    return {**gate, "coverage_a": float(answered.sum()) / n,
            "escalation_rate": float((~answered).sum()) / n}


def build_full_policies(inputs: TestInputs, cal: dict, *,
                        transfer: str = TRANSFER_PRIMARY) -> list[RouterPolicy]:
    """FULL TEST-IID: a_only (logreg), a_only_cnb, a_to_human @ tau, all_human."""
    art_a, art_cnb = inputs.art_a, inputs.art_cnb
    cal_t = cal[(threshold_opt.FAMILY_A_TO_HUMAN, threshold_opt.DATASET_FULL_CAL)]
    tau = (cal_t.tau_star if transfer == TRANSFER_PRIMARY
           else coverage_matched_tau(art_a.p_max, cal_t.target_coverage_a))
    source = {"file": cal_t.source_file, "sha256": cal_t.source_sha256,
              "cal_tau_star": cal_t.tau_star,
              "cal_target_coverage_a": cal_t.target_coverage_a}
    a_block = {"tier_a_logreg": inputs.blocks["tier_a_logreg"]}
    return [
        single_tier_policy("a_only", art_a, evaluation_set=EVAL_FULL,
                           gate=_gate("answer_all", model=TIER_A_TEST_CONFIG),
                           inputs=a_block),
        single_tier_policy("a_only_cnb", art_cnb, evaluation_set=EVAL_FULL,
                           gate=_gate("answer_all", model=TIER_A_CNB_TEST_CONFIG),
                           inputs={"tier_a_cnb": inputs.blocks["tier_a_cnb"]}),
        a_to_human_policy(
            art_a, tau, evaluation_set=EVAL_FULL,
            gate=_with_rates(_gate("tier_a_threshold", model=TIER_A_TEST_CONFIG, tau=tau,
                                   source=source, transfer=transfer), art_a.p_max, tau),
            inputs=a_block),
        all_human_policy(art_a.complaint_id, art_a.y_true, evaluation_set=EVAL_FULL,
                         inputs=a_block),
    ]


def build_paired_policies(inputs: TestInputs, cal: dict, *,
                          transfer: str = TRANSFER_PRIMARY) -> list[RouterPolicy]:
    """PAIRED 5,000 rows: a_only, c_only, a_to_human @ tau, a_to_c @ tau, all_human."""
    art_a, art_c, index = inputs.art_a, inputs.art_c, inputs.index_paired
    p_max = art_a.p_max[index]
    cal_h = cal[(threshold_opt.FAMILY_A_TO_HUMAN, threshold_opt.DATASET_PAIRED)]
    cal_c = cal[(threshold_opt.FAMILY_A_TO_C, threshold_opt.DATASET_PAIRED)]

    def tau_for(cal_t):
        return (cal_t.tau_star if transfer == TRANSFER_PRIMARY
                else coverage_matched_tau(p_max, cal_t.target_coverage_a))

    def source(cal_t):
        return {"file": cal_t.source_file, "sha256": cal_t.source_sha256,
                "cal_tau_star": cal_t.tau_star,
                "cal_target_coverage_a": cal_t.target_coverage_a}

    tau_h, tau_c = tau_for(cal_h), tau_for(cal_c)
    a_block = {"tier_a_logreg": inputs.blocks["tier_a_logreg"]}
    c_block = {"tier_c_haiku": inputs.blocks["tier_c_haiku"]}
    both = {**a_block, **c_block}
    return [
        single_tier_policy("a_only", art_a, evaluation_set=EVAL_PAIRED, index=index,
                           gate=_gate("answer_all", model=TIER_A_TEST_CONFIG),
                           inputs=a_block),
        single_tier_policy("c_only", art_c, evaluation_set=EVAL_PAIRED,
                           api_cost_usd=inputs.api_cost_usd,
                           to_human=inputs.parse_failed,
                           gate=_gate("answer_all_parsefail_human",
                                      model=TIER_C_TEST_CONFIG),
                           inputs=c_block),
        a_to_human_policy(
            art_a, tau_h, evaluation_set=EVAL_PAIRED, index=index,
            gate=_with_rates(_gate("tier_a_threshold", model=TIER_A_TEST_CONFIG,
                                   tau=tau_h, source=source(cal_h), transfer=transfer),
                             p_max, tau_h),
            inputs=a_block),
        a_to_c_policy(
            art_a, index, art_c, tau=tau_c, api_cost_usd=inputs.api_cost_usd,
            parse_failed=inputs.parse_failed,
            gate=_with_rates(_gate("tier_a_threshold", model=TIER_A_TEST_CONFIG,
                                   tau=tau_c, source=source(cal_c), transfer=transfer),
                             p_max, tau_c),
            inputs=both),
        all_human_policy(art_c.complaint_id, art_c.y_true, evaluation_set=EVAL_PAIRED,
                         inputs=a_block),
    ]


# Comparisons per evaluation set: (A, B) with A the policy making the claim.
FULL_COMPARISONS = (("a_to_human", "a_only"), ("a_to_human", "a_only_cnb"),
                    ("a_to_human", "all_human"))
PAIRED_COMPARISONS = (("a_to_c_parsefail_human", "a_only"),
                      ("a_to_c_parsefail_human", "c_only"),
                      ("a_to_c_parsefail_human", "a_to_human"),
                      ("a_to_c_parsefail_human", "all_human"),
                      ("a_to_human", "a_only"))


def build_evaluation(name, policies, comparisons, inputs: TestInputs,
                     cfg: cost_model.CostConfig, cal: dict, *, secondary_policies,
                     n_resamples: int = harness.N_RESAMPLES,
                     seed: int = harness.BOOTSTRAP_SEED,
                     op_version: str = OP_V1) -> dict:
    """Assemble one evaluation set: policies, paired deltas, secondary transfer block."""
    labels = list(inputs.art_a.class_labels)
    by_name = {p.name: p for p in policies}
    evaluated = {p.name: evaluate_policy(p, cfg, labels, n_resamples=n_resamples,
                                         seed=seed) for p in policies}
    deltas = [
        paired_comparison(by_name[a], by_name[b], cfg, n_resamples=n_resamples, seed=seed)
        for a, b in comparisons
    ]
    secondary_by_name = {p.name: p for p in secondary_policies}
    secondary = {
        "transfer_mode": TRANSFER_SECONDARY,
        "label": "SECONDARY sensitivity only — tau re-chosen on TEST p_max",
        "contract": COVERAGE_MATCH_NOTE,
        "policies": {
            p.name: evaluate_policy(p, cfg, labels, n_resamples=n_resamples, seed=seed)
            for p in secondary_policies if p.gate["kind"] == "tier_a_threshold"
        },
        "coverage_matching": [
            {
                "policy": p.name,
                "target_coverage_a": _round(p.gate["tau_source"]["cal_target_coverage_a"]),
                "realized_coverage_a": _round(p.gate["coverage_a"]),
                "abs_coverage_error": _round(
                    abs(p.gate["coverage_a"]
                        - p.gate["tau_source"]["cal_target_coverage_a"])),
                "coverage_quantum": _round(1.0 / len(p)),
                "tau": p.gate["tau"],
            }
            for p in secondary_policies if p.gate["kind"] == "tier_a_threshold"
        ],
        "comparisons": [
            paired_comparison(secondary_by_name[a], secondary_by_name[b], cfg,
                              n_resamples=n_resamples, seed=seed)
            for a, b in comparisons
            if secondary_by_name[a].gate["kind"] == "tier_a_threshold"
        ],
    }
    transfer_rows = []
    for policy in policies:
        if policy.gate["kind"] != "tier_a_threshold":
            continue
        src = policy.gate["tau_source"]
        realized = policy.gate["coverage_a"]
        target = src["cal_target_coverage_a"]
        secondary_gate = secondary_by_name[policy.name].gate
        transfer_rows.append({
            "policy": policy.name,
            "cal_tau_star": src["cal_tau_star"],
            "cal_target_coverage_a": _round(target),
            "applied_tau": policy.gate["tau"],
            "realized_coverage_a": _round(realized),
            "coverage_gap": _round(realized - target),
            "coverage_matched_tau": secondary_gate["tau"],
            "coverage_matched_realized_coverage_a": _round(secondary_gate["coverage_a"]),
            "coverage_matched_abs_error": _round(abs(secondary_gate["coverage_a"] - target)),
            "cal_source_file": src["file"],
            "cal_source_sha256": src["sha256"],
        })

    # v1 router files predate the op-version field and are never rewritten (a test asserts
    # byte-identity), so only v2 carries it.
    version_block = {} if op_version == OP_V1 else {
        "operating_point_version": op_version,
        "tau_derivation_note": threshold_opt.ISOCAL_IN_SAMPLE_NOTE,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        **version_block,
        "evaluation_set": name,
        "n_examples": len(policies[0]),
        "class_labels": labels,
        "cost_config": cost_model.config_block(cfg),
        "inputs": {k: v for p in policies for k, v in p.inputs.items()},
        "transfer": {
            "primary_mode": TRANSFER_PRIMARY,
            "note": TRANSFER_NOTE,
            "rows": transfer_rows,
        },
        "policies": evaluated,
        "paired_deltas": deltas,
        "secondary_coverage_matched": secondary,
        "bootstrap": {
            "n_resamples": int(n_resamples),
            "seed": int(seed),
            "method": (
                f"percentile [{harness.CI_LOWER_PCT}, {harness.CI_UPPER_PCT}] over "
                "resampled example indices (one integers(0, n, size=n) draw per replicate, "
                "shared across policies within a paired delta)"
            ),
        },
        "notes": {
            "human_credit": HUMAN_CREDIT_NOTE,
            "sign_convention": SIGN_CONVENTION_NOTE,
            "parse_failures": PARSE_FAIL_NOTE,
            "human_assumption": cost_model.HUMAN_ASSUMPTION,
            "amendment": threshold_opt.AMENDMENT_NOTE,
        },
    }


def build_all(cfg: cost_model.CostConfig, *, preds_dir=DEFAULT_PREDS_DIR,
              results_path=DEFAULT_RESULTS_PATH,
              thresholds_dir=DEFAULT_THRESHOLDS_DIR,
              n_resamples: int = harness.N_RESAMPLES,
              seed: int = harness.BOOTSTRAP_SEED,
              op_version: str = OP_V1,
              verify_replay: bool = True) -> dict[str, dict]:
    """Both evaluation sets, keyed by name. Nothing is written; the caller writes.

    `verify_replay` is threaded only so a synthetic fixture without CAL artifacts can
    exercise the metadata gate; production callers leave it on and the replay gate is
    additionally covered against the shipped files.
    """
    inputs = load_test_inputs(preds_dir, results_path)
    cal = load_cal_thresholds(thresholds_dir, cost_sha256=cfg.sha256,
                              results_path=results_path, derivation=op_version,
                              cost_config=cfg if verify_replay else None,
                              preds_dir=preds_dir, verify_replay=verify_replay)
    out = {}
    for name, builder, comparisons in (
        (EVAL_FULL, build_full_policies, FULL_COMPARISONS),
        (EVAL_PAIRED, build_paired_policies, PAIRED_COMPARISONS),
    ):
        out[name] = build_evaluation(
            name, builder(inputs, cal), comparisons, inputs, cfg, cal,
            secondary_policies=builder(inputs, cal, transfer=TRANSFER_SECONDARY),
            n_resamples=n_resamples, seed=seed, op_version=op_version,
        )
    return out


def result_filename(name: str, cfg: cost_model.CostConfig, op_version: str) -> str:
    """`<name>[__opv2]__cost-<sha8>.json` — v1 keeps its original, unversioned name."""
    return f"{name}{OP_VERSIONS[op_version]['suffix']}__cost-{cfg.sha256[:8]}.json"


def _dominance_row(evaluation: dict, router: str) -> dict:
    """Which baselines this router beats on cost with a paired CI excluding zero."""
    rows = [d for d in evaluation["paired_deltas"] if d["a"] == router]
    beaten = [d["b"] for d in rows
              if d["delta_cost_per_1k"]["excludes_zero"]
              and d["delta_cost_per_1k"]["point"] < 0]
    return {
        "compared_against": [d["b"] for d in rows],
        "dominated": sorted(beaten),
        "dominated_model_baselines": sorted(set(beaten) & MODEL_BASELINES),
        "n_model_baselines_dominated": len(set(beaten) & MODEL_BASELINES),
        "not_dominated": sorted({d["b"] for d in rows} - set(beaten)),
    }


def build_summary(evaluations: dict[str, dict], cfg: cost_model.CostConfig, *,
                  op_version: str = OP_V1) -> dict:
    """Headline table + the two verdicts Phase 4 acceptance turns on."""
    paired = evaluations[EVAL_PAIRED]
    router = "a_to_c_parsefail_human"
    decision_1 = next(d for d in paired["paired_deltas"]
                      if d["a"] == router and d["b"] == "a_to_human")
    dominance = {
        f"{name}/{a}": _dominance_row(ev, a)
        for name, ev in evaluations.items()
        for a in dict.fromkeys(d["a"] for d in ev["paired_deltas"])
    }
    headline = dominance[f"{EVAL_PAIRED}/{router}"]
    version_block = {} if op_version == OP_V1 else {
        "operating_point_version": op_version,
        "tau_derivation_note": threshold_opt.ISOCAL_IN_SAMPLE_NOTE,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "summary",
        **version_block,
        "cost_config": cost_model.config_block(cfg),
        "headline_router": router,
        "evaluations": {
            name: {
                "n_examples": ev["n_examples"],
                "policies": {
                    pname: {
                        "expected_cost_per_1k": p["expected_cost_per_1k"]["total"],
                        "coverage_machine": p["routing"]["coverage_machine"],
                        "human_rate": p["routing"]["human_rate"],
                        "accuracy_machine": p["accuracy_machine"],
                        "accuracy_system": p["accuracy_system"],
                        "macro_f1_answered": p["macro_f1_answered"],
                        "macro_f1_system": p["macro_f1_system"],
                    }
                    for pname, p in ev["policies"].items()
                },
                "transfer": ev["transfer"]["rows"],
                "file": result_filename(name, cfg, op_version),
            }
            for name, ev in evaluations.items()
        },
        "paired_deltas": {
            name: [
                {"a": d["a"], "b": d["b"],
                 "delta_cost_per_1k": d["delta_cost_per_1k"],
                 "delta_accuracy_system": d["delta_accuracy_system"],
                 "mcnemar_p_value": d["mcnemar_machine_rows"]["p_value"]}
                for d in ev["paired_deltas"]
            ]
            for name, ev in evaluations.items()
        },
        "owner_decision_1_cross_family": {
            "question": (
                "Does the Haiku-terminal cascade beat the A->human cascade on the full "
                "n=5,000 paired TEST-IID rows?"
            ),
            "delta_cost_per_1k": decision_1["delta_cost_per_1k"],
            "n_examples": decision_1["n_examples"],
            "verdict": (
                "supported (paired CI excludes zero)"
                if decision_1["delta_cost_per_1k"]["excludes_zero"]
                else "DIRECTIONAL ONLY — paired CI includes zero"
            ),
        },
        "dominance": {
            "criterion": (
                "router is cheaper than the baseline with a paired bootstrap CI on the "
                "cost/1k difference that excludes zero (CLAUDE.md comparison rule)"
            ),
            "model_baselines": sorted(MODEL_BASELINES),
            "note": DOMINANCE_NOTE,
            "by_router": dominance,
            "headline_router_model_baselines_dominated":
                headline["dominated_model_baselines"],
            "headline_router_meets_two_model_baselines":
                headline["n_model_baselines_dominated"] >= 2,
        },
        "n_parse_failed_tier_c_test_iid":
            paired["inputs"]["tier_c_haiku"]["n_parse_failed"],
        "notes": {
            "transfer": TRANSFER_NOTE,
            "human_credit": HUMAN_CREDIT_NOTE,
            "sign_convention": SIGN_CONVENTION_NOTE,
            "parse_failures": PARSE_FAIL_NOTE,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m triage_lab.router_sim")
    parser.add_argument("--preds-dir", type=Path, default=DEFAULT_PREDS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ROUTER_DIR)
    parser.add_argument("--cost-config", type=Path, default=cost_model.DEFAULT_COST_CONFIG)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--thresholds-dir", type=Path, default=DEFAULT_THRESHOLDS_DIR)
    parser.add_argument(
        "--no-verify-replay", dest="verify_replay", action="store_false",
        help="skip the tau-replay gate (metadata checks only); for fixtures without CAL "
             "artifacts, never for reported runs",
    )
    parser.add_argument(
        "--op-version", choices=sorted(OP_VERSIONS), default=OP_V1,
        help="which CAL threshold derivation to transfer; v1-raw is the default so its "
             "committed files regenerate byte-identically",
    )
    args = parser.parse_args(argv)

    cfg = cost_model.load_cost_config(args.cost_config)
    print(f"cost config {cfg.path.name} ({cfg.version}) sha256={cfg.sha256[:12]} "
          f"c_misroute=${cfg.c_misroute_usd:.2f} c_human=${cfg.c_human_usd:.2f}")

    # Compute everything before writing anything (batch atomicity, as in cost_model).
    evaluations = build_all(cfg, preds_dir=args.preds_dir, results_path=args.results,
                            thresholds_dir=args.thresholds_dir,
                            op_version=args.op_version,
                            verify_replay=args.verify_replay)
    summary = build_summary(evaluations, cfg, op_version=args.op_version)

    for name, ev in evaluations.items():
        path = cost_model.write_result_json(
            ev, args.out_dir / result_filename(name, cfg, args.op_version))
        print(f"\n=== {name} (n={ev['n_examples']}) -> {path.name}")
        for pname, p in ev["policies"].items():
            total = p["expected_cost_per_1k"]["total"]
            acc = p["accuracy_machine"]
            cov_a = p["routing"]["coverage_a"]
            print(f"  {pname:24s} covA={'  n/a ' if cov_a is None else f'{cov_a:.4f}'} "
                  f"mach={p['routing']['coverage_machine']:.4f} "
                  f"hum={p['routing']['human_rate']:.4f} "
                  f"acc_mach={'  n/a ' if acc is None else f'{acc:.4f}'} "
                  f"acc_sys={p['accuracy_system']:.4f} "
                  f"cost/1k=${total['point']:8.2f} "
                  f"[{total['ci_lo']:8.2f}, {total['ci_hi']:8.2f}]")
        for d in ev["paired_deltas"]:
            band = d["delta_cost_per_1k"]
            mark = "✓" if band["excludes_zero"] else "·"
            print(f"  {mark} {d['a']} - {d['b']:24s} "
                  f"cost {band['point']:9.2f} [{band['ci_lo']:9.2f}, {band['ci_hi']:9.2f}]"
                  f"  acc {d['delta_accuracy_system']['point']:+.4f}")

    summary_path = cost_model.write_result_json(
        summary, args.out_dir / result_filename("summary", cfg, args.op_version))
    dom = summary["dominance"]
    print("\ndominance (paired cost CI excludes zero; model baselines only):")
    for key, row in dom["by_router"].items():
        print(f"  {key:52s} beats {row['dominated_model_baselines']} "
              f"(n={row['n_model_baselines_dominated']}); "
              f"not {row['not_dominated']}")
    print(f"headline router meets >=2 model baselines: "
          f"{dom['headline_router_meets_two_model_baselines']}")
    print(f"owner decision 1: {summary['owner_decision_1_cross_family']['verdict']}")
    print(f"summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    import sys

    from triage_lab.router_sim import main as _main

    sys.exit(_main())
