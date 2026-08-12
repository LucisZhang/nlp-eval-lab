"""Drift charts + escalation-rate-over-time from the frozen results log (Phase 5, §7.2).

The last Phase 5 exhibit. Everything upstream measured one drift channel at a time --
yearly evals (runs.jsonl), prior shift, OOV, perturbations. This module is the *rollup*:
one machine-readable ``results/drift/summary.json`` that carries the whole 2022-H2 ->
2026-H1 timeline, and three committed SVGs rendered from it. Phase 6's demo timeline reads
the same JSON, so the charts and the demo can never disagree about a number.

Strictly derivative and strictly offline: no model is fitted, no API is called, nothing is
appended to ``results/runs.jsonl``. Two kinds of number go in, and they are never mixed:

- **Logged metrics** (macro-F1, ECE, accuracy with their CIs) are COPIED from the run
  records. They are not recomputed here -- a chart that re-derives a headline number is a
  second source of truth, and the one that disagrees is always the chart. The artifacts are
  still opened, because ``cost_model.load_artifact_verified`` re-runs the repo's full
  provenance + structural + aggregate gate (which pins recomputed accuracy/macro-F1 against
  the record at 1e-9); we consume the gate, not its arithmetic.
- **Escalation rates and policy metrics** are computed FRESH from those gate-verified
  artifacts under the frozen Phase 4 thresholds, because no run record contains them: a
  threshold is a policy decision applied after the fact, and it has never been applied to
  the yearly slices before.

**Calibration space is the load-bearing precondition.** The v1 lesson (EXPERIMENT_LOG,
Phase 4) was a tau fit on raw CAL probabilities and applied to an isotonic-calibrated TEST
artifact -- the same number in two different spaces. It is not repeated here: only the
``v2-isocal`` derivation is loaded, and the yearly Tier A runs are isotonic-calibrated in
the same space as the CAL rung the tau was fit on. There is deliberately no v1 code path,
so a v1 drift number cannot be produced by passing a flag.

**Five escalation arms, five thresholds, one per (policy family, support).** A tau* is the
argmin of a *specific* policy's cost curve; transplanting one family's constant onto another
family is the same class of error as transplanting across calibration spaces. So each arm
gets the tau the sweep derived for it, loaded from ``results/thresholds/`` via the router's
own validating loader (never transcribed):

- ``a_to_human`` on the FULL slices -- tau from ``(a_to_human, full_cal)``, CAL escalation
  0.0994. The deployment-shaped question: what fraction of an incoming year does the shipped
  Tier A gate hand to a human?
- ``a_to_human`` on the PAIRED 1,500-row subsets -- tau from ``(a_to_human, paired_subset)``,
  CAL escalation 0.1067. Same policy, Tier C's support; it is what makes the cascade arm
  readable, since a support change and a policy change would otherwise be confounded.
- ``a_to_c_parsefail_human`` on the PAIRED subsets -- tau from ``(a_to_c, paired_subset)``,
  CAL escalation 0.0400. Tier C is an unconditional terminal stop and ``parse_failed`` on
  the row's receipt is the only Tier C -> human signal (UPGRADE_PLAN §4.2 as narrowed by the
  2026-08-07 amendment). Its CAL optimum escalates far LESS than the human arm's, because on
  CAL the LLM is not more accurate than Tier A on the rows Tier A is unsure about, so
  escalation buys error rather than removing it.
- ``a_to_b`` on the FULL slices -- tau from ``(a_to_b, full_cal)``, CAL escalation 0.3165 --
  and on the PAIRED subsets -- tau from ``(a_to_b, paired_subset)``, CAL escalation 0.5500.
  Tier B2 (DistilBERT) is the cascade rung: it is the certified top Tier B point on TEST-IID
  and the deployment point the demo ships. Its CAL optimum escalates by far the MOST of any
  arm, which is the whole finding: unlike the LLM, B2 *is* more accurate than Tier A on the
  rows Tier A is unsure about, and at amortized-compute prices sending a third of the traffic
  to it is cheaper than being wrong. The paired duplicate exists for the same reason the
  ``a_to_human`` one does -- so the Tier B and Tier C cascades can be read against each other
  on identical rows, with support held fixed.

**There is no tau_B in this module, and that is a frozen fact, not an omission.** The Phase 4
``a_to_b`` family is ``escalation_arm: tier_b_terminal`` -- ONE gate. Tier B2 answers every
escalated row; the residual human rate is structurally 0.0 and is reported as such (with its
degenerate CI) rather than hidden. The only frozen tau_B in the repo belongs to ``a_to_b_to_c``
(0.5769749672571918, paired subset), where the rows BELOW it go to Tier C at Tier C's measured
prices -- not to a human at ``c_human``. Re-terminating that constant on a human queue would
be a cross-family transplant of exactly the kind the paragraph above refuses, so an
``a_to_b_to_human`` series is not produced here; it would need its own CAL sweep and its own
threshold file first.

**Cost model v2.** Tier B has a price only under ``configs/cost_model_v2.yaml``, so that is
this module's default cost config and the generation its tau constants are loaded from. v2's
two dollar parameters are byte-identical to v1's by design, and the ``a_to_human`` /
``a_to_c`` tau* files under both generations carry the same numbers -- so the three Tier A/C
arms are unchanged by the switch, and only the *set of available policies* grows.

**Parse failure is read from receipts, never from the artifact.** A parse-failed row's
``y_pred`` in the Tier C parquet is the frozen *fallback* label (``card``), indistinguishable
from a real prediction. The convention this module reuses verbatim is
``cost_model.join_parse_failed(ids, raw_log_path, record=...)``: the per-row boolean comes
off the same receipts that already passed the cost gate, joined ON ``complaint_id``, and a
missing or non-boolean flag is a hard failure rather than a default. Scoring follows
``router_sim.a_to_c_policy``: a parse-failed escalated row goes to a human and its fallback
label is DISCARDED, not scored.

**Both accuracy views, always** (``router_sim.HUMAN_CREDIT_NOTE``). ``*_system`` credits
every human-routed row as correct, matching the cost model's ``P(error|human)=0``
assumption; ``*_machine`` / ``*_answered`` covers machine-answered rows only, where no
assumption applies. Sonnet's parse-fail rate roughly doubles across the timeline, so the two
views separate over time and quoting either alone would narrate the assumption as a result.

**Supports differ and are labeled everywhere.** Tier A's slices are full (104,443 at
test_iid, 20,000 per year); Tier C's are uniform-random subsamples (5,000 at test_iid for
Haiku, 1,500 everywhere else). The paired arms are pinned to the 1,500 ids of the Sonnet
artifact for that slice -- verified to be exactly the Haiku ids on the yearly slices and a
verified subset of Haiku's 5,000 at test_iid -- so every paired number in the timeline sits
on identical rows across models and slices. Every chart states this in a footnote; a
macro-F1 curve on 20,000 rows plotted next to one on 1,500 without that label is a
different-population comparison dressed as a trend.

**Bootstrap** follows the harness contract (``N_RESAMPLES``, ``BOOTSTRAP_SEED``, percentile).
One index draw per replicate feeds every statistic of that arm/slice, so escalation rate and
answered-set quality move together; ``default_rng(seed)`` depends only on (seed, n), so
Haiku and Sonnet -- equal-sized artifacts on identical rows -- receive byte-identical index
vectors and their arms are paired for free.

Tier B1 (ModernBERT) is ``pending`` by construction, not omitted: a slot with an explicit
evidence class is the honest report of a tier that has not landed. Tier B2 has landed and
carries a full series.
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from triage_lab import cost_model, harness, metrics, predictions, router_sim, threshold_opt

REPO_ROOT = harness.REPO_ROOT
DEFAULT_PREDS_DIR = harness.DEFAULT_PREDS_DIR
DEFAULT_RESULTS_PATH = harness.DEFAULT_RESULTS_PATH
DEFAULT_SPLITS_STATS_PATH = harness.DEFAULT_SPLITS_STATS_PATH
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "drift"
DEFAULT_CHARTS_DIRNAME = "charts"
# v2, not cost_model's own v1 default: the a_to_b arm's tau exists only under a cost config
# that prices Tier B, and mixing two cost generations inside one summary would make the
# `cost_model` block a claim about half the file. See the module docstring.
DEFAULT_COST_CONFIG = REPO_ROOT / "configs" / "cost_model_v2.yaml"

SCHEMA_VERSION = 1
JSON_ROUND = 10

# --- frozen protocol constants (none of these is a CLI flag) -----------------------------

# The timeline, oldest first. test_iid (2022-H2) is the baseline the drift years are read
# against: it is the slice the reported Phase 1-4 numbers live on.
SLICES: tuple[tuple[str, str], ...] = (
    ("test_iid", "2022-H2"),
    ("test_drift_2023", "2023"),
    ("test_drift_2024", "2024"),
    ("test_drift_2025", "2025"),
    ("test_drift_2026h1", "2026-H1"),
)
SLICE_ORDER: tuple[str, ...] = tuple(s for s, _ in SLICES)
SLICE_LABELS: dict[str, str] = dict(SLICES)

# Config stem per (tier, slice): the TEST-IID final, and the yearly template.
TIER_CONFIGS: dict[str, tuple[str, str]] = {
    "tier_a": ("tier_a_logreg_test_iid", "tier_a_logreg_test_drift_{year}"),
    "tier_b2": ("tier_b2_distilbert_s0", "tier_b2_distilbert_s0_test_drift_{year}"),
    "tier_c_haiku": ("tier_c_haiku_zeroshot_test_iid",
                     "tier_c_haiku_zeroshot_test_drift_{year}"),
    "tier_c_sonnet": ("tier_c_sonnet_zeroshot_test_iid",
                      "tier_c_sonnet_zeroshot_test_drift_{year}"),
}
TIER_ORDER: tuple[str, ...] = ("tier_a", "tier_b2", "tier_c_haiku", "tier_c_sonnet")
TIER_DISPLAY: dict[str, str] = {
    "tier_a": "Tier A (LogReg word+char, isotonic)",
    "tier_b2": "Tier B2 DistilBERT (temperature-scaled)",
    "tier_c_haiku": "Tier C Haiku 4.5 (zero-shot)",
    "tier_c_sonnet": "Tier C Sonnet 5 (zero-shot)",
}
GATE_TIER = "tier_a"                 # the only tier with a confidence gate (§4.2 amendment)
CASCADE_TIER = "tier_b2"             # the Tier B rung of the a_to_b arm (router_sim §4.2)
PAIRED_ID_SOURCE = "tier_c_sonnet"   # 1,500 rows on EVERY slice, incl. test_iid
PAIRED_N = 1500
TERMINAL_MODELS: tuple[str, ...] = ("tier_c_haiku", "tier_c_sonnet")
# Tiers whose p_max is a real confidence signal, so ECE is defined on it. Tier C is excluded
# by construction (ECE_TIER_C_EXCLUSION_NOTE); Tier A is isotonic-calibrated, B2 is
# temperature-scaled on CAL, and both are gate-relevant: the router thresholds this number.
ECE_TIERS: tuple[str, ...] = ("tier_a", "tier_b2")

LOGGED_METRICS: tuple[str, ...] = ("macro_f1", "ece", "accuracy")

# Which frozen tau belongs to which arm. Keys are this module's arm names; values are the
# (policy_family, dataset) key of the threshold file the Phase 4 sweep wrote.
ARM_A_TO_HUMAN_FULL = "a_to_human__full_slice"
ARM_A_TO_HUMAN_PAIRED = "a_to_human__paired_subset"
ARM_A_TO_C = "a_to_c_parsefail_human__paired_subset"
ARM_A_TO_B_FULL = "a_to_b__full_slice"
ARM_A_TO_B_PAIRED = "a_to_b__paired_subset"
ARM_THRESHOLD_KEYS: dict[str, tuple[str, str]] = {
    ARM_A_TO_HUMAN_FULL: (threshold_opt.FAMILY_A_TO_HUMAN, threshold_opt.DATASET_FULL_CAL),
    ARM_A_TO_HUMAN_PAIRED: (threshold_opt.FAMILY_A_TO_HUMAN, threshold_opt.DATASET_PAIRED),
    ARM_A_TO_C: (threshold_opt.FAMILY_A_TO_C, threshold_opt.DATASET_PAIRED),
    ARM_A_TO_B_FULL: (threshold_opt.FAMILY_A_TO_B, threshold_opt.DATASET_FULL_CAL),
    ARM_A_TO_B_PAIRED: (threshold_opt.FAMILY_A_TO_B, threshold_opt.DATASET_PAIRED),
}
ARM_ORDER: tuple[str, ...] = (ARM_A_TO_HUMAN_FULL, ARM_A_TO_HUMAN_PAIRED, ARM_A_TO_C,
                              ARM_A_TO_B_FULL, ARM_A_TO_B_PAIRED)

# v2-isocal ONLY. The v1 raw-CAL derivation is the documented calibration-mismatch lesson
# and is deliberately unreachable from here (see module docstring).
OP_VERSION = router_sim.OP_V2

METRIC_AGREEMENT_TOL = 1e-9

# The label-drift event the whole timeline is read against. Dates are the ones frozen in
# taxonomy_map.yaml's era-boundary comment; announcement and first observation differ, so
# both are carried and the charts annotate the boundary rather than pretending to a date.
TAXONOMY_CHANGE: dict = {
    "event": "CFPB credit-reporting product consolidation",
    "announced": "2023-04",
    "observed_in_data": "2023-08",
    "era_boundary": "B2",
    "first_affected_slice": "test_drift_2023",
    "last_unaffected_slice": "test_iid",
    "affected_class": "credit_reporting",
    "source": "taxonomy_map.yaml (frozen with the snapshot); UPGRADE_PLAN.md §5",
    "note": (
        "Announced 2023-04, first observed in the frozen snapshot at 2023-08, so the "
        "test_drift_2023 slice STRADDLES the change rather than sitting after it. Charts "
        "draw the marker at the test_iid|2023 boundary (the last slice fully before the "
        "change) and shade the 2023 slice as the straddling one; a single dated vertical "
        "line inside a yearly bucket would claim a resolution the x-axis does not have."
    ),
    "evidence_class": "documented",
}

TIER_B_PENDING: dict = {
    "status": "pending",
    "tiers": ["tier_b1_modernbert"],
    "landed": ["tier_b2_distilbert"],
    "note": (
        "Tier B2 (DistilBERT) HAS landed: it carries a yearly macro-F1 and ECE series and "
        "is the Tier B rung of the a_to_b escalation arm. It was run first, ahead of B1, "
        "because it is the certified top Tier B point on TEST-IID (Phase 4 backfill) and "
        "the cascade rung the router and the demo actually ship. Tier B1 (ModernBERT) has "
        "no yearly drift runs and stays an explicit PENDING slot: its series costs roughly "
        "8h on MPS (4 yearly slices x ~107k rows of CAL + slice inference) and whether to "
        "spend that is an OPEN OWNER DECISION, not a measurement that was attempted and "
        "failed. Reported as a slot rather than omitted: 'not measured' and 'measured to "
        "be absent' are different claims. When B1 lands, one entry in TIER_CONFIGS adds "
        "its series with no other change."
    ),
    "evidence_class": "pending",
}

ECE_TIER_C_EXCLUSION_NOTE = (
    "Tier C is excluded from the ECE chart. Its p_max is degenerate: structured output "
    "returns one label, the artifact's probability vector is one-hot, so every prediction "
    "carries confidence 1.0 and ECE collapses to the error rate rather than measuring "
    "calibration (Phase 4 finding). Plotting it beside Tier A's isotonic-calibrated ECE "
    "would invite a comparison that is not defined."
)

SUPPORT_NOTE = (
    "Supports differ by tier and are NOT comparable as populations: Tier A and Tier B2 are "
    "the full slice (104,443 rows at 2022-H2, 20,000 per drift year); Tier C is a "
    "uniform-random subsample (5,000 at 2022-H2 for Haiku, 1,500 elsewhere). Paired arms "
    "are pinned to the 1,500 ids of that slice's Sonnet artifact, verified identical to "
    "Haiku's on the drift years and a verified subset of Haiku's 5,000 at 2022-H2."
)

TIER_B_TERMINAL_NOTE = (
    "The a_to_b arm has ONE gate by construction: the frozen Phase 4 family is "
    "`escalation_arm: tier_b_terminal` -- per router_sim.a_to_b_policy, a local classifier "
    "always emits a label, so there is no analogue of Tier C's parse failure. Tier B2 "
    "answers every escalated row and the residual human rate is structurally 0.0 (reported "
    "with its degenerate CI, not hidden). "
    "`escalation_rate` is therefore the escalate-to-B2 rate and `human_rate` is the B2->human "
    "rate. The only frozen tau_B in the repo is a_to_b_to_c's (0.5769749672571918, paired "
    "subset), where rows below it go to Tier C at Tier C's measured prices rather than to a "
    "human at c_human; re-terminating that constant on a human queue would be the "
    "cross-family transplant this module refuses everywhere else, so no a_to_b_to_human "
    "series is produced. Consequences for the accuracy views: with an empty human arm "
    "accuracy_machine == accuracy_system and macro_f1_answered == macro_f1_system on this "
    "arm, and no P(error|human)=0 assumption is load-bearing for it -- unlike the a_to_human "
    "arms, whose *_system views assume it on a tenth of the traffic."
)

HUMAN_CREDIT_NOTE = router_sim.HUMAN_CREDIT_NOTE
PARSE_FAIL_CONVENTION_NOTE = (
    "Per-row parse failure is read from the run's committed per-call receipts via "
    "cost_model.join_parse_failed (joined ON complaint_id, missing/non-boolean flag = hard "
    "failure), NOT from the predictions parquet: a parse-failed row's y_pred there is the "
    "frozen fallback label ('card') and is indistinguishable from a real prediction. "
    "Scoring follows router_sim.a_to_c_policy: an escalated parse-failed row routes to a "
    "human and its fallback label is discarded, never scored."
)

EVIDENCE_CLASS: dict[str, str] = {
    "series.logged.*.macro_f1": "measured",
    "series.logged.*.ece": "measured",
    "series.logged.*.accuracy": "measured",
    "series.logged.*.n": "measured",
    "series.escalation.*.escalation_rate":
        "measured (derived from frozen artifacts under frozen tau)",
    "series.escalation.*.coverage":
        "measured (derived from frozen artifacts under frozen tau)",
    "series.escalation.*.human_rate":
        "measured (derived from frozen artifacts under frozen tau)",
    "series.escalation.*.accuracy_answered":
        "measured (derived from frozen artifacts under frozen tau)",
    "series.escalation.*.macro_f1_answered":
        "measured (derived from frozen artifacts under frozen tau)",
    "series.escalation.*.accuracy_machine":
        "measured (derived from frozen artifacts under frozen tau)",
    "series.escalation.*.accuracy_system":
        "measured (derived from frozen artifacts under frozen tau); credits human-routed "
        "rows as correct per the cost model's P(error|human)=0 ASSUMPTION",
    "series.escalation.*.macro_f1_system":
        "measured (derived from frozen artifacts under frozen tau); credits human-routed "
        "rows as correct per the cost model's P(error|human)=0 ASSUMPTION",
    "series.escalation.*.n_parse_failed":
        "measured (per-call receipts, committed under results/tier_c_raw/)",
    "thresholds.*.tau_star":
        "derived on CAL (cost argmin) under cost-model parameters that are ESTIMATED",
    "thresholds.*.cal_escalation_rate": "derived on CAL",
    "cost_model.params.c_misroute_usd": "estimated",
    "cost_model.params.c_human_usd": "estimated",
    "cost_model.api_cost.tier_a": "estimated",
    "cost_model.api_cost.tier_b1": "estimated",
    "cost_model.api_cost.tier_b2": "estimated",
    "cost_model.api_cost.tier_c": "measured",
    "taxonomy_change": "documented",
    "series.logged.tier_b2": "measured",
    "series.escalation.a_to_b":
        "measured (derived from frozen artifacts under frozen tau)",
    "series.logged.tier_b1": "pending",
    "series.escalation.tier_b1": "pending",
}


# ---------------------------------------------------------------------------
# Run + artifact resolution
# ---------------------------------------------------------------------------

def config_stem(tier: str, split: str) -> str:
    """Config stem of the run that evaluates `tier` on `split`."""
    if tier not in TIER_CONFIGS:
        raise ValueError(f"unknown tier {tier!r}; known: {sorted(TIER_CONFIGS)}")
    if split not in SLICE_LABELS:
        raise ValueError(f"unknown slice {split!r}; known: {list(SLICE_ORDER)}")
    iid_stem, drift_template = TIER_CONFIGS[tier]
    if split == "test_iid":
        return iid_stem
    return drift_template.format(year=split.removeprefix("test_drift_"))


def load_slice_artifacts(records: dict, preds_dir) -> dict[tuple[str, str], object]:
    """Gate-verified artifact for every (tier, slice) the timeline needs.

    Missing run record or missing parquet is a hard failure naming the cell, not a series
    that silently disappears from a chart. ``load_artifact_verified`` is the repo's full
    gate: provenance (run_id / config hash / split hash / git sha / snapshot hash), then
    ``predictions.verify_artifact`` -- ids unique and inside the frozen split, y_true
    agreeing with it, p_max exactly the row max, y_pred the argmax, and every recomputed
    aggregate matching the logged record to 1e-9.
    """
    out: dict[tuple[str, str], object] = {}
    missing: list[str] = []
    for tier in TIER_ORDER:
        for split in SLICE_ORDER:
            stem = config_stem(tier, split)
            record = records.get(stem)
            if record is None:
                missing.append(f"{tier}/{split}: no run record for config {stem!r}")
                continue
            art_path = Path(preds_dir) / f"{record['run_id']}.parquet"
            if not art_path.exists():
                missing.append(
                    f"{tier}/{split}: no prediction artifact at {art_path} "
                    f"(run {record['run_id'][:8]}); run `make preds` first"
                )
                continue
            out[(tier, split)] = cost_model.load_artifact_verified(record, preds_dir)
    if missing:
        raise ValueError(
            "the drift timeline is incomplete — " + "; ".join(missing) + ". Every "
            "(tier, slice) cell must resolve to a verified artifact; a chart with a "
            "silently dropped series is not a drift chart"
        )
    return out


def _metric_block(record: dict, key: str) -> dict | None:
    block = (record.get("metrics") or {}).get(key)
    if block is None:
        return None
    return {
        "point": float(block["point"]),
        "ci_lo": float(block["ci_lo"]),
        "ci_hi": float(block["ci_hi"]),
    }


def _support_kind(tier: str, n: int, n_split: int) -> str:
    if n == n_split:
        return "full_slice"
    return f"uniform_random_subsample_{n}"


def check_metric_agreement(art, record: dict) -> dict:
    """Recompute accuracy from the artifact and demand the logged point back at 1e-9.

    ``load_artifact_verified`` already enforces this inside its gate; it is repeated here,
    and its result written into the summary, so the guarantee is visible in the artifact a
    reader holds instead of being an invisible property of a loader they have to trust.
    """
    logged = _metric_block(record, "accuracy")
    if logged is None:
        raise ValueError(
            f"run {record['run_id'][:8]} logs no accuracy metric; the artifact cannot be "
            "cross-checked against the record it is charted under"
        )
    computed = float(metrics.accuracy(art.y_true, art.y_pred, list(art.class_labels)))
    delta = abs(computed - logged["point"])
    if delta > METRIC_AGREEMENT_TOL:
        raise ValueError(
            f"run {record['run_id'][:8]}: artifact-recomputed accuracy {computed!r} "
            f"disagrees with the logged {logged['point']!r} (|delta| = {delta:.3e} > "
            f"{METRIC_AGREEMENT_TOL:g}); the artifact does not describe this run"
        )
    return {"check": "accuracy_recomputed_vs_logged", "computed": computed,
            "logged": logged["point"], "abs_delta": delta, "tol": METRIC_AGREEMENT_TOL,
            "ok": True}


def logged_series(records: dict, artifacts: dict, split_sizes: dict[str, int]) -> list[dict]:
    """One row per (tier, slice): the LOGGED metrics, copied, plus provenance and support."""
    rows: list[dict] = []
    for tier in TIER_ORDER:
        for split in SLICE_ORDER:
            stem = config_stem(tier, split)
            record = records[stem]
            art = artifacts[(tier, split)]
            n = len(art)
            row = {
                "tier": tier,
                "tier_display": TIER_DISPLAY[tier],
                "slice": split,
                "slice_label": SLICE_LABELS[split],
                "run_id": record["run_id"],
                "config_name": stem,
                "config_sha256": record.get("config_sha256", ""),
                "git_sha": record.get("git_sha", ""),
                "split": (record.get("dataset") or {}).get("split", ""),
                "split_sha256": (record.get("dataset") or {}).get("split_sha256", ""),
                "n": n,
                "n_split_full": split_sizes[split],
                "support": _support_kind(tier, n, split_sizes[split]),
                "evidence_class": "measured",
                "metric_agreement": check_metric_agreement(art, record),
            }
            for key in LOGGED_METRICS:
                row[key] = _metric_block(record, key)
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Paired-subset resolution
# ---------------------------------------------------------------------------

def paired_ids(split: str, artifacts: dict) -> np.ndarray:
    """The frozen 1,500 complaint_ids the paired arms live on, for one slice.

    Defined as the Sonnet artifact's ids -- the only Tier C artifact that is 1,500 rows on
    EVERY slice -- and then verified against Haiku's: identical on the drift years, a strict
    subset at test_iid (where Haiku ran 5,000). Verifying rather than assuming is the whole
    content of the word "paired": if the id sets ever diverge, a per-row comparison across
    models is comparing different complaints.
    """
    ids = np.sort(np.asarray(artifacts[(PAIRED_ID_SOURCE, split)].complaint_id,
                             dtype=np.int64))
    if len(ids) != PAIRED_N:
        raise ValueError(
            f"{split}: the paired id source {PAIRED_ID_SOURCE!r} has {len(ids)} rows, not "
            f"the frozen {PAIRED_N}; the paired arms are not defined on this slice"
        )
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{split}: the paired id source has duplicate complaint_ids")
    for tier in TERMINAL_MODELS:
        other = np.asarray(artifacts[(tier, split)].complaint_id, dtype=np.int64)
        if not np.isin(ids, other).all():
            n_absent = int((~np.isin(ids, other)).sum())
            raise ValueError(
                f"{split}: {n_absent} of the {PAIRED_N} paired ids are absent from the "
                f"{tier} artifact; the 'identical rows' pairing claim is false"
            )
    return ids


# ---------------------------------------------------------------------------
# Policy arms
# ---------------------------------------------------------------------------

def _codes(art, ids_index, class_labels) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(art.y_true)[ids_index]
    y_pred = np.asarray(art.y_pred)[ids_index]
    return (metrics.encode_labels(y_true, class_labels),
            metrics.encode_labels(y_pred, class_labels))


def _arm_stats(true_codes: np.ndarray, pred_codes: np.ndarray, answered: np.ndarray,
               to_human: np.ndarray, n_classes: int) -> dict:
    """Every rate and quality number of one arm, from one (possibly resampled) row set.

    ``answered`` = the machine produced the scored label (Tier A above tau, or Tier C
    terminally). ``to_human`` = no machine answered. For ``a_to_human`` the two are exact
    complements; for the cascade they are not, which is why both are carried.
    """
    n = len(true_codes)
    machine = ~to_human
    n_machine = int(machine.sum())
    correct = true_codes == pred_codes
    system_pred = np.where(to_human, true_codes, pred_codes)
    out = {
        "coverage_a": float(answered.mean()),
        "escalation_rate": float((~answered).mean()),
        "human_rate": float(to_human.mean()),
        "coverage_machine": float(n_machine / n),
        "accuracy_system": float(np.where(to_human, True, correct).mean()),
        "macro_f1_system": metrics.macro_f1_from_codes(true_codes, system_pred, n_classes),
    }
    if n_machine:
        out["accuracy_machine"] = float(correct[machine].mean())
        out["macro_f1_answered"] = metrics.macro_f1_from_codes(
            true_codes[machine], pred_codes[machine], n_classes)
    else:
        out["accuracy_machine"] = math.nan
        out["macro_f1_answered"] = math.nan
    return out


BOOTSTRAPPED_KEYS: tuple[str, ...] = (
    "coverage_a", "escalation_rate", "human_rate", "coverage_machine",
    "accuracy_machine", "macro_f1_answered", "accuracy_system", "macro_f1_system",
)


def bootstrap_arm(true_codes, pred_codes, answered, to_human, n_classes: int, *,
                  n_resamples: int = harness.N_RESAMPLES,
                  seed: int = harness.BOOTSTRAP_SEED) -> dict[str, np.ndarray]:
    """Percentile-bootstrap replicates of every arm statistic, one draw per replicate.

    One index vector feeds all statistics of a replicate, so escalation rate and
    answered-set quality co-move exactly as they do in the data. The stream depends only on
    (seed, n), so two models on identical rows get identical draws and their arms are paired
    without extra machinery.
    """
    n = len(true_codes)
    reps = {k: np.empty(n_resamples, dtype=np.float64) for k in BOOTSTRAPPED_KEYS}
    rng = np.random.default_rng(seed)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        stats = _arm_stats(true_codes[idx], pred_codes[idx], answered[idx], to_human[idx],
                           n_classes)
        for key in BOOTSTRAPPED_KEYS:
            reps[key][i] = stats[key]
    return reps


def _ci_block(point: float, values: np.ndarray) -> dict:
    if math.isnan(point):
        return {"point": None, "ci_lo": None, "ci_hi": None}
    lo, hi = np.percentile(values, [harness.CI_LOWER_PCT, harness.CI_UPPER_PCT])
    return {"point": float(point), "ci_lo": float(lo), "ci_hi": float(hi)}


def build_a_to_human_arm(split: str, *, dataset: str, tau: float, art_a, record_a,
                         index, n_resamples: int, seed: int) -> dict:
    """Tier A answers above tau; every escalated row goes to a human, terminally."""
    class_labels = list(art_a.class_labels)
    p_max = np.asarray(art_a.p_max, dtype=np.float64)[index]
    answered = p_max >= tau
    to_human = ~answered
    true_codes, pred_codes = _codes(art_a, index, class_labels)
    point = _arm_stats(true_codes, pred_codes, answered, to_human, len(class_labels))
    reps = bootstrap_arm(true_codes, pred_codes, answered, to_human, len(class_labels),
                         n_resamples=n_resamples, seed=seed)
    n = len(p_max)
    return {
        "policy": "a_to_human",
        "dataset": dataset,
        "slice": split,
        "slice_label": SLICE_LABELS[split],
        "terminal_model": "human",
        "tau": float(tau),
        "n": n,
        "n_artifact": len(art_a),
        "n_answered_a": int(answered.sum()),
        "n_escalated": int((~answered).sum()),
        "n_to_human": int(to_human.sum()),
        "n_parse_failed_escalated": None,
        "n_parse_failed_slice": None,
        "gate_run_id": record_a["run_id"],
        "gate_config": None,
        "terminal_run_id": None,
        **{k: _ci_block(point[k], reps[k]) for k in BOOTSTRAPPED_KEYS},
    }


def build_a_to_b_arm(split: str, *, dataset: str, tau: float, art_a, record_a, index_a,
                     art_b, record_b, index_b, n_resamples: int, seed: int) -> dict:
    """Tier A above tau, else Tier B2 TERMINALLY. One gate; the human arm is empty.

    Row-level construction is `router_sim.a_to_b_policy`'s: the escalated rows take Tier B2's
    label, nothing routes to a human, and the two artifacts are paired ON complaint_id (never
    on row order) with `y_true` cross-checked, so a reordered artifact cannot mix one
    complaint's confidence with another's label while the aggregates stay plausible.

    See TIER_B_TERMINAL_NOTE for why there is no tau_B here.
    """
    class_labels = list(art_a.class_labels)
    if list(art_b.class_labels) != class_labels:
        raise ValueError(
            f"{split}/{CASCADE_TIER}: Tier A and Tier B artifacts declare different class "
            "label orders; a cascade over them would mix two label spaces"
        )
    p_max = np.asarray(art_a.p_max, dtype=np.float64)[index_a]
    answered_by_a = p_max >= tau

    true_a, pred_a = _codes(art_a, index_a, class_labels)
    true_b, pred_b = _codes(art_b, index_b, class_labels)
    if not np.array_equal(true_a, true_b):
        n_bad = int(np.count_nonzero(true_a != true_b))
        raise ValueError(
            f"{split}/{CASCADE_TIER}: Tier A and Tier B disagree on y_true for {n_bad} of "
            f"{len(true_a)} joined rows; the two are not scored against the same answer key"
        )
    pred_codes = np.where(answered_by_a, pred_a, pred_b)
    to_human = np.zeros(len(pred_codes), dtype=bool)   # tier_b_terminal: no human arm

    point = _arm_stats(true_a, pred_codes, answered_by_a, to_human, len(class_labels))
    reps = bootstrap_arm(true_a, pred_codes, answered_by_a, to_human, len(class_labels),
                         n_resamples=n_resamples, seed=seed)
    return {
        "policy": threshold_opt.FAMILY_A_TO_B,
        "dataset": dataset,
        "slice": split,
        "slice_label": SLICE_LABELS[split],
        "terminal_model": CASCADE_TIER,
        "tau": float(tau),
        "n": len(p_max),
        "n_artifact": len(art_b),
        "n_answered_a": int(answered_by_a.sum()),
        "n_escalated": int((~answered_by_a).sum()),
        "n_to_human": int(to_human.sum()),
        "n_parse_failed_escalated": None,
        "n_parse_failed_slice": None,
        "gate_run_id": record_a["run_id"],
        "gate_config": config_stem(GATE_TIER, split),
        "terminal_run_id": record_b["run_id"],
        **{k: _ci_block(point[k], reps[k]) for k in BOOTSTRAPPED_KEYS},
    }


def build_a_to_c_arm(split: str, terminal_tier: str, *, tau: float, art_a, record_a,
                     index_a, art_c, record_c, index_c, parse_failed,
                     n_resamples: int, seed: int) -> dict:
    """Tier A above tau, else Tier C TERMINALLY; parse failure is the only Tier C->human arm.

    Row-level construction is `router_sim.a_to_c_policy` verbatim, including the decision to
    discard a parse-failed row's fallback label rather than score it.
    """
    class_labels = list(art_a.class_labels)
    if list(art_c.class_labels) != class_labels:
        raise ValueError(
            f"{split}/{terminal_tier}: Tier A and Tier C artifacts declare different class "
            "label orders; a cascade over them would mix two label spaces"
        )
    p_max = np.asarray(art_a.p_max, dtype=np.float64)[index_a]
    answered_by_a = p_max >= tau

    true_a, pred_a = _codes(art_a, index_a, class_labels)
    true_c, pred_c = _codes(art_c, index_c, class_labels)
    if not np.array_equal(true_a, true_c):
        n_bad = int(np.count_nonzero(true_a != true_c))
        raise ValueError(
            f"{split}/{terminal_tier}: Tier A and Tier C disagree on y_true for {n_bad} of "
            f"{len(true_a)} paired rows; the two are not scored against the same answer key"
        )
    parse_failed = np.asarray(parse_failed, dtype=bool)
    pred_codes = np.where(answered_by_a, pred_a, pred_c)
    to_human = (~answered_by_a) & parse_failed

    point = _arm_stats(true_a, pred_codes, answered_by_a, to_human, len(class_labels))
    reps = bootstrap_arm(true_a, pred_codes, answered_by_a, to_human, len(class_labels),
                         n_resamples=n_resamples, seed=seed)
    return {
        "policy": "a_to_c_parsefail_human",
        "dataset": threshold_opt.DATASET_PAIRED,
        "slice": split,
        "slice_label": SLICE_LABELS[split],
        "terminal_model": terminal_tier,
        "tau": float(tau),
        "n": len(p_max),
        "n_artifact": len(art_c),
        "n_answered_a": int(answered_by_a.sum()),
        "n_escalated": int((~answered_by_a).sum()),
        "n_to_human": int(to_human.sum()),
        "n_parse_failed_escalated": int(to_human.sum()),
        "n_parse_failed_slice": int(parse_failed.sum()),
        "gate_run_id": record_a["run_id"],
        "gate_config": config_stem(GATE_TIER, split),
        "terminal_run_id": record_c["run_id"],
        **{k: _ci_block(point[k], reps[k]) for k in BOOTSTRAPPED_KEYS},
    }


def escalation_series(records: dict, artifacts: dict, thresholds: dict, *,
                      n_resamples: int = harness.N_RESAMPLES,
                      seed: int = harness.BOOTSTRAP_SEED) -> list[dict]:
    """Every arm x slice (x terminal model) row of the escalation-rate-over-time exhibit."""
    rows: list[dict] = []
    tau_full = thresholds[ARM_THRESHOLD_KEYS[ARM_A_TO_HUMAN_FULL]].tau_star
    tau_paired_human = thresholds[ARM_THRESHOLD_KEYS[ARM_A_TO_HUMAN_PAIRED]].tau_star
    tau_paired_c = thresholds[ARM_THRESHOLD_KEYS[ARM_A_TO_C]].tau_star
    tau_full_b = thresholds[ARM_THRESHOLD_KEYS[ARM_A_TO_B_FULL]].tau_star
    tau_paired_b = thresholds[ARM_THRESHOLD_KEYS[ARM_A_TO_B_PAIRED]].tau_star

    for split in SLICE_ORDER:
        art_a = artifacts[(GATE_TIER, split)]
        record_a = records[config_stem(GATE_TIER, split)]
        if (CASCADE_TIER, split) not in artifacts:
            raise ValueError(
                f"{split}: the a_to_b escalation arm needs the {CASCADE_TIER} artifact for "
                f"this slice (config {config_stem(CASCADE_TIER, split)!r}) and it did not "
                "resolve; the Tier B2 yearly drift runs must land in results/runs.jsonl and "
                "`make preds` must have written their parquet before this exhibit can be "
                "built. A missing rung is a hard failure, never a dropped series"
            )
        art_b = artifacts[(CASCADE_TIER, split)]
        record_b = records[config_stem(CASCADE_TIER, split)]
        keep = paired_ids(split, artifacts)
        index_a = threshold_opt.restrict_to_ids(art_a, keep)
        if len(index_a) != PAIRED_N:
            raise ValueError(
                f"{split}: restricting the Tier A artifact to the paired ids yielded "
                f"{len(index_a)} rows, not {PAIRED_N}"
            )

        rows.append(build_a_to_human_arm(
            split, dataset=threshold_opt.DATASET_FULL_CAL, tau=tau_full, art_a=art_a,
            record_a=record_a, index=slice(None), n_resamples=n_resamples, seed=seed))
        rows.append(build_a_to_human_arm(
            split, dataset=threshold_opt.DATASET_PAIRED, tau=tau_paired_human, art_a=art_a,
            record_a=record_a, index=index_a, n_resamples=n_resamples, seed=seed))

        # a_to_b, both supports. The full-slice arm is the deployment-shaped question; the
        # paired one exists so the Tier B and Tier C cascades can be read against each other
        # on identical rows (support held fixed, only the policy varying).
        ids_full = np.asarray(art_a.complaint_id, dtype=np.int64)
        rows.append(build_a_to_b_arm(
            split, dataset=threshold_opt.DATASET_FULL_CAL, tau=tau_full_b, art_a=art_a,
            record_a=record_a, index_a=slice(None), art_b=art_b, record_b=record_b,
            index_b=threshold_opt.restrict_to_ids(art_b, ids_full),
            n_resamples=n_resamples, seed=seed))
        rows.append(build_a_to_b_arm(
            split, dataset=threshold_opt.DATASET_PAIRED, tau=tau_paired_b, art_a=art_a,
            record_a=record_a, index_a=index_a, art_b=art_b, record_b=record_b,
            index_b=threshold_opt.restrict_to_ids(art_b, keep),
            n_resamples=n_resamples, seed=seed))

        for tier in TERMINAL_MODELS:
            art_c = artifacts[(tier, split)]
            record_c = records[config_stem(tier, split)]
            index_c = threshold_opt.restrict_to_ids(art_c, keep)
            if len(index_c) != PAIRED_N:
                raise ValueError(
                    f"{split}/{tier}: restricting the Tier C artifact to the paired ids "
                    f"yielded {len(index_c)} rows, not {PAIRED_N}"
                )
            raw_log_path = (record_c.get("extra") or {}).get("raw_log_path")
            if not raw_log_path:
                raise ValueError(
                    f"{split}/{tier}: run {record_c['run_id'][:8]} records no "
                    "extra.raw_log_path, so per-row parse failure cannot be read from its "
                    "receipts; the only Tier C -> human signal would have to be defaulted"
                )
            parse_failed = cost_model.join_parse_failed(keep, raw_log_path, record=record_c)
            logged_pf = (record_c.get("extra") or {}).get("parse_failures")
            if logged_pf is not None and len(art_c) == PAIRED_N \
                    and int(parse_failed.sum()) != int(logged_pf):
                raise ValueError(
                    f"{split}/{tier}: receipts report {int(parse_failed.sum())} parse "
                    f"failures over the paired rows but the run record logs "
                    f"{int(logged_pf)} over its {len(art_c)} rows; the receipts joined here "
                    "are not this run's"
                )
            rows.append(build_a_to_c_arm(
                split, tier, tau=tau_paired_c, art_a=art_a, record_a=record_a,
                index_a=index_a, art_c=art_c, record_c=record_c, index_c=index_c,
                parse_failed=parse_failed, n_resamples=n_resamples, seed=seed))
    return rows


# ---------------------------------------------------------------------------
# Summary assembly
# ---------------------------------------------------------------------------

def _tier_b_cal_rung(cal, thresholds_dir) -> dict | None:
    """Which Tier B CAL run the a_to_b tau was fit against, read back off its own file.

    A cascade constant is only interpretable with BOTH rungs named: the tau escalates to a
    specific Tier B checkpoint's CAL confidences, and the arm then applies it to that
    checkpoint's TEST predictions. The identity is read from the (already validated,
    already sha256'd) threshold file rather than transcribed here, for the reason
    ``router_sim.load_cal_thresholds`` gives: a number typed into a module is a second
    source of truth that no gate compares against the file it came from. ``None`` for the
    families that have no Tier B rung — an explicit null, not a missing key.
    """
    obj = json.loads((Path(thresholds_dir) / cal.source_file).read_text())
    tier_b = (obj.get("inputs") or {}).get("tier_b")
    if tier_b is None:
        return None
    return {k: tier_b[k] for k in ("config_name", "run_id", "split", "n_examples",
                                   "per_example_usd", "evidence_class") if k in tier_b}


def threshold_block(thresholds: dict,
                    thresholds_dir=router_sim.DEFAULT_THRESHOLDS_DIR) -> dict:
    """The frozen tau constants, each with the file it was loaded from and its CAL point."""
    out: dict[str, dict] = {}
    for arm in ARM_ORDER:
        cal = thresholds[ARM_THRESHOLD_KEYS[arm]]
        out[arm] = {
            "tier_b_cal_rung": _tier_b_cal_rung(cal, thresholds_dir),
            "policy_family": cal.policy_family,
            "cal_dataset": cal.dataset,
            "tau_star": cal.tau_star,
            "cal_coverage_a": cal.target_coverage_a,
            "cal_escalation_rate": 1.0 - cal.target_coverage_a,
            "cal_n_answered_at_tau_star": cal.n_answered_at_tau_star,
            "tier_a_cal_config": cal.tier_a_config,
            "tier_a_cal_run_id": cal.tier_a_run_id,
            "source_file": f"results/thresholds/{cal.source_file}",
            "source_sha256": cal.source_sha256,
            "evidence_class": EVIDENCE_CLASS["thresholds.*.tau_star"],
        }
    return out


def build_summary(*, records: dict, artifacts: dict, thresholds: dict,
                  cost_config, split_sizes: dict[str, int],
                  thresholds_dir=router_sim.DEFAULT_THRESHOLDS_DIR,
                  n_resamples: int = harness.N_RESAMPLES,
                  seed: int = harness.BOOTSTRAP_SEED,
                  generated_at: str | None = None,
                  git_sha: str | None = None) -> dict:
    """The whole ``results/drift/summary.json`` object. See the module docstring."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "git_sha": git_sha if git_sha is not None else harness._git_sha(),
        "repro_command": (
            "uv run --extra charts python -m triage_lab.drift_charts --all"
        ),
        "op_version": OP_VERSION,
        "slice_order": list(SLICE_ORDER),
        "slice_labels": dict(SLICE_LABELS),
        "slice_sizes": dict(split_sizes),
        "tier_order": list(TIER_ORDER),
        "arm_order": list(ARM_ORDER),
        "taxonomy_change": TAXONOMY_CHANGE,
        "tier_b": TIER_B_PENDING,
        "thresholds": threshold_block(thresholds, thresholds_dir),
        "cost_model": cost_model.config_block(cost_config),
        "series": {
            "logged": logged_series(records, artifacts, split_sizes),
            "escalation": escalation_series(records, artifacts, thresholds,
                                            n_resamples=n_resamples, seed=seed),
        },
        "bootstrap": {
            "n_resamples": n_resamples,
            "seed": seed,
            "method": harness.BOOTSTRAP_METHOD,
            "ci_pct": [harness.CI_LOWER_PCT, harness.CI_UPPER_PCT],
            "resample": "rows_within_slice",
            "note": "one index draw per replicate feeds every statistic of that arm/slice",
            "applies_to": (
                "escalation series ONLY; the logged series carries the CIs the harness "
                "recorded at run time and is not re-bootstrapped here"
            ),
        },
        "notes": {
            "supports": SUPPORT_NOTE,
            "human_credit": HUMAN_CREDIT_NOTE,
            "parse_failures": PARSE_FAIL_CONVENTION_NOTE,
            "ece_tier_c_excluded": ECE_TIER_C_EXCLUSION_NOTE,
            "calibration_space": (
                "Only the v2-isocal derivation is used. The yearly Tier A runs are isotonic-"
                "calibrated in the same p_max space as the CAL rung the tau was fit on "
                f"({thresholds[ARM_THRESHOLD_KEYS[ARM_A_TO_HUMAN_FULL]].tier_a_config}), so "
                "the constants transfer without crossing a probability-space boundary. The "
                "v1 raw-CAL derivation is the documented calibration-mismatch lesson and is "
                "not reachable from this module."
            ),
            "tau_per_family": (
                "Each arm uses the tau its OWN policy family's CAL sweep produced. The "
                "a_to_c cascade's tau is NOT the a_to_human tau: at CAL prices the cascade's "
                "argmin escalates 4.0% where the human arm escalates 10.7% on the same rows. "
                "The a_to_b cascade's tau is a third constant again, and the furthest from "
                "the others — its CAL argmin escalates MORE than either, because Tier B2 "
                "(unlike the LLM) is more accurate than Tier A on the rows Tier A is unsure "
                "about while costing amortized compute rather than per-call spend."
            ),
            "tier_b_terminal": TIER_B_TERMINAL_NOTE,
        },
        "evidence_class": dict(EVIDENCE_CLASS),
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

# Fixed rendering constants: the SVGs are committed, so a figure size, dpi or font change
# rewrites every byte of every chart. Frozen, not configurable.
FIG_SIZE = (9.0, 5.6)
FIG_DPI = 100
HASH_SALT = "triage-lab-drift-charts-v1"
SERIES_STYLE: dict[str, dict] = {
    "tier_a": {"color": "#1f4e79", "marker": "o", "linestyle": "-"},
    "tier_b2": {"color": "#6a3d9a", "marker": "D", "linestyle": "-"},
    "tier_c_haiku": {"color": "#b8531a", "marker": "s", "linestyle": "--"},
    "tier_c_sonnet": {"color": "#2f6b3a", "marker": "^", "linestyle": "-."},
}
ARM_STYLE: dict[str, dict] = {
    ARM_A_TO_HUMAN_FULL: {"color": "#1f4e79", "marker": "o", "linestyle": "-"},
    ARM_A_TO_HUMAN_PAIRED: {"color": "#5b8db8", "marker": "o", "linestyle": ":"},
    ARM_A_TO_B_FULL: {"color": "#6a3d9a", "marker": "D", "linestyle": "-"},
    ARM_A_TO_B_PAIRED: {"color": "#9d7bbf", "marker": "D", "linestyle": ":"},
    "a_to_c__tier_c_haiku": {"color": "#b8531a", "marker": "s", "linestyle": "--"},
    "a_to_c__tier_c_sonnet": {"color": "#2f6b3a", "marker": "^", "linestyle": "-."},
}
# Dash pattern of each arm's CAL operating-point rule on the escalation chart, in ARM_ORDER.
ARM_HLINE_STYLE: dict[str, str] = {
    ARM_A_TO_HUMAN_FULL: "-",
    ARM_A_TO_HUMAN_PAIRED: ":",
    ARM_A_TO_C: "--",
    ARM_A_TO_B_FULL: "-",
    ARM_A_TO_B_PAIRED: ":",
}
TAXONOMY_LINE_X = 0.5   # the test_iid | 2023 boundary
TAXONOMY_BAND = (0.5, 1.5)  # the 2023 slice, which straddles the observed change
FOOTNOTE_WRAP_COLS = 158

# Chart copy, frozen alongside the figures it is committed with. Every chart carries its
# evidence class and its support caveat ON the chart, because a chart travels away from the
# JSON that qualifies it. No literal '$' is allowed in any of this text: matplotlib parses a
# pair of them as mathtext and would silently italicise everything between.
FOOTNOTE_MACRO_F1: list[str] = [
    ("Evidence class: MEASURED — every point and CI is copied from its run record in "
     "results/runs.jsonl (n=1,000 percentile bootstrap, seed 20260805); nothing here is "
     "recomputed."),
    ("SUPPORTS DIFFER: Tier A and Tier B2 = full slice (104,443 rows at 2022-H2, 20,000 per "
     "drift year); Tier C = uniform-random subsample (5,000 at 2022-H2 for Haiku, 1,500 "
     "elsewhere). Not the same population."),
    ("Tier B1 (ModernBERT): PENDING — no yearly drift runs exist, so it has no series here. "
     "B2 was run first as the certified top Tier B point and the shipped cascade rung."),
]
FOOTNOTE_ECE: list[str] = [
    "Evidence class: MEASURED — copied from results/runs.jsonl run records.",
    ("TIER C EXCLUDED BY CONSTRUCTION: structured output returns one label, so its p_max is "
     "a degenerate one-hot 1.0 on every row and ECE collapses to the error rate instead of "
     "measuring calibration (Phase 4 finding)."),
    ("Tier A is isotonic-calibrated, Tier B2 temperature-scaled — both fit on CAL, both "
     "gate-relevant: the router thresholds exactly this p_max. Tier B1: PENDING."),
]
FOOTNOTE_ESCALATION: list[str] = [
    ("Evidence class: MEASURED (derived from frozen prediction artifacts under frozen τ; "
     "95% percentile bootstrap over rows, n=1,000, seed 20260805). τ itself is DERIVED on "
     "CAL under ESTIMATED cost-model parameters (c_misroute = 6.00 USD, c_human = 2.50 USD; "
     "cost model v2, which prices Tier B compute as an ESTIMATE)."),
    ("Each arm uses the τ its OWN policy family's CAL sweep produced — the a_to_c argmin "
     "escalates far LESS than the human arm's (Tier C is not more accurate than Tier A on "
     "the rows Tier A is unsure about) and the a_to_b argmin far MORE (Tier B2 is)."),
    ("a_to_b has ONE gate: Tier B2 answers every escalated row, so its human rate is "
     "structurally 0.0 and its escalation rate is the escalate-to-B2 rate. No frozen τ_B "
     "for an A→B→human policy exists; see notes.tier_b_terminal in summary.json."),
    ("SUPPORTS DIFFER: the full-slice arms are 104,443 / 20,000 rows; the paired arms are "
     "the same 1,500 ids per slice. Tier B1: PENDING."),
]


def _require_matplotlib():
    try:
        import matplotlib
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "matplotlib is not installed. Chart rendering lives in the OPTIONAL `charts` "
            "extra so the core reproduction path stays lean:\n"
            "    uv sync --frozen --extra charts\n"
            "    uv run --extra charts python -m triage_lab.drift_charts --all\n"
            "(or pass --skip-charts to write summary.json only)"
        ) from exc
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = HASH_SALT
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["axes.grid"] = True
    matplotlib.rcParams["grid.alpha"] = 0.3
    matplotlib.rcParams["grid.linewidth"] = 0.5
    import matplotlib.pyplot as plt

    return plt


def _series_points(rows: list[dict], key: str) -> tuple[list[float], list[float], list[float]]:
    by_slice = {r["slice"]: r for r in rows}
    point, lo, hi = [], [], []
    for split in SLICE_ORDER:
        block = by_slice[split][key]
        point.append(math.nan if block is None else block["point"])
        lo.append(math.nan if block is None else block["ci_lo"])
        hi.append(math.nan if block is None else block["ci_hi"])
    return point, lo, hi


def _annotate_taxonomy(ax, summary: dict) -> None:
    tx = summary["taxonomy_change"]
    ax.axvspan(*TAXONOMY_BAND, color="#c0392b", alpha=0.06, zorder=0)
    ax.axvline(TAXONOMY_LINE_X, color="#c0392b", linestyle="--", linewidth=1.1, zorder=1)
    ax.annotate(
        f"taxonomy consolidation\nannounced {tx['announced']}, in data {tx['observed_in_data']}\n"
        "(2023 slice straddles it)",
        xy=(TAXONOMY_LINE_X, 0.02), xycoords=("data", "axes fraction"),
        xytext=(6, 0), textcoords="offset points",
        fontsize=7, color="#c0392b", va="bottom", ha="left",
    )


def _footnote(fig, lines: list[str]) -> list[str]:
    """Hard-wrap at a fixed column so no footnote can run off the canvas edge.

    Wrapping is done here rather than by matplotlib's `wrap=True`, which measures against
    the figure width at draw time and would silently re-flow (and so re-render) if the
    figure size ever changed. A fixed column keeps the committed SVGs stable.
    """
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=FOOTNOTE_WRAP_COLS) or [""])
    fig.text(0.008, 0.006, "\n".join(wrapped), fontsize=6.4, color="#444444",
             va="bottom", ha="left")
    return wrapped


def _finish(fig, ax, path: Path, *, title: str, ylabel: str, footnote: list[str],
            legend_loc: str = "best", headroom: float = 0.0) -> Path:
    ax.set_xticks(range(len(SLICE_ORDER)))
    ax.set_xticklabels([SLICE_LABELS[s] for s in SLICE_ORDER])
    ax.set_xlim(-0.35, len(SLICE_ORDER) - 0.65)
    ax.set_xlabel("evaluation slice (temporal)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    if headroom:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + headroom * (hi - lo))
    ax.legend(fontsize=7.5, loc=legend_loc, framealpha=0.92)
    n_lines = sum(max(1, len(textwrap.wrap(line, width=FOOTNOTE_WRAP_COLS)))
                  for line in footnote)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.93, bottom=0.115 + 0.021 * n_lines)
    _footnote(fig, footnote)
    path.parent.mkdir(parents=True, exist_ok=True)
    # metadata Date=None: matplotlib otherwise stamps the render time into the SVG, which
    # would make a committed chart differ on every regeneration for no substantive reason.
    fig.savefig(path, format="svg", metadata={"Date": None})
    return path


def render_macro_f1_chart(summary: dict, out_path: Path) -> Path:
    plt = _require_matplotlib()
    logged = summary["series"]["logged"]
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
    _annotate_taxonomy(ax, summary)
    x = np.arange(len(SLICE_ORDER), dtype=float)
    for tier in TIER_ORDER:
        rows = [r for r in logged if r["tier"] == tier]
        point, lo, hi = _series_points(rows, "macro_f1")
        by_slice = {r["slice"]: r for r in rows}
        # Read the support off the row rather than assuming it from the tier: Tier B2 is a
        # full-slice tier that is not the gate, and a mislabeled support turns a
        # different-population comparison into an unqualified one.
        yearly = by_slice["test_drift_2023"]
        support = ("full slice" if yearly["support"] == "full_slice"
                   else "uniform-random subsample")
        label = f"{TIER_DISPLAY[tier]} — {support} (n={yearly['n']:,} yearly)"
        style = SERIES_STYLE[tier]
        ax.fill_between(x, lo, hi, color=style["color"], alpha=0.12, linewidth=0)
        ax.errorbar(x, point, yerr=[np.array(point) - np.array(lo),
                                    np.array(hi) - np.array(point)],
                    capsize=3, elinewidth=0.9, label=label, markersize=5, **style)
    _finish(
        fig, ax, out_path,
        title="Macro-F1 over time (2022-H2 → 2026-H1), 95% bootstrap CI",
        ylabel="macro-F1 (9 routing classes)",
        legend_loc="upper left", headroom=0.22,
        footnote=FOOTNOTE_MACRO_F1,
    )
    plt.close(fig)
    return out_path


def render_ece_chart(summary: dict, out_path: Path) -> Path:
    plt = _require_matplotlib()
    logged = summary["series"]["logged"]
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
    _annotate_taxonomy(ax, summary)
    x = np.arange(len(SLICE_ORDER), dtype=float)
    for tier in ECE_TIERS:
        rows = [r for r in logged if r["tier"] == tier]
        point, lo, hi = _series_points(rows, "ece")
        style = SERIES_STYLE[tier]
        ax.fill_between(x, lo, hi, color=style["color"], alpha=0.12, linewidth=0)
        ax.errorbar(x, point, yerr=[np.array(point) - np.array(lo),
                                    np.array(hi) - np.array(point)],
                    capsize=3, elinewidth=0.9, markersize=5,
                    label=f"{TIER_DISPLAY[tier]} — full slice", **style)
    ax.set_ylim(bottom=0.0)
    _finish(
        fig, ax, out_path,
        title="Expected calibration error over time — gate-relevant tiers, 95% bootstrap CI",
        ylabel="ECE (15-bin, confidence = p_max)",
        legend_loc="upper right",
        footnote=FOOTNOTE_ECE,
    )
    plt.close(fig)
    return out_path


def render_escalation_chart(summary: dict, out_path: Path) -> Path:
    plt = _require_matplotlib()
    esc = summary["series"]["escalation"]
    thresholds = summary["thresholds"]
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
    _annotate_taxonomy(ax, summary)
    x = np.arange(len(SLICE_ORDER), dtype=float)

    series: list[tuple[str, str, list[dict]]] = [
        (ARM_A_TO_HUMAN_FULL, "a_to_human, full slice (τ={tau:.4f})",
         [r for r in esc if r["policy"] == "a_to_human"
          and r["dataset"] == threshold_opt.DATASET_FULL_CAL]),
        (ARM_A_TO_HUMAN_PAIRED, "a_to_human, paired n=1,500 (τ={tau:.4f})",
         [r for r in esc if r["policy"] == "a_to_human"
          and r["dataset"] == threshold_opt.DATASET_PAIRED]),
        (ARM_A_TO_B_FULL, "a_to_b→B2 terminal, full slice (τ={tau:.4f})",
         [r for r in esc if r["policy"] == threshold_opt.FAMILY_A_TO_B
          and r["dataset"] == threshold_opt.DATASET_FULL_CAL]),
        (ARM_A_TO_B_PAIRED, "a_to_b→B2 terminal, paired n=1,500 (τ={tau:.4f})",
         [r for r in esc if r["policy"] == threshold_opt.FAMILY_A_TO_B
          and r["dataset"] == threshold_opt.DATASET_PAIRED]),
    ]
    for tier in TERMINAL_MODELS:
        series.append((
            f"a_to_c__{tier}",
            f"a_to_c→{tier.replace('tier_c_', '')}, paired n=1,500 (τ={{tau:.4f}})",
            [r for r in esc if r["policy"] == "a_to_c_parsefail_human"
             and r["terminal_model"] == tier],
        ))

    for key, label_fmt, rows in series:
        point, lo, hi = _series_points(rows, "escalation_rate")
        style = ARM_STYLE[key]
        ax.fill_between(x, lo, hi, color=style["color"], alpha=0.12, linewidth=0)
        ax.errorbar(x, point, yerr=[np.array(point) - np.array(lo),
                                    np.array(hi) - np.array(point)],
                    capsize=3, elinewidth=0.9, markersize=5,
                    label=label_fmt.format(tau=rows[0]["tau"]), **style)

    for arm in ARM_ORDER:
        rate = thresholds[arm]["cal_escalation_rate"]
        ax.axhline(rate, color="#777777", linestyle=ARM_HLINE_STYLE[arm], linewidth=0.9,
                   zorder=0)
        ax.annotate(f"CAL operating point {rate:.4f} ({thresholds[arm]['policy_family']}, "
                    f"{thresholds[arm]['cal_dataset']})",
                    xy=(len(SLICE_ORDER) - 1, rate), xytext=(-4, 3),
                    textcoords="offset points", fontsize=6.4, color="#555555", ha="right",
                    # opaque backing: these labels sit on top of the series by construction
                    # (they mark the y-value the series is being read against)
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8,
                          "pad": 1.0})

    ax.set_ylim(bottom=0.0)
    _finish(
        fig, ax, out_path,
        title="Escalation rate over time under the frozen Phase 4 thresholds (v2-isocal)",
        ylabel="escalation rate (share of rows Tier A does not answer)",
        legend_loc="upper center", headroom=0.20,
        footnote=FOOTNOTE_ESCALATION,
    )
    plt.close(fig)
    return out_path


CHART_RENDERERS = {
    "macro_f1_over_time.svg": render_macro_f1_chart,
    "ece_over_time.svg": render_ece_chart,
    "escalation_over_time.svg": render_escalation_chart,
}


def render_charts(summary: dict, charts_dir: Path) -> list[Path]:
    return [renderer(summary, Path(charts_dir) / name)
            for name, renderer in CHART_RENDERERS.items()]


# ---------------------------------------------------------------------------
# Deterministic JSON output (same contract as prior_shift/oov)
# ---------------------------------------------------------------------------

# Keys whose float value is a DECISION BOUNDARY, not a measurement, and must round-trip
# exactly. Rounding a tau to 10 dp moves rows across the inclusive `p_max >= tau` gate, so a
# reader replaying the published constant would reproduce a different answered set than the
# file claims — the exact failure `threshold_opt._tau_json` documents (it hit 6 of 9
# threshold files before it was fixed there). json.dumps writes float64 via repr, which
# round-trips.
NO_ROUND_KEYS = frozenset({"tau", "tau_star"})


def _round_tree(value):
    """Round floats to JSON_ROUND; NaN/inf -> None (valid JSON, honest about undefined).

    ``NO_ROUND_KEYS`` are passed through at full float64 precision -- see that constant.
    """
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, dict):
        return {
            k: (float(v) if k in NO_ROUND_KEYS and isinstance(v, float) else _round_tree(v))
            for k, v in value.items()
        }
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


def split_sizes(splits_stats_path=DEFAULT_SPLITS_STATS_PATH) -> dict[str, int]:
    """Frozen full-slice row counts, so "full slice" is a checked claim and not a guess."""
    import yaml

    stats = yaml.safe_load(Path(splits_stats_path).read_text())
    return {split: int(stats["splits"][split]["n_selected"]) for split in SLICE_ORDER}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m triage_lab.drift_charts")
    parser.add_argument("--all", action="store_true",
                        help="the full rollup: summary.json + all three charts (required; "
                             "the summary is a single object, so there is no partial mode)")
    parser.add_argument("--skip-charts", action="store_true",
                        help="write summary.json only (no matplotlib needed)")
    parser.add_argument("--generated-at",
                        help="pin the generated_at stamp (byte-identical output across runs)")
    parser.add_argument("--preds-dir", type=Path, default=DEFAULT_PREDS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--charts-dir", type=Path, default=None)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--thresholds-dir", type=Path,
                        default=router_sim.DEFAULT_THRESHOLDS_DIR)
    parser.add_argument("--cost-config", type=Path, default=DEFAULT_COST_CONFIG,
                        help="cost generation the tau constants are loaded from; defaults "
                             "to v2, the only one that prices Tier B (see module docstring)")
    parser.add_argument("--splits-stats", type=Path, default=DEFAULT_SPLITS_STATS_PATH)
    args = parser.parse_args(argv)

    if not args.all:
        parser.error("give --all (the rollup is a single object; there is no partial mode)")
    charts_dir = args.charts_dir or (args.out_dir / DEFAULT_CHARTS_DIRNAME)

    cfg = cost_model.load_cost_config(args.cost_config)
    thresholds = router_sim.load_cal_thresholds(
        args.thresholds_dir, cost_sha256=cfg.sha256, results_path=args.results,
        derivation=OP_VERSION, cost_config=cfg, preds_dir=args.preds_dir,
    )
    records = predictions.records_by_config(args.results)
    artifacts = load_slice_artifacts(records, args.preds_dir)
    sizes = split_sizes(args.splits_stats)

    summary = build_summary(
        records=records, artifacts=artifacts, thresholds=thresholds, cost_config=cfg,
        split_sizes=sizes, thresholds_dir=args.thresholds_dir,
        generated_at=args.generated_at,
    )
    summary_path = write_json(summary, args.out_dir / "summary.json")

    for row in summary["series"]["escalation"]:
        pf = "" if row["n_parse_failed_escalated"] is None else (
            f"  parse_fail_human={row['n_parse_failed_escalated']}"
            f"/{row['n_parse_failed_slice']}")
        print(
            f"[{row['policy']:22s} {row['terminal_model']:14s} {row['dataset']:13s} "
            f"{row['slice_label']:8s}] n={row['n']:6d} tau={row['tau']:.6f} "
            f"esc={row['escalation_rate']['point']:.4f} "
            f"[{row['escalation_rate']['ci_lo']:.4f},{row['escalation_rate']['ci_hi']:.4f}]"
            f"  acc_ans={row['accuracy_machine']['point']:.4f}"
            f"  mF1_ans={row['macro_f1_answered']['point']:.4f}"
            f"  acc_sys={row['accuracy_system']['point']:.4f}{pf}"
        )
    print(f"summary: {len(summary['series']['logged'])} logged rows, "
          f"{len(summary['series']['escalation'])} escalation rows -> {summary_path}")

    if args.skip_charts:
        print("charts skipped (--skip-charts)")
        return 0
    for path in render_charts(summary, charts_dir):
        print(f"chart: {path}")
    return 0


if __name__ == "__main__":
    import sys

    from triage_lab.drift_charts import main as _main

    sys.exit(_main())
