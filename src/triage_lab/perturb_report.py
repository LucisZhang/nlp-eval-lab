"""Perturbation-robustness report (Phase 5, UPGRADE_PLAN §6.3.4).

Joins each perturbed TEST-IID run to its **clean baseline on identical rows** and reports
the macro-F1 / accuracy drop as a paired bootstrap delta. Everything here is a read of
already-logged runs: no model is fitted, no API is called, nothing is appended to
``results/runs.jsonl``. The derived output is committed under ``results/perturbation/``,
like ``results/oov/`` and ``results/prior_shift/``.

**Why paired, and what the pairing buys.** A perturbed run and its clean baseline differ in
exactly one thing -- the eval text -- and cover the *same complaint_ids* with the *same*
labels and the *same* fitted model. So the delta can be bootstrapped with one shared
resample index per replicate (``harness.paired_bootstrap_delta``), which cancels the
between-document variance that dominates an unpaired difference of two ~40,000-row macro-F1
estimates. Identical systems give delta 0 on every replicate and a CI of exactly [0, 0],
which is not a degenerate case here but a *prediction*: see the case arm below. A drop is
only claimed where the paired CI excludes zero (§6.1).

The join is asserted, never assumed: perturbation rewrites text, so it must leave the id
set bit-identical. A mismatch means the two runs are not on the same rows and the paired
delta would be meaningless, so it is a hard error rather than an inner join.

**Four arms.**

- ``logreg_wordchar`` / ``cnb_wordchar`` -- the two frozen TEST-IID finals (runs 8e4d6345
  and c20cd14a), perturbed at 0.05 and 0.10 in each family.
- ``logreg_word_only`` -- the sensitivity ladder: word n-grams only, 0.10 only. Its purpose
  is to be *differenced against* ``logreg_wordchar`` at the same rate, since the only
  configuration difference is ``features.char.enabled``. That difference-of-differences is
  deliberately NOT computed here (see ``methods_notes.char_shield``).
- ``tier_c_haiku`` -- Claude Haiku 4.5 zero-shot at 0.10, on the frozen 1,500-row paired
  subset, against clean run 70a1b0c4 (5,000 rows). This arm joins by *containment* rather
  than equality and is ``optional``: until its runs are logged it is omitted from the report
  entirely (``skipped``) rather than counted as a hole, because it costs real money and its
  absence is a schedule fact, not a defect. See ``methods_notes.tier_c_join``.

**The case arm is a predicted structural zero for Tier A only** and is labelled as such in
the Tier A rows. Both TF-IDF blocks are built with ``lowercase=True``, so flipping case and
then lowercasing is the identity: the feature matrix is bit-identical and the delta must be
exactly 0.0 with CI [0, 0]. It is reported rather than skipped because a *non*-zero case
delta would be a bug in the perturbation plumbing, which makes this arm the cheapest
available end-to-end control on that plumbing. The label is per-arm, not per-family: LLM
subword tokenizers are case-sensitive, so the ``tier_c_haiku`` case row is an ordinary
measurement (``methods_notes.tier_c_case_arm``).

Every artifact is read through ``cost_model.load_artifact_verified``, i.e. the repo's full
provenance + structural + aggregate gate, because a permuted id column is invisible to
aggregate metrics and would silently corrupt exactly the per-row join this module performs.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from triage_lab import harness, perturb, predictions

REPO_ROOT = harness.REPO_ROOT
DEFAULT_PREDS_DIR = harness.DEFAULT_PREDS_DIR
DEFAULT_RESULTS_PATH = harness.DEFAULT_RESULTS_PATH
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "perturbation"

SCHEMA_VERSION = 1
JSON_ROUND = 10
EVIDENCE_CLASS = "measured"

# Metrics differenced, in report order. macro_f1 is the headline (§6.1); accuracy is the
# companion that says how much of the macro-F1 move is just the majority classes.
DELTA_METRICS: tuple[str, ...] = ("macro_f1", "accuracy")

# Frozen protocol: the families and the two rates, and the config-name tag each rate uses.
FAMILY_ORDER: tuple[str, ...] = perturb.FAMILIES
RATE_TAGS: dict[str, float] = {"05": 0.05, "10": 0.10}


JOIN_IDENTICAL = "identical_rows"
JOIN_SUBSET = "perturbed_subset_of_clean"


@dataclass(frozen=True)
class Arm:
    """One model configuration, its clean baseline config, and the rates it was perturbed at.

    ``join`` distinguishes the two shapes this report has to handle. Tier A perturbed runs
    cover the whole TEST-IID split, so clean and perturbed are the SAME rows and any
    disagreement is a fault. Tier C runs on a seeded ``eval_rows_cap`` subsample, and the
    perturbed arms were capped at 1,500 against a clean run capped at 5,000 -- with the same
    ``cap_seed``, so the 1,500 are a byte-identical prefix-of-the-same-permutation subset.
    That case restricts the clean side and checks ``expected_n``, which is what turns "a
    subset" into "*the* subset" rather than a silent inner join.
    """

    key: str
    label: str
    clean_config: str
    perturbed_template: str
    rate_tags: tuple[str, ...]
    join: str = JOIN_IDENTICAL
    expected_n: int | None = None
    # Families whose delta is a structural zero for THIS model class (see CASE_STRUCTURAL_NOTE).
    structural_zero_families: tuple[str, ...] = ()
    # Optional arms are skipped (not reported as holes) when their runs are not logged yet.
    optional: bool = False

    def perturbed_config(self, family: str, rate_tag: str) -> str:
        return self.perturbed_template.format(family=family, rate_tag=rate_tag)


ARMS: tuple[Arm, ...] = (
    Arm(
        key="logreg_wordchar",
        label="LogReg word+char (TEST-IID final)",
        clean_config="tier_a_logreg_test_iid",
        perturbed_template="tier_a_logreg_test_iid_perturb_{family}_{rate_tag}",
        rate_tags=("05", "10"),
        structural_zero_families=("case",),
    ),
    Arm(
        key="cnb_wordchar",
        label="ComplementNB word+char (TEST-IID final)",
        clean_config="tier_a_cnb_test_iid",
        perturbed_template="tier_a_cnb_test_iid_perturb_{family}_{rate_tag}",
        rate_tags=("05", "10"),
        structural_zero_families=("case",),
    ),
    Arm(
        key="logreg_word_only",
        label="LogReg word-only (sensitivity arm)",
        clean_config="tier_a_logreg_word_test_iid",
        perturbed_template="tier_a_logreg_word_test_iid_perturb_{family}_{rate_tag}",
        rate_tags=("10",),
        structural_zero_families=("case",),
    ),
    Arm(
        key="tier_c_haiku",
        label="Claude Haiku 4.5 zero-shot (1,500-row paired subset)",
        clean_config="tier_c_haiku_zeroshot_test_iid",
        perturbed_template="tier_c_haiku_zeroshot_test_iid_perturb_{family}_{rate_tag}",
        rate_tags=("10",),
        join=JOIN_SUBSET,
        expected_n=1500,
        # Deliberately empty: an LLM tokenizer is case-SENSITIVE, so `case` is a real
        # measurement here, not the Tier A structural zero.
        structural_zero_families=(),
        optional=True,
    ),
)
ARMS_BY_KEY = {arm.key: arm for arm in ARMS}

CASE_STRUCTURAL_NOTE = (
    "Predicted STRUCTURAL ZERO for Tier A: both TF-IDF blocks use lowercase=True, so a "
    "case flip is annihilated before featurization and the delta must be exactly 0.0 with "
    "CI [0, 0]. Reported as an end-to-end control on the perturbation plumbing, not as a "
    "robustness finding; the informative version of this arm is Tier B / Tier C."
)
TIER_C_CASE_NOTE = (
    "The Tier A structural-zero argument does NOT extend to Tier C: an LLM's subword "
    "tokenizer is case-sensitive, so case mangling changes the token sequence, the token "
    "count and therefore both the prediction and the per-call cost. The tier_c case row is "
    "a real measurement and is not labelled structural."
)
TIER_C_JOIN_NOTE = (
    "The Tier C perturbed arms were capped at 1,500 rows against a clean run capped at "
    "5,000, with the SAME cap_seed. tier_c.subsample_eval takes the first `cap` entries of "
    "one default_rng(cap_seed).permutation(n), so the 1,500 are a byte-identical subset of "
    "the 5,000 (and identical to the Sonnet paired subset). The clean side is restricted to "
    "those ids by containment -- every perturbed id must exist in the clean run, and the "
    "matched count must equal expected_n -- so this is a pairing claim, not an inner join."
)
CHAR_SHIELD_NOTE = (
    "Whether the char_wb 3-5-grams BUY robustness is the difference between the "
    "logreg_wordchar and logreg_word_only deltas at the same family and rate. That "
    "difference-of-differences is not computed here: it spans four artifacts and two "
    "different fitted models, so an honest CI needs one joint bootstrap over the shared "
    "rows rather than a subtraction of two independently-bootstrapped intervals (whose "
    "widths do not compose). The point estimates are all in `rows`; read the sign, and "
    "compute the joint interval before making it a claim."
)
PAIRING_NOTE = (
    "One resample index vector per replicate is applied to BOTH systems "
    "(harness.paired_bootstrap_delta), so the delta CI is a paired interval on identical "
    "rows. Sign convention: delta = perturbed - clean, so a negative delta is degradation."
)


# ---------------------------------------------------------------------------
# Run + artifact resolution
# ---------------------------------------------------------------------------

class MissingRun(Exception):
    """A requested cell has no logged run (or no artifact) yet. Reported, never guessed at."""


def _load_verified(record: dict, preds_dir):
    """The repo's full artifact gate (provenance + structural + aggregate).

    Imported lazily, exactly as prior_shift does, to keep this module's import graph light.
    The verified loader rather than the raw reader because this module JOINS artifacts row
    by row, and a permuted id column -- the one fault aggregate metrics cannot see -- would
    silently produce a wrong paired delta.
    """
    from triage_lab import cost_model

    art_path = Path(preds_dir) / f"{record['run_id']}.parquet"
    if not art_path.exists():
        raise MissingRun(
            f"artifact {art_path} for run {record['run_id'][:8]} "
            f"({Path(record.get('config_path', '')).stem})"
        )
    return cost_model.load_artifact_verified(record, preds_dir, allowed_splits={"test_iid"})


def resolve_clean_record(arm: Arm, records_by_config: dict, records: list[dict],
                         overrides: dict[str, str]) -> dict:
    """The arm's clean baseline record: an explicit --clean-run override, else its config.

    The override exists so the report still runs if a baseline was logged under a config
    name this module does not know (a re-run, a superseding record). It selects by run_id
    prefix and must match exactly one record.
    """
    override = overrides.get(arm.key)
    if override is not None:
        matches = [r for r in records if r["run_id"].startswith(override)]
        if len(matches) != 1:
            raise ValueError(
                f"--clean-run {arm.key}={override!r} matches {len(matches)} run(s); it must "
                "match exactly one"
            )
        return matches[0]
    if arm.clean_config not in records_by_config:
        raise MissingRun(f"clean baseline config {arm.clean_config}")
    return records_by_config[arm.clean_config]


def check_recorded_perturbation(record: dict, family: str, rate: float) -> dict:
    """The perturbed run's own `extra.perturbation`, checked against what the report assumes.

    The report addresses runs by CONFIG NAME; this is the assertion that the name is not
    lying. A run whose record does not carry the perturbation it is being filed under is a
    hard error, not a footnote.
    """
    applied = (record.get("extra") or {}).get(perturb.CONFIG_KEY)
    if not isinstance(applied, dict):
        # A missing block is a data problem, not a type problem: this run is simply not a
        # perturbed run, whatever its config name says.
        raise ValueError(  # noqa: TRY004
            f"run {record['run_id'][:8]} ({record.get('config_path')}) carries no "
            f"extra.{perturb.CONFIG_KEY}; it cannot be reported as a perturbed run"
        )
    if applied.get("family") != family or float(applied.get("rate", -1)) != float(rate):
        raise ValueError(
            f"run {record['run_id'][:8]} records perturbation {applied} but is filed under "
            f"family={family!r} rate={rate}; the config name and the record disagree"
        )
    return applied


# ---------------------------------------------------------------------------
# The paired join
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AlignedPair:
    """Two artifacts restricted to the same complaint_ids, in the same (sorted) order."""

    ids: np.ndarray
    y_true: np.ndarray
    clean_pred: np.ndarray
    clean_probs: np.ndarray
    perturbed_pred: np.ndarray
    perturbed_probs: np.ndarray
    class_labels: list


def align_pair(clean, perturbed, *, join: str = JOIN_IDENTICAL,
               expected_n: int | None = None) -> AlignedPair:
    """Sort both artifacts by complaint_id and reduce them to one comparable row set.

    ``join=JOIN_IDENTICAL``: the id sets must be equal. Perturbation rewrites text; it
    cannot add, drop or relabel a row, so a disagreement is not something to reconcile with
    an inner join -- it means one of the two runs is not what it claims to be.

    ``join=JOIN_SUBSET``: every perturbed id must be present in the clean run, and the clean
    side is restricted to exactly those rows. This is the Tier C shape (1,500-row perturbed
    arms against a 5,000-row clean run drawn from the same seeded permutation). It is a
    *containment* check, not an intersection: an id the clean run never scored is a hard
    error, because it would mean the two caps were not drawn from the same stream. When
    ``expected_n`` is given the matched count must hit it exactly, which is what stops a
    partially-overlapping pair from quietly producing a small-n delta.
    """
    if join not in (JOIN_IDENTICAL, JOIN_SUBSET):
        raise ValueError(f"unknown join mode {join!r}")
    if list(clean.class_labels) != list(perturbed.class_labels):
        raise ValueError(
            f"class_labels differ between the clean ({clean.class_labels}) and perturbed "
            f"({perturbed.class_labels}) artifacts; their probability columns are not "
            "comparable"
        )
    c_order = np.argsort(clean.complaint_id, kind="stable")
    p_order = np.argsort(perturbed.complaint_id, kind="stable")
    c_ids, p_ids = clean.complaint_id[c_order], perturbed.complaint_id[p_order]

    if join == JOIN_SUBSET:
        pos = np.searchsorted(c_ids, p_ids)
        pos_clipped = np.minimum(pos, max(len(c_ids) - 1, 0))
        matched = len(c_ids) > 0 and np.array_equal(c_ids[pos_clipped], p_ids)
        if not matched:
            n_absent = int(np.setdiff1d(p_ids, c_ids).size)
            raise ValueError(
                f"{n_absent} of {len(p_ids)} perturbed id(s) are absent from the clean run "
                f"({len(c_ids)} rows); the perturbed subsample is not a subset of the clean "
                "one, so the two caps were not drawn from the same seeded permutation"
            )
        c_order = c_order[pos]
        c_ids = clean.complaint_id[c_order]
    elif not np.array_equal(c_ids, p_ids):
        only_clean = int(np.setdiff1d(c_ids, p_ids).size)
        only_pert = int(np.setdiff1d(p_ids, c_ids).size)
        raise ValueError(
            f"clean and perturbed runs do not cover identical rows: {len(c_ids)} vs "
            f"{len(p_ids)} ids, {only_clean} only-clean and {only_pert} only-perturbed; "
            "perturbation must not change the eval row set"
        )
    if expected_n is not None and len(p_ids) != expected_n:
        raise ValueError(
            f"expected {expected_n} paired rows for this arm but matched {len(p_ids)}; "
            "the frozen subsample size is part of the run's identity"
        )
    c_true, p_true = clean.y_true[c_order], perturbed.y_true[p_order]
    if not np.array_equal(c_true, p_true):
        n_bad = int(np.count_nonzero(c_true != p_true))
        raise ValueError(
            f"{n_bad} row(s) carry different y_true in the clean and perturbed artifacts; "
            "perturbation rewrites narratives only and must never touch labels"
        )
    return AlignedPair(
        ids=c_ids,
        y_true=c_true,
        clean_pred=clean.y_pred[c_order],
        clean_probs=clean.probs[c_order],
        perturbed_pred=perturbed.y_pred[p_order],
        perturbed_probs=perturbed.probs[p_order],
        class_labels=list(clean.class_labels),
    )


def pair_deltas(
    pair: AlignedPair,
    *,
    metric_names: tuple[str, ...] = DELTA_METRICS,
    n_resamples: int = harness.N_RESAMPLES,
    seed: int = harness.BOOTSTRAP_SEED,
) -> dict:
    """{metric: {clean, perturbed, delta, ci_lo, ci_hi, ci_excludes_zero}} for one pair.

    Point values are recomputed from the artifacts with `harness.evaluate` rather than read
    from the run records, so the point and the interval come from the same rows and the
    same code path (the records' own points are separately gated to 1e-9 by the artifact
    verification these artifacts already passed).
    """
    clean_points = harness.evaluate(
        pair.y_true, pair.clean_pred, pair.clean_probs, pair.class_labels
    )
    pert_points = harness.evaluate(
        pair.y_true, pair.perturbed_pred, pair.perturbed_probs, pair.class_labels
    )
    out: dict[str, dict] = {}
    for name in metric_names:
        delta = harness.paired_bootstrap_delta(
            pair.y_true,
            pair.perturbed_pred,
            pair.clean_pred,
            pair.perturbed_probs,
            pair.clean_probs,
            name,
            pair.class_labels,
            n_resamples=n_resamples,
            seed=seed,
        )
        out[name] = {
            "clean": float(clean_points[name]),
            "perturbed": float(pert_points[name]),
            "delta": delta["delta"],
            "ci_lo": delta["ci_lo"],
            "ci_hi": delta["ci_hi"],
            "ci_excludes_zero": bool(delta["ci_lo"] > 0.0 or delta["ci_hi"] < 0.0),
        }
    return out


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------

def build_row(
    arm: Arm,
    family: str,
    rate_tag: str,
    *,
    clean_record: dict,
    perturbed_record: dict,
    pair: AlignedPair,
    n_resamples: int = harness.N_RESAMPLES,
    seed: int = harness.BOOTSTRAP_SEED,
) -> dict:
    """One flat (arm x family x rate) result row."""
    rate = RATE_TAGS[rate_tag]
    return {
        "arm": arm.key,
        "arm_label": arm.label,
        "family": family,
        "rate": rate,
        "rate_tag": rate_tag,
        "n_rows": len(pair.ids),
        "join": arm.join,
        "clean_run_id": clean_record["run_id"],
        "clean_config": Path(clean_record.get("config_path", "")).stem,
        "perturbed_run_id": perturbed_record["run_id"],
        "perturbed_config": Path(perturbed_record.get("config_path", "")).stem,
        "recorded_perturbation": (perturbed_record.get("extra") or {}).get(
            perturb.CONFIG_KEY
        ),
        "metrics": pair_deltas(
            pair, n_resamples=n_resamples, seed=seed
        ),
        "structural_expectation": (
            CASE_STRUCTURAL_NOTE if family in arm.structural_zero_families else None
        ),
    }


def jobs(arms=ARMS, families=FAMILY_ORDER) -> list[tuple[Arm, str, str]]:
    """Every (arm, family, rate_tag) cell, in frozen report order."""
    return [
        (arm, family, tag)
        for arm in arms
        for family in families
        for tag in arm.rate_tags
    ]


def select_jobs(arm_keys, families) -> list[tuple[Arm, str, str]]:
    selected = [a for a in ARMS if not arm_keys or a.key in set(arm_keys)]
    fams = tuple(f for f in FAMILY_ORDER if not families or f in set(families))
    return jobs(tuple(selected), fams)


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


def protocol_block() -> dict:
    """The perturbation protocol, copied out of the code that implements it.

    Emitted into the summary so a reader of ``results/perturbation/summary.json`` can see
    what "typo at 0.10" meant without opening the source.
    """
    return {
        "rate_semantics": (
            "independent per-eligible-SITE probability, not the fraction of the document "
            "rewritten; sites are non-whitespace chars (typo), confusion-table matches "
            "(ocr, greedy longest-first and non-overlapping), alphabetic chars (case)"
        ),
        "families": list(perturb.FAMILIES),
        "rates": sorted(set(RATE_TAGS.values())),
        "default_seed": perturb.DEFAULT_SEED,
        "doc_rng_key": "blake2b(f'{seed}:{family}:{rate}:{complaint_id}', digest_size=16)",
        "typo_ops": dict(zip(perturb.TYPO_OPS, perturb.TYPO_OP_WEIGHTS, strict=True)),
        "ocr_bigram_confusions": dict(perturb.OCR_BIGRAM_CONFUSIONS),
        "ocr_unigram_confusions": dict(perturb.OCR_UNIGRAM_CONFUSIONS),
        "applied_to": "eval split narrative only; TRAIN fit and CAL calibration stay clean",
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(
    selected_jobs,
    *,
    results_path=DEFAULT_RESULTS_PATH,
    preds_dir=DEFAULT_PREDS_DIR,
    out_dir=DEFAULT_OUT_DIR,
    clean_overrides: dict[str, str] | None = None,
    write_summary: bool = True,
    n_resamples: int = harness.N_RESAMPLES,
    seed: int = harness.BOOTSTRAP_SEED,
    log=print,
) -> dict:
    """Build every requested row, print the table, and (optionally) write summary.json.

    Missing runs are reported as `missing` entries rather than crashing the whole report --
    the natural state of this exhibit while `make perturb` is still running is "some cells
    exist" -- but the CLI exits nonzero unless `--allow-missing`, because a report with
    holes is not the report.
    """
    records = predictions.load_records(results_path)
    by_config = predictions.records_by_config(results_path)
    clean_overrides = clean_overrides or {}
    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    git_sha = harness._git_sha()

    # Clean baselines are loaded once per arm: each is shared by up to six perturbed cells
    # and the verification gate is the expensive part.
    clean_cache: dict[str, tuple[dict, object]] = {}
    rows: list[dict] = []
    missing: list[dict] = []
    skipped: list[dict] = []

    for arm, family, rate_tag in selected_jobs:
        cell = f"{arm.key}/{family}/{rate_tag}"
        pert_config = arm.perturbed_config(family, rate_tag)
        # The perturbed run is resolved FIRST: an optional arm with no logged run must not
        # pay for (or fail on) its clean baseline's verification.
        if pert_config not in by_config:
            entry = {"cell": cell, "arm": arm.key, "family": family, "rate_tag": rate_tag,
                     "missing": f"perturbed config {pert_config}"}
            if arm.optional:
                skipped.append(entry)
                log(f"[{cell:34s}] not run yet (optional arm) -- omitted from the report")
            else:
                missing.append(entry)
                log(f"[{cell:34s}] MISSING: no logged run for {pert_config}")
            continue
        try:
            pert_record = by_config[pert_config]
            check_recorded_perturbation(pert_record, family, RATE_TAGS[rate_tag])
            pert_art = _load_verified(pert_record, preds_dir)
            if arm.key not in clean_cache:
                clean_record = resolve_clean_record(arm, by_config, records, clean_overrides)
                clean_cache[arm.key] = (clean_record, _load_verified(clean_record, preds_dir))
            clean_record, clean_art = clean_cache[arm.key]
        except MissingRun as exc:
            missing.append({"cell": cell, "arm": arm.key, "family": family,
                            "rate_tag": rate_tag, "missing": str(exc)})
            log(f"[{cell:34s}] MISSING: no logged run / artifact for {exc}")
            continue

        pair = align_pair(clean_art, pert_art, join=arm.join, expected_n=arm.expected_n)
        row = build_row(
            arm, family, rate_tag,
            clean_record=clean_record, perturbed_record=pert_record, pair=pair,
            n_resamples=n_resamples, seed=seed,
        )
        rows.append(row)
        _log_row(row, log)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "git_sha": git_sha,
        "evidence_class": EVIDENCE_CLASS,
        "analysis": "perturbation_robustness",
        "repro_command": "uv run python -m triage_lab.perturb_report --all",
        "split": "test_iid",
        "arms": [
            {"key": a.key, "label": a.label, "clean_config": a.clean_config,
             "rates": [RATE_TAGS[t] for t in a.rate_tags], "join": a.join,
             "expected_n": a.expected_n, "optional": a.optional,
             "structural_zero_families": list(a.structural_zero_families)}
            for a in ARMS
        ],
        "protocol": protocol_block(),
        "bootstrap": {
            "n_resamples": n_resamples,
            "seed": seed,
            "method": harness.BOOTSTRAP_METHOD,
            "ci_pct": [harness.CI_LOWER_PCT, harness.CI_UPPER_PCT],
            "pairing": PAIRING_NOTE,
        },
        "methods_notes": {
            "sign": "delta = perturbed - clean; negative = degradation",
            "case_arm": CASE_STRUCTURAL_NOTE,
            "tier_c_case_arm": TIER_C_CASE_NOTE,
            "tier_c_join": TIER_C_JOIN_NOTE,
            "char_shield": CHAR_SHIELD_NOTE,
            "artifact_gate": (
                "every artifact is read through cost_model.load_artifact_verified (the full "
                "provenance + structural + aggregate gate), because this module joins "
                "artifacts row by row and a permuted id column is invisible to aggregates"
            ),
        },
        "rows": rows,
        "missing": missing,
        "skipped": skipped,
    }
    if write_summary:
        path = write_json(summary, Path(out_dir) / "summary.json")
        log(f"summary: {len(rows)} row(s), {len(missing)} missing, "
            f"{len(skipped)} skipped -> {path}")
    return summary


def _log_row(row: dict, log) -> None:
    f1 = row["metrics"]["macro_f1"]
    flag = "yes" if f1["ci_excludes_zero"] else "no"
    log(
        f"[{row['arm']}/{row['family']}/{row['rate_tag']:2s}]".ljust(36)
        + f" n={row['n_rows']:>6d} "
        f"clean_F1={f1['clean']:.4f} perturbed_F1={f1['perturbed']:.4f} "
        f"delta={f1['delta']:+.4f} [{f1['ci_lo']:+.4f},{f1['ci_hi']:+.4f}] "
        f"CI!=0:{flag}"
    )


def format_table(summary: dict) -> str:
    """The human-readable family x rate x model table (macro-F1 and accuracy)."""
    header = (
        f"{'model':<20s} {'family':<6s} {'rate':>5s} {'metric':<9s} "
        f"{'clean':>8s} {'perturb':>8s} {'delta':>9s} {'95% CI':>21s}  CI!=0"
    )
    lines = [header, "-" * len(header)]
    for row in summary["rows"]:
        for name in DELTA_METRICS:
            m = row["metrics"][name]
            ci = f"[{m['ci_lo']:+.4f}, {m['ci_hi']:+.4f}]"
            lines.append(
                f"{row['arm']:<20s} {row['family']:<6s} {row['rate']:>5.2f} {name:<9s} "
                f"{m['clean']:>8.4f} {m['perturbed']:>8.4f} {m['delta']:>+9.4f} {ci:>21s}"
                f"  {'yes' if m['ci_excludes_zero'] else 'no':<3s}"
                f"{'   [structural zero expected]' if row['structural_expectation'] else ''}"
            )
    for miss in summary["missing"]:
        lines.append(f"{miss['cell']:<20s} MISSING ({miss['missing']})")
    for skip in summary.get("skipped", []):
        lines.append(f"{skip['cell']:<20s} not run yet ({skip['missing']})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_override(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"--clean-run expects <arm>=<run_id_prefix>, got {value!r}"
        )
    arm, run_id = value.split("=", 1)
    if arm not in ARMS_BY_KEY:
        raise argparse.ArgumentTypeError(
            f"unknown arm {arm!r}; choose from {sorted(ARMS_BY_KEY)}"
        )
    return arm, run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m triage_lab.perturb_report")
    parser.add_argument("--all", action="store_true",
                        help="every (arm x family x rate) cell in the frozen order")
    parser.add_argument("--arm", action="append", choices=sorted(ARMS_BY_KEY),
                        help="restrict to this arm (repeatable)")
    parser.add_argument("--family", action="append", choices=list(FAMILY_ORDER),
                        help="restrict to this perturbation family (repeatable)")
    parser.add_argument("--clean-run", action="append", type=_parse_override, default=[],
                        metavar="ARM=RUN_ID",
                        help="override an arm's clean baseline by run_id prefix, for when "
                             "the baseline was not logged under its expected config name")
    parser.add_argument("--allow-missing", action="store_true",
                        help="exit 0 even if some cells have no logged run yet")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--preds-dir", type=Path, default=DEFAULT_PREDS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    if not args.all and not (args.arm or args.family):
        parser.error("give --all or at least one of --arm/--family")

    selected = jobs() if args.all else select_jobs(args.arm, args.family)
    summary = run(
        selected,
        results_path=args.results,
        preds_dir=args.preds_dir,
        out_dir=args.out_dir,
        clean_overrides=dict(args.clean_run),
        write_summary=args.all,
    )
    print()
    print(format_table(summary))
    if not args.all:
        print("\nsummary.json not rewritten (partial selection; use --all)")
    if summary["missing"] and not args.allow_missing:
        print(f"\n{len(summary['missing'])} cell(s) missing; run `make perturb` first "
              "(or pass --allow-missing)")
        return 1
    return 0


if __name__ == "__main__":
    import sys

    from triage_lab.perturb_report import main as _main

    sys.exit(_main())
