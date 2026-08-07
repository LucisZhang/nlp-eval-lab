"""Router-simulator tests: hand-computed cascade costs under BOTH tau-transfer modes,
paired-delta sign convention and shared-index pairing, subset alignment and parse-fail
routing, the registered config-documentation hash exception, CLI determinism, and a
replay regression over the shipped results/router_sim/ files."""

from __future__ import annotations

import json
import math
from types import MappingProxyType

import duckdb
import numpy as np
import pytest
import yaml

from triage_lab import cost_model, harness, predictions, router_sim, threshold_opt

C_MISROUTE = 6.00
C_HUMAN = 2.50

SLUG = "anthropic/claude-haiku-4.5"
PROMPT_RATE = 1e-6
COMPLETION_RATE = 5e-6
PRICING = {"slug": SLUG, "prompt_usd_per_token": PROMPT_RATE,
           "completion_usd_per_token": COMPLETION_RATE}
RECEIPT_COST = 995 * PROMPT_RATE + 1 * COMPLETION_RATE  # $0.001 per escalated row

# 12 TEST rows, confidence descending, all > 0.5 so y_pred really is the argmax.
P_MAX = [0.99, 0.95, 0.91, 0.87, 0.83, 0.79, 0.75, 0.71, 0.67, 0.63, 0.59, 0.55]
IDS = list(range(100, 112))
LABELS = ["a", "b"]
Y_TRUE = ["a", "b"] * 6
# Tier A is right on the 6 most confident rows and wrong on the 6 least confident. On the
# PAIRED rows (even positions: p_max .99 .91 .83 .75 .67 .59) that is [T,T,T,F,F,F].
A_CORRECT = [True] * 6 + [False] * 6
# Tier C on the 6 paired rows: [T,F,T,T,F,T]; the 5th is the parse-failed one, whose
# fallback label is deliberately WRONG so the test proves it is never scored.
C_CORRECT_PAIRED = [True, False, True, True, False, True]
PARSE_FAILED_PAIRED = [False, False, False, False, True, False]

_REAL_ROUTER_FILES = sorted(
    p for p in router_sim.DEFAULT_ROUTER_DIR.glob("*__cost-*.json")
    if not p.name.startswith("summary__")
)
_HAS_REAL = bool(_REAL_ROUTER_FILES) and router_sim.DEFAULT_PREDS_DIR.exists() and \
    harness.DEFAULT_RESULTS_PATH.exists()
_needs_real = pytest.mark.skipif(not _HAS_REAL, reason="real preds/router_sim not present")


# ---------------------------------------------------------------------------
# Synthetic TEST-IID mini-repo
# ---------------------------------------------------------------------------

def _write_split(tmp_path, ids, y_true, *, name="test_iid"):
    splits = tmp_path / "splits"
    splits.mkdir(exist_ok=True)
    path = splits / f"{name}.parquet"
    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.register("_s", {"complaint_id": np.asarray(ids, dtype=np.int64),
                            "class": np.asarray([str(v) for v in y_true], dtype=object)})
        con.execute(f'COPY (SELECT complaint_id, "class" FROM _s) TO \'{path}\' '
                    "(FORMAT parquet)")
    finally:
        con.close()
    return splits


def _write_config(tmp_path, name, runner, splits_dir, *, split="test_iid"):
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        f"model:\n  runner: {runner}\n  name: {name}\n"
        f"data:\n  split: {split}\n  splits_dir: {splits_dir}\n"
        "  label_column: class\n  order_column: complaint_id\n"
    )
    return path


def _probs_for(y_pred, p_max, labels):
    index = {lbl: i for i, lbl in enumerate(labels)}
    probs = np.zeros((len(y_pred), len(labels)), dtype=np.float64)
    for i, pred in enumerate(y_pred):
        p = 1.0 if p_max is None else float(p_max[i])
        probs[i, :] = (1.0 - p) / max(len(labels) - 1, 1)
        probs[i, index[pred]] = p
    return probs


def _write_artifact(tmp_path, run_id, ids, y_true, y_pred, labels, p_max=None, *,
                    split="test_iid", split_sha256="splithash", config_sha256="cfghash"):
    probs = _probs_for(y_pred, p_max, labels)
    prov = predictions.ArtifactProvenance(
        run_id=run_id, config_sha256=config_sha256, split=split,
        split_sha256=split_sha256, class_labels=list(labels),
    )
    predictions.write_artifact(
        tmp_path / f"{run_id}.parquet", ids=np.asarray(ids, dtype=np.int64),
        y_true=y_true, y_pred=y_pred, probs=probs, class_labels=list(labels),
        provenance=prov,
    )
    return probs


def _metrics_block(y_true, y_pred, probs, labels):
    points = harness.evaluate(np.asarray(y_true, dtype=object),
                              np.asarray(y_pred, dtype=object), probs, list(labels))
    return {name: {"point": float(points[name])} for name in predictions.VERIFY_METRICS}


def _flip(label):
    return LABELS[1] if label == LABELS[0] else LABELS[0]


def _preds_from_correct(y_true, correct):
    return [t if ok else _flip(t) for t, ok in zip(y_true, correct, strict=True)]


def _receipt(cid, *, parse_failed=False):
    return {
        "complaint_id": cid, "prompt_tokens": 995, "completion_tokens": 1,
        "total_tokens": 996, "computed_cost_usd": RECEIPT_COST,
        "parse_failed": parse_failed, "slug": SLUG, "content": '{"label": "a"}',
    }


def _cost_config(tmp_path, *, c_misroute=C_MISROUTE, c_human=C_HUMAN):
    path = tmp_path / "cost.yaml"
    path.write_text(
        f"version: v1\nparams:\n  c_misroute_usd: {c_misroute}\n"
        f"  c_human_usd: {c_human}\napi_cost:\n  tier_a:\n    mode: amortized_zero\n"
        "    per_example_usd: 0.0\n    evidence_class: estimated\n    note: amortized\n"
        "  tier_c:\n    mode: measured_receipts\n    evidence_class: measured\n"
        "    note: receipts\nevidence_class:\n  params.c_misroute_usd: estimated\n"
    )
    return cost_model.load_cost_config(path)


CAL_RUN_ID = "ca" * 32
CAL_CONFIG_SHA = "calcfghash"
THRESHOLD_N_EXAMPLES = 100


def _write_threshold_file(dirpath, *, family, dataset, tau_star, target_coverage,
                          cost_sha256, tier_a_run_id=CAL_RUN_ID,
                          tier_a_config_sha=CAL_CONFIG_SHA, tier_a_split="cal",
                          n_examples=THRESHOLD_N_EXAMPLES, n_answered=None, suffix=""):
    """A results/thresholds/ file carrying every field the router now validates."""
    dirpath.mkdir(parents=True, exist_ok=True)
    if n_answered is None:
        # A non-finite target has no consistent count; write 0 so the file still parses
        # and the range check (not an arithmetic error) is what rejects it.
        n_answered = (round(target_coverage * n_examples)
                      if math.isfinite(target_coverage) else 0)
    obj = {
        "policy_family": family,
        "dataset": dataset,
        "is_primary": True,
        "tau_star": tau_star,
        "n_examples": n_examples,
        "n_answered_at_tau_star": n_answered,
        "target_coverage_a": target_coverage,
        "cost_config": {"sha256": cost_sha256},
        "inputs": {"tier_a": {"config_name": threshold_opt.PRIMARY_TIER_A_CONFIG,
                              "run_id": tier_a_run_id,
                              "config_sha256": tier_a_config_sha,
                              "split": tier_a_split}},
    }
    path = dirpath / f"{family}__{dataset}__abcadd53{suffix}__cost-{cost_sha256[:8]}.json"
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n")
    return path


def _mini_repo(tmp_path, *, cost_sha256, tau_full=0.83, tau_paired_human=0.83,
               tau_paired_c=0.75, target_full=0.5, target_paired_human=0.5,
               target_paired_c=0.5, paired_ids=None):
    """Three verified TEST-IID artifacts + Haiku receipts + CAL threshold constants."""
    # Ids requested for the paired subset but absent from the Tier A artifact still have
    # to exist in the frozen split, or the Tier C artifact fails membership before the
    # subset-alignment check this fixture is set up to exercise.
    extra = [i for i in (paired_ids or []) if i not in IDS]
    splits_dir = _write_split(tmp_path, IDS + extra, Y_TRUE + [Y_TRUE[0]] * len(extra))
    a_id, cnb_id, c_id = "aa" * 32, "bb" * 32, "cc" * 32

    a_cfg = _write_config(tmp_path, router_sim.TIER_A_TEST_CONFIG, "tier_a", splits_dir)
    cnb_cfg = _write_config(tmp_path, router_sim.TIER_A_CNB_TEST_CONFIG, "tier_a",
                            splits_dir)
    c_cfg = _write_config(tmp_path, router_sim.TIER_C_TEST_CONFIG, "tier_c", splits_dir)
    shas = {p: harness.config_sha256(p) for p in (a_cfg, cnb_cfg, c_cfg)}

    y_pred_a = _preds_from_correct(Y_TRUE, A_CORRECT)
    a_probs = _write_artifact(tmp_path, a_id, IDS, Y_TRUE, y_pred_a, LABELS, P_MAX,
                              config_sha256=shas[a_cfg])
    # CNB: uniformly worse than LogReg (wrong on 8 of 12), its own flat p_max shape.
    y_pred_cnb = _preds_from_correct(Y_TRUE, [True] * 4 + [False] * 8)
    cnb_probs = _write_artifact(tmp_path, cnb_id, IDS, Y_TRUE, y_pred_cnb, LABELS,
                                [0.9] * 12, config_sha256=shas[cnb_cfg])

    c_ids = list(paired_ids) if paired_ids is not None else IDS[::2]
    c_true = [Y_TRUE[IDS.index(i)] for i in c_ids] if paired_ids is None else \
        [Y_TRUE[0]] * len(c_ids)
    c_pred = _preds_from_correct(c_true, C_CORRECT_PAIRED[:len(c_ids)])
    c_probs = _write_artifact(tmp_path, c_id, c_ids, c_true, c_pred, LABELS,
                              config_sha256=shas[c_cfg])

    receipts = [_receipt(cid, parse_failed=pf)
                for cid, pf in zip(c_ids, PARSE_FAILED_PAIRED, strict=False)]
    log = tmp_path / "calls.jsonl"
    log.write_text("".join(json.dumps(r) + "\n" for r in receipts))

    records = [
        # The CAL rung the thresholds are fit on: the router binds every tau* to this
        # record. Only the record is needed (no artifact) — the router never re-evaluates
        # the CAL run, it only proves the constants came from it.
        {"run_id": CAL_RUN_ID,
         "config_path": f"configs/{threshold_opt.PRIMARY_TIER_A_CONFIG}.yaml",
         "config_sha256": CAL_CONFIG_SHA,
         "dataset": {"split": "cal", "split_sha256": "calsplithash"}},
        {"run_id": a_id, "config_path": str(a_cfg), "config_sha256": shas[a_cfg],
         "dataset": {"split": "test_iid", "split_sha256": "splithash"},
         "metrics": _metrics_block(Y_TRUE, y_pred_a, a_probs, LABELS)},
        {"run_id": cnb_id, "config_path": str(cnb_cfg), "config_sha256": shas[cnb_cfg],
         "dataset": {"split": "test_iid", "split_sha256": "splithash"},
         "metrics": _metrics_block(Y_TRUE, y_pred_cnb, cnb_probs, LABELS)},
        {"run_id": c_id, "config_path": str(c_cfg), "config_sha256": shas[c_cfg],
         "dataset": {"split": "test_iid", "split_sha256": "splithash"},
         "metrics": _metrics_block(c_true, c_pred, c_probs, LABELS),
         "cost_usd": sum(r["computed_cost_usd"] for r in receipts),
         "extra": {"raw_log_path": str(log), "model_slug": SLUG,
                   "pricing_snapshot": PRICING}},
    ]
    results = tmp_path / "runs.jsonl"
    results.write_text("".join(json.dumps(r) + "\n" for r in records))

    thresholds = tmp_path / "thresholds"
    for family, dataset, tau, target in (
        (threshold_opt.FAMILY_A_TO_HUMAN, threshold_opt.DATASET_FULL_CAL, tau_full,
         target_full),
        (threshold_opt.FAMILY_A_TO_HUMAN, threshold_opt.DATASET_PAIRED,
         tau_paired_human, target_paired_human),
        (threshold_opt.FAMILY_A_TO_C, threshold_opt.DATASET_PAIRED, tau_paired_c,
         target_paired_c),
    ):
        _write_threshold_file(thresholds, family=family, dataset=dataset, tau_star=tau,
                              target_coverage=target, cost_sha256=cost_sha256)
    return {"results": results, "thresholds": thresholds, "log": log,
            "a_id": a_id, "cnb_id": cnb_id, "c_id": c_id, "c_ids": c_ids}


def _build(tmp_path, **kwargs):
    cfg = _cost_config(tmp_path)
    repo = _mini_repo(tmp_path, cost_sha256=cfg.sha256, **kwargs)
    evaluations = router_sim.build_all(
        cfg, preds_dir=tmp_path, results_path=repo["results"],
        thresholds_dir=repo["thresholds"], n_resamples=25)
    return repo, cfg, evaluations


# ---------------------------------------------------------------------------
# Hand-computed cascade, threshold transfer (PRIMARY)
# ---------------------------------------------------------------------------

def test_hand_computed_paired_policies_threshold_transfer(tmp_path):
    _, _, ev = _build(tmp_path)
    paired = ev[router_sim.EVAL_PAIRED]
    assert paired["n_examples"] == 6
    def per1k(usd):
        return usd * 1000 / 6

    pol = paired["policies"]
    # a_only: Tier A wrong on 3 of the 6 paired rows -> 3 x $6
    assert pol["a_only"]["expected_cost_per_1k"]["total"]["point"] == \
        pytest.approx(per1k(18.0))
    # c_only: 6 receipts x $0.001, one parse-failed row -> $2.50, one C miss -> $6
    assert pol["c_only"]["expected_cost_per_1k"]["total"]["point"] == \
        pytest.approx(per1k(6.0 + 2.5 + 6 * RECEIPT_COST))
    assert pol["c_only"]["routing"]["n_to_human"] == 1
    # a_to_human @ tau=0.83: answers the 3 most confident (all correct), 3 to human
    assert pol["a_to_human"]["routing"]["n_answered_machine"] == 3
    assert pol["a_to_human"]["expected_cost_per_1k"]["total"]["point"] == \
        pytest.approx(per1k(7.5))
    # a_to_c @ tau=0.75: A answers 4 (1 wrong -> $6); 2 escalate ($0.002); the escalated
    # parse-failed row adds $2.50 and contributes NO misroute despite a wrong fallback.
    router = pol["a_to_c_parsefail_human"]
    assert router["routing"]["coverage_a"] == pytest.approx(4 / 6)
    assert router["routing"]["n_to_human"] == 1
    assert router["expected_cost_per_1k"]["total"]["point"] == \
        pytest.approx(per1k(6.0 + 2.5 + 2 * RECEIPT_COST))
    comps = router["expected_cost_per_1k"]
    assert comps["misroute"]["point"] == pytest.approx(per1k(6.0))
    assert comps["human"]["point"] == pytest.approx(per1k(2.5))
    assert comps["api"]["point"] == pytest.approx(per1k(2 * RECEIPT_COST))
    assert pol["all_human"]["expected_cost_per_1k"]["total"]["point"] == \
        pytest.approx(2500.0)


def test_hand_computed_full_policies_threshold_transfer(tmp_path):
    _, _, ev = _build(tmp_path)
    full = ev[router_sim.EVAL_FULL]
    assert full["n_examples"] == 12
    def per1k(usd):
        return usd * 1000 / 12
    pol = full["policies"]
    # a_only: wrong on the 6 least confident rows
    assert pol["a_only"]["expected_cost_per_1k"]["total"]["point"] == \
        pytest.approx(per1k(36.0))
    # a_only_cnb: wrong on 8
    assert pol["a_only_cnb"]["expected_cost_per_1k"]["total"]["point"] == \
        pytest.approx(per1k(48.0))
    # a_to_human @ tau=0.83 -> answers 5 (all correct), 7 to human
    a2h = pol["a_to_human"]
    assert a2h["routing"]["n_answered_machine"] == 5
    assert a2h["routing"]["n_to_human"] == 7
    assert a2h["expected_cost_per_1k"]["total"]["point"] == pytest.approx(per1k(17.5))
    assert a2h["accuracy_machine"] == 1.0
    assert a2h["accuracy_system"] == 1.0  # human rows credited


def test_parse_failed_row_is_routed_to_human_and_never_scored(tmp_path):
    _, _, ev = _build(tmp_path)
    paired = ev[router_sim.EVAL_PAIRED]
    router = paired["policies"]["a_to_c_parsefail_human"]
    # The parse-failed row's Tier C fallback label is WRONG in this fixture. If it were
    # scored as a machine answer it would add a $6 misroute; it must not.
    assert router["expected_cost_per_1k"]["misroute"]["point"] == \
        pytest.approx(6.0 * 1000 / 6)
    assert router["routing"]["n_to_human"] == 1
    # machine accuracy is over the 5 machine-answered rows: A right on 3 of 4, C right on 1
    assert router["accuracy_machine"] == pytest.approx(4 / 5)
    # system accuracy credits the human row
    assert router["accuracy_system"] == pytest.approx(5 / 6)
    assert router["macro_f1_answered"] is not None
    assert router["macro_f1_system"] >= router["macro_f1_answered"]


# ---------------------------------------------------------------------------
# Coverage-matched transfer (SECONDARY)
# ---------------------------------------------------------------------------

def test_coverage_matched_tau_picks_the_target_quantile():
    p_max = np.array([0.99, 0.91, 0.83, 0.75, 0.67, 0.59])
    assert router_sim.coverage_matched_tau(p_max, 0.5) == 0.83   # 3 of 6
    assert router_sim.coverage_matched_tau(p_max, 1.0) == 0.59   # all
    assert router_sim.coverage_matched_tau(p_max, 0.0) == float("inf")  # none
    # ties: the whole tie block is answered, so realized coverage can exceed the target
    tied = np.array([0.9, 0.9, 0.9, 0.5])
    tau = router_sim.coverage_matched_tau(tied, 0.25)
    assert tau == 0.9
    assert int((tied >= tau).sum()) == 3


def test_hand_computed_coverage_matched_secondary_block(tmp_path):
    _, _, ev = _build(tmp_path)
    secondary = ev[router_sim.EVAL_PAIRED]["secondary_coverage_matched"]
    assert secondary["transfer_mode"] == router_sim.TRANSFER_SECONDARY
    assert "SECONDARY" in secondary["label"]
    # target coverage 0.5 on 6 rows -> k=3 -> tau = 0.83, so a_to_c answers 3 by Tier A
    # (all correct) and escalates 3: api 3 x $0.001, one parse-fail -> $2.50, C right on
    # the other two -> no misroute at all.
    router = secondary["policies"]["a_to_c_parsefail_human"]
    assert router["gate"]["tau"] == pytest.approx(0.83)
    assert router["routing"]["coverage_a"] == pytest.approx(0.5)
    assert router["expected_cost_per_1k"]["total"]["point"] == \
        pytest.approx((2.5 + 3 * RECEIPT_COST) * 1000 / 6)
    assert router["expected_cost_per_1k"]["misroute"]["point"] == 0.0
    # only threshold policies appear in the secondary block
    assert set(secondary["policies"]) == {"a_to_human", "a_to_c_parsefail_human"}


def test_transfer_block_reports_realized_vs_target_coverage(tmp_path):
    _, _, ev = _build(tmp_path)
    rows = {r["policy"]: r for r in ev[router_sim.EVAL_PAIRED]["transfer"]["rows"]}
    router = rows["a_to_c_parsefail_human"]
    assert router["cal_tau_star"] == pytest.approx(0.75)
    assert router["applied_tau"] == pytest.approx(0.75)
    assert router["cal_target_coverage_a"] == pytest.approx(0.5)
    assert router["realized_coverage_a"] == pytest.approx(4 / 6)
    assert router["coverage_gap"] == pytest.approx(4 / 6 - 0.5)
    assert router["coverage_matched_tau"] == pytest.approx(0.83)
    assert router["cal_source_sha256"]
    assert "isotonic" in ev[router_sim.EVAL_PAIRED]["transfer"]["note"]


# ---------------------------------------------------------------------------
# Paired deltas
# ---------------------------------------------------------------------------

def test_paired_delta_sign_convention_and_hand_computed_value(tmp_path):
    _, _, ev = _build(tmp_path)
    paired = ev[router_sim.EVAL_PAIRED]
    by_pair = {(d["a"], d["b"]): d for d in paired["paired_deltas"]}
    d = by_pair[("a_to_c_parsefail_human", "a_only")]
    pol = paired["policies"]
    expected = (pol["a_to_c_parsefail_human"]["expected_cost_per_1k"]["total"]["point"]
                - pol["a_only"]["expected_cost_per_1k"]["total"]["point"])
    assert d["delta_cost_per_1k"]["point"] == pytest.approx(expected, abs=1e-6)
    assert d["delta_cost_per_1k"]["point"] < 0          # A cheaper -> negative
    assert d["cheaper"] == "a_to_c_parsefail_human"
    # accuracy delta: system accuracy 5/6 vs 3/6
    assert d["delta_accuracy_system"]["point"] == pytest.approx(5 / 6 - 3 / 6)


def test_paired_delta_is_paired_not_two_marginals(tmp_path):
    # A policy compared against itself must give an exactly-zero delta with a degenerate
    # band: two independent draws would not.
    _, cfg, _ = _build(tmp_path)
    repo = _mini_repo(tmp_path, cost_sha256=cfg.sha256)
    inputs = router_sim.load_test_inputs(tmp_path, repo["results"])
    cal = router_sim.load_cal_thresholds(repo["thresholds"], cost_sha256=cfg.sha256,
                                         results_path=repo["results"])
    policies = {p.name: p for p in router_sim.build_paired_policies(inputs, cal)}
    same = router_sim.paired_comparison(policies["a_only"], policies["a_only"], cfg,
                                        n_resamples=20)
    assert same["delta_cost_per_1k"] == {"point": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
                                         "excludes_zero": False}
    assert same["delta_accuracy_system"]["point"] == 0.0
    assert same["mcnemar_machine_rows"]["n_discordant"] == 0
    assert same["cheaper"] == "neither"


def test_paired_delta_refuses_misaligned_policies(tmp_path):
    _, cfg, _ = _build(tmp_path)
    repo = _mini_repo(tmp_path, cost_sha256=cfg.sha256)
    inputs = router_sim.load_test_inputs(tmp_path, repo["results"])
    cal = router_sim.load_cal_thresholds(repo["thresholds"], cost_sha256=cfg.sha256,
                                         results_path=repo["results"])
    full = {p.name: p for p in router_sim.build_full_policies(inputs, cal)}
    paired = {p.name: p for p in router_sim.build_paired_policies(inputs, cal)}
    with pytest.raises(ValueError, match="their ids differ"):
        router_sim.paired_comparison(full["a_only"], paired["a_only"], cfg,
                                     n_resamples=5)


def test_mcnemar_uses_only_rows_both_policies_answered(tmp_path):
    _, _, ev = _build(tmp_path)
    by_pair = {(d["a"], d["b"]): d for d in ev[router_sim.EVAL_PAIRED]["paired_deltas"]}
    # a_to_c answers 5 by machine, a_to_human answers 3 -> 3 rows in common
    m = by_pair[("a_to_c_parsefail_human", "a_to_human")]["mcnemar_machine_rows"]
    assert m["n_both_machine"] == 3
    assert m["n_excluded_human"] == 3
    assert 0.0 <= m["p_value"] <= 1.0
    # against all_human there is no machine-vs-machine row at all
    m2 = by_pair[("a_to_c_parsefail_human", "all_human")]["mcnemar_machine_rows"]
    assert m2["n_both_machine"] == 0
    assert m2["p_value"] == 1.0


# ---------------------------------------------------------------------------
# Alignment + input gates
# ---------------------------------------------------------------------------

def test_paired_subset_id_mismatch_is_a_hard_failure(tmp_path):
    cfg = _cost_config(tmp_path)
    with pytest.raises(ValueError, match="absent from the Tier A artifact"):
        repo = _mini_repo(tmp_path, cost_sha256=cfg.sha256, paired_ids=[100, 999])
        router_sim.load_test_inputs(tmp_path, repo["results"])


def test_non_test_iid_artifact_is_refused(tmp_path):
    # A consistent CAL run (artifact and record agree) must still be refused: the router
    # reports on TEST-IID, and a CAL artifact reaching it would be a silent slice swap.
    run_id = "dd" * 32
    _write_artifact(tmp_path, run_id, [1], ["a"], ["a"], LABELS, [0.9], split="cal")
    record = {"run_id": run_id, "config_path": "configs/x.yaml",
              "config_sha256": "cfghash",
              "dataset": {"split": "cal", "split_sha256": "splithash"}}
    with pytest.raises(ValueError, match="outside the allowed set"):
        cost_model.load_artifact_verified(record, tmp_path,
                                          allowed_splits=router_sim.ALLOWED_SPLITS)


def test_threshold_file_from_a_different_cost_config_is_refused(tmp_path):
    cfg = _cost_config(tmp_path)
    repo = _mini_repo(tmp_path, cost_sha256="0" * 64)
    with pytest.raises(ValueError, match="was fit under cost config"):
        router_sim.load_cal_thresholds(repo["thresholds"], cost_sha256=cfg.sha256,
                                         results_path=repo["results"])


def test_missing_threshold_files_are_a_hard_failure(tmp_path):
    empty = tmp_path / "none"
    empty.mkdir()
    with pytest.raises(ValueError, match="no primary-rung threshold files"):
        router_sim.load_cal_thresholds(empty)


# ---------------------------------------------------------------------------
# Registered config-documentation hash exception
# ---------------------------------------------------------------------------

def test_registry_is_structurally_immutable():
    """A provenance exemption list that code can append to at runtime is a disabled gate."""
    registry = predictions.CONFIG_DOC_CORRECTIONS
    assert isinstance(registry, MappingProxyType)
    with pytest.raises(TypeError):
        registry["new-run"] = {"recorded_sha256": "a", "corrected_sha256": "b",
                               "reason": "sneaking one in"}
    with pytest.raises(TypeError):
        del registry[next(iter(registry))]
    with pytest.raises(AttributeError):
        registry.update({"new-run": {}})
    with pytest.raises(AttributeError):
        registry.clear()
    assert len(registry) == 1


def test_registry_is_closed_over_run_id_and_both_hashes(tmp_path):
    """The exemption applies to one run and one before/after pair — nothing else.

    Exercised without mutating the registry (it cannot be mutated): a synthetic record
    reuses the REGISTERED run id but points at a different config file, so each of the
    three binding conditions can be broken independently.
    """
    registered_run = next(iter(predictions.CONFIG_DOC_CORRECTIONS))
    entry = predictions.CONFIG_DOC_CORRECTIONS[registered_run]

    other = tmp_path / "other.yaml"
    other.write_text("model:\n  runner: tier_a\ndata:\n  split: cal\n")

    # registered run id, registered recorded hash, but a file with a THIRD hash
    third = {"run_id": registered_run, "config_path": str(other),
             "config_sha256": entry["recorded_sha256"]}
    assert predictions.config_doc_correction(third) is None
    with pytest.raises(ValueError, match="config hash mismatch"):
        predictions.load_config_checked(third)

    # the real corrected file, but a record whose recorded hash is not the registered one
    real_path = predictions._repo_path(
        next(r["config_path"] for r in predictions.load_records()
             if r["run_id"] == registered_run))
    wrong_recorded = {"run_id": registered_run, "config_path": str(real_path),
                      "config_sha256": "q" * 64}
    assert predictions.config_doc_correction(wrong_recorded) is None
    with pytest.raises(ValueError, match="config hash mismatch"):
        predictions.load_config_checked(wrong_recorded)

    # the registered pair, but attributed to a DIFFERENT run id
    other_run = {"run_id": "y" * 64, "config_path": str(real_path),
                 "config_sha256": entry["recorded_sha256"]}
    assert predictions.config_doc_correction(other_run) is None
    with pytest.raises(ValueError, match="config hash mismatch"):
        predictions.load_config_checked(other_run)


def test_registry_is_inert_when_the_file_still_matches(tmp_path):
    cfg_path = tmp_path / "toy.yaml"
    cfg_path.write_text("model:\n  runner: tier_a\ndata:\n  split: cal\n")
    sha = harness.config_sha256(cfg_path)
    record = {"run_id": "w" * 64, "config_path": str(cfg_path), "config_sha256": sha}
    assert predictions.load_config_checked(record)  # matching hash short-circuits
    assert predictions.config_doc_correction(record) is None


def test_shipped_registry_entry_matches_the_repo_state():
    """The real entry must describe the real files, or it is documentation of nothing."""
    records = {r["run_id"]: r for r in predictions.load_records()}
    for run_id, entry in predictions.CONFIG_DOC_CORRECTIONS.items():
        record = records[run_id]
        assert record["config_sha256"] == entry["recorded_sha256"]
        path = predictions._repo_path(record["config_path"])
        assert harness.config_sha256(path) == entry["corrected_sha256"]
        assert predictions.config_doc_correction(record) == entry
        config = harness.load_config(path)
        assert config["data"]["split"] == record["dataset"]["split"]
        assert config["model"]["name"] == path.stem


def _strip_comments(text: str) -> str:
    return "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith("#"))


def test_registered_correction_changed_prose_only():
    """The corrected config must parse to the SAME object as before the edit.

    Two independent checks, because each is weak alone:

    - against git HEAD, the pre-edit bytes parse to an identical object (skipped once the
      correction is committed, when HEAD and the working tree agree by construction);
    - durably, stripping every comment line from the current file leaves the parsed object
      unchanged — i.e. this config carries no semantics in its comments, so a
      comment-only diff cannot have moved a number.
    """
    import subprocess

    for run_id in predictions.CONFIG_DOC_CORRECTIONS:
        record = next(r for r in predictions.load_records() if r["run_id"] == run_id)
        path = predictions._repo_path(record["config_path"])
        current_text = path.read_text()
        current = yaml.safe_load(current_text)

        assert yaml.safe_load(_strip_comments(current_text)) == current

        rel = path.relative_to(harness.REPO_ROOT)
        proc = subprocess.run(  # fixed argv, no shell
            ["git", "show", f"HEAD:{rel}"], cwd=harness.REPO_ROOT,
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            pytest.skip(f"{rel} not in git HEAD")
        head_text = proc.stdout
        if head_text == current_text:
            pytest.skip("correction already committed; HEAD == working tree")
        assert yaml.safe_load(head_text) == current, (
            "the registered 'documentation' correction changed a parsed value"
        )


# ---------------------------------------------------------------------------
# Determinism + shipped-file replay
# ---------------------------------------------------------------------------

def test_cli_is_byte_deterministic(tmp_path):
    cfg = _cost_config(tmp_path)
    repo = _mini_repo(tmp_path, cost_sha256=cfg.sha256)
    outs = []
    for name in ("out1", "out2"):
        out_dir = tmp_path / name
        assert router_sim.main([
            "--preds-dir", str(tmp_path), "--out-dir", str(out_dir),
            "--results", str(repo["results"]), "--cost-config", str(tmp_path / "cost.yaml"),
            "--thresholds-dir", str(repo["thresholds"]),
        ]) == 0
        outs.append(out_dir)
    files = sorted(p.name for p in outs[0].glob("*.json"))
    assert len(files) == 3  # two evaluation sets + summary
    for name in files:
        assert (outs[0] / name).read_bytes() == (outs[1] / name).read_bytes()


def test_summary_reports_dominance_and_decision_1(tmp_path):
    _, cfg, ev = _build(tmp_path)
    summary = router_sim.build_summary(ev, cfg)
    assert summary["headline_router"] == "a_to_c_parsefail_human"
    dom = summary["dominance"]
    assert set(dom["model_baselines"]) == {"a_only", "a_only_cnb", "c_only"}
    assert "all_human" not in dom["model_baselines"]
    for row in dom["by_router"].values():
        assert set(row["dominated"]) <= set(row["compared_against"])
        assert row["n_model_baselines_dominated"] == len(row["dominated_model_baselines"])
    d1 = summary["owner_decision_1_cross_family"]
    assert d1["n_examples"] == 6
    assert ("supported" in d1["verdict"]) == d1["delta_cost_per_1k"]["excludes_zero"]


@_needs_real
def test_shipped_router_files_replay_exactly():
    """Regression over the COMMITTED results/router_sim/ files.

    Every published operating point is rebuilt from the artifacts it names and must
    reproduce its own routing counts and cost — the same self-check the threshold files
    carry, extended to the router's applied taus.
    """
    cfg = cost_model.load_cost_config()
    inputs = router_sim.load_test_inputs()
    cal = router_sim.load_cal_thresholds(cost_sha256=cfg.sha256)
    builders = {router_sim.EVAL_FULL: router_sim.build_full_policies,
                router_sim.EVAL_PAIRED: router_sim.build_paired_policies}

    for path in _REAL_ROUTER_FILES:
        obj = json.loads(path.read_text())
        name = obj["evaluation_set"]
        for transfer, block in ((router_sim.TRANSFER_PRIMARY, obj["policies"]),
                                (router_sim.TRANSFER_SECONDARY,
                                 obj["secondary_coverage_matched"]["policies"])):
            policies = {p.name: p for p in builders[name](inputs, cal, transfer=transfer)}
            for pname, published in block.items():
                policy = policies[pname]
                assert len(policy) == obj["n_examples"], f"{path.name}:{pname}"
                assert int(policy.machine.sum()) == \
                    published["routing"]["n_answered_machine"], f"{path.name}:{pname}"
                assert int(policy.to_human.sum()) == \
                    published["routing"]["n_to_human"], f"{path.name}:{pname}"
                cost = cost_model.expected_cost_per_1k(
                    policy.correct_for_cost, policy.api_cost_usd, policy.to_human,
                    c_misroute=cfg.c_misroute_usd, c_human=cfg.c_human_usd)["total"]
                assert cost == pytest.approx(
                    published["expected_cost_per_1k"]["total"]["point"], abs=1e-9
                ), f"{path.name}:{pname}"
                assert int((policy.machine | policy.to_human).sum()) == \
                    obj["n_examples"], f"{path.name}:{pname}"


@_needs_real
def test_shipped_router_files_carry_full_provenance():
    for path in _REAL_ROUTER_FILES:
        obj = json.loads(path.read_text())
        assert obj["cost_config"]["sha256"]
        for block in obj["inputs"].values():
            assert block["run_id"] and block["config_sha256"] and block["split_sha256"]
            assert block["split"] == "test_iid"
        tier_c = obj["inputs"].get("tier_c_haiku")
        if tier_c:
            assert tier_c["receipts_sha256"] == \
                cost_model.receipts_sha256(tier_c["raw_log_path"])
            assert tier_c["cost_sum_check"]["ok"] is True
        for row in obj["transfer"]["rows"]:
            assert row["cal_source_sha256"]
            assert (router_sim.DEFAULT_THRESHOLDS_DIR / row["cal_source_file"]).exists()


@_needs_real
def test_shipped_macro_f1_views_are_both_reported():
    for path in _REAL_ROUTER_FILES:
        obj = json.loads(path.read_text())
        for pname, p in obj["policies"].items():
            assert "macro_f1_system" in p and "macro_f1_answered" in p
            if p["routing"]["n_answered_machine"] == 0:
                assert p["macro_f1_answered"] is None
                assert p["accuracy_machine"] is None
            else:
                assert p["macro_f1_answered"] is not None
            # the system view can only be >= the answered view when humans are credited
            if p["routing"]["n_to_human"] > 0 and p["macro_f1_answered"] is not None:
                assert p["macro_f1_system"] >= p["macro_f1_answered"], pname
        assert "credit" in obj["notes"]["human_credit"].lower()


# ---------------------------------------------------------------------------
# Threshold-artifact validation
# ---------------------------------------------------------------------------

def _thresholds_for_validation(tmp_path, cfg, **overrides):
    """A valid three-file threshold set, with one file's fields overridable."""
    repo = _mini_repo(tmp_path, cost_sha256=cfg.sha256)
    target = overrides.pop("target_family", threshold_opt.FAMILY_A_TO_C)
    dataset = overrides.pop("target_dataset", threshold_opt.DATASET_PAIRED)
    if overrides:
        _write_threshold_file(repo["thresholds"], family=target, dataset=dataset,
                              tau_star=overrides.pop("tau_star", 0.75),
                              target_coverage=overrides.pop("target_coverage", 0.5),
                              cost_sha256=cfg.sha256, **overrides)
    return repo


def test_duplicate_primary_threshold_file_is_a_hard_failure(tmp_path):
    cfg = _cost_config(tmp_path)
    repo = _mini_repo(tmp_path, cost_sha256=cfg.sha256)
    # a second file claiming the same (family, dataset), differing only in file name
    _write_threshold_file(repo["thresholds"], family=threshold_opt.FAMILY_A_TO_C,
                          dataset=threshold_opt.DATASET_PAIRED, tau_star=0.61,
                          target_coverage=0.5, cost_sha256=cfg.sha256, suffix="_stale")
    with pytest.raises(ValueError, match="two primary threshold files claim"):
        router_sim.load_cal_thresholds(repo["thresholds"], cost_sha256=cfg.sha256,
                                       results_path=repo["results"])


@pytest.mark.parametrize(("override", "pattern"), [
    ({"tier_a_run_id": "zz" * 32}, "not bound to the CAL run"),
    ({"tier_a_config_sha": "different"}, "not bound to the CAL run"),
    ({"tier_a_split": "test_iid"}, "not bound to the CAL run"),
])
def test_threshold_file_not_bound_to_the_cal_run_is_refused(tmp_path, override, pattern):
    cfg = _cost_config(tmp_path)
    repo = _thresholds_for_validation(tmp_path, cfg, **override)
    with pytest.raises(ValueError, match=pattern):
        router_sim.load_cal_thresholds(repo["thresholds"], cost_sha256=cfg.sha256,
                                       results_path=repo["results"])


@pytest.mark.parametrize(("override", "pattern"), [
    ({"tau_star": float("nan")}, "not a finite probability threshold"),
    ({"tau_star": float("inf")}, "not a finite probability threshold"),
    ({"tau_star": 0.0}, "not a finite probability threshold"),
    ({"tau_star": -0.5}, "not a finite probability threshold"),
    ({"tau_star": 1.5}, "not a finite probability threshold"),
    ({"target_coverage": 0.0}, "not a finite coverage"),
    ({"target_coverage": 1.5}, "not a finite coverage"),
    ({"target_coverage": float("nan")}, "not a finite coverage"),
])
def test_out_of_range_threshold_fields_are_refused(tmp_path, override, pattern):
    cfg = _cost_config(tmp_path)
    # NaN/inf have no JSON literal; json.dumps emits NaN/Infinity, which json.loads reads
    # back as floats — exactly the shape a hand-edited or numpy-serialized file would have.
    repo = _thresholds_for_validation(tmp_path, cfg, **override)
    with pytest.raises(ValueError, match=pattern):
        router_sim.load_cal_thresholds(repo["thresholds"], cost_sha256=cfg.sha256,
                                       results_path=repo["results"])


@pytest.mark.parametrize(("n_examples", "n_answered", "pattern"), [
    (100, 90, "disagrees with itself"),     # 0.90 != target 0.50
    (100, 101, "inconsistent counts"),      # answered more than examined
    (0, 0, "inconsistent counts"),          # empty
    (100, -1, "inconsistent counts"),
])
def test_self_inconsistent_threshold_counts_are_refused(tmp_path, n_examples, n_answered,
                                                        pattern):
    cfg = _cost_config(tmp_path)
    repo = _thresholds_for_validation(tmp_path, cfg, n_examples=n_examples,
                                      n_answered=n_answered)
    with pytest.raises(ValueError, match=pattern):
        router_sim.load_cal_thresholds(repo["thresholds"], cost_sha256=cfg.sha256,
                                       results_path=repo["results"])


def test_incomplete_threshold_set_is_refused(tmp_path):
    cfg = _cost_config(tmp_path)
    repo = _mini_repo(tmp_path, cost_sha256=cfg.sha256)
    stale = next(repo["thresholds"].glob(f"{threshold_opt.FAMILY_A_TO_C}__*.json"))
    stale.unlink()
    with pytest.raises(ValueError, match="not the expected one"):
        router_sim.load_cal_thresholds(repo["thresholds"], cost_sha256=cfg.sha256,
                                       results_path=repo["results"])


def test_missing_cal_run_record_is_refused(tmp_path):
    cfg = _cost_config(tmp_path)
    repo = _mini_repo(tmp_path, cost_sha256=cfg.sha256)
    kept = [json.loads(x) for x in repo["results"].read_text().splitlines() if x.strip()]
    kept = [r for r in kept if r["run_id"] != CAL_RUN_ID]
    repo["results"].write_text("".join(json.dumps(r) + "\n" for r in kept))
    with pytest.raises(ValueError, match="no run record for the CAL rung"):
        router_sim.load_cal_thresholds(repo["thresholds"], cost_sha256=cfg.sha256,
                                       results_path=repo["results"])


@_needs_real
def test_shipped_thresholds_pass_validation_against_the_real_records():
    cfg = cost_model.load_cost_config()
    cal = router_sim.load_cal_thresholds(cost_sha256=cfg.sha256)
    assert set(cal) == set(router_sim.EXPECTED_THRESHOLD_KEYS)
    cal_record = predictions.records_by_config()[threshold_opt.PRIMARY_TIER_A_CONFIG]
    for entry in cal.values():
        assert entry.tier_a_run_id == cal_record["run_id"]
        assert 0.0 < entry.tau_star <= 1.0
        assert 0.0 < entry.target_coverage_a <= 1.0
        assert (router_sim.DEFAULT_THRESHOLDS_DIR / entry.source_file).exists()


# ---------------------------------------------------------------------------
# Coverage-matching contract
# ---------------------------------------------------------------------------

def test_coverage_matching_reports_the_nearest_achievable_coverage():
    # 5 rows: coverage is quantised to multiples of 0.2, so a 0.5 target is unreachable
    # and round() lands BELOW it (k = round(2.5) = 2 -> 0.4).
    p_max = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    tau = router_sim.coverage_matched_tau(p_max, 0.5)
    realized = float((p_max >= tau).sum()) / len(p_max)
    assert realized == pytest.approx(0.4)
    assert abs(realized - 0.5) == pytest.approx(0.1)
    # a reachable target is hit exactly
    tau = router_sim.coverage_matched_tau(p_max, 0.6)
    assert float((p_max >= tau).sum()) / len(p_max) == pytest.approx(0.6)


def test_secondary_block_records_the_coverage_matching_error(tmp_path):
    _, _, ev = _build(tmp_path)
    for name in (router_sim.EVAL_FULL, router_sim.EVAL_PAIRED):
        secondary = ev[name]["secondary_coverage_matched"]
        assert "NEAREST ACHIEVABLE" in secondary["contract"]
        rows = {r["policy"]: r for r in secondary["coverage_matching"]}
        assert rows
        for policy, row in rows.items():
            realized = secondary["policies"][policy]["routing"]["coverage_a"]
            assert row["realized_coverage_a"] == pytest.approx(realized)
            assert row["abs_coverage_error"] == pytest.approx(
                abs(realized - row["target_coverage_a"]))
            # the error can never exceed one coverage quantum plus a tie block
            assert row["abs_coverage_error"] >= 0.0
            assert row["coverage_quantum"] == pytest.approx(
                1.0 / ev[name]["n_examples"])
        for row in ev[name]["transfer"]["rows"]:
            assert row["coverage_matched_abs_error"] == pytest.approx(
                abs(row["coverage_matched_realized_coverage_a"]
                    - row["cal_target_coverage_a"]))
