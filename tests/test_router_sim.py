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

# --- Tier B fixture -------------------------------------------------------
# B2 is wrong exactly where Tier A is most confident and right everywhere else, so the two
# tiers disagree in the region a gate has to find. On the PAIRED rows (even positions) that
# is [F, F, T, T, T, T].
B2_CORRECT = [False] * 3 + [True] * 9
# B2's own confidence, unrelated to Tier A's ordering (odd positions are filler): the paired
# rows see 0.95, 0.85, 0.75, 0.65, 0.60, 0.55.
B2_P_MAX = [0.95, 0.9, 0.85, 0.9, 0.75, 0.9, 0.65, 0.9, 0.60, 0.9, 0.55, 0.9]
# The three B1 seeds are identical to each other here: they are single-tier frontier points
# in this fixture, never cascade rungs, so only their existence matters.
B1_CORRECT = [True] * 8 + [False] * 4
B1_P_MAX = [0.9] * 12
B_PER_EXAMPLE = 0.05   # huge next to the real ~$5.6e-6, so it is visible in the arithmetic

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
                    split="test_iid", split_sha256="splithash", config_sha256="cfghash",
                    git_sha="gitsha", input_sha256="inputsha"):
    probs = _probs_for(y_pred, p_max, labels)
    prov = predictions.ArtifactProvenance(
        run_id=run_id, config_sha256=config_sha256, split=split,
        split_sha256=split_sha256, class_labels=list(labels),
        git_sha=git_sha, input_sha256=input_sha256,
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


def _cost_config(tmp_path, *, c_misroute=C_MISROUTE, c_human=C_HUMAN, tier_b=False,
                 name="cost.yaml"):
    """Synthetic cost config; `tier_b=True` gives it the v2 shape (tier_b1 + tier_b2)."""
    path = tmp_path / name
    tier_b_block = "" if not tier_b else (
        "  tier_b1:\n    mode: amortized_estimate\n"
        f"    per_example_usd: {B_PER_EXAMPLE}\n    evidence_class: estimated\n"
        "    note: amortized gpu\n"
        "  tier_b2:\n    mode: amortized_estimate\n"
        f"    per_example_usd: {B_PER_EXAMPLE}\n    evidence_class: estimated\n"
        "    note: amortized gpu\n"
    )
    path.write_text(
        f"version: v1\nparams:\n  c_misroute_usd: {c_misroute}\n"
        f"  c_human_usd: {c_human}\napi_cost:\n  tier_a:\n    mode: amortized_zero\n"
        "    per_example_usd: 0.0\n    evidence_class: estimated\n    note: amortized\n"
        + tier_b_block +
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
                          n_examples=THRESHOLD_N_EXAMPLES, n_answered=None, suffix="",
                          tier_b_gate=None):
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
        **({} if tier_b_gate is None else {"tier_b_gate": tier_b_gate}),
    }
    path = dirpath / f"{family}__{dataset}__abcadd53{suffix}__cost-{cost_sha256[:8]}.json"
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n")
    return path


def _tier_b_runs(tmp_path, splits_dir, ids, y_true, records):
    """Four Tier B TEST-IID runs (3 x B1 seeds + B2), appended to `records`.

    The b1 seeds are only frontier POINTS here; B2 is the cascade rung, so it is the one
    whose per-row correctness and confidence are shaped to make the gate arithmetic
    interesting.
    """
    run_ids = {}
    for config_name, _, _ in router_sim.TIER_B_TEST_RUNGS:
        is_b2 = config_name == router_sim.TIER_B_CASCADE_CONFIG
        run_id = f"{'b2' if is_b2 else config_name[-2:]}" * 32
        cfg_path = _write_config(tmp_path, config_name, "tier_b", splits_dir)
        sha = harness.config_sha256(cfg_path)
        y_pred = _preds_from_correct(y_true, B2_CORRECT if is_b2 else B1_CORRECT)
        probs = _write_artifact(tmp_path, run_id, ids, y_true, y_pred, LABELS,
                                B2_P_MAX if is_b2 else B1_P_MAX, config_sha256=sha)
        records.append({
            "run_id": run_id, "config_path": str(cfg_path), "config_sha256": sha,
            "git_sha": "gitsha",
            "dataset": {"split": "test_iid", "split_sha256": "splithash",
                        "input_sha256": "inputsha"},
            "metrics": _metrics_block(y_true, y_pred, probs, LABELS),
        })
        run_ids[config_name] = run_id
    return run_ids


def _mini_repo(tmp_path, *, cost_sha256, tau_full=0.83, tau_paired_human=0.83,
               tau_paired_c=0.75, target_full=0.5, target_paired_human=0.5,
               target_paired_c=0.5, paired_ids=None, tier_b=False,
               tau_a_to_b_full=0.83, tau_a_to_b_paired=0.83, tau_abc=0.75, tau_b=0.58):
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
         "config_sha256": CAL_CONFIG_SHA, "git_sha": "gitsha",
         "dataset": {"split": "cal", "split_sha256": "calsplithash",
                     "input_sha256": "inputsha"}},
        {"run_id": a_id, "config_path": str(a_cfg), "config_sha256": shas[a_cfg],
         "git_sha": "gitsha",
         "dataset": {"split": "test_iid", "split_sha256": "splithash",
                     "input_sha256": "inputsha"},
         "metrics": _metrics_block(Y_TRUE, y_pred_a, a_probs, LABELS)},
        {"run_id": cnb_id, "config_path": str(cnb_cfg), "config_sha256": shas[cnb_cfg],
         "git_sha": "gitsha",
         "dataset": {"split": "test_iid", "split_sha256": "splithash",
                     "input_sha256": "inputsha"},
         "metrics": _metrics_block(Y_TRUE, y_pred_cnb, cnb_probs, LABELS)},
        {"run_id": c_id, "config_path": str(c_cfg), "config_sha256": shas[c_cfg],
         "git_sha": "gitsha",
         "dataset": {"split": "test_iid", "split_sha256": "splithash",
                     "input_sha256": "inputsha"},
         "metrics": _metrics_block(c_true, c_pred, c_probs, LABELS),
         "cost_usd": sum(r["computed_cost_usd"] for r in receipts),
         "extra": {"raw_log_path": str(log), "model_slug": SLUG,
                   "pricing_snapshot": PRICING}},
    ]
    b_ids = _tier_b_runs(tmp_path, splits_dir, IDS, Y_TRUE, records) if tier_b else {}
    results = tmp_path / "runs.jsonl"
    results.write_text("".join(json.dumps(r) + "\n" for r in records))

    thresholds = tmp_path / "thresholds"
    families = [
        (threshold_opt.FAMILY_A_TO_HUMAN, threshold_opt.DATASET_FULL_CAL, tau_full,
         target_full, None),
        (threshold_opt.FAMILY_A_TO_HUMAN, threshold_opt.DATASET_PAIRED,
         tau_paired_human, target_paired_human, None),
        (threshold_opt.FAMILY_A_TO_C, threshold_opt.DATASET_PAIRED, tau_paired_c,
         target_paired_c, None),
    ]
    if tier_b:
        families += [
            (threshold_opt.FAMILY_A_TO_B, threshold_opt.DATASET_FULL_CAL,
             tau_a_to_b_full, target_full, None),
            (threshold_opt.FAMILY_A_TO_B, threshold_opt.DATASET_PAIRED,
             tau_a_to_b_paired, target_paired_human, None),
            (threshold_opt.FAMILY_A_TO_B_TO_C, threshold_opt.DATASET_PAIRED,
             tau_abc, target_paired_c,
             {"tau_b_star": tau_b, "n_answered_b_at_tau_star": 1,
              "coverage_b_marginal": 0.5, "fit": "joint_2d_argmin"}),
        ]
    for family, dataset, tau, target, gate_b in families:
        _write_threshold_file(thresholds, family=family, dataset=dataset, tau_star=tau,
                              target_coverage=target, cost_sha256=cost_sha256,
                              tier_b_gate=gate_b)
    return {"results": results, "thresholds": thresholds, "log": log,
            "a_id": a_id, "cnb_id": cnb_id, "c_id": c_id, "c_ids": c_ids,
            "b_ids": b_ids}


def _build(tmp_path, **kwargs):
    cfg = _cost_config(tmp_path)
    repo = _mini_repo(tmp_path, cost_sha256=cfg.sha256, **kwargs)
    # The synthetic repo carries no CAL artifacts, so this exercises the METADATA gate;
    # the tau-replay gate is covered against the real shipped files further down.
    evaluations = router_sim.build_all(
        cfg, preds_dir=tmp_path, results_path=repo["results"],
        thresholds_dir=repo["thresholds"], n_resamples=25, verify_replay=False)
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
# Tier B policies (present only under a cost config that prices Tier B)
# ---------------------------------------------------------------------------

def _build_tier_b(tmp_path, **kwargs):
    cfg = _cost_config(tmp_path, tier_b=True)
    repo = _mini_repo(tmp_path, cost_sha256=cfg.sha256, tier_b=True, **kwargs)
    evaluations = router_sim.build_all(
        cfg, preds_dir=tmp_path, results_path=repo["results"],
        thresholds_dir=repo["thresholds"], n_resamples=25, verify_replay=False)
    return repo, cfg, evaluations


def test_tier_b_policies_exist_only_when_the_cost_config_prices_tier_b(tmp_path):
    """The switch is the PRICES, not a flag: an unpriced tier cannot be a frontier point."""
    _, _, without = _build(tmp_path)
    assert not any(p.startswith(("b", "a_to_b"))
                   for p in without[router_sim.EVAL_FULL]["policies"])

    priced = tmp_path / "priced"
    priced.mkdir()
    _, cfg, ev = _build_tier_b(priced)
    assert cost_model.prices_tier_b(cfg)
    full = set(ev[router_sim.EVAL_FULL]["policies"])
    paired = set(ev[router_sim.EVAL_PAIRED]["policies"])
    assert {"b1_only_sa", "b1_only_sb", "b1_only_sc", "b2_only", "a_to_b"} <= full
    assert "a_to_b_to_c" not in full          # Tier C is not defined on the full slice
    assert {"b1_only_sa", "b2_only", "a_to_b", "a_to_b_to_c"} <= paired
    assert router_sim.model_baselines(cfg) == router_sim.MODEL_BASELINES | {
        "b1_only_sa", "b1_only_sb", "b1_only_sc", "b2_only"}
    assert router_sim.expected_threshold_keys(cfg) == (
        router_sim.EXPECTED_THRESHOLD_KEYS | router_sim.TIER_B_THRESHOLD_KEYS)


def test_hand_computed_a_to_b_full_slice(tmp_path):
    _, _, ev = _build_tier_b(tmp_path)
    pol = ev[router_sim.EVAL_FULL]["policies"]

    def per1k(usd):
        return usd * 1000 / 12

    # b2_only: wrong on the 3 rows Tier A is most confident about; every row pays compute.
    assert pol["b2_only"]["expected_cost_per_1k"]["total"]["point"] == \
        pytest.approx(per1k(3 * 6.0 + 12 * B_PER_EXAMPLE))
    assert pol["b2_only"]["routing"]["n_to_human"] == 0
    # a_to_b @ tau=0.83: A answers the 5 most confident (all correct); B takes the other 7
    # and is right on all of them -> no misroute at all, just 7 x $0.05 of compute.
    a2b = pol["a_to_b"]
    assert a2b["routing"]["coverage_a"] == pytest.approx(5 / 12)
    assert a2b["routing"]["n_to_human"] == 0          # Tier B never defers
    assert a2b["expected_cost_per_1k"]["misroute"]["point"] == 0.0
    assert a2b["expected_cost_per_1k"]["human"]["point"] == 0.0
    assert a2b["expected_cost_per_1k"]["total"]["point"] == \
        pytest.approx(per1k(7 * B_PER_EXAMPLE))
    assert a2b["accuracy_system"] == 1.0


def test_hand_computed_a_to_b_to_c_paired_subset(tmp_path):
    """The full §4.2 cascade, including the row Tier B rescues from a Tier C parse failure.

    Paired rows (Tier A p_max .99 .91 .83 .75 .67 .59; A correct on the first three).
    tau_A = 0.75 -> A answers 4 (the 4th is WRONG -> $6). The remaining two escalate and
    each pays Tier B's $0.05 compute. Their Tier B confidences are 0.60 and 0.55, so at
    tau_B = 0.58 the first is answered by Tier B (correctly) and the second falls through
    to Tier C (which is right, and costs one receipt).

    The Tier-B-answered row is the parse-failed one in the Tier C receipts: because Tier C
    was never asked, no human charge may fire.
    """
    _, _, ev = _build_tier_b(tmp_path)
    router = ev[router_sim.EVAL_PAIRED]["policies"]["a_to_b_to_c"]

    def per1k(usd):
        return usd * 1000 / 6

    assert router["routing"]["coverage_a"] == pytest.approx(4 / 6)
    assert router["gate"]["coverage_b"] == pytest.approx(1 / 6)
    assert router["gate"]["tier_c_rate"] == pytest.approx(1 / 6)
    assert router["routing"]["n_to_human"] == 0
    comps = router["expected_cost_per_1k"]
    assert comps["misroute"]["point"] == pytest.approx(per1k(6.0))
    assert comps["human"]["point"] == 0.0
    assert comps["api"]["point"] == pytest.approx(per1k(2 * B_PER_EXAMPLE + RECEIPT_COST))
    assert comps["total"]["point"] == \
        pytest.approx(per1k(6.0 + 2 * B_PER_EXAMPLE + RECEIPT_COST))


def test_a_to_b_to_c_pays_both_tiers_when_a_row_falls_through(tmp_path):
    """Incurred spend composes: a row that runs B and then C pays for both.

    Raising tau_B above every Tier B confidence sends both escalated rows to Tier C — and
    the parse-failed one now DOES reach a human, still having paid for both calls.
    """
    _, _, ev = _build_tier_b(tmp_path, tau_b=0.99)
    router = ev[router_sim.EVAL_PAIRED]["policies"]["a_to_b_to_c"]

    def per1k(usd):
        return usd * 1000 / 6

    assert router["gate"]["coverage_b"] == 0.0
    assert router["routing"]["n_to_human"] == 1               # the parse-failed row
    comps = router["expected_cost_per_1k"]
    assert comps["api"]["point"] == pytest.approx(per1k(2 * B_PER_EXAMPLE + 2 * RECEIPT_COST))
    assert comps["human"]["point"] == pytest.approx(per1k(2.5))
    # the parse-failed row's fallback label is never scored, so the only misroute is Tier A's
    assert comps["misroute"]["point"] == pytest.approx(per1k(6.0))


def test_tier_b_comparisons_and_dominance_census_include_the_new_points(tmp_path):
    _, cfg, ev = _build_tier_b(tmp_path)
    summary = router_sim.build_summary(ev, cfg)
    dom = summary["dominance"]["by_router"]
    assert f"{router_sim.EVAL_FULL}/a_to_b" in dom
    assert f"{router_sim.EVAL_FULL}/b2_only" in dom
    assert f"{router_sim.EVAL_PAIRED}/a_to_b_to_c" in dom
    # the INCUMBENT router is now measured against the new baseline too, so its census row
    # cannot report a dominance count that silently omits Tier B
    assert "b2_only" in dom[f"{router_sim.EVAL_FULL}/a_to_human"]["compared_against"]
    assert "b2_only" in summary["dominance"]["model_baselines"]
    assert "b1_only_sa" in summary["dominance"]["note"]
    for row in dom.values():
        assert set(row["dominated"]) <= set(row["compared_against"])


def test_tier_b_transfer_rows_report_both_gates(tmp_path):
    _, _, ev = _build_tier_b(tmp_path)
    rows = {r["policy"]: r for r in ev[router_sim.EVAL_PAIRED]["transfer"]["rows"]}
    abc = rows["a_to_b_to_c"]
    assert abc["cal_tau_b_star"] == pytest.approx(0.58)
    assert abc["applied_tau_b"] == pytest.approx(0.58)
    assert abc["realized_coverage_b"] == pytest.approx(1 / 6)
    assert abc["realized_tier_c_rate"] == pytest.approx(1 / 6)
    assert abc["coverage_matched_tau_b"] is not None
    # one-gate policies do not grow a phantom second gate
    assert "applied_tau_b" not in rows["a_to_human"]


def test_tier_b_run_missing_under_a_tier_b_cost_config_is_a_hard_failure(tmp_path):
    cfg = _cost_config(tmp_path, tier_b=True)
    repo = _mini_repo(tmp_path, cost_sha256=cfg.sha256, tier_b=False)
    with pytest.raises(ValueError, match="prices Tier B but there is no run record"):
        router_sim.load_test_inputs(tmp_path, repo["results"], cost_config=cfg)


def test_relabelled_tier_b_artifact_is_refused(tmp_path):
    """A Tier B artifact scored against a different answer key never reaches a cascade.

    Ids alone cannot catch this: the artifact carries the right complaints with the wrong
    ground truth, and a cascade built on it would credit Tier B for answers Tier A is being
    marked down for. The repo's own verification gate is what fires — `_load_tier_b_rungs`
    runs `load_artifact_verified` on every rung before aligning it, so the Tier B rungs are
    held to exactly the standard Tier A and Tier C are.
    """
    cfg = _cost_config(tmp_path, tier_b=True)
    repo = _mini_repo(tmp_path, cost_sha256=cfg.sha256, tier_b=True)
    run_id = repo["b_ids"][router_sim.TIER_B_CASCADE_CONFIG]
    records = [json.loads(x) for x in repo["results"].read_text().splitlines() if x.strip()]
    record = next(r for r in records if r["run_id"] == run_id)
    flipped = [_flip(t) for t in Y_TRUE]
    y_pred = _preds_from_correct(flipped, B2_CORRECT)
    probs = _write_artifact(tmp_path, run_id, IDS, flipped, y_pred, LABELS, B2_P_MAX,
                            config_sha256=record["config_sha256"])
    record["metrics"] = _metrics_block(flipped, y_pred, probs, LABELS)
    repo["results"].write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(ValueError, match="y_true_matches_split"):
        router_sim.load_test_inputs(tmp_path, repo["results"], cost_config=cfg)


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
                                         results_path=repo["results"],
                                         verify_replay=False)
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
                                         results_path=repo["results"],
                                         verify_replay=False)
    full = {p.name: p for p in router_sim.build_full_policies(inputs, cal)}
    paired = {p.name: p for p in router_sim.build_paired_policies(inputs, cal)}
    with pytest.raises(ValueError, match="different\n?\\s*numbers of rows"):
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
              "config_sha256": "cfghash", "git_sha": "gitsha",
              "dataset": {"split": "cal", "split_sha256": "splithash",
                          "input_sha256": "inputsha"}}
    with pytest.raises(ValueError, match="outside the allowed set"):
        cost_model.load_artifact_verified(record, tmp_path,
                                          allowed_splits=router_sim.ALLOWED_SPLITS)


def test_threshold_file_whose_name_and_body_disagree_on_cost_is_refused(tmp_path):
    """A file NAMED for our cost model but FIT under another is tampering, not a generation.

    This is the case the cost binding exists for: the name is what selects a generation, so
    a file that lies about which one it belongs to would otherwise smuggle a tau fit under
    different prices into this router's operating points.
    """
    cfg = _cost_config(tmp_path)
    repo = _mini_repo(tmp_path, cost_sha256=cfg.sha256)
    path = next(repo["thresholds"].glob(f"{threshold_opt.FAMILY_A_TO_C}__*.json"))
    obj = json.loads(path.read_text())
    obj["cost_config"]["sha256"] = "0" * 64          # body says another cost model
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n")  # name still says ours
    with pytest.raises(ValueError, match="was fit under cost config"):
        router_sim.load_cal_thresholds(repo["thresholds"], cost_sha256=cfg.sha256,
                                       results_path=repo["results"],
                                       verify_replay=False)


def test_threshold_files_of_another_cost_generation_are_skipped_not_fatal(tmp_path):
    """v1-cost and v2-cost threshold sets coexist: v1 evidence is never deleted.

    A file that says the same thing twice — a foreign cost hash in BOTH its name and its
    body — belongs to another cost generation and is simply not this router's input. What
    must still fail is the requested generation being absent entirely.
    """
    cfg = _cost_config(tmp_path)
    repo = _mini_repo(tmp_path, cost_sha256=cfg.sha256)
    foreign = "0" * 64
    for family, dataset in sorted(router_sim.EXPECTED_THRESHOLD_KEYS):
        _write_threshold_file(repo["thresholds"], family=family, dataset=dataset,
                              tau_star=0.11, target_coverage=0.5, cost_sha256=foreign)
    cal = router_sim.load_cal_thresholds(repo["thresholds"], cost_sha256=cfg.sha256,
                                         results_path=repo["results"],
                                         verify_replay=False)
    assert set(cal) == set(router_sim.EXPECTED_THRESHOLD_KEYS)
    assert all(entry.tau_star != 0.11 for entry in cal.values())
    # ...and asking for the generation that is NOT there still fails loudly
    with pytest.raises(ValueError, match="no primary-rung .* threshold files"):
        router_sim.load_cal_thresholds(repo["thresholds"], cost_sha256="1" * 64,
                                       results_path=repo["results"],
                                       verify_replay=False)


def test_missing_threshold_files_are_a_hard_failure(tmp_path):
    empty = tmp_path / "none"
    empty.mkdir()
    with pytest.raises(ValueError, match="no primary-rung .* threshold files"):
        router_sim.load_cal_thresholds(empty, verify_replay=False)


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
            "--thresholds-dir", str(repo["thresholds"]), "--no-verify-replay",
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
    # Keyed by (cost generation, operating-point version): a file must be replayed at the
    # prices it was written under, and there is now more than one generation on disk.
    cals: dict = {}
    cfgs: dict = {}
    inputs: dict = {}
    builders = {router_sim.EVAL_FULL: router_sim.build_full_policies,
                router_sim.EVAL_PAIRED: router_sim.build_paired_policies}

    for path in _REAL_ROUTER_FILES:
        obj = json.loads(path.read_text())
        cost_path = obj["cost_config"]["path"]
        if cost_path not in cfgs:
            cfgs[cost_path] = cost_model.load_cost_config(harness.REPO_ROOT / cost_path)
            assert cfgs[cost_path].sha256 == obj["cost_config"]["sha256"], path.name
            inputs[cost_path] = router_sim.load_test_inputs(
                cost_config=cfgs[cost_path])
        cfg = cfgs[cost_path]
        op_version = obj.get("operating_point_version", router_sim.OP_V1)
        if (cost_path, op_version) not in cals:
            cals[(cost_path, op_version)] = router_sim.load_cal_thresholds(
                cost_sha256=cfg.sha256, derivation=op_version, cost_config=cfg)
        cal = cals[(cost_path, op_version)]
        name = obj["evaluation_set"]
        for transfer, block in ((router_sim.TRANSFER_PRIMARY, obj["policies"]),
                                (router_sim.TRANSFER_SECONDARY,
                                 obj["secondary_coverage_matched"]["policies"])):
            policies = {p.name: p
                        for p in builders[name](inputs[cost_path], cal,
                                                transfer=transfer)}
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
                                       results_path=repo["results"],
                                       verify_replay=False)


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
                                       results_path=repo["results"],
                                       verify_replay=False)


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
                                       results_path=repo["results"],
                                       verify_replay=False)


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
                                       results_path=repo["results"],
                                       verify_replay=False)


def test_incomplete_threshold_set_is_refused(tmp_path):
    cfg = _cost_config(tmp_path)
    repo = _mini_repo(tmp_path, cost_sha256=cfg.sha256)
    stale = next(repo["thresholds"].glob(f"{threshold_opt.FAMILY_A_TO_C}__*.json"))
    stale.unlink()
    with pytest.raises(ValueError, match="not the expected one"):
        router_sim.load_cal_thresholds(repo["thresholds"], cost_sha256=cfg.sha256,
                                       results_path=repo["results"],
                                       verify_replay=False)


def test_missing_cal_run_record_is_refused(tmp_path):
    cfg = _cost_config(tmp_path)
    repo = _mini_repo(tmp_path, cost_sha256=cfg.sha256)
    kept = [json.loads(x) for x in repo["results"].read_text().splitlines() if x.strip()]
    kept = [r for r in kept if r["run_id"] != CAL_RUN_ID]
    repo["results"].write_text("".join(json.dumps(r) + "\n" for r in kept))
    with pytest.raises(ValueError, match="no run record for the CAL rung"):
        router_sim.load_cal_thresholds(repo["thresholds"], cost_sha256=cfg.sha256,
                                       results_path=repo["results"],
                                       verify_replay=False)


@_needs_real
def test_shipped_thresholds_pass_validation_against_the_real_records():
    cfg = cost_model.load_cost_config()
    cal = router_sim.load_cal_thresholds(cost_sha256=cfg.sha256, cost_config=cfg)
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


# ---------------------------------------------------------------------------
# v1 / v2 operating-point coexistence
# ---------------------------------------------------------------------------

def _add_v2_thresholds(repo, cfg, *, taus=(0.60, 0.60, 0.55)):
    """Write a v2 threshold set alongside the v1 one, bound to a v2 CAL record."""
    v2_run = "v2" * 32
    records = [json.loads(x) for x in repo["results"].read_text().splitlines() if x.strip()]
    records.append({
        "run_id": v2_run,
        "config_path": f"configs/{threshold_opt.V2_TIER_A_CAL_CONFIG}.yaml",
        "config_sha256": "v2calcfg", "git_sha": "gitsha",
        "dataset": {"split": "cal", "split_sha256": "calsplithash",
                    "input_sha256": "inputsha"},
    })
    repo["results"].write_text("".join(json.dumps(r) + "\n" for r in records))
    keys = [(threshold_opt.FAMILY_A_TO_HUMAN, threshold_opt.DATASET_FULL_CAL),
            (threshold_opt.FAMILY_A_TO_HUMAN, threshold_opt.DATASET_PAIRED),
            (threshold_opt.FAMILY_A_TO_C, threshold_opt.DATASET_PAIRED)]
    for (family, dataset), tau in zip(keys, taus, strict=True):
        path = _write_threshold_file(
            repo["thresholds"], family=family, dataset=dataset, tau_star=tau,
            target_coverage=0.5, cost_sha256=cfg.sha256, tier_a_run_id=v2_run,
            tier_a_config_sha="v2calcfg", suffix="_v2")
        obj = json.loads(path.read_text())
        obj["derivation"] = threshold_opt.DERIVATION_V2
        obj["is_primary_v2"] = True
        obj["inputs"]["tier_a"]["config_name"] = threshold_opt.V2_TIER_A_CAL_CONFIG
        path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n")
    return v2_run


def test_v1_and_v2_threshold_sets_coexist_without_colliding(tmp_path):
    cfg = _cost_config(tmp_path)
    repo = _mini_repo(tmp_path, cost_sha256=cfg.sha256)
    v2_run = _add_v2_thresholds(repo, cfg)

    v1 = router_sim.load_cal_thresholds(repo["thresholds"], cost_sha256=cfg.sha256,
                                        results_path=repo["results"],
                                        derivation=router_sim.OP_V1, verify_replay=False)
    v2 = router_sim.load_cal_thresholds(repo["thresholds"], cost_sha256=cfg.sha256,
                                        results_path=repo["results"],
                                        derivation=router_sim.OP_V2, verify_replay=False)
    assert set(v1) == set(v2) == set(router_sim.EXPECTED_THRESHOLD_KEYS)
    assert {e.tier_a_run_id for e in v1.values()} == {CAL_RUN_ID}
    assert {e.tier_a_run_id for e in v2.values()} == {v2_run}
    # the two sets carry different taus, i.e. neither leaked into the other
    key = (threshold_opt.FAMILY_A_TO_C, threshold_opt.DATASET_PAIRED)
    assert v1[key].tau_star != v2[key].tau_star


def test_duplicate_hard_fail_is_per_derivation(tmp_path):
    cfg = _cost_config(tmp_path)
    repo = _mini_repo(tmp_path, cost_sha256=cfg.sha256)
    v2_run = _add_v2_thresholds(repo, cfg)
    # a SECOND v2 file for one key: must fail for v2 while v1 still loads cleanly
    path = _write_threshold_file(
        repo["thresholds"], family=threshold_opt.FAMILY_A_TO_C,
        dataset=threshold_opt.DATASET_PAIRED, tau_star=0.31, target_coverage=0.5,
        cost_sha256=cfg.sha256, tier_a_run_id=v2_run, tier_a_config_sha="v2calcfg",
        suffix="_v2dup")
    obj = json.loads(path.read_text())
    obj["derivation"] = threshold_opt.DERIVATION_V2
    obj["inputs"]["tier_a"]["config_name"] = threshold_opt.V2_TIER_A_CAL_CONFIG
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n")

    assert router_sim.load_cal_thresholds(repo["thresholds"], cost_sha256=cfg.sha256,
                                          results_path=repo["results"],
                                          derivation=router_sim.OP_V1, verify_replay=False)
    with pytest.raises(ValueError, match="two primary threshold files claim"):
        router_sim.load_cal_thresholds(repo["thresholds"], cost_sha256=cfg.sha256,
                                       results_path=repo["results"],
                                       derivation=router_sim.OP_V2, verify_replay=False)


def test_v2_router_outputs_are_versioned_and_marked(tmp_path):
    cfg = _cost_config(tmp_path)
    repo = _mini_repo(tmp_path, cost_sha256=cfg.sha256)
    _add_v2_thresholds(repo, cfg)
    evaluations = router_sim.build_all(
        cfg, preds_dir=tmp_path, results_path=repo["results"],
        thresholds_dir=repo["thresholds"], n_resamples=10,
        op_version=router_sim.OP_V2, verify_replay=False)
    for ev in evaluations.values():
        assert ev["operating_point_version"] == router_sim.OP_V2
        assert "in-sample" in ev["tau_derivation_note"]
    assert router_sim.result_filename("paired_subset", cfg, router_sim.OP_V2) == \
        f"paired_subset__opv2__cost-{cfg.sha256[:8]}.json"
    # v1 keeps its original, unversioned filename so committed evidence is not renamed
    assert router_sim.result_filename("paired_subset", cfg, router_sim.OP_V1) == \
        f"paired_subset__cost-{cfg.sha256[:8]}.json"
    v1_eval = router_sim.build_all(
        cfg, preds_dir=tmp_path, results_path=repo["results"],
        thresholds_dir=repo["thresholds"], n_resamples=10, verify_replay=False)
    for ev in v1_eval.values():
        assert "operating_point_version" not in ev


def test_pairing_refuses_identical_ids_with_different_y_true(tmp_path):
    """Same ids, different ground truth: the failure ids alone cannot catch.

    Two artifacts can carry identical id vectors while disagreeing on labels (a re-cut
    split, a relabelled taxonomy). A paired delta across that disagreement would score
    each system against a different answer key while every shape check passed.
    """
    cfg = _cost_config(tmp_path)
    ids = [1, 2, 3]
    a = router_sim.RouterPolicy(
        name="a", evaluation_set="fixture", ids=np.asarray(ids, dtype=np.int64),
        y_true=np.array(["a", "b", "a"], dtype=object),
        y_pred=np.array(["a", "b", "a"], dtype=object),
        to_human=np.zeros(3, dtype=bool), api_cost_usd=np.zeros(3),
        gate={"kind": "answer_all", "tau": None})
    b = router_sim.RouterPolicy(
        name="b", evaluation_set="fixture", ids=np.asarray(ids, dtype=np.int64),
        y_true=np.array(["a", "a", "a"], dtype=object),   # <- one label differs
        y_pred=np.array(["a", "b", "a"], dtype=object),
        to_human=np.zeros(3, dtype=bool), api_cost_usd=np.zeros(3),
        gate={"kind": "answer_all", "tau": None})
    with pytest.raises(ValueError, match="disagree on y_true for 1 row"):
        router_sim.paired_comparison(a, b, cfg, n_resamples=5)
    # length mismatch is reported as such, not as an id difference
    short = router_sim.RouterPolicy(
        name="short", evaluation_set="fixture", ids=np.asarray([1, 2], dtype=np.int64),
        y_true=np.array(["a", "b"], dtype=object),
        y_pred=np.array(["a", "b"], dtype=object),
        to_human=np.zeros(2, dtype=bool), api_cost_usd=np.zeros(2),
        gate={"kind": "answer_all", "tau": None})
    with pytest.raises(ValueError, match="different numbers of rows"):
        router_sim.paired_comparison(a, short, cfg, n_resamples=5)


# ---------------------------------------------------------------------------
# tau-replay gate (against the real committed threshold files)
# ---------------------------------------------------------------------------

_REAL_THRESHOLD_FILES = sorted(
    p for p in router_sim.DEFAULT_THRESHOLDS_DIR.glob("*__cost-*.json")
    if not p.name.startswith("summary__")
)
_needs_thresholds = pytest.mark.skipif(
    not (_REAL_THRESHOLD_FILES and router_sim.DEFAULT_PREDS_DIR.exists()),
    reason="real thresholds/preds not present")


# The cost generations present on disk. The tamper tests run against each of them: a gate
# that fired only for the generation the file happened to sort first would be no gate at all
# for the other.
_REAL_COST_CONFIG_PATHS = sorted({
    json.loads(p.read_text())["cost_config"]["path"] for p in _REAL_THRESHOLD_FILES
})


def _tampered_threshold_dir(tmp_path, derivation, mutate, cfg):
    """Copy the real threshold set, apply `mutate` to one file of THIS cost generation.

    The generation matters: a file belonging to another cost model is skipped by the
    loader (both name and body say so), so tampering with one would prove nothing.
    """
    out = tmp_path / "thresholds"
    out.mkdir()
    touched = False
    for path in _REAL_THRESHOLD_FILES:
        obj = json.loads(path.read_text())
        if (obj.get("derivation", threshold_opt.DERIVATION_V1) == derivation
                and obj.get("cost_config", {}).get("sha256") == cfg.sha256
                and not touched and obj.get("is_primary")
                and obj["policy_family"] == threshold_opt.FAMILY_A_TO_HUMAN
                and obj["dataset"] == threshold_opt.DATASET_FULL_CAL):
            mutate(obj)
            touched = True
        (out / path.name).write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n")
    assert touched, "no file was tampered; the fixture proves nothing"
    return out


@_needs_thresholds
@pytest.mark.parametrize("cost_path", _REAL_COST_CONFIG_PATHS)
def test_shipped_thresholds_pass_the_tau_replay_gate(cost_path):
    cfg = cost_model.load_cost_config(harness.REPO_ROOT / cost_path)
    derivations = ((router_sim.OP_V2,) if cost_model.prices_tier_b(cfg)
                   else (router_sim.OP_V1, router_sim.OP_V2))
    for derivation in derivations:
        cal = router_sim.load_cal_thresholds(cost_sha256=cfg.sha256, cost_config=cfg,
                                             derivation=derivation)
        assert set(cal) == set(router_sim.expected_threshold_keys(cfg))


@_needs_thresholds
@pytest.mark.parametrize("cost_path", _REAL_COST_CONFIG_PATHS)
def test_tau_replay_catches_an_edited_answered_count(tmp_path, cost_path):
    def mutate(obj):
        obj["n_answered_at_tau_star"] += 1
        obj["target_coverage_a"] = obj["n_answered_at_tau_star"] / obj["n_examples"]
    cfg = cost_model.load_cost_config(harness.REPO_ROOT / cost_path)
    bad = _tampered_threshold_dir(tmp_path, router_sim.OP_V2, mutate, cfg)
    with pytest.raises(ValueError, match="does not replay.*selects"):
        router_sim.load_cal_thresholds(bad, cost_sha256=cfg.sha256, cost_config=cfg,
                                       derivation=router_sim.OP_V2)


@_needs_thresholds
@pytest.mark.parametrize("cost_path", _REAL_COST_CONFIG_PATHS)
def test_tau_replay_catches_an_edited_tau(tmp_path, cost_path):
    # Move tau to a value that still satisfies every METADATA check (finite, in range,
    # counts self-consistent) but selects a different set of rows.
    def mutate(obj):
        obj["tau_star"] = min(obj["tau_star"] + 0.05, 1.0)
    cfg = cost_model.load_cost_config(harness.REPO_ROOT / cost_path)
    bad = _tampered_threshold_dir(tmp_path, router_sim.OP_V2, mutate, cfg)
    with pytest.raises(ValueError, match="does not replay"):
        router_sim.load_cal_thresholds(bad, cost_sha256=cfg.sha256, cost_config=cfg,
                                       derivation=router_sim.OP_V2)


@_needs_thresholds
@pytest.mark.parametrize("cost_path", _REAL_COST_CONFIG_PATHS)
def test_tau_replay_catches_an_edited_objective(tmp_path, cost_path):
    def mutate(obj):
        point = obj["operating_points"]["tau_star"]["expected_cost_per_1k"]["total"]
        point["point"] = point["point"] + 1.0
    cfg = cost_model.load_cost_config(harness.REPO_ROOT / cost_path)
    bad = _tampered_threshold_dir(tmp_path, router_sim.OP_V2, mutate, cfg)
    with pytest.raises(ValueError, match="does not replay.*objective"):
        router_sim.load_cal_thresholds(bad, cost_sha256=cfg.sha256, cost_config=cfg,
                                       derivation=router_sim.OP_V2)


def test_replay_without_a_cost_config_is_refused(tmp_path):
    cfg = _cost_config(tmp_path)
    repo = _mini_repo(tmp_path, cost_sha256=cfg.sha256)
    with pytest.raises(ValueError, match="verify_replay needs the cost config"):
        router_sim.load_cal_thresholds(repo["thresholds"], cost_sha256=cfg.sha256,
                                       results_path=repo["results"])


@_needs_real
def test_v1_router_outputs_regenerate_byte_identically(tmp_path):
    """Byte-level regeneration of the committed v1 router evidence."""
    out_dir = tmp_path / "router"
    assert router_sim.main(["--out-dir", str(out_dir),
                            "--op-version", router_sim.OP_V1]) == 0
    regenerated = sorted(out_dir.glob("*.json"))
    assert regenerated
    for path in regenerated:
        committed = router_sim.DEFAULT_ROUTER_DIR / path.name
        assert committed.exists(), f"{path.name} is not committed"
        assert path.read_bytes() == committed.read_bytes(), path.name
        assert "__opv2__" not in path.name
