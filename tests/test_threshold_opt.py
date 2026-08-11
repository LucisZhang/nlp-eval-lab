"""Threshold-optimization tests: hand-computed sweeps for both policy families (including
a parse-failed row that pays incurred API + c_human and never a misroute), the fast
prefix-sum sweep vs the row-by-row cost_model reference at EVERY grid threshold, subset
id-alignment and receipt-gate hard failures, CI-only-at-operating-points, the sensitivity
grid's exact-defaults cell, and end-to-end CLI determinism over a synthetic mini-repo."""

from __future__ import annotations

import json

import duckdb
import numpy as np
import pytest

from triage_lab import cost_model, harness, predictions, threshold_opt

# The tau*-replay regression runs against the SHIPPED results/thresholds/ files and the
# artifacts they were built from; it skips (rather than fails) where that data is absent,
# matching tests/test_tier_c_prompt.py's convention for real-data-dependent checks.
_REAL_THRESHOLDS = sorted(threshold_opt.DEFAULT_THRESHOLDS_DIR.glob("*__cost-*.json"))
_REAL_POLICY_FILES = [p for p in _REAL_THRESHOLDS if not p.name.startswith("summary__")]
_HAS_REAL = bool(_REAL_POLICY_FILES) and threshold_opt.DEFAULT_PREDS_DIR.exists() and \
    harness.DEFAULT_RESULTS_PATH.exists()
_needs_real = pytest.mark.skipif(not _HAS_REAL, reason="real preds/thresholds not present")

C_MISROUTE = 6.00
C_HUMAN = 2.50

SLUG = "anthropic/claude-haiku-4.5"
PROMPT_RATE = 1e-6
COMPLETION_RATE = 5e-6
PRICING = {
    "slug": SLUG,
    "prompt_usd_per_token": PROMPT_RATE,
    "completion_usd_per_token": COMPLETION_RATE,
}

# 4-row fixture shared by the hand-computed cases. Confidence descending, the two
# least-confident rows are the ones Tier A gets wrong (the situation a gate exists for).
P_MAX = np.array([0.9, 0.7, 0.5, 0.3])
CORRECT_A = np.array([True, True, False, False])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _policy(family, p_max, correct_a, arm, dataset="fixture"):
    return threshold_opt.PolicyData(
        family=family,
        dataset=dataset,
        ids=np.arange(len(p_max), dtype=np.int64),
        p_max=np.asarray(p_max, dtype=np.float64),
        correct_a=np.asarray(correct_a, dtype=bool),
        arm=arm,
        inputs={},
    )


def _sweep(policy, *, c_misroute=C_MISROUTE, c_human=C_HUMAN):
    return threshold_opt.sweep(threshold_opt.build_grid(policy),
                               c_misroute=c_misroute, c_human=c_human)


B_PER_EXAMPLE = 0.05   # deliberately huge next to the real ~$5.6e-6, so it is VISIBLE


def _cost_config(tmp_path, *, c_misroute=C_MISROUTE, c_human=C_HUMAN, tier_b=False,
                 name="cost.yaml"):
    """Synthetic cost config; `tier_b=True` adds the two Tier B tiers (the v2 shape)."""
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


def _write_split(tmp_path, ids, y_true, *, name="cal"):
    """Frozen-split stand-in the predictions verification gate joins against (DuckDB)."""
    splits = tmp_path / "splits"
    splits.mkdir(exist_ok=True)
    path = splits / f"{name}.parquet"
    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.register("_s", {
            "complaint_id": np.asarray(ids, dtype=np.int64),
            "class": np.asarray([str(v) for v in y_true], dtype=object),
        })
        con.execute(f"COPY (SELECT complaint_id, \"class\" FROM _s) TO '{path}' "
                    "(FORMAT parquet)")
    finally:
        con.close()
    return splits


def _write_config(tmp_path, name, runner, splits_dir, *, split="cal"):
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        f"model:\n  runner: {runner}\n  name: {name}\n"
        f"data:\n  split: {split}\n  splits_dir: {splits_dir}\n"
        "  label_column: class\n  order_column: complaint_id\n"
    )
    return path


def _write_artifact(tmp_path, run_id, ids, y_true, y_pred, labels, p_max=None,
                    *, split="cal", split_sha256="splithash", config_sha256="cfghash",
                    git_sha="gitsha", input_sha256="inputsha"):
    """Artifact with controllable p_max (probs are built to realize it). Returns probs."""
    index = {lbl: i for i, lbl in enumerate(labels)}
    probs = np.zeros((len(ids), len(labels)), dtype=np.float64)
    for i, pred in enumerate(y_pred):
        p = 1.0 if p_max is None else float(p_max[i])
        probs[i, index[pred]] = p
        rest = (1.0 - p) / max(len(labels) - 1, 1)
        for j in range(len(labels)):
            if j != index[pred]:
                probs[i, j] = rest
    prov = predictions.ArtifactProvenance(
        run_id=run_id, config_sha256=config_sha256, split=split,
        split_sha256=split_sha256, class_labels=list(labels),
        git_sha=git_sha, input_sha256=input_sha256,
    )
    path = tmp_path / f"{run_id}.parquet"
    predictions.write_artifact(
        path, ids=np.asarray(ids, dtype=np.int64), y_true=y_true, y_pred=y_pred,
        probs=probs, class_labels=list(labels), provenance=prov,
    )
    return probs


def _receipt(cid, *, prompt=995, completion=1, parse_failed=False, slug=SLUG):
    return {
        "complaint_id": cid,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "computed_cost_usd": prompt * PROMPT_RATE + completion * COMPLETION_RATE,
        "parse_failed": parse_failed,
        "slug": slug,
        "content": '{"label": "a"}',
    }


def _write_receipts(path, receipts):
    path.write_text("".join(json.dumps(r) + "\n" for r in receipts), encoding="utf-8")
    return path


def _record(run_id, config_name, *, split="cal", split_sha256="splithash",
            config_sha256="cfghash", cost_usd=None, raw_log_path=None,
            config_path=None, metrics=None, git_sha="gitsha", input_sha256="inputsha"):
    rec = {
        "run_id": run_id,
        "config_path": str(config_path) if config_path else f"configs/{config_name}.yaml",
        "config_sha256": config_sha256,
        "git_sha": git_sha,
        "dataset": {"split": split, "split_sha256": split_sha256,
                    "input_sha256": input_sha256},
    }
    if metrics is not None:
        rec["metrics"] = metrics
    if cost_usd is not None:
        rec["cost_usd"] = cost_usd
    if raw_log_path is not None:
        rec["extra"] = {
            "raw_log_path": str(raw_log_path),
            "model_slug": SLUG,
            "pricing_snapshot": PRICING,
        }
    return rec


def _metrics_block(y_true, y_pred, probs, labels):
    """Point metrics the predictions verification gate diffs the artifact against."""
    points = harness.evaluate(np.asarray(y_true, dtype=object),
                              np.asarray(y_pred, dtype=object), probs, list(labels))
    return {name: {"point": float(points[name])} for name in predictions.VERIFY_METRICS}


def _mini_repo(tmp_path, *, receipt_overrides=None, artifact_ids=None,
               second_rung=None, tier_a_config=None, tier_b=False):
    """Synthetic CAL mini-repo complete enough for the FULL predictions gate.

    That means real config YAMLs (re-hashed and compared), a frozen-split parquet the
    artifacts are joined against, and recomputed point metrics in each record — the
    fixture has to be a real run's worth of provenance, because `load_artifact_checked`
    now runs `predictions.verify_artifact` and a stub would simply fail it.
    """
    labels = ["a", "b"]
    a_id = "aa" * 32
    c_id = "cc" * 32
    ids = list(range(100, 112))
    y_true = ["a", "b"] * 6
    y_pred_a = ["a", "b", "a", "b", "a", "b", "b", "a", "a", "b", "b", "a"]
    # Kept above 0.5 so the predicted class really is the argmax: with two labels a
    # "confidence" below 0.5 would make the OTHER column the row's p_max.
    p_max = np.linspace(0.95, 0.55, 12)
    splits_dir = _write_split(tmp_path, ids, y_true)
    tier_a_config = tier_a_config or threshold_opt.PRIMARY_TIER_A_CONFIG
    a_cfg = _write_config(tmp_path, tier_a_config, "tier_a", splits_dir)
    c_cfg = _write_config(tmp_path, threshold_opt.TIER_C_CAL_CONFIG, "tier_c", splits_dir)
    a_sha = harness.config_sha256(a_cfg)
    c_sha = harness.config_sha256(c_cfg)
    a_probs = _write_artifact(tmp_path, a_id, artifact_ids or ids, y_true, y_pred_a,
                              labels, p_max, config_sha256=a_sha)

    c_ids = ids[::2]  # 6 paired ids
    c_true = [y_true[ids.index(i)] for i in c_ids]
    c_pred = ["a", "a", "b", "a", "a", "b"]
    c_probs = _write_artifact(tmp_path, c_id, c_ids, c_true, c_pred, labels,
                              config_sha256=c_sha)

    overrides = receipt_overrides or {}
    receipts = [_receipt(cid, prompt=995 + 100 * n, parse_failed=(n == 1),
                         **overrides.get(cid, {}))
                for n, cid in enumerate(c_ids)]
    log = _write_receipts(tmp_path / "calls.jsonl", receipts)
    total = sum(r["computed_cost_usd"] for r in receipts)

    results = tmp_path / "runs.jsonl"
    records = [
        _record(a_id, tier_a_config, config_path=a_cfg,
                config_sha256=a_sha,
                metrics=_metrics_block(y_true, y_pred_a, a_probs, labels)),
        _record(c_id, threshold_opt.TIER_C_CAL_CONFIG, config_path=c_cfg,
                config_sha256=c_sha,
                metrics=_metrics_block(c_true, c_pred, c_probs, labels),
                cost_usd=total, raw_log_path=log),
    ]
    if tier_b:
        # The B2 CAL rung the Tier B families escalate to. Correct wherever Tier A is
        # wrong (and wrong on two rows Tier A gets right), with its own confidence
        # ordering, so neither gate can be fit by accident from the other's ranking.
        b_id = "b2" * 32
        b_cfg = _write_config(tmp_path, threshold_opt.TIER_B_CAL_CONFIG, "tier_b",
                              splits_dir)
        b_sha = harness.config_sha256(b_cfg)
        b_correct = [False, False] + [True] * 10
        b_pred = [t if ok else ("b" if t == "a" else "a")
                  for t, ok in zip(y_true, b_correct, strict=True)]
        b_p_max = np.linspace(0.6, 0.95, 12)
        b_probs = _write_artifact(tmp_path, b_id, ids, y_true, b_pred, labels, b_p_max,
                                  config_sha256=b_sha)
        records.append(_record(b_id, threshold_opt.TIER_B_CAL_CONFIG, config_path=b_cfg,
                               config_sha256=b_sha,
                               metrics=_metrics_block(y_true, b_pred, b_probs, labels)))

    if second_rung:
        # A second Tier A rung whose artifact carries a duplicate complaint_id: the first
        # rung builds fine, so the batch only fails AFTER real work is in memory — which
        # is what makes it a batch-atomicity fixture rather than an early-exit one.
        b_id = "bb" * 32
        b_cfg = _write_config(tmp_path, second_rung, "tier_a", splits_dir)
        b_sha = harness.config_sha256(b_cfg)
        dup_ids = list(ids)
        dup_ids[5] = dup_ids[4]
        b_probs = _write_artifact(tmp_path, b_id, dup_ids, y_true, y_pred_a, labels,
                                  p_max, config_sha256=b_sha)
        records.append(_record(b_id, second_rung, config_path=b_cfg, config_sha256=b_sha,
                               metrics=_metrics_block(y_true, y_pred_a, b_probs, labels)))

    results.write_text("".join(json.dumps(r) + "\n" for r in records))
    return {"a_id": a_id, "c_id": c_id, "results": results, "log": log,
            "splits_dir": splits_dir, "ids": ids, "labels": labels,
            "y_true": y_true, "y_pred_a": y_pred_a, "p_max": p_max}


# ---------------------------------------------------------------------------
# Hand-computed sweeps
# ---------------------------------------------------------------------------

def test_a_to_human_hand_computed_sweep():
    # 4 rows, c_misroute $6, c_human $2.50 -> each row is $250/1k per dollar.
    #   k=0 all human      : 4 x 2.50            = $10.00 -> 2500/1k
    #   k=1 A answers r0   : 3 x 2.50            = $ 7.50 -> 1875/1k
    #   k=2 A answers r0,r1: 2 x 2.50            = $ 5.00 -> 1250/1k   <- argmin
    #   k=3 + r2 (wrong)   : 6.00 + 2.50         = $ 8.50 -> 2125/1k
    #   k=4 all A, r2+r3 wrong: 2 x 6.00         = $12.00 -> 3000/1k
    policy = _policy(threshold_opt.FAMILY_A_TO_HUMAN, P_MAX, CORRECT_A,
                     threshold_opt.human_arm(4))
    rows = _sweep(policy)
    assert rows["n_answered_a"].tolist() == [0, 1, 2, 3, 4]
    assert rows["tau"][0] == np.inf
    assert rows["tau"][1:].tolist() == [0.9, 0.7, 0.5, 0.3]
    assert rows["cost_per_1k"] == pytest.approx([2500.0, 1875.0, 1250.0, 2125.0, 3000.0])
    assert rows["coverage_a"] == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert rows["human_rate"] == pytest.approx([1.0, 0.75, 0.5, 0.25, 0.0])
    assert rows["api_per_1k"] == pytest.approx([0.0] * 5)

    j = threshold_opt.argmin_index(rows["cost_per_1k"])
    assert j == 2
    assert rows["tau"][j] == 0.7
    assert rows["coverage_a"][j] == 0.5
    assert rows["cost_per_1k"][j] == pytest.approx(1250.0)
    # accuracy of the machine-answered set at tau*: both A-answered rows are right
    assert rows["accuracy_machine"][j] == pytest.approx(1.0)
    # ...and the system accuracy credits the human rows as resolved
    assert rows["accuracy_system"][j] == pytest.approx(1.0)


def _tier_c_fixture_arm(*, fallback_pred="b"):
    """Tier C arm for the 4-row fixture; row 2 is the parse-failed one.

    row0: parsed, C wrong  -> api 0.10 + misroute
    row1: parsed, C right  -> api 0.20
    row2: PARSE FAILED     -> api 0.30 + c_human  (its y_pred is the fallback label)
    row3: parsed, C right  -> api 0.40
    """
    y_true = np.array(["a", "b", "a", "b"], dtype=object)
    y_pred = np.array(["b", "b", fallback_pred, "b"], dtype=object)
    return threshold_opt.tier_c_arm(
        y_true, y_pred,
        api_cost_usd=[0.10, 0.20, 0.30, 0.40],
        parse_failed=[False, False, True, False],
    )


def test_a_to_c_hand_computed_sweep():
    #   k=0 (c_only): api 1.00 + misroute 6.00 (r0) + human 2.50 (r2) = $9.50 -> 2375/1k
    #   k=1: A takes r0 (right); escalate r1,r2,r3: api 0.90 + human 2.50 = $3.40 -> 850/1k
    #   k=2: A takes r0,r1;      escalate r2,r3   : api 0.70 + human 2.50 = $3.20 -> 800/1k
    #   k=3: A takes r0,r1,r2 (r2 WRONG -> 6.00); escalate r3: api 0.40   = $6.40 -> 1600/1k
    #   k=4: A takes all; r2,r3 wrong                                     = $12.00 -> 3000/1k
    policy = _policy(threshold_opt.FAMILY_A_TO_C, P_MAX, CORRECT_A, _tier_c_fixture_arm())
    rows = _sweep(policy)
    assert rows["cost_per_1k"] == pytest.approx([2375.0, 850.0, 800.0, 1600.0, 3000.0])
    assert rows["api_per_1k"] == pytest.approx([250.0, 225.0, 175.0, 100.0, 0.0])
    assert rows["human_per_1k"] == pytest.approx([625.0, 625.0, 625.0, 0.0, 0.0])
    assert rows["misroute_per_1k"] == pytest.approx([1500.0, 0.0, 0.0, 1500.0, 3000.0])
    assert rows["human_rate"] == pytest.approx([0.25, 0.25, 0.25, 0.0, 0.0])

    j = threshold_opt.argmin_index(rows["cost_per_1k"])
    assert j == 2
    assert rows["tau"][j] == 0.7
    assert rows["coverage_a"][j] == 0.5
    assert rows["escalation_rate"][j] == 0.5
    assert rows["cost_per_1k"][j] == pytest.approx(800.0)


def test_parse_failed_row_pays_api_plus_human_and_never_misroutes():
    # The parse-failed row's y_pred is a fallback label. Whether that label happens to be
    # right ("a") or wrong ("b"), the row must cost exactly api + c_human: it was answered
    # by a human, so it can neither incur a misroute nor count as a machine answer.
    wrong_fallback = _policy(threshold_opt.FAMILY_A_TO_C, P_MAX, CORRECT_A,
                             _tier_c_fixture_arm(fallback_pred="b"))
    right_fallback = _policy(threshold_opt.FAMILY_A_TO_C, P_MAX, CORRECT_A,
                             _tier_c_fixture_arm(fallback_pred="a"))
    rows_w = _sweep(wrong_fallback)
    rows_r = _sweep(right_fallback)
    for key in ("cost_per_1k", "misroute_per_1k", "api_per_1k", "human_per_1k",
                "accuracy_machine", "accuracy_system"):
        assert rows_w[key] == pytest.approx(rows_r[key]), key

    # At k=0 every row escalates: the parse-failed row contributes its $0.30 of incurred
    # spend AND $2.50 of human review, and contributes nothing to the misroute term
    # (which is $6.00 from row 0 alone).
    correct, api, to_human = threshold_opt.materialize(wrong_fallback, np.inf)
    assert to_human.tolist() == [False, False, True, False]
    assert api == pytest.approx([0.10, 0.20, 0.30, 0.40])
    comps = cost_model.cost_components(correct, api, to_human,
                                       c_misroute=C_MISROUTE, c_human=C_HUMAN)
    assert comps["human"].tolist() == [0.0, 0.0, 2.5, 0.0]
    assert comps["api"][2] == pytest.approx(0.30)
    assert comps["misroute"].tolist() == [6.0, 0.0, 0.0, 0.0]


def test_human_arm_correctness_flag_is_inert_for_cost():
    # Rows resolved by a human are recorded correct (the cost model's assumption). Flipping
    # that flag must not move any cost, because the misroute charge is gated on the row
    # being machine-answered.
    arm = threshold_opt.human_arm(4)
    flipped = threshold_opt.EscalationArm(
        name=arm.name, api_cost_usd=arm.api_cost_usd, to_human=arm.to_human,
        correct=np.zeros(4, dtype=bool),
    )
    base = _sweep(_policy(threshold_opt.FAMILY_A_TO_HUMAN, P_MAX, CORRECT_A, arm))
    other = _sweep(_policy(threshold_opt.FAMILY_A_TO_HUMAN, P_MAX, CORRECT_A, flipped))
    assert base["cost_per_1k"] == pytest.approx(other["cost_per_1k"])


def test_argmin_tie_break_prefers_more_tier_a_coverage():
    cost = np.array([5.0, 3.0, 3.0, 7.0])
    assert threshold_opt.argmin_index(cost) == 2  # last minimum = highest coverage
    assert threshold_opt.argmin_index(np.array([1.0, 2.0, 3.0])) == 0


# ---------------------------------------------------------------------------
# Tier B families: a_to_b (one gate) and a_to_b_to_c (joint two-gate fit)
# ---------------------------------------------------------------------------

# Tier B on the 4-row fixture: wrong on the row Tier A is most confident about, right on
# the three others — i.e. the tiers disagree in the region the gate has to find.
B_CORRECT = [False, True, True, True]
# Tier B's own confidence, deliberately NOT ordered like Tier A's, so a fit that silently
# reused the Tier A ranking would land somewhere else.
P_MAX_B = np.array([0.4, 0.8, 0.6, 0.2])


def _tier_b_fixture_arm(per_example=B_PER_EXAMPLE):
    y_true = np.array(["a", "b", "a", "b"], dtype=object)
    y_pred = np.array([t if ok else ("b" if t == "a" else "a")
                       for t, ok in zip(y_true, B_CORRECT, strict=True)], dtype=object)
    return threshold_opt.tier_b_arm(y_true, y_pred, per_example)


def test_a_to_b_hand_computed_sweep():
    # Tier B compute $0.05/escalated row, c_misroute $6. Tier B is wrong only on r0.
    #   k=0 all to B      : 4 x 0.05 + 6.00 (r0)   = $6.20 -> 1550/1k   (= b_only)
    #   k=1 A answers r0  : 3 x 0.05               = $0.15 ->   37.5/1k
    #   k=2 A answers r0r1: 2 x 0.05               = $0.10 ->   25/1k   <- argmin
    #   k=3 + r2 (A wrong): 6.00 + 0.05            = $6.05 -> 1512.5/1k
    #   k=4 all A         : 2 x 6.00               = $12.00 -> 3000/1k
    policy = _policy(threshold_opt.FAMILY_A_TO_B, P_MAX, CORRECT_A, _tier_b_fixture_arm())
    rows = _sweep(policy)
    assert rows["cost_per_1k"] == pytest.approx([1550.0, 37.5, 25.0, 1512.5, 3000.0])
    assert rows["human_rate"] == pytest.approx([0.0] * 5)   # Tier B never defers
    assert rows["api_per_1k"] == pytest.approx([50.0, 37.5, 25.0, 12.5, 0.0])
    j = threshold_opt.argmin_index(rows["cost_per_1k"])
    assert rows["tau"][j] == 0.7
    # System accuracy at tau*: A right on the 2 it answers, B right on the 2 it takes.
    assert rows["accuracy_system"][j] == pytest.approx(1.0)


def test_tier_b_then_c_arm_charges_b_always_and_c_only_on_fallthrough():
    """The per-row semantics the whole two-gate family rests on.

    At tau_B = 0.6, Tier B answers r1 (0.8) and r2 (0.6); r0 (0.4) and r3 (0.2) fall
    through to Tier C. Row 2 is Tier C's parse-failed row, but Tier B answered it here, so
    the parse-failure never fires — the arm must not leak a human charge for a row Tier C
    was never asked about.
    """
    y_true = np.array(["a", "b", "a", "b"], dtype=object)
    b_pred = np.array(["b", "b", "a", "b"], dtype=object)     # B wrong on r0 only
    c_pred = np.array(["a", "b", "b", "a"], dtype=object)     # C wrong on r3
    arm = threshold_opt.tier_b_then_c_arm(
        y_true, b_pred, P_MAX_B, 0.6, B_PER_EXAMPLE, c_pred,
        c_api_cost_usd=[0.10, 0.20, 0.30, 0.40],
        parse_failed=[False, False, True, False])
    # r0, r3 pay B's compute AND their Tier C call; r1, r2 pay B's compute alone.
    assert arm.api_cost_usd == pytest.approx([0.15, 0.05, 0.05, 0.45])
    assert arm.to_human.tolist() == [False, False, False, False]
    assert arm.correct.tolist() == [True, True, True, False]   # r0 -> C right; r3 -> C wrong
    assert arm.b_answered.tolist() == [False, True, True, False]

    # Now drop the Tier B gate so r2 falls through: its parse failure fires, it routes to a
    # human, its fallback label is discarded, and BOTH spends still stand.
    arm2 = threshold_opt.tier_b_then_c_arm(
        y_true, b_pred, P_MAX_B, 0.9, B_PER_EXAMPLE, c_pred,
        c_api_cost_usd=[0.10, 0.20, 0.30, 0.40],
        parse_failed=[False, False, True, False])
    assert arm2.to_human.tolist() == [False, False, True, False]
    assert arm2.api_cost_usd[2] == pytest.approx(0.35)
    assert bool(arm2.correct[2]) is True       # human-resolved, never scored as a miss


def _abc_make_policy(*, c_api=(0.10, 0.20, 0.30, 0.40),
                     parse_failed=(False, False, True, False)):
    y_true = np.array(["a", "b", "a", "b"], dtype=object)
    b_pred = np.array(["b", "b", "a", "b"], dtype=object)
    c_pred = np.array(["a", "b", "b", "a"], dtype=object)

    def make(tau_b):
        return _policy(threshold_opt.FAMILY_A_TO_B_TO_C, P_MAX, CORRECT_A,
                       threshold_opt.tier_b_then_c_arm(
                           y_true, b_pred, P_MAX_B, tau_b, B_PER_EXAMPLE, c_pred,
                           c_api_cost_usd=list(c_api), parse_failed=list(parse_failed)))
    return make


def test_joint_sweep_matches_a_brute_force_search_over_the_whole_2d_grid():
    """The joint argmin must be the argmin — checked against the row-by-row cost reference.

    `joint_sweep` reuses the prefix-sum machinery inside an outer loop; the reference here
    goes through `cost_at`, i.e. `cost_model.expected_cost_per_1k` per candidate pair, so
    a reassociation bug in the fast path cannot hide.
    """
    make = _abc_make_policy()
    tau_bs = threshold_opt.tau_b_candidates(P_MAX_B)
    assert tau_bs[0] == np.inf
    assert tau_bs[1:].tolist() == [0.8, 0.6, 0.4, 0.2]

    fit = threshold_opt.joint_sweep(make, tau_bs, c_misroute=C_MISROUTE, c_human=C_HUMAN)

    best = None
    for tau_b in tau_bs:
        policy = make(float(tau_b))
        for tau_a in threshold_opt.build_grid(policy).tau:
            total = threshold_opt.cost_at(policy, tau_a, c_misroute=C_MISROUTE,
                                          c_human=C_HUMAN)["total"]
            if best is None or total < best[0] - 1e-12:
                best = (total, tau_b, tau_a)
    assert fit.rows["cost_per_1k"][fit.j_star] == pytest.approx(best[0], abs=1e-9)
    # every grid entry reports the best tau_A at ITS tau_B, and the argmin points at one
    assert len(fit.grid) == len(tau_bs)
    assert fit.grid[fit.j_grid_star]["cost_per_1k"] == pytest.approx(best[0], abs=1e-9)
    assert min(row["cost_per_1k"] for row in fit.grid) == pytest.approx(best[0], abs=1e-9)


def test_joint_fit_breaks_ties_toward_more_tier_a_then_more_tier_b():
    """With a free, perfect Tier B and a perfect Tier A, every gate costs $0.

    Every (tau_A, tau_B) pair then ties at zero, so the tie-break is the ONLY thing
    choosing: it must answer everything at Tier A (largest coverage_a) and, among the
    remaining ties, prefer the Tier B gate that sends fewest rows to the paid tier.
    """
    y_true = np.array(["a", "b", "a", "b"], dtype=object)

    def make(tau_b):
        return _policy(
            threshold_opt.FAMILY_A_TO_B_TO_C, P_MAX, [True] * 4,
            threshold_opt.tier_b_then_c_arm(
                y_true, y_true, P_MAX_B, tau_b, 0.0, y_true,
                c_api_cost_usd=[0.0] * 4, parse_failed=[False] * 4))

    fit = threshold_opt.joint_sweep(make, threshold_opt.tau_b_candidates(P_MAX_B),
                                    c_misroute=C_MISROUTE, c_human=C_HUMAN)
    assert fit.rows["cost_per_1k"][fit.j_star] == pytest.approx(0.0)
    assert fit.rows["coverage_a"][fit.j_star] == pytest.approx(1.0)
    assert fit.tau_b == pytest.approx(float(P_MAX_B.min()))   # lowest gate = widest Tier B
    assert fit.coverage_b_marginal == pytest.approx(1.0)


def test_no_gate_endpoint_is_named_per_family():
    assert threshold_opt.NO_GATE_LABEL[threshold_opt.FAMILY_A_TO_B] == "b_only"
    assert threshold_opt.NO_GATE_LABEL[threshold_opt.FAMILY_A_TO_B_TO_C] == "b_to_c"
    # and the pre-Tier-B families keep the names their committed files use
    assert threshold_opt.NO_GATE_LABEL[threshold_opt.FAMILY_A_TO_HUMAN] == "all_human"
    assert threshold_opt.NO_GATE_LABEL[threshold_opt.FAMILY_A_TO_C] == "c_only"


# ---------------------------------------------------------------------------
# Fast sweep == row-by-row cost_model reference
# ---------------------------------------------------------------------------

def _random_a_to_c_policy(seed, n=137):
    rng = np.random.default_rng(seed)
    p_max = rng.random(n).round(3)  # rounding forces ties into the grid
    correct_a = rng.random(n) < 0.6
    parse_failed = rng.random(n) < 0.1
    labels = np.array(["a", "b", "c"], dtype=object)
    y_true = labels[rng.integers(0, 3, size=n)]
    y_pred = labels[rng.integers(0, 3, size=n)]
    arm = threshold_opt.tier_c_arm(y_true, y_pred, rng.random(n) * 0.005, parse_failed)
    return _policy(threshold_opt.FAMILY_A_TO_C, p_max, correct_a, arm)


def test_fast_sweep_matches_the_reference_at_every_threshold():
    policy = _random_a_to_c_policy(7)
    rows = _sweep(policy)
    for j, tau in enumerate(rows["tau"]):
        ref = threshold_opt.cost_at(policy, tau, c_misroute=C_MISROUTE, c_human=C_HUMAN)
        assert rows["cost_per_1k"][j] == pytest.approx(ref["total"], abs=1e-9), j
        assert rows["misroute_per_1k"][j] == pytest.approx(ref["misroute"], abs=1e-9), j
        assert rows["api_per_1k"][j] == pytest.approx(ref["api"], abs=1e-9), j
        assert rows["human_per_1k"][j] == pytest.approx(ref["human"], abs=1e-9), j


def test_grid_is_tie_aware():
    # Duplicate p_max values collapse to one threshold whose k covers the whole tie block.
    policy = _policy(threshold_opt.FAMILY_A_TO_HUMAN,
                     [0.9, 0.9, 0.5, 0.5, 0.5], [True] * 5, threshold_opt.human_arm(5))
    rows = _sweep(policy)
    assert rows["tau"][1:].tolist() == [0.9, 0.5]
    assert rows["n_answered_a"].tolist() == [0, 2, 5]


def test_sweep_endpoints_are_the_reference_policies():
    human = _policy(threshold_opt.FAMILY_A_TO_HUMAN, P_MAX, CORRECT_A,
                    threshold_opt.human_arm(4))
    rows = _sweep(human)
    assert rows["cost_per_1k"][0] == pytest.approx(C_HUMAN * 1000)      # all_human
    assert rows["cost_per_1k"][-1] == pytest.approx(C_MISROUTE * 1000 * 0.5)  # a_only
    tier_c = _policy(threshold_opt.FAMILY_A_TO_C, P_MAX, CORRECT_A, _tier_c_fixture_arm())
    rows_c = _sweep(tier_c)
    assert rows_c["cost_per_1k"][-1] == pytest.approx(rows["cost_per_1k"][-1])  # same a_only


def test_empty_policy_is_rejected():
    empty = _policy(threshold_opt.FAMILY_A_TO_HUMAN, [], [], threshold_opt.human_arm(0))
    with pytest.raises(ValueError, match="empty policy"):
        threshold_opt.build_grid(empty)


# ---------------------------------------------------------------------------
# Alignment + gate hard failures
# ---------------------------------------------------------------------------

def test_subset_id_alignment_hard_fails_on_missing_id(tmp_path):
    labels = ["a", "b"]
    _write_artifact(tmp_path, "aa" * 32, [1, 2, 3], ["a", "b", "a"], ["a", "b", "a"],
                    labels)
    art_a = predictions.read_artifact(tmp_path / f"{'aa' * 32}.parquet")
    # exact subset in the requested order
    idx = threshold_opt.restrict_to_ids(art_a, [3, 1])
    assert idx.tolist() == [2, 0]
    with pytest.raises(ValueError, match="absent from the Tier A artifact"):
        threshold_opt.restrict_to_ids(art_a, [1, 99])


def test_paired_build_hard_fails_when_y_true_disagrees(tmp_path):
    labels = ["a", "b"]
    _write_artifact(tmp_path, "aa" * 32, [1, 2], ["a", "b"], ["a", "b"], labels)
    _write_artifact(tmp_path, "cc" * 32, [1, 2], ["b", "b"], ["b", "b"], labels)
    art_a = predictions.read_artifact(tmp_path / f"{'aa' * 32}.parquet")
    art_c = predictions.read_artifact(tmp_path / f"{'cc' * 32}.parquet")
    rec_a = _record("aa" * 32, "tier_a_x")
    rec_c = _record("cc" * 32, "tier_c_x")
    with pytest.raises(ValueError, match="disagree on y_true"):
        threshold_opt.build_a_to_c(art_a, rec_a, "tier_a_x", art_c, rec_c, "tier_c_x",
                                   api_cost_usd=[0.001, 0.001],
                                   parse_failed=[False, False], cost_sum_check={})


def test_receipt_gate_corruption_propagates_through_this_module(tmp_path):
    # One corrupted receipt (a slug from a different model) must fail the whole build:
    # the same cost_model gate, reached through the threshold optimizer.
    repo = _mini_repo(tmp_path, receipt_overrides={100: {"slug": "other/model"}})
    cfg = _cost_config(tmp_path)
    with pytest.raises(ValueError, match="slug is not the run's model"):
        threshold_opt.build_all(cfg, preds_dir=tmp_path, results_path=repo["results"],
                                tier_a_configs=(threshold_opt.PRIMARY_TIER_A_CONFIG,))


def test_missing_parse_failed_flag_is_a_hard_failure(tmp_path):
    repo = _mini_repo(tmp_path)
    lines = [json.loads(x) for x in repo["log"].read_text().splitlines() if x.strip()]
    for line in lines:
        line.pop("parse_failed")
    _write_receipts(repo["log"], lines)
    cfg = _cost_config(tmp_path)
    with pytest.raises(ValueError, match="missing/non-boolean parse_failed"):
        threshold_opt.build_all(cfg, preds_dir=tmp_path, results_path=repo["results"],
                                tier_a_configs=(threshold_opt.PRIMARY_TIER_A_CONFIG,))


def test_non_cal_artifact_is_refused(tmp_path):
    _write_artifact(tmp_path, "aa" * 32, [1], ["a"], ["a"], ["a", "b"], split="test_iid")
    record = _record("aa" * 32, "tier_a_x", split="test_iid")
    with pytest.raises(ValueError, match="outside the allowed set"):
        threshold_opt.load_artifact_checked(record, tmp_path)


def test_provenance_gate_applies_to_consumed_artifacts(tmp_path):
    _write_artifact(tmp_path, "aa" * 32, [1], ["a"], ["a"], ["a", "b"],
                    split_sha256="wrong")
    record = _record("aa" * 32, "tier_a_x")
    with pytest.raises(ValueError, match="provenance mismatch"):
        threshold_opt.load_artifact_checked(record, tmp_path)


# ---------------------------------------------------------------------------
# Result objects: CIs only at operating points, sensitivity grid, determinism
# ---------------------------------------------------------------------------

def _built(tmp_path, **kwargs):
    repo = _mini_repo(tmp_path)
    cfg = _cost_config(tmp_path)
    results = threshold_opt.build_all(
        cfg, preds_dir=tmp_path, results_path=repo["results"],
        tier_a_configs=(threshold_opt.PRIMARY_TIER_A_CONFIG,),
        n_resamples=kwargs.pop("n_resamples", 25), **kwargs)
    return repo, cfg, results


def test_build_all_produces_both_families_on_the_right_datasets(tmp_path):
    _, _, results = _built(tmp_path)
    keys = [(r["policy_family"], r["dataset"], r["n_examples"]) for r in results]
    assert keys == [
        (threshold_opt.FAMILY_A_TO_HUMAN, threshold_opt.DATASET_FULL_CAL, 12),
        (threshold_opt.FAMILY_A_TO_HUMAN, threshold_opt.DATASET_PAIRED, 6),
        (threshold_opt.FAMILY_A_TO_C, threshold_opt.DATASET_PAIRED, 6),
    ]
    assert all(r["is_primary"] for r in results)
    assert results[2]["inputs"]["tier_c"]["n_parse_failed"] == 1
    assert results[2]["inputs"]["tier_c"]["cost_sum_check"]["ok"] is True


def _built_tier_b(tmp_path, **kwargs):
    repo = _mini_repo(tmp_path, tier_b=True)
    cfg = _cost_config(tmp_path, tier_b=True)
    results = threshold_opt.build_all(
        cfg, preds_dir=tmp_path, results_path=repo["results"],
        tier_a_configs=(threshold_opt.PRIMARY_TIER_A_CONFIG,),
        n_resamples=kwargs.pop("n_resamples", 25), **kwargs)
    return repo, cfg, results


def test_build_all_adds_the_tier_b_families_when_the_cost_config_prices_them(tmp_path):
    _, _, results = _built_tier_b(tmp_path)
    assert [(r["policy_family"], r["dataset"], r["n_examples"]) for r in results] == [
        (threshold_opt.FAMILY_A_TO_HUMAN, threshold_opt.DATASET_FULL_CAL, 12),
        (threshold_opt.FAMILY_A_TO_HUMAN, threshold_opt.DATASET_PAIRED, 6),
        (threshold_opt.FAMILY_A_TO_C, threshold_opt.DATASET_PAIRED, 6),
        (threshold_opt.FAMILY_A_TO_B, threshold_opt.DATASET_FULL_CAL, 12),
        (threshold_opt.FAMILY_A_TO_B, threshold_opt.DATASET_PAIRED, 6),
        (threshold_opt.FAMILY_A_TO_B_TO_C, threshold_opt.DATASET_PAIRED, 6),
    ]
    a_to_b = results[3]
    assert set(a_to_b["operating_points"]) == {"tau_star", "a_only", "b_only"}
    assert a_to_b["escalation_arm"] == "tier_b_terminal"
    assert a_to_b["inputs"]["tier_b"]["config_name"] == threshold_opt.TIER_B_CAL_CONFIG
    assert a_to_b["inputs"]["tier_b"]["per_example_usd"] == B_PER_EXAMPLE
    # a one-gate family must not grow a second gate
    assert "tier_b_gate" not in a_to_b


def test_joint_family_records_both_gates_and_the_realized_routing_mix(tmp_path):
    _, _, results = _built_tier_b(tmp_path)
    abc = results[5]
    assert set(abc["operating_points"]) == {"tau_star", "a_only", "b_to_c"}
    gate = abc["tier_b_gate"]
    assert gate["fit"] == "joint_2d_argmin"
    mix = gate["routing_mix_at_joint_operating_point"]
    # every complaint is accounted for exactly once across the four destinations
    assert (mix["answered_tier_a"] + mix["answered_tier_b"] + mix["sent_to_tier_c"]
            == abc["n_examples"])
    assert mix["to_human_parse_failed"] <= mix["sent_to_tier_c"]
    assert mix["answered_tier_a"] == abc["n_answered_at_tau_star"]
    assert gate["grid"]["rows"][gate["grid"]["argmin_index"]]["cost_per_1k"] == \
        pytest.approx(abc["operating_points"]["tau_star"]["cost_per_1k"])
    # the joint sensitivity re-fits BOTH gates in every cell, so tau_b is a cell field
    assert all("tau_b_star" in cell for cell in abc["sensitivity"]["cells"])
    assert abc["notes"]["joint_sweep"].startswith("a_to_b_to_c's")


def test_joint_operating_point_replays_through_the_row_by_row_cost_reference(tmp_path):
    """The published (tau_A, tau_B) must reproduce the published objective from scratch."""
    repo, cfg, results = _built_tier_b(tmp_path)
    abc = results[5]
    records = threshold_opt._records_by_config(repo["results"])
    art_a = threshold_opt.load_artifact_checked(
        records[threshold_opt.PRIMARY_TIER_A_CONFIG], tmp_path)
    art_b = threshold_opt.load_artifact_checked(
        records[threshold_opt.TIER_B_CAL_CONFIG], tmp_path)
    art_c = threshold_opt.load_artifact_checked(
        records[threshold_opt.TIER_C_CAL_CONFIG], tmp_path)
    api, parse_failed, check, _ = threshold_opt.load_tier_c_arm_inputs(
        art_c, records[threshold_opt.TIER_C_CAL_CONFIG])
    policy = threshold_opt.build_a_to_b_to_c(
        art_a, records[threshold_opt.PRIMARY_TIER_A_CONFIG],
        threshold_opt.PRIMARY_TIER_A_CONFIG, art_b,
        records[threshold_opt.TIER_B_CAL_CONFIG], threshold_opt.TIER_B_CAL_CONFIG,
        art_c, records[threshold_opt.TIER_C_CAL_CONFIG], threshold_opt.TIER_C_CAL_CONFIG,
        tau_b=abc["tier_b_gate"]["tau_b_star"], b_per_example_usd=B_PER_EXAMPLE,
        api_cost_usd=api, parse_failed=parse_failed, cost_sum_check=check)
    total = threshold_opt.cost_at(policy, abc["tau_star"], c_misroute=cfg.c_misroute_usd,
                                 c_human=cfg.c_human_usd)["total"]
    assert total == pytest.approx(
        abc["operating_points"]["tau_star"]["expected_cost_per_1k"]["total"]["point"],
        abs=1e-9)


def test_summary_gains_tier_b_columns_only_when_the_families_exist(tmp_path):
    _, cfg, results = _built_tier_b(tmp_path)
    summary = threshold_opt.build_summary(results, cfg)
    assert summary["tier_b_config"] == threshold_opt.TIER_B_CAL_CONFIG
    cells = summary["sensitivity_comparison_paired_subset"]["cells"]
    assert len(cells) == 36
    for cell in cells:
        assert {"cost_per_1k_a_to_b", "cost_per_1k_b_only", "cost_per_1k_a_to_b_to_c",
                "tau_b_star_a_to_b_to_c"} <= set(cell)
        # the winner is chosen over every family present, Tier B included
        assert cell["winner_family"] in {
            threshold_opt.FAMILY_A_TO_HUMAN, threshold_opt.FAMILY_A_TO_C,
            threshold_opt.FAMILY_A_TO_B, threshold_opt.FAMILY_A_TO_B_TO_C}

    # ...and a cost config without Tier B keeps exactly the columns it shipped with
    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    _, plain_cfg, plain_results = _built(plain_dir)
    plain = threshold_opt.build_summary(plain_results, plain_cfg)
    assert "tier_b_config" not in plain
    assert all("cost_per_1k_a_to_b" not in cell
               for cell in plain["sensitivity_comparison_paired_subset"]["cells"])


def test_tier_b_cal_run_missing_under_a_tier_b_cost_config_is_a_hard_failure(tmp_path):
    repo = _mini_repo(tmp_path, tier_b=False)
    cfg = _cost_config(tmp_path, tier_b=True)
    with pytest.raises(ValueError, match="no run record for config"):
        threshold_opt.build_all(
            cfg, preds_dir=tmp_path, results_path=repo["results"],
            tier_a_configs=(threshold_opt.PRIMARY_TIER_A_CONFIG,), n_resamples=5)


def test_tier_b_cli_is_byte_deterministic(tmp_path):
    repo = _mini_repo(tmp_path, tier_b=True)
    _cost_config(tmp_path, tier_b=True, name="cost_b.yaml")
    outs = []
    for name in ("out1", "out2"):
        out_dir = tmp_path / name
        assert threshold_opt.main([
            "--preds-dir", str(tmp_path), "--out-dir", str(out_dir),
            "--results", str(repo["results"]),
            "--cost-config", str(tmp_path / "cost_b.yaml"), "--max-points", "8",
            "--tier-a-config", threshold_opt.PRIMARY_TIER_A_CONFIG,
        ]) == 0
        outs.append(out_dir)
    files = sorted(p.name for p in outs[0].glob("*.json"))
    assert len(files) == 7  # 6 policy files + summary
    assert any(f.startswith(f"{threshold_opt.FAMILY_A_TO_B_TO_C}__") for f in files)
    for name in files:
        assert (outs[0] / name).read_bytes() == (outs[1] / name).read_bytes()


def test_cis_appear_only_at_operating_points(tmp_path):
    _, _, results = _built(tmp_path)
    for result in results:
        labels = set(result["operating_points"])
        expected = {"tau_star", "a_only",
                    "c_only" if result["policy_family"] == threshold_opt.FAMILY_A_TO_C
                    else "all_human"}
        assert labels == expected
        for point in result["operating_points"].values():
            bands = point["expected_cost_per_1k"]
            assert set(bands) == {"total", "misroute", "api", "human"}
            for band in bands.values():
                assert set(band) == {"point", "ci_lo", "ci_hi"}
                assert band["ci_lo"] <= band["point"] <= band["ci_hi"]
        for row in result["grid"]["rows"]:
            assert "expected_cost_per_1k" not in row
            assert set(row) == {
                "tau", "n_answered_a", "coverage_a", "escalation_rate", "human_rate",
                "accuracy_machine", "accuracy_system", "cost_per_1k", "misroute_per_1k",
                "api_per_1k", "human_per_1k",
            }
        # the no-gate row serializes tau as null (no finite threshold answers nothing)
        assert result["grid"]["rows"][0]["tau"] is None


def test_operating_point_costs_match_the_grid_points(tmp_path):
    _, _, results = _built(tmp_path)
    for result in results:
        rows = {r["coverage_a"]: r for r in result["grid"]["rows"]}
        for point in result["operating_points"].values():
            row = rows[point["coverage_a"]]
            assert point["expected_cost_per_1k"]["total"]["point"] == \
                pytest.approx(row["cost_per_1k"], abs=1e-9)


def test_paired_deltas_are_paired_and_self_comparison_is_exactly_zero(tmp_path):
    _, cfg, results = _built(tmp_path)
    for result in results:
        reference = ("c_only" if result["policy_family"] == threshold_opt.FAMILY_A_TO_C
                     else "all_human")
        assert set(result["paired_deltas"]) == {
            "tau_star_minus_a_only", f"tau_star_minus_{reference}"}
        star = result["operating_points"]["tau_star"]["expected_cost_per_1k"]["total"]
        for label, delta in result["paired_deltas"].items():
            ref_label = label.removeprefix("tau_star_minus_")
            ref = result["operating_points"][ref_label]["expected_cost_per_1k"]["total"]
            # the paired point estimate IS the difference of the two point costs
            assert delta["delta_cost_per_1k"]["point"] == pytest.approx(
                star["point"] - ref["point"], abs=1e-6)
            band = delta["delta_cost_per_1k"]
            assert band["ci_lo"] <= band["point"] <= band["ci_hi"]
            assert delta["excludes_zero"] == (band["ci_lo"] > 0 or band["ci_hi"] < 0)

    # A policy compared against itself must give an exactly-zero delta with a degenerate
    # band — the property that proves the resampling is paired and not two independent
    # draws (two independent draws would give a nonzero spread).
    policy = _policy(threshold_opt.FAMILY_A_TO_HUMAN, P_MAX, CORRECT_A,
                     threshold_opt.human_arm(4))
    same = threshold_opt.paired_delta(policy, 0.7, 0.7, cfg=cfg, n_resamples=20)
    assert same["delta_cost_per_1k"] == {"point": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    assert same["excludes_zero"] is False
    assert same["favors"] == "neither"
    # ...and a real difference reproduces the hand-computed gap (1250 - 3000 = -1750/1k)
    real = threshold_opt.paired_delta(policy, 0.7, 0.3, cfg=cfg, n_resamples=50)
    assert real["delta_cost_per_1k"]["point"] == pytest.approx(-1750.0)
    assert real["favors"] == "tau_star"


def test_cross_family_paired_delta_is_attached_and_id_aligned(tmp_path):
    _, _, results = _built(tmp_path)
    a_to_c = results[2]
    delta = a_to_c["cross_family_paired_delta"]
    assert delta["vs"].startswith(threshold_opt.FAMILY_A_TO_HUMAN)
    star_c = a_to_c["operating_points"]["tau_star"]["expected_cost_per_1k"]["total"]
    star_h = results[1]["operating_points"]["tau_star"]["expected_cost_per_1k"]["total"]
    assert delta["delta_cost_per_1k"]["point"] == pytest.approx(
        star_c["point"] - star_h["point"], abs=1e-6)
    assert delta["favors"] in {threshold_opt.FAMILY_A_TO_C,
                               threshold_opt.FAMILY_A_TO_HUMAN, "neither"}
    band = delta["delta_cost_per_1k"]
    assert band["ci_lo"] <= band["point"] <= band["ci_hi"]
    # only the a_to_c result carries it: it is the side making the claim
    assert "cross_family_paired_delta" not in results[0]
    assert "cross_family_paired_delta" not in results[1]


def test_cross_family_pairing_refuses_mismatched_ids():
    a = _policy(threshold_opt.FAMILY_A_TO_HUMAN, P_MAX, CORRECT_A,
                threshold_opt.human_arm(4))
    b = _policy(threshold_opt.FAMILY_A_TO_HUMAN, P_MAX[:3], CORRECT_A[:3],
                threshold_opt.human_arm(3))
    cfg = None
    with pytest.raises(ValueError, match="their ids differ"):
        threshold_opt.paired_delta_across_policies(a, 0.5, b, 0.5, cfg=cfg,
                                                   n_resamples=5)


def test_sensitivity_grid_contains_the_exact_defaults_cell(tmp_path):
    _, cfg, results = _built(tmp_path)
    for result in results:
        cells = result["sensitivity"]["cells"]
        assert len(cells) == len(threshold_opt.SENSITIVITY_C_MISROUTE) * \
            len(threshold_opt.SENSITIVITY_C_HUMAN)
        defaults = [c for c in cells if c["is_cost_config_default"]]
        assert len(defaults) == 1
        cell = defaults[0]
        assert (cell["c_misroute_usd"], cell["c_human_usd"]) == \
            (cfg.c_misroute_usd, cfg.c_human_usd) == (6.00, 2.50)
        # the defaults cell must agree with the headline tau*
        assert cell["tau_star"] == result["tau_star"]
        assert cell["cost_per_1k"] == pytest.approx(
            result["operating_points"]["tau_star"]["expected_cost_per_1k"]["total"]["point"],
            abs=1e-9)
        assert "estimated parameters, measured predictions" in \
            result["sensitivity"]["evidence_class_note"]


def test_grid_downsampling_keeps_endpoints_and_tau_star(tmp_path):
    repo = _mini_repo(tmp_path)
    cfg = _cost_config(tmp_path)
    results = threshold_opt.build_all(
        cfg, preds_dir=tmp_path, results_path=repo["results"],
        tier_a_configs=(threshold_opt.PRIMARY_TIER_A_CONFIG,), max_points=3, n_resamples=5)
    full = results[0]["grid"]["n_thresholds_full"]
    rows = results[0]["grid"]["rows"]
    assert full == 13  # 12 distinct p_max + the no-gate row
    assert len(rows) <= 5  # <=3 sampled plus the always-kept endpoints/tau*
    assert rows[0]["tau"] is None
    assert rows[-1]["coverage_a"] == 1.0
    assert any(r["tau"] == results[0]["tau_star"] for r in rows)


def test_notes_and_amendment_ride_in_every_output(tmp_path):
    _, _, results = _built(tmp_path)
    for result in results:
        notes = result["notes"]
        assert "TERMINAL" in notes["amendment"]
        assert "self-consistency" in notes["amendment"]
        assert "selection optimism" in notes["selection_optimism"].lower()
        assert "isotonic" in notes["p_max_space"]
        assert result["target_coverage_a"] is not None


def test_cli_is_byte_deterministic(tmp_path):
    repo = _mini_repo(tmp_path)
    cost_cfg = tmp_path / "cost.yaml"
    _cost_config(tmp_path)  # writes cost.yaml
    outs = []
    for name in ("out1", "out2"):
        out_dir = tmp_path / name
        assert threshold_opt.main([
            "--preds-dir", str(tmp_path), "--out-dir", str(out_dir),
            "--results", str(repo["results"]), "--cost-config", str(cost_cfg),
            "--max-points", "8",
            "--tier-a-config", threshold_opt.PRIMARY_TIER_A_CONFIG,
        ]) == 0
        outs.append(out_dir)
    files = sorted(p.name for p in outs[0].glob("*.json"))
    assert len(files) == 4  # 3 policy files + summary
    assert any(f.startswith("summary__cost-") for f in files)
    for name in files:
        assert (outs[0] / name).read_bytes() == (outs[1] / name).read_bytes()


def test_summary_compares_families_only_on_the_paired_subset(tmp_path):
    _, cfg, results = _built(tmp_path)
    summary = threshold_opt.build_summary(results, cfg)
    assert summary["primary_tier_a_config"] == threshold_opt.PRIMARY_TIER_A_CONFIG
    assert summary["n_parse_failed_tier_c_cal"] == 1
    cells = summary["sensitivity_comparison_paired_subset"]["cells"]
    assert len(cells) == 36
    assert sum(c["is_cost_config_default"] for c in cells) == 1
    for cell in cells:
        assert cell["winner_family"] in {threshold_opt.FAMILY_A_TO_HUMAN,
                                         threshold_opt.FAMILY_A_TO_C}
        best = min(cell["cost_per_1k_a_to_human"], cell["cost_per_1k_a_to_c"])
        assert cell["beats_a_only"] == (best < cell["cost_per_1k_a_only"])
    assert "paired" in summary["sensitivity_comparison_paired_subset"]["note"]


# ---------------------------------------------------------------------------
# Serialized-threshold reproducibility
# ---------------------------------------------------------------------------

def test_tau_star_is_serialized_exactly_not_rounded(tmp_path):
    # The published tau must be the float that was optimized. Rounding it to JSON_ROUND
    # would move rows across the inclusive `p_max >= tau` gate; this asserts the value
    # survives a real JSON text round-trip bit-for-bit.
    repo = _mini_repo(tmp_path)
    cfg = _cost_config(tmp_path)
    results = threshold_opt.build_all(
        cfg, preds_dir=tmp_path, results_path=repo["results"],
        tier_a_configs=(threshold_opt.PRIMARY_TIER_A_CONFIG,), n_resamples=5)
    for result in results:
        written = json.loads(json.dumps(result, sort_keys=True))
        tau = written["tau_star"]
        if tau is None:
            continue
        exact = [float(t) for t in repo["p_max"] if float(t) == tau]
        assert exact, "tau_star is not one of the artifact's actual p_max values"
        assert repr(tau) == repr(exact[0])  # full float64 precision survived the text


def _replay(policy, tau, cfg):
    """Answered count + cost/1k obtained by replaying a published threshold."""
    n_answered = int(np.count_nonzero(policy.p_max >= tau))
    cost = threshold_opt.cost_at(policy, tau, c_misroute=cfg.c_misroute_usd,
                                 c_human=cfg.c_human_usd)
    return n_answered, cost["total"]


def test_serialized_thresholds_replay_on_a_synthetic_repo(tmp_path):
    repo = _mini_repo(tmp_path)
    cfg = _cost_config(tmp_path)
    out_dir = tmp_path / "out"
    assert threshold_opt.main([
        "--preds-dir", str(tmp_path), "--out-dir", str(out_dir),
        "--results", str(repo["results"]), "--cost-config", str(tmp_path / "cost.yaml"),
        "--tier-a-config", threshold_opt.PRIMARY_TIER_A_CONFIG,
    ]) == 0
    records = threshold_opt._records_by_config(repo["results"])
    art_c = threshold_opt.load_artifact_checked(records[threshold_opt.TIER_C_CAL_CONFIG],
                                                tmp_path)
    art_a = threshold_opt.load_artifact_checked(records[threshold_opt.PRIMARY_TIER_A_CONFIG],
                                                tmp_path)
    for path in sorted(out_dir.glob("*.json")):
        if path.name.startswith("summary__"):
            continue
        obj = json.loads(path.read_text())
        policy = _rebuild_policy(obj, records, art_a, art_c, tmp_path)
        tau = np.inf if obj["tau_star"] is None else obj["tau_star"]
        n_answered, cost = _replay(policy, tau, cfg)
        assert n_answered == obj["n_answered_at_tau_star"], path.name
        star = obj["operating_points"]["tau_star"]["expected_cost_per_1k"]["total"]
        assert cost == pytest.approx(star["point"], abs=1e-9), path.name


def _rebuild_policy(obj, records, art_a, art_c, preds_dir, art_b=None):
    """Rebuild the exact PolicyData a written result describes (no bootstraps).

    Two-gate files carry their fitted tau_B in `tier_b_gate`, so the rebuild is pinned to
    the published pair — replaying a joint file at a re-derived tau_B would be testing the
    optimizer twice instead of testing the file.
    """
    family = obj["policy_family"]
    name_a = obj["inputs"]["tier_a"]["config_name"]
    record_a = records[name_a]
    paired_index = threshold_opt.restrict_to_ids(art_a, art_c.complaint_id)
    index = (None if obj["dataset"] == threshold_opt.DATASET_FULL_CAL else paired_index)

    if family == threshold_opt.FAMILY_A_TO_HUMAN:
        return threshold_opt.build_a_to_human(art_a, record_a, name_a,
                                              dataset=obj["dataset"], index=index)
    if family == threshold_opt.FAMILY_A_TO_B:
        name_b = obj["inputs"]["tier_b"]["config_name"]
        return threshold_opt.build_a_to_b(
            art_a, record_a, name_a, art_b, records[name_b], name_b,
            b_per_example_usd=obj["inputs"]["tier_b"]["per_example_usd"],
            dataset=obj["dataset"], index=index)

    name_c = obj["inputs"]["tier_c"]["config_name"]
    record_c = records[name_c]
    api, parse_failed, check, _ = threshold_opt.load_tier_c_arm_inputs(art_c, record_c)
    if family == threshold_opt.FAMILY_A_TO_C:
        return threshold_opt.build_a_to_c(
            art_a, record_a, name_a, art_c, record_c, name_c, api_cost_usd=api,
            parse_failed=parse_failed, cost_sum_check=check)
    name_b = obj["inputs"]["tier_b"]["config_name"]
    gate_b = obj["tier_b_gate"]
    return threshold_opt.build_a_to_b_to_c(
        art_a, record_a, name_a, art_b, records[name_b], name_b, art_c, record_c, name_c,
        tau_b=(np.inf if gate_b["tau_b_star"] is None else gate_b["tau_b_star"]),
        b_per_example_usd=obj["inputs"]["tier_b"]["per_example_usd"],
        api_cost_usd=api, parse_failed=parse_failed, cost_sum_check=check)


@_needs_real
def test_shipped_threshold_files_replay_exactly():
    """Regression over the COMMITTED results/thresholds/ files.

    Every published threshold — tau* and each reference operating point — is replayed
    against the artifact it was computed from and must reproduce the answered count and
    the cost it claims. This is the check that would have caught the rounded-tau bug: the
    file said 1,279 answers / $960.33 while its own published tau replayed to 1,278 /
    $962.00.
    """
    records = threshold_opt._records_by_config()
    art_c = threshold_opt.load_artifact_checked(records[threshold_opt.TIER_C_CAL_CONFIG])
    arts_a: dict = {}   # resolved per file, so v1 and v2 rungs are both covered
    cfgs: dict = {}     # ...and per COST generation, so a v2-cost file is not replayed
    art_b = None        # at v1 prices

    for path in _REAL_POLICY_FILES:
        obj = json.loads(path.read_text())
        cost_path = obj["cost_config"]["path"]
        if cost_path not in cfgs:
            cfgs[cost_path] = cost_model.load_cost_config(
                threshold_opt.REPO_ROOT / cost_path)
            assert cfgs[cost_path].sha256 == obj["cost_config"]["sha256"], path.name
        cfg = cfgs[cost_path]
        name_a = obj["inputs"]["tier_a"]["config_name"]
        if name_a not in arts_a:
            arts_a[name_a] = threshold_opt.load_artifact_checked(records[name_a])
        art_a = arts_a[name_a]
        if "tier_b" in obj["inputs"] and art_b is None:
            art_b = threshold_opt.load_artifact_checked(
                records[obj["inputs"]["tier_b"]["config_name"]])
        policy = _rebuild_policy(obj, records, art_a, art_c,
                                 threshold_opt.DEFAULT_PREDS_DIR, art_b=art_b)
        assert len(policy) == obj["n_examples"], path.name

        tau = np.inf if obj["tau_star"] is None else obj["tau_star"]
        n_answered, cost = _replay(policy, tau, cfg)
        assert n_answered == obj["n_answered_at_tau_star"], path.name
        star = obj["operating_points"]["tau_star"]["expected_cost_per_1k"]["total"]
        assert cost == pytest.approx(star["point"], abs=1e-9), path.name

        # every published operating point, not just tau*
        for label, point in obj["operating_points"].items():
            tau = np.inf if point["tau"] is None else point["tau"]
            n_answered, cost = _replay(policy, tau, cfg)
            assert n_answered == point["n_answered_a"], f"{path.name}:{label}"
            assert cost == pytest.approx(
                point["expected_cost_per_1k"]["total"]["point"], abs=1e-9
            ), f"{path.name}:{label}"


@_needs_real
def test_shipped_files_carry_receipt_and_artifact_provenance():
    for path in _REAL_POLICY_FILES:
        obj = json.loads(path.read_text())
        assert obj["inputs"]["tier_a"]["config_sha256"]
        assert obj["inputs"]["tier_a"]["split_sha256"]
        if obj["policy_family"] == threshold_opt.FAMILY_A_TO_C:
            tier_c = obj["inputs"]["tier_c"]
            expected = cost_model.receipts_sha256(tier_c["raw_log_path"])
            assert tier_c["receipts_sha256"] == expected
            assert len(tier_c["receipts_sha256"]) == 64


# ---------------------------------------------------------------------------
# Blind spots: a_to_human equivalence, NaN p_max, duplicate ids
# ---------------------------------------------------------------------------

def test_fast_sweep_matches_the_reference_for_a_to_human():
    rng = np.random.default_rng(21)
    n = 173
    p_max = rng.random(n).round(3)  # rounding forces ties into the grid
    policy = _policy(threshold_opt.FAMILY_A_TO_HUMAN, p_max, rng.random(n) < 0.55,
                     threshold_opt.human_arm(n))
    rows = _sweep(policy)
    for j, tau in enumerate(rows["tau"]):
        ref = threshold_opt.cost_at(policy, tau, c_misroute=C_MISROUTE, c_human=C_HUMAN)
        assert rows["cost_per_1k"][j] == pytest.approx(ref["total"], abs=1e-9), j
        assert rows["misroute_per_1k"][j] == pytest.approx(ref["misroute"], abs=1e-9), j
        assert rows["human_per_1k"][j] == pytest.approx(ref["human"], abs=1e-9), j
        assert rows["api_per_1k"][j] == 0.0


def test_non_finite_p_max_is_a_hard_failure():
    # NaN >= tau is False at EVERY threshold, so a NaN row would silently escalate always
    # and never appear as a gate. Refuse instead.
    for bad in (np.nan, np.inf):
        policy = _policy(threshold_opt.FAMILY_A_TO_HUMAN, [0.9, bad, 0.5],
                         [True, True, False], threshold_opt.human_arm(3))
        with pytest.raises(ValueError, match="non-finite p_max"):
            threshold_opt.build_grid(policy)


def test_nan_probability_artifact_fails_the_verification_gate(tmp_path):
    repo = _mini_repo(tmp_path)
    labels = repo["labels"]
    ids, y_true = repo["ids"], repo["y_true"]
    probs = np.full((len(ids), len(labels)), 0.5)
    probs[3, 0] = np.nan
    records = threshold_opt._records_by_config(repo["results"])
    record = records[threshold_opt.PRIMARY_TIER_A_CONFIG]
    prov = predictions.ArtifactProvenance(
        run_id=record["run_id"], config_sha256=record["config_sha256"], split="cal",
        split_sha256="splithash", class_labels=labels,
        git_sha="gitsha", input_sha256="inputsha",
    )
    predictions.write_artifact(
        tmp_path / f"{record['run_id']}.parquet", ids=np.asarray(ids, dtype=np.int64),
        y_true=y_true, y_pred=repo["y_pred_a"], probs=probs, class_labels=labels,
        provenance=prov,
    )
    with pytest.raises(ValueError, match="verification gate"):
        threshold_opt.load_artifact_checked(record, tmp_path)


def test_duplicate_complaint_id_artifact_fails_through_this_module(tmp_path):
    ids = list(range(100, 112))
    dup = list(ids)
    dup[5] = dup[4]  # one id repeated
    repo = _mini_repo(tmp_path, artifact_ids=dup)
    records = threshold_opt._records_by_config(repo["results"])
    with pytest.raises(ValueError, match="ids_unique_nonnull|verification gate"):
        threshold_opt.load_artifact_checked(
            records[threshold_opt.PRIMARY_TIER_A_CONFIG], tmp_path)


def test_cli_writes_nothing_when_a_later_rung_fails(tmp_path):
    # Batch atomicity: the primary rung scores fine, the second rung's artifact has a
    # duplicate complaint_id, so the whole publish must abort with an empty out-dir.
    repo = _mini_repo(tmp_path, second_rung="tier_a_logreg_word_cal")
    _cost_config(tmp_path)
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="verification gate"):
        threshold_opt.main([
            "--preds-dir", str(tmp_path), "--out-dir", str(out_dir),
            "--results", str(repo["results"]),
            "--cost-config", str(tmp_path / "cost.yaml"),
            "--tier-a-config", threshold_opt.PRIMARY_TIER_A_CONFIG,
            "--tier-a-config", "tier_a_logreg_word_cal",
        ])
    assert not out_dir.exists() or not list(out_dir.glob("*.json"))


# ---------------------------------------------------------------------------
# v1 / v2 derivations
# ---------------------------------------------------------------------------

def test_v1_results_carry_no_derivation_fields(tmp_path):
    """v1 files are never rewritten, so their schema must not gain keys.

    The derivation block is emitted only for v2; a v1 regeneration has to stay
    byte-identical to what is committed (a separate real-data diff asserts that).
    """
    _, _, results = _built(tmp_path)
    for result in results:
        assert "derivation" not in result
        assert "is_primary_v2" not in result
        assert "derivation_note" not in result


def test_v2_results_are_marked_and_bound_to_the_isocal_rung(tmp_path):
    repo = _mini_repo(tmp_path, tier_a_config=threshold_opt.V2_TIER_A_CAL_CONFIG)
    cfg = _cost_config(tmp_path)
    results = threshold_opt.build_all(
        cfg, preds_dir=tmp_path, results_path=repo["results"],
        derivation=threshold_opt.DERIVATION_V2, n_resamples=10)
    assert len(results) == 3
    for result in results:
        assert result["derivation"] == threshold_opt.DERIVATION_V2
        assert result["is_primary_v2"] is True
        assert result["is_primary"] is True
        assert "in-sample" in result["derivation_note"]
        assert result["inputs"]["tier_a"]["config_name"] == \
            threshold_opt.V2_TIER_A_CAL_CONFIG
    summary = threshold_opt.build_summary(results, cfg,
                                          derivation=threshold_opt.DERIVATION_V2)
    assert summary["derivation"] == threshold_opt.DERIVATION_V2
    assert summary["primary_tier_a_config"] == threshold_opt.V2_TIER_A_CAL_CONFIG
    assert summary["tier_a_configs_swept"] == [threshold_opt.V2_TIER_A_CAL_CONFIG]


def test_derivation_profiles_have_distinct_summary_names():
    names = {d: threshold_opt.DERIVATIONS[d]["summary_name"]
             for d in threshold_opt.DERIVATIONS}
    assert len(set(names.values())) == len(names)
    assert "v2-isocal" in names[threshold_opt.DERIVATION_V2]


@_needs_real
def test_v1_thresholds_regenerate_byte_identically(tmp_path):
    """The committed v1 evidence must come back byte-for-byte from today's code.

    Not a normalized-JSON comparison and not a claim in a report: the CLI is re-run into a
    scratch directory against the real inputs and the resulting FILES are compared byte
    for byte with what is committed. v1 is the documented calibration-mismatch lesson, so
    it has to stay reproducible while v2 evolves around it.
    """
    out_dir = tmp_path / "thresholds"
    assert threshold_opt.main(["--out-dir", str(out_dir),
                               "--derivation", threshold_opt.DERIVATION_V1]) == 0
    regenerated = sorted(out_dir.glob("*.json"))
    assert regenerated, "no v1 threshold files were produced"
    for path in regenerated:
        committed = threshold_opt.DEFAULT_THRESHOLDS_DIR / path.name
        assert committed.exists(), f"{path.name} is not committed"
        assert path.read_bytes() == committed.read_bytes(), path.name
    # ...and the v2 files are untouched by a v1 run
    assert not any(p.name.endswith("__v2-isocal__cost-f76ad15a.json")
                   for p in regenerated)
