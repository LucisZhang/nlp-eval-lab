"""`make reproduce-headline` driver tests.

All of these run on the COMMITTED tree alone — `results/`, `configs/` and `demo/data/` are
in git, `data/` is not — so they are exactly the checks a CI smoke job can make on a bare
clone: the chain's run set resolves from the committed derivation artifacts, the gated file
list is the committed one, the masking rule masks one named field and nothing else, and the
preflight failure paths say what to do.

The expensive half (re-deriving artifacts, re-running the derivation) is deliberately not
exercised here; it is the target's own job and takes hours.
"""

from __future__ import annotations

import json

import pytest

from triage_lab import cost_model, demo_build, harness, predictions, reproduce_headline

# The chain, as the committed artifacts name it. Written out so a silent change to the run
# set (a rung added, a config renamed) fails a test instead of quietly re-deriving a
# different chain — the list is asserted against the artifacts, never used in place of them.
EXPECTED_CHAIN_CONFIGS = {
    "tier_a_cnb_test_iid",
    "tier_a_logreg_test_iid",
    "tier_a_logreg_wordchar_cal",
    "tier_a_logreg_wordchar_isocal_cal",
    "tier_b1_modernbert_sa",
    "tier_b1_modernbert_sb",
    "tier_b1_modernbert_sc",
    "tier_b2_distilbert_s0",
    "tier_b2_distilbert_s0_cal",
    "tier_c_haiku_ablation_zeroshot_cal",
    "tier_c_haiku_zeroshot_test_iid",
}


@pytest.fixture(scope="module")
def plan():
    return reproduce_headline.build_plan()


# ---------------------------------------------------------------------------
# Chain resolution
# ---------------------------------------------------------------------------

def test_chain_runs_resolve_from_the_committed_derivation_artifacts(plan):
    assert {r.config_name for r in plan.runs} == EXPECTED_CHAIN_CONFIGS
    records = predictions.records_by_config()
    for run in plan.runs:
        assert records[run.config_name]["run_id"] == run.run_id
        assert run.split == (records[run.config_name].get("dataset") or {})["split"]


def test_every_chain_run_but_the_demo_exhibit_is_named_by_a_committed_artifact(plan):
    """The resolution is evidence-driven: only `DEMO_ONLY_CONFIGS` is a code-side addition."""
    referenced = reproduce_headline.referenced_runs(
        reproduce_headline.chain_artifact_paths(plan.cfg))
    from_artifacts = {r.run_id for r in plan.runs if r.source == "chain-artifact"}
    assert from_artifacts == set(referenced)
    extras = {r.config_name for r in plan.runs if r.source == "demo-payload"}
    assert extras == set(reproduce_headline.DEMO_ONLY_CONFIGS)
    assert extras == {demo_build.TIER_A_RAW_CAL}


def test_chain_module_constants_agree_with_the_artifacts(plan):
    """`resolve_chain_runs` hard-fails on divergence; this pins the agreeing state."""
    assert reproduce_headline.chain_module_configs() == EXPECTED_CHAIN_CONFIGS
    assert {r.config_name for r in plan.runs} == reproduce_headline.chain_module_configs()


def test_referenced_runs_refuses_two_names_for_one_run(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(
        {"inputs": {"tier_a": {"run_id": "abc", "config_name": "one", "split": "cal"}}}))
    (tmp_path / "b.json").write_text(json.dumps(
        {"inputs": {"tier_a": {"run_id": "abc", "config_name": "two", "split": "cal"}}}))
    with pytest.raises(ValueError, match="named 'one' and 'two'"):
        reproduce_headline.referenced_runs(sorted(tmp_path.glob("*.json")))


def test_resolve_fails_loudly_when_a_committed_chain_file_is_missing(plan, tmp_path):
    with pytest.raises(ValueError, match="committed derivation file"):
        reproduce_headline.resolve_chain_runs(plan.cfg, frontier_dir=tmp_path)


# ---------------------------------------------------------------------------
# Gated files
# ---------------------------------------------------------------------------

def test_gated_files_are_committed_and_cover_the_whole_chain(plan):
    rels = [reproduce_headline._rel(p) for p in plan.gated]
    assert len(rels) == len(set(rels))
    for path in plan.gated:
        assert path.exists(), f"{path} is gated but not committed"
    assert "docs/DATASHEET.md" in rels
    assert f"results/frontier/frontier__opv2__cost-{plan.cfg.sha256[:8]}.json" in rels
    demo_files = {r for r in rels if r.startswith("demo/data/")}
    assert demo_files == {reproduce_headline._rel(p)
                          for p in demo_build.DEFAULT_OUT_DIR.glob("*.json")}
    assert len(demo_files) == 10
    # Nothing from the superseded v1 cost generation is rewritten by this command.
    v1 = cost_model.load_cost_config(cost_model.DEFAULT_COST_CONFIG)
    assert not [r for r in rels if v1.sha256[:8] in r]


def test_tier_b_cost_files_are_the_chain_runs_not_the_config_prefix_sweep(plan):
    """Why the driver passes run ids: `--config-prefix tier_b` now over-selects.

    Phase 5 added `tier_b2_distilbert_s0_test_drift_*` runs. They start with `tier_b`, so
    the prefix selection in `make tier-b-frontier` would price them too and write
    `results/cost_model/` files that HEAD does not carry (and would hard-fail on a clone
    with no artifact for them). The headline command must reproduce the committed set.
    """
    chain_ids = set(reproduce_headline.tier_b_run_ids(plan.runs))
    assert len(chain_ids) == 5
    prefix_ids = {r["run_id"] for r in predictions.load_records()
                  if reproduce_headline._rel(r.get("config_path", "")).split("/")[-1]
                  .startswith("tier_b")}
    assert chain_ids < prefix_ids, "the prefix no longer over-selects; revisit the driver"
    for run_id in chain_ids:
        assert (cost_model.DEFAULT_COST_DIR / f"{run_id}.json").exists()
    for extra in prefix_ids - chain_ids:
        assert not (cost_model.DEFAULT_COST_DIR / f"{extra}.json").exists(), (
            "a non-chain Tier B run now has a committed cost_model file; it must be either "
            "in the chain or out of the gate, not silently both")


# ---------------------------------------------------------------------------
# Hash gate + masking
# ---------------------------------------------------------------------------

def test_masked_field_changes_the_compare_digest_for_nothing_else(tmp_path, monkeypatch):
    path = tmp_path / "meta.json"
    monkeypatch.setitem(reproduce_headline.MASKED_FIELDS, str(path), ("git_sha",))
    path.write_text(json.dumps({"git_sha": "aaa", "snapshot_sha256": "s"}))
    raw_a, cmp_a = reproduce_headline.file_digests(path)
    path.write_text(json.dumps({"git_sha": "bbb", "snapshot_sha256": "s"}))
    raw_b, cmp_b = reproduce_headline.file_digests(path)
    assert raw_a != raw_b and cmp_a == cmp_b
    path.write_text(json.dumps({"git_sha": "bbb", "snapshot_sha256": "MOVED"}))
    _, cmp_c = reproduce_headline.file_digests(path)
    assert cmp_c != cmp_b


def test_file_digests_refuse_a_stale_mask(tmp_path, monkeypatch):
    path = tmp_path / "meta.json"
    monkeypatch.setitem(reproduce_headline.MASKED_FIELDS, str(path), ("git_sha",))
    path.write_text(json.dumps({"snapshot_sha256": "s"}))
    with pytest.raises(ValueError, match="no field 'git_sha' to mask"):
        reproduce_headline.file_digests(path)


def test_unmasked_files_compare_on_raw_bytes(tmp_path):
    path = tmp_path / "frontier.json"
    path.write_text(json.dumps({"git_sha": "aaa"}))
    raw, compare = reproduce_headline.file_digests(path)
    assert raw == compare


def test_git_sha_is_the_only_meta_field_that_moves_between_builds(monkeypatch):
    """Proves the mask cannot hide a number: everything else in meta is input-determined."""
    records = predictions.load_records()
    cfg = cost_model.load_cost_config(demo_build.DEMO_COST_CONFIG)
    monkeypatch.setattr(harness, "_git_sha", lambda: "sha-one")
    first = demo_build.build_meta(records, cfg)
    monkeypatch.setattr(harness, "_git_sha", lambda: "sha-two")
    second = demo_build.build_meta(records, cfg)
    differing = {k for k in first if first[k] != second[k]}
    assert differing == {"git_sha"}
    assert set(reproduce_headline.MASKED_FIELDS) == {"demo/data/meta.json"}
    assert reproduce_headline.MASKED_FIELDS["demo/data/meta.json"] == ("git_sha",)


def test_compare_state_reports_change_loss_and_unexpected_output():
    before = {"files": {"a.json": ("r1", "c1"), "b.json": ("r2", "c2")},
              "listings": {"d": ["a.json", "b.json"]}}
    after = {"files": {"a.json": ("r9", "c9"), "b.json": None},
             "listings": {"d": ["a.json", "new.json"]}}
    rows = reproduce_headline.compare_state(before, after)
    assert {(r["path"], r["kind"]) for r in rows} == {
        ("a.json", "changed"), ("b.json", "missing"),
        ("new.json", "unexpected-new"), ("b.json", "deleted"),
    }
    assert reproduce_headline.compare_state(before, before) == []


def test_masked_only_changes_names_the_file_that_moved_under_the_mask():
    before = {"files": {"m.json": ("r1", "c1")}, "listings": {}}
    after = {"files": {"m.json": ("r2", "c1")}, "listings": {}}
    assert reproduce_headline.masked_only_changes(before, after) == ["m.json"]
    assert reproduce_headline.compare_state(before, after) == []


# ---------------------------------------------------------------------------
# Stage wiring
# ---------------------------------------------------------------------------

def test_stage_commands_force_the_artifacts_and_pin_the_v2_derivation(plan):
    stages = dict(reproduce_headline.stage_commands(plan))
    assert list(stages) == ["data", "preds", "cost-model", "thresholds", "router-sim",
                            "frontier", "demo-data"]
    assert stages["data"][-1] == "data"

    preds = stages["preds"]
    assert "--force" in preds
    assert set(plan.run_ids) == set(preds[preds.index("--force") + 1:])

    cost = stages["cost-model"]
    assert "--config-prefix" not in cost
    assert set(reproduce_headline.tier_b_run_ids(plan.runs)) <= set(cost)

    for name in ("cost-model", "thresholds", "router-sim", "frontier"):
        assert str(plan.cfg.path) in stages[name]
    assert reproduce_headline.DERIVATION in stages["thresholds"]
    for name in ("router-sim", "frontier"):
        assert reproduce_headline.OP_VERSION in stages[name]
    # Every chain CLI runs in THIS interpreter, so the Makefile's `--extra tierb` carries.
    for name, argv in stages.items():
        if name != "data":
            assert argv[0].endswith("python") or "python" in argv[0]


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def test_committed_inputs_pass_on_this_tree(plan):
    rows = reproduce_headline.check_committed_inputs(plan.cfg, plan.runs, plan.gated)
    assert all(row["ok"] for row in rows), [r for r in rows if not r["ok"]]
    assert {r["check"] for r in rows} == {
        "results_log", "cost_config", "gated_files_present", "run_records",
        "config_hashes", "tier_c_receipts"}


def test_committed_inputs_flag_a_missing_gated_file(plan, tmp_path):
    rows = reproduce_headline.check_committed_inputs(
        plan.cfg, plan.runs, [*plan.gated, tmp_path / "gone.json"])
    row = next(r for r in rows if r["check"] == "gated_files_present")
    assert not row["ok"] and "gone.json" in row["detail"]


def test_local_prerequisites_name_the_missing_extra_and_the_missing_checkpoints(
        plan, tmp_path, monkeypatch):
    """The two fresh-clone blockers, both failing in seconds instead of hours in."""
    monkeypatch.setattr(reproduce_headline, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(reproduce_headline.importlib.util, "find_spec",
                        lambda name: None if name in {"torch", "transformers"} else object())
    rows = {r["check"]: r for r in reproduce_headline.check_local_prerequisites(plan.runs)}
    assert not rows["tierb_extra"]["ok"]
    assert "--extra tierb" in rows["tierb_extra"]["detail"]
    assert not rows["tier_b_checkpoints"]["ok"]
    detail = rows["tier_b_checkpoints"]["detail"]
    assert "TIER_B_RUNBOOK" in detail and "data/checkpoints/tier_b1_sa" in detail


# ---------------------------------------------------------------------------
# Split byte-identity gate
# ---------------------------------------------------------------------------

def _stats_yaml(path, splits: dict, input_sha256: str) -> None:
    body = [f'input_sha256: "{input_sha256}"', "splits:"]
    for name, sha in splits.items():
        body += [f"  {name}:", f'    sha256: "{sha}"']
    path.write_text("\n".join(body) + "\n")


def _chain_split_shas(plan) -> tuple[dict, str]:
    records = {r["run_id"]: r for r in predictions.load_records()}
    splits, inputs = {}, set()
    for run in plan.runs:
        dataset = records[run.run_id]["dataset"]
        splits[dataset["split"]] = dataset["split_sha256"]
        inputs.add(dataset["input_sha256"])
    assert len(inputs) == 1
    return splits, inputs.pop()


def test_split_identity_passes_when_the_splits_match_the_frozen_records(plan, tmp_path):
    splits, input_sha = _chain_split_shas(plan)
    stats = tmp_path / "splits_stats.yaml"
    _stats_yaml(stats, splits, input_sha)
    rows = reproduce_headline.check_split_identity(plan.runs, splits_stats_path=stats)
    assert {r["check"] for r in rows} == {f"split::{s}" for s in splits}
    assert all(r["ok"] for r in rows)


def test_split_identity_catches_a_split_that_moved(plan, tmp_path):
    splits, input_sha = _chain_split_shas(plan)
    moved = dict(splits)
    moved["cal"] = "0" * 64
    stats = tmp_path / "splits_stats.yaml"
    _stats_yaml(stats, moved, input_sha)
    rows = reproduce_headline.check_split_identity(plan.runs, splits_stats_path=stats)
    bad = [r for r in rows if not r["ok"]]
    assert [r["check"] for r in bad] == ["split::cal"]
    assert "000000000000" in bad[0]["detail"]


def test_split_identity_catches_a_different_snapshot(plan, tmp_path):
    splits, _ = _chain_split_shas(plan)
    stats = tmp_path / "splits_stats.yaml"
    _stats_yaml(stats, splits, "f" * 64)
    rows = reproduce_headline.check_split_identity(plan.runs, splits_stats_path=stats)
    assert not any(r["ok"] for r in rows)


def test_split_identity_fails_when_make_data_produced_nothing(plan, tmp_path):
    rows = reproduce_headline.check_split_identity(
        plan.runs, splits_stats_path=tmp_path / "absent.yaml")
    assert rows == [rows[0]] and not rows[0]["ok"]
    assert "make data" in rows[0]["detail"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_plan_mode_is_offline_and_green_on_a_bare_clone(capsys):
    assert reproduce_headline.main(["--plan"]) == 0
    out = capsys.readouterr().out
    assert "preflight OK (no stage was run)" in out
    assert "frontier__opv2__cost-" in out
    for config in sorted(EXPECTED_CHAIN_CONFIGS):
        assert config in out
