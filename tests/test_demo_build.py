"""Demo payload tests: traceability, contract shape, receipt consistency, freeze, determinism.

The demo is the one artifact a reader meets without the repo around it, so these tests are
mostly *traceability* tests rather than arithmetic tests: every run id under `demo/data/`
must exist in the append-only log, and every number the payload claims to have COPIED must
equal its source byte-for-byte. A demo that quietly recomputes a headline is a second
source of truth, and the one that disagrees is always the demo.

Three tiers of test, by what they need on disk:

- **committed-only** (always run in CI): traceability, shape, pending slots, the receipts
  drawer vs `results/tier_c_raw/**/calls.jsonl`, and the curated set's pool freeze. These
  are the tests that matter for a clone with no `data/`.
- **needs `data/`** (skipped otherwise): anything replaying a per-example artifact or a
  split — the full curated-selection replay, the calibration bins, and the two-build
  determinism check.
- **pure unit** (always): run resolution and the stratified allocator, on synthetic input.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from triage_lab import cost_model, demo_build, harness, predictions, router_sim

DEMO_DIR = demo_build.DEFAULT_OUT_DIR
RESULTS_PATH = harness.DEFAULT_RESULTS_PATH

CONTRACT_FILES = (
    "meta.json",
    "runs_index.json",
    "frontier.json",
    "policies.json",
    "drift.json",
    "calibration.json",
    "samples.json",
    "curated_ids.json",
    "receipts.json",
    "case_study.json",
)

_HAS_DEMO = DEMO_DIR.exists() and all((DEMO_DIR / n).exists() for n in CONTRACT_FILES)
_HAS_DATA = (demo_build.DEFAULT_PREDS_DIR.exists()
             and (demo_build.DEFAULT_SPLITS_DIR / "test_iid.parquet").exists())

_needs_demo = pytest.mark.skipif(not _HAS_DEMO, reason="demo/data payload not built")
_needs_data = pytest.mark.skipif(
    not _HAS_DATA, reason="gitignored data/preds + data/splits not present")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _load(name: str) -> dict:
    return json.loads((DEMO_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def payload() -> dict:
    return {name: _load(name) for name in CONTRACT_FILES}


@pytest.fixture(scope="module")
def records() -> list[dict]:
    return predictions.load_records(RESULTS_PATH)


@pytest.fixture(scope="module")
def resolved(records) -> dict:
    return demo_build.resolve_records(records)


def _walk(obj, path=()):
    """Every (key-path, value) pair in a nested JSON structure."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _walk(value, (*path, key))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from _walk(value, (*path, i))
    else:
        yield path, obj


def _is_run_id_key(key) -> bool:
    """Key names that carry a run id, excluding the sha256 fields that look like one."""
    if not isinstance(key, str):
        return False
    return key.endswith("run_id") or key in {"run_ids", "run_refs"}


def _collect_run_ids(obj) -> set[str]:
    out: set[str] = set()
    for path, value in _walk(obj, ()):
        if not isinstance(value, str) or len(value) != 64:
            continue
        # `run_refs`/`run_ids` are lists, so the id sits one level under the key.
        for key in reversed(path):
            if isinstance(key, str):
                if _is_run_id_key(key):
                    out.add(value)
                break
    return out


def _receipt_lines(raw_log_path) -> list[dict]:
    path = demo_build.REPO_ROOT / raw_log_path
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _receipts_by_id(raw_log_path) -> dict[int, dict]:
    return {int(r["complaint_id"]): r for r in _receipt_lines(raw_log_path)}


# ---------------------------------------------------------------------------
# Contract shape
# ---------------------------------------------------------------------------

@_needs_demo
def test_all_ten_contract_files_exist():
    for name in CONTRACT_FILES:
        assert (DEMO_DIR / name).is_file(), f"missing contract file {name}"


@_needs_demo
def test_files_are_deterministically_serialized(payload):
    """Sorted keys, 2-space indent, ensure_ascii=False, trailing newline — no exceptions."""
    for name, obj in payload.items():
        text = (DEMO_DIR / name).read_text(encoding="utf-8")
        expected = json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
        assert text == expected, f"{name} is not in the contract's canonical serialization"


@_needs_demo
def test_meta_shape(payload, records):
    meta = payload["meta.json"]
    assert meta["schema_version"] == "demo-v1"
    assert len(meta["git_sha"]) == 40
    assert meta["op_version"] == "v2-isocal"
    assert meta["snapshot_sha256"] == demo_build.snapshot_sha256(records)
    cfg = cost_model.load_cost_config(demo_build.DEMO_COST_CONFIG)
    assert meta["cost_model"] == {"path": "configs/cost_model_v2.yaml", "sha256": cfg.sha256}
    assert set(meta["evidence_classes"]) == {"measured", "estimated", "projected", "derived"}
    # Owner decision 2026-08-12, executed in this payload.
    assert meta["headline_router"]["policy"] == "a_to_b"
    assert meta["headline_router"]["evaluation_set"] == router_sim.EVAL_FULL
    assert meta["pending_tier_b"] == list(demo_build.PENDING_TIER_B_SLOTS) == ["tier_b1"]


@_needs_demo
def test_tier_b_backfill_left_exactly_one_labeled_pending_slot(payload):
    """Tier B backfilled 2026-08-12: the only pending slot left is the drift panel's B1
    yearly series (descoped by owner), still a real labeled object, never an omitted key."""
    assert payload["frontier.json"]["pending_points"] == []
    assert payload["calibration.json"]["pending"] == []

    pending = payload["drift.json"]["pending_series"]
    assert [p["slot"] for p in pending] == ["tier_b1"]
    assert pending[0]["pending"] is True
    assert "descoped" in pending[0]["label"]
    assert set(payload["meta.json"]["pending_tier_b"]) == {"tier_b1"}

    # The former sample-card slots are real per-tier data now.
    for sample in payload["samples.json"]["samples"]:
        for key in ("tier_b1", "tier_b2"):
            tier = sample["tiers"][key]
            assert "pending" not in tier
            assert set(tier) == {"label", "p_max", "correct", "run_id"}


@_needs_demo
def test_every_exhibit_declares_an_evidence_class(payload):
    allowed = set(demo_build.EVIDENCE_LEGEND)
    seen = 0
    for name in ("frontier.json", "policies.json", "calibration.json", "samples.json"):
        for path, value in _walk(payload[name], ()):
            if path and path[-1] == "evidence_class" and isinstance(value, str):
                assert value.split(" ")[0] in allowed, f"{name} {path}: {value!r}"
                seen += 1
    assert seen >= 10


# ---------------------------------------------------------------------------
# Traceability: every run id resolves, every copied number is exact
# ---------------------------------------------------------------------------

@_needs_demo
def test_every_run_id_under_demo_data_exists_in_the_results_log(payload, records):
    known = {r["run_id"] for r in records}
    found: set[str] = set()
    for name, obj in payload.items():
        ids = _collect_run_ids(obj)
        if name == "runs_index.json":
            ids |= set(obj)
        if name == "receipts.json":
            ids |= set(obj["runs"])
        unknown = ids - known
        assert not unknown, f"{name} names run id(s) absent from runs.jsonl: {sorted(unknown)}"
        found |= ids
    assert len(found) >= 10, "traceability test found suspiciously few run ids"


@_needs_demo
def test_runs_index_has_one_verbatim_entry_per_record(payload, records):
    index = payload["runs_index.json"]
    assert len(index) == len(records)
    for record in records:
        entry = index[record["run_id"]]
        assert entry["metrics"] == record["metrics"]
        assert entry["dataset"] == record["dataset"]
        assert entry["cost_usd"] == record["cost_usd"]
        assert entry["config_sha256"] == record["config_sha256"]
        assert entry["slice"] == record["dataset"]["split"]
        assert entry["extra"] == (record.get("extra") or {})
        assert entry["tier"] in {"A", "B", "C"}
        assert entry["model_label"]


@_needs_demo
def test_frontier_claims_are_a_verbatim_copy_of_the_primary_file(payload):
    # Selected by the cost config the payload was built under: one frontier file exists
    # per cost generation, so "the only one there" stops being a selection.
    source = demo_build.primary_frontier_path(
        cost_model.load_cost_config(demo_build.DEMO_COST_CONFIG))
    claims = dict(payload["frontier.json"]["claims"])
    assert claims.pop("source") == "results/frontier/" + source.name
    assert claims == json.loads(source.read_text(encoding="utf-8"))


@_needs_demo
def test_frontier_single_tier_points_copy_macro_f1_and_cost_exactly(payload, resolved):
    points = {p["key"]: p for p in payload["frontier.json"]["points"]}
    expected = {
        "tier_a_logreg": (demo_build.TIER_A_LOGREG_TEST, "test_iid"),
        "tier_a_cnb": (demo_build.TIER_A_CNB_TEST, "test_iid"),
        "tier_c_haiku": (demo_build.HAIKU_TEST, "test_iid"),
        "tier_c_sonnet": (demo_build.SONNET_TEST, "test_iid"),
        **{key: (config, "test_iid") for config, key, _ in demo_build.TIER_B_TESTS},
    }
    for key, (config_name, slice_name) in expected.items():
        point = points[key]
        record = demo_build.record_for(resolved, config_name, slice_name)
        assert point["run_id"] == record["run_id"]

        logged = record["metrics"]["macro_f1"]
        assert point["macro_f1"] == {"point": logged["point"], "ci_lo": logged["ci_lo"],
                                     "ci_hi": logged["ci_hi"]}

        artifact = json.loads(
            (cost_model.DEFAULT_COST_DIR / f"{record['run_id']}.json").read_text())
        bands = artifact["expected_cost_per_1k"]
        for field, band in (("cost_per_1k_usd", bands["total"]),
                            ("api_cost_per_1k_usd", bands["api"])):
            assert point[field] == {"point": band["point"], "ci_lo": band["ci_lo"],
                                    "ci_hi": band["ci_hi"]}, f"{key}.{field}"
        assert point["cost_model_source"] == f"results/cost_model/{record['run_id']}.json"


ROUTER_EXHIBITS = (
    # (payload key, evaluation set, router_sim policy name, headline?)
    ("a_to_human", router_sim.EVAL_FULL, "a_to_human", False),
    ("a_to_b", router_sim.EVAL_FULL, "a_to_b", True),
    ("a_to_c_haiku", router_sim.EVAL_PAIRED, "a_to_c_parsefail_human", False),
    ("a_to_b_to_c", router_sim.EVAL_PAIRED, "a_to_b_to_c", False),
)


@_needs_demo
def test_frontier_router_points_copy_router_sim_exactly(payload):
    points = {p["key"]: p for p in payload["frontier.json"]["points"]}
    cfg = cost_model.load_cost_config(demo_build.DEMO_COST_CONFIG)
    for key, eval_set, policy_name, headline in ROUTER_EXHIBITS:
        path = router_sim.DEFAULT_ROUTER_DIR / router_sim.result_filename(
            eval_set, cfg, demo_build.OP_VERSION)
        policy = json.loads(path.read_text(encoding="utf-8"))["policies"][policy_name]
        point = points[key]
        band = policy["expected_cost_per_1k"]["total"]
        assert point["cost_per_1k_usd"] == {"point": band["point"], "ci_lo": band["ci_lo"],
                                            "ci_hi": band["ci_hi"]}
        assert point["macro_f1"] == {"point": policy["macro_f1_system"]}
        assert point["n"] == policy["n_examples"]
        assert point["cost_model_source"] == f"results/router_sim/{path.name}"
        assert point["headline"] is headline
    # Exactly one headline point, and it is the owner-decided a_to_b (2026-08-12).
    headline_keys = [p["key"] for p in payload["frontier.json"]["points"]
                     if p.get("headline")]
    assert headline_keys == ["a_to_b"]


@_needs_demo
def test_policies_copy_router_sim_and_the_frozen_thresholds(payload):
    cfg = cost_model.load_cost_config(demo_build.DEMO_COST_CONFIG)
    policies_doc = payload["policies.json"]
    assert policies_doc["op_version"] == demo_build.OP_VERSION
    assert policies_doc["cost_defaults"]["c_misroute"] == cfg.c_misroute_usd
    assert policies_doc["cost_defaults"]["c_human"] == cfg.c_human_usd
    assert policies_doc["cost_defaults"]["sha256"] == cfg.sha256

    by_key = {p["key"]: p for p in policies_doc["policies"]}
    assert list(by_key) == [k for k, _, _, _ in ROUTER_EXHIBITS]
    assert [k for k, p in by_key.items() if p["headline"]] == ["a_to_b"]
    for key, eval_set, policy_name, _headline in ROUTER_EXHIBITS:
        path = router_sim.DEFAULT_ROUTER_DIR / router_sim.result_filename(
            eval_set, cfg, demo_build.OP_VERSION)
        policy = json.loads(path.read_text(encoding="utf-8"))["policies"][policy_name]
        block = by_key[key]
        assert block["router_sim_policy"] == policy_name
        assert block["evaluation_set"] == eval_set
        # The two-gate cascade also publishes its frozen second threshold.
        if "tau_b" in policy["gate"]:
            assert block["tau_b"]["value"] == policy["gate"]["tau_b"]
        else:
            assert "tau_b" not in block

        assert block["expected_cost_per_1k"] == policy["expected_cost_per_1k"]
        assert block["rates"] == {
            "answered_a": policy["routing"]["coverage_a"],
            "escalated": policy["routing"]["escalation_rate"],
            "human": policy["routing"]["human_rate"],
        }
        assert block["p_error_machine"] == pytest.approx(
            1.0 - policy["accuracy_machine"], abs=1e-9)
        assert block["macro_f1_system"] == {"point": policy["macro_f1_system"]}
        assert block["source"] == f"results/router_sim/{path.name}"

        # The tau is the CAL constant, not a demo-local number: it must equal the
        # thresholds file's tau_star and that file's hash must still be its hash.
        tau_path = demo_build.REPO_ROOT / block["tau"]["source"]
        threshold = json.loads(tau_path.read_text(encoding="utf-8"))
        assert block["tau"]["value"] == threshold["tau_star"]
        assert block["tau"]["cal_tau_star"] == threshold["tau_star"]
        assert block["tau"]["sha256"] == cost_model.sha256_file(tau_path)
        assert threshold["derivation"] == demo_build.OP_VERSION


@_needs_demo
def test_drift_is_a_verbatim_copy_plus_annotations(payload):
    drift = payload["drift.json"]
    source = json.loads(demo_build.DEFAULT_DRIFT_SUMMARY.read_text(encoding="utf-8"))
    assert drift["summary"] == source
    assert drift["source"] == "results/drift/summary.json"
    assert [a["x"] for a in drift["annotations"]] == ["2023-04", "2026-H1"]
    assert [s["slot"] for s in drift["pending_series"]] == ["tier_b1"]
    # The copied rollup must actually carry the Tier B series the panel renders.
    assert "tier_b2" in drift["summary"]["tier_order"]
    assert {"a_to_b__full_slice", "a_to_b__paired_subset"} <= set(
        drift["summary"]["arm_order"])


@_needs_demo
def test_calibration_ece_and_brier_are_copied_from_the_run_records(payload, resolved):
    exhibits = payload["calibration.json"]["exhibits"]
    assert len(exhibits) >= 2
    # The four temperature-scaled Tier B TEST-IID finals ship alongside the Tier A trio.
    assert {key for _, key, _ in demo_build.TIER_B_TESTS} <= {e["key"] for e in exhibits}
    by_run = {r["run_id"]: r for r in predictions.load_records(RESULTS_PATH)}
    for exhibit in exhibits:
        record = by_run[exhibit["run_id"]]
        for key in ("ece", "brier"):
            logged = record["metrics"][key]
            assert exhibit[key] == {"point": logged["point"], "ci_lo": logged["ci_lo"],
                                    "ci_hi": logged["ci_hi"]}, f"{exhibit['key']}.{key}"
        assert exhibit["slice"] == record["dataset"]["split"]
        assert exhibit["calibration"] in {"raw", "isotonic", "temperature"}


@_needs_demo
def test_calibration_bins_are_the_bins_behind_the_logged_ece(payload):
    """15 equal-width bins that sum to n and reproduce the logged ECE."""
    for exhibit in payload["calibration.json"]["exhibits"]:
        bins = exhibit["bins"]
        assert len(bins) == 15
        assert sum(b["n"] for b in bins) == exhibit["n"]
        for i, b in enumerate(bins):
            assert b["lo"] == pytest.approx(i / 15, abs=1e-9)
            assert b["hi"] == pytest.approx((i + 1) / 15, abs=1e-9)
            if b["n"] == 0:
                assert b["conf_mean"] is None and b["acc"] is None
            else:
                assert b["lo"] - 1e-9 <= b["conf_mean"] <= b["hi"] + 1e-9
        replay = demo_build._ece_from_bins(bins, exhibit["n"])
        assert replay == pytest.approx(exhibit["ece"]["point"], abs=1e-6)


@_needs_demo
def test_tier_c_calibration_note_names_the_two_tier_c_runs(payload, resolved):
    note = payload["calibration.json"]["tier_c_note"]
    assert "degenerate one-hot" in note["text"]
    expected = {demo_build.record_for(resolved, demo_build.HAIKU_TEST, "test_iid")["run_id"],
                demo_build.record_for(resolved, demo_build.SONNET_TEST, "test_iid")["run_id"]}
    assert set(note["run_ids"]) == expected


# ---------------------------------------------------------------------------
# Receipts: committed-only, must pass in a clone with no data/
# ---------------------------------------------------------------------------

@_needs_demo
def test_sample_tier_c_fields_match_the_committed_receipts(payload, resolved):
    """>=20 curated rows, Haiku AND Sonnet, checked line-by-line against calls.jsonl.

    Deliberately independent of `data/`: samples.json and the receipts are both committed,
    so this is the test that proves the demo's per-call figures are auditable from the repo
    alone. The label is re-parsed here with the plain JSON contract the prompt specifies
    rather than by calling the production parser, so a change in that parser cannot make
    this test agree with the payload by construction.
    """
    samples = payload["samples.json"]["samples"]
    assert len(samples) >= 20

    for tier_key, config_name in (("haiku", demo_build.HAIKU_TEST),
                                  ("sonnet", demo_build.SONNET_TEST)):
        record = demo_build.record_for(resolved, config_name, "test_iid")
        raw_log_path = record["extra"]["raw_log_path"]
        fallback = record["extra"]["fallback_label"]
        by_id = _receipts_by_id(raw_log_path)
        checked = 0
        for sample in samples:
            receipt = by_id[sample["complaint_id"]]
            tier = sample["tiers"][tier_key]

            content = receipt.get("content")
            try:
                parsed = json.loads(content).get("label")
            except (TypeError, ValueError, AttributeError):
                parsed = None
            expected_label = parsed if parsed in payload["samples.json"]["class_labels"] \
                else fallback

            assert tier["run_id"] == record["run_id"]
            assert tier["label"] == expected_label
            assert tier["cost_usd"] == receipt["computed_cost_usd"]
            assert tier["latency_ms"] == receipt["latency_ms"]
            assert tier["provider"] == receipt["provider"]
            assert tier["prompt_tokens"] == receipt["prompt_tokens"]
            assert tier["completion_tokens"] == receipt["completion_tokens"]
            assert tier["parse_failed"] == receipt["parse_failed"]
            assert tier["correct"] == (tier["label"] == sample["y_true"])
            checked += 1
        assert checked >= 20, f"{tier_key}: only {checked} rows checked"


@_needs_demo
def test_receipts_rollup_recomputes_from_calls_jsonl(payload, records):
    drawer = payload["receipts.json"]
    assert drawer["repro"]["results_log"] == "results/runs.jsonl"

    by_run = {r["run_id"]: r for r in records}
    expected_runs = {r["run_id"] for r in records if (r.get("extra") or {}).get("raw_log_path")}
    assert set(drawer["runs"]) == expected_runs

    for run_id, entry in drawer["runs"].items():
        record = by_run[run_id]
        lines = _receipt_lines(entry["raw_log_path"])
        assert entry["raw_log_path"] == record["extra"]["raw_log_path"]
        assert entry["model"] == record["extra"]["model_slug"]
        assert entry["n_calls"] == len(lines)
        assert entry["total_cost_usd"] == pytest.approx(
            sum(r["computed_cost_usd"] for r in sorted(
                lines, key=lambda r: int(r["complaint_id"]))), abs=1e-12)
        assert entry["total_cost_usd"] == pytest.approx(record["cost_usd"], abs=1e-6)
        assert entry["provider_mix"] == dict(sorted(Counter(
            r["provider"] for r in lines).items()))
        assert entry["token_totals"] == {
            "prompt": sum(r["prompt_tokens"] for r in lines),
            "completion": sum(r["completion_tokens"] for r in lines),
        }
        assert entry["parse_failures"] == sum(1 for r in lines if r["parse_failed"])
        assert entry["parse_failures"] == record["extra"]["parse_failures"]
        assert entry["receipts_sha256"] == cost_model.receipts_sha256(entry["raw_log_path"])


# ---------------------------------------------------------------------------
# Curated set: frozen
# ---------------------------------------------------------------------------

@_needs_demo
def test_curated_ids_shape_and_pool_freeze_from_committed_receipts(payload, resolved):
    """The pool half of the freeze, checkable without `data/`.

    The draw itself needs `y_true` (gitignored split), so the receipts-only check is that
    the committed selection is a subset of exactly the pool the committed receipts still
    describe — pinned by `pool_sha256` so a receipts change cannot silently re-pool.
    """
    curated = payload["curated_ids.json"]
    assert curated["version"] == "v1"
    assert curated["seed"] == 20260806
    assert curated["n"] == 200 == len(curated["complaint_ids"])
    assert curated["complaint_ids"] == sorted(set(curated["complaint_ids"]))

    haiku = demo_build.record_for(resolved, demo_build.HAIKU_TEST, "test_iid")
    sonnet = demo_build.record_for(resolved, demo_build.SONNET_TEST, "test_iid")
    pool = sorted(set(_receipts_by_id(haiku["extra"]["raw_log_path"]))
                  & set(_receipts_by_id(sonnet["extra"]["raw_log_path"])))
    assert curated["pool_n"] == len(pool) == 1500
    import hashlib
    canonical = json.dumps(pool, separators=(",", ":")).encode("utf-8")
    assert curated["pool_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert set(curated["complaint_ids"]) <= set(pool)


@_needs_demo
def test_samples_selection_matches_curated_ids(payload):
    curated = payload["curated_ids.json"]
    samples = payload["samples.json"]
    assert [s["complaint_id"] for s in samples["samples"]] == curated["complaint_ids"]
    selection = samples["selection"]
    assert selection["narrative_source"] == demo_build.NARRATIVE_SOURCE
    for key, value in curated.items():
        assert selection[key] == value


@_needs_demo
def test_samples_router_paths_use_the_frozen_op(payload):
    """The per-sample path is the HEADLINE router (a_to_b, owner decision 2026-08-12):
    B2-terminal, so no human arm exists and every path ends answered."""
    allowed = {("A", "answered"),
               ("A", "escalated", "B2", "answered")}
    tau = payload["policies.json"]
    frozen_tau = next(p["tau"]["value"] for p in tau["policies"] if p["key"] == "a_to_b")
    for sample in payload["samples.json"]["samples"]:
        router = sample["router"]
        assert router["op_version"] == "v2-isocal"
        assert router["policy"] == "a_to_b"
        assert router["tau"] == frozen_tau
        assert tuple(router["path"]) in allowed
        # The gate decision must agree with the Tier A confidence the same file publishes.
        answered = sample["tiers"]["tier_a_logreg"]["p_max"] >= frozen_tau
        assert (tuple(router["path"]) == ("A", "answered")) is bool(answered)


@_needs_demo
def test_samples_carry_narrative_and_tier_a_provenance(payload, resolved):
    logreg = demo_build.record_for(resolved, demo_build.TIER_A_LOGREG_TEST, "test_iid")
    labels = set(payload["samples.json"]["class_labels"])
    for sample in payload["samples.json"]["samples"]:
        assert sample["narrative"].strip()
        assert sample["y_true"] in labels
        tier_a = sample["tiers"]["tier_a_logreg"]
        assert tier_a["run_id"] == logreg["run_id"]
        assert tier_a["label"] in labels
        assert 0.0 <= tier_a["p_max"] <= 1.0
        assert tier_a["correct"] == (tier_a["label"] == sample["y_true"])


@_needs_demo
def test_samples_carry_tier_b_provenance(payload, resolved):
    """Both Tier B cards trace to the frozen TEST-IID finals (tier_b1 shows seed sa)."""
    b1 = demo_build.record_for(resolved, demo_build.TIER_B1_SAMPLE_CONFIG, "test_iid")
    b2 = demo_build.record_for(resolved, demo_build.TIER_B2_SAMPLE_CONFIG, "test_iid")
    labels = set(payload["samples.json"]["class_labels"])
    assert "seed sa" in payload["samples.json"]["tier_b1_note"]
    for sample in payload["samples.json"]["samples"]:
        for key, record in (("tier_b1", b1), ("tier_b2", b2)):
            tier = sample["tiers"][key]
            assert tier["run_id"] == record["run_id"]
            assert tier["label"] in labels
            assert 0.0 <= tier["p_max"] <= 1.0
            assert tier["correct"] == (tier["label"] == sample["y_true"])


@_needs_demo
@_needs_data
def test_curated_selection_regenerates_byte_identically(payload, resolved):
    """The full freeze: replay the seeded draw from receipts + the frozen split."""
    haiku = demo_build.record_for(resolved, demo_build.HAIKU_TEST, "test_iid")
    sonnet = demo_build.record_for(resolved, demo_build.SONNET_TEST, "test_iid")
    pool = demo_build.paired_pool(haiku, sonnet)
    rows = demo_build.load_split_rows(pool, "test_iid")
    y_true_by_id = {cid: label for cid, (_, label) in rows.items()}

    regenerated = demo_build.build_curated_ids(pool, y_true_by_id)
    assert regenerated == payload["curated_ids.json"]

    text = json.dumps(regenerated, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    assert text == (DEMO_DIR / "curated_ids.json").read_text(encoding="utf-8")


@_needs_demo
@_needs_data
def test_curated_set_is_class_stratified_with_every_class_present(payload, resolved):
    counts = Counter(s["y_true"] for s in payload["samples.json"]["samples"])
    labels = payload["samples.json"]["class_labels"]
    assert set(counts) == set(labels), "every class must appear at least once"
    assert sum(counts.values()) == 200
    assert all(v >= 1 for v in counts.values())


@_needs_data
def test_building_twice_yields_byte_identical_output(tmp_path):
    """Determinism: no wall-clock stamp, no dict-order dependence, no RNG drift."""
    first, second = tmp_path / "a", tmp_path / "b"
    demo_build.build_all(first)
    demo_build.build_all(second)
    for name in CONTRACT_FILES:
        a = (first / name).read_bytes()
        b = (second / name).read_bytes()
        assert a == b, f"{name} is not byte-identical across two builds"


@_needs_demo
@_needs_data
def test_freeze_check_rejects_a_changed_selection(tmp_path, payload):
    path = tmp_path / "curated_ids.json"
    demo_build.write_json(payload["curated_ids.json"], path)
    demo_build.freeze_check(payload["curated_ids.json"], path)  # unchanged: passes

    mutated = {**payload["curated_ids.json"], "seed": 1}
    with pytest.raises(ValueError, match="FROZEN SELECTION CHANGED"):
        demo_build.freeze_check(mutated, path)


# ---------------------------------------------------------------------------
# Case study: the page where prose and numbers share a sentence
# ---------------------------------------------------------------------------
#
# The case study is the highest-risk exhibit in the repo: it is the one place a number is
# written INSIDE a sentence, which is exactly where an unattributed or stale figure is
# cheapest to produce and hardest to spot. So it gets four gates rather than one:
#
#   (a) every run id it names exists in the append-only log (shared with every other file);
#   (b) every COPIED value appears, at exact float equality, among the numeric leaves of
#       the artifact its `source` names — plus hand-written spot checks on the headlines,
#       so the generic gate cannot be satisfied by a coincidentally-present float;
#   (c) every `display` string reads back to its own `value` under its declared `unit`,
#       component by component and at the display's own precision — the page cannot show
#       "0.76" for a value of 0.7605270747 or drop a CI bound;
#   (d) every numeric token in the PROSE is one of that section's declared displays.
#
# Gate (d) needs a tokenizer, and a tokenizer needs an exemption rule. The rule is:
# **a numeric token is exempt only if it is an IDENTIFIER, never if it is a quantity.**
# Concretely — calendar years and half-year slice labels (2015, 2026-H1, 2022-H2), ISO
# dates, hex digests, the standard name SHA-256, the model names "Haiku 4.5" / "Sonnet 5",
# the unit suffix "1k", and digits that are part of a word (A6000, bf16, int8, fp32, v2,
# QInt8, B2), which fall out of the "not preceded by a letter or digit" lookbehind rather
# than needing a list. Nothing that is a rate, count, dollar figure, CI bound or p-value is
# exempt: those must be declared or the test fails.
#
# The known blind spot, stated rather than hidden: a genuine measurement that happens to
# look like a bare calendar year (e.g. an unseparated "2000") would be silently exempted.
# Every count on the page is thousands-separated, which keeps that case out of reach.

_CS_IDENTIFIER_PATTERNS = (
    r"\d{4}-\d{2}-\d{2}",                                   # ISO dates
    r"\b(?:19|20)\d{2}(?:-H[12])?\b",                       # years / half-year slices
    r"\b(?=[0-9a-f]{8,}\b)[0-9a-f]*[a-f][0-9a-f]*\b",       # hex digests
    r"\bSHA-256\b",
    r"\b(?:Claude\s+)?(?:Haiku\s+4\.5|Sonnet\s+5)\b",       # model product names
    r"/1k\b|\b1k\b",                                        # the per-1k unit suffix
)
_CS_IDENTIFIER_RE = re.compile("|".join(_CS_IDENTIFIER_PATTERNS))
# Not preceded by a letter or digit: "A6000", "bf16", "int8", "fp32", "v2", "B2" are words.
_CS_NUMBER_RE = re.compile(r"(?<![0-9A-Za-z])\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?")

_CS_ARTIFACT_SOURCES = {  # sources the value gate can open and search
    "results/runs.jsonl",
    "SNAPSHOT_MANIFEST.yaml",
}
_CS_UNCHECKABLE_SOURCES = {  # derived-from-a-glob or declared-in-code; other gates cover
    "results/tier_c_raw/**/calls.jsonl",
    "src/triage_lab/demo_build.py",
}


def _cs_numeric_tokens(text: str) -> set[str]:
    """Numeric tokens in prose, identifiers removed, commas normalized away."""
    stripped = _CS_IDENTIFIER_RE.sub(" ", text)
    return {m.group(0).replace(",", "") for m in _CS_NUMBER_RE.finditer(stripped)}


def _cs_display_tokens(display: str) -> list[str]:
    """Unsigned tokens, for the prose gate — signs live in the sentence, not the number."""
    return [m.group(0).replace(",", "") for m in _CS_NUMBER_RE.finditer(display)]


# Signed form, for the read-back gate only: a dropped or flipped minus is exactly the kind
# of error the display gate exists to catch, so that one comparison is sign-aware.
_CS_SIGNED_RE = re.compile(
    r"(?P<sign>[-+]?)\$?(?P<num>\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def _cs_display_signed(display: str) -> list[str]:
    return [m.group("sign") + m.group("num").replace(",", "")
            for m in _CS_SIGNED_RE.finditer(display)]


def _cs_numeric_leaves(obj) -> set[float]:
    out: set[float] = set()
    for _path, value in _walk(obj, ()):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        out.add(float(value))
    return out


def _cs_source_leaves(entry: dict, records: list[dict]) -> set[float] | None:
    """Numeric leaves of the artifact an entry's `source` names, or None if uncheckable."""
    source = entry["source"]
    if source in _CS_UNCHECKABLE_SOURCES:
        return None
    path = demo_build.REPO_ROOT / source
    if source == "results/runs.jsonl":
        by_run = {r["run_id"]: r for r in records}
        named = [by_run[rid] for rid in entry["run_ids"]]
        assert named, f"{entry['label']}: sources runs.jsonl but names no run id"
        return set().union(*[_cs_numeric_leaves(r) for r in named])
    assert path.is_file(), f"{entry['label']}: source {source} does not exist"
    if path.suffix == ".yaml":
        import yaml
        return _cs_numeric_leaves(yaml.safe_load(path.read_text(encoding="utf-8")))
    return _cs_numeric_leaves(json.loads(path.read_text(encoding="utf-8")))


def _cs_value_components(value) -> list[float]:
    if isinstance(value, dict):
        return [float(value["point"]), float(value["ci_lo"]), float(value["ci_hi"])]
    return [float(value)]


@pytest.fixture(scope="module")
def case_study(payload) -> dict:
    return payload["case_study.json"]


@pytest.fixture(scope="module")
def cs_sections(case_study) -> dict:
    return {s["id"]: s for s in case_study["sections"]}


@_needs_demo
def test_case_study_shape(case_study, cs_sections):
    assert case_study["schema_version"] == "case-study-v1"
    assert case_study["title"] == demo_build.CASE_STUDY_TITLE
    assert case_study["source_note"]
    assert set(case_study["evidence_classes"]) == set(demo_build.EVIDENCE_LEGEND)
    assert list(cs_sections) == [
        "intro", "tiers", "drift", "thresholds", "router", "robustness", "negatives",
        "verification", "limits", "provenance",
    ]
    allowed_evidence = set(demo_build.EVIDENCE_LEGEND) | {"pending"}
    for section in case_study["sections"]:
        assert section["kind"] in {"narrative", "verification", "limits", "pending"}
        assert section["title"]
        assert isinstance(section["paragraphs"], list)
        assert isinstance(section["numbers"], list)
        assert isinstance(section["items"], list)
        assert isinstance(section["gaps"], list)
        labels = [n["label"] for n in section["numbers"]]
        assert len(labels) == len(set(labels)), f"{section['id']}: duplicate number labels"
        for entry in section["numbers"]:
            assert entry["unit"] in {"raw", "usd", "pct", "pctpoint", "count", "pvalue"}
            assert entry["basis"] in {"copied", "derived", "declared"}
            assert entry["evidence_class"] in allowed_evidence
            assert entry["source"]
            assert entry["display"]
        # The section's chips are exactly the ordered union of its numbers' run ids.
        expected: list[str] = []
        for entry in section["numbers"]:
            for run_id in entry["run_ids"]:
                if run_id not in expected:
                    expected.append(run_id)
        assert section["run_ids"] == expected


@_needs_demo
def test_case_study_pending_slots_are_labeled_objects(case_study, cs_sections):
    slots = {p["slot"] for p in case_study["pending"]}
    assert slots == {"reproduce_headline", "provenance_seeds"}
    for slot in case_study["pending"]:
        assert slot["pending"] is True and slot["label"]
    # ...and both are visible where the reader meets them, not only in the footer.
    verification_pending = [i for i in cs_sections["verification"]["items"] if i.get("pending")]
    assert [i["pending"]["slot"] for i in verification_pending] == ["reproduce_headline"]
    assert cs_sections["provenance"]["pending"]["slot"] == "provenance_seeds"


@_needs_demo
def test_case_study_run_ids_all_trace_to_the_results_log(case_study, records):
    known = {r["run_id"] for r in records}
    seen: set[str] = set()
    for section in case_study["sections"]:
        for entry in section["numbers"]:
            for run_id in entry["run_ids"]:
                assert run_id in known, f"{entry['label']}: unknown run id {run_id}"
                seen.add(run_id)
        for item in section["items"]:
            for run_id in item.get("run_ids", []):
                assert run_id in known, f"item {item.get('n')}: unknown run id {run_id}"
                seen.add(run_id)
    assert len(seen) >= 15, "case study cites suspiciously few runs"


@_needs_demo
def test_case_study_copied_values_exist_in_their_source_artifact(case_study, records):
    """Generic gate (b): a copied number must literally be in the file it names."""
    checked = 0
    for section in case_study["sections"]:
        for entry in section["numbers"]:
            if entry["basis"] != "copied":
                continue
            leaves = _cs_source_leaves(entry, records)
            if leaves is None:
                continue
            for component in _cs_value_components(entry["value"]):
                assert component in leaves, (
                    f"{section['id']}.{entry['label']}: {component!r} is not a value in "
                    f"{entry['source']} — the page is not copying, it is asserting")
                checked += 1
    assert checked >= 100, "value gate covered suspiciously few components"


@_needs_demo
def test_case_study_headline_values_match_a_hand_written_lookup(case_study, cs_sections,
                                                                resolved):
    """Targeted gate (b): the numbers a reader would quote, looked up independently.

    Deliberately not driven by `source`/`label`: this table is written from the artifacts,
    so a build that renamed a source or picked the wrong row still fails here.
    """
    def _num(section_id, label):
        return {n["label"]: n for n in cs_sections[section_id]["numbers"]}[label]

    def _artifact(rel):
        return json.loads((demo_build.REPO_ROOT / rel).read_text(encoding="utf-8"))

    logreg = demo_build.record_for(resolved, demo_build.TIER_A_LOGREG_TEST, "test_iid")
    b2 = demo_build.record_for(resolved, demo_build.TIER_B2_SAMPLE_CONFIG, "test_iid")
    for label, record in (("tier_a_macro_f1", logreg), ("b2_macro_f1", b2)):
        logged = record["metrics"]["macro_f1"]
        assert _num("tiers", label)["value"] == {
            "point": logged["point"], "ci_lo": logged["ci_lo"], "ci_hi": logged["ci_hi"]}

    compare = _artifact("results/tier_b_compare/summary.json")
    b1_vs_b2 = next(c for c in compare["comparisons"]
                    if c["a"] == "b1_sa" and c["b"] == "b2")
    assert _num("tiers", "b1_minus_b2_macro_f1")["value"] == {
        "point": b1_vs_b2["macro_f1"]["delta"],
        "ci_lo": b1_vs_b2["macro_f1"]["ci_lo"],
        "ci_hi": b1_vs_b2["macro_f1"]["ci_hi"]}
    assert _num("tiers", "b1_sa_macro_f1")["value"] == \
        compare["provenance"]["b1_sa"]["macro_f1_point"]

    paired = _artifact("results/prior_shift/paired_within_tier_a_vs_tier_b2_2026h1.json")
    assert paired["ci_excludes_zero"] is True, "the certified phrasing needs an excl-0 CI"
    assert _num("drift", "paired_within_delta")["value"] == {
        "point": paired["delta"]["point"], "ci_lo": paired["delta"]["ci_lo"],
        "ci_hi": paired["delta"]["ci_hi"]}

    frontier_doc = _artifact(
        demo_build._rel(demo_build.primary_frontier_path(
            cost_model.load_cost_config(demo_build.DEMO_COST_CONFIG))))
    claim = next(c for c in frontier_doc["claims"]
                 if c.get("router") == "a_to_b" and c.get("baseline") == "b2_only"
                 and c.get("evaluation_set") == router_sim.EVAL_FULL)
    assert claim["gate"]["certified"] is True, "the headline two-axis claim must be certified"
    for label, key in (("a_to_b_vs_b2_cost", "delta_cost_per_1k"),
                       ("a_to_b_vs_b2_f1", "delta_macro_f1_system")):
        band = claim[key]
        assert _num("router", label)["value"] == {
            "point": band["point"], "ci_lo": band["ci_lo"], "ci_hi": band["ci_hi"]}

    drift = _artifact("results/drift/summary.json")
    cliff = next(r for r in drift["series"]["logged"]
                 if r["tier"] == "tier_a" and r["slice"] == "test_drift_2026h1")
    assert _num("drift", "a_2026")["value"] == {
        "point": cliff["macro_f1"]["point"], "ci_lo": cliff["macro_f1"]["ci_lo"],
        "ci_hi": cliff["macro_f1"]["ci_hi"]}

    oov = _artifact("results/oov/summary.json")
    peak = next(r for r in oov["rows"] if r["slice"] == "test_drift_2025"
                and r["metric"] == "tfidf_centroid_cosine_distance")
    year = next(r for r in oov["rows"] if r["slice"] == "test_drift_2026h1"
                and r["metric"] == "tfidf_centroid_cosine_distance")
    assert _num("negatives", "centroid_2025")["value"]["point"] == peak["point"]
    assert _num("negatives", "centroid_2026")["value"]["point"] == year["point"]
    # The prose says the intervals are DISJOINT; if they ever overlap, the claim is wrong.
    assert year["ci_hi"] < peak["ci_lo"], "OOV prose claims disjoint intervals"


@_needs_demo
def test_case_study_display_reads_back_to_its_value(case_study):
    """Gate (c): the digits shown are the value, at the precision shown."""
    for section in case_study["sections"]:
        for entry in section["numbers"]:
            tokens = _cs_display_signed(entry["display"])
            components = _cs_value_components(entry["value"])
            where = f"{section['id']}.{entry['label']}"
            assert len(tokens) == len(components), (
                f"{where}: display {entry['display']!r} has {len(tokens)} numeric tokens "
                f"for {len(components)} value component(s)")
            for token, component in zip(tokens, components, strict=True):
                shown = float(token)
                if entry["unit"] == "pct":
                    component = component * 100.0
                if entry["unit"] == "pvalue":
                    assert component == pytest.approx(shown, rel=0.05), where
                    continue
                digits = len(token.split(".")[1]) if "." in token else 0
                assert shown == pytest.approx(round(component, digits), abs=10 ** -9), (
                    f"{where}: display shows {token!r} but the value is {component!r}")


@_needs_demo
def test_every_numeric_token_in_the_prose_is_a_declared_number(case_study):
    """Gate (d): no number reaches a sentence without a source behind it."""
    total_tokens = 0
    for section in case_study["sections"]:
        declared: set[str] = set()
        for entry in section["numbers"]:
            declared.update(_cs_display_tokens(entry["display"]))
        texts = list(section["paragraphs"])
        texts += [item["text"] for item in section["items"] if "text" in item]
        for text in texts:
            tokens = _cs_numeric_tokens(text)
            total_tokens += len(tokens)
            undeclared = tokens - declared
            assert not undeclared, (
                f"{section['id']}: prose contains number(s) {sorted(undeclared)} that no "
                f"`numbers` entry declares (declared: {sorted(declared)}) — either copy "
                f"them from an artifact or take them out of the sentence.\n{text}")
    assert total_tokens >= 60, "prose gate matched suspiciously few tokens"


@_needs_demo
def test_case_study_tokenizer_still_catches_an_undeclared_number():
    """The gate above is only worth its runtime if it can fail. Prove it can."""
    assert _cs_numeric_tokens("macro-F1 0.7605 on 2026-H1 with SHA-256 deadbeefcafe") == \
        {"0.7605"}
    # identifiers, not quantities: exempt
    assert _cs_numeric_tokens("Haiku 4.5 and Sonnet 5 on an A6000 in bf16, int8 v2") == set()
    assert _cs_numeric_tokens("downloaded 2026-08-05 across 2015 to 2026") == set()
    assert _cs_numeric_tokens("$1.315/1k calls") == {"1.315"}
    # quantities: never exempt
    assert _cs_numeric_tokens("n=104,443 rows at 99.81% and p=6.3e-08") == \
        {"104443", "99.81", "6.3e-08"}


@_needs_demo
def test_case_study_derived_numbers_recompute_from_their_sources(cs_sections, records,
                                                                 resolved, payload):
    """Gate for `basis: derived`: recomputed here from the artifacts, not from the module."""
    def _num(section_id, label):
        return {n["label"]: n for n in cs_sections[section_id]["numbers"]}[label]

    drift = json.loads((demo_build.REPO_ROOT / "results/drift/summary.json")
                       .read_text(encoding="utf-8"))
    ci_lo, ci_hi = drift["bootstrap"]["ci_pct"]
    for section_id in ("intro", "verification"):
        assert _num(section_id, "n_run_records")["value"] == len(records)
        assert _num(section_id, "ci_level")["value"] == ci_hi - ci_lo

    def _esc(policy, slice_name):
        return next(r for r in drift["series"]["escalation"]
                    if r["policy"] == policy and r["slice"] == slice_name
                    and r["dataset"] == "full_cal")["escalation_rate"]["point"]

    for label, policy in (("human_rise", "a_to_human"), ("b_rise", "a_to_b")):
        expected = _esc(policy, "test_drift_2026h1") / _esc(policy, "test_drift_2025") - 1
        assert _num("thresholds", label)["value"] == pytest.approx(expected, abs=1e-12)

    perturb = json.loads((demo_build.REPO_ROOT / "results/perturbation/summary.json")
                         .read_text(encoding="utf-8"))

    def _delta(arm):
        return next(r for r in perturb["rows"] if r["arm"] == arm and r["family"] == "typo"
                    and r["rate"] == 0.1)["metrics"]["macro_f1"]["delta"]

    shield = 1.0 - _delta("logreg_wordchar") / _delta("logreg_word_only")
    assert _num("robustness", "char_shield_share")["value"] == pytest.approx(shield,
                                                                             abs=1e-12)

    # Prompt-token inflation: recomputed straight off the committed receipt lines, with no
    # help from cost_model's loader, so a change in that loader cannot make this agree.
    clean = _receipts_by_id(demo_build.record_for(
        resolved, demo_build.HAIKU_TEST, "test_iid")["extra"]["raw_log_path"])
    for family in ("typo", "ocr", "case"):
        row = next(r for r in perturb["rows"] if r["arm"] == "tier_c_haiku"
                   and r["family"] == family and r["rate"] == 0.1)
        perturbed = _receipts_by_id(demo_build.record_for(
            resolved, row["perturbed_config"], "test_iid")["extra"]["raw_log_path"])
        ids = sorted(perturbed)
        expected = (sum(perturbed[i]["prompt_tokens"] for i in ids)
                    / sum(clean[i]["prompt_tokens"] for i in ids) - 1.0)
        assert _num("robustness", f"inflation_{family}")["value"] == \
            pytest.approx(expected, abs=1e-12)

    providers: Counter = Counter()
    for entry in payload["receipts.json"]["runs"].values():
        providers.update(entry["provider_mix"])
    total = sum(providers.values())
    assert _num("verification", "n_tier_c_calls")["value"] == total
    assert _num("verification", "bedrock_share")["value"] == pytest.approx(
        providers["Amazon Bedrock"] / total, abs=1e-12)


@_needs_demo
def test_case_study_declared_suite_counts_are_the_module_constant(cs_sections):
    """The one number with no results/ artifact: it must at least be a named constant."""
    numbers = {n["label"]: n for n in cs_sections["verification"]["numbers"]}
    for field in ("passed", "skipped", "failed"):
        entry = numbers[f"suite_{field}"]
        assert entry["basis"] == "declared"
        assert entry["value"] == demo_build.SUITE_RESULT[field]
        assert entry["repro"] == demo_build.SUITE_RESULT["command"]
    assert numbers["suite_failed"]["value"] == 0, "the page may not claim a green suite"


@_needs_demo
def test_case_study_records_the_numbers_it_deliberately_does_not_show(cs_sections):
    """`gaps` is load-bearing: an omission with a reason, not a silent absence.

    The three `tier_c_compare` gaps closed on 2026-08-13 when that tool learned to write a
    committed artifact, so the page now STATES those comparisons. This test pins the
    closure: no gap may cite tier_c_compare again without the artifact also disappearing,
    which the value gate would catch first.
    """
    gap_text = " ".join(g for s in cs_sections.values() for g in s["gaps"])
    assert "tier_c_compare" not in gap_text, (
        "a tier_c_compare gap is back: either an artifact went missing or a paired "
        "comparison was silently dropped from the prose")
    # The one surviving omission is a real one: a superseded cost generation's figure.
    assert cs_sections["negatives"]["gaps"], "the cost-generation omission must stay stated"
    assert "generation" in " ".join(cs_sections["negatives"]["gaps"])
    for section_id in ("tiers", "drift", "negatives"):
        assert not any("EXPERIMENT_LOG.md" in g for g in cs_sections[section_id]["gaps"]), (
            f"{section_id}: EXPERIMENT_LOG.md is no longer a substitute for an artifact")


@_needs_demo
def test_case_study_states_the_three_paired_tier_c_comparisons(cs_sections):
    """The arc the gaps used to swallow: tied in distribution, decisive off it.

    Checked against the artifacts rather than the prose, and with the SLICE pinned — the
    POSTCUTOFF delta is not a 2026-H1 drift-slice number and the page must not imply it is.
    """
    def _art(key):
        return json.loads((demo_build.DEFAULT_TIER_C_COMPARE_DIR / f"{key}.json")
                          .read_text(encoding="utf-8"))

    expected = {
        ("tiers", "sonnet_minus_haiku_iid"): (demo_build.TIER_C_COMPARE_SONNET_IID,
                                              "test_iid", False),
        ("drift", "sonnet_minus_haiku_pc"): (demo_build.TIER_C_COMPARE_SONNET_POSTCUTOFF,
                                             "test_postcutoff", True),
        ("negatives", "fewshot_minus_zeroshot"): (demo_build.TIER_C_COMPARE_FEWSHOT,
                                                  "cal", False),
    }
    for (section_id, prefix), (key, split, excludes_zero) in expected.items():
        artifact = _art(key)
        assert artifact["split"] == split, f"{key}: wrong slice"
        band = artifact["deltas"]["macro_f1"]
        assert band["excludes_zero"] is excludes_zero, (
            f"{key}: the page's verdict depends on this and it flipped")
        numbers = {n["label"]: n for n in cs_sections[section_id]["numbers"]}
        assert numbers[f"{prefix}_delta"]["value"] == {
            "point": band["delta"], "ci_lo": band["ci_lo"], "ci_hi": band["ci_hi"]}
        assert numbers[f"{prefix}_p"]["value"] == artifact["mcnemar"]["p_value"]
        assert numbers[f"{prefix}_n"]["value"] == artifact["n_examples"] == 1500
        assert numbers[f"{prefix}_delta"]["source"] == \
            f"results/tier_c_compare/{key}.json"
        assert numbers[f"{prefix}_delta"]["repro"] == artifact["repro_command"]
        # Both arms' run ids, resolved by the tool from the receipts, reach the chips.
        assert set(numbers[f"{prefix}_delta"]["run_ids"]) == {
            artifact["arm_a"]["run_id"], artifact["arm_b"]["run_id"]}
    # Slice hygiene: the drift section must name POSTCUTOFF where it quotes that delta.
    drift_prose = " ".join(cs_sections["drift"]["paragraphs"])
    assert "TEST-POSTCUTOFF" in drift_prose


@_needs_demo
def test_case_study_sections_carry_receipts(cs_sections):
    for section_id in ("tiers", "drift", "thresholds", "router", "robustness",
                       "negatives"):
        section = cs_sections[section_id]
        assert section["repro"], f"{section_id}: no reproduction command"
        assert section["run_ids"] or section_id == "router", (
            f"{section_id}: no run chips")
    # The verification checklist is the page's spine: every item states a source.
    for item in cs_sections["verification"]["items"]:
        assert item["source"] and item["title"] and item["text"]


@_needs_demo
def test_case_study_is_reachable_from_the_site(case_study):
    """A payload no panel loads is not an exhibit. Pin the wiring, not the styling."""
    index = (demo_build.REPO_ROOT / "demo" / "index.html").read_text(encoding="utf-8")
    app = (demo_build.REPO_ROOT / "demo" / "assets" / "app.js").read_text(encoding="utf-8")
    assert 'data-panel="casestudy"' in index
    assert 'id="panel-casestudy"' in index
    for element_id in ("casestudy-body", "casestudy-banner-slot", "casestudy-source-note",
                       "casestudy-title"):
        assert f'id="{element_id}"' in index, f"index.html has no #{element_id}"
        assert f'"{element_id}"' in app, f"app.js never reads #{element_id}"
    assert '"casestudy"' in app and "initCaseStudy()" in app
    assert 'loadJSON("case_study.json")' in app


# ---------------------------------------------------------------------------
# Unit: run resolution + stratified allocation (no repo state)
# ---------------------------------------------------------------------------

def _record(run_id, config, split, ts, **extra):
    return {
        "run_id": run_id,
        "config_path": f"configs/{config}.yaml",
        "dataset": {"split": split},
        "timestamp_utc": ts,
        **({"extra": extra} if extra else {}),
    }


def test_resolve_records_keys_on_config_and_slice():
    records = [
        _record("a" * 64, "tier_a_logreg_test_iid", "test_iid", "2026-01-01T00:00:00Z"),
        _record("b" * 64, "tier_a_logreg_test_iid", "cal", "2026-01-02T00:00:00Z"),
    ]
    resolved = demo_build.resolve_records(records)
    assert resolved[("tier_a_logreg_test_iid", "test_iid")]["run_id"] == "a" * 64
    assert resolved[("tier_a_logreg_test_iid", "cal")]["run_id"] == "b" * 64


def test_resolve_records_prefers_the_latest_duplicate():
    records = [
        _record("a" * 64, "c", "cal", "2026-01-01T00:00:00Z"),
        _record("b" * 64, "c", "cal", "2026-02-01T00:00:00Z"),
    ]
    assert demo_build.resolve_records(records)[("c", "cal")]["run_id"] == "b" * 64


def test_resolve_records_honours_an_explicit_supersede():
    """A later record can point at an EARLIER one; the pointed-at record loses regardless."""
    records = [
        _record("a" * 64, "c", "cal", "2026-03-01T00:00:00Z"),
        _record("b" * 64, "c", "cal", "2026-02-01T00:00:00Z", supersedes="a" * 64),
    ]
    assert demo_build.resolve_records(records)[("c", "cal")]["run_id"] == "b" * 64


def test_resolve_records_refuses_an_unbreakable_tie():
    records = [
        _record("a" * 64, "c", "cal", "2026-01-01T00:00:00Z"),
        _record("b" * 64, "c", "cal", "2026-01-01T00:00:00Z"),
    ]
    with pytest.raises(ValueError, match="cannot resolve"):
        demo_build.resolve_records(records)


def test_record_for_names_the_available_runs_when_missing():
    with pytest.raises(ValueError, match="no run record for config"):
        demo_build.record_for({}, "nope", "test_iid")


@pytest.mark.parametrize("n", [10, 200, 999])
def test_stratified_allocation_sums_and_floors(n):
    counts = {"a": 1000, "b": 500, "c": 3, "d": 1}
    alloc = demo_build.stratified_allocation(counts, n)
    assert sum(alloc.values()) == n
    assert all(alloc[c] >= 1 for c in counts)
    assert all(alloc[c] <= counts[c] for c in counts)


def test_stratified_allocation_is_order_independent():
    counts = {"a": 700, "b": 201, "c": 99}
    forward = demo_build.stratified_allocation(counts, 200)
    reversed_ = demo_build.stratified_allocation(dict(reversed(counts.items())), 200)
    assert forward == reversed_


def test_stratified_allocation_refuses_an_impossible_request():
    with pytest.raises(ValueError, match="cannot give each"):
        demo_build.stratified_allocation({c: 10 for c in "abcde"}, 3)
    with pytest.raises(ValueError, match="cannot draw"):
        demo_build.stratified_allocation({"a": 2, "b": 2}, 10)


def test_select_curated_ids_is_stable_and_sorted():
    pool = list(range(1000, 1300))
    y_true = {cid: ("x" if cid % 3 else "y") for cid in pool}
    first = demo_build.select_curated_ids(pool, y_true, n=50, seed=demo_build.CURATED_SEED)
    shuffled = list(reversed(pool))
    second = demo_build.select_curated_ids(shuffled, y_true, n=50,
                                           seed=demo_build.CURATED_SEED)
    assert first == second == sorted(first)
    assert len(first) == 50


def test_select_curated_ids_changes_with_the_seed():
    pool = list(range(1000, 1300))
    y_true = {cid: ("x" if cid % 3 else "y") for cid in pool}
    assert (demo_build.select_curated_ids(pool, y_true, n=50, seed=1)
            != demo_build.select_curated_ids(pool, y_true, n=50, seed=2))


def test_reliability_bins_shape_and_empty_bins():
    p_max = [0.05, 0.5, 0.5, 1.0]
    correct = [False, True, False, True]
    bins = demo_build.reliability_bins(p_max, correct)
    assert len(bins) == 15
    assert sum(b["n"] for b in bins) == 4
    assert bins[0]["n"] == 1 and bins[0]["acc"] == 0.0
    assert bins[14]["n"] == 1, "p = 1.0 must fold into the top bin"
    empty = [b for b in bins if b["n"] == 0]
    assert empty and all(b["conf_mean"] is None and b["acc"] is None for b in empty)


def test_write_json_is_canonical_and_atomic(tmp_path):
    path = demo_build.write_json({"b": 1, "a": "é"}, tmp_path / "x.json")
    assert path.read_text(encoding="utf-8") == '{\n  "a": "é",\n  "b": 1\n}\n'
    assert not list(tmp_path.glob("*.tmp"))


def test_snapshot_sha256_refuses_disagreeing_records():
    ok = [{"dataset": {"input_sha256": "a" * 64}}, {"dataset": {"input_sha256": "a" * 64}}]
    assert demo_build.snapshot_sha256(ok) == "a" * 64
    bad = [{"dataset": {"input_sha256": "a" * 64}}, {"dataset": {"input_sha256": "b" * 64}}]
    with pytest.raises(ValueError, match="exactly one dataset input_sha256"):
        demo_build.snapshot_sha256(bad)


def test_model_label_covers_every_committed_run():
    for record in predictions.load_records(RESULTS_PATH):
        label = demo_build.model_label(record)
        assert label.startswith("Tier ")
        assert demo_build.tier_of(record) in {"A", "B", "C"}


def test_pending_slot_is_a_real_object():
    assert demo_build.pending_slot("x", "L") == {"pending": True, "slot": "x", "label": "L"}
    assert demo_build.pending_slot("x") == {"pending": True, "slot": "x"}


def test_repo_paths_are_relative_and_posix():
    assert demo_build._rel(Path(RESULTS_PATH)) == "results/runs.jsonl"
