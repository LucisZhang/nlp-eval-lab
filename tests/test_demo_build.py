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
def test_all_nine_contract_files_exist():
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
    cfg = cost_model.load_cost_config()
    assert meta["cost_model"] == {"path": "configs/cost_model_v1.yaml", "sha256": cfg.sha256}
    assert set(meta["evidence_classes"]) == {"measured", "estimated", "projected", "derived"}
    assert meta["pending_tier_b"] == list(demo_build.PENDING_TIER_B_SLOTS)


@_needs_demo
def test_pending_tier_b_slots_are_real_objects(payload):
    """Every Tier B slot is an object with `pending: true`, never an omitted key."""
    slots = set()
    for obj in (payload["frontier.json"]["pending_points"],
                payload["calibration.json"]["pending"],
                payload["drift.json"]["pending_series"]):
        for entry in obj:
            assert entry["pending"] is True
            slots.add(entry["slot"])

    for sample in payload["samples.json"]["samples"]:
        for key in ("tier_b1", "tier_b2"):
            slot = sample["tiers"][key]
            assert slot["pending"] is True and slot["slot"] == key
            slots.add(key)

    assert slots <= set(payload["meta.json"]["pending_tier_b"])
    assert {"tier_b1_modernbert", "tier_b2_distilbert", "router_a_b_c"} <= slots


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
    source = demo_build.primary_frontier_path(cost_model.load_cost_config())
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


@_needs_demo
def test_frontier_router_points_copy_router_sim_exactly(payload):
    points = {p["key"]: p for p in payload["frontier.json"]["points"]}
    cfg = cost_model.load_cost_config()
    for key, eval_set, policy_name in (
        ("a_to_human", router_sim.EVAL_FULL, "a_to_human"),
        ("a_to_c_haiku", router_sim.EVAL_PAIRED, "a_to_c_parsefail_human"),
    ):
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


@_needs_demo
def test_policies_copy_router_sim_and_the_frozen_thresholds(payload):
    cfg = cost_model.load_cost_config()
    policies_doc = payload["policies.json"]
    assert policies_doc["op_version"] == demo_build.OP_VERSION
    assert policies_doc["cost_defaults"]["c_misroute"] == cfg.c_misroute_usd
    assert policies_doc["cost_defaults"]["c_human"] == cfg.c_human_usd
    assert policies_doc["cost_defaults"]["sha256"] == cfg.sha256

    by_key = {p["key"]: p for p in policies_doc["policies"]}
    for key, eval_set, policy_name in (
        ("a_to_human", router_sim.EVAL_FULL, "a_to_human"),
        ("a_to_c_haiku", router_sim.EVAL_PAIRED, "a_to_c_parsefail_human"),
    ):
        path = router_sim.DEFAULT_ROUTER_DIR / router_sim.result_filename(
            eval_set, cfg, demo_build.OP_VERSION)
        policy = json.loads(path.read_text(encoding="utf-8"))["policies"][policy_name]
        block = by_key[key]

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
    assert [s["slot"] for s in drift["pending_series"]] == ["tier_b1", "tier_b2"]


@_needs_demo
def test_calibration_ece_and_brier_are_copied_from_the_run_records(payload, resolved):
    exhibits = payload["calibration.json"]["exhibits"]
    assert len(exhibits) >= 2
    by_run = {r["run_id"]: r for r in predictions.load_records(RESULTS_PATH)}
    for exhibit in exhibits:
        record = by_run[exhibit["run_id"]]
        for key in ("ece", "brier"):
            logged = record["metrics"][key]
            assert exhibit[key] == {"point": logged["point"], "ci_lo": logged["ci_lo"],
                                    "ci_hi": logged["ci_hi"]}, f"{exhibit['key']}.{key}"
        assert exhibit["slice"] == record["dataset"]["split"]
        assert exhibit["calibration"] in {"raw", "isotonic"}


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
    allowed = {("A", "answered"),
               ("A", "escalated", "C", "answered"),
               ("A", "escalated", "C", "human")}
    tau = payload["policies.json"]
    frozen_tau = next(p["tau"]["value"] for p in tau["policies"] if p["key"] == "a_to_c_haiku")
    for sample in payload["samples.json"]["samples"]:
        router = sample["router"]
        assert router["op_version"] == "v2-isocal"
        assert router["policy"] == "a_to_c_haiku"
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
