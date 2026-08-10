"""Validate an extracted Tier B training-results bundle before placing checkpoints.

Usage:
    uv run python scripts/validate_tier_b_bundle.py <extracted_bundle_dir>

Checks, fail-loud (non-zero exit on any mismatch):
  1. every file listed in the bundle's manifest.json exists, size- and sha256-matches;
  2. no file exists in the bundle that the manifest does not list;
  3. the four bundled config YAMLs are byte-identical to the repo's frozen configs/;
  4. each manifest job's config_sha256 equals the frozen config's sha and is
     verified_completed;
  5. each run's training_meta.json seed / base model / data-manifest sha matches the
     frozen config and the frozen tier_b_kit manifest, and status.json says completed.

Used for the 2026-08-10 ingest of tier_b_training_results_20260807T191924Z.tar.gz
(EXPERIMENT_LOG.md); rerunnable against a re-extraction of the retained tarball.
"""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CONFIG_BY_RUN = {
    "tier_b1_sa": "tier_b1_modernbert_sa.yaml",
    "tier_b1_sb": "tier_b1_modernbert_sb.yaml",
    "tier_b1_sc": "tier_b1_modernbert_sc.yaml",
    "tier_b2_s0": "tier_b2_distilbert_s0.yaml",
}
EXPECTED = {
    "tier_b1_sa": {"seed": 20260805, "base": "answerdotai/ModernBERT-base"},
    "tier_b1_sb": {"seed": 20260806, "base": "answerdotai/ModernBERT-base"},
    "tier_b1_sc": {"seed": 20260807, "base": "answerdotai/ModernBERT-base"},
    "tier_b2_s0": {"seed": 20260805, "base": "distilbert-base-uncased"},
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(staging: Path) -> int:
    failures: list[str] = []
    manifest = json.loads((staging / "manifest.json").read_text())

    listed = set()
    for entry in manifest["files"]:
        p = staging / entry["archive_path"]
        listed.add(entry["archive_path"])
        if not p.exists():
            failures.append(f"MISSING: {entry['archive_path']}")
            continue
        if p.stat().st_size != entry["size"]:
            failures.append(f"SIZE MISMATCH: {entry['archive_path']}")
        if sha256(p) != entry["sha256"]:
            failures.append(f"SHA MISMATCH: {entry['archive_path']}")
    print(f"[1] manifest re-hash: {len(manifest['files'])} files checked")

    on_disk = {str(p.relative_to(staging)) for p in staging.rglob("*") if p.is_file()}
    extra = on_disk - listed - {"manifest.json"}
    if extra:
        failures.append(f"UNLISTED FILES IN ARCHIVE: {sorted(extra)}")
    print(f"[2] unlisted-file check: extra={sorted(extra) if extra else 'none'}")

    for name in CONFIG_BY_RUN.values():
        if (staging / "metadata/configs" / name).read_bytes() != (REPO / "configs" / name).read_bytes():
            failures.append(f"CONFIG BYTE MISMATCH vs frozen configs/: {name}")
    print("[3] config byte-compare done")

    job_by_name = {j["name"]: j for j in manifest["jobs"]}
    for run, cfg in CONFIG_BY_RUN.items():
        j = job_by_name.get(run)
        if j is None:
            failures.append(f"JOB MISSING IN MANIFEST: {run}")
            continue
        if j["config_sha256"] != sha256(REPO / "configs" / cfg):
            failures.append(f"JOB CONFIG SHA MISMATCH: {run}")
        if not j["verified_completed"]:
            failures.append(f"JOB NOT verified_completed: {run}")
    print("[4] job config-sha + completion flags done")

    kit_manifest_sha = sha256(REPO / "data/tier_b_kit/manifest.json")
    if manifest["data_manifest_sha256"] != kit_manifest_sha:
        failures.append("BUNDLE data_manifest_sha256 != frozen data/tier_b_kit/manifest.json")
    for run, exp in EXPECTED.items():
        d = staging / "checkpoints" / run
        meta = json.loads((d / "training_meta.json").read_text())
        status = json.loads((d / "status.json").read_text())
        if meta.get("seed") != exp["seed"]:
            failures.append(f"SEED MISMATCH {run}: meta={meta.get('seed')}")
        if meta.get("base_model") != exp["base"]:
            failures.append(f"BASE MODEL MISMATCH {run}: meta={meta.get('base_model')}")
        if meta.get("data_manifest_sha256") != manifest["data_manifest_sha256"]:
            failures.append(f"DATA MANIFEST SHA MISMATCH {run}")
        if status.get("exit_status") != "completed" or meta.get("run_status") != "completed":
            failures.append(f"RUN NOT completed {run}")
    print("[5] per-run meta/status checks done")

    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for f in failures:
            print(" -", f)
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(Path(sys.argv[1])))
