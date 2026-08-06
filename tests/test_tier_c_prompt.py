"""Tier C prompt bundle: hashing, schema<->taxonomy lock, exemplar freeze + leakage.

The data-dependent checks (exemplar membership / leakage) need the real splits and
are skipped when data/splits is absent, mirroring how the rest of the suite keeps
CI green without the multi-GB corpus.
"""

from __future__ import annotations

import json
import subprocess
import sys

import duckdb
import pytest

from triage_lab import tier_c_prompt as tcp
from triage_lab.taxonomy import DEFAULT_TAXONOMY_PATH, load_taxonomy

VERSION = "v1"
SPLITS_DIR = tcp.DEFAULT_SPLITS_DIR
_HAS_TRAIN = (SPLITS_DIR / "train.parquet").exists() and (
    SPLITS_DIR / "splits_stats.yaml"
).exists()
_needs_data = pytest.mark.skipif(not _HAS_TRAIN, reason="real data/splits not present")


def _schema_validate(obj: dict, schema: dict) -> bool:
    """Minimal strict check: exactly {'label': <enum member>} and nothing else."""
    if set(obj) != {"label"}:
        return False
    return obj["label"] in schema["properties"]["label"]["enum"]


# ---------------------------------------------------------------------------
# Hash determinism
# ---------------------------------------------------------------------------

def test_hashes_are_deterministic():
    a = tcp.load_prompt_bundle(VERSION)
    b = tcp.load_prompt_bundle(VERSION)
    assert a.file_sha256 == b.file_sha256
    assert a.bundle_sha256 == b.bundle_sha256
    assert len(a.bundle_sha256) == 64


def test_bundle_sha_covers_all_three_files():
    bundle = tcp.load_prompt_bundle(VERSION)
    assert set(bundle.file_sha256) == {
        tcp.PROMPT_FILE,
        tcp.SCHEMA_FILE,
        tcp.EXEMPLARS_FILE,
    }
    # bundle_sha256 is a pure function of the three per-file hashes (sorted).
    assert bundle.bundle_sha256 == tcp._bundle_sha256(bundle.file_sha256)
    # Perturbing any file hash changes the bundle hash.
    tampered = dict(bundle.file_sha256)
    tampered[tcp.SCHEMA_FILE] = "0" * 64
    assert tcp._bundle_sha256(tampered) != bundle.bundle_sha256


# ---------------------------------------------------------------------------
# Schema <-> taxonomy lock
# ---------------------------------------------------------------------------

def test_schema_enum_matches_taxonomy_exactly():
    bundle = tcp.load_prompt_bundle(VERSION)
    enum = bundle.schema["properties"]["label"]["enum"]
    expected = sorted(load_taxonomy(DEFAULT_TAXONOMY_PATH).classes)
    assert enum == expected


def test_schema_is_strict_mode_shaped():
    bundle = tcp.load_prompt_bundle(VERSION)
    s = bundle.schema
    assert s["type"] == "object"
    assert s["additionalProperties"] is False
    assert s["required"] == ["label"]
    assert set(s["properties"]) == {"label"}


# ---------------------------------------------------------------------------
# build_messages structure
# ---------------------------------------------------------------------------

def test_build_messages_zero_shot():
    bundle = tcp.load_prompt_bundle(VERSION)
    msgs = tcp.build_messages(bundle, "some narrative text", num_exemplars=0)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == bundle.system
    assert "some narrative text" in msgs[-1]["content"]


def test_build_messages_full_shot_alternates_and_is_schema_valid():
    bundle = tcp.load_prompt_bundle(VERSION)
    k = len(bundle.exemplars)
    msgs = tcp.build_messages(bundle, "query narrative", num_exemplars=k)
    roles = [m["role"] for m in msgs]
    # system, then k*(user, assistant) pairs, then the final user query.
    assert roles[0] == "system"
    assert roles[-1] == "user"
    assert roles[1 : 1 + 2 * k] == ["user", "assistant"] * k
    assert len(msgs) == 1 + 2 * k + 1
    # Every assistant exemplar turn parses as JSON valid under schema.json.
    for m in msgs:
        if m["role"] == "assistant":
            obj = json.loads(m["content"])
            assert _schema_validate(obj, bundle.schema)


def test_build_messages_rejects_too_many_exemplars():
    bundle = tcp.load_prompt_bundle(VERSION)
    with pytest.raises(ValueError, match="exceeds available"):
        tcp.build_messages(bundle, "x", num_exemplars=len(bundle.exemplars) + 1)


# ---------------------------------------------------------------------------
# Exemplar coverage + freeze behavior
# ---------------------------------------------------------------------------

def test_exemplars_one_per_class():
    bundle = tcp.load_prompt_bundle(VERSION)
    labels = sorted(load_taxonomy(DEFAULT_TAXONOMY_PATH).classes)
    got = sorted(e["label"] for e in bundle.exemplars)
    assert got == labels  # exactly one per class, no dup, no miss


def test_generate_refuses_to_overwrite_existing(monkeypatch, tmp_path, capsys):
    """--generate-exemplars must never clobber a frozen file (CLAUDE.md rule 4)."""
    # Point PROMPTS_ROOT at a scratch dir seeded with a sentinel exemplars.json.
    fake_root = tmp_path / "prompts" / "tier_c"
    (fake_root / VERSION).mkdir(parents=True)
    sentinel = fake_root / VERSION / tcp.EXEMPLARS_FILE
    sentinel.write_text('{"sentinel": true}\n')
    monkeypatch.setattr(tcp, "PROMPTS_ROOT", fake_root)

    rc = tcp.main(["--version", VERSION, "--generate-exemplars"])
    assert rc == 1
    assert "refusing to overwrite" in capsys.readouterr().err
    assert sentinel.read_text() == '{"sentinel": true}\n'  # untouched


# ---------------------------------------------------------------------------
# Data-dependent: leakage + byte-identical regeneration
# ---------------------------------------------------------------------------

def _split_ids(name: str) -> set[int]:
    path = SPLITS_DIR / f"{name}.parquet"
    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        rows = con.execute(
            f"SELECT complaint_id FROM read_parquet('{path}')"
        ).fetchall()
    finally:
        con.close()
    return {int(r[0]) for r in rows}


@_needs_data
def test_exemplar_ids_in_train_and_absent_from_eval_splits():
    bundle = tcp.load_prompt_bundle(VERSION)
    ex_ids = {int(e["complaint_id"]) for e in bundle.exemplars}
    train_ids = _split_ids("train")
    assert ex_ids <= train_ids  # every exemplar is a TRAIN row
    for eval_split in ("cal", "test_iid", "test_drift_2023", "test_drift_2024",
                       "test_drift_2025", "test_drift_2026h1", "test_postcutoff"):
        assert not (ex_ids & _split_ids(eval_split)), f"exemplar leaked into {eval_split}"


@_needs_data
def test_frozen_exemplars_regenerate_byte_identically():
    rc = tcp.main(["--version", VERSION, "--verify-exemplars"])
    assert rc == 0


@_needs_data
def test_selection_reports_train_sha_from_stats():
    _, selection = tcp.select_exemplars(SPLITS_DIR)
    import yaml

    stats = yaml.safe_load((SPLITS_DIR / "splits_stats.yaml").read_text())
    assert selection["train_sha256"] == stats["splits"]["train"]["sha256"]


@_needs_data
def test_selection_rejects_tampered_train_sha(tmp_path, monkeypatch):
    """The integrity gate must fire if TRAIN's sha drifts from splits_stats.yaml."""
    import shutil

    import yaml

    fake = tmp_path / "splits"
    fake.mkdir()
    shutil.copy(SPLITS_DIR / "train.parquet", fake / "train.parquet")
    stats = yaml.safe_load((SPLITS_DIR / "splits_stats.yaml").read_text())
    stats["splits"]["train"]["sha256"] = "0" * 64
    (fake / "splits_stats.yaml").write_text(yaml.safe_dump(stats))
    with pytest.raises(ValueError, match="integrity check failed"):
        tcp.select_exemplars(fake)


# ---------------------------------------------------------------------------
# CLI hash printing
# ---------------------------------------------------------------------------

def test_cli_prints_bundle_hash():
    proc = subprocess.run(
        [sys.executable, "-m", "triage_lab.tier_c_prompt", "--version", VERSION],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    bundle = tcp.load_prompt_bundle(VERSION)
    assert bundle.bundle_sha256 in proc.stdout
