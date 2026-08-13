"""Live in-browser inference export tests: schema/shape checks over demo/live/.

CI-safe: everything under `demo/` is committed, so these tests run WITHOUT `data/`
present. They only read files, never regenerate them (regeneration is `make demo-live`,
a ~35 min SAGA refit that needs `data/`).
"""

from __future__ import annotations

import base64
import hashlib
import json

import pytest

from triage_lab import demo_build

REPO_ROOT = demo_build.REPO_ROOT
LIVE_DIR = REPO_ROOT / "demo" / "live"

CLASS_LABELS = [
    "card",
    "credit_reporting",
    "debt_collection",
    "deposit_account",
    "money_service",
    "mortgage",
    "payday_personal_loan",
    "student_loan",
    "vehicle_loan",
]

N_CLASSES = len(CLASS_LABELS)
VOCAB_SIZE = 150_000


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# demo/live/tier_a/tier_a_live.json
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tier_a_live():
    return _load(LIVE_DIR / "tier_a" / "tier_a_live.json")


def test_tier_a_live_exists_and_parses(tier_a_live):
    assert isinstance(tier_a_live, dict)


def test_tier_a_live_schema_version(tier_a_live):
    assert tier_a_live["schema_version"] == "live-v1"


def test_tier_a_live_class_labels(tier_a_live):
    assert tier_a_live["class_labels"] == CLASS_LABELS


def test_tier_a_live_vocab_sizes(tier_a_live):
    assert len(tier_a_live["word"]["vocab"]) == VOCAB_SIZE
    assert len(tier_a_live["char"]["vocab"]) == VOCAB_SIZE


def test_tier_a_live_idf_b64_byte_lengths(tier_a_live):
    for block in ("word", "char"):
        idf = base64.b64decode(tier_a_live[block]["idf_b64"])
        assert len(idf) == 4 * VOCAB_SIZE


def test_tier_a_live_coef_b64_byte_length(tier_a_live):
    coef = base64.b64decode(tier_a_live["coef_b64"])
    assert len(coef) == 4 * N_CLASSES * 300_000


def test_tier_a_live_intercept_length(tier_a_live):
    assert len(tier_a_live["intercept"]) == N_CLASSES


def test_tier_a_live_calibration_shape(tier_a_live):
    cal = tier_a_live["calibration"]
    assert cal["method"] == "isotonic"
    per_class = cal["per_class"]
    assert len(per_class) == N_CLASSES
    for entry in per_class:
        x, y = entry["x"], entry["y"]
        assert len(x) == len(y)
        assert all(x[i] < x[i + 1] for i in range(len(x) - 1)), "x must be strictly increasing"


def test_tier_a_live_provenance(tier_a_live):
    prov = tier_a_live["provenance"]
    assert prov["source_run_id"].startswith("8e4d6345")
    assert prov["verify"]["label_exact"] == 200


# ---------------------------------------------------------------------------
# demo/live/tier_b2/live_config.json
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tier_b2_live_config():
    return _load(LIVE_DIR / "tier_b2" / "live_config.json")


def test_tier_b2_live_config_temperature(tier_b2_live_config):
    temp = tier_b2_live_config["temperature"]
    assert isinstance(temp, float)
    assert 1.0 < temp < 2.0


def test_tier_b2_live_config_class_labels(tier_b2_live_config):
    assert tier_b2_live_config["class_labels"] == CLASS_LABELS


def test_tier_b2_live_config_provenance_parity_present(tier_b2_live_config):
    assert "parity" in tier_b2_live_config["provenance"]


# ---------------------------------------------------------------------------
# demo/live/tier_b2/python_int8_curated.json
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tier_b2_python_int8_curated():
    return _load(LIVE_DIR / "tier_b2" / "python_int8_curated.json")


def test_tier_b2_python_int8_curated_has_200_predictions(tier_b2_python_int8_curated):
    preds = tier_b2_python_int8_curated["predictions"]
    assert len(preds) == 200
    for entry in preds:
        assert "complaint_id" in entry
        assert entry["label"] in CLASS_LABELS
        assert 0.0 < entry["p_max"] <= 1.0


# ---------------------------------------------------------------------------
# demo/live/tier_b2/model.int8.onnx
# ---------------------------------------------------------------------------

def test_tier_b2_onnx_sha256_matches_live_config(tier_b2_live_config):
    onnx_path = LIVE_DIR / "tier_b2" / "model.int8.onnx"
    assert onnx_path.is_file()
    digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    expected = tier_b2_live_config["provenance"]["parity"]["model_int8_onnx_sha256"]
    assert digest == expected


# ---------------------------------------------------------------------------
# Vendored onnxruntime-web assets
# ---------------------------------------------------------------------------

def test_vendor_ort_assets_present():
    ort_dir = REPO_ROOT / "demo" / "vendor" / "ort"
    for name in ("ort.wasm.min.js", "ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"):
        assert (ort_dir / name).is_file(), f"missing vendored asset {name}"


# ---------------------------------------------------------------------------
# Live UI assets
# ---------------------------------------------------------------------------

def test_live_js_and_agreement_html_present():
    assert (REPO_ROOT / "demo" / "assets" / "live.js").is_file()
    assert (LIVE_DIR / "agreement.html").is_file()


# ---------------------------------------------------------------------------
# demo/live/agreement_report.json
#
# NOTE: this file is generated by an in-browser run, not by `make demo-live`. As of
# writing it does not yet exist on disk, but it WILL exist before this change is
# committed. This assertion is deliberately hard (no skip): a missing/malformed
# agreement report is a real gap, not an environment limitation.
# ---------------------------------------------------------------------------

def test_agreement_report_exists_and_has_minimal_schema():
    path = LIVE_DIR / "agreement_report.json"
    assert path.is_file(), "demo/live/agreement_report.json must be committed"
    report = _load(path)

    tier_a_agreement = report["tier_a"]["label_agreement_vs_official"]
    assert isinstance(tier_a_agreement, (int, float))
    assert 0.0 <= tier_a_agreement <= 1.0

    tier_b2_agreement = report["tier_b2"]["vs_official_fp32"]["label_agreement_vs_official"]
    assert isinstance(tier_b2_agreement, (int, float))
    assert 0.0 <= tier_b2_agreement <= 1.0
