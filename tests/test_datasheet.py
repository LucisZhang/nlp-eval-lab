"""Phase-0: the dataset datasheet generator is deterministic and complete.

Renders docs/DATASHEET.md from small fixture YAMLs (never the real multi-GB data),
mirroring the fixture pattern in test_splits.py / test_leakage.py. Asserts (a) the
generator is byte-for-byte deterministic across repeated runs and (b) the
§5-mandated facts are present: snapshot SHA-256, dedup rate, the scrubbing note
(mentioning the `XXXX` mask and that the lab did no redaction), and every taxonomy
class.
"""

import pytest
import yaml

from triage_lab import datasheet

# --------------------------------------------------------------------------- #
# Fixture artifacts (schemas match the real pipeline outputs, minimized).
# --------------------------------------------------------------------------- #

_MANIFEST = {
    "url": "https://files.consumerfinance.gov/ccdb/complaints.csv.zip",
    "filename": "complaints.csv.zip",
    "download_date": "2026-08-05",
    "sha256": "b4d1eac8ef9f2e7710224848d321f355c480ff95baa742a2f6d1e3c704705600",
    "size_bytes": 1408215102,
}

_TAXONOMY = {
    "version": 1,
    "classes": {
        "mortgage": {
            "products": ["Mortgage"],
            "notes": "Single stable label across 2015-2026; no era crossing.",
        },
        "credit_reporting": {
            "products": [
                "Credit reporting",
                "Credit reporting or other personal consumer reports",
            ],
            "notes": "Renamed concept folded together for TRAIN continuity.",
        },
    },
    "dropped_products": [
        {"product": "Other financial service", "reason": "Pre-2017 residual grab-bag."}
    ],
}

_INGEST_STATS = {
    "total_rows_parsed": 16872863,
    "parser_rejected_lines": 1,
    "malformed_key_rows_dropped": 3,
    "narrative_rows": 3830206,
    "min_date_received": "2015-03-19",
    "max_date_received": "2026-07-27",
    "output_sha256": "ce97d5caabebe746d187853fe9524cd02242aef3dab0ecdcf28168a4b67543e1",
}

_DEDUP_STATS = {
    "normalization": "v1",
    "input_rows": 3830206,
    "output_rows": 1983496,
    "removed_rows": 1846710,
    "dedup_rate": 0.4821437802562055,
    "n_clusters": 1983496,
    "params": {
        "num_perm": 128,
        "minhash_seed": 1,
        "hashfunc": "sha1_hash32",
        "shingle_size": 5,
        "num_bands": 8,
        "rows_per_band": 16,
        "lsh_threshold": 0.9,
        "chunk_size": 512,
        "datasketch_version": "2.0.0",
    },
}

_SPLITS_STATS = {
    "seed": 20260805,
    "quota_scheme": "Largest-remainder apportionment; SEED = 20260805.",
    "postcutoff_rationale": (
        "TEST-POSTCUTOFF = date_received >= 2026-02-01. Tier C uses "
        "claude-haiku-4-5-20251001; boundary set after the latest training cutoff."
    ),
    "overlap_note": (
        "test_drift_2026h1 ∩ test_postcutoff = 15409 complaint_ids by design."
    ),
    "input_sha256": "170f66cd95f8ba2ff47929d3cbe903030400a960e3d1657df53484290a22d89f",
    "taxonomy_version": 1,
    "rows_deduped_total": 1983496,
    "rows_dropped_product": 291,
    "rows_mapped_total": 1983205,
    "rows_before_train_start": 18657,
    "rows_in_no_split": 1360593,
    "splits": {
        "train": {
            "start": "2015-07-01",
            "end": "2021-12-31",
            "strata": "class_year",
            "target": 300000,
            "n_candidates": 678370,
            "n_selected": 300000,
            "class_year_counts": {
                "mortgage": {2015: 3435, 2016: 6947},
                "credit_reporting": {2015: 2683, 2016: 5856},
            },
            "sha256": "939186e72a78c680107dd3ae0087485fabd5f50dc1cca8dab8ca0205956a342b",
        },
        "cal": {
            "start": "2022-01-01",
            "end": "2022-06-30",
            "strata": "none",
            "target": None,
            "n_candidates": 86972,
            "n_selected": 86972,
            "class_year_counts": {
                "mortgage": {2022: 6394},
                "credit_reporting": {2022: 40609},
            },
            "sha256": "d7c24d6db05f337ac3fca1922b16c4324775e2007833af1eae684ddc37533c94",
        },
        "test_postcutoff": {
            "start": "2026-02-01",
            "end": None,
            "strata": "none",
            "target": None,
            "n_candidates": 66606,
            "n_selected": 66606,
            "class_year_counts": {
                "mortgage": {2026: 4945},
                "credit_reporting": {2026: 1970},
            },
            "sha256": "ccbd779deff812e8b41dc6aba8c8d9c5c1cf0dcb294063cda0ef6af50dcdc530",
        },
    },
}


def _write_fixtures(tmp_path, *, with_ingest=True):
    paths = {}
    for name, obj in [
        ("SNAPSHOT_MANIFEST.yaml", _MANIFEST),
        ("taxonomy_map.yaml", _TAXONOMY),
        ("dedup_stats.yaml", _DEDUP_STATS),
        ("splits_stats.yaml", _SPLITS_STATS),
    ]:
        p = tmp_path / name
        p.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True))
        paths[name] = p
    if with_ingest:
        p = tmp_path / "ingest_stats.yaml"
        p.write_text(yaml.safe_dump(_INGEST_STATS, sort_keys=False))
        paths["ingest_stats.yaml"] = p
    else:
        paths["ingest_stats.yaml"] = tmp_path / "missing_ingest_stats.yaml"
    return paths


def _run(tmp_path, out_name="DATASHEET.md", **kw):
    paths = _write_fixtures(tmp_path, **kw)
    out = tmp_path / out_name
    return datasheet.run(
        out_path=out,
        manifest_path=paths["SNAPSHOT_MANIFEST.yaml"],
        taxonomy_path=paths["taxonomy_map.yaml"],
        ingest_stats_path=paths["ingest_stats.yaml"],
        dedup_stats_path=paths["dedup_stats.yaml"],
        splits_stats_path=paths["splits_stats.yaml"],
    ), out


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #

def test_double_run_byte_identical(tmp_path):
    text_a, out_a = _run(tmp_path, out_name="a.md")
    text_b, out_b = _run(tmp_path, out_name="b.md")
    assert out_a.read_bytes() == out_b.read_bytes()
    assert text_a == text_b
    # Returned text equals what was written to disk.
    assert out_a.read_text() == text_a


def test_no_generation_timestamp_leaks(tmp_path):
    # Determinism guard: nothing time-derived should appear. The only date in the
    # document is the frozen snapshot download date from the manifest.
    text, _ = _run(tmp_path)
    assert "2026-08-05" in text  # manifest download_date, allowed
    # A regeneration must still be byte-identical (covered above); here assert the
    # generator carries no obvious wall-clock words.
    for banned in ("Generated on", "generated at", "datetime", "today"):
        assert banned not in text


# --------------------------------------------------------------------------- #
# Mandated content
# --------------------------------------------------------------------------- #

def test_snapshot_provenance_present(tmp_path):
    text, _ = _run(tmp_path)
    assert _MANIFEST["sha256"] in text
    assert _MANIFEST["url"] in text
    assert _MANIFEST["download_date"] in text


def test_scrubbing_note_present(tmp_path):
    text, _ = _run(tmp_path)
    assert "XXXX" in text
    # The lab must NOT claim it performed redaction.
    assert "opt-in" in text
    lowered = text.lower()
    assert "no redaction" in lowered
    assert "cfpb" in lowered


def test_license_present(tmp_path):
    text, _ = _run(tmp_path)
    assert "17 U.S.C. §105" in text
    assert "public domain" in text.lower()


def test_dedup_rate_present(tmp_path):
    text, _ = _run(tmp_path)
    # 0.4821437802562055 -> 48.21%
    assert "48.21%" in text
    assert "0.4821" in text


def test_every_taxonomy_class_and_dropped_present(tmp_path):
    text, _ = _run(tmp_path)
    for cls in _TAXONOMY["classes"]:
        assert cls in text
    # Dropped product + its rationale surface.
    assert "Other financial service" in text
    assert "residual grab-bag" in text


def test_class_year_matrix_and_split_table_present(tmp_path):
    text, _ = _run(tmp_path)
    # Split boundaries and per-split counts render.
    assert "2015-07-01" in text
    assert "2026-02-01" in text
    assert "300,000" in text  # train n_selected, comma-formatted
    # Postcutoff + overlap rationale carried through.
    assert "claude-haiku-4-5-20251001" in text
    assert "15409" in text or "15,409" in text


def test_optional_ingest_stats_absent_still_renders(tmp_path):
    text, _ = _run(tmp_path, with_ingest=False)
    # Core mandated facts still present without ingest_stats.
    assert _MANIFEST["sha256"] in text
    assert "48.21%" in text
    # Ingest-only field must be absent.
    assert "Rows parsed from CSV" not in text


def test_missing_required_input_fails_loud(tmp_path):
    paths = _write_fixtures(tmp_path)
    with pytest.raises(FileNotFoundError):
        datasheet.run(
            out_path=tmp_path / "out.md",
            manifest_path=tmp_path / "does_not_exist.yaml",
            taxonomy_path=paths["taxonomy_map.yaml"],
            ingest_stats_path=paths["ingest_stats.yaml"],
            dedup_stats_path=paths["dedup_stats.yaml"],
            splits_stats_path=paths["splits_stats.yaml"],
        )
