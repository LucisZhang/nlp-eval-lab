import pytest
import yaml

from triage_lab import taxonomy

REPO_ROOT_TAXONOMY = taxonomy.DEFAULT_TAXONOMY_PATH


def test_load_real_taxonomy_map():
    tax = taxonomy.load_taxonomy(REPO_ROOT_TAXONOMY)
    assert tax.version == 1
    assert len(tax.classes) == 9
    assert len(tax.product_to_class) == 20
    assert len(tax.dropped) == 1
    assert len(tax.product_to_class) + len(tax.dropped) == 21
    expected_classes = {
        "credit_reporting",
        "debt_collection",
        "mortgage",
        "card",
        "deposit_account",
        "money_service",
        "student_loan",
        "vehicle_loan",
        "payday_personal_loan",
    }
    assert set(tax.classes) == expected_classes


def _write_map(tmp_path, content: dict, name="taxonomy_map.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(content, sort_keys=False))
    return path


def test_duplicate_product_across_classes_raises(tmp_path):
    content = {
        "version": 1,
        "classes": {
            "a": {"products": ["X"]},
            "b": {"products": ["X"]},
        },
        "dropped_products": [],
    }
    path = _write_map(tmp_path, content)
    with pytest.raises(ValueError, match="more than one class"):
        taxonomy.load_taxonomy(path)


def test_product_mapped_and_dropped_raises(tmp_path):
    content = {
        "version": 1,
        "classes": {
            "a": {"products": ["X"]},
        },
        "dropped_products": [{"product": "X", "reason": "test"}],
    }
    path = _write_map(tmp_path, content)
    with pytest.raises(ValueError, match="both mapped and dropped"):
        taxonomy.load_taxonomy(path)


def test_empty_class_raises(tmp_path):
    content = {
        "version": 1,
        "classes": {
            "a": {"products": []},
        },
        "dropped_products": [],
    }
    path = _write_map(tmp_path, content)
    with pytest.raises(ValueError, match="empty class"):
        taxonomy.load_taxonomy(path)


def test_validate_coverage_raises_on_unknown_product(tmp_path):
    content = {
        "version": 1,
        "classes": {
            "a": {"products": ["X"]},
        },
        "dropped_products": [{"product": "Y", "reason": "test"}],
    }
    path = _write_map(tmp_path, content)
    tax = taxonomy.load_taxonomy(path)
    with pytest.raises(ValueError, match="not covered by taxonomy map"):
        taxonomy.validate_coverage(tax, {"X", "Y", "Z"})


def test_validate_coverage_passes_when_all_covered(tmp_path):
    content = {
        "version": 1,
        "classes": {
            "a": {"products": ["X"]},
        },
        "dropped_products": [{"product": "Y", "reason": "test"}],
    }
    path = _write_map(tmp_path, content)
    tax = taxonomy.load_taxonomy(path)
    # Should not raise, including the dropped product Y.
    taxonomy.validate_coverage(tax, {"X", "Y"})
