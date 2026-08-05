"""Taxonomy harmonization: maps historical CFPB `Product` values into stable
routing classes defined in taxonomy_map.yaml (UPGRADE_PLAN.md §5)."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAXONOMY_PATH = REPO_ROOT / "taxonomy_map.yaml"
DEFAULT_PARQUET_PATH = REPO_ROOT / "data" / "ingest" / "narratives.parquet"
DEFAULT_STATS_PATH = REPO_ROOT / "data" / "ingest" / "taxonomy_stats.yaml"


@dataclass(frozen=True)
class Taxonomy:
    product_to_class: dict[str, str]
    dropped: frozenset[str]
    classes: tuple[str, ...]
    version: int


def load_taxonomy(path: Path) -> Taxonomy:
    """Load and validate a taxonomy map YAML file."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())

    version = raw["version"]
    raw_classes = raw.get("classes", {})
    raw_dropped = raw.get("dropped_products", [])

    product_to_class: dict[str, str] = {}
    for class_name, class_def in raw_classes.items():
        products = class_def.get("products", [])
        if not products:
            raise ValueError(f"class {class_name!r} has no products (empty class)")
        for product in products:
            if product in product_to_class:
                raise ValueError(
                    f"product {product!r} appears in more than one class: "
                    f"{product_to_class[product]!r} and {class_name!r}"
                )
            product_to_class[product] = class_name

    dropped = frozenset(entry["product"] for entry in raw_dropped)
    overlap = set(product_to_class) & dropped
    if overlap:
        raise ValueError(f"product(s) both mapped and dropped: {sorted(overlap)}")

    classes = tuple(sorted(raw_classes.keys()))

    return Taxonomy(
        product_to_class=product_to_class,
        dropped=dropped,
        classes=classes,
        version=version,
    )


def validate_coverage(taxonomy: Taxonomy, products: set[str]) -> None:
    """Raise ValueError listing any product not mapped and not dropped."""
    known = set(taxonomy.product_to_class) | taxonomy.dropped
    unknown = products - known
    if unknown:
        raise ValueError(
            f"{len(unknown)} product(s) not covered by taxonomy map: {sorted(unknown)}"
        )


def compute_stats(taxonomy: Taxonomy, parquet_path: Path) -> dict:
    """Query narratives.parquet via DuckDB and compute deterministic taxonomy stats."""
    parquet_path = Path(parquet_path)
    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=true")

        products = {
            r[0]
            for r in con.execute(
                f"SELECT DISTINCT product FROM read_parquet('{parquet_path}')"
            ).fetchall()
        }
        validate_coverage(taxonomy, products)

        rows_total = con.execute(
            f"SELECT count(*) FROM read_parquet('{parquet_path}')"
        ).fetchone()[0]

        product_year_counts = con.execute(
            f"""
            SELECT product, extract(year FROM date_received) AS year, count(*)
            FROM read_parquet('{parquet_path}')
            GROUP BY product, year
            ORDER BY product, year
            """
        ).fetchall()
    finally:
        con.close()

    class_counts: dict[str, int] = {cls: 0 for cls in taxonomy.classes}
    class_year_matrix: dict[str, dict[int, int]] = {cls: {} for cls in taxonomy.classes}
    rows_mapped = 0
    rows_dropped = 0
    n_products_mapped = 0
    n_products_dropped = 0
    seen_products: set[str] = set()

    for product, year, count in product_year_counts:
        seen_products.add(product)
        if product in taxonomy.product_to_class:
            cls = taxonomy.product_to_class[product]
            class_counts[cls] += count
            class_year_matrix[cls][int(year)] = class_year_matrix[cls].get(int(year), 0) + count
            rows_mapped += count
        elif product in taxonomy.dropped:
            rows_dropped += count

    n_products_mapped = len(set(taxonomy.product_to_class) & seen_products)
    n_products_dropped = len(taxonomy.dropped & seen_products)

    return {
        "map_version": taxonomy.version,
        "n_classes": len(taxonomy.classes),
        "n_products_mapped": n_products_mapped,
        "n_products_dropped": n_products_dropped,
        "rows_total": int(rows_total),
        "rows_mapped": int(rows_mapped),
        "rows_dropped": int(rows_dropped),
        "class_counts": dict(sorted(class_counts.items())),
        "class_year_matrix": {
            cls: dict(sorted(years.items()))
            for cls, years in sorted(class_year_matrix.items())
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m triage_lab.taxonomy")
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY_PATH)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_STATS_PATH)
    args = parser.parse_args(argv)

    taxonomy = load_taxonomy(args.taxonomy)
    stats = compute_stats(taxonomy, args.parquet)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(stats, sort_keys=True))

    print(
        f"coverage: {stats['n_products_mapped']} mapped + "
        f"{stats['n_products_dropped']} dropped products; "
        f"{stats['rows_mapped']}/{stats['rows_total']} rows mapped, "
        f"{stats['rows_dropped']} rows dropped"
    )
    print("class_counts:")
    for cls, count in stats["class_counts"].items():
        print(f"  {cls:24s} {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
