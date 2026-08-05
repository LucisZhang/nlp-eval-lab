"""Deterministic dataset datasheet generator (UPGRADE_PLAN.md §5, Phase 0).

Renders ``docs/DATASHEET.md`` purely from committed / pipeline-derived artifacts:

- ``SNAPSHOT_MANIFEST.yaml``      — source URL, download date, SHA-256, size
- ``taxonomy_map.yaml``          — every class/product mapping + dropped products
- ``data/ingest/ingest_stats.yaml``  — parse/filter accounting (optional)
- ``data/dedup/dedup_stats.yaml``    — dedup rate + MinHash/LSH parameters
- ``data/splits/splits_stats.yaml``  — per-split counts, boundaries, class×year

Output is byte-for-byte deterministic: no generation timestamp, no dict-order
nondeterminism (every mapping is iterated in explicit sorted order), and every
static sentence lives in the template below. The download date that appears in
the document is a property of the *snapshot* (from the manifest), never the time
this generator ran. It fails loudly if any required input is missing.

Follows "Datasheets for Datasets" (Gebru et al., 2021) section structure loosely
— Motivation, Composition, Collection, Preprocessing, Uses, Distribution /
License, Maintenance — but the §5-mandated facts are the contract.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "SNAPSHOT_MANIFEST.yaml"
DEFAULT_TAXONOMY_PATH = REPO_ROOT / "taxonomy_map.yaml"
DEFAULT_INGEST_STATS_PATH = REPO_ROOT / "data" / "ingest" / "ingest_stats.yaml"
DEFAULT_DEDUP_STATS_PATH = REPO_ROOT / "data" / "dedup" / "dedup_stats.yaml"
DEFAULT_SPLITS_STATS_PATH = REPO_ROOT / "data" / "splits" / "splits_stats.yaml"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "docs" / "DATASHEET.md"

# Canonical ordering of the split slices in the rendered tables (matches the
# temporal pipeline order in splits.SPECS; kept literal so the datasheet does not
# import the heavyweight splits module just for names).
_SPLIT_ORDER = (
    "train",
    "cal",
    "test_iid",
    "test_drift_2023",
    "test_drift_2024",
    "test_drift_2025",
    "test_drift_2026h1",
    "test_postcutoff",
)


# --------------------------------------------------------------------------- #
# Static prose (all sentences that are not derived from an artifact live here).
# --------------------------------------------------------------------------- #

_LICENSE_TEXT = (
    "**U.S. Government work — public domain (17 U.S.C. §105).** The CFPB Consumer "
    "Complaint Database is a work of the United States federal government and is not "
    "subject to domestic copyright protection. Redistribution of individual records "
    "and derived samples (including in the project demo) is therefore unrestricted. "
    "No additional dataset license applies."
)

_SCRUBBING_TEXT = (
    "Consumer narratives are **opt-in**: they appear only when the complainant "
    "affirmatively consented to publication. Before publication, **CFPB scrubs the "
    "narratives and masks personally identifiable information as the literal token "
    "`XXXX`** (variable-length runs such as `XXXX`/`XXXXXXXX` denote redacted spans). "
    "**This lab performed NO redaction, de-identification, or PII removal of its own** "
    "— it consumes the already-scrubbed CFPB text verbatim and makes no claim about "
    "the completeness of CFPB's masking. The dedup normalizer does collapse runs of "
    "the `x` mask to a single token for near-duplicate detection only; it never "
    "rewrites the stored narrative."
)

_MAPPING_FRAMING = (
    "CFPB renamed and merged its `Product` taxonomy several times over 2015–2026 "
    "(notably the 2017-04 revision and the 2023 credit-reporting consolidation). To "
    "keep the routing target stable across the temporal splits, historical `Product` "
    "values are collapsed into fixed routing **classes** by the frozen "
    "`taxonomy_map.yaml`. Every mapping decision and its rationale is reproduced "
    "below verbatim from that file; changing any mapping invalidates all downstream "
    "splits and results. The taxonomy churn itself is retained as a measured "
    "\"label-drift\" exhibit rather than hidden."
)

_DEDUP_FRAMING = (
    "CFPB narratives contain large volumes of near-duplicate template complaints "
    "(credit-report disputes especially). Near-duplicates are removed **before** "
    "splitting so that resubmitted or templated complaints cannot leak across the "
    "train/test boundary. Detection uses MinHash + LSH banding (datasketch): each "
    "narrative is normalized (lowercased, mask runs collapsed, non-alphanumerics "
    "stripped), shingled into word 5-grams, and signed with a fixed-seed 128-permutation "
    "MinHash; signatures are banded and union-found into near-duplicate clusters. Per "
    "cluster the earliest-dated record (tie-break: smallest complaint id) is kept and "
    "the rest dropped. The procedure is deterministic across runs and worker counts."
)

_MOTIVATION_TEXT = (
    "This datasheet documents the frozen dataset underlying a three-tier "
    "consumer-complaint triage evaluation lab. The task is to predict a stable routing "
    "class from a consumer's free-text complaint narrative, and to measure how model "
    "quality and routing cost degrade under real 2015→2026 distribution drift. The "
    "dataset is a single frozen snapshot of the public CFPB Consumer Complaint "
    "Database; no new data was collected and no consumers were contacted."
)

_USES_TEXT = (
    "The dataset supports supervised text classification (narrative → routing class), "
    "model calibration, confidence-cascade router evaluation against an explicit cost "
    "model, and temporal drift / prior-shift measurement across the frozen TEST-DRIFT "
    "and TEST-POSTCUTOFF slices. **Not suitable** for: any attempt to re-identify "
    "individuals (the text is opt-in and CFPB-scrubbed, and re-identification is out of "
    "scope and discouraged); demographic or fairness auditing (no demographic labels "
    "are present); or treating `Product` as ground-truth consumer intent (it is the "
    "company/CFPB routing category, harmonized here across taxonomy revisions)."
)

_MAINTENANCE_TEXT = (
    "The snapshot is **frozen**: it is pinned by SHA-256 in `SNAPSHOT_MANIFEST.yaml` "
    "and never refreshed within this lab. The temporal splits, sub-sampling seed list, "
    "and few-shot exemplar selections freeze the moment they are first materialized and "
    "are reproduced byte-identically by `make data`. Corrections are made by appending "
    "new artifacts, never by editing frozen ones. This datasheet is regenerated "
    "deterministically from the artifacts above and carries no independent state."
)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _load_required(path: Path, label: str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"datasheet: required input {label} not found at {path}. "
            "Run the Phase-0 pipeline (make data) first."
        )
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(  # noqa: TRY004 — malformed artifact, not a caller type error
            f"datasheet: {label} at {path} did not parse to a mapping"
        )
    return data


def _load_optional(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text())


def _collapse_ws(text: str) -> str:
    """Normalize a (possibly multi-line folded YAML) string to a single line.

    Deterministic: splits on any whitespace and rejoins with single spaces.
    """
    return " ".join(str(text).split())


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #

def _fmt_int(n) -> str:
    return f"{int(n):,}"


def _aggregate_class_year(splits_stats: dict) -> tuple[list[str], list[int], dict]:
    """Sum every split's class_year_counts into one class×year matrix.

    Returns (sorted classes, sorted years, matrix[class][year]). Cells sum the
    frozen per-split *selected* counts, so the documented test_drift_2026h1 ∩
    test_postcutoff overlap is counted in both — this is stated in the rendered note.
    """
    matrix: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    years: set[int] = set()
    for split in splits_stats["splits"].values():
        for cls, year_counts in split.get("class_year_counts", {}).items():
            for year, count in year_counts.items():
                matrix[cls][int(year)] += int(count)
                years.add(int(year))
    classes = sorted(matrix)
    return classes, sorted(years), matrix


def _render_class_year_table(splits_stats: dict) -> str:
    classes, years, matrix = _aggregate_class_year(splits_stats)
    header = "| class | " + " | ".join(str(y) for y in years) + " | total |"
    sep = "|---|" + "|".join("---:" for _ in years) + "|---:|"
    lines = [header, sep]
    for cls in classes:
        row_total = sum(matrix[cls].values())
        cells = [
            _fmt_int(matrix[cls][y]) if y in matrix[cls] else "·" for y in years
        ]
        lines.append(f"| {cls} | " + " | ".join(cells) + f" | {_fmt_int(row_total)} |")
    # Column totals row.
    col_totals = [
        _fmt_int(sum(matrix[cls].get(y, 0) for cls in classes)) for y in years
    ]
    grand = sum(sum(matrix[cls].values()) for cls in classes)
    lines.append(
        "| **total** | " + " | ".join(f"**{c}**" for c in col_totals)
        + f" | **{_fmt_int(grand)}** |"
    )
    return "\n".join(lines)


def _render_split_table(splits_stats: dict) -> str:
    header = (
        "| split | date_received range | strata | target | candidates | selected | sha256 |"
    )
    sep = "|---|---|---|---:|---:|---:|---|"
    lines = [header, sep]
    splits = splits_stats["splits"]
    for name in _SPLIT_ORDER:
        if name not in splits:
            continue
        s = splits[name]
        start = s["start"]
        end = s["end"] if s["end"] is not None else "open-ended"
        target = _fmt_int(s["target"]) if s["target"] is not None else "all"
        lines.append(
            f"| {name} | {start} → {end} | {s['strata']} | {target} | "
            f"{_fmt_int(s['n_candidates'])} | {_fmt_int(s['n_selected'])} | "
            f"`{s['sha256'][:12]}` |"
        )
    return "\n".join(lines)


def _render_taxonomy(taxonomy_raw: dict) -> str:
    lines: list[str] = []
    raw_classes = taxonomy_raw.get("classes", {})
    for class_name in sorted(raw_classes):
        class_def = raw_classes[class_name]
        products = list(class_def.get("products", []))
        notes = _collapse_ws(class_def.get("notes", "")) if class_def.get("notes") else ""
        lines.append(f"#### `{class_name}`")
        lines.append("")
        lines.append("Historical `Product` values mapped here:")
        lines.append("")
        for product in products:
            lines.append(f"- {product!r}")
        lines.append("")
        if notes:
            lines.append(f"*Rationale.* {notes}")
            lines.append("")
    return "\n".join(lines).rstrip()


def _render_dropped(taxonomy_raw: dict) -> str:
    dropped = taxonomy_raw.get("dropped_products", [])
    if not dropped:
        return "_None: every observed `Product` value is mapped to a class._"
    lines: list[str] = []
    for entry in sorted(dropped, key=lambda e: e["product"]):
        reason = _collapse_ws(entry.get("reason", ""))
        lines.append(f"- **{entry['product']!r}** — {reason}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Datasheet assembly
# --------------------------------------------------------------------------- #

def render_datasheet(
    manifest: dict,
    taxonomy_raw: dict,
    dedup_stats: dict,
    splits_stats: dict,
    ingest_stats: dict | None,
) -> str:
    """Return the full DATASHEET.md text. Pure function of its inputs."""
    dedup_rate = float(dedup_stats["dedup_rate"])
    params = dedup_stats.get("params", {})

    # Composition figures.
    rows_deduped_total = splits_stats["rows_deduped_total"]
    rows_mapped_total = splits_stats["rows_mapped_total"]
    rows_dropped_product = splits_stats["rows_dropped_product"]
    rows_before_train_start = splits_stats["rows_before_train_start"]
    rows_in_no_split = splits_stats["rows_in_no_split"]

    parts: list[str] = []

    parts.append("# Datasheet — CFPB Consumer Complaint Triage Dataset")
    parts.append("")
    parts.append(
        "> Deterministically generated by `python -m triage_lab.datasheet` from the "
        "frozen snapshot manifest and pipeline stats artifacts. Do not edit by hand; "
        "regenerate via `make datasheet`."
    )
    parts.append("")

    # --- Motivation ---
    parts.append("## 1. Motivation")
    parts.append("")
    parts.append(_MOTIVATION_TEXT)
    parts.append("")

    # --- Collection / Source ---
    parts.append("## 2. Collection & Source")
    parts.append("")
    parts.append(f"- **Source URL:** {manifest['url']}")
    parts.append(f"- **Snapshot file:** `{manifest['filename']}`")
    parts.append(f"- **Download date (frozen):** {manifest['download_date']}")
    parts.append(f"- **Snapshot SHA-256:** `{manifest['sha256']}`")
    parts.append(f"- **Snapshot size:** {_fmt_int(manifest['size_bytes'])} bytes")
    parts.append("")
    if ingest_stats is not None:
        parts.append(
            f"- **Rows parsed from CSV:** {_fmt_int(ingest_stats['total_rows_parsed'])}"
        )
        parts.append(
            f"- **Parser-rejected lines (malformed RFC-4180):** "
            f"{_fmt_int(ingest_stats['parser_rejected_lines'])}"
        )
        parts.append(
            f"- **Rows dropped for missing key fields:** "
            f"{_fmt_int(ingest_stats['malformed_key_rows_dropped'])}"
        )
        parts.append(
            f"- **Rows with a non-empty consumer narrative (kept):** "
            f"{_fmt_int(ingest_stats['narrative_rows'])}"
        )
        parts.append(
            f"- **`date_received` range:** {ingest_stats['min_date_received']} → "
            f"{ingest_stats['max_date_received']}"
        )
        parts.append("")
    parts.append(
        "Ingest keeps only rows with a parseable complaint id, a parseable "
        "`date_received`, and a non-empty `Consumer complaint narrative`; all such "
        "drops are counted above, never silent."
    )
    parts.append("")

    # --- Scrubbing / privacy ---
    parts.append("### 2.1 Privacy & scrubbing")
    parts.append("")
    parts.append(_SCRUBBING_TEXT)
    parts.append("")

    # --- Composition ---
    parts.append("## 3. Composition")
    parts.append("")
    parts.append(
        f"- **Narratives after dedup (corpus used downstream):** "
        f"{_fmt_int(rows_deduped_total)}"
    )
    parts.append(
        f"- **Rows dropped by taxonomy (dropped `Product`):** "
        f"{_fmt_int(rows_dropped_product)}"
    )
    parts.append(
        f"- **Rows carrying a routing class (mapped):** {_fmt_int(rows_mapped_total)}"
    )
    parts.append(
        f"- **Mapped rows before the TRAIN start boundary (no split):** "
        f"{_fmt_int(rows_before_train_start)}"
    )
    parts.append(
        f"- **Mapped rows in no split (outside every slice window):** "
        f"{_fmt_int(rows_in_no_split)}"
    )
    parts.append(f"- **Number of routing classes:** {len(taxonomy_raw.get('classes', {}))}")
    parts.append("")
    parts.append("### 3.1 Class × year matrix (frozen split slices)")
    parts.append("")
    parts.append(
        "Counts are of **selected** rows summed across the frozen split slices. Because "
        "`test_drift_2026h1` and `test_postcutoff` share their Feb–Jun 2026 rows by "
        "design, those rows are counted in both slices here; the `class` and column "
        "totals therefore exceed the deduped-corpus counts and are a slice census, not "
        "a corpus census. The dominance of `credit_reporting` in later years is the "
        "measured prior shift retained for the drift chapter."
    )
    parts.append("")
    parts.append(_render_class_year_table(splits_stats))
    parts.append("")

    # --- Preprocessing: taxonomy ---
    parts.append("## 4. Preprocessing")
    parts.append("")
    parts.append("### 4.1 Taxonomy harmonization")
    parts.append("")
    parts.append(_MAPPING_FRAMING)
    parts.append("")
    parts.append(f"Taxonomy map version: **{taxonomy_raw.get('version')}**.")
    parts.append("")
    parts.append(_render_taxonomy(taxonomy_raw))
    parts.append("")
    parts.append("#### Dropped products")
    parts.append("")
    parts.append(_render_dropped(taxonomy_raw))
    parts.append("")

    # --- Preprocessing: dedup ---
    parts.append("### 4.2 Near-duplicate removal (dedup)")
    parts.append("")
    parts.append(_DEDUP_FRAMING)
    parts.append("")
    parts.append(f"- **Input rows (pre-dedup):** {_fmt_int(dedup_stats['input_rows'])}")
    parts.append(f"- **Output rows (post-dedup):** {_fmt_int(dedup_stats['output_rows'])}")
    parts.append(f"- **Removed rows:** {_fmt_int(dedup_stats['removed_rows'])}")
    parts.append(
        f"- **Dedup rate:** {dedup_rate:.4f} ({dedup_rate * 100:.2f}% of input rows removed)"
    )
    parts.append(f"- **Near-duplicate clusters:** {_fmt_int(dedup_stats['n_clusters'])}")
    if params:
        parts.append(
            f"- **MinHash/LSH parameters:** num_perm={params.get('num_perm')}, "
            f"minhash_seed={params.get('minhash_seed')}, hashfunc={params.get('hashfunc')}, "
            f"shingle_size={params.get('shingle_size')}, num_bands={params.get('num_bands')}, "
            f"rows_per_band={params.get('rows_per_band')}, "
            f"lsh_threshold={params.get('lsh_threshold')}, "
            f"datasketch={params.get('datasketch_version')}"
        )
    parts.append("")

    # --- Splits ---
    parts.append("## 5. Temporal splits")
    parts.append("")
    parts.append(
        f"Splits are materialized deterministically (sub-sampling seed "
        f"**{splits_stats['seed']}**, order-and-take, no RNG). Targets, strata, and "
        "boundaries are frozen; `make data` reproduces every split parquet "
        "byte-identically. Per-split row counts and date boundaries:"
    )
    parts.append("")
    parts.append(_render_split_table(splits_stats))
    parts.append("")
    parts.append(f"*TEST-POSTCUTOFF rationale.* {_collapse_ws(splits_stats['postcutoff_rationale'])}")
    parts.append("")
    parts.append(f"*Cross-split overlap.* {_collapse_ws(splits_stats['overlap_note'])}")
    parts.append("")
    parts.append(
        "No near-duplicate pair survives across split boundaries at the LSH threshold; "
        "this is enforced by dedup running before the split and checked in CI on a "
        "planted-leak fixture."
    )
    parts.append("")

    # --- Uses ---
    parts.append("## 6. Uses")
    parts.append("")
    parts.append(_USES_TEXT)
    parts.append("")

    # --- Distribution / License ---
    parts.append("## 7. Distribution & License")
    parts.append("")
    parts.append(_LICENSE_TEXT)
    parts.append("")

    # --- Maintenance ---
    parts.append("## 8. Maintenance")
    parts.append("")
    parts.append(_MAINTENANCE_TEXT)
    parts.append("")

    return "\n".join(parts) + "\n"


def run(
    out_path: Path = DEFAULT_OUTPUT_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    taxonomy_path: Path = DEFAULT_TAXONOMY_PATH,
    ingest_stats_path: Path = DEFAULT_INGEST_STATS_PATH,
    dedup_stats_path: Path = DEFAULT_DEDUP_STATS_PATH,
    splits_stats_path: Path = DEFAULT_SPLITS_STATS_PATH,
) -> str:
    """Render and write docs/DATASHEET.md. Returns the written text."""
    manifest = _load_required(manifest_path, "SNAPSHOT_MANIFEST.yaml")
    taxonomy_raw = _load_required(taxonomy_path, "taxonomy_map.yaml")
    dedup_stats = _load_required(dedup_stats_path, "data/dedup/dedup_stats.yaml")
    splits_stats = _load_required(splits_stats_path, "data/splits/splits_stats.yaml")
    ingest_stats = _load_optional(ingest_stats_path)

    text = render_datasheet(manifest, taxonomy_raw, dedup_stats, splits_stats, ingest_stats)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m triage_lab.datasheet")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY_PATH)
    parser.add_argument("--ingest-stats", type=Path, default=DEFAULT_INGEST_STATS_PATH)
    parser.add_argument("--dedup-stats", type=Path, default=DEFAULT_DEDUP_STATS_PATH)
    parser.add_argument("--splits-stats", type=Path, default=DEFAULT_SPLITS_STATS_PATH)
    args = parser.parse_args(argv)

    run(
        out_path=args.out,
        manifest_path=args.manifest,
        taxonomy_path=args.taxonomy,
        ingest_stats_path=args.ingest_stats,
        dedup_stats_path=args.dedup_stats,
        splits_stats_path=args.splits_stats,
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
