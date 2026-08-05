"""MinHash/LSH near-dup removal on CFPB narratives (ingest -> dedup, pre-split).

Runs before the temporal split to kill train/test leakage from resubmitted and
templated complaints. Output is required to be byte-identical across runs and
across worker counts. That holds because:

- Each row's 128-perm MinHash signature is a pure function of its narrative text
  (normalization + word-5-shingles + datasketch with seed/hashfunc pinned
  explicitly, never library defaults). Worker count only changes how chunks are
  distributed, never a chunk's result, and ordered `imap` reassembles the
  signature matrix in input order.
- Banding builds, per band, a `band_bytes -> first_index` map scanning rows in
  ascending index order, so the recorded first index is always the smallest, and
  union-find (iterative, path-halving) yields the same set partition regardless of
  edge order. The cluster representative is chosen by (earliest date_received,
  smallest complaint_id), independent of union direction.
- The DuckDB writes are single-threaded with preserved insertion order and a
  fixed row-group size, over inputs already sorted by complaint_id.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import re
import sys
from pathlib import Path

import datasketch
import duckdb
import numpy as np
import yaml
from datasketch import MinHash
from datasketch.hashfunc import sha1_hash32

from triage_lab.snapshot import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "ingest" / "narratives.parquet"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "dedup"

# Fixed for deterministic parquet output; DuckDB's default, pinned explicitly.
ROW_GROUP_SIZE = 122880

# Signing/LSH parameters — all pinned, all recorded in the stats YAML.
NORMALIZATION_VERSION = "v1"
SHINGLE_SIZE = 5
NUM_PERM = 128
MINHASH_SEED = 1
NUM_BANDS = 8
ROWS_PER_BAND = 16  # NUM_BANDS * ROWS_PER_BAND == NUM_PERM
# Nominal Jaccard threshold; the 8x16 banding S-curve midpoint is ~0.878.
# Conservative because removal is irreversible pre-split and sub-threshold
# merges only over-remove; genuine misses are caught by the later leakage check.
LSH_THRESHOLD = 0.9

CHUNK_SIZE = 512  # docs per signing task
READ_BATCH = 50_000  # rows per DuckDB fetchmany
HONESTY_SAMPLE = 1_000  # first-N removed rows sampled for estimated-Jaccard receipt

# 0 normalized tokens -> this sentinel shingle (content-free rows collapse together).
_SENTINEL_SHINGLE = b"\x00EMPTY\x00"

_MASK_RE = re.compile(r"x{2,}")
_NONALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> list[str]:
    """lowercase -> collapse CFPB redaction masks (xx+ -> x) -> strip non-alnum -> split."""
    t = text.lower()
    t = _MASK_RE.sub("x", t)
    t = _NONALNUM_RE.sub(" ", t)
    return t.split()


def _shingles(tokens: list[str]) -> tuple[list[bytes], bool]:
    """Word 5-gram shingle bytes; short seq -> one shingle; empty -> sentinel."""
    if not tokens:
        return [_SENTINEL_SHINGLE], True
    if len(tokens) < SHINGLE_SIZE:
        return [" ".join(tokens).encode("utf-8")], False
    return (
        [
            " ".join(tokens[i : i + SHINGLE_SIZE]).encode("utf-8")
            for i in range(len(tokens) - SHINGLE_SIZE + 1)
        ],
        False,
    )


def sign_texts(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Pure signing worker: texts -> (uint32 (k,128) signatures, bool (k,) empty mask).

    Module-level so the spawn pool can pickle it by reference. A pure function of
    the input texts, so worker count cannot change output bytes.
    """
    empty = np.zeros(len(texts), dtype=bool)
    sets: list[list[bytes]] = []
    for j, text in enumerate(texts):
        shingles, is_empty = _shingles(normalize(text))
        empty[j] = is_empty
        sets.append(shingles)
    minhashes = MinHash.bulk(
        sets, num_perm=NUM_PERM, seed=MINHASH_SEED, hashfunc=sha1_hash32
    )
    sig = np.empty((len(texts), NUM_PERM), dtype=np.uint32)
    for j, m in enumerate(minhashes):
        sig[j] = m.hashvalues
    return sig, empty


def _find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]  # path halving
        x = parent[x]
    return x


def _union(parent: list[int], a: int, b: int) -> None:
    ra, rb = _find(parent, a), _find(parent, b)
    if ra < rb:
        parent[rb] = ra
    elif rb < ra:
        parent[ra] = rb


def cluster(signatures: np.ndarray) -> np.ndarray:
    """Band signatures into 8x16, union on per-band collision, return root per row."""
    n = signatures.shape[0]
    parent = list(range(n))
    for b in range(NUM_BANDS):
        start = b * ROWS_PER_BAND
        band = signatures[:, start : start + ROWS_PER_BAND]
        seen: dict[bytes, int] = {}
        for i in range(n):
            key = band[i].tobytes()
            first = seen.get(key)
            if first is None:
                seen[key] = i
            else:
                _union(parent, first, i)
    return np.fromiter((_find(parent, i) for i in range(n)), dtype=np.int64, count=n)


def representatives(
    roots: np.ndarray, date_ord: np.ndarray, cid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per cluster keep earliest date_received, tie-break smallest complaint_id.

    Returns (cluster_id per row, kept mask, representative index per row).
    """
    n = len(roots)
    best: dict[int, int] = {}
    for i in range(n):
        r = int(roots[i])
        cur = best.get(r)
        if cur is None or (int(date_ord[i]), int(cid[i])) < (
            int(date_ord[cur]),
            int(cid[cur]),
        ):
            best[r] = i
    rep_index = np.fromiter(
        (best[int(roots[i])] for i in range(n)), dtype=np.int64, count=n
    )
    cluster_id = cid[rep_index]
    kept = np.arange(n) == rep_index
    return cluster_id, kept, rep_index


def _removed_sample_jaccard(
    signatures: np.ndarray, kept: np.ndarray, rep_index: np.ndarray
) -> dict:
    """Estimated Jaccard (fraction of equal minhash values) for the first
    HONESTY_SAMPLE removed rows by complaint_id vs their cluster representative."""
    removed = np.nonzero(~kept)[0][:HONESTY_SAMPLE]
    if removed.size == 0:
        return {"n": 0, "mean": None, "min": None, "p05": None}
    est = np.array(
        [float(np.mean(signatures[i] == signatures[rep_index[i]])) for i in removed]
    )
    return {
        "n": int(est.size),
        "mean": float(est.mean()),
        "min": float(est.min()),
        "p05": float(np.percentile(est, 5)),
    }


def _read_input(input_path: Path) -> tuple[int, np.ndarray, np.ndarray, list[str]]:
    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=true")
        n = con.execute(
            f"SELECT count(*) FROM read_parquet('{input_path}')"
        ).fetchone()[0]
        cid = np.empty(n, dtype=np.int64)
        date_ord = np.empty(n, dtype=np.int64)
        narratives: list[str] = [""] * n
        cur = con.execute(
            f"""
            SELECT complaint_id,
                   datediff('day', DATE '1970-01-01', date_received),
                   narrative
            FROM read_parquet('{input_path}')
            ORDER BY complaint_id
            """
        )
        offset = 0
        while batch := cur.fetchmany(READ_BATCH):
            for cid_v, do_v, nar in batch:
                cid[offset] = cid_v
                date_ord[offset] = do_v
                narratives[offset] = nar
                offset += 1
    finally:
        con.close()
    return n, cid, date_ord, narratives


def _sign_all(narratives: list[str], workers: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(narratives)
    sig = np.empty((n, NUM_PERM), dtype=np.uint32)
    empty = np.empty(n, dtype=bool)
    chunks = [narratives[i : i + CHUNK_SIZE] for i in range(0, n, CHUNK_SIZE)]

    def _absorb(results) -> None:
        offset = 0
        for chunk_sig, chunk_empty in results:
            k = chunk_sig.shape[0]
            sig[offset : offset + k] = chunk_sig
            empty[offset : offset + k] = chunk_empty
            offset += k

    if workers <= 1:
        _absorb(sign_texts(c) for c in chunks)
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(workers) as pool:
            _absorb(pool.imap(sign_texts, chunks))
    return sig, empty


def run_dedup(input_path: Path, out_dir: Path, workers: int = 1) -> dict:
    """Sign, cluster, and write the deduped narratives + audit map + stats YAML."""
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    map_path = out_dir / "dedup_map.parquet"
    deduped_path = out_dir / "narratives_deduped.parquet"
    stats_path = out_dir / "dedup_stats.yaml"

    n, cid, date_ord, narratives = _read_input(input_path)
    signatures, empty = _sign_all(narratives, workers)
    del narratives

    roots = cluster(signatures)
    cluster_id, kept, rep_index = representatives(roots, date_ord, cid)
    removed_sample = _removed_sample_jaccard(signatures, kept, rep_index)

    _, sizes = np.unique(roots, return_counts=True)
    size_hist = np.bincount(sizes)
    cluster_size_histogram = {
        int(s): int(c) for s, c in enumerate(size_hist) if s > 0 and c > 0
    }
    n_clusters = int(sizes.size)
    output_rows = int(kept.sum())

    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=true")
        con.register(
            "dedup_map_view",
            {"complaint_id": cid, "cluster_id": cluster_id, "kept": kept},
        )
        con.execute(
            f"""
            COPY (
                SELECT complaint_id, cluster_id, kept
                FROM dedup_map_view
                ORDER BY complaint_id
            ) TO '{map_path}'
            (FORMAT PARQUET, COMPRESSION 'snappy', ROW_GROUP_SIZE {ROW_GROUP_SIZE})
            """
        )
        con.execute(
            f"""
            COPY (
                SELECT n.*
                FROM read_parquet('{input_path}') n
                JOIN dedup_map_view m ON n.complaint_id = m.complaint_id
                WHERE m.kept
                ORDER BY n.complaint_id
            ) TO '{deduped_path}'
            (FORMAT PARQUET, COMPRESSION 'snappy', ROW_GROUP_SIZE {ROW_GROUP_SIZE})
            """
        )
    finally:
        con.close()

    stats = {
        "normalization": NORMALIZATION_VERSION,
        "input_rows": int(n),
        "output_rows": output_rows,
        "removed_rows": int(n) - output_rows,
        "dedup_rate": (int(n) - output_rows) / int(n) if n else 0.0,
        "empty_normalized_rows": int(empty.sum()),
        "n_clusters": n_clusters,
        "cluster_size_histogram": cluster_size_histogram,
        "params": {
            "num_perm": NUM_PERM,
            "minhash_seed": MINHASH_SEED,
            "hashfunc": "sha1_hash32",
            "shingle_size": SHINGLE_SIZE,
            "num_bands": NUM_BANDS,
            "rows_per_band": ROWS_PER_BAND,
            "lsh_threshold": LSH_THRESHOLD,
            "chunk_size": CHUNK_SIZE,
            "datasketch_version": datasketch.__version__,
        },
        "removed_sample_jaccard": removed_sample,
        "input_sha256": sha256_file(input_path),
        "output_sha256": sha256_file(deduped_path),
        "dedup_map_sha256": sha256_file(map_path),
    }
    stats_path.write_text(yaml.safe_dump(stats, sort_keys=False))
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m triage_lab.dedup")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    # Worker count changes only wall-clock, never output bytes (see module docstring).
    parser.add_argument(
        "--workers", type=int, default=min(os.cpu_count() or 1, 8)
    )
    args = parser.parse_args(argv)

    stats = run_dedup(args.input, args.out_dir, args.workers)

    print(f"input_rows:            {stats['input_rows']}")
    print(f"output_rows:           {stats['output_rows']}")
    print(f"removed_rows:          {stats['removed_rows']}")
    print(f"dedup_rate:            {stats['dedup_rate']:.6f}")
    print(f"empty_normalized_rows: {stats['empty_normalized_rows']}")
    print(f"n_clusters:            {stats['n_clusters']}")
    print(f"output_sha256:         {stats['output_sha256']}")
    print(f"dedup_map_sha256:      {stats['dedup_map_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
