"""Tier C prompt + structured-output schema: versioned, content-hashed, frozen.

Tier C (Claude via OpenRouter, CLAUDE.md rule 6) classifies a complaint narrative
into exactly one harmonized product class. Everything the model sees — the system
instructions, the user template, the few-shot exemplars, and the JSON schema that
constrains the output to the fixed label enum — lives under
``prompts/tier_c/<version>/`` and is frozen (CLAUDE.md rule 4): a change of any kind
is a new version directory, never an in-place edit.

Three frozen files make up a version:

- ``prompt.yaml``   — chat template: ``system`` string, ``user_template`` (with a
  ``{narrative}`` placeholder), and ``exemplar_format`` documenting how each frozen
  exemplar becomes a user/assistant message pair.
- ``schema.json``   — the structured-output JSON Schema (OpenRouter
  ``response_format: {type: "json_schema", ...}`` inner schema): one required
  ``label`` whose enum is exactly the harmonized taxonomy labels; strict-mode
  compatible (``additionalProperties: false``).
- ``exemplars.json``— the frozen k-shot exemplars, one per class, deterministically
  drawn from the TRAIN split (see ``select_exemplars``).

``load_prompt_bundle`` reads a version, hashes each file with
``snapshot.sha256_file`` (identical hasher to the rest of the pipeline), and derives
a single ``bundle_sha256`` that every Tier C run record will carry as its prompt
identity.

This module makes NO API calls and touches only the TRAIN split (never TEST-*).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import yaml

from triage_lab.snapshot import sha256_file
from triage_lab.taxonomy import DEFAULT_TAXONOMY_PATH, load_taxonomy

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_ROOT = REPO_ROOT / "prompts" / "tier_c"
DEFAULT_SPLITS_DIR = REPO_ROOT / "data" / "splits"

# Frozen exemplar-selection constants. Distinct from the split RNG salt so the two
# concerns never alias; recorded verbatim in exemplars.json's selection block.
EXEMPLAR_SEED = 20260806
EXEMPLAR_MIN_CHARS = 200
EXEMPLAR_MAX_CHARS = 1200
EXEMPLAR_TRAIN_SPLIT = "train"

# The three frozen files that constitute a prompt version.
PROMPT_FILE = "prompt.yaml"
SCHEMA_FILE = "schema.json"
EXEMPLARS_FILE = "exemplars.json"
BUNDLE_FILES = (EXEMPLARS_FILE, PROMPT_FILE, SCHEMA_FILE)  # sorted-filename order


# ---------------------------------------------------------------------------
# Taxonomy label enum
# ---------------------------------------------------------------------------

def harmonized_labels(taxonomy_path: Path = DEFAULT_TAXONOMY_PATH) -> list[str]:
    """The sorted harmonized class labels — the single source for the schema enum."""
    return sorted(load_taxonomy(taxonomy_path).classes)


# ---------------------------------------------------------------------------
# Deterministic serialization (byte-reproducible frozen files)
# ---------------------------------------------------------------------------

def _dumps_json(obj) -> str:
    """Pretty JSON with sorted keys + trailing newline; regeneration is byte-stable."""
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# Structured-output schema
# ---------------------------------------------------------------------------

def build_schema(labels: list[str]) -> dict:
    """The strict-mode JSON Schema constraining output to a single enum ``label``."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["label"],
        "properties": {
            "label": {
                "type": "string",
                "description": (
                    "The single harmonized CFPB product class for this complaint "
                    "narrative. Must be exactly one of the enum values."
                ),
                "enum": list(labels),
            }
        },
    }


# ---------------------------------------------------------------------------
# Exemplar selection (deterministic, TRAIN-only)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Collapse whitespace runs to a single space and strip ends."""
    return " ".join(str(text).split())


def _verify_train_integrity(splits_dir: Path) -> tuple[Path, str]:
    """Return (train_parquet, sha256) after checking it against splits_stats.yaml.

    Same fail-loud integrity gate pattern as tier_a: refuse to select exemplars
    from a TRAIN parquet whose bytes drifted from the frozen splits_stats.yaml.
    """
    stats_path = splits_dir / "splits_stats.yaml"
    train_path = splits_dir / f"{EXEMPLAR_TRAIN_SPLIT}.parquet"
    stats = yaml.safe_load(stats_path.read_text())
    expected = stats["splits"][EXEMPLAR_TRAIN_SPLIT]["sha256"]
    actual = sha256_file(train_path)
    if actual != expected:
        raise ValueError(
            f"integrity check failed for split {EXEMPLAR_TRAIN_SPLIT!r}: parquet "
            f"sha256 {actual} != frozen splits_stats.yaml {expected}"
        )
    return train_path, actual


def select_exemplars(
    splits_dir: Path = DEFAULT_SPLITS_DIR,
    *,
    seed: int = EXEMPLAR_SEED,
    min_chars: int = EXEMPLAR_MIN_CHARS,
    max_chars: int = EXEMPLAR_MAX_CHARS,
    taxonomy_path: Path = DEFAULT_TAXONOMY_PATH,
) -> tuple[list[dict], dict]:
    """Deterministically draw one exemplar per harmonized class from TRAIN.

    Reads only the TRAIN parquet (after verifying its sha256), keeps narratives whose
    whitespace-normalized length is in ``[min_chars, max_chars]``, groups by class, and
    within each class takes a single seeded draw over the id-sorted candidate list.
    Numpy's PCG64 is stable across platforms, so `same TRAIN -> same exemplars` holds
    byte-for-byte. Returns (exemplars sorted by label, selection metadata).
    """
    import numpy as np

    splits_dir = Path(splits_dir)
    train_path, train_sha = _verify_train_integrity(splits_dir)
    classes = harmonized_labels(taxonomy_path)

    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=true")
        rows = con.execute(
            f'SELECT complaint_id, narrative, "class" '
            f"FROM read_parquet('{train_path}') "
            f"ORDER BY complaint_id"
        ).fetchall()
    finally:
        con.close()

    # Group id-sorted, length-filtered candidates by class.
    candidates: dict[str, list[tuple[int, str]]] = {cls: [] for cls in classes}
    for cid, narrative, cls in rows:
        if cls not in candidates:
            continue
        norm = _normalize(narrative)
        if min_chars <= len(norm) <= max_chars:
            candidates[cls].append((int(cid), norm))

    rng = np.random.default_rng(seed)
    exemplars: list[dict] = []
    for cls in classes:  # ascending class name order fixes the RNG draw sequence
        pool = candidates[cls]
        if not pool:
            raise ValueError(
                f"no TRAIN narrative in [{min_chars}, {max_chars}] chars for class {cls!r}"
            )
        pick = int(rng.integers(0, len(pool)))
        cid, narrative = pool[pick]
        exemplars.append({"complaint_id": cid, "label": cls, "narrative": narrative})

    exemplars.sort(key=lambda e: e["label"])
    selection = {
        "source_split": EXEMPLAR_TRAIN_SPLIT,
        "train_sha256": train_sha,
        "seed": seed,
        "filter": {
            "normalized_char_min": min_chars,
            "normalized_char_max": max_chars,
            "normalization": "collapse whitespace runs to a single space, strip ends",
        },
        "candidate_order": "ascending complaint_id",
        "method": (
            "one numpy.random.default_rng(seed) draw per class, classes visited in "
            "ascending name order over the id-sorted candidate list"
        ),
        "n_classes": len(classes),
    }
    return exemplars, selection


def build_exemplars_doc(
    splits_dir: Path = DEFAULT_SPLITS_DIR, *, created: str = "2026-08-06"
) -> dict:
    """Assemble the full exemplars.json document (header keys + exemplars)."""
    exemplars, selection = select_exemplars(splits_dir)
    return {
        "version": "v1",
        "created": created,
        "note": (
            "FROZEN few-shot exemplars for Tier C prompt v1 (CLAUDE.md rule 4). Do not "
            "edit; a change requires a new prompt version directory. Regenerate/verify "
            "byte-identically via `python -m triage_lab.tier_c_prompt --verify-exemplars`."
        ),
        "selection": selection,
        "exemplars": exemplars,
    }


# ---------------------------------------------------------------------------
# Prompt bundle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptBundle:
    """A loaded, hashed prompt version. ``bundle_sha256`` is its prompt identity."""

    version: str
    system: str
    user_template: str
    exemplar_format: dict
    schema: dict
    exemplars: list[dict]
    file_sha256: dict[str, str]
    bundle_sha256: str
    selection: dict = field(default_factory=dict)

    @property
    def labels(self) -> list[str]:
        return list(self.schema["properties"]["label"]["enum"])


def _bundle_sha256(file_sha256: dict[str, str]) -> str:
    """sha256 over (filename + ':' + sha + '\\n') for the three files, sorted by name."""
    parts = [f"{name}:{file_sha256[name]}\n" for name in sorted(file_sha256)]
    return hashlib.sha256("".join(parts).encode()).hexdigest()


def load_prompt_bundle(version: str = "v1") -> PromptBundle:
    """Load + hash a frozen prompt version from ``prompts/tier_c/<version>/``."""
    vdir = PROMPTS_ROOT / version
    prompt_path = vdir / PROMPT_FILE
    schema_path = vdir / SCHEMA_FILE
    exemplars_path = vdir / EXEMPLARS_FILE
    for p in (prompt_path, schema_path, exemplars_path):
        if not p.exists():
            raise FileNotFoundError(f"prompt version {version!r} missing file: {p}")

    prompt = yaml.safe_load(prompt_path.read_text())
    schema = json.loads(schema_path.read_text())
    exemplars_doc = json.loads(exemplars_path.read_text())

    file_sha256 = {
        PROMPT_FILE: sha256_file(prompt_path),
        SCHEMA_FILE: sha256_file(schema_path),
        EXEMPLARS_FILE: sha256_file(exemplars_path),
    }
    return PromptBundle(
        version=version,
        system=prompt["system"],
        user_template=prompt["user_template"],
        exemplar_format=prompt.get("exemplar_format", {}),
        schema=schema,
        exemplars=list(exemplars_doc["exemplars"]),
        file_sha256=file_sha256,
        bundle_sha256=_bundle_sha256(file_sha256),
        selection=exemplars_doc.get("selection", {}),
    )


# ---------------------------------------------------------------------------
# Chat-message rendering
# ---------------------------------------------------------------------------

def _exemplar_assistant_content(label: str) -> str:
    """The assistant turn for an exemplar: a compact JSON object matching schema.json."""
    return json.dumps({"label": label}, separators=(",", ":"), ensure_ascii=False)


def build_messages(bundle: PromptBundle, narrative: str, num_exemplars: int) -> list[dict]:
    """Render the OpenAI-compatible chat messages for one classification query.

    Structure: a single system message; then ``num_exemplars`` user/assistant pairs
    (each exemplar's narrative rendered through ``user_template``, answered by a
    schema-valid JSON object); then the final user message with ``narrative``.
    ``num_exemplars=0`` is the §6.2 zero-shot variant (same system + template, no
    exemplar block). Raises if ``num_exemplars`` exceeds the frozen exemplar count.
    """
    available = len(bundle.exemplars)
    if num_exemplars < 0:
        raise ValueError(f"num_exemplars must be >= 0, got {num_exemplars}")
    if num_exemplars > available:
        raise ValueError(
            f"num_exemplars={num_exemplars} exceeds available exemplars ({available})"
        )

    messages: list[dict] = [{"role": "system", "content": bundle.system}]
    for ex in bundle.exemplars[:num_exemplars]:
        messages.append(
            {"role": "user", "content": bundle.user_template.format(narrative=ex["narrative"])}
        )
        messages.append(
            {"role": "assistant", "content": _exemplar_assistant_content(ex["label"])}
        )
    messages.append(
        {"role": "user", "content": bundle.user_template.format(narrative=narrative)}
    )
    return messages


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_hashes(version: str) -> int:
    bundle = load_prompt_bundle(version)
    print(f"prompt version: {version}")
    for name in sorted(bundle.file_sha256):
        print(f"  {name:16s} {bundle.file_sha256[name]}")
    print(f"  {'bundle_sha256':16s} {bundle.bundle_sha256}")
    return 0


def _generate_exemplars(version: str, splits_dir: Path) -> int:
    out_path = PROMPTS_ROOT / version / EXEMPLARS_FILE
    if out_path.exists():
        print(
            f"refusing to overwrite frozen {out_path} — exemplars are frozen "
            "(CLAUDE.md rule 4); a change requires a NEW prompt version directory.",
            file=sys.stderr,
        )
        return 1
    doc = build_exemplars_doc(splits_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_dumps_json(doc))
    print(f"wrote {out_path} ({len(doc['exemplars'])} exemplars)")
    return 0


def _verify_exemplars(version: str, splits_dir: Path) -> int:
    out_path = PROMPTS_ROOT / version / EXEMPLARS_FILE
    if not out_path.exists():
        print(f"ERROR: {out_path} does not exist; nothing to verify", file=sys.stderr)
        return 1
    on_disk = out_path.read_text()
    created = json.loads(on_disk).get("created", "2026-08-06")
    regenerated = _dumps_json(build_exemplars_doc(splits_dir, created=created))
    if regenerated != on_disk:
        print(
            f"ERROR: regenerated exemplars differ from frozen {out_path} — TRAIN split "
            "or selection logic drifted.",
            file=sys.stderr,
        )
        return 1
    print(f"OK: {out_path} regenerates byte-identically")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m triage_lab.tier_c_prompt")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--generate-exemplars",
        action="store_true",
        help="write exemplars.json from data/ only if it does not already exist",
    )
    group.add_argument(
        "--verify-exemplars",
        action="store_true",
        help="regenerate exemplars from data/ and exit nonzero on any byte difference",
    )
    args = parser.parse_args(argv)

    if args.generate_exemplars:
        return _generate_exemplars(args.version, args.splits_dir)
    if args.verify_exemplars:
        return _verify_exemplars(args.version, args.splits_dir)
    return _print_hashes(args.version)


if __name__ == "__main__":
    sys.exit(main())
