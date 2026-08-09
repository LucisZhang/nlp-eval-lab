"""Deterministic input perturbations for the robustness protocol (Phase 5, UPGRADE_PLAN §6.3.4).

Three noise families are applied to the **eval text only**, at fixed rates, so that
"perturbed vs clean" is a paired comparison on identical rows, identical labels and an
identical fitted model. Nothing here ever touches TRAIN or CAL: Tier A refits on clean
TRAIN and calibrates on clean CAL exactly as in the frozen finals, and only the eval
split's ``narrative`` column is rewritten between loading and featurization. That is the
whole point of the exhibit -- it measures *input* robustness of a fixed system, not the
robustness of a training procedure.

**Rate semantics (one definition, three eligibility sets).** ``rate`` is the independent
per-*site* perturbation probability, where a site is one element of the family's eligible
set:

- ``typo``  -- every non-whitespace character is a site.
- ``ocr``   -- every character or bigram that is a key of the confusion table is a site.
  Sites are found by a **greedy longest-match left-to-right scan** (so ``"cl"`` is one
  2-char site, never ``c`` followed by ``l``), and the segmentation is computed *before*
  any random draw, so the site set is a pure function of the text and independent of the
  rate and of the RNG.
- ``case``  -- every alphabetic character is a site.

So ``rate=0.10`` means "each eligible site is independently perturbed with probability
0.1", NOT "10% of the document is rewritten". The realized fraction of *characters*
changed is lower than ``rate`` for ``ocr`` (only table characters are eligible) and can
exceed it in *length* terms for ``typo`` (insert/delete change length by one). ``rate=0``
is the exact identity and ``rate=1`` applies the family at every site.

**Determinism.** Every document gets its own RNG, seeded from
``blake2b(f"{seed}:{family}:{rate}:{doc_key}", digest_size=16)`` -- the same salted-hash
keying convention as ``splits.py`` (``_rank_key``), for the same reason: a hash is not an
RNG stream, so nothing depends on row order, batch size, chunking, or how many documents
were processed before this one. ``perturb_text`` is a pure function of its five arguments.
The frozen module seed is ``DEFAULT_SEED = 20260805``, the repo-wide constant.

The rate is part of the key because a run is identified by ``(family, rate, seed)``. The
consequence, stated because it costs statistical power: the 0.05 and 0.10 arms of a family
are **independent** draws, not nested ones (the 0.05 perturbation is not a subset of the
0.10 one). Coupling them -- dropping ``rate`` from the key and thresholding one shared
uniform vector -- would make the 0.05-vs-0.10 contrast lower-variance, but would also make
each arm's noise depend on which other arms exist. Arm-independence was chosen; the
rate-vs-rate contrast is not a headline claim.

**Within a document, all uniform draws are vectorised up front** (fire decisions, then
op choice, then key choice), so the op assigned to site *i* does not depend on how many
earlier sites fired. Only the fire vector is thresholded by the rate.

**Typo op mix.** At a firing site one of four ops is chosen from a frozen weight vector
``TYPO_OP_WEIGHTS`` (substitute 0.40 / delete 0.20 / transpose 0.20 / insert 0.20):
substitution is weighted highest because it is the dominant real keyboard error, and the
three length-changing ops are held equal. ``substitute`` and ``insert`` use a
QWERTY-adjacent key; ``transpose`` swaps with the next character; ``insert`` places the
adjacent key immediately after the character (a double-strike). Two documented edge cases:
a ``transpose`` at the last character is a no-op, and ``substitute``/``insert`` on a
character with no entry in ``QWERTY_ADJACENT`` (rare symbols) leave it unchanged, so the
realized change rate sits marginally below the nominal one on symbol-dense text.

**A note that matters for reading the Tier A results.** Both Tier A TF-IDF blocks are built
with ``lowercase=True``, so the ``case`` family is a **structural zero** for Tier A: swapping
case and then lowercasing is the identity, the feature matrix is bit-identical, and the
paired delta is exactly 0 with CI [0, 0]. The ``case`` arm is kept anyway, as (a) the
plumbing control proving the perturbation path does not leak into anything it should not
touch, and (b) the arm that is actually informative for Tier B (subword vocabularies are
case-sensitive) and Tier C. It is labelled as such in ``perturb_report``.

This module imports nothing from the harness or any tier, so the harness can import it
without a cycle and any tier runner can call it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

# Frozen protocol constants. Changing any of these changes every perturbed document, so
# none of them is a CLI flag or a config knob beyond what the run YAML already names.
DEFAULT_SEED = 20260805
FAMILIES: tuple[str, ...] = ("typo", "ocr", "case")

# The config key, under `data:`, that turns perturbation on. Named once so the harness,
# the tiers and the report all agree.
CONFIG_KEY = "perturbation"
_SPEC_FIELDS = ("family", "rate", "seed")

TYPO_OPS: tuple[str, ...] = ("substitute", "delete", "transpose", "insert")
TYPO_OP_WEIGHTS: tuple[float, ...] = (0.40, 0.20, 0.20, 0.20)
TYPO_OP_CUM: tuple[float, ...] = (0.40, 0.60, 0.80, 1.00)
_OP_SUBSTITUTE, _OP_DELETE, _OP_TRANSPOSE, _OP_INSERT = range(4)

# US-QWERTY physical layout, one string per row, plus each row's horizontal offset in
# key-widths. Adjacency is DERIVED from this geometry rather than typed out key by key:
# the layout is auditable at a glance, the derivation is a pure function, and a typo in a
# 47-entry hand-written table would be invisible. Two keys are adjacent if they are
# side-by-side in a row, or in neighbouring rows within _ROW_SPAN key-widths.
_QWERTY_ROWS: tuple[tuple[str, float], ...] = (
    ("`1234567890-=", 0.00),
    ("qwertyuiop[]\\", 0.50),
    ("asdfghjkl;'", 0.75),
    ("zxcvbnm,./", 1.25),
)
_ROW_SPAN = 0.9


def _build_qwerty_adjacency() -> dict[str, str]:
    """{key: sorted neighbouring keys} from the _QWERTY_ROWS geometry (lowercase/unshifted)."""
    placed = [
        (r, i, ch, i + offset)
        for r, (row, offset) in enumerate(_QWERTY_ROWS)
        for i, ch in enumerate(row)
    ]
    adjacency: dict[str, str] = {}
    for r, i, ch, pos in placed:
        neighbours = {
            other
            for r2, i2, other, pos2 in placed
            if (r2 == r and abs(i2 - i) == 1)
            or (abs(r2 - r) == 1 and abs(pos2 - pos) <= _ROW_SPAN)
        }
        adjacency[ch] = "".join(sorted(neighbours))
    return adjacency


QWERTY_ADJACENT: dict[str, str] = _build_qwerty_adjacency()

# OCR confusion table (ASCII only, by design: an accented or Unicode look-alike would be a
# tokenizer test, not an OCR test). Bigram keys are matched greedily before unigram keys.
# The table is symmetric wherever the inverse reading is itself OCR-plausible, which is why
# `rn -> m` is paired with `m -> rn` and `cl -> d` with `d -> cl`; the character pairs
# (l/1, O/0, S/5, B/8, e/c) are the classic segmentation-free glyph confusions.
OCR_BIGRAM_CONFUSIONS: dict[str, str] = {
    "rn": "m",
    "cl": "d",
}
OCR_UNIGRAM_CONFUSIONS: dict[str, str] = {
    "l": "1",
    "1": "l",
    "O": "0",
    "0": "O",
    "S": "5",
    "5": "S",
    "B": "8",
    "8": "B",
    "m": "rn",
    "d": "cl",
    "e": "c",
    "c": "e",
}
OCR_MAX_KEY_LEN = 2


# ---------------------------------------------------------------------------
# Per-document RNG keying
# ---------------------------------------------------------------------------

def _key_token(doc_key) -> str:
    """Canonical string for a document key. Integer-like keys (complaint_id, including
    numpy integers) render as their decimal digits, so the hash never depends on which
    integer type the caller happened to hold."""
    if isinstance(doc_key, bool):
        raise TypeError("doc_key must be a complaint_id-like value, not a bool")
    if isinstance(doc_key, int | np.integer):
        return str(int(doc_key))
    return str(doc_key)


def _rate_token(rate: float) -> str:
    """Canonical string for a rate. `repr` of a float is the shortest round-tripping
    decimal in CPython, so 0.05 keys as "0.05" on every platform -- and, crucially, the
    same rate written `0.05` or `5.0e-2` in a YAML keys identically."""
    return repr(float(rate))


def doc_rng(family: str, rate: float, seed: int, doc_key) -> np.random.Generator:
    """Per-document Generator seeded from blake2b(f"{seed}:{family}:{rate}:{doc_key}").

    Mirrors splits.py's `_rank_key`: the hash is a salt, not a stream, so the perturbation
    of a document is independent of row order, batching and every other document.
    """
    digest = hashlib.blake2b(
        f"{seed}:{family}:{_rate_token(rate)}:{_key_token(doc_key)}".encode(),
        digest_size=16,
    ).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


# ---------------------------------------------------------------------------
# Site enumeration (RNG-free, rate-free: a pure function of the text)
# ---------------------------------------------------------------------------

def typo_sites(text: str) -> list[int]:
    """Positions of every non-whitespace character."""
    return [i for i, ch in enumerate(text) if not ch.isspace()]


def case_sites(text: str) -> list[int]:
    """Positions of every alphabetic character."""
    return [i for i, ch in enumerate(text) if ch.isalpha()]


def ocr_sites(text: str) -> list[tuple[int, str]]:
    """(start, key) of every confusion-table match, greedy longest-first, non-overlapping.

    Advancing by the full key length on a *non*-firing 2-char site as well as a firing one
    is deliberate: it keeps the site set independent of the draw, which is what makes the
    realized site count -- and therefore the meaning of `rate` -- a property of the text
    alone.
    """
    sites: list[tuple[int, str]] = []
    i, n = 0, len(text)
    while i < n:
        two = text[i : i + OCR_MAX_KEY_LEN]
        if len(two) == OCR_MAX_KEY_LEN and two in OCR_BIGRAM_CONFUSIONS:
            sites.append((i, two))
            i += OCR_MAX_KEY_LEN
            continue
        one = text[i]
        if one in OCR_UNIGRAM_CONFUSIONS:
            sites.append((i, one))
        i += 1
    return sites


# ---------------------------------------------------------------------------
# The three families
# ---------------------------------------------------------------------------

def _adjacent_key(ch: str, u: float) -> str:
    """A QWERTY-adjacent key for `ch`, case-preserved. Unknown characters map to themselves
    (documented no-op; see the module docstring)."""
    neighbours = QWERTY_ADJACENT.get(ch.lower())
    if not neighbours:
        return ch
    pick = neighbours[min(int(u * len(neighbours)), len(neighbours) - 1)]
    return pick.upper() if ch.isupper() else pick


def _perturb_typo(text: str, rate: float, rng: np.random.Generator) -> str:
    sites = typo_sites(text)
    if not sites:
        return text
    # All three uniform vectors are drawn up front, in this order, at full site length, so
    # the op at a site never depends on how many earlier sites fired.
    fire = rng.random(len(sites)) < rate
    ops = np.searchsorted(TYPO_OP_CUM, rng.random(len(sites)), side="right")
    picks = rng.random(len(sites))

    fire_at: dict[int, tuple[int, float]] = {
        pos: (int(ops[k]), float(picks[k])) for k, pos in enumerate(sites) if fire[k]
    }
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        drawn = fire_at.get(i)
        if drawn is None:
            out.append(ch)
            i += 1
            continue
        op, u = drawn
        if op == _OP_SUBSTITUTE:
            out.append(_adjacent_key(ch, u))
            i += 1
        elif op == _OP_DELETE:
            i += 1
        elif op == _OP_TRANSPOSE:
            if i + 1 < n:
                # Consumes the next position, whose own draw (if it had one) is skipped --
                # unavoidable for a swap, and deterministic given the fire vector.
                out.append(text[i + 1])
                out.append(ch)
                i += 2
            else:
                out.append(ch)  # nothing to swap with at end of string
                i += 1
        else:  # _OP_INSERT — a double-strike of the neighbouring key
            out.append(ch)
            out.append(_adjacent_key(ch, u))
            i += 1
    return "".join(out)


def _perturb_ocr(text: str, rate: float, rng: np.random.Generator) -> str:
    sites = ocr_sites(text)
    if not sites:
        return text
    fire = rng.random(len(sites)) < rate
    out: list[str] = []
    cursor = 0
    for k, (start, key) in enumerate(sites):
        out.append(text[cursor:start])
        if fire[k]:
            out.append(
                OCR_BIGRAM_CONFUSIONS[key] if len(key) == OCR_MAX_KEY_LEN
                else OCR_UNIGRAM_CONFUSIONS[key]
            )
        else:
            out.append(key)
        cursor = start + len(key)
    out.append(text[cursor:])
    return "".join(out)


def _perturb_case(text: str, rate: float, rng: np.random.Generator) -> str:
    sites = case_sites(text)
    if not sites:
        return text
    fire = rng.random(len(sites)) < rate
    chars = list(text)
    for k, pos in enumerate(sites):
        if not fire[k]:
            continue
        flipped = chars[pos].swapcase()
        # Guard the handful of code points whose case flip is not length-1 (e.g. 'ß' ->
        # 'SS'): this family must change case, never length.
        if len(flipped) == 1:
            chars[pos] = flipped
    return "".join(chars)


_FAMILY_FNS = {
    "typo": _perturb_typo,
    "ocr": _perturb_ocr,
    "case": _perturb_case,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate(family: str, rate: float) -> None:
    """Fail loud on an unknown family or an out-of-range rate (config typos die here)."""
    if family not in FAMILIES:
        raise ValueError(f"unknown perturbation family {family!r}; choose from {list(FAMILIES)}")
    if not 0.0 <= float(rate) <= 1.0:
        raise ValueError(f"perturbation rate {rate!r} outside [0, 1]")


def perturb_text(text: str, family: str, rate: float, seed: int, doc_key) -> str:
    """Perturb one document. Pure: same (text, family, rate, seed, doc_key) -> same string.

    `rate == 0` short-circuits to the identity, which is the same answer the RNG path would
    give (`rng.random() < 0` is never true) but costs nothing.
    """
    validate(family, rate)
    if text is None:
        return ""
    if rate == 0.0 or not text:
        return text
    return _FAMILY_FNS[family](text, float(rate), doc_rng(family, rate, seed, doc_key))


def perturb_texts(texts, doc_keys, family: str, rate: float, seed: int) -> list[str]:
    """Apply `perturb_text` down a loaded eval frame, one document key per row.

    The keys must be the eval split's `complaint_id`s, id-aligned to `texts`; that is what
    makes the output independent of the order the frame happens to be in.
    """
    validate(family, rate)
    texts = list(texts)
    doc_keys = list(doc_keys)
    if len(texts) != len(doc_keys):
        raise ValueError(
            f"perturb_texts: {len(texts)} texts but {len(doc_keys)} doc keys; the keys must "
            "be the eval split's complaint_ids, row-aligned to the texts"
        )
    return [
        perturb_text(text, family, rate, seed, key)
        for text, key in zip(texts, doc_keys, strict=True)
    ]


# ---------------------------------------------------------------------------
# Config plumbing (shared by every tier runner + the harness gate)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PerturbationSpec:
    """The validated `data.perturbation` block of a run config."""

    family: str
    rate: float
    seed: int

    def as_dict(self) -> dict:
        """The block as it is stamped into the run record's `extra.perturbation`."""
        return {"family": self.family, "rate": float(self.rate), "seed": int(self.seed)}


def spec_from_config(config: dict) -> PerturbationSpec | None:
    """Parse + validate `data.perturbation`; None when the config declares no perturbation.

    Unknown keys inside the block are a hard error rather than being ignored: a silently
    dropped `familly:` would produce a run that looks perturbed in its config and is not.
    """
    data = config.get("data") or {}
    block = data.get(CONFIG_KEY)
    if block is None:
        return None
    if not isinstance(block, dict):
        raise TypeError(f"data.{CONFIG_KEY} must be a mapping, got {type(block).__name__}")
    unknown = sorted(set(block) - set(_SPEC_FIELDS))
    if unknown:
        raise ValueError(
            f"data.{CONFIG_KEY} has unknown key(s) {unknown}; allowed: {list(_SPEC_FIELDS)}"
        )
    missing = [k for k in ("family", "rate") if k not in block]
    if missing:
        raise ValueError(f"data.{CONFIG_KEY} is missing required key(s) {missing}")
    family = str(block["family"])
    rate = float(block["rate"])
    validate(family, rate)
    return PerturbationSpec(family=family, rate=rate, seed=int(block.get("seed", DEFAULT_SEED)))


def apply_spec(texts, doc_keys, spec: PerturbationSpec | None) -> list[str]:
    """`perturb_texts` under a spec; a None spec is the identity (returns a list copy)."""
    if spec is None:
        return list(texts)
    return perturb_texts(texts, doc_keys, spec.family, spec.rate, spec.seed)
