"""
Train/eval decontamination via MinHash + LSH.

Uses the ``aligntune_fast`` Rust extension when available (backend="auto" or
"rust"), falling back to a bit-identical pure-Python implementation otherwise.

Public API
----------
decontaminate(train_docs, eval_docs, ...) -> DeconReport
clean_dataset(train_docs, eval_docs, ...) -> (kept_docs, DeconReport)
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------
_RUST_AVAILABLE = False
try:
    import aligntune_fast as _af

    _RUST_AVAILABLE = True
except ImportError:
    _af = None  # type: ignore[assignment]

MERSENNE_P = (1 << 61) - 1


def _resolve_backend(backend: str) -> str:
    if backend == "auto":
        return "rust" if _RUST_AVAILABLE else "python"
    if backend == "rust":
        if not _RUST_AVAILABLE:
            raise ImportError(
                "backend='rust' requested but aligntune_fast is not installed. "
                "Build it with: cd rust_ext && maturin develop --release"
            )
        return "rust"
    if backend == "python":
        return "python"
    raise ValueError(f"Unknown backend: {backend!r}. Use 'auto', 'rust', or 'python'.")


# ---------------------------------------------------------------------------
# Deterministic coefficients
# ---------------------------------------------------------------------------

def _make_coeffs(
    num_perm: int, seed: int
) -> tuple[list[int], list[int]]:
    """Generate deterministic hash-family coefficients from a seed.

    Uses numpy RandomState for reproducibility.  Returns plain Python lists
    so they can be passed directly to the Rust FFI or the Python fallback.
    """
    rs = np.random.RandomState(seed)
    coeff_a = rs.randint(1, MERSENNE_P, size=num_perm, dtype=np.uint64).tolist()
    coeff_b = rs.randint(0, MERSENNE_P, size=num_perm, dtype=np.uint64).tolist()
    return coeff_a, coeff_b


# ---------------------------------------------------------------------------
# Pure-Python fallback (bit-identical to Rust path)
# ---------------------------------------------------------------------------

def _sha1_u32(data: bytes) -> int:
    """SHA-1 -> first 4 bytes as u32 little-endian."""
    d = hashlib.sha1(data).digest()
    return int.from_bytes(d[:4], "little")


def _py_minhash_signatures(
    docs: list[str],
    coeff_a: list[int],
    coeff_b: list[int],
    shingle: int,
) -> list[list[int]]:
    num_perm = len(coeff_a)
    result = []
    for doc in docs:
        tokens = doc.split()
        if len(tokens) < shingle:
            result.append([0xFFFFFFFF] * num_perm)
            continue
        shingle_set: set[int] = set()
        for i in range(len(tokens) - shingle + 1):
            joined = " ".join(tokens[i : i + shingle])
            shingle_set.add(_sha1_u32(joined.encode("utf-8")))
        if not shingle_set:
            result.append([0xFFFFFFFF] * num_perm)
            continue
        sig = [0xFFFFFFFF] * num_perm
        for h in shingle_set:
            for j in range(num_perm):
                hv = ((coeff_a[j] * h + coeff_b[j]) % MERSENNE_P) & 0xFFFFFFFF
                if hv < sig[j]:
                    sig[j] = hv
        result.append(sig)
    return result


def _py_estimate_jaccard(sig1: list[int], sig2: list[int]) -> float:
    if len(sig1) != len(sig2) or not sig1:
        return 0.0
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


def _py_band_key(sig: list[int], start: int, end: int) -> tuple[int, ...]:
    """Bucket key for one band slice — a tuple of the row values."""
    return tuple(sig[start:end])


def _py_lsh_candidate_pairs(
    sig_a: list[list[int]],
    sig_b: list[list[int]],
    bands: int,
    rows: int,
) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    for band in range(bands):
        start = band * rows
        end = start + rows
        b_map: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for j, sig in enumerate(sig_b):
            key = _py_band_key(sig, start, end)
            b_map[key].append(j)
        for i, sig in enumerate(sig_a):
            key = _py_band_key(sig, start, end)
            if key in b_map:
                for j in b_map[key]:
                    seen.add((i, j))
    return list(seen)


# ---------------------------------------------------------------------------
# Exact Jaccard via shingle sets (used by verify=True)
# ---------------------------------------------------------------------------

def _shingle_set(doc: str, shingle: int) -> set[int]:
    """Build the set of shingle hashes for a document.

    Uses the same SHA-1-first-4-bytes-LE hash as the MinHash path so that
    exact Jaccard computed here is consistent with the MinHash estimate.
    """
    tokens = doc.split()
    s: set[int] = set()
    for i in range(len(tokens) - shingle + 1):
        joined = " ".join(tokens[i : i + shingle])
        s.add(_sha1_u32(joined.encode("utf-8")))
    return s


def exact_jaccard(doc_a: str, doc_b: str, shingle: int = 5) -> float:
    """Compute exact Jaccard similarity of two documents' shingle sets."""
    sa = _shingle_set(doc_a, shingle)
    sb = _shingle_set(doc_b, shingle)
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------

def _compute_signatures(
    docs: list[str],
    coeff_a: list[int],
    coeff_b: list[int],
    shingle: int,
    backend: str,
) -> list[list[int]]:
    if backend == "rust":
        return _af.minhash_signatures(docs, coeff_a, coeff_b, shingle)
    return _py_minhash_signatures(docs, coeff_a, coeff_b, shingle)


def _lsh_candidates(
    sig_a: list[list[int]],
    sig_b: list[list[int]],
    bands: int,
    rows: int,
    backend: str,
) -> list[tuple[int, int]]:
    if backend == "rust":
        return _af.lsh_candidate_pairs(sig_a, sig_b, bands, rows)
    return _py_lsh_candidate_pairs(sig_a, sig_b, bands, rows)


def _est_jaccard(sig1: list[int], sig2: list[int], backend: str) -> float:
    if backend == "rust":
        return _af.estimate_jaccard(sig1, sig2)
    return _py_estimate_jaccard(sig1, sig2)


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class DeconReport:
    """Results of a decontamination run."""

    matches: list[tuple[int, int, float]]
    """(train_idx, eval_idx, jaccard) for each flagged pair.

    The jaccard value is the *estimated* MinHash Jaccard when ``verify=False``,
    or the *exact* set Jaccard when ``verify=True``.
    """

    contaminated_train_indices: set[int]
    """Train indices that matched at least one eval doc."""

    n_train: int
    n_eval: int

    timing: dict[str, float] = field(default_factory=dict)
    """Wall-clock seconds for each phase: signatures, lsh, filter, verify."""

    backend_used: str = "unknown"
    verified: bool = False
    """Whether exact-Jaccard verification was performed."""


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def decontaminate(
    train_docs: Sequence[str],
    eval_docs: Sequence[str],
    *,
    num_perm: int = 128,
    shingle: int = 5,
    bands: int = 16,
    rows: int = 8,
    threshold: float = 0.8,
    estimate_margin: float = 0.05,
    seed: int = 0,
    backend: str = "auto",
    verify: bool = False,
) -> DeconReport:
    """Detect train/eval overlap via MinHash + LSH.

    Parameters
    ----------
    threshold : float
        Jaccard similarity threshold for flagging a pair.
    estimate_margin : float
        Candidates with ``estimate >= threshold - margin`` are kept.
        This avoids rejecting true-positives whose MinHash estimate sits
        just below the threshold due to estimator variance.
    backend : str
        ``"auto"`` (try Rust, else Python), ``"rust"``, or ``"python"``.
    verify : bool
        If True, recompute exact Jaccard (from shingle sets) for every
        candidate that survives the estimate filter and discard those
        below ``threshold``.  The match tuples then carry the exact
        Jaccard instead of the estimate.  This eliminates false positives
        at a small additional cost (candidates are few).

    Returns
    -------
    DeconReport
    """
    if bands * rows != num_perm:
        raise ValueError(f"bands*rows ({bands}*{rows}={bands*rows}) != num_perm ({num_perm})")

    be = _resolve_backend(backend)
    train_list = list(train_docs)
    eval_list = list(eval_docs)
    coeff_a, coeff_b = _make_coeffs(num_perm, seed)
    timing: dict[str, float] = {}

    # Signatures
    t0 = time.perf_counter()
    train_sigs = _compute_signatures(train_list, coeff_a, coeff_b, shingle, be)
    eval_sigs = _compute_signatures(eval_list, coeff_a, coeff_b, shingle, be)
    timing["signatures"] = time.perf_counter() - t0

    # LSH candidate pairs
    t0 = time.perf_counter()
    raw_pairs = _lsh_candidates(train_sigs, eval_sigs, bands, rows, be)
    timing["lsh"] = time.perf_counter() - t0

    # Filter by estimated Jaccard
    cutoff = threshold - estimate_margin
    t0 = time.perf_counter()
    matches: list[tuple[int, int, float]] = []
    contaminated: set[int] = set()
    for i, j in raw_pairs:
        jac = _est_jaccard(list(train_sigs[i]), list(eval_sigs[j]), be)
        if jac >= cutoff:
            matches.append((int(i), int(j), jac))
            contaminated.add(int(i))
    timing["filter"] = time.perf_counter() - t0

    # Exact verification
    if verify:
        t0 = time.perf_counter()
        shingle_cache: dict[tuple[str, int], set[int]] = {}

        def _get_shingles(corpus: str, docs: list[str], idx: int) -> set[int]:
            key = (corpus, idx)
            if key not in shingle_cache:
                shingle_cache[key] = _shingle_set(docs[idx], shingle)
            return shingle_cache[key]

        verified_matches: list[tuple[int, int, float]] = []
        verified_contaminated: set[int] = set()
        for ti, ei, _ in matches:
            sa = _get_shingles("train", train_list, ti)
            sb = _get_shingles("eval", eval_list, ei)
            union = len(sa | sb)
            ej = len(sa & sb) / union if union else 0.0
            if ej >= threshold:
                verified_matches.append((ti, ei, ej))
                verified_contaminated.add(ti)
        matches = verified_matches
        contaminated = verified_contaminated
        timing["verify"] = time.perf_counter() - t0

    return DeconReport(
        matches=matches,
        contaminated_train_indices=contaminated,
        n_train=len(train_list),
        n_eval=len(eval_list),
        timing=timing,
        backend_used=be,
        verified=verify,
    )


def clean_dataset(
    train_docs: Sequence[str],
    eval_docs: Sequence[str],
    **kwargs: Any,
) -> tuple[list[str], DeconReport]:
    """Run decontamination and return train with contaminated rows removed."""
    report = decontaminate(train_docs, eval_docs, **kwargs)
    kept = [
        doc
        for i, doc in enumerate(train_docs)
        if i not in report.contaminated_train_indices
    ]
    return kept, report
