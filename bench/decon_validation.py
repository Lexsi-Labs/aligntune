#!/usr/bin/env python3
"""
Decontamination API validation with realistic graded-overlap corpus.

Exercises aligntune.data.decontamination through the public API:
  - Equivalence: Rust vs Python backends produce identical results
  - Recall curve: binned by true Jaccard for two LSH configurations
  - Precision and timing

Run from repo root:
    .venv/Scripts/python.exe bench/decon_validation.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import random
import sys
import time
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Surgical import of decontamination module (bypass heavy aligntune.__init__)
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"


def _stub_pkg(name: str, path: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = [path]
    mod.__package__ = name
    sys.modules[name] = mod
    return mod


def _load_mod(fqn: str, filepath: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(fqn, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fqn] = mod
    spec.loader.exec_module(mod)
    return mod


_stub_pkg("aligntune", str(SRC / "aligntune"))
_stub_pkg("aligntune.data", str(SRC / "aligntune" / "data"))

decon = _load_mod(
    "aligntune.data.decontamination",
    str(SRC / "aligntune" / "data" / "decontamination.py"),
)

decontaminate = decon.decontaminate
import aligntune_fast as af

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_PERM = 128
SHINGLE = 5
SEED_COEFFS = 0
N_EVAL = 4000
N_CLEAN = 4000
N_CONTAM = 1500
N_TRAIN = N_CLEAN + N_CONTAM

# ---------------------------------------------------------------------------
# Vocab: ~5000 pseudo-words, Zipf-ish
# ---------------------------------------------------------------------------
rng_vocab = random.Random(77)
BASE_CHARS = "abcdefghijklmnopqrstuvwxyz"
VOCAB: list[str] = []
seen_w: set[str] = set()
while len(VOCAB) < 5000:
    wlen = rng_vocab.randint(3, 10)
    w = "".join(rng_vocab.choices(BASE_CHARS, k=wlen))
    if w not in seen_w:
        seen_w.add(w)
        VOCAB.append(w)

ZIPF_W = [1.0 / (i + 1) ** 0.8 for i in range(len(VOCAB))]

# ---------------------------------------------------------------------------
# Corpus generation
# ---------------------------------------------------------------------------
random.seed(42)


def zipf_doc(lo: int = 40, hi: int = 80) -> str:
    return " ".join(random.choices(VOCAB, weights=ZIPF_W, k=random.randint(lo, hi)))


def _sha1_u32(data: bytes) -> int:
    d = hashlib.sha1(data).digest()
    return int.from_bytes(d[:4], "little")


def shingle_set(doc: str) -> set[int]:
    tokens = doc.split()
    s: set[int] = set()
    for i in range(len(tokens) - SHINGLE + 1):
        joined = " ".join(tokens[i : i + SHINGLE])
        s.add(_sha1_u32(joined.encode("utf-8")))
    return s


def true_jaccard(doc_a: str, doc_b: str) -> float:
    sa, sb = shingle_set(doc_a), shingle_set(doc_b)
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0


def contaminate_to_target_jaccard(
    source: str, target_jac: float, max_iters: int = 30
) -> tuple[str, float]:
    """Binary-search word-replacement count to hit target Jaccard ± 0.03."""
    words = source.split()
    n = len(words)
    lo_k, hi_k = 0, n
    best_doc, best_jac = source, 1.0
    rng = random.Random(hash(source) & 0xFFFFFFFF)
    for _ in range(max_iters):
        k = (lo_k + hi_k) // 2
        if k == 0:
            candidate = source
        else:
            w = list(words)
            positions = rng.sample(range(n), min(k, n))
            for pos in positions:
                w[pos] = rng.choice(VOCAB)
            candidate = " ".join(w)
        jac = true_jaccard(source, candidate)
        if abs(jac - target_jac) < abs(best_jac - target_jac):
            best_doc, best_jac = candidate, jac
        if abs(jac - target_jac) < 0.03:
            return best_doc, best_jac
        if jac > target_jac:
            lo_k = k + 1
        else:
            hi_k = k - 1
        if lo_k > hi_k:
            break
    return best_doc, best_jac


print("=" * 70)
print("  Decontamination API Validation")
print(f"  eval={N_EVAL}  train={N_TRAIN} (clean={N_CLEAN}, contam={N_CONTAM})")
print(f"  NUM_PERM={NUM_PERM}  SHINGLE={SHINGLE}")
print(f"  CPU cores: {af.rayon_num_threads()}")
print("=" * 70)

# Build eval
print("\nGenerating eval docs ...", end=" ", flush=True)
eval_docs = [zipf_doc() for _ in range(N_EVAL)]
print("done")

# Build train: clean + contaminated with uniform spread across 0.5-1.0
print("Generating train docs (clean + contaminated) ...", end=" ", flush=True)
train_docs: list[str] = [zipf_doc() for _ in range(N_CLEAN)]

# For each contaminated doc, pick a target Jaccard uniformly in [0.5, 1.0)
contam_meta: list[dict] = []
target_jacs = [0.5 + (i / N_CONTAM) * 0.5 for i in range(N_CONTAM)]
random.shuffle(target_jacs)

for target_jac in target_jacs:
    eval_idx = random.randint(0, N_EVAL - 1)
    contam_doc, achieved_jac = contaminate_to_target_jaccard(
        eval_docs[eval_idx], target_jac
    )
    train_idx = len(train_docs)
    train_docs.append(contam_doc)
    contam_meta.append({
        "train_idx": train_idx,
        "eval_idx": eval_idx,
        "true_jac": achieved_jac,
    })

print("done")

# Distribution check
BINS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
print("\n  Contaminated doc distribution by true Jaccard:")
for lo, hi in BINS:
    cnt = sum(1 for m in contam_meta if lo <= m["true_jac"] < hi)
    print(f"    [{lo:.1f}, {hi:.1f}): {cnt}")

# ===================================================================
# 1. EQUIVALENCE: Rust vs Python on first 500 docs
# ===================================================================
print("\n" + "=" * 70)
print("  [1] Backend Equivalence (Rust vs Python, first 500 docs)")
print("=" * 70)

EQUIV_N = 500
print(f"Running decontaminate with backend='rust' ...", end=" ", flush=True)
r_rust = decontaminate(
    train_docs[:EQUIV_N], eval_docs[:EQUIV_N],
    num_perm=NUM_PERM, shingle=SHINGLE, bands=16, rows=8,
    threshold=0.8, seed=SEED_COEFFS, backend="rust",
)
print("done")

print(f"Running decontaminate with backend='python' ...", end=" ", flush=True)
r_py = decontaminate(
    train_docs[:EQUIV_N], eval_docs[:EQUIV_N],
    num_perm=NUM_PERM, shingle=SHINGLE, bands=16, rows=8,
    threshold=0.8, seed=SEED_COEFFS, backend="python",
)
print("done")

# Compare match sets
rust_matches = {(ti, ei) for ti, ei, _ in r_rust.matches}
py_matches = {(ti, ei) for ti, ei, _ in r_py.matches}

if rust_matches == py_matches:
    print(f"  Signatures & matches: PASS (identical, {len(rust_matches)} matches)")
else:
    only_rust = rust_matches - py_matches
    only_py = py_matches - rust_matches
    print(f"  FAIL: {len(only_rust)} only in Rust, {len(only_py)} only in Python")

# ===================================================================
# 2. RECALL CURVE: two LSH configs
# ===================================================================
print("\n" + "=" * 70)
print("  [2] Recall Curve by True Jaccard Bucket")
print("=" * 70)

LSH_CONFIGS = [
    {"bands": 16, "rows": 8, "label": "b=16,r=8 (threshold~0.5)"},
    {"bands": 32, "rows": 4, "label": "b=32,r=4 (threshold~0.2)"},
]

for cfg in LSH_CONFIGS:
    print(f"\n  Config: {cfg['label']}")
    t0 = time.perf_counter()
    report = decontaminate(
        train_docs, eval_docs,
        num_perm=NUM_PERM, shingle=SHINGLE,
        bands=cfg["bands"], rows=cfg["rows"],
        threshold=0.8, estimate_margin=0.05,
        seed=SEED_COEFFS, backend="rust",
    )
    t_total = time.perf_counter() - t0

    flagged_set = {(ti, ei) for ti, ei, _ in report.matches}

    # Overall precision: fraction of flagged pairs with true Jaccard >= 0.8
    true_pos = 0
    for ti, ei, est_jac in report.matches:
        tj = true_jaccard(train_docs[ti], eval_docs[ei])
        if tj >= 0.8:
            true_pos += 1
    overall_prec = true_pos / len(report.matches) if report.matches else 0.0

    # Recall by bucket
    print(f"  {'True Jac bucket':<18} {'contam':>7} {'caught':>7} {'recall':>8}")
    print("  " + "-" * 42)
    for lo, hi in BINS:
        bucket = [m for m in contam_meta if lo <= m["true_jac"] < hi]
        caught = sum(
            1 for m in bucket if (m["train_idx"], m["eval_idx"]) in flagged_set
        )
        rec = caught / len(bucket) if bucket else 0.0
        print(f"  [{lo:.1f}, {hi:.1f})          {len(bucket):>7} {caught:>7} {rec:>8.3f}")

    # Overall stats
    all_contam_08 = [m for m in contam_meta if m["true_jac"] >= 0.8]
    caught_08 = sum(
        1 for m in all_contam_08 if (m["train_idx"], m["eval_idx"]) in flagged_set
    )
    rec_08 = caught_08 / len(all_contam_08) if all_contam_08 else 0.0

    print(f"\n  Flagged pairs:     {len(report.matches)}")
    print(f"  Precision (>=0.8): {overall_prec:.4f}  ({true_pos}/{len(report.matches)})")
    print(f"  Recall (>=0.8):    {rec_08:.4f}  ({caught_08}/{len(all_contam_08)})")
    print(f"  Time:              {t_total:.2f}s  (sigs={report.timing['signatures']:.2f}s "
          f"lsh={report.timing['lsh']:.2f}s filter={report.timing['filter']:.2f}s)")
    print(f"  Backend:           {report.backend_used}")

# ===================================================================
# 3. SPEED: Rust vs Python backend
# ===================================================================
print("\n" + "=" * 70)
print("  [3] Speed: Rust vs Python backend (full corpus)")
print("=" * 70)

# Rust (already ran above, reuse timing from b=16,r=8 config)
print(f"\n  Running Rust backend ...", end=" ", flush=True)
t0 = time.perf_counter()
r_rust_full = decontaminate(
    train_docs, eval_docs,
    num_perm=NUM_PERM, shingle=SHINGLE, bands=16, rows=8,
    threshold=0.8, seed=SEED_COEFFS, backend="rust",
)
t_rust = time.perf_counter() - t0
print(f"{t_rust:.2f}s")

# Python on a 1000-doc subset, extrapolate
PY_N = 1000
print(f"  Running Python backend ({PY_N} docs, extrapolated) ...", end=" ", flush=True)
t0 = time.perf_counter()
_ = decontaminate(
    train_docs[:PY_N], eval_docs[:PY_N],
    num_perm=NUM_PERM, shingle=SHINGLE, bands=16, rows=8,
    threshold=0.8, seed=SEED_COEFFS, backend="python",
)
t_py_sub = time.perf_counter() - t0
# Signature cost scales linearly, LSH scales ~quadratically with corpus
# Use linear extrapolation (conservative for Python)
scale = (N_TRAIN + N_EVAL) / (PY_N * 2)
t_py_est = t_py_sub * scale
print(f"{t_py_sub:.2f}s (subset), ~{t_py_est:.1f}s (extrapolated)")

print(f"\n  {'Backend':<30} {'time (s)':>10} {'speedup':>10}")
print("  " + "-" * 48)
print(f"  {'Python (extrapolated)':<30} {t_py_est:>10.1f} {'1.0x':>10}")
print(f"  {'Rust (' + str(af.rayon_num_threads()) + 'T)':<30} {t_rust:>10.2f} {t_py_est/t_rust:>9.1f}x")

print("=" * 70)
