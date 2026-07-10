#!/usr/bin/env python3
"""
MinHash benchmark: Rust (rayon-parallel) vs naive Python vs datasketch.

Corpus: 20 000 synthetic docs, NUM_PERM=128, SHINGLE=5.
Correctness: Rust vs pure-Python reference on first 500 docs (bit-exact).
Jaccard sanity: 200 random pairs, MAE of MinHash estimate vs true Jaccard.
Speed: docs/sec and speedup of Rust vs each baseline.

Run from repo root:
    .venv/Scripts/python.exe bench/minhash_rust_vs_python.py
"""

from __future__ import annotations

import hashlib
import os
import random
import time
from collections import defaultdict

import aligntune_fast
from datasketch import MinHash as DSMinHash

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_DOCS = 20_000
NUM_PERM = 128
SHINGLE = 5
SEED = 42
MERSENNE_P = (1 << 61) - 1

VOCAB = (
    "the quick brown fox jumps over lazy dog a b c d e f g h i j k l m n "
    "o p q r s t u v w x y z alpha beta gamma delta sigma omega pi theta "
    "compute evaluate transform model pipeline reward function training "
    "dataset batch epoch gradient descent optimizer learning rate scheduler "
    "excellent wonderful amazing great fantastic superb perfect beautiful "
    "terrible horrible awful bad poor dreadful ugly disgusting negative "
    "analysis research experiment hypothesis theory proof lemma corollary "
    "server client database query index table column row join filter sort "
).split()

# ---------------------------------------------------------------------------
# Corpus generator (deterministic)
# ---------------------------------------------------------------------------
random.seed(SEED)


def random_doc(min_words: int = 40, max_words: int = 120) -> str:
    length = random.randint(min_words, max_words)
    return " ".join(random.choices(VOCAB, k=length))


print("=" * 70)
print("  MinHash Benchmark: Rust (rayon) vs Python vs datasketch")
print(f"  N_DOCS={N_DOCS}  NUM_PERM={NUM_PERM}  SHINGLE={SHINGLE}")
print(f"  CPU cores (rayon): {aligntune_fast.rayon_num_threads()}")
print(f"  os.cpu_count():    {os.cpu_count()}")
print("=" * 70)
print()

print("Generating corpus ...", end=" ", flush=True)
docs = [random_doc() for _ in range(N_DOCS)]
print(f"done ({N_DOCS} docs)")

# ---------------------------------------------------------------------------
# Coefficients (shared across Rust and Python)
# ---------------------------------------------------------------------------
rng = random.Random(123)
coeff_a = [rng.randint(1, MERSENNE_P - 1) for _ in range(NUM_PERM)]
coeff_b = [rng.randint(0, MERSENNE_P - 1) for _ in range(NUM_PERM)]

# ---------------------------------------------------------------------------
# Pure-Python reference MinHash (identical algorithm to Rust)
# ---------------------------------------------------------------------------


def _sha1_u32(data: bytes) -> int:
    """SHA-1 hash -> first 4 bytes as u32 little-endian."""
    d = hashlib.sha1(data).digest()
    return int.from_bytes(d[:4], "little")


def py_minhash_one(doc: str) -> list[int]:
    """Compute one MinHash signature using the exact same algorithm as Rust."""
    tokens = doc.split()
    if len(tokens) < SHINGLE:
        return [0xFFFFFFFF] * NUM_PERM

    # Build shingle hash set
    shingle_set: set[int] = set()
    for i in range(len(tokens) - SHINGLE + 1):
        joined = " ".join(tokens[i : i + SHINGLE])
        shingle_set.add(_sha1_u32(joined.encode("utf-8")))

    if not shingle_set:
        return [0xFFFFFFFF] * NUM_PERM

    sig = [0xFFFFFFFF] * NUM_PERM
    for h in shingle_set:
        for i in range(NUM_PERM):
            hv = ((coeff_a[i] * h + coeff_b[i]) % MERSENNE_P) & 0xFFFFFFFF
            if hv < sig[i]:
                sig[i] = hv
    return sig


def py_minhash_all(documents: list[str]) -> list[list[int]]:
    return [py_minhash_one(d) for d in documents]


# ---------------------------------------------------------------------------
# 1. CORRECTNESS: Rust vs pure-Python on first 500 docs
# ---------------------------------------------------------------------------
CHECK_N = 500
print(f"\n[1] Correctness check: Rust vs Python on first {CHECK_N} docs ...", end=" ", flush=True)

rust_sigs_check = aligntune_fast.minhash_signatures(
    docs[:CHECK_N], coeff_a, coeff_b, SHINGLE
)
py_sigs_check = py_minhash_all(docs[:CHECK_N])

mismatches = 0
for i in range(CHECK_N):
    if list(rust_sigs_check[i]) != py_sigs_check[i]:
        mismatches += 1
        if mismatches <= 3:
            # Print first few mismatches for debugging
            for j in range(NUM_PERM):
                if rust_sigs_check[i][j] != py_sigs_check[i][j]:
                    print(
                        f"\n  doc {i} perm {j}: rust={rust_sigs_check[i][j]} "
                        f"py={py_sigs_check[i][j]}"
                    )
                    break

if mismatches == 0:
    print(f"PASS ({CHECK_N} docs, {NUM_PERM} perms each, bit-exact)")
else:
    print(f"FAIL ({mismatches}/{CHECK_N} docs differ)")

# ---------------------------------------------------------------------------
# 2. JACCARD SANITY: MinHash estimate vs true Jaccard on 200 random pairs
# ---------------------------------------------------------------------------
PAIR_N = 200
print(f"\n[2] Jaccard sanity: {PAIR_N} random pairs ...")

# Precompute shingle sets for first 1000 docs (for true Jaccard)
JACCARD_POOL = 1000
shingle_sets: list[set[int]] = []
for doc in docs[:JACCARD_POOL]:
    tokens = doc.split()
    s: set[int] = set()
    for i in range(len(tokens) - SHINGLE + 1):
        joined = " ".join(tokens[i : i + SHINGLE])
        s.add(_sha1_u32(joined.encode("utf-8")))
    shingle_sets.append(s)

rust_sigs_pool = aligntune_fast.minhash_signatures(
    docs[:JACCARD_POOL], coeff_a, coeff_b, SHINGLE
)

pair_rng = random.Random(999)
abs_errors = []
for _ in range(PAIR_N):
    i = pair_rng.randint(0, JACCARD_POOL - 1)
    j = pair_rng.randint(0, JACCARD_POOL - 1)
    if i == j:
        continue

    # True Jaccard
    inter = len(shingle_sets[i] & shingle_sets[j])
    union = len(shingle_sets[i] | shingle_sets[j])
    true_jac = inter / union if union > 0 else 1.0

    # MinHash estimate
    matches = sum(
        1 for k in range(NUM_PERM) if rust_sigs_pool[i][k] == rust_sigs_pool[j][k]
    )
    mh_jac = matches / NUM_PERM

    abs_errors.append(abs(true_jac - mh_jac))

mae = sum(abs_errors) / len(abs_errors)
print(f"    Mean absolute error: {mae:.4f}  (expect < ~0.05)")
if mae < 0.05:
    print("    PASS")
else:
    print("    WARNING: MAE higher than expected")

# ---------------------------------------------------------------------------
# 3. SPEED: all 20 000 docs
# ---------------------------------------------------------------------------
print(f"\n[3] Speed benchmark over {N_DOCS} docs\n")

# (a) Naive Python loop
print("    [a] naive Python loop ...", end=" ", flush=True)
t0 = time.perf_counter()
_ = py_minhash_all(docs)
t_py = time.perf_counter() - t0
print(f"{t_py:.2f}s")

# (b) datasketch MinHash (num_perm=128, default sha1)
print("    [b] datasketch MinHash ...", end=" ", flush=True)
t0 = time.perf_counter()
for doc in docs:
    m = DSMinHash(num_perm=NUM_PERM)
    tokens = doc.split()
    for i in range(len(tokens) - SHINGLE + 1):
        shingle_str = " ".join(tokens[i : i + SHINGLE])
        m.update(shingle_str.encode("utf-8"))
t_ds = time.perf_counter() - t0
print(f"{t_ds:.2f}s")

# (c) Rust minhash_signatures (single call, rayon-parallel)
print("    [c] Rust minhash_signatures ...", end=" ", flush=True)
t0 = time.perf_counter()
_ = aligntune_fast.minhash_signatures(docs, coeff_a, coeff_b, SHINGLE)
t_rust = time.perf_counter() - t0
print(f"{t_rust:.2f}s")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
dps_py = N_DOCS / t_py
dps_ds = N_DOCS / t_ds
dps_rust = N_DOCS / t_rust

print()
print("=" * 70)
print("  SUMMARY")
print("=" * 70)
hdr = f"  {'Method':<35} {'time (s)':>10} {'docs/sec':>12} {'vs loop':>10}"
print(hdr)
print("  " + "-" * 65)
print(
    f"  {'naive Python (loop)':<35} {t_py:>10.2f} {dps_py:>12.0f} {'1.00x':>10}"
)
print(
    f"  {'datasketch MinHash':<35} {t_ds:>10.2f} {dps_ds:>12.0f} {t_py/t_ds:>9.1f}x"
)
print(
    f"  {'Rust rayon (aligntune_fast)':<35} {t_rust:>10.2f} {dps_rust:>12.0f} {t_py/t_rust:>9.1f}x"
)
print("=" * 70)
