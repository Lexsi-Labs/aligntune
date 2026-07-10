#!/usr/bin/env python3
"""
Real-dataset decontamination smoke test on GSM8K.

Loads gsm8k train (~7.5k) and test (~1.3k) questions from HuggingFace,
runs the decontamination API, and prints the top matches so we can
eyeball genuine near-duplicates.

Run from repo root:
    .venv/Scripts/python.exe bench/decon_real_data.py
"""

from __future__ import annotations

import importlib.util
import sys
import time
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Surgical import of decontamination (bypass heavy aligntune.__init__)
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"


def _stub_pkg(name: str, path: str) -> None:
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    mod.__path__ = [path]
    mod.__package__ = name
    sys.modules[name] = mod


def _load_mod(fqn: str, filepath: str) -> types.ModuleType:
    if fqn in sys.modules:
        return sys.modules[fqn]
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
exact_jaccard = decon.exact_jaccard

# ---------------------------------------------------------------------------
# Load GSM8K
# ---------------------------------------------------------------------------
from datasets import load_dataset  # noqa: E402

print("=" * 75)
print("  Real-Data Decontamination Smoke Test: GSM8K")
print("=" * 75)

print("\nLoading gsm8k train split ...", end=" ", flush=True)
ds_train = load_dataset("openai/gsm8k", "main", split="train")
train_questions = [row["question"] for row in ds_train]
print(f"done ({len(train_questions)} rows)")

print("Loading gsm8k test split ...", end=" ", flush=True)
ds_test = load_dataset("openai/gsm8k", "main", split="test")
eval_questions = [row["question"] for row in ds_test]
print(f"done ({len(eval_questions)} rows)")

# Use all train rows — Rust is fast enough
print(f"Using all {len(train_questions)} train rows")

# ---------------------------------------------------------------------------
# Run decontamination (threshold=0.7, verify=True)
# ---------------------------------------------------------------------------
print(f"\nRunning decontaminate(threshold=0.7, verify=True) ...")
t0 = time.perf_counter()
report = decontaminate(
    train_questions,
    eval_questions,
    num_perm=128,
    shingle=5,
    bands=16,
    rows=8,
    threshold=0.7,
    estimate_margin=0.05,
    verify=True,
    backend="auto",
)
t_total = time.perf_counter() - t0

n_total = report.n_train + report.n_eval
print(f"Done in {t_total:.2f}s ({n_total / t_total:.0f} docs/sec)")

print(f"\n  Backend:           {report.backend_used}")
print(f"  Verified:          {report.verified}")
print(f"  Train docs:        {report.n_train}")
print(f"  Eval docs:         {report.n_eval}")
print(f"  Flagged pairs:     {len(report.matches)}")
print(f"  Contaminated train indices: {len(report.contaminated_train_indices)}")
print(f"  Timing: sigs={report.timing.get('signatures', 0):.2f}s "
      f"lsh={report.timing.get('lsh', 0):.2f}s "
      f"filter={report.timing.get('filter', 0):.2f}s "
      f"verify={report.timing.get('verify', 0):.2f}s")

# ---------------------------------------------------------------------------
# Top 10 matches by exact Jaccard
# ---------------------------------------------------------------------------
def trunc(s: str, n: int = 120) -> str:
    s = s.replace("\n", " ").strip()
    return s[:n] + "..." if len(s) > n else s

if report.matches:
    sorted_matches = sorted(report.matches, key=lambda x: x[2], reverse=True)
    top = sorted_matches[:10]

    print(f"\n  Top {len(top)} matches by exact Jaccard:")
    print("  " + "-" * 72)
    for rank, (ti, ei, jac) in enumerate(top, 1):
        print(f"\n  #{rank}  Jaccard={jac:.3f}  train[{ti}] <-> eval[{ei}]")
        print(f"    TRAIN: {trunc(train_questions[ti])}")
        print(f"    EVAL:  {trunc(eval_questions[ei])}")
else:
    print("\n  No matches found at threshold=0.7")

# ---------------------------------------------------------------------------
# Rerun at threshold=0.5
# ---------------------------------------------------------------------------
print(f"\n{'=' * 75}")
print("  Rerun at threshold=0.5")
print("=" * 75)

t0 = time.perf_counter()
report_05 = decontaminate(
    train_questions,
    eval_questions,
    num_perm=128,
    shingle=5,
    bands=16,
    rows=8,
    threshold=0.5,
    estimate_margin=0.05,
    verify=True,
    backend="auto",
)
t_05 = time.perf_counter() - t0

print(f"\n  Threshold=0.5: {len(report_05.matches)} flagged pairs "
      f"({len(report_05.contaminated_train_indices)} train indices) "
      f"in {t_05:.2f}s")

if report_05.matches:
    sorted_05 = sorted(report_05.matches, key=lambda x: x[2], reverse=True)
    top_05 = sorted_05[:10]
    print(f"\n  Top {len(top_05)} matches at threshold=0.5:")
    print("  " + "-" * 72)
    for rank, (ti, ei, jac) in enumerate(top_05, 1):
        print(f"\n  #{rank}  Jaccard={jac:.3f}  train[{ti}] <-> eval[{ei}]")
        print(f"    TRAIN: {trunc(train_questions[ti])}")
        print(f"    EVAL:  {trunc(eval_questions[ei])}")

# ---------------------------------------------------------------------------
# Closest-pairs probe: run LSH at very low threshold to find ANY overlap
# ---------------------------------------------------------------------------
print(f"\n{'=' * 75}")
print("  Closest-pairs probe (threshold=0.3, shingle=3)")
print("=" * 75)

t0 = time.perf_counter()
report_low = decontaminate(
    train_questions,
    eval_questions,
    num_perm=128,
    shingle=3,
    bands=16,
    rows=8,
    threshold=0.3,
    estimate_margin=0.1,
    verify=True,
    backend="auto",
)
t_low = time.perf_counter() - t0

print(f"\n  shingle=3, threshold=0.3: {len(report_low.matches)} flagged pairs in {t_low:.2f}s")

if report_low.matches:
    sorted_low = sorted(report_low.matches, key=lambda x: x[2], reverse=True)
    top_low = sorted_low[:10]
    print(f"\n  Top {len(top_low)} closest train/eval pairs:")
    print("  " + "-" * 72)
    for rank, (ti, ei, jac) in enumerate(top_low, 1):
        print(f"\n  #{rank}  Jaccard={jac:.3f}  train[{ti}] <-> eval[{ei}]")
        print(f"    TRAIN: {trunc(train_questions[ti])}")
        print(f"    EVAL:  {trunc(eval_questions[ei])}")
else:
    print("  No pairs found even at threshold=0.3 — dataset is very clean")

print("=" * 75)
