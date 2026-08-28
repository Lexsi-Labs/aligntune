#!/usr/bin/env python3
"""
Sentiment-reward batching benchmark — loop vs naive batch vs real batch.

Reuses the same 128 synthetic completions and SentimentReward setup from
reward_baseline.py.  Detects CUDA and reports device.  Measures:

  1. loop        — .compute() per completion  (current trainer path)
  2. batch_naive — _batch_compute_pipeline    (pipeline(list), no batch_size)
  3. batch_real  — pipeline(texts, batch_size=32, padding=True, truncation=True)

Asserts all three produce identical scores (atol=1e-4).

Run from repo root:
    .venv/Scripts/python.exe bench/reward_batch_gpu.py
"""

from __future__ import annotations

import importlib.util
import random
import sys
import time
import types
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Surgical imports — bypass heavy aligntune.__init__
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"


def _stub_package(name: str, path: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = [path]
    mod.__package__ = name
    sys.modules[name] = mod
    return mod


def _load_module(fqn: str, filepath: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(fqn, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fqn] = mod
    spec.loader.exec_module(mod)
    return mod


_stub_package("aligntune", str(SRC / "aligntune"))
_stub_package("aligntune.utils", str(SRC / "aligntune" / "utils"))

_load_module(
    "aligntune.utils.math_grading",
    str(SRC / "aligntune" / "utils" / "math_grading.py"),
)
core = _load_module(
    "aligntune.rewards.core",
    str(SRC / "aligntune" / "rewards" / "core.py"),
)

RewardType = core.RewardType
RewardConfig = core.RewardConfig
SentimentReward = core.SentimentReward

# ---------------------------------------------------------------------------
# Helpers (same vocab / RNG as reward_baseline.py)
# ---------------------------------------------------------------------------
VOCAB = (
    "the quick brown fox jumps over lazy dog a b c d e f g h i j k l m n "
    "o p q r s t u v w x y z alpha beta gamma delta sigma omega pi theta "
    "compute evaluate transform model pipeline reward function training "
    "dataset batch epoch gradient descent optimizer learning rate scheduler "
    "excellent wonderful amazing great fantastic superb perfect beautiful "
    "terrible horrible awful bad poor dreadful ugly disgusting negative "
).split()

N = 128
random.seed(42)


def random_text(min_words: int = 40, max_words: int = 120) -> str:
    length = random.randint(min_words, max_words)
    return " ".join(random.choices(VOCAB, k=length))


def ms_per_step(elapsed: float, n: int) -> float:
    return (elapsed / n) * 1000


def fmt(val: float) -> str:
    return f"{val:8.2f}"


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_IDX = 0 if DEVICE == "cuda" else -1

print("=" * 65)
print("  Sentiment Reward — Batch Strategy Benchmark")
print(f"  N = {N}, device = {DEVICE}, Python {sys.version.split()[0]}")
if DEVICE == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  (no CUDA GPU detected — running on CPU)")
print("=" * 65)
print()

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
completions = [random_text() for _ in range(N)]

# ---------------------------------------------------------------------------
# Build reward + warm up
# ---------------------------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
TARGET = "positive"

cfg = RewardConfig(
    reward_type=RewardType.SENTIMENT,
    model_name=MODEL_NAME,
    device=DEVICE,
    params={"target_sentiment": TARGET},
)
sentiment = SentimentReward(cfg)

print(f"Loading model ({MODEL_NAME}) ...", end=" ", flush=True)
_ = sentiment.compute(completions[0])  # warm-up: lazy model load
print("done.\n")

# Grab the raw HF pipeline for batch_real
hf_pipeline = sentiment._get_sentiment_pipeline()

# Pre-truncate once (shared across all three paths)
truncated = [sentiment._truncate_text(t) for t in completions]


def map_scores(results: list, target: str, weight: float) -> list[float]:
    """Map HF pipeline output dicts to reward floats (same logic as core.py)."""
    scores = []
    for r in results:
        if isinstance(r, list):
            r = r[0]
        if r["label"].lower() == target.lower():
            scores.append(r["score"] * weight)
        else:
            scores.append((1.0 - r["score"]) * weight)
    return scores


# ---------------------------------------------------------------------------
# 1. loop — .compute() per completion
# ---------------------------------------------------------------------------
print("[1] loop (.compute per item) ...", end=" ", flush=True)
t0 = time.perf_counter()
scores_loop = [sentiment.compute(c) for c in completions]
t_loop = time.perf_counter() - t0
print(f"{t_loop:.2f}s")

# ---------------------------------------------------------------------------
# 2. batch_naive — _batch_compute_pipeline (no batch_size kwarg)
# ---------------------------------------------------------------------------
print("[2] batch_naive (_batch_compute_pipeline) ...", end=" ", flush=True)
t0 = time.perf_counter()
scores_naive = sentiment.batch_compute(completions)
t_naive = time.perf_counter() - t0
print(f"{t_naive:.2f}s")

# ---------------------------------------------------------------------------
# 3. batch_real — pipeline(texts, batch_size=32, padding=True, truncation=True)
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
print(f"[3] batch_real (batch_size={BATCH_SIZE}, padding+truncation) ...", end=" ", flush=True)
t0 = time.perf_counter()
raw_results = hf_pipeline(
    truncated,
    batch_size=BATCH_SIZE,
    padding=True,
    truncation=True,
)
scores_real = map_scores(raw_results, TARGET, cfg.weight)
t_real = time.perf_counter() - t0
print(f"{t_real:.2f}s")

# ---------------------------------------------------------------------------
# Correctness assertion
# ---------------------------------------------------------------------------
print("\nScore agreement check (atol=1e-4):")
max_diff_naive = max(abs(a - b) for a, b in zip(scores_loop, scores_naive))
max_diff_real = max(abs(a - b) for a, b in zip(scores_loop, scores_real))
print(f"  loop vs batch_naive : max_diff = {max_diff_naive:.6f}", end="")
assert max_diff_naive < 1e-4, f"FAIL: max_diff={max_diff_naive}"
print("  PASS")
print(f"  loop vs batch_real  : max_diff = {max_diff_real:.6f}", end="")
assert max_diff_real < 1e-4, f"FAIL: max_diff={max_diff_real}"
print("  PASS")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
ms_loop_v = ms_per_step(t_loop, N)
ms_naive_v = ms_per_step(t_naive, N)
ms_real_v = ms_per_step(t_real, N)

speedup_naive = t_loop / t_naive if t_naive > 0 else float("inf")
speedup_real = t_loop / t_real if t_real > 0 else float("inf")

print()
print("=" * 65)
print(f"  SUMMARY  (device={DEVICE}, N={N})")
print("=" * 65)
hdr = f"  {'Strategy':<30} {'ms/step':>10} {'vs loop':>10}"
print(hdr)
print("  " + "-" * 52)
print(f"  {'loop (.compute)':<30} {fmt(ms_loop_v):>10} {'1.00x':>10}")
print(f"  {'batch_naive (no batch_size)':<30} {fmt(ms_naive_v):>10} {f'{speedup_naive:.2f}x':>10}")
print(f"  {'batch_real (bs=32, pad+trunc)':<30} {fmt(ms_real_v):>10} {f'{speedup_real:.2f}x':>10}")
print("=" * 65)
