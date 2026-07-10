#!/usr/bin/env python3
"""
Reward-path micro-benchmark — no training loop.

Measures wall-clock cost of real reward computation over 128 synthetic
completions, covering:
  1. Model-based sentiment reward  (loop vs batch)
  2. Math correctness grading      (sympy path)
  3. Pure-text rewards              (diversity, length)

Run from repo root:
    .venv/Scripts/python.exe bench/reward_baseline.py
"""

from __future__ import annotations

import importlib.util
import os
import random
import sys
import time
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Surgical imports — bypass heavy aligntune.__init__ (needs vllm etc.)
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

# Stub top-level packages so sub-module absolute imports resolve
_stub_package("aligntune", str(SRC / "aligntune"))
_stub_package("aligntune.utils", str(SRC / "aligntune" / "utils"))

# Load math_grading first (dependency of rewards/core.py)
math_grading = _load_module(
    "aligntune.utils.math_grading",
    str(SRC / "aligntune" / "utils" / "math_grading.py"),
)

# Load rewards/core.py
core = _load_module(
    "aligntune.rewards.core",
    str(SRC / "aligntune" / "rewards" / "core.py"),
)

RewardType = core.RewardType
RewardConfig = core.RewardConfig
RewardFunctionFactory = core.RewardFunctionFactory
SentimentReward = core.SentimentReward
DiversityReward = core.DiversityReward
LengthReward = core.LengthReward
MathCorrectnessReward = core.MathCorrectnessReward

grade_math_answer = math_grading.grade_math_answer

# ---------------------------------------------------------------------------
# Helpers
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


def make_arithmetic_string() -> tuple[str, str]:
    """Return (completion_with_equation, reference_answer)."""
    a = random.randint(1, 100)
    b = random.randint(1, 100)
    op = random.choice(["+", "-", "*"])
    result = eval(f"{a}{op}{b}")
    completion = f"The answer is {a} {op} {b} = {result}"
    reference = str(result)
    return completion, reference


def ms_per_step(elapsed: float, n: int) -> float:
    return (elapsed / n) * 1000


def fmt(val: float) -> str:
    return f"{val:8.2f}"


# ---------------------------------------------------------------------------
# Generate data
# ---------------------------------------------------------------------------
print("=" * 65)
print("  AlignTune Reward-Path Micro-Benchmark")
print(f"  N = {N} completions, Python {sys.version.split()[0]}")
print("=" * 65)
print()

completions = [random_text() for _ in range(N)]
references = [random_text() for _ in range(N)]
math_completions = []
math_references = []
for _ in range(N):
    c, r = make_arithmetic_string()
    math_completions.append(c)
    math_references.append(r)

# ---------------------------------------------------------------------------
# 1. MODEL-BASED REWARD: Sentiment  (loop vs batch)
# ---------------------------------------------------------------------------
print("[1] Sentiment reward (model-based)")
print(f"    model: distilbert-base-uncased-finetuned-sst-2-english")
print(f"    loading model ...", end=" ", flush=True)

sentiment_cfg = RewardConfig(
    reward_type=RewardType.SENTIMENT,
    model_name="distilbert-base-uncased-finetuned-sst-2-english",
    device="cpu",
)
sentiment = SentimentReward(sentiment_cfg)

# Warm up (trigger lazy model load + JIT)
_ = sentiment.compute(completions[0])
print("done.")

# (a) Python loop — one .compute() per completion
t0 = time.perf_counter()
loop_scores = [sentiment.compute(c) for c in completions]
t_loop = time.perf_counter() - t0
ms_loop = ms_per_step(t_loop, N)

# (b) batch_compute
t0 = time.perf_counter()
batch_scores = sentiment.batch_compute(completions)
t_batch = time.perf_counter() - t0
ms_batch = ms_per_step(t_batch, N)

speedup = t_loop / t_batch if t_batch > 0 else float("inf")

print(f"    loop   : {fmt(ms_loop)} ms/step  (total {t_loop:.2f}s)")
print(f"    batch  : {fmt(ms_batch)} ms/step  (total {t_batch:.2f}s)")
print(f"    speedup: {speedup:.2f}x")
print()

# ---------------------------------------------------------------------------
# 2. MATH CORRECTNESS REWARD (sympy path)
# ---------------------------------------------------------------------------
print("[2] Math correctness reward (sympy grading)")

math_cfg = RewardConfig(
    reward_type=RewardType.MATH_CORRECTNESS,
    device="cpu",
)
math_reward = MathCorrectnessReward(math_cfg)

t0 = time.perf_counter()
math_scores = [math_reward.compute(c, r) for c, r in zip(math_completions, math_references)]
t_math = time.perf_counter() - t0
ms_math = ms_per_step(t_math, N)
avg_math = sum(math_scores) / len(math_scores)

# Also benchmark the raw grade_math_answer function directly
t0 = time.perf_counter()
grading_results = [grade_math_answer(c, r) for c, r in zip(math_completions, math_references)]
t_grade = time.perf_counter() - t0
ms_grade = ms_per_step(t_grade, N)

print(f"    MathCorrectnessReward.compute : {fmt(ms_math)} ms/step  (avg score {avg_math:.3f})")
print(f"    grade_math_answer (raw)       : {fmt(ms_grade)} ms/step  (acc {sum(grading_results)/N:.3f})")
print()

# ---------------------------------------------------------------------------
# 3. PURE-TEXT REWARDS: Diversity & Length
# ---------------------------------------------------------------------------
print("[3] Pure-text rewards (single-threaded)")

# Diversity
div_cfg = RewardConfig(reward_type=RewardType.DIVERSITY, device="cpu")
div_reward = DiversityReward(div_cfg)

t0 = time.perf_counter()
div_scores = [div_reward.compute(c) for c in completions]
t_div = time.perf_counter() - t0
ms_div = ms_per_step(t_div, N)
avg_div = sum(div_scores) / len(div_scores)

# Length
len_cfg = RewardConfig(
    reward_type=RewardType.LENGTH,
    device="cpu",
    params={"min_length": 10, "max_length": 500},
)
len_reward = LengthReward(len_cfg)

t0 = time.perf_counter()
len_scores = [len_reward.compute(c) for c in completions]
t_len = time.perf_counter() - t0
ms_len = ms_per_step(t_len, N)
avg_len = sum(len_scores) / len(len_scores)

print(f"    DiversityReward   : {fmt(ms_div)} ms/step  (avg score {avg_div:.3f})")
print(f"    LengthReward      : {fmt(ms_len)} ms/step  (avg score {avg_len:.3f})")
print()

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("=" * 65)
print("  SUMMARY")
print("=" * 65)
header = f"  {'Reward':<35} {'ms/step':>10} {'note':>15}"
print(header)
print("  " + "-" * 62)
rows = [
    ("Sentiment (loop, per-item)", ms_loop, "model-based"),
    ("Sentiment (batch_compute)", ms_batch, f"{speedup:.1f}x faster"),
    ("MathCorrectness (.compute)", ms_math, "regex+eval"),
    ("grade_math_answer (raw)", ms_grade, "sympy path"),
    ("Diversity (pure text)", ms_div, f"avg={avg_div:.3f}"),
    ("Length (pure text)", ms_len, f"avg={avg_len:.3f}"),
]
for name, ms, note in rows:
    print(f"  {name:<35} {fmt(ms):>10} {note:>15}")
print("=" * 65)
