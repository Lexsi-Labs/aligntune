# Universal Evaluation Framework

A backend-agnostic evaluation system for Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) models, combining modular evaluation with legacy benchmark support via `lm-eval`.

## Key Features

* **Unified Interface:** one entry point for generation (SFT), comparative (RL), and standard benchmarks (MMLU/GSM8K).
* **Backend Agnostic:** evaluates models regardless of training origin (TRL, Unsloth, HF).
* **Safe Compute:** a per-sample wrapper catches individual failures (NaNs, empty strings, missing dependencies) and returns an error key instead of crashing the run.
* **Smart Caching:** results are cached based on a hash of (model + dataset + config).
* **Specialized Metrics:** built-in support for math (regex extraction) and code (sandboxed execution).

## Quick Start

1. **Evaluating an SFT Model (Generation)**

Use `BaseEvaluator` for standard text generation. Supports **Perplexity**, **ROUGE**, **BLEU**, and **Custom Metrics**.

```{python}
from aligntune.eval import BaseEvaluator
from aligntune.eval.metrics import RougeMetric, BleuMetric
from datasets import load_dataset

# 1. Load Model (Standard HF)
# ... (load model/tokenizer code) ...

# 2. Initialize Evaluator
evaluator = BaseEvaluator(
    metrics=[RougeMetric(), BleuMetric()],
    batch_size=8,
    device="cuda"
)

# 3. Run Evaluation
results = evaluator.evaluate(
    model=model,
    tokenizer=tokenizer,
    dataset=load_dataset("imdb", split="test[:10]"),
    task_name="text_generation"
)
print(results)
# Output: {'perplexity': 12.4, 'rouge1': 0.45, ...}
```


2. **Evaluating an RL Model (PPO/GRPO)**

Use `RLEvaluator` to assess training stability. It compares a Policy Model against a **Reference Model** and scores outputs using a **Reward Model**.

**Key Metrics:** KL Divergence, Reward Accuracy, Policy Entropy.

```{python}
from aligntune.eval import RLEvaluator

evaluator = RLEvaluator(batch_size=4)

metrics = evaluator.evaluate_rl(
    policy_model=policy_model,
    reference_model=reference_model,  # Required for KL Divergence
    tokenizer=tokenizer,
    dataset=dataset,                  # Expects 'prompt', 'chosen', 'rejected'
    reward_model=my_reward_func       # Optional
)
print(metrics)
```


3. **Running Standard Benchmarks (Legacy Integration)**

The framework integrates `lm-eval` to run standard benchmarks like GSM8K or MMLU easily.

```{python}
# Uses the legacy lm_eval_integration under the hood
benchmark_results = evaluator.evaluate_benchmark(
    model_name_or_path="meta-llama/Llama-3.2-1B",
    tasks=["gsm8k", "mmlu", "arc_easy"]
)
```

## Architecture

The system uses a Strategy Pattern for metrics and separates the new modular logic from legacy support.

```
src/aligntune/eval/
├── evaluator.py        # BaseEvaluator: Main entry point (renamed from core)
├── rl_evaluator.py     # RLEvaluator: Handles RL-specific logic
├── core.py             # (Legacy) EvalRunner for backward compatibility
├── lm_eval_integration.py # (Legacy) Wrappers for EleutherAI lm-eval
├── safe_executor.py    # Sandbox for code execution
└── metrics/            # Strategy Pattern Implementations
    ├── base.py         # Abstract base + safe_compute wrapper
    ├── generic.py      # Perplexity, Accuracy
    ├── text.py         # ROUGE, BLEU
    ├── rl.py           # KL Divergence, Reward Acc
    ├── math.py         # Robust Regex extraction for Math
    └── code.py         # Pass@K using safe_executor
```


## Safe Compute in Detail

All metrics use a `safe_compute` wrapper. If a metric fails (e.g., missing library, math error), the evaluator logs a warning and returns a failure key (`_error`) instead of crashing the process.

Example Response on Failure:

```{python}
{
  "perplexity": 15.2,
  "accuracy": 0.88,
  "rouge_error": 0.0   # Indicates ROUGE calculation failed
}
```


## Custom Metrics

You can inject custom logic without modifying the library.

```{python}
from aligntune.eval import Metric
import numpy as np

class DiversityMetric(Metric):
    def __init__(self):
        super().__init__("diversity")

    def compute(self, predictions, references, **kwargs):
        # Your custom logic here
        scores = [len(set(p.split())) / len(p.split()) for p in predictions]
        return {"diversity": np.mean(scores)}

# Usage
evaluator.add_metric(DiversityMetric())
```


## Requirements

* **Core:** `torch`, `transformers`, `datasets`

* **Text Metrics:** `rouge_score`, `nltk`

* **Benchmarks:** `lm_eval` (optional)

If optional dependencies are missing, specific metrics will be disabled gracefully via Safe Compute.