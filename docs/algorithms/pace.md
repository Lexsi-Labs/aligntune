# PACE (Baseline-Optimized Learning Technique)

PACE is a relative preference algorithmic strategy that incorporates learned baselines per-prompt directly into the scoring mechanism.

## Overview

PACE dynamically tracks the historical baseline reward for specific prompts and penalizes generations that fail to beat their historical average. This curriculum-style baseline tracking makes the gradient updates significantly less noisy than standard GRPO or PPO.

### When to use PACE?
- When your reward signals have high variance across different prompts.
- When you want to continuously track and improve upon historical bests in specific reasoning tasks.

## Configuration

To use PACE, pass `algorithm="pace"` to the factory method:

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="openai/gsm8k",
    algorithm="pace",
    backend="trl",
    reward_functions=["math_correctness"],
    reward_function_weights=[1.0],
    num_epochs=1,
    batch_size=4,
    learning_rate=1e-6,
    num_generations=8,
    max_completion_length=512,
    temperature=0.7,
    top_p=0.95,
    use_baseline_advantages=True,
    baseline_enabled=True,
    curriculum_enabled=False,
)

trainer.train()
```

`reward_functions` is required for a meaningful PACE run. PACE combines the
configured registry rewards or TRL-compatible custom callables, then compares
each prompt's current reward with its learned historical baseline.
`reward_function_weights` optionally supplies one weight per reward. See the
[reward functions guide](../user-guide/reward-functions.md) for registry and
custom-callable examples.

PACE expects prompt-oriented data with a `prompt` column. A ground-truth
column such as `reference`, `answer`, or `solution` is forwarded to rewards
that require it.

`use_baseline_advantages=True` uses the learned per-prompt baseline for the
advantage calculation. `baseline_enabled` persists the baseline table.
`curriculum_enabled` (uncertainty-based prompt sampling) is not available in
this build and is always forced to `False`.

### Configuration Options

PACE subclasses the GRPO trainer, so it inherits every parameter on the
[GRPO page](grpo.md#configuration-parameters), plus:

--8<-- "docs/PARAMETERS.md:pace-bolt"

## See Also

- [Algorithms Overview](overview.md)
- [PACE Parameters](../PARAMETERS.md#pace-bolt) - Full parameter reference, including TRL vs. Unsloth backend differences
