# GSPO (Group Sequence Policy Optimization)

GSPO is not an independent implementation, it is a thin subclass of the GRPO trainer that changes the
importance-sampling granularity from per-token to per-sequence.

## Overview

Group Sequence Policy Optimization (GSPO) extends GRPO by computing importance sampling at the
sequence level rather than the token level, which can give more stable gradients for long completions.
Both backends (TRL and Unsloth) implement GSPO as a subclass of their GRPO trainer that only overrides
one or two config defaults (if not already set by the caller) before calling the GRPO parent unchanged.

## Key Features

- **Sequence-level importance sampling**: Aggregates importance weights over the whole sequence instead of per-token
- **Multi-backend support**: Available in both TRL and Unsloth backends (see `backends/unsloth/rl/gspo/` for the Unsloth implementation)

## Usage

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="gspo",
 backend="trl",
 reward_functions=["length"],
 reward_function_weights=[1.0],
 num_epochs=1,
 batch_size=2,
 learning_rate=1e-6,
 num_generations=4,
 max_completion_length=256,
 temperature=0.7,
 top_p=0.95,
 beta=0.1,
 epsilon=0.2,
)

trainer.train()
```

`reward_functions` is required, just as it is for GRPO. It accepts registry
names such as `"math_correctness"` and TRL-compatible Python callables.
`reward_function_weights` optionally supplies one weight per reward. See the
[reward functions guide](../user-guide/reward-functions.md) for registry and
custom-callable examples.

GSPO expects prompt-oriented data with a `prompt` column. A ground-truth
column such as `reference`, `answer`, or `solution` is only needed by the
selected reward function and is forwarded to it by the data pipeline.

## Algorithm Details

GSPO differs from base GRPO in exactly one way:

1. **Sequence-level importance sampling**: `importance_sampling_level` is set to `'sequence'` instead of `'token'`
2. Everything else, group generation, group scoring, relative-advantage normalization, the clipped
   policy-gradient update, is identical to GRPO

The trainer generates `num_generations` completions per prompt, scores each
completion with the configured rewards, and applies the resulting group
rewards using sequence-level importance sampling.

## Configuration Options

GSPO's full parameter surface **is** GRPO's surface, see [GRPO](grpo.md) for the complete table. GSPO
only overrides these defaults (and only if the caller hasn't already set them):

| Parameter | Override |
|---|---|
| `importance_sampling_level` | `'token'` → `'sequence'` |
| `loss_type` | `'grpo'` → `'dapo'` |

## Limitations

- Higher computational cost than standard GRPO in some configurations
- May require re-tuning `epsilon`/clip range relative to base GRPO since clipping now applies at the sequence level

## See Also

- [GRPO Algorithm](grpo.md) - Base group optimization and full parameter reference
- [Algorithms Overview](overview.md) - All supported algorithms
