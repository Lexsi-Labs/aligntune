# DAPO (Decouple Clip and Dynamic sAmpling Policy Optimization)

DAPO is not an independent implementation, it is a thin subclass of the GRPO trainer that changes
the `loss_type` used for the policy-gradient loss.

## Overview

Decouple Clip and Dynamic sAmpling Policy Optimization (DAPO) addresses limitations in vanilla GRPO's
loss formulation. Both backends (TRL and Unsloth) implement DAPO as a subclass of their GRPO trainer
that only overrides one config default (if not already set by the caller) before calling the GRPO
parent unchanged.

## Key Features

- **DAPO loss**: Uses the `'dapo'` loss variant instead of the default `'grpo'` loss
- **Improved stability**: Better gradient behavior than standard PPO
- **Multi-backend support**: Available in both TRL and Unsloth backends

## Usage

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dapo",
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

DAPO expects prompt-oriented data with a `prompt` column. A ground-truth
column such as `reference`, `answer`, or `solution` is only needed by the
selected reward function and is forwarded to it by the data pipeline.

## Algorithm Details

DAPO differs from base GRPO in exactly one way:

1. **Loss type override**: `loss_type` is set to `'dapo'` instead of `'grpo'`
2. Everything else, group generation, group scoring, relative-advantage normalization, clipping, is
   identical to GRPO

The trainer generates `num_generations` completions per prompt, scores them
with the configured rewards, and applies the group rewards using the DAPO
loss variant.

## Configuration Options

DAPO's full parameter surface **is** GRPO's surface, see [GRPO](grpo.md) for the complete table. DAPO
only overrides this default (and only if the caller hasn't already set it):

| Parameter | Override |
|---|---|
| `loss_type` | `'grpo'` → `'dapo'` |

## Advantages

- More stable training than standard PPO
- Better performance on complex tasks
- Reduced hyperparameter sensitivity

## See Also

- [PPO Algorithm](ppo.md) - Base policy optimization
- [GRPO Algorithm](grpo.md) - Group-based optimization
- [Algorithms Overview](overview.md) - All supported algorithms
