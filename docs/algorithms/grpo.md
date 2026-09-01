# GRPO (Group Relative Policy Optimization)

GRPO is an advanced RLHF algorithm that optimizes policies using group-based relative comparisons.

## Overview

Group Relative Policy Optimization (GRPO) is a reinforcement learning algorithm that performs policy optimization by comparing groups of generated samples rather than individual samples. This approach provides more stable training and better performance on complex tasks.

## Key Features

- **Group-based optimization**: Compares groups of samples for more stable gradients
- **Relative comparisons**: Uses relative rewards within groups rather than absolute values
- **Memory efficient**: Reduces variance through group normalization
- **Scalable**: Works well with large models and complex tasks

## Usage

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="grpo",
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
 loss_type="grpo",
 importance_sampling_level="token",
)

trainer.train()
```

`reward_functions` is required. It can contain registry names such as
`"math_correctness"` or TRL-compatible Python callables. Use
`reward_function_weights` to assign one weight per reward. See the
[reward functions guide](../user-guide/reward-functions.md) for registry and
custom-callable examples.

GRPO expects prompt-oriented data with a `prompt` column. A ground-truth
column such as `reference`, `answer`, or `solution` is only required when the
selected reward function uses it; it is forwarded to the reward function by
the data pipeline.

## Algorithm Details

GRPO optimizes policies by:

1. **Group Generation**: Generate multiple samples for each prompt
2. **Group Scoring**: Score all samples in a group
3. **Relative Normalization**: Normalize rewards within each group
4. **Policy Update**: Update policy based on group-relative advantages

During each update, the trainer generates `num_generations` completions for
each prompt, scores them with every configured reward function, combines the
scores using `reward_function_weights`, and passes the resulting group rewards
to the GRPO objective.

## Configuration Parameters

GRPO, GSPO, DAPO, and Dr. GRPO are not independent implementations. GSPO, DAPO, and Dr. GRPO are thin
subclasses of the GRPO trainer that only override one or two config defaults before calling the GRPO
parent unchanged (see [GSPO](gspo.md), [DAPO](dapo.md), [Dr. GRPO](dr-grpo.md)). Config class:
`trl.GRPOConfig`: also shared by GBMPO, Counterfactual GRPO, and PACE (each documented on
its own page with its extra parameters on top of this table). GRPO uses the common RL parameters plus:

--8<-- "docs/PARAMETERS.md:grpo-family"

The most important controls are `num_generations`,
`max_completion_length`, `temperature`, and `top_p` for rollout generation;
`beta` and `epsilon` for KL regularization and clipping; and `loss_type` plus
`importance_sampling_level` for GRPO-family objective variants. `scale_rewards`
and `mask_truncated_completions` control reward normalization and handling of
unfinished completions. `epsilon_high` can be used for an asymmetric upper
clip when supported by the installed TRL version.

## Best Practices

- Tune `num_generations` (samples per prompt/group) for better performance, more generations give a
  better relative-reward estimate at the cost of more compute
- Combine with diverse reward functions
- Monitor reward variance within each group to ensure stable training

## See Also

- [DPO Algorithm](dpo.md) - Direct Preference Optimization
- [PPO Algorithm](ppo.md) - Proximal Policy Optimization
- [Algorithms Overview](overview.md) - All supported algorithms
