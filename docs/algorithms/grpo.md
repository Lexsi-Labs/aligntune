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
 num_epochs=1,
 batch_size=2,
 learning_rate=1e-6
)

trainer.train()
```

## Algorithm Details

GRPO optimizes policies by:

1. **Group Generation**: Generate multiple samples for each prompt
2. **Group Scoring**: Score all samples in a group
3. **Relative Normalization**: Normalize rewards within each group
4. **Policy Update**: Update policy based on group-relative advantages

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `group_size` | int | 8 | Number of samples per group |
| `beta` | float | 0.01 | KL penalty coefficient |
| `cliprange` | float | 0.2 | PPO-style clipping range |
| `temperature` | float | 0.7 | Sampling temperature |

## Best Practices

- Use larger group sizes (8-16) for better performance
- Combine with diverse reward functions
- Monitor group diversity to ensure stable training

## See Also

- [DPO Algorithm](dpo.md) - Direct Preference Optimization
- [PPO Algorithm](ppo.md) - Proximal Policy Optimization
- [Algorithms Overview](overview.md) - All supported algorithms