# DAPO (Decouple Clip and Dynamic sAmpling Policy Optimization)

DAPO is an advanced RLHF algorithm that decouples clipping and dynamic sampling for improved training stability.

## Overview

Decouple Clip and Dynamic sAmpling Policy Optimization (DAPO) addresses limitations in traditional PPO by separating the clipping mechanism from dynamic sampling, providing more stable and efficient training.

## Key Features

- **Decoupled optimization**: Separate clipping and sampling mechanisms
- **Dynamic sampling**: Adaptive sample generation based on training progress
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
 num_epochs=1,
 batch_size=2,
 learning_rate=1e-6
)

trainer.train()
```

## Algorithm Details

DAPO improves upon PPO by:

1. **Decoupled clipping**: Separate value and policy clipping
2. **Dynamic sampling**: Adjusts sample distribution during training
3. **Adaptive optimization**: Better handling of reward scaling

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip_range` | float | 0.2 | Policy clipping range |
| `value_clip_range` | float | 0.2 | Value function clipping range |
| `dynamic_sampling` | bool | true | Enable dynamic sampling |
| `sampling_temperature` | float | 1.0 | Sampling temperature |

## Advantages

- More stable training than standard PPO
- Better performance on complex tasks
- Reduced hyperparameter sensitivity

## See Also

- [PPO Algorithm](ppo.md) - Base policy optimization
- [GRPO Algorithm](grpo.md) - Group-based optimization
- [Algorithms Overview](overview.md) - All supported algorithms