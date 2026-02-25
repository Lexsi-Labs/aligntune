# GSPO (Group Sequential Policy Optimization)

GSPO is a variant of GRPO that performs sequential optimization within groups.

## Overview

Group Sequential Policy Optimization (GSPO) extends GRPO by performing sequential policy updates within each group, allowing for more fine-grained optimization of complex behaviors.

## Key Features

- **Sequential updates**: Multiple optimization steps per group
- **Fine-grained control**: Better handling of complex reward landscapes
- **TRL-only**: Currently only available in TRL backend

## Usage

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="gspo",
 backend="trl", # GSPO only available in TRL
 num_epochs=1,
 batch_size=2,
 learning_rate=1e-6
)

trainer.train()
```

## Algorithm Details

GSPO performs multiple sequential policy updates within each group, allowing for:

1. **Progressive refinement**: Each step improves upon the previous
2. **Complex optimization**: Better handling of multi-objective rewards
3. **Stability**: Reduced variance through sequential updates

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `group_size` | int | 4 | Number of samples per group |
| `seq_steps` | int | 3 | Number of sequential steps |
| `beta` | float | 0.01 | KL penalty coefficient |

## Limitations

- Currently only available in TRL backend
- Higher computational cost than standard GRPO
- May require tuning of sequential parameters

## See Also

- [GRPO Algorithm](grpo.md) - Base group optimization
- [Algorithms Overview](overview.md) - All supported algorithms