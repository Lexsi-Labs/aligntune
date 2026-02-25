# Dr. GRPO (GRPO Done Right)

Dr. GRPO is a corrected and improved version of the original GRPO algorithm.

## Overview

GRPO Done Right (Dr. GRPO) addresses optimization biases identified in the original GRPO implementation, providing more accurate and stable training.

## Key Features

- **Bias correction**: Fixes optimization biases in original GRPO
- **Improved stability**: More reliable convergence
- **Enhanced performance**: Better results on benchmark tasks
- **Multi-backend support**: Available in both TRL and Unsloth

## Usage

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dr-grpo",
 backend="trl",
 num_epochs=1,
 batch_size=2,
 learning_rate=1e-6
)

trainer.train()
```

## Algorithm Details

Dr. GRPO corrects issues in the original GRPO by:

1. **Bias correction**: Proper normalization of group advantages
2. **Improved clipping**: Better handling of extreme values
3. **Enhanced sampling**: More efficient sample utilization

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `group_size` | int | 8 | Number of samples per group |
| `bias_correction` | bool | true | Enable bias correction |
| `clip_range` | float | 0.2 | Advantage clipping range |

## Benefits

- More accurate optimization than original GRPO
- Better convergence properties
- Improved performance on complex tasks

## See Also

- [GRPO Algorithm](grpo.md) - Original group optimization
- [Algorithms Overview](overview.md) - All supported algorithms