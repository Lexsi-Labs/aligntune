# SPIN (Self-Play Fine-Tuning)

SPIN (Self-Play Fine-Tuning) enables a Large Language Model to improve its own performance through self-play, removing the need for external preference labels from human annotators or larger models (like GPT-4).

## Overview

SPIN pits a language model against iterations of itself. The model learns to differentiate between its newly generated (synthetic) responses and ground-truth human responses from an SFT dataset, iteratively pushing the model's distribution closer to the human data.

### When to use SPIN?
- When you have a high-quality SFT dataset but no preference ranking data.
- When you want the model to iteratively self-improve beyond its baseline supervised performance.

## Configuration

To use SPIN, pass `algorithm="spin"` to the factory method:

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="your_sft_dataset",
    algorithm="spin",
    backend="trl"
)

trainer.train()
```

### Configuration Options

--8<-- "docs/PARAMETERS.md:spin"

## See Also

- [Algorithms Overview](overview.md)
- [SPIN Parameters](../PARAMETERS.md#spin-self-play-fine-tuning) - Full parameter reference
