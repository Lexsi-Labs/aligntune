# Online Iterative DPO (Online-DPO)

Online-DPO takes standard Direct Preference Optimization and runs it iteratively inside an active-learning feedback loop.

## Overview

In standard DPO, you require a static dataset of fixed pairs (chosen, rejected) gathered offline. In Online-DPO, the model receives prompt-only rows, generates responses on the fly, and sends them to a neural reward model or configured reward function. The resulting scores are used to build the online preference signal, and the model is iteratively tuned on its own generated distribution.

### When to use Online-DPO?
- When you have a neural reward model or a registry/custom reward function.
- When offline DPO plateaus because the model's generated distribution drifts away from the static dataset's distribution.

## Configuration

To use Online-DPO, pass `algorithm="online_dpo"` to the factory method:

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="Anthropic/hh-rlhf",
    algorithm="online_dpo",
    backend="trl",
    reward_model_name="your-active-reward-model",
)

trainer.train()
```

### Reward Sources

Online-DPO requires at least one reward source: a neural reward model, a
registry reward, or a custom reward function.

#### Neural Reward Model

```python
trainer = create_rl_trainer(
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="Anthropic/hh-rlhf",
    algorithm="online_dpo",
    backend="trl",
    reward_model_name="your-reward-model",
)
```

#### Registry Reward

```python
trainer = create_rl_trainer(
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="openai/gsm8k",
    algorithm="online_dpo",
    backend="trl",
    reward_functions=["math_correctness"],
    reward_function_weights=[1.0],
)
```

#### Custom Reward

```python
def my_reward(text, reference=None, **kwargs):
    return 1.0 if "correct" in text.lower() else 0.0

trainer = create_rl_trainer(
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="your/dataset",
    algorithm="online_dpo",
    backend="trl",
    rewards=[my_reward],
)
```

You must provide either `reward_model_name` or at least one configured reward
function.

### Dataset Format

Online-DPO uses the GRPO data path internally. After processing, each training
row must contain a prompt suitable for live generation:

```python
{
    "prompt": [
        {"role": "user", "content": "Solve this problem."}
    ]
}
```

`chosen` and `rejected` responses are not required for online generation. If a
source dataset contains preference pairs, DataManager converts the source into
prompt-compatible rows before training.

### Configuration Options

--8<-- "docs/PARAMETERS.md:online-dpo"

The main factory-level reward and dataset options are:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reward_model_name` | `None` | Neural reward model used to score generated completions |
| `reward_functions` | `None` | Registry reward names, such as `["math_correctness"]` |
| `reward_function_weights` | `None` | Weights for configured registry rewards |
| `rewards` | `None` | Custom reward callables or reward configurations |
| `max_samples` | `None` | Limit the number of training examples |
| `val_split_ratio` | `None` | Fraction reserved for validation |
| `test_split_ratio` | `None` | Fraction reserved for testing |

### Backend Note

Online-DPO uses TRL's experimental `OnlineDPOTrainer` on both backends. The
Unsloth backend accelerates model loading and inference; it does not use a
separate Online-DPO algorithm implementation.

## See Also

- [Algorithms Overview](overview.md)
- [Online-DPO Parameters](../PARAMETERS.md#online-dpo) - Full parameter reference
