# Backend Factory API

The Backend Factory is the main entry point for creating trainers in AlignTune.

---

## Overview

The `BackendFactory` provides functions to create SFT and RL trainers with automatic backend selection and fallback handling.

---

## BackendFactory Class

Complete API reference for the `BackendFactory` class.

::: core.backend_factory.BackendFactory
 options:
 show_source: true
 heading_level: 3

---

## Factory Functions

### `create_sft_trainer()`

Create a Supervised Fine-Tuning trainer.

::: core.backend_factory.create_sft_trainer
 options:
 show_source: true
 heading_level: 3

**Example**:

```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
 model_name="meta-llama/Llama-3.2-3B-Instruct",
 dataset_name="tatsu-lab/alpaca",
 backend="trl",
 num_epochs=3,
 batch_size=4
)
trainer.train()
```

### `create_rl_trainer()`

Create a Reinforcement Learning trainer.

::: core.backend_factory.create_rl_trainer
 options:
 show_source: true
 heading_level: 3

**Example**:

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dpo",
 backend="trl",
 num_epochs=1
)
trainer.train()
```

### `get_backend_status()`

Get the availability status of all backends.

::: core.backend_factory.get_backend_status
 options:
 show_source: true
 heading_level: 3

**Example**:

```python
from aligntune.core.backend_factory import get_backend_status

status = get_backend_status()
print(f"TRL available: {status['trl_available']}")
print(f"Unsloth available: {status['unsloth_available']}")
```

### `list_backends()`

List all available backends.

::: core.backend_factory.list_backends
 options:
 show_source: true
 heading_level: 3

---

## Enums and Types

### `BackendType` Enum

::: core.backend_factory.BackendType
 options:
 show_source: true
 heading_level: 3

### `TrainingType` Enum

::: core.backend_factory.TrainingType
 options:
 show_source: true
 heading_level: 3

### `RLAlgorithm` Enum

::: core.backend_factory.RLAlgorithm
 options:
 show_source: true
 heading_level: 3

### `BackendConfig` Dataclass

::: core.backend_factory.BackendConfig
 options:
 show_source: true
 heading_level: 3

## Examples

### SFT with TRL Backend

```python
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 backend="trl"
)
```

### SFT with Unsloth Backend

```python
trainer = create_sft_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="tatsu-lab/alpaca",
 backend="unsloth"
)
```

### DPO Training

```python
trainer = create_rl_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dpo",
 backend="trl"
)
```

### PPO Training

```python
trainer = create_rl_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="ppo",
 backend="unsloth"
)
```

## Error Handling

### Backend Not Available

```python
try:
 trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 backend="unsloth"
 )
except ValueError as e:
 print(f"Backend not available: {e}")
 # Falls back to TRL automatically
```

### Invalid Algorithm

```python
try:
 trainer = create_rl_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="invalid"
 )
except ValueError as e:
 print(f"Invalid algorithm: {e}")
```

## Next Steps

- [Configuration Classes](configuration.md) - Configuration options
- [Trainers](trainers.md) - Trainer methods
- [User Guide](../user-guide/sft.md) - Usage guide