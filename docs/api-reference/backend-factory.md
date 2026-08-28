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

### `create_distill_trainer()`

Create a knowledge distillation trainer (offline or online, including SDFT self-distillation).

::: core.backend_factory.create_distill_trainer
    options:
      show_source: true
      heading_level: 3

**Example**:

```python
from aligntune.core.backend_factory import create_distill_trainer

trainer = create_distill_trainer(
    student_model="Qwen/Qwen2.5-0.5B-Instruct",
    teacher_model="Qwen/Qwen2.5-7B-Instruct",
    dataset_name="wikitext",
    backend="trl",
    temperature=3.0,
    alpha=0.5,
    loss_type="kl",
)
trainer.train()
```

### `create_es_trainer()`

Create an Evolution Strategies (gradient-free, population-based) trainer for LoRA optimization.

::: core.backend_factory.create_es_trainer
    options:
      show_source: true
      heading_level: 3

**Example**:

```python
from aligntune.core.backend_factory import create_es_trainer

trainer = create_es_trainer(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    dataset_name="openai/gsm8k",
    backend="es",
    population_size=64,
    sigma=0.5,
    num_iterations=1000,
    reward_type="math_correctness",
)
trainer.train()
```

### `create_raft_trainer()`

Create a RAFT (Retrieval Augmented Fine-Tuning) trainer. Note that RAFT is only reachable via this
standalone function, it is not wired into `BackendFactory`'s algorithm dispatch like other RL algorithms.

::: core.backend_factory.create_raft_trainer
    options:
      show_source: true
      heading_level: 3

**Example**:

```python
from aligntune.core.backend_factory import create_raft_trainer

trainer = create_raft_trainer(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    train_examples=[
        {
            "question": "What is the capital of France?",
            "answer": "Paris",
            "golden_docs": [{"title": "France", "text": "The capital of France is Paris."}],
            "distractor_docs": [],
        },
    ],
    backend="trl",
)
trainer.train()
```

### `create_tokenization_trainer()`

Create a tokenization trainer for vocabulary adaptation (continued-BPE extension or naive extension,
with optional pruning).

::: core.backend_factory.create_tokenization_trainer
    options:
      show_source: true
      heading_level: 3

**Example**:

```python
from aligntune.core.backend_factory import create_tokenization_trainer

trainer = create_tokenization_trainer(
    base_model="meta-llama/Llama-2-7b-hf",
    target_languages=["hi"],
    dataset_name="wikimedia/wikipedia",
    config_name="20231101.hi",
    num_new_tokens=20000,
)
result = trainer.train()
```

### `merge_models()`

Merge multiple models or LoRA adapters using mergekit (linear, task_arithmetic, ram).

::: core.backend_factory.merge_models
    options:
      show_source: true
      heading_level: 3

**Example**:

```python
from aligntune.core.backend_factory import merge_models

merged_path = merge_models(
    models=["finetuned_model1", "finetuned_model2"],
    method="linear",
    weights=[0.5, 0.5],
    output_path="./merged",
)
```

### `merge_models_from_yaml()`

Merge models using an existing mergekit YAML config file, for advanced features (weight gradients,
filters, layer ranges) not exposed by `merge_models()`'s keyword API.

::: core.backend_factory.merge_models_from_yaml
    options:
      show_source: true
      heading_level: 3

**Example**:

```python
from aligntune.core.backend_factory import merge_models_from_yaml

merged_path = merge_models_from_yaml(
    yaml_path="merge_config.yaml",
    output_path="./merged",
)
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