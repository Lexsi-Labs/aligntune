# Trainers API Reference

Complete API reference for AlignTune trainer classes.

---

## Base Trainers

### `TrainerBase` (RL)

Abstract base trainer for RL training with lifecycle management.

::: core.rl.trainer_base.TrainerBase
 options:
 show_source: true
 heading_level: 3

**Example**:

```python
from aligntune.core.rl.trainer_base import TrainerBase
from aligntune.core.rl.config import UnifiedConfig

# TrainerBase is abstract - use concrete implementations
# See Backend Trainers below
```

### `SFTTrainerBase`

Abstract base trainer for SFT training with lifecycle management.

::: core.sft.trainer_base.SFTTrainerBase
 options:
 show_source: true
 heading_level: 3

**Example**:

```python
from aligntune.core.sft.trainer_base import SFTTrainerBase
from aligntune.core.sft.config import SFTConfig

# SFTTrainerBase is abstract - use concrete implementations
# See Backend Trainers below
```

### `TrainingState`

Training state tracking dataclass.

::: core.rl.trainer_base.TrainingState
 options:
 show_source: true
 heading_level: 3

---

## Backend Trainers

### TRL Backends

#### `TRLSFTTrainer`

TRL backend for Supervised Fine-Tuning. Use [`create_sft_trainer()`](backend-factory.md#create_sft_trainer) with `backend="trl"` instead of instantiating directly.

**Example**:

```python
from aligntune.backends.trl.sft.sft import TRLSFTTrainer
from aligntune.core.sft.config import SFTConfig

config = SFTConfig(...)
trainer = TRLSFTTrainer(config)
trainer.train()
```

#### `TRLDPOTrainer`

TRL backend for Direct Preference Optimization. Use [`create_rl_trainer()`](backend-factory.md#create_rl_trainer) with `algorithm="dpo"` and `backend="trl"`.

**Example**:

```python
from aligntune.backends.trl.rl.dpo.dpo import TRLDPOTrainer
from aligntune.core.rl.config import UnifiedConfig, AlgorithmType

config = UnifiedConfig(algo=AlgorithmType.DPO, ...)
trainer = TRLDPOTrainer(config)
trainer.train()
```

#### `TRLPPOTrainer`

TRL backend for Proximal Policy Optimization.

TRL backend for Proximal Policy Optimization. Use [`create_rl_trainer()`](backend-factory.md#create_rl_trainer) with `algorithm="ppo"` and `backend="trl"`.

**Example**:

```python
from aligntune.backends.trl.rl.ppo.ppo import TRLPPOTrainer
from aligntune.core.rl.config import UnifiedConfig

config = UnifiedConfig(algo=AlgorithmType.PPO, ...)
trainer = TRLPPOTrainer(config)
trainer.train()
```

#### `TRLGRPOTrainer`

TRL backend for Group Relative Policy Optimization.

TRL backend for Group Relative Policy Optimization. Use [`create_rl_trainer()`](backend-factory.md#create_rl_trainer) with `algorithm="grpo"` and `backend="trl"`.

### Unsloth Backends

#### `UnslothSFTTrainer`

Unsloth backend for Supervised Fine-Tuning (faster).

Unsloth backend for Supervised Fine-Tuning (faster). Use [`create_sft_trainer()`](backend-factory.md#create_sft_trainer) with `backend="unsloth"`.

**Example**:

```python
from aligntune.backends.unsloth.sft.sft import UnslothSFTTrainer
from aligntune.core.sft.config import SFTConfig

config = SFTConfig(...)
trainer = UnslothSFTTrainer(config)
trainer.train()
```

#### `UnslothDPOTrainer`

Unsloth backend for Direct Preference Optimization.

Unsloth backend for Direct Preference Optimization. Use [`create_rl_trainer()`](backend-factory.md#create_rl_trainer) with `algorithm="dpo"` and `backend="unsloth"`.

#### `UnslothPPOTrainer`

Unsloth backend for Proximal Policy Optimization.

Unsloth backend for Proximal Policy Optimization. Use [`create_rl_trainer()`](backend-factory.md#create_rl_trainer) with `algorithm="ppo"` and `backend="unsloth"`.

#### `UnslothGRPOTrainer`

Unsloth backend for Group Relative Policy Optimization.

Unsloth backend for Group Relative Policy Optimization. Use [`create_rl_trainer()`](backend-factory.md#create_rl_trainer) with `algorithm="grpo"` and `backend="unsloth"`.

---

## Specialized Trainers

### `ClassificationTrainer`

Specialized trainer for text classification tasks.

Specialized trainer for text classification. Use with TRL backend via [`create_sft_trainer()`](backend-factory.md#create_sft_trainer) and task type configuration.

**Note:** Only available for TRL backend.

---

## Usage Examples

### SFT Training

```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
 model_name="meta-llama/Llama-3.2-3B-Instruct",
 dataset_name="tatsu-lab/alpaca",
 backend="trl"
)

# Train
results = trainer.train()

# Evaluate
metrics = trainer.evaluate()

# Save
model_path = trainer.save_model()

# Predict
result = trainer.predict("What is AI?")
```

### RL Training

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dpo",
 backend="trl"
)

# Train
results = trainer.train()

# Evaluate
metrics = trainer.evaluate()

# Save
model_path = trainer.save_model()

# Push to Hub
url = trainer.push_to_hub("username/my-model")
```

---

## Next Steps

- [Backend Factory](backend-factory.md) - Trainer creation functions
- [Configuration Classes](configuration.md) - Configuration options
- [User Guide](../user-guide/sft.md) - Usage guide