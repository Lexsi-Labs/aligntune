# API Reference Overview

AlignTune provides a comprehensive API for fine-tuning models. This section covers all available APIs.

## Core APIs

### Backend Factory

The main entry point for creating trainers:

```python
from aligntune.core.backend_factory import (
 create_sft_trainer,
 create_rl_trainer,
 BackendType,
 get_backend_status
)
```

**Functions**:
- `create_sft_trainer()` - Create SFT trainer
- `create_rl_trainer()` - Create RL trainer
- `get_backend_status()` - Check backend availability

See [Backend Factory](backend-factory.md) for details.

### Configuration Classes

Configuration classes for all training parameters:

```python
from aligntune.core.sft.config import SFTConfig, TaskType
from aligntune.core.rl.config import UnifiedConfig, AlgorithmType
```

See [Configuration Classes](configuration.md) for details.

### Trainers

Base trainer classes with common methods:

```python
from aligntune.core.sft.trainer_base import SFTTrainerBase
from aligntune.core.rl.trainer_base import TrainerBase
```

**Common Methods**:
- `train()` - Start training
- `evaluate()` - Evaluate model
- `save_model()` - Save model
- `load_checkpoint()` - Load checkpoint
- `predict()` - Generate predictions
- `push_to_hub()` - Push to HuggingFace Hub

See [Trainers](trainers.md) for details.

## Quick Reference

### SFT Training

```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 backend="trl",
 num_epochs=3
)
trainer.train()
```

### RL Training

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

## API Documentation

- [Unified API](unified-api.md) - Unified RLHF API system
- [Core API](core.md) - Core functions and utilities
- [Backend Factory](backend-factory.md) - Trainer creation
- [Configuration Classes](configuration.md) - All configuration options
- [Trainers](trainers.md) - Trainer classes and methods
- [Reward Functions Reference](reward-functions-reference.md) - Complete reward functions list
- [Reward Model Training Reference](reward-model-training-reference.md) - Reward model system architecture

## Next Steps

- [Getting Started](../getting-started/quickstart.md) - Quick start guide
- [User Guide](../user-guide/sft.md) - Detailed usage guide
- [Examples](../examples/overview.md) - Code examples