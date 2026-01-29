# Configuration Classes API Reference

Complete API reference for AlignTune configuration classes.

---

## SFT Configuration

### `SFTConfig`

Main configuration class for SFT training.

::: core.sft.config.SFTConfig
 options:
 show_source: true
 heading_level: 3

**Example**:

```python
from aligntune.core.sft.config import SFTConfig, ModelConfig, DatasetConfig, TrainingConfig, LoggingConfig

config = SFTConfig(
 model=ModelConfig(name_or_path="meta-llama/Llama-3.2-3B-Instruct"),
 dataset=DatasetConfig(name="tatsu-lab/alpaca"),
 train=TrainingConfig(epochs=3, learning_rate=5e-5),
 logging=LoggingConfig(output_dir="./output")
)
```

### `ModelConfig` (SFT)

Model configuration for SFT training.

::: core.sft.config.ModelConfig
 options:
 show_source: true
 heading_level: 3

### `DatasetConfig` (SFT)

Dataset configuration for SFT training.

::: core.sft.config.DatasetConfig
 options:
 show_source: true
 heading_level: 3

### `TrainingConfig` (SFT)

Training configuration for SFT.

::: core.sft.config.TrainingConfig
 options:
 show_source: true
 heading_level: 3

### `EvaluationConfig` (SFT)

Evaluation configuration for SFT.

::: core.sft.config.EvaluationConfig
 options:
 show_source: true
 heading_level: 3

### `LoggingConfig` (SFT)

Logging configuration for SFT.

::: core.sft.config.LoggingConfig
 options:
 show_source: true
 heading_level: 3

### `TaskType` Enum (SFT)

Task type enumeration for SFT.

::: core.sft.config.TaskType
 options:
 show_source: true
 heading_level: 3

### `PrecisionType` Enum

Model precision type enumeration.

::: core.sft.config.PrecisionType
 options:
 show_source: true
 heading_level: 3

---

## RL Configuration

### `UnifiedConfig`

Main configuration class for RL training.

::: core.rl.config.UnifiedConfig
 options:
 show_source: true
 heading_level: 3

**Example**:

```python
from aligntune.core.rl.config import UnifiedConfig, AlgorithmType, ModelConfig, DatasetConfig, TrainingConfig, LoggingConfig

config = UnifiedConfig(
 algo=AlgorithmType.DPO,
 model=ModelConfig(name_or_path="microsoft/DialoGPT-medium"),
 datasets=[DatasetConfig(name="Anthropic/hh-rlhf")],
 train=TrainingConfig(epochs=1, learning_rate=1e-6),
 logging=LoggingConfig(output_dir="./output")
)
```

### `ModelConfig` (RL)

Model configuration for RL training.

::: core.rl.config.ModelConfig
 options:
 show_source: true
 heading_level: 3

### `DatasetConfig` (RL)

Dataset configuration for RL training.

::: core.rl.config.DatasetConfig
 options:
 show_source: true
 heading_level: 3

### `TrainingConfig` (RL)

Training configuration for RL.

::: core.rl.config.TrainingConfig
 options:
 show_source: true
 heading_level: 3

### `RewardConfig`

Reward function configuration.

::: core.rl.config.RewardConfig
 options:
 show_source: true
 heading_level: 3

### `RewardModelSourceConfig`

Reward model source configuration.

::: core.rl.config.RewardModelSourceConfig
 options:
 show_source: true
 heading_level: 3

### `RewardModelTrainingConfig`

Reward model training configuration.

::: core.rl.config.RewardModelTrainingConfig
 options:
 show_source: true
 heading_level: 3

### `LoggingConfig` (RL)

Logging configuration for RL.

::: core.rl.config.LoggingConfig
 options:
 show_source: true
 heading_level: 3

### `SampleLoggingConfig`

Sample logging configuration for qualitative generation.

::: core.rl.config.SampleLoggingConfig
 options:
 show_source: true
 heading_level: 3

### `DistributedConfig`

Distributed training configuration.

::: core.rl.config.DistributedConfig
 options:
 show_source: true
 heading_level: 3

### `AlgorithmType` Enum

RL algorithm type enumeration.

::: core.rl.config.AlgorithmType
 options:
 show_source: true
 heading_level: 3

### `PrecisionType` Enum (RL)

Model precision type enumeration for RL.

::: core.rl.config.PrecisionType
 options:
 show_source: true
 heading_level: 3

### `BackendType` Enum (RL)

Distributed training backend enumeration.

::: core.rl.config.BackendType
 options:
 show_source: true
 heading_level: 3

---

## Configuration Loaders

### `SFTConfigLoader`

Configuration loader for SFT configs.

::: core.sft.config_loader.SFTConfigLoader
 options:
 show_source: true
 heading_level: 3

### `ConfigLoader` (RL)

Configuration loader for RL configs.

::: core.rl.config_loader.ConfigLoader
 options:
 show_source: true
 heading_level: 3

---

## Helper Functions

### `create_instruction_following_config()`

Create configuration for instruction following.

::: core.sft.config.create_instruction_following_config
 options:
 show_source: true
 heading_level: 3

### `create_text_classification_config()`

Create configuration for text classification.

::: core.sft.config.create_text_classification_config
 options:
 show_source: true
 heading_level: 3

---

## Usage Examples

### Creating SFT Configuration

```python
from aligntune.core.sft.config import SFTConfig, ModelConfig, DatasetConfig, TrainingConfig, LoggingConfig

config = SFTConfig(
 model=ModelConfig(
 name_or_path="meta-llama/Llama-3.2-3B-Instruct",
 precision="bf16",
 max_seq_length=512
 ),
 dataset=DatasetConfig(
 name="tatsu-lab/alpaca",
 max_samples=1000,
 task_type="instruction_following"
 ),
 train=TrainingConfig(
 epochs=3,
 per_device_batch_size=4,
 learning_rate=5e-5
 ),
 logging=LoggingConfig(
 output_dir="./output",
 run_name="my_sft_training"
 )
)
```

### Creating RL Configuration

```python
from aligntune.core.rl.config import (
 UnifiedConfig, AlgorithmType, ModelConfig, 
 DatasetConfig, TrainingConfig, LoggingConfig
)

config = UnifiedConfig(
 algo=AlgorithmType.DPO,
 model=ModelConfig(
 name_or_path="microsoft/DialoGPT-medium",
 use_peft=True,
 lora_r=16
 ),
 datasets=[
 DatasetConfig(
 name="Anthropic/hh-rlhf",
 max_samples=1000
 )
 ],
 train=TrainingConfig(
 epochs=1,
 per_device_batch_size=1,
 learning_rate=1e-6,
 beta=0.1 # DPO parameter
 ),
 logging=LoggingConfig(
 output_dir="./output",
 run_name="my_dpo_training"
 )
)
```

### Loading from YAML

```python
from aligntune.core.rl.config_loader import ConfigLoader

config = ConfigLoader.load_from_yaml("config.yaml")
```

---

## Next Steps

- [Backend Factory](backend-factory.md) - Trainer creation
- [Trainers](trainers.md) - Trainer classes
- [User Guide](../user-guide/sft.md) - Usage guide