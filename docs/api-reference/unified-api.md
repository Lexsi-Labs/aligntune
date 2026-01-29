# Unified RLHF API Documentation

This document describes the unified RLHF training system that provides a consistent interface for training with PPO, DPO, GRPO, GSPO, DAPO, and Dr. GRPO algorithms.

## Overview

The unified system provides:
- **Consistent API**: Same interface for all RLHF algorithms
- **No Interactive Prompts**: Everything is configuration-driven
- **No Placeholder Defaults**: All critical parameters must be explicitly provided
- **Distributed Training**: Support for DDP, FSDP, and DeepSpeed
- **Comprehensive Logging**: TensorBoard and WandB integration
- **Dataset Caching**: Efficient caching with hash-based keys
- **Extensible**: Easy to add new algorithms, datasets, and rewards

## Quick Start

### Python Library Usage

```python
from aligntune import (
 UnifiedConfig, AlgorithmType, ModelConfig, DatasetConfig,
 TrainingConfig, ConfigLoader, TrainerFactory
)

# Create configuration
config = UnifiedConfig(
 algo=AlgorithmType.PPO,
 model=ModelConfig(name_or_path="microsoft/DialoGPT-small"),
 datasets=[DatasetConfig(name="Anthropic/hh-rlhf", percent=1)],
 train=TrainingConfig(max_steps=100)
)

# Create and run trainer
trainer = TrainerFactory.create_trainer(config)
trainer.train()
```

### CLI Usage

```bash
# Train with YAML config
python -m aligntune.cli.train_unified --config examples/configs/ppo_minimal.yaml

# Train with CLI arguments
python -m aligntune.cli.train_unified \
 --algo ppo \
 --model microsoft/DialoGPT-small \
 --dataset Anthropic/hh-rlhf:train#percent=1 \
 --max-steps 100
```

## Configuration

### UnifiedConfig

The main configuration class that contains all training parameters:

```python
@dataclass
class UnifiedConfig:
 algo: AlgorithmType # Algorithm to use
 model: ModelConfig # Model configuration
 datasets: List[DatasetConfig] # Dataset configurations
 tasks: List[Dict[str, Any]] # Task definitions
 rewards: List[RewardConfig] # Reward functions
 train: TrainingConfig # Training parameters
 distributed: DistributedConfig # Distributed training
 logging: LoggingConfig # Logging configuration
 chat_template: Optional[str] # Chat template
 caching: Dict[str, Any] # Caching configuration
```

### Algorithm Types

```python
class AlgorithmType(Enum):
 PPO = "ppo" # Proximal Policy Optimization
 DPO = "dpo" # Direct Preference Optimization
 GRPO = "grpo" # Group Relative Policy Optimization
 GSPO = "gspo" # Generalized Scoring Proximal Objective
 DAPO = "dapo" # Decouple Clip and Dynamic sAmpling Policy Optimization
 DRGRPO = "drgrpo" # Dr. GRPO (GRPO Done Right)
```

### Model Configuration

```python
@dataclass
class ModelConfig:
 name_or_path: str # Model name or path (required)
 sft_path: Optional[str] # SFT checkpoint path
 reward_path: Optional[str] # Reward model path
 precision: PrecisionType # Model precision
 quantization: Dict[str, Any] # Quantization settings
 attn_implementation: str # Attention implementation
 gradient_checkpointing: bool # Enable gradient checkpointing
 max_memory: Optional[Dict[str, str]] # Memory limits per device
```

### Dataset Configuration

```python
@dataclass
class DatasetConfig:
 name: str # Dataset name (required)
 split: str # Dataset split
 percent: Optional[float] # Percentage of dataset to use
 max_samples: Optional[int] # Maximum number of samples
 column_mapping: Dict[str, str] # Column mapping
 task_type: str # Task type
 weight: float # Dataset weight
 chat_template: Optional[str] # Chat template
```

### Training Configuration

```python
@dataclass
class TrainingConfig:
 per_device_batch_size: int # Batch size per device
 gradient_accumulation_steps: int # Gradient accumulation steps
 max_steps: Optional[int] # Maximum training steps
 epochs: Optional[int] # Number of epochs
 eval_interval: int # Evaluation interval
 save_interval: int # Save interval
 rollout_batch_size: int # Rollout batch size
 kl_coef: float # KL coefficient
 cliprange: float # PPO clip range
 num_ppo_epochs: Optional[int] # PPO epochs
 learning_rate: float # Learning rate
```

## Dataset Specifications

### Format

```
name[:split][#percent=N|max=N][?map.key=value]
```

### Examples

```bash
# Basic dataset
Anthropic/hh-rlhf:train#percent=25

# With column mapping
HuggingFaceH4/ultrafeedback_binarized:train?map.prompt=prompt&map.chosen=chosen&map.rejected=rejected

# With max samples
gsm8k:train#max=1000
```

## Reward Specifications

### Format

```
type:weight:param1=value1:param2=value2
```

### Examples

```bash
# Length reward
length:weight=1.0:min_length=10:max_length=200

# Math reward
math:weight=0.8:tolerance=1e-6

# Safety reward with shield
safety:weight=0.5:strict=true
```

## Built-in Components

### Dataset Loaders

- `huggingface`: Load from HuggingFace Hub
- `local`: Load from local files

### Schema Adapters

- `chat`: Chat format (messages)
- `hh-rlhf`: Anthropic HH-RLHF format
- `ultrafeedback`: UltraFeedback format
- `alpaca`: Alpaca format

### Reward Functions

- `length`: Text length reward
- `sentiment`: Sentiment analysis reward
- `coherence`: Text coherence reward
- `math`: Mathematical correctness reward
- `code`: Code quality reward
- `safety`: Safety check reward

### Tasks

- `conversation`: Conversational AI
- `classification`: Text classification
- `summarization`: Text summarization
- `math`: Mathematical reasoning
- `code`: Code generation

## Distributed Training

### Backends

- `single`: Single GPU/CPU training
- `ddp`: DistributedDataParallel via accelerate
- `fsdp`: FullyShardedDataParallel
- `deepspeed`: DeepSpeed ZeRO-2/3

### Example Configuration

```yaml
distributed:
 backend: ddp
 nodes: 1
 gpus_per_node: 2
 seed: 42
```

## Logging

### Supported Loggers

- `tensorboard`: TensorBoard logging
- `wandb`: Weights & Biases logging

### Example Configuration

```yaml
logging:
 loggers: ["tensorboard", "wandb"]
 output_dir: "./output"
 run_name: "my_training_run"
```

## Caching

### Configuration

```yaml
caching:
 root: "./cache"
 enabled: true
```

### Features

- Hash-based cache keys
- Automatic cache validation
- Cache cleanup utilities
- Support for multiple datasets

## Examples

### Minimal PPO Training

```yaml
algo: ppo
model:
 name_or_path: "microsoft/DialoGPT-small"
 precision: bf16
datasets:
 - name: "Anthropic/hh-rlhf"
 split: "train"
 percent: 1
train:
 per_device_batch_size: 1
 max_steps: 100
 eval_interval: 50
distributed:
 backend: single
logging:
 output_dir: "./output"
```

### Multi-task Training

```yaml
algo: ppo
model:
 name_or_path: "microsoft/DialoGPT-medium"
 precision: bf16
datasets:
 - name: "gsm8k"
 split: "train"
 max_samples: 5000
 task_type: "math"
 weight: 0.4
 - name: "sahil2801/CodeAlpaca-20k"
 split: "train"
 percent: 20
 task_type: "code"
 weight: 0.3
 - name: "Anthropic/hh-rlhf"
 split: "train"
 percent: 10
 task_type: "conversation"
 weight: 0.3
tasks:
 - name: "math"
 weight: 0.4
 - name: "code"
 weight: 0.3
 - name: "conversation"
 weight: 0.3
rewards:
 - type: "math"
 weight: 0.4
 - type: "code"
 weight: 0.3
 - type: "length"
 weight: 0.2
 - type: "safety"
 weight: 0.1
 shield: true
```

## Migration Guide

### From Old API

1. **Replace interactive prompts**: Use YAML configuration files
2. **Update imports**: Use unified system imports
3. **Update configuration**: Use `UnifiedConfig` instead of algorithm-specific configs
4. **Update training**: Use `TrainerFactory.create_trainer()`

### Example Migration

**Old (Interactive)**:
```python
from aligntune import PPOTrainer

trainer = PPOTrainer()
trainer.setup_interactive() # Interactive prompts
trainer.train()
```

**New (Unified)**:
```python
from aligntune import UnifiedConfig, TrainerFactory

config = UnifiedConfig(
 algo=AlgorithmType.PPO,
 model=ModelConfig(name_or_path="model_name"),
 datasets=[DatasetConfig(name="dataset_name")],
 train=TrainingConfig(max_steps=1000)
)

trainer = TrainerFactory.create_trainer(config)
trainer.train()
```

## Troubleshooting

### Common Issues

1. **Missing required fields**: All critical parameters must be explicitly provided
2. **Invalid dataset specs**: Check dataset specification format
3. **Invalid reward specs**: Check reward specification format
4. **Import errors**: Ensure all dependencies are installed

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Validation

Validate configuration before training:

```python
from aligntune import ConfigLoader

ConfigLoader.validate_config(config)
```

## API Reference

### Core Classes

- `UnifiedConfig`: Main configuration class
- `ConfigLoader`: Configuration loading and validation
- `TrainerFactory`: Trainer creation factory
- `TrainerBase`: Abstract base trainer class

### Registries

- `DatasetRegistry`: Dataset loader registry
- `RewardRegistry`: Reward function registry
- `TaskRegistry`: Task definition registry

### Utilities

- `UnifiedLogger`: Logging system
- `UnifiedEvaluator`: Evaluation system
- `DatasetCache`: Dataset caching system
- `RolloutEngine`: Rollout generation engine

### Model Wrappers

- `PolicyModel`: Policy model wrapper
- `ReferenceModel`: Reference model wrapper
- `ValueModel`: Value model wrapper