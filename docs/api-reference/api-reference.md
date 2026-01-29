# AlignTune API Reference

## Overview

This document provides comprehensive documentation for all AlignTune APIs, including detailed parameter descriptions, configuration options, and usage examples.

## Core Configuration Classes

### **UnifiedConfig**

The main configuration class that unifies all training parameters.

```python
from aligntune import UnifiedConfig, AlgorithmType, ModelConfig, DatasetConfig, TrainingConfig

config = UnifiedConfig(
 algo=AlgorithmType.DPO, # Required: Training algorithm
 model=ModelConfig(...), # Required: Model configuration
 datasets=[DatasetConfig(...)], # Required: Dataset configuration(s)
 train=TrainingConfig(...), # Optional: Training parameters
 rewards=[RewardConfig(...)], # Optional: Reward functions
 tasks=[...], # Optional: Task definitions
 distributed=DistributedConfig(...), # Optional: Distributed training
 logging=LoggingConfig(...), # Optional: Logging configuration
 chat_template="auto", # Optional: Chat template
 caching={}, # Optional: Caching configuration
)
```

### **AlgorithmType**

Supported training algorithms.

```python
from aligntune import AlgorithmType

# Available algorithms
AlgorithmType.PPO # Proximal Policy Optimization
AlgorithmType.DPO # Direct Preference Optimization
AlgorithmType.GRPO # Group Relative Policy Optimization
AlgorithmType.GSPO # Group Sequential Policy Optimization
```

## Model Configuration

### **ModelConfig**

```python
from aligntune import ModelConfig, PrecisionType

model_config = ModelConfig(
 name_or_path="microsoft/DialoGPT-medium", # Required: Model name or path
 sft_path=None, # Optional: Path to SFT model
 reward_path=None, # Optional: Path to reward model
 precision=PrecisionType.BF16, # Optional: Model precision
 quantization={}, # Optional: Quantization config
 attn_implementation="auto", # Optional: Attention implementation
 gradient_checkpointing=True, # Optional: Enable gradient checkpointing
 max_memory=None, # Optional: Memory mapping
 use_unsloth=False, # Optional: Enable Unsloth
 max_seq_length=2048 # Optional: Maximum sequence length
)
```

#### **ModelConfig Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name_or_path` | `str` | **Required** | HuggingFace model name or local path |
| `sft_path` | `str` | `None` | Path to pre-trained SFT model |
| `reward_path` | `str` | `None` | Path to reward model for PPO |
| `precision` | `PrecisionType` | `BF16` | Model precision (BF16, FP16, FP32) |
| `quantization` | `Dict` | `{}` | Quantization configuration |
| `attn_implementation` | `str` | `"auto"` | Attention implementation |
| `gradient_checkpointing` | `bool` | `True` | Enable gradient checkpointing |
| `max_memory` | `Dict` | `None` | Memory mapping configuration |
| `use_unsloth` | `bool` | `False` | Enable Unsloth acceleration |
| `max_seq_length` | `int` | `2048` | Maximum sequence length |

### **PrecisionType**

```python
from aligntune import PrecisionType

PrecisionType.BF16 # Brain Float 16 (recommended for modern GPUs)
PrecisionType.FP16 # Float 16 (good for older GPUs)
PrecisionType.FP32 # Float 32 (CPU training)
```

## Dataset Configuration

### **DatasetConfig**

```python
from aligntune import DatasetConfig

dataset_config = DatasetConfig(
 name="Anthropic/hh-rlhf", # Required: Dataset name
 split="train", # Optional: Dataset split
 percent=100, # Optional: Percentage to use
 max_samples=None, # Optional: Maximum samples
 task_type="conversation", # Optional: Task type
 weight=1.0, # Optional: Dataset weight
 column_mapping={}, # Optional: Column mapping
 preprocessing={} # Optional: Preprocessing config
)
```

#### **DatasetConfig Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | **Required** | HuggingFace dataset name or local path |
| `split` | `str` | `"train"` | Dataset split to use |
| `percent` | `int` | `100` | Percentage of dataset to use (1-100) |
| `max_samples` | `int` | `None` | Maximum number of samples |
| `task_type` | `str` | `"conversation"` | Type of task |
| `weight` | `float` | `1.0` | Weight for multi-dataset training |
| `column_mapping` | `Dict` | `{}` | Map dataset columns to expected names |
| `preprocessing` | `Dict` | `{}` | Preprocessing configuration |

## Training Configuration

### **TrainingConfig**

```python
from aligntune import TrainingConfig

training_config = TrainingConfig(
 per_device_batch_size=1, # Required: Batch size per device
 gradient_accumulation_steps=1, # Optional: Gradient accumulation
 max_steps=1000, # Optional: Maximum training steps
 eval_interval=100, # Optional: Evaluation interval
 save_interval=500, # Optional: Save interval
 learning_rate=5e-5, # Optional: Learning rate
 weight_decay=0.01, # Optional: Weight decay
 warmup_steps=100, # Optional: Warmup steps
 max_length=512, # Optional: Maximum sequence length
 max_prompt_length=256, # Optional: Maximum prompt length
 temperature=0.7, # Optional: Generation temperature
 beta=0.1, # Optional: DPO beta parameter
 kl_coef=0.1, # Optional: PPO KL coefficient
 cliprange=0.2, # Optional: PPO clip range
 rollout_batch_size=1, # Optional: PPO rollout batch size
 ppo_epochs=4, # Optional: PPO epochs
 vf_coef=0.1, # Optional: PPO value function coefficient
 ent_coef=0.0, # Optional: PPO entropy coefficient
 target_kl=0.01, # Optional: PPO target KL
 init_kl_coef=0.2, # Optional: PPO initial KL coefficient
 adap_kl_ctrl=True, # Optional: PPO adaptive KL control
 gamma=1.0, # Optional: PPO discount factor
 lam=0.95, # Optional: PPO GAE lambda
 whiten_rewards=False # Optional: PPO reward whitening
)
```

#### **TrainingConfig Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `per_device_batch_size` | `int` | **Required** | Batch size per device |
| `gradient_accumulation_steps` | `int` | `1` | Gradient accumulation steps |
| `max_steps` | `int` | `1000` | Maximum training steps |
| `eval_interval` | `int` | `100` | Steps between evaluations |
| `save_interval` | `int` | `500` | Steps between saves |
| `learning_rate` | `float` | `5e-5` | Learning rate |
| `weight_decay` | `float` | `0.01` | Weight decay |
| `warmup_steps` | `int` | `100` | Warmup steps |
| `max_length` | `int` | `512` | Maximum sequence length |
| `max_prompt_length` | `int` | `256` | Maximum prompt length |
| `temperature` | `float` | `0.7` | Generation temperature |
| `beta` | `float` | `0.1` | DPO beta parameter |
| `kl_coef` | `float` | `0.1` | PPO KL coefficient |
| `cliprange` | `float` | `0.2` | PPO clip range |
| `rollout_batch_size` | `int` | `1` | PPO rollout batch size |
| `ppo_epochs` | `int` | `4` | PPO epochs per update |
| `vf_coef` | `float` | `0.1` | PPO value function coefficient |
| `ent_coef` | `float` | `0.0` | PPO entropy coefficient |
| `target_kl` | `float` | `0.01` | PPO target KL divergence |
| `init_kl_coef` | `float` | `0.2` | PPO initial KL coefficient |
| `adap_kl_ctrl` | `bool` | `True` | PPO adaptive KL control |
| `gamma` | `float` | `1.0` | PPO discount factor |
| `lam` | `float` | `0.95` | PPO GAE lambda |
| `whiten_rewards` | `bool` | `False` | PPO reward whitening |

## Reward Configuration

### **RewardConfig**

```python
from aligntune import RewardConfig, RewardType

reward_config = RewardConfig(
 reward_type=RewardType.SAFETY, # Required: Reward type
 weight=1.0, # Optional: Reward weight
 params={}, # Optional: Reward parameters
 model_name=None, # Optional: Model for reward
 device="auto", # Optional: Device for reward
 cache_dir=None # Optional: Cache directory
)
```

#### **RewardConfig Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reward_type` | `RewardType` | **Required** | Type of reward function |
| `weight` | `float` | `1.0` | Weight of this reward |
| `params` | `Dict` | `{}` | Parameters for reward function |
| `model_name` | `str` | `None` | Model name for model-based rewards |
| `device` | `str` | `"auto"` | Device for reward computation |
| `cache_dir` | `str` | `None` | Cache directory for models |

## Distributed Configuration

### **DistributedConfig**

```python
from aligntune import DistributedConfig, BackendType

distributed_config = DistributedConfig(
 backend=BackendType.SINGLE, # Required: Distributed backend
 num_processes=1, # Optional: Number of processes
 master_addr="localhost", # Optional: Master address
 master_port=29500, # Optional: Master port
 seed=42, # Optional: Random seed
 deepspeed_config={}, # Optional: DeepSpeed config
 fsdp_config={} # Optional: FSDP config
)
```

#### **DistributedConfig Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `BackendType` | **Required** | Distributed backend type |
| `num_processes` | `int` | `1` | Number of processes |
| `master_addr` | `str` | `"localhost"` | Master node address |
| `master_port` | `int` | `29500` | Master node port |
| `seed` | `int` | `42` | Random seed |
| `deepspeed_config` | `Dict` | `{}` | DeepSpeed configuration |
| `fsdp_config` | `Dict` | `{}` | FSDP configuration |

### **BackendType**

```python
from aligntune import BackendType

BackendType.SINGLE # Single GPU/CPU training
BackendType.DDP # Distributed Data Parallel
BackendType.FSDP # Fully Sharded Data Parallel
BackendType.DEEPSPEED # DeepSpeed ZeRO
```

## Logging Configuration

### **LoggingConfig**

```python
from aligntune import LoggingConfig

logging_config = LoggingConfig(
 output_dir="./output", # Required: Output directory
 run_name="my_experiment", # Optional: Run name
 loggers=["tensorboard"], # Optional: Logging backends
 level="INFO", # Optional: Log level
 save_steps=500, # Optional: Save interval
 eval_steps=100, # Optional: Eval interval
 report_to=["tensorboard"], # Optional: Reporting backends
 logging_dir="./logs", # Optional: Logging directory
 disable_tqdm=False, # Optional: Disable progress bars
 push_to_hub=False, # Optional: Push to HuggingFace Hub
 hub_model_id=None, # Optional: Hub model ID
 hub_token=None # Optional: Hub token
)
```

#### **LoggingConfig Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `str` | **Required** | Output directory for checkpoints |
| `run_name` | `str` | `None` | Name for this training run |
| `loggers` | `List[str]` | `["tensorboard"]` | Logging backends |
| `level` | `str` | `"INFO"` | Logging level |
| `save_steps` | `int` | `500` | Steps between saves |
| `eval_steps` | `int` | `100` | Steps between evaluations |
| `report_to` | `List[str]` | `["tensorboard"]` | Reporting backends |
| `logging_dir` | `str` | `"./logs"` | Logging directory |
| `disable_tqdm` | `bool` | `False` | Disable progress bars |
| `push_to_hub` | `bool` | `False` | Push to HuggingFace Hub |
| `hub_model_id` | `str` | `None` | Hub model ID |
| `hub_token` | `str` | `None` | Hub authentication token |

## Backend Factory

### **BackendFactory**

```python
from aligntune import BackendFactory, BackendConfig, TrainingType, BackendType, RLAlgorithm

# Create trainer with specific backend
backend_config = BackendConfig(
 training_type=TrainingType.RL,
 backend=BackendType.TRL,
 algorithm=RLAlgorithm.DPO
)

trainer = BackendFactory.create_trainer(config, backend_config)
```

### **Convenience Functions**

```python
from aligntune import create_sft_trainer, create_rl_trainer

# Create SFT trainer
sft_trainer = create_sft_trainer(config, backend="trl")

# Create RL trainer
rl_trainer = create_rl_trainer(config, algorithm="dpo", backend="unsloth")
```

## Configuration Loading

### **ConfigLoader**

```python
from aligntune import ConfigLoader

# Load from YAML
config = ConfigLoader.load_from_yaml("config.yaml")

# Load from dictionary
config = ConfigLoader.load_from_dict(config_dict)

# Validate configuration
ConfigLoader.validate_config(config)

# Save resolved configuration
ConfigLoader.save_resolved_config(config, output_dir)
```

## Complete Example

```python
from aligntune import (
 UnifiedConfig, AlgorithmType, ModelConfig, DatasetConfig, 
 TrainingConfig, RewardConfig, RewardType, DistributedConfig, 
 BackendType, LoggingConfig, create_rl_trainer
)

# Create complete configuration
config = UnifiedConfig(
 algo=AlgorithmType.DPO,
 model=ModelConfig(
 name_or_path="microsoft/DialoGPT-medium",
 precision=PrecisionType.BF16,
 use_unsloth=True,
 max_seq_length=2048
 ),
 datasets=[
 DatasetConfig(
 name="Anthropic/hh-rlhf",
 split="train",
 percent=10,
 max_samples=1000
 )
 ],
 train=TrainingConfig(
 per_device_batch_size=2,
 gradient_accumulation_steps=4,
 max_steps=1000,
 learning_rate=5e-5,
 beta=0.1,
 temperature=0.7
 ),
 rewards=[
 RewardConfig(
 reward_type=RewardType.SAFETY,
 weight=2.0,
 params={"strict": True}
 ),
 RewardConfig(
 reward_type=RewardType.HELPFULNESS,
 weight=1.5
 )
 ],
 distributed=DistributedConfig(
 backend=BackendType.SINGLE,
 seed=42
 ),
 logging=LoggingConfig(
 output_dir="./output/dpo_experiment",
 run_name="dpo_hh_rlhf",
 loggers=["tensorboard", "wandb"]
 )
)

# Create and train
trainer = create_rl_trainer(config, algorithm="dpo", backend="unsloth")
trainer.train()
```

---
 For more examples and use cases, see the [Examples Documentation](../examples/overview.md) and [Reward Functions Reference](reward-functions-reference.md).