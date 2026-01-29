# Configuration Files

Detailed guide for creating and using YAML configuration files with the AlignTune CLI.

## Overview

Configuration files provide a declarative way to specify complex training setups, making it easy to:

- Share training configurations
- Version control experiments
- Reproduce results
- Manage complex setups

## Basic Structure

```yaml
# Basic configuration structure
algo: dpo # Algorithm (dpo, ppo, grpo, etc.)

model:
 name_or_path: "microsoft/DialoGPT-medium"
 max_seq_length: 512

datasets:
 - name: "Anthropic/hh-rlhf"
 max_samples: 1000

train:
 max_steps: 1000
 learning_rate: 5e-5
 per_device_batch_size: 4

logging:
 output_dir: "./output"
 run_name: "my_experiment"
```

## Configuration Sections

### Algorithm Configuration

```yaml
algo: dpo # Options: dpo, ppo, grpo, gspo, dapo, dr-grpo, gbmpo, counterfactual-grpo, bolt
```

### Model Configuration

```yaml
model:
 name_or_path: "microsoft/DialoGPT-medium" # Required: Model identifier
 sft_path: null # Optional: Pre-trained SFT model path
 reward_path: null # Optional: Reward model path
 precision: "bf16" # Optional: bf16, fp16, fp32
 max_seq_length: 2048 # Optional: Maximum sequence length
 use_unsloth: false # Optional: Enable Unsloth
 gradient_checkpointing: true # Optional: Enable gradient checkpointing
```

### Dataset Configuration

```yaml
datasets:
 - name: "Anthropic/hh-rlhf" # Required: Dataset name
 split: "train" # Optional: Dataset split
 percent: 100 # Optional: Percentage to use (1-100)
 max_samples: null # Optional: Maximum samples
 task_type: "conversation" # Optional: Task type
 weight: 1.0 # Optional: Dataset weight for multi-dataset training
```

### Training Configuration

```yaml
train:
 per_device_batch_size: 4 # Required: Batch size per device
 gradient_accumulation_steps: 1 # Optional: Gradient accumulation
 max_steps: 1000 # Optional: Maximum training steps
 learning_rate: 5e-5 # Optional: Learning rate
 weight_decay: 0.01 # Optional: Weight decay
 warmup_steps: 100 # Optional: Warmup steps
 max_length: 512 # Optional: Maximum sequence length
 temperature: 0.7 # Optional: Generation temperature

 # RL-specific parameters
 beta: 0.1 # Optional: DPO beta parameter
 kl_coef: 0.1 # Optional: PPO KL coefficient
 cliprange: 0.2 # Optional: PPO clip range
```

### Reward Configuration (RL only)

```yaml
rewards:
 - reward_type: "safety" # Required: Reward function type
 weight: 1.0 # Optional: Reward weight
 params: {} # Optional: Reward parameters
 model_name: null # Optional: Model for reward function
 device: "auto" # Optional: Device for computation
```

### Logging Configuration

```yaml
logging:
 output_dir: "./output" # Required: Output directory
 run_name: null # Optional: Run name
 loggers: ["tensorboard"] # Optional: Logging backends
 level: "INFO" # Optional: Log level
 save_steps: 500 # Optional: Checkpoint save interval
 eval_steps: 100 # Optional: Evaluation interval
 report_to: ["tensorboard"] # Optional: Reporting backends
 push_to_hub: false # Optional: Push to HuggingFace Hub
 hub_model_id: null # Optional: Hub model ID
```

### Sample Logging Configuration

```yaml
sample_logging:
 enabled: true # Enable qualitative sample logging
 num_samples: 2 # Number of samples to generate
 max_new_tokens: 64 # Maximum tokens per sample
 interval_steps: null # Log every N steps
 percent_of_max_steps: 0.5 # Log at percentage of training
 prompts: null # Custom prompts (uses defaults if null)
 temperature: 0.7 # Sampling temperature
 top_p: 0.9 # Nucleus sampling parameter
```

## Complete Examples

### SFT Configuration

```yaml
algo: sft

model:
 name_or_path: "microsoft/DialoGPT-small"
 max_seq_length: 512

datasets:
 - name: "tatsu-lab/alpaca"
 max_samples: 1000

train:
 per_device_batch_size: 4
 max_steps: 1000
 learning_rate: 5e-5
 weight_decay: 0.01
 warmup_steps: 100

logging:
 output_dir: "./output/sft_training"
 run_name: "sft_experiment"
 save_steps: 500
 eval_steps: 100
```

### DPO Configuration

```yaml
algo: dpo

model:
 name_or_path: "microsoft/DialoGPT-medium"
 max_seq_length: 512

datasets:
 - name: "Anthropic/hh-rlhf"
 max_samples: 1000

train:
 per_device_batch_size: 2
 max_steps: 1000
 learning_rate: 1e-6
 beta: 0.1
 weight_decay: 0.01
 warmup_steps: 50

rewards:
 - reward_type: "safety"
 weight: 2.0
 - reward_type: "helpfulness"
 weight: 1.5

logging:
 output_dir: "./output/dpo_training"
 run_name: "dpo_experiment"
 save_steps: 500
 eval_steps: 100
```

### PPO Configuration

```yaml
algo: ppo

model:
 name_or_path: "microsoft/DialoGPT-medium"
 max_seq_length: 512

datasets:
 - name: "Anthropic/hh-rlhf"
 max_samples: 500

train:
 per_device_batch_size: 1
 max_steps: 1000
 learning_rate: 1e-6
 kl_coef: 0.1
 cliprange: 0.2
 vf_coef: 0.1
 ent_coef: 0.0

rewards:
 - reward_type: "length"
 weight: 0.5
 - reward_type: "sentiment"
 weight: 1.0
 - reward_type: "safety"
 weight: 2.0

sample_logging:
 enabled: true
 num_samples: 2
 max_new_tokens: 64
 percent_of_max_steps: 0.5

logging:
 output_dir: "./output/ppo_training"
 run_name: "ppo_experiment"
 save_steps: 500
 eval_steps: 100
```

## Using Configuration Files

### Command Line Usage

```bash
# Train with configuration file
aligntune train --config config.yaml

# Override specific values
aligntune train --config config.yaml --max-steps 2000 --learning-rate 1e-5
```



## Best Practices

### Organization

- Use descriptive filenames: `dpo_experiment_001.yaml`
- Group related configs in directories
- Version control your configurations

### Parameter Tuning

- Start with provided examples
- Gradually modify parameters
- Keep records of what worked

### Reproducibility

- Set explicit random seeds
- Record exact configurations used
- Include environment information

## See Also

- [CLI Commands](commands.md) - Command line interface
- [CLI Overview](overview.md) - Main CLI overview
- [Examples](../examples/overview.md) - Configuration examples