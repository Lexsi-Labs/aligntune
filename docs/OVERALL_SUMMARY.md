# AlignTune Summary


## Backend Support Matrix

| Algorithm | TRL Backend | Unsloth Backend |
|-----------|-------------|-----------------|
| **SFT** | Yes | Yes |
| **DPO** | Yes | Yes |
| **PPO** | Yes | Yes |
| **GRPO** | Yes | Yes |
| **GSPO** | Yes | No |
| **DAPO** | Yes | Yes |
| **Dr. GRPO** | Yes | Yes |

---

## Quick Start

### Supervised Fine-Tuning (SFT)

```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="gsm8k",
 backend="trl",
 output_dir="./output/sft_model",
 max_seq_length=512,
 use_unsloth=False,
 peft_enabled=False,
 batch_size=2,
 learning_rate=1e-5,
 
 # Data Configuration
 subset="main",
 task_type="instruction_following",
 column_mapping={"question": "instruction", "answer": "response"},
 
 # Training Configuration
 max_steps=5,
 num_epochs=1,
 
 # Logging
 run_name="sft_training_run"
)

# Start training
trainer.train()
```

### Reinforcement Learning Training

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="openai/gsm8k",
 config_name="main",
 algorithm="grpo",
 backend="trl",
 output_dir="./output/grpo_model",
 batch_size=8,
)

# Start training
trainer.train()
```

---

## Detailed Usage

### 1. Basic RL Training

The simplest way to start training with any RL algorithm:

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="openai/gsm8k",
 config_name="main",
 algorithm="grpo", # Options: dpo, ppo, grpo, gspo, dapo, etc.
 backend="trl", # Options: trl, unsloth
 output_dir="./output/my_model",
 batch_size=8,
)
```

### 2. Using Configuration Files

You can organize your training settings in YAML or JSON configuration files for better reproducibility and version control:

#### YAML Configuration Example

**config/grpo_training.yaml:**
```yaml
model_name: "Qwen/Qwen3-0.6B"
dataset_name: "openai/gsm8k"
config_name: "main"
algorithm: "grpo"
backend: "trl"
output_dir: "./output/grpo_model"
batch_size: 8
max_steps: 100
learning_rate: 2e-4
max_seq_length: 512

# Column mapping for dataset
column_mapping:
 question: "prompt"
 answer: "response"

# Reward configuration
rewards:
 - type: "accuracy"
 weight: 1.0
```

**Using the config file:**
```python
from aligntune.core.backend_factory import create_rl_trainer

# Method 1: Pass config file path
trainer = create_rl_trainer(config="config/grpo_training.yaml")

# Method 2: Override specific parameters
trainer = create_rl_trainer(
 config="config/grpo_training.yaml",
 batch_size=16, # Override the config value
 max_steps=200 # Override the config value
)

trainer.train()
```

#### Python Dictionary Configuration

```python
from aligntune.core.backend_factory import create_rl_trainer

config = {
 "model_name": "Qwen/Qwen3-0.6B",
 "dataset_name": "openai/gsm8k",
 "config_name": "main",
 "algorithm": "dpo",
 "backend": "trl",
 "output_dir": "./output/dpo_model",
 "batch_size": 8,
 "max_steps": 100,
 "learning_rate": 1e-5,
}

# Method 1: Pass as config parameter
trainer = create_rl_trainer(config=config)

# Method 2: Unpack as kwargs
trainer = create_rl_trainer(**config)

# Method 3: Mix config with overrides
trainer = create_rl_trainer(
 config=config,
 batch_size=16 # Override
)

trainer.train()
```

#### SFT Configuration Example

**config/sft_training.yaml:**
```yaml
model_name: "Qwen/Qwen3-0.6B"
dataset_name: "openai/gsm8k"
backend: "trl"
output_dir: "./output/sft_model"
batch_size: 8
num_epochs: 3
learning_rate: 2e-4
max_seq_length: 512

# Dataset configuration
subset: "main"
task_type: "instruction_following"
column_mapping:
 question: "instruction"
 answer: "response"

# Training settings
gradient_accumulation_steps: 4
warmup_ratio: 0.1
```

**Using the SFT config:**
```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(config="config/sft_training.yaml")
trainer.train()
```

### 3. Custom Data Processing

Define a custom preprocessing function to transform your data:

```python
def preprocess_function(example):
 """Transform raw data into the required format"""
 return {
 "prompt": example["question"],
 "chosen": example["answer"],
 "rejected": "None, 0"
 }

trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="openai/gsm8k",
 config_name="main",
 algorithm="dpo",
 backend="trl",
 output_dir="./output/dpo_model",
 batch_size=8,
 processing_fn=preprocess_function,
)
```

### 4. Column Mapping

Map dataset columns to expected format without writing a preprocessing function:

```python
trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="google-research-datasets/mbpp",
 algorithm="grpo",
 column_mapping={
 "text": "prompt",
 "code": "response"
 },
 backend="trl",
 output_dir="./output/grpo_model",
 batch_size=8,
 num_generations=4,
)
```

### 5. Custom Reward Functions

For algorithms like GRPO, GSPO, and Neural Mirror GRPO:

```python
def my_math_reward(text, reference=None, **kwargs):
 """
 Custom reward function for math problems.
 Returns 1.0 if the answer contains numbers, 0.0 otherwise.
 """
 has_numbers = any(char.isdigit() for char in text)
 return 1.0 if has_numbers else 0.0

trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="openai/gsm8k",
 config_name="main",
 algorithm="grpo",
 backend="unsloth",
 output_dir="./output/grpo_custom_reward",
 batch_size=4,
 rewards=[
 {
 "type": "custom",
 "weight": 1.0,
 "params": {
 "reward_function": my_math_reward
 }
 }
 ]
)
```

#### Multiple Reward Functions

You can combine multiple reward functions with different weights:

```python
def accuracy_reward(text, reference=None, **kwargs):
 """Check if output matches reference"""
 if reference and text.strip() == reference.strip():
 return 1.0
 return 0.0

def length_penalty(text, reference=None, **kwargs):
 """Penalize overly long responses"""
 max_length = 200
 return max(0.0, 1.0 - (len(text) - max_length) / max_length) if len(text) > max_length else 1.0

trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="openai/gsm8k",
 config_name="main",
 algorithm="grpo",
 backend="trl",
 output_dir="./output/grpo_multi_reward",
 batch_size=4,
 rewards=[
 {
 "type": "custom",
 "weight": 0.7,
 "params": {"reward_function": accuracy_reward}
 },
 {
 "type": "custom",
 "weight": 0.3,
 "params": {"reward_function": length_penalty}
 }
 ]
)
```

---

## Configuration

### Common Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model_name` | str | HuggingFace model identifier | Required |
| `dataset_name` | str | HuggingFace dataset identifier | Required |
| `algorithm` | str | RL algorithm to use | Required |
| `backend` | str | Training backend: `"trl"` or `"unsloth"` | `"trl"` |
| `output_dir` | str | Directory for saving outputs | Required |
| `batch_size` | int | Training batch size | 8 |
| `learning_rate` | float | Learning rate | 1e-5 |
| `max_seq_length` | int | Maximum sequence length | 512 |
| `num_epochs` | int | Number of training epochs | 1 |
| `max_steps` | int | Maximum training steps (overrides epochs) | None |
| `config` | str/dict | Path to config file or config dictionary | None |

### Configuration File Formats

AlignTune supports two configuration file formats:

- **YAML**: `.yaml` or `.yml` files
- **JSON**: `.json` files

Both formats support the same parameters and can be used interchangeably.

### Advanced Parameters

For a complete list of available parameters, see [PARAMETERS.md](PARAMETERS.md).

---

## Best Practices

### 1. Dataset Configuration

When working with datasets that have multiple subsets:

```python
trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="openai/gsm8k",
 config_name="main", # Important: specify the subset
 algorithm="dpo",
 backend="trl",
 output_dir="./output/model",
)
```

### 2. Dataset Loading Changes

**Note:** Newer versions of the `datasets` library no longer support `.py` scripts in the data loader. Make sure to:
- Use datasets with proper configuration files
- Update your dataset loading code accordingly
- Use `config_name` parameter for dataset subsets

### 3. Column Mapping Strategy

Always map your dataset columns to match the expected prompt format:

```python
# Example: Mapping code dataset columns
column_mapping = {
 "text": "prompt", # Input text/question
 "code": "response" # Expected output/code
}

# Example: Mapping Q&A dataset columns
column_mapping = {
 "question": "instruction",
 "answer": "response"
}
```

### 4. Configuration File Best Practices

- **Version Control**: Keep config files in version control for reproducibility
- **Environment-Specific Configs**: Create separate configs for dev/prod environments
- **Parameter Overrides**: Use command-line/code overrides for quick experiments
- **Documentation**: Add comments to YAML configs to document parameter choices

Example directory structure:
```
project/
 configs/
 base_grpo.yaml # Base configuration
 grpo_small_model.yaml # Small model variant
 grpo_production.yaml # Production settings
 train.py
 evaluate.py
```

---

## Evaluation

For unified evaluation and custom metrics, see the [Evaluation Guide](user-guide/evaluation.md).

---

## Troubleshooting

### Common Issues

1. **Import Errors**
 ```bash
 # Ensure all dependencies are installed
 pip install -r requirements.txt
 ```

2. **Dataset Loading Issues**
 - Check that `config_name` matches your dataset's subset name
 - Verify dataset exists on HuggingFace Hub
 - Ensure proper column mapping

3. **Memory Issues**
 - Reduce `batch_size`
 - Enable `peft_enabled=True` for parameter-efficient training
 - Use gradient checkpointing
 - Reduce `max_seq_length`

4. **Backend Compatibility**
 - Check the [Backend Support Matrix](#backend-support-matrix) for algorithm-backend compatibility
 - GSPO is not currently supported with Unsloth backend

5. **Configuration File Issues**
 - Ensure YAML syntax is correct (proper indentation)
 - Verify file path is correct when using `config=` parameter
 - Check that all required parameters are present in the config

### Getting Help

- Check [Examples](examples/overview.md) for working examples
- Review [PARAMETERS.md](PARAMETERS.md) for parameter documentation
- Open an issue on GitHub with:
 - Error message
 - Code snippet
 - Environment details (Python version, GPU info, etc.)

---

## Additional Resources

- **Evaluation Guide**: [#](#)
- **Parameters Reference**: [PARAMETERS.md](PARAMETERS.md)
- **Examples**: [examples/overview.md](examples/overview.md) - Working examples