# Neural Mirror GRPO Training with AlignTune

**Neural Mirror GRPO (NMGRPO)** - An advanced RL algorithm that extends GRPO with neural mirror descent for improved policy optimization through learnable mirror maps.

## Installation

```bash
git clone https://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@github.com/Lexsi-Labs/aligntune.git
cd Finetunehub
git checkout -b 28_merge origin/28_merge
pip install -e .
```

If using TIR e2e with nemo image, uninstall conflicting dependencies:
```bash
pip uninstall causal_conv1d
pip uninstall mamba_ssm
```

## Quick Start

```python
from aligntune import create_rl_trainer

trainer = create_rl_trainer(
    model_name="Qwen/Qwen3-1.7B",
    dataset_name="openai/gsm8k",
    algorithm="nmgrpo",
    backend="trl",
    output_dir="./output/nmgrpo_experiment",
    
    # Neural Mirror GRPO specific parameters
    mirror_coefficient=0.0001,
    mirror_init_scale=0.01,
    mirror_seed=42,
    divergence_type="neural_mirror",
    
    # Training parameters
    num_epochs=3,
    batch_size=16,
    learning_rate=1e-5,
    
    # Reward configuration
    reward_functions=["math_correctness"],
    reward_function_weights=[1.0]
)

results = trainer.train()
```

## Neural Mirror GRPO Overview

Neural Mirror GRPO introduces a **learnable mirror map** that adapts the optimization geometry during training. Unlike standard GRPO which uses fixed divergence measures, NMGRPO learns the optimal mirror structure for your specific task.

### Key Concepts

- **Mirror Map**: A neural network that transforms the policy parameter space
- **Adaptive Geometry**: The mirror map learns the best optimization landscape
- **Stable Updates**: Mirror descent ensures stable policy improvements
- **Task-Specific**: The mirror adapts to the characteristics of your task

## Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | Required | HuggingFace model name or path |
| `dataset_name` | str | Required | HuggingFace dataset name |
| `algorithm` | str | Required | Must be `"nmgrpo"` |
| `backend` | str | `"trl"` | Training backend (only TRL supported) |
| `output_dir` | str | `"./output"` | Output directory for checkpoints |
| `num_epochs` | int | `1` | Number of training epochs |
| `max_steps` | int | `1600` | Maximum training steps |
| `batch_size` | int | `16` | Per-device batch size |
| `learning_rate` | float | `1e-5` | Learning rate |
| `max_seq_length` | int | `512` | Maximum sequence length |

## Neural Mirror Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mirror_coefficient` | float | `0.0001` | Strength of mirror regularization |
| `mirror_init_scale` | float | `0.01` | Initialization scale for mirror parameters |
| `mirror_seed` | int | `42` | Random seed for mirror initialization |
| `divergence_type` | str | `"neural_mirror"` | Divergence measure type |
| `num_generations` | int | `32` | Number of completions per prompt |
| `max_completion_length` | int | `256` | Maximum length of generated completions |
| `max_prompt_length` | int | `768` | Maximum length of prompts |
| `temperature` | float | `0.8` | Sampling temperature |
| `top_p` | float | `0.9` | Nucleus sampling parameter |

### Mirror Coefficient Guidelines

- **Small tasks (< 1K samples)**: `0.0001` - `0.001`
- **Medium tasks (1K-10K samples)**: `0.001` - `0.01`
- **Large tasks (> 10K samples)**: `0.01` - `0.1`

Higher coefficients provide stronger regularization but may slow convergence.

## Reward Functions

```python
# Single reward
reward_functions=["math_correctness"]
reward_function_weights=[1.0]

# Multiple rewards
reward_functions=["math_correctness", "math_reasoning", "safety"]
reward_function_weights=[0.6, 0.3, 0.1]
```

### Available Rewards

**Math/Logic**: `math_correctness`, `math_reasoning`, `logical_consistency`, `causal_reasoning`, `counterfactual_reasoning`

**Code**: `code_syntax`, `code_quality`, `code_execution`, `code_completeness`, `code_correctness`

**Quality**: `coherence`, `fluency`, `readability`, `engagement`, `conciseness`, `brevity`

**Safety**: `safety`, `toxicity`, `bias`, `harmlessness`, `hallucination`, `honesty`

**Task-Specific**: `instruction_following`, `helpfulness`, `politeness`, `relevance`, `context_relevance`

**Metrics**: `bleu`, `rouge`, `meteor`, `bertscore`, `semantic_similarity`

**Domain**: `medical_accuracy`, `legal_compliance`, `financial_accuracy`

## Advanced Configuration

```python
trainer = create_rl_trainer(
    # Model
    model_name="Qwen/Qwen3-1.7B",
    max_seq_length=512,
    
    # Dataset
    dataset_name="openai/gsm8k",
    split="train[:1%]",
    max_samples=1000,
    system_prompt="You are a helpful math tutor.",
    
    # Training
    algorithm="nmgrpo",
    backend="trl",
    num_epochs=3,
    max_steps=1600,
    batch_size=16,
    num_generations=32,
    learning_rate=1e-5,
    gradient_accumulation_steps=8,
    max_grad_norm=1.0,
    
    # Neural Mirror Configuration
    mirror_coefficient=0.0001,
    mirror_init_scale=0.01,
    mirror_seed=42,
    divergence_type="neural_mirror",
    
    # Generation Settings
    max_prompt_length=768,
    max_completion_length=256,
    temperature=0.8,
    top_p=0.9,
    
    # LoRA Configuration
    use_lora=True,
    lora_r=64,
    lora_alpha=64,
    lora_dropout=0.05,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    
    # Precision
    precision="auto",  # or "bf16", "fp16", "fp32"
    
    # Optimizer & Scheduler
    optimizer="adamw_torch",
    lr_scheduler="cosine",
    weight_decay=0.0,
    warmup_steps=10,
    
    # Rewards
    reward_functions=["code_quality", "code_correctness", "length", "diversity"],
    reward_function_weights=[0.3, 0.4, 0.15, 0.15],
    
    # Logging
    output_dir="./output/nmgrpo_advanced",
    wandb_project="nmgrpo-experiments",
    wandb_run_name="qwen3-neural-mirror-gsm8k",
    loggers=["tensorboard", "wandb"]
)

results = trainer.train()
```

## Alternative Reward Configuration

```python
trainer = create_rl_trainer(
    model_name="Qwen/Qwen3-1.7B",
    dataset_name="openai/gsm8k",
    algorithm="nmgrpo",
    backend="trl",
    mirror_coefficient=0.0001,
    
    # Use rewards dict instead of reward_functions
    rewards=[
        {
            "type": "code_quality",
            "weight": 0.3,
            "params": {}
        },
        {
            "type": "code_correctness",
            "weight": 0.4,
            "params": {}
        },
        {
            "type": "length",
            "weight": 0.15,
            "params": {
                "min_length": 20,
                "max_length": 256
            }
        },
        {
            "type": "diversity",
            "weight": 0.15,
            "params": {}
        }
    ]
)
```

## CLI Usage

### Using Command Line Arguments
```bash
aligntune train \
  --model "Qwen/Qwen3-1.7B" \
  --dataset "openai/gsm8k" \
  --type nmgrpo \
  --backend trl \
  --split "train[:1%]" \
  --output "output/nmgrpo_experiment" \
  --epochs 3 \
  --max-steps 1600 \
  --batch-size 16 \
  --num-generations 32 \
  --lr 1e-5 \
  --max-length 512 \
  --reward-functions "math_correctness,math_reasoning" \
  --mirror-coefficient 0.0001 \
  --mirror-init-scale 0.01 \
  --mirror-seed 42
```

### Using YAML Configuration

**Create config file: `configs/nmgrpo_gsm8k.yaml`**
```yaml
algo: nmgrpo

model:
  name_or_path: "Qwen/Qwen3-1.7B"
  precision: auto
  gradient_checkpointing: true
  max_seq_length: 512
  
  # LoRA configuration
  use_peft: true
  lora_r: 64
  lora_alpha: 64
  lora_dropout: 0.05
  lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

datasets:
  - name: "openai/gsm8k"
    split: "train[:1%]"
    column_mapping:
      prompt: "question"
      text: "answer"
    system_prompt: "You are a helpful math tutor."

train:
  epochs: 3
  max_steps: 1600
  per_device_batch_size: 16
  learning_rate: 0.00001
  num_generations: 32
  gradient_accumulation_steps: 8
  
  # Neural Mirror GRPO parameters
  mirror_coefficient: 0.0001
  mirror_init_scale: 0.01
  mirror_seed: 42
  divergence_type: "neural_mirror"
  
  # Optimizer & Scheduler
  optimizer: "adamw_torch"
  lr_scheduler: "cosine"
  weight_decay: 0.0
  warmup_steps: 10
  max_grad_norm: 1.0
  
  # Generation settings
  max_prompt_length: 768
  max_completion_length: 256
  temperature: 0.8
  top_p: 0.9
  
  # Reward configuration
  rewards:
    - type: "code_quality"
      weight: 0.3
      params: {}
    
    - type: "code_correctness"
      weight: 0.4
      params: {}
    
    - type: "length"
      weight: 0.15
      params:
        min_length: 20
        max_length: 256
    
    - type: "diversity"
      weight: 0.15
      params: {}

logging:
  output_dir: "output/nmgrpo_advanced"
  run_name: "nmgrpo_neural_mirror_gsm8k"
  loggers: ["tensorboard", "wandb"]
  logging_steps: 10
  save_steps: 100
  eval_steps: 100

distributed:
  backend: single

caching:
  root: "cache"
  enabled: true

chat_template: auto
```

**Run with config:**
```bash
aligntune train --config configs/nmgrpo_gsm8k.yaml
```

## Minimal Example

```python
from aligntune import create_rl_trainer

# Simplest possible Neural Mirror GRPO training
trainer = create_rl_trainer(
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="openai/gsm8k",
    algorithm="nmgrpo",
    reward_functions=["math_correctness"]
)

trainer.train()
```

## Production Example

```python
from aligntune import create_rl_trainer

# Production-ready Neural Mirror GRPO configuration
trainer = create_rl_trainer(
    # Model
    model_name="Qwen/Qwen3-4B",
    max_seq_length=1024,
    
    # Dataset
    dataset_name="mbpp",  # Code generation task
    split="train",
    
    # Neural Mirror GRPO Algorithm
    algorithm="nmgrpo",
    backend="trl",
    mirror_coefficient=0.001,  # Higher for larger dataset
    mirror_init_scale=0.01,
    mirror_seed=42,
    divergence_type="neural_mirror",
    
    # Training
    num_epochs=5,
    max_steps=5000,
    batch_size=32,
    num_generations=64,
    learning_rate=1e-5,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    
    # LoRA
    use_lora=True,
    lora_r=128,
    lora_alpha=256,
    lora_dropout=0.05,
    
    # Precision
    precision="bf16",
    
    # Optimizer
    optimizer="adamw_torch",
    lr_scheduler="cosine",
    weight_decay=0.0,
    warmup_steps=50,
    
    # Generation
    max_prompt_length=512,
    max_completion_length=512,
    temperature=0.7,
    top_p=0.9,
    
    # Rewards
    reward_functions=[
        "code_quality",
        "code_correctness",
        "code_execution",
        "safety"
    ],
    reward_function_weights=[0.3, 0.4, 0.2, 0.1],
    
    # Logging
    output_dir="./output/production_nmgrpo",
    wandb_project="nmgrpo-production",
    wandb_run_name="qwen3-4b-mbpp-neural-mirror",
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
)

results = trainer.train()
print(f"Training completed: {results}")
```


## Implementation Details

### Trainer Location
```
Finetunehub/src/aligntune/backends/trl/rl/neural_mirror_grpo/
├── NMGrpo.py              # TRL interface wrapper
└── neural_mirror_grpo.py  # Core Neural Mirror GRPO trainer (external dependency)
```

### Configuration Location
```
Finetunehub/src/aligntune/core/rl/config.py  # RL configuration classes
Finetunehub/src/aligntune/core/backend_factory.py  # Backend registration
```

### Reward Functions Location
```
Finetunehub/src/aligntune/rewards/
├── core.py       # All reward function implementations
└── registry.py   # Reward function registration
```

## Key Features

✅ **Learnable mirror maps** for adaptive optimization geometry  
✅ **Stable policy updates** through mirror descent  
✅ **Task-specific adaptation** - mirror learns from your data  
✅ **LoRA support** for efficient fine-tuning  
✅ **Multiple reward functions** with automatic weighting  
✅ **Flexible precision** (fp32, fp16, bf16, auto)  
✅ **WandB/TensorBoard logging** built-in  
✅ **YAML and Python API** for configuration  
✅ **Unified DataManager** for dataset preprocessing  

## Dataset Handling

Neural Mirror GRPO uses the unified DataManager for robust dataset processing:

```python
# Automatic format detection
trainer = create_rl_trainer(
    dataset_name="openai/gsm8k",  # Auto-detects format
    column_mapping={
        "prompt": "question",  # Maps dataset columns
        "text": "answer"
    }
)

# Custom preprocessing
def custom_preprocessing(example):
    example["prompt"] = f"Question: {example['question']}"
    return example

trainer = create_rl_trainer(
    dataset_name="my_dataset",
    processing_fn=custom_preprocessing,
    processing_batched=False
)
```

## Troubleshooting

### Model name not a string error
**Problem**: `model_name must be a string` error during setup

**Solution**: Ensure your config properly specifies the model name:
```python
# Correct
model_name="Qwen/Qwen3-1.7B"

# Or in YAML
model:
  name_or_path: "Qwen/Qwen3-1.7B"
```

### Reward function not found
**Problem**: `Unknown reward function` error

**Solution**: Use correct reward names from the available rewards list:
```python
# Correct
reward_functions=["code_quality"]

# Incorrect
reward_functions=["code_score"]  # This doesn't exist
```

### Training instability
**Solutions**:
- Increase `mirror_coefficient` (e.g., from 0.0001 to 0.001)
- Reduce `learning_rate` (e.g., from 1e-5 to 5e-6)
- Increase `gradient_accumulation_steps`
- Reduce `temperature` for more conservative generation

### OOM (Out of Memory)
**Solutions**:
- Reduce `batch_size`
- Increase `gradient_accumulation_steps`
- Reduce `num_generations`
- Reduce `max_completion_length`
- Use smaller model
- Enable LoRA: `use_lora=True`
- Use quantization: `load_in_8bit=True`

### Slow convergence
**Solutions**:
- Reduce `mirror_coefficient` (but monitor stability)
- Increase `learning_rate` (but monitor stability)
- Reduce `gradient_accumulation_steps`
- Increase `warmup_steps` for smoother start


## Advanced: Custom Reward Functions

```python
# Define custom reward
def my_custom_reward(text, reference=None, **kwargs):
    """
    Custom reward function.
    
    Args:
        text: Generated completion
        reference: Optional reference text
        **kwargs: Additional context
    
    Returns:
        float: Reward score (higher is better)
    """
    score = 0.0
    
    # Example: reward based on length
    score += min(len(text.split()) / 50, 1.0)
    
    # Example: check for specific patterns
    if "step-by-step" in text.lower():
        score += 0.5
    
    return score

# Use in training
trainer = create_rl_trainer(
    model_name="Qwen/Qwen3-1.7B",
    dataset_name="openai/gsm8k",
    algorithm="nmgrpo",
    rewards=[
        {
            "type": "custom",
            "weight": 0.5,
            "params": {
                "reward_function": my_custom_reward
            }
        },
        {
            "type": "math_correctness",
            "weight": 0.5,
            "params": {}
        }
    ]
)
```

## Performance Tips

1. **Start Small**: Begin with a small dataset subset to validate your setup
2. **Monitor Metrics**: Watch reward trends and loss curves carefully
3. **Tune Mirror Coefficient**: This is the most important hyperparameter
4. **Balance Rewards**: Ensure reward weights sum to 1.0 for interpretability
5. **Use LoRA**: Almost always beneficial for efficiency
6. **Gradient Accumulation**: Use to simulate larger batch sizes
7. **Warmup Steps**: Critical for stable optimization
8. **Save Frequently**: Set reasonable `save_steps` to avoid losing progress





