# Reinforcement Learning (RL) Training Guide

Complete guide to Reinforcement Learning from Human Feedback (RLHF) training with AlignTune, covering all supported RLHF algorithms.

## Overview

Reinforcement Learning training aligns language models with human preferences using various algorithms. AlignTune supports multiple RLHF algorithms:

1. **DPO** (Direct Preference Optimization) - No reward model needed
2. **PPO** (Proximal Policy Optimization) - Flexible policy optimization with reward models
3. **GRPO** (Group Relative Policy Optimization) - Multi-criteria optimization
4. **GSPO** (Group Sequential Policy Optimization) - Sequential group learning
5. **DAPO** (Decouple Clip and Dynamic sAmpling Policy Optimization) - Scaled RL for LLMs while addressing key limitations in GRPO
6. **Dr. GRPO** (GRPO Done Right) - Unbiased GRPO variant that corrects optimization biases

## Quick Start

### Basic DPO Training

```python
from aligntune.core.backend_factory import create_rl_trainer

# Create and train DPO model
trainer = create_rl_trainer(
 model_name="meta-llama/Llama-3.2-3B",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dpo",
 backend="trl", # Use TRL for GPT2 models
 num_epochs=1,
 batch_size=4,
 learning_rate=5e-5,
 max_seq_length=512,
 max_samples=1000,
 # For GPT2 models, add: lora_target_modules=["c_attn", "c_proj"]
 # Or use a different model like Qwen/Qwen3-0.6B or Llama models
)

# Train the model
trainer.train()
```

### Basic PPO Training

```python
trainer = create_rl_trainer(
    model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    dataset_name="Anthropic/hh-rlhf",
    algorithm="ppo",
    backend="unsloth",
    reward_model_name="Skywork/Skywork-Reward-V2-Qwen3-0.6B",
    num_epochs=1,
    batch_size=1,
    learning_rate=1e-6,
)

trainer.train()
```

## Algorithms

### 1. DPO (Direct Preference Optimization)

DPO trains models directly on preference pairs without requiring a separate reward model.

**Advantages:**
- No reward model needed
- Simpler training pipeline
- Direct preference learning

**Use Cases:**
- Preference alignment
- Human feedback integration
- General RLHF training

#### Example

```python
trainer = create_rl_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dpo",
 backend="trl", # Use TRL for GPT2 models
 num_epochs=1,
 batch_size=4,
 learning_rate=5e-5,
 max_seq_length=512,
 # For GPT2 models, add: lora_target_modules=["c_attn", "c_proj"]
 # Or use a different model like Qwen/Qwen3-0.6B
 # DPO-specific parameters
 beta=0.1, # KL penalty coefficient
 reference_free=False, # Use reference model
 loss_type="sigmoid" # Loss function type
)

trainer.train()
```

**Dataset Format:**
```json
{
 "prompt": "What is machine learning?",
 "chosen": "Machine learning is a subset of AI...",
 "rejected": "I don't know."
}
```

### 2. PPO (Proximal Policy Optimization)

PPO optimizes policies using reward models and value functions.

**Advantages:**
- Flexible reward shaping
- Can use custom reward models
- Supports complex reward landscapes

**Use Cases:**
- Custom reward model training
- Complex reward functions
- Production RLHF systems

#### Example

```python
trainer = create_rl_trainer(
    model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    dataset_name="Anthropic/hh-rlhf",
    algorithm="ppo",
    backend="unsloth",
    # Reward model configuration
    reward_model_name="Skywork/Skywork-Reward-V2-Qwen3-0.6B",
    # PPO-specific parameters
    num_epochs=1,
    batch_size=1,
    learning_rate=1e-6,
    kl_coef=0.1, # KL penalty coefficient
    cliprange=0.2, # PPO clip range
    vf_coef=0.1, # Value function coefficient
    gamma=1.0, # Discount factor
    lam=0.95 # GAE lambda
)

trainer.train()
```

#### Custom Reward Model Training

```python
def load_training_texts():
    """Load training texts for reward model training."""
    return [
        "This is a helpful and informative response that addresses the question clearly.",
        "I'm not sure about this answer.",
        "This response is clear, concise, and well-structured.",
    ]

trainer = create_rl_trainer(
    model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    dataset_name="HuggingFaceH4/ultrafeedback_binarized",
    algorithm="ppo",
    backend="unsloth",
    # Custom reward model training
    train_custom_reward_model=True,
    reward_training_texts=load_training_texts(),
    reward_functions=["length", "sentiment", "safety", "coherence"],
    reward_function_weights=[0.2, 0.3, 0.3, 0.2],
    reward_training_base_model="microsoft/DialoGPT-medium",
    reward_training_output_dir="./reward_models/custom",
    # PPO configuration
    num_epochs=1,
    batch_size=1,
    learning_rate=2e-4
)

trainer.train()
```

### 3. GRPO (Group Relative Policy Optimization)

GRPO optimizes policies using group-based relative comparisons.

**Advantages:**
- Multi-criteria optimization
- Group-based learning
- Flexible reward combinations

**Use Cases:**
- Multi-objective optimization
- Complex reward landscapes
- Group-based preference learning

#### Example

```python
trainer = create_rl_trainer(
    model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    dataset_name="Anthropic/hh-rlhf",
    algorithm="grpo",
    backend="unsloth",
    num_epochs=1,
    batch_size=2, # GRPO requires minimum batch_size=2 for generations
    learning_rate=1e-6,
    loss_type='grpo'
)

trainer.train()

```

### 4. GSPO (Group Sequential Policy Optimization)

GSPO uses sequential group learning for policy optimization.

**Note:** GSPO is only supported by TRL backend, not Unsloth.

**Advantages:**
- Sequential learning
- Group-based optimization
- Structured policy updates

**Use Cases:**
- Sequential preference learning
- Structured policy optimization
- Group-based RLHF

#### Example

```python
trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="gspo",
 backend="trl", # GSPO only works with TRL
 num_epochs=1,
 batch_size=4,
 learning_rate=5e-5,
 # Note: group_size and sequential_steps are not in standard GRPO config
 # These parameters are specific to GSPO implementation
)

trainer.train()
```

## Backend Selection

### TRL Backend

**Use TRL when:**
- Need maximum compatibility
- Using GSPO (TRL only)
- Working with standard models
- Need reliable, battle-tested training

```python
trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dpo",
 backend="trl"
)
```

### Unsloth Backend

**Use Unsloth when:**
- Need faster training
- Working with large models
- Need memory efficiency
- Training PPO, DPO, or GRPO

```python
trainer = create_rl_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="ppo",
 backend="unsloth"
)
```

## Configuration

### Model Configuration

```python
trainer = create_rl_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="ppo",
 # Model settings
 max_seq_length=2048,
 quantization={"load_in_4bit": True},
 use_peft=True, # Enable LoRA
 lora_r=16,
 lora_alpha=32,
 lora_dropout=0.05,
 use_gradient_checkpointing=True,
 # Reward model settings (for PPO)
 reward_model_name="Skywork/Skywork-Reward-V2-Qwen3-0.6B",
 reward_model_quantization={"load_in_4bit": True}
)
```

### Training Configuration

```python
trainer = create_rl_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dpo",
 # Training settings
 num_epochs=1,
 max_steps=None, # Use epochs
 batch_size=4,
 gradient_accumulation_steps=4,
 learning_rate=5e-5,
 weight_decay=0.01,
 warmup_steps=100,
 max_grad_norm=1.0,
 # Algorithm-specific
 beta=0.1, # DPO: KL coefficient
 kl_coef=0.1, # PPO: KL coefficient
 cliprange=0.2, # PPO: clip range
 temperature=0.7 # Generation temperature
)
```

### Dataset Configuration

```python
trainer = create_rl_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dpo",
 # Dataset settings
 max_samples=1000,
 percent=10.0,
 split="train",
 # Field mappings
 column_mapping={
 "prompt": "prompt",
 "chosen": "chosen",
 "rejected": "rejected"
 },
 truncation_mode="keep_end",
 padding_free=False
)
```

## Reward Models

### Using Pre-trained Reward Models

```python
# From HuggingFace Hub
trainer = create_rl_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="ppo",
 reward_model_name="Skywork/Skywork-Reward-V2-Qwen3-0.6B"
)

# From local path
trainer = create_rl_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="ppo",
 reward_model_path="./reward_models/my_reward_model"
)
```

### Training Custom Reward Models

See [Reward Model Training Guide](reward-model-training.md) for detailed instructions.

```python
trainer = create_rl_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="HuggingFaceH4/ultrafeedback_binarized",
 algorithm="ppo",
 backend="unsloth",
 # Custom reward model training
 train_custom_reward_model=True,
 reward_training_texts=training_texts,
 reward_functions=["length", "sentiment", "safety"],
 reward_function_weights=[0.3, 0.4, 0.3],
 reward_training_base_model="microsoft/DialoGPT-medium",
 reward_training_output_dir="./reward_models/custom"
)
```

## Advanced Features

### Model Family Consistency (PPO)

AlignTune automatically checks that all models in PPO training belong to the same family:

```python
# Correct: All Qwen models
trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 algorithm="ppo",
 reward_model_name="Skywork/Skywork-Reward-V2-Qwen3-0.6B"
)

# Error: Mixed families
trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 algorithm="ppo",
 reward_model_name="meta-llama/Llama-2-7b-hf" # Different family!
)
```

### LoRA/QLoRA Fine-Tuning

```python
trainer = create_rl_trainer(
 model_name="meta-llama/Llama-2-7b-hf",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="ppo",
 # LoRA configuration
 use_peft=True,
 lora_r=16,
 lora_alpha=32,
 lora_dropout=0.05,
 lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
 # Quantization
 quantization={"load_in_4bit": True}
)
```

### Distributed Training

```python
# Distributed training
trainer = create_rl_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="ppo",
 # Distributed settings (configure via accelerate)
 # See distributed training guide
)
```

## Evaluation

### Basic Evaluation

```python
# Evaluate on validation set
metrics = trainer.evaluate()
print(metrics)
```

### Custom Evaluation

```python
from datasets import load_dataset

eval_dataset = load_dataset("Anthropic/hh-rlhf", split="test")
metrics = trainer.evaluate(eval_dataset=eval_dataset)
```

### Zero-Shot Evaluation

```python
# Generate predictions
prompts = [
 "What is machine learning?",
 "Explain deep learning"
]

results = trainer.predict(prompts)
for prompt, result in zip(prompts, results):
 print(f"Q: {prompt}")
 print(f"A: {result}\n")
```

## Best Practices

### 1. Algorithm Selection

- **DPO**: Start here for preference alignment, simpler setup
- **PPO**: Use for custom rewards, complex scenarios
- **GRPO**: Multi-criteria optimization
- **GSPO**: Sequential learning (TRL only)

### 2. Backend Selection

- **TRL**: Maximum compatibility, GSPO support
- **Unsloth**: Faster training (Faster), memory efficient

### 3. Reward Models

- Use pre-trained models when available
- Train custom models for domain-specific tasks
- Ensure model family consistency for PPO

### 4. Hyperparameters

```python
# DPO recommended settings
beta=0.1 # KL coefficient
learning_rate=5e-5,
batch_size=4

# PPO recommended settings
kl_coef=0.1
cliprange=0.2
learning_rate=1e-6,
batch_size=1 # Smaller for PPO
```

### 5. Memory Optimization

```python
# For large models
trainer = create_rl_trainer(
 model_name="meta-llama/Llama-2-7b-hf",
 algorithm="ppo",
 quantization={"load_in_4bit": True},
 use_peft=True,
 use_gradient_checkpointing=True,
 batch_size=1,
 gradient_accumulation_steps=8
)
```

## Troubleshooting

### Out of Memory

```python
# Reduce batch size, use quantization
trainer = create_rl_trainer(
 model_name="meta-llama/Llama-2-7b-hf",
 algorithm="ppo",
 batch_size=1, # Reduce
 gradient_accumulation_steps=8, # Compensate
 quantization={"load_in_4bit": True},
 use_gradient_checkpointing=True
)
```

### Model Family Mismatch (PPO)

```python
# Ensure all models are same family
# Correct
model_name="Qwen/Qwen3-0.6B"
reward_model_name="Skywork/Skywork-Reward-V2-Qwen3-0.6B"

# Wrong
model_name="Qwen/Qwen3-0.6B"
reward_model_name="meta-llama/Llama-2-7b-hf" # Different family!
```

### Slow Training

```python
# Use Unsloth backend
trainer = create_rl_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 algorithm="ppo",
 backend="unsloth" # faster
)
```

### Poor Convergence

```python
# Adjust learning rate and KL coefficient
trainer = create_rl_trainer(
 model_name="microsoft/DialoGPT-medium",
 algorithm="dpo",
 learning_rate=1e-5, # Lower learning rate
 beta=0.05, # Lower KL penalty
 num_epochs=2 # More epochs
)
```

## Complete Examples

### DPO Training

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dpo",
 backend="trl",
 num_epochs=1,
 batch_size=4,
 learning_rate=5e-5,
 max_seq_length=512,
 max_samples=1000,
 beta=0.1
)

trainer.train()
metrics = trainer.evaluate()
model_path = trainer.save_model()
```

### PPO with Custom Reward Model

```python
trainer = create_rl_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="HuggingFaceH4/ultrafeedback_binarized",
 algorithm="ppo",
 backend="unsloth",
 train_custom_reward_model=True,
 reward_training_texts=load_training_texts(),
 reward_functions=["length", "sentiment", "safety"],
 reward_function_weights=[0.3, 0.4, 0.3],
 reward_training_base_model="microsoft/DialoGPT-medium",
 reward_training_output_dir="./reward_models/custom",
 num_epochs=1,
 batch_size=1,
 learning_rate=2e-4
)

trainer.train()
```

## Next Steps

- [Reward Functions Guide](reward-functions.md) - Explore reward functions
- [Reward Model Training](reward-model-training.md) - Train custom reward models
- [Evaluation Guide](evaluation.md) - Comprehensive evaluation
- [SFT Guide](sft.md) - Supervised fine-tuning

## Additional Resources

- [API Reference](../api-reference/core.md) - Complete API documentation
- [Examples](../examples/rl.md) - More RL examples
- [Backend Selection](../getting-started/backend-selection.md) - Backend guide
- [Unsloth Compatibility](../unsloth_compatibility.md) - Unsloth setup