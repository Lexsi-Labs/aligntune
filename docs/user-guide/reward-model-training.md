# Reward Model Training Guide

Complete guide to training custom reward models from rule-based reward functions and integrating them into PPO training.

## Overview

Reward models learn to score text quality based on training data generated from rule-based reward functions. AlignTune provides a complete system for training reward models and seamlessly integrating them into PPO training pipelines.

## Quick Start

### Standalone Reward Model Training

```python
from aligntune.rewards.training import RewardModelTrainer
from aligntune.rewards.registry import RewardRegistry

# Get reward functions
registry = RewardRegistry()
length_func = registry.get_reward_function("length")
sentiment_func = registry.get_reward_function("sentiment")
safety_func = registry.get_reward_function("safety")

# Create trainer
trainer = RewardModelTrainer(
    base_model_name="microsoft/DialoGPT-medium",
    reward_functions=[length_func, sentiment_func, safety_func],
    composite_weights=[0.3, 0.4, 0.3]
)

# Generate training data
training_texts = [
 "This is a helpful and informative response.",
 "I'm not sure, but here's what I think.",
 "That's a great question! Let me explain step by step."
]

training_data = trainer.generate_training_data(
    texts=training_texts,
    batch_size=32
)

# Train reward model
model_path = trainer.train_reward_model(
    training_data=training_data,
    output_dir="./reward_models/custom",
    num_epochs=3,
    learning_rate=1e-5,
    batch_size=8
)

print(f"Reward model saved to: {model_path}")
```

### PPO with Custom Reward Model Training

```python
from aligntune.core.backend_factory import create_rl_trainer

def load_training_texts():
 """Load your training texts."""
 return [
 "This is a helpful response.",
 "I'm not sure about this.",
 "That's a great question!",
 # ... more texts
 ]

# Create PPO trainer with custom reward model training
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
 reward_training_output_dir="./reward_models/custom_ppo",
 # PPO configuration
 num_epochs=1,
 batch_size=1,
 learning_rate=2e-4
)

# Train (reward model is trained first, then PPO)
trainer.train()
```

## Training Modes

### 1. Standalone Training

Train reward models independently for later use:

```python
from aligntune.rewards.training import RewardModelTrainer

trainer = RewardModelTrainer(
 base_model_name="microsoft/DialoGPT-medium",
 reward_functions=["length", "sentiment", "safety"],
 composite_weights=[0.3, 0.4, 0.3]
)

# Generate training data
training_data = trainer.generate_training_data(
 texts=your_training_texts,
 batch_size=32
)

# Train
model_path = trainer.train_reward_model(
 training_data=training_data,
 output_dir="./reward_models/my_model",
 num_epochs=3,
 learning_rate=1e-5,
 batch_size=8
)
```

### 2. Integrated PPO Training

Train reward model as part of PPO training:

```python
trainer = create_rl_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 algorithm="ppo",
 backend="unsloth",
 train_custom_reward_model=True,
 reward_training_texts=training_texts,
 reward_functions=["length", "sentiment", "safety"],
 reward_function_weights=[0.3, 0.4, 0.3],
 reward_training_base_model="microsoft/DialoGPT-medium",
 reward_training_output_dir="./reward_models/custom"
)
```

## Configuration

### Reward Model Training Config

```python
from aligntune.core.rl.config import RewardModelTrainingConfig

config = RewardModelTrainingConfig(
 base_model_name="microsoft/DialoGPT-medium", # Required
 training_texts=your_texts, # Required
 reward_functions=["length", "sentiment"], # Required
 output_dir="./reward_models/custom", # Required
 # Optional parameters
 reward_weights=[0.5, 0.5],
 num_epochs=3,
 learning_rate=1e-5,
 batch_size=8,
 gradient_accumulation_steps=4,
 max_length=512
)
```

### Reward Model Source Config

```python
from aligntune.core.rl.config import RewardModelSourceConfig, RewardModelTrainingConfig

# For custom training
training_config = RewardModelTrainingConfig(
 base_model_name="microsoft/DialoGPT-medium",
 training_texts=texts,
 reward_functions=["length", "safety"],
 output_dir="./reward_models/custom"
)

source_config = RewardModelSourceConfig(
 source_type="custom_trained",
 training_config=training_config
)

# For HuggingFace Hub
source_config = RewardModelSourceConfig(
 source_type="pretrained_hf",
 model_name="Skywork/Skywork-Reward-V2-Qwen3-0.6B"
)

# For local model
source_config = RewardModelSourceConfig(
 source_type="pretrained_local",
 model_path="./reward_models/my_model"
)
```

## Using Pre-trained Reward Models

### From HuggingFace Hub

```python
trainer = create_rl_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 algorithm="ppo",
 backend="unsloth",
 reward_model_name="Skywork/Skywork-Reward-V2-Qwen3-0.6B"
)
```

### From Local Path

```python
trainer = create_rl_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 algorithm="ppo",
 backend="unsloth",
 reward_model_path="./reward_models/my_custom_model"
)
```

## Training Data Generation

### Basic Generation

```python
from aligntune.rewards.training import RewardModelTrainer

trainer = RewardModelTrainer(
 base_model_name="microsoft/DialoGPT-medium",
 reward_functions=["length", "sentiment", "safety"],
 composite_weights=[0.3, 0.4, 0.3]
)

# Generate training data
training_data = trainer.generate_training_data(
 texts=your_texts,
 batch_size=32
)
```

### With Reference Texts

```python
training_data = trainer.generate_training_data(
 texts=your_texts,
 reference_texts=reference_texts, # Optional reference texts
 batch_size=32
)
```

## Best Practices

### 1. Training Data Quality

- Use diverse, representative texts
- Include both good and bad examples
- Ensure sufficient quantity (100+ texts recommended)

### 2. Reward Function Selection

- Choose functions relevant to your task
- Balance weights appropriately
- Start with a few key functions, expand as needed

### 3. Model Selection

- Use base models compatible with your policy model
- Consider model family consistency (for PPO)
- Smaller models work well for reward models

### 4. Training Parameters

```python
# Recommended settings
num_epochs=3 # 3-5 epochs usually sufficient
learning_rate=1e-5 # Lower learning rate for reward models
batch_size=8 # Adjust based on model size
gradient_accumulation_steps=4 # Effective batch size = 32
```

### 5. Validation

- Validate reward model on held-out data
- Test reward scores on sample texts
- Ensure rewards align with expectations

## Complete Example

```python
from aligntune.core.backend_factory import create_rl_trainer
from aligntune.rewards.registry import RewardRegistry

def load_training_texts():
 """Load training texts for reward model."""
 return [
 "This is a helpful and informative response that addresses the question.",
 "I'm not entirely sure, but I think the answer might be related to this.",
 "That's a great question! Let me break it down step by step for you.",
 "I don't have enough information to provide a complete answer.",
 "Here's a comprehensive explanation of the topic you asked about.",
 # ... more diverse texts
 ]

def main():
 # Load training data
 training_texts = load_training_texts()
 
 # Create PPO trainer with custom reward model
 trainer = create_rl_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="HuggingFaceH4/ultrafeedback_binarized",
 algorithm="ppo",
 backend="unsloth",
 # Custom reward model training
 train_custom_reward_model=True,
 reward_training_texts=training_texts,
 reward_functions=[
 "length", # Appropriate length
 "sentiment", # Positive sentiment
 "safety", # Safety (high priority)
 "coherence", # Logical flow
 "helpfulness" # Helpful responses
 ],
 reward_function_weights=[
 0.15, # length
 0.15, # sentiment
 0.35, # safety (highest)
 0.20, # coherence
 0.15 # helpfulness
 ],
 reward_training_base_model="microsoft/DialoGPT-medium",
 reward_training_output_dir="./reward_models/custom_ppo",
 # Training parameters
 reward_training_epochs=3,
 reward_training_lr=1e-5,
 reward_training_batch_size=8,
 # PPO configuration
 num_epochs=1,
 batch_size=1,
 learning_rate=2e-4,
 max_samples=100
 )
 
 # Train (reward model trained first, then PPO)
 print("Starting training...")
 trainer.train()
 
 # Save model
 model_path = trainer.save_model()
 print(f"Model saved to: {model_path}")

if __name__ == "__main__":
 main()
```

## Validation

### Strict Validation Rules

AlignTune enforces strict validation:

1. **Exactly One Source**: Must specify exactly one reward source
2. **No Empty Fields**: All required fields must be non-empty
3. **Minimum Data**: At least 10 training texts required
4. **Valid Functions**: All reward functions must exist in registry
5. **Positive Weights**: All reward weights must be positive

### Error Handling

- **No Fallbacks**: System fails fast with clear errors
- **Clear Messages**: Errors include actionable information
- **Comprehensive Validation**: All inputs validated before processing

## Troubleshooting

### Insufficient Training Data

```python
# Ensure at least 10 texts
if len(training_texts) < 10:
 raise ValueError("Need at least 10 training texts")
```

### Invalid Reward Functions

```python
from aligntune.rewards.registry import RewardRegistry

registry = RewardRegistry()
available = registry.list_reward_functions()

# Check if function exists
if "my_function" not in available:
 raise ValueError(f"Reward function not found. Available: {available}")
```

### Model Family Mismatch (PPO)

```python
# Ensure reward model matches policy model family
# Correct: Both Qwen
model_name="Qwen/Qwen3-0.6B"
reward_training_base_model="Qwen/Qwen3-0.6B"

# Wrong: Different families
model_name="Qwen/Qwen3-0.6B"
reward_training_base_model="meta-llama/Llama-2-7b-hf"
```

## Next Steps

- [Reward Functions Guide](reward-functions.md) - Explore reward functions
- [RL Training Guide](rl.md) - Complete RL training guide
- [Reward Model Training System](../api-reference/reward-model-training-reference.md) - Detailed system docs

## Additional Resources

- [API Reference](../api-reference/core.md) - API documentation
- [Examples](../examples/rl.md) - Code examples
- [Reward Functions Reference](../api-reference/reward-functions-reference.md) - Complete function list