# Reward Functions Guide

Complete guide to using reward functions in AlignTune for RLHF training.

## Overview

Reward functions score generated completions during online RL training. AlignTune
provides **50+ registered rewards** covering quality, safety, task-specific
metrics, and more. This is separate from training a neural reward model: PPO
can consume a reward model, while GRPO, Online-DPO, and related trainers
can consume registry rewards or custom callables directly.

## Quick Start

### Using Registry Rewards in Online RL

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="openai/gsm8k",
 algorithm="grpo",  # also works for other GRPO-family trainers and Online-DPO
 backend="trl",
 reward_functions=["math_correctness"],
 reward_function_weights=[1.0],
 num_generations=4,
 max_completion_length=256,
 num_epochs=1,
 batch_size=2,
)

trainer.train()
```

`reward_functions` is the convenience factory argument for registered names.
The factory converts it to the canonical `rewards` configuration. You can
also pass that configuration directly:

```python
trainer = create_rl_trainer(
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="openai/gsm8k",
    algorithm="grpo",
    rewards=[
        {
            "type": "math_correctness",
            "weight": 1.0,
            "params": {},
        }
    ],
)
```

For a custom reward, provide a TRL-style callable through a `custom` reward
spec. It must return one numeric score per completion in the batch:

```python
def short_answer_reward(completions, **kwargs):
    return [1.0 if len(text.split()) <= 100 else 0.0 for text in completions]

trainer = create_rl_trainer(
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="openai/gsm8k",
    algorithm="grpo",
    rewards=[
        {"type": "custom", "params": {"function": short_answer_reward}, "weight": 1.0}
    ],
)
```

TRL calls rewards batch-wise with `prompts`, `completions`, and any preserved
dataset columns. Do not return one scalar for a multi-example batch. Keep
reference columns in the dataset when the reward needs them.

## Define and Pass a Custom Reward

Custom rewards must use TRL's batch interface. Return exactly one numeric score
for every completion in the batch:

```python
def my_reward(prompts, completions, **kwargs):
    return [
        1.0 if "final answer" in completion.lower() else 0.0
        for completion in completions
    ]

trainer = create_rl_trainer(
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="openai/gsm8k",
    algorithm="grpo",
    rewards=[my_reward],
)
```

You can also pass the callable using an explicit custom reward specification:

```python
rewards=[
    {
        "type": "custom",
        "weight": 1.0,
        "params": {"function": my_reward},
    }
]
```

If the reward needs a target or reference, include that column in the dataset
and accept it as a batch argument:

```python
def exact_match_reward(completions, reference, **kwargs):
    return [
        1.0 if prediction.strip() == target.strip() else 0.0
        for prediction, target in zip(completions, reference)
    ]

trainer = create_rl_trainer(
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="your-dataset",
    algorithm="grpo",
    rewards=[exact_match_reward],
)
```

Registry rewards such as `"math_correctness"` can still be passed by name in
`reward_functions=[...]` or as `{"type": "math_correctness", ...}` in
`rewards=[...]`; AlignTune adapts those built-in scalar `compute(...)` methods
to TRL's batch interface automatically.

### Using Reward Functions in Reward Model Training

```python
from aligntune.rewards.training import RewardModelTrainer
from aligntune.rewards.registry import RewardRegistry

# Get reward functions from registry
registry = RewardRegistry()
length_func = registry.get_reward_function("length")
sentiment_func = registry.get_reward_function("sentiment")
safety_func = registry.get_reward_function("safety")

# Create trainer with reward functions
trainer = RewardModelTrainer(
 base_model_name="microsoft/DialoGPT-medium",
 reward_functions=[length_func, sentiment_func, safety_func],
 composite_weights=[0.3, 0.4, 0.3]
)

# Generate training data
training_data = trainer.generate_training_data(
 texts=["This is a good response.", "This is a bad response."],
 batch_size=32
)

# Train reward model
model_path = trainer.train_reward_model(
 training_data=training_data,
 output_dir="./reward_models/custom",
 num_epochs=3
)
```

## Reward Function Categories

### Basic Quality Rewards

#### Length Reward
Encourages appropriate response length.

```python
reward_functions=["length"]
# With parameters
reward_config = {
 "type": "length",
 "params": {"min_length": 10, "max_length": 500},
 "weight": 1.0
}
```

#### Coherence Reward
Measures logical flow and coherence.

```python
reward_functions=["coherence"]
```

#### Fluency Reward
Measures grammatical correctness.

```python
reward_functions=["fluency"]
```

### Task-Specific Rewards

#### Sentiment Reward
Encourages specific sentiment.

```python
reward_functions=["sentiment"]
# With target sentiment
reward_config = {
 "type": "sentiment",
 "params": {"target_sentiment": "positive"},
 "weight": 1.0
}
```

#### Safety Reward
Detects and penalizes harmful content.

```python
reward_functions=["safety"]
# With strict mode
reward_config = {
 "type": "safety",
 "params": {"strict": True},
 "weight": 2.0
}
```

#### Factuality Reward
Measures factual accuracy.

```python
reward_functions=["factuality"]
```

### Generation Quality Metrics

#### BLEU Reward
Measures n-gram overlap with reference.

```python
reward_functions=["bleu"]
reward_config = {
 "type": "bleu",
 "params": {"n_gram": 4},
 "weight": 0.8
}
```

#### ROUGE Reward
Measures recall-oriented evaluation.

```python
reward_functions=["rouge"]
reward_config = {
 "type": "rouge",
 "params": {"rouge_type": "rouge-l"},
 "weight": 1.0
}
```

### Code Quality Rewards

#### Code Syntax Reward
Validates code syntax correctness.

```python
reward_functions=["code_syntax"]
reward_config = {
 "type": "code_syntax",
 "params": {"language": "python"},
 "weight": 1.0
}
```

#### Code Execution Reward
Tests if code runs without errors.

```python
reward_functions=["code_execution"]
reward_config = {
 "type": "code_execution",
 "params": {"timeout": 5, "safe_mode": True},
 "weight": 1.5
}
```

### Math and Reasoning Rewards

#### Math Correctness Reward
Validates mathematical correctness.

```python
reward_functions=["math_correctness"]
reward_config = {
 "type": "math_correctness",
 "params": {"tolerance": 1e-6},
 "weight": 1.0
}
```

#### Logical Consistency Reward
Measures logical consistency.

```python
reward_functions=["logical_consistency"]
```

## Composite Rewards

Combine multiple registry rewards with weights:

```python
trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="openai/gsm8k",
 algorithm="grpo",
 rewards=[
  {"type": "math_correctness", "weight": 0.7, "params": {}},
  {"type": "length", "weight": 0.3,
   "params": {"min_length": 1, "max_length": 256}},
 ],
)
```

## Using Reward Registry

### Get Available Functions

```python
from aligntune.rewards.registry import RewardRegistry

registry = RewardRegistry()

# List all available functions
available = registry.list_rewards()
print(available)

# Get specific function
length_func = registry.get_reward_function("length")
sentiment_func = registry.get_reward_function("sentiment")
```

### Train a Neural Reward Model (separate workflow)

`RewardModelTrainer` trains a model from text and reward features. It is not
the same as passing a reward callable to GRPO/Online-DPO. Use this path
when you need a reusable neural reward model, then pass its saved path to a
trainer that supports `reward_model_name` or `reward_model_path`.

### Create Custom Reward Function for Online RL

```python
from aligntune.rewards.core import RewardFunction, RewardConfig, RewardType
from aligntune.rewards.registry import RewardRegistry
from aligntune.core.backend_factory import create_rl_trainer

def custom_reward(completions, **kwargs):
    # Return exactly one numeric score per completion.
    return [1.0 if "final answer" in text.lower() else 0.0
            for text in completions]

trainer = create_rl_trainer(
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="openai/gsm8k",
    algorithm="grpo",
    rewards=[
        {"type": "custom", "params": {"function": custom_reward}, "weight": 1.0}
    ],
)
```

For a reusable named reward, register it with `RewardRegistry` and then use
its name in `reward_functions` or `rewards`.

## Best Practices

### 1. Choose Relevant Rewards

Select rewards that match your task:

- **Customer Service**: sentiment, politeness, helpfulness
- **Code Generation**: code_syntax, code_execution, code_completeness
- **Math Problems**: math_correctness, logical_consistency
- **Safety-Critical**: safety, toxicity, bias

### 2. Weight Balancing

Balance reward weights based on importance:

```python
# Safety is critical - higher weight
reward_function_weights=[0.1, 0.1, 0.5, 0.2, 0.1]
# length, sentiment, safety, coherence, helpfulness
```

### 3. Start Simple

Begin with a few key rewards, then expand:

```python
# Start with basics
reward_functions=["length", "safety"]

# Add more as needed
reward_functions=["length", "safety", "sentiment", "coherence"]
```

### 4. Test Reward Functions

Test rewards independently before combining:

```python
from aligntune.rewards.registry import RewardRegistry

registry = RewardRegistry()
length_func = registry.get_reward_function("length")

# Test on sample text
test_text = "This is a test response."
score = length_func.compute(test_text)
print(f"Length reward: {score}")
```

## Complete Example

```python
from aligntune.core.backend_factory import create_rl_trainer

def load_training_texts():
    """Load training texts for reward model."""
    return [
    "This is a helpful and informative response.",
    "I'm not sure, but here's what I think.",
    "That's a great question! Let me explain.",
    # ... more texts
    ]

# Create trainer with custom reward model
trainer = create_rl_trainer(
    model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    dataset_name="HuggingFaceH4/ultrafeedback_binarized",
    algorithm="ppo",
    backend="unsloth",
    # Custom reward model with multiple functions
    train_custom_reward_model=True,
    reward_training_texts=load_training_texts(),
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
    0.35, # safety (highest weight)
    0.20, # coherence
    0.15 # helpfulness
    ],
    reward_training_base_model="microsoft/DialoGPT-medium",
    reward_training_output_dir="./reward_models/custom",
    # PPO configuration
    num_epochs=1,
    batch_size=1,
    learning_rate=2e-4
)

# Train
trainer.train()
```

## Available Reward Functions

For a complete list of all 50+ reward functions, see the [Reward Functions Reference](../api-reference/reward-functions-reference.md).

**Categories:**
- Basic Quality (length, coherence, fluency)
- Task-Specific (sentiment, safety, factuality, bias)
- Generation Metrics (BLEU, ROUGE, METEOR, BERTScore)
- Code Quality (syntax, execution, completeness)
- Math & Reasoning (correctness, logical consistency, commonsense)
- Specialized (hallucination, toxicity, politeness, helpfulness, honesty)

## Next Steps

- [Reward Model Training](reward-model-training.md) - Train custom reward models
- [RL Training Guide](rl.md) - Complete RL training guide
- [Reward Functions Reference](../api-reference/reward-functions-reference.md) - Complete function list

## Additional Resources

- [API Reference](../api-reference/core.md) - API documentation
- [Examples](../examples/rl.md) - Code examples
- [Reward Model Training System](../api-reference/reward-model-training-reference.md) - Detailed system docs
