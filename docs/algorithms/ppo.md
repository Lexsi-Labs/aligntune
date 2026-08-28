# PPO (Proximal Policy Optimization)

Proximal Policy Optimization (PPO) is a powerful RLHF algorithm that uses a reward model to optimize language model behavior through policy gradient methods.

---

## Overview

PPO optimizes the policy by generating responses, evaluating them with a reward model, and updating the policy to maximize expected reward while staying close to the reference policy.

### Key Features

- **Reward Model**: Uses neural reward models for fine-grained control
- **Online Learning**: Generates and evaluates responses during training
- **Both Backends**: Supported by TRL and Unsloth
- **Flexible**: Works with custom reward models and functions

---

## How It Works

PPO alternates between:
1. **Rollout Phase**: Generate responses from the current policy
2. **Evaluation Phase**: Score responses with reward model
3. **Update Phase**: Update policy to maximize reward (with KL constraint)

### Mathematical Formulation

The PPO objective maximizes:

```
L_PPO = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
```

Where:
- `r_t`: Probability ratio between policy and reference
- `A_t`: Advantage estimate
- `ε`: Clipping parameter

---

## Usage

### Basic PPO Training

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="ppo",
 backend="trl",
 reward_model_name="your-reward-model", # Required
 num_epochs=1,
 batch_size=1,
 learning_rate=1e-6
)

trainer.train()
```

### PPO with Unsloth Backend

```python
trainer = create_rl_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="ppo",
 backend="unsloth",
 reward_model_name="your-reward-model",
 num_epochs=1,
 batch_size=1,
 learning_rate=1e-6
)

trainer.train()
```

### PPO with Custom Reward Model

```python
# Train custom reward model first
from aligntune.rewards.training import RewardModelTrainer
from aligntune.rewards.registry import RewardRegistry

# RewardModelTrainer expects RewardFunction objects, not name strings --
# fetch them from the registry first
registry = RewardRegistry()
helpfulness_func = registry.get_reward_function("helpfulness")
safety_func = registry.get_reward_function("safety")
coherence_func = registry.get_reward_function("coherence")

reward_trainer = RewardModelTrainer(
 base_model_name="Qwen/Qwen3-0.6B",
 reward_functions=[helpfulness_func, safety_func, coherence_func],
 composite_weights=[0.4, 0.3, 0.3]
)

# Generate training data, then train the reward model
# (RewardModelTrainer has no .train() -- use generate_training_data() +
# train_reward_model())
training_data = reward_trainer.generate_training_data(
 texts=your_training_texts,
 batch_size=32
)
reward_model_path = reward_trainer.train_reward_model(
 training_data=training_data,
 output_dir="./reward_models/custom_ppo",
 num_epochs=3
)

# Use in PPO training
ppo_trainer = create_rl_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="ppo",
 backend="unsloth",
 reward_model_path=reward_model_path,
 num_epochs=1,
 batch_size=1,
 learning_rate=1e-6
)
```

### PPO with Reward Functions

```python
trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="ppo",
 backend="trl",
 reward_functions=["helpfulness", "safety"],
 reward_function_weights=[0.6, 0.4],
 num_epochs=1,
 batch_size=1,
 learning_rate=1e-6
)
```

---

## Configuration Parameters

Config class: `trl.experimental.ppo.PPOConfig`. PPO also accepts every
parameter in the
[common RL parameters](../PARAMETERS.md#rl-reinforcement-learning-common-parameters)
reference, any `TrainingConfig` field matching a real `PPOConfig` field is
forwarded automatically, even if not listed below.

--8<-- "docs/PARAMETERS.md:ppo"

### Reward Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reward_model_name` | `str` | `None` | HuggingFace reward model ID |
| `reward_model_path` | `str` | `None` | Local reward model path |
| `reward_functions` | `list` | `[]` | List of reward function names |
| `reward_function_weights` | `list` | `[]` | Weights for reward functions |

---

## Dataset Format

PPO requires prompts (not preference pairs):

```python
{
 "prompt": "What is machine learning?",
 # Responses are generated during training
}
```

### Supported Datasets

- **Anthropic/hh-rlhf**: Can extract prompts
- **OpenAssistant/oasst1**: Conversation prompts
- Custom datasets with `prompt` column

---

## Best Practices

### 1. Reward Model Selection

- **Use pre-trained models**: HuggingFace reward models
- **Train custom models**: For domain-specific tasks
- **Use reward functions**: For rule-based rewards

### 2. Clipping Range

- **Conservative (0.1-0.2)**: Stable training, slower updates
- **Aggressive (0.2-0.3)**: Faster updates, risk of instability

### 3. KL Coefficient

- **Low (0.01-0.1)**: More exploration, risk of divergence
- **High (0.1-0.5)**: More conservative, stays close to reference

### 4. Number of Generations

- **Few (4-8)**: Faster training, less exploration
- **Many (8-16)**: More exploration, slower training

---

## Evaluation

PPO evaluation includes:

```python
config = EvalConfig(
    model_path="./output/ppo_model",
    output_dir="./eval_results/ppo",

    # Task configuration
    task_type="text",
    data_task_type="sft",

    # Metrics
    metrics=["perplexity", "reward_accuracy"],

    # Dataset
    dataset_name="HuggingFaceH4/ultrachat_200k",
    split="test_sft",
    max_samples=100,

    # Generation
    max_length=512,
    temperature=0.7,
    batch_size=8
)

results = run_eval(config)
Note: For Base model we don't need to pass model_path also if model is not trained using lora no need to pass base_model also
```

---

## Common Issues

### Reward Model Not Found

**Solutions**:
- Verify reward model name/path
- Check HuggingFace authentication
- Train custom reward model

### Training Instability

**Solutions**:
- Reduce learning rate
- Increase KL coefficient
- Reduce clipping range
- Use smaller batch size

### Low Rewards

**Solutions**:
- Check reward model quality
- Verify reward model is appropriate for task
- Adjust reward function weights
- Increase number of generations

---

## Comparison with Other Algorithms

| Feature | PPO | DPO | GRPO |
|---------|-----|-----|------|
| Reward Model | Required | Not needed | Not needed |
| Training Speed | Slow | Fast | Medium |
| Setup Complexity | High | Low | Medium |
| Reward Control | Fine-grained | Coarse | Medium |

---

## References

- [Proximal Policy Optimization Paper](https://arxiv.org/abs/1707.06347)
- [TRL PPO Documentation](https://huggingface.co/docs/trl/ppo_trainer)

---

## Next Steps

- **[DPO Guide](dpo.md)** - Learn about Direct Preference Optimization
- **[GRPO Guide](grpo.md)** - Learn about Group Relative Policy Optimization
- **[Reward Model Training](../user-guide/reward-model-training.md)** - Train custom reward models