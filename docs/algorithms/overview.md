# RL Algorithms Overview

AlignTune supports a comprehensive set of Reinforcement Learning algorithms for aligning Large Language Models with human preferences and optimizing for specific objectives.

---

## Algorithm Comparison

| Algorithm | TRL Backend | Unsloth Backend | Reward Model | Description |
|-----------|-------------|-----------------|--------------|-------------|
| **DPO** | Yes | Yes | No | Direct Preference Optimization - No reward model needed |
| **PPO** | Yes | Yes | Yes | Proximal Policy Optimization - Requires reward model |
| **GRPO** | Yes | Yes | No | Group Relative Policy Optimization - Multi-criteria optimization |
| **GSPO** | Yes | No | No | Group Sequential Policy Optimization - Sequential group learning (TRL only) |
| **DAPO** | Yes | Yes | No | Decouple Clip and Dynamic sAmpling Policy Optimization |
| **Dr. GRPO** | Yes | Yes | No | GRPO Done Right - Unbiased GRPO variant |

---

## Algorithm Selection Guide

### When to Use DPO

**Use DPO when:**
- You have preference data (chosen vs rejected pairs)
- You want to avoid training a separate reward model
- You need fast training with minimal setup
- You're working with preference datasets like Anthropic/hh-rlhf

**Example**:
```python
trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dpo",
 backend="trl"
)
```

### When to Use PPO

**Use PPO when:**
- You have a reward model (or want to train one)
- You need fine-grained control over reward signals
- You're doing online learning with environment interaction
- You want to optimize for complex, multi-objective rewards

**Example**:
```python
trainer = create_rl_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="ppo",
 backend="unsloth",
 reward_model_name="your-reward-model"
)
```

### When to Use GRPO

**Use GRPO when:**
- You want multi-criteria optimization
- You're working with group-based preferences
- You need to optimize for multiple objectives simultaneously
- You want to avoid reward model training

**Example**:
```python
trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="grpo",
 backend="trl"
)
```

### When to Use GSPO

**Use GSPO when:**
- You need sequential group learning
- You're working with sequential decision-making tasks
- You want TRL backend (Unsloth not supported)

**Example**:
```python
trainer = create_rl_trainer(
 model_name="Qwen/Qwen2-1.5B-Instruct",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="gspo",
 backend="trl" # Only TRL supports GSPO
)
```

### When to Use DAPO

**Use DAPO when:**
- You want to address GRPO limitations
- You need decoupled clipping and dynamic sampling
- You're working with large-scale RLHF

**Example**:
```python
trainer = create_rl_trainer(
 model_name="Qwen/Qwen2-1.5B-Instruct",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dapo",
 backend="trl"
)
```

### When to Use Dr. GRPO

**Use Dr. GRPO when:**
- You want an unbiased GRPO variant
- You need to correct optimization biases
- You're working with group-based preferences

**Example**:
```python
trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="drgrpo",
 backend="trl"
)
```

### When to Use Dr. GRPO

**Use Dr. GRPO when:**
- You want an unbiased GRPO variant
- You need to correct optimization biases
- You're working with group-based preferences

**Example**:
```python
trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="drgrpo",
 backend="trl"
)
```

---

## Algorithm Details

- **[DPO](dpo.md)** - Direct Preference Optimization
- **[PPO](ppo.md)** - Proximal Policy Optimization
- **[GRPO](grpo.md)** - Group Relative Policy Optimization
- **[GSPO](gspo.md)** - Group Sequential Policy Optimization
- **[DAPO](dapo.md)** - Decouple Clip and Dynamic sAmpling Policy Optimization
- **[Dr. GRPO](dr-grpo.md)** - GRPO Done Right

---

## Quick Reference

### Algorithm Requirements

| Algorithm | Dataset Format | Reward Model | Special Requirements |
|-----------|---------------|--------------|---------------------|
| DPO | Preference pairs | | Chosen/rejected pairs |
| PPO | Prompts | | Reward model required |
| GRPO | Group preferences | | Group structure |
| GSPO | Sequential groups | | Sequential structure |
| DAPO | Group preferences | | Group structure |
| Dr. GRPO | Group preferences | | Group structure |

---

## Next Steps

- **[DPO Guide](dpo.md)** - Learn about Direct Preference Optimization
- **[PPO Guide](ppo.md)** - Learn about Proximal Policy Optimization
- **[GRPO Guide](grpo.md)** - Learn about Group Relative Policy Optimization
- **[Backend Selection](../getting-started/backend-selection.md)** - Choose the right backend