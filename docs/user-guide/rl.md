# Reinforcement Learning (RL) Training Guide

Complete guide to Reinforcement Learning from Human Feedback (RLHF) training with AlignTune, covering all supported RLHF algorithms.

## Overview

Reinforcement Learning training aligns language models with human preferences using various algorithms. AlignTune supports multiple RLHF algorithms:

1. **DPO / Online-DPO** - Direct Preference Optimization
2. **PPO** - Proximal Policy Optimization
3. **GRPO / GSPO / DAPO / Dr. GRPO** - Group-relative/sequential methods
4. **C-GRPO / GBMPO / PACE** - Advanced GRPO variants
5. **ORPO** - Highly efficient DPO alternative
6. **SPIN** - Self-play method

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
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dpo",
 backend="trl",
 column_mapping={
     "prompt": "prompt",
     "chosen": "chosen",
     "rejected": "rejected",
 },
 num_epochs=1,
 batch_size=4,
 learning_rate=5e-5,
 max_seq_length=1024,
 # DPO-specific parameters
 beta=0.1,                 # Reference-model regularization strength
 loss_type="sigmoid",    # DPO objective
 label_smoothing=0.0,     # Preference-label smoothing
 truncation_mode="keep_end",
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

The current DPO configuration uses `max_seq_length` as the overall
`max_length` limit. Prompt and completion limits are not separate DPOConfig
fields. If the source dataset uses different column names, update
`column_mapping` accordingly.

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
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="openai/gsm8k",
    algorithm="grpo",
    backend="trl",
    reward_functions=["math_correctness"],
    reward_function_weights=[1.0],
    num_epochs=1,
    batch_size=4,
    learning_rate=1e-6,
    num_generations=4,
    max_completion_length=256,
    temperature=0.7,
    top_p=0.95,
    beta=0.1,
    epsilon=0.2,
    loss_type="grpo",
    importance_sampling_level="token",
    mask_truncated_completions=False,
    scale_rewards="group",
)

trainer.train()

```

GRPO-family trainers share this configuration surface:

- `reward_functions` and `reward_function_weights` are required for GRPO
  training.
- `num_generations` controls the number of completions sampled per prompt and
  must be compatible with the effective training batch size.
- `max_completion_length`, `temperature`, and `top_p` control rollout
  generation.
- `beta` controls the reference-model KL penalty and `epsilon` controls the
  clipping range.
- `loss_type` selects the GRPO-family objective.
- `importance_sampling_level` accepts `"token"` or `"sequence"`.
- `scale_rewards` and `mask_truncated_completions` control reward scaling and
  handling of unfinished completions.

### 4. GSPO (Group Sequence Policy Optimization)

GSPO is a thin subclass of GRPO that switches importance sampling from per-token to per-sequence.
It is available on **both** TRL and Unsloth backends.

**Advantages:**
- Sequence-level importance sampling
- Group-based optimization
- More stable gradients for long completions

**Use Cases:**
- Sequence-level preference learning
- Structured policy optimization
- Group-based RLHF

#### Example

```python
trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="openai/gsm8k",
 algorithm="gspo",
 backend="trl",
 reward_functions=["math_correctness"],
 reward_function_weights=[1.0],
 num_epochs=1,
 batch_size=4,
 learning_rate=1e-6,
 num_generations=4,
 max_completion_length=256,
 importance_sampling_level="sequence",
 loss_type="dapo",
)

trainer.train()
```

GSPO is implemented as the GRPO trainer with sequence-level importance
sampling. Its defaults are `importance_sampling_level="sequence"` and
`loss_type="dapo"`; all other parameters come from GRPO. This is a GRPO-family
approximation rather than a separate native GSPO trainer. `group_size` and
`sequential_steps` are not valid configuration fields.

### 5. DAPO (Decouple Clip and Dynamic sAmpling Policy Optimization)

A thin GRPO subclass that only overrides `loss_type -> 'dapo'` to address
length-normalization behavior in the GRPO loss. Available on both backends.

**Use Cases:** dynamic-sampling-style GRPO training without switching trainers.

```python
trainer = create_rl_trainer(
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="openai/gsm8k",
    algorithm="dapo",
    backend="trl",
    reward_functions=["math_correctness"],
    reward_function_weights=[1.0],
    num_generations=4,
    max_completion_length=256,
    loss_type="dapo",
)
trainer.train()
```
Full parameter surface is GRPO's, see [DAPO](../algorithms/dapo.md).

### 6. Dr. GRPO (GRPO Done Right)

A thin GRPO subclass that only overrides `loss_type -> 'dr_grpo'`, addressing
length-normalization bias in the GRPO loss. Available on both backends.

**Use Cases:** more reliable convergence than vanilla GRPO with no extra setup.

```python
trainer = create_rl_trainer(
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="openai/gsm8k",
    algorithm="drgrpo",
    backend="trl",
    reward_functions=["math_correctness"],
    reward_function_weights=[1.0],
    num_generations=4,
    max_completion_length=256,
    loss_type="dr_grpo",
)
trainer.train()
```
Full parameter surface is GRPO's. In this implementation, using
`algorithm="grpo", loss_type="dr_grpo"` selects the same underlying GRPO
objective as `algorithm="drgrpo"`.

### 7. GBMPO (Group-Based Mirror Policy Optimization)

Extends GRPO with a divergence-based regularization term added to the policy
loss, without requiring a value network. Available on both backends.

**Use Cases:** standard GRPO collapses from over-optimization; you want PPO/TRPO-style stability without the memory cost of a value model.

```python
trainer = create_rl_trainer(
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="openai/gsm8k",
    algorithm="gbmpo",
    backend="trl",
    reward_functions=["math_correctness"],
    reward_function_weights=[1.0],
    num_generations=4,
    max_completion_length=256,
    temperature=0.7,
    top_p=0.95,
    gbmpo_divergence_type="l2",
    gbmpo_l2_coefficient=0.0001,
    beta=0.0,
    num_epochs=1,
    batch_size=4,
    learning_rate=1e-6,
)
trainer.train()
```

GBMPO inherits the shared GRPO parameters. Its additional controls are:

- `gbmpo_divergence_type`: `"l2"`, `"l2kl"`, `"prob_l2"`, or `"prob_l2kl"`.
- `gbmpo_l2_coefficient`: strength of the added L2 regularization term.

For `l2` and `prob_l2`, set `beta=0.0` to disable the separate KL term. The
factory makes a tiny compatibility adjustment internally because the parent
TRL GRPO trainer expects a nonzero beta value. If `gbmpo_divergence_type` is
omitted, the factory defaults to `"l2kl"`.

See [GBMPO](../algorithms/gbmpo.md) for the full divergence variant table.

### 8. Counterfactual GRPO (C-GRPO)

Extends GRPO with a baseline-swapping penalty that neutralizes reward hacking (e.g. length hacking) without an explicit length penalty. Available on both backends.

**Use Cases:** GRPO training converges to models that exploit the reward (e.g. abnormally long responses) for a higher score.

```python
trainer = create_rl_trainer(
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="Anthropic/hh-rlhf",
    algorithm="counterfact_grpo",
    backend="trl",  # also available with backend="unsloth"
)
trainer.train()
```
See [Counterfactual GRPO](../algorithms/counterfactual-grpo.md) for full configuration.

### 9. PACE (Baseline-Optimized Learning Technique)

Tracks a learned, per-prompt historical baseline reward and penalizes generations that fail to beat it, giving less noisy gradients than standard GRPO/PPO. Available on both backends.

**Use Cases:** reward signals have high variance across prompts; you want continuous improvement tracked per-prompt on reasoning tasks.

```python
trainer = create_rl_trainer(
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="openai/gsm8k",
    algorithm="pace",
    backend="trl",  # also available with backend="unsloth"
)
trainer.train()
```
See [PACE](../algorithms/pace.md) for the curriculum/baseline parameter table.

### 10. Online-DPO

Runs DPO iteratively inside an active-learning loop: the model generates responses on the fly, a reward model (or LLM-as-judge) scores/pairs them, and the model is tuned on its own generated distribution instead of a static offline dataset.

**Use Cases:** you have a reward model or synthetic evaluator; offline DPO has plateaued because the model's generations have drifted from the static dataset.

```python
trainer = create_rl_trainer(
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="Anthropic/hh-rlhf",
    algorithm="online_dpo",
    backend="trl",  # also available with backend="unsloth"
    reward_model_name="your-active-reward-model",
    max_new_tokens=64,
    max_length=512,
    beta=0.1,
    loss_type="sigmoid",
)
trainer.train()
```

Online DPO requires either `reward_model_name` or at least one configured
registry/custom reward. For example, instead of a reward model you can pass:

```python
rewards=[
    {
        "type": "math_correctness",
        "weight": 1.0,
        "params": {},
    }
]
```

The dataset is prompt-only: Online DPO generates completions during training,
scores them with the reward model or reward functions, and constructs the
preference pairs internally.

See [Online-DPO](../algorithms/online-dpo.md) for full configuration.

### 11. SPIN (Self-Play Fine-Tuning)

SPIN uses an SFT dataset to create preference pairs online:

- The dataset response becomes `chosen`.
- The current model generates the `rejected` response.
- TRL's `DPOTrainer` trains on those synthetic preference pairs.
- No reward model or reward function is used.

SPIN expects SFT rows containing either:

- `prompt` and `completion`/`answer`/`response`, or
- conversational `messages` with an assistant response.

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="your_sft_dataset",
    algorithm="spin",
    backend="trl",  # or "unsloth"

    column_mapping={
        "prompt": "prompt",
        "completion": "completion",
    },

    num_rounds=3,
    samples_per_round=256,
    dpo_steps_per_round=50,

    batch_size=4,
    learning_rate=1e-5,
    gradient_accumulation_steps=1,

    max_seq_length=1024,
    beta=0.1,
    loss_type="sigmoid",

    generation_batch_size=8,
    generation_max_prompt_length=512,
    generation_max_length=256,
    generation_temperature=0.7,
    generation_top_p=0.95,
    generation_top_k=0,
    generation_do_sample=True,
    enable_thinking=False,

    eval_strategy="no",
    save_strategy="steps",
    save_steps=50,
    report_to="none",
)

result = trainer.train()
```

`max_steps` overrides `dpo_steps_per_round` and applies separately to each
round. If `samples_per_round` is set, the dataset must contain at least
`num_rounds * samples_per_round` training rows. Set `eval_strategy` to
`"steps"` to generate validation preference pairs each round; use
`eval_samples` to limit validation cost.

SPIN does not use `reward_model_name`, `reward_functions`, or `rewards`.

See [SPIN](../algorithms/spin.md) for full configuration.

### 12. RAFT (Retrieval Augmented Fine-Tuning)

Document-grounded SFT: trains the model to answer using a mix of golden (relevant) and distractor (irrelevant) documents in context, so it learns to identify and cite the right source. Uses its own factory function, `create_raft_trainer`, separate from `create_rl_trainer`.

**Use Cases:** RAG-style deployments where the model needs to learn to ignore distractor context and cite the right document.

```python
from aligntune.core.backend_factory import create_raft_trainer

trainer = create_raft_trainer(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    train_examples=train_examples,
    backend="trl",  # or "unsloth"
    max_golden_docs=3,
    max_distractor_docs=5,
)
trainer.train()
```
See [RAFT](../algorithms/raft.md) for full configuration, including the citation-loss placeholder caveat.

### 13. ORPO (Odds Ratio Preference Optimization)

Combines SFT and preference alignment into a single training objective by adding an odds-ratio penalty directly to the NLL loss, no separate SFT phase and no reference model needed.

**Use Cases:** you want instruction-tuning and alignment in one pass; you're memory-constrained and can't host a frozen reference model.

```python
trainer = create_rl_trainer(
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="Anthropic/hh-rlhf",
    algorithm="orpo",
    backend="trl",  # also available with backend="unsloth"
    column_mapping={
        "prompt": "prompt",
        "chosen": "chosen",
        "rejected": "rejected",
    },
    beta=0.1,
    max_seq_length=1024,
    num_epochs=1,
    batch_size=4,
    learning_rate=5e-6,
)
trainer.train()
```

ORPO requires paired preference rows containing `prompt`, `chosen`, and
`rejected`. It combines the language-model loss with the odds-ratio penalty in
one pass, so it does not require a reward model or a separate reference model.
The installed TRL configuration uses `max_seq_length` as the overall sequence
limit; `max_prompt_length` and `truncation_mode` are not valid ORPO parameters.

See [ORPO](../algorithms/orpo.md) for full configuration.

## Backend Selection

### TRL Backend

**Use TRL when:**
- Need maximum compatibility
- Need vLLM-backed rollout (`rollout_backend='vllm'`), which is TRL-only
- Working with standard models
- Want the most-tested training path

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
- Training PPO, DPO, GRPO, GSPO, DAPO, or Dr. GRPO

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
