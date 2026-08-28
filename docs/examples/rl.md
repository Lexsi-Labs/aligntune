# RL Examples

Comprehensive examples for Reinforcement Learning training with AlignTune.

## Basic Examples

### 1. DPO Training

Basic DPO training example:

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dpo",
 backend="trl", # Use TRL for GPT2 models
 num_epochs=1,
 batch_size=4,
 learning_rate=5e-5,
 max_seq_length=512,
 max_samples=1000,
 beta=0.1,
 # For GPT2 models, add: lora_target_modules=["c_attn", "c_proj"]
 # Or use a different model like Qwen/Qwen3-0.6B or Llama models
)

trainer.train()
model_path = trainer.save_model()
```

### ORPO Training

ORPO combines SFT and preference alignment into a single training pass — no
separate reference model needed. It uses the same `create_rl_trainer()` entry
point and the same preference-pair columns (`prompt`, `chosen`, `rejected`)
as DPO:

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="orpo",
 backend="trl", # also available with backend="unsloth"
 column_mapping={
     "prompt": "prompt",
     "chosen": "chosen",
     "rejected": "rejected",
 },
 num_epochs=1,
 batch_size=4,
 learning_rate=5e-6,
 max_seq_length=512,
 max_samples=1000,
 beta=0.1,
)

trainer.train()
model_path = trainer.save_model()
```

### GRPO Family: GRPO, GSPO, DAPO, and Dr. GRPO

These algorithms share the GRPO prompt-and-reward workflow. The factory
selects the algorithm-specific trainer from the `algorithm` value; the reward
function and generation settings remain the same.

```python
from aligntune.core.backend_factory import create_rl_trainer

COMMON_GRPO = dict(
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="openai/gsm8k",
    backend="trl",
    task_type="grpo",
    column_mapping={
        "prompt": "question",
        "reference": "answer",
    },
    reward_functions=["math_correctness"],
    num_epochs=1,
    batch_size=4,
    num_generations=4,
    max_seq_length=1024,
    max_prompt_length=512,
    max_completion_length=256,
    learning_rate=1e-6,
    temperature=0.7,
    top_p=0.95,
)

# Select one algorithm at a time.
grpo_trainer = create_rl_trainer(
    **COMMON_GRPO,
    algorithm="grpo",
    loss_type="grpo",
)

gspo_trainer = create_rl_trainer(
    **COMMON_GRPO,
    algorithm="gspo",
    loss_type="dapo",
)

dapo_trainer = create_rl_trainer(
    **COMMON_GRPO,
    algorithm="dapo",
    loss_type="dapo",
)

drgrpo_trainer = create_rl_trainer(
    **COMMON_GRPO,
    algorithm="drgrpo",
    loss_type="dr_grpo",
)

result = grpo_trainer.train()
print(result)
```

### GBMPO

GBMPO adds a configurable divergence and L2 regularization term to the policy
update. The reward and dataset pipeline remains the same as other GRPO-family
trainers.

```python
from aligntune.core.backend_factory import create_rl_trainer

gbmpo_trainer = create_rl_trainer(
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="openai/gsm8k",
    backend="trl",
    algorithm="gbmpo",
    task_type="grpo",
    column_mapping={"prompt": "question", "reference": "answer"},
    reward_functions=["math_correctness"],
    num_generations=4,
    batch_size=4,
    max_seq_length=1024,
    max_prompt_length=512,
    max_completion_length=256,
    learning_rate=1e-6,
    gbmpo_l2_coefficient=0.01,
    gbmpo_divergence_type="kl",
    num_epochs=1,
)

result = gbmpo_trainer.train()
print(result)
```

### Counterfactual GRPO

Counterfactual GRPO reweights generated spans using counterfactual importance.
In addition to normal rewards, it exposes controls for span weighting and
debugging.

```python
from aligntune.core.backend_factory import create_rl_trainer

counterfactual_grpo_trainer = create_rl_trainer(
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="openai/gsm8k",
    backend="trl",
    algorithm="counterfact_grpo",
    task_type="grpo",
    column_mapping={"prompt": "question", "reference": "answer"},
    reward_functions=["math_correctness"],
    num_generations=4,
    batch_size=4,
    max_seq_length=1024,
    max_prompt_length=512,
    max_completion_length=256,
    learning_rate=1e-6,
    method_name="counterfactual",
    max_spans=4,
    boost_factor=2.0,
    min_weight=0.5,
    num_epochs=1,
)

result = counterfactual_grpo_trainer.train()
print(result)
```

### Algorithm Differences

- **GRPO**: group-relative advantages from multiple sampled completions.
- **GSPO**: sequence-level importance sampling for the GRPO update.
- **DAPO**: GRPO-family objective with DAPO clipping and token-handling
  options.
- **Dr. GRPO**: GRPO with the Dr. GRPO loss normalization.
- **GBMPO**: GRPO-style updates with configurable divergence and L2
  regularization.
- **Counterfactual GRPO**: counterfactual span weighting on top of
  generated-response rewards.

**Expected Output:**
```
Training started...
Epoch 1/1: 100%|| 250/250 [08:45<00:00, loss=0.234]
Model saved to: ./output/model
```

### 2. PPO with Pre-trained Reward Model

PPO training with HuggingFace reward model:

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="ppo",
 backend="unsloth",
 reward_model_name="Skywork/Skywork-Reward-V2-Qwen3-0.6B",
 num_epochs=1,
 batch_size=1,
 learning_rate=1e-6,
 kl_coef=0.1,
 cliprange=0.2
)

trainer.train()
```

### 3. PPO with Custom Reward Model

Train custom reward model during PPO:

```python
from aligntune.core.backend_factory import create_rl_trainer

def load_training_texts():
 return [
 "This is a helpful response.",
 "I'm not sure about this.",
 "That's a great question!",
 "Here is a concise explanation.",
 "The calculation is shown step by step.",
 "This answer includes useful context.",
 "I cannot verify that claim from the available information.",
 "The main result is summarized below.",
 "This is an example response for reward-model training.",
 "The response addresses the user's question directly.",
 ]

trainer = create_rl_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="HuggingFaceH4/ultrafeedback_binarized",
 algorithm="ppo",
 backend="unsloth",
 train_custom_reward_model=True,
 reward_training_texts=load_training_texts(),
 reward_functions=["length", "sentiment", "safety", "coherence"],
 reward_function_weights=[0.2, 0.3, 0.3, 0.2],
 reward_training_base_model="microsoft/DialoGPT-medium",
 reward_training_output_dir="./reward_models/custom",
 num_epochs=1,
 batch_size=1,
 learning_rate=2e-4
)

trainer.train()
```

## Advanced Examples

### 4. GRPO Training

Group Relative Policy Optimization:

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="grpo",
 backend="unsloth",
 reward_functions=["length"],
 num_epochs=1,
 batch_size=2, # GRPO requires minimum batch_size=2 for generations
 learning_rate=1e-6,
 loss_type='grpo',
)

trainer.train()
```

### 5. GSPO Training

Group Sequential Policy Optimization (TRL only):

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="gspo",
 backend="trl", # GSPO only works with TRL
 num_epochs=1,
 batch_size=4,
 learning_rate=5e-5,

)

trainer.train()
```

### 6. Multi-Stage Pipeline

SFT → DPO → PPO pipeline:

```python
# Stage 1: SFT
from aligntune.core.backend_factory import create_rl_trainer, create_sft_trainer

sft_trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="tatsu-lab/alpaca",
 backend="trl",
 num_epochs=3
)
sft_trainer.train()
sft_path = sft_trainer.save_model()

# Stage 2: DPO
dpo_trainer = create_rl_trainer(
 model_name=sft_path, # Start from SFT model
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dpo",
 backend="trl",
 num_epochs=1
)
dpo_trainer.train()
dpo_path = dpo_trainer.save_model()

# Stage 3: PPO
ppo_trainer = create_rl_trainer(
 model_name=dpo_path, # Start from DPO model
 dataset_name="Anthropic/hh-rlhf",
 algorithm="ppo",
 backend="unsloth",
 reward_model_name="Skywork/Skywork-Reward-V2-Qwen3-0.6B",
 num_epochs=1
)
ppo_trainer.train()
```

## Complete Workflow Example

```python
from aligntune.core.backend_factory import create_rl_trainer
# Create trainer
trainer = create_rl_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dpo",
 backend="trl", # Use TRL for GPT2 models
 num_epochs=1,
 batch_size=4,
 learning_rate=5e-5,
 max_seq_length=512,
 max_samples=1000,
 beta=0.1,
 lora_target_modules=["c_attn", "c_proj"], # or set use_peft = False
 eval_interval=100,
 save_interval=500
)

# Train
print("Starting training...")
results = trainer.train()
print(f"Training completed: {results}")

# Save model
model_path = trainer.save_model()
print(f"Model saved to: {model_path}")
```

## Running Examples

### From Command Line

```bash
# DPO training
python examples/trl_dpo_1.py

# PPO training
python examples/trl_ppo1.py

# GRPO training
python examples/trl_grpo_1.py
```

## Tips

1. **Start with DPO**: Simpler setup, no reward model needed
2. **Model Family Consistency**: For PPO, ensure all models same family
3. **Reward Models**: Use pre-trained when available, train custom for domains
4. **Backend Selection**: Unsloth for speed, TRL for compatibility/GSPO
5. **Memory Management**: Use smaller batch sizes for PPO

## Next Steps

- [SFT Examples](sft.md) - SFT training examples
- [Advanced Examples](advanced.md) - Advanced use cases
- [RL Guide](../user-guide/rl.md) - Complete RL guide
