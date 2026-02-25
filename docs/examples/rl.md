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

**Expected Output:**
```
Training started...
Epoch 1/1: 100%|| 250/250 [08:45<00:00, loss=0.234]
Model saved to: ./output/model
```

### 2. PPO with Pre-trained Reward Model

PPO training with HuggingFace reward model:

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
 kl_coef=0.1,
 cliprange=0.2
)

trainer.train()
```

### 3. PPO with Custom Reward Model

Train custom reward model during PPO:

```python
def load_training_texts():
 return [
 "This is a helpful response.",
 "I'm not sure about this.",
 "That's a great question!",
 # ... more texts
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
trainer = create_rl_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="grpo",
 backend="unsloth",
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
from aligntune.core.backend_factory import create_sft_trainer

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
from datasets import load_dataset



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
 lora_target_modules=["c_attn", "c_proj"] # or set use_peft = False
 eval_interval=100,
 save_interval=500
)

# Train
print("Starting training...")
results = trainer.train()
print(f"Training completed: {results}")

# Evaluate
print("Evaluating...")
metrics = trainer.evaluate()
print(f"Evaluation metrics: {metrics}")

# Save model
model_path = trainer.save_model()
print(f"Model saved to: {model_path}")

# Test predictions
print("Testing predictions...")
prompts = [
 "What is machine learning?",
 "Explain deep learning"
]

results = trainer.predict(prompts)
for prompt, result in zip(prompts, results):
 print(f"Q: {prompt}")
 print(f"A: {result}\n")
```

## Running Examples

### From Command Line

```bash
# DPO training
python examples/dpo_intel_orca_trl/train_dpo_intel_orca_trl.py

# PPO training
python examples/ppo_ultrachat_unsloth/train_ppo_ultrachat_unsloth.py

# GRPO training
python examples/grpo_gsm8k_trl/train_grpo_direct_api.py
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