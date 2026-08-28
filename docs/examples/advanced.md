# Advanced Examples

Advanced examples demonstrating sophisticated use cases with AlignTune.

## Distributed Training

### distributed training DDP Training

```python
from aligntune.core.backend_factory import create_sft_trainer
import torch

# Configure for DDP
trainer = create_sft_trainer(
 model_name="meta-llama/Llama-2-7b-hf",
 dataset_name="tatsu-lab/alpaca",
 backend="trl",
 num_epochs=3,
 batch_size=2, # Per GPU
 gradient_accumulation_steps=4
)

# Run with accelerate
# accelerate launch --multi_gpu train.py
trainer.train()
```

### FSDP Training

```python
# Configure for FSDP
trainer = create_sft_trainer(
 model_name="meta-llama/Llama-2-7b-hf",
 dataset_name="tatsu-lab/alpaca",
 backend="trl",
 num_epochs=3,
 batch_size=1,
 gradient_accumulation_steps=8
)

# Run with accelerate FSDP
# accelerate launch --fsdp train.py
trainer.train()
```

## Custom Reward Functions

### Creating Custom Rewards

```python
from aligntune.rewards.types import RewardFunction, RewardType
from aligntune.rewards.registry import RewardRegistry

class DomainSpecificReward(RewardFunction):
    def __init__(self):
        
    
    def compute(self, text: str, **kwargs) -> float:
        # Your custom logic
        score = 0.0
        if "domain_keyword" in text.lower():
        score += 0.5
        # ... more logic
        return score

# Register
RewardRegistry.register_reward("domain_specific", DomainSpecificReward)

# Use in training
trainer = create_rl_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 algorithm="ppo",
 reward_functions=["domain_specific", "length", "safety"],
 reward_function_weights=[0.5, 0.3, 0.2]
)
```

## Production Pipeline

### Complete Production Setup

```python
from aligntune.core.backend_factory import create_sft_trainer, create_rl_trainer
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def production_pipeline():
 """Complete production training pipeline."""
 
 # Stage 1: SFT
 logger.info("Stage 1: SFT Training")
 sft_trainer = create_sft_trainer(
    model_name="microsoft/DialoGPT-medium",
    dataset_name="tatsu-lab/alpaca",
    backend="trl",
    num_epochs=3,
    batch_size=4,
    output_dir="./output/sft",
    eval_interval=100,
    save_interval=500
 )
 sft_trainer.train()
 sft_path = sft_trainer.save_model()
 logger.info(f"SFT model saved to: {sft_path}")
 
 # Stage 2: DPO
 logger.info("Stage 2: DPO Training")
 dpo_trainer = create_rl_trainer(
 model_name=sft_path,
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dpo",
 backend="trl",
 num_epochs=1,
 batch_size=4,
 output_dir="./output/dpo"
 )
 dpo_trainer.train()
 dpo_path = dpo_trainer.save_model()
 logger.info(f"DPO model saved to: {dpo_path}")
 
 # Stage 3: Evaluation
 logger.info("Stage 3: Evaluation")
 metrics = dpo_trainer.evaluate()
 logger.info(f"Final metrics: {metrics}")
 
 # Stage 4: Push to Hub
 logger.info("Stage 4: Pushing to Hub")
 url = dpo_trainer.push_to_hub(
 repo_id="username/production-model",
 private=False
 )
 logger.info(f"Model available at: {url}")

if __name__ == "__main__":
 production_pipeline()
```

## Custom Domain Training

### Medical Domain Example

```python
trainer = create_sft_trainer(
    model_name="microsoft/DialoGPT-medium",
    dataset_name="medical-dataset",
    backend="trl",
    task_type="instruction_following",
    num_epochs=5,
    batch_size=4,
    learning_rate=1e-5,
    max_seq_length=1024,
    # Custom column mappings
    column_mapping={
    "instruction": "medical_question",
    "output": "medical_answer",
    "input": "patient_context"
    }
)

trainer.train()
```

## Integration Examples

### Weights & Biases Integration

```python
trainer = create_sft_trainer(
    model_name="microsoft/DialoGPT-medium",
    dataset_name="tatsu-lab/alpaca",
    backend="trl",
    loggers=["wandb"], # Enable W&B
    run_name="my-experiment"
)

trainer.train()
```

### TensorBoard Integration

```python
trainer = create_sft_trainer(
    model_name="microsoft/DialoGPT-medium",
    dataset_name="tatsu-lab/alpaca",
    backend="trl",
    loggers=["tensorboard"],
    output_dir="./output/tensorboard"
)

trainer.train()
# View with: tensorboard --logdir ./output/tensorboard
```

## Memory Optimization

### Large Model Training

```python
trainer = create_sft_trainer(
    model_name="meta-llama/Llama-2-7b-hf",
    dataset_name="tatsu-lab/alpaca",
    backend="unsloth",
    # Memory optimizations
    quantization={"load_in_4bit": True},
    peft_enabled=True,
    lora_r=16,
    lora_alpha=32,
    use_gradient_checkpointing=True,
    batch_size=1,
    gradient_accumulation_steps=8,
    activation_offloading=True
)

trainer.train()
```

## Next Steps

- [SFT Examples](sft.md) - Basic SFT examples
- [RL Examples](rl.md) - RL training examples
- [Advanced Topics](../advanced/architecture.md) - Architecture details