# TRL Backend

Complete guide to using the TRL (Transformers Reinforcement Learning) backend in AlignTune.

---

## Overview

The **TRL Backend** is the standard, battle-tested implementation based on HuggingFace's TRL library. It provides reliable, production-ready training for all supported algorithms.

---

## Key Features

- **Reliable**: Extensively tested, production-ready
- **Complete**: Supports all algorithms (SFT, DPO, PPO, GRPO, GSPO, etc.)
- **Compatible**: Works with all HuggingFace models
- **CPU Support**: Can run on CPU (slower)
- **Stable**: No compatibility issues

---

## When to Use TRL Backend

Use TRL backend when:

1. **Maximum Reliability Needed**
 - Production deployments
 - Critical applications
 - Long-running training jobs

2. **Algorithm Requirements**
 - Using GSPO (Unsloth doesn't support)
 - Need all algorithm support

3. **Hardware Constraints**
 - CPU-only environments
 - CUDA compatibility issues
 - Limited GPU memory

4. **Simplicity**
 - Minimal setup required
 - Standard HuggingFace workflow

---

## Usage

### SFT Training

```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 backend="trl",
 num_epochs=3,
 batch_size=4,
 learning_rate=5e-5
)

trainer.train()
```

### DPO Training

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dpo",
 backend="trl",
 num_epochs=1,
 batch_size=1,
 learning_rate=1e-6
)

trainer.train()
```

### PPO Training

```python
trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="ppo",
 backend="trl",
 reward_model_name="your-reward-model",
 num_epochs=1,
 batch_size=1,
 learning_rate=1e-6
)

trainer.train()
```

---

## Algorithm Support

TRL backend supports all algorithms:

| Algorithm | Support | Notes |
|-----------|---------|-------|
| SFT | | Full support |
| DPO | | Full support |
| PPO | | Full support |
| GRPO | | Full support |
| GSPO | | **Only TRL supports GSPO** |
| DAPO | | Full support |
| Dr. GRPO | | Full support |

---

## Configuration

### Basic Configuration

```python
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 backend="trl",
 num_epochs=3,
 batch_size=4,
 learning_rate=5e-5,
 max_seq_length=512
)
```

### Advanced Configuration

```python
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 backend="trl",
 num_epochs=3,
 batch_size=4,
 learning_rate=5e-5,
 gradient_checkpointing=True,
 use_peft=True,
 lora_r=8,
 lora_alpha=16,
 lora_dropout=0.05,
 warmup_steps=100,
 weight_decay=0.01
)
```

---

## Performance

### Training Speed

TRL backend provides standard training speed:
- Baseline performance
- No special optimizations
- Reliable and consistent

### Memory Usage

TRL backend uses standard memory:
- Full fine-tuning: Baseline memory usage
- LoRA: Reduced memory with PEFT
- Gradient checkpointing: Further memory reduction

---

## Best Practices

### 1. Use String-Based Selection

```python
# CORRECT
backend="trl"

# AVOID
from aligntune.core.backend_factory import BackendType
backend=BackendType.TRL # May cause Unsloth interference
```

### 2. Enable Gradient Checkpointing

```python
gradient_checkpointing=True # Reduces memory usage
```

### 3. Use LoRA for Large Models

```python
use_peft=True,
lora_r=8,
lora_alpha=16
```

### 4. Monitor Training

```python
logging_steps=10, # Log every 10 steps
save_steps=500, # Save checkpoint every 500 steps
```

---

## Troubleshooting

### Common Issues

1. **Slow Training**: Normal for TRL backend, consider Unsloth for speed
2. **Memory Issues**: Enable gradient checkpointing or use LoRA
3. **CUDA Errors**: TRL is more compatible, check PyTorch version

See [Troubleshooting Guide](../user-guide/troubleshooting.md) for more details.

---

## Comparison with Unsloth

| Feature | TRL | Unsloth |
|---------|-----|---------|
| Speed | Baseline | faster |
| Memory | Baseline | 60-80% less |
| Algorithm Support | All | Most |
| CPU Support | Yes | No |
| Reliability | Battle-tested | Production-ready |

See [Backend Comparison](comparison.md) for detailed comparison.

---

## Next Steps

- **[Backend Comparison](comparison.md)** - Compare with Unsloth
- **[Unsloth Backend](unsloth.md)** - Learn about Unsloth backend
- **[Backend Selection](../getting-started/backend-selection.md)** - Choose the right backend