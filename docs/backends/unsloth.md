# Unsloth Backend

Complete guide to using the Unsloth backend in AlignTune for faster training.

---

## Overview

The **Unsloth Backend** provides optimized training with significant speed improvements and memory efficiency through CUDA-specific optimizations. It's ideal for GPU-based training when speed is critical.

---

## Key Features

- **Fast**: faster training
- **Memory Efficient**: Reduction in memory
- **GPU Optimized**: CUDA-specific optimizations
- **LoRA/QLoRA**: Automatic PEFT configuration
- **GPU Required**: Needs CUDA-capable GPU
- **Limited**: Some algorithms not supported

---

## When to Use Unsloth Backend

Use Unsloth backend when:

1. **Speed is Critical**
 - Fast iteration needed
 - Large-scale training
 - Time-constrained projects

2. **Memory Constraints**
 - Limited GPU memory
 - Training large models
 - Need memory efficiency

3. **Supported Algorithms**
 - Using DPO, PPO, GRPO, etc.
 - Not using GSPO

4. **GPU Available**
 - CUDA-capable GPU
 - Optimized CUDA setup

---

## Usage

### SFT Training

```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="tatsu-lab/alpaca",
 backend="unsloth",
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
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dpo",
 backend="unsloth",
 num_epochs=1,
 batch_size=1,
 learning_rate=1e-6
)

trainer.train()
```

### PPO Training

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

---

## Algorithm Support

Unsloth backend supports most algorithms:

| Algorithm | Support | Notes |
|-----------|---------|-------|
| SFT | Yes | Full support |
| DPO | Yes | Full support |
| PPO | Yes | Full support |
| GRPO | Yes | Full support |
| GSPO | No | **Not supported** |
| DAPO | Yes | Full support |
| Dr. GRPO | Yes | Full support |

---

## Configuration

### Basic Configuration

```python
trainer = create_sft_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="tatsu-lab/alpaca",
 backend="unsloth",
 num_epochs=3,
 batch_size=4,
 learning_rate=5e-5
)
```

### Advanced Configuration with QLoRA

```python
trainer = create_sft_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="tatsu-lab/alpaca",
 backend="unsloth",
 num_epochs=3,
 batch_size=4,
 learning_rate=5e-5,
 use_peft=True,
 lora_r=8,
 lora_alpha=16,
 lora_dropout=0.05
)
```

---

## Performance

### Training Speed

Unsloth backend provides faster training:
- **Small models (100M-1B)**: faster
- **Medium models (1B-7B)**: faster
- **Large models (7B+)**: faster

### Memory Usage
Unsloth backend uses  less memory overall 
<!-- Unsloth backend uses  less memory:
- **Full fine-tuning**: 60-70% less memory
- **LoRA**: 70-80% less memory
- **QLoRA (4-bit)**: 80-90% less memory -->

---

## Best Practices

### 1. Use Quantized Models

```python
# Use 4-bit quantized models for maximum memory efficiency
model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
```

### 2. Enable LoRA/QLoRA

```python
use_peft=True,
lora_r=8,
lora_alpha=16,
lora_dropout=0.05
```

### 3. Optimize Batch Size

```python
# Start with small batch size, increase if memory allows
batch_size=1 # For large models
batch_size=4 # For smaller models
```

### 4. Use String-Based Selection

```python
# CORRECT
backend="unsloth"

# AVOID
from aligntune.core.backend_factory import BackendType
backend=BackendType.UNSLOTH # May cause interference
```

---

## Troubleshooting

### Common Issues

1. **Unsloth Not Available**: Check CUDA installation and compatibility
2. **CUDA Errors**: Verify PyTorch and CUDA versions match
3. **Memory Issues**: Use quantized models or reduce batch size

See [Troubleshooting Guide](../user-guide/troubleshooting.md) and [Unsloth Compatibility](../unsloth_compatibility.md) for more details.

---

## Comparison with TRL

| Feature | Unsloth | TRL |
|---------|---------|-----|
| Speed | faster | Baseline |
| Memory | Less memory | Baseline |
| Algorithm Support | Most | All |
| CPU Support | No | Yes |
| GPU Required | Yes | No |

See [Backend Comparison](comparison.md) for detailed comparison.

---

## Next Steps

- **[Backend Comparison](comparison.md)** - Compare with TRL
- **[TRL Backend](trl.md)** - Learn about TRL backend
- **[Unsloth Compatibility](../unsloth_compatibility.md)** - Compatibility guide
- **[Backend Selection](../getting-started/backend-selection.md)** - Choose the right backend