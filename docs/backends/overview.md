# Backends Overview

AlignTune provides two backend implementations for training: **TRL** (reliable, battle-tested) and **Unsloth** (faster, memory efficient).

---

## Backend Comparison

| Feature | TRL Backend | Unsloth Backend |
|---------|-------------|-----------------|
| **Speed** | Standard | Faster Faster |
| **Memory** | Standard | Optimized |
| **Reliability** | Battle-tested | Production-ready |
| **GPU Required** | Works on CPU | CUDA required |
| **Algorithm Support** | All algorithms | Most algorithms |
| **Model Compatibility** | All HuggingFace models | Optimized models |
| **Setup Complexity** | Low | Medium |

---

## TRL Backend

### Overview

The **TRL** (Transformers Reinforcement Learning) backend is the standard, battle-tested implementation based on HuggingFace's TRL library.

### Key Features

- **Reliable**: Extensively tested, production-ready
- **Complete**: Supports all algorithms (SFT, DPO, PPO, GRPO, GSPO, etc.)
- **Compatible**: Works with all HuggingFace models
- **CPU Support**: Can run on CPU (slower)
- **Stable**: No compatibility issues

### When to Use TRL

- You need maximum reliability
- You're using algorithms not supported by Unsloth (GSPO)
- You're on CPU or have CUDA compatibility issues
- You want the most stable training experience

### Example

```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="tatsu-lab/alpaca",
 backend="trl", # Explicit TRL backend
 num_epochs=3
)
```

---

## Unsloth Backend

### Overview

The **Unsloth** backend provides optimized training with significant speed improvements and memory efficiency through CUDA-specific optimizations.

### Key Features

- **Fast**: faster training
- **Memory Efficient**: Memory reduction
- **GPU Optimized**: CUDA-specific optimizations
- **LoRA/QLoRA**: Automatic PEFT configuration
- **GPU Required**: Needs CUDA-capable GPU
- **Limited**: Some algorithms not supported

### When to Use Unsloth

- You have a CUDA-capable GPU
- You want maximum training speed
- You're using supported algorithms (DPO, PPO, GRPO, etc.)
- You need memory-efficient training

### Example

```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="tatsu-lab/alpaca",
 backend="unsloth", # Unsloth backend
 num_epochs=3
)
```

---

## Algorithm Support Matrix

| Algorithm | TRL Backend | Unsloth Backend |
|-----------|-------------|-----------------|
| **SFT** | Yes | Yes |
| **DPO** | Yes | Yes |
| **PPO** | Yes | Yes |
| **GRPO** | Yes | Yes |
| **GSPO** | Yes | No |
| **DAPO** | Yes | Yes |
| **Dr. GRPO** | Yes | Yes |

---

## Backend Selection

### Automatic Selection

AlignTune can automatically select the best backend:

```python
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 backend="auto" # Automatic selection
)
```

**Selection Logic**:
1. Checks if Unsloth is available and compatible
2. Falls back to TRL if Unsloth unavailable
3. Considers algorithm support

### Explicit Selection

You can explicitly choose a backend:

```python
# TRL backend
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 backend="trl"
)

# Unsloth backend
trainer = create_sft_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="tatsu-lab/alpaca",
 backend="unsloth"
)
```

---

## Performance Comparison

### Training Speed

| Model Size | TRL Backend | Unsloth Backend | Speedup |
|------------|-------------|-----------------|---------|
| Small (100M-1B) | Baseline | faster | 2-3x |
| Medium (1B-7B) | Baseline | faster | 3-4x |
| Large (7B+) | Baseline | faster | Faster |

### Memory Usage

| Configuration | TRL Backend | Unsloth Backend | Reduction |
|---------------|-------------|-----------------|-----------|
| Full Fine-tuning | Baseline | 60-70% less | 60-70% |
| LoRA | Baseline | 70-80% less | 70-80% |
| QLoRA (4-bit) | Baseline | 80-90% less | 80-90% |

---

## Best Practices

### 1. Use String-Based Selection

Always use string-based backend selection to prevent interference:

```python
# CORRECT
backend="trl" # or "unsloth"

# AVOID
from aligntune.core.backend_factory import BackendType
backend=BackendType.TRL # May cause Unsloth interference
```

### 2. Check Backend Availability

```bash
# Run diagnostics
aligntune diagnose

# List available backends
aligntune list-backends-cmd
```

### 3. Handle Backend Errors

```python
try:
 trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 backend="unsloth"
 )
except ImportError as e:
 # Fallback to TRL
 print(f"Unsloth not available: {e}")
 trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 backend="trl"
 )
```

---

## Migration Guide

### From TRL to Unsloth

1. **Check Compatibility**:
 ```bash
 aligntune diagnose
 ```

2. **Update Model Name**:
 ```python
 # TRL
 model_name="microsoft/DialoGPT-small"
 
 # Unsloth (if available)
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
 ```

3. **Update Backend**:
 ```python
 backend="unsloth"
 ```

4. **Test Training**:
 ```python
 # Start with small dataset
 max_samples=100
 ```

---

## Troubleshooting

### Unsloth Not Available

See [Troubleshooting Guide](../user-guide/troubleshooting.md#backend-issues) for solutions.

### Backend Interference

See [Troubleshooting Guide](../user-guide/troubleshooting.md#backend-interference) for solutions.

---

## Next Steps

- **[TRL Backend](trl.md)** - Detailed TRL backend guide
- **[Unsloth Backend](unsloth.md)** - Detailed Unsloth backend guide
- **[Backend Comparison](comparison.md)** - Side-by-side comparison
- **[Backend Selection](../getting-started/backend-selection.md)** - Choose the right backend