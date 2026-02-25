# Backend Selection

AlignTune supports multiple backends for training. Choose the one that best fits your needs.

## Available Backends

### TRL Backend

- **Reliability**: Battle-tested, stable
- **Compatibility**: Works everywhere
- **Performance**: Standard speed
- **Use Case**: Production, reliability-focused

### Unsloth Backend

- **Speed**: faster training
- **Memory**: Optimized memory usage
- **Compatibility**: Requires GPU, specific setup
- **Use Case**: Fast iteration, research

## Backend Support Matrix

| Algorithm | TRL Backend | Unsloth Backend |
|-----------|-------------|-----------------|
| **SFT** | Yes | Yes |
| **DPO** | Yes | Yes |
| **PPO** | Yes | Yes |
| **GRPO** | Yes | Yes |
| **GSPO** | Yes | No |
| **DAPO** | Yes | Yes |
| **Dr. GRPO** | Yes | Yes |

## Automatic Backend Selection

AlignTune can automatically select the best backend:

```python
from aligntune.core.backend_factory import create_sft_trainer

# Auto-select backend (recommended)
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 backend="auto" # Automatically chooses best backend
)
```

## Manual Backend Selection

### TRL Backend

```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 backend="trl" # Explicitly use TRL
)
```

### Unsloth Backend

```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="tatsu-lab/alpaca",
 backend="unsloth" # Explicitly use Unsloth
)
```

## Backend Selection Logic

When using `backend="auto"`, AlignTune:

1. Checks if Unsloth is available
2. Checks if model is compatible with Unsloth
3. Falls back to TRL if Unsloth unavailable or incompatible
4. Logs the selected backend

## Checking Backend Status

```python
from aligntune.core.backend_factory import get_backend_status

status = get_backend_status()
print(f"TRL available: {status['trl']['available']}")
print(f"Unsloth available: {status['unsloth']['available']}")
```

## When to Use Each Backend

### Use TRL Backend When:

- You need maximum reliability
- You're in production
- You don't have GPU or Unsloth setup
- You need GSPO algorithm
- You want battle-tested code

### Use Unsloth Backend When:

- You need faster training (faster training)
- You have GPU available
- You're doing research/experimentation
- You want memory optimization
- You're fine-tuning large models

## Backend-Specific Configuration

### TRL Backend

```python
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 backend="trl",
 # Standard configuration works
 num_epochs=3,
 batch_size=4
)
```

### Unsloth Backend

```python
trainer = create_sft_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="tatsu-lab/alpaca",
 backend="unsloth",
 # Unsloth-specific optimizations enabled
 num_epochs=3,
 batch_size=4
)
```

## Troubleshooting

### Unsloth Not Available

If Unsloth is not available, AlignTune automatically falls back to TRL:

```python
# This will use TRL if Unsloth unavailable
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 backend="unsloth" # Falls back to TRL if Unsloth unavailable
)
```

### Force TRL Backend

```python
# Explicitly force TRL
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 backend="trl" # Always uses TRL
)
```

## Next Steps

- [SFT Guide](../user-guide/sft.md) - SFT training details
- [RL Guide](../user-guide/rl.md) - RL training details
- [Unsloth Compatibility](../unsloth_compatibility.md) - Unsloth setup and troubleshooting