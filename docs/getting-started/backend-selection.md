# Backend Selection

AlignTune supports multiple backends for training. Choose the one that best fits your needs.

## Available Backends

### TRL Backend

- **Reliability**: the reference implementation; runs on CPU or GPU, no extra setup
- **Performance**: standard `transformers`/`peft` training speed
- **Use Case**: production, or any environment without a compatible GPU

### Unsloth Backend

- **Speed**: 2x faster training, ~60% lower memory use, via custom kernels
- **Compatibility**: requires a CUDA-compatible GPU
- **Use Case**: fast iteration, research, large models on limited VRAM

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

AlignTune can select the backend for you:

```python
from aligntune.core.backend_factory import create_sft_trainer

# Auto-select backend (recommended)
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 backend="auto" # Automatically chooses best backend
)
```

When `backend="auto"`, AlignTune:

1. Checks if Unsloth is available
2. Checks if the model is compatible with Unsloth
3. Falls back to TRL if Unsloth is unavailable or incompatible
4. Logs the selected backend

## Selecting a Backend Explicitly

```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 backend="trl", # or "unsloth" (use unsloth/... model checkpoints with the Unsloth backend)
 num_epochs=3,
 batch_size=4,
)
```

Unsloth requires an Unsloth-format checkpoint (e.g. `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`); TRL works with any standard HF checkpoint. No other config changes are needed to switch backends.

## Checking Backend Status

```python
from aligntune.core.backend_factory import get_backend_status

status = get_backend_status()
print(f"TRL available: {status['trl_available']}")
print(f"Unsloth available: {status['unsloth_available']}")
```

## When to Use Each Backend

### Use TRL Backend When:

- You're in production and need the most reliable path
- You don't have a GPU, or don't have Unsloth set up
- You need the GSPO algorithm (Unsloth doesn't support it)

### Use Unsloth Backend When:

- You want faster training and lower VRAM use
- You have a CUDA-compatible GPU
- You're doing research or fast iteration
- You're fine-tuning a large model on limited hardware

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
