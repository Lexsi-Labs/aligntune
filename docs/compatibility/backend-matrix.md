# Backend Support Matrix

Complete compatibility matrix for AlignTune backends and algorithms.

## Algorithm Support

| Algorithm | TRL Backend | Unsloth Backend | Notes |
|-----------|-------------|-----------------|-------|
| **SFT** | Yes | Yes | Both backends fully supported |
| **DPO** | Yes | Yes | Both backends fully supported |
| **PPO** | Yes | Yes | Both backends fully supported |
| **GRPO** | Yes | Yes | Both backends fully supported |
| **GSPO** | Yes | No | TRL only, Unsloth not supported |
| **DAPO** | Yes | Yes | Both backends fully supported |
| **Dr. GRPO** | Yes | Yes | Both backends fully supported |

## Backend Comparison

### TRL Backend

**Advantages:**
- Maximum compatibility
- Supports all algorithms (including GSPO)
- Battle-tested reliability
- Works on CPU and GPU
- No special requirements

**Use When:**
- Need GSPO support
- Maximum compatibility required
- Working with standard models
- CPU-only environments

### Unsloth Backend

**Advantages:**
- faster training
- Memory efficient
- Optimized kernels
- Automatic optimizations

**Limitations:**
- GSPO not supported
- Requires GPU
- CUDA compatibility needed

**Use When:**
- Need faster training
- Working with large models
- Memory constraints
- GPU available

## Model Compatibility

### TRL Backend

| Model Type | Support | Notes |
|------------|---------|-------|
| Causal LM | Yes | Full support |
| Sequence-to-Sequence | Yes | Full support |
| Encoder-Decoder | Yes | Full support |
| Classification | Yes | Full support |

### Unsloth Backend

| Model Type | Support | Notes |
|------------|---------|-------|
| Causal LM | Yes | Full support |
| Sequence-to-Sequence | Limited | Limited support |
| Encoder-Decoder | Limited | Limited support |
| Classification | No | Not recommended |

## Task Type Support

### SFT Tasks

| Task Type | TRL | Unsloth | Notes |
|-----------|-----|---------|-------|
| Instruction Following | Yes | Yes | Both supported |
| Supervised Fine-Tuning | Yes | Yes | Both supported |
| Text Classification | Yes | Limited | TRL recommended |
| Token Classification | Yes | Limited | TRL recommended |
| Text Generation | Yes | Yes | Both supported |
| Chat Completion | Yes | Yes | Both supported |

## Requirements

### TRL Backend

- Python 3.8+
- PyTorch 1.13+
- Transformers 4.35+
- TRL 0.7+
- CPU or GPU

### Unsloth Backend

- Python 3.8+
- PyTorch 2.0+ (CUDA)
- CUDA 11.8+ or 12.1+
- GPU required
- Unsloth package

## Performance Comparison

### Training Speed

| Model Size | TRL | Unsloth | Speedup |
|------------|-----|---------|---------|
| Small (<1B) | Baseline | 2-3x | 2-3x |
| Medium (1-7B) | Baseline | 3-4x | 3-4x |
| Large (7B+) | Baseline | Faster | Faster |

### Memory Usage

| Configuration | TRL | Unsloth | Reduction |
|---------------|-----|---------|-----------|
| Full Precision | Baseline | -20% | 20% |
| LoRA | Baseline | -30% | 30% |
| QLoRA (4-bit) | Baseline | -50% | 50% |

## Migration Guide

### From TRL to Unsloth

```python
# TRL
trainer = create_sft_trainer(
 model_name="model",
 backend="trl"
)

# Unsloth (if compatible)
trainer = create_sft_trainer(
 model_name="unsloth/model-bnb-4bit", # Use Unsloth model
 backend="unsloth"
)
```

### From Unsloth to TRL

```python
# Unsloth
trainer = create_sft_trainer(
 model_name="unsloth/model",
 backend="unsloth"
)

# TRL (always works)
trainer = create_sft_trainer(
 model_name="model", # Standard model
 backend="trl"
)
```

## Troubleshooting

### Unsloth Not Available

```python
# Automatic fallback to TRL
trainer = create_sft_trainer(
 model_name="model",
 backend="auto" # Falls back to TRL if Unsloth unavailable
)
```

### GSPO with Unsloth

```python
# This will fail
trainer = create_rl_trainer(
 algorithm="gspo",
 backend="unsloth" # Error: GSPO not supported
)

# Use TRL instead
trainer = create_rl_trainer(
 algorithm="gspo",
 backend="trl" # Works
)
```

## Next Steps

- [Backend Selection](../getting-started/backend-selection.md) - Choosing backends
- [Unsloth Compatibility](../unsloth_compatibility.md) - Unsloth setup
- [Performance](../advanced/performance.md) - Performance optimization