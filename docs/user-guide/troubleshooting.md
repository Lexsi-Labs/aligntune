# Troubleshooting

This guide helps you resolve common issues when using AlignTune.

---

## Backend Issues

### Unsloth Not Available

**Error**: `Unsloth not available` or `ImportError: Unsloth backend not available`

**Causes**:
- Unsloth not installed
- CUDA not available
- PyTorch version mismatch
- CUDA version incompatibility

**Solutions**:

1. **Install Unsloth**:
 ```bash
 pip install unsloth
 ```

2. **Check CUDA Availability**:
 ```python
 import torch
 print(torch.cuda.is_available())
 print(torch.version.cuda)
 ```

3. **Use TRL Backend Instead**:
 ```python
 trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 backend="trl" # Use TRL instead
 )
 ```

4. **Run Diagnostics**:
 ```bash
 aligntune diagnose
 ```

### Backend Interference

**Error**: Unsloth patches interfering with TRL backend

**Cause**: Importing `BackendType` enum triggers Unsloth loading

**Solution**: Use string-based backend selection

```python
# CORRECT - String-based selection
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dpo",
 backend="trl" # Use string, not BackendType enum
)
```

```python
# AVOID - Importing BackendType causes Unsloth interference
from aligntune.core.backend_factory import create_rl_trainer, BackendType
# This import triggers Unsloth loading even when using TRL backend
```

---

## CUDA and GPU Issues

### CUDA Out of Memory (OOM)

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:

1. **Reduce Batch Size**:
 ```python
 trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 batch_size=1 # Reduce from default
 )
 ```

2. **Enable Gradient Checkpointing**:
 ```python
 trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 gradient_checkpointing=True
 )
 ```

3. **Use LoRA/QLoRA**:
 ```python
 trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 use_peft=True,
 lora_r=8, # Lower rank = less memory
 lora_alpha=16
 )
 ```

4. **Use 4-bit Quantization** (Unsloth):
 ```python
 trainer = create_sft_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="tatsu-lab/alpaca",
 backend="unsloth"
 )
 ```

5. **Use CPU**:
 ```python
 trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 device_map="cpu"
 )
 ```

### CUDA Symbol Errors

**Error**: `CUDA symbol errors` or `undefined symbol: cublasLtMatmul`

**Cause**: PyTorch and CUDA version mismatch

**Solutions**:

1. **Use TRL Backend**:
 ```python
 backend="trl" # TRL is more compatible
 ```

2. **Reinstall PyTorch with Matching CUDA**:
 ```bash
 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
 ```

3. **Check Compatibility**:
 ```bash
 aligntune diagnose
 ```

---

## Configuration Issues

### Invalid Configuration

**Error**: `ConfigurationError` or validation errors

**Solutions**:

1. **Validate Configuration**:
 ```bash
 aligntune validate config.yaml
 ```

2. **Check Required Parameters**:
 - `model_name`: Must be a valid HuggingFace model ID
 - `dataset_name`: Must be a valid HuggingFace dataset ID
 - `algorithm`: Must be one of: `dpo`, `ppo`, `grpo`, `gspo`, `dapo`, `drgrpo`

3. **Check Parameter Types**:
 ```python
 # CORRECT
 num_epochs=3 # int
 learning_rate=5e-5 # float
 batch_size=4 # int
 
 # INCORRECT
 num_epochs="3" # string
 learning_rate="5e-5" # string
 ```

### Missing Required Parameters

**Error**: `ValueError: Missing required parameter`

**Common Missing Parameters**:

1. **RL Training without Algorithm**:
 ```python
 # MISSING algorithm
 trainer = create_rl_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="Anthropic/hh-rlhf"
 )
 
 # CORRECT
 trainer = create_rl_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dpo"
 )
 ```

2. **PPO without Reward Model**:
 ```python
 # PPO requires reward model
 trainer = create_rl_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="ppo",
 reward_model_name="your-reward-model" # Required for PPO
 )
 ```

---

## Dataset Issues

### Dataset Not Found

**Error**: `DatasetNotFoundError` or `FileNotFoundError`

**Solutions**:

1. **Check Dataset Name**:
 ```python
 # Use correct HuggingFace dataset ID
 dataset_name="tatsu-lab/alpaca" # 
 dataset_name="alpaca" # May not work
 ```

2. **Check Dataset Access**:
 - Some datasets require authentication
 - Use `huggingface-cli login` to authenticate

3. **Use Local Dataset**:
 ```python
 from datasets import load_dataset
 
 dataset = load_dataset("path/to/local/dataset")
 ```

### Dataset Format Issues

**Error**: `KeyError` or missing columns

**Solutions**:

1. **Specify Column Mapping**:
 ```python
 trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 column_mapping={
 "instruction": "prompt",
 "output": "response"
 }
 )
 ```

2. **Check Dataset Structure**:
 ```python
 from datasets import load_dataset
 
 dataset = load_dataset("tatsu-lab/alpaca")
 print(dataset["train"][0]) # Check structure
 ```

---

## Training Issues

### Loss Not Decreasing

**Symptoms**: Loss stays constant or increases

**Solutions**:

1. **Reduce Learning Rate**:
 ```python
 learning_rate=1e-6 # Try lower learning rate
 ```

2. **Increase Warmup Steps**:
 ```python
 warmup_steps=100
 warmup_ratio=0.1
 ```

3. **Check Data Quality**:
 - Verify dataset is correct
 - Check for data preprocessing issues

4. **Use Different Optimizer**:
 ```python
 optimizer="adamw_torch" # or "lion", "adafactor"
 ```

### Training Too Slow

**Solutions**:

1. **Use Unsloth Backend**:
 ```python
 backend="unsloth" # faster
 ```

2. **Increase Batch Size**:
 ```python
 batch_size=8 # If memory allows
 gradient_accumulation_steps=4
 ```

3. **Use Mixed Precision**:
 ```python
 precision="bf16" # or "fp16"
 ```

4. **Enable Flash Attention**:
 ```python
 use_flash_attention_2=True
 ```

### Training Crashes

**Error**: Training process crashes or hangs

**Solutions**:

1. **Check Logs**:
 ```bash
 # Check training logs
 tail -f output/training.log
 ```

2. **Reduce Batch Size**:
 ```python
 batch_size=1 # Minimal batch size
 ```

3. **Disable Gradient Checkpointing** (if causing issues):
 ```python
 gradient_checkpointing=False
 ```

4. **Use TRL Backend** (more stable):
 ```python
 backend="trl"
 ```

---

## Model Issues

### Model Not Found

**Error**: `ModelNotFoundError` or `OSError: Can't load model`

**Solutions**:

1. **Check Model Name**:
 ```python
 # Use correct HuggingFace model ID
 model_name="microsoft/DialoGPT-small" # 
 model_name="DialoGPT-small" # May not work
 ```

2. **Check Model Access**:
 - Some models require authentication
 - Use `huggingface-cli login` to authenticate

3. **Use Local Model**:
 ```python
 model_name="./path/to/local/model"
 ```

### Model Loading Errors

**Error**: `RuntimeError` during model loading

**Solutions**:

1. **Check Model Compatibility**:
 - Verify model architecture is supported
 - Check model size vs. available memory

2. **Use Device Map**:
 ```python
 device_map="auto" # Automatic device placement
 ```

3. **Use Quantization**:
 ```python
 quantization={"load_in_4bit": True} # 4-bit quantization
 ```

---

## Evaluation Issues

### Evaluation Fails

**Error**: `EvaluationError` or evaluation crashes

**Solutions**:

1. **Check Evaluation Dataset**:
 ```python
 # Ensure eval dataset exists
 if trainer.eval_dataset is None:
 print("No evaluation dataset available")
 ```

2. **Reduce Evaluation Samples**:
 ```python
 max_samples_for_quality_metrics=50 # Limit samples
 ```

3. **Skip Quality Metrics**:
 ```python
 compute_rouge=False
 compute_bleu=False
 ```

---

## Memory Issues

### Out of Memory (OOM)

**Solutions**:

1. **Reduce Model Size**:
 - Use smaller models
 - Use quantized models (4-bit, 8-bit)

2. **Optimize Training**:
 ```python
 gradient_checkpointing=True
 activation_offloading=True
 use_peft=True # LoRA reduces memory
 ```

3. **Reduce Sequence Length**:
 ```python
 max_seq_length=256 # Reduce from 512
 ```

4. **Use Gradient Accumulation**:
 ```python
 batch_size=1
 gradient_accumulation_steps=8 # Simulates batch_size=8
 ```

---

## Getting Help

### Diagnostic Commands

```bash
# Run comprehensive diagnostics
aligntune diagnose

# Validate configuration
aligntune validate config.yaml

# Check backend availability
aligntune list-backends-cmd
```

### Useful Resources

- [Unsloth Compatibility Guide](../unsloth_compatibility.md) - Unsloth-specific issues
- [Backend Selection Guide](../getting-started/backend-selection.md) - Backend comparison
- [Configuration Guide](../getting-started/configuration.md) - Configuration options
- [GitHub Issues](https://github.com/Lexsi-Labs/aligntune/issues) - Report bugs

### Reporting Issues

When reporting issues, please include:

1. **Error Message**: Full error traceback
2. **Configuration**: Your configuration file or code
3. **Environment**: Python version, PyTorch version, CUDA version
4. **Diagnostics**: Output of `aligntune diagnose`

---

## Common Error Messages

### `Backend not available`

**Solution**: Use a different backend or install missing dependencies

### `Algorithm not supported`

**Solution**: Check algorithm name and backend compatibility

### `Dataset format error`

**Solution**: Check dataset structure and column mappings

### `Model loading failed`

**Solution**: Verify model name and access permissions

### `CUDA out of memory`

**Solution**: Reduce batch size, enable gradient checkpointing, or use quantization

---

## Prevention Tips

1. **Always Validate Configuration**: Use `aligntune validate` before training
2. **Start Small**: Test with small datasets and models first
3. **Monitor Resources**: Watch GPU memory and CPU usage
4. **Use Auto Backend**: Let AlignTune select the best backend
5. **Check Compatibility**: Run `aligntune diagnose` regularly

---

## Next Steps

- [Backend Selection](../getting-started/backend-selection.md) - Choose the right backend
- [Configuration Guide](../getting-started/configuration.md) - Configure training properly
- [Performance Optimization](../advanced/performance.md) - Optimize training speed and memory