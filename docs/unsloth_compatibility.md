# Unsloth Compatibility Guide

This document provides comprehensive information about Unsloth compatibility with AlignTune, including supported versions, known issues, and troubleshooting steps.

## Overview

Unsloth provides faster training with memory optimizations, but requires specific environment configurations. If Unsloth isn't available or isn't compatible with the requested model, AlignTune detects that and falls back to the TRL backend instead of failing the run.

## Supported Versions

### Recommended Combinations

| PyTorch | CUDA | Unsloth | Status |
|---------|------|---------|--------|
| 2.0.0-2.7.0 | 11.8, 12.1, 12.4 | latest | Optimal |
| 2.8.0+ | 12.8+ | latest | Known Issues |

### Python Requirements
- Python 3.11+ (matching `pyproject.toml`'s `requires-python`)
- CUDA-capable GPU (for GPU training)

## Currently Pinned Versions

`unsloth` and `unsloth_zoo` (`2026.7.2`) are **vendored inside the AlignTune
package** (`third_party/unsloth/`, `third_party/unsloth_zoo/`) and built into the
same wheel - `pip install aligntune` (or `pip install -e .`) is all you need, and
you should **not** `pip install unsloth` separately.

They run against `trl==1.7.1` (AlignTune's pin). Upstream Unsloth's own package
metadata still declares `trl<=0.24.0` as a ceiling, which is why it is vendored
rather than resolved by pip - in practice `2026.7.2` works correctly against
`trl==1.7.1`: verified for model loading, LoRA patching, and DPO/ORPO/GRPO
(+ all GRPO-family variants)/PPO/Online-DPO/SDFT training (see
[Algorithm-Specific Findings](#algorithm-specific-findings-trl-171-unsloth-202672)
below for a narrower ORPO/Unsloth issue found and fixed).

**Note on SDFT**: if `privileged_context` ends up empty for your dataset (training completes at `global_step=0` with no error), your dataset likely has no context-like column for the automatic alias scan to match. Set `privileged_context_column` explicitly to the column that holds the context.

## Algorithm-Specific Findings (trl 1.7.1 + unsloth 2026.7.2)

These were found and fixed via end-to-end training runs, not code review alone. Documented here since they're specific to this version combination and could resurface if either package is upgraded again.

| Algorithm | Backend | Issue | Status |
|---|---|---|---|
| ORPO | Unsloth | Unsloth silently truncates a forward pass's returned `logits` to its configured `max_seq_length`, but `ORPOTrainer.concatenated_forward` builds `labels` from the untruncated input, mismatches once a concatenated chosen+rejected batch exceeds `max_seq_length` (more likely for ORPO/CPO than single-sequence trainers) | **Fixed**: `labels` realigned to Unsloth's actual returned length |

## Known Compatibility Issues

### 1. CUDA Symbol Errors
**Error**: `undefined symbol: _ZN3c104cuda9SetDeviceEa`

**Cause**: Version incompatibility between PyTorch 2.8.0+ and Unsloth's CUDA extensions.

**Solutions**:
- Use PyTorch 2.7.0 or earlier
- Use TRL backends instead: `--backend trl`

### 2. Flash Attention Issues
**Error**: Flash Attention 2 installation broken

**Cause**: CUDA version incompatibility with Flash Attention.

**Solutions**:
- Unsloth automatically falls back to Xformers
- Update Flash Attention: `pip install --upgrade flash-attn`
- Use TRL backends: `--backend trl`

### 3. Version Mismatches
**Error**: Version incompatibility between dependencies

**Solutions**:
- Check compatibility matrix above
- Align PyTorch with your CUDA version (`unsloth` / `unsloth_zoo` are pinned and vendored — don't `pip install` them separately)
- Use TRL backends: `--backend trl`

## Environment Setup

### Optimal Setup
```bash
# Install PyTorch with CUDA support
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install AlignTune (bundles the pinned unsloth + unsloth_zoo)
pip install -e .
```

### Alternative Setup (if CUDA issues persist)
```bash
# Use TRL backends instead
pip install torch transformers trl
pip install -e .
```

## Troubleshooting

### 1. Check Environment
```bash
# Run comprehensive diagnostics
aligntune diagnose system

# Check basic info with verbose output
aligntune info --verbose
```

### 2. Common Issues and Solutions

#### Unsloth Not Detected

Unsloth is bundled with AlignTune, so this is almost always a GPU/CUDA problem at
import time rather than a missing package.

```bash
# Check the vendored Unsloth imports on this machine
python -c "import unsloth; print('Unsloth available')"

# If it errors, check CUDA compatibility
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
```

#### CUDA Symbol Errors
- **Solution 1**: Downgrade PyTorch to 2.7.0
- **Solution 2**: Use TRL backends: `--backend trl`

#### Flash Attention Issues
- Unsloth automatically falls back to Xformers
- No action required, but performance may be slightly reduced

### 3. Fallback to TRL Backends

If Unsloth is not available, AlignTune automatically falls back to TRL backends:

```bash
# Explicitly use TRL backends
aligntune train --model microsoft/DialoGPT-medium --dataset tatsu-lab/alpaca --backend trl

# Or let AlignTune auto-select
aligntune train --model microsoft/DialoGPT-medium --dataset tatsu-lab/alpaca --backend auto
```

## Performance Comparison

| Backend | Speed | Memory | Compatibility |
|---------|-------|--------|---------------|
| Unsloth | faster | 80% less | Requires specific setup |
| TRL | Standard | Standard | Universal |
| Legacy | Standard | Standard | Universal |

## Diagnostic Commands

### Basic Information
```bash
# Show system status
aligntune info

# Show detailed diagnostics
aligntune info --verbose
```

### Comprehensive Diagnostics
```bash
# Run full environment check
aligntune diagnose system
```

### Backend Selection
```bash
# List available backends
aligntune list-backends-cmd

# Validate model compatibility
aligntune validate model microsoft/DialoGPT-medium
```

## Error Messages and Solutions

### "Unsloth not available: cuda_symbol_error"
- **Cause**: CUDA version incompatibility
- **Solution**: Use TRL backends or fix CUDA setup

### "Unsloth not available: flash_attention_error"
- **Cause**: Flash Attention compatibility issues
- **Solution**: Unsloth will auto-fallback to Xformers

### "Unsloth not available: missing_dependency"
- **Cause**: A dependency the vendored Unsloth needs (e.g. a CUDA-enabled `torch`) failed to import
- **Solution**: Fix the CUDA/PyTorch install, or use TRL backends

## Best Practices

1. **Always test with `aligntune diagnose system`** before training
2. **Use `--backend auto`** to let AlignTune choose the best available backend
3. **Check logs** for detailed error information
4. **Fallback to TRL** if Unsloth has issues
5. **Update dependencies** regularly for best compatibility

## Getting Help

If you encounter issues not covered in this guide:

1. Run `aligntune diagnose system` and share the output
2. Check the logs for detailed error messages
3. Try using TRL backends as a fallback
4. Report issues with full diagnostic output

---

For more information, see the [main README](https://github.com/Lexsi-Labs/aligntune/blob/main/README.md) or run `aligntune info --help`.