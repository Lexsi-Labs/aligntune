# Unsloth Compatibility Guide

This document provides comprehensive information about Unsloth compatibility with AlignTune, including supported versions, known issues, and troubleshooting steps.

## Overview

Unsloth provides faster training with memory optimizations, but requires specific environment configurations. AlignTune includes robust detection and fallback mechanisms to ensure training works even when Unsloth is unavailable.

## Supported Versions

### Recommended Combinations

| PyTorch | CUDA | Unsloth | Status |
|---------|------|---------|--------|
| 2.0.0-2.7.0 | 11.8, 12.1, 12.4 | latest | Optimal |
| 2.8.0+ | 12.8+ | latest | Known Issues |

### Python Requirements
- Python 3.8-3.12
- CUDA-capable GPU (for GPU training)

## Known Compatibility Issues

### 1. CUDA Symbol Errors
**Error**: `undefined symbol: _ZN3c104cuda9SetDeviceEa`

**Cause**: Version incompatibility between PyTorch 2.8.0+ and Unsloth's CUDA extensions.

**Solutions**:
- Use PyTorch 2.7.0 or earlier
- Update Unsloth: `pip install --upgrade unsloth`
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
- Update all dependencies: `pip install --upgrade torch unsloth`
- Use TRL backends: `--backend trl`

## Environment Setup

### Optimal Setup
```bash
# Install PyTorch with CUDA support
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Unsloth
pip install unsloth

# Install AlignTune
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
aligntune diagnose

# Check basic info with verbose output
aligntune info --verbose
```

### 2. Common Issues and Solutions

#### Unsloth Not Detected
```bash
# Check if Unsloth is installed
python -c "import unsloth; print('Unsloth available')"

# If error occurs, check CUDA compatibility
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
```

#### CUDA Symbol Errors
- **Solution 1**: Downgrade PyTorch to 2.7.0
- **Solution 2**: Use TRL backends: `--backend trl`
- **Solution 3**: Update Unsloth: `pip install --upgrade unsloth`

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
aligntune diagnose
```

### Backend Selection
```bash
# List available backends
aligntune list-backends

# Validate model compatibility
aligntune validate --model microsoft/DialoGPT-medium --backend auto
```

## Error Messages and Solutions

### "Unsloth not available: cuda_symbol_error"
- **Cause**: CUDA version incompatibility
- **Solution**: Use TRL backends or fix CUDA setup

### "Unsloth not available: flash_attention_error"
- **Cause**: Flash Attention compatibility issues
- **Solution**: Unsloth will auto-fallback to Xformers

### "Unsloth not available: missing_dependency"
- **Cause**: Unsloth not installed
- **Solution**: `pip install unsloth` or use TRL backends

## Best Practices

1. **Always test with `aligntune diagnose`** before training
2. **Use `--backend auto`** to let AlignTune choose the best available backend
3. **Check logs** for detailed error information
4. **Fallback to TRL** if Unsloth has issues
5. **Update dependencies** regularly for best compatibility

## Getting Help

If you encounter issues not covered in this guide:

1. Run `aligntune diagnose` and share the output
2. Check the logs for detailed error messages
3. Try using TRL backends as a fallback
4. Report issues with full diagnostic output

## Version History

- **v0.2.0**: Enhanced error detection and diagnostics
- **v0.1.0**: Initial Unsloth support with basic detection

---

For more information, see the [main README](https://github.com/Lexsi-Labs/aligntune/blob/main/README.md) or run `aligntune info --help`.