# Known Issues and Troubleshooting

This document tracks known issues, common problems, and their solutions.

## Reporting Issues

If you encounter an issue not listed here, please report it:

- **GitHub Issues**: [Open an issue](https://github.com/Lexsi-Labs/aligntune/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Lexsi-Labs/aligntune/discussions)

## Common Issues

### Import Errors

**Problem**: `ImportError: cannot import name 'create_sft_trainer'`

**Solution**: Ensure you're using the correct import path:
```python
from aligntune.core.backend_factory import create_sft_trainer
```

### Backend Availability

**Problem**: Unsloth backend not available

**Solution**: 
- Check GPU compatibility
- Install Unsloth: `pip install unsloth`
- Use TRL backend as fallback: `backend="trl"`

### Memory Issues

**Problem**: Out of memory errors during training

**Solution**:
- Reduce `batch_size`
- Enable PEFT/LoRA: `peft_enabled=True`
- Use gradient checkpointing
- Reduce `max_seq_length`
- Use quantization: `quantization={"load_in_4bit": True}`

## Backend-Specific Issues

### TRL Backend

- See [TRL Backend Documentation](backends/trl.md) for TRL-specific issues

### Unsloth Backend

- See [Unsloth Compatibility Guide](unsloth_compatibility.md) for detailed troubleshooting

## Getting Help

For more detailed troubleshooting, see:
- [Troubleshooting Guide](user-guide/troubleshooting.md)
- [Unsloth Compatibility](unsloth_compatibility.md)
- [Backend Support Matrix](compatibility/backend-matrix.md)