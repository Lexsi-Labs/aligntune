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

## Algorithm-Specific Known Issues

**Problem**: `ValueError: No training dataset loaded` when using a HuggingFace slice-notation split (e.g. `split="train[:16]"`)

**Status**: Not a bug, use `split="train"` with `max_samples=N` instead. **Cause**: `DataManager`'s split-name matching does exact string comparison and doesn't recognize the sliced form as its base split.

**Problem**: RAFT's citation loss has no effect, and `create_raft_trainer()` can't be reached through `BackendFactory`

**Status**: Known limitation, unresolved. **Cause**: `use_citation_loss=True` only enables citation tracking, the citation loss term itself is a documented placeholder, not numerically implemented (it would require generation during training). Separately, RAFT isn't wired into `BackendFactory` at all; it's only reachable via the standalone `create_raft_trainer()` function, unlike every other algorithm. See [RAFT parameters](PARAMETERS.md#raft-retrieval-augmented-fine-tuning).

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