# Distributed Training

Run AlignTune across multiple GPUs and nodes.

## Overview

AlignTune supports distributed training via the underlying backends (TRL, Unsloth) and `accelerate`. Configuration is passed through the unified config so you can use the same YAML/code for single-GPU and multi-GPU runs.

## Using Accelerate

1. Configure with `accelerate config` (multi-GPU, multi-node, etc.).
2. Launch with `accelerate launch`:

   ```bash
   accelerate launch --num_processes 4 aligntune train config.yaml
   ```

3. Set distributed options in your config under `distributed` (e.g., rank, world size, backend) when supported by the loader.

## Backend Behavior

- **TRL**: Uses Hugging Face Trainer / Accelerate; supports DDP, DeepSpeed, FSDP when configured.
- **Unsloth**: Follow Unsloth’s documentation for multi-GPU; use the same `accelerate launch` pattern where applicable.

## Next Steps

- [Architecture](architecture.md) - System design
- [Custom Backends](custom-backends.md) - Adding new backends
- [Performance](performance.md) - Optimization
