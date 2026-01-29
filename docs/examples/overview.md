# Examples Overview

This section contains comprehensive examples for using AlignTune.

## Quick Links

- [SFT Examples](sft.md) - Supervised Fine-Tuning examples
- [RL Examples](rl.md) - Reinforcement Learning examples
- [Advanced Examples](advanced.md) - Advanced use cases

## Example Categories

### Basic Examples

- Simple SFT training
- Basic DPO training
- Model evaluation
- Model saving and loading

### Intermediate Examples

- Custom reward functions
- Multi-dataset training
- Custom evaluation metrics
- Checkpoint management

### Advanced Examples

- Distributed training
- Custom backends
- Reward model training
- Performance optimization

## Running Examples

All examples can be run directly:

```bash
# Run SFT example
python examples/sft_customer_support_trl/train_sft_direct_api.py

# Run DPO example
python examples/dpo_intel_orca_trl/train_dpo_intel_orca_trl.py

```

## Example Structure

Examples are organized by:

- **Task Type**: SFT, DPO, PPO, GRPO, GSPO
- **Backend**: TRL, Unsloth
- **Complexity**: Basic, Intermediate, Advanced

## Next Steps

- [SFT Examples](sft.md) - Start with SFT
- [RL Examples](rl.md) - Learn RL training
- [Advanced Examples](advanced.md) - Explore advanced features