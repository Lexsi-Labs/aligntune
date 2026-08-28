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
python examples/sft_trl_1.py

# Run DPO example
python examples/trl_dpo_1.py
```

## Example Scripts (`examples/`)

The `examples/` directory holds three kinds of scripts:

1. **Standalone feature examples**, demonstrate one specific capability end-to-end and aren't tied to a Colab notebook:

    | Script | Demonstrates |
    |---|---|
    | `curriculum_rl_example.py` | Curriculum learning during RLHF training |
    | `alignment_audit_example.py` | `AlignmentAuditor` / `AlignmentAuditCallback` for tracking alignment metrics during training |
    | `reward_tracking_example.py` | Per-component reward tracking and visualization |
    | `load_raw_files_example.py` | Raw file loaders (loading local text/JSON/CSV data instead of a Hub dataset) |

2. **Colab notebook sources**, scripts like `sft_trl_1.py`, `trl_dpo_1.py`, `trl_grpo_1.py`, `unsloth_grpo_1.py`, `retail_banking_sft.py`, `wealth_management_sft.py`, `wealth_dpo_training_evaluation_full.py`, `dapo_trl__code.py`, `drgrpo_unsloth.py`, `grpo_code_gen_trl_mbpp.py`, `gspo_generic_demo_unsloth.py`, `txgemma_trialbench_sft_trl_backend_eval.py`, and the `unsloth1*`/`(1).py`-suffixed files are the exported source of the notebooks already linked from the [Demo Notebooks](../notebooks/demo.md) table, open the Colab link there rather than running these directly, since they include Colab-only setup cells (`!git clone`, `!pip install`).

3. **`custom_tasks/`**, custom `lm-eval` task definitions (YAML + helper `utils.py`) for domain-specific evaluation, e.g. `bitext_2/`, `bitext_3/`, `bitext_insurance/`. See each subfolder's own `README.md`.

## Example Structure

Examples are organized by:

- **Task Type**: SFT, DPO, PPO, GRPO, GSPO
- **Backend**: TRL, Unsloth
- **Complexity**: Basic, Intermediate, Advanced

## Next Steps

- [SFT Examples](sft.md) - Start with SFT
- [RL Examples](rl.md) - Learn RL training
- [Advanced Examples](advanced.md) - Explore advanced features