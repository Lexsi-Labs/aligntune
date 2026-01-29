# CLI Overview

AlignTune provides a comprehensive command-line interface for training and managing models without writing Python code.

## Quick Start

```bash
# Show system information
aligntune info

# Train a model
aligntune train --model "microsoft/DialoGPT-small" --dataset "tatsu-lab/alpaca" 


```

## Main Commands

### `aligntune info`

Display system information, versions, and compatibility status.

```bash
aligntune info
```

**Output includes:**
- AlignTune version
- Python version
- Available backends (TRL, Unsloth)
- GPU information
- Memory status

### `aligntune list-backends-cmd`

List all available backends and their current status.

```bash
aligntune list-backends-cmd
```

**Shows:**
- Backend availability
- Version information
- GPU compatibility
- Installation status


### `aligntune train`

Train models with comprehensive options.

```bash
aligntune train --help # To get all parameters
```
```bash
# SFT Training
aligntune train --model "microsoft/DialoGPT-small" --dataset "tatsu-lab/alpaca" --type sft

# DPO Training
aligntune train --model "microsoft/DialoGPT-small" --dataset "Anthropic/hh-rlhf" --type dpo --backend trl --split train[:100] 

# PPO Training
aligntune train --model "Qwen/Qwen3-0.6B" --dataset "Anthropic/hh-rlhf" --type ppo --backend unsloth --split "train[:100]"
```

**Common Options:**

| Option | Description | Example |
|--------|-------------|---------|
| `--model`, `-m` | Model name or path | `--model "microsoft/DialoGPT-small"` |
| `--dataset`, `-d` | Dataset name | `--dataset "tatsu-lab/alpaca"` |
| `--type`, `-t` | Training type (sft, dpo, ppo, etc.) | `--type dpo` |
| `--backend`, `-b` | Backend (trl, unsloth, auto) | `--backend auto` |
| `--epochs`, `-e` | Number of epochs | `--epochs 3` |
| `--batch-size` | Batch size | `--batch-size 4` |
| `--learning-rate`, `--lr` | Learning rate | `--learning-rate 5e-5` |
| `--max-length` | Maximum sequence length | `--max-length 512` |
| `--output-dir`, `-o` | Output directory | `--output-dir ./output` |
| `--4bit` | Load model in 4-bit quantization | `--4bit` |
| `--8bit` | Load model in 8-bit quantization | `--8bit` |
| `--wandb-project` | Weights & Biases project name | `--wandb-project my-project` |


## Configuration Files

The CLI supports YAML configuration files for complex setups. See [Configuration Files](configuration.md) for detailed format documentation.

**Example configuration:**

```yaml
algo: dpo
model:
 name_or_path: "microsoft/DialoGPT-small"
 max_seq_length: 512
datasets:
 - name: "Anthropic/hh-rlhf"
 max_samples: 1000
train:
 max_steps: 1000
 learning_rate: 5e-5
 per_device_batch_size: 4
logging:
 output_dir: "./output"
 run_name: "my_experiment"
```

**Using configuration files:**

```bash
# Train with config file
aligntune train --config my_config.yaml

```

## Examples

### Quick SFT Training

```bash
aligntune train examples/sft_customer_support_unsloth/config_sft_customer_support.yaml
```

### Quick RL Training

```bash
aligntune train examples/grpo_gsm8k_trl/config_grpo_gsm8k.yaml
```


## Advanced Usage

For detailed command reference, see:
- [CLI Commands](commands.md) - Complete command reference
- [CLI Configuration](configuration.md) - Configuration file format

## See Also

- [Getting Started](../getting-started/installation.md) - Installation and setup
- [User Guide](../user-guide/overview.md) - Detailed usage guides
- [API Reference](../api-reference/overview.md) - Python API reference
- [Examples](../examples/overview.md) - Code examples