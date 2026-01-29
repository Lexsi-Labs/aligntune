# CLI Reference

AlignTune provides a comprehensive command-line interface for training and managing models.

## Overview

The CLI provides easy access to all AlignTune functionality without writing Python code.

## Main Commands

### `aligntune info`

Display system information and compatibility status.

```bash
aligntune info
```

### `aligntune list-backends-cmd`

List all available backends and their status.

```bash
aligntune list-backends-cmd
```



## Training Commands

### `aligntune train`

Train a model with specified parameters.

```bash
# SFT Training
aligntune train --model "microsoft/DialoGPT-small" --dataset "tatsu-lab/alpaca" 

# DPO Training
aligntune train --model "microsoft/DialoGPT-small" --dataset "Anthropic/hh-rlhf" --type dpo --backend trl

# PPO Training
aligntune train --model "microsoft/DialoGPT-small" --dataset "Anthropic/hh-rlhf" --type ppo --backend unsloth
```

### Training Options

| Option | Description | Example |
|--------|-------------|---------|
| `--model` | Model name or path | `--model "microsoft/DialoGPT-small"` |
| `--dataset` | Dataset name | `--dataset "tatsu-lab/alpaca"` |
| `--type` | Training type (sft, dpo, ppo, etc.) | `--type dpo` |
| `--backend` | Backend (trl, unsloth, auto) | `--backend auto` |
| `--epochs` | Number of epochs | `--epochs 3` |
| `--batch-size` | Batch size | `--batch-size 4` |
| `--learning-rate` | Learning rate | `--learning-rate 5e-5` |
| `--max-length` | Maximum sequence length | `--max-length 512` |
| `--output-dir` | Output directory | `--output-dir ./output` |


## Configuration Files

The CLI supports YAML configuration files for complex setups:

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

## Examples

### Quick SFT Training

```bash
aligntune train \
 --model microsoft/DialoGPT-small \
 --dataset tatsu-lab/alpaca \
 --type sft \
 --backend auto \
 --epochs 3 \
 --batch-size 4 \
 --learning-rate 5e-5
```


## See Also

- [Getting Started](getting-started/installation.md) - Installation and setup
- [User Guide](user-guide/overview.md) - Detailed usage guides
- [API Reference](api-reference/overview.md) - Python API reference