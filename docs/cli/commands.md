# CLI Commands

Detailed reference for all AlignTune CLI commands and options.

## Core Commands

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



## Training Commands

### `aligntune train`

Main training command with flexible configuration options.

```bash
aligntune train [OPTIONS]
```
```bash
aligntune train --help # To get all parameters
```

#### Required Options

| Option | Description | Example |
|--------|-------------|---------|
| `--model` | Model name or path | `--model "microsoft/DialoGPT-small"` |
| `--dataset` | Dataset name | `--dataset "tatsu-lab/alpaca"` |
| `--type` | Training type | `--type sft` |

#### Optional Options

| Option | Default | Description |
|--------|---------|-------------|
| `--backend` | `auto` | Backend to use (trl, unsloth, auto) |
| `--epochs` | `1` | Number of training epochs |
| `--batch-size` | `4` | Batch size per device |
| `--learning-rate` | `5e-5` | Learning rate |
| `--max-length` | `512` | Maximum sequence length |
| `--output-dir` | `./output` | Output directory |
| `--seed` | `42` | Random seed |
| `--config` | - | YAML configuration file |

#### Examples

```bash
# Basic SFT training
aligntune train \
 --model microsoft/DialoGPT-small \
 --dataset tatsu-lab/alpaca \

# DPO training with specific backend
aligntune train \
 --model microsoft/DialoGPT-medium \
 --dataset Anthropic/hh-rlhf \
 --type dpo \
 --backend trl \
 --epochs 1 \
 --batch-size 2 \
 --learning-rate 1e-6

# Training with configuration file
aligntune train --config examples/configs/dpo_minimal.yaml
```


## Configuration File Support

The CLI supports YAML configuration files for complex training setups:

```yaml
# config.yaml
algo: dpo
model:
 name_or_path: "microsoft/DialoGPT-medium"
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


## Error Handling

The CLI provides helpful error messages and suggestions:

- **Model not found**: Lists similar available models
- **Backend unavailable**: Shows installation instructions
- **Configuration errors**: Points to specific validation issues
- **Memory issues**: Suggests batch size adjustments

## See Also

- [CLI Overview](overview.md) - Main CLI overview
- [Configuration Files](configuration.md) - YAML configuration format
- [Getting Started](../getting-started/quickstart.md) - Quick start guide