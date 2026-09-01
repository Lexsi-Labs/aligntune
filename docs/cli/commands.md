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
aligntune train --config src/aligntune/recipes/configs/dpo/mistral_hhrlhf.yaml
```

## Recipe Commands

### `aligntune recipes`

Browse, inspect, and run the built-in library of pre-tuned training recipes (`src/aligntune/recipes/`).

```bash
# List all recipes
aligntune recipes list

# Filter by algorithm or tags
aligntune recipes list --algorithm dpo
aligntune recipes list --tag llama --tag memory-efficient

# Search recipes
aligntune recipes list --search "math"

# Show full details for one recipe
aligntune recipes show llama3-instruction-tuning

# Copy a recipe to a local file for customization
aligntune recipes copy llama3-instruction-tuning --output my_custom_recipe.yaml

# Run a recipe directly, optionally overriding values
aligntune recipes run llama3-instruction-tuning --override train.learning_rate=1e-4

# Create a new recipe from an existing config file
aligntune recipes create my-recipe --description "My custom recipe" --config config.yaml --tag custom
```

**Subcommands:** `list`, `show <name>`, `copy <name>`, `run <name>`, `create <name>`

## Validation Commands

### `aligntune validate`

Validate configs, models, datasets, and estimate memory requirements before you spend GPU time.

```bash
# Validate a training config (and optionally check model/dataset access)
aligntune validate config my_config.yaml --verbose --check-access

# Check system compatibility
aligntune validate system --detailed

# Check whether a model is accessible (and optionally try downloading it)
aligntune validate model meta-llama/Llama-3-8B-Instruct --check-download

# Check whether a dataset is accessible
aligntune validate dataset anthropic/hh-rlhf --check-download

# Estimate memory requirements from a config or manually
aligntune validate memory --config my_config.yaml
aligntune validate memory --model meta-llama/Llama-3-8B --batch-size 8 --seq-length 4096
```

**Subcommands:** `config <file>`, `system`, `model <name>`, `dataset <name>`, `memory`

## Diagnostics Commands

### `aligntune diagnose`

Run system/config/training diagnostics, complementary to `validate`, focused on troubleshooting and live monitoring.

```bash
# Quick system check, or extended monitoring over time
aligntune diagnose system
aligntune diagnose system --duration 60 --interval 2 --output diagnostics.json

# Diagnose a config file for potential issues
aligntune diagnose config my_config.yaml --output config_diagnostics.json

# Set up training-time diagnostics/monitoring
aligntune diagnose training my_config.yaml --monitor --output-dir ./diagnostics

# List available diagnostic tools
aligntune diagnose info
```

**Subcommands:** `system`, `config <file>`, `training <file>`, `info`

## Advisor Commands

### `aligntune advise`

The **Utility Advisor**: deterministic VRAM/time/cost/carbon estimation and algorithm recommendations, without needing GPU access.

```bash
# Estimate VRAM, time, cost, and carbon for a training run
aligntune advise estimate --model "Qwen/Qwen2.5-7B" --dataset-size 10000 --algorithm dpo
aligntune advise estimate --model "meta-llama/Llama-2-70b-hf" --dataset-size 50000 \
    --algorithm lora --batch-size 8 --gpu h100 --region us-west-2

# Recommend algorithms for a task, optionally under a budget
aligntune advise recommend --task alignment --dataset-size 10000
aligntune advise recommend --task speed --dataset-size 50000 --budget 10.0

# Get optimization suggestions (precision, GPU, VRAM pressure)
aligntune advise optimize --model-size 7b --precision fp32 --gpu a100-40gb
aligntune advise optimize --model-size 70b --vram-tight --dataset-size 50000

# List available GPU profiles with specs and pricing
aligntune advise list-gpus
```

**Subcommands:** `estimate`, `recommend`, `optimize`, `list-gpus`

## Model Merging Commands

### `aligntune merge`

Merge multiple models via mergekit-backed methods (linear, task arithmetic) or a dependency-free LoRA adapter merge.

```bash
# Linear merge
aligntune merge --method linear \
    --models org/model-a org/model-b \
    --output ./merged_linear --weights 0.5 0.5

# LoRA adapter merge (no mergekit dependency needed)
aligntune merge --method lora-merge \
    --models org/base-model --adapter ./my_lora_adapter --output ./merged_full
```

**Methods:** `linear`, `task_arithmetic` (via mergekit) and `lora-merge` (via PEFT).

## Interactive Training. Aligner

### `aligntune aligner`

Start an interactive training session with live metric inspection and hyperparameter adjustment while training runs.

```bash
aligntune aligner --config config.yaml
aligntune aligner --config config.yaml --model gpt2
aligntune aligner --config config.yaml --no-dashboard
```

## Export & Verification Commands

### `aligntune export`

Export fine-tuned checkpoints to deployment formats.

```bash
# Export to GGUF (llama.cpp / Ollama)
aligntune export gguf ./checkpoint --output ./models -q Q4_K_M -c llama-cpp

# Export/create an Ollama Modelfile
aligntune export ollama ./checkpoint --create
aligntune export ollama ./model.gguf --name my-model:latest --create

# Upload to the HuggingFace Hub (full weights or adapter-only)
aligntune export hf_hub ./checkpoint --repo username/my-model
aligntune export hf_hub ./checkpoint --repo username/my-lora --adapter-only

# Merge a LoRA adapter into the base model weights
aligntune export merge_adapter ./checkpoint --output ./merged_model
```

**Subcommands:** `gguf`, `ollama`, `hf_hub`, `merge_adapter`

!!! warning "Known issue"
    In some environments `aligntune export` fails to load with `ImportError: cannot import name 'HfFolder' from 'huggingface_hub'` due to a `huggingface_hub` version mismatch (the command was written against an older API). Pin `huggingface_hub<0.26` or patch the import if you hit this, it is a dependency issue, not a missing feature.

### `aligntune verify-export`

Regression-test exported artifacts (GGUF, quantized HF checkpoints, etc.) against a baseline checkpoint by re-running the alignment audit/eval and comparing metrics against configurable thresholds.

```bash
# Verify specific artifacts against a baseline
aligntune verify-export run ./checkpoint \
    -a Q4_K_M:gguf:./exports/gguf/model.gguf \
    -a int8:hf_8bit:./exports/hf_int8 \
    --probe-set probes.json --eval-config eval.yaml

# Auto-discover artifacts under a directory tree and verify all of them
aligntune verify-export auto-discover ./checkpoint ./exports \
    --probe-set probes.json --eval-config eval.yaml
```

**Subcommands:** `run`, `auto-discover`. Exit code `0` = all artifacts pass, `1` = any artifact fails or config error, safe to use as a CI gate before shipping a quantized export.

## Adapter Management Commands

### `aligntune adapters`

Inspect and generate LoRA adapters.

```bash
# Show rank, target modules, and parameter count
aligntune adapters info ./my_lora_adapter

# Generate a LoRA adapter from a task description (Text2LoRA) or a document (Doc2LoRA)
aligntune adapters generate --adapter-type text2lora --description "Summarize legal contracts concisely"
aligntune adapters generate --adapter-type doc2lora --document ./spec.pdf --output ./generated_adapter
```

**Subcommands:** `info <adapter>`, `generate`

## Composition Commands

### `aligntune compose`

Run and inspect multi-stage training compositions (e.g. `SFT → MoA → ES → DPO → audit`), with checkpoints threaded automatically between stages. See [Model Merging](../advanced/merging.md) and [Advanced Adapters](../advanced/adapters.md) for the building blocks a composition can chain together.

```bash
# Run a composition pipeline defined in YAML
aligntune compose run recipes/configs/compositions/full_stack.yaml

# Continue past a failed stage, target a device, and set log verbosity
aligntune compose run recipes/configs/compositions/full_stack.yaml --device cuda --skip-failed --log-level DEBUG

# List available composition templates
aligntune compose list
aligntune compose list --search "full"

# Inspect a composition file's stages in detail
aligntune compose inspect recipes/configs/compositions/full_stack.yaml
```

**Subcommands:** `run <file>`, `list`, `inspect <file>`

## Indic Evaluation Commands

### `aligntune indic-eval`

Evaluate models on Indic-language benchmarks (MILU, IndicXTREME, IndicGenBench, Sarvam tasks) across Hindi, Tamil, Bengali, and other scripts.

```bash
# Evaluate on all languages and benchmarks
aligntune indic-eval run --model meta-llama/Llama-2-7b

# Evaluate specific languages and a specific benchmark
aligntune indic-eval run --model meta-llama/Llama-2-7b --languages hi,ta --benchmarks milu

# Quick smoke test with a sample limit
aligntune indic-eval run --model meta-llama/Llama-2-7b --limit 10

# List available tasks (optionally filtered by language)
aligntune indic-eval list
aligntune indic-eval list --language hi
```

**Subcommands:** `run`, `list`

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