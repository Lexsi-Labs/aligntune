# Installation

This guide will help you install AlignTune and set up your environment.

## Requirements

- Python 3.11 or higher
- PyTorch 2.0 or higher
- Git (required, `pip install aligntune` resolves direct `git+https` dependencies such as `tokenizer-extension` and `curatorkit`, so `git` must be installed and on your `PATH`)
- CUDA-capable GPU (recommended for training, optional for inference)

## Installation Methods

### Install from PyPI (Recommended)

```bash
pip install aligntune
```

### Install from Source

```bash
# Clone the repository
git clone https://github.com/Lexsi-Labs/aligntune.git
cd aligntune

# Install in development mode
pip install -e .

# Or include development tooling (tests, linters, etc.)
pip install -e ".[dev]"
```

### Install with Optional Dependencies

AlignTune does not currently define `eval` or `all` extras, only `dev`
and `docs` are declared in `pyproject.toml`. Evaluation tooling (`lm-eval`,
`scikit-learn`, `rouge-score`, `sacrebleu`, `nltk`, `evaluate`) is already included
in the base install, so a plain `pip install aligntune` covers it.

Unsloth (faster training on CUDA GPUs) needs no separate install — it is
vendored inside the `aligntune` package. See [Unsloth](#unsloth) below.

### Install with `uv`

```bash
pip install uv
uv pip install -e .
```

CuratorKIT (data curation: schema gating, cleaning, dedup) and `mergekit`
(model merging, notebooks 35-42) are both already included in `dependencies`
and install automatically with the single `pip install -e .` /
`uv pip install -e .` above, no separate step needed.

### Model Merging (mergekit)

`mergekit` is vendored *as part of the `aligntune` package itself* (under
`third_party/mergekit`, built into the same wheel/editable install rather
than installed as a second distribution) with two small patches for
compatibility with this project's transformers/pydantic versions (see
`third_party/mergekit/PATCH_NOTES.md`). It needs no separate install step ,
`pip install -e .` or `uv pip install -e .` from [above](#install-with-uv)
already includes it.

### Unsloth

`unsloth` and `unsloth_zoo` (`2026.7.2`) are vendored the same way as mergekit
— under `third_party/unsloth`, `third_party/unsloth_zoo`, built into the same
wheel/editable install — so faster training needs no separate install step.
`pip install aligntune` / `pip install -e .` already includes it.

Vendoring is required here, not just convenient: Unsloth's published metadata
caps `transformers<=5.5.0` and `trl<=0.24.0`, while this project pins
`transformers==5.14.1` and `trl==1.7.1`. Letting `pip`/`uv` resolve Unsloth's
declared dependencies normally would downgrade the whole transformers/trl stack
(breaking everything else) or fail to resolve at all. Do **not** run
`pip install unsloth` yourself — a separate copy would shadow the vendored one
and drag in those incompatible pins.

Unsloth still requires a CUDA-capable GPU at run time; without one AlignTune
falls back to the TRL backend. See
[Unsloth Compatibility](../unsloth_compatibility.md).

## Verify Installation

```bash
# Check installation
python -c "import aligntune; print(aligntune.__version__)"

# Check CLI
aligntune --help

# Check system information
aligntune info
```

## Dependencies

### Core Dependencies

- `transformers` - HuggingFace Transformers library
- `trl` - Transformer Reinforcement Learning
- `datasets` - HuggingFace Datasets
- `torch` - PyTorch
- `numpy`, `pandas` - Data processing
- `pyyaml` - Configuration management
- `tqdm` - Progress bars

### Optional Dependencies

- `wandb` - Weights & Biases logging
- `tensorboard` - TensorBoard logging
- `lm-eval` - Language model evaluation
- `scikit-learn` - Evaluation metrics
- `seqeval` - Sequence evaluation

## GPU Setup

### CUDA Installation

AlignTune works with CUDA-enabled GPUs. Install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Verify GPU Access

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

## Unsloth Setup

Unsloth is vendored inside AlignTune (see [Unsloth](#unsloth) above) — there is
nothing extra to install. It is used automatically when a CUDA-capable GPU is
available; otherwise AlignTune falls back to the TRL backend.

### Verify Unsloth

```python
from aligntune.core.backend_factory import get_backend_status

status = get_backend_status()
print(f"Unsloth available: {status['unsloth_available']}")
```

## Environment Variables

Set these environment variables for optimal performance:

```bash
# HuggingFace cache directory
export HF_HOME=/path/to/cache

# CUDA device selection
export CUDA_VISIBLE_DEVICES=0,1

# Disable tokenizers parallelism (if you see warnings)
export TOKENIZERS_PARALLELISM=false

# HuggingFace offline mode (if needed)
export HF_DATASETS_OFFLINE=1
```

## Troubleshooting

### Import Errors

```bash
# Ensure proper Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Reinstall in development mode
pip install -e .
```

### CUDA Issues

```bash
# Check CUDA version
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Unsloth Compatibility

If Unsloth has compatibility issues, AlignTune automatically falls back to TRL backends. See [Unsloth Compatibility](../unsloth_compatibility.md) for details.

## Next Steps

- [Quick Start Guide](quickstart.md) - Get started with your first training
- [Configuration Guide](configuration.md) - Learn about configuration options
- [Backend Selection](backend-selection.md) - Choose the right backend