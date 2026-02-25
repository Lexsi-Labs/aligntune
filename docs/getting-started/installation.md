# Installation

This guide will help you install AlignTune and set up your environment.

## Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher
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

# Or install with all optional dependencies
pip install -e ".[all]"
```

### Install with Optional Dependencies

```bash
# Install with Unsloth support (for faster training)
pip install aligntune[unsloth]

# Install with evaluation tools
pip install aligntune[eval]

# Install with all optional dependencies
pip install aligntune[all]
```

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

- `unsloth` - Fast training acceleration (requires GPU)
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

## Unsloth Setup (Optional)

Unsloth provides faster training but requires specific setup:

```bash
# Install Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Or for local installation
pip install unsloth
```

### Verify Unsloth

```python
from aligntune.core.backend_factory import BackendType
from aligntune.core.backend_factory import get_backend_status

status = get_backend_status()
print(f"Unsloth available: {status['unsloth']['available']}")
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