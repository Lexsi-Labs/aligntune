# Configuration

AlignTune supports flexible configuration through Python code, YAML files, or CLI arguments.

## Configuration Methods

### 1. Python Code (Recommended for Development)

```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 num_epochs=3,
 batch_size=4,
 learning_rate=5e-5,
 output_dir="./output"
)
```

### 2. YAML Files (Recommended for Production)

```yaml
# config.yaml
model:
 name_or_path: "microsoft/DialoGPT-small"
 precision: fp32

datasets:
 - name: "tatsu-lab/alpaca"
 split: "train"
 max_samples: 1000

train:
 per_device_batch_size: 4
 num_epochs: 3
 learning_rate: 5e-5

logging:
 output_dir: "./output"
 run_name: "my_experiment"
```

### 3. CLI Arguments

```bash
aligntune train \
 --model microsoft/DialoGPT-small \
 --dataset tatsu-lab/alpaca \
 --epochs 3 \
 --batch-size 4 \
 --lr 5e-5
```

## Configuration Structure

### Model Configuration

```python
model_config = {
 "name_or_path": "microsoft/DialoGPT-small", # Required
 "precision": "fp32", # fp32, fp16, bf16
 "max_seq_length": 2048,
 "gradient_checkpointing": True,
 "use_unsloth": False
}
```

### Dataset Configuration

```python
dataset_config = {
 "name": "tatsu-lab/alpaca", # Required
 "split": "train",
 "max_samples": 1000,
 "percent": 100
}
```

### Training Configuration

```python
training_config = {
 "per_device_batch_size": 4, # Required
 "num_epochs": 3,
 "learning_rate": 5e-5,
 "gradient_accumulation_steps": 1,
 "max_steps": None,
 "warmup_steps": 100,
 "weight_decay": 0.01
}
```

### Logging Configuration

```python
logging_config = {
 "output_dir": "./output", # Required
 "run_name": "my_experiment",
 "loggers": ["tensorboard"],
 "save_steps": 500,
 "eval_steps": 100
}
```

## Complete Example

### YAML Configuration

```yaml
# Complete SFT configuration
model:
 name_or_path: "microsoft/DialoGPT-small"
 precision: fp32
 max_seq_length: 2048
 gradient_checkpointing: true

datasets:
 - name: "tatsu-lab/alpaca"
 split: "train"
 max_samples: 1000

train:
 per_device_batch_size: 4
 gradient_accumulation_steps: 1
 num_epochs: 3
 learning_rate: 5e-5
 warmup_steps: 100
 weight_decay: 0.01
 max_steps: null
 eval_interval: 100
 save_interval: 500

logging:
 output_dir: "./output"
 run_name: "sft_experiment"
 loggers: ["tensorboard"]
 save_steps: 500
 eval_steps: 100
```

### Python Code

```python
from aligntune.core.sft.config_loader import SFTConfigLoader
from aligntune.core.backend_factory import create_sft_trainer


# Load configuration
config = SFTConfigLoader.load_from_yaml("config.yaml")


# Create trainer from config
trainer = create_sft_trainer(config=config)


trainer.train()
```

## Configuration Validation

```python
from aligntune.core.rl.config_loader import ConfigLoader

from aligntune.core.sft.config_loader import SFTConfigLoader

# Load and validate
config = ConfigLoader.load_from_yaml("config.yaml")
ConfigLoader.validate_config(config)
```

## Environment Variables

You can override configuration with environment variables:

```bash
export ALIGNTUNE_OUTPUT_DIR="./custom_output"
export ALIGNTUNE_LEARNING_RATE=1e-4
export ALIGNTUNE_BATCH_SIZE=8
```

## Next Steps

- [Backend Selection](backend-selection.md) - Choose the right backend
- [SFT Guide](../user-guide/sft.md) - SFT-specific configuration
- [RL Guide](../user-guide/rl.md) - RL-specific configuration