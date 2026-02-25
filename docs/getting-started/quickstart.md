# Quick Start

This quick start guide demonstrates how to run a complete end-to-end workflow with AlignTune in just a few steps.

---

## 1. Prepare Your Environment

Ensure you have installed AlignTune and its dependencies as per the [Installation Guide](installation.md).

Activate your virtual environment:
```bash
# If using venv
source aligntune-env/bin/activate

# If using conda
conda activate aligntune
```

Check installation:
```bash
python -c "import aligntune; print('AlignTune installed successfully')"
```

---

## 2. Load a Dataset

We'll use the **Alpaca** dataset for this example.

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train")

# Preview the data
print(dataset[0])
# Output: {'instruction': '...', 'input': '...', 'output': '...'}
```

---

## 3. Initialize and Configure the Trainer

Use the `create_sft_trainer` function to define your model, dataset, and training configuration.

### Supervised Fine-Tuning (SFT)

```python
from aligntune.core.backend_factory import create_sft_trainer

# Create SFT trainer
trainer = create_sft_trainer(
 model_name="meta-llama/Llama-3.2-3B-Instruct",
 dataset_name="tatsu-lab/alpaca",
 backend="trl", # or "unsloth" for faster training
 num_epochs=3,
 batch_size=4,
 learning_rate=5e-5,
 max_samples=1000 # Limit for quick start
)
```

---

## 4. Train the Model

Train your pipeline on the training data.

```python
# Train the model
trainer.train()
```

During training, AlignTune will automatically handle data preprocessing and apply the chosen training strategy. You'll see progress bars and training metrics.

---

## 5. Evaluate and Predict

After training, evaluate performance and generate predictions:

```python
# Evaluate on test set (if available)
metrics = trainer.evaluate()
print("Evaluation metrics:", metrics)
# Output: {'eval_loss': 2.34, 'eval_perplexity': 10.4, ...}

# Make predictions
result = trainer.predict("What is machine learning?")
print("Prediction:", result)
```

Supported metrics include: **Loss**, **Perplexity**, **BLEU**, and **ROUGE** scores.

---

## 6. Save and Load the Model

Persist your trained model for later use:

```python
# Save to disk
model_path = trainer.save_model("./output/my_model")
print(f"Model saved to: {model_path}")

# Load from disk (in a new session)
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./output/my_model")
tokenizer = AutoTokenizer.from_pretrained("./output/my_model")

# Push to HuggingFace Hub (optional)
# trainer.push_to_hub("username/my-model", token="hf_...")
```

---

## 7. Try Reinforcement Learning (DPO)

Switch to reinforcement learning with minimal code changes:

### Reinforcement Learning (DPO)

```python
from aligntune.core.backend_factory import create_rl_trainer

# Create DPO trainer
trainer = create_rl_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dpo", # Direct Preference Optimization
 backend="trl",
 num_epochs=1,
 batch_size=1,
 learning_rate=1e-6,
 max_samples=500 # Limit for quick start
)

# Train
trainer.train()

# Evaluate
metrics = trainer.evaluate()
print("DPO metrics:", metrics)

# Save model
trainer.save_model()
```

---

### Next Steps

- Explore advanced configurations in the [User Guide](../user-guide/overview.md)
- Learn about different algorithms in [Algorithms Overview](../algorithms/overview.md)
- Compare backends with [Backend Comparison](../backends/comparison.md)
- Dive into RL training in [RL Guide](../user-guide/rl.md)

## Using YAML Configuration

Create a configuration file `config.yaml`:

```yaml
# SFT Configuration
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
 output_dir: "./output"

logging:
 output_dir: "./output"
 run_name: "my_first_training"
```

Then train:

```python
from aligntune.core.config_loader import ConfigLoader
from aligntune.core.backend_factory import create_sft_trainer

# Load configuration
config = ConfigLoader.load_from_yaml("config.yaml")

# Create trainer
trainer = create_sft_trainer(config)

# Train
trainer.train()
```

## CLI Quick Start

```bash
# Train SFT model
aligntune train \
 --model microsoft/DialoGPT-small \
 --dataset tatsu-lab/alpaca \
 --type sft \
 --epochs 3 \
 --batch-size 4

# Train DPO model
aligntune train \
 --model microsoft/DialoGPT-medium \
 --dataset Anthropic/hh-rlhf \
 --type dpo \
 --epochs 1 \
 --batch-size 1
```

## What's Next?

- [Configuration Guide](configuration.md) - Learn about all configuration options
- [Backend Selection](backend-selection.md) - Choose between TRL and Unsloth
- [SFT Guide](../user-guide/sft.md) - Deep dive into SFT training
- [RL Guide](../user-guide/rl.md) - Deep dive into RL training