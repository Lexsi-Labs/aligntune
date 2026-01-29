# Model Management

Learn how to save, load, and share your trained models with AlignTune.

## Saving Models

### Basic Save

```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(...)
trainer.train()

# Save model
path = trainer.save_model()
print(f"Model saved to: {path}")
```

### Custom Save Path

```python
# Save to custom directory
path = trainer.save_model(output_dir="./my_models/experiment_1")
```

### What Gets Saved

When you call `save_model()`, AlignTune saves:

- Model weights (`pytorch_model.bin` or `model.safetensors`)
- Tokenizer files (`tokenizer.json`, `tokenizer_config.json`, etc.)
- Model configuration (`config.json`)
- Training configuration (`training_config.yaml`)
- Task metadata (if applicable)

## Loading Models

### Load Saved Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./output/my_model")
tokenizer = AutoTokenizer.from_pretrained("./output/my_model")
```

### Load with AlignTune

```python
from aligntune.core.backend_factory import create_sft_trainer

# Create trainer and load model
trainer = create_sft_trainer(
 model_name="./output/my_model", # Path to saved model
 dataset_name="tatsu-lab/alpaca"
)

# Model is automatically loaded
trainer.predict("Hello!")
```

## Pushing to HuggingFace Hub

### Basic Push

```python
trainer = create_sft_trainer(...)
trainer.train()
trainer.save_model()

# Push to Hub
url = trainer.push_to_hub(
 repo_id="username/my-model",
 private=False
)
print(f"Model available at: {url}")
```

### Private Repository

```python
url = trainer.push_to_hub(
 repo_id="username/my-private-model",
 private=True,
 token="hf_..." # Your HuggingFace token
)
```

### Custom Commit Message

```python
url = trainer.push_to_hub(
 repo_id="username/my-model",
 commit_message="Trained on Alpaca dataset with 3 epochs"
)
```

## Generating Predictions

### Single Prediction

```python
# Generate from single input
result = trainer.predict("What is machine learning?")
print(result)
```

### Batch Predictions

```python
# Generate from multiple inputs
results = trainer.predict([
 "What is AI?",
 "Explain deep learning",
 "What is NLP?"
])
for result in results:
 print(result)
```

### Custom Generation Parameters

```python
result = trainer.predict(
 "What is machine learning?",
 max_new_tokens=200,
 temperature=0.7,
 top_p=0.9,
 do_sample=True
)
```

## Model Evaluation

### Basic Evaluation

```python
# Evaluate on default validation set
metrics = trainer.evaluate()
print(metrics)
```

### Custom Evaluation Dataset

```python
from datasets import load_dataset

# Load custom evaluation dataset
eval_dataset = load_dataset("tatsu-lab/alpaca", split="test")

# Evaluate
metrics = trainer.evaluate(eval_dataset=eval_dataset)
print(metrics)
```

### Custom Metric Prefix

```python
metrics = trainer.evaluate(
 eval_dataset=eval_dataset,
 metric_key_prefix="test"
)
# Metrics will have "test/" prefix
```

## Checkpointing

### Save Checkpoints

Checkpoints are automatically saved during training:

```python
trainer = create_sft_trainer(
 ...,
 save_interval=500 # Save every 500 steps
)
trainer.train()
```

### Load Checkpoint

```python
# Resume from checkpoint
trainer.load_checkpoint("./output/checkpoint-1000")
```

## Model Information

### Get Model Path

```python
# After saving
path = trainer.save_model()
print(f"Model path: {path}")
```

### Check Model Status

```python
# Check if model is loaded
if trainer.model is not None:
 print("Model is loaded")
 print(f"Device: {next(trainer.model.parameters()).device}")
```

## Best Practices

### 1. Save Regularly

```python
# Save after training
trainer.train()
trainer.save_model()

# Also save checkpoints during training
# (automatic with save_interval)
```

### 2. Version Your Models

```python
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
path = trainer.save_model(output_dir=f"./models/experiment_{timestamp}")
```

### 3. Push Important Models

```python
# Push production models to Hub
if is_production_model:
 trainer.push_to_hub(
 repo_id="username/production-model",
 private=False
 )
```

### 4. Document Your Models

```python
# Save with metadata
trainer.save_model()
# Edit training_config.yaml to add notes
```

## Troubleshooting

### Model Not Found

```python
# Check if model exists
from pathlib import Path

model_path = Path("./output/my_model")
if not model_path.exists():
 print("Model not found!")
```

### Push to Hub Fails

```python
# Ensure you're logged in
from huggingface_hub import login

login(token="hf_...")

# Then push
trainer.push_to_hub("username/my-model")
```

### Memory Issues

```python
# Use smaller batch size for prediction
result = trainer.predict(
 "Your input",
 max_new_tokens=50 # Reduce if needed
)
```

## Next Steps

- [Evaluation Guide](evaluation.md) - Learn about evaluation metrics
- [SFT Guide](sft.md) - SFT-specific model management
- [RL Guide](rl.md) - RL-specific model management