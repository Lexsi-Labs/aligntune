# Supervised Fine-Tuning (SFT) Guide

Complete guide to Supervised Fine-Tuning with AlignTune, covering all task types, configurations, and best practices.

## Overview

Supervised Fine-Tuning (SFT) is the process of training a pre-trained language model on a labeled dataset to adapt it for specific tasks. AlignTune provides a unified interface for SFT across multiple task types with support for both TRL and Unsloth backends.

### Supported Task Types

AlignTune supports six main SFT task types:

1. **Instruction Following** - Teach models to follow instructions
2. **Supervised Fine-Tuning** - General-purpose fine-tuning
3. **Text Classification** - Classify text into categories
4. **Token Classification** - Named Entity Recognition (NER), POS tagging
5. **Text Generation** - Generate coherent text
6. **Chat Completion** - Conversational AI training

## Quick Start

### Basic SFT Training

```python
from aligntune.core.backend_factory import create_sft_trainer

# Create and train SFT model
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="tatsu-lab/alpaca",
 backend="trl", # or "unsloth" for faster training
 num_epochs=3,
 batch_size=4,
 learning_rate=2e-4,
 max_seq_length=512,
 max_samples=1000
)

# Train the model
trainer.train()

# Save the model
model_path = trainer.save_model()
print(f"Model saved to: {model_path}")
```

### Using YAML Configuration

```yaml
# config.yaml
model:
 name_or_path: "microsoft/DialoGPT-medium"
 max_seq_length: 512

dataset:
 name: "tatsu-lab/alpaca"
 max_samples: 1000
 task_type: "supervised_fine_tuning"

train:
 epochs: 3
 per_device_batch_size: 4
 learning_rate: 2e-4

logging:
 output_dir: "./output"
```

```python
from aligntune.core.sft.config_loader import SFTConfigLoader
from aligntune.core.backend_factory import create_sft_trainer

# Load configuration
config = SFTConfigLoader.load_from_yaml("config.yaml")

# Create trainer from config
trainer = create_sft_trainer(
 model_name=config.model.name_or_path,
 dataset_name=config.dataset.name,
 backend="trl",
 num_epochs=config.train.epochs,
 batch_size=config.train.per_device_batch_size,
 learning_rate=config.train.learning_rate,
 max_seq_length=config.model.max_seq_length,
 max_samples=config.dataset.max_samples,
 task_type=config.dataset.task_type.value
)

trainer.train()
```

## Task Types

### 1. Instruction Following

Train models to follow instructions and generate appropriate responses.

```python
from aligntune.core.backend_factory import create_sft_trainer


trainer = create_sft_trainer(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    dataset_name="tatsu-lab/alpaca",
    backend="unsloth", # Unsloth recommended for generation tasks
    task_type="instruction_following",
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-4,
    max_seq_length=1024, # Longer sequences for instructions
    max_samples=1000,
    # Column mappings for instruction datasets
    instruction_column="instruction",
    response_column="output",
    input_column="input"
)


trainer.train()
```

**Dataset Format:**
```json
{
 "instruction": "Explain machine learning",
 "input": "",
 "output": "Machine learning is a subset of AI..."
}
```

### 2. Supervised Fine-Tuning

General-purpose fine-tuning for any text-to-text task.

```python
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="databricks/databricks-dolly-15k",
 backend="trl",
 task_type="supervised_fine_tuning",
 num_epochs=3,
 batch_size=4,
 learning_rate=2e-4,
 max_seq_length=512,
 auto_detect_fields=True # Automatically detect dataset fields
)

trainer.train()
```

### 3. Text Classification

Classify text into predefined categories (sentiment, topic, etc.).

```python
trainer = create_sft_trainer(
 model_name="distilbert-base-uncased",
 dataset_name="imdb",
 backend="trl", # TRL recommended for classification
 task_type="text_classification",
 num_epochs=3,
 batch_size=8,
 learning_rate=5e-5,
 max_seq_length=512,
 max_samples=5000,
 # Classification-specific
 text_column="text",
 label_column="label",
 num_labels=2 # Binary classification
)

trainer.train()
```

**Dataset Format:**
```json
{
 "text": "This movie is fantastic!",
 "label": 1
}
```

### 4. Token Classification

Named Entity Recognition (NER), Part-of-Speech tagging, etc.

```python
trainer = create_sft_trainer(
 model_name="bert-base-uncased",
 dataset_name="lhoestq/conll2003",
 backend="trl", # TRL recommended for token classification
 task_type="token_classification",
 num_epochs=3,
 batch_size=8,
 learning_rate=5e-5,
 max_seq_length=512,
 split="train", # Dataset split
 tokens_column="tokens",
 tags_column="ner_tags",
 num_labels=9 # Number of NER tag classes
)

trainer.train()
```

**Dataset Format:**
```json
{
 "tokens": ["Apple", "is", "a", "company"],
 "ner_tags": [1, 0, 0, 0] # B-ORG, O, O, O
}
```

### 5. Text Generation

Generate coherent, context-aware text.

```python
trainer = create_sft_trainer(
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="Salesforce/wikitext",
    backend="trl", # Unsloth recommended for generation
    task_type="text_generation",
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-4,
    max_seq_length=1024,
    text_column="text"
)


trainer.train()
```

### 6. Chat Completion

Train conversational AI models with chat templates.

```python
trainer = create_sft_trainer(
 model_name="mistralai/Mistral-7B-v0.1",
 dataset_name="HuggingFaceH4/ultrachat_200k",
 backend="unsloth",
 task_type="chat_completion",
 split="train_sft", # Dataset split
 num_epochs=2,
 batch_size=2,
 learning_rate=2e-4,
 max_seq_length=2048, # Longer for conversations
 gradient_accumulation_steps=4,
 messages_column="messages",
 chat_template="auto" # Auto-detect chat template
)

trainer.train()
```

**Dataset Format:**
```json
{
 "messages": [
 {"role": "user", "content": "Hello!"},
 {"role": "assistant", "content": "Hi! How can I help?"}
 ]
}
```

## Backend Selection

### TRL Backend (Recommended for Classification)

**Use TRL when:**
- Training classification models
- Training token classification (NER)
- Need maximum compatibility
- Working with smaller models

```python
trainer = create_sft_trainer(
 model_name="distilbert-base-uncased",
 dataset_name="imdb",
 backend="trl", # Explicit TRL backend
 task_type="text_classification"
)
```

### Unsloth Backend (Recommended for Generation)

**Use Unsloth when:**
- Training generation models (faster)
- Working with large models
- Need memory efficiency
- Training instruction following or chat models

```python
trainer = create_sft_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="tatsu-lab/alpaca",
 backend="unsloth", # Unsloth for speed
 task_type="instruction_following"
)
```

### Auto Backend Selection

Let AlignTune choose the best backend:

```python
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="tatsu-lab/alpaca",
 backend="auto", # Automatic selection
 task_type="instruction_following"
)
```

## Configuration Options

### Model Configuration

```python
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="tatsu-lab/alpaca",
 # Model settings
 max_seq_length=512,
 quantization={"load_in_4bit": True}, # 4-bit quantization
 peft_enabled=True, # Enable LoRA
 lora_r=16, # LoRA rank
 lora_alpha=32, # LoRA alpha
 lora_dropout=0.1,
 # For GPT2 models, use: lora_target_modules=["c_attn", "c_proj"]
 use_gradient_checkpointing=True, # Memory optimization
 use_flash_attention_2=True # Flash Attention (if available)
)
```

### Training Configuration

```python
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="tatsu-lab/alpaca",
 # Training settings
 num_epochs=3,
 max_steps=None, # Use epochs instead
 batch_size=4,
 gradient_accumulation_steps=4, # Effective batch size = 16
 learning_rate=2e-4,
 weight_decay=0.01,
 warmup_steps=100,
 warmup_ratio=0.1,
 max_grad_norm=1.0,
 # Advanced settings
 packing=True, # Sequence packing for efficiency
 # Note: For GPT2 models, use backend="trl" as GPT2 is not supported by Unsloth
 packing_strategy="bfd", # "bfd" or "wrapped"
 loss_type="nll", # "nll" or "dft"
 completion_only_loss=True, # Only compute loss on completion
 activation_offloading=False # Offload activations to CPU
)
```

### Dataset Configuration

```python
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="tatsu-lab/alpaca",
 # Dataset settings
 max_samples=1000, # Limit dataset size
 percent=10.0, # Use 10% of dataset
 split="train", # Dataset split
 # Note: For GPT2 models, use backend="trl" as GPT2 is not supported by Unsloth
 # Column mappings
 column_mapping={
 "instruction": "instruction",
 "output": "response"
 },
 # Field detection
 auto_detect_fields=True, # Auto-detect dataset fields
 dataset_num_proc=4 # Parallel processing
)
```

## Advanced Features

### LoRA/QLoRA Fine-Tuning

Efficient fine-tuning with parameter-efficient methods:

```python
trainer = create_sft_trainer(
 model_name="meta-llama/Llama-2-7b-hf",
 dataset_name="tatsu-lab/alpaca",
 backend="unsloth",
 # LoRA configuration
 peft_enabled=True,
 lora_r=16,
 lora_alpha=32,
 lora_dropout=0.1,
 lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
 # Quantization
 quantization={
 "load_in_4bit": True,
 "bnb_4bit_compute_dtype": "float16",
 "bnb_4bit_quant_type": "nf4"
 }
)
```

### Sequence Packing

Pack multiple sequences into a single batch for efficiency:

```python
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="tatsu-lab/alpaca",
 backend="trl", # Use TRL for GPT2 models or models not supported by Unsloth
 packing=True, # Enable packing
 packing_strategy="bfd", # "bfd" (best-fit-decreasing) or "wrapped"
 padding_free=True, # No padding needed with packing
 max_seq_length=2048 # Longer sequences with packing
)
```

### Mixed Precision Training

Use FP16 or BF16 for faster training:

```python
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="tatsu-lab/alpaca",
 # Precision settings (handled automatically by backend)
 # Unsloth uses optimal precision automatically
 # TRL: set fp16=True or bf16=True in training config
)
```

### Gradient Checkpointing

Reduce memory usage during training:

```python
trainer = create_sft_trainer(
 model_name="meta-llama/Llama-2-7b-hf",
 dataset_name="tatsu-lab/alpaca",
 use_gradient_checkpointing=True, # Enable checkpointing
 activation_offloading=True # Offload activations to CPU
)
```

## Evaluation

### Basic Evaluation

```python
# Evaluate on default validation set
metrics = trainer.evaluate()
print(metrics)
```

### Custom Evaluation Dataset

```python
from datasets import load_dataset

# Load evaluation dataset
eval_dataset = load_dataset("tatsu-lab/alpaca", split="test")

# Evaluate
metrics = trainer.evaluate(eval_dataset=eval_dataset)
print(f"Loss: {metrics['eval_loss']}")
print(f"Perplexity: {metrics.get('eval_perplexity', 'N/A')}")
```

### Zero-Shot Evaluation

```python
# Generate predictions for test prompts
# Note: Make sure to call trainer.train() or trainer.setup_model() before using predict()
test_prompts = [
 "What is machine learning?",
 "Explain deep learning",
 "What is NLP?"
]

results = trainer.predict(test_prompts)
for prompt, result in zip(test_prompts, results):
 print(f"Q: {prompt}")
 print(f"A: {result}\n")
```

## Model Management

### Saving Models

```python
# Save after training
model_path = trainer.save_model()
print(f"Model saved to: {model_path}")

# Save to custom directory
model_path = trainer.save_model(output_dir="./my_models/experiment_1")
```

### Pushing to HuggingFace Hub

```python
# Push to Hub
url = trainer.push_to_hub(
 repo_id="username/my-sft-model",
 private=False
)
print(f"Model available at: {url}")

# Push with custom commit message
url = trainer.push_to_hub(
 repo_id="username/my-sft-model",
 commit_message="Trained on Alpaca dataset with 3 epochs"
)
```

### Loading Saved Models

```python
# Load saved model
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
 model_name="./output/my_model", # Path to saved model
 dataset_name="tatsu-lab/alpaca"
)

# Use for inference
result = trainer.predict("What is AI?")
```

## Best Practices

### 1. Choose the Right Task Type

- **Instruction Following**: For teaching models to follow instructions
- **Text Classification**: For categorization tasks
- **Token Classification**: For NER, POS tagging
- **Chat Completion**: For conversational AI
- **Text Generation**: For general text generation

### 2. Backend Selection

- **TRL**: Use for classification, token classification, maximum compatibility
- **Unsloth**: Use for generation tasks (faster), large models

### 3. Memory Optimization

```python
# For large models, use:
trainer = create_sft_trainer(
 model_name="meta-llama/Llama-2-7b-hf",
 dataset_name="tatsu-lab/alpaca",
 backend="unsloth",
 peft_enabled=True, # LoRA
 quantization={"load_in_4bit": True}, # 4-bit quantization
 use_gradient_checkpointing=True, # Gradient checkpointing
 batch_size=1, # Smaller batch size
 gradient_accumulation_steps=8 # Compensate with accumulation
)
```

### 4. Learning Rate Scheduling

```python
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="tatsu-lab/alpaca",
 learning_rate=2e-4,
 warmup_steps=100, # Warmup for stable training
 warmup_ratio=0.1 # Or use ratio instead
)
```

### 5. Dataset Size

- Start with a subset (`max_samples=1000`) for testing
- Gradually increase dataset size
- Use `percent` parameter for quick experiments

### 6. Sequence Length

- **Classification**: 512 tokens is usually sufficient
- **Instruction Following**: 1024-2048 tokens
- **Chat Completion**: 2048+ tokens for conversations
- **Text Generation**: 1024-2048 tokens

### 7. Regular Checkpointing

```python
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="tatsu-lab/alpaca",
 save_interval=500, # Save every 500 steps
 eval_interval=100 # Evaluate every 100 steps
)
```

## Troubleshooting

### Out of Memory Errors

```python
# Reduce batch size and use gradient accumulation
trainer = create_sft_trainer(
 model_name="meta-llama/Llama-2-7b-hf",
 dataset_name="tatsu-lab/alpaca",
 batch_size=1, # Reduce batch size
 gradient_accumulation_steps=8, # Increase accumulation
 use_gradient_checkpointing=True, # Enable checkpointing
 quantization={"load_in_4bit": True} # Use quantization
)
```

### Slow Training

```python
# Use Unsloth backend for generation tasks
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="tatsu-lab/alpaca",
 backend="unsloth", # faster
 packing=True, # Enable sequence packing
 dataset_num_proc=4 # Parallel data processing
)
```

### Poor Model Performance

```python
# Increase training duration and adjust learning rate
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="tatsu-lab/alpaca",
 num_epochs=5, # More epochs
 learning_rate=1e-4, # Lower learning rate
 warmup_steps=200, # More warmup
 max_samples=None # Use full dataset
)
```

### Dataset Format Issues

```python
# Use column mapping for custom datasets
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="my-custom-dataset",
 column_mapping={
 "instruction": "prompt", # Map custom column names
 "output": "completion"
 },
 auto_detect_fields=False # Disable auto-detection
)
```

## Examples

### Complete Instruction Following Example

```python
from aligntune.core.backend_factory import create_sft_trainer

# Create trainer
trainer = create_sft_trainer(
 model_name="gpt2", # Use Qwen/Llama instead of GPT2 for Unsloth
 dataset_name="tatsu-lab/alpaca",
 backend="trl", # Use TRL for GPT2 models
 task_type="instruction_following",
 column_mapping={"output": "response"}, # Map output column to response
 num_epochs=3,
 batch_size=4,
 learning_rate=2e-4,
 max_seq_length=1024,
 max_samples=1000,
 peft_enabled=True,
 lora_target_modules=["c_attn", "c_proj"], # For GPT2 models
 quantization={"load_in_4bit": True}
)

# Train
print("Starting training...")
trainer.train()

# Evaluate
print("Evaluating...")
metrics = trainer.evaluate()
print(f"Evaluation metrics: {metrics}")

# Save
model_path = trainer.save_model()
print(f"Model saved to: {model_path}")

# Test
result = trainer.predict("What is machine learning?")
print(f"Prediction: {result}")
```

### Complete Classification Example

```python
from aligntune.core.backend_factory import create_sft_trainer

# Create trainer
trainer = create_sft_trainer(
 model_name="distilbert-base-uncased",
 dataset_name="imdb",
 backend="trl", # TRL for classification
 task_type="text_classification",
 num_epochs=3,
 batch_size=8,
 learning_rate=5e-5,
 max_seq_length=512,
 max_samples=5000,
 text_column="text",
 label_column="label",
 num_labels=2
)

# Train
trainer.train()

# Evaluate
metrics = trainer.evaluate()
print(f"Accuracy: {metrics.get('eval_accuracy', 'N/A')}")

# Save
model_path = trainer.save_model()
```

## Next Steps

- [RL Training Guide](rl.md) - Learn about reinforcement learning training
- [Reward Functions Guide](reward-functions.md) - Explore reward functions for RL
- [Evaluation Guide](evaluation.md) - Comprehensive evaluation guide
- [Model Management](model-management.md) - Advanced model management

## Additional Resources

- [API Reference](../api-reference/core.md) - Complete API documentation
- [Examples](../examples/sft.md) - More SFT examples
- [Backend Selection](../getting-started/backend-selection.md) - Detailed backend guide
- [Configuration Guide](../getting-started/configuration.md) - Configuration reference