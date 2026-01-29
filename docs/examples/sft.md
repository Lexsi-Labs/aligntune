# SFT Examples

Comprehensive examples for Supervised Fine-Tuning with AlignTune.

## Basic Examples

### 1. Basic SFT Training

Simple SFT training example:

```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="tatsu-lab/alpaca",
 backend="trl",
 num_epochs=3,
 batch_size=4,
 learning_rate=2e-4,
 max_seq_length=512,
 max_samples=1000,
lora_target_modules=["c_attn", "c_proj"], # For GPT2 based models
)

trainer.train()
model_path = trainer.save_model()
```

**Expected Output:**
```
Training started...
Epoch 1/3: 100%|| 250/250 [05:23<00:00, loss=2.34]
Epoch 2/3: 100%|| 250/250 [05:18<00:00, loss=1.89]
Epoch 3/3: 100%|| 250/250 [05:21<00:00, loss=1.67]
Model saved to: ./output/model
```

### 2. Instruction Following

Train model to follow instructions:

```python
trainer = create_sft_trainer(
 model_name="Qwen/Qwen3-0.6B", # Use Qwen/Llama instead of GPT2
 dataset_name="tatsu-lab/alpaca",
 backend="trl", # Use TRL for GPT2 models
 task_type="instruction_following",
 column_mapping={"output": "response"}, # Map output column to response
 num_epochs=3,
 batch_size=4,
 learning_rate=2e-4,
 max_seq_length=1024,
 peft_enabled=True,
 instruction_column="instruction",
 response_column="output",
 input_column="input"
)

trainer.train()
```

### 3. Text Classification

**Note:** For GPT2 models, use `backend="trl"` and `lora_target_modules=["c_attn", "c_proj"]`. GPT2 is not supported by Unsloth.

Train classification model:

```python
trainer = create_sft_trainer(
 model_name="distilbert-base-uncased",
 dataset_name="imdb",
 backend="trl",
 task_type="text_classification",
 num_epochs=3,
 batch_size=8,
 learning_rate=5e-5,
 max_seq_length=512,
 text_column="text",
 label_column="label",
 num_labels=2
)

trainer.train()
metrics = trainer.evaluate()
print(f"Accuracy: {metrics.get('eval_accuracy', 'N/A')}")
```

## Advanced Examples

### 4. LoRA Fine-Tuning

Efficient fine-tuning with LoRA:

```python
trainer = create_sft_trainer(
 model_name="meta-llama/Llama-2-7b-hf",
 dataset_name="tatsu-lab/alpaca",
 backend="unsloth",
 peft_enabled=True,
 lora_r=16,
 lora_alpha=32,
 lora_dropout=0.1,
 quantization={"load_in_4bit": True},
 num_epochs=3,
 batch_size=2,
 learning_rate=2e-4
)

trainer.train()
```

### 5. Sequence Packing

Efficient training with sequence packing:

```python
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="tatsu-lab/alpaca",
 backend="trl", # Use TRL for GPT2 models
 packing=True,
 packing_strategy="bfd",
 padding_free=True,
 max_seq_length=2048,
 num_epochs=3,
 batch_size=4
)

trainer.train()
```

### 6. Chat Completion

Train conversational model:

```python
trainer = create_sft_trainer(
 model_name="mistralai/Mistral-7B-v0.1",
 dataset_name="HuggingFaceH4/ultrachat_200k",
 split="train_sft", # Dataset split
 backend="unsloth",
 task_type="chat_completion",
 num_epochs=2,
 batch_size=2,
 learning_rate=2e-4,
 max_seq_length=2048,
 gradient_accumulation_steps=4,
 messages_column="messages",
 chat_template="auto"
)

trainer.train()
```

## Complete Workflow Example

```python
from aligntune.core.backend_factory import create_sft_trainer
from datasets import load_dataset

# Load dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train")

# Create trainer
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="tatsu-lab/alpaca",
 backend="trl",
 num_epochs=3,
 batch_size=4,
 learning_rate=2e-4,
 max_seq_length=512,
 max_samples=1000,
 eval_interval=100,
 save_interval=500
)

# Train
print("Starting training...")
results = trainer.train()
print(f"Training completed: {results}")

# Evaluate
print("Evaluating...")
test_dataset = load_dataset("tatsu-lab/alpaca", split="test")
metrics = trainer.evaluate(eval_dataset=test_dataset)
print(f"Evaluation metrics: {metrics}")

# Save model
model_path = trainer.save_model()
print(f"Model saved to: {model_path}")

# Test predictions
print("Testing predictions...")
prompts = [
 "What is machine learning?",
 "Explain deep learning",
 "What is NLP?"
]

results = trainer.predict(prompts)
for prompt, result in zip(prompts, results):
 print(f"Q: {prompt}")
 print(f"A: {result}\n")

# Push to Hub (optional)
# url = trainer.push_to_hub("username/my-model")
# print(f"Model available at: {url}")
```

## Running Examples

### From Command Line

```bash
# Basic SFT
python examples/sft_customer_support_trl/train_sft_direct_api.py

# Instruction following
python examples/sft_financial_summarization_trl/train_sft_direct_api.py

```


## Tips

1. **Start Small**: Use `max_samples=1000` for testing
2. **Monitor Training**: Set `eval_interval` to track progress
3. **Save Regularly**: Use `save_interval` for checkpoints
4. **Choose Backend**: Use Unsloth for speed, TRL for compatibility
5. **Memory Management**: Use LoRA and quantization for large models

## Next Steps

- [RL Examples](rl.md) - RL training examples
- [Advanced Examples](advanced.md) - Advanced use cases
- [SFT Guide](../user-guide/sft.md) - Complete SFT guide