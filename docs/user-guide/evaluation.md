# Evaluation Guide

Complete guide to evaluating fine-tuned models with AlignTune, covering training evaluation, standalone evaluation, benchmarks, and custom evaluation tasks.

## Overview

AlignTune provides comprehensive evaluation capabilities:

1. **Training Evaluation** - Evaluate during training
2. **Standalone Evaluation** - Evaluate saved models
3. **Benchmark Evaluation** - Standard benchmarks (lm-eval)
4. **Custom Evaluation** - Task-specific evaluation
5. **RL Evaluation** - DPO, PPO, GRPO evaluation

## Quick Start

### Basic Training Evaluation

```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
    model_name="microsoft/DialoGPT-medium",
    dataset_name="tatsu-lab/alpaca",
    backend="trl",
    num_epochs=3
)

# Train with automatic evaluation
trainer.train()

# Evaluate on validation set
metrics = trainer.evaluate()
print(metrics)
```

### Standalone Model Evaluation

```python
from datasets import load_dataset

# Load evaluation dataset
eval_dataset = load_dataset("tatsu-lab/alpaca", split="test")

# Evaluate
metrics = trainer.evaluate(eval_dataset=eval_dataset)
print(f"Loss: {metrics['eval_loss']}")
print(f"Perplexity: {metrics.get('eval_perplexity', 'N/A')}")
```

### RL Model Evaluation

```python
from aligntune.eval.core import EvalConfig, run_eval

# Evaluate DPO model
config = EvalConfig(
    model_path="./output/dpo_model",
    output_dir="./eval_results",
    task_type="dpo",
    data_task_type="dpo",
    metrics=["reward_margin", "preference_accuracy", "win_rate"],
    dataset_name="Anthropic/hh-rlhf",
    split="test",
    max_samples=100,
    use_lora=True,
    base_model="microsoft/phi-2"
)

results = run_eval(config)
print(f"Win Rate: {results.get('win_rate', 0.0):.2%}")
```

## Evaluation Types

### 1. Training Evaluation

Automatic evaluation during training:

```python
trainer = create_sft_trainer(
    model_name="microsoft/DialoGPT-medium",
    dataset_name="tatsu-lab/alpaca",
    backend="trl",
    num_epochs=3,
    eval_interval=100,  # Evaluate every 100 steps
    save_interval=500   # Save every 500 steps
)

trainer.train()  # Evaluation runs automatically
```

### 2. Standalone Evaluation

Evaluate after training:

```python
# After training
metrics = trainer.evaluate()

# With custom dataset
from datasets import load_dataset
eval_dataset = load_dataset("tatsu-lab/alpaca", split="test")
metrics = trainer.evaluate(eval_dataset=eval_dataset)

# With custom metric prefix
metrics = trainer.evaluate(
    eval_dataset=eval_dataset,
    metric_key_prefix="test"
)
```

### 3. Zero-Shot Evaluation

Generate predictions for test prompts:

```python
# Single prediction
result = trainer.predict("What is machine learning?")
print(result)

# Batch predictions
prompts = [
    "What is AI?",
    "Explain deep learning",
    "What is NLP?"
]

results = trainer.predict(prompts)
for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}")
    print(f"A: {result}\n")
```

### 4. Benchmark Evaluation

Run standard benchmarks with lm-eval:

```python
from aligntune.eval.lm_eval_integration import LMEvalRunner

runner = LMEvalRunner()

# Run standard benchmarks
results = runner.run_benchmark(
    model_name="your-model-name",
    tasks=["hellaswag", "arc", "mmlu"],
    batch_size=1
)

print(results)
```

### 5. Custom Evaluation

Create custom evaluation tasks:

```python
from aligntune.eval.core import EvalRunner, EvalTask, TaskCategory

# Create custom task
task = EvalTask(
    name="custom_sentiment",
    category=TaskCategory.SENTIMENT_ANALYSIS,
    description="Custom sentiment analysis task",
    dataset_name="imdb",
    input_column="text",
    target_column="label",
    metrics=["accuracy", "f1"]
)

# Run evaluation
runner = EvalRunner()
results = runner.run_evaluation(model, [task])
```

### 6. RL Evaluation

Evaluate RL-trained models (DPO, PPO, GRPO):

#### DPO Evaluation

```python
from aligntune.eval.core import EvalConfig, run_eval

# Evaluate DPO model
config = EvalConfig(
    model_path="./output/dpo_model",
    output_dir="./eval_results/dpo",
    device="cuda",
    
    # Task configuration
    task_type="dpo",
    data_task_type="dpo",
    
    # DPO metrics
    metrics=["reward_margin", "preference_accuracy", "win_rate"],
    reference_model_path="microsoft/phi-2",  # For KL divergence
    
    # Dataset
    dataset_name="Anthropic/hh-rlhf",
    split="test",
    max_samples=100,
    
    # Generation
    max_length=1536,
    temperature=0.0,
    batch_size=2,
    
    # Model loading
    use_lora=True,
    base_model="microsoft/phi-2",
    use_unsloth=True
)

results = run_eval(config)
print(f"Win Rate: {results.get('win_rate', 0.0):.2%}")
```

#### PPO Evaluation

```python
config = EvalConfig(
    model_path="./output/ppo_model",
    output_dir="./eval_results/ppo",
    
    # Task configuration
    task_type="text",
    data_task_type="sft",
    
    # Metrics
    metrics=["perplexity", "reward_accuracy"],
    
    # Dataset
    dataset_name="HuggingFaceH4/ultrachat_200k",
    split="test_sft",
    max_samples=100,
    
    # Generation
    max_length=512,
    temperature=0.7,
    batch_size=8
)

results = run_eval(config)
print(f"Perplexity: {results.get('perplexity', 0.0):.4f}")
```

#### Before/After Comparison

```python
from aligntune.core.backend_factory import create_rl_trainer

# Train DPO model
trainer = create_rl_trainer(
    model_name="microsoft/phi-2",
    dataset_name="Anthropic/hh-rlhf",
    algorithm="dpo",
    backend="trl",
    num_epochs=1,
    use_lora=True
)
trainer.train()
trained_path = trainer.save_model()

# Evaluate base model
base_config = EvalConfig(
    model_path="microsoft/phi-2",
    task_type="dpo",
    data_task_type="dpo",
    metrics=["reward_margin", "win_rate"],
    dataset_name="Anthropic/hh-rlhf",
    split="test",
    use_lora=False,
    output_dir="./eval_results/base"
)
base_results = run_eval(base_config, trainer.dataset_dict)

# Evaluate trained model
trained_config = EvalConfig(
    model_path=trained_path,
    task_type="dpo",
    data_task_type="dpo",
    metrics=["reward_margin", "win_rate"],
    reference_model_path="microsoft/phi-2",
    dataset_name="Anthropic/hh-rlhf",
    split="test",
    use_lora=True,
    base_model="microsoft/phi-2",
    output_dir="./eval_results/trained"
)
trained_results = run_eval(trained_config, trainer.dataset_dict)

# Compare
print(f"Base: {base_results['win_rate']:.2%}")
print(f"Trained: {trained_results['win_rate']:.2%}")
print(f"Improvement: {trained_results['win_rate'] - base_results['win_rate']:.2%}")
```

## Evaluation Metrics

### Classification Metrics

```python
# For text classification
metrics = trainer.evaluate(eval_dataset=eval_dataset)
# Returns: accuracy, f1, precision, recall
```

### Generation Metrics

```python
# For text generation
metrics = trainer.evaluate(eval_dataset=eval_dataset)
# Returns: perplexity, loss, BLEU, ROUGE (if applicable)
```

### RL Metrics

#### DPO Metrics

```python
metrics = [
    "reward_margin",       # Difference between chosen/rejected rewards
    "preference_accuracy", # How often model prefers chosen
    "win_rate",           # % matching human preferences
    "kl_divergence",      # KL from reference (needs reference_model_path)
    "log_ratio",          # Log probability ratio
    "implicit_reward",    # Implicit reward signal
    "calibration"         # Calibration score
]
```

#### PPO Metrics

```python
metrics = [
    "perplexity",        # Language modeling quality
    "reward_accuracy",   # Alignment with reward model
    "policy_entropy"     # Policy output diversity
]
```

#### RL Metrics

```python
metrics = [
    "kl_divergence",     # KL divergence from reference
    "reward_accuracy",   # Reward model alignment
    "policy_entropy"     # Output diversity
]
```

### Custom Metrics

```python
from aligntune.eval.core import EvalTask, TaskCategory

task = EvalTask(
    name="custom_task",
    category=TaskCategory.TEXT_GENERATION,
    metrics=["accuracy", "f1", "custom_metric"],
    custom_metrics=[your_custom_metric_function]
)
```

## Evaluation Configuration

### Training Evaluation Config

```python
trainer = create_sft_trainer(
    model_name="microsoft/DialoGPT-medium",
    dataset_name="tatsu-lab/alpaca",
    # Evaluation settings
    eval_interval=100,              # Evaluate every N steps
    save_interval=500,              # Save every N steps
    load_best_model_at_end=True,   # Load best model
    metric_for_best_model="eval_loss",  # Metric to track
    greater_is_better=False         # Lower is better for loss
)
```

### Standalone Evaluation Config

```python
from aligntune.eval.core import EvalConfig, EvalType, TaskCategory

config = EvalConfig(
    eval_type=EvalType.STANDALONE,
    task_categories=[TaskCategory.TEXT_GENERATION],
    metrics=["accuracy", "f1", "bleu", "rouge"],
    batch_size=32,
    max_samples=1000,
    device="auto",
    precision="bf16",
    output_dir="./eval_results",
    save_predictions=True,
    save_metrics=True
)
```

### RL Evaluation Config

```python
from aligntune.eval.core import EvalConfig, run_eval

config = EvalConfig(
    # Model configuration
    model_path="./model",
    output_dir="./eval_results",
    device="cuda",
    
    # Task configuration
    task_type="dpo",              # Evaluation task: dpo, ppo, grpo, text, math, code
    data_task_type="dpo",         # Data schema: dpo, sft, grpo
    
    # Metrics
    metrics=["reward_margin", "win_rate"],
    reference_model_path=None,    # Reference model (for KL)
    reward_model=None,            # Reward model (optional)
    
    # Dataset
    dataset_name="dataset_name",
    dataset_config=None,
    split="test",
    max_samples=100,
    
    # Generation
    max_length=2048,              # Context window
    max_new_tokens=512,           # Generation length
    temperature=0.0,              # Sampling temperature
    batch_size=8,
    
    # Model loading
    use_lora=False,
    base_model=None,
    use_unsloth=False,
    use_vllm=False,
    precision="bf16",
    
    # Advanced
    column_mapping=None,
    trust_remote_code=True
)

results = run_eval(config)
```

## Task Type Reference

Supported `task_type` values for evaluation:

| Task Type | Description | Default Metrics | Data Schema |
|-----------|-------------|----------------|-------------|
| `"text"` | Text generation | perplexity, accuracy | sft |
| `"math"` | Math problems | math_accuracy, pass_at_k | sft |
| `"code"` | Code generation | pass_at_k | code |
| `"dpo"` | DPO models | reward_margin, preference_accuracy, win_rate | dpo |
| `"ppo"` | PPO models | perplexity, reward_accuracy | sft |
| `"generic"` | Generic evaluation | perplexity, accuracy | sft |

**Note**: For RL evaluation, `task_type` and `data_task_type` must match (e.g., both `"dpo"` for DPO models).

## Complete Examples

### SFT Evaluation

```python
from aligntune.core.backend_factory import create_sft_trainer
from datasets import load_dataset

# Create trainer
trainer = create_sft_trainer(
    model_name="microsoft/DialoGPT-medium",
    dataset_name="tatsu-lab/alpaca",
    backend="trl",
    num_epochs=3,
    eval_interval=100
)

# Train
trainer.train()

# Evaluate on test set
test_dataset = load_dataset("tatsu-lab/alpaca", split="test")
metrics = trainer.evaluate(eval_dataset=test_dataset)
print(f"Test metrics: {metrics}")

# Zero-shot evaluation
prompts = ["What is AI?", "Explain machine learning"]
results = trainer.predict(prompts)
for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}\nA: {result}\n")
```

### RL Evaluation

```python
from aligntune.core.backend_factory import create_rl_trainer
from aligntune.eval.core import EvalConfig, run_eval

MODEL_NAME = "microsoft/phi-2"
DATASET = "Anthropic/hh-rlhf"
OUTPUT_DIR = "./output/dpo_phi2"

# Train DPO model
trainer = create_rl_trainer(
    model_name=MODEL_NAME,
    dataset_name=DATASET,
    algorithm="dpo",
    backend="trl",
    output_dir=OUTPUT_DIR,
    num_epochs=1,
    use_lora=True
)
trainer.train()
trained_path = trainer.save_model()

# Evaluate base model
base_config = EvalConfig(
    model_path=MODEL_NAME,
    output_dir=f"{OUTPUT_DIR}/eval_base",
    task_type="dpo",
    data_task_type="dpo",
    metrics=["reward_margin", "preference_accuracy", "win_rate"],
    dataset_name=DATASET,
    split="test",
    max_samples=100,
    use_lora=False,
    use_unsloth=True
)
base_results = run_eval(base_config, trainer.dataset_dict)

# Evaluate trained model
trained_config = EvalConfig(
    model_path=trained_path,
    output_dir=f"{OUTPUT_DIR}/eval_trained",
    task_type="dpo",
    data_task_type="dpo",
    metrics=["reward_margin", "preference_accuracy", "win_rate"],
    reference_model_path=MODEL_NAME,
    dataset_name=DATASET,
    split="test",
    max_samples=100,
    use_lora=True,
    base_model=MODEL_NAME,
    use_unsloth=True
)
trained_results = run_eval(trained_config, trainer.dataset_dict)

# Print comparison
print(f"\nBase Model:")
print(f"  Win Rate: {base_results.get('win_rate', 0.0):.2%}")
print(f"\nTrained Model:")
print(f"  Win Rate: {trained_results.get('win_rate', 0.0):.2%}")
print(f"\nImprovement: {trained_results['win_rate'] - base_results['win_rate']:.2%}")
```

### Benchmark Evaluation

```python
from aligntune.eval.lm_eval_integration import LMEvalRunner

runner = LMEvalRunner()

# Run multiple benchmarks
results = runner.run_benchmark(
    model_name="your-model-name",
    tasks=[
        "hellaswag",    # Commonsense reasoning
        "arc",          # Science questions
        "mmlu",         # Multitask language understanding
        "truthfulqa"    # Truthful question answering
    ],
    batch_size=1,
    limit=100  # Limit samples per task
)

# Print results
for task, metrics in results.items():
    print(f"{task}: {metrics}")
```

### Custom Task Evaluation

```python
from aligntune.eval.core import EvalRunner, EvalTask, TaskCategory

# Create custom tasks
tasks = [
    EvalTask(
        name="sentiment_analysis",
        category=TaskCategory.SENTIMENT_ANALYSIS,
        description="IMDB sentiment analysis",
        dataset_name="imdb",
        input_column="text",
        target_column="label",
        metrics=["accuracy", "f1"]
    ),
    EvalTask(
        name="text_generation",
        category=TaskCategory.TEXT_GENERATION,
        description="Alpaca instruction following",
        dataset_name="tatsu-lab/alpaca",
        input_column="instruction",
        target_column="output",
        metrics=["bleu", "rouge"]
    )
]

# Run evaluation
runner = EvalRunner()
results = runner.run_evaluation(model, tasks)

# Print results
for task_name, metrics in results.items():
    print(f"{task_name}: {metrics}")
```

## Best Practices

### 1. Evaluation During Training

- Set appropriate `eval_interval` (100-500 steps)
- Use validation split for evaluation
- Track best model with `load_best_model_at_end`

### 2. Comprehensive Evaluation

- Evaluate on multiple metrics
- Use both automatic and manual evaluation
- Test on diverse datasets

### 3. Zero-Shot Evaluation

- Test on unseen prompts
- Evaluate qualitative samples
- Check for common failure modes

### 4. Benchmark Evaluation

- Use standard benchmarks for comparison
- Run multiple benchmarks
- Document results for reproducibility

### 5. Custom Evaluation

- Create task-specific evaluation
- Use domain-specific metrics
- Validate on real-world data

### 6. RL Evaluation

- Always compare base and trained models
- Use consistent evaluation settings
- Use greedy decoding (temperature=0.0) for deterministic results
- Reuse dataset_dict when possible
- Match task_type and data_task_type for RL models

## Troubleshooting

### Evaluation Too Slow

```python
# Reduce batch size or max samples
metrics = trainer.evaluate(
    eval_dataset=eval_dataset,
    batch_size=16,    # Reduce from default
    max_samples=500   # Limit samples
)
```

### Out of Memory

```python
# Use smaller batch size
config = EvalConfig(
    batch_size=8,      # Smaller batch
    max_samples=100,   # Limit samples
    precision="fp16"   # Lower precision
)
```

### Missing Metrics

```python
# Specify metrics explicitly
task = EvalTask(
    name="my_task",
    category=TaskCategory.TEXT_CLASSIFICATION,
    metrics=["accuracy", "f1", "precision", "recall"]
)
```

### RL Evaluation Issues

```python
# Missing required columns for DPO
# Fix: Ensure task types match
config = EvalConfig(
    task_type="dpo",
    data_task_type="dpo"  # Must match
)

# KL divergence requires reference model
config = EvalConfig(
    metrics=["kl_divergence"],
    reference_model_path="microsoft/phi-2"
)

# LoRA adapter not loading
config = EvalConfig(
    model_path="./lora_adapter",
    use_lora=True,
    base_model="microsoft/phi-2"
)
```

## Next Steps

- [SFT Guide](sft.md) - SFT training and evaluation
- [RL Guide](rl.md) - RL training and evaluation
- [Model Management](model-management.md) - Model saving and loading

## Additional Resources

- [API Reference](../api-reference/core.md) - API documentation
- [Examples](../examples/sft.md) - Code examples
- [Evaluation System](../api-reference/core.md) - Detailed system docs