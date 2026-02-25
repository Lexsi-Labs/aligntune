# User Guide Overview

Welcome to the AlignTune User Guide! This section provides comprehensive tutorials and guides for using AlignTune effectively.

---

## What is AlignTune?

AlignTune is a production-ready fine-tuning library for Large Language Models (LLMs) that supports:

- **Supervised Fine-Tuning (SFT)**: Adapt pre-trained models to specific tasks
- **Reinforcement Learning (RL)**: Align models with human preferences using RLHF algorithms
- **Multi-Backend Support**: Choose between TRL (reliable) and Unsloth (faster) backends
- **RL Algorithms**: DPO, PPO, GRPO, GSPO, DAPO, Dr. GRPO

---

## Core Components

### 1. Backend Factory

The `BackendFactory` is the main entry point for creating trainers:

```python
from aligntune.core.backend_factory import create_sft_trainer, create_rl_trainer

# Create SFT trainer
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 backend="trl"
)

# Create RL trainer
trainer = create_rl_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dpo",
 backend="trl"
)
```

### 2. Configuration System

AlignTune supports three configuration methods:

- **Python API**: Direct function calls with keyword arguments
- **YAML Files**: Declarative configuration files
- **CLI**: Command-line interface

### 3. Reward System

27+ built-in reward functions for quality, safety, style, and task-specific metrics.

### 4. Evaluation System

Comprehensive evaluation with basic metrics, quality metrics, and safety metrics.

---

## Getting Started

### For SFT Training

1. **Start with [SFT Guide](sft.md)**: Complete guide to Supervised Fine-Tuning
2. **Learn [Configuration](../getting-started/configuration.md)**: Understand configuration options
3. **Explore [Examples](../examples/sft.md)**: See real-world examples

### For RL Training

1. **Start with [RL Guide](rl.md)**: Complete guide to Reinforcement Learning
2. **Learn [Reward Functions](reward-functions.md)**: Understand reward functions
3. **Explore [RL Examples](../examples/rl.md)**: See RL training examples

---

## Guide Structure

### Supervised Fine-Tuning (SFT)

- **[SFT Guide](sft.md)**: Complete SFT training guide
 - Task types (instruction following, classification, chat)
 - Configuration options
 - Best practices
 - Examples

### Reinforcement Learning (RL)

- **[RL Guide](rl.md)**: Complete RL training guide
 - Algorithm overview (DPO, PPO, GRPO, etc.)
 - Configuration options
 - Best practices
 - Examples

### Reward Functions

- **[Reward Functions](reward-functions.md)**: Using reward functions
 - Built-in reward functions
 - Custom reward functions
 - Composite rewards

### Reward Model Training

- **[Reward Model Training](reward-model-training.md)**: Training custom reward models
 - Training from rule-based functions
 - Integration with PPO
 - Best practices

### Evaluation

- **[Evaluation](evaluation.md)**: Model evaluation
 - Basic metrics
 - Quality metrics
 - Safety metrics
 - Task-specific metrics

### Model Management

- **[Model Management](model-management.md)**: Saving, loading, and sharing models
 - Local model saving
 - HuggingFace Hub integration
 - Checkpoint management

### Sample Logging

- **[Sample Logging](sample-logging.md)**: Qualitative sample generation
 - Configuration
 - Best practices
 - Examples

### Troubleshooting

- **[Troubleshooting](troubleshooting.md)**: Common issues and solutions
 - Backend issues
 - CUDA/GPU issues
 - Configuration issues
 - Training issues

---

## Quick Reference

### Common Patterns

#### SFT Training Pattern

```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 backend="trl",
 num_epochs=3
)
trainer.train()
trainer.evaluate()
trainer.save_model()
```

#### DPO Training Pattern

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dpo",
 backend="trl",
 num_epochs=1
)
trainer.train()
trainer.evaluate()
trainer.save_model()
```

#### PPO Training Pattern

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="ppo",
 backend="unsloth",
 reward_model_name="your-reward-model",
 num_epochs=1
)
trainer.train()
trainer.evaluate()
trainer.save_model()
```

---

## Next Steps

1. **[Getting Started](../getting-started/installation.md)**: Installation and setup
2. **[Basic Concepts](../getting-started/basic-concepts.md)**: Core concepts
3. **[SFT Guide](sft.md)**: Start with SFT training
4. **[RL Guide](rl.md)**: Explore RL training
5. **[Examples](../examples/overview.md)**: See real-world examples

---

## Additional Resources

- **[API Reference](../api-reference/overview.md)**: Complete API documentation
- **[CLI Overview](../cli/overview.md)**: Command-line interface
- **[API Reference](../api-reference/overview.md)**: Complete API documentation
- **[Examples](../examples/overview.md)**: Code examples and tutorials
- **[Advanced Topics](../advanced/architecture.md)**: Architecture and advanced usage
- **[Contributing](../contributing/guide.md)**: Contribute to AlignTune

---

**Ready to start? Begin with [SFT Guide](sft.md) or [RL Guide](rl.md)!**