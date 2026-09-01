# User Guide Overview

Welcome to the AlignTune User Guide! This section provides comprehensive tutorials and guides for using AlignTune effectively.

---

## What is AlignTune?

AlignTune is a production-ready fine-tuning library for Large Language Models (LLMs) that supports:

- **Supervised Fine-Tuning (SFT)**: Including classification tasks
- **Reinforcement Learning (RL)**: Align models using 13+ RLHF algorithms — DPO, PPO, GRPO, GSPO, PACE, ORPO, SPIN, RAFT, and more; see the [Algorithms Overview](../algorithms/overview.md) for the full list
- **Multi-Backend Support**: Choose between TRL, Unsloth, and ES backends
- **Distillation**: Train student models natively
- **Advanced Adapters**: MoA, Text2LoRA/Doc2LoRA
- **Long Context**: Built-in support for RoPE scaling and attention variants
- **Compositions**: Multi-stage training pipelines that chain SFT/RL/distillation stages together (see [Production Compositions](../advanced/composition.md))
- **Model Merging**: Combine multiple fine-tuned models or LoRA adapters via mergekit (linear, task arithmetic, etc.): see [Model Merging](../advanced/merging.md)
- **Advisor CLI**: Deterministic VRAM/time/cost/carbon estimation and algorithm recommendations without needing GPU access (see [CLI Commands](../cli/commands.md#advisor-commands))
- **Alignment Auditing**: Automated alignment-drift detection and audit reporting via the `AlignmentAuditor` callback (see [examples/alignment_audit_example.py](https://github.com/Lexsi-Labs/aligntune/blob/main/examples/alignment_audit_example.py))

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

- **YAML files / JSON**: Declarative configuration passed to `create_*_trainer(config="path/to.yaml")`
- **Python API**: Direct function calls with keyword arguments
- **CLI**: Command-line interface

### 3. Reward System

50+ built-in reward functions for quality, safety, style, and task-specific metrics.

### 4. Evaluation System

Comprehensive evaluation with basic metrics, quality metrics, and safety metrics.

---

## Getting Started

### For SFT Training

1. **Start with [SFT Guide](sft.md)**: Complete guide to Supervised Fine-Tuning
2. **Learn [Configuration](../getting-started/configuration.md)**: Understand configuration options
3. **Explore [Examples](../examples/sft.md)**: See real-world examples

### For Distillation Training

1. **Start with [Distillation Guide](distillation.md)**: Choose Standard or SDFT
2. **Review [Distillation Parameters](../PARAMETERS.md#distillation-parameters)**: Check method-specific controls
3. **Use [Distillation Internals](../advanced/distillation.md)**: Method-routing logic and implementation gotchas

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

### Advanced Topics

- **[Adapters v3.3](../advanced/adapters.md)**: MoA, Text2LoRA/Doc2LoRA
- **[Long Context](../advanced/long-context.md)**: RoPE scaling and packing
- **[Distillation Internals](../advanced/distillation.md)**: Method-routing logic and TRL config classes

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
- **[Examples](../examples/overview.md)**: Code examples and tutorials
- **[Advanced Topics](../advanced/architecture.md)**: Architecture and advanced usage
- **[Contributing](../contributing/guide.md)**: Contribute to AlignTune

---

**Ready to start? Begin with [SFT Guide](sft.md) or [RL Guide](rl.md)!**
