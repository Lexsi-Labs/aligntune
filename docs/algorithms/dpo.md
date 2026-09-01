# DPO (Direct Preference Optimization)

Direct Preference Optimization (DPO) is a popular RLHF algorithm that optimizes language models directly on preference data without requiring a separate reward model.

---

## Overview

DPO eliminates the need for a reward model by directly optimizing the policy on preference pairs (chosen vs rejected responses). This makes it simpler and faster than PPO-based approaches.

### Key Features

- **No Reward Model**: Works directly with preference pairs
- **Fast Training**: Simpler optimization than PPO
- **Both Backends**: Supported by TRL and Unsloth
- **Widely Used**: Industry-standard for preference learning

---

## How It Works

DPO optimizes the policy to increase the likelihood of chosen responses while decreasing the likelihood of rejected responses, using a KL divergence constraint to prevent the policy from deviating too far from the reference model.

### Mathematical Formulation

The DPO objective maximizes:

```
L_DPO = E[log σ(β * (log π_θ(y_w | x) - log π_ref(y_w | x) - log π_θ(y_l | x) + log π_ref(y_l | x)))]
```

Where:
- `π_θ`: Policy model
- `π_ref`: Reference model
- `y_w`: Chosen response
- `y_l`: Rejected response
- `β`: Temperature parameter

---

## Usage

### Basic DPO Training

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dpo",
 backend="trl",
 num_epochs=1,
 batch_size=4,
 learning_rate=1e-6,
 beta=0.1 # DPO temperature parameter
)

trainer.train()
```

### DPO with Unsloth Backend

```python
trainer = create_rl_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dpo",
 backend="unsloth",
 num_epochs=1,
 batch_size=1,
 learning_rate=1e-6,
 beta=0.1
)

trainer.train()
```

### DPO with Custom Configuration

```python
trainer = create_rl_trainer(
 model_name="Qwen/Qwen3-0.6B",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dpo",
 backend="trl",
 num_epochs=3,
 batch_size=4,
 learning_rate=5e-6,
 beta=0.1,
 max_seq_length=512,
 label_smoothing=0.0,
 loss_type="sigmoid", # TRL normalizes this to ["sigmoid"]
 # For combined losses:
 # loss_type=["sigmoid", "sft"],
 # loss_weights=[0.8, 0.2],
)
```

---

## Configuration Parameters

Config class: `trl.DPOConfig`. DPO also accepts every parameter in the
[common RL parameters](../PARAMETERS.md#rl-reinforcement-learning-common-parameters)
reference (batch size, learning rate, epochs, LoRA settings, etc.): any
`TrainingConfig` field matching a real `DPOConfig` field is forwarded
automatically, even if not listed below.

Use `max_seq_length` as the overall TRL `DPOConfig.max_length` limit.
`max_prompt_length` and `max_completion_length` are not fields in the pinned
TRL `DPOConfig`. DPO uses a reference model through
`DPOTrainer(ref_model=...)` by default; `reference_free` is not a supported
field in this TRL DPO configuration.

--8<-- "docs/PARAMETERS.md:dpo"

---

## Dataset Format

DPO requires preference pairs in the following format:

```python
{
 "prompt": "What is machine learning?",
 "chosen": "Machine learning is a subset of AI...",
 "rejected": "I don't know about that topic."
}
```

### Supported Datasets

- **Anthropic/hh-rlhf**: Human preference dataset
- **OpenAssistant/oasst1**: OpenAssistant conversations
- Custom datasets with `prompt`, `chosen`, `rejected` columns

---

## Best Practices

### 1. Beta Parameter Tuning

The `beta` parameter controls the strength of the KL penalty:
- **Low beta (0.01-0.1)**: Stronger preference learning, risk of overfitting
- **Medium beta (0.1-0.5)**: Balanced learning
- **High beta (0.5-1.0)**: More conservative, stays closer to reference

```python
# Conservative training
beta=0.5

# Aggressive training
beta=0.01
```

### 2. Learning Rate

DPO typically requires lower learning rates than SFT:
- Start with `1e-6` to `5e-6`
- Adjust based on loss convergence

### 3. Batch Size

- Use batch size 1-4 for smaller models
- Use batch size 4-8 for larger models
- Consider gradient accumulation for effective larger batches

### 4. Number of Epochs

- Usually 1-3 epochs are sufficient
- Monitor validation loss to avoid overfitting

---

## Evaluation

DPO evaluation includes:

```python
# Note: aligntune.eval.core also defines an unrelated EvalConfig class
# (used with EvalRunner/EvalTask); run_eval() and the model_path-style
# EvalConfig used here both live in aligntune.eval.runner.
from aligntune.eval.runner import EvalConfig, run_eval

MODEL_NAME = "microsoft/phi-2"
DATASET = "Anthropic/hh-rlhf"
OUTPUT_DIR = "./output/dpo_phi2"
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
trained_results = run_eval(trained_config)
# Note: For the base model you don't need to pass model_path; also, if the
# model wasn't trained with LoRA, you don't need to pass base_model either.
```

The `dpo_eval_*` settings control optional AlignTune-side DPO evaluation and
are separate from TRL's periodic `DPOTrainer` validation metrics.

---

## Common Issues

### Loss Not Decreasing

**Solutions**:
- Reduce learning rate
- Increase beta (stronger KL penalty)
- Check dataset quality
- Verify preference pairs are correct

### Overfitting

**Solutions**:
- Increase beta
- Reduce number of epochs
- Use more regularization
- Increase dataset size

---

## Comparison with Other Algorithms

| Feature | DPO | PPO | GRPO |
|---------|-----|-----|------|
| Reward Model | No | Yes | No |
| Training Speed | Fast | Slow | Medium |
| Setup Complexity | Low | High | Medium |
| Preference Data | Required | Optional | Required |

---

## References

- [Direct Preference Optimization Paper](https://arxiv.org/abs/2305.18290)
- [TRL DPO Documentation](https://huggingface.co/docs/trl/dpo_trainer)

---

## Next Steps

- **[PPO Guide](ppo.md)** - Learn about Proximal Policy Optimization
- **[GRPO Guide](grpo.md)** - Learn about Group Relative Policy Optimization
- **[Backend Selection](../getting-started/backend-selection.md)** - Choose the right backend
