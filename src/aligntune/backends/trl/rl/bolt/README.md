# BOLT: Baseline-Optimized Learning Technique

A GRPO backend extension that implements curriculum learning and persistent baselines for improved reinforcement learning from human feedback (RLHF) training.

## Overview

BOLT enhances standard GRPO training with two key innovations:

1. **Uncertainty-based Curriculum Sampling**: Focuses training on prompts where the model is most uncertain, maximizing learning efficiency
2. **Persistent Per-Prompt Baselines**: Maintains a running estimate v̂(x) of success probability for each prompt, enabling stable advantage computation even with small group sizes (K=1-2)

Both features can be enabled independently or together.

---

## Theoretical Background

### Curriculum Sampling

Standard GRPO samples prompts uniformly. BOLT instead samples proportionally to uncertainty:

```
w(x) = √(v̂(x)(1 - v̂(x))) + ε
```

Where:
- `v̂(x)` = estimated success probability for prompt x
- `ε` = floor to ensure all prompts have some probability (default: 0.05)

This focuses training on:
- **Learning edge prompts** (v̂ ≈ 0.5): Maximum uncertainty, maximum learning signal
- Avoids wasting compute on:
  - Already mastered prompts (v̂ → 1)
  - Too difficult prompts (v̂ → 0)

### Persistent Baseline with KL-Adaptive Forgetting

Standard GRPO computes advantages within a group:
```
A_i = r_i - mean(r_group)
```

This requires large group sizes (K=8+) for stable estimates. BOLT instead uses persistent baselines:
```
A = r - v̂(x)
```

Where v̂(x) is updated with exponential moving average and **KL-adaptive forgetting** (SPO Equation 5):

```
ρ(KL) = 2^(-KL / D_half)
```

- When KL is low (policy stable): ρ → 1, slow forgetting, use history
- When KL is high (policy shifting): ρ → 0, fast forgetting, adapt quickly

This enables:
- **Small group sizes (K=1-2)**: Baseline provides stability
- **Faster dataset iteration**: Cover more prompts per epoch
- **Automatic adaptation**: KL-based forgetting tracks policy changes

---

## Architecture

```
aligntune/backends/trl/rl/bolt/
├── __init__.py          # Module exports
├── baseline.py          # BaselineStore, UnifiedBaseline
├── curriculum.py        # Sampler, Dataset wrapper, Callbacks
├── bolt.py              # TRLBoltTrainer
└── README.md            # This file

aligntune/backends/unsloth/rl/bolt/
├── __init__.py          # Module exports
└── bolt.py              # UnslothBoltTrainer (reuses TRL components)

examples/bolt_training/
└── precompute_baseline.py  # Warm-start baseline computation
```

### Component Details

#### `baseline.py`

**`BaselineStore`**: Core Beta-Bernoulli tracker
- Maintains (α, β, kl_ema) per prompt
- `get(key)` → (v̂, N_eff) - pre-update read
- `update(key, reward, kl)` → applies KL-adaptive forgetting
- Save/load support (pickle format)

**`UnifiedBaseline`**: Extended baseline for curriculum + advantages
- Inherits BaselineStore
- `get_sampling_weight(key)` → √(v̂(1-v̂)) + ε
- `get_curriculum_stats()` → metrics for logging
- `load_from_json(path)` → warm-start from v̂ dictionary

**`make_prompt_key(prompt)`**: Creates stable keys for prompts

#### `curriculum.py`

**`DynamicWeightedSampler`**: Weighted sampling from baseline
- Computes weights from `baseline.get_sampling_weight()`
- Supports epoch-based caching
- `update_weights()` → refresh from current baseline state

**`DynamicallySampledDataset`**: Dataset wrapper with refreshable indices
- `refresh_indices()` → called by callback to resample

**`BaselineRewardWrapper`**: Computes advantages A = r - v̂(x)
- Pre-update reads for advantage computation
- Queues updates for post-step application
- Optional normalization

**`CurriculumCallback`**: HuggingFace TrainerCallback
- Updates weights every N steps
- Logs curriculum statistics

**`BaselineUpdateCallback`**: TrainerCallback for baseline updates
- Applies pending updates after optimizer step
- Logs baseline statistics

#### `bolt.py`

**`BoltConfig`**: Dataclass for BOLT configuration
- Extracted from TrainingConfig

**`TRLBoltTrainer`**: Main trainer class
- Extends `TRLGRPOTrainer`
- Overrides: `setup_data()`, `setup_rewards()`, `train()`
- Adds curriculum/baseline callbacks
- Saves baseline at end of training

---

## Configuration

BOLT adds these fields to `TrainingConfig`:

```python
# Curriculum sampling
curriculum_enabled: bool = False      # Enable uncertainty-based sampling
curriculum_epsilon: float = 0.05      # Floor for sampling weights
curriculum_update_freq: int = 10      # Steps between weight updates

# Persistent baseline
baseline_enabled: bool = False        # Enable persistent baseline tracking
baseline_rho_min: float = 0.875       # Min forgetting factor (fast adaptation)
baseline_rho_max: float = 0.96        # Max forgetting factor (slow adaptation)
baseline_D_half: float = 0.5          # KL half-life for adaptive forgetting
baseline_warm_start: Optional[str]    # Path to JSON/PKL for warm-start

# Advantage computation
use_baseline_advantages: bool = False # Use A = r - v̂(x) vs group mean
```

### YAML Configuration Example

```yaml
algo: BOLT
model:
  name_or_path: "Qwen/Qwen2.5-3B-Instruct"
  use_peft: true
  lora_r: 32
  lora_alpha: 32

datasets:
  - name: "gsm8k"
    split: "train"

rewards:
  - type: "math_reasoning"
    weight: 1.0

train:
  per_device_batch_size: 16
  num_generations: 8          # K=8 for curriculum-only
  learning_rate: 5e-5
  max_steps: 2500

  # BOLT: Curriculum
  curriculum_enabled: true
  curriculum_epsilon: 0.05
  curriculum_update_freq: 10

  # BOLT: Persistent baseline
  baseline_enabled: true
  baseline_rho_min: 0.875
  baseline_rho_max: 0.96
  baseline_D_half: 0.5
  baseline_warm_start: "./baselines/gsm8k_qwen25.json"

  # BOLT: Advantage computation
  use_baseline_advantages: true
```

---

## Usage Modes

### Mode 1: Curriculum Only
Focus training on uncertain prompts, use standard group-mean advantages.

```yaml
curriculum_enabled: true
baseline_enabled: true        # Needed to track v̂ for sampling
use_baseline_advantages: false
num_generations: 8            # Standard K=8
```

### Mode 2: Baseline Advantages Only
Use persistent baselines for stable advantages with small groups.

```yaml
curriculum_enabled: false
baseline_enabled: true
use_baseline_advantages: true
num_generations: 2            # Can use small K with baseline
```

### Mode 3: Full BOLT (Curriculum + Baseline Advantages)
Maximum efficiency: curriculum sampling + small groups.

```yaml
curriculum_enabled: true
baseline_enabled: true
use_baseline_advantages: true
num_generations: 2            # Small K enabled by baseline
```

---

## Warm-Start Baselines

Pre-computing baselines enables:
1. **Faster convergence**: Start with informed sampling weights
2. **Reproducibility**: Same starting point across experiments
3. **Transfer**: Use baselines from similar models/tasks

### Precompute Script

```bash
cd examples/bolt_training

# GSM8K with base model

python precompute_baseline.py \

    --model "Qwen/Qwen2.5-3B-Instruct" \
    --dataset gsm8k \
    --num_samples 8 \
    --output baselines/gsm8k_qwen25.json

# MATH with checkpoint


python precompute_baseline.py \

    --model "Qwen/Qwen2.5-3B-Instruct" \
    --lora_path ./checkpoint-500 \
    --dataset math \
    --num_samples 8 \
    --output baselines/math_step500.json
```

### Output Format

```json
{
    "metadata": {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "dataset": "gsm8k",
        "num_samples": 8,
        "n_prompts": 7473,
        "mean_v_hat": 0.423,
        "timestamp": "2024-12-01T12:00:00"
    },
    "baselines": {
        "gsm8k_0": 0.75,
        "gsm8k_1": 0.125,
        "gsm8k_2": 0.875,
        ...
    }
}
```

---

## Logged Metrics

BOLT adds these metrics to training logs:

### Baseline Metrics
- `baseline/avg_v_hat`: Mean success probability across prompts
- `baseline/avg_N_eff`: Mean effective sample size
- `baseline/avg_kl_ema`: Mean KL divergence EMA
- `baseline/num_prompts`: Number of tracked prompts
- `baseline/min_v_hat`, `baseline/max_v_hat`: Range of v̂ values

### Curriculum Metrics
- `curriculum/n_tracked`: Number of prompts with baselines
- `curriculum/mean_v_hat`: Mean success probability
- `curriculum/std_v_hat`: Standard deviation of v̂
- `curriculum/mean_uncertainty`: Mean √(v̂(1-v̂))
- `curriculum/mean_weight`: Mean sampling weight
- `curriculum/pct_easy`: % prompts with v̂ > 0.8
- `curriculum/pct_hard`: % prompts with v̂ < 0.2
- `curriculum/pct_learning_edge`: % prompts with 0.3 ≤ v̂ ≤ 0.7

---

## Implementation Details

### Critical: TRL Advantage Override

TRL's GRPOTrainer **always** computes advantages as:
```python
advantages = rewards - mean_grouped_rewards  # Line 1604 in grpo_trainer.py
```

There is **no config option** to disable this. Any approach that tries to compute
baseline advantages in a reward wrapper will fail because TRL will subtract the
group mean again, cancelling out the baseline term.

**The correct solution**: Override `_generate_and_score_completions` to replace
TRL's computed advantages with our baseline advantages AFTER TRL computes them.

```python
class _BoltGRPOTrainer(GRPOTrainer):
    def _generate_and_score_completions(self, inputs):
        # Let TRL compute everything (including wrong advantages)
        output = super()._generate_and_score_completions(inputs)

        # Replace with our baseline advantages
        advantages = rewards - baselines  # A = r - v̂(x)
        output["advantages"] = normalize(advantages)

        return output
```

This is implemented in `BoltGRPOTrainer.create_trainer_class()` which dynamically
creates a GRPOTrainer subclass with the correct override.

### Pre-Update Read Semantics

Critical for correct advantage computation:

1. **Before generating**: Read v̂(x) from baseline
2. **Compute advantage**: A = r - v̂(x) using OLD baseline
3. **After optimizer step**: Update baseline with new reward

This ensures advantages reflect historical performance, not current batch.

### KL-Adaptive Forgetting

The forgetting factor ρ adapts to policy changes:

```python
ρ = 2^(-KL / D_half)
ρ = max(rho_min, min(rho_max, ρ))

# Update rule
α_new = ρ * α + reward
β_new = ρ * β + (1 - reward)
```

With defaults (rho_min=0.875, rho_max=0.96, D_half=0.5):
- KL = 0: ρ = 0.96 (slow forgetting, stable policy)
- KL = 0.5: ρ = 0.96 (clamped to max)
- KL = 2.0: ρ = 0.875 (fast forgetting, policy shifting)

### Prompt Key Generation

Prompts are identified by truncated content + hash:
```python
def make_prompt_key(prompt: str) -> str:
    if len(prompt) <= 200:
        return prompt
    return f"{prompt[:100]}...{hash(prompt)}"
```

---

## Comparison with Standard GRPO

| Aspect | Standard GRPO | BOLT |
|--------|--------------|------|
| Sampling | Uniform | Uncertainty-weighted |
| Advantage | Group mean | Persistent baseline |
| Min group size | K ≥ 8 | K ≥ 1 |
| Dataset coverage | 1x per epoch | 2x+ (oversampling) |
| Adaptation | None | KL-adaptive forgetting |
| Warm-start | No | Yes (JSON baselines) |

---

## References

- **SPO (Self-Play Optimization)**: KL-adaptive forgetting (Equation 5)
- **Curriculum Learning**: Bengio et al., "Curriculum Learning" (2009)
- **GRPO**: Shao et al., "DeepSeekMath" (2024)

---

## File Sizes

| File | Lines | Description |
|------|-------|-------------|
| `baseline.py` | 445 | BaselineStore, UnifiedBaseline |
| `curriculum.py` | 356 | Sampler, callbacks, reward wrapper |
| `bolt.py` | 441 | TRLBoltTrainer |
| `__init__.py` | 50 | Module exports |
| **Total** | ~1,290 | |

**Utility:**
| File | Lines | Purpose |
|------|-------|---------|
| `examples/bolt_training/precompute_baseline.py` | 969 | Warm-start baseline computation |

---

## Changelog

### v1.0.0 (2024-12)
- Initial implementation
- TRL and Unsloth backends
- Curriculum sampling with uncertainty-based weights
- Persistent baselines with KL-adaptive forgetting
- Warm-start support via JSON/PKL
- Precompute script for baseline generation
