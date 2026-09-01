# Backends Overview

AlignTune routes training through one of three backends behind a single config format, so switching between them (say, from TRL to Unsloth for speed) doesn't require rewriting your training code.

## Supported Backends

### 1. TRL Backend (Transformer Reinforcement Learning)
The most comprehensive and battle-tested backend. It supports every single SFT and RL algorithm offered by AlignTune.
- **Best for:** Most standard training workloads, CPU usage, and experimental algorithms not yet available on Unsloth.
- **Identifier:** `backend="trl"`

### 2. Unsloth Backend
Optimized for memory efficiency and raw speed. Unsloth rewrites backpropagation and cross-entropy kernels to drastically reduce VRAM constraints.
- **Best for:** Low-VRAM environments, fast rapid-iteration, QLoRA workflows.
- **Identifier:** `backend="unsloth"`

### 3. ES Backend (Evolution Strategies)
A gradient-free optimization backend that aligns models without standard backward passes, relying on evolutionary sampling and variation perturbations.
- **Best for:** Non-differentiable reward scenarios and experimental neuro-evolution experiments.
- **Identifier:** `backend="es"`
- As of PR #33, rollout/generation is abstracted behind a `BaseRolloutBackend`
  interface (`core/rollout/`), with two implementations: `HFRolloutBackend`
  (standard `transformers` generation) and `VLLMRolloutBackend` (vLLM-backed,
  ~5-10x faster generation via continuous batching, LoRA-adapter swapping, and
  PagedAttention). **Note:** the `ESTrainer` itself
  currently hardcodes `VLLMRolloutBackend`: `HFRolloutBackend` exists and is
  unit-tested but isn't yet wired in as a selectable option for ES training.

```python
from aligntune.core.backend_factory import create_es_trainer

trainer = create_es_trainer(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    dataset_name="openai/gsm8k",
    population_size=64,
    sigma=0.5,
    reward_type="math_correctness",
)
trainer.train()  # internally uses VLLMRolloutBackend for generation
```

`HFRolloutBackend`/`VLLMRolloutBackend` can also be used standalone for plain
generation outside of ES training:

```python
from aligntune.core.rollout import VLLMRolloutBackend

backend = VLLMRolloutBackend(
    model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
    gpu_memory_utilization=0.7,
)
backend.initialize()
completions = backend.generate(["What is 12 * 7?"], max_new_tokens=64, temperature=0.7)
backend.cleanup()
```

---

## Automatic Backend Selection

AlignTune utilizes an intelligent automatic fallback system. If you omit the `backend` argument, the `BackendFactory` resolves the optimal backend via the following logic:

1. Triggers Unsloth if the environment has a compatible CUDA GPU and the algorithm chosen is supported by Unsloth.
2. Falls back to TRL if Unsloth is missing or unsupported for the given configuration.

```python
trainer = create_sft_trainer(
    model_name="microsoft/DialoGPT-small",
    dataset_name="tatsu-lab/alpaca",
    backend="auto", # Factory handles it natively based on environment
    ...
)
```