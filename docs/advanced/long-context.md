# Long Context Utilities

Fine-tuning with long context windows increases memory use and can reduce the
quality of positional representations. AlignTune provides several ways to
extend context efficiently: RoPE scaling, S2-Attention, Sliding Window
Attention (SWA), sequence packing, and attention-implementation helpers.

## Choosing a technique

| Technique | Cost | Best use case |
|---|---|---|
| RoPE scaling | Cheapest at inference, needs the position-embedding math to hold up at the new length | Extending a 4k–8k base model to 32k–128k+ context for RAG/long-document tasks |
| S2-Attention (shifted sparse attention) | Cheap short-window training, full attention at inference | Fine-tuning for long context on a limited GPU budget. LongLoRA-style, extended Llama-2-7B to 100k tokens on a single 8×A100 node in the original paper |
| Sliding Window Attention (SWA) | Linear cost at both train and inference time | Very long sequences where dependencies are mostly local (streaming, long chat history): cheapest of the three when you don't need full-history attention |

## RoPE scaling

The SFT factory supports these RoPE strategies:

| `rope_type` | What it does | Notes |
|---|---|---|
| `default` | Keeps the model's native positional encoding | No context extension; useful for explicitly disabling scaling in a shared config |
| `linear` | Linearly interpolates positions | Simple and broadly compatible baseline |
| `dynamic` | Applies dynamic NTK-aware scaling | Useful when short and long contexts are mixed |
| `yarn` | Uses YaRN frequency-domain interpolation/extrapolation | Good default for larger extensions; supports `rope_beta_fast`, `rope_beta_slow`, and `rope_attention_factor` |
| `longrope` | Uses per-dimension short/long frequency factors | Requires model-specific `rope_short_factor` and `rope_long_factor` values when needed |
| `llama3` | Uses Llama 3 frequency-aware scaling | Requires the model-compatible low/high-frequency settings |

Pass the settings directly to the TRL SFT factory. It validates the internal
RoPE configuration before loading the model:

```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    dataset_name="tatsu-lab/alpaca",
    backend="trl",
    max_seq_length=16384,
    rope_type="yarn",                    # linear, dynamic, yarn, longrope, llama3
    rope_factor=4.0,
    rope_target_max_seq_length=16384,
    # Optional when the model config does not expose its native length:
    rope_original_max_position_embeddings=4096,
)
trainer.train()
```

For strategy-specific settings, pass the corresponding factory arguments:

```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    dataset_name="tatsu-lab/alpaca",
    backend="trl",
    max_seq_length=32768,
    rope_type="yarn",
    rope_target_max_seq_length=32768,
    rope_factor=8.0,
    rope_beta_fast=32,
    rope_beta_slow=1,
    rope_attention_factor=1.0,
)
trainer.train()
```

The lower-level `RopeScalingConfig`/`RopeScalingApplier` helper is also
available for direct Hugging Face loading. That helper exposes the legacy
strategies `linear`, `dynamic`, `yarn`, and `ntk` (`ntk` is translated to
Hugging Face's `longrope` key). Prefer the factory for SFT so the model and
trainer receive the same `max_seq_length` and RoPE settings:

```python
from aligntune.core.long_context import RopeScalingConfig, RopeScalingApplier
from transformers import AutoModelForCausalLM

rope_config = RopeScalingConfig(
    type="yarn",
    factor=4.0,
    original_max_position=4096,
    target_max_position=16384,
)
RopeScalingApplier.validate_config(rope_config)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    rope_scaling=RopeScalingApplier.build_rope_config(rope_config),
)
```

## S2-Attention

```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    dataset_name="tatsu-lab/alpaca",
    backend="trl",
    attn_implementation="s2",   # registered by aligntune.core.long_context on import
    s2_group_size_ratio=0.25,
    s2_min_seq_length=64,
    s2_shift_ratio=0.5,
)
trainer.train()
```

## Sliding Window Attention

```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    dataset_name="tatsu-lab/alpaca",
    backend="trl",
    attn_implementation="swa",  # registered by aligntune.core.long_context on import
    sliding_window=4096,         # window size in tokens
)
trainer.train()
```

Mistral-family models have SWA built into their architecture natively (no `attn_implementation`
override needed): see `recipes/configs/sft/mistral_7b_32k_sliding_window.yaml`.

## Sequence packing & attention overrides

- **Sequence Packing** (`DocumentPacker`): efficient batch compilation via
  `pack_sequences_best_fit`, drastically reducing zero-padding when stacking many
  variable-length conversations/documents.
- **Attention Overrides**: `LongContextAttentionHelper` auto-detects the best available
  attention implementation (`flash_attention_2` → `sdpa` → `eager`) and patches it to respect
  sequence-packing boundaries correctly.

## Notebooks

- [`notebooks/26_rope_scaling.ipynb`](https://github.com/Lexsi-Labs/aligntune/blob/main/notebooks/26_rope_scaling.ipynb)
- [`notebooks/27_s2_attention.ipynb`](https://github.com/Lexsi-Labs/aligntune/blob/main/notebooks/27_s2_attention.ipynb)
- [`notebooks/28_sliding_window_attention.ipynb`](https://github.com/Lexsi-Labs/aligntune/blob/main/notebooks/28_sliding_window_attention.ipynb)
