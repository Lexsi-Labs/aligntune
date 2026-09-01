# AlignTune Configuration Parameters

This document provides a comprehensive reference for all configuration parameters available in AlignTune, organized by training type (SFT vs RL) and algorithm-specific settings.

Rows marked ⚠️ call out fields whose effective default or valid values depend on the specific algorithm, read the row's note before relying on that field's listed default.

---

## Table of Contents

1. [SFT (Supervised Fine-Tuning) Parameters](#sft-supervised-fine-tuning-parameters)
2. [RL (Reinforcement Learning) Common Parameters](#rl-reinforcement-learning-common-parameters)
3. [Algorithm-Specific RL Parameters](#algorithm-specific-rl-parameters)
   - [PPO](#ppo-proximal-policy-optimization)
   - [DPO](#dpo-direct-preference-optimization)
   - [Online-DPO](#online-dpo)
   - [SPIN](#spin-self-play-fine-tuning)
   - [GRPO (base) / GSPO / DAPO / Dr. GRPO](#grpo-base-gspo-dapo-dr-grpo)
   - [GBMPO](#gbmpo-group-based-mirror-po)
   - [Counterfactual GRPO](#counterfactual-grpo)
   - [PACE](#pace-bolt)
   - [ES (non-meta)](#es-evolution-strategies-non-meta)
4. [Distillation Parameters](#distillation-parameters)
   - [Standard / Offline Distillation](#standard-offline-distillation)
   - [SDFT](#sdft-self-distillation-fine-tuning)
5. [RAFT](#raft-retrieval-augmented-fine-tuning)

---

## SFT (Supervised Fine-Tuning) Parameters

### Model Configuration (`ModelConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name_or_path` | str | **Required** | HuggingFace model name or local path |
| `precision` | PrecisionType | `auto` | Model precision (`bf16`, `fp16`, `fp32`, `auto`) |
| `quantization` | Dict | `{}` | Quantization config (e.g., `{"load_in_4bit": True}`) |
| `attn_implementation` | str | `"auto"` | Attention implementation (`auto`, `flash_attention_2`, `sdpa`) |
| `gradient_checkpointing` | bool | `True` | Enable gradient checkpointing for memory efficiency |
| `max_memory` | Dict | `None` | Max memory per device (e.g., `{"0": "20GB"}`) |
| `device_map` | str/Dict | `"auto"` | Device mapping strategy |
| `use_unsloth` | bool | `False` | Enable Unsloth acceleration |
| `max_seq_length` | int | `2048` | Maximum sequence length |
| `use_peft` | bool | `False` | Enable PEFT/LoRA |
| `lora_r` | int | `16` | LoRA rank |
| `lora_alpha` | int | `32` | LoRA alpha scaling |
| `lora_dropout` | float | `0.1` | LoRA dropout rate |
| `lora_target_modules` | List[str] | `None` | LoRA target modules (e.g., `["q_proj", "v_proj"]`) |
| `lora_bias` | str | `"none"` | Bias training (`none`, `all`, `lora_only`) |
| `trust_remote_code` | bool | `False` | Allow custom model/tokenizer code from the model repository |
| `tokenizer_name_or_path` | str | `None` | Optional tokenizer path; defaults to the model path |
| `rope_type` | str | `None` | **TRL-only** RoPE scaling type; required when other RoPE options are set |
| `rope_target_max_seq_length` | int | `None` | **TRL-only** Target context length for RoPE scaling |
| `rope_factor` | float | `None` | **TRL-only** RoPE scaling factor |
| `rope_theta` | float | `None` | **TRL-only** Advanced RoPE theta override |
| `rope_original_max_position_embeddings` | int | `None` | **TRL-only** Original context length for RoPE scaling |
| `rope_partial_rotary_factor` | float | `None` | **TRL-only** Partial rotary factor |
| `rope_attention_factor` | float | `None` | **TRL-only** RoPE attention scaling factor |
| `rope_beta_fast` | float | `None` | **TRL-only** Fast beta parameter for YARN-style RoPE |
| `rope_beta_slow` | float | `None` | **TRL-only** Slow beta parameter for YARN-style RoPE |
| `rope_short_factor` | float/list | `None` | **TRL-only** Short-context RoPE factor |
| `rope_long_factor` | float/list | `None` | **TRL-only** Long-context RoPE factor |
| `rope_low_freq_factor` | float | `None` | **TRL-only** Low-frequency RoPE factor |
| `rope_high_freq_factor` | float | `None` | **TRL-only** High-frequency RoPE factor |
| `sliding_window` | int | `None` | **TRL-only** Sliding-window attention size |
| `s2_group_size_ratio` | float | `0.25` | **TRL-only** S2 attention group-size ratio |
| `s2_min_seq_length` | int | `64` | **TRL-only** Minimum length before S2 attention is applied |
| `s2_shift_ratio` | float | `0.5` | **TRL-only** S2 attention shift ratio |
| `train_embeddings` | bool | `False` | **TRL backend only for now**, train input embeddings instead of or alongside adapter weights |
| `embedding_init_method` | str | `"random"` | **TRL backend only for now**, new embedding initialization (`random`, `mean`, `mean_of_constituents`) |
| `embedding_pad_to_multiple_of` | int | `None` | **TRL backend only for now**, pad embedding vocabulary to a multiple of this value |
| `use_gradient_checkpointing` | bool | `True` | Enable gradient checkpointing (legacy) |
| `num_labels` | int | `None` | Number of labels (classification tasks) |
| `model_init_kwargs` | Dict | `{}` | Additional model initialization arguments |

**Backend note:** The advanced RoPE, S2/sliding-window, and embedding controls
listed above are supported by the **TRL SFT backend only for now**. The
Unsloth SFT backend does not provide equivalent wiring for these options.

### Dataset Configuration (`DatasetConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | **Required** | HuggingFace dataset name or local path |
| `split` | str | `None` | Source split to select. Use a plain split name such as `"train"`; HF slice syntax such as `"train[:16]"` is unsupported. Use `max_samples` for row limits. When omitted, all available splits are processed. |
| `config_name` / `subset` / `config` | str | `None` | Hugging Face dataset configuration/subset. `subset` is normalized to `config_name` by the loader. |
| `max_samples` | int | `None` | Maximum rows per split, applied before CuratorKIT processing. This is not a global limit across all splits. |
| `column_mapping` | Dict[str, str] | `{}` | Map source columns to the canonical columns required by the selected task. |
| `format_type` | str | `None` | Expected CuratorKIT format (for example, `alpaca`, `sharegpt`, `dpo`, or `grpo`). `None` enables format detection/fallback handling. |
| `system_prompt` | str | `None` | Optional system prompt injected after curation and column normalization. |
| `processing_fn` | Callable | `None` | Custom row preprocessing function run during the CuratorKIT stage. Return `None` to reject a row. |
| `processing_batched` | bool | `False` | Whether `processing_fn` receives a batch instead of one row at a time. |
| `keep_columns` | bool | Task-dependent | Preserve source columns in addition to CuratorKIT canonical columns. Defaults to enabled for GRPO and distillation tasks. |
| `val_split_ratio` | float | `None` | Create a `validation` split from the source data. Must be between 0 and 1. |
| `test_split_ratio` | float | `None` | Create a `test` split from the source data. Must be between 0 and 1; the two ratios must sum to less than 1. |
| `split_seed` | int | `42` | Seed used for deterministic ratio-based splitting. |
| `privileged_context_column` | str | `None` | Distillation-only source column for hints/feedback. Aliases such as `hint`, `feedback`, `context`, and `reference` are also recognized. |
| `curator_schema_gate` | bool | `True` | Enable CuratorKIT row-level schema validation. |
| `curator_clean` | bool | `False` | Enable CuratorKIT text cleaning. |
| `curator_dedup` | str | `"none"` | CuratorKIT deduplication mode: `none`, `exact`, or `minhash`. |
| `curator_use_tiktoken` | bool | `False` | Use tiktoken for CuratorKIT token counting. |
| `curator_max_tokens` | int | `1_000_000` | Maximum token count used by CuratorKIT schema checks. |

**SFT column-mapping example:** mappings use the form `source_column` to
canonical SFT column. For instruction data, the source dataset can map
`prompt`, `context`, and `response` to `instruction`, `input`, and `output`.
For conversational data, map a conversation column to `messages` instead.

```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    dataset_name="your-org/your-dataset",
    system_prompt="Answer accurately and concisely.",
    column_mapping={
        "prompt": "instruction",
        "context": "input",
        "response": "output",
        # Use this for conversational rows shaped as role/content messages.
        # "conversation": "messages",
    },
)
```

**DataManager flow:** the loader first normalizes the source into a
`DatasetDict`, normalizes split aliases (`dev`/`valid` to `validation`), applies
`max_samples` and requested split ratios, then runs CuratorKIT independently on
each final split. AlignTune subsequently applies column mapping, adds
distillation privileged context, injects the system prompt, and returns the
processed `DatasetDict` to the trainer.

### Training Configuration (`TrainingConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `per_device_batch_size` | int | `1` | Batch size per device |
| `per_device_eval_batch_size` | int | `None` | Evaluation batch size per device; defaults to the training batch size in the factory |
| `gradient_accumulation_steps` | int | `1` | Gradient accumulation steps |
| `max_steps` | int | `None` | Maximum training steps |
| `epochs` | int | `None` | Number of training epochs; defaults to `3` when `max_steps` is unset |
| `learning_rate` | float | `1e-5` | Learning rate |
| `weight_decay` | float | `0.01` | Weight decay |
| `warmup_steps` | int | `0` | Warmup steps (or calculated from warmup_ratio) |
| `warmup_ratio` | float | `0.1` | Warmup ratio of total steps |
| `eval_interval` | int | `100` | Evaluation interval (steps) |
| `save_interval` | int | `500` | Save checkpoint interval (steps) |
| `max_grad_norm` | float | `1.0` | Maximum gradient norm for clipping |
| `fp16` | bool | `False` | Use FP16 training |
| `bf16` | bool | `False` | Use BF16 training |
| `dataloader_num_workers` | int | `0` | Number of dataloader workers |
| `remove_unused_columns` | bool | `False` | Remove unused dataset columns |
| `optimizer` | str | `"adamw_torch"` | Optimizer type |
| `lr_scheduler` | str | `"cosine"` | Learning rate scheduler |
| `group_by_length` | bool | `False` (`True` in factory) | Group sequences by length |
| `dataloader_drop_last` | bool | `False` | Drop last incomplete batch |
| `eval_accumulation_steps` | int | `None` | Evaluation accumulation steps |
| `label_smoothing_factor` | float | `0.0` | Label smoothing factor |
| `early_stopping_patience` | int | `None` | Early stopping patience |
| `early_stopping_threshold` | float | `0.0` | Early stopping threshold |
| `load_best_model_at_end` | bool | `True` | Load best model at end |
| `metric_for_best_model` | str | `"eval_loss"` | Metric for best model selection |
| `greater_is_better` | bool | `False` | Whether higher metric is better |
| `packing` | bool | `False` | Enable sequence packing |
| `packing_strategy` | str | `"bfd"` | Packing strategy (`bfd`, `wrapped`) |
| `eval_packing` | bool | `None` | Enable packing for evaluation |
| `padding_free` | bool | `False` | Padding-free training |
| `pad_to_multiple_of` | int | `None` | Pad sequences to multiple of N |
| `completion_only_loss` | bool | `None` | Compute loss only on completions |
| `assistant_only_loss` | bool | `False` | Compute loss only on assistant turns |
| `loss_type` | str | `"nll"` | Loss type (`nll`, `dft`) |
| `activation_offloading` | bool | `False` | Enable activation offloading |
| `use_flash_attention_2` | bool | `None` | Use Flash Attention 2 |
| `gradient_checkpointing` | bool | `False` | Enable gradient checkpointing |
| `gradient_checkpointing_kwargs` | Dict | `{}` | Gradient checkpointing arguments |
| `enable_thinking` | bool | `False` | Enable model thinking/reasoning mode when supported by the tokenizer/model |
| `use_liger_kernel` | bool | `False` | Enable Liger kernels when supported by the TRL backend |
| `seed` | int | `42` | Training and split reproducibility seed |
| `data_seed` | int | `None` | Optional separate data-loader seed |
| `extra_params` | Dict | `{}` | Additional factory keyword arguments forwarded to backend configuration |

### Evaluation Configuration (`EvaluationConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `compute_perplexity` | bool | `True` | Compute perplexity metric |
| `compute_rouge` | bool | `True` | Compute ROUGE scores |
| `compute_bleu` | bool | `True` | Compute BLEU scores |
| `compute_meteor` | bool | `False` | Compute METEOR scores (requires nltk) |
| `compute_bertscore` | bool | `False` | Compute BERTScore (requires bert-score) |
| `compute_semantic_similarity` | bool | `False` | Compute semantic similarity |
| `compute_codebleu` | bool | `False` | Compute CodeBLEU (for code tasks) |
| `custom_metrics` | List[Callable] | `None` | Custom metric functions |
| `max_samples_for_quality_metrics` | int | `50` | Max samples for quality metrics |
| `bertscore_model` | str | `"microsoft/deberta-xlarge-mnli"` | Model for BERTScore |
| `semantic_similarity_model` | str | `"sentence-transformers/all-MiniLM-L6-v2"` | Model for semantic similarity |

### Logging Configuration (`LoggingConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | `"./output"` | Output directory |
| `run_name` | str | `None` | Run name for logging |
| `loggers` | List[str] | `["tensorboard"]` | Logger types (`tensorboard`, `wandb`) |
| `log_level` | str | `"INFO"` | Logging level |
| `log_interval` | int | `10` | Logging interval (steps) |
| `save_strategy` | str | `"steps"` | Save strategy |
| `eval_strategy` | str | `"steps"` | Evaluation strategy passed to the trainer. |
| `report_to` | str | `"none"` | Reporting destination |

---

## RL (Reinforcement Learning) Common Parameters

### Model Configuration (`ModelConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name_or_path` | str | **Required** | Policy model name or path |
| `sft_path` | str | `None` | SFT checkpoint path |
| `reward_path` | str | `None` | Reward model path |
| `reward_model_name` | str | `None` | Separate reward model name |
| `reward_model_source` | RewardModelSourceConfig | `None` | Reward model source configuration |
| `precision` | PrecisionType | `AUTO` | Model precision, `"auto"` resolves per-GPU via `PrecisionHandler.get_training_args_precision()`, which checks `torch.cuda.is_bf16_supported()` and falls back to `fp16` on non-Ampere GPUs (T4, V100, etc.) |
| `quantization` | Dict | `{}` | Quantization config |
| `attn_implementation` | str | `"auto"` | Attention implementation |
| `gradient_checkpointing` | bool | `False` | Enable gradient checkpointing |
| `max_memory` | Dict | `None` | Max memory per device |
| `device_map` | str/Dict | `None` | Device mapping |
| `use_unsloth` | bool | `False` | Enable Unsloth acceleration |
| `max_seq_length` | int | `2048` | Maximum sequence length |
| `reward_value_model` | str | `"meta-llama/Llama-3.2-1B-Instruct"` | Reward/value model name |
| `reward_value_loading_type` | str | `None` | Loading type (`unsloth`, `standard`) |
| `reward_model_quantization` | Dict | `{}` | Reward model quantization |
| `value_model_quantization` | Dict | `{}` | Value model quantization |
| `use_peft` | bool | `True` | Enable PEFT/LoRA |
| `lora_r` | int | `16` | LoRA rank |
| `lora_alpha` | int | `32` | LoRA alpha |
| `lora_dropout` | float | `0.05` | LoRA dropout |
| `lora_target_modules` | List[str] | `["q_proj", "k_proj", "v_proj", "o_proj"]` | LoRA target modules |
| `rslora_enabled` | bool | `False` | Enable rank-stabilized LoRA |
| `loftq_init` | bool | `False` | Initialize LoRA weights with LoftQ when supported by the installed PEFT version |
| `pissa_init` | bool | `False` | Initialize LoRA weights with PiSSA |
| `trust_remote_code` | bool | `True` | Trust remote code |
| `model_init_kwargs` | Dict | `{}` | Model initialization arguments |
| `ref_model_init_kwargs` | Dict | `{}` | Reference model init arguments |
| `model_adapter_name` | str | `None` | Model adapter name |
| `ref_adapter_name` | str | `None` | Reference adapter name |
| `force_use_ref_model` | bool | `False` | Force use of reference model |
| `disable_dropout` | bool | `True` | Disable dropout |
| `use_logits_to_keep` | bool | `False` | Use logits_to_keep optimization |
| `reward_device` | str | `"auto"` | Reward model device (`auto`, `cpu`, `cuda`) |

### Dataset Configuration (`DatasetConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | **Required** | Dataset name or path |
| `split` | str | `None` | Source split to select. Use a plain split name such as `"train"`; HF slice syntax such as `"train[:16]"` is unsupported. When omitted, all available splits are processed. |
| `config_name` | str | `None` | Hugging Face dataset configuration/subset name |
| `max_samples` | int | `None` | Maximum rows per split, applied before CuratorKIT processing |
| `column_mapping` | Dict[str, str] | `{}` | Map source columns to canonical columns. Prompt-only/GRPO-style tasks map a source question to `prompt`; preference tasks map fields to `prompt`, `chosen`, and `rejected`. |
| `format_type` | str | `None` | Expected CuratorKIT format, such as `dpo`, `grpo`, or `alpaca` |
| `system_prompt` | str | `None` | Optional system prompt injected after curation and column normalization |
| `processing_fn` | Callable | `None` | Custom row preprocessing function run during curation |
| `processing_batched` | bool | `False` | Whether `processing_fn` receives batches |
| `keep_columns` | bool | Task-dependent | Preserve source columns in addition to canonical columns |
| `val_split_ratio` | float | `None` | Create a validation split; must be between 0 and 1 |
| `test_split_ratio` | float | `None` | Create a test split; must be between 0 and 1, with both ratios summing to less than 1 |
| `split_seed` | int | `42` | Seed for deterministic ratio-based splitting |
| `curator_schema_gate` | bool | `True` | Enable CuratorKIT's row-level format/schema validation gate. Set `False` only for datasets already in exact canonical shape that CuratorKIT's format-detector doesn't yet recognize (see [Known Issues](ISSUES.md)): most format gaps have a proper fallback-schema fix instead, so this is a last resort. |
| `curator_clean` | bool | `False` | Enable CuratorKIT text cleaning |
| `curator_dedup` | str | `"none"` | CuratorKIT dedup mode |
| `curator_use_tiktoken` | bool | `False` | Use tiktoken for CuratorKIT token counting |
| `curator_max_tokens` | int | `1_000_000` | Max tokens for CuratorKIT schema length checks |

### Training Configuration (`TrainingConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `per_device_batch_size` | int | `1` | Batch size per device |
| `per_device_eval_batch_size` | int | `1` | Eval batch size per device |
| `gradient_accumulation_steps` | int | `1` | Gradient accumulation steps |
| `max_steps` | int | `-1` | Maximum training steps; `-1` means train for the configured epochs |
| `epochs` | int | `3` | Number of epochs |
| `eval_interval` | int | `100` | Evaluation interval |
| `save_interval` | int | `500` | Save interval |
| `learning_rate` | float | `1e-5` | Learning rate |
| `max_grad_norm` | float | `1.0` | Max gradient norm |
| `weight_decay` | float | `0.01` | Weight decay |
| `use_cache` | bool | `True` | Enable model key/value cache when supported by the trainer |
| `optimizer` | str | `"adamw_torch"` | Optimizer type |
| `lr_scheduler` | str | `"cosine"` | LR scheduler type |
| `warmup_steps` | int | `0` | Warmup steps |
| `warmup_ratio` | float | `0.0` | Warmup ratio |
| `rollout_batch_size` | int | `1` | Rollout batch size |
| `kl_coef` | float | `0.1` | KL divergence coefficient |
| `cliprange` | float | `0.2` | PPO clip range |
| `cliprange_value` | float | `0.2` | Value function clip range |
| `num_ppo_epochs` | int | `None` | PPO epochs per batch |
| `temperature` | float | `0.6` | Sampling temperature |
| `whiten_rewards` | bool | `False` | Whiten rewards |
| `kl_estimator` | str | `"k1"` | KL estimator type |
| `vf_coef` | float | `0.1` | Value function coefficient |
| `gamma` | float | `1.0` | Discount factor |
| `lam` | float | `0.95` | GAE lambda |
| `response_length` | int | `128` | Response length |
| `stop_token` | str | `"eos"` | Stop token |
| `missing_eos_penalty` | float | `1.0` | Missing EOS penalty |
| `ds3_gather_for_generation` | bool | `True` | DeepSpeed Stage 3 gather |
| `generation_kwargs` | Dict | `None` | Generation arguments |
| `max_length` | int | `1024` | Max sequence length |
| `max_prompt_length` | int | `512` | Max prompt length. **Removed from `GRPOConfig`/`DPOConfig`/`ORPOConfig`/`CPOConfig` in the pinned `trl==1.7.1`**, still read for logging/internal truncation in every GRPO-family and DPO trainer, but never forwarded to the underlying TRL config object; passing it directly (bypassing the wrapper) raises `TypeError: ...__init__() got an unexpected keyword argument 'max_prompt_length'`. |
| `max_target_length` | int | `None` | Max target length |
| `max_completion_length` | int | `256` | Max completion length |
| `top_p` | float | `0.95` | Nucleus sampling parameter |
| `padding_free` | bool | `False` | Padding-free training |
| `truncation_mode` | str | `"keep_end"` | Truncation mode, **also removed from `ORPOConfig`/`CPOConfig`** in trl 1.7.1, same fate as `max_prompt_length` for those two algorithms only |
| `beta` | float | `0.1` | DPO/GRPO-family beta parameter (KL coefficient / inverse temperature depending on algorithm) |
| `loss_type` | str | `None` | Loss type. This single field means different valid value-sets for different algorithms (DPO: `sigmoid`/`hinge`/`ipo`/... as a `list[str]`; GRPO-family: `grpo`/`dapo`/`bnpo`/`dr_grpo`/`cispo`/`sapo`/`luspo`/`vespo`): setting a value valid for one algorithm but not another gets silently discarded/reset by that algorithm's post-backfill guard, if it has one. |
| `loss_weights` | Dict | `None` | Loss weights |
| `f_divergence_type` | str | `"reverse_kl"` | F-divergence type |
| `f_alpha_divergence_coef` | float | `1.0` | Alpha divergence coefficient |
| `reference_free` | bool | `False` | Reference-free training |
| `label_smoothing` | float | `0.0` | Label smoothing |
| `use_weighting` | bool | `False` | Use importance weighting |
| `rpo_alpha` | float | `None` | RPO alpha parameter |
| `ld_alpha` | float | `None` | LD alpha parameter |
| `discopop_tau` | float | `0.05` | DiscoPOP tau parameter |
| `sync_ref_model` | bool | `False` | Sync reference model |
| `ref_model_mixup_alpha` | float | `0.6` | Reference model mixup alpha |
| `ref_model_sync_steps` | int | `512` | Reference sync steps |
| `grpo_alpha` | float | `0.1` | GRPO alpha parameter |
| `grpo_beta` | float | `0.1` | GRPO beta parameter |
| `gspo_gamma` | float | `0.1` | GSPO gamma parameter |
| `gspo_delta` | float | `0.1` | GSPO delta parameter |
| `eval_steps` | int | `100` | Evaluation steps |
| `eval_strategy` | str | `"no"` | Evaluation strategy |
| `dpo_eval_enabled` | bool | `False` | Enable preference/DPO evaluation for supported preference trainers |
| `dpo_eval_max_samples` | int | `None` | Maximum samples used by DPO evaluation |
| `dpo_zero_shot_max_samples` | int | `50` | Maximum zero-shot DPO evaluation samples |
| `dpo_few_shot_max_samples` | int | `30` | Maximum few-shot DPO evaluation samples |
| `save_steps` | int | `500` | Save steps |
| `save_strategy` | str | `"steps"` | Save strategy |
| `save_total_limit` | int | `None` | Max checkpoints to keep |
| `load_best_model_at_end` | bool | `False` | Load best model at end |
| `metric_for_best_model` | str | `None` | Best model metric |
| `greater_is_better` | bool | `False` | Whether higher is better |
| `logging_steps` | int | `10` | Logging steps |
| `logging_strategy` | str | `"steps"` | Logging strategy |
| `num_generations` | int | `None` | Number of generations per prompt |
| `mask_truncated_completions` | bool | **`False`** | Whether TRL should mask out any sample that didn't emit EOS before hitting `max_completion_length`. **Changed from `True` to `False` (matching TRL's own `GRPOConfig` default)**, the old `True` default caused TRL to zero out the *entire* completion mask (not just the overflow tokens) for any such sample, which drove loss and every parameter's gradient to exactly `0.0` whenever a chunk of a batch didn't finish within `max_completion_length`: very common for undertrained models on a short completion budget. |
| `scale_rewards` | str | `"group"` | Reward scaling (`group`, `batch`). **Not exposed by the TRL backend at all** for GRPO/GSPO/DAPO/Dr.GRPO, only Unsloth reads/forwards this field for those four algorithms. |
| `reward_weights` | List[float] | `None` | Reward function weights |
| `enable_thinking` | bool | `False` | Enable Qwen3 thinking mode |
| `fast_inference` | bool | `False` | Enable Unsloth vLLM (faster) |
| `vllm_gpu_memory_utilization` | float | `0.7` | vLLM GPU memory (0.95 for max) |
| `seed` | int | `42` | Random seed |
| `data_seed` | int | `47` | Data seed |
| `use_liger_kernel` | bool | `False` | Use Liger kernel |
| `use_liger_loss` | bool | `None` | Use Liger loss |
| `gradient_checkpointing_kwargs` | Dict | `{"use_reentrant": False}` | Gradient checkpointing args |
| `group_by_length` | bool | `False` | Group sequences by length |

### Sample Logging Configuration (`SampleLoggingConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `False` | Enable sample logging |
| `prompts` | List[str] | `None` | Prompts for sample generation |
| `interval_steps` | int | `None` | Steps between samples |
| `percent_of_max_steps` | float | `None` | Percent of max steps (0-1) |
| `max_new_tokens` | int | `80` | Max tokens to generate |
| `temperature` | float | `0.6` | Generation temperature |
| `top_p` | float | `0.9` | Nucleus sampling parameter |
| `num_samples` | int | `3` | Number of samples per prompt |

### Logging Configuration (`LoggingConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loggers` | List[str] | `["tensorboard"]` | Logger types |
| `run_name` | str | `None` | Run name |
| `output_dir` | str | `"./output"` | Output directory |
| `log_level` | str | `"INFO"` | Logging level |
| `sample_logging` | SampleLoggingConfig | See above | Sample logging config |
| `report_to` | str | `"none"` | Reporting destination |

### Reward Configuration (`RewardConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | str | **Required** | Reward function type |
| `weight` | float | `1.0` | Reward weight |
| `params` | Dict | `{}` | Reward function parameters |
| `shield` | bool | `False` | Enable safety shield |
| `clip` | float | `None` | Clip reward values |
| `normalize` | bool | `False` | Normalize rewards |

### Reward Model Training Configuration (`RewardModelTrainingConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_model_name` | str | **Required** | Base model for reward training |
| `training_texts` | List[str] | **Required** | Training texts (min 10) |
| `reward_functions` | List[str] | **Required** | Reward functions to use |
| `output_dir` | str | **Required** | Output directory |
| `reference_texts` | List[str] | `None` | Reference texts (optional) |
| `reward_weights` | List[float] | `None` | Reward function weights |
| `num_epochs` | int | `3` | Training epochs |
| `learning_rate` | float | `1e-5` | Learning rate |
| `batch_size` | int | `8` | Batch size |
| `gradient_accumulation_steps` | int | `4` | Gradient accumulation |
| `max_length` | int | `512` | Max sequence length |

### Reward Model Source Configuration (`RewardModelSourceConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source_type` | str | **Required** | Source type (`pretrained_hf`, `pretrained_local`, `custom_trained`) |
| `model_name` | str | `None` | HuggingFace model name (for `pretrained_hf`) |
| `model_path` | str | `None` | Local model path (for `pretrained_local`) |
| `training_config` | RewardModelTrainingConfig | `None` | Training config (for `custom_trained`) |
| `fine_tune_with_rewards` | bool | `False` | Fine-tune pretrained with reward functions |

---

## Algorithm-Specific RL Parameters

<!-- --8<-- [start:ppo] -->
### PPO (Proximal Policy Optimization)

PPO uses the common RL parameters plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_ppo_epochs` | int | `4` | Number of PPO epochs per batch |
| `cliprange` | float | `0.2` | PPO policy clip range |
| `cliprange_value` | float | `0.2` | PPO value function clip range |
| `vf_coef` | float | `0.1` | Value function coefficient |
| `gamma` | float | `1.0` | Discount factor (GAE) |
| `lam` | float | `0.95` | GAE lambda parameter |
| `kl_coef` | float | `0.1` | KL penalty coefficient |
| `kl_estimator` | str | `"k1"` | KL estimator (`k1`, `k2`, `k3`) |
| `whiten_rewards` | bool | `False` | Whiten advantages/rewards |
| `response_length` | int | `128` | Maximum generated response length for PPO rollouts |
| `temperature` | float | `0.6` | Rollout sampling temperature |
| `stop_token` | str | `"eos"` | Stop-token behavior for rollout generation |
| `missing_eos_penalty` | float | `1.0` | Penalty applied when a rollout does not emit EOS |

**Reward requirement:** PPO requires either a configured reward model, registered/custom
reward functions, or an enabled custom reward-model training configuration. The
policy, reference, reward, and value models should normally use compatible model
families and tokenizers.

**Rollout backend limitation:** the installed `trl.experimental.ppo.PPOConfig`
does not define `use_vllm`, `rollout_backend`, `vllm_gpu_memory_utilization`, or
`vllm_tensor_parallel_size`. PPO therefore uses the Transformers rollout path;
the shared vLLM settings used by GRPO-family trainers do not enable vLLM for PPO.

**Unsloth-specific fixes/quirks**: `first_true_indices`/`SIMPLE_CHAT_TEMPLATE` (used internally for reward-model sequence-length detection and chat-template fallback) were removed from `trl.trainer.utils` in trl>=1.0, the Unsloth backend now has local fallback implementations for both. Precision selection now correctly checks `torch.cuda.is_bf16_supported()` (via `PrecisionHandler`) instead of assuming `bf16` whenever `precision="auto"`, which previously crashed on non-Ampere GPUs (e.g. Tesla T4). The `apply_qkv`/`apply_o` attention monkey-patch (needed for some architectures under Unsloth) now covers `Qwen2Attention` in addition to `Qwen3Attention`.

<!-- --8<-- [end:ppo] -->
---

<!-- --8<-- [start:dpo] -->
### DPO (Direct Preference Optimization)

DPO uses the common RL parameters plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | float | `0.1` | DPO beta (inverse temperature) |
| `loss_type` | str or `list[str]` | `"sigmoid"` | DPO loss type(s). TRL accepts a string or list and normalizes a string to a list. Valid values: `sigmoid`, `hinge`, `ipo`, `exo_pair`, `nca_pair`, `robust`, `bco_pair`, `sppo_hard`, `aot`, `aot_unpaired`, `apo_zero`, `apo_down`, `discopop`, `sft`, `sigmoid_norm`. |
| `loss_weights` | list[float] | `None` | Optional weights for combined loss types; length must match `loss_type`. |
| `label_smoothing` | float | `0.0` | Label smoothing factor |
| `precompute_ref_log_probs` | bool | `False` | Precompute reference log probs |
| `sync_ref_model` | bool | `False` | Sync reference model periodically |
| `ref_model_mixup_alpha` | float | `0.6` | Reference model mixup alpha |
| `ref_model_sync_steps` | int | `512` | Steps between reference syncs |

**DPO Evaluation Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dpo_eval_enabled` | bool | `False` | Enable DPO evaluation |
| `dpo_eval_max_samples` | int | `None` | Max samples for eval |
| `dpo_zero_shot_max_samples` | int | `50` | Zero-shot eval samples |
| `dpo_few_shot_max_samples` | int | `30` | Few-shot eval samples |
| `dpo_few_shot_examples_text` | str | `None` | Few-shot examples |

**Dataset note**: for preference datasets that only ship full `chosen`/`rejected` conversation strings with no separate `prompt` column (e.g. `Anthropic/hh-rlhf`), aligntune derives `prompt` automatically by splitting at the last `"\n\nAssistant:"` turn marker, the standard hh-rlhf preprocessing recipe.

<!-- --8<-- [end:dpo] -->
---

<!-- --8<-- [start:orpo] -->
### ORPO (Odds Ratio Preference Optimization)

Config: `ORPOConfig`. ORPO folds preference optimization into a single SFT-style
pass — no separate reference model and no reward model. It trains on
`chosen`/`rejected` pairs, adding an odds-ratio penalty on the rejected response
to the standard next-token loss. Available through both the TRL and Unsloth
backends.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | float | `0.1` | Weight of the odds-ratio preference term added to the SFT loss |
| `max_seq_length` | int | `1024` | Total sequence limit; mapped to TRL's `max_length` |
| `batch_size` | int | `8` | Per-device train batch size |
| `gradient_accumulation_steps` | int | `1` | Gradient accumulation steps |
| `learning_rate` | float | `5e-6` | Optimizer learning rate |
| `num_epochs` | int | `1` | Number of training epochs |
| `max_steps` | int | `-1` | Hard cap on optimizer steps; takes precedence over `num_epochs` when positive |

The installed TRL `ORPOConfig` (`trl==1.7.1`) does **not** accept
`max_prompt_length` or `truncation_mode`; set the overall limit with
`max_seq_length` only. Passing either directly (bypassing the wrapper) raises
`TypeError`.

<!-- --8<-- [end:orpo] -->
---

<!-- --8<-- [start:online-dpo] -->
### Online-DPO

Config: `OnlineDPOConfig` (`trl.experimental.online_dpo`). Structurally different from DPO, uses **prompt-only** data (`task_type="grpo"`) and generates completions live during training, requiring reward functions/reward models rather than precomputed preference pairs.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | `3` | Training epochs (higher default than other algorithms) |
| `gradient_accumulation_steps` | int | `2` | Grad accumulation (higher default than other algorithms) |
| `learning_rate` | float | `5e-7` | LR, much lower than other algorithms' `5e-5`, appropriate for online RL |
| `max_new_tokens` | int | `64` | Tokens to generate per completion |
| `max_length` | int | `512` | Max total sequence length |
| `temperature` | float | `0.9` | Generation temperature |
| `top_p` | float | `1.0` | Nucleus sampling |
| `top_k` | int | `0` | Top-k sampling |
| `repetition_penalty` | float | `1.0` | Repetition penalty |
| `beta` | float | `0.1` | KL/temperature coefficient |
| `loss_type` | str | `'sigmoid'` | Online DPO loss variant |
| `missing_eos_penalty` | float | `1.0` | Penalty for completions missing EOS |
| `eval_steps` | int | `500` | Higher default than other algorithms' `100` |
| `save_steps` | int | `500` | Higher default than other algorithms' `100` |
| `save_total_limit` | int | `3` | Different default than other algorithms' `None` |

⚠️ **TRL-backend-only bug**: `OnlineDPOConfig` defaults `bf16=True` unconditionally unless `fp16` is explicitly set, the **Unsloth backend already fixes this** (resolves the actual flags via `PrecisionHandler` for the detected GPU), but the **TRL backend does not**, so Online-DPO via the TRL backend can crash on pre-Ampere GPUs (V100, T4, etc.).

**Unsloth-only fixes**: generalizes the `apply_qkv`/`apply_o` attention patch to `Qwen2Attention` (fixes `'Qwen2Attention' object has no attribute 'apply_qkv'` during generation) and wraps Unsloth-loaded causal-LM reward models with a pooling `.score` head, since `FastLanguageModel.from_pretrained` doesn't wire one up automatically (governed by `config.model.reward_value_loading_type`).

<!-- --8<-- [end:online-dpo] -->
---


<!-- --8<-- [start:spin] -->
### SPIN (Self-Play Fine-Tuning)

Config: `DPOConfig`. SPIN starts from an SFT-style dataset, generates synthetic rejected responses, uses the dataset completion as the chosen response, and runs DPO training each round. It is available through both the TRL and Unsloth backends.

SPIN expects data that can be normalized to a prompt and a reference/completion; it does not require pre-built `chosen`/`rejected` pairs.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_rounds` | int | `2` | Number of self-play rounds |
| `samples_per_round` | int/None | `None` | Number of training rows used in each round. Each round consumes a non-overlapping slice; the dataset must contain at least `num_rounds * samples_per_round` rows. |
| `generation_batch_size` | int | `per_device_batch_size` (or `8`) | Batch size used for synthetic response generation |
| `generation_temperature` | float | `0.7` | Temperature for opponent-response generation |
| `generation_max_length` | int | `512` | Max new tokens for opponent generation |
| `generation_max_prompt_length` | int/None | `max_prompt_length` (or `512`) | Prompt token limit during generation |
| `generation_top_p` | float | `top_p` (or `0.95`) | Nucleus-sampling parameter |
| `generation_top_k` | int | `top_k` (or `0`) | Top-k sampling parameter |
| `generation_repetition_penalty` | float | `repetition_penalty` (or `1.0`) | Repetition penalty during generation |
| `generation_do_sample` | bool/None | `None` | Sampling override; when unset, sampling follows whether temperature is positive |
| `generation_kwargs` | dict | `{}` | Additional generation arguments; these override SPIN defaults |
| `enable_thinking` | bool | `False` | Enables thinking mode when supported by the tokenizer/chat template |
| `dpo_steps_per_round` | int | `100` | DPO training steps per round. A positive `max_steps` takes precedence. |
| `max_length`/`max_seq_length` | int | `2048` | Total sequence length |
| `beta` | float | `0.1` | DPO KL-regularization strength |
| `loss_type` | str/list | `'sigmoid'` (wrapped in a list) | Same list-wrapping requirement as DPO |
| `label_smoothing` | float | `0.0` | DPO label-smoothing value |
| `eval_samples` | int/None | `None` | Maximum validation rows used for response generation and DPO evaluation |

**Evaluation:** Set `eval_strategy="no"` to disable validation-pair generation and DPO evaluation. Otherwise, validation pairs are generated from a fixed validation subset each round; `eval_samples` can cap its size.

**Performance:** Synthetic responses are generated in batches at the start of each round. Runtime is primarily determined by the number of rows per round, generation batch size, number of rounds, and generation length. Use a small `samples_per_round` and completion length for smoke tests.

<!-- --8<-- [end:spin] -->
---

<!-- --8<-- [start:grpo-family] -->
### GRPO (base) / GSPO / DAPO / Dr. GRPO

GSPO, DAPO, and Dr. GRPO are **not independent implementations**, both backends implement them as thin subclasses of the GRPO trainer that only override one or two config defaults (if not already set by the caller) before calling the GRPO parent unchanged. Their full hyperparameter surface **is** GRPO's surface below, plus:

| Algorithm | Override(s) |
|---|---|
| **GSPO** | `importance_sampling_level` → `'sequence'`; `loss_type` → `'dapo'` |
| **DAPO** | `loss_type` → `'dapo'` |
| **Dr. GRPO** | `loss_type` → `'dr_grpo'` |

GRPO uses the common RL parameters plus:

| Parameter | Type | TRL Default | Unsloth Default | Description |
|-----------|------|---------|---------|-------------|
| `learning_rate` | float | `1e-6` | `2e-4` | Optimizer LR |
| `per_device_batch_size` | int | `1` | `4` | Per-device train batch size |
| `gradient_accumulation_steps` | int | `32` | `1` | Grad accumulation |
| `max_grad_norm` | float | `1.0` | `0.5` | Gradient clipping |
| `warmup_steps` | int | `10` | `0` | LR warmup steps |
| `beta` / `kl_coef` | float | `0.1` | `0.1` | KL coefficient |
| `epsilon` / `cliprange` | float | `0.2` | `0.2` | Clip range |
| `epsilon_high` | float or `None` | `None` | `None` | Optional asymmetric upper clip bound; available through generic config backfill |
| `loss_type` | str | `'grpo'` | `'grpo'` | Exactly one of `grpo`, `dapo`, `bnpo`, `dr_grpo`, `cispo`, `sapo`, `luspo`, `vespo`; unsupported values (including DPO `'sigmoid'`) reset to `'grpo'` |
| `importance_sampling_level` | str | `'token'` | `'token'` | `'token'` or `'sequence'` (GSPO sets `'sequence'`) |
| `num_generations` | int | `4` | `= per_device_batch_size` | Generations per prompt |
| `max_completion_length` | int | `256` | derived: 40% of `max_seq_length` | Max completion tokens |
| `temperature` | float | `0.7` | `0.7` | Sampling temperature |
| `top_p` | float | `0.95` | `0.9` | Nucleus sampling |
| `top_k` | int | `0` | `0` | Top-k sampling; `0` disables it |
| `reward_weights` | list[float] | `None` | `None` | One weight per configured reward function |
| `max_steps` | int | `500` | `500` | Max training steps |
| `rollout_backend` | str | `'hf'` | *(not exposed)* | `'hf'` or `'vllm'`: **TRL-only** |
| `vllm_gpu_memory_utilization` | float | `0.7` | *(not exposed)* | vLLM GPU mem fraction. **TRL-only** |
| `vllm_tensor_parallel_size` | int | `1` | *(not exposed)* | vLLM tensor-parallel degree. **TRL-only** |
| `scale_rewards` | str | `'group'` | `'group'` | Reward normalization scope; Unsloth forwards explicitly, TRL receives it through generic backfill |
| `mask_truncated_completions` | bool | `False` | `True` | Whether to mask truncated completions; Unsloth forwards explicitly, TRL receives it through generic backfill |

⚠️ **TRL-vs-Unsloth capability gap**: Unsloth reads `scale_rewards` and `mask_truncated_completions` explicitly, while TRL relies on generic config backfill for these fields. `epsilon_high` is also available through that backfill path rather than the explicit constructor arguments.

<!-- --8<-- [end:grpo-family] -->
---


<!-- --8<-- [start:gbmpo] -->
### GBMPO (Group-Based Mirror PO)

Both backends **subclass the full GRPO trainer** and patch `_compute_loss`: GBMPO inherits every GRPO parameter above, plus:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `gbmpo_divergence_type` / `divergence_type` | str | `'l2'` at the trainer level, but **`create_rl_trainer(algorithm="gbmpo")` defaults it to `'l2kl'`** | One of `l2`, `l2kl`, `prob_l2`, `prob_l2kl` |
| `gbmpo_l2_coefficient` / `l2_coefficient` | float | `0.0001` | λ coefficient for the added L2 penalty term |

If `divergence_type` is L2-only (`l2`/`prob_l2`) and `beta==0.0`, `beta` is silently bumped to `1e-10` for TRL compatibility.

<!-- --8<-- [end:gbmpo] -->
---

<!-- --8<-- [start:counterfactual-grpo] -->
### Counterfactual GRPO

Config: `GRPOConfig`. This has the largest TRL/Unsloth backend divergence of any algorithm in this doc, **learning rate, gradient accumulation, precision-auto-mapping, and max_completion_length/max_prompt_length defaults all differ between backends**; check the table below carefully before assuming parity.

| Parameter | Type | TRL Default | Unsloth Default |
|---|---|---|---|
| `learning_rate` | float | `1e-7` | `5e-5` |
| `gradient_accumulation_steps` | int | `32` | `1` |
| `max_grad_norm` | float | `0.0` (disables clipping) | `0.0` |
| `beta` / `kl_coef` | float | `0.005` | `0.005` |
| `loss_type` | str | `'dapo'` (sigmoid backfill guarded) | `'dapo'` (sigmoid backfill guarded) |
| `max_completion_length` | int | `256` | `768` |
| `max_prompt_length` | int | `512` (not forwarded) | `256` (not forwarded) |
| `precision="auto"` maps to | n/a | `bf16` | `fp16` |

**Counterfactual-importance-weighting parameters** (identical names/defaults in both backends):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `boost_factor` | float | `2.0` | Max importance-weight multiplier for high-importance tokens |
| `min_weight` | float | `0.5` | Minimum importance-weight multiplier |
| `answer_weight` | float | `1.5` | Weight applied to the "answer region" of the completion |
| `max_spans` | int | `10` | Max spans considered per completion for span-based importance detection |
| `method_name` | str | `'counterfactual'` | `"baseline"` maps to vanilla GRPO weighting |
| `random_importance` | bool | `False` | Use random (not counterfactual) importance weights |
| `invert_importance` | bool | `False` | Invert importance direction |
| `enable_gradient_conservation` | bool | `True` via `create_rl_trainer()`/the trainer class default (⚠️ the trainer's own config-read fallback is `False`: only matters if going around the factory) | Normalize weights to mean=1.0 to conserve total gradient magnitude |
| `weight_debug` | bool | `False` | Print detailed per-step weight-computation debug info |
| `weighting_mode` | str | derived from the legacy flags above if unset | `"counterfactual"`, `"random"`, `"inverted"`, or `"vanilla"` |
| `extra_verbose` | bool | `False` | Per-sample JSONL logging (not exposed by `create_rl_trainer()`'s own signature) |
| `extra_verbose_sample_rate` | float | `0.1` | Fraction of samples logged when `extra_verbose=True` |

⚠️ **Unsloth backend does not read `random_importance`/`invert_importance` at all** in `setup_trainer()`: setting either directly has no effect there unless `weighting_mode` is set explicitly.

⚠️ **Design note, not a bug**: `TRLCounterFactGRPOTrainer`'s reward computation is hardcoded to domain-specific math/code-execution logic and never uses `self.reward_functions`/user-configured rewards at all, regardless of what's passed via `rewards=[...]`.

<!-- --8<-- [end:counterfactual-grpo] -->
---

<!-- --8<-- [start:pace-bolt] -->
### PACE

PACE subclasses the GRPO trainer.

**PACE-specific parameters** (identical in both backends):

Curriculum sampling (`curriculum_enabled`) is not available in this build; it is
always forced to `False` regardless of what's passed. Set `baseline_enabled`
and/or `use_baseline_advantages` instead.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `baseline_enabled` | bool | `False` | Enables persistent per-prompt baseline tracking |
| `baseline_rho_min` | float | `0.875` | Min EMA forgetting factor |
| `baseline_rho_max` | float | `0.96` | Max EMA forgetting factor |
| `baseline_D_half` | float | `0.5` | KL half-life for adaptive forgetting |
| `baseline_warm_start` | str/None | `None` | Path to a JSON/PKL warm-start file for the baseline table |
| `use_baseline_advantages` | bool | `False` | If `True`, replaces `A = r - mean(r_group)` with `A = r - v̂(x)` (SPO-style) |

**Generic GRPO-family parameters, TRL backend**: `max_steps` (**10**, unusually low), `num_generations` (**8**), `max_completion_length` (**512**), `save_safetensors` (`True`, **not forwarded**, removed from the installed `GRPOConfig`, same fate as `max_prompt_length`): remaining fields match base GRPO's TRL defaults.

**Generic GRPO-family parameters, Unsloth backend**: reads a smaller subset than TRL (no `save_strategy`/`save_total_limit`/`load_best_model_at_end`/`save_safetensors` forwarding), and `max_steps` defaults to **1** (not 10, differs from TRL PACE). Does **not** call the config-backfill mechanism at all (unlike TRL PACE), so no implicit `config.train` field gap-filling happens for Unsloth PACE.

<!-- --8<-- [end:pace-bolt] -->
---

<!-- --8<-- [start:es-non-meta] -->
### ES (Evolution Strategies, non-meta)

A from-scratch black-box optimizer over LoRA weights, no TRL trainer class is involved at all (directly perturbs `peft` LoRA A/B matrices and evaluates via a vLLM rollout backend).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `population_size` | int | `64` | Perturbed LoRA variants generated per iteration |
| `sigma` | float | `0.5` | Std-dev of Gaussian noise added to LoRA A/B matrices |
| `learning_rate` | float | `0.01` | ES weight-update learning rate |
| `prompt_batch_size` | int | `4` | Prompts sampled per iteration |
| `max_new_tokens` | int | `256` | Max generation length |
| `temperature` | float | `0.7` | Sampling temperature |
| `top_p` | float | `0.95` | Nucleus sampling |
| `top_k` | int | `-1` | Top-k sampling; `-1` disables |
| `num_return_sequences` | int | `1` | Generations per prompt per adapter |
| `num_iterations` | int | `1000` | Total ES iterations |
| `save_freq` | int | `100` | Checkpoint save interval (iterations) |
| `tensor_parallel_size` | int | `1` | vLLM tensor parallelism |
| `dtype` | str | `"auto"` | vLLM model dtype |
| `seed` | int | `42` | Reproducibility seed |
| `reward_type` | str | `"math_correctness"` | Default reward fn if `config.rewards` isn't provided |
| `use_unsloth` | bool | `False` | Load the base model backbone via Unsloth |

Related nested configs via `create_es_trainer(...)`: `ESPeftConfig` (`rank=8, alpha=16, dropout=0.05, peft_type="lora"`), `ESModelConfig` (`dtype="auto", max_seq_length=2048, use_peft=True`), `ESDatasetConfig` (standard dataset/curator params, same shape as the RL common Dataset Configuration above).


<!-- --8<-- [end:es-non-meta] -->
---

## Distillation Parameters

Both distillation methods share a base config resolution: Standard Distillation always routes through the `"distillation_offline"` task type regardless of `on_policy`; SDFT uses its own fixed task type. Every method's underlying TRL config class is in `trl.experimental.*`.

Distillation data always uses the `distillation_offline` task schema so TRL's collator receives the required `messages` structure. When no validation dataset is available, the distillation backends automatically set `eval_strategy="no"` and disable evaluation.

<!-- --8<-- [start:distill-standard] -->
### Standard / Offline Distillation

Config: `trl.experimental.distillation.DistillationConfig`.

**Backend status:** use `backend="trl"` as the recommended Standard Distillation path. `backend="unsloth"` is experimental and intended for acceleration; review the Unsloth-specific teacher-loading and on-policy generation caveats below before using it in production.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `learning_rate` | float | `5e-5` (overrides TRL's own default of `1e-6`) | LR |
| `on_policy` | bool | `False` | `False` = offline teacher/data completions; `True` = online student samples. This is the recommended binary API. |
| `lmbda` | float | derived from `on_policy` (`0.0` offline, `1.0` online) | Explicit `lmbda` takes precedence; values between 0 and 1 mix offline and on-policy batches |
| `beta` | float | `1.0` (overrides TRL's own default, also `1.0` coincidentally) | Divergence type: 0=forward KL, 0.5=JSD, 1=reverse KL |
| `temperature` | float | `3.0` (via config-backfill; overrides TRL's own default of `1.0`) | KD softmax temperature |
| `reverse_kl_top_1_mode` | str | `'sampled'` | Reverse-KL top-1 selection: `'sampled'` or `'argmax'`; applies when `beta > 0` and `loss_top_k == 1` |
| `loss_top_k` | int | `1` | Number of top tokens used for the KL/JSD support; `0` uses the full vocabulary |
| `loss_add_tail` | bool | `True` | Include the remaining probability mass outside the selected top-k support |
| `max_length` | int | `1024` (TRL) | Total prompt plus completion length; AlignTune's model default is commonly `512` |
| `max_prompt_length` | int or `None` | `None` | Prompt truncation limit; auto-derived from `max_length - max_completion_length` when unset |
| `max_completion_length` | int | `512` | Maximum generated completion length for on-policy training |
| `num_generations` | int | `1` | Completions generated per prompt in on-policy mode |
| `generation_batch_size` | int or `None` | `None` | Unique prompts generated per optimizer step; auto-derived when unset |
| `top_p` | float | `0.95` | Nucleus sampling for on-policy generation |
| `top_k` | int | `0` | Top-k sampling for on-policy generation; `0` disables it |
| `disable_dropout` | bool | `True` | Disable dropout in student and teacher models |
| `use_vllm` | bool | `False` | Use vLLM for on-policy student generation |
| `vllm_mode` | str | `'colocate'` | vLLM mode: `'colocate'` or `'server'` |
| `vllm_gpu_memory_utilization` | float | `0.3` | GPU memory fraction for colocated vLLM |
| `vllm_tensor_parallel_size` | int | `1` | vLLM tensor-parallel degree |
| `vllm_server_base_url` | str or `None` | `None` | External vLLM server URL when `vllm_mode='server'` |
| `vllm_max_model_length` | int or `None` | `None` | Maximum sequence length for the vLLM engine |
| `vllm_sync_frequency` | int | `1` | Student-weight synchronization frequency |
| `vllm_enable_sleep_mode` | bool | `False` | Offload vLLM weights during optimizer steps |
| `log_completions` | bool | `False` | Log generated prompt/completion samples |
| `log_completions_steps` | int | `100` | Logging interval for generated samples |
| `num_completions_to_print` | int or `None` | `None` | Number of generated samples to print |
| `alpha` | n/a | **not forwarded**. `DistillationConfig` has no `alpha` field, so this is validated by aligntune's own dataclass but always dropped as an invalid param |

**Off-policy vs on-policy:**

- **Off-policy (default):** `on_policy=False` resolves to `lmbda=0.0` and trains from dataset/teacher completions.
- **On-policy:** `on_policy=True` resolves to `lmbda=1.0` and generates completions from the student during training.
- **Mixed:** pass an explicit `lmbda` between `0.0` and `1.0`; an explicit value takes precedence over `on_policy`.

**Unsloth-only**: `max_length` pinned equal to `config.model.max_seq_length` (avoids a logits/labels length desync); `max_completion_length` capped to `min(configured value or 256, max_seq_length // 2)`; precision resolved via `PrecisionHandler` for pre-Ampere safety.

**Teacher-loading quirk (Unsloth backend only)**: the teacher is loaded through Unsloth too (auto-detected via matching `AutoConfig.model_type` with the student): loading the student through Unsloth first monkey-patches that architecture's attention/norm classes process-wide, so a plain-HF-loaded teacher of the *same* architecture family would crash (`'Qwen2Attention' object has no attribute 'apply_qkv'`). Different-family pairs default to plain HF loading; override via `config.model.teacher_use_unsloth`.

**Online distillation (`on_policy=True`), Unsloth backend only, fixed**: Unsloth's patched `model.generate()` forces `torch.inference_mode()` internally for speed, so the generated completion tokens come back as inference tensors. Any later autograd op that touches them, the student log-probs gather in TRL's on-policy divergence loss, crashed with `RuntimeError: Inference tensors cannot be saved for backward`. Unsloth's own RL integration works around this for its supported trainers (GRPO etc.) by cloning the `generate()` output before use; `DistillationTrainer` isn't one of those, so it never got the fix. `UnslothDistillationTrainer.train()` now wraps `self.student_model.generate` with the same clone-on-output pattern for online mode.

<!-- --8<-- [end:distill-standard] -->

<!-- --8<-- [start:distill-sdft] -->
### SDFT (Self-Distillation Fine-Tuning)

Config: `trl.experimental.sdft.SDFTConfig`: its own experimental trainer, not `SFTConfig`-based despite the name.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `teacher_model_kind` | str | `"base"` | `"base"` = frozen base checkpoint as teacher; `"live"`/`"ema"` = EMA of current student |
| `distillation_mode` | str | `"topk_logits"` | One of `"sampled_token"`, `"topk_logits"`, `"full_logits"` |
| `distillation_alpha` | float | `0.5` | Weight blending CE loss vs. distillation loss |
| `distillation_topk` | int | `100` | Number of teacher logits retained for `topk_logits` mode |
| `teacher_update_rate` | float | `0.05` | EMA teacher update rate for live/EMA teacher modes |
| `teacher_sync_steps` | int | `1` | Optimizer-step interval for synchronizing the live/EMA teacher |
| `generate_from_teacher` | bool | `False` | Use the privileged-context teacher prompt for rollout generation instead of the plain student prompt |
| `num_generations` | int | `8` | Sampled generations per example |
| `max_prompt_length` | int | `512` | Maximum prompt length used for SDFT rollouts |
| `max_completion_length` | int | `256` | Max generated completion length |
| `temperature` | float | `1.0` | Rollout sampling temperature |
| `top_p` / `top_k` | float / int | `1.0` / `0` | Rollout sampling controls |
| `repetition_penalty` | float | `1.0` | Rollout repetition penalty |
| `generation_batch_size` | int/None | `None` | Batch size used for rollout generation |
| `steps_per_generation` | int/None | `None` | Reuse a rollout batch for this many optimizer steps |

When no validation dataset is supplied, the SDFT backends force
`eval_strategy="no"` and disable evaluation. Set an evaluation strategy only
when a validation split is available.

Any other `SDFTConfig` field (e.g. `distillation_topk`, `teacher_update_rate`, `teacher_sync_steps`, `max_prompt_length`) is only reachable via the generic backfill mechanism or `extra_params`.

SDFT requires a `privileged_context` column after DataManager processing. For
datasets using another name, set `privileged_context_column` to the raw hint,
feedback, explanation, or retrieval-context column (for example, `"input"` for
an Alpaca-shaped dataset).

<!-- --8<-- [end:distill-sdft] -->
---


<!-- --8<-- [start:raft] -->
## RAFT (Retrieval Augmented Fine-Tuning)

Config: `RaftTrainerConfig`, subclassing `trl.SFTTrainer` directly. ⚠️ **Not wired into `BackendFactory`**, only reachable via the standalone `create_raft_trainer()` function, unlike every other algorithm in this doc.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_golden_docs` | int | `3` | Max relevant documents included per example |
| `max_distractor_docs` | int | `5` | Max irrelevant distractor documents included per example |
| `doc_context_template` | str | `"[DOC {idx}] {title}: {text}"` | Format string for each document block |
| `use_citation_loss` | bool | `True` | ⚠️ Enables citation tracking, but **the citation loss term itself is a documented placeholder**, not numerically implemented (requires generation during training) |
| `citation_loss_weight` | float | `0.1` | Weight for citation loss (currently unused since the loss is a stub) |
| `use_doc_ranking_loss` | bool | `False` | Enable auxiliary document-ranking loss, declared but not referenced in the loss computation |
| `include_doc_ids_in_output` | bool | `False` | Whether answers are expected to contain `[DOC X]` citations |
| `track_citation_quality` | bool | `True` | Enables a string-matching citation-quality metric |

`create_raft_trainer()` also accepts standard `TrainingArguments`/`SFTConfig`-style kwargs (`num_epochs`, `batch_size`, `learning_rate`, `warmup_steps`, `weight_decay`, `logging_steps`, `eval_strategy`, `save_strategy`, `save_steps`, `gradient_accumulation_steps`, `max_grad_norm`).

**Unsloth-only quirks**: forces `packing=True, packing_strategy="bfd"` (Unsloth's compiled `SFTTrainer` otherwise raises "max_length is not enforced" whenever `packing=False`); builds an `SFTConfig` rather than plain `TrainingArguments` (a plain one carries fields that don't exist on this trl version's `SFTConfig`).

<!-- --8<-- [end:raft] -->
---

## Backend Selection

AlignTune supports two backends with automatic fallback:

| Backend | Description | Availability |
|---------|-------------|--------------|
| **TRL** | HuggingFace Transformers RL library | Standard, widely supported |
| **Unsloth** | Optimized training with memory efficiency | Requires Unsloth installation |

**Backend Priority:**
- SFT: Unsloth → TRL
- RL: Unsloth → TRL

**Setting Backend:**
```python
# Auto-select best available
trainer = create_sft_trainer(..., backend="auto")

# Explicit selection
trainer = create_sft_trainer(..., backend="trl")
trainer = create_rl_trainer(..., backend="unsloth", algorithm="grpo")
```

**Note**: When TRL is selected, Unsloth is disabled to prevent interference. GSPO algorithm is TRL-only. See [Unsloth Compatibility](unsloth_compatibility.md) for a full per-algorithm breakdown of TRL-vs-Unsloth issues found and fixed against the currently pinned `trl==1.7.1` + `unsloth==2026.7.2`.

---

## Task Types (SFT)

| Task Type | Description |
|-----------|-------------|
| `INSTRUCTION_FOLLOWING` | Instruction-response pairs |
| `SUPERVISED_FINE_TUNING` | General supervised training |
| `TEXT_CLASSIFICATION` | Text classification tasks |
| `TOKEN_CLASSIFICATION` | Token-level classification (NER) |
| `TEXT_GENERATION` | General text generation |
| `CHAT_COMPLETION` | Multi-turn chat completion |

---

## Convenience Functions

### SFT Training
```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
 model_name="meta-llama/Llama-2-7b-hf",
 dataset_name="tatsu-lab/alpaca",
 backend="auto",
 output_dir="./output",
 num_epochs=3,
 batch_size=4,
 learning_rate=2e-4,
 max_seq_length=512,
 max_samples=1000
)
```

### RL Training
```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
 model_name="meta-llama/Llama-2-7b-hf",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="grpo",
 backend="auto",
 output_dir="./output",
 num_epochs=3,
 batch_size=4,
 learning_rate=2e-4,
 reward_model_name="OpenAssistant/reward-model-deberta-v3-large-v2"
)
```

---

## Notes

- All parameters with **Required** must be explicitly provided
- Parameters with defaults can be omitted to use default values
- Enum parameters accept both enum values and strings (e.g., `"bf16"` or `PrecisionType.BF16`)
- For classification tasks, `num_labels` must be set in ModelConfig
- Reward model configuration supports three modes: pretrained HF, pretrained local, or custom trained
- DPO evaluation is optional and controlled by `dpo_eval_enabled`
- Sample logging is optional for monitoring generation quality during training
- **Split strings must be plain split names** (`"train"`), not HF slice notation (`"train[:N]"`): use `max_samples` for row limits instead
- Many algorithm-specific "advanced" knobs are only reachable via `config.train.extra_params={...}` rather than a first-class `TrainingConfig` field, this doc flags every case found so far, but if a parameter you expect to work silently has no effect, check whether it needs `extra_params` instead

---

## Additional Resources

- [Unsloth Compatibility Guide](unsloth_compatibility.md): per-algorithm TRL/Unsloth version-specific issues
- [Known Issues](ISSUES.md): common problems and their solutions
- [Changelog](CHANGELOG.md): detailed, PR-by-PR change history
- [Algorithm Zoo](algorithms/overview.md): conceptual overview and selection guide
