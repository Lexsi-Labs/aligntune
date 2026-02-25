# AlignTune Configuration Parameters

This document provides a comprehensive reference for all configuration parameters available in AlignTune, organized by training type (SFT vs RL) and algorithm-specific settings.

---

## Table of Contents

1. [SFT (Supervised Fine-Tuning) Parameters](#sft-supervised-fine-tuning-parameters)
2. [RL (Reinforcement Learning) Common Parameters](#rl-reinforcement-learning-common-parameters)
3. [Algorithm-Specific RL Parameters](#algorithm-specific-rl-parameters)
 - [PPO (Proximal Policy Optimization)](#ppo-proximal-policy-optimization)
 - [DPO (Direct Preference Optimization)](#dpo-direct-preference-optimization)
 - [GRPO (Group Relative Policy Optimization)](#grpo-group-relative-policy-optimization)
 - [DAPO / DRGRPO](#dapo-drgrpo)
 - [GSPO (Generalized Scoring Proximal Objective)](#gspo-generalized-scoring-proximal-objective)
 - [Neural Mirror GRPO](#neural-mirror-grpo)
 - [Meta-ES](#meta-es)

---

## SFT (Supervised Fine-Tuning) Parameters

### Model Configuration (`ModelConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name_or_path` | str | **Required** | HuggingFace model name or local path |
| `precision` | PrecisionType | `bf16` | Model precision (`bf16`, `fp16`, `fp32`, `auto`) |
| `quantization` | Dict | `{}` | Quantization config (e.g., `{"load_in_4bit": True}`) |
| `attn_implementation` | str | `"auto"` | Attention implementation (`auto`, `flash_attention_2`, `sdpa`) |
| `gradient_checkpointing` | bool | `True` | Enable gradient checkpointing for memory efficiency |
| `max_memory` | Dict | `None` | Max memory per device (e.g., `{"0": "20GB"}`) |
| `device_map` | str/Dict | `"auto"` | Device mapping strategy |
| `use_unsloth` | bool | `False` | Enable Unsloth acceleration |
| `max_seq_length` | int | `2048` | Maximum sequence length |
| `peft_enabled` | bool | `False` | Enable PEFT/LoRA |
| `lora_rank` | int | `16` | LoRA rank |
| `lora_alpha` | int | `32` | LoRA alpha scaling |
| `lora_dropout` | float | `0.1` | LoRA dropout rate |
| `target_modules` | List[str] | `None` | LoRA target modules (e.g., `["q_proj", "v_proj"]`) |
| `bias` | str | `"none"` | Bias training (`none`, `all`, `lora_only`) |
| `use_gradient_checkpointing` | bool | `True` | Enable gradient checkpointing (legacy) |
| `num_labels` | int | `None` | Number of labels (classification tasks) |
| `model_init_kwargs` | Dict | `{}` | Additional model initialization arguments |

### Dataset Configuration (`DatasetConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | **Required** | HuggingFace dataset name or local path |
| `split` | str | `"train"` | Dataset split to use |
| `subset` / `config` | str | `None` | Dataset subset/config name |
| `percent` | float | `None` | Percentage of data to use (0-100) |
| `max_samples` | int | `None` | Maximum number of samples |
| `column_mapping` | Dict | `{}` | Column name mappings |
| `task_type` | TaskType | `SUPERVISED_FINE_TUNING` | Task type enum |
| `auto_detect_fields` | bool | `False` | Auto-detect dataset fields |
| `format_type` | str | `None` | Dataset format (`alpaca`, `dolly`, etc.) |
| `text_column` | str | `"text"` | Text column name |
| `instruction_column` | str | `"instruction"` | Instruction column name |
| `response_column` | str | `"response"` | Response column name |
| `output_column` | str | `"output"` | Output column name |
| `input_column` | str | `"input"` | Input column name |
| `context_column` | str | `"context"` | Context column name |
| `label_column` | str | `"label"` | Label column (classification) |
| `tokens_column` | str | `"tokens"` | Tokens column (token classification) |
| `tags_column` | str | `"ner_tags"` | Tags column (NER) |
| `messages_column` | str | `"messages"` | Messages column (chat) |
| `dataset_text_field` | str | `"text"` | Dataset text field for trainers |
| `chat_template` | str | `None` | Chat template string |
| `dataset_num_proc` | int | `None` | Number of parallel processes |
| `pad_token` | str | `None` | Padding token |
| `preserve_columns` | List[str] | `None` | Columns to preserve during processing |
| `processing_fn` | Callable | `None` | Custom processing function |
| `processing_batched` | bool | `False` | Whether processing is batched |
| `processing_fn_kwargs` | Dict | `{}` | Arguments for processing function |

### Training Configuration (`TrainingConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `per_device_batch_size` | int | `1` | Batch size per device |
| `gradient_accumulation_steps` | int | `1` | Gradient accumulation steps |
| `max_steps` | int | `None` | Maximum training steps |
| `epochs` | int | `3` | Number of training epochs (if max_steps not set) |
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
| `group_by_length` | bool | `False` | Group sequences by length |
| `dataloader_drop_last` | bool | `False` | Drop last incomplete batch |
| `eval_accumulation_steps` | int | `None` | Evaluation accumulation steps |
| `label_smoothing_factor` | float | `0.0` | Label smoothing factor |
| `early_stopping_patience` | int | `None` | Early stopping patience |
| `early_stopping_threshold` | float | `0.0` | Early stopping threshold |
| `load_best_model_at_end` | bool | `True` | Load best model at end |
| `metric_for_best_model` | str | `"eval_loss"` | Metric for best model selection |
| `greater_is_better` | bool | `False` | Whether higher metric is better |
| `use_trl` | bool | `False` | Use TRL backend |
| `dataset_num_proc` | int | `None` | Dataset processing processes |
| `dataset_kwargs` | Dict | `{}` | Additional dataset arguments |
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
| `eval_strategy` | str | `"steps"` | Evaluation strategy |
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
| `precision` | PrecisionType | `AUTO` | Model precision |
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
| `split` | str | `"train"` | Dataset split |
| `percent` | float | `None` | Percentage of data (0-100) |
| `max_samples` | int | `None` | Maximum samples |
| `field_mappings` | Dict | `{}` | Field name mappings |
| `format_type` | str | `None` | Dataset format type |
| `auto_detect_fields` | bool | `True` | Auto-detect fields |
| `column_mapping` | Dict | `{}` | Column mappings (legacy) |
| `task_type` | str | `"conversation"` | Task type |
| `weight` | float | `1.0` | Dataset weight |
| `chat_template` | str | `None` | Chat template |
| `system_prompt` | str | `None` | System prompt |
| `dataset_num_proc` | int | `None` | Processing processes |
| `pad_token` | str | `None` | Padding token |
| `label_pad_token_id` | int | `-100` | Label padding token ID |
| `truncation_mode` | str | `"keep_end"` | Truncation mode |
| `padding_free` | bool | `False` | Padding-free training |
| `precompute_ref_log_probs` | bool | `False` | Precompute reference log probs |
| `precompute_ref_batch_size` | int | `None` | Batch size for precomputation |
| `tools` | Any | `None` | Tools configuration |
| `preserve_columns` | List[str] | `None` | Columns to preserve |
| `processing_fn` | Callable | `None` | Custom processing function |
| `processing_batched` | bool | `False` | Batched processing |
| `processing_fn_kwargs` | Dict | `{}` | Processing function arguments |
| `config_name` | str | `None` | Dataset config name |

### Training Configuration (`TrainingConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `per_device_batch_size` | int | `1` | Batch size per device |
| `per_device_eval_batch_size` | int | `1` | Eval batch size per device |
| `gradient_accumulation_steps` | int | `1` | Gradient accumulation steps |
| `max_steps` | int | `None` | Maximum training steps |
| `epochs` | int | `None` | Number of epochs |
| `eval_interval` | int | `100` | Evaluation interval |
| `save_interval` | int | `500` | Save interval |
| `learning_rate` | float | `1e-5` | Learning rate |
| `max_grad_norm` | float | `1.0` | Max gradient norm |
| `weight_decay` | float | `0.01` | Weight decay |
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
| `max_prompt_length` | int | `512` | Max prompt length |
| `max_target_length` | int | `None` | Max target length |
| `max_completion_length` | int | `256` | Max completion length |
| `top_p` | float | `0.95` | Nucleus sampling parameter |
| `padding_free` | bool | `False` | Padding-free training |
| `truncation_mode` | str | `"keep_end"` | Truncation mode |
| `beta` | float | `0.1` | DPO beta parameter |
| `loss_type` | str | `None` | Loss type |
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
| `save_steps` | int | `500` | Save steps |
| `save_strategy` | str | `"steps"` | Save strategy |
| `save_total_limit` | int | `None` | Max checkpoints to keep |
| `load_best_model_at_end` | bool | `False` | Load best model at end |
| `metric_for_best_model` | str | `None` | Best model metric |
| `greater_is_better` | bool | `False` | Whether higher is better |
| `logging_steps` | int | `10` | Logging steps |
| `logging_strategy` | str | `"steps"` | Logging strategy |
| `num_generations` | int | `None` | Number of generations per prompt |
| `mask_truncated_completions` | bool | `True` | Mask truncated completions |
| `scale_rewards` | str | `"group"` | Reward scaling (`group`, `batch`) |
| `reward_weights` | List[float] | `None` | Reward function weights |
| `enable_thinking` | bool | `False` | Enable Qwen3 thinking mode |
| `fast_inference` | bool | `False` | Enable Unsloth vLLM (faster) |
| `vllm_gpu_memory_utilization` | float | `0.7` | vLLM GPU memory (0.95 for max) |
| `seed` | int | `42` | Random seed |
| `data_seed` | int | `47` | Data seed |
| `use_liger_kernel` | bool | `False` | Use Liger kernel |
| `use_liger_loss` | bool | `None` | Use Liger loss |
| `gradient_checkpointing_kwargs` | Dict | `{"use_reentrant": False}` | Gradient checkpointing args |
| `group_by_length` | bool | `True` | Group sequences by length |

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

**Note**: For optimal performance, ensure policy_model, reward_model, and value_model are from the same model family.

---

### DPO (Direct Preference Optimization)

DPO uses the common RL parameters plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | float | `0.1` | DPO beta (inverse temperature) |
| `loss_type` | str | `"sigmoid"` | Loss type (`sigmoid`, `hinge`, `ipo`, `kto`) |
| `label_smoothing` | float | `0.0` | Label smoothing factor |
| `reference_free` | bool | `False` | Reference-free DPO |
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

---

### GRPO (Group Relative Policy Optimization)

GRPO uses the common RL parameters plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grpo_alpha` | float | `0.1` | GRPO alpha parameter |
| `grpo_beta` | float | `0.1` | GRPO beta parameter |
| `num_generations` | int | `batch_size` | Generations per prompt |
| `scale_rewards` | str | `"group"` | Reward scaling (`group`, `batch`) |
| `kl_coef` | float | `0.1` | KL penalty coefficient |
| `cliprange` | float | `0.2` | Policy clip range |

---

### DAPO / DRGRPO

**DAPO** (Direct Advantage Policy Optimization) and **DRGRPO** (Distributional Robust GRPO) share identical parameters with GRPO:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grpo_alpha` | float | `0.1` | Alpha parameter |
| `grpo_beta` | float | `0.1` | Beta parameter |
| `num_generations` | int | `batch_size` | Generations per prompt |
| `scale_rewards` | str | `"group"` | Reward scaling |
| `kl_coef` | float | `0.1` | KL penalty coefficient |
| `cliprange` | float | `0.2` | Policy clip range |

---

### GSPO (Generalized Scoring Proximal Objective)

GSPO uses scoring-based policy optimization:

**Note**: GSPO is only supported by TRL backend, not Unsloth.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gspo_gamma` | float | `0.1` | GSPO gamma parameter |
| `gspo_delta` | float | `0.1` | GSPO delta parameter |
| `num_generations` | int | `batch_size` | Generations per prompt |
| `scale_rewards` | str | `"group"` | Reward scaling strategy |
| `kl_coef` | float | `0.1` | KL penalty coefficient |
| `cliprange` | float | `0.2` | Policy clip range |

---


### Neural Mirror GRPO

Neural Mirror GRPO combines GRPO with neural mirror descent for improved optimization:

**Base GRPO Parameters** plus neural mirror-specific optimizations.

Uses standard GRPO parameters with enhanced optimization through mirror descent. No additional unique parameters.

---

### Meta-ES

Meta-ES uses evolution strategies for meta-learning the reward function:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `meta_iterations` | int | `15` | Number of meta-iterations |
| `patience` | int | `5` | Early stopping patience |
| `min_delta` | float | `0.001` | Minimum improvement for early stopping |
| `init_scale` | float | `0.01` | Initial parameter scale |
| `N` | int | `10` | Population size (must be even) |
| `T` | int | `100` | Training steps per evaluation |
| `sigma` | float | `0.01` | ES noise standard deviation |
| `sigma_decay` | float | `0.99` | ES sigma decay per iteration |
| `alpha` | float | `0.01` | ES learning rate |
| `mirror_coefficient` | float | `0.0001` | Mirror descent coefficient |
| `debug_mode` | bool | `False` | Enable debug logging |
| `eval_timeout` | int | `5` | Evaluation timeout (seconds) |
| `eval_max_tokens` | int | `512` | Max tokens for evaluation |
| `eval_k` | int | `1` | Top-k sampling for evaluation |
| `eval_temperature` | float | `0.8` | Temperature for evaluation |
| `num_workers` | int | `1` | Number of parallel workers |
| `no_wandb` | bool | `False` | Disable Weights & Biases logging |
| `wandb_project` | str | `"neural-mirror-es"` | W&B project name |
| `resume` | str | `None` | Path to checkpoint for resuming |

**Usage Notes:**
- Meta-ES optimizes the reward function itself through evolution
- Population size `N` must be even for symmetric perturbations
- `T` controls how long each candidate is trained before evaluation
- `sigma` controls exploration; decay helps convergence
- Suitable for scenarios where reward design is critical

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

**Note**: When TRL is selected, Unsloth is disabled to prevent interference. GSPO algorithm is TRL-only.

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

---

## Additional Resources

- [FinetuneHub Documentation](#)
- [Backend Comparison Guide](#)
- [Algorithm Selection Guide](#)