# Changelog

All notable changes to AlignTune will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **SFT Customer Support Chatbot Example (Unsloth Backend)**:
  - New example at `examples/sft_customer_support_unsloth/` for instruction-following customer support chatbot
  - Uses `meta-llama/Llama-3.2-3B-Instruct` model and `bitext/Bitext-customer-support-llm-chatbot-training-dataset`
  - Uses Unsloth backend for faster training and 30-40% less memory usage
  - Includes pre/post-training evaluation with perplexity, ROUGE, and BLEU metrics
  - Memory-optimized training script with 4-bit quantization, LoRA, and gradient accumulation
  - Comprehensive documentation
  - Two training approaches: direct API (`train_sft_direct_api.py`) and config-based (`train_sft_with_eval.py`)
- **SFT Customer Support Chatbot Example (TRL Backend)**:
  - New example at `examples/sft_customer_support_trl/` for instruction-following customer support chatbot
  - Uses `google/gemma-3-4b-it` model and `bitext/Bitext-customer-support-llm-chatbot-training-dataset`
  - Uses TRL backend for maximum compatibility with all HuggingFace models
  - Includes pre/post-training evaluation with perplexity, ROUGE, and BLEU metrics
  - Two training approaches: direct API (`train_sft_direct_api.py`) and config-based (`train_sft_with_eval.py`)
  - ⚠️ **Note**: Not yet fully tested - use with caution
- **SFT Financial Report Summarization Example (TRL Backend)**:
  - New example at `examples/sft_financial_summarization_trl/` for financial report summarization
  - Uses `meta-llama/Llama-3.1-8B-Instruct` model and `FinGPT/fingpt-sentiment-train` dataset
  - Uses TRL backend for maximum compatibility
  - Includes ROUGE-L and BLEU metrics for summarization quality
  - Pre/post-training evaluation with comprehensive metrics
  - ⚠️ **Note**: Not yet fully tested - use with caution
- **SFT Healthcare Diagnosis Assistant Example (TRL Backend)**:
  - New example at `examples/sft_healthcare_diagnosis_trl/` for healthcare diagnosis assistance
  - Uses `microsoft/BioGPT-Large` model and `medalpaca/medical_meadow_medical_flashcards` dataset
  - Uses TRL backend for maximum compatibility
  - Includes ROUGE-L and BLEU metrics
  - Medical disclaimer and safety considerations documented
  - Pre/post-training evaluation with comprehensive metrics
  - ⚠️ **Note**: Not yet fully tested - use with caution

### Fixed
- **Critical Training Duration Bug**: Fixed `max_steps` defaulting to 10 steps instead of using provided value or calculating from epochs
  - `max_steps` now properly calculates from dataset size and epochs when not specified
  - Training will now run for the full duration as intended
- **PEFT/LoRA Evaluation Issues**:
  - Fixed vocab size mismatch when loading PEFT adapters (resize embeddings before loading adapter)
  - Fixed base model loading to use same quantization config as training
  - Fixed adapter loading to use `is_trainable=False` for inference mode
  - Fixed evaluation caching to ensure fresh results
- **Unsloth Model Evaluation**: Fixed `AttributeError: 'LlamaAttention' object has no attribute 'apply_qkv'` when evaluating Unsloth-trained models
  - Evaluation now properly detects Unsloth models and uses `FastLanguageModel.from_pretrained()` to preserve Unsloth patches
  - Auto-detection via backend parameter or `training_config.yaml` check
  - Both direct API and config-based evaluation scripts updated
- **Evaluation Config Loading**: Fixed `SFTConfigLoader` to properly load `evaluation` section from YAML config
  - Added support for `enabled`, `pre_training`, `post_training`, `eval_dataset`, and `metrics` fields
  - Evaluation now runs correctly when configured in YAML
- **Parameter Passing**: Fixed `create_sft_trainer` to properly pass all LoRA parameters (`lora_r`, `lora_alpha`, `lora_target_modules`) to model config
- **Logging and Saving**: Fixed hardcoded `logging_steps=10` and `save_steps=500` to use config values

### Changed
- **Comprehensive Parameter Support**: Added all missing parameters to `create_sft_trainer`:
  - ModelConfig: `bias`, `attn_implementation`, `max_memory`, `use_gradient_checkpointing`
  - TrainingConfig: `eval_interval`, `max_grad_norm`, `fp16`/`bf16`, `dataloader_num_workers`, `remove_unused_columns`, `optimizer`, `lr_scheduler`, `dataloader_drop_last`, `eval_accumulation_steps`, `label_smoothing_factor`, early stopping parameters, model selection parameters
  - LoggingConfig: `log_level`, `save_strategy`, `eval_strategy`
- **Better Default Hyperparameters**: Updated SFT customer support example with improved defaults:
  - LoRA rank: 4 → 32 (more capacity)
  - LoRA alpha: 4 → 32 (proper 1:1 ratio, standard for higher ranks)
  - Target modules: `[q_proj, v_proj]` → `[q_proj, k_proj, v_proj, o_proj]` (all attention layers)
  - Learning rate: 2e-4 → 1e-4 (more stable)
  - Added warmup_ratio: 0.1
  - Max sequence length: 256 → 512 (better for instruction following)
  - Gradient accumulation: 4 → 8 (effective batch size = 32)
  - Max steps: 500 → 250 (better convergence with larger effective batch)
- **Unified Evaluation System**: Both `train_sft_direct_api.py` and `train_sft_with_eval.py` now use the same improved evaluation:
  - Uses `aligntune.eval.evaluator.BaseEvaluator` for consistent metrics
  - Proper PEFT/LoRA model loading with quantization config matching
  - Embedding resizing for vocab size changes
  - Cache clearing for fresh evaluation results
  - ROUGE and BLEU metrics in addition to perplexity
  - Unsloth model support with proper FastLanguageModel loading
- **Example Organization**: Reorganized SFT examples with backend-specific naming:
  - `sft_customer_support_trl/` → `sft_customer_support_unsloth/` (changed to Unsloth backend)
  - `sft_financial_summarization_trl/` (TRL backend)
  - `sft_healthcare_diagnosis_trl/` (TRL backend)
  - Updated examples README with backend comparison and clear naming convention

## [0.2.0] - 2026-01-18

### Changed (Major)
- **Renamed library from `finetunehub` to `aligntune`**
  - Package directory: `src/finetunehub/` → `src/aligntune/`
  - All imports updated: `from finetunehub.*` → `from aligntune.*`
  - CLI commands: `aligntune` (primary), `at` (short alias)
  - Entry points and pyproject.toml fully updated
  - All documentation, examples, and tests migrated

- **License changed from MIT to AlignTune Source Available License (ASAL) v1.0**
  - New source-available license based on Lexsi Labs LSAL structure
  - Free for research, academic, and personal use
  - Commercial use requires separate license
  - Updated across all documentation and configuration files

### Added
- **Comprehensive Documentation System** (59 markdown files):
  - Complete documentation structure integrated into main project
  - Getting Started guides (5 docs): Installation, Quick Start, Basic Concepts, Configuration, Backend Selection
  - User Guide (9 docs): Overview, SFT, RL, Evaluation, Reward Functions, Model Management, Sample Logging, Troubleshooting
  - Algorithm Documentation (10 docs): DPO, PPO, GRPO, GSPO, DAPO, Dr. GRPO, GBMPO, Counterfactual GRPO, BOLT with detailed explanations
  - Backend Documentation (4 docs): Overview, TRL Backend, Unsloth Backend, Comparison
  - API Reference (6 docs): Complete API documentation with all parameters
  - Examples (4 categories): Overview, SFT, RL, Advanced examples
  - Advanced Topics (4 docs): Architecture, Custom Backends, Distributed Training, Performance
  - Contributing guides (3 docs): Guide, Code Style, Testing
  - Notebooks section for interactive tutorials

- **Enhanced Documentation Infrastructure**:
  - MkDocs configuration with Cinder theme
  - Mermaid diagram support for architecture visualization
  - Jupyter notebook integration
  - Automatic API documentation generation with mkdocstrings
  - Custom HTML overrides and styling
  - Logo and branding assets (aligntune-logo.png, aligntune-banner.png)

### Removed (Code Cleanup)
- **Deleted unused/backup directories and files** (Total: ~7,500 lines removed):
  - `src/finetunehub/eval_old/` - Old evaluation framework (7 files, 2,913 lines)
  - `src/finetunehub/backends/trl/rl/ppo/ppo_old.py` - Old PPO implementation (1,437 lines)
  - `src/finetunehub/backends/trl/rl/grpo/grpo_old.py` - Old GRPO implementation (1,264 lines)
  - `src/finetunehub/backends/unsloth/rl/ppo/ppo_old.py` - Old Unsloth PPO (1,833 lines)
  - `src/finetunehub/cli_commands/` - Unused CLI modules
  - `src/finetunehub/cli/unified-old.py` - Old CLI backup
  - `src/finetunehub/rl/` and `src/finetunehub/sft/` - Backward compatibility wrappers
  - `src/finetunehub/scripts/` - Directory removed (contents moved)
- **Removed old `finetunehub` package directory** - Fully replaced by `aligntune`

### Changed
- **Reorganized BOLT utilities**:
  - Moved `precompute_baseline.py` from `src/finetunehub/scripts/` → `examples/bolt_training/`
  - Rationale: Co-locate BOLT-specific utilities with BOLT examples for better discoverability

### Breaking Changes
- **Library renamed**: All imports must change from `finetunehub` to `aligntune`
  - `from finetunehub.core.rl import *` → `from aligntune.core.rl import *`
  - `from finetunehub.eval import *` → `from aligntune.eval import *`
  - CLI: `finetunehub train` → `aligntune train`
- **Removed backward compatibility wrappers**: The deprecated import paths have been removed
  - All examples and documentation have been updated to use the new paths
  - All test files have been migrated to the new import structure

## [0.1.0] - 2026-01-18

### Added
- Initial release with comprehensive GRPO support
- Backend support for both TRL and Unsloth
- GSM8K math reasoning examples
- Multiple bug fixes and improvements
