# AlignTune Examples

This directory contains complete, production-ready examples for training language models with AlignTune.


### Supervised Fine-Tuning (SFT)
| Backend | Model | Dataset | Link |
|---------|-------|---------|------|
| TRL | Gemma-7b | philschmid/dolly-15k-oai-style (messages format) | [sft_trl_1.py](sft_trl_1.py) |
| TRL | GemmaTX | TrialBench adverse event prediction | [sft_trl_2.py](sft_trl_2.py) |
| Unsloth | Qwen-2.5-0.5-I | gaming | [unsloth_sft_1.py](unsloth_sft_1.py) |
| TRL | Qwen3-4B-Instruct | Retail banking chatbot | [retail_banking_sft.py](retail_banking_sft.py) |
| TRL | Qwen3-4B-Instruct-2507 | Wealth management chatbot | [wealth_management_sft.py](wealth_management_sft.py) |
| Unsloth | Qwen2.5-0.5B-Instruct | Instruction following | [unsloth1_instruction_following.py](unsloth1_instruction_following.py) |
| TRL | txgemma-2b-predict | TrialBench adverse-event-rate prediction | [txgemma_trialbench_sft_trl_backend_eval.py](txgemma_trialbench_sft_trl_backend_eval.py) |

### Reinforcement Learning (RL)

| Backend | Algorithm | Dataset | Link |
|---------|-----------|---------|------|
| TRL | DPO | hh-rlhf | [trl_dpo_1.py](trl_dpo_1.py) |
| TRL | GRPO | GSM8K | [trl_grpo_1.py](trl_grpo_1.py) |
| TRL | GRPO (math) | GSM8K | [trl_grpo_1_math.py](trl_grpo_1_math.py) |
| TRL | GRPO (code) | MBPP | [grpo_code_gen_trl_mbpp.py](grpo_code_gen_trl_mbpp.py) |
| TRL | PPO | openai_summarize_tldr | [trl_ppo1.py](trl_ppo1.py) |
| TRL | PPO (pretrained reward) | Summarization | [trl_ppo_pretrainedreward_summarization.py](trl_ppo_pretrainedreward_summarization.py) |
| TRL | DAPO | Code | [dapo_trl__code.py](dapo_trl__code.py) |
| TRL | DPO | Wealth management preference data | [wealth_dpo_training_evaluation_full.py](wealth_dpo_training_evaluation_full.py) |
| Unsloth | DPO | distilabel-intel-orca-dpo-pairs | [unsloth_dpo_1.py](unsloth_dpo_1.py) |
| Unsloth | DPO | Intel Orca (phi chat) | [unsloth_dpo_intel_orca_phi_chat.py](unsloth_dpo_intel_orca_phi_chat.py) |
| Unsloth | GRPO | GSM8K | [unsloth_grpo_1.py](unsloth_grpo_1.py) |
| Unsloth | Dr. GRPO | alpaca-cleaned | [drgrpo_unsloth.py](drgrpo_unsloth.py) |
| Unsloth | GSPO | Generic demo | [gspo_generic_demo_unsloth.py](gspo_generic_demo_unsloth.py) |
| Unsloth | PPO | ultrachat | [unsloth_ppo_ultrachat.py](unsloth_ppo_ultrachat.py) |

The `(1).py`-suffixed files above and the ones listed here are exported Colab notebook sources, see [docs/notebooks.md](../docs/notebooks.md#demo-notebooks-colab) for the matching runnable Colab links (these scripts include Colab-only setup cells like `!git clone`/`!pip install` and are not meant to be run standalone).

### Standalone Feature Examples

Scripts demonstrating one specific capability end-to-end, independent of any Colab notebook:

| Script | Demonstrates |
|--------|--------------|
| [curriculum_rl_example.py](curriculum_rl_example.py) | Curriculum learning during RLHF training |
| [alignment_audit_example.py](alignment_audit_example.py) | `AlignmentAuditor` / `AlignmentAuditCallback` for tracking alignment metrics during training |
| [reward_tracking_example.py](reward_tracking_example.py) | Per-component reward tracking and visualization |
| [load_raw_files_example.py](load_raw_files_example.py) | Raw file loaders (local text/JSON/CSV data instead of a Hub dataset) |

### Custom Evaluation Tasks

[`custom_tasks/`](custom_tasks/) contains custom `lm-eval` task definitions (YAML + helper `utils.py`) for domain-specific evaluation, see each subfolder's own README (`bitext_2/`, `bitext_3/`, `bitext_insurance/`).


