# AlignTune Examples

This directory contains complete, production-ready examples for training language models with AlignTune.


### Supervised Fine-Tuning (SFT)
| Backend | Model | Dataset | Link |
|---------|-------|---------|------|
| TRL | Gemma-7b | philschmid/dolly-15k-oai-style (messages format) | [sft_trl_1.py](sft_trl_1.py) |
| TRL | GemmaTX | TrialBench adverse event prediction | [sft_trl_2.py](sft_trl_2.py) |
| Unsloth | Qwen-2.5-0.5-I | gaming | [unsloth_sft_1.py](unsloth_sft_1.py) |

### Reinforcement Learning (RL)

| Backend | Algorithm | Dataset | Link |
|---------|-----------|---------|------|
| TRL | DPO | hh-rlhf | [trl_dpo_1.py](trl_dpo_1.py) |
| TRL | GRPO | GSM8K | [trl_grpo_1.py](trl_grpo_1.py) |
| TRL | PPO | openai_summarize_tldr | [trl_ppo1.py](trl_ppo1.py) |
| Unsloth | DPO | distilabel-intel-orca-dpo-pairs | [unsloth_dpo_1.py](unsloth_dpo_1.py) |
| Unsloth | GRPO | GSM8K | [unsloth_grpo_1.py](unsloth_grpo_1.py) |


