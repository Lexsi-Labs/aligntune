"""
Unified Registry for Task Types and Training Types.

This module provides shared enums used across SFT and RL trainers.
"""

from enum import Enum


class TaskType(Enum):
    """Unified task types for both SFT and RL training."""

    # SFT Tasks
    SFT = "sft"
    PRETRAINING = "pretraining"
    SUPERVISED_FINE_TUNING = "supervised_fine_tuning"
    TEXT_GENERATION = "text_generation"
    CHAT_COMPLETION = "chat_completion"
    TEXT_CLASSIFICATION = "text_classification"
    TOKEN_CLASSIFICATION = "token_classification"
    INSTRUCTION_FOLLOWING = "instruction_following"

    # RL Tasks (used by DataManager to load RL-specific datasets)
    DPO = "dpo"
    GRPO = "grpo"
    GSPO = "gspo"  # Generalized Scoring Proximal Objective
    GBMPO = "gbmpo"  # Generalized Bregman Mirror Descent Policy Optimization
    DAPO = "dapo"  # Difficulty-Aware Policy Optimization
    DRGRPO = "drgrpo"  # Doctor GRPO
    KTO = "kto"
    ORPO = "orpo"
    RLOO = "rloo"
    PPO = "ppo"
    SIMPO = "simpo"
    SPIN = "spin"
    ONLINE_DPO = "online_dpo"

    # Distillation Tasks
    DISTILLATION = "distillation"
    GOLD = "gold"  # Cross-tokenizer distillation
    SDFT = "sdft"  # Self-distillation fine-tuning
    SDPO = "sdpo"  # Self-distillation + RL rewards
    


class TrainingType(Enum):
    """Training types for backend selection."""
    SFT = "sft"
    RL = "rl"
