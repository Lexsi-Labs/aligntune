"""
Reward functions for RL training.

This module provides comprehensive reward functions for reinforcement learning
from human feedback (RLHF) training, covering all types of tasks and scenarios.
"""

from .core import (
    RewardType,
    RewardConfig,
    RewardFunction,
    RewardFunctionFactory,
    CompositeReward,
    LengthReward,
    CoherenceReward,
    SentimentReward,
    SafetyReward,
    BLEUReward,
    ROUGEReward,
    MathCorrectnessReward,
    CodeSyntaxReward,
    ToxicityReward,
)

# Import training classes from TRL backend (re-export for backward compatibility)
try:
    from ..backends.trl.rewards import (
        RewardModelTrainer,
        RewardModelDataset,
        RewardModelValidator,
        RewardModelLoader,
    )
except ImportError:
    # Fallback to local training module if TRL backend not available
    from .training import (
        RewardModelTrainer,
        RewardModelDataset,
        RewardModelValidator,
        RewardModelLoader,
    )

# Import factory module
from . import factory

# Register default reward functions
from . import registry

__all__ = [
    "RewardType",
    "RewardConfig", 
    "RewardFunction",
    "RewardFunctionFactory",
    "CompositeReward",
    "LengthReward",
    "CoherenceReward",
    "SentimentReward",
    "SafetyReward",
    "BLEUReward",
    "ROUGEReward",
    "MathCorrectnessReward",
    "CodeSyntaxReward",
    "ToxicityReward",
    "RewardModelTrainer",
    "RewardModelDataset",
    "RewardModelValidator",
    "RewardModelLoader",
    "factory",
    "registry",
]
