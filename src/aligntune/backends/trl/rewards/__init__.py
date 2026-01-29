"""
TRL Backend Rewards Module

This module contains reward model training components that are specific to the TRL backend.
Since reward model training uses TRL's RewardTrainer, these components logically belong
under the TRL backend structure.

Components:
- RewardModelTrainer: Orchestrates reward model training using TRL's RewardTrainer
- RewardModelDataset: PyTorch Dataset for reward model training data
- RewardModelValidator: Validates reward model configurations
- RewardModelLoader: Loads reward models from various sources
"""

from .training import (
    RewardModelTrainer,
    RewardModelDataset,
    RewardModelValidator,
    RewardModelLoader,
)

__all__ = [
    "RewardModelTrainer",
    "RewardModelDataset", 
    "RewardModelValidator",
    "RewardModelLoader",
]
