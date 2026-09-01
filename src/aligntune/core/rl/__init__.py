"""
RL core module - moved from /rl/core/ to /core/rl/

This module contains all RL-related core functionality including:
- Configuration classes
- Trainer base classes and factories
- Model management
- Evaluation and logging
- Distributed training support
"""

from .config import (
    UnifiedConfig,
    AlgorithmType,
    PrecisionType,
    BackendType,
    ModelConfig,
    DatasetConfig,
    RewardConfig,
    RewardModelTrainingConfig,
    TrainingConfig,
    LoggingConfig,
    DistributedConfig,
)

from .trainer_base import RLTrainerBase, TrainingState
# Alias for backwards compatibility
TrainerBase = RLTrainerBase
from .trainer_factory import create_trainer_from_config, TrainerFactory
from .registries import DatasetRegistry, RewardRegistry, TaskRegistry
from .evaluator import UnifiedEvaluator
from .logging_utils import UnifiedLogger
from .config_loader import ConfigLoader
from .function_based_reward_model import FunctionBasedRewardModel
from .models import PolicyModel, ReferenceModel, ValueModel, ModelManager
from .rollout import RolloutEngine

__all__ = [
    # Config classes
    "UnifiedConfig",
    "AlgorithmType",
    "PrecisionType",
    "BackendType",
    "ModelConfig",
    "DatasetConfig",
    "RewardConfig",
    "RewardModelTrainingConfig",
    "TrainingConfig",
    "LoggingConfig",
    "DistributedConfig",
    # Core classes
    "TrainerBase",
    "RLTrainerBase",
    "TrainingState",
    "create_trainer_from_config",
    "TrainerFactory",
    "DatasetRegistry",
    "RewardRegistry",
    "TaskRegistry",
    "UnifiedEvaluator",
    "UnifiedLogger",
    "ConfigLoader",
    "FunctionBasedRewardModel",
    "PolicyModel",
    "ReferenceModel",
    "ValueModel",
    "ModelManager",
    "RolloutEngine",
]