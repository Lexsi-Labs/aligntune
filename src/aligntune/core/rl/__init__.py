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

from .trainer_base import TrainerBase, TrainingState
from .trainer_factory import create_trainer_from_config, TrainerFactory
from .registries import DatasetRegistry, RewardRegistry, TaskRegistry
from .models import ModelManager, PolicyModel, ReferenceModel, ValueModel
from .rollout import RolloutEngine
from .evaluator import UnifiedEvaluator
from .logging import UnifiedLogger
from .config_loader import ConfigLoader
from .function_based_reward_model import FunctionBasedRewardModel

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
    "TrainingState",
    "create_trainer_from_config",
    "DatasetRegistry",
    "RewardRegistry", 
    "TaskRegistry",
    "ModelManager",
    "PolicyModel",
    "ReferenceModel",
    "ValueModel",
    "RolloutEngine",
    "UnifiedEvaluator",
    "UnifiedLogger",
    "ConfigLoader",
    "FunctionBasedRewardModel",
]
