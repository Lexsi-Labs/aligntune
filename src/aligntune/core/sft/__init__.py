"""
SFT core module - moved from /sft/core/ to /core/sft/

This module contains all SFT-related core functionality including:
- Configuration classes
- Trainer base classes and factories
- Evaluation and logging
- Model management
"""

from .config import (
    SFTConfig,
    TaskType,
    PrecisionType,
    BackendType,
    ModelConfig,
    DatasetConfig,
    TrainingConfig,
    LoggingConfig,
)

from .trainer_base import SFTTrainerBase
from .trainer_factory import SFTTrainerFactory
from .evaluator import SFTEvaluator
from .logging import SFTLogger
from .config_loader import SFTConfigLoader

__all__ = [
    # Config classes
    "SFTConfig",
    "TaskType",
    "PrecisionType", 
    "BackendType",
    "ModelConfig",
    "DatasetConfig",
    "TrainingConfig",
    "LoggingConfig",
    # Core classes
    "SFTTrainerBase",
    "SFTTrainerFactory",
    "SFTEvaluator",
    "SFTLogger",
    "SFTConfigLoader",
]
