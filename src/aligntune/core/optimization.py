"""
Optimization utilities for AlignTune.


This module provides registries for optimizers and schedulers with support for
TRL-compatible configurations and multiple backend implementations.
"""

import logging
from typing import Dict, Any, Optional, Callable, Union, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizerType(Enum):
    """Supported optimizer types."""
    ADAMW_TORCH = "adamw_torch"
    ADAMW_8BIT = "adamw_8bit"
    ADAMW_BNB_8BIT = "adamw_bnb_8bit"
    ADAMW_PAGED = "adamw_paged"
    LION_8BIT = "lion_8bit"
    ADAFACTOR = "adafactor"
    SGD = "sgd"
    RMSPROP = "rmsprop"


class SchedulerType(Enum):
    """Supported scheduler types."""
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    INVERSE_SQRT = "inverse_sqrt"
    REDUCE_LR_ON_PLATEAU = "reduce_lr_on_plateau"


@dataclass
class OptimizerConfig:
    """Configuration for an optimizer."""
    class_name: str
    params: Dict[str, Any]
    description: str
    best_for: List[str]
    requires_bitsandbytes: bool = False
    requires_unsloth: bool = False


@dataclass
class SchedulerConfig:
    """Configuration for a learning rate scheduler."""
    class_name: str
    params: Dict[str, Any]
    description: str
    best_for: List[str]


class OptimizerRegistry:
    """Registry of available optimizers with TRL-compatible configurations."""

    _registry: Dict[str, OptimizerConfig] = {}

    @classmethod
    def register(cls, name: str, config: OptimizerConfig) -> None:
        """Register an optimizer configuration."""
        cls._registry[name] = config
        logger.debug(f"Registered optimizer: {name}")

    @classmethod
    def get(cls, name: str) -> Optional[OptimizerConfig]:
        """Get optimizer configuration by name."""
        return cls._registry.get(name)

    @classmethod
    def list_available(cls) -> Dict[str, OptimizerConfig]:
        """List all available optimizers."""
        return cls._registry.copy()

    @classmethod
    def is_available(cls, name: str) -> bool:
        """Check if optimizer is available."""
        return name in cls._registry

    @classmethod
    def get_trl_compatible(cls, name: str, learning_rate: float, weight_decay: float = 0.01) -> Dict[str, Any]:
        """Get TRL-compatible optimizer configuration."""
        config = cls.get(name)
        if not config:
            raise ValueError(f"Unknown optimizer: {name}")

        # Merge learning rate and weight decay with base params
        params = config.params.copy()
        params['lr'] = learning_rate
        params['weight_decay'] = weight_decay

        return {
            'optimizer': config.class_name,
            'optimizer_kwargs': params
        }


class SchedulerRegistry:
    """Registry of available learning rate schedulers."""

    _registry: Dict[str, SchedulerConfig] = {}

    @classmethod
    def register(cls, name: str, config: SchedulerConfig) -> None:
        """Register a scheduler configuration."""
        cls._registry[name] = config
        logger.debug(f"Registered scheduler: {name}")

    @classmethod
    def get(cls, name: str) -> Optional[SchedulerConfig]:
        """Get scheduler configuration by name."""
        return cls._registry.get(name)

    @classmethod
    def list_available(cls) -> Dict[str, SchedulerConfig]:
        """List all available schedulers."""
        return cls._registry.copy()

    @classmethod
    def is_available(cls, name: str) -> bool:
        """Check if scheduler is available."""
        return name in cls._registry

    @classmethod
    def get_trl_compatible(cls, name: str, num_training_steps: int, warmup_steps: int = 0) -> Dict[str, Any]:
        """Get TRL-compatible scheduler configuration."""
        config = cls.get(name)
        if not config:
            raise ValueError(f"Unknown scheduler: {name}")

        # Merge training steps with base params
        params = config.params.copy()
        params['num_training_steps'] = num_training_steps
        params['num_warmup_steps'] = warmup_steps

        return {
            'lr_scheduler': config.class_name,
            'lr_scheduler_kwargs': params
        }


# Register default optimizers
OptimizerRegistry.register(
    OptimizerType.ADAMW_TORCH.value,
    OptimizerConfig(
        class_name="torch.optim.AdamW",
        params={'betas1': 0.9,'betas2':0.999, 'eps': 1e-8},
        description="Standard PyTorch AdamW optimizer",
        best_for=["general", "sft", "instruction_following", "classification"],
        requires_bitsandbytes=False
    )
)

OptimizerRegistry.register(
    OptimizerType.ADAMW_8BIT.value,
    OptimizerConfig(
        class_name="bitsandbytes.optim.AdamW8bit",
        params={'betas': (0.9, 0.999), 'eps': 1e-8},
        description="8-bit AdamW with bitsandbytes",
        best_for=["memory_efficient", "large_models", "long_training"],
        requires_bitsandbytes=True
    )
)

OptimizerRegistry.register(
    OptimizerType.ADAMW_BNB_8BIT.value,
    OptimizerConfig(
        class_name="bitsandbytes.optim.AdamW8bit",
        params={'betas': (0.9, 0.999), 'eps': 1e-8},
        description="BitsAndBytes 8-bit AdamW",
        best_for=["memory_efficient", "large_models"],
        requires_bitsandbytes=True
    )
)

OptimizerRegistry.register(
    OptimizerType.ADAMW_PAGED.value,
    OptimizerConfig(
        class_name="bitsandbytes.optim.AdamW8bit",
        params={'betas': (0.9, 0.999), 'eps': 1e-8, 'is_paged': True},
        description="Paged 8-bit AdamW for very large models",
        best_for=["very_large_models", "extreme_memory_efficiency"],
        requires_bitsandbytes=True
    )
)

# Register TRL-compatible alias for paged_adamw_8bit
OptimizerRegistry.register(
    "paged_adamw_8bit",
    OptimizerConfig(
        class_name="paged_adamw_8bit",  # TRL accepts this as a string directly
        params={'betas': (0.9, 0.999), 'eps': 1e-8},
        description="Paged 8-bit AdamW (TRL-compatible name)",
        best_for=["very_large_models", "extreme_memory_efficiency"],
        requires_bitsandbytes=True
    )
)

OptimizerRegistry.register(
    OptimizerType.LION_8BIT.value,
    OptimizerConfig(
        class_name="bitsandbytes.optim.Lion8bit",
        params={},
        description="Lion optimizer (memory efficient, fast convergence)",
        best_for=["large_models", "long_training", "memory_constrained"],
        requires_bitsandbytes=True
    )
)

OptimizerRegistry.register(
    OptimizerType.ADAFACTOR.value,
    OptimizerConfig(
        class_name="transformers.Adafactor",
        params={
            'scale_parameter': True,
            'relative_step': True,
            'warmup_init': False
        },
        description="Adafactor (adaptive, memory efficient)",
        best_for=["memory_constrained", "t5_models", "long_sequences"],
        requires_bitsandbytes=False
    )
)

OptimizerRegistry.register(
    OptimizerType.SGD.value,
    OptimizerConfig(
        class_name="torch.optim.SGD",
        params={'momentum': 0.9},
        description="Stochastic Gradient Descent",
        best_for=["simple_models", "debugging"],
        requires_bitsandbytes=False
    )
)

OptimizerRegistry.register(
    OptimizerType.RMSPROP.value,
    OptimizerConfig(
        class_name="torch.optim.RMSprop",
        params={'alpha': 0.99, 'eps': 1e-8},
        description="RMSprop optimizer",
        best_for=["rnn_models", "some_cv_tasks"],
        requires_bitsandbytes=False
    )
)

# Register default schedulers
SchedulerRegistry.register(
    SchedulerType.LINEAR.value,
    SchedulerConfig(
        class_name="transformers.get_linear_schedule_with_warmup",
        params={},
        description="Linear decay with warmup",
        best_for=["general", "most_tasks"]
    )
)

SchedulerRegistry.register(
    SchedulerType.COSINE.value,
    SchedulerConfig(
        class_name="transformers.get_cosine_schedule_with_warmup",
        params={},
        description="Cosine annealing with warmup",
        best_for=["large_models", "long_training"]
    )
)

SchedulerRegistry.register(
    SchedulerType.COSINE_WITH_RESTARTS.value,
    SchedulerConfig(
        class_name="transformers.get_cosine_with_hard_restarts_schedule_with_warmup",
        params={'num_cycles': 3},
        description="Cosine annealing with restarts and warmup",
        best_for=["very_long_training", "curriculum_learning"]
    )
)

SchedulerRegistry.register(
    SchedulerType.POLYNOMIAL.value,
    SchedulerConfig(
        class_name="transformers.get_polynomial_decay_schedule_with_warmup",
        params={'power': 1.0},
        description="Polynomial decay with warmup",
        best_for=["fine_tuning", "short_training"]
    )
)

SchedulerRegistry.register(
    SchedulerType.CONSTANT.value,
    SchedulerConfig(
        class_name="transformers.get_constant_schedule",
        params={},
        description="Constant learning rate",
        best_for=["debugging", "simple_experiments"]
    )
)

SchedulerRegistry.register(
    SchedulerType.CONSTANT_WITH_WARMUP.value,
    SchedulerConfig(
        class_name="transformers.get_constant_schedule_with_warmup",
        params={},
        description="Constant learning rate after warmup",
        best_for=["some_fine_tuning", "stable_training"]
    )
)

SchedulerRegistry.register(
    SchedulerType.INVERSE_SQRT.value,
    SchedulerConfig(
        class_name="transformers.get_inverse_sqrt_schedule",
        params={},
        description="Inverse square root decay",
        best_for=["transformers", "language_modeling"]
    )
)

SchedulerRegistry.register(
    SchedulerType.REDUCE_LR_ON_PLATEAU.value,
    SchedulerConfig(
        class_name="torch.optim.lr_scheduler.ReduceLROnPlateau",
        params={'mode': 'min', 'factor': 0.1, 'patience': 10},
        description="Reduce LR when metric plateaus",
        best_for=["validation_based", "some_cv_tasks"]
    )
)


def get_optimizer_for_config(optimizer_name: str, learning_rate: float, weight_decay: float = 0.01) -> Dict[str, Any]:
    """
    Get optimizer configuration for TRL training.

    Args:
        optimizer_name: Name of optimizer from registry
        learning_rate: Learning rate
        weight_decay: Weight decay

    Returns:
        Dictionary with optimizer configuration for TRL
    """
    return OptimizerRegistry.get_trl_compatible(optimizer_name, learning_rate, weight_decay)


def get_scheduler_for_config(scheduler_name: str, num_training_steps: int, warmup_steps: int = 0) -> Dict[str, Any]:
    """
    Get scheduler configuration for TRL training.

    Args:
        scheduler_name: Name of scheduler from registry
        num_training_steps: Total training steps
        warmup_steps: Warmup steps

    Returns:
        Dictionary with scheduler configuration for TRL
    """
    return SchedulerRegistry.get_trl_compatible(scheduler_name, num_training_steps, warmup_steps)


def validate_optimizer_availability(optimizer_name: str) -> bool:
    """
    Check if optimizer is available with current dependencies.

    Args:
        optimizer_name: Name of optimizer

    Returns:
        True if available, False otherwise
    """
    config = OptimizerRegistry.get(optimizer_name)
    if not config:
        return False

    # Check bitsandbytes availability
    if config.requires_bitsandbytes:
        try:
            import bitsandbytes
        except ImportError:
            logger.warning(f"Optimizer {optimizer_name} requires bitsandbytes but it's not installed")
            return False

    # Check unsloth availability
    if config.requires_unsloth:
        try:
            import unsloth
        except ImportError:
            logger.warning(f"Optimizer {optimizer_name} requires unsloth but it's not installed")
            return False

    return True


def validate_scheduler_availability(scheduler_name: str) -> bool:
    """
    Check if scheduler is available.

    Args:
        scheduler_name: Name of scheduler

    Returns:
        True if available, False otherwise
    """
    return SchedulerRegistry.is_available(scheduler_name)


# Export utilities
__all__ = [
    'OptimizerType',
    'SchedulerType',
    'OptimizerConfig',
    'SchedulerConfig',
    'OptimizerRegistry',
    'SchedulerRegistry',
    'get_optimizer_for_config',
    'get_scheduler_for_config',
    'validate_optimizer_availability',
    'validate_scheduler_availability'
]
