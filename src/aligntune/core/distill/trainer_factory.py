"""
Trainer factory for Knowledge Distillation.

Automatically selects the appropriate distillation trainer based on config.
Delegates to Backend Factory for backend selection.
"""

import logging
from typing import Any

from .config import UnifiedDistillConfig

logger = logging.getLogger(__name__)


class TrainerFactory:
    """Factory for creating distillation trainers - delegates to Backend Factory."""

    @classmethod
    def create_trainer(cls, config: UnifiedDistillConfig) -> Any:
        """
        Create a distillation trainer by delegating to Backend Factory.

        Args:
            config: UnifiedDistillConfig configuration

        Returns:
            Trainer instance from Backend Factory
        """
        # Import here to avoid circular imports
        from ..backend_factory import (
            BackendFactory, BackendConfig, BackendType, TrainingType,
            _enable_unsloth_backend, _disable_unsloth_backend,
        )

        # Extract backend from config - default to TRL
        backend = BackendType.TRL
        if hasattr(config.model, 'backend'):
            backend_str = config.model.backend
            if isinstance(backend_str, str):
                try:
                    backend = BackendType(backend_str.lower())
                except ValueError:
                    logger.warning(
                        f"Unknown backend '{backend_str}' in config.model.backend, "
                        f"falling back to 'trl'. Valid options: {[b.value for b in BackendType]}"
                    )
                    backend = BackendType.TRL
            elif hasattr(backend_str, 'value'):
                backend = backend_str

        # Get distillation type from config
        distill_type = config.get_distillation_type()

        # Create backend config with distillation type as algorithm
        backend_config = BackendConfig(
            training_type=TrainingType.DISTILL,
            backend=backend,
            algorithm=distill_type.value  # Use distillation type as algorithm for routing
        )

        logger.info(f"Creating distillation trainer with backend: {backend.value}")
        logger.info(f"Distillation method: {distill_type.value}")

        # Set PURE_TRL_MODE only when TRL backend is being used, mirroring
        # create_distill_trainer()/create_rl_trainer() in backend_factory.py -
        # otherwise a prior TRL-mode call in the same process can leave
        # PURE_TRL_MODE=1 set and silently break Unsloth trainer construction.
        if backend == BackendType.TRL or backend == BackendType.VERL:
            _disable_unsloth_backend()
        else:
            _enable_unsloth_backend()

        # Delegate to Backend Factory
        return BackendFactory.create_trainer(config, backend_config)


def create_trainer_from_config(config: UnifiedDistillConfig) -> Any:
    """
    Create distillation trainer from config.

    Args:
        config: UnifiedDistillConfig object

    Returns:
        Trainer instance (TRL or future backends)

    Raises:
        ValueError: If backend is not supported for distillation
    """
    return TrainerFactory.create_trainer(config)
