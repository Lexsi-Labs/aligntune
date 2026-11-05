"""Compatibility shims between TRL's custom trainers and different transformers versions."""
from typing import Any


def ensure_save_model_compat(trainer: Any) -> Any:
    """Give a trainer the attributes ``Trainer.save_model`` reads in newer transformers.

    TRL's experimental ``PPOTrainer`` has its own ``__init__`` and never calls
    ``transformers.Trainer.__init__``, so it lacks ``is_distributed_loading_by_transformers``.
    transformers 5.17 does not read it in ``save_model``; 5.22 does and raises ``AttributeError``.
    Setting the default (False, what ``Trainer.__init__`` uses for an ordinary model) is harmless
    on every version.
    """
    if trainer is not None and not hasattr(trainer, "is_distributed_loading_by_transformers"):
        trainer.is_distributed_loading_by_transformers = False
    return trainer
