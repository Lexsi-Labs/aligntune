"""
GBMPO - Simplified GBMPO Implementation via Loss Patching

This package provides GBMPO (Generalized Bregman Mirror Descent Policy Optimization)
using a clean, simple architecture:

Architecture:
- TRLGBMPOTrainer: Extends TRLGRPOTrainer, patches loss to add L2
- GBMPOConfig: Still used for old tests (can be deprecated later)

Key Benefits:
- No custom trainer class needed (just patch TRL GRPOTrainer)
- Minimal code (~370 lines vs 2849 in old implementation)
- Easy to maintain (patch is self-contained)
- All divergence types supported (l2, l2kl, prob_l2, prob_l2kl)

Usage:
    from aligntune.core.backend_factory import create_rl_trainer

    trainer = create_rl_trainer(
        algorithm="gbmpo",
        gbmpo_divergence_type="l2",
        gbmpo_l2_coefficient=0.0001,
        model_name="gpt2",
        dataset_name="openai/gsm8k",
        ...
    )
    results = trainer.train()
"""

from .gbmpo import TRLGBMPOTrainer
from .config import GBMPOConfig, VALID_DIVERGENCE_TYPES

__all__ = [
    "TRLGBMPOTrainer",
    "GBMPOConfig",  # Keep for backward compat with old tests
    "VALID_DIVERGENCE_TYPES",
]

__version__ = "2.0.0"  # Simplified version
