"""
Unsloth DAPO Backend - Thin Wrapper over Unsloth GRPO

DAPO (Difficulty-Aware Policy Optimization) is a GRPO variant that normalizes
gradients by active tokens in the global accumulated batch, removing length bias.

This is a thin wrapper that inherits from UnslothGRPOTrainer and sets loss_type='dapo',
with Unsloth optimizations for 2-5x faster training.

Reference: DAPO paper recommends this normalization for better training stability.
"""

import logging
from typing import Dict, Any
from aligntune.core.rl.config import UnifiedConfig
from ..grpo.grpo import UnslothGRPOTrainer

logger = logging.getLogger(__name__)


class UnslothDAPOTrainer(UnslothGRPOTrainer):
    """
    Unsloth DAPO trainer - inherits from UnslothGRPOTrainer with loss_type='dapo'.

    DAPO normalizes by active tokens in the global accumulated batch (introduced in DAPO paper),
    which removes length bias that exists in standard GRPO normalization.

    Uses Unsloth optimizations for 2-5x faster training compared to standard TRL.

    All functionality is inherited from UnslothGRPOTrainer - this class only overrides
    the default loss_type configuration parameter.
    """

    def __init__(self, config: UnifiedConfig):
        """Initialize Unsloth DAPO trainer with loss_type='dapo'."""
        # Override default loss_type to 'dapo' if not explicitly set
        if getattr(config.train, 'loss_type', None) is None:
            config.train.loss_type = 'dapo'

        # Call parent constructor (does all the work)
        super().__init__(config)

        logger.info("=" * 80)
        logger.info("Initialized Unsloth DAPO trainer (GRPO with loss_type='dapo')")
        logger.info("DAPO removes length bias by normalizing with active tokens")
        logger.info("Using Unsloth optimizations for 2-5x speed improvement")
        logger.info("=" * 80)

    @classmethod
    def is_available(cls) -> bool:
        """Check if Unsloth, TRL and dependencies are available."""
        return UnslothGRPOTrainer.is_available()

    def setup_rewards(self) -> None:
        """Setup reward functions using parent GRPO implementation."""
        super().setup_rewards()

    def train_step(self, batch):
        """Train step (handled by TRL trainer)."""
        pass
