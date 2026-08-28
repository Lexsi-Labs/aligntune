"""
Unsloth Dr. GRPO Backend - Thin Wrapper over Unsloth GRPO

Dr. GRPO (Doctor GRPO) is a GRPO variant that normalizes by a global constant
(approximately max_completion_length), removing length bias.

This is a thin wrapper that inherits from UnslothGRPOTrainer and sets loss_type='dr_grpo',
with Unsloth optimizations for 2-5x faster training.

Reference: Dr. GRPO paper introduces this constant normalization approach.
"""

import logging
from typing import Dict, Any
from aligntune.core.rl.config import UnifiedConfig
from ..grpo.grpo import UnslothGRPOTrainer

logger = logging.getLogger(__name__)


class UnslothDRGRPOTrainer(UnslothGRPOTrainer):
    """
    Unsloth Dr. GRPO trainer - inherits from UnslothGRPOTrainer with loss_type='dr_grpo'.

    Dr. GRPO normalizes by a global constant (typically max_completion_length),
    which removes length bias while being simpler than DAPO's batch-based normalization.

    Uses Unsloth optimizations for 2-5x faster training compared to standard TRL.

    All functionality is inherited from UnslothGRPOTrainer - this class only overrides
    the default loss_type configuration parameter.
    """

    def __init__(self, config: UnifiedConfig):
        """Initialize Unsloth Dr. GRPO trainer with loss_type='dr_grpo'."""
        # Override default loss_type to 'dr_grpo' if not explicitly set
        if getattr(config.train, 'loss_type', None) is None:
            config.train.loss_type = 'dr_grpo'

        # Call parent constructor (does all the work)
        super().__init__(config)

        logger.info("=" * 80)
        logger.info("Initialized Unsloth Dr. GRPO trainer (GRPO with loss_type='dr_grpo')")
        logger.info("Dr. GRPO removes length bias by normalizing with global constant")
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
