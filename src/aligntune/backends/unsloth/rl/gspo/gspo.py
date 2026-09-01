"""
Unsloth GSPO Backend - Thin Wrapper over Unsloth GRPO

GSPO (Generalized Scoring Proximal Objective) uses sequence-level importance sampling
with length normalization: w^GSPO_i = [π_θ(y_i|x) / π_θ_old(y_i|x)]^(1/|y_i|)

This is a thin wrapper that inherits from UnslothGRPOTrainer and sets
importance_sampling_level='sequence' for more stable sequence-level rewards,
with Unsloth optimizations for 2-5x faster training.

Note: TRL's GRPOTrainer doesn't have native GSPO implementation, but setting
importance_sampling_level='sequence' provides similar benefits for sequence rewards.

Reference: GSPO paper shows sequence-level sampling is more stable for sequence-level rewards.
"""

import logging
from typing import Dict, Any
from aligntune.core.rl.config import UnifiedConfig
from ..grpo.grpo import UnslothGRPOTrainer

logger = logging.getLogger(__name__)


class UnslothGSPOTrainer(UnslothGRPOTrainer):
    """
    Unsloth GSPO trainer - inherits from UnslothGRPOTrainer with importance_sampling_level='sequence'.

    GSPO uses sequence-level importance sampling instead of token-level sampling,
    which provides more stable training when using sequence-level rewards.

    Uses Unsloth optimizations for 2-5x faster training compared to standard TRL.

    All functionality is inherited from UnslothGRPOTrainer - this class only overrides
    the default importance_sampling_level configuration parameter.
    """

    def __init__(self, config: UnifiedConfig):
        """Initialize Unsloth GSPO trainer with importance_sampling_level='sequence'."""
        # Override default importance sampling to 'sequence' if not explicitly set
        if getattr(config.train, 'importance_sampling_level', None) is None:
            config.train.importance_sampling_level = 'sequence'

        # GSPO paper also recommends using DAPO loss for better stability
        if getattr(config.train, 'loss_type', None) is None:
            config.train.loss_type = 'dapo'

        # Call parent constructor (does all the work)
        super().__init__(config)

        logger.info("=" * 80)
        logger.info("Initialized Unsloth GSPO trainer (importance_sampling_level='sequence')")
        logger.info("GSPO uses sequence-level sampling for more stable sequence rewards")
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
